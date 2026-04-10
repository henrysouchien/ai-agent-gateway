import asyncio
import os
import time
from pathlib import Path

from agent_gateway import (
  AgentRunner,
  AnthropicProvider,
  ApprovalDecision,
  ChatRuntime,
  GatewayServerConfig,
  McpClientManager,
  ToolDispatcher,
  create_gateway_app,
)


BASE_DIR = Path(__file__).parent
NOTES_DIR = BASE_DIR / "approved_notes"
NOTES_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL = "claude-sonnet-4-6"


def build_auth_config() -> dict[str, object]:
  api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
  auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
  if auth_token and not api_key:
    auth_mode = "oauth"
  else:
    auth_mode = "api"
  return {
    "auth_mode": auth_mode,
    "api_key": api_key,
    "auth_token": auth_token if auth_mode == "oauth" else "",
    "model": DEFAULT_MODEL,
    "max_tokens": 16_000,
  }


def _safe_path(filename: str) -> Path:
  cleaned = Path(filename).name
  if not cleaned:
    raise ValueError("filename is required")
  return NOTES_DIR / cleaned


async def write_note(tool_input, **_kwargs):
  filename = str(tool_input.get("filename") or "").strip()
  text = str(tool_input.get("text") or "").strip()
  if not filename or not text:
    return None, {"code": "invalid_input", "message": "filename and text are required"}

  path = _safe_path(filename)
  path.write_text(text + "\n", encoding="utf-8")
  return {"ok": True, "path": str(path), "message": f"Wrote {path.name}"}, None


def write_note_tool_def() -> dict[str, object]:
  return {
    "name": "write_note",
    "description": "Write a note to the approved_notes directory. Requires approval in this example.",
    "input_schema": {
      "type": "object",
      "properties": {
        "filename": {"type": "string", "description": "File name such as release-plan.txt."},
        "text": {"type": "string", "description": "Text content to store."},
      },
      "required": ["filename", "text"],
    },
  }


def make_request_approval(session, event_log):
  async def request_approval(payload):
    approval_queue = asyncio.Queue(maxsize=1)
    session.pending_tools[payload.tool_call_id] = {
      "nonce": payload.nonce,
      "requested_at": int(time.time()),
      "status": "approval_pending",
      "tool_name": payload.tool_name,
      "resolved_qualifier": payload.resolved_qualifier,
    }
    session.approval_queues[payload.tool_call_id] = approval_queue

    event_log.append(
      {
        "type": "tool_approval_request",
        "tool_call_id": payload.tool_call_id,
        "nonce": payload.nonce,
        "tool_name": payload.tool_name,
        "tool_input": payload.tool_input,
        "resolved_qualifier": payload.resolved_qualifier,
      }
    )

    try:
      approval = await approval_queue.get()
    finally:
      session.pending_tools.pop(payload.tool_call_id, None)
      session.approval_queues.pop(payload.tool_call_id, None)

    return ApprovalDecision(
      approved=bool(approval.get("approved")),
      allow_tool_type=bool(approval.get("allow_tool_type")),
    )

  return request_approval


provider = AnthropicProvider()
mcp_client = McpClientManager(config_path=None)
AUTH_CONFIG = build_auth_config()
TOOL_DEFINITIONS = [write_note_tool_def()]
LOCAL_HANDLERS = {"write_note": write_note}


async def build_chat_runtime(session, request, channel, auth_manager) -> ChatRuntime:
  _ = channel, auth_manager

  def get_tool_definitions():
    return list(TOOL_DEFINITIONS)

  def build_runner(event_log, session_id):
    dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=LOCAL_HANDLERS,
      needs_approval=lambda tool_name, _tool_input, _qualifier: tool_name == "write_note",
      request_approval=make_request_approval(session, event_log),
      approved_tool_types=session.approved_tool_types,
      event_log=event_log,
      session_id=session_id,
    )
    return AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id=session_id,
      provider=provider,
      auth_config=AUTH_CONFIG,
      mcp_client=mcp_client,
      get_tool_definitions=get_tool_definitions,
    )

  return ChatRuntime(
    system_prompt=(
      "You can write release notes with the write_note tool. "
      "When a user asks to persist text to disk, call the tool instead of pretending it was saved."
    ),
    build_runner=build_runner,
    get_tool_definitions=get_tool_definitions,
    provider=provider,
    model_override=request.model or DEFAULT_MODEL,
    max_turns=6,
  )


app = create_gateway_app(
  GatewayServerConfig(
    valid_api_keys={"demo-key"},
    cors_origins=["*"],
    allowed_models={DEFAULT_MODEL},
    build_chat_runtime=build_chat_runtime,
    default_provider=provider,
    auth_config=AUTH_CONFIG,
    mcp_client=mcp_client,
    on_startup=mcp_client.startup,
    on_shutdown=mcp_client.shutdown,
  )
)
