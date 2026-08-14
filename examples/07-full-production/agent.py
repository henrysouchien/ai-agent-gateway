import asyncio
import json
import os
import time
from pathlib import Path

from agent_gateway import (
  AgentRunner,
  AnthropicProvider,
  ApprovalDecision,
  ChatRuntime,
  CredentialHandle,
  GatewayServerConfig,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
  McpClientManager,
  ToolDispatcher,
  UsageEvent,
  create_gateway_app,
  load_rate_table,
)
from agent_gateway.server import MaterializedCredential


BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL = INITIAL_MODEL_REGISTRY.require(
  "anthropic.claude-opus-5"
).upstream_model
DEFAULT_EFFORT = "high"
TENANT_ID = "example-07-full-production"
SERVICE_CREDENTIAL_HANDLE = CredentialHandle(
  handle_id="example-07-anthropic-service",
  provider="anthropic",
  principal="service",
  tenant_id=TENANT_ID,
  actor_id=None,
)


def build_auth_config(*, rate_table_version: str) -> dict[str, object]:
  api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
  auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
  if auth_token and not api_key:
    auth_mode = "oauth"
  else:
    auth_mode = "api"
  return {
    "provider": "anthropic",
    "auth_mode": auth_mode,
    "api_key": api_key,
    "auth_token": auth_token if auth_mode == "oauth" else "",
    "max_tokens": 16_000,
    "billing_mode": "byok",
    "rate_table_version": rate_table_version,
  }


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
  with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, default=str) + "\n")


def _safe_report_path(filename: str) -> Path:
  cleaned = Path(filename).name
  if not cleaned:
    raise ValueError("filename is required")
  return REPORTS_DIR / cleaned


async def write_report(tool_input, **_kwargs):
  filename = str(tool_input.get("filename") or "").strip()
  content = str(tool_input.get("content") or "").strip()
  if not filename or not content:
    return None, {"code": "invalid_input", "message": "filename and content are required"}

  path = _safe_report_path(filename)
  path.write_text(content + "\n", encoding="utf-8")
  return {"ok": True, "path": str(path), "report": path.name}, None


async def list_reports(_tool_input, **_kwargs):
  files = sorted(path.name for path in REPORTS_DIR.glob("*") if path.is_file())
  return {"files": files, "count": len(files)}, None


def report_tool_definitions() -> list[dict[str, object]]:
  return [
    {
      "name": "write_report",
      "description": "Write a markdown report to the reports directory. Requires approval in this example.",
      "input_schema": {
        "type": "object",
        "properties": {
          "filename": {"type": "string", "description": "Report file name such as launch-checklist.md."},
          "content": {"type": "string", "description": "Markdown content for the report."},
        },
        "required": ["filename", "content"],
      },
    },
    {
      "name": "list_reports",
      "description": "List reports currently stored on disk.",
      "input_schema": {
        "type": "object",
        "properties": {},
      },
    },
  ]


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


def on_event(event: dict[str, object], session_id: str) -> None:
  append_jsonl(LOGS_DIR / "events.jsonl", {"session_id": session_id, "event": event})


def on_usage(event: UsageEvent) -> None:
  append_jsonl(LOGS_DIR / "usage.jsonl", event.__dict__)


def on_tool_timing(
  session_id: str,
  tool_name: str,
  server: str | None,
  duration_ms: int,
  is_error: bool,
  result_bytes: int,
  *,
  user_id: str | None = None,
) -> None:
  append_jsonl(
    LOGS_DIR / "tool_timing.jsonl",
    {
      "session_id": session_id,
      "tool_name": tool_name,
      "server": server,
      "duration_ms": duration_ms,
      "is_error": is_error,
      "result_bytes": result_bytes,
      "user_id": user_id,
    },
  )


rate_table = load_rate_table()
provider = AnthropicProvider(rate_table=rate_table)
mcp_client = McpClientManager(config_path=None)
AUTH_CONFIG = build_auth_config(rate_table_version=rate_table.version)
TOOL_DEFINITIONS = report_tool_definitions()
LOCAL_HANDLERS = {
  "write_report": write_report,
  "list_reports": list_reports,
}


def admit_demo_session(session, _api_key, _request) -> None:
  session.role = "owner"
  session.channel = "web"
  session.model_entitled_capabilities = frozenset({"session.driver"})
  session.model_entitled_keys = frozenset({"anthropic.claude-opus-5"})


def materialize_service_credential(
  handle: CredentialHandle,
) -> MaterializedCredential:
  if handle is not SERVICE_CREDENTIAL_HANDLE:
    raise RuntimeError("Unknown service credential handle")
  return MaterializedCredential(handle=handle, auth_config=AUTH_CONFIG)


async def build_chat_runtime(session, request, channel, auth_manager) -> ChatRuntime:
  _ = auth_manager
  capability_execution = request.capability_execution
  if capability_execution is None:
    raise RuntimeError("Runtime requires a prepared session.driver turn")
  capability_execution.validate()
  bound_auth_config = capability_execution.auth_config
  system_prompt = (
    "You are a production-style launch assistant. "
    "Use list_reports to inspect existing deliverables. "
    "Use write_report when the user explicitly asks to persist a report or checklist."
  )
  if channel == "web":
    system_prompt += " Keep responses concise for browser users."

  def get_tool_definitions():
    return list(TOOL_DEFINITIONS)

  def build_runner(event_log, session_id, started_at):
    dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=LOCAL_HANDLERS,
      needs_approval=lambda tool_name, _tool_input, _qualifier: tool_name == "write_report",
      request_approval=make_request_approval(session, event_log),
      approved_tool_types=session.approved_tool_types,
      event_log=event_log,
      session_id=session_id,
    )
    return AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id=session_id,
      started_at=started_at,
      capability_execution=capability_execution,
      mcp_client=mcp_client,
      get_tool_definitions=get_tool_definitions,
      on_usage=on_usage,
      on_tool_timing=on_tool_timing,
      per_turn_timeout=90,
      max_budget_usd=0.50,
      user_id=session.user_id,
      request_id=request.request_id,
      billing_mode=str(bound_auth_config["billing_mode"]),
      rate_table_version=str(bound_auth_config["rate_table_version"]),
      channel=channel,
    )

  return ChatRuntime(
    system_prompt=system_prompt,
    build_runner=build_runner,
    capability_execution=capability_execution,
    get_tool_definitions=get_tool_definitions,
    max_turns=8,
  )


app = create_gateway_app(
  GatewayServerConfig(
    tenant_id=TENANT_ID,
    allow_service_credentials_for_interactive=True,
    valid_api_keys={"demo-key"},
    session_ttl=7_200,
    cors_origins=["http://localhost:3000", "https://app.example.com"],
    on_session_created=admit_demo_session,
    model_registry=INITIAL_MODEL_REGISTRY,
    model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    default_provider=provider,
    service_provider_handles={"anthropic": SERVICE_CREDENTIAL_HANDLE},
    service_auth_config_resolver=materialize_service_credential,
    build_chat_runtime=build_chat_runtime,
    mcp_client=mcp_client,
    on_event=on_event,
    on_startup=mcp_client.startup,
    on_shutdown=mcp_client.shutdown,
    transcript_dir=TRANSCRIPTS_DIR,
    log_name="agent_gateway.production_example",
  )
)
