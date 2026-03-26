from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
import json as json_mod
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .event_log import EventLog
from .mcp_client import McpClientManager
from .providers.agent_sdk import AgentSDKConfig
from .providers import AnthropicProvider, ModelProvider
from .runner import AgentRunner
from .sdk_runner import AgentSDKRunner
from .session import AuthManager, Session, SessionStore
from .tool_dispatcher import ApprovalDecision, ApprovalRequest


SystemPrompt = str | List[Tuple[str, bool]]
ExecutionLocationResolver = Callable[[str], Optional[str]]
BuildChatRuntime = Callable[[Session, "ChatRequest", Optional[str], AuthManager], Awaitable["ChatRuntime"]]
RequestApproval = Callable[[ApprovalRequest], Awaitable[Optional[ApprovalDecision]]]
BuildRunner = Callable[[EventLog, str], AgentRunner | AgentSDKRunner]


class ChatInitRequest(BaseModel):
  api_key: str = Field(..., min_length=1)


class ChatInitResponse(BaseModel):
  session_token: str
  session_id: str
  expires_at: int


class ChatMessage(BaseModel):
  role: str
  content: str


class ChatRequest(BaseModel):
  messages: List[ChatMessage]
  context: Dict[str, Any] = Field(default_factory=dict)
  model: Optional[str] = None


class ToolResultRequest(BaseModel):
  tool_call_id: str
  nonce: str
  result: Optional[Dict[str, Any]] = None
  error: Optional[Dict[str, Any]] = None


class ToolApprovalRequest(BaseModel):
  tool_call_id: str
  nonce: str
  approved: bool
  allow_tool_type: bool = False


@dataclass
class ChatRuntime:
  system_prompt: SystemPrompt
  build_runner: BuildRunner
  get_tool_definitions: Callable[[], List[Dict[str, Any]]] = field(default_factory=lambda: [])
  build_dispatcher: Callable[["RequestContext"], Any] | None = None
  provider: ModelProvider | None = None
  model_override: Optional[str] = None
  excluded_tools: Optional[Set[str]] = None
  execution_location: Optional[ExecutionLocationResolver] = None
  on_usage: Optional[Callable[..., Any]] = None
  on_tool_result: Optional[Callable[..., Any]] = None
  on_tool_timing: Optional[Callable[..., Any]] = None
  post_runner_init: Optional[Callable[[Any], None]] = None
  max_turns: Optional[int] = None
  compaction_trigger: int | None = None
  compaction_instructions: str | None = None


@dataclass
class RequestContext:
  session: Session
  event_log: EventLog
  request_approval: RequestApproval
  result_queue: asyncio.Queue
  mcp_client: Optional[McpClientManager]


@dataclass
class GatewayServerConfig:
  auth_manager: Optional[AuthManager] = None
  default_provider: ModelProvider | None = None
  jwt_secret: str = "dev-secret-change-me"
  session_ttl: int = 3600
  valid_api_keys: Set[str] = field(default_factory=set)
  cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3002"])
  cors_allow_headers: List[str] = field(
    default_factory=lambda: ["Authorization", "Content-Type", "X-MCP-Secret"]
  )
  cors_allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
  auth_config: Dict[str, Any] = field(default_factory=dict)
  mcp_client: Optional[McpClientManager] = None
  sdk_config: AgentSDKConfig | None = None
  per_turn_timeout: int = 300
  compaction_trigger: int | None = None
  compaction_instructions: str | None = None
  allowed_models: Set[str] = field(default_factory=lambda: {"claude-sonnet-4-6", "claude-opus-4-6"})
  build_chat_runtime: Optional[BuildChatRuntime] = None
  on_event: Optional[Callable[..., Any]] = None
  on_tool_result: Optional[Callable[..., Any]] = None
  on_usage: Optional[Callable[..., Any]] = None
  on_tool_timing: Optional[Callable[..., Any]] = None
  on_startup: Optional[Callable[..., Any]] = None
  on_shutdown: Optional[Callable[..., Any]] = None
  transcript_dir: Optional[Path] = None
  log_name: str = "gateway"
  prefix: str = "/api"


def _model_to_dict(model: Any) -> Dict[str, Any]:
  if hasattr(model, "model_dump"):
    return model.model_dump()
  return model.dict()


def _normalize_prefix(prefix: str) -> str:
  cleaned = (prefix or "").strip()
  if not cleaned or cleaned == "/":
    return ""
  return "/" + cleaned.strip("/")


def _route_path(prefix: str, suffix: str) -> str:
  normalized = _normalize_prefix(prefix)
  return f"{normalized}{suffix}" if normalized else suffix


def _resolve_compaction_trigger(runtime_val: int | None, config_val: int | None) -> int | None:
  """Resolve compaction trigger: runtime overrides config. 0 or negative = explicitly disable."""
  raw = runtime_val if runtime_val is not None else config_val
  if raw is None or raw <= 0:
    return None
  return raw


def _sanitize_for_json(obj: Any) -> Any:
  if isinstance(obj, float) and not math.isfinite(obj):
    return None
  if isinstance(obj, dict):
    return {key: _sanitize_for_json(value) for key, value in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_sanitize_for_json(value) for value in obj]
  if isinstance(obj, (set, frozenset)):
    return [_sanitize_for_json(value) for value in obj]
  return obj


def _json_dumps(payload: Dict[str, Any]) -> str:
  sanitized = _sanitize_for_json(payload)
  return JSONResponse(content=sanitized).body.decode("utf-8")


def _drain_result_queue(queue: Optional[asyncio.Queue]) -> None:
  if queue is None:
    return
  while True:
    try:
      queue.get_nowait()
    except asyncio.QueueEmpty:
      break


def _write_transcript(transcript_dir: Optional[Path], session_id: str, entry: Dict[str, Any]) -> None:
  if transcript_dir is None:
    return
  payload = dict(entry)
  payload["ts"] = time.time()
  path = transcript_dir / f"{session_id}.jsonl"
  try:
    with open(path, "a", encoding="utf-8") as handle:
      handle.write(json_mod.dumps(payload, default=str) + "\n")
  except Exception:
    pass


async def _cleanup_sessions_loop(session_store: SessionStore) -> None:
  while True:
    await asyncio.sleep(300)
    await session_store.cleanup_expired_async()


async def _maybe_await(callback: Optional[Callable[..., Any]]) -> None:
  if callback is None:
    return
  result = callback()
  if inspect.isawaitable(result):
    await result


def _make_request_approval(session: Session, event_log: EventLog) -> RequestApproval:
  async def request_approval(payload: ApprovalRequest) -> Optional[ApprovalDecision]:
    expires_at = int(time.time() + payload.timeout)
    approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    session.pending_tools[payload.tool_call_id] = {
      "nonce": payload.nonce,
      "requested_at": int(time.time()),
      "expires_at": expires_at,
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
        "expires_at": expires_at,
        "tool_name": payload.tool_name,
        "tool_input": payload.tool_input,
        "resolved_qualifier": payload.resolved_qualifier,
      }
    )

    try:
      approval = await asyncio.wait_for(approval_queue.get(), timeout=payload.timeout)
    except asyncio.TimeoutError:
      return None
    finally:
      session.pending_tools.pop(payload.tool_call_id, None)
      session.approval_queues.pop(payload.tool_call_id, None)

    return ApprovalDecision(
      approved=bool(approval.get("approved")),
      allow_tool_type=bool(approval.get("allow_tool_type")),
    )

  return request_approval


def create_gateway_app(config: GatewayServerConfig) -> FastAPI:
  if config.build_chat_runtime is None:
    raise ValueError("GatewayServerConfig.build_chat_runtime is required")

  auth = config.auth_manager or AuthManager(
    secret=config.jwt_secret,
    valid_keys=config.valid_api_keys,
    session_store=SessionStore(ttl=config.session_ttl),
  )
  transcript_dir = Path(config.transcript_dir) if config.transcript_dir is not None else None
  if transcript_dir is not None:
    transcript_dir.mkdir(parents=True, exist_ok=True)
  log = logging.getLogger(config.log_name)
  route_prefix = _normalize_prefix(config.prefix)

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_cleanup_sessions_loop(auth.session_store))
    startup_complete = False
    try:
      await _maybe_await(config.on_startup)
      startup_complete = True
      yield
    finally:
      if startup_complete:
        await _maybe_await(config.on_shutdown)
      cleanup_task.cancel()
      await asyncio.gather(cleanup_task, return_exceptions=True)

  app = FastAPI(lifespan=lifespan)
  app.state.auth = auth
  app.state.gateway_config = config

  app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.cors_origins),
    allow_credentials=False,
    allow_methods=list(config.cors_allow_methods),
    allow_headers=list(config.cors_allow_headers),
  )

  router = APIRouter(prefix=route_prefix)

  @router.post("/chat/init", response_model=ChatInitResponse)
  async def chat_init(payload: ChatInitRequest) -> ChatInitResponse:
    auth.validate_api_key(payload.api_key)
    session = auth.session_store.create_session(api_key_hash=AuthManager.hash_api_key(payload.api_key))
    token = auth.issue_token(session)
    log.info("Session created: %s", session.session_id)
    return ChatInitResponse(
      session_token=token,
      session_id=session.session_id,
      expires_at=session.expires_at,
    )

  @router.post("/chat")
  async def chat_stream(request: Request, body: ChatRequest = Body(...)) -> StreamingResponse:
    token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
    session = auth.verify_token(token)
    raw_channel = (body.context or {}).get("channel")
    channel = raw_channel.strip().lower() if isinstance(raw_channel, str) else None

    if session.stream_active:
      raise HTTPException(status_code=409, detail="A chat stream is already active for this session")
    session.stream_active = True

    messages = [_model_to_dict(message) for message in body.messages]
    last_user = next((message["content"] for message in reversed(messages) if message.get("role") == "user"), "")
    sid = session.session_id
    log.info("Chat request | session=%s | msgs=%d | user=%s", sid, len(messages), last_user[:200])
    _write_transcript(
      transcript_dir=transcript_dir,
      session_id=sid,
      entry={"type": "chat_request", "messages": messages, "context": body.context},
    )

    headers = {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "X-Accel-Buffering": "no",
      "Connection": "keep-alive",
    }

    try:
      runtime = await config.build_chat_runtime(
        session=session,
        request=body,
        channel=channel,
        auth_manager=auth,
      )
      resolved_model = runtime.model_override or body.model or str(config.auth_config.get("model", "")).strip() or None
      if resolved_model:
        if config.allowed_models and resolved_model not in config.allowed_models:
          raise HTTPException(status_code=400, detail=f"Invalid model: {resolved_model}")
      event_log = EventLog(on_event=config.on_event, session_id=sid)
      runner = runtime.build_runner(event_log, sid)
    except Exception:
      session.stream_active = False
      raise

    async def run_agent() -> None:
      try:
        await runner.run(
          messages=messages,
          system_prompt=runtime.system_prompt,
          model_override=runtime.model_override,
          max_turns=runtime.max_turns,
        )
      except asyncio.CancelledError:
        raise
      except Exception as exc:
        event_log.append({"type": "error", "error": str(exc)})
      finally:
        event_log.close()

    async def heartbeat() -> None:
      while True:
        await asyncio.sleep(15)
        if event_log.closed:
          return
        event_log.append({"type": "heartbeat", "timestamp": int(time.time())})

    async def event_generator():
      runner_task = asyncio.create_task(run_agent())
      heartbeat_task = asyncio.create_task(heartbeat())
      try:
        async for entry in event_log.iter_from(0):
          if await request.is_disconnected():
            break

          event = dict(entry.event)
          if event.get("type") in {"tool_call_start", "tool_call_complete"}:
            tool_name = event.get("tool_name")
            if isinstance(tool_name, str) and runtime.execution_location is not None:
              execution_location = runtime.execution_location(tool_name)
              if execution_location is not None:
                event["execution_location"] = execution_location

          if event.get("type") != "heartbeat":
            _write_transcript(transcript_dir=transcript_dir, session_id=sid, entry=event)

          try:
            yield f"data: {_json_dumps(event)}\n\n".encode("utf-8")
          except Exception as ser_exc:
            log.error(
              "SSE serialization failed for event type=%s: %s",
              event.get("type"),
              ser_exc,
              exc_info=True,
            )
            try:
              error_event = {"type": "stream_error", "error": f"SSE serialization failed: {ser_exc}"}
              yield f"data: {_json_dumps(error_event)}\n\n".encode("utf-8")
            except Exception:
              pass
            break
      finally:
        runner_task.cancel()
        heartbeat_task.cancel()
        await asyncio.gather(runner_task, heartbeat_task, return_exceptions=True)
        event_log.close("stream closed")
        session.pending_tools.clear()
        session.approval_queues.clear()
        _drain_result_queue(session.result_queue)
        session.stream_active = False

    return StreamingResponse(event_generator(), headers=headers)

  @router.post("/chat/tool-result")
  async def tool_result(request: Request, payload: ToolResultRequest) -> JSONResponse:
    token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
    session = auth.verify_token(token)

    pending = session.pending_tools.get(payload.tool_call_id)
    if not pending:
      return JSONResponse({"error": "Unknown tool_call_id"}, status_code=404)

    if pending.get("status") == "received":
      return JSONResponse({"error": "Result already submitted"}, status_code=409)

    if pending.get("status") != "pending":
      return JSONResponse({"error": "Invalid pending tool state for result submission"}, status_code=409)

    if pending.get("nonce") != payload.nonce:
      return JSONResponse({"error": "Nonce mismatch"}, status_code=409)

    if int(time.time()) > int(pending.get("expires_at", 0)):
      session.pending_tools.pop(payload.tool_call_id, None)
      return JSONResponse({"error": "Tool call expired"}, status_code=410)

    pending["status"] = "received"
    if session.result_queue is None:
      session.result_queue = asyncio.Queue()
    await session.result_queue.put({"result": payload.result, "error": payload.error})

    tool_name = pending.get("tool_name", "?")
    if payload.error:
      log.warning("Tool result: %s | error=%s", tool_name, payload.error)
    else:
      log.info("Tool result: %s | success", tool_name)
    return JSONResponse({"status": "ok"})

  @router.post("/chat/tool-approval")
  async def tool_approval(request: Request, payload: ToolApprovalRequest) -> JSONResponse:
    token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
    session = auth.verify_token(token)

    pending = session.pending_tools.get(payload.tool_call_id)
    if not pending:
      return JSONResponse({"error": "Unknown tool_call_id"}, status_code=404)

    if pending.get("status") == "approval_received":
      return JSONResponse({"error": "Approval already submitted"}, status_code=409)

    if pending.get("status") != "approval_pending":
      return JSONResponse({"error": "Invalid pending tool state for approval submission"}, status_code=409)

    if pending.get("nonce") != payload.nonce:
      return JSONResponse({"error": "Nonce mismatch"}, status_code=409)

    if int(time.time()) > int(pending.get("expires_at", 0)):
      session.pending_tools.pop(payload.tool_call_id, None)
      session.approval_queues.pop(payload.tool_call_id, None)
      return JSONResponse({"error": "Tool approval expired"}, status_code=410)

    approval_queue = session.approval_queues.get(payload.tool_call_id)
    if approval_queue is None:
      return JSONResponse({"error": "Missing approval queue for tool call"}, status_code=404)

    pending["status"] = "approval_received"
    await approval_queue.put(
      {
        "approved": payload.approved,
        "allow_tool_type": payload.allow_tool_type,
      }
    )

    tool_name = pending.get("tool_name", "?")
    log.info(
      "Tool approval: %s | approved=%s allow_tool_type=%s",
      tool_name,
      payload.approved,
      payload.allow_tool_type,
    )
    return JSONResponse({"status": "ok"})

  @router.get("/health")
  async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

  app.include_router(router)
  app.state.gateway_chat_init = chat_init
  app.state.gateway_chat_stream = chat_stream
  app.state.gateway_tool_result = tool_result
  app.state.gateway_tool_approval = tool_approval
  app.state.gateway_health = health
  return app


__all__ = [
  "ChatInitRequest",
  "ChatInitResponse",
  "ChatMessage",
  "ChatRequest",
  "ChatRuntime",
  "GatewayServerConfig",
  "RequestContext",
  "ToolApprovalRequest",
  "ToolResultRequest",
  "create_gateway_app",
]
