from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
import json as json_mod
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .auth import (
  CredentialsResolver,
  CredentialsTimeoutError,
  CrossUserReuseError,
  MissingUserIdError,
  NoCredentialError,
  StrictModeDefaultUserError,
)
from .event_log import EventLog
from .mcp_client import McpClientManager
from .providers.agent_sdk import AgentSDKConfig
from .providers import AnthropicProvider, ModelProvider
from .runner import AgentRunner
from .sdk_runner import AgentSDKRunner
from .session import AuthManager, GatewaySession, SessionStore
from .tool_dispatcher import ApprovalDecision, ApprovalRequest


SystemPrompt = str | List[Tuple[str, bool]]
ExecutionLocationResolver = Callable[[str], Optional[str]]
BuildChatRuntime = Callable[[GatewaySession, "ChatRequest", Optional[str], AuthManager], Awaitable["ChatRuntime"]]
RequestApproval = Callable[[ApprovalRequest], Awaitable[Optional[ApprovalDecision]]]
BuildRunner = Callable[[EventLog, str], AgentRunner | AgentSDKRunner]


class ChatInitRequest(BaseModel):
  """Request body for `POST /chat/init`."""

  api_key: str = Field(..., min_length=1)
  user_id: str | None = None
  context: Dict[str, Any] = Field(default_factory=dict)


class ModelCatalog(BaseModel):
  default_model: str
  allowed_models: List[str]
  display_names: Dict[str, str]


class ChatInitResponse(BaseModel):
  """Response body returned after a session token is issued."""

  session_token: str
  session_id: str
  expires_at: int
  model_catalog: Optional[ModelCatalog] = None


class ChatMessage(BaseModel):
  """Single plain-text chat message sent to the gateway."""

  role: str
  content: str


class ChatRequest(BaseModel):
  """Request body for `POST /chat`.

  `messages` is the full message list for the current turn. `context` is
  intentionally free-form; `context["channel"]` is the most common field used
  to shape runtime behavior per frontend.
  """

  messages: List[ChatMessage]
  user_id: str | None = None
  request_id: str | None = None
  context: Dict[str, Any] = Field(default_factory=dict)
  model: Optional[str] = None


class ToolResultRequest(BaseModel):
  """Submit the result of a client-executed tool call.

  The built-in backend tools do not normally use this path, but the endpoint is
  available for custom runtimes that stage tool execution on the client side.
  """

  tool_call_id: str
  nonce: str
  result: Optional[Dict[str, Any]] = None
  error: Optional[Dict[str, Any]] = None


class ToolApprovalRequest(BaseModel):
  """Approve or deny a pending tool call."""

  tool_call_id: str
  nonce: str
  approved: bool
  allow_tool_type: bool = False


@dataclass
class ChatRuntime:
  """Per-request runtime wiring returned by `build_chat_runtime`.

  Attributes:
    system_prompt: Prompt string or cached prompt blocks for the request.
    build_runner: Factory that receives `(event_log, session_id)` and returns an
      `AgentRunner` or `AgentSDKRunner`.
    get_tool_definitions: Callback returning the tool schemas visible to the
      model for this request.
    build_dispatcher: Reserved callback slot for custom dispatcher builders.
    provider: Provider handling the request, typically Anthropic or OpenAI.
    model_override: Request-scoped model override.
    excluded_tools: Optional tool names hidden from the runtime.
    execution_location: Optional resolver used to annotate tool events with
      metadata such as `backend`, `client`, or a channel-specific location.
    on_usage: Optional request-scoped usage hook.
    on_tool_result: Optional request-scoped tool result hook.
    on_tool_timing: Optional request-scoped tool timing hook.
    disconnect_handler: Optional request-scoped disconnect hook invoked when the
      client connection is closed mid-stream.
    post_runner_init: Optional callback invoked after runner construction.
    max_turns: Optional cap on loop iterations for this request.
    compaction_trigger: Optional token threshold for provider-side compaction.
    compaction_instructions: Optional provider-specific compaction instructions.
  """

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
  disconnect_handler: Optional[Callable[[], Awaitable[None] | None]] = None
  post_runner_init: Optional[Callable[[Any], None]] = None
  max_turns: Optional[int] = None
  compaction_trigger: int | None = None
  compaction_instructions: str | None = None
  _disconnect_called: bool = field(default=False, init=False, repr=False)

  async def on_disconnect(self) -> None:
    if self._disconnect_called:
      return
    self._disconnect_called = True
    if self.disconnect_handler is None:
      return
    try:
      result = self.disconnect_handler()
      if inspect.isawaitable(result):
        await result
    except Exception as exc:
      logging.getLogger("agent_gateway.server").warning("ChatRuntime.on_disconnect failed: %s", exc)


@dataclass
class RequestContext:
  """Mutable request-scoped objects shared across dispatch layers."""

  session: GatewaySession
  event_log: EventLog
  request_approval: RequestApproval
  result_queue: asyncio.Queue
  mcp_client: Optional[McpClientManager]


@dataclass
class GatewayServerConfig:
  """Configuration for `create_gateway_app()`.

  Attributes:
    auth_manager: Optional pre-built `AuthManager`. When omitted, the server
      creates one from `jwt_secret`, `valid_api_keys`, and `session_ttl`.
    default_provider: Informational default provider for the app.
    jwt_secret: JWT signing secret used when `auth_manager` is omitted.
    session_ttl: Session lifetime in seconds.
    valid_api_keys: API keys accepted by `POST /chat/init`.
    cors_origins: Allowed CORS origins.
    cors_allow_headers: Allowed CORS headers.
    cors_allow_methods: Allowed CORS methods.
    auth_config: Default provider auth/model config used by convenience flows and
      by request validation when no runtime override is present.
    mcp_client: Optional shared `McpClientManager`.
    sdk_config: Optional `AgentSDKConfig` when using `AgentSDKRunner`.
    per_turn_timeout: Default per-turn timeout in seconds.
    compaction_trigger: Default compaction threshold.
    compaction_instructions: Default compaction instructions.
    allowed_models: Model allowlist enforced at request time.
    model_catalog: Optional model discovery metadata returned by `POST /chat/init`.
    build_chat_runtime: Required async callback that returns a `ChatRuntime`.
    on_event: Optional event observer invoked for every appended `EventLog`
      entry.
    on_tool_result: Optional app-level tool result hook.
    on_usage: Optional app-level usage hook.
    on_tool_timing: Optional app-level tool timing hook.
    on_startup: Optional startup callback.
    on_shutdown: Optional shutdown callback.
    transcript_dir: Optional directory where request and event transcripts are
      written as JSONL files.
    log_name: Logger name for the server.
    prefix: Route prefix, usually `/api`.
  """

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
  credentials_resolver: CredentialsResolver | None = None
  resolver_timeout_seconds: float = 5.0
  mcp_client: Optional[McpClientManager] = None
  sdk_config: AgentSDKConfig | None = None
  per_turn_timeout: int = 300
  compaction_trigger: int | None = None
  compaction_instructions: str | None = None
  allowed_models: Set[str] = field(default_factory=lambda: {"claude-sonnet-4-6", "claude-opus-4-6"})
  model_catalog: Optional[ModelCatalog] = None
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


def _error_payload(
  exc: Exception,
  *,
  user_id: str | None = None,
  session_id: str | None = None,
  request_user: str | None = None,
  session_user: str | None = None,
  timeout_seconds: float | None = None,
) -> tuple[int, Dict[str, Any]]:
  if isinstance(exc, CredentialsTimeoutError):
    payload: Dict[str, Any] = {
      "error": "credentials_timeout",
      "message": str(exc),
    }
    if user_id is not None:
      payload["user_id"] = user_id
    if timeout_seconds is not None:
      payload["timeout_seconds"] = timeout_seconds
    return 504, payload

  if isinstance(exc, StrictModeDefaultUserError):
    payload = {"error": "strict_mode_default_user", "message": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
    return 400, payload

  if isinstance(exc, MissingUserIdError):
    payload = {"error": "missing_user_id", "message": str(exc)}
    if session_id is not None:
      payload["session_id"] = session_id
    return 400, payload

  if isinstance(exc, CrossUserReuseError):
    payload = {"error": "cross_user_reuse", "message": str(exc)}
    if session_id is not None:
      payload["session_id"] = session_id
    if session_user is not None:
      payload["session_user"] = session_user
    if request_user is not None:
      payload["request_user"] = request_user
    return 401, payload

  if isinstance(exc, NoCredentialError):
    payload = {"error": "credentials_unavailable", "message": str(exc), "reason": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
    return 401, payload

  payload = {"error": "credentials_unavailable", "message": str(exc), "reason": str(exc)}
  if user_id is not None:
    payload["user_id"] = user_id
  return 500, payload


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


def _make_request_approval(session: GatewaySession, event_log: EventLog) -> RequestApproval:
  async def request_approval(payload: ApprovalRequest) -> Optional[ApprovalDecision]:
    approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
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
        "reason": payload.reason,
        "allow_persistent_approval": payload.allow_persistent_approval,
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


def create_gateway_app(config: GatewayServerConfig) -> FastAPI:
  """Create a FastAPI gateway application from explicit runtime configuration.

  The returned app exposes:

  - `POST {prefix}/chat/init`
  - `POST {prefix}/chat`
  - `POST {prefix}/chat/tool-result`
  - `POST {prefix}/chat/tool-approval`
  - `GET  {prefix}/health`

  `create_gateway_app()` is the extensibility point for production deployments.
  Unlike `create_agent()`, it does not assume Anthropic, a particular tool
  policy, or a particular runtime builder. You provide the `build_chat_runtime`
  callback and the server handles the HTTP, session, SSE, and approval loop.

  Args:
    config: `GatewayServerConfig` describing auth, routing, callbacks, and the
      runtime factory.

  Returns:
    A FastAPI application ready to serve the gateway HTTP API.
  """
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
    resolver = config.credentials_resolver
    resolved_user_id = payload.user_id.strip() if isinstance(payload.user_id, str) else payload.user_id
    if resolved_user_id == "":
      resolved_user_id = None
    if resolved_user_id is None:
      context_user_id = (payload.context or {}).get("user_id")
      if isinstance(context_user_id, str) and context_user_id.strip():
        resolved_user_id = context_user_id.strip()
        log.warning("Chat init used deprecated context.user_id fallback; move user_id to the top-level request field.")
    if resolved_user_id is None:
      resolved_user_id = "_default"

    resolved_auth_config: dict[str, Any] | None = None
    if resolver is not None:
      if resolved_user_id == "_default":
        status, error_payload = _error_payload(
          StrictModeDefaultUserError(
            "Gateway is in strict multi-user mode (resolver configured). Request must include a non-default user_id. "
            "Did you forget to thread user_id from the consumer's auth middleware to the gateway request?"
          ),
          user_id=resolved_user_id,
        )
        return JSONResponse(error_payload, status_code=status)
      try:
        auth_config = await asyncio.wait_for(
          resolver(resolved_user_id, payload),
          timeout=config.resolver_timeout_seconds,
        )
      except asyncio.TimeoutError as exc:
        status, error_payload = _error_payload(
          CredentialsTimeoutError(
            f"Credential resolution for user '{resolved_user_id}' timed out after "
            f"{config.resolver_timeout_seconds:.1f}s. Check the resolver latency or raise resolver_timeout_seconds."
          ),
          user_id=resolved_user_id,
          timeout_seconds=config.resolver_timeout_seconds,
        )
        return JSONResponse(error_payload, status_code=status)
      except Exception as exc:
        status, error_payload = _error_payload(exc, user_id=resolved_user_id)
        return JSONResponse(error_payload, status_code=status)
      resolved_auth_config = auth_config.to_dict()

    session = auth.session_store.create_session(
      api_key_hash=AuthManager.hash_api_key(payload.api_key),
      user_id=resolved_user_id,
      auth_config=resolved_auth_config,
    )
    token = auth.issue_token(session)
    log.info("Session created: %s", session.session_id)
    return ChatInitResponse(
      session_token=token,
      session_id=session.session_id,
      expires_at=session.expires_at,
      model_catalog=config.model_catalog,
    )

  @router.post("/chat")
  async def chat_stream(request: Request, body: ChatRequest = Body(...)) -> StreamingResponse:
    token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
    if isinstance(body.user_id, str):
      body.user_id = body.user_id.strip() or None
    if isinstance(body.request_id, str):
      body.request_id = body.request_id.strip() or None
    session, claims = auth.verify_token_with_payload(token)
    jwt_user_id = str(claims.get("user_id") or session.user_id)
    strict_mode = config.credentials_resolver is not None
    if strict_mode and jwt_user_id == "_default":
      status, error_payload = _error_payload(
        StrictModeDefaultUserError(
          "Gateway is in strict multi-user mode (resolver configured). Request must include a non-default user_id. "
          "Did you forget to thread user_id from the consumer's auth middleware to the gateway request?"
        ),
        user_id=jwt_user_id,
      )
      return JSONResponse(error_payload, status_code=status)
    if strict_mode:
      if body.user_id is None:
        status, error_payload = _error_payload(MissingUserIdError(), session_id=session.session_id)
        return JSONResponse(error_payload, status_code=status)
      if body.user_id != jwt_user_id:
        status, error_payload = _error_payload(
          CrossUserReuseError(
            f"Session for user '{jwt_user_id}' cannot be reused for user '{body.user_id}'. "
            "Create a separate gateway session per end user."
          ),
          session_id=session.session_id,
          session_user=jwt_user_id,
          request_user=body.user_id,
        )
        return JSONResponse(error_payload, status_code=status)
    else:
      body.user_id = body.user_id or jwt_user_id
    body.request_id = body.request_id or str(uuid.uuid4())
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
      session_auth_config = session.auth_config or config.auth_config
      resolved_model = runtime.model_override or body.model or str(session_auth_config.get("model", "")).strip() or None
      if resolved_model:
        if config.allowed_models and resolved_model not in config.allowed_models:
          raise HTTPException(status_code=400, detail=f"Invalid model: {resolved_model}")
      event_log = EventLog(on_event=config.on_event, session_id=sid)
      runner = runtime.build_runner(event_log, sid)
      if runtime.disconnect_handler is None:
        runner_on_disconnect = getattr(runner, "on_disconnect", None)
        if callable(runner_on_disconnect):
          runtime.disconnect_handler = runner_on_disconnect
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

    async def _safe_fire_disconnect() -> None:
      try:
        await runtime.on_disconnect()
      except Exception as exc:
        log.warning("on_disconnect failed for %s: %s", sid, exc)

    runner_task = asyncio.create_task(run_agent())
    heartbeat_task = asyncio.create_task(heartbeat())

    async def response_cleanup() -> None:
      disconnect_task = asyncio.create_task(_safe_fire_disconnect())
      runner_task.cancel()
      heartbeat_task.cancel()
      await asyncio.gather(runner_task, heartbeat_task, disconnect_task, return_exceptions=True)
      event_log.close("stream closed")
      session.pending_tools.clear()
      session.approval_queues.clear()
      _drain_result_queue(session.result_queue)
      session.stream_active = False

    async def event_generator():
      async for entry in event_log.iter_from(0):
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

    return StreamingResponse(event_generator(), headers=headers, background=BackgroundTask(response_cleanup))

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
  "ModelCatalog",
  "RequestContext",
  "ToolApprovalRequest",
  "ToolResultRequest",
  "create_gateway_app",
]
