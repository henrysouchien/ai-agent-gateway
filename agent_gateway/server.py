from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import hashlib
import hmac
import inspect
import json as json_mod
import logging
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Set, Tuple

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from ._provider_utils import _get_allowed_models_for_provider_name
from .artifact_paths import (
  ArtifactPath,
  ArtifactPathError,
  artifact_json_path_for_request,
  latest_artifact_json_path_for_request,
  letter_docx_path_for_request,
  reject_unsafe_path,
  ticker_artifact_index_for_request,
)
from .auth import (
  ChannelMismatchError,
  CredentialRefreshRequest,
  CredentialsResolver,
  CredentialsRefreshResolver,
  CredentialsTimeoutError,
  CrossUserReuseError,
  MissingUserIdError,
  NoCredentialError,
  ProviderCredentialFailure,
)
from .autonomous_runner import AutonomousRegistry
from .control_plane import create_control_plane_router
from .control_plane.middleware import add_control_plane_version_header_middleware
from .event_log import EventLog, UserEventBus
from .approval_audit import ApprovalAuditEmitter
from .approvals import ApprovalActionError, _record_vote_and_unblock
from .approval_resolver import resolve_policy
from .approval_store import SQLiteApprovalStore, expire_pending_loop
from .audit_resolver import resolve_audit_writer
from .agent_session_log import _atomic_write_sidecar
from .mcp_client import McpClientManager
from .package_info import package_health
from .product_config import gateway_product_id
from .providers.agent_sdk import AgentSDKConfig
from .providers import AnthropicProvider, ModelProvider, StreamEvent
from .runner import AgentRunner
from .sdk_runner import AgentSDKRunner
from .session import AuthManager, GatewaySession, SessionStore
from .tool_dispatcher import ApprovalDecision, ApprovalRequest
from .tool_redaction import (
  get_audit_hmac_key_id,
  get_audit_hmac_secret,
  redact_tool_input as default_redact_tool_input,
)


SystemPrompt = str | List[Tuple[str, bool]]
ExecutionLocationResolver = Callable[[str], Optional[str]]
BuildChatRuntime = Callable[[GatewaySession, "ChatRequest", Optional[str], AuthManager], Awaitable["ChatRuntime"]]
RequestApproval = Callable[[ApprovalRequest], Awaitable[Optional[ApprovalDecision]]]
BuildRunner = Callable[[EventLog, str], AgentRunner | AgentSDKRunner]

_AGENT_API_CLAIM_AUDIENCE = "agent_api_v1"
_AGENT_API_CLAIM_CLOCK_SKEW_SECONDS = 60
_AGENT_API_CLAIM_NONCE_HEX_LENGTH = 32
_AGENT_API_CLAIM_HEADERS = {
  "audience": "X-Agent-Claim-Audience",
  "issued_at": "X-Agent-Claim-Issued-At",
  "expiry": "X-Agent-Claim-Expiry",
  "user_id": "X-Agent-Claim-User-Id",
  "user_email": "X-Agent-Claim-User-Email",
  "nonce": "X-Agent-Claim-Nonce",
  "signature": "X-Agent-Claim-Signature",
}
_AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT = 600
_ARTIFACT_DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_DEFAULT_CHAT_PROFILE = "analyst"
_CHAT_PROFILE_ALIASES = {"hank": "analyst"}
_SIDECAR_SLUG_RE = re.compile(r"[^a-z0-9-]+")
log = logging.getLogger("agent_gateway.server")


class ChatInitRequest(BaseModel):
  """Request body for `POST /chat/init`."""

  api_key: str = Field(..., min_length=1)
  user_id: str | None = None
  user_email: str | None = None
  context: Dict[str, Any] = Field(default_factory=dict)
  anthropic_auth_mode: Optional[str] = None
  anthropic_api_key: Optional[str] = None
  anthropic_auth_token: Optional[str] = None


class ModelCatalog(BaseModel):
  default_model: str
  allowed_models: List[str]
  display_names: Dict[str, str]


class ChatInitResponse(BaseModel):
  """Response body returned after a session token is issued."""

  user_id: str
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
  metadata: Dict[str, Any] = Field(default_factory=dict)
  model: Optional[str] = None


def _resolve_chat_profile_name(context: Mapping[str, Any] | None) -> str:
  raw_profile = None
  for key in ("profile", "profile_name", "agent_id", "agent", "system_prompt_key"):
    raw_profile = (context or {}).get(key)
    if raw_profile is not None:
      break
  if raw_profile is None:
    return _DEFAULT_CHAT_PROFILE
  if not isinstance(raw_profile, str):
    raise HTTPException(status_code=400, detail="context.profile must be a string when provided")
  profile_name = raw_profile.strip().lower()
  if not profile_name:
    return _DEFAULT_CHAT_PROFILE
  return _CHAT_PROFILE_ALIASES.get(profile_name, profile_name)


@dataclass
class ChatTurnInputs:
  messages: list[ChatMessage]
  request_id: str | None
  context: dict[str, Any] | None
  metadata: dict[str, Any] | None
  model: str | None


@dataclass
class ChatTurnResult:
  session_id: str
  request_id: str | None
  state: str
  events: list[dict[str, Any]]


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
    credentials_resolver: Optional init-time resolver for per-user credentials.
    credentials_refresh_resolver: Optional stream-time resolver used to rotate
      credentials after provider rate-limit, billing, or auth failures.
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
    on_session_created: Optional app-level hook invoked after session creation
      and before the init response is issued.
    on_startup: Optional startup callback.
    on_shutdown: Optional shutdown callback.
    transcript_dir: Optional directory where request and event transcripts are
      written as JSONL files.
    control_skills_dir: Optional directory backing control-plane skill list/read
      endpoints. When omitted, the package uses `AGENT_GATEWAY_SKILLS_DIR` if
      set, otherwise an empty package-local directory.
    audit_hmac_secret_resolver: Callback returning the approval-audit HMAC
      secret bytes. Defaults to agent_gateway's environment-backed resolver.
    audit_hmac_key_id_resolver: Callback returning the approval-audit HMAC key
      id. Defaults to agent_gateway's environment-backed resolver.
    tool_input_redactor: Optional app-specific redactor used before approval
      audit persistence. Package consumers can omit this; arguments are copied
      without redaction when no app policy is provided.
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
    default_factory=lambda: [
      "Authorization",
      "Content-Type",
      *_AGENT_API_CLAIM_HEADERS.values(),
    ]
  )
  cors_allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
  auth_config: Dict[str, Any] = field(default_factory=dict)
  credentials_resolver: CredentialsResolver | None = None
  credentials_refresh_resolver: CredentialsRefreshResolver | None = None
  resolver_timeout_seconds: float = 5.0
  mcp_client: Optional[McpClientManager] = None
  sdk_config: AgentSDKConfig | None = None
  per_turn_timeout: int = 300
  compaction_trigger: int | None = None
  compaction_instructions: str | None = None
  allowed_models: Set[str] | None = None
  model_catalog: Optional[ModelCatalog] = None
  build_chat_runtime: Optional[BuildChatRuntime] = None
  on_event: Optional[Callable[..., Any]] = None
  on_tool_result: Optional[Callable[..., Any]] = None
  on_usage: Optional[Callable[..., Any]] = None
  on_tool_timing: Optional[Callable[..., Any]] = None
  on_session_created: Optional[Callable[[GatewaySession, str, ChatInitRequest], None]] = None
  on_startup: Optional[Callable[..., Any]] = None
  on_shutdown: Optional[Callable[..., Any]] = None
  transcript_dir: Optional[Path] = None
  control_skills_dir: Optional[Path] = None
  audit_hmac_secret_resolver: Callable[[], bytes] = get_audit_hmac_secret
  audit_hmac_key_id_resolver: Callable[[], str] = get_audit_hmac_key_id
  tool_input_redactor: Optional[Callable[..., dict[str, Any]]] = None
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


def _default_control_skills_dir() -> Path:
  configured = os.getenv("AGENT_GATEWAY_SKILLS_DIR", "").strip()
  if configured:
    return Path(configured).expanduser()
  return Path(__file__).resolve().parent / "_no_control_skills"


def _default_autonomous_api_dir() -> Path:
  return Path(__file__).resolve().parents[3] / "api"


def _default_autonomous_log_dir() -> Path | None:
  explicit = os.getenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", "").strip()
  if explicit:
    return Path(explicit).expanduser()
  gateway_log_dir = os.getenv("GATEWAY_LOG_DIR", "").strip()
  if gateway_log_dir:
    return Path(gateway_log_dir).expanduser() / "autonomous"
  legacy_agents_log_dir = os.getenv("AGENTS_MCP_LOG_DIR", "").strip()
  if legacy_agents_log_dir:
    return Path(legacy_agents_log_dir).expanduser()
  return None


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


def _claim_ttl_ceiling_seconds() -> int:
  raw = os.getenv("AGENT_API_CLAIM_MAX_TTL_SECONDS", "").strip()
  if not raw:
    return _AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT
  try:
    value = int(raw)
  except ValueError:
    return _AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT
  return value if value > 0 else _AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT


def _verify_signed_user_claim(request: Request) -> dict[str, Any]:
  claim_headers = _extract_agent_claim_headers(request.headers)
  if claim_headers is None:
    raise HTTPException(status_code=401, detail="Signed user claim required")

  hmac_key = os.getenv("AGENT_API_USER_CLAIM_HMAC_KEY", "").strip()
  if not hmac_key:
    raise HTTPException(
      status_code=503,
      detail="Agent API signed claim verifier not configured (AGENT_API_USER_CLAIM_HMAC_KEY not set)",
    )

  verified = _verify_agent_claim_headers(
    hmac_key,
    claim_headers,
    ttl_ceiling=_claim_ttl_ceiling_seconds(),
  )
  if verified is None:
    raise HTTPException(status_code=401, detail="Invalid signed user claim")
  return verified


def _artifact_auth_dependency(request: Request) -> str:
  authorization = request.headers.get("Authorization")
  if authorization is not None:
    token = AuthManager.get_bearer_token(authorization)
    auth_manager = getattr(request.app.state, "auth", None)
    if auth_manager is None:
      raise HTTPException(status_code=503, detail="Gateway auth manager unavailable")
    session, _claims = auth_manager.verify_token_with_payload(token)
    return session.user_id

  claim = _verify_signed_user_claim(request)
  return str(claim["user_id"])


def _extract_agent_claim_headers(headers: Mapping[str, Any]) -> dict[str, str] | None:
  claim_headers: dict[str, str] = {}
  for field_name, header_name in _AGENT_API_CLAIM_HEADERS.items():
    value = headers.get(header_name)
    if value is None:
      return None
    claim_headers[field_name] = str(value)
  return claim_headers


def _verify_agent_claim_headers(
  hmac_key: str,
  claim_headers: Mapping[str, str],
  *,
  ttl_ceiling: int,
  now: int | None = None,
) -> dict[str, Any] | None:
  if claim_headers.get("audience") != _AGENT_API_CLAIM_AUDIENCE:
    return None
  try:
    issued_at = int(claim_headers.get("issued_at", ""))
    expiry = int(claim_headers.get("expiry", ""))
  except (TypeError, ValueError):
    return None

  current_time = int(time.time()) if now is None else int(now)
  if issued_at > current_time + _AGENT_API_CLAIM_CLOCK_SKEW_SECONDS:
    return None
  if current_time > expiry:
    return None
  if expiry - issued_at > ttl_ceiling:
    return None

  user_id = str(claim_headers.get("user_id") or "")
  user_email = str(claim_headers.get("user_email") or "")
  nonce = str(claim_headers.get("nonce") or "")
  signature = str(claim_headers.get("signature") or "")
  if not user_id or not user_email:
    return None
  if len(nonce) != _AGENT_API_CLAIM_NONCE_HEX_LENGTH:
    return None
  try:
    bytes.fromhex(nonce)
  except ValueError:
    return None

  canonical = f"{_AGENT_API_CLAIM_AUDIENCE}\n{issued_at}\n{expiry}\n{user_id}\n{user_email}\n{nonce}".encode("utf-8")
  expected = hmac.new(hmac_key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
  if not hmac.compare_digest(expected, signature):
    return None
  return {
    **dict(claim_headers),
    "issued_at": issued_at,
    "expiry": expiry,
    "user_id": user_id,
    "user_email": user_email,
  }


def _artifact_json_response(artifact: ArtifactPath) -> JSONResponse:
  path = _assert_artifact_path_still_safe(artifact)
  if not path.is_file():
    raise HTTPException(status_code=404, detail="Artifact not found")
  with path.open("r", encoding="utf-8") as handle:
    payload = json_mod.load(handle)
  return JSONResponse(content=payload, headers=_file_cache_headers(path))


def _assert_artifact_path_still_safe(artifact: ArtifactPath) -> Path:
  try:
    resolved = artifact.path.resolve()
    resolved.relative_to(artifact.workspace_root.resolve())
  except ValueError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
  return resolved


def _file_cache_headers(path: Path) -> dict[str, str]:
  stat = path.stat()
  return {
    "Cache-Control": "private, max-age=0",
    "ETag": f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"',
  }


def _letter_filename(ticker: str, artifact_id: str) -> str:
  date = artifact_id[:10] if len(artifact_id) >= 10 else artifact_id
  return f"LP-letter-{ticker}-{date}.docx"


def _normalize_request_user_id(user_id: str | None) -> str | None:
  normalized = user_id.strip() if isinstance(user_id, str) else user_id
  if normalized == "":
    return None
  if normalized == "_default":
    raise MissingUserIdError("user_id '_default' is reserved; supply a stable end-user id.")
  return normalized


def _resolver_contract_payload(message: str, *, user_id: str | None = None) -> tuple[int, Dict[str, Any]]:
  payload: Dict[str, Any] = {"error": "credential_resolver_invalid", "message": message}
  if user_id is not None:
    payload["user_id"] = user_id
  return 400, payload


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

  if isinstance(exc, MissingUserIdError):
    payload = {"error": "missing_user_id", "message": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
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

  if isinstance(exc, ChannelMismatchError):
    payload = {"error": "channel_mismatch", "message": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
    return 400, payload

  if isinstance(exc, HTTPException):
    payload = {
      "error": "auth_failed",
      "message": str(exc.detail) if exc.detail is not None else "Authentication failed",
    }
    if user_id is not None:
      payload["user_id"] = user_id
    return exc.status_code, payload

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


def _now_iso() -> str:
  return datetime.now(UTC).isoformat()


def _sidecar_slug(value: str | None) -> str | None:
  if value is None:
    return None
  slug = _SIDECAR_SLUG_RE.sub("-", str(value).strip().lower().replace("_", "-")).strip("-")[:64]
  return slug or None


def _maybe_write_chat_log_meta(
  transcript_dir: Path,
  session_id: str,
  *,
  user_id: str | None,
  channel: str | None,
) -> None:
  meta_path = transcript_dir / f"{session_id}.meta.json"
  if meta_path.exists():
    return
  _atomic_write_sidecar(
    meta_path,
    {
      "schema_version": 1,
      "agent_session_id": session_id,
      "agent_id": None,
      "user_id": user_id,
      "product_id": gateway_product_id() or None,
      "file_kind": None,
      "channel": _sidecar_slug(channel),
      "profile": None,
      "created_at": _now_iso(),
    },
  )


def _write_transcript(
  transcript_dir: Optional[Path],
  session_id: str,
  entry: Dict[str, Any],
  *,
  user_id: str | None = None,
  channel: str | None = None,
) -> None:
  if transcript_dir is None:
    return
  try:
    _maybe_write_chat_log_meta(transcript_dir, session_id, user_id=user_id, channel=channel)
  except Exception:
    log.warning("Chat log sidecar write failed for %s (telemetry-only)", session_id, exc_info=True)
  payload = dict(entry)
  payload["ts"] = time.time()
  path = transcript_dir / f"{session_id}.jsonl"
  try:
    with open(path, "a", encoding="utf-8") as handle:
      handle.write(json_mod.dumps(payload, default=str) + "\n")
  except Exception:
    pass


def _redact_tool_input_for_event(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
  try:
    return default_redact_tool_input(
      tool_name,
      tool_input,
      deployment_secret=get_audit_hmac_secret(),
    )
  except Exception:
    return dict(tool_input)


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


def _legacy_request_approval(session: GatewaySession, event_log: EventLog) -> RequestApproval:
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
        "tool_input": _redact_tool_input_for_event(payload.tool_name, payload.tool_input),
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


def _make_request_approval(
  session: GatewaySession,
  event_log: EventLog,
  *,
  store: Any | None = None,
  policy: Any | None = None,
) -> RequestApproval:
  resolved_store = store or getattr(session, "approval_store", None)
  resolved_policy = policy or getattr(session, "approval_policy", None)
  if resolved_store is None or resolved_policy is None:
    return _legacy_request_approval(session, event_log)
  # Dispatcher-owned lifecycle is used when ToolDispatcher receives the same
  # store/policy bundle. Keep this callback as the legacy pending-tools
  # transport for older callers that have not been constructor-injected yet.
  return _legacy_request_approval(session, event_log)


def _chat_turn_state_from_events(events: list[dict[str, Any]]) -> str:
  for event in reversed(events):
    event_type = event.get("type")
    if event_type == "error":
      return "failed"
    if event_type == "stream_complete":
      return "completed"
  return "completed" if events else "starting"


async def _dispatch_chat_turn(
  session: GatewaySession,
  inputs: ChatTurnInputs,
  *,
  event_log: EventLog,
  on_event: Callable[[StreamEvent], Awaitable[None]],
  build_chat_runtime: BuildChatRuntime,
  credentials_resolver: CredentialsResolver | None,
  transcript_dir: Path | None,
) -> ChatTurnResult:
  """Run one chat turn outside the ASGI response lifecycle."""
  if session.kind != "chat":
    raise HTTPException(status_code=400, detail="control sessions cannot dispatch chat turns")
  if session.stream_active:
    raise HTTPException(status_code=409, detail="A chat stream is already active for this session")

  session.stream_active = True
  sid = session.session_id
  request_id = inputs.request_id.strip() if isinstance(inputs.request_id, str) else inputs.request_id
  request_id = request_id or str(uuid.uuid4())
  context = dict(inputs.context or {})
  context["profile"] = _resolve_chat_profile_name(context)
  request = ChatRequest(
    messages=list(inputs.messages),
    user_id=session.user_id,
    request_id=request_id,
    context=context,
    metadata=dict(inputs.metadata or {}),
    model=inputs.model,
  )

  raw_channel = context.get("channel")
  claimed_channel = raw_channel.strip().lower() if isinstance(raw_channel, str) else None
  channel = session.channel or claimed_channel
  log = getattr(build_chat_runtime, "_gateway_log", logging.getLogger("agent_gateway.server"))
  resolver_timeout_seconds = float(getattr(build_chat_runtime, "_gateway_resolver_timeout_seconds", 5.0))
  config_auth_config = getattr(build_chat_runtime, "_gateway_auth_config", {})
  allowed_models = getattr(build_chat_runtime, "_gateway_allowed_models", None)
  auth_manager = getattr(build_chat_runtime, "_gateway_auth_manager", None)

  previous_on_event = getattr(event_log, "_on_event", None)
  previous_session_id = getattr(event_log, "_session_id", "")
  fanout_stop = object()
  fanout_queue: asyncio.Queue[Any] = asyncio.Queue()

  async def _fanout_worker() -> None:
    while True:
      event = await fanout_queue.get()
      if event is fanout_stop:
        return
      try:
        await on_event(dict(event))
      except Exception as exc:
        log.warning("chat turn event fan-out failed for %s: %s", sid, exc)

  def _record_event(event: Dict[str, Any], event_session_id: str) -> None:
    event_copy = dict(event)
    session.event_history.append(event_copy)
    fanout_queue.put_nowait(event_copy)
    if previous_on_event is not None:
      try:
        previous_on_event(event, event_session_id)
      except Exception:
        pass

  async def _stop_fanout_worker() -> None:
    fanout_queue.put_nowait(fanout_stop)
    await asyncio.gather(fanout_task, return_exceptions=True)

  setattr(event_log, "_on_event", _record_event)
  setattr(event_log, "_session_id", sid)
  fanout_task = asyncio.create_task(_fanout_worker())

  runtime: ChatRuntime | None = None
  runner: Any | None = None
  runner_task: asyncio.Task[Any] | None = None
  heartbeat_task: asyncio.Task[Any] | None = None

  async def _credential_refresher(failure: ProviderCredentialFailure) -> Dict[str, Any] | None:
    resolver = credentials_resolver
    if resolver is None:
      return None
    session_auth_config = session.auth_config or config_auth_config
    refresh_request = CredentialRefreshRequest(
      user_id=session.user_id,
      user_email=session.user_email,
      session_id=session.session_id,
      api_key_hash=session.api_key_hash,
      channel=channel,
      provider=failure.provider,
      billing_mode=str(session_auth_config.get("billing_mode", "") or "") or None,
      model=str(session_auth_config.get("model", "") or "") or None,
      auth_mode=str(session_auth_config.get("auth_mode", "") or "") or None,
      request_id=request_id,
      failure=failure,
    )
    try:
      auth_config = await asyncio.wait_for(
        resolver(refresh_request),  # type: ignore[arg-type]
        timeout=resolver_timeout_seconds,
      )
    except Exception as exc:
      log.warning(
        "Credential refresh failed | session=%s user=%s provider=%s kind=%s detail=%s",
        session.session_id,
        session.user_id,
        failure.provider,
        failure.kind,
        exc,
      )
      return None
    if auth_config is None:
      log.info(
        "Credential refresh unavailable | session=%s user=%s provider=%s kind=%s",
        session.session_id,
        session.user_id,
        failure.provider,
        failure.kind,
      )
      return None
    if auth_config.provider != failure.provider:
      log.warning(
        "Credential refresh returned provider=%s for provider=%s; ignoring",
        auth_config.provider,
        failure.provider,
      )
      return None
    refreshed_auth_config = auth_config.to_dict()
    session.auth_config = refreshed_auth_config
    return dict(refreshed_auth_config)

  async def _safe_fire_disconnect() -> None:
    if runtime is None:
      return
    try:
      await runtime.on_disconnect()
    except Exception as exc:
      log.warning("on_disconnect failed for %s: %s", sid, exc)

  try:
    messages = [_model_to_dict(message) for message in inputs.messages]
    last_user = next((message["content"] for message in reversed(messages) if message.get("role") == "user"), "")
    if not session.initial_message and last_user:
      session.initial_message = str(last_user)
    log.info("Chat request | session=%s | msgs=%d | user=%s", sid, len(messages), last_user[:200])
    event = {"type": "chat_request", "messages": messages, "context": context}
    pid = gateway_product_id()
    if pid is not None:
      event["product_id"] = pid
    _write_transcript(
      transcript_dir=transcript_dir,
      session_id=sid,
      entry=event,
      user_id=session.user_id,
      channel=channel,
    )

    runtime = await build_chat_runtime(
      session=session,
      request=request,
      channel=channel,
      auth_manager=auth_manager,
    )
    session_auth_config = session.auth_config or config_auth_config
    resolved_model = runtime.model_override or request.model or str(session_auth_config.get("model", "")).strip() or None
    if resolved_model:
      if allowed_models and resolved_model not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Invalid model: {resolved_model}")
    runner = runtime.build_runner(event_log, sid)
    setattr(event_log, "_gateway_execution_location", runtime.execution_location)
    set_credential_refresher = getattr(runner, "set_credential_refresher", None)
    if callable(set_credential_refresher) and credentials_resolver is not None:
      set_credential_refresher(_credential_refresher)
    if runtime.disconnect_handler is None:
      runner_on_disconnect = getattr(runner, "on_disconnect", None)
      if callable(runner_on_disconnect):
        runtime.disconnect_handler = runner_on_disconnect

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

    runner_task = asyncio.create_task(run_agent())
    heartbeat_task = asyncio.create_task(heartbeat())
    # shield: a client-disconnect cancel of the enclosing dispatch_task must NOT
    # auto-propagate into runner_task here. The `except asyncio.CancelledError` block
    # below fires the cooperative disconnect (sets the tool abort_event and yields)
    # BEFORE cancelling runner_task, so an in-flight tool call gets the cooperative
    # abort handshake. Plain `await runner_task` cancels runner_task first, pre-empting
    # that handshake — the PR4 (47b91a31) regression this restores.
    await asyncio.shield(runner_task)
    session.stream_active = False
  except asyncio.CancelledError:
    await _safe_fire_disconnect()
    if runner_task is not None:
      runner_task.cancel()
    if heartbeat_task is not None:
      heartbeat_task.cancel()
    await asyncio.gather(
      *(task for task in (runner_task, heartbeat_task) if task is not None),
      return_exceptions=True,
    )
    event_log.close("stream closed")
    raise
  finally:
    if heartbeat_task is not None:
      heartbeat_task.cancel()
      await asyncio.gather(heartbeat_task, return_exceptions=True)
    event_log.close()
    await _stop_fanout_worker()
    setattr(event_log, "_on_event", previous_on_event)
    setattr(event_log, "_session_id", previous_session_id)
    session.pending_tools.clear()
    session.approval_queues.clear()
    _drain_result_queue(session.result_queue)
    session.stream_active = False

  events = [dict(entry.event) for entry in event_log.entries]
  return ChatTurnResult(
    session_id=sid,
    request_id=request_id,
    state=_chat_turn_state_from_events(events),
    events=events,
  )


def _init_approval_subsystem(app: FastAPI, config: GatewayServerConfig) -> None:
  audit_writer = resolve_audit_writer()
  audit_emitter = ApprovalAuditEmitter(
    writer=audit_writer,
    deployment_secret=config.audit_hmac_secret_resolver(),
    key_id=config.audit_hmac_key_id_resolver(),
    tool_input_redactor=config.tool_input_redactor,
  )
  store = SQLiteApprovalStore(audit_emitter=audit_emitter)
  policy = resolve_policy(store=store)
  app.state.gateway_approval_audit_writer = audit_writer
  app.state.gateway_approval_audit_emitter = audit_emitter
  app.state.gateway_approval_store = store
  app.state.gateway_approval_policy = policy


def create_gateway_app(config: GatewayServerConfig) -> FastAPI:
  """Create a FastAPI gateway application from explicit runtime configuration.

  The returned app exposes:

  - `POST {prefix}/chat/init`
  - `POST {prefix}/chat`
  - `POST {prefix}/chat/tool-result`
  - `POST {prefix}/chat/tool-approval`
  - `POST {prefix}/control/session`
  - `GET  {prefix}/control/health`
  - `GET  {prefix}/control/runs`
  - `GET  {prefix}/control/runs/{run_id}`
  - `GET  {prefix}/control/runs/{run_id}/logs`
  - `GET  {prefix}/control/schedules`
  - `GET  {prefix}/control/schedules/{name}`
  - `GET  {prefix}/control/schedules/{name}/logs`
  - `POST {prefix}/control/schedules`
  - `PUT  {prefix}/control/schedules/{name}/enabled`
  - `DELETE {prefix}/control/schedules/{name}`
  - `GET  {prefix}/control/skills`
  - `GET  {prefix}/control/skills/{name}`
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
  if config.allowed_models is None:
    provider_name = str(getattr(config.default_provider, "name", "anthropic") or "anthropic")
    config.allowed_models = _get_allowed_models_for_provider_name(provider_name)

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
    approval_expire_task = asyncio.create_task(expire_pending_loop(app.state.gateway_approval_store))
    startup_complete = False
    try:
      await _maybe_await(config.on_startup)
      startup_complete = True
      yield
    finally:
      if startup_complete:
        await _maybe_await(config.on_shutdown)
      subprocess_registry = getattr(app.state, "subprocess_registry", None)
      if subprocess_registry is not None:
        await subprocess_registry.shutdown()
      user_event_bus = getattr(app.state, "user_event_bus", None)
      if user_event_bus is not None:
        await user_event_bus.shutdown()
      cleanup_task.cancel()
      approval_expire_task.cancel()
      await asyncio.gather(cleanup_task, approval_expire_task, return_exceptions=True)

  app = FastAPI(lifespan=lifespan)
  app.state.auth = auth
  app.state.gateway_config = config

  async def _build_chat_runtime_for_dispatch(
    *,
    session: GatewaySession,
    request: ChatRequest,
    channel: str | None,
    auth_manager: AuthManager | None,
  ) -> ChatRuntime:
    _ = auth_manager
    return await config.build_chat_runtime(
      session=session,
      request=request,
      channel=channel,
      auth_manager=auth,
    )

  setattr(_build_chat_runtime_for_dispatch, "_gateway_auth_manager", auth)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_auth_config", config.auth_config)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_allowed_models", config.allowed_models)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_log", log)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_resolver_timeout_seconds", config.resolver_timeout_seconds)
  app.state.gateway_build_chat_runtime = _build_chat_runtime_for_dispatch
  app.state.user_event_bus = UserEventBus()
  app.state.subprocess_registry = AutonomousRegistry(
    api_dir=_default_autonomous_api_dir(),
    python_executable=os.getenv("AGENT_GATEWAY_AUTONOMOUS_PYTHON", "").strip() or sys.executable,
    log_dir=_default_autonomous_log_dir(),
    max_running=int(os.getenv("AGENT_GATEWAY_AUTONOMOUS_MAX_RUNNING", "2") or "2"),
  )
  _init_approval_subsystem(app, config)
  control_prefix = _route_path(route_prefix, "/control")
  add_control_plane_version_header_middleware(
    app,
    path_prefixes={control_prefix, "/control"},
  )

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
    try:
      resolved_user_id = _normalize_request_user_id(payload.user_id)
    except MissingUserIdError as exc:
      status, error_payload = _error_payload(exc)
      return JSONResponse(error_payload, status_code=status)
    resolved_user_email = (
      payload.user_email.strip() if isinstance(payload.user_email, str) else payload.user_email
    )
    if resolved_user_email == "":
      resolved_user_email = None

    resolved_auth_config: dict[str, Any] | None = None
    resolved_channel: str | None = None
    resolved_risk_user_id: int = 0
    resolved_role: str = "owner"
    if resolver is not None:
      try:
        result = await asyncio.wait_for(
          resolver(payload.api_key, payload),
          timeout=config.resolver_timeout_seconds,
        )
      except asyncio.TimeoutError as exc:
        status, error_payload = _error_payload(
          CredentialsTimeoutError(
            f"Credential resolution for user '{resolved_user_id or 'unresolved'}' timed out after "
            f"{config.resolver_timeout_seconds:.1f}s. Check the resolver latency or raise resolver_timeout_seconds."
          ),
          user_id=resolved_user_id,
          timeout_seconds=config.resolver_timeout_seconds,
        )
        return JSONResponse(error_payload, status_code=status)
      except Exception as exc:
        status, error_payload = _error_payload(exc, user_id=resolved_user_id)
        return JSONResponse(error_payload, status_code=status)
      try:
        resolved_user_id = _normalize_request_user_id(result.user_id)
      except MissingUserIdError as exc:
        status, error_payload = _error_payload(exc)
        return JSONResponse(error_payload, status_code=status)
      if resolved_user_id is None:
        status, error_payload = _error_payload(
          MissingUserIdError("credential resolver must return user_id.")
        )
        return JSONResponse(error_payload, status_code=status)
      resolved_channel = result.channel
      resolved_user_email = result.user_email or resolved_user_email
      try:
        resolved_risk_user_id = int(result.risk_user_id) if result.risk_user_id is not None else 0
      except (TypeError, ValueError):
        resolved_risk_user_id = 0
      if resolved_risk_user_id <= 0:
        status, error_payload = _resolver_contract_payload(
          "credential resolver must return a positive risk_user_id.",
          user_id=resolved_user_id,
        )
        return JSONResponse(error_payload, status_code=status)
      if result.role not in {"owner", "invite"}:
        status, error_payload = _resolver_contract_payload(
          "credential resolver must return role='owner' or role='invite'.",
          user_id=resolved_user_id,
        )
        return JSONResponse(error_payload, status_code=status)
      resolved_role = result.role
      resolved_auth_config = result.auth_config.to_dict()
      claimed_init_channel: str | None = None
      if isinstance(payload.context, dict):
        raw_claim = payload.context.get("channel")
        if isinstance(raw_claim, str) and raw_claim.strip():
          claimed_init_channel = raw_claim.strip().lower()
      if (
        claimed_init_channel is not None
        and resolved_channel is not None
        and claimed_init_channel != resolved_channel
      ):
        status, error_payload = _error_payload(
          ChannelMismatchError(
            f"context.channel={claimed_init_channel!r} does not match key channel={resolved_channel!r}. "
            "The API key is bound to a specific channel; the request must claim the same channel."
          ),
          user_id=resolved_user_id,
        )
        return JSONResponse(error_payload, status_code=status)
    elif resolved_user_id is None:
      status, error_payload = _error_payload(
        MissingUserIdError("user_id is required in /chat/init when no credentials resolver is configured.")
      )
      return JSONResponse(error_payload, status_code=status)

    session = auth.session_store.create_session(
      api_key_hash=AuthManager.hash_api_key(payload.api_key),
      user_id=resolved_user_id,
      user_email=resolved_user_email,
      risk_user_id=resolved_risk_user_id,
      role=resolved_role,  # type: ignore[arg-type]
      kind="chat",
      auth_config=resolved_auth_config,
    )
    session.channel = resolved_channel
    session.is_public = resolved_channel == "public"
    session.approval_store = app.state.gateway_approval_store
    session.approval_policy = app.state.gateway_approval_policy
    if config.on_session_created is not None:
      try:
        config.on_session_created(session, payload.api_key, payload)
      except Exception:
        auth.session_store.expire_session(session.session_id)
        raise
    token = auth.issue_token(session)
    log.info("Session created: %s", session.session_id)
    return ChatInitResponse(
      user_id=resolved_user_id,
      session_token=token,
      session_id=session.session_id,
      expires_at=session.expires_at,
      model_catalog=config.model_catalog,
    )

  @router.post("/chat")
  async def chat_stream(request: Request, body: ChatRequest = Body(...)) -> StreamingResponse:
    token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
    if isinstance(body.user_id, str):
      try:
        body.user_id = _normalize_request_user_id(body.user_id)
      except MissingUserIdError as exc:
        status, error_payload = _error_payload(exc)
        return JSONResponse(error_payload, status_code=status)
    if isinstance(body.request_id, str):
      body.request_id = body.request_id.strip() or None
    session, claims = auth.verify_token_with_payload(token)
    jwt_user_id = str(claims.get("user_id") or session.user_id).strip()
    if not jwt_user_id:
      status, error_payload = _error_payload(MissingUserIdError(), session_id=session.session_id)
      return JSONResponse(error_payload, status_code=status)
    strict_mode = config.credentials_resolver is not None
    if body.user_id is not None and body.user_id != jwt_user_id:
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
    if strict_mode:
      if body.user_id is None:
        status, error_payload = _error_payload(MissingUserIdError(), session_id=session.session_id)
        return JSONResponse(error_payload, status_code=status)
    body.user_id = body.user_id or jwt_user_id
    body.request_id = body.request_id or str(uuid.uuid4())
    if session.kind != "chat":
      return JSONResponse(
        {"error": "invalid_session_kind", "message": "control sessions cannot dispatch chat turns"},
        status_code=400,
      )
    raw_channel = (body.context or {}).get("channel")
    claimed_channel = raw_channel.strip().lower() if isinstance(raw_channel, str) else None
    channel = session.channel
    if channel is None:
      channel = claimed_channel
    elif claimed_channel and claimed_channel != channel:
      log.warning("Client claimed channel=%s; session bound to %s; using session", claimed_channel, channel)
    sid = session.session_id

    headers = {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "X-Accel-Buffering": "no",
      "Connection": "keep-alive",
    }
    user_event_bus = app.state.user_event_bus

    async def _on_chat_event(event: StreamEvent) -> None:
      event_dict = dict(event)  # type: ignore[arg-type]
      try:
        await user_event_bus.publish(
          user_id=session.user_id,
          control_run_id=sid,
          event=event_dict,
        )
      except Exception as exc:
        log.warning("UserEventBus publish failed for %s: %s", sid, exc)
      if config.on_event is not None:
        try:
          config.on_event(event_dict, sid)
        except Exception:
          pass

    event_log = EventLog(session_id=sid)
    inputs = ChatTurnInputs(
      messages=list(body.messages),
      request_id=body.request_id,
      context=dict(body.context or {}),
      metadata=dict(body.metadata or {}),
      model=body.model,
    )
    dispatch_task = asyncio.create_task(
      _dispatch_chat_turn(
        session,
        inputs,
        event_log=event_log,
        on_event=_on_chat_event,
        build_chat_runtime=app.state.gateway_build_chat_runtime,
        credentials_resolver=config.credentials_refresh_resolver,  # type: ignore[arg-type]
        transcript_dir=transcript_dir,
      )
    )
    await asyncio.sleep(0)
    if dispatch_task.done():
      exc = dispatch_task.exception()
      if exc is not None:
        raise exc

    async def response_cleanup() -> None:
      dispatch_task.cancel()
      await asyncio.gather(dispatch_task, return_exceptions=True)
      await user_event_bus.cleanup_run(session.user_id, sid)

    async def event_generator():
      async for entry in event_log.iter_from(0):
        event = dict(entry.event)
        pid = gateway_product_id()
        if pid is not None:
          event["product_id"] = pid
        if event.get("type") in {"tool_call_start", "tool_call_complete"}:
          tool_name = event.get("tool_name")
          execution_location_resolver = getattr(event_log, "_gateway_execution_location", None)
          if isinstance(tool_name, str) and execution_location_resolver is not None:
            execution_location = execution_location_resolver(tool_name)
            if execution_location is not None:
              event["execution_location"] = execution_location

        if event.get("type") != "heartbeat":
          _write_transcript(
            transcript_dir=transcript_dir,
            session_id=sid,
            entry=event,
            user_id=session.user_id,
            channel=channel,
          )

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

  @router.get("/artifacts/{ticker}/{skill}/latest")
  async def artifact_latest(request: Request, ticker: str, skill: str) -> JSONResponse:
    user_id = _artifact_auth_dependency(request)
    try:
      artifact = latest_artifact_json_path_for_request(
        user_id,
        ticker=ticker,
        skill=skill,
      )
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    if artifact is None:
      raise HTTPException(status_code=404, detail="Artifact not found")
    return _artifact_json_response(artifact)

  @router.get("/artifacts/{ticker}/{skill}/{artifact_id}")
  async def artifact_by_id(
    request: Request,
    ticker: str,
    skill: str,
    artifact_id: str,
  ) -> JSONResponse:
    user_id = _artifact_auth_dependency(request)
    try:
      artifact = artifact_json_path_for_request(
        user_id,
        ticker=ticker,
        skill=skill,
        artifact_id=artifact_id,
      )
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    return _artifact_json_response(artifact)

  @router.get("/artifacts/{ticker}")
  async def artifact_index(request: Request, ticker: str) -> JSONResponse:
    user_id = _artifact_auth_dependency(request)
    try:
      index = ticker_artifact_index_for_request(user_id, ticker=ticker)
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    return JSONResponse(content=index)

  @router.get("/letters/{ticker}/{artifact_id}")
  async def letter_by_id(request: Request, ticker: str, artifact_id: str) -> FileResponse:
    user_id = _artifact_auth_dependency(request)
    try:
      artifact = letter_docx_path_for_request(
        user_id,
        ticker=ticker,
        artifact_id=artifact_id,
      )
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc

    path = _assert_artifact_path_still_safe(artifact)
    if not path.is_file():
      raise HTTPException(status_code=404, detail="Letter artifact not found")
    headers = _file_cache_headers(path)
    headers["Content-Disposition"] = (
      f'attachment; filename="{_letter_filename(artifact.ticker, artifact.artifact_id or artifact_id)}"'
    )
    return FileResponse(path, media_type=_ARTIFACT_DOCX_MEDIA_TYPE, headers=headers)

  @router.get("/artifacts/{artifact_path:path}")
  async def artifact_path_guard(request: Request, artifact_path: str) -> JSONResponse:
    _artifact_auth_dependency(request)
    try:
      reject_unsafe_path(artifact_path)
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    raise HTTPException(status_code=404, detail="Artifact not found")

  @router.get("/letters/{letter_path:path}")
  async def letter_path_guard(request: Request, letter_path: str) -> JSONResponse:
    _artifact_auth_dependency(request)
    try:
      reject_unsafe_path(letter_path)
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    raise HTTPException(status_code=404, detail="Letter artifact not found")

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

    try:
      result = await _record_vote_and_unblock(
        target_session=session,
        pending_entry=pending,
        tool_call_id=payload.tool_call_id,
        nonce=payload.nonce,
        decider_id=session.user_id,
        decider_role=getattr(session, "role", None),
        approved=payload.approved,
        allow_tool_type=payload.allow_tool_type,
        reason=None,
        app_state=request.app.state,
      )
    except ApprovalActionError as exc:
      return JSONResponse(exc.payload, status_code=exc.status_code)
    return JSONResponse(result)

  @router.get("/health")
  async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "package": package_health()})

  app.include_router(router)
  app.include_router(
    create_control_plane_router(
      auth=auth,
      credentials_resolver=config.credentials_resolver,
      resolver_timeout_seconds=config.resolver_timeout_seconds,
      route_prefix=route_prefix,
      skills_dir=config.control_skills_dir or _default_control_skills_dir(),
      artifact_auth_dependency=_artifact_auth_dependency,
      autonomous_registry=app.state.subprocess_registry,
      approval_store=app.state.gateway_approval_store,
      approval_policy=app.state.gateway_approval_policy,
    ),
    prefix=control_prefix,
  )
  app.state.gateway_chat_init = chat_init
  app.state.gateway_chat_stream = chat_stream
  app.state.gateway_control_session = next(
    route.endpoint for route in app.routes if getattr(route, "path", None) == f"{control_prefix}/session"
  )
  app.state.gateway_control_health = next(
    route.endpoint for route in app.routes if getattr(route, "path", None) == f"{control_prefix}/health"
  )
  app.state.gateway_artifact_latest = artifact_latest
  app.state.gateway_artifact_by_id = artifact_by_id
  app.state.gateway_artifact_index = artifact_index
  app.state.gateway_letter_by_id = letter_by_id
  app.state.gateway_tool_result = tool_result
  app.state.gateway_tool_approval = tool_approval
  app.state.gateway_health = health
  return app


__all__ = [
  "ChatInitRequest",
  "ChatInitResponse",
  "ChatMessage",
  "ChatRequest",
  "ChatTurnInputs",
  "ChatTurnResult",
  "ChatRuntime",
  "GatewayServerConfig",
  "ModelCatalog",
  "RequestContext",
  "ToolApprovalRequest",
  "ToolResultRequest",
  "create_gateway_app",
]
