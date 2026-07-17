from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import logging
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
  TYPE_CHECKING,
  Any,
  Awaitable,
  Callable,
  Dict,
  List,
  Mapping,
  Optional,
  Protocol,
  Set,
  Tuple,
)

from fastapi import HTTPException
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic_core import PydanticCustomError

from .auth import CredentialsRefreshResolver, CredentialsResolver
from .thinking import parse_effort
from .commercial_work_start import (
  COMMERCIAL_CLAIM_HEADER,
  COMMERCIAL_WORK_AUTHORIZATION_HEADER,
)
from .commercial_claims import COMMERCIAL_CLAIM_ISSUER
from .commercial_work_authorization import WORK_AUTHORIZATION_ISSUER
from .event_log import EventLog
from .mcp_client import McpClientManager
from .providers import ModelProvider
from .providers.agent_sdk import AgentSDKConfig
from .runner import AgentRunner
from .sdk_runner import AgentSDKRunner
from .session import AuthManager, GatewaySession
from .tool_dispatcher import ApprovalDecision, ApprovalRequest
from .tool_redaction import get_audit_hmac_key_id, get_audit_hmac_secret

if TYPE_CHECKING:
  from .commercial_work_start import (
    CommercialWorkStartContext,
    CommercialWorkStartGate,
  )
  from .ui_blocks_run import UiBlocksRunContext

SystemPrompt = str | List[Tuple[str, bool]]
ExecutionLocationResolver = Callable[[str], Optional[str]]
BuildChatRuntime = Callable[[GatewaySession, "ChatRequest", Optional[str], AuthManager], Awaitable["ChatRuntime"]]
RequestApproval = Callable[[ApprovalRequest], Awaitable[Optional[ApprovalDecision]]]
BuildRunner = Callable[..., AgentRunner | AgentSDKRunner]
DispatchScopeValidator = Callable[
  [GatewaySession, dict[str, Any]],
  Awaitable[dict[str, Any] | None] | dict[str, Any] | None,
]


class ScheduledRetentionSweeper(Protocol):
  interval_seconds: float

  def sweep(self) -> Any: ...

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
_ARTIFACT_ORIGIN_VALUES = frozenset({"product", "harness", "import"})
_ARTIFACT_ORIGIN_FILTER_VALUES = frozenset({"all", *_ARTIFACT_ORIGIN_VALUES})
_ARTIFACT_VISIBILITY_VALUES = frozenset({"default", "sandbox", "archived"})
_ARTIFACT_VISIBILITY_FILTER_VALUES = frozenset({"default", "sandbox", "archived", "all"})
_ARTIFACT_INDEX_RECENT_LIMIT = 5
_DEFAULT_CHAT_PROFILE = "analyst"
_CHAT_PROFILE_ALIASES = {"hank": "analyst", "hank-community": "community"}
_ACTIVE_TURN_GRACE_SECONDS = 60.0
_STREAM_SUBSCRIBER_QUEUE_MAX = 256
_STREAM_SUBSCRIBER_KEEPALIVE_SECONDS = 15.0
_SIDECAR_SLUG_RE = re.compile(r"[^a-z0-9-]+")
log = logging.getLogger("agent_gateway.server")
_STREAM_SUBSCRIBER_DONE = object()
_COMMERCIAL_BODY_TOKEN_KEYS = frozenset({
  "commercial_claim",
  "commercial_claim_token",
  "commercial_execution_claim",
  "commercial_execution_claim_token",
  "commercial_work_authorization",
  "commercial_work_authorization_token",
  "hank_commercial_claim",
  "hank_work_authorization",
  "work_authorization",
  "work_authorization_token",
  "x_hank_commercial_claim",
  "x_hank_work_authorization",
})
_COMMERCIAL_TOKEN_ISSUERS = frozenset({
  COMMERCIAL_CLAIM_ISSUER,
  WORK_AUTHORIZATION_ISSUER,
})


def _normalized_body_key(value: str) -> str:
  return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _looks_like_commercial_jwt(value: str) -> bool:
  if not 1 <= len(value) <= 4096:
    return False
  segments = value.split(".")
  if len(segments) != 3 or any(not segment for segment in segments):
    return False
  try:
    payload_segment = segments[1]
    padding = "=" * (-len(payload_segment) % 4)
    payload = json.loads(base64.b64decode(
      payload_segment + padding,
      altchars=b"-_",
      validate=True,
    ))
  except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
    return False
  return isinstance(payload, dict) and payload.get("iss") in _COMMERCIAL_TOKEN_ISSUERS


def _assert_no_commercial_bearer_material(value: Any) -> None:
  stack = [value]
  seen: set[int] = set()
  while stack:
    current = stack.pop()
    if isinstance(current, Mapping):
      identity = id(current)
      if identity in seen:
        continue
      seen.add(identity)
      for key, item in current.items():
        if (
          isinstance(key, str)
          and _normalized_body_key(key) in _COMMERCIAL_BODY_TOKEN_KEYS
        ):
          raise PydanticCustomError(
            "commercial_bearer_material_forbidden",
            "commercial bearer material is accepted only in dedicated headers",
          )
        stack.append(item)
    elif isinstance(current, (list, tuple)):
      identity = id(current)
      if identity in seen:
        continue
      seen.add(identity)
      stack.extend(current)
    elif isinstance(current, str) and _looks_like_commercial_jwt(current):
      raise PydanticCustomError(
        "commercial_bearer_material_forbidden",
        "commercial bearer material is accepted only in dedicated headers",
      )


class ChatInitRequest(BaseModel):
  """Request body for `POST /chat/init`."""

  api_key: str = Field(..., min_length=1)
  schema_version: int | None = None
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
  providers: Dict[str, str] | None = None


class ChatInitResponse(BaseModel):
  """Response body returned after a session token is issued."""

  user_id: str
  session_token: str
  session_id: str
  expires_at: int
  schema_version: int
  model_catalog: Optional[ModelCatalog] = None


class ChatMessage(BaseModel):
  """Single plain-text chat message sent to the gateway."""

  role: str
  content: str


class UiBlocksContractPin(BaseModel):
  contract_version: int = Field(strict=True)
  manifest_digest: str = Field(strict=True)


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
  effort: Optional[str] = None
  ui_blocks_contract: UiBlocksContractPin | None = None
  drain_trailing: bool = False
  _commercial_work_start: "CommercialWorkStartContext | None" = PrivateAttr(
    default=None
  )
  _ui_blocks_run: "UiBlocksRunContext | None" = PrivateAttr(default=None)

  @model_validator(mode="before")
  @classmethod
  def _reject_commercial_bearer_material(cls, value: Any) -> Any:
    _assert_no_commercial_bearer_material(value)
    return value

  @model_validator(mode="before")
  @classmethod
  def _normalize_effort(cls, value: Any) -> Any:
    if isinstance(value, dict) and "effort" in value and value.get("effort") is not None:
      normalized = dict(value)
      normalized["effort"] = parse_effort(value.get("effort")).value
      return normalized
    return value

  @property
  def commercial_work_start(self) -> "CommercialWorkStartContext | None":
    return self._commercial_work_start

  def _bind_commercial_work_start(
    self, context: "CommercialWorkStartContext | None"
  ) -> None:
    self._commercial_work_start = context

  @property
  def ui_blocks_run(self) -> "UiBlocksRunContext | None":
    return self._ui_blocks_run

  def _bind_ui_blocks_run(self, context: "UiBlocksRunContext | None") -> None:
    self._ui_blocks_run = context


class ChatRecapRequest(BaseModel):
  """Request body for `POST /chat/recap`."""

  session_id: str = Field(..., min_length=1)
  scope: str = "active_turn"


class ChatCancelRequest(BaseModel):
  """Request body for `POST /chat/cancel`."""

  session_id: str = Field(..., min_length=1)


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
  effort: str | None = None
  ui_blocks_contract: UiBlocksContractPin | None = None
  commercial_work_start: "CommercialWorkStartContext | None" = field(
    default=None,
    repr=False,
  )
  commercial_dispatch_owner: object | None = field(default=None, repr=False)


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
  denied_by: Optional[str] = None

  @model_validator(mode="after")
  def _validate_denied_by(self) -> "ToolApprovalRequest":
    if self.approved:
      self.denied_by = None
      return self
    if self.denied_by not in (None, "relay_policy"):
      raise ValueError("denied_by must be omitted or 'relay_policy'")
    return self


@dataclass
class ChatRuntime:
  """Per-request runtime wiring returned by `build_chat_runtime`.

  Attributes:
    system_prompt: Prompt string or cached prompt blocks for the request.
    build_runner: Factory that receives `(event_log, session_id, started_at)`
      and returns an `AgentRunner` or `AgentSDKRunner`. Two-argument factories
      are still accepted for compatibility with older tests/custom runtimes.
    get_tool_definitions: Callback returning the tool schemas visible to the
      model for this request.
    build_dispatcher: Reserved callback slot for custom dispatcher builders.
    provider: Provider handling the request, typically Anthropic or OpenAI.
    model_override: Request-scoped model override.
    resolved_provider_name: Provider resolved for this request, when known.
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
  resolved_provider_name: Optional[str] = None
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


def _build_runner_with_started_at(
  build_runner: BuildRunner,
  event_log: EventLog,
  session_id: str,
  started_at: float,
) -> AgentRunner | AgentSDKRunner:
  try:
    signature = inspect.signature(build_runner)
  except (TypeError, ValueError):
    return build_runner(event_log, session_id, started_at)

  positional_params = [
    param
    for param in signature.parameters.values()
    if param.kind in {param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD}
  ]
  accepts_varargs = any(param.kind == param.VAR_POSITIONAL for param in signature.parameters.values())
  if accepts_varargs or len(positional_params) >= 3:
    return build_runner(event_log, session_id, started_at)
  return build_runner(event_log, session_id)


async def _call_build_chat_runtime(
  build_chat_runtime: BuildChatRuntime,
  *,
  session: GatewaySession,
  request: "ChatRequest",
  channel: str | None,
  auth_manager: AuthManager | None,
) -> "ChatRuntime":
  kwargs = {
    "session": session,
    "request": request,
    "channel": channel,
    "auth_manager": auth_manager,
  }
  try:
    signature = inspect.signature(build_chat_runtime)
  except (TypeError, ValueError):
    return await build_chat_runtime(session, request, channel, auth_manager)  # type: ignore[misc]

  params = signature.parameters
  accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
  accepts_named_kwargs = all(
    name in params
    and params[name].kind in {params[name].POSITIONAL_OR_KEYWORD, params[name].KEYWORD_ONLY}
    for name in kwargs
  )
  if accepts_kwargs or accepts_named_kwargs:
    return await build_chat_runtime(**kwargs)
  return await build_chat_runtime(session, request, channel, auth_manager)  # type: ignore[misc]


@dataclass
class RequestContext:
  """Mutable request-scoped objects shared across dispatch layers."""

  session: GatewaySession
  event_log: EventLog
  request_approval: RequestApproval
  result_queue: asyncio.Queue
  mcp_client: Optional[McpClientManager]
  run_context: Any | None = None


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
    dispatch_scope_validator: Optional dispatch-time validator for structured
      portfolio scopes. The callback may canonicalize the redacted scope or
      raise an HTTPException/ValueError to stop run creation.
    commercial_work_start_gate: Optional default-off gate that verifies header
      authority and persists one-time consumption before runtime construction.
    mcp_client: Optional shared `McpClientManager`.
    sdk_config: Optional `AgentSDKConfig` when using `AgentSDKRunner`.
    per_turn_timeout: Default per-turn timeout in seconds.
    compaction_trigger: Default compaction threshold.
    compaction_instructions: Default compaction instructions.
    allowed_models: Model allowlist enforced at request time.
    model_catalog: Optional model discovery metadata returned by `POST /chat/init`.
    channel_profile_allowlist: Optional mapping from authoritative session
      channel to profile names allowed on that channel. Channels absent from
      the mapping are unrestricted.
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
    transcript_retention_days: Number of days to retain chat transcript files.
    retention_sweeper: Optional synchronous retention sweeper injected by an
      application owner. Package consumers default to no sweep.
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
  jwt_secret: str = field(default_factory=lambda: secrets.token_hex(32))
  session_ttl: int = 3600
  valid_api_keys: Set[str] = field(default_factory=set)
  cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3002"])
  cors_allow_headers: List[str] = field(
    default_factory=lambda: [
      "Authorization",
      "Content-Type",
      COMMERCIAL_CLAIM_HEADER,
      COMMERCIAL_WORK_AUTHORIZATION_HEADER,
      *_AGENT_API_CLAIM_HEADERS.values(),
    ]
  )
  cors_allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
  auth_config: Dict[str, Any] = field(default_factory=dict)
  credentials_resolver: CredentialsResolver | None = None
  credentials_refresh_resolver: CredentialsRefreshResolver | None = None
  dispatch_scope_validator: DispatchScopeValidator | None = None
  commercial_work_start_gate: "CommercialWorkStartGate | None" = None
  resolver_timeout_seconds: float = 5.0
  mcp_client: Optional[McpClientManager] = None
  sdk_config: AgentSDKConfig | None = None
  per_turn_timeout: int = 300
  compaction_trigger: int | None = None
  compaction_instructions: str | None = None
  allowed_models: Set[str] | None = None
  model_catalog: Optional[ModelCatalog] = None
  channel_profile_allowlist: Mapping[str, frozenset[str]] | None = None
  build_chat_runtime: Optional[BuildChatRuntime] = None
  on_event: Optional[Callable[..., Any]] = None
  on_tool_result: Optional[Callable[..., Any]] = None
  on_usage: Optional[Callable[..., Any]] = None
  on_tool_timing: Optional[Callable[..., Any]] = None
  on_session_created: Optional[Callable[[GatewaySession, str, ChatInitRequest], None]] = None
  on_startup: Optional[Callable[..., Any]] = None
  on_shutdown: Optional[Callable[..., Any]] = None
  transcript_dir: Optional[Path] = None
  transcript_retention_days: int = 7
  retention_sweeper: ScheduledRetentionSweeper | None = None
  control_skills_dir: Optional[Path] = None
  audit_hmac_secret_resolver: Callable[[], bytes] = get_audit_hmac_secret
  audit_hmac_key_id_resolver: Callable[[], str] = get_audit_hmac_key_id
  tool_input_redactor: Optional[Callable[..., dict[str, Any]]] = None
  log_name: str = "gateway"
  prefix: str = "/api"
