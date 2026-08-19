from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import inspect
import json
import logging
import math
import re
import secrets
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
  TYPE_CHECKING,
  Any,
  Awaitable,
  Callable,
  Dict,
  List,
  Literal,
  Mapping,
  Optional,
  Protocol,
  Set,
  Tuple,
)

from fastapi import HTTPException
from pydantic import (
  BaseModel,
  ConfigDict,
  Field,
  PrivateAttr,
  field_validator,
  model_validator,
)
from pydantic_core import PydanticCustomError

from .auth import CredentialsResolver
from .capability_binding import (
  CapabilityBind,
  CredentialHandle,
)
from .capability_execution import (
  BoundCapabilityExecution,
  CapabilityAdapterResolver,
  CapabilityExecutionResolver,
  MaterializedCredential,
)
from .model_registry import (
  ModelLifecycle,
  ProductModelRegistry,
  ProductModelSelectionPolicy,
  SelectionSource,
)
from .model_preferences import ModelPreferenceStore
from .autonomous_capability_handoff import AutonomousCapabilityBindingResolver
from .claim_signing_authority import GatewayClaimSigningAuthority
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
from .selected_content import SelectedContentAdmission
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
SelectedContentAdmitter = Callable[
  [GatewaySession, "ChatRequest"],
  Awaitable[SelectedContentAdmission] | SelectedContentAdmission,
]
RequestApproval = Callable[[ApprovalRequest], Awaitable[Optional[ApprovalDecision]]]
BuildRunner = Callable[
  [EventLog, str, float],
  AgentRunner | AgentSDKRunner,
]
DispatchScopeValidator = Callable[
  [GatewaySession, dict[str, Any]],
  Awaitable[dict[str, Any] | None] | dict[str, Any] | None,
]
@dataclass(frozen=True)
class SessionExecutionPolicy:
  """Trusted per-turn runner limits; model selection belongs to product policy."""

  max_tokens: int | None = None
  max_turns: int | None = None
  max_budget_usd: float | None = None

  def __post_init__(self) -> None:
    for field_name in ("max_tokens", "max_turns"):
      value = getattr(self, field_name)
      if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
      ):
        raise ValueError(f"{field_name} must be a positive integer")
    budget = self.max_budget_usd
    if budget is not None and (
      isinstance(budget, bool)
      or not isinstance(budget, int | float)
      or not math.isfinite(float(budget))
      or float(budget) <= 0
    ):
      raise ValueError("max_budget_usd must be a finite positive number")


SessionExecutionPolicyResolver = Callable[
  [Mapping[str, Any]],
  SessionExecutionPolicy | None,
]


ServiceAuthConfigResolver = Callable[
  [CredentialHandle],
  MaterializedCredential,
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

  model_config = ConfigDict(extra="forbid")

  api_key: str = Field(..., min_length=1)
  schema_version: int | None = None
  user_id: str | None = None
  user_email: str | None = None
  request_id: str | None = None
  subject_assertion: str | None = None
  context: Dict[str, Any] = Field(default_factory=dict)
  anthropic_auth_mode: Optional[str] = None
  anthropic_api_key: Optional[str] = None
  anthropic_auth_token: Optional[str] = None
  capability_selections: Dict[str, "CapabilitySelection"] = Field(
    default_factory=dict
  )


class CapabilitySelection(BaseModel):
  """Strict stable-key run selection for one internal capability."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  model_key: str = Field(strict=True, min_length=1)
  effort: str | None = Field(default=None, strict=True)

  @model_validator(mode="after")
  def _normalize_and_require_selection(self) -> "CapabilitySelection":
    model_key = self.model_key.strip()
    if not model_key:
      raise ValueError("capability selection model_key must be non-empty")
    effort = (
      parse_effort(self.effort, field_name="capability selection effort").value
      if self.effort is not None
      else None
    )
    object.__setattr__(self, "model_key", model_key)
    object.__setattr__(self, "effort", effort)
    return self


class CapabilityChoice(BaseModel):
  """Presentation-only session-eligible stable model choice."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  model_key: str
  label: str
  supported_efforts: List[str]
  default_effort: str
  lifecycle: ModelLifecycle


class CapabilityChoiceSelection(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  model_key: str
  label: str
  effort: str
  reason: SelectionSource


class CapabilityChoiceNotice(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  code: str
  message: str
  model_key: str | None = None
  reason: str | None = None


class CapabilityChoiceResponse(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  capability: str
  catalog_revision: str
  policy_revision: str
  selected: CapabilityChoiceSelection | None
  notices: List[CapabilityChoiceNotice]
  choices: List[CapabilityChoice]


class ModelPreferenceUpdate(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  model_key: str = Field(min_length=1)
  effort: str | None = None
  catalog_revision: str | None = None

  @model_validator(mode="after")
  def _normalize(self) -> "ModelPreferenceUpdate":
    model_key = self.model_key.strip()
    if not model_key:
      raise ValueError("preference model_key must be non-empty")
    object.__setattr__(self, "model_key", model_key)
    if self.effort is not None:
      object.__setattr__(
        self,
        "effort",
        parse_effort(self.effort, field_name="preference effort").value,
      )
    if self.catalog_revision is not None:
      object.__setattr__(
        self,
        "catalog_revision",
        self.catalog_revision.strip() or None,
      )
    return self


class ModelPreferenceResponse(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  capability: str
  model_key: str | None
  effort: str | None


class ChatInitResponse(BaseModel):
  """Response body returned after a session token is issued."""

  user_id: str
  session_token: str
  session_id: str
  expires_at: int
  schema_version: int
  capability_choices: Dict[str, CapabilityChoiceResponse]


class ChatMessage(BaseModel):
  """Single plain-text chat message sent to the gateway."""

  role: str
  content: str


class UiBlocksContractPin(BaseModel):
  contract_version: int = Field(strict=True)


CHAT_ATTACHMENTS_CONTRACT = "chat-attachments-v1"
CHAT_ATTACHMENT_MAX_COUNT = 8
CHAT_ATTACHMENT_MAX_BYTES = 1024 * 1024
CHAT_ATTACHMENT_MAX_BASE64_BYTES = 1_398_104
CHAT_ATTACHMENT_MAX_TOTAL_BYTES = 4 * 1024 * 1024
CHAT_ATTACHMENT_MAX_TOTAL_BASE64_BYTES = 5_592_416
_CHAT_ATTACHMENT_NAME_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_CHAT_ATTACHMENT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHAT_ATTACHMENT_MEDIA_TYPES_BY_SUFFIX = {
  ".txt": "text/plain",
  ".md": "text/markdown",
  ".markdown": "text/markdown",
  ".json": "application/json",
  ".csv": "text/csv",
  ".tsv": "text/tab-separated-values",
}


def _expected_attachment_input_name(index: int) -> str:
  return "source_document" if index == 1 else f"source_document_{index}"


class ChatAttachmentV1(BaseModel):
  """Closed first-cohort exact UTF-8 attachment envelope."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  schema_version: Literal["chat-attachment/1"]
  input_name: str
  display_name: str
  media_type: str
  encoding: Literal["utf-8"]
  content_bytes: int = Field(gt=0, le=CHAT_ATTACHMENT_MAX_BYTES)
  content_sha256: str
  content_b64: str = Field(min_length=4, max_length=CHAT_ATTACHMENT_MAX_BASE64_BYTES)
  _decoded_content: bytes = PrivateAttr(default=b"")

  @field_validator("input_name")
  @classmethod
  def _validate_input_name(cls, value: str) -> str:
    if _CHAT_ATTACHMENT_NAME_RE.fullmatch(value) is None:
      raise ValueError("input_name is invalid")
    return value

  @field_validator("display_name")
  @classmethod
  def _validate_display_name(cls, value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).strip()
    if value != normalized:
      raise ValueError(
        "display_name must be NFC-normalized without surrounding whitespace"
      )
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
      raise ValueError("display_name must be a basename")
    if any(unicodedata.category(character) == "Cc" for character in value):
      raise ValueError("display_name contains control characters")
    if len(value.encode("utf-8")) > 255:
      raise ValueError("display_name exceeds 255 UTF-8 bytes")
    return value

  @field_validator("content_sha256")
  @classmethod
  def _validate_content_sha256(cls, value: str) -> str:
    if _CHAT_ATTACHMENT_SHA256_RE.fullmatch(value) is None:
      raise ValueError("content_sha256 must be 64 lowercase hex characters")
    return value

  @model_validator(mode="after")
  def _validate_content(self) -> "ChatAttachmentV1":
    suffix = next(
      (
        candidate
        for candidate in sorted(
          _CHAT_ATTACHMENT_MEDIA_TYPES_BY_SUFFIX,
          key=len,
          reverse=True,
        )
        if self.display_name.lower().endswith(candidate)
      ),
      None,
    )
    expected_media_type = (
      _CHAT_ATTACHMENT_MEDIA_TYPES_BY_SUFFIX.get(suffix)
      if suffix is not None
      else None
    )
    if expected_media_type is None or self.media_type != expected_media_type:
      raise ValueError(
        "media_type does not match the allowlisted display_name suffix"
      )
    if not self.content_b64 or self.content_b64.startswith("data:"):
      raise ValueError("content_b64 must be canonical base64 without a data URL prefix")
    try:
      decoded = base64.b64decode(self.content_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
      raise ValueError("content_b64 is not valid base64") from exc
    if base64.b64encode(decoded).decode("ascii") != self.content_b64:
      raise ValueError("content_b64 is not canonical base64")
    if len(decoded) != self.content_bytes:
      raise ValueError("content_bytes does not match decoded content")
    try:
      decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
      raise ValueError("attachment content must be valid UTF-8") from exc
    if hashlib.sha256(decoded).hexdigest() != self.content_sha256:
      raise ValueError("content_sha256 does not match decoded content")
    self._decoded_content = decoded
    return self

  def decoded_bytes(self) -> bytes:
    return self._decoded_content


class InvestmentArtifactSelection(BaseModel):
  """Untrusted coordinates for one explicit bounded Investment view."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  artifact_id: str = Field(min_length=1, max_length=256)
  view: Literal["summary", "excerpt"]

  @field_validator("artifact_id")
  @classmethod
  def _validate_artifact_id(cls, value: str) -> str:
    if value != value.strip() or any(
      ord(character) < 32 or ord(character) == 127 for character in value
    ):
      raise ValueError("artifact_id is invalid")
    return value


class ChatRequest(BaseModel):
  """Request body for `POST /chat`.

  `messages` is the full message list for the current turn. `context` is
  intentionally free-form; `context["channel"]` is the most common field used
  to shape runtime behavior per frontend.
  """

  model_config = ConfigDict(extra="forbid")

  messages: List[ChatMessage]
  user_id: str | None = None
  request_id: str | None = None
  context: Dict[str, Any] = Field(default_factory=dict)
  metadata: Dict[str, Any] = Field(default_factory=dict)
  model_key: Optional[str] = None
  effort: Optional[str] = None
  catalog_revision: Optional[str] = None
  ui_blocks_contract: UiBlocksContractPin | None = None
  attachments: tuple[ChatAttachmentV1, ...] = ()
  investment_artifact_selection: InvestmentArtifactSelection | None = None
  drain_trailing: bool = False
  _commercial_work_start: "CommercialWorkStartContext | None" = PrivateAttr(
    default=None
  )
  _ui_blocks_run: "UiBlocksRunContext | None" = PrivateAttr(default=None)
  _capability_execution: BoundCapabilityExecution | None = PrivateAttr(
    default=None
  )
  _capability_execution_resolver: CapabilityExecutionResolver | None = PrivateAttr(
    default=None
  )
  _session_execution_policy: SessionExecutionPolicy | None = PrivateAttr(
    default=None
  )

  @model_validator(mode="before")
  @classmethod
  def _reject_commercial_bearer_material(cls, value: Any) -> Any:
    _assert_no_commercial_bearer_material(value)
    return value

  @model_validator(mode="before")
  @classmethod
  def _normalize_selection(cls, value: Any) -> Any:
    if not isinstance(value, dict):
      return value
    normalized = dict(value)
    if "model_key" in value and value.get("model_key") is not None:
      model_key = str(value.get("model_key") or "").strip()
      if not model_key:
        raise ValueError("model_key must be non-empty when supplied")
      normalized["model_key"] = model_key
    if "effort" in value and value.get("effort") is not None:
      normalized["effort"] = parse_effort(value.get("effort")).value
    return normalized

  @model_validator(mode="after")
  def _require_complete_explicit_selection(self) -> "ChatRequest":
    if self.effort is not None and self.model_key is None:
      raise ValueError("effort requires an explicit model_key")
    if self.catalog_revision is not None and self.model_key is None:
      raise ValueError("catalog_revision requires an explicit model_key")
    if len(self.attachments) > CHAT_ATTACHMENT_MAX_COUNT:
      raise ValueError(
        f"attachments cannot contain more than {CHAT_ATTACHMENT_MAX_COUNT} files"
      )
    if sum(item.content_bytes for item in self.attachments) > CHAT_ATTACHMENT_MAX_TOTAL_BYTES:
      raise ValueError("attachments exceed the aggregate decoded byte limit")
    if sum(len(item.content_b64) for item in self.attachments) > (
      CHAT_ATTACHMENT_MAX_TOTAL_BASE64_BYTES
    ):
      raise ValueError("attachments exceed the aggregate base64 byte limit")
    for index, attachment in enumerate(self.attachments, start=1):
      expected_name = _expected_attachment_input_name(index)
      if attachment.input_name != expected_name:
        raise ValueError(
          f"attachments[{index - 1}].input_name must be {expected_name!r}"
        )
    return self

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

  @property
  def capability_execution(self) -> BoundCapabilityExecution | None:
    return self._capability_execution

  @property
  def capability_bind(self) -> CapabilityBind | None:
    execution = self._capability_execution
    return execution.bind if execution is not None else None

  @property
  def bound_provider(self) -> ModelProvider | None:
    execution = self._capability_execution
    return execution.provider if execution is not None else None

  @property
  def bound_auth_config(self) -> Mapping[str, Any] | None:
    execution = self._capability_execution
    return execution.auth_config if execution is not None else None

  @property
  def capability_execution_resolver(self) -> CapabilityExecutionResolver | None:
    return self._capability_execution_resolver

  @property
  def session_execution_policy(self) -> SessionExecutionPolicy | None:
    return self._session_execution_policy

  def _bind_session_execution_policy(
    self,
    policy: SessionExecutionPolicy | None,
  ) -> None:
    if self._session_execution_policy is not None:
      raise ValueError("session execution policy is already bound for this turn")
    if policy is not None and not isinstance(policy, SessionExecutionPolicy):
      raise TypeError("session execution policy must be SessionExecutionPolicy")
    self._session_execution_policy = policy

  def _bind_capability_execution_resolver(
    self,
    resolver: CapabilityExecutionResolver,
  ) -> None:
    if self._capability_execution_resolver is not None:
      raise ValueError("capability execution resolver is already bound for this turn")
    if not isinstance(resolver, CapabilityExecutionResolver):
      raise TypeError(
        "capability execution resolver must be CapabilityExecutionResolver"
      )
    execution = self._capability_execution
    if execution is None:
      raise ValueError(
        "session.driver must be bound before the capability execution resolver"
      )
    capability_bind = execution.bind
    if capability_bind.run_mode != resolver.auth_context.run_mode:
      raise ValueError(
        "capability execution resolver auth mode does not match session.driver"
      )
    self._capability_execution_resolver = resolver

  def _bind_session_driver(
    self,
    *,
    capability_execution: BoundCapabilityExecution,
    capability_execution_resolver: CapabilityExecutionResolver | None = None,
  ) -> None:
    if self._capability_execution is not None:
      raise ValueError("session.driver is already bound for this turn")
    if not isinstance(capability_execution, BoundCapabilityExecution):
      raise TypeError(
        "session.driver must be a BoundCapabilityExecution"
      )
    capability_execution.validate()
    capability_bind = capability_execution.bind
    if capability_bind.capability_id != "session.driver":
      raise ValueError("chat requests require a session.driver capability bind")
    self._capability_execution = capability_execution
    if capability_execution_resolver is not None:
      self._bind_capability_execution_resolver(capability_execution_resolver)


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
  model_key: str | None
  effort: str | None = None
  catalog_revision: str | None = None
  ui_blocks_contract: UiBlocksContractPin | None = None
  attachments: tuple[ChatAttachmentV1, ...] = ()
  investment_artifact_selection: InvestmentArtifactSelection | None = None
  commercial_work_start: "CommercialWorkStartContext | None" = field(
    default=None,
    repr=False,
  )
  commercial_dispatch_owner: object | None = field(default=None, repr=False)


@dataclass(frozen=True)
class PreparedChatTurn:
  session_id: str
  request: ChatRequest
  channel: str | None


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
    build_runner: Factory that receives `(event_log, session_id, started_at)`
      and returns an `AgentRunner` or `AgentSDKRunner`.
    get_tool_definitions: Callback returning the tool schemas visible to the
      model for this request.
    build_dispatcher: Reserved callback slot for custom dispatcher builders.
    capability_execution: Exact immutable provider, credential, model, effort,
      bind, and transport consumed by the runtime.
    purpose: Request purpose used for definition-time policy filtering.
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
  capability_execution: BoundCapabilityExecution
  get_tool_definitions: Callable[[], List[Dict[str, Any]]] = field(default_factory=lambda: [])
  build_dispatcher: Callable[["RequestContext"], Any] | None = None
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
  purpose: str | None = None
  _disconnect_called: bool = field(default=False, init=False, repr=False)

  def __post_init__(self) -> None:
    execution = self.capability_execution
    if not isinstance(execution, BoundCapabilityExecution):
      raise TypeError(
        "ChatRuntime.capability_execution must be BoundCapabilityExecution"
      )
    execution.validate()
    if execution.bind.capability_id != "session.driver":
      raise ValueError(
        "ChatRuntime requires a session.driver capability execution"
      )

  @property
  def capability_bind(self) -> CapabilityBind:
    return self.capability_execution.bind

  @property
  def provider(self) -> ModelProvider:
    return self.capability_execution.provider

  @property
  def resolved_provider_name(self) -> str:
    return self.capability_execution.bind.provider

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
  return build_runner(event_log, session_id, started_at)


async def _call_build_chat_runtime(
  build_chat_runtime: BuildChatRuntime,
  *,
  session: GatewaySession,
  request: "ChatRequest",
  channel: str | None,
  auth_manager: AuthManager | None,
  storage_root: Path | None = None,
) -> "ChatRuntime":
  kwargs = {
    "session": session,
    "request": request,
    "channel": channel,
    "auth_manager": auth_manager,
  }
  if storage_root is not None:
    kwargs["storage_root"] = storage_root
  try:
    signature = inspect.signature(build_chat_runtime)
  except (TypeError, ValueError):
    return await build_chat_runtime(session, request, channel, auth_manager)  # type: ignore[misc]

  params = signature.parameters
  accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
  legacy_names = ("session", "request", "channel", "auth_manager")
  legacy_names_are_keyword_capable = all(
    name in params
    and params[name].kind in {params[name].POSITIONAL_OR_KEYWORD, params[name].KEYWORD_ONLY}
    for name in legacy_names
  )
  if accepts_kwargs:
    return await build_chat_runtime(**kwargs)
  if legacy_names_are_keyword_capable:
    supported_kwargs = {
      name: value
      for name, value in kwargs.items()
      if name in params
      and params[name].kind
      in {params[name].POSITIONAL_OR_KEYWORD, params[name].KEYWORD_ONLY}
    }
    return await build_chat_runtime(**supported_kwargs)
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
    tenant_id: Validated deployment tenant used for secret-free credential
      provenance. Deployments that bind capabilities must set this explicitly.
    allow_service_credentials_for_interactive: Server-owned default policy for
      interactive use of service credentials when no resolver overrides it.
    credentials_resolver: Optional init-time resolver for per-user credentials.
    dispatch_scope_validator: Optional dispatch-time validator for structured
      portfolio scopes. The callback may canonicalize the redacted scope or
      raise an HTTPException/ValueError to stop run creation.
    commercial_work_start_gate: Optional default-off gate that verifies header
      authority and persists one-time consumption before runtime construction.
    mcp_client: Optional shared `McpClientManager`.
    mcp_meta_inject_servers: Optional immutable set of MCP servers that receive
      server-derived invocation identity metadata.
    sdk_config: Optional `AgentSDKConfig` when using `AgentSDKRunner`.
    per_turn_timeout: Default per-turn timeout in seconds.
    compaction_trigger: Default compaction threshold.
    compaction_instructions: Default compaction instructions.
    model_registry: Required stable execution-identity authority.
    model_selection_policy: Required product default and selection policy.
    model_preference_store: Durable account-wide per-capability preference
      store. Preferences remain intent and are re-admitted for each session.
    session_execution_policy_resolver: Trusted per-turn runner-limit resolver;
      it cannot select a model or effort.
    service_provider_handles: Secret-free service credential provenance by
      provider family.
    service_auth_config_resolver: Trusted server callback that materializes the
      selected service credential after principal resolution.
    capability_adapter_resolver: Trusted callback that returns the exact
      adapter implementation for a bound adapter ID. Preparation uses it to
      validate credential material and exact effort before any durable or
      billable side effect. Startup closure resolves every adapter named by a
      gateway-executed registry entry through it at construction; the callback
      vouches for the implementations it maps, whereas without it only
      installed adapters with matching `adapter_route_support` declarations
      are admitted.
    autonomous_capability_binding_resolver: Trusted server callback that
      resolves and materializes the exact profile/skill-aware session-driver
      bind before an autonomous subprocess is launched. Resume requests carry
      the persisted bind as an exact requirement.
    claim_signing_authority: Process-local claim signer loaded from a one-shot
      descriptor before the gateway application is imported.
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
  cors_allow_methods: List[str] = field(
    default_factory=lambda: [
      "GET",
      "POST",
      "PUT",
      "DELETE",
      "PATCH",
      "OPTIONS",
    ]
  )
  tenant_id: str | None = None
  allow_service_credentials_for_interactive: bool = False
  credentials_resolver: CredentialsResolver | None = None
  dispatch_scope_validator: DispatchScopeValidator | None = None
  commercial_work_start_gate: "CommercialWorkStartGate | None" = None
  resolver_timeout_seconds: float = 5.0
  mcp_client: Optional[McpClientManager] = None
  mcp_meta_inject_servers: frozenset[str] | None = None
  sdk_config: AgentSDKConfig | None = None
  per_turn_timeout: int = 300
  compaction_trigger: int | None = None
  compaction_instructions: str | None = None
  model_registry: ProductModelRegistry | None = None
  model_selection_policy: ProductModelSelectionPolicy | None = None
  model_preference_store: ModelPreferenceStore | None = None
  session_execution_policy_resolver: SessionExecutionPolicyResolver | None = None
  service_provider_handles: Mapping[str, CredentialHandle] = field(default_factory=dict)
  service_auth_config_resolver: ServiceAuthConfigResolver | None = None
  capability_adapter_resolver: CapabilityAdapterResolver | None = None
  autonomous_capability_binding_resolver: (
    AutonomousCapabilityBindingResolver | None
  ) = None
  claim_signing_authority: GatewayClaimSigningAuthority | None = None
  channel_profile_allowlist: Mapping[str, frozenset[str]] | None = None
  build_chat_runtime: Optional[BuildChatRuntime] = None
  selected_content_admitter: SelectedContentAdmitter | None = None
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
