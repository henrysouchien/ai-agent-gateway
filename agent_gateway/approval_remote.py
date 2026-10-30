"""Versioned approval egress contract and placement-bound transport.

Owner: EDGE-LANES. Consumer: RUNTIME-ALIGN W1d. Base:
0d50be342dfba3a71df9283784384fde0b141ba0.

Transport: POST <placement-bound approvals base>/<operation> with the versioned
JSON envelope below. No second approval route family is added.

The server derives storage_id and placement_generation from the handler
closure. The client does not send either. AgentCore authorizes by that
closure, with no redundant claim check. LocalProcess mounts the same handler
after the edge resolves a fresh derived claim to a ready placement.
LocalProcess is not an OS tenant-isolation boundary.

TypeAdapters for store-owned types are built on first codec use, never at
module load. This module does not import agent_gateway.approval_store at
module load: W1d's store imports this module, and a load-time import would
cycle. Codec functions import those names lazily when reconstructing them.
"""
from __future__ import annotations

import base64
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import MISSING, fields
from datetime import datetime
from functools import cache
from typing import (
  TYPE_CHECKING,
  Annotated,
  Any,
  ContextManager,
  Literal,
  NoReturn,
  Protocol,
  TypeAlias,
  cast,
  get_args,
  get_origin,
  get_type_hints,
)

from pydantic import (
  BaseModel,
  BeforeValidator,
  ConfigDict,
  Field,
  JsonValue,
  PlainSerializer,
  TypeAdapter,
  ValidationError,
  create_model,
  field_validator,
  model_validator,
)

from agent_gateway.approval_policy import (
  ApprovalConstraintError,
  ApprovalRequest,
  ApprovalReuseMode,
  ApprovalState,
  ApprovalVote,
  DelegationGrant,
  PersistentGrant,
)
from agent_gateway.prepared_business_model_store import (
  PreparedBusinessModelChange,
  PreparedBusinessModelError,
  PreparedBusinessModelLifecycle,
)
from agent_gateway.raw_patch_authorization_store import (
  RawPatchAuthorizationClaim,
  RawPatchAuthorizationConsumption,
  RawPatchAuthorizationError,
  RawPatchPreparedAuthorization,
)
from agent_gateway.tool_dispatcher_helpers import ToolResult
from agent_gateway.tool_policy_registry import PreparedToolCall

if TYPE_CHECKING:
  from agent_gateway.approval_store import (
    ApprovalRequestLocator,
    ExecutableApprovalBinding,
    PolicyGeneration,
    RootTurnRecord,
    TargetedPreparedReconciliationResult,
  )


APPROVALS_EGRESS_URL_ENV = "HANK_APPROVALS_EGRESS_URL"
APPROVALS_EGRESS_URL = "http://127.0.0.1:8090/egress/approvals"
APPROVAL_REMOTE_PROTOCOL_VERSION = "hank.approval-remote.v1"

ExecutionAuditOutcome: TypeAlias = Literal["success", "tool_error", "cancelled"]

ApprovalRemoteOperation: TypeAlias = Literal[
  "request.create", "request.create_raw_patch", "request.create_or_get",
  "request.create_or_get_prepared_bm", "request.get", "request.get_by_tool_call",
  "request.update", "request.transition", "request.cancel", "request.abort",
  "request.vote", "request.vote_with_delivery",
  "locator.locate", "locator.list_parked", "locator.for_decision",
  "locator.root_turn_for_approval",
  "policy.record", "policy.read", "policy.current", "policy.record_root_turn",
  "policy.read_root_turn", "policy.link_executable", "policy.reissue",
  "executable.record", "executable.park", "executable.read",
  "executable.read_completed", "executable.claim", "executable.complete",
  "grant.create_persistent", "grant.find_persistent", "grant.revoke_persistent",
  "grant.revoke_persistent_for_approval", "grant.fence_cancellation",
  "grant.create_delegation", "grant.get_delegation", "grant.claim_delegation",
  "grant.revoke_delegation",
  "prepared_bm.insert_or_verify", "prepared_bm.get", "prepared_bm.transition",
  "prepared_bm.reconcile",
  "raw_patch.resolve", "raw_patch.bind_prepared", "raw_patch.get_prepared",
  "raw_patch.consume", "raw_patch.claim", "raw_patch.release",
  "raw_patch.heartbeat", "raw_patch.get_consumption",
  "outbox.get", "outbox.ensure_audited", "outbox.acknowledge", "outbox.failure",
  "outbox.quarantine", "outbox.list_pending", "outbox.high_water",
  "outbox.recovery_window", "publish.reserve", "publish.commit", "publish.abort",
  "publish.reconcile_duplicate",
  "notification.enqueue", "notification.retry_failed", "notification.get_projection",
  "notification.list_outbox",
  "audit.execution_outcome",
]

# Payload keys are exactly the method parameter names excluding self. The two
# context managers have no operation; they call the publish operations below.
#
# Server dispatch is not getattr(store, OPERATION_METHODS[operation]) for every
# key. 63 operations dispatch to the same-named method on the closure's
# per-storage SQLiteApprovalStore. publish.reserve, publish.commit, and
# publish.abort are not store methods: they are approval-service operations
# over the reservation table in that storage's DB, and the store writers are
# reservation-aware. audit.execution_outcome is not a store method name: it
# dispatches to that store's configured audit_emitter.emit_execution_outcome.
# A None emitter on that explicit RPC is unavailable, not a silent success and
# not a dropped getattr.
OPERATION_METHODS: Mapping[str, str] = {
  "request.create": "create",
  "request.create_raw_patch": "create_raw_patch_authorization",
  "request.create_or_get": "create_or_get_by_tool_call_id",
  "request.create_or_get_prepared_bm": "create_or_get_with_prepared_business_model_change",
  "request.get": "get",
  "request.get_by_tool_call": "get_by_tool_call_id",
  "request.update": "update_request",
  "request.transition": "transition_state",
  "request.cancel": "terminalize_pending_for_cancellation",
  "request.abort": "abort_unpublished_approval",
  "request.vote": "record_vote",
  "request.vote_with_delivery": "record_vote_with_autonomous_delivery",
  "locator.locate": "locate_request",
  "locator.list_parked": "list_parked_requests",
  "locator.for_decision": "_locate_request_for_decision",
  "locator.root_turn_for_approval": "_locate_root_turn_for_approval",
  "policy.record": "record_policy_generation",
  "policy.read": "read_policy_generation",
  "policy.current": "current_policy_generation",
  "policy.record_root_turn": "record_root_turn",
  "policy.read_root_turn": "_read_root_turn_for_executor",
  "policy.link_executable": "link_root_turn_executable_binding",
  "policy.reissue": "reissue_parked_approval",
  "executable.record": "record_executable_binding",
  "executable.park": "park_executable_binding",
  "executable.read": "_read_executable_binding_for_executor",
  "executable.read_completed": "_read_completed_executable_result_for_executor",
  "executable.claim": "_claim_executable_binding_for_executor",
  "executable.complete": "_complete_executable_binding_for_executor",
  "grant.create_persistent": "create_persistent_grant",
  "grant.find_persistent": "find_persistent_grant",
  "grant.revoke_persistent": "revoke_persistent_grant",
  "grant.revoke_persistent_for_approval": "revoke_persistent_grants_for_approval",
  "grant.fence_cancellation": "fence_persistent_grants_for_cancellation",
  "grant.create_delegation": "create_delegation_grant",
  "grant.get_delegation": "get_delegation_grant",
  "grant.claim_delegation": "claim_delegation_grant",
  "grant.revoke_delegation": "revoke_delegation_grant",
  "prepared_bm.insert_or_verify": "insert_or_verify_prepared_business_model_change",
  "prepared_bm.get": "get_prepared_business_model_change",
  "prepared_bm.transition": "transition_prepared_business_model_change",
  "prepared_bm.reconcile": "reconcile_prepared_business_model_change",
  "raw_patch.resolve": "resolve_raw_patch_authorization_reference",
  "raw_patch.bind_prepared": "bind_raw_patch_prepared_authorization",
  "raw_patch.get_prepared": "get_raw_patch_prepared_authorization",
  "raw_patch.consume": "consume_raw_patch_authorization",
  "raw_patch.claim": "claim_raw_patch_authorization",
  "raw_patch.release": "release_raw_patch_authorization_claim",
  "raw_patch.heartbeat": "heartbeat_raw_patch_authorization_claim",
  "raw_patch.get_consumption": "get_raw_patch_authorization_consumption",
  "outbox.get": "get_autonomous_approval_delivery",
  "outbox.ensure_audited": "ensure_autonomous_approval_delivery_audited",
  "outbox.acknowledge": "acknowledge_autonomous_approval_delivery",
  "outbox.failure": "record_autonomous_approval_delivery_failure",
  "outbox.quarantine": "quarantine_autonomous_approval_delivery",
  "outbox.list_pending": "list_pending_autonomous_approval_deliveries",
  "outbox.high_water": "autonomous_approval_delivery_high_water",
  "outbox.recovery_window": "autonomous_approval_delivery_recovery_window",
  "publish.reserve": "reserve_autonomous_approval_publication",
  "publish.commit": "commit_autonomous_approval_publication",
  "publish.abort": "abort_autonomous_approval_publication",
  "publish.reconcile_duplicate": "reconcile_autonomous_approval_delivery_duplicate",
  "notification.enqueue": "enqueue_pending_approval_notification",
  "notification.retry_failed": "retry_failed_approval_notifications",
  "notification.get_projection": "get_approval_notification_projection",
  "notification.list_outbox": "list_approval_notification_outbox",
  "audit.execution_outcome": "audit_emitter.emit_execution_outcome",
}

# 67 operations. These four are the dispatch exceptions above. The other 63
# values are SQLiteApprovalStore method names.
STORE_METHOD_DISPATCH_COUNT = 63
RESERVATION_TABLE_OPERATIONS: tuple[str, ...] = (
  "publish.reserve",
  "publish.commit",
  "publish.abort",
)
AUDIT_EMITTER_OPERATION = "audit.execution_outcome"
WIRE_INTERNAL_PUBLISH_METHODS: tuple[str, ...] = (
  "reserve_autonomous_approval_publication",
  "commit_autonomous_approval_publication",
  "abort_autonomous_approval_publication",
)

# Creation-record owner fields. Must equal the closure storage_id, else 403
# runtime_identity_mismatch. Never select a database from the body.
CREATION_OWNER_FIELDS: Mapping[str, str] = {
  "request.create": "request.user_id",
  "request.create_raw_patch": "request.user_id",
  "request.create_or_get": "request.user_id",
  "request.create_or_get_prepared_bm": "request.user_id",
  "policy.record_root_turn": "tenant_storage_id",
  "executable.record": "tenant_storage_id",
  "executable.park": "tenant_storage_id",
  "policy.link_executable": "tenant_storage_id",
  "policy.reissue": "tenant_storage_id",
  "grant.create_persistent": "grant.user_id",
  "grant.create_delegation": "grant.delegator_user_id",
  "notification.enqueue": "request.user_id",
  "audit.execution_outcome": "request.user_id",
}

# While a reservation is active for the same outbox row, these writers return
# 429 reservation_active (retryable). They do not wait. fence and terminalize
# match the reservation decision_id, which is the store payload's approval_id
# (the existing approval id). The other three match
# (decision_id, tool_call_id, nonce). reconcile is not in this tuple:
# it blocks until the reservation releases, the remote form of BEGIN IMMEDIATE.
# A 429 would fail a reconcile the local lock would have waited out.
RESERVATION_BLOCKED_WRITERS: tuple[str, ...] = (
  "fence_persistent_grants_for_cancellation",
  "terminalize_pending_for_cancellation",
  "acknowledge_autonomous_approval_delivery",
  "record_autonomous_approval_delivery_failure",
  "quarantine_autonomous_approval_delivery",
)

# isinstance order on the server. Subclasses before their bases, or the wire
# name collapses and consumers cannot catch the subclass.
SERVER_EXCEPTION_MATCH_ORDER: tuple[str, ...] = (
  "ApprovalExecutableBindingError",
  "RootTurnRecordError",
  "ApprovalConstraintError",
  "ValueError",
  "PreparedBusinessModelError",
  "RawPatchAuthorizationError",
  "PersistentGrantCancellationFenced",
  "RuntimeError",
  "KeyError",
)

# pydantic.JsonValue, imported above. A quoted recursive alias makes model
# construction recurse. Binary fields are not JsonValue; they use StandardBase64.
def _standard_b64_bytes(value: object) -> bytes:
  """Raw domain bytes, or a standard base64 string with padding.

  pydantic Base64Bytes in this runtime decodes Python bytes as base64, so it
  cannot carry domain bytes. This is that serializer's equivalent: JSON is
  standard base64 with padding; non-UTF-8 bytes are legal; unpadded and
  non-alphabet input is rejected. TypeAdapter(bytes) is not this serializer.
  """

  if isinstance(value, str):
    try:
      return base64.b64decode(value, validate=True)
    except Exception as exc:
      raise ValueError("standard base64 with padding required") from exc
  if isinstance(value, bytes):
    return value
  raise ValueError("expected raw bytes or standard base64")


StandardBase64: TypeAlias = Annotated[
  bytes,
  BeforeValidator(_standard_b64_bytes),
  PlainSerializer(
    lambda value: base64.b64encode(value).decode("ascii"),
    return_type=str,
    when_used="json",
  ),
]


class ApprovalRemoteRequest(BaseModel):
  """No storage_id and no placement_generation. Both are closure-derived."""

  model_config = ConfigDict(extra="forbid", frozen=True)
  version: Literal["hank.approval-remote.v1"] = APPROVAL_REMOTE_PROTOCOL_VERSION
  operation: ApprovalRemoteOperation
  payload: dict[str, JsonValue]


class ApprovalRemoteResponse(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)
  version: Literal["hank.approval-remote.v1"] = APPROVAL_REMOTE_PROTOCOL_VERSION
  result: JsonValue


class _ErrorBody(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)
  version: Literal["hank.approval-remote.v1"] = APPROVAL_REMOTE_PROTOCOL_VERSION
  message: str


class ApprovalExecutableBindingErrorBody(_ErrorBody):
  """domain_code is the exception's .code. Client calls
  ApprovalExecutableBindingError(domain_code) and ignores message."""

  exception: Literal["ApprovalExecutableBindingError"]
  code: str
  domain_code: str
  retryable: bool = False

  @field_validator("code", "domain_code")
  @classmethod
  def producer_code(cls, value: str) -> str:
    return _adapter(_approval_store().ApprovalExecutableBindingErrorCode).validate_python(
      value, strict=True,
    )

  @model_validator(mode="after")
  def code_matches_domain(self) -> ApprovalExecutableBindingErrorBody:
    if self.code != self.domain_code:
      raise ValueError("code must equal domain_code")
    return self


class RootTurnRecordErrorBody(_ErrorBody):
  """domain_code is the exception's .code. Client calls
  RootTurnRecordError(domain_code) and ignores message."""

  exception: Literal["RootTurnRecordError"]
  code: str
  domain_code: str
  retryable: bool = False

  @field_validator("code", "domain_code")
  @classmethod
  def producer_code(cls, value: str) -> str:
    return _adapter(_approval_store().RootTurnRecordErrorCode).validate_python(
      value, strict=True,
    )

  @model_validator(mode="after")
  def code_matches_domain(self) -> RootTurnRecordErrorBody:
    if self.code != self.domain_code:
      raise ValueError("code must equal domain_code")
    return self


class StoreExceptionErrorBody(_ErrorBody):
  """Store exceptions other than the two coded subclasses.

  domain_code is absent. Client reconstructs ExceptionType(message).
  Status is not how the client picks the class: ValueError state conflicts
  cannot be split into 400 vs 409 by class. Any non-200 with this exception
  name re-raises that exception.
  """

  exception: Literal[
    "KeyError",
    "ValueError",
    "RuntimeError",
    "ApprovalConstraintError",
    "PersistentGrantCancellationFenced",
    "PreparedBusinessModelError",
    "RawPatchAuthorizationError",
  ]
  code: Literal["store_exception"]
  domain_code: None = None
  retryable: bool = False


class ApprovalRemoteIdentityMismatch(RuntimeError):
  """Creation-record owner id != closure storage_id. Never selects a DB."""

  status = 403
  retryable = False
  code = "runtime_identity_mismatch"

  def __init__(self, message: str = "runtime_identity_mismatch") -> None:
    self.code = "runtime_identity_mismatch"
    super().__init__(message)


class ApprovalRemoteMigrationIncomplete(RuntimeError):
  """This storage's approval RPCs are refused until migration completes."""

  status = 423
  retryable = True
  code = "migration_incomplete"

  def __init__(self, message: str = "migration_incomplete") -> None:
    self.code = "migration_incomplete"
    super().__init__(message)


class ApprovalRemoteReservationActive(RuntimeError):
  """A publication reservation blocks this writer. Retryable. Not a wait."""

  status = 429
  retryable = True
  code = "reservation_active"

  def __init__(self, message: str = "reservation_active") -> None:
    self.code = "reservation_active"
    super().__init__(message)


class ApprovalRemoteStaleReservation(RuntimeError):
  """Reservation generation != closure generation. Do not publish or release."""

  status = 409
  retryable = False
  code = "stale_placement_generation"

  def __init__(self, message: str = "stale_placement_generation") -> None:
    self.code = "stale_placement_generation"
    super().__init__(message)


class ApprovalRemoteReservationMismatch(RuntimeError):
  """A reservation field did not match the stored row."""

  status = 409
  retryable = False
  code = "reservation_mismatch"

  def __init__(self, message: str = "reservation_mismatch") -> None:
    self.code = "reservation_mismatch"
    super().__init__(message)


class IdentityMismatchErrorBody(_ErrorBody):
  exception: Literal["ApprovalRemoteIdentityMismatch"]
  code: Literal["runtime_identity_mismatch"]
  retryable: Literal[False] = False
  domain_code: None = None


class MigrationIncompleteErrorBody(_ErrorBody):
  exception: Literal["ApprovalRemoteMigrationIncomplete"]
  code: Literal["migration_incomplete"]
  retryable: Literal[True] = True
  domain_code: None = None


class ReservationActiveErrorBody(_ErrorBody):
  exception: Literal["ApprovalRemoteReservationActive"]
  code: Literal["reservation_active"]
  retryable: Literal[True] = True
  domain_code: None = None


class StaleReservationErrorBody(_ErrorBody):
  exception: Literal["ApprovalRemoteStaleReservation"]
  code: Literal["stale_placement_generation"]
  retryable: Literal[False] = False
  domain_code: None = None


class ReservationMismatchErrorBody(_ErrorBody):
  exception: Literal["ApprovalRemoteReservationMismatch"]
  code: Literal["reservation_mismatch"]
  retryable: Literal[False] = False
  domain_code: None = None


class ApprovalRemoteProtocolError(RuntimeError):
  """Transport, version, schema, operation-mismatch, or unknown-server-error."""

  def __init__(self, message: str, *, code: str = "malformed_envelope") -> None:
    self.code = code
    super().__init__(message)


class ProtocolErrorBody(_ErrorBody):
  exception: Literal["ApprovalRemoteProtocolError"]
  code: Literal[
    "malformed_envelope",
    "version_mismatch",
    "operation_mismatch",
    "unknown_exception",
    "unavailable",
    "unhandled",
  ]
  domain_code: None = None
  retryable: bool = False


ApprovalRemoteErrorBody: TypeAlias = Annotated[
  ApprovalExecutableBindingErrorBody
  | RootTurnRecordErrorBody
  | StoreExceptionErrorBody
  | IdentityMismatchErrorBody
  | MigrationIncompleteErrorBody
  | ReservationActiveErrorBody
  | StaleReservationErrorBody
  | ReservationMismatchErrorBody
  | ProtocolErrorBody,
  Field(discriminator="exception"),
]


class AutonomousPublicationReservation(BaseModel):
  """Publication lease. Structured key is
  (decision_id, tool_call_id, nonce, placement_generation).
  decision_id is the existing approval id under every retained grammar,
  including UUID and approval-*. placement_generation is the closure
  generation, a version, not an id and not a client-supplied field.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)
  decision_id: str
  tool_call_id: str
  nonce: str
  approved: bool
  placement_generation: int


class PublishReserveResult(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)
  reservation: AutonomousPublicationReservation


class PublishCommitResult(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)
  state: Literal["published"]


class PublishAbortResult(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)
  released: bool


class PreparedToolCallWire(BaseModel):
  """Named wire form of PreparedToolCall.

  prepared_input is PreparedToolCall.materialize_input(), a plain dict.
  exact_backend is the dataclass field, not part of that dict. Do not
  TypeAdapter(PreparedToolCall): prepared_input is a mappingproxy and pydantic
  raises PydanticSerializationError.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)
  prepared_input: dict[str, JsonValue]
  exact_backend: str | None = None


class ExecutableBindingOperandsWire(BaseModel):
  """Binding prepared_input and dispatch_context as plain dicts.

  Each is PreparedToolCall(prepared_input=mapping).materialize_input().
  Other ExecutableApprovalBinding fields serialize as their declared types.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)
  prepared_input: dict[str, JsonValue]
  dispatch_context: dict[str, JsonValue]


class CreateRawPatchAuthorizationBytesWire(BaseModel):
  """create_raw_patch_authorization.prepared_payload. Standard base64."""

  model_config = ConfigDict(extra="forbid", frozen=True)
  prepared_payload: StandardBase64


class BindRawPatchPreparedAuthorizationBytesWire(BaseModel):
  """bind_raw_patch_prepared_authorization.prepared_payload. Standard base64."""

  model_config = ConfigDict(extra="forbid", frozen=True)
  prepared_payload: StandardBase64


class ConsumeRawPatchAuthorizationBytesWire(BaseModel):
  """consume_raw_patch_authorization.receipt. Standard base64."""

  model_config = ConfigDict(extra="forbid", frozen=True)
  receipt: StandardBase64


class TransitionPreparedBusinessModelBytesWire(BaseModel):
  """transition_prepared_business_model_change.execution_receipt."""

  model_config = ConfigDict(extra="forbid", frozen=True)
  execution_receipt: StandardBase64 | None = None


def _dataclass_wire(
  producer: Any,
  name: str,
  overrides: Mapping[str, Any],
) -> type[BaseModel]:
  """Keep producer fields/types/defaults; override only transport representations."""

  hints = get_type_hints(producer)
  definitions: dict[str, Any] = {}
  for field in fields(producer):
    annotation = overrides.get(field.name, hints[field.name])
    if field.default is not MISSING:
      default = field.default
    elif field.default_factory is not MISSING:
      default = Field(default_factory=field.default_factory)
    else:
      default = ...
    definitions[field.name] = (annotation, default)
  return create_model(
    name,
    __config__=ConfigDict(extra="forbid", frozen=True),
    **definitions,
  )


PreparedBusinessModelChangeWire = _dataclass_wire(
  PreparedBusinessModelChange,
  "PreparedBusinessModelChangeWire",
  {
    "prepared_payload": StandardBase64,
    "execution_receipt": StandardBase64 | None,
  },
)


class RawPatchPreparedAuthorizationWire(BaseModel):
  """RawPatchPreparedAuthorization.prepared_payload as standard base64."""

  model_config = ConfigDict(extra="forbid", frozen=True)
  approval_id: str
  prepared_payload: StandardBase64
  prepared_payload_digest: str
  created_at: str


class RawPatchAuthorizationConsumptionWire(BaseModel):
  """RawPatchAuthorizationConsumption.receipt as standard base64."""

  model_config = ConfigDict(extra="forbid", frozen=True)
  approval_id: str
  change_set_id: str
  change_hash: str
  receipt: StandardBase64
  receipt_digest: str
  consumed_at: str


# The only base64 fields. payload: dict[str, JsonValue] does not mark which
# strings are base64; these named DTOs do. No other field is base64.
BASE64_WIRE_DTO: Mapping[str, str] = {
  "create_raw_patch_authorization.prepared_payload": "CreateRawPatchAuthorizationBytesWire",
  "bind_raw_patch_prepared_authorization.prepared_payload": "BindRawPatchPreparedAuthorizationBytesWire",
  "consume_raw_patch_authorization.receipt": "ConsumeRawPatchAuthorizationBytesWire",
  "transition_prepared_business_model_change.execution_receipt": "TransitionPreparedBusinessModelBytesWire",
  "PreparedBusinessModelChange.prepared_payload": "PreparedBusinessModelChangeWire",
  "PreparedBusinessModelChange.execution_receipt": "PreparedBusinessModelChangeWire",
  "RawPatchPreparedAuthorization.prepared_payload": "RawPatchPreparedAuthorizationWire",
  "RawPatchAuthorizationConsumption.receipt": "RawPatchAuthorizationConsumptionWire",
}


class ApprovalRemoteTransport(Protocol):
  """Async placement-bound transport. AgentCore tunnel only.

  Constructed bound to one placement. Posts to the tunnel URL projected as
  HANK_APPROVALS_EGRESS_URL by AgentCoreRuntimeBackend._environment, whose
  value is APPROVALS_EGRESS_URL (http://127.0.0.1:8090/egress/approvals).
  The tunnel handler closure already selected the tenant. This transport
  sends no storage_id, no placement_generation, and no claim headers.
  """

  async def post(
    self, operation: ApprovalRemoteOperation, body: bytes,
  ) -> tuple[int, bytes]: ...


class SyncApprovalRemoteTransport(Protocol):
  """Blocking placement-bound transport with a finite timeout.

  Safe to call under a running event loop. Never a tunnel coroutine
  submitted to the caller's loop, and never asyncio.run_coroutine_threadsafe
  of the tunnel onto that loop.

  AgentCore posts to the HANK_APPROVALS_EGRESS_URL tunnel (APPROVALS_EGRESS_URL).
  LocalProcess has no tunnel and no 8090 listener. It posts to the edge
  router URL whose path is /egress/approvals/<operation>, attaching a fresh
  derived claim on every request via
  claim_signing_authority.UserClaimHttpAuth or
  agent_api_claim_http_headers. A claim captured at process start is not
  fresh. There is no unauthenticated listener that trusts body identity.
  """

  def post(
    self, operation: ApprovalRemoteOperation, body: bytes,
  ) -> tuple[int, bytes]: ...


class ApprovalExecutionAuditEmitter(Protocol):
  """Wire adapter matching ApprovalAuditEmitter.emit_execution_outcome.

  boundary_sanitizer is not a parameter. The runtime applies
  tool_input_redactor and boundary_sanitizer before transport; the payload
  carries redacted args only. The edge emitter owns HMAC signing.
  Callers pass these arguments by keyword, as emit_execution_audit does.
  """

  async def emit_execution_outcome(
    self,
    request: ApprovalRequest,
    raw_tool_args: dict[str, Any],
    outcome: ExecutionAuditOutcome,
    error_summary: str | None = None,
  ) -> None: ...


class ApprovalStoreAuditEmitter(Protocol):
  """Optional consumer audit emitter. Compatible with both emitters.

  approval_store._ApprovalAuditEmitter requires emit_audit_for_lifecycle_event
  and does not declare emit_execution_outcome. ApprovalExecutionAuditEmitter
  requires emit_execution_outcome and not emit_audit_for_lifecycle_event.
  Requiring either method would reject the other emitter, so this protocol
  requires neither. SQLiteApprovalStore.audit_emitter returns
  _ApprovalAuditEmitter | None. Consumers probe
  getattr(store, "audit_emitter", None) and
  getattr(emitter, "emit_execution_outcome", None) and handle None.
  The property itself may be None.
  """


# Amended ApprovalRequestStore members. Current protocol members except
# maintain_pending, plus store methods product consumers call that are not
# already on that protocol. Not every ApprovalRemoteClient method. Not
# WIRE_INTERNAL_PUBLISH_METHODS. Not schedule_approval_notification_delivery
# (not an RPC; the runtime getattr no-op stays). Not store methods only tests
# or the store itself call.
AMENDED_APPROVAL_REQUEST_STORE_METHODS: frozenset[str] = frozenset({
  "audit_emitter",
  "create",
  "create_or_get_by_tool_call_id",
  "locate_request",
  "_locate_request_for_decision",
  "record_executable_binding",
  "park_executable_binding",
  "_read_executable_binding_for_executor",
  "_read_completed_executable_result_for_executor",
  "_claim_executable_binding_for_executor",
  "_complete_executable_binding_for_executor",
  "get",
  "get_by_tool_call_id",
  "update_request",
  "transition_state",
  "terminalize_pending_for_cancellation",
  "abort_unpublished_approval",
  "record_vote",
  "create_persistent_grant",
  "find_persistent_grant",
  "revoke_persistent_grant",
  "revoke_persistent_grants_for_approval",
  "fence_persistent_grants_for_cancellation",
  "create_delegation_grant",
  "get_delegation_grant",
  "claim_delegation_grant",
  "revoke_delegation_grant",
  "reconcile_prepared_business_model_change",
  "create_raw_patch_authorization",
  "create_or_get_with_prepared_business_model_change",
  "record_vote_with_autonomous_delivery",
  "list_parked_requests",
  "record_policy_generation",
  "record_root_turn",
  "_read_root_turn_for_executor",
  "_locate_root_turn_for_approval",
  "link_root_turn_executable_binding",
  "reissue_parked_approval",
  "get_prepared_business_model_change",
  "transition_prepared_business_model_change",
  "enqueue_pending_approval_notification",
  "get_autonomous_approval_delivery",
  "ensure_autonomous_approval_delivery_audited",
  "acknowledge_autonomous_approval_delivery",
  "record_autonomous_approval_delivery_failure",
  "quarantine_autonomous_approval_delivery",
  "list_pending_autonomous_approval_deliveries",
  "autonomous_approval_delivery_recovery_window",
  "autonomous_approval_delivery_append_transaction",
  "autonomous_approval_delivery_duplicate_transaction",
  "retry_failed_approval_notifications",
  "resolve_raw_patch_authorization_reference",
  "get_raw_patch_prepared_authorization",
  "consume_raw_patch_authorization",
  "claim_raw_patch_authorization",
  "release_raw_patch_authorization_claim",
  "heartbeat_raw_patch_authorization_claim",
  "get_raw_patch_authorization_consumption",
})


class ApprovalRemoteClient(Protocol):
  """Wire-internal client. Never a consumer annotation.

  W1d's remote class implements approval_store.ApprovalRequestStore as
  amended in the same cutover, not this protocol. The amendment is
  AMENDED_APPROVAL_REQUEST_STORE_METHODS only: store methods product
  consumers call, plus an optional audit_emitter. It does not include
  reserve/commit/abort_autonomous_approval_publication; those stay
  wire-internal on this protocol. get and get_by_tool_call_id widen to the
  owner_user_id keyword already on SQLiteApprovalStore. maintain_pending
  leaves ApprovalRequestStore. maintain_pending_loop is retyped to the
  edge-owned SQLite store. Do not add a second consumer protocol or cast shim.

  audit_emitter on the amended protocol is ApprovalStoreAuditEmitter | None,
  not a required ApprovalExecutionAuditEmitter. Both the local
  _ApprovalAuditEmitter and the remote adapter are assignable. Consumers
  probe getattr and handle None.

  Async methods use ApprovalRemoteTransport on AgentCore. On LocalProcess
  they use SyncApprovalRemoteTransport (fresh derived claim, edge router
  URL). Sync methods always use SyncApprovalRemoteTransport.
  """

  @property
  def audit_emitter(self) -> ApprovalStoreAuditEmitter | None: ...

  async def create(self, request: ApprovalRequest) -> ApprovalRequest: ...
  async def create_raw_patch_authorization(
    self, request: ApprovalRequest, *, prepared_payload: bytes,
  ) -> ApprovalRequest: ...
  async def create_or_get_by_tool_call_id(
    self, request: ApprovalRequest,
  ) -> tuple[ApprovalRequest, bool]: ...
  async def create_or_get_with_prepared_business_model_change(
    self, request: ApprovalRequest, record: PreparedBusinessModelChange,
  ) -> tuple[ApprovalRequest, PreparedBusinessModelChange, bool]: ...
  async def get(
    self, approval_id: str, *,
    owner_user_id: str | tuple[str, ...] | list[str] | set[str] | None = None,
  ) -> ApprovalRequest | None: ...
  async def get_by_tool_call_id(
    self, tool_call_id: str, *,
    owner_user_id: str | tuple[str, ...] | list[str] | set[str] | None = None,
  ) -> ApprovalRequest | None: ...
  async def update_request(self, request: ApprovalRequest) -> ApprovalRequest: ...
  async def transition_state(
    self, approval_id: str, state: ApprovalState, *,
    expected_state_version: int | None = None, expires_at: datetime | None = None,
    decider_id: str | None = None, decider_role: str | None = None,
    decision: str | None = None, decision_reason: str | None = None,
  ) -> ApprovalRequest: ...
  async def terminalize_pending_for_cancellation(
    self, approval_id: str, *, expected_tool_call_id: str, expected_user_id: str,
    expected_request_id: str, expected_run_id: str, expected_session_id: str,
    expected_channel: str | None, decider_id: str | None = None,
    decider_role: str | None = None, decision_reason: str,
    autonomous_delivery: Mapping[str, Any] | None = None,
  ) -> tuple[ApprovalRequest, bool, bool]: ...
  async def abort_unpublished_approval(
    self, approval_id: str, *, expected_tool_call_id: str, expected_user_id: str,
    expected_request_id: str, expected_run_id: str, expected_session_id: str,
    expected_channel: str | None, decision_reason: str,
  ) -> tuple[ApprovalRequest, bool, bool]: ...
  async def record_vote(self, approval_id: str, vote: ApprovalVote) -> ApprovalRequest: ...
  async def record_vote_with_autonomous_delivery(
    self, approval_id: str, vote: ApprovalVote, *, delivery: Mapping[str, Any],
  ) -> ApprovalRequest: ...

  async def locate_request(
    self, *, approval_id: str, tenant_storage_id: str,
  ) -> ApprovalRequestLocator | None: ...
  async def list_parked_requests(self, tenant_storage_id: str) -> list[ApprovalRequest]: ...
  async def _locate_request_for_decision(
    self, *, tenant_storage_id: str, approval_id: str | None = None,
    tool_call_id: str | None = None, nonce: str | None = None,
  ) -> tuple[ApprovalRequest, ApprovalRequestLocator, str | None] | None: ...
  async def _locate_root_turn_for_approval(
    self, *, approval_id: str, tenant_storage_id: str,
  ) -> tuple[str, str, str, str] | None:
    """(job_id, tool_call_id, nonce, turn_id). Not a RootTurnRecord."""

  async def record_policy_generation(
    self, *, policy_id: str, policy_version: str, policy_bundle_hash: str,
    make_current: bool = True,
  ) -> PolicyGeneration: ...
  async def read_policy_generation(self, *, generation_ref: str) -> PolicyGeneration | None: ...
  async def current_policy_generation(self) -> PolicyGeneration | None: ...
  async def record_root_turn(
    self, *, job_id: str, turn_id: str, tenant_storage_id: str,
    capability_bind: Mapping[str, object], credential_generation_ref: str | None,
    effective_grant: Mapping[str, object], research_binding: Mapping[str, object] | None,
    selection_snapshot: Mapping[str, object], approval_route_kind: str,
    policy_generation_ref: str, created_at: datetime | None = None,
  ) -> None: ...
  async def _read_root_turn_for_executor(
    self, *, job_id: str, turn_id: str, tenant_storage_id: str,
  ) -> RootTurnRecord | None: ...
  async def link_root_turn_executable_binding(
    self, *, job_id: str, turn_id: str, tool_call_id: str, nonce: str,
    tenant_storage_id: str,
  ) -> None: ...
  async def reissue_parked_approval(
    self, *, approval_id: str, job_id: str, tool_call_id: str, nonce: str,
    turn_id: str, tenant_storage_id: str, generation: PolicyGeneration,
  ) -> tuple[ApprovalRequest, str, str, bool]: ...

  async def record_executable_binding(
    self, *, approval_id: str, job_id: str, tool_call_id: str, nonce: str,
    tenant_storage_id: str, tool_name: str, prepared_call: PreparedToolCall,
    dispatch_context: Mapping[str, object], credential_generation_ref: str | None,
  ) -> None: ...
  async def park_executable_binding(
    self, *, approval_id: str, job_id: str, tool_call_id: str, nonce: str,
    tenant_storage_id: str, tool_name: str, prepared_call: PreparedToolCall,
    dispatch_context: Mapping[str, object], credential_generation_ref: str | None,
    policy_generation_ref: str | None = None,
  ) -> str: ...
  async def _read_executable_binding_for_executor(
    self, *, job_id: str, tool_call_id: str, nonce: str, tenant_storage_id: str,
  ) -> tuple[ExecutableApprovalBinding, ApprovalRequest]: ...
  async def _read_completed_executable_result_for_executor(
    self, *, binding: ExecutableApprovalBinding,
  ) -> ToolResult | None: ...
  async def _claim_executable_binding_for_executor(
    self, *, binding: ExecutableApprovalBinding, decision_state_version: int,
    claim_token: str,
  ) -> ToolResult | None: ...
  async def _complete_executable_binding_for_executor(
    self, *, binding: ExecutableApprovalBinding, decision_state_version: int,
    claim_token: str, tool_result: ToolResult,
  ) -> None: ...

  async def create_persistent_grant(self, grant: PersistentGrant) -> PersistentGrant: ...
  async def find_persistent_grant(
    self, *, user_id: str, tool_name: str, scope_hint: str,
    now: datetime | None = None, approval_constraint: str = "standard",
    approval_reuse_mode: ApprovalReuseMode = "legacy",
    approval_reuse_key: str | None = None,
  ) -> PersistentGrant | None: ...
  async def revoke_persistent_grant(
    self, grant_id: str, *, revoked_at: datetime | None = None,
  ) -> None: ...
  async def revoke_persistent_grants_for_approval(
    self, approval_id: str, *, revoked_at: datetime | None = None,
  ) -> int: ...
  async def fence_persistent_grants_for_cancellation(
    self, approval_id: str, *, expected_tool_call_id: str, expected_user_id: str,
    expected_request_id: str, expected_run_id: str, expected_session_id: str,
    expected_channel: str | None,
  ) -> tuple[ApprovalRequest, bool]: ...
  async def create_delegation_grant(self, grant: DelegationGrant) -> DelegationGrant: ...
  async def get_delegation_grant(self, delegation_id: str) -> DelegationGrant | None: ...
  async def claim_delegation_grant(
    self, *, delegation_id: str, bound_relay_request_id: str,
    bound_excel_session_id: str, now: datetime | None = None,
  ) -> DelegationGrant | None: ...
  async def revoke_delegation_grant(
    self, delegation_id: str, *, revoked_at: datetime | None = None,
  ) -> None: ...

  async def insert_or_verify_prepared_business_model_change(
    self, record: PreparedBusinessModelChange,
  ) -> tuple[PreparedBusinessModelChange, bool]: ...
  async def get_prepared_business_model_change(
    self, *, caller_kind: str, user_scope: str, idempotency_locator: str,
  ) -> PreparedBusinessModelChange | None: ...
  async def transition_prepared_business_model_change(
    self, *, caller_kind: str, user_scope: str, idempotency_locator: str,
    expected: PreparedBusinessModelLifecycle, target: PreparedBusinessModelLifecycle,
    approval_id: str | None = None, approval_chain_id: str | None = None,
    execution_receipt: bytes | None = None, restoration_digest: str | None = None,
    checkpoint_id: str | None = None, consumed_at: str | None = None,
  ) -> PreparedBusinessModelChange: ...
  async def reconcile_prepared_business_model_change(
    self, *, caller_kind: str, user_scope: str, idempotency_locator: str,
  ) -> TargetedPreparedReconciliationResult: ...

  async def resolve_raw_patch_authorization_reference(
    self, reference: str,
  ) -> ApprovalRequest | None: ...
  async def bind_raw_patch_prepared_authorization(
    self, *, approval_id: str, prepared_payload: bytes,
  ) -> tuple[RawPatchPreparedAuthorization, bool]: ...
  async def get_raw_patch_prepared_authorization(
    self, *, approval_id: str,
  ) -> RawPatchPreparedAuthorization | None: ...
  async def consume_raw_patch_authorization(
    self, *, approval_id: str, approval_chain_id: str, authorization_mode: str,
    grant_reference: str | None, cache_reference: str | None, user_id: str,
    tool_name: str, args_hash: str, change_set_id: str, change_hash: str,
    base_vector_hash: str, reviewed_change_binding_digest: str,
    execution_semantics_digest: str, prepared_payload_digest: str,
    claim_token: str, receipt: bytes,
  ) -> tuple[RawPatchAuthorizationConsumption, bool]: ...
  async def claim_raw_patch_authorization(
    self, *, approval_id: str, approval_chain_id: str, authorization_mode: str,
    grant_reference: str | None, cache_reference: str | None, user_id: str,
    tool_name: str, args_hash: str, change_set_id: str, change_hash: str,
    base_vector_hash: str, reviewed_change_binding_digest: str,
    execution_semantics_digest: str, prepared_payload_digest: str,
  ) -> RawPatchAuthorizationClaim: ...
  async def release_raw_patch_authorization_claim(
    self, *, approval_id: str, approval_chain_id: str, authorization_mode: str,
    grant_reference: str | None, cache_reference: str | None, user_id: str,
    tool_name: str, args_hash: str, change_set_id: str, change_hash: str,
    base_vector_hash: str, reviewed_change_binding_digest: str,
    execution_semantics_digest: str, prepared_payload_digest: str, claim_token: str,
  ) -> None: ...
  async def heartbeat_raw_patch_authorization_claim(
    self, *, approval_id: str, claim_token: str, lease_seconds: float = 120.0,
  ) -> RawPatchAuthorizationClaim: ...
  async def get_raw_patch_authorization_consumption(
    self, *, approval_id: str,
  ) -> RawPatchAuthorizationConsumption | None: ...

  async def get_autonomous_approval_delivery(
    self, approval_id: str, *, tool_call_id: str, nonce: str,
  ) -> dict[str, Any] | None: ...
  async def ensure_autonomous_approval_delivery_audited(
    self, approval_id: str, *, tool_call_id: str, nonce: str,
  ) -> dict[str, Any]: ...
  async def acknowledge_autonomous_approval_delivery(
    self, approval_id: str, *, task_id: str, control_run_id: str, session_id: str,
    channel_id: str, tool_call_id: str, nonce: str, approved: bool, decided_at_ns: int,
  ) -> dict[str, Any]: ...
  async def record_autonomous_approval_delivery_failure(
    self, approval_id: str, *, tool_call_id: str, nonce: str, error: str,
  ) -> dict[str, Any]: ...
  async def quarantine_autonomous_approval_delivery(
    self, approval_id: str, *, tool_call_id: str, nonce: str, error: str,
  ) -> dict[str, Any]: ...
  async def list_pending_autonomous_approval_deliveries(
    self, *, after_sequence: int = 0, through_sequence: int | None = None,
  ) -> list[dict[str, Any]]: ...
  async def autonomous_approval_delivery_high_water(self) -> int: ...
  async def autonomous_approval_delivery_recovery_window(self) -> dict[str, int]: ...
  def reconcile_autonomous_approval_delivery_duplicate(
    self, approval_id: str, *, tool_call_id: str, nonce: str, approved: bool,
  ) -> dict[str, Any]:
    """Sync. Uses SyncApprovalRemoteTransport and publish.reconcile_duplicate.
    One atomic CAS. No reserve. Not async: the duplicate context manager
    calls it synchronously before yield."""
    ...

  def reserve_autonomous_approval_publication(
    self, decision_id: str, *, tool_call_id: str, nonce: str, approved: bool,
  ) -> PublishReserveResult:
    """Sync. Wire result is PublishReserveResult, not a bare reservation.
    Wire-internal. Not an ApprovalRequestStore method.

    Before inserting a reservation, performs the local append transaction's
    pre-yield checks and raises the same exceptions. A failed check does not
    insert a reservation, so the child-inbox append does not run:
    - type(approved) is not bool: ValueError
      "autonomous approval delivery approved is invalid"
    - outbox row missing: KeyError
      "autonomous approval delivery outbox record not found"
    - row.approved is not the requested approved: ValueError
      "autonomous approval delivery decision changed"
    - audit_state != "ready": RuntimeError
      "autonomous approval delivery audit receipt is not ready"
    - state != "pending": RuntimeError
      "autonomous approval delivery is not pending"
    - approved and a cancellation fence exists for that approval id:
      PersistentGrantCancellationFenced
      "approved autonomous delivery is fenced for cancellation"

    Idempotent for the same (decision_id, tool_call_id, nonce, approved)
    under the closure placement generation when those checks still pass:
    returns the existing reservation. A lost reserve response must not 429
    against its own reservation. placement_generation on the returned
    reservation is the closure generation."""
    ...

  def commit_autonomous_approval_publication(
    self, reservation: AutonomousPublicationReservation,
  ) -> PublishCommitResult:
    """Sync. Wire result is PublishCommitResult, never null.
    Wire-internal. Not an ApprovalRequestStore method.

    Re-verify every reservation field against the stored row and the closure
    generation. Generation mismatch is 409 stale_placement_generation and
    does not publish. Any other field mismatch is 409 reservation_mismatch.
    First success performs the local append CAS: the outbox row must have
    state pending and audit_state ready; the update sets state to published,
    increments attempt_count by 1, clears last_error, and sets updated_at
    and published_at to the same timestamp. A repeated commit of that same
    reservation returns state published and does not increment attempt_count
    again. Commit after abort (row gone) is 409 reservation_mismatch. CAS
    failure while the reservation is still active is RuntimeError, not a
    silent published. Reconcile cannot cause that failure: it does not
    publish while the reservation is active."""
    ...

  def abort_autonomous_approval_publication(
    self, reservation: AutonomousPublicationReservation,
  ) -> PublishAbortResult:
    """Sync. Wire result is PublishAbortResult; released is not discarded.
    Wire-internal. Not an ApprovalRequestStore method.

    Re-verify every reservation field against the stored row and the closure
    generation. Generation mismatch against a still-stored row is 409
    stale_placement_generation and does not release that row. Abort of an
    active matching reservation returns released true. Abort after commit of
    the same reservation returns released false and does not reverse the
    published CAS. Abort when the row is already gone (prior abort, or
    observed stop/replace of that generation) returns released false and is
    not an error."""
    ...

  def autonomous_approval_delivery_append_transaction(
    self, approval_id: str, *, tool_call_id: str, nonce: str, approved: bool,
  ) -> ContextManager[None]:
    """Sync context manager over SyncApprovalRemoteTransport. No operation
    of its own. Reserve, yield, commit on normal exit, abort on
    BaseException. If commit raises, abort that reservation and then
    re-raise: there is no wall-clock expiry, so a failed commit must not
    pin the reservation. Duplicate reconcile is not this path."""
    ...

  def autonomous_approval_delivery_duplicate_transaction(
    self, approval_id: str, *, tool_call_id: str, nonce: str, approved: bool,
  ) -> ContextManager[None]:
    """Sync context manager. No operation of its own. Calls
    reconcile_autonomous_approval_delivery_duplicate synchronously before
    yield (publish.reconcile_duplicate). No reserve, commit, or abort."""
    ...

  async def enqueue_pending_approval_notification(
    self, request: ApprovalRequest,
  ) -> dict[str, Any] | None: ...
  async def retry_failed_approval_notifications(self, approval_id: str) -> dict[str, Any]: ...
  async def get_approval_notification_projection(
    self, approval_id: str,
  ) -> dict[str, Any] | None: ...
  async def list_approval_notification_outbox(
    self, approval_id: str | None = None,
  ) -> list[dict[str, Any]]: ...


# --- Rules that are not expressible as signatures ---
#
# Closure. W1c's make_egress_handler and EgressFactory gain the placement
# generation they do not take today (current signature is identity,
# storage_id, config). The closure selects the per-storage DB and mints
# AutonomousPublicationReservation.placement_generation. A body field naming
# another tenant or generation is not a selector; those fields are absent.
#
# URL. POST <base>/<operation>. AgentCore base is HANK_APPROVALS_EGRESS_URL.
# LocalProcess base is the edge router's /egress/approvals. The URL's final
# segment must equal envelope.operation, else 400 operation_mismatch
# (ApprovalRemoteProtocolError). 200 parses ApprovalRemoteResponse. Any
# non-200 parses ApprovalRemoteErrorBody and re-raises the named exception
# regardless of which non-200 status arrived. Unknown exception names or
# codes are ApprovalRemoteProtocolError, never a coerced ValueError or a
# store success. Body parse failure is ApprovalRemoteProtocolError.
#
# Advisory server status, not a client branch: KeyError 404; migration 423
# retryable; reservation_active 429 retryable; runtime_identity_mismatch 403;
# malformed_envelope / version_mismatch / operation_mismatch 400; unavailable
# 503; unhandled 500; stale_placement_generation and reservation_mismatch 409.
# ValueError may be 400 or 409. The client still re-raises ValueError for
# either. Other store exceptions are sent as 409; the client still re-raises
# by exception name on any non-200.
#
# Reconstruction. ApprovalExecutableBindingError(domain_code) and
# RootTurnRecordError(domain_code), ignoring message. Callers compare
# exc.code, not exc.domain_code. Other store exceptions reconstruct as
# ExceptionType(message). Non-store refusals reconstruct as the named
# contract exception (identity, migration, reservation active, stale
# reservation, reservation mismatch), whose .code and .status are the class
# attributes above.
#
# Payload. Keys are exactly the method parameter names excluding self,
# including keyword-only names. No extra nesting. A method with no parameters
# sends {}. Each value serializes per that parameter's declared type, and
# ApprovalRemoteResponse.result serializes per the return annotation, except:
#   - PreparedToolCall uses PreparedToolCallWire (materialize_input() plus
#     exact_backend). Reconstruct PreparedToolCall(prepared_input=...,
#     exact_backend=...).
#   - ExecutableApprovalBinding.prepared_input and .dispatch_context use
#     ExecutableBindingOperandsWire (plain dicts from materialize_input()).
#   - ToolResult is the str returned by approval_store._executable_result_json
#     (canonical [{'kind': kind, 'value': value}, error], kind in json /
#     AgentCompletionEnvelope / TaskResult). Restore with
#     _restore_executable_result on element 0, paired with element 1. None
#     is JSON null, not a string. Do not TypeAdapter(ToolResult): that drops
#     the envelope kind. Those two functions stay authoritative; do not
#     reimplement them.
#   - The eight bytes fields in BASE64_WIRE_DTO use the named DTO's
#     StandardBase64 field (standard base64 with padding; the Base64Bytes
#     equivalent, because this runtime's Base64Bytes decodes Python bytes).
#     No other field is base64. set[str] is a JSON array, restored as set,
#     via the declared type's adapter. datetime is that adapter's ISO-8601 form.
#
# Publication. Reservation key is (decision_id, tool_call_id, nonce,
# placement_generation). decision_id is the existing approval id under every
# retained grammar, including UUID and approval-*. placement_generation is a
# version, not an id. No wall-clock expiry. Reservations release only
# on commit, abort, or observed stop/replace of their placement generation.
# publish.reconcile_duplicate stays one atomic CAS and needs no reserve.
# It is not a 429 writer. While a reservation is active for the same
# (decision_id, tool_call_id, nonce), the approval service blocks reconcile
# and does not invoke reconcile_autonomous_approval_delivery_duplicate until
# that reservation is released. That is the remote form of BEGIN IMMEDIATE
# against the append transaction: reconcile cannot publish, increment
# attempt_count, or return a projection during the child-inbox append. A
# transport timeout while blocked is not a publish and not an increment.
# After release, the store method is the local one:
#   - approved not bool: ValueError, no increment
#   - row missing: KeyError, no increment
#   - decision changed: ValueError, no increment
#   - audit_state not ready: RuntimeError, no increment
#   - state published or acknowledged: return that projection, attempt_count
#     unchanged
#   - state not pending: RuntimeError, no increment
#   - approved and fenced: PersistentGrantCancellationFenced, no increment
#   - else one CAS: state published, attempt_count + 1, last_error cleared,
#     updated_at and published_at set to the same timestamp, return the
#     projection. rowcount != 1 is RuntimeError and no partial publish.
# RESERVATION_BLOCKED_WRITERS return 429 while a reservation for that row is
# active. They do not wait. Other writers are not blocked and do not wait.
#
# Notifications. notification.enqueue and notification.retry_failed are RPCs.
# After each, the edge approval service calls
# SQLiteApprovalStore.schedule_approval_notification_delivery on its own
# per-storage store, in the runtime_approval_service loop, and owns
# deliver_pending_approval_notifications. Those are not runtime RPCs. The
# runtime getattr(store, 'schedule_approval_notification_delivery') no-op is
# expected; the edge must not no-op.
#
# Not RPCs: maintain_pending, deliver_pending_approval_notifications,
# schedule_approval_notification_delivery, and their loops. Edge-global
# maintenance runs against each tenant store on the approval-service loop.
#
# Execution audit. audit.execution_outcome is the missing family-8 operation.
# It is not a store method. The server dispatches it to that storage's
# configured audit_emitter.emit_execution_outcome. The runtime redacts before
# transport. The edge emitter HMAC-signs. The amended ApprovalRequestStore
# exposes optional audit_emitter (ApprovalStoreAuditEmitter | None). Consumers
# still probe getattr and skip when it is None. An explicit
# audit.execution_outcome RPC with no configured emitter is unavailable, not
# a skipped success.


_BYTES_PARAMETER_DTO: dict[tuple[str, str], type[BaseModel]] = {
  ("create_raw_patch_authorization", "prepared_payload"): CreateRawPatchAuthorizationBytesWire,
  ("bind_raw_patch_prepared_authorization", "prepared_payload"): BindRawPatchPreparedAuthorizationBytesWire,
  ("consume_raw_patch_authorization", "receipt"): ConsumeRawPatchAuthorizationBytesWire,
  ("transition_prepared_business_model_change", "execution_receipt"): TransitionPreparedBusinessModelBytesWire,
}
_PROTOCOL_CODES = frozenset({
  "malformed_envelope",
  "version_mismatch",
  "operation_mismatch",
  "unknown_exception",
  "unavailable",
  "unhandled",
})
_PROTOCOL_STATUS: dict[str, int] = {
  "malformed_envelope": 400,
  "version_mismatch": 400,
  "operation_mismatch": 400,
  "unknown_exception": 400,
  "unavailable": 503,
  "unhandled": 500,
}
_STORE_STATUS: dict[str, int] = {
  "KeyError": 404,
  "ValueError": 409,
  "RuntimeError": 409,
  "ApprovalConstraintError": 409,
  "PersistentGrantCancellationFenced": 409,
  "PreparedBusinessModelError": 409,
  "RawPatchAuthorizationError": 409,
  "ApprovalExecutableBindingError": 409,
  "RootTurnRecordError": 409,
}
_adapters: dict[Any, TypeAdapter[Any]] = {}
_hint_cache: dict[Any, dict[str, Any]] = {}
_specs: dict[str, "_CallSpec"] | None = None
_error_body_adapter: TypeAdapter[Any] | None = None


class _CallSpec:
  __slots__ = ("operation", "method_name", "function")

  def __init__(self, operation: str, method_name: str, function: Any) -> None:
    self.operation = operation
    self.method_name = method_name
    self.function = function


def _approval_store() -> Any:
  import agent_gateway.approval_store as store
  return store


def _annotation_namespace() -> dict[str, Any]:
  store = _approval_store()
  namespace = dict(globals())
  namespace.update({
    "ApprovalRequestLocator": store.ApprovalRequestLocator,
    "ExecutableApprovalBinding": store.ExecutableApprovalBinding,
    "PolicyGeneration": store.PolicyGeneration,
    "RootTurnRecord": store.RootTurnRecord,
    "TargetedPreparedReconciliationResult": store.TargetedPreparedReconciliationResult,
  })
  return namespace


def _hints(function: Any) -> dict[str, Any]:
  cached = _hint_cache.get(function)
  if cached is None:
    cached = get_type_hints(function, globalns=_annotation_namespace())
    _hint_cache[function] = cached
  return cached


def _parameter_names(function: Any) -> tuple[str, ...]:
  return tuple(
    name for name, parameter in inspect.signature(function).parameters.items()
    if name != "self" and parameter.kind is not inspect.Parameter.VAR_POSITIONAL
    and parameter.kind is not inspect.Parameter.VAR_KEYWORD
  )


def _call_specs() -> dict[str, _CallSpec]:
  global _specs
  if _specs is None:
    specs: dict[str, _CallSpec] = {}
    for operation, method in OPERATION_METHODS.items():
      if operation == AUDIT_EMITTER_OPERATION:
        function = ApprovalExecutionAuditEmitter.emit_execution_outcome
        method_name = "emit_execution_outcome"
      else:
        function = getattr(BoundApprovalRemoteClient, method)
        method_name = method
      specs[operation] = _CallSpec(operation, method_name, function)
    _specs = specs
  return _specs


def operation_parameter_names(operation: str) -> tuple[str, ...]:
  """Parameter names excluding self. Payload keys are exactly these names."""

  return _parameter_names(_call_specs()[operation].function)


def _adapter(annotation: Any) -> TypeAdapter[Any]:
  cached = _adapters.get(annotation)
  if cached is None:
    cached = TypeAdapter(annotation)
    _adapters[annotation] = cached
  return cached


def _error_adapter() -> TypeAdapter[Any]:
  global _error_body_adapter
  if _error_body_adapter is None:
    _error_body_adapter = TypeAdapter(ApprovalRemoteErrorBody)
  return _error_body_adapter


def _union_members(annotation: Any) -> tuple[Any, ...] | None:
  import types
  from typing import Union

  origin = get_origin(annotation)
  if origin in (Union, types.UnionType):
    return get_args(annotation)
  return None


def _allows_none(annotation: Any) -> bool:
  if annotation is type(None) or annotation is None:
    return True
  members = _union_members(annotation)
  return members is not None and type(None) in members


def _non_optional(annotation: Any) -> Any:
  members = _union_members(annotation)
  if members is None:
    return annotation
  rest = tuple(member for member in members if member is not type(None))
  if len(rest) == 1 and len(rest) != len(members):
    return rest[0]
  return annotation


def _is_enum(annotation: Any) -> bool:
  from enum import Enum

  return isinstance(annotation, type) and issubclass(annotation, Enum)


def _model_bytes(model: BaseModel) -> bytes:
  return model.model_dump_json().encode("utf-8")


def _exception_message(exc: BaseException) -> str:
  if exc.args and isinstance(exc.args[0], str):
    return exc.args[0]
  return str(exc)


def _loads_object(body: bytes) -> dict[str, Any]:
  try:
    value = json.loads(body)
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ApprovalRemoteProtocolError(
      "malformed envelope", code="malformed_envelope",
    ) from exc
  if type(value) is not dict:
    raise ApprovalRemoteProtocolError(
      "malformed envelope", code="malformed_envelope",
    )
  return value


def _require_version(value: Mapping[str, Any]) -> None:
  if value.get("version") != APPROVAL_REMOTE_PROTOCOL_VERSION:
    raise ApprovalRemoteProtocolError(
      "version mismatch", code="version_mismatch",
    )


def _encode_parameter_bytes(value: Any, method: str, field: str) -> JsonValue:
  dto = _BYTES_PARAMETER_DTO[(method, field)]
  parsed = dto.model_validate({field: value})
  dumped = parsed.model_dump(mode="json")
  encoded = dumped[field]
  if encoded is not None and not isinstance(encoded, str):
    raise ApprovalRemoteProtocolError(
      "binary field must be standard base64", code="malformed_envelope",
    )
  return encoded


def _decode_parameter_bytes(value: object, method: str, field: str) -> bytes | None:
  dto = _BYTES_PARAMETER_DTO[(method, field)]
  parsed = dto.model_validate({field: value})
  return getattr(parsed, field)


def _encode_prepared_tool_call(value: PreparedToolCall) -> dict[str, JsonValue]:
  wire = PreparedToolCallWire(
    prepared_input=cast(dict[str, JsonValue], value.materialize_input()),
    exact_backend=value.exact_backend,
  )
  return wire.model_dump(mode="json")


def _decode_prepared_tool_call(value: object) -> PreparedToolCall:
  wire = PreparedToolCallWire.model_validate(value)
  return PreparedToolCall(
    prepared_input=wire.prepared_input,
    exact_backend=wire.exact_backend,
  )


def _dataclass_values(value: Any) -> dict[str, Any]:
  values = {field.name: getattr(value, field.name) for field in fields(value)}
  # Include instance additions too: producer drift must not be silently stripped.
  if hasattr(value, "__dict__"):
    values.update(vars(value))
  return values


@cache
def _binding_wire() -> type[BaseModel]:
  return _dataclass_wire(
    _approval_store().ExecutableApprovalBinding,
    "ExecutableApprovalBindingWire",
    {
      name: field.annotation
      for name, field in ExecutableBindingOperandsWire.model_fields.items()
    },
  )


def _encode_binding(value: Any) -> dict[str, JsonValue]:
  values = _dataclass_values(value)
  for name in ExecutableBindingOperandsWire.model_fields:
    values[name] = PreparedToolCall(
      prepared_input=values[name],
    ).materialize_input()
  return _binding_wire().model_validate(values).model_dump(mode="json")


def _decode_binding(value: object) -> Any:
  wire = _binding_wire().model_validate(value, strict=True)
  return _approval_store().ExecutableApprovalBinding(**wire.model_dump())


def _encode_prepared_business_model(
  value: PreparedBusinessModelChange,
) -> dict[str, JsonValue]:
  return PreparedBusinessModelChangeWire.model_validate(
    _dataclass_values(value),
  ).model_dump(mode="json")


def _decode_prepared_business_model(value: object) -> PreparedBusinessModelChange:
  wire = PreparedBusinessModelChangeWire.model_validate(value)
  return PreparedBusinessModelChange(**wire.model_dump())


def _encode_raw_patch_prepared(
  value: RawPatchPreparedAuthorization,
) -> dict[str, JsonValue]:
  return RawPatchPreparedAuthorizationWire(
    approval_id=value.approval_id,
    prepared_payload=value.prepared_payload,
    prepared_payload_digest=value.prepared_payload_digest,
    created_at=value.created_at,
  ).model_dump(mode="json")


def _decode_raw_patch_prepared(value: object) -> RawPatchPreparedAuthorization:
  wire = RawPatchPreparedAuthorizationWire.model_validate(value)
  return RawPatchPreparedAuthorization(
    approval_id=wire.approval_id,
    prepared_payload=wire.prepared_payload,
    prepared_payload_digest=wire.prepared_payload_digest,
    created_at=wire.created_at,
  )


def _encode_raw_patch_consumption(
  value: RawPatchAuthorizationConsumption,
) -> dict[str, JsonValue]:
  return RawPatchAuthorizationConsumptionWire(
    approval_id=value.approval_id,
    change_set_id=value.change_set_id,
    change_hash=value.change_hash,
    receipt=value.receipt,
    receipt_digest=value.receipt_digest,
    consumed_at=value.consumed_at,
  ).model_dump(mode="json")


def _decode_raw_patch_consumption(value: object) -> RawPatchAuthorizationConsumption:
  wire = RawPatchAuthorizationConsumptionWire.model_validate(value)
  return RawPatchAuthorizationConsumption(
    approval_id=wire.approval_id,
    change_set_id=wire.change_set_id,
    change_hash=wire.change_hash,
    receipt=wire.receipt,
    receipt_digest=wire.receipt_digest,
    consumed_at=wire.consumed_at,
  )


def _encode_tool_result(value: ToolResult) -> str:
  from agent_gateway.approval_store import ApprovalExecutableBindingError
  if (
    type(value) is not tuple
    or len(value) != 2
    or (value[1] is not None and not isinstance(value[1], dict))
  ):
    raise ApprovalExecutableBindingError("result_mismatch")
  try:
    return _approval_store()._executable_result_json(value)
  except (TypeError, ValueError):
    raise ApprovalExecutableBindingError("result_mismatch") from None


def _decode_tool_result(value: object) -> ToolResult:
  if not isinstance(value, str):
    raise ApprovalRemoteProtocolError(
      "tool result must be the canonical string", code="malformed_envelope",
    )
  parsed = json.loads(value)
  if (
    not isinstance(parsed, list)
    or len(parsed) != 2
    or not isinstance(parsed[0], dict)
  ):
    raise ApprovalRemoteProtocolError(
      "tool result string is not the canonical pair",
      code="malformed_envelope",
    )
  restored = _approval_store()._restore_executable_result(parsed[0])
  return restored, parsed[1]


def _encode_targeted(value: Any) -> dict[str, JsonValue]:
  record = value.record
  conflict = value.conflict
  return {
    "record": None if record is None else _encode_prepared_business_model(record),
    "transitioned": value.transitioned,
    "conflict": None if conflict is None else conflict.value,
  }


def _decode_targeted(value: object) -> Any:
  if not isinstance(value, dict) or set(value) != {"record", "transitioned", "conflict"}:
    raise ApprovalRemoteProtocolError(
      "malformed reconciliation result", code="malformed_envelope",
    )
  store = _approval_store()
  record = value["record"]
  conflict = value["conflict"]
  return store.TargetedPreparedReconciliationResult(
    record=None if record is None else _decode_prepared_business_model(record),
    transitioned=value["transitioned"],
    conflict=(
      None if conflict is None else store.PreparedReconciliationConflict(conflict)
    ),
  )


def _encode_tuple(
  value: tuple[Any, ...], annotation: Any, *, method: str, field: str,
) -> list[JsonValue]:
  args = get_args(annotation)
  if len(args) == 2 and args[1] is Ellipsis:
    return [
      _encode_value(item, args[0], method=method, field=field) for item in value
    ]
  return [
    _encode_value(item, arg, method=method, field=field)
    for item, arg in zip(value, args, strict=True)
  ]


def _decode_tuple(
  value: object, annotation: Any, *, method: str, field: str,
) -> tuple[Any, ...]:
  args = get_args(annotation)
  if not isinstance(value, list):
    raise ApprovalRemoteProtocolError(
      "malformed tuple", code="malformed_envelope",
    )
  if len(args) == 2 and args[1] is Ellipsis:
    return tuple(
      _decode_value(item, args[0], method=method, field=field) for item in value
    )
  if len(value) != len(args):
    raise ApprovalRemoteProtocolError(
      "malformed tuple", code="malformed_envelope",
    )
  return tuple(
    _decode_value(item, arg, method=method, field=field)
    for item, arg in zip(value, args)
  )


def _materialize_mappings(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {key: _materialize_mappings(item) for key, item in value.items()}
  if type(value) is list:
    return [_materialize_mappings(item) for item in value]
  if type(value) is tuple:
    return tuple(_materialize_mappings(item) for item in value)
  return value


def _dump_declared(annotation: Any, value: Any) -> JsonValue:
  return _adapter(annotation).dump_python(_materialize_mappings(value), mode="json")


def _validate_declared(annotation: Any, value: Any) -> Any:
  fragment = json.dumps(_materialize_mappings(value), allow_nan=False)
  return _adapter(annotation).validate_json(fragment, strict=True, extra="forbid")


def _collection_arm(members: tuple[Any, ...], origin: Any) -> Any | None:
  for member in members:
    if get_origin(member) is origin:
      return member
  return None


def _encode_union(value: Any, annotation: Any, members: tuple[Any, ...]) -> JsonValue:
  if type(value) is set:
    arm = _collection_arm(members, set)
    if arm is not None:
      return _dump_declared(arm, value)
  if type(value) is frozenset:
    arm = _collection_arm(members, frozenset)
    if arm is not None:
      return _dump_declared(arm, value)
  return _dump_declared(annotation, value)


def _decode_union(value: Any, annotation: Any, members: tuple[Any, ...]) -> Any:
  if isinstance(value, list):
    set_arm = _collection_arm(members, set)
    if set_arm is not None:
      try:
        restored = _validate_declared(set_arm, value)
      except ValidationError:
        restored = None
      if restored is not None:
        return set(restored)
    frozen_arm = _collection_arm(members, frozenset)
    if set_arm is None and frozen_arm is not None:
      try:
        restored = _validate_declared(frozen_arm, value)
      except ValidationError:
        restored = None
      if restored is not None:
        return frozenset(restored)
  return _validate_declared(annotation, value)


def _encode_value(
  value: Any, annotation: Any, *, method: str, field: str,
) -> JsonValue:

  if value is None:
    if _allows_none(annotation):
      return None
    raise TypeError(f"{method}.{field} is required")
  if (method, field) in _BYTES_PARAMETER_DTO:
    return _encode_parameter_bytes(value, method, field)
  core = _non_optional(annotation)
  if core is PreparedToolCall:
    return _encode_prepared_tool_call(value)
  if core is ToolResult:
    return _encode_tool_result(value)
  if core is PreparedBusinessModelChange:
    return _encode_prepared_business_model(value)
  if core is RawPatchPreparedAuthorization:
    return _encode_raw_patch_prepared(value)
  if core is RawPatchAuthorizationConsumption:
    return _encode_raw_patch_consumption(value)
  store = _approval_store()
  if core is store.ExecutableApprovalBinding:
    return _encode_binding(value)
  if core is store.TargetedPreparedReconciliationResult:
    return _encode_targeted(value)
  members = _union_members(annotation)
  if members is not None and not (
    len(members) == 2 and type(None) in members
  ):
    return _encode_union(value, annotation, members)
  origin = get_origin(core)
  if origin is set or origin is frozenset:
    return _dump_declared(core, value)
  if origin is tuple:
    return _encode_tuple(value, core, method=method, field=field)
  if origin is list:
    (item_type,) = get_args(core)
    return [
      _encode_value(item, item_type, method=method, field=field) for item in value
    ]
  if isinstance(core, type) and issubclass(core, BaseModel):
    model = value if isinstance(value, core) else core.model_validate(value)
    return model.model_dump(mode="json")
  if _is_enum(core):
    return value.value
  return _dump_declared(core, value)


def _decode_value(
  value: Any, annotation: Any, *, method: str, field: str,
) -> Any:
  if value is None and _allows_none(annotation):
    return None
  if (method, field) in _BYTES_PARAMETER_DTO:
    return _decode_parameter_bytes(value, method, field)
  core = _non_optional(annotation)
  if core is PreparedToolCall:
    return _decode_prepared_tool_call(value)
  if core is ToolResult:
    return _decode_tool_result(value)
  if core is PreparedBusinessModelChange:
    return _decode_prepared_business_model(value)
  if core is RawPatchPreparedAuthorization:
    return _decode_raw_patch_prepared(value)
  if core is RawPatchAuthorizationConsumption:
    return _decode_raw_patch_consumption(value)
  store = _approval_store()
  if core is store.ExecutableApprovalBinding:
    return _decode_binding(value)
  if core is store.TargetedPreparedReconciliationResult:
    return _decode_targeted(value)
  members = _union_members(annotation)
  if members is not None and not (
    len(members) == 2 and type(None) in members
  ):
    return _decode_union(value, annotation, members)
  origin = get_origin(core)
  if origin is set:
    return set(_validate_declared(core, value))
  if origin is frozenset:
    return frozenset(_validate_declared(core, value))
  if origin is tuple:
    return _decode_tuple(value, core, method=method, field=field)
  if origin is list:
    if not isinstance(value, list):
      raise ApprovalRemoteProtocolError(
        "malformed list", code="malformed_envelope",
      )
    (item_type,) = get_args(core)
    return [
      _decode_value(item, item_type, method=method, field=field) for item in value
    ]
  if isinstance(core, type) and issubclass(core, BaseModel):
    return core.model_validate(value, strict=True)
  if _is_enum(core):
    return core(value)
  return _validate_declared(core, value)


def _operation_for_method(method: str) -> str:
  for operation, name in OPERATION_METHODS.items():
    if name == method:
      return operation
  raise KeyError(method)


def encode_operation_call(
  operation: str, arguments: Mapping[str, Any],
) -> bytes:
  """Serialize one operation's parameter dict into the versioned envelope."""

  spec = _call_specs()[operation]
  names = _parameter_names(spec.function)
  if set(arguments) != set(names):
    raise ApprovalRemoteProtocolError(
      "payload keys must equal the method parameter names",
      code="malformed_envelope",
    )
  hints = _hints(spec.function)
  payload = {
    name: _encode_value(
      arguments[name], hints[name], method=spec.method_name, field=name,
    )
    for name in names
  }
  return _model_bytes(ApprovalRemoteRequest(
    operation=cast(ApprovalRemoteOperation, operation),
    payload=payload,
  ))


def decode_operation_arguments(
  operation: str, payload: Mapping[str, Any],
) -> dict[str, Any]:
  """Restore domain arguments. Extra or missing keys are a malformed envelope."""

  spec = _call_specs()[operation]
  names = _parameter_names(spec.function)
  if set(payload) != set(names):
    raise ApprovalRemoteProtocolError(
      "payload keys must equal the method parameter names",
      code="malformed_envelope",
    )
  hints = _hints(spec.function)
  try:
    return {
      name: _decode_value(
        payload[name], hints[name], method=spec.method_name, field=name,
      )
      for name in names
    }
  except ApprovalRemoteProtocolError:
    raise
  except Exception as exc:
    raise ApprovalRemoteProtocolError(
      "malformed payload", code="malformed_envelope",
    ) from exc


def encode_operation_result(operation: str, result: Any) -> bytes:
  """Serialize a method return value as ApprovalRemoteResponse."""

  spec = _call_specs()[operation]
  hints = _hints(spec.function)
  encoded = _encode_value(
    result, hints["return"], method=spec.method_name, field="",
  )
  return _model_bytes(ApprovalRemoteResponse(result=encoded))


def decode_operation_result(operation: str, body: bytes) -> Any:
  """Parse a 200 body and restore the domain return value."""

  raw = _loads_object(body)
  _require_version(raw)
  try:
    response = ApprovalRemoteResponse.model_validate(raw)
  except ValidationError as exc:
    raise ApprovalRemoteProtocolError(
      "malformed response", code="malformed_envelope",
    ) from exc
  spec = _call_specs()[operation]
  hints = _hints(spec.function)
  try:
    return _decode_value(
      response.result, hints["return"], method=spec.method_name, field="",
    )
  except ApprovalRemoteProtocolError:
    raise
  except Exception as exc:
    raise ApprovalRemoteProtocolError(
      "malformed result", code="malformed_envelope",
    ) from exc


def parse_remote_request(
  body: bytes, *, url_operation: str,
) -> tuple[ApprovalRemoteOperation, dict[str, Any]]:
  """Parse one POST body. The URL's final segment must equal envelope.operation."""

  raw = _loads_object(body)
  _require_version(raw)
  if raw.get("operation") != url_operation:
    raise ApprovalRemoteProtocolError(
      "operation mismatch", code="operation_mismatch",
    )
  try:
    envelope = ApprovalRemoteRequest.model_validate(raw)
  except ValidationError as exc:
    raise ApprovalRemoteProtocolError(
      "malformed envelope", code="malformed_envelope",
    ) from exc
  arguments = decode_operation_arguments(envelope.operation, envelope.payload)
  return envelope.operation, arguments



def _store_exception_body(
  exception: Literal[
    "KeyError",
    "ValueError",
    "RuntimeError",
    "ApprovalConstraintError",
    "PersistentGrantCancellationFenced",
    "PreparedBusinessModelError",
    "RawPatchAuthorizationError",
  ],
  exc: BaseException,
) -> tuple[int, bytes]:
  return _STORE_STATUS[exception], _model_bytes(StoreExceptionErrorBody(
    exception=exception,
    code="store_exception",
    message=_exception_message(exc),
  ))


def _encode_protocol_error(exc: ApprovalRemoteProtocolError) -> tuple[int, bytes]:
  message = _exception_message(exc)
  if exc.code == "version_mismatch":
    code: Literal[
      "malformed_envelope", "version_mismatch", "operation_mismatch",
      "unknown_exception", "unavailable", "unhandled",
    ] = "version_mismatch"
  elif exc.code == "operation_mismatch":
    code = "operation_mismatch"
  elif exc.code == "unknown_exception":
    code = "unknown_exception"
  elif exc.code == "unavailable":
    code = "unavailable"
  elif exc.code == "unhandled":
    code = "unhandled"
  elif exc.code == "malformed_envelope":
    code = "malformed_envelope"
  else:
    code = "unhandled"
  return _PROTOCOL_STATUS[code], _model_bytes(ProtocolErrorBody(
    exception="ApprovalRemoteProtocolError",
    code=code,
    message=message,
  ))


def encode_remote_error(exc: BaseException) -> tuple[int, bytes]:
  """Advisory status plus the versioned error body. Class is the exception name."""

  if type(exc) is ApprovalRemoteIdentityMismatch:
    return ApprovalRemoteIdentityMismatch.status, _model_bytes(IdentityMismatchErrorBody(
      exception="ApprovalRemoteIdentityMismatch",
      code="runtime_identity_mismatch",
      message=_exception_message(exc),
      retryable=False,
    ))
  if type(exc) is ApprovalRemoteMigrationIncomplete:
    return ApprovalRemoteMigrationIncomplete.status, _model_bytes(MigrationIncompleteErrorBody(
      exception="ApprovalRemoteMigrationIncomplete",
      code="migration_incomplete",
      message=_exception_message(exc),
      retryable=True,
    ))
  if type(exc) is ApprovalRemoteReservationActive:
    return ApprovalRemoteReservationActive.status, _model_bytes(ReservationActiveErrorBody(
      exception="ApprovalRemoteReservationActive",
      code="reservation_active",
      message=_exception_message(exc),
      retryable=True,
    ))
  if type(exc) is ApprovalRemoteStaleReservation:
    return ApprovalRemoteStaleReservation.status, _model_bytes(StaleReservationErrorBody(
      exception="ApprovalRemoteStaleReservation",
      code="stale_placement_generation",
      message=_exception_message(exc),
      retryable=False,
    ))
  if type(exc) is ApprovalRemoteReservationMismatch:
    return ApprovalRemoteReservationMismatch.status, _model_bytes(ReservationMismatchErrorBody(
      exception="ApprovalRemoteReservationMismatch",
      code="reservation_mismatch",
      message=_exception_message(exc),
      retryable=False,
    ))
  if type(exc) is ApprovalRemoteProtocolError:
    return _encode_protocol_error(exc)
  from agent_gateway.approval_store import (
    ApprovalExecutableBindingError,
    PersistentGrantCancellationFenced,
    RootTurnRecordError,
  )
  if isinstance(exc, ApprovalExecutableBindingError):
    return 409, _model_bytes(ApprovalExecutableBindingErrorBody.model_validate({
      "exception": "ApprovalExecutableBindingError",
      "code": exc.code,
      "domain_code": exc.code,
      "message": _exception_message(exc),
    }))
  if isinstance(exc, RootTurnRecordError):
    return 409, _model_bytes(RootTurnRecordErrorBody.model_validate({
      "exception": "RootTurnRecordError",
      "code": exc.code,
      "domain_code": exc.code,
      "message": _exception_message(exc),
    }))
  if isinstance(exc, ApprovalConstraintError):
    return _store_exception_body("ApprovalConstraintError", exc)
  if isinstance(exc, ValueError):
    return _store_exception_body("ValueError", exc)
  if isinstance(exc, PreparedBusinessModelError):
    return _store_exception_body("PreparedBusinessModelError", exc)
  if isinstance(exc, RawPatchAuthorizationError):
    return _store_exception_body("RawPatchAuthorizationError", exc)
  if isinstance(exc, PersistentGrantCancellationFenced):
    return _store_exception_body("PersistentGrantCancellationFenced", exc)
  if isinstance(exc, RuntimeError):
    return _store_exception_body("RuntimeError", exc)
  if isinstance(exc, KeyError):
    return _store_exception_body("KeyError", exc)
  return _PROTOCOL_STATUS["unhandled"], _model_bytes(ProtocolErrorBody(
    exception="ApprovalRemoteProtocolError",
    code="unhandled",
    message=_exception_message(exc),
  ))


def _reconstruct_error(body: Any) -> BaseException:
  if isinstance(body, ApprovalExecutableBindingErrorBody):
    return _approval_store().ApprovalExecutableBindingError(body.domain_code)
  if isinstance(body, RootTurnRecordErrorBody):
    return _approval_store().RootTurnRecordError(body.domain_code)
  if isinstance(body, StoreExceptionErrorBody):
    types = _store_exception_types()
    return types[body.exception](body.message)
  if isinstance(body, IdentityMismatchErrorBody):
    return ApprovalRemoteIdentityMismatch(body.message)
  if isinstance(body, MigrationIncompleteErrorBody):
    return ApprovalRemoteMigrationIncomplete(body.message)
  if isinstance(body, ReservationActiveErrorBody):
    return ApprovalRemoteReservationActive(body.message)
  if isinstance(body, StaleReservationErrorBody):
    return ApprovalRemoteStaleReservation(body.message)
  if isinstance(body, ReservationMismatchErrorBody):
    return ApprovalRemoteReservationMismatch(body.message)
  if isinstance(body, ProtocolErrorBody):
    return ApprovalRemoteProtocolError(body.message, code=body.code)
  raise ApprovalRemoteProtocolError(
    "unknown exception", code="unknown_exception",
  )


def _store_exception_types() -> dict[str, type[Exception]]:
  store = _approval_store()
  return {
    "KeyError": KeyError,
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "ApprovalConstraintError": ApprovalConstraintError,
    "PersistentGrantCancellationFenced": store.PersistentGrantCancellationFenced,
    "PreparedBusinessModelError": PreparedBusinessModelError,
    "RawPatchAuthorizationError": RawPatchAuthorizationError,
  }


def decode_remote_error(body: bytes) -> BaseException:
  """Reconstruct the named exception. HTTP status does not select the class."""

  raw = _loads_object(body)
  _require_version(raw)
  try:
    parsed = _error_adapter().validate_python(raw)
  except ValidationError as exc:
    raise ApprovalRemoteProtocolError(
      "malformed error", code="malformed_envelope",
    ) from exc
  return _reconstruct_error(parsed)


def interpret_remote_reply(operation: str, status: int, body: bytes) -> Any:
  """200 restores the return value. Any other status re-raises the named exception."""

  if status == 200:
    return decode_operation_result(operation, body)
  raise decode_remote_error(body)



def _join_operation_url(base_url: str, operation: str) -> str:
  return base_url.rstrip("/") + "/" + operation


class AsyncApprovalsEgressTransport:
  """Async placement-bound POST. AgentCore tunnel only. No claim headers."""

  def __init__(self, base_url: str) -> None:
    self._base_url = base_url

  async def post(
    self, operation: ApprovalRemoteOperation, body: bytes,
  ) -> tuple[int, bytes]:
    import httpx

    try:
      async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
        response = await client.post(
          _join_operation_url(self._base_url, operation),
          content=body,
          headers={"content-type": "application/json"},
        )
    except httpx.HTTPError as exc:
      raise ApprovalRemoteProtocolError(str(exc), code="unavailable") from exc
    return response.status_code, response.content


class BlockingApprovalsEgressTransport:
  """Blocking placement-bound POST.

  Safe under a running event loop: this is a synchronous client, never a
  tunnel coroutine submitted to that loop. LocalProcess passes mint_claim;
  each post attaches a fresh derived claim. AgentCore omits mint_claim.
  """

  def __init__(
    self,
    base_url: str,
    *,
    mint_claim: Callable[[], Mapping[str, str]] | None = None,
  ) -> None:
    self._base_url = base_url
    self._mint_claim = mint_claim

  def post(
    self, operation: ApprovalRemoteOperation, body: bytes,
  ) -> tuple[int, bytes]:
    import httpx

    headers = {"content-type": "application/json"}
    if self._mint_claim is not None:
      from agent_gateway.claim_signing_authority import (
        agent_api_claim_http_headers,
      )
      headers.update(agent_api_claim_http_headers(self._mint_claim()))
    try:
      with httpx.Client(trust_env=False, timeout=None) as client:
        response = client.post(
          _join_operation_url(self._base_url, operation),
          content=body,
          headers=headers,
        )
    except httpx.HTTPError as exc:
      raise ApprovalRemoteProtocolError(str(exc), code="unavailable") from exc
    return response.status_code, response.content


_DEFAULT_AUDIT = object()


class RemoteExecutionAuditEmitter:
  """Posts audit.execution_outcome. Does not skip when the edge has no emitter."""

  def __init__(self, client: "BoundApprovalRemoteClient") -> None:
    self._client = client

  async def emit_execution_outcome(
    self,
    request: ApprovalRequest,
    raw_tool_args: dict[str, Any],
    outcome: ExecutionAuditOutcome,
    error_summary: str | None = None,
  ) -> None:
    body = encode_operation_call(AUDIT_EMITTER_OPERATION, {
      "request": request,
      "raw_tool_args": raw_tool_args,
      "outcome": outcome,
      "error_summary": error_summary,
    })
    status, response = await self._client._post_async(
      AUDIT_EMITTER_OPERATION, body,
    )
    self._client._finish(AUDIT_EMITTER_OPERATION, status, response)


def _abort_then_reraise(
  client: "BoundApprovalRemoteClient",
  reservation: AutonomousPublicationReservation,
  original: BaseException,
) -> NoReturn:
  try:
    client.abort_autonomous_approval_publication(reservation)
  except BaseException as abort_error:
    raise abort_error from original
  raise original


class _AppendPublication:
  def __init__(
    self,
    client: "BoundApprovalRemoteClient",
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    approved: bool,
  ) -> None:
    self._client = client
    self._approval_id = approval_id
    self._tool_call_id = tool_call_id
    self._nonce = nonce
    self._approved = approved
    self._reservation: AutonomousPublicationReservation | None = None

  def __enter__(self) -> None:
    reserved = self._client.reserve_autonomous_approval_publication(
      self._approval_id,
      tool_call_id=self._tool_call_id,
      nonce=self._nonce,
      approved=self._approved,
    )
    self._reservation = reserved.reservation
    return None

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: object,
  ) -> bool:
    reservation = self._reservation
    if reservation is None:
      return False
    if exc is not None:
      _abort_then_reraise(self._client, reservation, exc)
    try:
      self._client.commit_autonomous_approval_publication(reservation)
    except BaseException as commit_error:
      _abort_then_reraise(self._client, reservation, commit_error)
    return False


class _DuplicatePublication:
  def __init__(
    self,
    client: "BoundApprovalRemoteClient",
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    approved: bool,
  ) -> None:
    self._client = client
    self._approval_id = approval_id
    self._tool_call_id = tool_call_id
    self._nonce = nonce
    self._approved = approved

  def __enter__(self) -> None:
    self._client.reconcile_autonomous_approval_delivery_duplicate(
      self._approval_id,
      tool_call_id=self._tool_call_id,
      nonce=self._nonce,
      approved=self._approved,
    )
    return None

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: object,
  ) -> bool:
    return False


class BoundApprovalRemoteClient:
  """Placement-bound wire client. Sends no storage_id and no generation.

  Async methods use async_transport when the caller bound an AgentCore
  tunnel. LocalProcess passes only sync_transport; async methods offload
  that blocking transport to a worker thread. Sync methods always use sync_transport.
  """

  def __init__(
    self,
    *,
    sync_transport: SyncApprovalRemoteTransport,
    async_transport: ApprovalRemoteTransport | None = None,
    audit_emitter: ApprovalStoreAuditEmitter | None | object = _DEFAULT_AUDIT,
  ) -> None:
    self._sync = sync_transport
    self._async = async_transport
    if audit_emitter is _DEFAULT_AUDIT:
      self._audit_emitter: ApprovalStoreAuditEmitter | None = (
        RemoteExecutionAuditEmitter(self)
      )
    else:
      self._audit_emitter = cast(
        ApprovalStoreAuditEmitter | None, audit_emitter,
      )

  @property
  def audit_emitter(self) -> ApprovalStoreAuditEmitter | None:
    return self._audit_emitter

  def _encode_method(
    self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any],
  ) -> tuple[str, bytes]:
    operation = _operation_for_method(method)
    signature = inspect.signature(getattr(BoundApprovalRemoteClient, method))
    bound = signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    arguments = {
      name: value for name, value in bound.arguments.items() if name != "self"
    }
    return operation, encode_operation_call(operation, arguments)

  def _post_sync(self, operation: str, body: bytes) -> tuple[int, bytes]:
    try:
      return self._sync.post(cast(ApprovalRemoteOperation, operation), body)
    except ApprovalRemoteProtocolError:
      raise
    except Exception as exc:
      raise ApprovalRemoteProtocolError(str(exc), code="unavailable") from exc

  async def _post_async(self, operation: str, body: bytes) -> tuple[int, bytes]:
    import asyncio
    try:
      if self._async is not None:
        return await self._async.post(
          cast(ApprovalRemoteOperation, operation), body,
        )
      return await asyncio.to_thread(
        self._sync.post, cast(ApprovalRemoteOperation, operation), body,
      )
    except ApprovalRemoteProtocolError:
      raise
    except Exception as exc:
      raise ApprovalRemoteProtocolError(str(exc), code="unavailable") from exc

  def _finish(self, operation: str, status: int, body: bytes) -> Any:
    return interpret_remote_reply(operation, status, body)

  def _scall(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
    operation, body = self._encode_method(method, args, kwargs)
    status, response = self._post_sync(operation, body)
    return self._finish(operation, status, response)

  async def _acall(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
    operation, body = self._encode_method(method, args, kwargs)
    status, response = await self._post_async(operation, body)
    return self._finish(operation, status, response)

  async def create(self, request: ApprovalRequest) -> ApprovalRequest:
    return await self._acall("create", request)

  async def create_raw_patch_authorization(
    self, request: ApprovalRequest, *, prepared_payload: bytes,
  ) -> ApprovalRequest:
    return await self._acall(
      "create_raw_patch_authorization",
      request,
      prepared_payload=prepared_payload,
    )

  async def create_or_get_by_tool_call_id(
    self, request: ApprovalRequest,
  ) -> tuple[ApprovalRequest, bool]:
    return await self._acall("create_or_get_by_tool_call_id", request)

  async def create_or_get_with_prepared_business_model_change(
    self, request: ApprovalRequest, record: PreparedBusinessModelChange,
  ) -> tuple[ApprovalRequest, PreparedBusinessModelChange, bool]:
    return await self._acall(
      "create_or_get_with_prepared_business_model_change", request, record,
    )

  async def get(
    self, approval_id: str, *,
    owner_user_id: str | tuple[str, ...] | list[str] | set[str] | None = None,
  ) -> ApprovalRequest | None:
    return await self._acall("get", approval_id, owner_user_id=owner_user_id)

  async def get_by_tool_call_id(
    self, tool_call_id: str, *,
    owner_user_id: str | tuple[str, ...] | list[str] | set[str] | None = None,
  ) -> ApprovalRequest | None:
    return await self._acall(
      "get_by_tool_call_id", tool_call_id, owner_user_id=owner_user_id,
    )

  async def update_request(self, request: ApprovalRequest) -> ApprovalRequest:
    return await self._acall("update_request", request)

  async def transition_state(
    self, approval_id: str, state: ApprovalState, *,
    expected_state_version: int | None = None, expires_at: datetime | None = None,
    decider_id: str | None = None, decider_role: str | None = None,
    decision: str | None = None, decision_reason: str | None = None,
  ) -> ApprovalRequest:
    return await self._acall(
      "transition_state", approval_id, state,
      expected_state_version=expected_state_version, expires_at=expires_at,
      decider_id=decider_id, decider_role=decider_role, decision=decision,
      decision_reason=decision_reason,
    )

  async def terminalize_pending_for_cancellation(
    self, approval_id: str, *, expected_tool_call_id: str, expected_user_id: str,
    expected_request_id: str, expected_run_id: str, expected_session_id: str,
    expected_channel: str | None, decider_id: str | None = None,
    decider_role: str | None = None, decision_reason: str,
    autonomous_delivery: Mapping[str, Any] | None = None,
  ) -> tuple[ApprovalRequest, bool, bool]:
    return await self._acall(
      "terminalize_pending_for_cancellation", approval_id,
      expected_tool_call_id=expected_tool_call_id,
      expected_user_id=expected_user_id,
      expected_request_id=expected_request_id, expected_run_id=expected_run_id,
      expected_session_id=expected_session_id, expected_channel=expected_channel,
      decider_id=decider_id, decider_role=decider_role,
      decision_reason=decision_reason, autonomous_delivery=autonomous_delivery,
    )

  async def abort_unpublished_approval(
    self, approval_id: str, *, expected_tool_call_id: str, expected_user_id: str,
    expected_request_id: str, expected_run_id: str, expected_session_id: str,
    expected_channel: str | None, decision_reason: str,
  ) -> tuple[ApprovalRequest, bool, bool]:
    return await self._acall(
      "abort_unpublished_approval", approval_id,
      expected_tool_call_id=expected_tool_call_id,
      expected_user_id=expected_user_id,
      expected_request_id=expected_request_id, expected_run_id=expected_run_id,
      expected_session_id=expected_session_id, expected_channel=expected_channel,
      decision_reason=decision_reason,
    )

  async def record_vote(
    self, approval_id: str, vote: ApprovalVote,
  ) -> ApprovalRequest:
    return await self._acall("record_vote", approval_id, vote)

  async def record_vote_with_autonomous_delivery(
    self, approval_id: str, vote: ApprovalVote, *, delivery: Mapping[str, Any],
  ) -> ApprovalRequest:
    return await self._acall(
      "record_vote_with_autonomous_delivery", approval_id, vote, delivery=delivery,
    )

  async def locate_request(
    self, *, approval_id: str, tenant_storage_id: str,
  ) -> ApprovalRequestLocator | None:
    return await self._acall(
      "locate_request", approval_id=approval_id, tenant_storage_id=tenant_storage_id,
    )

  async def list_parked_requests(
    self, tenant_storage_id: str,
  ) -> list[ApprovalRequest]:
    return await self._acall("list_parked_requests", tenant_storage_id)

  async def _locate_request_for_decision(
    self, *, tenant_storage_id: str, approval_id: str | None = None,
    tool_call_id: str | None = None, nonce: str | None = None,
  ) -> tuple[ApprovalRequest, ApprovalRequestLocator, str | None] | None:
    return await self._acall(
      "_locate_request_for_decision", tenant_storage_id=tenant_storage_id,
      approval_id=approval_id, tool_call_id=tool_call_id, nonce=nonce,
    )

  async def _locate_root_turn_for_approval(
    self, *, approval_id: str, tenant_storage_id: str,
  ) -> tuple[str, str, str, str] | None:
    return await self._acall(
      "_locate_root_turn_for_approval",
      approval_id=approval_id, tenant_storage_id=tenant_storage_id,
    )

  async def record_policy_generation(
    self, *, policy_id: str, policy_version: str, policy_bundle_hash: str,
    make_current: bool = True,
  ) -> PolicyGeneration:
    return await self._acall(
      "record_policy_generation", policy_id=policy_id,
      policy_version=policy_version, policy_bundle_hash=policy_bundle_hash,
      make_current=make_current,
    )

  async def read_policy_generation(
    self, *, generation_ref: str,
  ) -> PolicyGeneration | None:
    return await self._acall(
      "read_policy_generation", generation_ref=generation_ref,
    )

  async def current_policy_generation(self) -> PolicyGeneration | None:
    return await self._acall("current_policy_generation")

  async def record_root_turn(
    self, *, job_id: str, turn_id: str, tenant_storage_id: str,
    capability_bind: Mapping[str, object], credential_generation_ref: str | None,
    effective_grant: Mapping[str, object], research_binding: Mapping[str, object] | None,
    selection_snapshot: Mapping[str, object], approval_route_kind: str,
    policy_generation_ref: str, created_at: datetime | None = None,
  ) -> None:
    return await self._acall(
      "record_root_turn", job_id=job_id, turn_id=turn_id,
      tenant_storage_id=tenant_storage_id, capability_bind=capability_bind,
      credential_generation_ref=credential_generation_ref,
      effective_grant=effective_grant, research_binding=research_binding,
      selection_snapshot=selection_snapshot, approval_route_kind=approval_route_kind,
      policy_generation_ref=policy_generation_ref, created_at=created_at,
    )

  async def _read_root_turn_for_executor(
    self, *, job_id: str, turn_id: str, tenant_storage_id: str,
  ) -> RootTurnRecord | None:
    return await self._acall(
      "_read_root_turn_for_executor", job_id=job_id, turn_id=turn_id,
      tenant_storage_id=tenant_storage_id,
    )

  async def link_root_turn_executable_binding(
    self, *, job_id: str, turn_id: str, tool_call_id: str, nonce: str,
    tenant_storage_id: str,
  ) -> None:
    return await self._acall(
      "link_root_turn_executable_binding", job_id=job_id, turn_id=turn_id,
      tool_call_id=tool_call_id, nonce=nonce, tenant_storage_id=tenant_storage_id,
    )

  async def reissue_parked_approval(
    self, *, approval_id: str, job_id: str, tool_call_id: str, nonce: str,
    turn_id: str, tenant_storage_id: str, generation: PolicyGeneration,
  ) -> tuple[ApprovalRequest, str, str, bool]:
    return await self._acall(
      "reissue_parked_approval", approval_id=approval_id, job_id=job_id,
      tool_call_id=tool_call_id, nonce=nonce, turn_id=turn_id,
      tenant_storage_id=tenant_storage_id, generation=generation,
    )

  async def record_executable_binding(
    self, *, approval_id: str, job_id: str, tool_call_id: str, nonce: str,
    tenant_storage_id: str, tool_name: str, prepared_call: PreparedToolCall,
    dispatch_context: Mapping[str, object], credential_generation_ref: str | None,
  ) -> None:
    return await self._acall(
      "record_executable_binding", approval_id=approval_id, job_id=job_id,
      tool_call_id=tool_call_id, nonce=nonce, tenant_storage_id=tenant_storage_id,
      tool_name=tool_name, prepared_call=prepared_call,
      dispatch_context=dispatch_context,
      credential_generation_ref=credential_generation_ref,
    )

  async def park_executable_binding(
    self, *, approval_id: str, job_id: str, tool_call_id: str, nonce: str,
    tenant_storage_id: str, tool_name: str, prepared_call: PreparedToolCall,
    dispatch_context: Mapping[str, object], credential_generation_ref: str | None,
    policy_generation_ref: str | None = None,
  ) -> str:
    return await self._acall(
      "park_executable_binding", approval_id=approval_id, job_id=job_id,
      tool_call_id=tool_call_id, nonce=nonce, tenant_storage_id=tenant_storage_id,
      tool_name=tool_name, prepared_call=prepared_call,
      dispatch_context=dispatch_context,
      credential_generation_ref=credential_generation_ref,
      policy_generation_ref=policy_generation_ref,
    )

  async def _read_executable_binding_for_executor(
    self, *, job_id: str, tool_call_id: str, nonce: str, tenant_storage_id: str,
  ) -> tuple[ExecutableApprovalBinding, ApprovalRequest]:
    return await self._acall(
      "_read_executable_binding_for_executor", job_id=job_id,
      tool_call_id=tool_call_id, nonce=nonce, tenant_storage_id=tenant_storage_id,
    )

  async def _read_completed_executable_result_for_executor(
    self, *, binding: ExecutableApprovalBinding,
  ) -> ToolResult | None:
    return await self._acall(
      "_read_completed_executable_result_for_executor", binding=binding,
    )

  async def _claim_executable_binding_for_executor(
    self, *, binding: ExecutableApprovalBinding, decision_state_version: int,
    claim_token: str,
  ) -> ToolResult | None:
    return await self._acall(
      "_claim_executable_binding_for_executor", binding=binding,
      decision_state_version=decision_state_version, claim_token=claim_token,
    )

  async def _complete_executable_binding_for_executor(
    self, *, binding: ExecutableApprovalBinding, decision_state_version: int,
    claim_token: str, tool_result: ToolResult,
  ) -> None:
    return await self._acall(
      "_complete_executable_binding_for_executor", binding=binding,
      decision_state_version=decision_state_version, claim_token=claim_token,
      tool_result=tool_result,
    )

  async def create_persistent_grant(
    self, grant: PersistentGrant,
  ) -> PersistentGrant:
    return await self._acall("create_persistent_grant", grant)

  async def find_persistent_grant(
    self, *, user_id: str, tool_name: str, scope_hint: str,
    now: datetime | None = None, approval_constraint: str = "standard",
    approval_reuse_mode: ApprovalReuseMode = "legacy",
    approval_reuse_key: str | None = None,
  ) -> PersistentGrant | None:
    return await self._acall(
      "find_persistent_grant", user_id=user_id, tool_name=tool_name,
      scope_hint=scope_hint, now=now, approval_constraint=approval_constraint,
      approval_reuse_mode=approval_reuse_mode,
      approval_reuse_key=approval_reuse_key,
    )

  async def revoke_persistent_grant(
    self, grant_id: str, *, revoked_at: datetime | None = None,
  ) -> None:
    return await self._acall(
      "revoke_persistent_grant", grant_id, revoked_at=revoked_at,
    )

  async def revoke_persistent_grants_for_approval(
    self, approval_id: str, *, revoked_at: datetime | None = None,
  ) -> int:
    return await self._acall(
      "revoke_persistent_grants_for_approval", approval_id, revoked_at=revoked_at,
    )

  async def fence_persistent_grants_for_cancellation(
    self, approval_id: str, *, expected_tool_call_id: str, expected_user_id: str,
    expected_request_id: str, expected_run_id: str, expected_session_id: str,
    expected_channel: str | None,
  ) -> tuple[ApprovalRequest, bool]:
    return await self._acall(
      "fence_persistent_grants_for_cancellation", approval_id,
      expected_tool_call_id=expected_tool_call_id,
      expected_user_id=expected_user_id,
      expected_request_id=expected_request_id, expected_run_id=expected_run_id,
      expected_session_id=expected_session_id, expected_channel=expected_channel,
    )

  async def create_delegation_grant(
    self, grant: DelegationGrant,
  ) -> DelegationGrant:
    return await self._acall("create_delegation_grant", grant)

  async def get_delegation_grant(
    self, delegation_id: str,
  ) -> DelegationGrant | None:
    return await self._acall("get_delegation_grant", delegation_id)

  async def claim_delegation_grant(
    self, *, delegation_id: str, bound_relay_request_id: str,
    bound_excel_session_id: str, now: datetime | None = None,
  ) -> DelegationGrant | None:
    return await self._acall(
      "claim_delegation_grant", delegation_id=delegation_id,
      bound_relay_request_id=bound_relay_request_id,
      bound_excel_session_id=bound_excel_session_id, now=now,
    )

  async def revoke_delegation_grant(
    self, delegation_id: str, *, revoked_at: datetime | None = None,
  ) -> None:
    return await self._acall(
      "revoke_delegation_grant", delegation_id, revoked_at=revoked_at,
    )

  async def insert_or_verify_prepared_business_model_change(
    self, record: PreparedBusinessModelChange,
  ) -> tuple[PreparedBusinessModelChange, bool]:
    return await self._acall(
      "insert_or_verify_prepared_business_model_change", record,
    )

  async def get_prepared_business_model_change(
    self, *, caller_kind: str, user_scope: str, idempotency_locator: str,
  ) -> PreparedBusinessModelChange | None:
    return await self._acall(
      "get_prepared_business_model_change", caller_kind=caller_kind,
      user_scope=user_scope, idempotency_locator=idempotency_locator,
    )

  async def transition_prepared_business_model_change(
    self, *, caller_kind: str, user_scope: str, idempotency_locator: str,
    expected: PreparedBusinessModelLifecycle, target: PreparedBusinessModelLifecycle,
    approval_id: str | None = None, approval_chain_id: str | None = None,
    execution_receipt: bytes | None = None, restoration_digest: str | None = None,
    checkpoint_id: str | None = None, consumed_at: str | None = None,
  ) -> PreparedBusinessModelChange:
    return await self._acall(
      "transition_prepared_business_model_change", caller_kind=caller_kind,
      user_scope=user_scope, idempotency_locator=idempotency_locator,
      expected=expected, target=target, approval_id=approval_id,
      approval_chain_id=approval_chain_id, execution_receipt=execution_receipt,
      restoration_digest=restoration_digest, checkpoint_id=checkpoint_id,
      consumed_at=consumed_at,
    )

  async def reconcile_prepared_business_model_change(
    self, *, caller_kind: str, user_scope: str, idempotency_locator: str,
  ) -> TargetedPreparedReconciliationResult:
    return await self._acall(
      "reconcile_prepared_business_model_change", caller_kind=caller_kind,
      user_scope=user_scope, idempotency_locator=idempotency_locator,
    )

  async def resolve_raw_patch_authorization_reference(
    self, reference: str,
  ) -> ApprovalRequest | None:
    return await self._acall(
      "resolve_raw_patch_authorization_reference", reference,
    )

  async def bind_raw_patch_prepared_authorization(
    self, *, approval_id: str, prepared_payload: bytes,
  ) -> tuple[RawPatchPreparedAuthorization, bool]:
    return await self._acall(
      "bind_raw_patch_prepared_authorization", approval_id=approval_id,
      prepared_payload=prepared_payload,
    )

  async def get_raw_patch_prepared_authorization(
    self, *, approval_id: str,
  ) -> RawPatchPreparedAuthorization | None:
    return await self._acall(
      "get_raw_patch_prepared_authorization", approval_id=approval_id,
    )

  async def consume_raw_patch_authorization(
    self, *, approval_id: str, approval_chain_id: str, authorization_mode: str,
    grant_reference: str | None, cache_reference: str | None, user_id: str,
    tool_name: str, args_hash: str, change_set_id: str, change_hash: str,
    base_vector_hash: str, reviewed_change_binding_digest: str,
    execution_semantics_digest: str, prepared_payload_digest: str,
    claim_token: str, receipt: bytes,
  ) -> tuple[RawPatchAuthorizationConsumption, bool]:
    return await self._acall(
      "consume_raw_patch_authorization", approval_id=approval_id,
      approval_chain_id=approval_chain_id, authorization_mode=authorization_mode,
      grant_reference=grant_reference, cache_reference=cache_reference,
      user_id=user_id, tool_name=tool_name, args_hash=args_hash,
      change_set_id=change_set_id, change_hash=change_hash,
      base_vector_hash=base_vector_hash,
      reviewed_change_binding_digest=reviewed_change_binding_digest,
      execution_semantics_digest=execution_semantics_digest,
      prepared_payload_digest=prepared_payload_digest, claim_token=claim_token,
      receipt=receipt,
    )

  async def claim_raw_patch_authorization(
    self, *, approval_id: str, approval_chain_id: str, authorization_mode: str,
    grant_reference: str | None, cache_reference: str | None, user_id: str,
    tool_name: str, args_hash: str, change_set_id: str, change_hash: str,
    base_vector_hash: str, reviewed_change_binding_digest: str,
    execution_semantics_digest: str, prepared_payload_digest: str,
  ) -> RawPatchAuthorizationClaim:
    return await self._acall(
      "claim_raw_patch_authorization", approval_id=approval_id,
      approval_chain_id=approval_chain_id, authorization_mode=authorization_mode,
      grant_reference=grant_reference, cache_reference=cache_reference,
      user_id=user_id, tool_name=tool_name, args_hash=args_hash,
      change_set_id=change_set_id, change_hash=change_hash,
      base_vector_hash=base_vector_hash,
      reviewed_change_binding_digest=reviewed_change_binding_digest,
      execution_semantics_digest=execution_semantics_digest,
      prepared_payload_digest=prepared_payload_digest,
    )

  async def release_raw_patch_authorization_claim(
    self, *, approval_id: str, approval_chain_id: str, authorization_mode: str,
    grant_reference: str | None, cache_reference: str | None, user_id: str,
    tool_name: str, args_hash: str, change_set_id: str, change_hash: str,
    base_vector_hash: str, reviewed_change_binding_digest: str,
    execution_semantics_digest: str, prepared_payload_digest: str, claim_token: str,
  ) -> None:
    return await self._acall(
      "release_raw_patch_authorization_claim", approval_id=approval_id,
      approval_chain_id=approval_chain_id, authorization_mode=authorization_mode,
      grant_reference=grant_reference, cache_reference=cache_reference,
      user_id=user_id, tool_name=tool_name, args_hash=args_hash,
      change_set_id=change_set_id, change_hash=change_hash,
      base_vector_hash=base_vector_hash,
      reviewed_change_binding_digest=reviewed_change_binding_digest,
      execution_semantics_digest=execution_semantics_digest,
      prepared_payload_digest=prepared_payload_digest, claim_token=claim_token,
    )

  async def heartbeat_raw_patch_authorization_claim(
    self, *, approval_id: str, claim_token: str, lease_seconds: float = 120.0,
  ) -> RawPatchAuthorizationClaim:
    return await self._acall(
      "heartbeat_raw_patch_authorization_claim", approval_id=approval_id,
      claim_token=claim_token, lease_seconds=lease_seconds,
    )

  async def get_raw_patch_authorization_consumption(
    self, *, approval_id: str,
  ) -> RawPatchAuthorizationConsumption | None:
    return await self._acall(
      "get_raw_patch_authorization_consumption", approval_id=approval_id,
    )

  async def get_autonomous_approval_delivery(
    self, approval_id: str, *, tool_call_id: str, nonce: str,
  ) -> dict[str, Any] | None:
    return await self._acall(
      "get_autonomous_approval_delivery", approval_id,
      tool_call_id=tool_call_id, nonce=nonce,
    )

  async def ensure_autonomous_approval_delivery_audited(
    self, approval_id: str, *, tool_call_id: str, nonce: str,
  ) -> dict[str, Any]:
    return await self._acall(
      "ensure_autonomous_approval_delivery_audited", approval_id,
      tool_call_id=tool_call_id, nonce=nonce,
    )

  async def acknowledge_autonomous_approval_delivery(
    self, approval_id: str, *, task_id: str, control_run_id: str, session_id: str,
    channel_id: str, tool_call_id: str, nonce: str, approved: bool, decided_at_ns: int,
  ) -> dict[str, Any]:
    return await self._acall(
      "acknowledge_autonomous_approval_delivery", approval_id, task_id=task_id,
      control_run_id=control_run_id, session_id=session_id, channel_id=channel_id,
      tool_call_id=tool_call_id, nonce=nonce, approved=approved,
      decided_at_ns=decided_at_ns,
    )

  async def record_autonomous_approval_delivery_failure(
    self, approval_id: str, *, tool_call_id: str, nonce: str, error: str,
  ) -> dict[str, Any]:
    return await self._acall(
      "record_autonomous_approval_delivery_failure", approval_id,
      tool_call_id=tool_call_id, nonce=nonce, error=error,
    )

  async def quarantine_autonomous_approval_delivery(
    self, approval_id: str, *, tool_call_id: str, nonce: str, error: str,
  ) -> dict[str, Any]:
    return await self._acall(
      "quarantine_autonomous_approval_delivery", approval_id,
      tool_call_id=tool_call_id, nonce=nonce, error=error,
    )

  async def list_pending_autonomous_approval_deliveries(
    self, *, after_sequence: int = 0, through_sequence: int | None = None,
  ) -> list[dict[str, Any]]:
    return await self._acall(
      "list_pending_autonomous_approval_deliveries",
      after_sequence=after_sequence, through_sequence=through_sequence,
    )

  async def autonomous_approval_delivery_high_water(self) -> int:
    return await self._acall("autonomous_approval_delivery_high_water")

  async def autonomous_approval_delivery_recovery_window(self) -> dict[str, int]:
    return await self._acall("autonomous_approval_delivery_recovery_window")

  def reconcile_autonomous_approval_delivery_duplicate(
    self, approval_id: str, *, tool_call_id: str, nonce: str, approved: bool,
  ) -> dict[str, Any]:
    return self._scall(
      "reconcile_autonomous_approval_delivery_duplicate", approval_id,
      tool_call_id=tool_call_id, nonce=nonce, approved=approved,
    )

  def reserve_autonomous_approval_publication(
    self, decision_id: str, *, tool_call_id: str, nonce: str, approved: bool,
  ) -> PublishReserveResult:
    return self._scall(
      "reserve_autonomous_approval_publication", decision_id,
      tool_call_id=tool_call_id, nonce=nonce, approved=approved,
    )

  def commit_autonomous_approval_publication(
    self, reservation: AutonomousPublicationReservation,
  ) -> PublishCommitResult:
    return self._scall(
      "commit_autonomous_approval_publication", reservation,
    )

  def abort_autonomous_approval_publication(
    self, reservation: AutonomousPublicationReservation,
  ) -> PublishAbortResult:
    return self._scall("abort_autonomous_approval_publication", reservation)

  def autonomous_approval_delivery_append_transaction(
    self, approval_id: str, *, tool_call_id: str, nonce: str, approved: bool,
  ) -> ContextManager[None]:
    return _AppendPublication(
      self, approval_id, tool_call_id=tool_call_id, nonce=nonce, approved=approved,
    )

  def autonomous_approval_delivery_duplicate_transaction(
    self, approval_id: str, *, tool_call_id: str, nonce: str, approved: bool,
  ) -> ContextManager[None]:
    return _DuplicatePublication(
      self, approval_id, tool_call_id=tool_call_id, nonce=nonce, approved=approved,
    )

  async def enqueue_pending_approval_notification(
    self, request: ApprovalRequest,
  ) -> dict[str, Any] | None:
    return await self._acall("enqueue_pending_approval_notification", request)

  async def retry_failed_approval_notifications(
    self, approval_id: str,
  ) -> dict[str, Any]:
    return await self._acall("retry_failed_approval_notifications", approval_id)

  async def get_approval_notification_projection(
    self, approval_id: str,
  ) -> dict[str, Any] | None:
    return await self._acall("get_approval_notification_projection", approval_id)

  async def list_approval_notification_outbox(
    self, approval_id: str | None = None,
  ) -> list[dict[str, Any]]:
    return await self._acall(
      "list_approval_notification_outbox", approval_id,
    )

