from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import json
import math
import re
import time
import traceback
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, Mapping, Protocol


ToolClass = Literal[
  "read",
  "pure_transform",
  "artifact_write",
  "state_write",
  "external_write",
  "portfolio_config",
  "irreversible",
]

ApprovalIdentitySource = Literal["change_set", "reviewed_change_binding"]
ApprovalConstraint = Literal[
  "standard",
  "fresh_human_owner",
  "legacy_unknown",
]
AuthorizationMode = Literal[
  "HUMAN",
  "AUTO_ALLOW",
  "CACHE_HIT",
  "PERSISTENT_GRANT",
  "OWNER_CONTROL_PLANE",
]

_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_REVIEWED_CHANGE_ID_RE = re.compile(r"^reviewed-change:v1:[0-9a-f]{64}$")

ApprovalNotificationPolicy = Literal["auto", "interrupt", "skip"]

ApprovalState = Literal[
  "created",
  "auto_approved",
  "auto_denied",
  "pending_user",
  "routed_external",
  "approved",
  "denied",
  "expired",
]

ApprovalOutcome = Literal[
  "auto_approve",
  "auto_deny",
  "request_user_approval",
  "route_external",
]

TerminalDecision = Literal["approved", "denied", "auto_approved", "auto_denied", "expired"]


def utc_now() -> datetime:
  return datetime.now(UTC)


def new_approval_id() -> str:
  return str(uuid.uuid4())


def canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def canonical_json_strict(value: Any) -> str:
  _require_plain_json(value)
  return json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
  )


def _require_plain_json(value: Any) -> None:
  if value is None or isinstance(value, str | bool | int):
    return
  if isinstance(value, float):
    if not math.isfinite(value):
      raise ValueError("non-finite JSON number")
    return
  if isinstance(value, list):
    for item in value:
      _require_plain_json(item)
    return
  if isinstance(value, dict):
    for key, item in value.items():
      if not isinstance(key, str):
        raise TypeError("JSON object keys must be strings")
      _require_plain_json(item)
    return
  raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _reviewed_change_contract() -> Any:
  """Load the monorepo's dependency-light identity owner on exact-plan paths."""

  errors: list[ModuleNotFoundError] = []
  for module_name in (
    "research.reviewed_change_binding",
    "api.research.reviewed_change_binding",
  ):
    try:
      return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
      if exc.name not in {module_name, module_name.split(".")[0]}:
        raise
      errors.append(exc)
  raise ValueError(
    "reviewed change identity contract is unavailable"
  ) from errors[-1]


def _normalize_review_reference(value: dict[str, Any]) -> dict[str, Any]:
  contract = _reviewed_change_contract()
  try:
    return contract.ReviewReference.from_dict(value).to_dict()
  except contract.ReviewedChangeBindingError as exc:
    raise ValueError("review_reference must use the typed review contract") from exc


def sha256_args(value: Any) -> str:
  return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ApprovalPolicyError(RuntimeError):
  """Sanitized wrapper for policy failures.

  The original exception message and raw tool arguments are deliberately not
  included because SDK and SSE paths stringify callback failures.
  """


class ApprovalConstraintError(ValueError):
  """A durable approval constraint is missing, invalid, or cannot authorize."""


class ApprovalRequestPayload:
  """Ephemeral approval payload carrying raw args only in memory."""

  __slots__ = ("_approval_id", "_tool_name", "_tool_class", "_tool_args", "_cleared")

  def __init__(
    self,
    approval_id: str,
    tool_name: str,
    tool_class: ToolClass,
    tool_args: dict[str, Any],
  ) -> None:
    self._approval_id = approval_id
    self._tool_name = tool_name
    self._tool_class = tool_class
    self._tool_args = tool_args
    self._cleared = False

  @property
  def approval_id(self) -> str:
    return self._approval_id

  @property
  def tool_name(self) -> str:
    return self._tool_name

  @property
  def tool_class(self) -> ToolClass:
    return self._tool_class

  @property
  def tool_args(self) -> dict[str, Any]:
    if self._cleared:
      raise RuntimeError("tool_args accessed after clear_raw_args()")
    return self._tool_args

  def clear_raw_args(self) -> None:
    self._tool_args = {}
    self._cleared = True

  def __repr__(self) -> str:
    return (
      "ApprovalRequestPayload("
      f"approval_id={self._approval_id!r}, "
      f"tool_name={self._tool_name!r}, "
      f"tool_class={self._tool_class!r}, "
      f"tool_args=<REDACTED>, cleared={self._cleared})"
    )

  __str__ = __repr__

  def __getstate__(self) -> None:
    raise TypeError("ApprovalRequestPayload is not serializable")

  def __reduce__(self) -> None:
    raise TypeError("ApprovalRequestPayload is not picklable")

  def __copy__(self) -> None:
    raise TypeError("ApprovalRequestPayload is not copyable")

  def __deepcopy__(self, memo: dict[int, Any]) -> None:
    raise TypeError("ApprovalRequestPayload is not deepcopyable")


@dataclass(kw_only=True)
class RunContext:
  user_id: str
  request_id: str
  session_id: str | None = None
  run_id: str | None = None
  profile: str = "chat"
  channel: str = "web"
  skill: str | None = None
  decider_role: str | None = None
  tenant_id: str | None = None
  parent_approval_id: str | None = None
  delegation: DelegationGrant | None = None
  model_id: str | None = None
  model_version: str | None = None
  system_prompt_hash: str | None = None
  tool_schema_version: str | None = None
  mcp_server_version: str | None = None
  policy_bundle_hash: str = "unknown"


@dataclass(kw_only=True)
class ApprovalRequest:
  approval_id: str
  tool_call_id: str
  parent_approval_id: str | None
  approval_chain_id: str
  delegation_id: str | None = None
  request_id: str
  session_id: str | None
  run_id: str | None
  user_id: str
  profile: str
  channel: str
  tool_name: str
  tool_class: ToolClass
  tool_args_redacted: dict[str, Any]
  args_hash: str
  reason: str | None
  blast_radius_summary: str
  state: ApprovalState
  requested_at: datetime
  decided_at: datetime | None = None
  expires_at: datetime | None = None
  decider_id: str | None = None
  decider_role: str | None = None
  decision: TerminalDecision | None = None
  decision_reason: str | None = None
  required_decider_count: int = 1
  eligible_decider_count: int = 1
  votes_received_count: int = 0
  state_version: int = 0
  args_predicate: dict[str, Any] | None = None
  chain_trust_window_seconds: int | None = None
  route_target: str | None = None
  route_target_type: str | None = None
  external_callback_id: str | None = None
  policy_id: str = "single-user"
  policy_version: str = "1"
  policy_bundle_hash: str = "unknown"
  persistent_grant_scope: str | None = None
  tenant_id: str | None = None
  model_id: str | None = None
  model_version: str | None = None
  system_prompt_hash: str | None = None
  tool_schema_version: str | None = None
  mcp_server_version: str | None = None
  skill: str | None = None
  notification_policy: ApprovalNotificationPolicy = "auto"
  notification: dict[str, Any] | None = None
  approval_constraint: ApprovalConstraint = "legacy_unknown"
  required_owner_user_id: str | None = None
  identity_source: ApprovalIdentitySource | None = None
  change_set_id: str | None = None
  change_hash: str | None = None
  base_vector_hash: str | None = None
  reviewed_change_binding_digest: str | None = None
  review_reference: dict[str, Any] | None = None
  execution_semantics_digest: str | None = None
  authorization_mode: AuthorizationMode = "HUMAN"
  grant_reference: str | None = None
  cache_reference: str | None = None

  def __post_init__(self) -> None:
    if self.approval_constraint not in {
      "standard",
      "fresh_human_owner",
      "legacy_unknown",
    }:
      raise ApprovalConstraintError("unsupported approval_constraint")
    if self.approval_constraint == "fresh_human_owner":
      owner_user_id = self.required_owner_user_id
      if (
        not isinstance(owner_user_id, str)
        or not owner_user_id
        or owner_user_id != owner_user_id.strip()
      ):
        raise ApprovalConstraintError(
          "fresh_human_owner approval requires a canonical owner identity"
        )
    elif self.required_owner_user_id is not None:
      raise ApprovalConstraintError(
        "required_owner_user_id requires fresh_human_owner approval"
      )
    if self.authorization_mode not in {
      "HUMAN",
      "AUTO_ALLOW",
      "CACHE_HIT",
      "PERSISTENT_GRANT",
      "OWNER_CONTROL_PLANE",
    }:
      raise ValueError("unsupported authorization_mode")
    if self.authorization_mode == "CACHE_HIT" and not self.cache_reference:
      raise ValueError("cache-hit approval requires cache_reference")
    if self.authorization_mode == "PERSISTENT_GRANT" and not self.grant_reference:
      raise ValueError("persistent-grant approval requires grant_reference")
    if self.authorization_mode != "CACHE_HIT" and self.cache_reference is not None:
      raise ValueError("cache_reference requires CACHE_HIT authorization_mode")
    if self.authorization_mode != "PERSISTENT_GRANT" and self.grant_reference is not None:
      raise ValueError("grant_reference requires PERSISTENT_GRANT authorization_mode")
    identity_values = (
      self.change_set_id,
      self.change_hash,
      self.base_vector_hash,
      self.reviewed_change_binding_digest,
      self.review_reference,
      self.execution_semantics_digest,
    )
    if self.identity_source is None:
      if any(value is not None for value in identity_values):
        raise ValueError("approval identity fields require identity_source")
      return
    if self.identity_source not in {"change_set", "reviewed_change_binding"}:
      raise ValueError("unsupported approval identity_source")
    if self.review_reference is not None:
      if not isinstance(self.review_reference, dict):
        raise ValueError("review_reference must be an object")
      try:
        canonical_json_strict(self.review_reference)
      except (TypeError, ValueError) as exc:
        raise ValueError("review_reference must contain strict JSON values") from exc
      self.review_reference = _normalize_review_reference(self.review_reference)
    if self.identity_source == "change_set":
      for label, value in (
        ("change_set_id", self.change_set_id),
        ("change_hash", self.change_hash),
        ("base_vector_hash", self.base_vector_hash),
      ):
        if not isinstance(value, str) or _HEX_DIGEST_RE.fullmatch(value) is None:
          raise ValueError(f"{label} must be a lowercase sha256 hex digest")
      if self.reviewed_change_binding_digest is not None:
        raise ValueError("change_set identity cannot carry a reviewed binding digest")
      if self.execution_semantics_digest is not None:
        raise ValueError("change_set identity cannot carry execution semantics digest")
      return
    for label, value in (
      ("change_hash", self.change_hash),
      ("base_vector_hash", self.base_vector_hash),
    ):
      if not isinstance(value, str) or _SHA256_DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 digest")
    if (
      not isinstance(self.change_set_id, str)
      or _REVIEWED_CHANGE_ID_RE.fullmatch(self.change_set_id) is None
    ):
      raise ValueError("reviewed binding identity requires a canonical change_set_id")
    for label, value in (
      ("reviewed_change_binding_digest", self.reviewed_change_binding_digest),
      ("execution_semantics_digest", self.execution_semantics_digest),
    ):
      if not isinstance(value, str) or _SHA256_DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 digest")
    contract = _reviewed_change_contract()
    try:
      contract.verify_reviewed_change_identity_v1(
        reviewed_change_binding_digest=self.reviewed_change_binding_digest,
        change_set_id=self.change_set_id,
        change_hash=self.change_hash,
        base_vector_hash=self.base_vector_hash,
        execution_semantics_digest=self.execution_semantics_digest,
      )
    except contract.ReviewedChangeBindingError as exc:
      raise ValueError(
        "reviewed binding identity fields are not canonically linked"
      ) from exc


def revalidate_approval_request(request: ApprovalRequest) -> ApprovalRequest:
  """Return a freshly validated copy before persistence or audit emission."""

  if not isinstance(request, ApprovalRequest):
    raise TypeError("request must be an ApprovalRequest")
  return replace(request)


@dataclass(frozen=True, kw_only=True)
class ApprovalDecision:
  outcome: ApprovalOutcome
  reason: str
  route_target: str | None = None
  route_target_type: str | None = None
  required_decider_count: int = 1
  eligible_decider_count: int = 1
  expiry_seconds: float | int | None = None
  allow_persistent_grant: bool = False
  persistent_grant_scope_hint: str | None = None
  redacted_args_for_audit: dict[str, Any] | None = None
  modified_tool_args: dict[str, Any] | None = None
  chain_trust_window_seconds: int | None = None
  args_predicate: dict[str, Any] | None = None
  notification_policy: ApprovalNotificationPolicy = "auto"
  policy_id: str = "single-user"
  policy_version: str = "1"
  grant_reference: str | None = None


@dataclass(frozen=True, kw_only=True)
class ApprovalVote:
  vote_id: str
  approval_id: str
  decider_id: str
  decider_role: str | None
  decision: Literal["approved", "denied"]
  decision_reason: str | None
  decided_at: datetime
  external_callback_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class PersistentGrant:
  grant_id: str
  user_id: str
  tool_name: str
  scope_hint: str
  args_predicate: dict[str, Any] | None
  granted_at: datetime
  expires_at: datetime | None
  revoked_at: datetime | None
  granted_via_approval_id: str
  policy_id: str


@dataclass(frozen=True, kw_only=True)
class DelegationGrant:
  delegation_id: str
  delegator_user_id: str
  delegator_run_id: str | None
  delegator_session_id: str | None
  delegator_profile: str
  delegator_channel: str
  bound_excel_session_id: str
  bound_relay_request_id: str
  bound_workbook: str | None
  tool_class_ceiling: frozenset[ToolClass]
  args_predicate: dict[str, Any] | None
  window_seconds: int
  exclude_external_write_bypass: bool = True
  created_at: datetime
  expires_at: datetime | None = None
  revoked_at: datetime | None = None
  consumed_at: datetime | None = None


class ApprovalPolicy(Protocol):
  async def decide(
    self,
    *,
    payload: ApprovalRequestPayload,
    request: ApprovalRequest,
    run_context: RunContext,
  ) -> ApprovalDecision: ...

  async def on_resolve(self, *, request: ApprovalRequest) -> None: ...

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None: ...

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: ToolClass) -> bool: ...


def build_approval_request(
  *,
  tool_call_id: str,
  tool_name: str,
  tool_class: ToolClass,
  tool_args_redacted: dict[str, Any],
  args_hash: str,
  run_context: RunContext,
  reason: str | None = None,
  state: ApprovalState = "created",
  approval_identity: Mapping[str, Any] | None = None,
  approval_constraint: ApprovalConstraint = "standard",
  required_owner_user_id: str | None = None,
) -> ApprovalRequest:
  identity = dict(approval_identity or {})
  allowed_identity_fields = {
    "identity_source",
    "change_set_id",
    "change_hash",
    "base_vector_hash",
    "reviewed_change_binding_digest",
    "review_reference",
    "execution_semantics_digest",
  }
  unexpected = set(identity).difference(allowed_identity_fields)
  if unexpected:
    raise ValueError(
      f"unsupported approval identity fields: {sorted(unexpected)!r}"
    )
  approval_id = new_approval_id()
  parent_id = run_context.parent_approval_id
  return ApprovalRequest(
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    parent_approval_id=parent_id,
    approval_chain_id=parent_id or approval_id,
    delegation_id=run_context.delegation.delegation_id if run_context.delegation is not None else None,
    request_id=run_context.request_id,
    session_id=run_context.session_id,
    run_id=run_context.run_id,
    skill=run_context.skill,
    user_id=run_context.user_id,
    profile=run_context.profile,
    channel=run_context.channel,
    tool_name=tool_name,
    tool_class=tool_class,
    tool_args_redacted=dict(tool_args_redacted),
    args_hash=args_hash,
    reason=reason,
    blast_radius_summary=f"{tool_class}:{tool_name}",
    state=state,
    requested_at=utc_now(),
    tenant_id=run_context.tenant_id,
    model_id=run_context.model_id,
    model_version=run_context.model_version,
    system_prompt_hash=run_context.system_prompt_hash,
    tool_schema_version=run_context.tool_schema_version,
    mcp_server_version=run_context.mcp_server_version,
    policy_bundle_hash=run_context.policy_bundle_hash,
    approval_constraint=approval_constraint,
    required_owner_user_id=required_owner_user_id,
    identity_source=identity.get("identity_source"),
    change_set_id=identity.get("change_set_id"),
    change_hash=identity.get("change_hash"),
    base_vector_hash=identity.get("base_vector_hash"),
    reviewed_change_binding_digest=identity.get(
      "reviewed_change_binding_digest"
    ),
    review_reference=copy.deepcopy(identity.get("review_reference")),
    execution_semantics_digest=identity.get("execution_semantics_digest"),
  )


def constrain_approval_decision(
  request: ApprovalRequest,
  decision: ApprovalDecision,
) -> ApprovalDecision:
  """Apply the durable constraint before policy output can affect lifecycle state."""

  if request.approval_constraint == "standard":
    return decision
  if request.approval_constraint == "legacy_unknown":
    return replace(
      decision,
      outcome="auto_deny",
      reason=(
        "Approval constraint is unknown; replan and obtain a fresh approval"
      ),
      allow_persistent_grant=False,
      persistent_grant_scope_hint=None,
      grant_reference=None,
      modified_tool_args=None,
    )
  if decision.outcome == "auto_deny":
    return replace(
      decision,
      allow_persistent_grant=False,
      persistent_grant_scope_hint=None,
      grant_reference=None,
    )
  return replace(
    decision,
    outcome="request_user_approval",
    reason="Exact promotion requires a fresh decision by its frozen owner",
    route_target=None,
    route_target_type="pending_tools",
    allow_persistent_grant=False,
    persistent_grant_scope_hint=None,
    grant_reference=None,
    modified_tool_args=None,
  )


def approval_is_executable(request: ApprovalRequest) -> bool:
  """Return whether the durable request can authorize its exact mutation."""

  if request.approval_constraint == "standard":
    return request.state in {"approved", "auto_approved"}
  if request.approval_constraint == "legacy_unknown":
    return False
  return (
    request.state == "approved"
    and request.decision == "approved"
    and request.authorization_mode in {"HUMAN", "OWNER_CONTROL_PLANE"}
    and request.decider_role == "owner"
    and request.decider_id == request.required_owner_user_id
  )


def apply_decision_to_request(request: ApprovalRequest, decision: ApprovalDecision) -> ApprovalRequest:
  decision = constrain_approval_decision(request, decision)
  authorization_mode: AuthorizationMode = request.authorization_mode
  grant_reference = request.grant_reference
  if decision.grant_reference is not None:
    authorization_mode = "PERSISTENT_GRANT"
    grant_reference = decision.grant_reference
  elif decision.outcome == "auto_approve" and authorization_mode == "HUMAN":
    authorization_mode = "AUTO_ALLOW"
  return replace(
    request,
    required_decider_count=max(1, int(decision.required_decider_count or 1)),
    eligible_decider_count=max(1, int(decision.eligible_decider_count or 1)),
    args_predicate=copy.deepcopy(decision.args_predicate),
    chain_trust_window_seconds=decision.chain_trust_window_seconds,
    persistent_grant_scope=decision.persistent_grant_scope_hint,
    policy_id=decision.policy_id,
    policy_version=decision.policy_version,
    route_target=decision.route_target,
    route_target_type=decision.route_target_type,
    reason=decision.reason,
    notification_policy=decision.notification_policy,
    authorization_mode=authorization_mode,
    grant_reference=grant_reference,
  )


async def call_policy_safely(
  policy: ApprovalPolicy,
  request: ApprovalRequest,
  raw_tool_args: dict[str, Any],
  run_context: RunContext,
) -> ApprovalDecision:
  """Call `policy.decide()` without leaking raw args in propagated exceptions."""

  _alias = raw_tool_args
  del raw_tool_args
  args_copy: dict[str, Any] | None = None
  payload: ApprovalRequestPayload | None = None
  caught: BaseException | None = None
  try:
    args_copy = dict(_alias)
    payload = ApprovalRequestPayload(
      request.approval_id,
      request.tool_name,
      request.tool_class,
      args_copy,
    )
    return await policy.decide(payload=payload, request=request, run_context=run_context)
  except BaseException as exc:
    caught = exc
  finally:
    if caught is not None and caught.__traceback__ is not None:
      traceback.clear_frames(caught.__traceback__)
    if payload is not None:
      payload.clear_raw_args()
    if args_copy is not None:
      args_copy.clear()
    del args_copy
    del payload
    del _alias

  if isinstance(caught, asyncio.CancelledError):
    del caught
    raise asyncio.CancelledError from None
  if caught is not None:
    del caught
  raise ApprovalPolicyError(
    "approval policy failed (raw args and original exception detail redacted)"
  ) from None


def decision_latency_ms(request: ApprovalRequest, *, now: datetime | None = None) -> int | None:
  decided_at = now or request.decided_at
  if decided_at is None:
    return None
  return max(0, int((decided_at.timestamp() - request.requested_at.timestamp()) * 1000))


def unix_ms() -> int:
  return int(time.time() * 1000)
