from __future__ import annotations

import copy
import hashlib
import json
import time
import traceback
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol


ToolClass = Literal[
  "read",
  "pure_transform",
  "artifact_write",
  "state_write",
  "external_write",
  "portfolio_config",
  "irreversible",
]

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


def sha256_args(value: Any) -> str:
  return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ApprovalPolicyError(RuntimeError):
  """Sanitized wrapper for policy failures.

  The original exception message and raw tool arguments are deliberately not
  included because SDK and SSE paths stringify callback failures.
  """


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


@dataclass(frozen=True, kw_only=True)
class ApprovalDecision:
  outcome: ApprovalOutcome
  reason: str
  route_target: str | None = None
  route_target_type: str | None = None
  required_decider_count: int = 1
  eligible_decider_count: int = 1
  expiry_seconds: int | None = None
  allow_persistent_grant: bool = False
  persistent_grant_scope_hint: str | None = None
  redacted_args_for_audit: dict[str, Any] | None = None
  modified_tool_args: dict[str, Any] | None = None
  chain_trust_window_seconds: int | None = None
  args_predicate: dict[str, Any] | None = None
  policy_id: str = "single-user"
  policy_version: str = "1"


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
) -> ApprovalRequest:
  approval_id = new_approval_id()
  parent_id = run_context.parent_approval_id
  return ApprovalRequest(
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    parent_approval_id=parent_id,
    approval_chain_id=parent_id or approval_id,
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
  )


def apply_decision_to_request(request: ApprovalRequest, decision: ApprovalDecision) -> ApprovalRequest:
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
