from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from datetime import timedelta
from typing import Any, Mapping, Protocol

from agent_gateway.approval_policy import (
  approval_request_from_projection,
  utc_now,
)
from agent_gateway.autonomous_runner import (
  AutonomousRegistry,
  AutonomousTask,
)
from agent_gateway.autonomous_approval_channel import (
  AutonomousApprovalChannelParent,
)


class _EnsureAutonomousApprovalDeliveryAudited(Protocol):
  async def __call__(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
  ) -> dict[str, Any]: ...


class _AutonomousApprovalDeliveryTransaction(Protocol):
  def __call__(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    approved: bool,
  ) -> AbstractContextManager[None]: ...


class _RecordAutonomousApprovalDeliveryFailure(Protocol):
  async def __call__(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    error: str,
  ) -> dict[str, Any]: ...


# The durable pending_user window an operator gets on a delegated request; the
# child's wait is bounded by the same value.
DELEGATED_APPROVAL_EXPIRY_SECONDS = 600


def autonomous_run_accepts_approval_decisions(
  record: AutonomousTask,
) -> bool:
  if record.state not in {"running", "approval_pending", "remediating"}:
    return False
  if record.proc is not None and record.proc.returncode is not None:
    return False
  return True


def autonomous_decision_unavailable(
  record: AutonomousTask,
) -> str | None:
  if not autonomous_run_accepts_approval_decisions(record):
    return "Autonomous run is not running"
  if type(getattr(
    record,
    "approval_channel",
    None,
  )) is not AutonomousApprovalChannelParent:
    return "Autonomous approval channel unavailable"
  return None


def autonomous_approval_delivery_context(
  record: AutonomousTask,
  *,
  tool_call_id: str,
  nonce: str,
) -> dict[str, str]:
  return {
    "task_id": record.task_id,
    "control_run_id": record.control_run_id,
    "session_id": record.session_id,
    "channel_id": record.channel_id,
    "tool_call_id": tool_call_id,
    "nonce": nonce,
  }


def autonomous_approval_authoritative_identity(
  record: AutonomousTask,
  *,
  approval_id: str,
  tool_call_id: str,
) -> dict[str, str | None]:
  owner_user_id = str(record.owner_user_id or "").strip()
  if not owner_user_id:
    raise ValueError("autonomous run owner_user_id is required")
  return {
    "approval_id": approval_id,
    "tool_call_id": tool_call_id,
    "user_id": owner_user_id,
    "request_id": record.control_run_id,
    "run_id": record.control_run_id,
    "session_id": record.session_id,
    "channel": record.channel,
  }


async def create_delegated_autonomous_approval(
  *,
  store: Any,
  policy: Any | None,
  record: AutonomousTask,
  event: dict[str, Any],
) -> Any:
  """Record the durable row for a request a delegated child authored.

  The child owns no ledger, so this is the only writer of its approval request.
  Every identity field is stamped from the run record before the row exists: a
  child cannot name another run, another session or another user's approval.
  """

  if not isinstance(event, dict):
    raise RuntimeError("autonomous approval request event is invalid")
  approval_id = str(event.get("approval_id") or "").strip()
  tool_call_id = str(event.get("tool_call_id") or "").strip()
  nonce = str(event.get("nonce") or "").strip()
  if not approval_id or not tool_call_id or not nonce:
    raise RuntimeError(
      "autonomous approval request event is missing its identity"
    )
  try:
    request = approval_request_from_projection(event.get("durable_request"))
  except (TypeError, ValueError) as exc:
    raise RuntimeError(
      "autonomous approval request projection is invalid"
    ) from exc
  if (
    request.approval_id != approval_id
    or request.tool_call_id != tool_call_id
  ):
    raise RuntimeError(
      "autonomous approval request projection disagrees with its event"
    )
  authoritative_identity = autonomous_approval_authoritative_identity(
    record,
    approval_id=approval_id,
    tool_call_id=tool_call_id,
  )
  requested_at = utc_now()
  request = replace(
    request,
    **authoritative_identity,
    state="pending_user",
    authorization_mode="HUMAN",
    cache_reference=None,
    grant_reference=None,
    requested_at=requested_at,
    expires_at=requested_at + timedelta(
      seconds=DELEGATED_APPROVAL_EXPIRY_SECONDS,
    ),
    decided_at=None,
    decider_id=None,
    decider_role=None,
    decision=None,
    decision_reason=None,
    state_version=0,
    # The child projects these, and before this they were written into the parent's
    # ledger verbatim. They are not descriptive: approvals.py mints a durable
    # PersistentGrant carrying `args_predicate` when the constraint permits reuse, so a
    # child could name the predicate its own future writes would be pre-approved under.
    # The parent neutralises every reuse/grant field; a delegated approval is decided
    # once, for one write, and grants nothing forward.
    approval_reuse_mode="legacy",
    approval_reuse_key=None,
    persistent_grant_scope=None,
    args_predicate=None,
    # The child must not choose whether its own write needs a frozen-owner approval.
    # Pinned to the value delegated requests effectively carry today, so no live flow
    # changes; the point is that the PARENT decides it. `legacy_unknown` would make
    # every delegated approval un-approvable (409) and `fresh_human_owner` would add
    # owner-role checks to every decision, so neither is a safe unilateral default.
    approval_constraint="standard",
    required_owner_user_id=None,
    policy_id=str(
      getattr(policy, "policy_id", None) or request.policy_id
    ),
    policy_version=str(
      getattr(policy, "policy_version", None) or request.policy_version
    ),
  )
  stored, _created = await store.create_or_get_by_tool_call_id(request)
  return stored


def require_matching_autonomous_delivery(
  record: AutonomousTask,
  delivery: Mapping[str, Any],
  request_record: Any,
  *,
  approval_id: str,
  tool_call_id: str,
  nonce: str,
  approved: bool,
) -> None:
  expected_delivery = {
    "approval_id": approval_id,
    "tool_call_id": tool_call_id,
    "nonce": nonce,
    "task_id": record.task_id,
    "control_run_id": record.control_run_id,
    "session_id": record.session_id,
    "channel_id": record.channel_id,
    "allow_tool_type": False,
  }
  if any(
    delivery.get(field_name) != expected
    for field_name, expected in expected_delivery.items()
  ):
    raise RuntimeError(
      "Autonomous approval delivery outbox identity mismatch"
    )
  if delivery.get("approved") != approved:
    raise ValueError(
      "Approval decision conflicts with the durable autonomous decision"
    )
  expected_request = {
    **autonomous_approval_authoritative_identity(
      record,
      approval_id=approval_id,
      tool_call_id=tool_call_id,
    ),
    "decider_id": str(record.owner_user_id or "").strip(),
  }
  if any(
    getattr(request_record, field_name, None) != expected
    for field_name, expected in expected_request.items()
  ):
    raise RuntimeError(
      "Autonomous approval durable identity mismatch"
    )
  expected_state = "approved" if approved else "denied"
  if getattr(request_record, "state", None) != expected_state:
    raise RuntimeError(
      "Autonomous approval delivery disagrees with durable state"
    )


async def deliver_autonomous_approval_outbox(
  *,
  registry: AutonomousRegistry,
  store: Any,
  record: AutonomousTask,
  request_record: Any,
  delivery: Mapping[str, Any],
  approval_id: str,
  tool_call_id: str,
  nonce: str,
  approved: bool,
  user_id: str,
  channel: str | None,
) -> None:
  require_matching_autonomous_delivery(
    record,
    delivery,
    request_record,
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    nonce=nonce,
    approved=approved,
  )
  delivery_state = delivery.get("state")
  if delivery_state in {"published", "acknowledged"}:
    return
  if delivery_state != "pending":
    raise RuntimeError(
      "Autonomous approval delivery outbox state is invalid"
    )
  ensure_audited: _EnsureAutonomousApprovalDeliveryAudited | None = getattr(
    store,
    "ensure_autonomous_approval_delivery_audited",
    None,
  )
  if not callable(ensure_audited):
    raise RuntimeError(
      "Autonomous approval delivery audit gate unavailable"
    )
  delivery = await ensure_audited(
    approval_id,
    tool_call_id=tool_call_id,
    nonce=nonce,
  )
  require_matching_autonomous_delivery(
    record,
    delivery,
    request_record,
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    nonce=nonce,
    approved=approved,
  )
  if delivery.get("audit_state") != "ready":
    raise RuntimeError(
      "Autonomous approval delivery audit receipt is not ready"
    )
  if delivery.get("state") != "pending":
    if delivery.get("state") in {"published", "acknowledged"}:
      return
    raise RuntimeError(
      "Autonomous approval delivery outbox state is invalid"
    )
  unavailable = autonomous_decision_unavailable(record)
  if unavailable is not None:
    raise RuntimeError(unavailable)
  append_transaction: _AutonomousApprovalDeliveryTransaction | None = getattr(
    store,
    "autonomous_approval_delivery_append_transaction",
    None,
  )
  if not callable(append_transaction):
    raise RuntimeError(
      "Autonomous approval cancellation fence unavailable"
    )
  duplicate_transaction: _AutonomousApprovalDeliveryTransaction | None = getattr(
    store,
    "autonomous_approval_delivery_duplicate_transaction",
    None,
  )
  if not callable(duplicate_transaction):
    raise RuntimeError(
      "Autonomous approval duplicate recovery unavailable"
    )

  async def publish_to_child_inbox() -> Any:
    return await registry.send_approval_decision(
      record.control_run_id,
      user_id=user_id,
      channel=channel,
      approval_id=approval_id,
      tool_call_id=tool_call_id,
      nonce=nonce,
      approved=approved,
      decided_at_ns=int(delivery["decided_at_ns"]),
      delivery_sequence=int(delivery["delivery_sequence"]),
      publication_transaction=lambda: append_transaction(
        approval_id,
        tool_call_id=tool_call_id,
        nonce=nonce,
        approved=approved,
      ),
      sent_reconciliation=lambda: duplicate_transaction(
        approval_id,
        tool_call_id=tool_call_id,
        nonce=nonce,
        approved=approved,
      ),
    )

  try:
    await publish_to_child_inbox()
  except BaseException as exc:
    record_failure: _RecordAutonomousApprovalDeliveryFailure | None = getattr(
      store,
      "record_autonomous_approval_delivery_failure",
      None,
    )
    if callable(record_failure) and isinstance(exc, Exception):
      await record_failure(
        approval_id,
        tool_call_id=tool_call_id,
        nonce=nonce,
        error=f"{type(exc).__name__}: {exc}",
      )
    raise


__all__ = [
  "DELEGATED_APPROVAL_EXPIRY_SECONDS",
  "autonomous_approval_authoritative_identity",
  "create_delegated_autonomous_approval",
  "autonomous_approval_delivery_context",
  "autonomous_decision_unavailable",
  "autonomous_run_accepts_approval_decisions",
  "deliver_autonomous_approval_outbox",
  "require_matching_autonomous_delivery",
]
