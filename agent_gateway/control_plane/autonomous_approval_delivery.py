from __future__ import annotations

from typing import Any

from agent_gateway.autonomous_runner import (
  AutonomousRegistry,
  AutonomousTask,
)
from agent_gateway.autonomous_approval_channel import (
  AutonomousApprovalChannelParent,
)


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


def require_matching_autonomous_delivery(
  record: AutonomousTask,
  delivery: dict[str, Any],
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
  delivery: dict[str, Any],
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
  ensure_audited = getattr(
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
  append_transaction = getattr(
    store,
    "autonomous_approval_delivery_append_transaction",
    None,
  )
  if not callable(append_transaction):
    raise RuntimeError(
      "Autonomous approval cancellation fence unavailable"
    )
  duplicate_transaction = getattr(
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
    record_failure = getattr(
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
  "autonomous_approval_authoritative_identity",
  "autonomous_approval_delivery_context",
  "autonomous_decision_unavailable",
  "autonomous_run_accepts_approval_decisions",
  "deliver_autonomous_approval_outbox",
  "require_matching_autonomous_delivery",
]
