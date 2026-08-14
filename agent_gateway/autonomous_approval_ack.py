from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .autonomous_approval_channel import (
  AutonomousApprovalChannelAuthority,
  AutonomousApprovalChannelParent,
  AutonomousApprovalDecision,
)


_ACK_FIELDS = frozenset({
  "type",
  "launch_nonce",
  "delivery_sequence",
  "approval_id",
  "tool_call_id",
  "nonce",
  "task_id",
  "control_run_id",
  "session_id",
  "channel_id",
  "approved",
  "allow_tool_type",
  "decided_at_ns",
})


def _canonical_text(value: Any, *, field_name: str) -> str:
  if (
    type(value) is not str
    or not value
    or value != value.strip()
    or len(value) > 512
    or any(ord(character) < 0x20 for character in value)
  ):
    raise RuntimeError(
      f"autonomous approval acknowledgement {field_name} is invalid"
    )
  return value


def _epoch_ns(value: datetime) -> int:
  normalized = value.astimezone(UTC)
  epoch = datetime(1970, 1, 1, tzinfo=UTC)
  delta = normalized - epoch
  return (
    ((delta.days * 86_400) + delta.seconds) * 1_000_000_000
    + delta.microseconds * 1_000
  )


async def require_durable_autonomous_approval_acknowledgement(
  *,
  store: Any,
  record: Any,
  event: dict[str, Any],
) -> dict[str, Any]:
  """CAS an exact live-session child ACK into parent-owned SQLite."""
  if not isinstance(event, dict) or set(event) != _ACK_FIELDS:
    raise RuntimeError(
      "autonomous approval acknowledgement fields are invalid"
    )
  if event.get("type") != "approval_delivery_acknowledged":
    raise RuntimeError(
      "autonomous approval acknowledgement type is invalid"
    )
  approval_id = _canonical_text(
    event.get("approval_id"),
    field_name="approval_id",
  )
  tool_call_id = _canonical_text(
    event.get("tool_call_id"),
    field_name="tool_call_id",
  )
  nonce = _canonical_text(
    event.get("nonce"),
    field_name="nonce",
  )
  approved = event.get("approved")
  delivery_sequence = event.get("delivery_sequence")
  decided_at_ns = event.get("decided_at_ns")
  if (
    type(approved) is not bool
    or type(delivery_sequence) is not int
    or delivery_sequence < 1
    or event.get("allow_tool_type") is not False
    or type(decided_at_ns) is not int
    or decided_at_ns < 1
  ):
    raise RuntimeError(
      "autonomous approval acknowledgement decision is invalid"
    )
  expected_run_authority = {
    "task_id": getattr(record, "task_id", None),
    "control_run_id": getattr(record, "control_run_id", None),
    "session_id": getattr(record, "session_id", None),
    "channel_id": getattr(record, "channel_id", None),
  }
  if any(
    event.get(field_name) != expected
    for field_name, expected in expected_run_authority.items()
  ):
    raise RuntimeError(
      "autonomous approval acknowledgement changed run authority"
    )
  try:
    channel_authority = AutonomousApprovalChannelAuthority(
      launch_nonce=event.get("launch_nonce"),
      task_id=event.get("task_id"),
      control_run_id=event.get("control_run_id"),
      session_id=event.get("session_id"),
      channel_id=event.get("channel_id"),
    )
    decision = AutonomousApprovalDecision(
      authority=channel_authority,
      delivery_sequence=delivery_sequence,
      approval_id=approval_id,
      tool_call_id=tool_call_id,
      nonce=nonce,
      approved=approved,
      decided_at_ns=decided_at_ns,
    )
  except (TypeError, ValueError) as exc:
    raise RuntimeError(
      "autonomous approval acknowledgement fields are invalid"
    ) from exc
  if channel_authority.launch_nonce != getattr(
    record,
    "launch_nonce",
    None,
  ):
    raise RuntimeError(
      "autonomous approval acknowledgement changed launch authority"
    )
  approval_channel = getattr(record, "approval_channel", None)
  if type(approval_channel) is not AutonomousApprovalChannelParent:
    raise RuntimeError(
      "autonomous approval acknowledgement channel is unavailable"
    )
  approval_channel.require_sent(decision)
  get_delivery = getattr(
    store,
    "get_autonomous_approval_delivery",
    None,
  )
  get_request = getattr(store, "get", None)
  acknowledge = getattr(
    store,
    "acknowledge_autonomous_approval_delivery",
    None,
  )
  if (
    not callable(get_delivery)
    or not callable(get_request)
    or not callable(acknowledge)
  ):
    raise RuntimeError(
      "autonomous approval acknowledgement store is unavailable"
    )
  delivery = await get_delivery(
    approval_id,
    tool_call_id=tool_call_id,
    nonce=nonce,
  )
  if not isinstance(delivery, dict):
    raise RuntimeError(
      "autonomous approval acknowledgement has no durable delivery"
    )
  expected_delivery: dict[str, Any] = {
    "approval_id": approval_id,
    "tool_call_id": tool_call_id,
    "nonce": nonce,
    **expected_run_authority,
    "approved": approved,
    "allow_tool_type": False,
    "decided_at_ns": decided_at_ns,
    "delivery_sequence": delivery_sequence,
    "audit_state": "ready",
  }
  if any(
    delivery.get(field_name) != expected
    for field_name, expected in expected_delivery.items()
  ) or delivery.get("state") not in {"published", "acknowledged"}:
    raise RuntimeError(
      "autonomous approval acknowledgement has no exact published delivery"
    )
  request = await get_request(approval_id)
  if request is None:
    raise RuntimeError(
      "autonomous approval acknowledgement request is missing"
    )
  owner_user_id = (
    getattr(record, "owner_user_id", None)
    or getattr(record, "user_id", None)
  )
  expected_state = "approved" if approved else "denied"
  expected_request = {
    "approval_id": approval_id,
    "tool_call_id": tool_call_id,
    "user_id": owner_user_id,
    "request_id": getattr(record, "control_run_id", None),
    "run_id": getattr(record, "control_run_id", None),
    "session_id": getattr(record, "session_id", None),
    "channel": getattr(record, "channel", None),
    "decider_id": owner_user_id,
    "state": expected_state,
    "decision": expected_state,
  }
  if any(
    getattr(request, field_name, None) != expected
    for field_name, expected in expected_request.items()
  ):
    raise RuntimeError(
      "autonomous approval acknowledgement request authority changed"
    )
  decided_at = getattr(request, "decided_at", None)
  if (
    not isinstance(decided_at, datetime)
    or _epoch_ns(decided_at) != decided_at_ns
  ):
    raise RuntimeError(
      "autonomous approval acknowledgement decision time changed"
    )
  if delivery["state"] == "published":
    delivery = await acknowledge(
      approval_id,
      task_id=event["task_id"],
      control_run_id=event["control_run_id"],
      session_id=event["session_id"],
      channel_id=event["channel_id"],
      tool_call_id=tool_call_id,
      nonce=nonce,
      approved=approved,
      decided_at_ns=decided_at_ns,
    )
  if (
    not isinstance(delivery, dict)
    or delivery.get("state") != "acknowledged"
    or delivery.get("acknowledged_at") is None
    or any(
      delivery.get(field_name) != expected
      for field_name, expected in expected_delivery.items()
    )
  ):
    raise RuntimeError(
      "autonomous approval acknowledgement was not durably joined"
    )
  return delivery


__all__ = [
  "require_durable_autonomous_approval_acknowledgement",
]
