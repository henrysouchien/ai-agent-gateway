from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway.autonomous_approval_ack import (
  require_durable_autonomous_approval_acknowledgement,
)
from agent_gateway.autonomous_approval_channel import (
  AutonomousApprovalChannelAuthority,
  AutonomousApprovalDecision,
  create_autonomous_approval_channel,
)


DECIDED_AT = datetime(2026, 7, 25, tzinfo=UTC)
DECIDED_AT_NS = int(DECIDED_AT.timestamp()) * 1_000_000_000
LAUNCH_NONCE = "a" * 32


def _authority() -> AutonomousApprovalChannelAuthority:
  return AutonomousApprovalChannelAuthority(
    launch_nonce=LAUNCH_NONCE,
    task_id="bg-1",
    control_run_id="run-1",
    session_id="bg-1",
    channel_id="ab" * 32,
  )


def _event() -> dict[str, Any]:
  return {
    "type": "approval_delivery_acknowledged",
    "launch_nonce": LAUNCH_NONCE,
    "delivery_sequence": 7,
    "approval_id": "approval-1",
    "tool_call_id": "tool-1",
    "nonce": "nonce-1",
    "task_id": "bg-1",
    "control_run_id": "run-1",
    "session_id": "bg-1",
    "channel_id": "ab" * 32,
    "approved": True,
    "allow_tool_type": False,
    "decided_at_ns": DECIDED_AT_NS,
  }


def _decision() -> AutonomousApprovalDecision:
  event = _event()
  return AutonomousApprovalDecision(
    authority=_authority(),
    delivery_sequence=event["delivery_sequence"],
    approval_id=event["approval_id"],
    tool_call_id=event["tool_call_id"],
    nonce=event["nonce"],
    approved=event["approved"],
    decided_at_ns=event["decided_at_ns"],
  )


def _record(*, send: bool = True):
  pair = create_autonomous_approval_channel(
    authority=_authority(),
  )
  if send:
    pair.parent.send(_decision())
  return (
    SimpleNamespace(
      task_id="bg-1",
      control_run_id="run-1",
      session_id="bg-1",
      channel_id="ab" * 32,
      launch_nonce=LAUNCH_NONCE,
      approval_channel=pair.parent,
      owner_user_id="owner",
      user_id="owner",
      channel="tui",
    ),
    pair,
  )


class Store:
  def __init__(self, *, state: str = "published") -> None:
    self.delivery = {
      **_event(),
      "state": state,
      "audit_state": "ready",
      "acknowledged_at": (
        "2026-07-25T00:00:01+00:00"
        if state == "acknowledged"
        else None
      ),
    }
    self.request = SimpleNamespace(
      approval_id="approval-1",
      tool_call_id="tool-1",
      user_id="owner",
      request_id="run-1",
      run_id="run-1",
      session_id="bg-1",
      channel="tui",
      decider_id="owner",
      state="approved",
      decision="approved",
      decided_at=DECIDED_AT,
    )
    self.acknowledge_calls: list[dict[str, Any]] = []

  async def get_autonomous_approval_delivery(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
  ) -> dict[str, Any] | None:
    if (
      approval_id,
      tool_call_id,
      nonce,
    ) != ("approval-1", "tool-1", "nonce-1"):
      return None
    return dict(self.delivery)

  async def get(self, approval_id: str) -> Any:
    return self.request if approval_id == "approval-1" else None

  async def acknowledge_autonomous_approval_delivery(
    self,
    approval_id: str,
    **authority: Any,
  ) -> dict[str, Any]:
    self.acknowledge_calls.append({
      "approval_id": approval_id,
      **authority,
    })
    self.delivery["state"] = "acknowledged"
    self.delivery["acknowledged_at"] = (
      "2026-07-25T00:00:01+00:00"
    )
    return dict(self.delivery)


def test_parent_cas_joins_only_exact_published_acknowledgement() -> None:
  store = Store()
  record, pair = _record()
  try:
    delivery = asyncio.run(
      require_durable_autonomous_approval_acknowledgement(
        store=store,
        record=record,
        event=_event(),
      )
    )
  finally:
    pair.close()

  assert delivery["state"] == "acknowledged"
  assert len(store.acknowledge_calls) == 1
  assert store.acknowledge_calls[0]["task_id"] == "bg-1"
  assert store.acknowledge_calls[0]["decided_at_ns"] == DECIDED_AT_NS


def test_parent_accepts_exact_idempotent_acknowledgement() -> None:
  store = Store(state="acknowledged")
  record, pair = _record()
  try:
    delivery = asyncio.run(
      require_durable_autonomous_approval_acknowledgement(
        store=store,
        record=record,
        event=_event(),
      )
    )
  finally:
    pair.close()

  assert delivery["state"] == "acknowledged"
  assert store.acknowledge_calls == []


@pytest.mark.parametrize(
  ("field_name", "wrong_value"),
  [
    ("launch_nonce", "c" * 32),
    ("delivery_sequence", 8),
    ("task_id", "bg-other"),
    ("control_run_id", "run-other"),
    ("session_id", "bg-other"),
    ("channel_id", "cd" * 32),
    ("approval_id", "approval-other"),
    ("tool_call_id", "tool-other"),
    ("nonce", "nonce-other"),
    ("approved", False),
    ("allow_tool_type", True),
    ("decided_at_ns", DECIDED_AT_NS + 1),
  ],
)
def test_parent_rejects_acknowledgement_authority_mismatch(
  field_name: str,
  wrong_value: Any,
) -> None:
  event = {**_event(), field_name: wrong_value}
  record, pair = _record()
  try:
    with pytest.raises(RuntimeError):
      asyncio.run(
        require_durable_autonomous_approval_acknowledgement(
          store=Store(),
          record=record,
          event=event,
        )
      )
  finally:
    pair.close()


@pytest.mark.parametrize(
  ("field_name", "wrong_value"),
  [
    ("state", "pending"),
    ("audit_state", "pending"),
    ("delivery_sequence", 8),
    ("approved", False),
    ("task_id", "bg-other"),
  ],
)
def test_parent_rejects_non_published_ack_projection(
  field_name: str,
  wrong_value: Any,
) -> None:
  store = Store()
  store.delivery[field_name] = wrong_value
  record, pair = _record()
  try:
    with pytest.raises(RuntimeError, match="published delivery"):
      asyncio.run(
        require_durable_autonomous_approval_acknowledgement(
          store=store,
          record=record,
          event=_event(),
        )
      )
  finally:
    pair.close()


def test_parent_rejects_ack_without_exact_live_send() -> None:
  record, pair = _record(send=False)
  try:
    with pytest.raises(RuntimeError, match="live-session send"):
      asyncio.run(
        require_durable_autonomous_approval_acknowledgement(
          store=Store(),
          record=record,
          event=_event(),
        )
      )
  finally:
    pair.close()


def test_parent_rejects_request_authority_mismatch() -> None:
  store = Store()
  store.request.decider_id = "other"
  record, pair = _record()
  try:
    with pytest.raises(RuntimeError, match="request authority"):
      asyncio.run(
        require_durable_autonomous_approval_acknowledgement(
          store=store,
          record=record,
          event=_event(),
        )
      )
  finally:
    pair.close()
