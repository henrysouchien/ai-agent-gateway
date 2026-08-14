from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from agent_gateway import approval_store as approval_store_module
from agent_gateway.approval_policy import ApprovalRequest, ApprovalVote
from agent_gateway.approval_store import SQLiteApprovalStore


def _request(*, deadline: datetime) -> ApprovalRequest:
  return ApprovalRequest(
    approval_id="approval-deadline",
    tool_call_id="tool-deadline",
    parent_approval_id=None,
    approval_chain_id="approval-deadline",
    request_id="run-deadline",
    session_id="session-deadline",
    run_id="run-deadline",
    user_id="owner",
    profile="analyst",
    channel="tui",
    tool_name="memory_write",
    tool_class="state_write",
    tool_args_redacted={},
    args_hash="deadline-hash",
    reason="deadline test",
    blast_radius_summary="state_write:memory_write",
    state="pending_user",
    requested_at=datetime(2034, 12, 31, tzinfo=UTC),
    expires_at=deadline,
    approval_constraint="standard",
  )


def _vote(*, decided_at: datetime) -> ApprovalVote:
  return ApprovalVote(
    vote_id="vote-deadline",
    approval_id="approval-deadline",
    decider_id="owner",
    decider_role="owner",
    decision="approved",
    decision_reason="reviewed",
    decided_at=decided_at,
  )


def _delivery() -> dict[str, str]:
  return {
    "task_id": "bg_deadline",
    "control_run_id": "run-deadline",
    "session_id": "session-deadline",
    "channel_id": "a" * 64,
    "tool_call_id": "tool-deadline",
    "nonce": "nonce-deadline",
  }


@pytest.mark.parametrize("trusted_offset_ns", [0, 1])
def test_vote_at_or_after_deadline_expires_atomically_without_vote_or_outbox(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path,
  trusted_offset_ns: int,
) -> None:
  deadline = datetime(2035, 1, 1, tzinfo=UTC)
  deadline_ns = approval_store_module._datetime_to_epoch_ns(deadline)
  monkeypatch.setattr(
    approval_store_module.time,
    "time_ns",
    lambda: deadline_ns + trusted_offset_ns,
  )
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  asyncio.run(store.create(_request(deadline=deadline)))

  resolved = asyncio.run(
    store.record_vote_with_autonomous_delivery(
      "approval-deadline",
      _vote(decided_at=deadline),
      delivery=_delivery(),
    )
  )

  assert resolved.state == "expired"
  assert resolved.decision == "expired"
  assert resolved.votes_received_count == 0
  with store._connection() as conn:
    vote_count = conn.execute(
      "SELECT COUNT(*) FROM approval_votes WHERE approval_id = ?",
      ("approval-deadline",),
    ).fetchone()[0]
  assert vote_count == 0
  assert asyncio.run(
    store.get_autonomous_approval_delivery(
      "approval-deadline",
      tool_call_id="tool-deadline",
      nonce="nonce-deadline",
    )
  ) is None


def test_vote_one_nanosecond_before_deadline_is_accepted(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path,
) -> None:
  deadline = datetime(2035, 1, 1, tzinfo=UTC)
  deadline_ns = approval_store_module._datetime_to_epoch_ns(deadline)
  monkeypatch.setattr(
    approval_store_module.time,
    "time_ns",
    lambda: deadline_ns - 1,
  )
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  asyncio.run(store.create(_request(deadline=deadline)))

  resolved = asyncio.run(
    store.record_vote_with_autonomous_delivery(
      "approval-deadline",
      _vote(decided_at=deadline),
      delivery=_delivery(),
    )
  )

  assert resolved.state == "approved"
  assert resolved.votes_received_count == 1
  delivery = asyncio.run(
    store.get_autonomous_approval_delivery(
      "approval-deadline",
      tool_call_id="tool-deadline",
      nonce="nonce-deadline",
    )
  )
  assert delivery is not None
  assert delivery["approved"] is True


def test_vote_refuses_server_clock_rollback_without_mutating_approval(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path,
) -> None:
  observed = datetime(2035, 1, 1, 12, tzinfo=UTC)
  deadline = datetime(2035, 1, 1, 13, tzinfo=UTC)
  observed_ns = approval_store_module._datetime_to_epoch_ns(observed)
  monkeypatch.setattr(
    approval_store_module.time,
    "time_ns",
    lambda: observed_ns,
  )
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  asyncio.run(store.create(_request(deadline=deadline)))
  monkeypatch.setattr(
    approval_store_module.time,
    "time_ns",
    lambda: observed_ns - 1,
  )

  with pytest.raises(RuntimeError, match="clock rollback"):
    asyncio.run(
      store.record_vote_with_autonomous_delivery(
        "approval-deadline",
        _vote(decided_at=observed),
        delivery=_delivery(),
      )
    )

  stored = asyncio.run(store.get("approval-deadline"))
  assert stored is not None
  assert stored.state == "pending_user"
  assert stored.votes_received_count == 0
  assert asyncio.run(
    store.get_autonomous_approval_delivery(
      "approval-deadline",
      tool_call_id="tool-deadline",
      nonce="nonce-deadline",
    )
  ) is None
