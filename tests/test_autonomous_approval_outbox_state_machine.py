from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import os
from pathlib import Path
import sqlite3
import stat
import threading
from typing import Any

import pytest

from agent_gateway import approval_store as approval_store_module
from agent_gateway.approval_policy import ApprovalRequest, ApprovalVote
from agent_gateway.approval_store import (
  BATCH_CANCELLED_REASON,
  CONVERSATION_CLOSED_REASON,
  GATEWAY_SHUTDOWN_REASON,
  RUN_CANCELLED_REASON,
  PersistentGrantCancellationFenced,
  SQLiteApprovalStore,
)
from agent_gateway.control_plane.autonomous_approval_drainer import (
  AutonomousApprovalDeliveryCoordinator,
)


class RecordingAuditEmitter:
  def __init__(
    self,
    *,
    fail_once_on: str | None = None,
  ) -> None:
    self.calls: list[dict[str, Any]] = []
    self.fail_once_on = fail_once_on

  async def emit_audit_for_lifecycle_event(
    self,
    **kwargs: Any,
  ) -> None:
    self.calls.append(dict(kwargs))
    if kwargs["event_type"] == self.fail_once_on:
      self.fail_once_on = None
      raise OSError("injected durable audit failure")


def _request(suffix: str) -> ApprovalRequest:
  return ApprovalRequest(
    approval_id=f"approval-{suffix}",
    tool_call_id=f"tool-{suffix}",
    parent_approval_id=None,
    approval_chain_id=f"approval-{suffix}",
    request_id=f"run-{suffix}",
    session_id=f"session-{suffix}",
    run_id=f"run-{suffix}",
    user_id="owner",
    profile="analyst",
    channel="tui",
    tool_name="memory_write",
    tool_class="state_write",
    tool_args_redacted={},
    args_hash=f"args-{suffix}",
    reason="state-machine test",
    blast_radius_summary="state_write:memory_write",
    state="pending_user",
    requested_at=datetime(2026, 1, 1, tzinfo=UTC),
    approval_constraint="standard",
  )


def _vote(
  suffix: str,
  *,
  approved: bool = True,
) -> ApprovalVote:
  return ApprovalVote(
    vote_id=f"vote-{suffix}",
    approval_id=f"approval-{suffix}",
    decider_id="owner",
    decider_role="owner",
    decision="approved" if approved else "denied",
    decision_reason="reviewed",
    decided_at=datetime(
      2026,
      1,
      2,
      3,
      4,
      5,
      678901,
      tzinfo=UTC,
    ),
  )


def _delivery(suffix: str) -> dict[str, str]:
  return {
    "task_id": f"bg_{suffix}",
    "control_run_id": f"run-{suffix}",
    "session_id": f"session-{suffix}",
    "channel_id": "a" * 64,
    "tool_call_id": f"tool-{suffix}",
    "nonce": f"nonce-{suffix}",
  }


def _get_delivery(
  store: SQLiteApprovalStore,
  suffix: str,
) -> dict[str, Any]:
  delivery = asyncio.run(
    store.get_autonomous_approval_delivery(
      f"approval-{suffix}",
      tool_call_id=f"tool-{suffix}",
      nonce=f"nonce-{suffix}",
    )
  )
  assert delivery is not None
  return delivery


def _create_decision(
  store: SQLiteApprovalStore,
  suffix: str,
  *,
  approved: bool = True,
) -> dict[str, Any]:
  asyncio.run(store.create(_request(suffix)))
  asyncio.run(
    store.record_vote_with_autonomous_delivery(
      f"approval-{suffix}",
      _vote(suffix, approved=approved),
      delivery=_delivery(suffix),
    )
  )
  return _get_delivery(store, suffix)


def _acknowledgment(
  suffix: str,
  delivery: dict[str, Any],
) -> dict[str, Any]:
  return {
    "approval_id": f"approval-{suffix}",
    **_delivery(suffix),
    "approved": bool(delivery["approved"]),
    "decided_at_ns": int(delivery["decided_at_ns"]),
  }


def test_empty_noncanonical_outbox_is_replaced(
  tmp_path: Path,
) -> None:
  database_path = tmp_path / "approvals.sqlite3"
  with sqlite3.connect(database_path) as conn:
    conn.execute(
      "CREATE TABLE autonomous_approval_delivery_outbox "
      "(legacy_id TEXT PRIMARY KEY)"
    )

  SQLiteApprovalStore(database_path)

  with sqlite3.connect(database_path) as conn:
    columns = {
      str(row[1])
      for row in conn.execute(
        "PRAGMA table_info(autonomous_approval_delivery_outbox)"
      )
    }
  assert columns == set(
    approval_store_module._AUTONOMOUS_APPROVAL_DELIVERY_OUTBOX_COLUMNS
  )


def test_noncanonical_outbox_with_rows_fails_closed(
  tmp_path: Path,
) -> None:
  database_path = tmp_path / "approvals.sqlite3"
  with sqlite3.connect(database_path) as conn:
    conn.execute(
      "CREATE TABLE autonomous_approval_delivery_outbox "
      "(legacy_id TEXT PRIMARY KEY)"
    )
    conn.execute(
      "INSERT INTO autonomous_approval_delivery_outbox VALUES ('legacy')"
    )

  with pytest.raises(
    RuntimeError,
    match="noncanonical autonomous approval delivery outbox contains rows",
  ):
    SQLiteApprovalStore(database_path)


def test_audit_failure_replays_deterministically_before_publication(
  tmp_path: Path,
) -> None:
  emitter = RecordingAuditEmitter(fail_once_on="approved")
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=emitter,
  )
  asyncio.run(store.create(_request("replay")))
  emitter.calls.clear()

  with pytest.raises(OSError, match="injected durable audit failure"):
    asyncio.run(
      store.record_vote_with_autonomous_delivery(
        "approval-replay",
        _vote("replay"),
        delivery=_delivery("replay"),
      )
    )

  pending = _get_delivery(store, "replay")
  assert pending["state"] == "pending"
  assert pending["audit_state"] == "pending"
  with pytest.raises(RuntimeError, match="audit receipt is not ready"):
    with store.autonomous_approval_delivery_append_transaction(
      "approval-replay",
      tool_call_id="tool-replay",
      nonce="nonce-replay",
      approved=True,
    ):
      pytest.fail("unaudited decision reached the append body")

  first_vote, first_terminal = emitter.calls
  retried_request = asyncio.run(
    store.record_vote_with_autonomous_delivery(
      "approval-replay",
      _vote("replay"),
      delivery=_delivery("replay"),
    )
  )
  assert retried_request.state == "approved"
  ready = _get_delivery(store, "replay")

  replay_vote, replay_terminal = emitter.calls[2:]
  assert ready["audit_state"] == "ready"
  assert replay_vote["entry_id"] == first_vote["entry_id"]
  assert replay_terminal["entry_id"] == first_terminal["entry_id"]
  assert replay_vote["event_ts"] == _vote("replay").decided_at
  assert replay_terminal["event_ts"] == _vote("replay").decided_at
  assert ready["vote_audit_entry_id"] == replay_vote["entry_id"]
  assert (
    ready["terminal_audit_entry_id"]
    == replay_terminal["entry_id"]
  )

  call_count = len(emitter.calls)
  same_receipt = asyncio.run(
    store.ensure_autonomous_approval_delivery_audited(
      "approval-replay",
      tool_call_id="tool-replay",
      nonce="nonce-replay",
    )
  )
  assert same_receipt == ready
  assert len(emitter.calls) == call_count


@pytest.mark.parametrize("operation", ["vote", "cancellation"])
def test_post_commit_audit_failure_wake_recovers_same_process(
  tmp_path: Path,
  operation: str,
) -> None:
  async def scenario() -> None:
    terminal_event = "approved" if operation == "vote" else "denied"
    suffix = f"recover-{operation}"
    emitter = RecordingAuditEmitter(
      fail_once_on=terminal_event
    )
    store = SQLiteApprovalStore(
      tmp_path / "approvals.sqlite3",
      audit_emitter=emitter,
    )
    await store.create(_request(suffix))
    coordinator = AutonomousApprovalDeliveryCoordinator(
      store=store,
      registry=object(),  # type: ignore[arg-type]
    )
    delivered = asyncio.Event()

    async def recover(delivery: dict[str, Any]) -> None:
      await store.ensure_autonomous_approval_delivery_audited(
        delivery["approval_id"],
        tool_call_id=delivery["tool_call_id"],
        nonce=delivery["nonce"],
      )
      with store.autonomous_approval_delivery_append_transaction(
        delivery["approval_id"],
        tool_call_id=delivery["tool_call_id"],
        nonce=delivery["nonce"],
        approved=bool(delivery["approved"]),
      ):
        pass
      delivered.set()

    coordinator._deliver_one = recover  # type: ignore[method-assign]
    coordinator.start()
    try:
      with pytest.raises(
        OSError,
        match="injected durable audit failure",
      ):
        if operation == "vote":
          await store.record_vote_with_autonomous_delivery(
            f"approval-{suffix}",
            _vote(suffix),
            delivery=_delivery(suffix),
          )
        else:
          await store.terminalize_pending_for_cancellation(
            f"approval-{suffix}",
            expected_tool_call_id=f"tool-{suffix}",
            expected_user_id="owner",
            expected_request_id=f"run-{suffix}",
            expected_run_id=f"run-{suffix}",
            expected_session_id=f"session-{suffix}",
            expected_channel="tui",
            decider_id="owner",
            decider_role="owner",
            decision_reason=RUN_CANCELLED_REASON,
            autonomous_delivery=_delivery(suffix),
          )
    finally:
      coordinator.wake()
    await asyncio.wait_for(delivered.wait(), timeout=1)
    await coordinator.shutdown()

    recovered = await store.get_autonomous_approval_delivery(
      f"approval-{suffix}",
      tool_call_id=f"tool-{suffix}",
      nonce=f"nonce-{suffix}",
    )
    assert recovered is not None
    assert recovered["audit_state"] == "ready"
    assert recovered["state"] == "published"

  asyncio.run(scenario())


def test_append_failure_rolls_back_and_sqlite_serializes_cancellation(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  ready = _create_decision(store, "append")
  assert ready["audit_state"] == "ready"

  with pytest.raises(OSError, match="injected append failure"):
    with store.autonomous_approval_delivery_append_transaction(
      "approval-append",
      tool_call_id="tool-append",
      nonce="nonce-append",
      approved=True,
    ):
      raise OSError("injected append failure")
  assert _get_delivery(store, "append")["state"] == "pending"

  competitor = sqlite3.connect(
    store.path,
    isolation_level=None,
    timeout=0,
  )
  try:
    with store.autonomous_approval_delivery_append_transaction(
      "approval-append",
      tool_call_id="tool-append",
      nonce="nonce-append",
      approved=True,
    ):
      with pytest.raises(sqlite3.OperationalError, match="locked"):
        competitor.execute("BEGIN IMMEDIATE")
  finally:
    competitor.close()
  assert _get_delivery(store, "append")["state"] == "published"


def test_cancellation_fence_wins_before_publication(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  _create_decision(store, "fenced")
  asyncio.run(
    store.fence_persistent_grants_for_cancellation(
      "approval-fenced",
      expected_tool_call_id="tool-fenced",
      expected_user_id="owner",
      expected_request_id="run-fenced",
      expected_run_id="run-fenced",
      expected_session_id="session-fenced",
      expected_channel="tui",
    )
  )

  with pytest.raises(PersistentGrantCancellationFenced):
    with store.autonomous_approval_delivery_append_transaction(
      "approval-fenced",
      tool_call_id="tool-fenced",
      nonce="nonce-fenced",
      approved=True,
    ):
      pytest.fail("fenced decision reached the append body")
  assert _get_delivery(store, "fenced")["state"] == "pending"


def _cancel(
  store: SQLiteApprovalStore,
  suffix: str,
  *,
  decider_id: str,
  decider_role: str | None = None,
  decision_reason: str,
) -> tuple[ApprovalRequest, bool, bool]:
  return asyncio.run(
    store.terminalize_pending_for_cancellation(
      f"approval-{suffix}",
      expected_tool_call_id=f"tool-{suffix}",
      expected_user_id="owner",
      expected_request_id=f"run-{suffix}",
      expected_run_id=f"run-{suffix}",
      expected_session_id=f"session-{suffix}",
      expected_channel="tui",
      decider_id=decider_id,
      decider_role=decider_role,
      decision_reason=decision_reason,
    )
  )


def _vote_row_count(store: SQLiteApprovalStore, suffix: str) -> int:
  connection = sqlite3.connect(store.path)
  try:
    row = connection.execute(
      "SELECT COUNT(*) FROM approval_votes WHERE approval_id = ?",
      (f"approval-{suffix}",),
    ).fetchone()
  finally:
    connection.close()
  return int(row[0])


def test_cancellation_denies_pending_without_vote_and_first_terminal_wins(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  asyncio.run(store.create(_request("pending")))
  cancelled, transitioned, identity_matches = _cancel(
    store,
    "pending",
    decider_id="owner",
    decider_role="owner",
    decision_reason=RUN_CANCELLED_REASON,
  )
  assert (transitioned, identity_matches) == (True, True)
  assert cancelled.state == "denied"
  assert cancelled.decision == "denied"
  assert cancelled.decision_reason == RUN_CANCELLED_REASON
  assert _vote_row_count(store, "pending") == 0

  repeated, transitioned, identity_matches = _cancel(
    store,
    "pending",
    decider_id="other",
    decision_reason=BATCH_CANCELLED_REASON,
  )
  assert (transitioned, identity_matches) == (False, True)
  assert repeated.state == "denied"
  assert repeated.decision_reason == RUN_CANCELLED_REASON

  _create_decision(store, "decided")
  decided, transitioned, identity_matches = _cancel(
    store,
    "decided",
    decider_id="owner",
    decision_reason=RUN_CANCELLED_REASON,
  )
  assert (transitioned, identity_matches) == (False, True)
  assert decided.state == "approved"
  assert decided.decision_reason == "reviewed"
  assert _vote_row_count(store, "decided") == 1


@pytest.mark.parametrize(
  "reason",
  [
    RUN_CANCELLED_REASON,
    BATCH_CANCELLED_REASON,
    GATEWAY_SHUTDOWN_REASON,
    CONVERSATION_CLOSED_REASON,
  ],
)
def test_cancellation_reasons_are_stored_verbatim_on_a_denied_request(
  tmp_path: Path,
  reason: str,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  asyncio.run(store.create(_request("pending")))
  cancelled, transitioned, identity_matches = _cancel(
    store,
    "pending",
    decider_id="owner",
    decision_reason=reason,
  )
  assert (transitioned, identity_matches) == (True, True)
  assert cancelled.state in approval_store_module.TERMINAL_STATES
  assert cancelled.decision == "denied"
  assert cancelled.decision_reason == reason


def test_exact_duplicate_reconciliation_is_idempotent(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  _create_decision(store, "duplicate")

  first = store.reconcile_autonomous_approval_delivery_duplicate(
    "approval-duplicate",
    tool_call_id="tool-duplicate",
    nonce="nonce-duplicate",
    approved=True,
  )
  second = store.reconcile_autonomous_approval_delivery_duplicate(
    "approval-duplicate",
    tool_call_id="tool-duplicate",
    nonce="nonce-duplicate",
    approved=True,
  )

  assert first["state"] == "published"
  assert second == first
  assert first["attempt_count"] == 1
  with pytest.raises(ValueError, match="decision changed"):
    store.reconcile_autonomous_approval_delivery_duplicate(
      "approval-duplicate",
      tool_call_id="tool-duplicate",
      nonce="nonce-duplicate",
      approved=False,
    )
  assert _get_delivery(store, "duplicate") == first


def test_two_duplicate_retriers_converge_on_one_publication(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  _create_decision(store, "duplicate-race")
  barrier = threading.Barrier(2)

  def reconcile() -> dict[str, Any]:
    barrier.wait(timeout=30)
    return store.reconcile_autonomous_approval_delivery_duplicate(
      "approval-duplicate-race",
      tool_call_id="tool-duplicate-race",
      nonce="nonce-duplicate-race",
      approved=True,
    )

  with ThreadPoolExecutor(max_workers=2) as executor:
    first_future = executor.submit(reconcile)
    second_future = executor.submit(reconcile)
    first = first_future.result(timeout=30)
    second = second_future.result(timeout=30)

  assert first["state"] == "published"
  assert second["state"] == "published"
  assert _get_delivery(
    store,
    "duplicate-race",
  )["attempt_count"] == 1


def test_duplicate_reconciliation_obeys_cancellation_fence(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  _create_decision(store, "duplicate-fenced")
  asyncio.run(
    store.fence_persistent_grants_for_cancellation(
      "approval-duplicate-fenced",
      expected_tool_call_id="tool-duplicate-fenced",
      expected_user_id="owner",
      expected_request_id="run-duplicate-fenced",
      expected_run_id="run-duplicate-fenced",
      expected_session_id="session-duplicate-fenced",
      expected_channel="tui",
    )
  )

  with pytest.raises(PersistentGrantCancellationFenced):
    store.reconcile_autonomous_approval_delivery_duplicate(
      "approval-duplicate-fenced",
      tool_call_id="tool-duplicate-fenced",
      nonce="nonce-duplicate-fenced",
      approved=True,
    )
  assert (
    _get_delivery(store, "duplicate-fenced")["state"]
    == "pending"
  )


def test_acknowledgment_requires_exact_authority_and_is_terminal(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  _create_decision(store, "ack")
  with store.autonomous_approval_delivery_append_transaction(
    "approval-ack",
    tool_call_id="tool-ack",
    nonce="nonce-ack",
    approved=True,
  ):
    pass
  published = _get_delivery(store, "ack")
  authority = _acknowledgment("ack", published)
  mismatches = {
    "approval_id": "approval-other",
    "task_id": "bg_other",
    "control_run_id": "run-other",
    "session_id": "session-other",
    "channel_id": "b" * 64,
    "tool_call_id": "tool-other",
    "nonce": "nonce-other",
    "approved": False,
    "decided_at_ns": published["decided_at_ns"] + 1,
  }
  for field_name, wrong_value in mismatches.items():
    wrong_authority = {**authority, field_name: wrong_value}
    with pytest.raises((KeyError, RuntimeError, ValueError)):
      asyncio.run(
        store.acknowledge_autonomous_approval_delivery(
          **wrong_authority
        )
      )
    assert _get_delivery(store, "ack")["state"] == "published"

  acknowledged = asyncio.run(
    store.acknowledge_autonomous_approval_delivery(**authority)
  )
  assert acknowledged["state"] == "acknowledged"
  assert acknowledged["acknowledged_at"] is not None
  assert (
    asyncio.run(
      store.acknowledge_autonomous_approval_delivery(**authority)
    )
    == acknowledged
  )

  unchanged = asyncio.run(
    store.record_autonomous_approval_delivery_failure(
      "approval-ack",
      tool_call_id="tool-ack",
      nonce="nonce-ack",
      error="late failure",
    )
  )
  assert unchanged == acknowledged


def test_pending_delivery_failure_stays_pending_and_can_later_publish(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  _create_decision(store, "retry")

  for expected_attempts in (1, 2):
    failed = asyncio.run(
      store.record_autonomous_approval_delivery_failure(
        "approval-retry",
        tool_call_id="tool-retry",
        nonce="nonce-retry",
        error=f"failure {expected_attempts}",
      )
    )

    assert failed["state"] == "pending"
    assert failed["attempt_count"] == expected_attempts
    assert failed["last_error"] == f"failure {expected_attempts}"
    assert failed["last_attempt_ns"] is not None
    assert failed["quarantined_at"] is None
    assert [
      row["approval_id"]
      for row in asyncio.run(
        store.list_pending_autonomous_approval_deliveries()
      )
    ] == ["approval-retry"]

  with store.autonomous_approval_delivery_append_transaction(
    "approval-retry",
    tool_call_id="tool-retry",
    nonce="nonce-retry",
    approved=True,
  ):
    pass

  published = _get_delivery(store, "retry")
  assert published["state"] == "published"
  assert published["attempt_count"] == 3
  assert asyncio.run(
    store.list_pending_autonomous_approval_deliveries()
  ) == []


def test_published_delivery_failure_keeps_its_state(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  _create_decision(store, "published-failure")
  with store.autonomous_approval_delivery_append_transaction(
    "approval-published-failure",
    tool_call_id="tool-published-failure",
    nonce="nonce-published-failure",
    approved=True,
  ):
    pass
  published = _get_delivery(store, "published-failure")

  failed = asyncio.run(
    store.record_autonomous_approval_delivery_failure(
      "approval-published-failure",
      tool_call_id="tool-published-failure",
      nonce="nonce-published-failure",
      error="late failure",
    )
  )

  assert failed["state"] == "published"
  assert failed["quarantined_at"] is None
  assert failed["attempt_count"] == published["attempt_count"] + 1
  assert failed["last_error"] == "late failure"


def test_pending_selector_pages_by_sequence_and_excludes_published(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  for suffix in ("one", "two", "three"):
    _create_decision(store, suffix)

  selected = asyncio.run(
    store.list_pending_autonomous_approval_deliveries()
  )
  assert len(selected) == 3
  first_sequence = selected[0]["delivery_sequence"]
  assert [
    row["delivery_sequence"]
    for row in asyncio.run(
      store.list_pending_autonomous_approval_deliveries(
        after_sequence=first_sequence,
      )
    )
  ] == [row["delivery_sequence"] for row in selected[1:]]
  assert [
    row["delivery_sequence"]
    for row in asyncio.run(
      store.list_pending_autonomous_approval_deliveries(
        through_sequence=first_sequence,
      )
    )
  ] == [first_sequence]
  assert asyncio.run(
    store.list_pending_autonomous_approval_deliveries(
      after_sequence=first_sequence,
      through_sequence=first_sequence,
    )
  ) == []
  with store.autonomous_approval_delivery_append_transaction(
    selected[0]["approval_id"],
    tool_call_id=selected[0]["tool_call_id"],
    nonce=selected[0]["nonce"],
    approved=selected[0]["approved"],
  ):
    pass
  remaining = asyncio.run(
    store.list_pending_autonomous_approval_deliveries()
  )
  assert len(remaining) == 2
  assert selected[0]["approval_id"] not in {
    delivery["approval_id"] for delivery in remaining
  }


def test_outbox_with_retry_columns_keeps_rows_on_open(
  tmp_path: Path,
) -> None:
  database_path = tmp_path / "approvals.sqlite3"
  store = SQLiteApprovalStore(
    database_path,
    audit_emitter=RecordingAuditEmitter(),
  )
  before = _create_decision(store, "legacy")
  with sqlite3.connect(database_path) as conn:
    conn.execute(
      "ALTER TABLE autonomous_approval_delivery_outbox "
      "ADD COLUMN retry_deadline_ns INTEGER NOT NULL DEFAULT 1"
    )
    conn.execute(
      "ALTER TABLE autonomous_approval_delivery_outbox "
      "ADD COLUMN next_attempt_ns INTEGER NOT NULL DEFAULT 1"
    )

  reopened = SQLiteApprovalStore(
    database_path,
    audit_emitter=RecordingAuditEmitter(),
  )

  with sqlite3.connect(database_path) as conn:
    columns = {
      str(row[1])
      for row in conn.execute(
        "PRAGMA table_info(autonomous_approval_delivery_outbox)"
      )
    }
  assert columns == set(
    approval_store_module._AUTONOMOUS_APPROVAL_DELIVERY_OUTBOX_COLUMNS
  )
  assert _get_delivery(reopened, "legacy") == before


def test_main_database_create_requires_parent_fsync(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  real_fsync = os.fsync
  failed = False

  def fail_directory_fsync(fd: int) -> None:
    nonlocal failed
    if not failed and stat.S_ISDIR(os.fstat(fd).st_mode):
      failed = True
      raise OSError("injected parent fsync failure")
    real_fsync(fd)

  database_path = tmp_path / "approvals.sqlite3"
  monkeypatch.setattr(
    approval_store_module.os,
    "fsync",
    fail_directory_fsync,
  )
  with pytest.raises(OSError, match="parent fsync failure"):
    SQLiteApprovalStore(database_path)
  assert not database_path.exists()

  monkeypatch.setattr(approval_store_module.os, "fsync", real_fsync)
  store = SQLiteApprovalStore(database_path)
  assert store.path == database_path


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_sidecar_create_requires_parent_fsync_and_retries_cleanly(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  suffix: str,
) -> None:
  sidecar_path = Path(f"{tmp_path / 'approvals.sqlite3'}{suffix}")
  real_fsync = os.fsync
  failed = False

  def fail_directory_fsync(fd: int) -> None:
    nonlocal failed
    if not failed and stat.S_ISDIR(os.fstat(fd).st_mode):
      failed = True
      raise OSError("injected sidecar parent fsync failure")
    real_fsync(fd)

  monkeypatch.setattr(
    approval_store_module.os,
    "fsync",
    fail_directory_fsync,
  )
  with pytest.raises(OSError, match="sidecar parent fsync failure"):
    approval_store_module._prepare_secure_sqlite_sidecar(
      sidecar_path
    )
  assert not sidecar_path.exists()

  monkeypatch.setattr(approval_store_module.os, "fsync", real_fsync)
  approval_store_module._prepare_secure_sqlite_sidecar(sidecar_path)
  assert stat.S_IMODE(sidecar_path.stat().st_mode) == 0o600


def test_existing_database_acceptance_reestablishes_parent_fsync(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  database_path = tmp_path / "approvals.sqlite3"
  approval_store_module._secure_create_sqlite_file(database_path)
  real_fsync = os.fsync

  def fail_directory_fsync(fd: int) -> None:
    if stat.S_ISDIR(os.fstat(fd).st_mode):
      raise OSError("injected existing parent fsync failure")
    real_fsync(fd)

  monkeypatch.setattr(
    approval_store_module.os,
    "fsync",
    fail_directory_fsync,
  )
  with pytest.raises(
    OSError,
    match="existing parent fsync failure",
  ):
    approval_store_module._require_secure_sqlite_file(
      database_path
    )
