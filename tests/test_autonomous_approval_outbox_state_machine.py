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
      batch_limit=1,
      fallback_retry_base_seconds=0.001,
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
            decision_reason="run_cancelled",
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
    barrier.wait(timeout=1)
    return store.reconcile_autonomous_approval_delivery_duplicate(
      "approval-duplicate-race",
      tool_call_id="tool-duplicate-race",
      nonce="nonce-duplicate-race",
      approved=True,
    )

  with ThreadPoolExecutor(max_workers=2) as executor:
    first_future = executor.submit(reconcile)
    second_future = executor.submit(reconcile)
    first = first_future.result(timeout=2)
    second = second_future.result(timeout=2)

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


def test_delivery_failures_back_off_then_quarantine(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  _create_decision(store, "retry")

  prior_next_attempt_ns = 0
  for attempt in range(1, 6):
    delivery = asyncio.run(
      store.record_autonomous_approval_delivery_failure(
        "approval-retry",
        tool_call_id="tool-retry",
        nonce="nonce-retry",
        error=f"failure {attempt}",
      )
    )
    assert delivery["attempt_count"] == attempt
    assert delivery["last_attempt_ns"] is not None
    assert delivery["next_attempt_ns"] >= delivery["last_attempt_ns"]
    if attempt < 5:
      assert delivery["next_attempt_ns"] >= prior_next_attempt_ns
    prior_next_attempt_ns = delivery["next_attempt_ns"]
    assert delivery["state"] == (
      "quarantined" if attempt == 5 else "pending"
    )

  assert delivery["quarantined_at"] is not None
  assert asyncio.run(
    store.list_pending_autonomous_approval_deliveries()
  ) == []
  with pytest.raises(RuntimeError, match="not pending"):
    with store.autonomous_approval_delivery_append_transaction(
      "approval-retry",
      tool_call_id="tool-retry",
      nonce="nonce-retry",
      approved=True,
    ):
      pass


def test_retry_deadline_quarantines_first_late_failure(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  ready = _create_decision(store, "deadline")
  monkeypatch.setattr(
    approval_store_module.time,
    "time_ns",
    lambda: ready["retry_deadline_ns"],
  )

  delivery = asyncio.run(
    store.record_autonomous_approval_delivery_failure(
      "approval-deadline",
      tool_call_id="tool-deadline",
      nonce="nonce-deadline",
      error="late first failure",
    )
  )

  assert delivery["attempt_count"] == 1
  assert delivery["state"] == "quarantined"


def test_pending_selector_is_bounded_and_excludes_published(
  tmp_path: Path,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3",
    audit_emitter=RecordingAuditEmitter(),
  )
  for suffix in ("one", "two", "three"):
    _create_decision(store, suffix)

  selected = asyncio.run(
    store.list_pending_autonomous_approval_deliveries(limit=2)
  )
  assert len(selected) == 2
  with store.autonomous_approval_delivery_append_transaction(
    selected[0]["approval_id"],
    tool_call_id=selected[0]["tool_call_id"],
    nonce=selected[0]["nonce"],
    approved=selected[0]["approved"],
  ):
    pass
  remaining = asyncio.run(
    store.list_pending_autonomous_approval_deliveries(limit=3)
  )
  assert len(remaining) == 2
  assert selected[0]["approval_id"] not in {
    delivery["approval_id"] for delivery in remaining
  }
  for invalid_limit in (0, 257, True):
    with pytest.raises(ValueError, match="1..256"):
      asyncio.run(
        store.list_pending_autonomous_approval_deliveries(
          limit=invalid_limit
        )
      )


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
