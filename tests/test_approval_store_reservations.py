from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal, TypedDict

import pytest

from agent_gateway.approval_policy import ApprovalRequest, ApprovalVote
from agent_gateway.approval_remote import (
  RESERVATION_BLOCKED_WRITERS,
  ApprovalRemoteReservationActive,
  AutonomousPublicationReservation,
)
from agent_gateway.approval_store import SQLiteApprovalStore


# Recovered from 22616ffb9a6816e95a2131c119f07b78aafb4084, retaining
# direct-store coverage rather than relying on the edge's earlier pre-check.
class RecordingAuditEmitter:
  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []

  async def emit_audit_for_lifecycle_event(self, **kwargs: Any) -> None:
    self.calls.append(dict(kwargs))


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
    reason="reservation test",
    blast_radius_summary="state_write:memory_write",
    state="pending_user",
    requested_at=datetime(2026, 1, 1, tzinfo=UTC),
    approval_constraint="standard",
  )


def _get_delivery(store: SQLiteApprovalStore, suffix: str) -> dict[str, Any]:
  delivery = asyncio.run(store.get_autonomous_approval_delivery(
    f"approval-{suffix}", tool_call_id=f"tool-{suffix}", nonce=f"nonce-{suffix}",
  ))
  assert delivery is not None
  return delivery


def _create_decision(store: SQLiteApprovalStore, suffix: str) -> dict[str, Any]:
  request = _request(suffix)
  asyncio.run(store.create(request))
  asyncio.run(store.record_vote_with_autonomous_delivery(
    request.approval_id,
    ApprovalVote(
      vote_id=f"vote-{suffix}",
      approval_id=request.approval_id,
      decider_id="owner",
      decider_role="owner",
      decision="approved",
      decision_reason="reviewed",
      decided_at=datetime(2026, 1, 2, tzinfo=UTC),
    ),
    delivery={
      "task_id": f"bg_{suffix}",
      "control_run_id": request.run_id,
      "session_id": request.session_id,
      "channel_id": "a" * 64,
      "tool_call_id": request.tool_call_id,
      "nonce": f"nonce-{suffix}",
    },
  ))
  return _get_delivery(store, suffix)


def _acknowledgment(delivery: dict[str, Any]) -> dict[str, Any]:
  # Use the stored approval-specific transport identity, not ambient runtime
  # metadata or a separately maintained copy of the acknowledgment fields.
  return {
    name: delivery[name]
    for name in SQLiteApprovalStore.acknowledge_autonomous_approval_delivery.__annotations__
    if name != "return"
  }


def _reserve(
  store: SQLiteApprovalStore,
  suffix: str,
  *,
  tool_call_id: str | None = None,
  nonce: str | None = None,
  generation: int = 1,
  state: Literal["active", "committed"] = "active",
) -> AutonomousPublicationReservation:
  reservation = AutonomousPublicationReservation(
    decision_id=f"approval-{suffix}",
    tool_call_id=f"tool-{suffix}" if tool_call_id is None else tool_call_id,
    nonce=f"nonce-{suffix}" if nonce is None else nonce,
    approved=True,
    placement_generation=generation,
  )
  # The edge owns these rows. Seed its actual structured record, not a
  # store-side reserve/commit/abort implementation or a mock of writer behavior.
  with store._connection() as conn:
    conn.execute(
      """
      INSERT INTO autonomous_publication_reservations (
        decision_id, tool_call_id, nonce, approved, placement_generation, state
      ) VALUES (?, ?, ?, ?, ?, ?)
      """,
      (
        reservation.decision_id, reservation.tool_call_id, reservation.nonce,
        int(reservation.approved), reservation.placement_generation, state,
      ),
    )
  return reservation


class _CancellationIdentity(TypedDict):
  expected_tool_call_id: str
  expected_user_id: str
  expected_request_id: str
  expected_run_id: str
  expected_session_id: str
  expected_channel: str


def _cancel_identity(suffix: str) -> _CancellationIdentity:
  return {
    "expected_tool_call_id": f"tool-{suffix}",
    "expected_user_id": "owner",
    "expected_request_id": f"run-{suffix}",
    "expected_run_id": f"run-{suffix}",
    "expected_session_id": f"session-{suffix}",
    "expected_channel": "tui",
  }


def _write(
  store: SQLiteApprovalStore,
  writer: str,
  suffix: str,
  delivery: dict[str, Any],
) -> Any:
  approval_id = f"approval-{suffix}"
  if writer == "fence_persistent_grants_for_cancellation":
    return asyncio.run(store.fence_persistent_grants_for_cancellation(
      approval_id, **_cancel_identity(suffix),
    ))
  if writer == "terminalize_pending_for_cancellation":
    return asyncio.run(store.terminalize_pending_for_cancellation(
      approval_id, **_cancel_identity(suffix), decision_reason="cancelled",
    ))
  if writer == "acknowledge_autonomous_approval_delivery":
    return asyncio.run(store.acknowledge_autonomous_approval_delivery(
      **_acknowledgment(delivery),
    ))
  return asyncio.run(getattr(store, writer)(
    approval_id, tool_call_id=f"tool-{suffix}", nonce=f"nonce-{suffix}",
    error="delivery failed",
  ))


def _prepare_writer(
  store: SQLiteApprovalStore, writer: str, suffix: str,
) -> dict[str, Any]:
  delivery = _create_decision(store, suffix)
  if writer == "acknowledge_autonomous_approval_delivery":
    with store.autonomous_approval_delivery_append_transaction(
      f"approval-{suffix}", tool_call_id=f"tool-{suffix}",
      nonce=f"nonce-{suffix}", approved=True,
    ):
      pass
    delivery = _get_delivery(store, suffix)
  return delivery


@pytest.mark.parametrize("writer", RESERVATION_BLOCKED_WRITERS)
@pytest.mark.parametrize("generation", [1, 2])
def test_active_reservation_blocks_conflicting_writer_without_mutation(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  writer: str,
  generation: int,
) -> None:
  emitter = RecordingAuditEmitter()
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3", audit_emitter=emitter)
  before = _prepare_writer(store, writer, "reserved")
  request_before = asyncio.run(store.get("approval-reserved"))
  _reserve(store, "reserved", generation=generation)
  audits_before = list(emitter.calls)
  connection = store._connection
  fence_transactions: list[bool] = []

  @contextmanager
  def traced_connection() -> Iterator[Any]:
    with connection() as conn:
      def trace(statement: str) -> None:
        if "SELECT 1 FROM autonomous_publication_reservations" in statement:
          fence_transactions.append(conn.in_transaction)
      conn.set_trace_callback(trace)
      yield conn

  monkeypatch.setattr(store, "_connection", traced_connection)
  with pytest.raises(ApprovalRemoteReservationActive) as caught:
    _write(store, writer, "reserved", before)

  assert caught.value.retryable is True
  assert caught.value.status == 429
  assert fence_transactions == [True]
  assert _get_delivery(store, "reserved") == before
  assert asyncio.run(store.get("approval-reserved")) == request_before
  assert emitter.calls == audits_before
  with store._connection() as conn:
    assert conn.execute(
      "SELECT COUNT(*) FROM persistent_grant_cancellation_fences"
    ).fetchone()[0] == 0
    assert conn.execute(
      "SELECT COUNT(*) FROM autonomous_publication_reservations WHERE state = 'active'"
    ).fetchone()[0] == 1


@pytest.mark.parametrize("writer", RESERVATION_BLOCKED_WRITERS)
def test_committed_reservation_does_not_block_writer(
  tmp_path: Path,
  writer: str,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3", audit_emitter=RecordingAuditEmitter(),
  )
  delivery = _prepare_writer(store, writer, "committed")
  _reserve(store, "committed", state="committed")

  result = _write(store, writer, "committed", delivery)

  if writer == "acknowledge_autonomous_approval_delivery":
    assert result["state"] == "acknowledged"
  elif writer == "quarantine_autonomous_approval_delivery":
    assert result["state"] == "quarantined"
  elif writer == "record_autonomous_approval_delivery_failure":
    assert result["attempt_count"] == delivery["attempt_count"] + 1
  elif writer == "fence_persistent_grants_for_cancellation":
    assert result[1] is True
  else:
    assert result[1:] == (False, True)


_DELIVERY_WRITERS = tuple(
  writer for writer in RESERVATION_BLOCKED_WRITERS
  if "expected_tool_call_id" not in getattr(SQLiteApprovalStore, writer).__annotations__
)
_CANCELLATION_WRITERS = tuple(
  writer for writer in RESERVATION_BLOCKED_WRITERS if writer not in _DELIVERY_WRITERS
)


@pytest.mark.parametrize("writer", _DELIVERY_WRITERS)
@pytest.mark.parametrize(
  ("reserved_tool_call_id", "reserved_nonce"),
  [("another-call", None), (None, "another-nonce")],
)
def test_delivery_writer_only_conflicts_with_exact_reserved_call(
  tmp_path: Path,
  writer: str,
  reserved_tool_call_id: str | None,
  reserved_nonce: str | None,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3", audit_emitter=RecordingAuditEmitter(),
  )
  delivery = _prepare_writer(store, writer, "exact")
  _reserve(
    store, "exact", tool_call_id=reserved_tool_call_id, nonce=reserved_nonce,
  )

  _write(store, writer, "exact", delivery)


@pytest.mark.parametrize("writer", _CANCELLATION_WRITERS)
def test_cancellation_conflicts_with_every_reserved_call_for_decision(
  tmp_path: Path,
  writer: str,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3", audit_emitter=RecordingAuditEmitter(),
  )
  delivery = _create_decision(store, "cancel")
  _reserve(store, "cancel", tool_call_id="another-call", nonce="another-nonce")

  with pytest.raises(ApprovalRemoteReservationActive):
    _write(store, writer, "cancel", delivery)


@pytest.mark.parametrize("writer", RESERVATION_BLOCKED_WRITERS)
def test_reservation_does_not_block_another_decision(
  tmp_path: Path,
  writer: str,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3", audit_emitter=RecordingAuditEmitter(),
  )
  _create_decision(store, "reserved")
  _reserve(store, "reserved")
  delivery = _prepare_writer(store, writer, "other")

  _write(store, writer, "other", delivery)


@pytest.mark.parametrize("writer", RESERVATION_BLOCKED_WRITERS)
def test_committed_generation_does_not_hide_another_active_generation(
  tmp_path: Path,
  writer: str,
) -> None:
  store = SQLiteApprovalStore(
    tmp_path / "approvals.sqlite3", audit_emitter=RecordingAuditEmitter(),
  )
  delivery = _prepare_writer(store, writer, "generations")
  _reserve(store, "generations", generation=1, state="committed")
  reservation = _reserve(store, "generations", generation=2)

  with pytest.raises(ApprovalRemoteReservationActive):
    _write(store, writer, "generations", delivery)
  assert _get_delivery(store, "generations") == delivery
  with store._connection() as conn:
    conn.execute(
      """DELETE FROM autonomous_publication_reservations
      WHERE decision_id = ? AND tool_call_id = ? AND nonce = ?
        AND placement_generation = ?""",
      (
        reservation.decision_id, reservation.tool_call_id,
        reservation.nonce, reservation.placement_generation,
      ),
    )
  _write(store, writer, "generations", delivery)


def test_reservation_release_restores_pending_cancellation(tmp_path: Path) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  asyncio.run(store.create(_request("pending")))
  reservation = _reserve(store, "pending")

  with pytest.raises(ApprovalRemoteReservationActive):
    asyncio.run(store.terminalize_pending_for_cancellation(
      "approval-pending", **_cancel_identity("pending"), decision_reason="cancelled",
    ))
  pending = asyncio.run(store.get("approval-pending"))
  assert pending is not None
  assert pending.state == "pending_user"
  with store._connection() as conn:
    conn.execute(
      """DELETE FROM autonomous_publication_reservations
      WHERE decision_id = ? AND tool_call_id = ? AND nonce = ?
        AND placement_generation = ?""",
      (
        reservation.decision_id, reservation.tool_call_id,
        reservation.nonce, reservation.placement_generation,
      ),
    )

  request, transitioned, matched = asyncio.run(store.terminalize_pending_for_cancellation(
    "approval-pending", **_cancel_identity("pending"), decision_reason="cancelled",
  ))
  assert (request.state, transitioned, matched) == ("denied", True, True)


def test_populated_database_gains_reservation_table_without_changing_decision(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  store = SQLiteApprovalStore(path, audit_emitter=RecordingAuditEmitter())
  delivery_before = _create_decision(store, "legacy")
  request_before = asyncio.run(store.get("approval-legacy"))
  with store._connection() as conn:
    conn.execute("DROP TABLE autonomous_publication_reservations")
    votes_before = conn.execute("SELECT * FROM approval_votes").fetchall()

  reopened = SQLiteApprovalStore(path, audit_emitter=RecordingAuditEmitter())

  assert _get_delivery(reopened, "legacy") == delivery_before
  assert asyncio.run(reopened.get("approval-legacy")) == request_before
  with reopened._connection() as conn:
    assert [tuple(row) for row in conn.execute("SELECT * FROM approval_votes")] == [
      tuple(row) for row in votes_before
    ]
    assert conn.execute(
      "SELECT COUNT(*) FROM autonomous_publication_reservations"
    ).fetchone()[0] == 0
  _reserve(reopened, "legacy")
  with pytest.raises(ApprovalRemoteReservationActive):
    _write(reopened, "record_autonomous_approval_delivery_failure", "legacy", delivery_before)


def test_migration_count_and_checkpoint_leave_ledger_ready_for_unchanged_move(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  store = SQLiteApprovalStore(path, audit_emitter=RecordingAuditEmitter())
  asyncio.run(store.create(_request("pending")))
  delivery = _create_decision(store, "decided")
  pending = asyncio.run(store.get("approval-pending"))
  assert store.pending_approval_count() == 1

  with store._connection() as conn:
    conn.execute(
      "INSERT INTO persistent_grant_cancellation_fences VALUES (?, ?)",
      ("approval-pending", "2026-10-29T00:00:00+00:00"),
    )
    conn.commit()
    assert Path(f"{path}-wal").stat().st_size > 0
    store.checkpoint_for_migration()
    assert Path(f"{path}-wal").stat().st_size == 0

  moved_path = path.rename(tmp_path / "moved.sqlite3")
  moved = SQLiteApprovalStore(moved_path, audit_emitter=RecordingAuditEmitter())
  assert moved.pending_approval_count() == 1
  assert asyncio.run(moved.get("approval-pending")) == pending
  assert _get_delivery(moved, "decided") == delivery
