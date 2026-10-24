from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
import threading

import pytest

from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.fork_ledger import (
  FORK_LEDGER_SCHEMA_VERSION,
  ForkLedger,
  ForkLedgerClockRollback,
  ForkLedgerUnavailable,
)

# Written by the v1 ``ForkLedger`` (9bc6620595) at 2026-07-27T12:00Z as
# process ``boot-1``: ``v1-settled`` settled at $0.50 of a $2.00 reservation,
# ``v1-abandoned`` abandoned at $2.00, ``v1-in-flight`` started with $2.00
# reserved, ``v1-refused`` abandoned at its $1.00 reservation after v1 refused
# to settle its $1.50 actual, and one pending receipt for ``v1-settled``.
_V1_FIXTURE = Path(__file__).parent / "fixtures" / "fork_ledger_v1.sqlite3"


def _ns(year: int, month: int, day: int, hour: int = 12) -> int:
  return int(
    datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp()
    * 1_000_000_000
  )


class MutableClock:
  def __init__(self, value: int) -> None:
    self.value = value

  def __call__(self) -> int:
    return self.value


def _ledger(
  tmp_path: Path,
  *,
  boot: str = "boot-1",
  clock: MutableClock | None = None,
) -> ForkLedger:
  return ForkLedger(
    tmp_path / "fork-ledger.sqlite3",
    process_instance_id=boot,
    clock_ns=clock or MutableClock(_ns(2026, 7, 27)),
  )


def _empty_journal(tmp_path: Path) -> AgentSessionLog:
  return AgentSessionLog(tmp_path / "empty-fork-journal.jsonl")


def _receipt(ledger: ForkLedger, fork_id: str = "fork-1") -> None:
  assert ledger.write_receipt(
    fork_id=fork_id,
    session_id="session-1",
    owner="owner-1",
    receipt_text=f"{fork_id} completed.",
  )


def _reserve(
  ledger: ForkLedger,
  fork_id: str,
  *,
  owner: str = "owner-1",
  per_fork: str = "2.00",
):
  return ledger.reserve_admission(
    fork_id=fork_id,
    owner=owner,
    max_reserved_usd=per_fork,
  )


def _live_instance_worker(path: str, connection) -> None:
  ledger = ForkLedger(
    path,
    process_instance_id="gateway-live",
    clock_ns=lambda: _ns(2026, 7, 27),
  )
  _reserve(ledger, "live-reserved")
  _reserve(ledger, "live-started")
  assert ledger.mark_admission_started(fork_id="live-started")
  _receipt(ledger, "live-receipt")
  claim = ledger.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="live-turn",
  )[0]
  connection.send(claim.claim_token)
  assert connection.recv() == "ack"
  connection.send(ledger.ack_receipt(
    fork_id=claim.fork_id,
    claim_token=claim.claim_token,
  ))
  connection.close()


def _hard_death_worker(path: str, connection) -> None:
  ledger = ForkLedger(
    path,
    process_instance_id="killed-instance",
    clock_ns=lambda: _ns(2026, 7, 27),
  )
  _reserve(ledger, "killed-admission")
  assert ledger.mark_admission_started(fork_id="killed-admission")
  _receipt(ledger, "killed-receipt")
  ledger.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="killed-turn",
  )
  connection.send("ready")
  connection.recv()


def _process_reserve_worker(
  path: str,
  process_number: int,
  barrier,
  connection,
) -> None:
  ledger = ForkLedger(
    path,
    process_instance_id=f"quota-process-{process_number}",
    clock_ns=lambda: _ns(2026, 7, 27),
  )
  barrier.wait()
  try:
    reservation = _reserve(ledger, f"process-fork-{process_number}")
  except BaseException as exc:
    result: object = f"error:{type(exc).__name__}:{exc}"
  else:
    result = reservation.day_invocations
  connection.send(result)
  connection.close()


def test_receipt_pending_claimed_acked_and_stale_token_cas(
  tmp_path: Path,
) -> None:
  ledger = _ledger(tmp_path)
  _receipt(ledger)

  claims = ledger.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-1",
  )

  assert len(claims) == 1
  assert claims[0].fork_id == "fork-1"
  assert not ledger.ack_receipt(
    fork_id="fork-1",
    claim_token="stale-token",
  )
  assert ledger.ack_receipt(
    fork_id="fork-1",
    claim_token=claims[0].claim_token,
  )
  assert ledger.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-2",
  ) == ()


def test_failed_notification_reverts_and_redelivers_once(
  tmp_path: Path,
) -> None:
  ledger = _ledger(tmp_path)
  _receipt(ledger)
  first = ledger.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-1",
  )[0]

  assert ledger.revert_receipt_claim(
    fork_id=first.fork_id,
    claim_token=first.claim_token,
  )
  second = ledger.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-2",
  )
  same_turn_retry = ledger.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-2",
  )

  assert [claim.fork_id for claim in second] == ["fork-1"]
  assert same_turn_retry == ()
  assert second[0].claim_token != first.claim_token


def test_two_concurrent_receipt_claims_have_one_winner(
  tmp_path: Path,
) -> None:
  ledger = _ledger(tmp_path)
  _receipt(ledger)
  barrier = threading.Barrier(2)

  def claim(turn: str):
    barrier.wait()
    return ledger.claim_pending_receipts(
      session_id="session-1",
      owner="owner-1",
      claiming_turn_id=turn,
    )

  with ThreadPoolExecutor(max_workers=2) as pool:
    results = tuple(pool.map(claim, ("turn-a", "turn-b")))

  assert sorted(len(result) for result in results) == [0, 1]
  assert sum(
    claim.fork_id == "fork-1"
    for result in results
    for claim in result
  ) == 1


def test_explicit_reconciliation_recovers_only_named_dead_instance(
  tmp_path: Path,
) -> None:
  clock = MutableClock(_ns(2026, 7, 27))
  first_boot = _ledger(tmp_path, boot="boot-old", clock=clock)
  _receipt(first_boot, "prior")
  prior = first_boot.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-old",
  )[0]

  current_boot = _ledger(tmp_path, boot="boot-new", clock=clock)
  assert current_boot.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-before-recovery",
  ) == ()
  assert current_boot.reconcile_startup(
    dead_process_instance_id="boot-old",
    fork_journal=lambda _fork_id: _empty_journal(tmp_path),
  ) == (1, 0)
  current = current_boot.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-new",
  )
  assert {claim.fork_id for claim in current} == {"prior"}
  assert current_boot.reconcile_startup(
    dead_process_instance_id="boot-old",
    fork_journal=lambda _fork_id: _empty_journal(tmp_path),
  ) == (0, 0)
  assert current_boot.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-new-retry",
  ) == ()
  assert prior.process_instance_id == "boot-old"


def test_second_process_opener_preserves_live_instance_state(
  tmp_path: Path,
) -> None:
  context = multiprocessing.get_context("spawn")
  parent, child = context.Pipe()
  path = str(tmp_path / "fork-ledger.sqlite3")
  process = context.Process(
    target=_live_instance_worker,
    args=(path, child),
  )
  process.start()
  child.close()
  assert parent.poll(10)
  claim_token = parent.recv()

  second = ForkLedger(
    path,
    process_instance_id="autonomous-live",
    clock_ns=lambda: _ns(2026, 7, 27),
  )
  assert second.reconcile_startup(
    live_process_instance_ids={"gateway-live", "autonomous-live"},
    fork_journal=lambda _fork_id: _empty_journal(tmp_path),
  ) == (0, 0)
  reserved = second.get_admission("live-reserved")
  started = second.get_admission("live-started")
  assert reserved is not None
  assert reserved.state == "reserved"
  assert reserved.settled_usd is None
  assert started is not None
  assert started.state == "started"
  assert started.settled_usd is None
  assert second.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="other-turn",
  ) == ()

  parent.send("ack")
  assert parent.poll(10)
  assert parent.recv() is True
  process.join(10)
  assert process.exitcode == 0
  assert claim_token


def test_receipt_fork_id_dedup_is_idempotent_but_rejects_rebinding(
  tmp_path: Path,
) -> None:
  ledger = _ledger(tmp_path)
  _receipt(ledger)

  assert not ledger.write_receipt(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    receipt_text="fork-1 completed.",
  )
  with pytest.raises(Exception, match="different content"):
    ledger.write_receipt(
      fork_id="fork-1",
      session_id="session-2",
      owner="owner-1",
      receipt_text="fork-1 completed.",
    )


def test_concurrent_owner_admissions_count_each_exactly_once(
  tmp_path: Path,
) -> None:
  ledger = _ledger(tmp_path)
  barrier = threading.Barrier(2)

  def reserve(fork_id: str) -> int:
    barrier.wait()
    return _reserve(ledger, fork_id).day_invocations

  with ThreadPoolExecutor(max_workers=2) as pool:
    results = tuple(pool.map(reserve, ("fork-a", "fork-b")))

  assert sorted(results) == [1, 2]
  assert ledger.get_admission("fork-a") is not None
  assert ledger.get_admission("fork-b") is not None


def test_day_totals_are_measured_past_any_former_quota_or_budget(
  tmp_path: Path,
) -> None:
  ledger = _ledger(tmp_path)
  reservations = [
    _reserve(ledger, f"fork-{index}", per_fork="2.00")
    for index in range(13)
  ]

  assert [item.day_invocations for item in reservations] == list(range(1, 14))
  assert reservations[-1].day_charged_microusd == 13 * 2_000_000
  assert reservations[-1].record.state == "reserved"
  assert ledger.get_admission("fork-12") is not None

  assert ledger.mark_admission_started(fork_id="fork-0")
  assert ledger.settle_admission(fork_id="fork-0", actual_cost_usd="0.50")
  other = _reserve(ledger, "after-settle", per_fork="2.00")
  assert other.day_charged_microusd == 500_000 + 13 * 2_000_000

  isolated = _reserve(ledger, "other-owner", owner="owner-2")
  assert isolated.day_invocations == 1


def test_settle_records_actual_cost(tmp_path: Path) -> None:
  ledger = _ledger(tmp_path)
  _reserve(ledger, "fork-1")

  assert ledger.mark_admission_started(fork_id="fork-1")
  assert ledger.settle_admission(
    fork_id="fork-1",
    actual_cost_usd="1.234567",
  )

  record = ledger.get_admission("fork-1")
  assert record is not None
  assert record.state == "settled"
  assert record.max_reserved_usd == Decimal("2.000000")
  assert record.settled_usd == Decimal("1.234567")


def test_settle_records_actual_above_its_reservation(tmp_path: Path) -> None:
  ledger = _ledger(tmp_path)
  _reserve(ledger, "fork-1", per_fork="2.00")

  assert ledger.mark_admission_started(fork_id="fork-1")
  assert ledger.settle_admission(fork_id="fork-1", actual_cost_usd="7.654321")

  record = ledger.get_admission("fork-1")
  assert record is not None
  assert record.state == "settled"
  assert record.max_reserved_usd == Decimal("2.000000")
  assert record.settled_usd == Decimal("7.654321")
  assert _reserve(ledger, "fork-2").day_charged_microusd == (
    7_654_321 + 2_000_000
  )


def _v1_ledger_file(tmp_path: Path) -> Path:
  path = tmp_path / "fork-ledger.sqlite3"
  path.write_bytes(_V1_FIXTURE.read_bytes())
  path.chmod(0o600)
  return path


def test_v1_ledger_migrates_to_v2_and_settles_above_reservation(
  tmp_path: Path,
) -> None:
  path = _v1_ledger_file(tmp_path)
  with sqlite3.connect(path) as connection:
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1

  ledger = _ledger(tmp_path, clock=MutableClock(_ns(2026, 7, 27, 13)))

  with sqlite3.connect(path) as connection:
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    assert connection.execute(
      "SELECT * FROM fork_ledger_metadata"
    ).fetchall() == [(1, FORK_LEDGER_SCHEMA_VERSION, "UTC", "micro_usd")]
    assert {
      name for (name,) in connection.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE '%_v1'"
      )
    } == set()
  preserved = {
    fork_id: ledger.get_admission(fork_id)
    for fork_id in ("v1-settled", "v1-abandoned", "v1-in-flight", "v1-refused")
  }
  assert {
    fork_id: (
      record.state,
      record.max_reserved_usd,
      record.settled_usd,
      record.process_instance_id,
    )
    for fork_id, record in preserved.items()
    if record is not None
  } == {
    "v1-settled": ("settled", Decimal("2.000000"), Decimal("0.500000"), "boot-1"),
    "v1-abandoned": (
      "abandoned", Decimal("2.000000"), Decimal("2.000000"), "boot-1",
    ),
    "v1-in-flight": ("started", Decimal("2.000000"), None, "boot-1"),
    "v1-refused": (
      "abandoned", Decimal("1.000000"), Decimal("1.000000"), "boot-1",
    ),
  }

  assert ledger.settle_admission(
    fork_id="v1-in-flight",
    actual_cost_usd="3.250000",
  )
  in_flight = ledger.get_admission("v1-in-flight")
  assert in_flight is not None
  assert in_flight.state == "settled"
  assert in_flight.settled_usd == Decimal("3.250000")

  reservation = _reserve(ledger, "v2-fork", per_fork="1.00")
  assert reservation.day_invocations == 5
  assert reservation.day_charged_microusd == (
    500_000 + 2_000_000 + 3_250_000 + 1_000_000 + 1_000_000
  )
  assert ledger.mark_admission_started(fork_id="v2-fork")
  assert ledger.settle_admission(fork_id="v2-fork", actual_cost_usd="4.00")
  settled = ledger.get_admission("v2-fork")
  assert settled is not None
  assert settled.settled_usd == Decimal("4.000000")

  claims = ledger.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-1",
  )
  assert [claim.receipt_text for claim in claims] == ["v1-settled completed."]


def test_unrecognized_v1_schema_is_refused_unchanged(tmp_path: Path) -> None:
  path = _v1_ledger_file(tmp_path)
  with sqlite3.connect(path) as connection:
    connection.execute("CREATE TABLE stray (value INTEGER)")

  with pytest.raises(ForkLedgerUnavailable, match="objects are incompatible"):
    _ledger(tmp_path, clock=MutableClock(_ns(2026, 7, 27, 13)))

  with sqlite3.connect(path) as connection:
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert connection.execute(
      "SELECT schema_version FROM fork_ledger_metadata"
    ).fetchone()[0] == 1


def test_fresh_boot_recovers_unknown_hard_death_and_redelivers_once(
  tmp_path: Path,
) -> None:
  context = multiprocessing.get_context("spawn")
  parent, child = context.Pipe()
  path = str(tmp_path / "fork-ledger.sqlite3")
  process = context.Process(
    target=_hard_death_worker,
    args=(path, child),
  )
  process.start()
  child.close()
  assert parent.poll(10)
  assert parent.recv() == "ready"
  pid = process.pid
  assert pid is not None
  os.kill(pid, signal.SIGKILL)
  process.join(10)
  assert process.exitcode == -signal.SIGKILL

  recovery = ForkLedger(
    path,
    process_instance_id="recovery-instance",
    clock_ns=lambda: _ns(2026, 7, 27),
  )
  before = recovery.get_admission("killed-admission")
  assert before is not None
  assert before.state == "started"
  assert recovery.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="before-recovery",
  ) == ()

  assert recovery.reconcile_startup(
    live_process_instance_ids={"recovery-instance"},
    fork_journal=lambda _fork_id: _empty_journal(tmp_path),
  ) == (1, 1)
  after = recovery.get_admission("killed-admission")
  assert after is not None
  assert after.state == "abandoned"
  assert after.settled_usd == after.max_reserved_usd
  claims = recovery.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="redelivery-turn",
  )
  assert [claim.fork_id for claim in claims] == ["killed-receipt"]
  assert recovery.ack_receipt(
    fork_id=claims[0].fork_id,
    claim_token=claims[0].claim_token,
  )
  assert recovery.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="no-second-redelivery",
  ) == ()


def test_cross_process_admissions_count_each_exactly_once(
  tmp_path: Path,
) -> None:
  context = multiprocessing.get_context("spawn")
  barrier = context.Barrier(12)
  path = str(tmp_path / "fork-ledger.sqlite3")
  _ledger(tmp_path, boot="schema-initializer")
  processes = []
  parents = []
  for process_number in range(12):
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
      target=_process_reserve_worker,
      args=(path, process_number, barrier, child),
    )
    processes.append(process)
    parents.append(parent)
    process.start()
    child.close()

  results = []
  for parent in parents:
    assert parent.poll(15)
    results.append(parent.recv())
  for process in processes:
    process.join(15)
    assert process.exitcode == 0

  # A worker that could not reserve sends its exception as `error:<type>:<message>`
  # (`_process_reserve_worker`); name it here instead of letting `sorted` raise a
  # TypeError that hides which worker failed and why.
  worker_errors = [result for result in results if isinstance(result, str)]
  assert not worker_errors, f"reserve workers failed: {worker_errors}"
  assert sorted(results) == list(range(1, 13))




def test_utc_day_rollover_resets_day_totals(tmp_path: Path) -> None:
  clock = MutableClock(_ns(2026, 7, 27, 23))
  ledger = _ledger(tmp_path, clock=clock)
  first = _reserve(ledger, "day-one")
  same_day = _reserve(ledger, "same-day")

  clock.value = _ns(2026, 7, 28, 0)
  second = _reserve(ledger, "day-two")

  assert first.record.date == "2026-07-27"
  assert same_day.day_invocations == 2
  assert second.record.date == "2026-07-28"
  assert second.day_invocations == 1


def test_clock_rollback_is_detected_and_writes_nothing(
  tmp_path: Path,
) -> None:
  clock = MutableClock(_ns(2026, 7, 27, 12))
  ledger = _ledger(tmp_path, clock=clock)
  _reserve(ledger, "first")
  clock.value -= 1

  with pytest.raises(ForkLedgerClockRollback):
    _reserve(ledger, "rolled-back")

  assert ledger.get_admission("rolled-back") is None
  with sqlite3.connect(tmp_path / "fork-ledger.sqlite3") as connection:
    assert connection.execute(
      "SELECT COUNT(*) FROM fork_admissions"
    ).fetchone()[0] == 1
