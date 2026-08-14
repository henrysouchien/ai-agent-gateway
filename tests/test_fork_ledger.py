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

from agent_gateway.fork_ledger import (
  ForkAdmissionBudgetExceeded,
  ForkAdmissionQuotaExceeded,
  ForkLedger,
  ForkLedgerClockRollback,
)


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
  daily: str = "10.00",
  quota: int = 12,
):
  return ledger.reserve_admission(
    fork_id=fork_id,
    owner=owner,
    max_reserved_usd=per_fork,
    daily_budget_usd=daily,
    daily_invocation_quota=quota,
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
    _reserve(ledger, f"process-fork-{process_number}", quota=5)
  except ForkAdmissionQuotaExceeded:
    result = "refused"
  except BaseException as exc:
    result = f"error:{type(exc).__name__}:{exc}"
  else:
    result = "admitted"
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
  ) == (1, 0)
  current = current_boot.claim_pending_receipts(
    session_id="session-1",
    owner="owner-1",
    claiming_turn_id="turn-new",
  )
  assert {claim.fork_id for claim in current} == {"prior"}
  assert current_boot.reconcile_startup(
    dead_process_instance_id="boot-old",
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


def test_atomic_owner_admission_allows_one_concurrent_caller(
  tmp_path: Path,
) -> None:
  ledger = _ledger(tmp_path)
  barrier = threading.Barrier(2)

  def reserve(fork_id: str) -> str:
    barrier.wait()
    try:
      _reserve(ledger, fork_id, quota=1)
    except ForkAdmissionQuotaExceeded:
      return "quota"
    return "reserved"

  with ThreadPoolExecutor(max_workers=2) as pool:
    results = tuple(pool.map(reserve, ("fork-a", "fork-b")))

  assert sorted(results) == ["quota", "reserved"]
  records = [
    ledger.get_admission("fork-a"),
    ledger.get_admission("fork-b"),
  ]
  assert sum(record is not None for record in records) == 1


def test_quota_and_budget_exhaustion_refuse_without_open_row(
  tmp_path: Path,
) -> None:
  ledger = _ledger(tmp_path)
  _reserve(ledger, "quota-first", quota=1)
  with pytest.raises(ForkAdmissionQuotaExceeded):
    _reserve(ledger, "quota-refused", quota=1)
  assert ledger.get_admission("quota-refused") is None

  _reserve(
    ledger,
    "budget-first",
    owner="owner-budget",
    per_fork="2.00",
    daily="3.00",
  )
  with pytest.raises(ForkAdmissionBudgetExceeded):
    _reserve(
      ledger,
      "budget-refused",
      owner="owner-budget",
      per_fork="2.00",
      daily="3.00",
    )
  assert ledger.get_admission("budget-refused") is None


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
  os.kill(process.pid, signal.SIGKILL)
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


def test_atomic_owner_admission_across_processes_admits_exactly_quota(
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

  assert not [result for result in results if result.startswith("error:")]
  assert results.count("admitted") == 5
  assert results.count("refused") == 7




def test_utc_day_rollover_resets_availability(tmp_path: Path) -> None:
  clock = MutableClock(_ns(2026, 7, 27, 23))
  ledger = _ledger(tmp_path, clock=clock)
  first = _reserve(ledger, "day-one", quota=1)
  with pytest.raises(ForkAdmissionQuotaExceeded):
    _reserve(ledger, "same-day", quota=1)

  clock.value = _ns(2026, 7, 28, 0)
  second = _reserve(ledger, "day-two", quota=1)

  assert first.date == "2026-07-27"
  assert second.date == "2026-07-28"


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
