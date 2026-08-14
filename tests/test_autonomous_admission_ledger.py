from __future__ import annotations

from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any

import pytest

from agent_gateway import autonomous_admission_ledger as ledger_module
from agent_gateway.autonomous_admission_ledger import (
  AUTONOMOUS_ADMISSION_LEDGER_SCHEMA_VERSION,
  AutonomousAdmissionLedgerCapacityExceeded,
  AutonomousAdmissionLedgerClockRollback,
  AutonomousAdmissionLedgerDuplicate,
  AutonomousAdmissionLedgerExpired,
  AutonomousAdmissionLedgerIdentity,
  AutonomousAdmissionLedgerIdentityError,
  AutonomousAdmissionLedgerUnavailable,
  OrdinaryAutonomousAdmissionReceipt,
  consume_ordinary_autonomous_launch_once,
  prepare_autonomous_admission_ledger,
)
from agent_gateway.autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE,
)


_ISSUED_AT_NS = 1_785_000_000_000_000_000
_EXPIRES_AT_NS = _ISSUED_AT_NS + 60_000_000_000
_ADMISSION_NS = _ISSUED_AT_NS + 1_000_000_000


def _path(tmp_path: Path, name: str = "admissions.sqlite3") -> Path:
  return tmp_path.resolve() / name


def _receipt(
  marker: int = 1,
  **changes: Any,
) -> OrdinaryAutonomousAdmissionReceipt:
  facts: dict[str, Any] = {
    "audience": AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE,
    "nonce": f"{marker:032x}",
    "task_id": f"task-{marker}",
    "control_run_id": f"run-{marker}",
    "owner_user_id": "owner-1",
    "channel_id": f"{marker:064x}",
    "issued_at_ns": _ISSUED_AT_NS,
    "expires_at_ns": _EXPIRES_AT_NS,
  }
  facts.update(changes)
  return OrdinaryAutonomousAdmissionReceipt(**facts)


def _row_count(path: Path) -> int:
  with sqlite3.connect(path) as connection:
    return connection.execute(
      "SELECT COUNT(*) FROM ordinary_autonomous_launch_admissions"
    ).fetchone()[0]


def _process_consume(
  identity_payload: dict[str, int | str],
  receipt_payload: dict[str, int | str],
  now_ns: int,
  output: multiprocessing.Queue,
) -> None:
  identity = AutonomousAdmissionLedgerIdentity.from_receipt(
    identity_payload
  )
  receipt = OrdinaryAutonomousAdmissionReceipt.from_receipt(
    receipt_payload
  )
  try:
    consume_ordinary_autonomous_launch_once(
      identity,
      receipt,
      clock_ns=lambda: now_ns,
    )
  except AutonomousAdmissionLedgerDuplicate:
    output.put("duplicate")
  except BaseException as exc:
    output.put(f"error:{type(exc).__name__}:{exc}")
  else:
    output.put("admitted")


def test_prepare_creates_closed_durable_canonical_identity(
  tmp_path: Path,
) -> None:
  path = _path(tmp_path)

  identity = prepare_autonomous_admission_ledger(path)

  assert identity == AutonomousAdmissionLedgerIdentity.from_receipt(
    identity.receipt()
  )
  assert identity.receipt() == {
    "schema_version": AUTONOMOUS_ADMISSION_LEDGER_SCHEMA_VERSION,
    "path": str(path),
    "device": path.stat().st_dev,
    "inode": path.stat().st_ino,
  }
  assert stat.S_IMODE(path.stat().st_mode) == 0o600
  assert path.stat().st_nlink == 1
  assert prepare_autonomous_admission_ledger(path) == identity
  with sqlite3.connect(path) as connection:
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    metadata = connection.execute(
      """
      SELECT schema_version, page_size, max_page_count,
             max_rows, cleanup_batch_size
        FROM autonomous_admission_ledger_metadata
      """
    ).fetchone()
  assert metadata == (
    ledger_module.AUTONOMOUS_ADMISSION_LEDGER_SCHEMA_VERSION,
    ledger_module.AUTONOMOUS_ADMISSION_LEDGER_PAGE_SIZE,
    ledger_module.AUTONOMOUS_ADMISSION_LEDGER_MAX_PAGE_COUNT,
    ledger_module.AUTONOMOUS_ADMISSION_LEDGER_MAX_ROWS,
    ledger_module.AUTONOMOUS_ADMISSION_LEDGER_CLEANUP_BATCH_SIZE,
  )


def test_prepare_repairs_mode_but_rejects_symlink_hardlink_and_nonregular(
  tmp_path: Path,
) -> None:
  insecure = _path(tmp_path, "insecure.sqlite3")
  insecure.touch(mode=0o644)
  os.chmod(insecure, 0o644)
  prepare_autonomous_admission_ledger(insecure)
  assert stat.S_IMODE(insecure.stat().st_mode) == 0o600

  target = _path(tmp_path, "target.sqlite3")
  target.touch(mode=0o600)
  symlink = _path(tmp_path, "symlink.sqlite3")
  symlink.symlink_to(target)
  with pytest.raises(
    AutonomousAdmissionLedgerIdentityError,
    match="securely",
  ):
    prepare_autonomous_admission_ledger(symlink)

  hardlink = _path(tmp_path, "hardlink.sqlite3")
  os.link(target, hardlink)
  with pytest.raises(
    AutonomousAdmissionLedgerIdentityError,
    match="identity is unsafe",
  ):
    prepare_autonomous_admission_ledger(target)

  directory = _path(tmp_path, "directory.sqlite3")
  directory.mkdir()
  with pytest.raises(AutonomousAdmissionLedgerIdentityError):
    prepare_autonomous_admission_ledger(directory)


def test_identity_and_receipt_contracts_are_closed_and_exact(
  tmp_path: Path,
) -> None:
  identity = prepare_autonomous_admission_ledger(_path(tmp_path))
  receipt = _receipt()

  with pytest.raises(ValueError, match="closed contract"):
    AutonomousAdmissionLedgerIdentity.from_receipt({
      **identity.receipt(),
      "extra": 1,
    })
  with pytest.raises(ValueError, match="inode"):
    AutonomousAdmissionLedgerIdentity(
      schema_version=1,
      path=identity.path,
      device=identity.device,
      inode=True,
    )
  with pytest.raises(ValueError, match="canonical absolute"):
    AutonomousAdmissionLedgerIdentity(
      schema_version=1,
      path="relative.sqlite3",
      device=1,
      inode=1,
    )
  with pytest.raises(ValueError, match="closed contract"):
    OrdinaryAutonomousAdmissionReceipt.from_receipt({
      **receipt.receipt(),
      "extra": None,
    })
  with pytest.raises(ValueError, match="nonce"):
    replace(receipt, nonce="A" * 32)
  with pytest.raises(ValueError, match="canonical identifier"):
    replace(receipt, task_id=" task")
  with pytest.raises(ValueError, match="channel_id"):
    replace(receipt, channel_id="A" * 64)
  with pytest.raises(ValueError, match="integer"):
    replace(receipt, issued_at_ns=True)
  with pytest.raises(ValueError, match="TTL"):
    replace(
      receipt,
      expires_at_ns=(
        receipt.issued_at_ns
        + 301_000_000_000
      ),
    )
  with pytest.raises(TypeError, match="verified launch envelope"):
    OrdinaryAutonomousAdmissionReceipt.from_verified_envelope(object())


def test_consume_commits_closed_receipt_before_return_and_replay_fails(
  tmp_path: Path,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  receipt = _receipt()

  record = consume_ordinary_autonomous_launch_once(
    identity,
    receipt,
    clock_ns=lambda: _ADMISSION_NS,
  )

  assert record.receipt == receipt
  assert record.admitted_at_ns == _ADMISSION_NS
  with sqlite3.connect(path) as connection:
    row = connection.execute(
      """
      SELECT audience, nonce, task_id, control_run_id, owner_user_id,
             channel_id, issued_at_ns, expires_at_ns, admitted_at_ns
        FROM ordinary_autonomous_launch_admissions
      """
    ).fetchone()
  assert row == (
    receipt.audience,
    receipt.nonce,
    receipt.task_id,
    receipt.control_run_id,
    receipt.owner_user_id,
    receipt.channel_id,
    receipt.issued_at_ns,
    receipt.expires_at_ns,
    _ADMISSION_NS,
  )
  with pytest.raises(AutonomousAdmissionLedgerDuplicate):
    consume_ordinary_autonomous_launch_once(
      identity,
      receipt,
      clock_ns=lambda: _ADMISSION_NS,
    )
  with pytest.raises(AutonomousAdmissionLedgerDuplicate):
    consume_ordinary_autonomous_launch_once(
      identity,
      replace(receipt, task_id="different-task"),
      clock_ns=lambda: _ADMISSION_NS,
    )
  assert _row_count(path) == 1


@pytest.mark.skipif(
  "fork" not in multiprocessing.get_all_start_methods(),
  reason="secure admission ledger is POSIX-only",
)
def test_concurrent_processes_using_independent_connections_have_one_winner(
  tmp_path: Path,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  receipt = _receipt()
  context = multiprocessing.get_context("fork")
  output = context.Queue()
  processes = [
    context.Process(
      target=_process_consume,
      args=(
        identity.receipt(),
        receipt.receipt(),
        _ADMISSION_NS,
        output,
      ),
    )
    for _ in range(2)
  ]
  for process in processes:
    process.start()
  for process in processes:
    process.join(timeout=15)
    if process.is_alive():
      process.terminate()
      process.join(timeout=5)
      pytest.fail("admission worker exceeded its bounded join")
    assert process.exitcode == 0

  assert sorted(output.get(timeout=2) for _ in processes) == [
    "admitted",
    "duplicate",
  ]
  assert _row_count(path) == 1


@pytest.mark.parametrize(
  ("now_ns", "expected_fragment"),
  (
    (_ISSUED_AT_NS - 1, "not yet valid"),
    (_EXPIRES_AT_NS, "expired"),
  ),
)
def test_wall_expiry_fails_before_consumption(
  tmp_path: Path,
  now_ns: int,
  expected_fragment: str,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)

  with pytest.raises(
    AutonomousAdmissionLedgerExpired,
    match=expected_fragment,
  ) as raised:
    consume_ordinary_autonomous_launch_once(
      identity,
      _receipt(),
      clock_ns=lambda: now_ns,
    )

  assert raised.value.consumed is False
  assert _row_count(path) == 0


def test_expiry_after_commit_fails_closed_and_leaves_nonce_consumed(
  tmp_path: Path,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  receipt = _receipt()
  observed_times = iter((
    _ADMISSION_NS,
    _ADMISSION_NS,
    receipt.expires_at_ns,
  ))

  with pytest.raises(
    AutonomousAdmissionLedgerExpired,
    match="expired",
  ) as raised:
    consume_ordinary_autonomous_launch_once(
      identity,
      receipt,
      clock_ns=lambda: next(observed_times),
    )

  assert raised.value.consumed is True
  assert _row_count(path) == 1


def test_clock_rollback_after_commit_fails_closed_and_consumes_nonce(
  tmp_path: Path,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  receipt = _receipt()
  observed_times = iter((
    _ADMISSION_NS,
    _ADMISSION_NS,
    _ADMISSION_NS - 1,
  ))

  with pytest.raises(
    AutonomousAdmissionLedgerClockRollback,
    match="backward",
  ):
    consume_ordinary_autonomous_launch_once(
      identity,
      receipt,
      clock_ns=lambda: next(observed_times),
    )

  assert _row_count(path) == 1


def test_concurrent_high_water_advance_is_not_a_false_clock_rollback(
  tmp_path: Path,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  first = _receipt(1)
  second = _receipt(2)
  clock_calls = 0

  def _first_clock() -> int:
    nonlocal clock_calls
    clock_calls += 1
    if clock_calls < 3:
      return _ADMISSION_NS
    observed_ns = _ADMISSION_NS + 1
    consume_ordinary_autonomous_launch_once(
      identity,
      second,
      clock_ns=lambda: _ADMISSION_NS + 2,
    )
    return observed_ns

  record = consume_ordinary_autonomous_launch_once(
    identity,
    first,
    clock_ns=_first_clock,
  )

  assert record.receipt == first
  assert _row_count(path) == 2


def test_durable_wall_high_water_prevents_replay_after_cleanup_and_rollback(
  tmp_path: Path,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  first = _receipt(
    1,
    expires_at_ns=_ISSUED_AT_NS + 15_000_000_000,
  )
  second = _receipt(
    2,
    issued_at_ns=_ISSUED_AT_NS + 16_000_000_000,
    expires_at_ns=_ISSUED_AT_NS + 30_000_000_000,
  )
  consume_ordinary_autonomous_launch_once(
    identity,
    first,
    clock_ns=lambda: _ISSUED_AT_NS + 10_000_000_000,
  )
  consume_ordinary_autonomous_launch_once(
    identity,
    second,
    clock_ns=lambda: _ISSUED_AT_NS + 20_000_000_000,
  )
  assert _row_count(path) == 1

  with pytest.raises(
    AutonomousAdmissionLedgerClockRollback,
    match="backward",
  ):
    consume_ordinary_autonomous_launch_once(
      identity,
      first,
      clock_ns=lambda: _ISSUED_AT_NS + 12_000_000_000,
    )

  assert _row_count(path) == 1


def test_cleanup_is_bounded_and_occurs_in_the_admission_transaction(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    ledger_module,
    "AUTONOMOUS_ADMISSION_LEDGER_CLEANUP_BATCH_SIZE",
    2,
  )
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  with sqlite3.connect(path) as connection:
    for marker in range(10, 15):
      connection.execute(
        """
        INSERT INTO ordinary_autonomous_launch_admissions (
          nonce, schema_version, audience, task_id, control_run_id,
          owner_user_id, channel_id, issued_at_ns, expires_at_ns,
          admitted_at_ns
        ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          f"{marker:032x}",
          AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE,
          f"expired-task-{marker}",
          f"expired-run-{marker}",
          "owner-1",
          f"{marker:064x}",
          _ISSUED_AT_NS - 20,
          _ISSUED_AT_NS - 10,
          _ISSUED_AT_NS - 15,
        ),
      )

  consume_ordinary_autonomous_launch_once(
    identity,
    _receipt(),
    clock_ns=lambda: _ADMISSION_NS,
  )

  with sqlite3.connect(path) as connection:
    expired_count = connection.execute(
      """
      SELECT COUNT(*)
        FROM ordinary_autonomous_launch_admissions
       WHERE expires_at_ns <= ?
      """,
      (_ADMISSION_NS,),
    ).fetchone()[0]
  assert expired_count == 3
  assert _row_count(path) == 4


def test_capacity_failure_rolls_back_the_uncommitted_nonce(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    ledger_module,
    "AUTONOMOUS_ADMISSION_LEDGER_MAX_ROWS",
    1,
  )
  monkeypatch.setattr(
    ledger_module,
    "AUTONOMOUS_ADMISSION_LEDGER_CLEANUP_BATCH_SIZE",
    1,
  )
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  first = _receipt(1)
  second = _receipt(2)
  consume_ordinary_autonomous_launch_once(
    identity,
    first,
    clock_ns=lambda: _ADMISSION_NS,
  )

  with pytest.raises(AutonomousAdmissionLedgerCapacityExceeded):
    consume_ordinary_autonomous_launch_once(
      identity,
      second,
      clock_ns=lambda: _ADMISSION_NS,
    )

  with sqlite3.connect(path) as connection:
    nonces = connection.execute(
      "SELECT nonce FROM ordinary_autonomous_launch_admissions"
    ).fetchall()
  assert nonces == [(first.nonce,)]


def test_schema_and_persisted_bound_mismatches_fail_closed(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  monkeypatch.setattr(
    ledger_module,
    "AUTONOMOUS_ADMISSION_LEDGER_MAX_ROWS",
    ledger_module.AUTONOMOUS_ADMISSION_LEDGER_MAX_ROWS + 1,
  )

  with pytest.raises(
    AutonomousAdmissionLedgerUnavailable,
    match="bounds",
  ):
    consume_ordinary_autonomous_launch_once(
      identity,
      _receipt(),
      clock_ns=lambda: _ADMISSION_NS,
    )
  assert _row_count(path) == 0


def test_schema_object_drift_fails_closed_without_self_migration(
  tmp_path: Path,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  with sqlite3.connect(path) as connection:
    connection.execute(
      "DROP INDEX idx_autonomous_admissions_expiry"
    )

  with pytest.raises(
    AutonomousAdmissionLedgerUnavailable,
    match="schema objects",
  ):
    consume_ordinary_autonomous_launch_once(
      identity,
      _receipt(),
      clock_ns=lambda: _ADMISSION_NS,
    )

  assert _row_count(path) == 0


def test_missing_or_replaced_database_never_falls_back_to_a_new_ledger(
  tmp_path: Path,
) -> None:
  missing_path = _path(tmp_path, "missing.sqlite3")
  missing_identity = prepare_autonomous_admission_ledger(missing_path)
  missing_path.unlink()
  with pytest.raises(AutonomousAdmissionLedgerIdentityError):
    consume_ordinary_autonomous_launch_once(
      missing_identity,
      _receipt(),
      clock_ns=lambda: _ADMISSION_NS,
    )
  assert not missing_path.exists()

  replaced_path = _path(tmp_path, "replaced.sqlite3")
  replaced_identity = prepare_autonomous_admission_ledger(replaced_path)
  old_path = _path(tmp_path, "replaced-old.sqlite3")
  replaced_path.rename(old_path)
  replaced_path.touch(mode=0o600)
  with pytest.raises(
    AutonomousAdmissionLedgerIdentityError,
    match="identity changed",
  ):
    consume_ordinary_autonomous_launch_once(
      replaced_identity,
      _receipt(2),
      clock_ns=lambda: _ADMISSION_NS,
    )
  assert replaced_path.stat().st_size == 0
  assert _row_count(old_path) == 0


def test_identity_replacement_after_commit_is_detected_before_admission(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  path = _path(tmp_path)
  identity = prepare_autonomous_admission_ledger(path)
  receipt = _receipt()
  original_verify = ledger_module._verify_identity
  replaced = False
  old_path = _path(tmp_path, "committed-old.sqlite3")

  def replace_after_commit(
    observed_identity: AutonomousAdmissionLedgerIdentity,
  ) -> os.stat_result:
    nonlocal replaced
    if not replaced:
      try:
        with sqlite3.connect(
          path.as_uri() + "?mode=ro",
          uri=True,
        ) as observer:
          committed_rows = observer.execute(
            "SELECT COUNT(*) "
            "FROM ordinary_autonomous_launch_admissions"
          ).fetchone()[0]
      except sqlite3.Error:
        committed_rows = 0
      if committed_rows == 1:
        path.rename(old_path)
        path.touch(mode=0o600)
        replaced = True
    return original_verify(observed_identity)

  monkeypatch.setattr(
    ledger_module,
    "_verify_identity",
    replace_after_commit,
  )

  with pytest.raises(
    AutonomousAdmissionLedgerIdentityError,
    match="identity changed",
  ):
    consume_ordinary_autonomous_launch_once(
      identity,
      receipt,
      clock_ns=lambda: _ADMISSION_NS,
    )

  assert replaced is True
  assert _row_count(old_path) == 1
  assert path.stat().st_size == 0


def test_permission_hardlink_and_sidecar_changes_fail_closed(
  tmp_path: Path,
) -> None:
  permission_path = _path(tmp_path, "permission.sqlite3")
  permission_identity = prepare_autonomous_admission_ledger(
    permission_path
  )
  os.chmod(permission_path, 0o640)
  with pytest.raises(
    AutonomousAdmissionLedgerIdentityError,
    match="0600",
  ):
    consume_ordinary_autonomous_launch_once(
      permission_identity,
      _receipt(),
      clock_ns=lambda: _ADMISSION_NS,
    )

  hardlink_path = _path(tmp_path, "hardlink-live.sqlite3")
  hardlink_identity = prepare_autonomous_admission_ledger(hardlink_path)
  os.link(hardlink_path, _path(tmp_path, "second-link.sqlite3"))
  with pytest.raises(
    AutonomousAdmissionLedgerIdentityError,
    match="hard link",
  ):
    consume_ordinary_autonomous_launch_once(
      hardlink_identity,
      _receipt(2),
      clock_ns=lambda: _ADMISSION_NS,
    )

  sidecar_path = _path(tmp_path, "sidecar.sqlite3")
  sidecar_identity = prepare_autonomous_admission_ledger(sidecar_path)
  unsafe_sidecar = Path(str(sidecar_path) + "-journal")
  unsafe_sidecar.symlink_to(permission_path)
  with pytest.raises(
    AutonomousAdmissionLedgerIdentityError,
    match="sidecar identity",
  ):
    consume_ordinary_autonomous_launch_once(
      sidecar_identity,
      _receipt(3),
      clock_ns=lambda: _ADMISSION_NS,
    )
