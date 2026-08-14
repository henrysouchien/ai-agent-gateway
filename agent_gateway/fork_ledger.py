"""Durable receipt delivery and paid-fork admission accounting.

The fork ledger deliberately has its own schema and lifecycle, but reuses the
autonomous admission ledger's hardened SQLite file and connection primitives.
All day windows are UTC calendar days.  Money is stored as integer micro-USD so
budget decisions never depend on SQLite floating-point aggregation.

Opening a ledger never reconciles another process instance's in-flight state.
Recovery is explicit: a lifecycle owner supplies the complete live process set
at startup, or a process instance whose death has been established.  Exclusion
recovery makes unidentified crash remnants recoverable without risking rows
owned by another known-live ledger opener.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import sqlite3
import time
from typing import Callable, Literal, TypeVar
from uuid import uuid4

from .autonomous_admission_ledger import (
  AutonomousAdmissionLedgerError,
  AutonomousAdmissionLedgerIdentityError,
  _canonical_absolute_path as _canonical_hardened_path,
  _configure_connection as _configure_hardened_connection,
  _database_stat as _hardened_database_stat,
  _prepare_database_file as _prepare_hardened_database_file,
  _verify_parent_directory as _verify_hardened_parent_directory,
  _verify_sidecars as _verify_hardened_sidecars,
)


FORK_LEDGER_SCHEMA_VERSION = 1
MICRO_USD = Decimal("0.000001")
_MAX_TEXT_LENGTH = 4_096

ReceiptState = Literal["pending", "claimed", "acked"]
AdmissionState = Literal["reserved", "started", "settled", "abandoned"]
_TransactionResult = TypeVar("_TransactionResult")

_METADATA_SQL = """
  CREATE TABLE IF NOT EXISTS fork_ledger_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    day_boundary TEXT NOT NULL CHECK (day_boundary = 'UTC'),
    money_unit TEXT NOT NULL CHECK (money_unit = 'micro_usd')
  )
"""
_CLOCK_SQL = """
  CREATE TABLE IF NOT EXISTS fork_ledger_clock (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    last_observed_wall_ns INTEGER NOT NULL CHECK (
      typeof(last_observed_wall_ns) = 'integer'
      AND last_observed_wall_ns >= 0
    )
  )
"""
_RECEIPTS_SQL = f"""
  CREATE TABLE IF NOT EXISTS fork_receipts (
    fork_id TEXT PRIMARY KEY CHECK (
      typeof(fork_id) = 'text'
      AND length(fork_id) BETWEEN 1 AND {_MAX_TEXT_LENGTH}
      AND fork_id = trim(fork_id)
    ),
    session_id TEXT NOT NULL CHECK (
      typeof(session_id) = 'text'
      AND length(session_id) BETWEEN 1 AND {_MAX_TEXT_LENGTH}
      AND session_id = trim(session_id)
    ),
    owner TEXT NOT NULL CHECK (
      typeof(owner) = 'text'
      AND length(owner) BETWEEN 1 AND {_MAX_TEXT_LENGTH}
      AND owner = trim(owner)
    ),
    receipt_text TEXT NOT NULL CHECK (
      typeof(receipt_text) = 'text'
      AND length(receipt_text) BETWEEN 1 AND {_MAX_TEXT_LENGTH}
      AND instr(receipt_text, char(10)) = 0
      AND instr(receipt_text, char(13)) = 0
    ),
    state TEXT NOT NULL CHECK (state IN ('pending', 'claimed', 'acked')),
    claim_token TEXT,
    claiming_turn_id TEXT,
    process_instance_id TEXT,
    created_at_ns INTEGER NOT NULL CHECK (created_at_ns > 0),
    updated_at_ns INTEGER NOT NULL CHECK (updated_at_ns >= created_at_ns),
    claimed_at_ns INTEGER,
    acked_at_ns INTEGER,
    CHECK (
      (state = 'pending' AND claim_token IS NULL
       AND claiming_turn_id IS NULL AND process_instance_id IS NULL)
      OR
      (state = 'claimed' AND claim_token IS NOT NULL
       AND claiming_turn_id IS NOT NULL AND process_instance_id IS NOT NULL
       AND claimed_at_ns IS NOT NULL)
      OR
      (state = 'acked' AND claim_token IS NOT NULL
       AND claiming_turn_id IS NOT NULL AND process_instance_id IS NOT NULL
       AND claimed_at_ns IS NOT NULL AND acked_at_ns IS NOT NULL)
    )
  ) WITHOUT ROWID
"""
_ADMISSIONS_SQL = f"""
  CREATE TABLE IF NOT EXISTS fork_admissions (
    fork_id TEXT PRIMARY KEY CHECK (
      typeof(fork_id) = 'text'
      AND length(fork_id) BETWEEN 1 AND {_MAX_TEXT_LENGTH}
      AND fork_id = trim(fork_id)
    ),
    owner TEXT NOT NULL CHECK (
      typeof(owner) = 'text'
      AND length(owner) BETWEEN 1 AND {_MAX_TEXT_LENGTH}
      AND owner = trim(owner)
    ),
    admission_date TEXT NOT NULL CHECK (
      length(admission_date) = 10
      AND admission_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
    ),
    max_reserved_microusd INTEGER NOT NULL CHECK (
      typeof(max_reserved_microusd) = 'integer'
      AND max_reserved_microusd > 0
    ),
    settled_microusd INTEGER CHECK (
      settled_microusd IS NULL
      OR (
        typeof(settled_microusd) = 'integer'
        AND settled_microusd >= 0
        AND settled_microusd <= max_reserved_microusd
      )
    ),
    state TEXT NOT NULL CHECK (
      state IN ('reserved', 'started', 'settled', 'abandoned')
    ),
    process_instance_id TEXT NOT NULL CHECK (
      typeof(process_instance_id) = 'text'
      AND length(process_instance_id) BETWEEN 1 AND {_MAX_TEXT_LENGTH}
      AND process_instance_id = trim(process_instance_id)
    ),
    created_at_ns INTEGER NOT NULL CHECK (created_at_ns > 0),
    updated_at_ns INTEGER NOT NULL CHECK (updated_at_ns >= created_at_ns),
    started_at_ns INTEGER,
    settled_at_ns INTEGER,
    CHECK (
      (state = 'reserved' AND started_at_ns IS NULL
       AND settled_at_ns IS NULL AND settled_microusd IS NULL)
      OR
      (state = 'started' AND started_at_ns IS NOT NULL
       AND settled_at_ns IS NULL AND settled_microusd IS NULL)
      OR
      (state IN ('settled', 'abandoned')
       AND settled_at_ns IS NOT NULL AND settled_microusd IS NOT NULL)
    )
  ) WITHOUT ROWID
"""
_RECEIPT_PENDING_INDEX_SQL = """
  CREATE INDEX IF NOT EXISTS idx_fork_receipts_pending
  ON fork_receipts (session_id, owner, state, created_at_ns, fork_id)
"""
_ADMISSION_OWNER_DAY_INDEX_SQL = """
  CREATE INDEX IF NOT EXISTS idx_fork_admissions_owner_day
  ON fork_admissions (owner, admission_date, state, fork_id)
"""
_SCHEMA_OBJECTS = {
  "fork_ledger_metadata": ("table", _METADATA_SQL),
  "fork_ledger_clock": ("table", _CLOCK_SQL),
  "fork_receipts": ("table", _RECEIPTS_SQL),
  "fork_admissions": ("table", _ADMISSIONS_SQL),
  "idx_fork_receipts_pending": ("index", _RECEIPT_PENDING_INDEX_SQL),
  "idx_fork_admissions_owner_day": (
    "index",
    _ADMISSION_OWNER_DAY_INDEX_SQL,
  ),
}


class ForkLedgerError(RuntimeError):
  """The durable fork ledger could not safely complete an operation."""


class ForkLedgerUnavailable(ForkLedgerError):
  """The ledger file, schema, or transaction is unavailable."""


class ForkLedgerIdentityError(ForkLedgerError):
  """The prepared ledger file identity is no longer authoritative."""


class ForkLedgerClockRollback(ForkLedgerError):
  """Wall time moved behind the ledger's durable monotonic watermark."""


class ForkLedgerDuplicate(ForkLedgerError):
  """A fork-keyed record already exists."""


class ForkAdmissionQuotaExceeded(ForkLedgerError):
  """The owner's UTC-day invocation quota is exhausted."""


class ForkAdmissionBudgetExceeded(ForkLedgerError):
  """The owner's UTC-day reserved/charged budget is exhausted."""


@dataclass(frozen=True, slots=True)
class ReceiptClaim:
  fork_id: str
  session_id: str
  owner: str
  receipt_text: str
  claim_token: str
  claiming_turn_id: str
  process_instance_id: str


@dataclass(frozen=True, slots=True)
class AdmissionRecord:
  fork_id: str
  owner: str
  date: str
  max_reserved_usd: Decimal
  settled_usd: Decimal | None
  state: AdmissionState
  process_instance_id: str


def _required_text(value: object, *, field_name: str) -> str:
  if not isinstance(value, str):
    raise TypeError(f"{field_name} must be text")
  if not value or value != value.strip() or len(value) > _MAX_TEXT_LENGTH:
    raise ValueError(f"{field_name} must be canonical non-empty text")
  return value


def _one_line(value: object) -> str:
  text = _required_text(value, field_name="receipt text")
  if "\n" in text or "\r" in text:
    raise ValueError("receipt text must be one line")
  return text


def _positive_int(value: object, *, field_name: str) -> int:
  if type(value) is not int or value <= 0:
    raise ValueError(f"{field_name} must be a positive integer")
  return value


def _usd_to_micros(
  value: Decimal | int | float | str,
  *,
  field_name: str,
  allow_zero: bool = False,
) -> int:
  try:
    amount = Decimal(str(value))
  except (InvalidOperation, ValueError) as exc:
    raise ValueError(f"{field_name} must be finite USD") from exc
  if not amount.is_finite() or amount < 0 or (amount == 0 and not allow_zero):
    qualifier = "non-negative" if allow_zero else "positive"
    raise ValueError(f"{field_name} must be finite and {qualifier}")
  quantized = amount.quantize(MICRO_USD, rounding=ROUND_HALF_UP)
  micros = int(quantized * 1_000_000)
  if micros == 0 and not allow_zero:
    raise ValueError(f"{field_name} must be at least one micro-USD")
  if micros > (1 << 63) - 1:
    raise ValueError(f"{field_name} is too large")
  return micros


def _micros_to_usd(value: int | None) -> Decimal | None:
  if value is None:
    return None
  return (Decimal(value) / Decimal(1_000_000)).quantize(MICRO_USD)


def _normalized_ddl(sql: str | None) -> str:
  return " ".join((sql or "").replace("IF NOT EXISTS ", "").split())


class ForkLedger:
  """One process lineage's durable fork receipt and admission authority."""

  def __init__(
    self,
    path: str | Path,
    *,
    process_instance_id: str,
    clock_ns: Callable[[], int] = time.time_ns,
  ) -> None:
    self.path = Path(_canonical_hardened_path(path))
    self.process_instance_id = _required_text(
      process_instance_id,
      field_name="process instance id",
    )
    if not callable(clock_ns):
      raise TypeError("fork ledger clock must be callable")
    self._clock_ns = clock_ns
    self._device = 0
    self._inode = 0
    self._prepare()

  def _translate_hardening_error(self, exc: BaseException) -> ForkLedgerError:
    message = str(exc).replace("autonomous admission ledger", "fork ledger")
    if isinstance(exc, AutonomousAdmissionLedgerIdentityError):
      return ForkLedgerIdentityError(message)
    if isinstance(exc, AutonomousAdmissionLedgerError):
      return ForkLedgerUnavailable(message)
    return ForkLedgerUnavailable(message)

  def _prepare(self) -> None:
    try:
      file_stat = _prepare_hardened_database_file(self.path)
      self._device = file_stat.st_dev
      self._inode = file_stat.st_ino
      connection = self._connect()
    except BaseException as exc:
      if isinstance(exc, ForkLedgerError):
        raise
      raise self._translate_hardening_error(exc) from exc
    try:
      connection.execute("BEGIN EXCLUSIVE")
      existing = connection.execute(
        """
        SELECT name FROM sqlite_master
         WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
         LIMIT 1
        """
      ).fetchone()
      if existing is None:
        for sql in (
          _METADATA_SQL,
          _CLOCK_SQL,
          _RECEIPTS_SQL,
          _ADMISSIONS_SQL,
          _RECEIPT_PENDING_INDEX_SQL,
          _ADMISSION_OWNER_DAY_INDEX_SQL,
        ):
          connection.execute(sql)
        connection.execute(
          """
          INSERT INTO fork_ledger_metadata (
            singleton, schema_version, day_boundary, money_unit
          ) VALUES (1, ?, 'UTC', 'micro_usd')
          """,
          (FORK_LEDGER_SCHEMA_VERSION,),
        )
        connection.execute(
          """
          INSERT INTO fork_ledger_clock (
            singleton, last_observed_wall_ns
          ) VALUES (1, 0)
          """
        )
        connection.execute(
          f"PRAGMA user_version={FORK_LEDGER_SCHEMA_VERSION}"
        )
      self._validate_schema(connection)
      connection.commit()
    except BaseException as exc:
      try:
        connection.rollback()
      except sqlite3.Error:
        pass
      if isinstance(exc, sqlite3.Error):
        raise ForkLedgerUnavailable(
          "fork ledger schema preparation failed"
        ) from exc
      raise
    finally:
      connection.close()
    self._verify_identity()

  def _verify_identity(self) -> None:
    try:
      _verify_hardened_parent_directory(self.path)
      file_stat = _hardened_database_stat(self.path)
      _verify_hardened_sidecars(self.path)
    except BaseException as exc:
      raise self._translate_hardening_error(exc) from exc
    if file_stat.st_dev != self._device or file_stat.st_ino != self._inode:
      raise ForkLedgerIdentityError("fork ledger file identity changed")

  def _connect(self) -> sqlite3.Connection:
    self._verify_identity()
    connection: sqlite3.Connection | None = None
    try:
      connection = sqlite3.connect(
        self.path.as_uri() + "?mode=rw",
        uri=True,
        timeout=5.0,
        isolation_level=None,
      )
      connection.row_factory = sqlite3.Row
      _configure_hardened_connection(connection)
      self._verify_identity()
      return connection
    except BaseException as exc:
      if connection is not None:
        connection.close()
      if isinstance(exc, ForkLedgerError):
        raise
      raise self._translate_hardening_error(exc) from exc

  def _validate_schema(self, connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version != FORK_LEDGER_SCHEMA_VERSION:
      raise ForkLedgerUnavailable("fork ledger schema version is incompatible")
    rows = connection.execute(
      """
      SELECT type, name, sql FROM sqlite_master
       WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
       ORDER BY name
      """
    ).fetchall()
    observed = {row["name"]: (row["type"], row["sql"]) for row in rows}
    if set(observed) != set(_SCHEMA_OBJECTS):
      raise ForkLedgerUnavailable("fork ledger schema objects are incompatible")
    for name, (kind, sql) in _SCHEMA_OBJECTS.items():
      actual_kind, actual_sql = observed[name]
      if (
        actual_kind != kind
        or _normalized_ddl(actual_sql) != _normalized_ddl(sql)
      ):
        raise ForkLedgerUnavailable(
          "fork ledger schema constraints are incompatible"
        )
    metadata = connection.execute(
      """
      SELECT singleton, schema_version, day_boundary, money_unit
        FROM fork_ledger_metadata
      """
    ).fetchall()
    clock = connection.execute(
      """
      SELECT singleton, last_observed_wall_ns FROM fork_ledger_clock
      """
    ).fetchall()
    if len(metadata) != 1 or tuple(metadata[0]) != (
      1,
      FORK_LEDGER_SCHEMA_VERSION,
      "UTC",
      "micro_usd",
    ):
      raise ForkLedgerUnavailable("fork ledger metadata is incompatible")
    if (
      len(clock) != 1
      or clock[0]["singleton"] != 1
      or type(clock[0]["last_observed_wall_ns"]) is not int
      or clock[0]["last_observed_wall_ns"] < 0
    ):
      raise ForkLedgerUnavailable("fork ledger clock is incompatible")

  def _now_ns(self) -> int:
    now = self._clock_ns()
    if type(now) is not int or now <= 0:
      raise ValueError("fork ledger clock must return a positive integer")
    return now

  @staticmethod
  def _utc_day(now_ns: int) -> str:
    return datetime.fromtimestamp(
      now_ns // 1_000_000_000,
      tz=timezone.utc,
    ).date().isoformat()

  def _observe_clock(
    self,
    connection: sqlite3.Connection,
    now_ns: int,
  ) -> str:
    row = connection.execute(
      """
      SELECT last_observed_wall_ns FROM fork_ledger_clock
       WHERE singleton = 1
      """
    ).fetchone()
    if row is None or type(row["last_observed_wall_ns"]) is not int:
      raise ForkLedgerUnavailable("fork ledger clock state is unavailable")
    previous = row["last_observed_wall_ns"]
    if now_ns < previous:
      raise ForkLedgerClockRollback("fork ledger wall clock moved backward")
    updated = connection.execute(
      """
      UPDATE fork_ledger_clock SET last_observed_wall_ns = ?
       WHERE singleton = 1 AND last_observed_wall_ns = ?
      """,
      (now_ns, previous),
    )
    if updated.rowcount != 1:
      raise ForkLedgerUnavailable("fork ledger clock update lost exclusivity")
    return self._utc_day(now_ns)

  def _transaction(
    self,
    operation: Callable[
      [sqlite3.Connection, int, str],
      _TransactionResult,
    ],
  ) -> _TransactionResult:
    connection = self._connect()
    try:
      connection.execute("BEGIN IMMEDIATE")
      self._verify_identity()
      self._validate_schema(connection)
      now_ns = self._now_ns()
      day = self._observe_clock(connection, now_ns)
      result = operation(connection, now_ns, day)
      self._verify_identity()
      connection.commit()
      return result
    except BaseException as exc:
      try:
        connection.rollback()
      except sqlite3.Error:
        pass
      if isinstance(exc, sqlite3.Error):
        raise ForkLedgerUnavailable(
          "fork ledger transaction failed"
        ) from exc
      raise
    finally:
      connection.close()

  def reconcile_startup(
    self,
    *,
    live_process_instance_ids: Collection[str] | None = None,
    dead_process_instance_id: str | None = None,
  ) -> tuple[int, int]:
    """Recover crash remnants outside a live set, or one established death."""

    if (live_process_instance_ids is None) == (
      dead_process_instance_id is None
    ):
      raise TypeError(
        "supply exactly one of live_process_instance_ids "
        "or dead_process_instance_id"
      )

    live_ids: frozenset[str] | None = None
    if live_process_instance_ids is not None:
      if isinstance(live_process_instance_ids, (str, bytes)):
        raise TypeError("live process instance ids must be a collection")
      live_ids = frozenset(
        _required_text(value, field_name="live process instance id")
        for value in live_process_instance_ids
      )
      if self.process_instance_id not in live_ids:
        raise ValueError(
          "live process instance ids must include the current instance"
        )
    else:
      dead_process_instance_id = _required_text(
        dead_process_instance_id,
        field_name="dead process instance id",
      )
      if dead_process_instance_id == self.process_instance_id:
        raise ValueError("cannot reconcile the current process instance")

    def operation(
      connection: sqlite3.Connection,
      now_ns: int,
      _day: str,
    ) -> tuple[int, int]:
      if live_ids is not None:
        connection.execute(
          """
          CREATE TEMP TABLE live_fork_process_instances (
            process_instance_id TEXT PRIMARY KEY
          ) WITHOUT ROWID
          """
        )
        connection.executemany(
          """
          INSERT INTO live_fork_process_instances (process_instance_id)
          VALUES (?)
          """,
          ((process_instance_id,) for process_instance_id in live_ids),
        )
        receipt_owner_predicate = """
          NOT EXISTS (
            SELECT 1 FROM live_fork_process_instances AS live
             WHERE live.process_instance_id =
                   fork_receipts.process_instance_id
          )
        """
        admission_owner_predicate = """
          NOT EXISTS (
            SELECT 1 FROM live_fork_process_instances AS live
             WHERE live.process_instance_id =
                   fork_admissions.process_instance_id
          )
        """
        receipt_parameters = (now_ns,)
        admission_parameters = (now_ns, now_ns)
      else:
        receipt_owner_predicate = "process_instance_id = ?"
        admission_owner_predicate = "process_instance_id = ?"
        receipt_parameters = (now_ns, dead_process_instance_id)
        admission_parameters = (
          now_ns,
          now_ns,
          dead_process_instance_id,
        )

      receipts = connection.execute(
        f"""
        UPDATE fork_receipts
           SET state = 'pending', claim_token = NULL,
               claiming_turn_id = NULL, process_instance_id = NULL,
               claimed_at_ns = NULL, updated_at_ns = ?
         WHERE state = 'claimed' AND {receipt_owner_predicate}
        """,
        receipt_parameters,
      ).rowcount
      admissions = connection.execute(
        f"""
        UPDATE fork_admissions
           SET state = 'abandoned',
               settled_microusd = max_reserved_microusd,
               settled_at_ns = ?, updated_at_ns = ?
         WHERE state IN ('reserved', 'started')
           AND {admission_owner_predicate}
        """,
        admission_parameters,
      ).rowcount
      return receipts, admissions

    return self._transaction(operation)

  def write_receipt(
    self,
    *,
    fork_id: str,
    session_id: str,
    owner: str,
    receipt_text: str,
  ) -> bool:
    fork_id = _required_text(fork_id, field_name="fork id")
    session_id = _required_text(session_id, field_name="session id")
    owner = _required_text(owner, field_name="owner")
    receipt_text = _one_line(receipt_text)

    def operation(
      connection: sqlite3.Connection,
      now_ns: int,
      _day: str,
    ) -> bool:
      inserted = connection.execute(
        """
        INSERT OR IGNORE INTO fork_receipts (
          fork_id, session_id, owner, receipt_text, state,
          created_at_ns, updated_at_ns
        ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
        """,
        (fork_id, session_id, owner, receipt_text, now_ns, now_ns),
      )
      if inserted.rowcount == 1:
        return True
      existing = connection.execute(
        """
        SELECT session_id, owner, receipt_text FROM fork_receipts
         WHERE fork_id = ?
        """,
        (fork_id,),
      ).fetchone()
      if existing is None or tuple(existing) != (
        session_id,
        owner,
        receipt_text,
      ):
        raise ForkLedgerDuplicate(
          "fork receipt id is already bound to different content"
        )
      return False

    return bool(self._transaction(operation))

  def claim_pending_receipts(
    self,
    *,
    session_id: str,
    owner: str,
    claiming_turn_id: str,
    limit: int = 100,
  ) -> tuple[ReceiptClaim, ...]:
    session_id = _required_text(session_id, field_name="session id")
    owner = _required_text(owner, field_name="owner")
    claiming_turn_id = _required_text(
      claiming_turn_id,
      field_name="claiming turn id",
    )
    limit = _positive_int(limit, field_name="receipt claim limit")

    def operation(
      connection: sqlite3.Connection,
      now_ns: int,
      _day: str,
    ) -> tuple[ReceiptClaim, ...]:
      rows = connection.execute(
        """
        SELECT fork_id, receipt_text FROM fork_receipts
         WHERE session_id = ? AND owner = ? AND state = 'pending'
         ORDER BY created_at_ns, fork_id LIMIT ?
        """,
        (session_id, owner, limit),
      ).fetchall()
      claims: list[ReceiptClaim] = []
      for row in rows:
        token = uuid4().hex
        updated = connection.execute(
          """
          UPDATE fork_receipts
             SET state = 'claimed', claim_token = ?,
                 claiming_turn_id = ?, process_instance_id = ?,
                 claimed_at_ns = ?, updated_at_ns = ?
           WHERE fork_id = ? AND state = 'pending'
          """,
          (
            token,
            claiming_turn_id,
            self.process_instance_id,
            now_ns,
            now_ns,
            row["fork_id"],
          ),
        )
        if updated.rowcount != 1:
          continue
        claims.append(ReceiptClaim(
          fork_id=row["fork_id"],
          session_id=session_id,
          owner=owner,
          receipt_text=row["receipt_text"],
          claim_token=token,
          claiming_turn_id=claiming_turn_id,
          process_instance_id=self.process_instance_id,
        ))
      return tuple(claims)

    return self._transaction(operation)

  def ack_receipt(self, *, fork_id: str, claim_token: str) -> bool:
    return self._finish_receipt_claim(
      fork_id=fork_id,
      claim_token=claim_token,
      ack=True,
    )

  def revert_receipt_claim(self, *, fork_id: str, claim_token: str) -> bool:
    return self._finish_receipt_claim(
      fork_id=fork_id,
      claim_token=claim_token,
      ack=False,
    )

  def _finish_receipt_claim(
    self,
    *,
    fork_id: str,
    claim_token: str,
    ack: bool,
  ) -> bool:
    fork_id = _required_text(fork_id, field_name="fork id")
    claim_token = _required_text(claim_token, field_name="claim token")

    def operation(
      connection: sqlite3.Connection,
      now_ns: int,
      _day: str,
    ) -> bool:
      if ack:
        result = connection.execute(
          """
          UPDATE fork_receipts
             SET state = 'acked', acked_at_ns = ?, updated_at_ns = ?
           WHERE fork_id = ? AND state = 'claimed' AND claim_token = ?
          """,
          (now_ns, now_ns, fork_id, claim_token),
        )
      else:
        result = connection.execute(
          """
          UPDATE fork_receipts
             SET state = 'pending', claim_token = NULL,
                 claiming_turn_id = NULL, process_instance_id = NULL,
                 claimed_at_ns = NULL, updated_at_ns = ?
           WHERE fork_id = ? AND state = 'claimed' AND claim_token = ?
          """,
          (now_ns, fork_id, claim_token),
        )
      return result.rowcount == 1

    return bool(self._transaction(operation))

  def reserve_admission(
    self,
    *,
    fork_id: str,
    owner: str,
    max_reserved_usd: Decimal | int | float | str,
    daily_budget_usd: Decimal | int | float | str,
    daily_invocation_quota: int,
  ) -> AdmissionRecord:
    fork_id = _required_text(fork_id, field_name="fork id")
    owner = _required_text(owner, field_name="owner")
    reserved = _usd_to_micros(
      max_reserved_usd,
      field_name="fork maximum reservation",
    )
    daily_budget = _usd_to_micros(
      daily_budget_usd,
      field_name="daily fork budget",
    )
    quota = _positive_int(
      daily_invocation_quota,
      field_name="daily fork invocation quota",
    )

    def operation(
      connection: sqlite3.Connection,
      now_ns: int,
      day: str,
    ) -> AdmissionRecord:
      if connection.execute(
        "SELECT 1 FROM fork_admissions WHERE fork_id = ?",
        (fork_id,),
      ).fetchone() is not None:
        raise ForkLedgerDuplicate("fork admission id already exists")
      totals = connection.execute(
        """
        SELECT COUNT(*) AS invocations,
               COALESCE(SUM(
                 CASE
                   WHEN state IN ('reserved', 'started')
                     THEN max_reserved_microusd
                   ELSE settled_microusd
                 END
               ), 0) AS charged_microusd
          FROM fork_admissions
         WHERE owner = ? AND admission_date = ?
        """,
        (owner, day),
      ).fetchone()
      if totals["invocations"] >= quota:
        raise ForkAdmissionQuotaExceeded(
          "daily fork invocation quota is exhausted"
        )
      if totals["charged_microusd"] + reserved > daily_budget:
        raise ForkAdmissionBudgetExceeded("daily fork budget is exhausted")
      connection.execute(
        """
        INSERT INTO fork_admissions (
          fork_id, owner, admission_date, max_reserved_microusd,
          state, process_instance_id, created_at_ns, updated_at_ns
        ) VALUES (?, ?, ?, ?, 'reserved', ?, ?, ?)
        """,
        (
          fork_id,
          owner,
          day,
          reserved,
          self.process_instance_id,
          now_ns,
          now_ns,
        ),
      )
      return AdmissionRecord(
        fork_id=fork_id,
        owner=owner,
        date=day,
        max_reserved_usd=_micros_to_usd(reserved),
        settled_usd=None,
        state="reserved",
        process_instance_id=self.process_instance_id,
      )

    return self._transaction(operation)

  def mark_admission_started(self, *, fork_id: str) -> bool:
    fork_id = _required_text(fork_id, field_name="fork id")

    def operation(
      connection: sqlite3.Connection,
      now_ns: int,
      _day: str,
    ) -> bool:
      result = connection.execute(
        """
        UPDATE fork_admissions
           SET state = 'started', started_at_ns = ?, updated_at_ns = ?
         WHERE fork_id = ? AND state = 'reserved'
           AND process_instance_id = ?
        """,
        (now_ns, now_ns, fork_id, self.process_instance_id),
      )
      return result.rowcount == 1

    return bool(self._transaction(operation))

  def settle_admission(
    self,
    *,
    fork_id: str,
    actual_cost_usd: Decimal | int | float | str,
  ) -> bool:
    fork_id = _required_text(fork_id, field_name="fork id")
    actual = _usd_to_micros(
      actual_cost_usd,
      field_name="actual fork cost",
      allow_zero=True,
    )

    def operation(
      connection: sqlite3.Connection,
      now_ns: int,
      _day: str,
    ) -> bool:
      result = connection.execute(
        """
        UPDATE fork_admissions
           SET state = 'settled', settled_microusd = ?,
               settled_at_ns = ?, updated_at_ns = ?
         WHERE fork_id = ? AND state IN ('reserved', 'started')
           AND process_instance_id = ?
           AND max_reserved_microusd >= ?
        """,
        (
          actual,
          now_ns,
          now_ns,
          fork_id,
          self.process_instance_id,
          actual,
        ),
      )
      return result.rowcount == 1

    return bool(self._transaction(operation))

  def abandon_admission(self, *, fork_id: str) -> bool:
    fork_id = _required_text(fork_id, field_name="fork id")

    def operation(
      connection: sqlite3.Connection,
      now_ns: int,
      _day: str,
    ) -> bool:
      result = connection.execute(
        """
        UPDATE fork_admissions
           SET state = 'abandoned',
               settled_microusd = max_reserved_microusd,
               settled_at_ns = ?, updated_at_ns = ?
         WHERE fork_id = ? AND state IN ('reserved', 'started')
           AND process_instance_id = ?
        """,
        (now_ns, now_ns, fork_id, self.process_instance_id),
      )
      return result.rowcount == 1

    return bool(self._transaction(operation))

  def get_admission(self, fork_id: str) -> AdmissionRecord | None:
    fork_id = _required_text(fork_id, field_name="fork id")
    connection = self._connect()
    try:
      self._validate_schema(connection)
      row = connection.execute(
        """
        SELECT fork_id, owner, admission_date, max_reserved_microusd,
               settled_microusd, state, process_instance_id
          FROM fork_admissions WHERE fork_id = ?
        """,
        (fork_id,),
      ).fetchone()
    finally:
      connection.close()
    if row is None:
      return None
    return AdmissionRecord(
      fork_id=row["fork_id"],
      owner=row["owner"],
      date=row["admission_date"],
      max_reserved_usd=_micros_to_usd(row["max_reserved_microusd"]),
      settled_usd=_micros_to_usd(row["settled_microusd"]),
      state=row["state"],
      process_instance_id=row["process_instance_id"],
    )


__all__ = [
  "AdmissionRecord",
  "ForkAdmissionBudgetExceeded",
  "ForkAdmissionQuotaExceeded",
  "ForkLedger",
  "ForkLedgerClockRollback",
  "ForkLedgerDuplicate",
  "ForkLedgerError",
  "ForkLedgerIdentityError",
  "ForkLedgerUnavailable",
  "ReceiptClaim",
]
