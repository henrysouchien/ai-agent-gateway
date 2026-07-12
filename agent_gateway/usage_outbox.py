"""Durable SQLite outbox for canonical commercial usage batches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from jsonschema import Draft202012Validator

from .commercial_contract import canonical_usage_payload_sha256


OUTBOX_SCHEMA_VERSION = 3
_V1_OUTBOX_TABLE_INFO = (
  ("event_id", "TEXT", 0, None, 1),
  ("payload_sha256", "TEXT", 1, None, 0),
  ("payload_json", "TEXT", 1, None, 0),
  ("state", "TEXT", 1, None, 0),
  ("attempt_count", "INTEGER", 1, "0", 0),
  ("next_attempt_at", "TEXT", 0, None, 0),
  ("sending_lease_token", "TEXT", 0, None, 0),
  ("sending_lease_expires_at", "TEXT", 0, None, 0),
  ("accepted_at", "TEXT", 0, None, 0),
  ("last_error", "TEXT", 0, None, 0),
  ("created_at", "TEXT", 1, None, 0),
)
_V1_OUTBOX_TABLE_SQL = """
  CREATE TABLE IF NOT EXISTS commercial_usage_outbox (
    event_id TEXT PRIMARY KEY,
    payload_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
      state IN ('pending', 'sending', 'accepted', 'retryable', 'dead')
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    next_attempt_at TEXT,
    sending_lease_token TEXT,
    sending_lease_expires_at TEXT,
    accepted_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    CHECK (
      (state = 'sending' AND sending_lease_token IS NOT NULL AND sending_lease_expires_at IS NOT NULL)
      OR (state <> 'sending' AND sending_lease_token IS NULL AND sending_lease_expires_at IS NULL)
    ),
    CHECK ((state = 'accepted' AND accepted_at IS NOT NULL) OR state <> 'accepted')
  )
"""
_OUTBOX_TABLE_INFO = (
  ("event_id", "TEXT", 0, None, 1),
  ("payload_sha256", "TEXT", 1, None, 0),
  ("payload_json", "TEXT", 1, None, 0),
  ("environment", "TEXT", 1, None, 0),
  ("source_product", "TEXT", 1, None, 0),
  ("occurred_at", "TEXT", 1, None, 0),
  ("request_id", "TEXT", 1, None, 0),
  ("session_id", "TEXT", 1, None, 0),
  ("state", "TEXT", 1, None, 0),
  ("attempt_count", "INTEGER", 1, "0", 0),
  ("next_attempt_at", "TEXT", 0, None, 0),
  ("sending_lease_token", "TEXT", 0, None, 0),
  ("sending_lease_expires_at", "TEXT", 0, None, 0),
  ("ingest_status", "TEXT", 0, None, 0),
  ("canonical_event_id", "TEXT", 0, None, 0),
  ("ingest_reason_code", "TEXT", 0, None, 0),
  ("ingest_decided_at", "TEXT", 0, None, 0),
  ("accepted_at", "TEXT", 0, None, 0),
  ("last_error", "TEXT", 0, None, 0),
  ("created_at", "TEXT", 1, None, 0),
)
_OUTBOX_TABLE_SQL = """
  CREATE TABLE IF NOT EXISTS commercial_usage_outbox (
    event_id TEXT PRIMARY KEY,
    payload_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    environment TEXT NOT NULL,
    source_product TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
      state IN ('pending', 'sending', 'accepted', 'retryable', 'dead')
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    next_attempt_at TEXT,
    sending_lease_token TEXT,
    sending_lease_expires_at TEXT,
    ingest_status TEXT CHECK (
      ingest_status IS NULL
      OR ingest_status IN (
        'accepted', 'duplicate', 'conflict', 'rejected_terminal', 'legacy_unknown'
      )
    ),
    canonical_event_id TEXT,
    ingest_reason_code TEXT,
    ingest_decided_at TEXT,
    accepted_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    CHECK (
      (state = 'sending' AND sending_lease_token IS NOT NULL AND sending_lease_expires_at IS NOT NULL)
      OR (state <> 'sending' AND sending_lease_token IS NULL AND sending_lease_expires_at IS NULL)
    ),
    CHECK (
      (state = 'accepted' AND accepted_at IS NOT NULL)
      OR (state <> 'accepted' AND accepted_at IS NULL)
    ),
    CHECK (
      (state NOT IN ('accepted', 'dead') AND ingest_status IS NULL
       AND canonical_event_id IS NULL AND ingest_reason_code IS NULL
       AND ingest_decided_at IS NULL)
      OR (
        state = 'accepted'
        AND (
          (ingest_status IN ('accepted', 'duplicate')
           AND canonical_event_id IS NOT NULL AND length(canonical_event_id) > 0
           AND ingest_decided_at IS NOT NULL)
          OR (ingest_status = 'legacy_unknown' AND canonical_event_id IS NULL
              AND ingest_reason_code IS NULL AND ingest_decided_at = accepted_at)
        )
      )
      OR (
        state = 'dead'
        AND (
          (ingest_status IN ('conflict', 'rejected_terminal')
           AND ingest_reason_code IS NOT NULL AND ingest_decided_at IS NOT NULL)
          OR (ingest_status IS NULL AND canonical_event_id IS NULL
              AND ingest_reason_code IS NULL AND ingest_decided_at IS NULL)
        )
      )
    )
  )
"""
_V1_OUTBOX_INDEX_SQL = {
  "idx_commercial_usage_outbox_ready": """
    CREATE INDEX IF NOT EXISTS idx_commercial_usage_outbox_ready
    ON commercial_usage_outbox(state, next_attempt_at, created_at)
  """,
  "idx_commercial_usage_outbox_lease": """
    CREATE INDEX IF NOT EXISTS idx_commercial_usage_outbox_lease
    ON commercial_usage_outbox(state, sending_lease_expires_at)
  """,
}
_OUTBOX_INDEX_SQL = {
  **_V1_OUTBOX_INDEX_SQL,
  "idx_commercial_usage_outbox_partition": """
    CREATE INDEX IF NOT EXISTS idx_commercial_usage_outbox_partition
    ON commercial_usage_outbox(environment, source_product, occurred_at, event_id)
  """,
  "idx_commercial_usage_outbox_session": """
    CREATE INDEX IF NOT EXISTS idx_commercial_usage_outbox_session
    ON commercial_usage_outbox(environment, request_id, session_id, event_id)
  """,
}
_REPORT_TABLE_SQL = """
  CREATE TABLE IF NOT EXISTS commercial_usage_reconciliation_reports (
    report_id TEXT PRIMARY KEY,
    environment TEXT NOT NULL,
    source_product TEXT NOT NULL,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    supersedes_report_sha256 TEXT,
    content_sha256 TEXT NOT NULL,
    report_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('match', 'mismatch', 'incomplete')),
    recorded_at TEXT NOT NULL,
    UNIQUE (environment, source_product, request_id, session_id, revision),
    UNIQUE (environment, source_product, request_id, session_id, content_sha256),
    UNIQUE (environment, source_product, request_id, session_id, report_sha256),
    CHECK (
      (revision = 1 AND supersedes_report_sha256 IS NULL)
      OR (revision > 1 AND supersedes_report_sha256 IS NOT NULL)
    )
  )
"""
_REPORT_INDEX_SQL = {
  "idx_commercial_usage_reconciliation_current": """
    CREATE INDEX IF NOT EXISTS idx_commercial_usage_reconciliation_current
    ON commercial_usage_reconciliation_reports(
      environment, source_product, request_id, session_id, revision DESC
    )
  """,
  "idx_commercial_usage_reconciliation_partition": """
    CREATE INDEX IF NOT EXISTS idx_commercial_usage_reconciliation_partition
    ON commercial_usage_reconciliation_reports(
      environment, source_product, recorded_at, report_id
    )
  """,
}
_REPORT_IMMUTABLE_TRIGGERS = {
  "trg_commercial_usage_reconciliation_linear_insert": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_reconciliation_linear_insert
  BEFORE INSERT ON commercial_usage_reconciliation_reports
  WHEN (
    NEW.revision = 1
    AND EXISTS (
      SELECT 1 FROM commercial_usage_reconciliation_reports
       WHERE environment = NEW.environment
         AND source_product = NEW.source_product
         AND request_id = NEW.request_id
         AND session_id = NEW.session_id
    )
  ) OR (
    NEW.revision > 1
    AND NOT EXISTS (
      SELECT 1 FROM commercial_usage_reconciliation_reports
       WHERE environment = NEW.environment
         AND source_product = NEW.source_product
         AND request_id = NEW.request_id
         AND session_id = NEW.session_id
         AND revision = NEW.revision - 1
         AND report_sha256 = NEW.supersedes_report_sha256
    )
  )
  BEGIN SELECT RAISE(ABORT, 'commercial usage reconciliation lineage must be linear'); END
  """,
  "trg_commercial_usage_reconciliation_no_update": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_reconciliation_no_update
  BEFORE UPDATE ON commercial_usage_reconciliation_reports
  BEGIN SELECT RAISE(ABORT, 'commercial usage reconciliation reports are append-only'); END
  """,
  "trg_commercial_usage_reconciliation_no_delete": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_reconciliation_no_delete
  BEFORE DELETE ON commercial_usage_reconciliation_reports
  BEGIN SELECT RAISE(ABORT, 'commercial usage reconciliation reports are append-only'); END
  """,
}
_RECONCILIATION_SHIPMENT_TABLE_SQL = """
  CREATE TABLE IF NOT EXISTS commercial_usage_reconciliation_shipments (
    report_id TEXT PRIMARY KEY REFERENCES commercial_usage_reconciliation_reports(report_id)
      ON DELETE RESTRICT,
    state TEXT NOT NULL CHECK (
      state IN ('held', 'pending', 'sending', 'accepted', 'retryable', 'dead')
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    next_attempt_at TEXT,
    sending_lease_token TEXT,
    sending_lease_expires_at TEXT,
    ingest_status TEXT CHECK (
      ingest_status IS NULL OR ingest_status IN ('accepted', 'duplicate', 'conflict')
    ),
    ingest_reason_code TEXT,
    ingest_decided_at TEXT,
    accepted_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    CHECK (
      (state = 'sending' AND sending_lease_token IS NOT NULL
       AND sending_lease_expires_at IS NOT NULL)
      OR (state <> 'sending' AND sending_lease_token IS NULL
          AND sending_lease_expires_at IS NULL)
    ),
    CHECK (
      (state = 'accepted' AND ingest_status IN ('accepted', 'duplicate')
       AND ingest_decided_at IS NOT NULL AND accepted_at IS NOT NULL)
      OR (state = 'dead' AND (
          (ingest_status = 'conflict' AND ingest_reason_code IS NOT NULL
           AND ingest_decided_at IS NOT NULL AND accepted_at IS NULL)
          OR (ingest_status IS NULL AND ingest_reason_code IS NULL
              AND ingest_decided_at IS NULL AND accepted_at IS NULL)
      ))
      OR (state NOT IN ('accepted', 'dead') AND ingest_status IS NULL
          AND ingest_reason_code IS NULL AND ingest_decided_at IS NULL
          AND accepted_at IS NULL)
    )
  )
"""
_RECONCILIATION_SHIPMENT_INDEX_SQL = {
  "idx_commercial_usage_reconciliation_shipments_ready": """
    CREATE INDEX IF NOT EXISTS idx_commercial_usage_reconciliation_shipments_ready
    ON commercial_usage_reconciliation_shipments(state, next_attempt_at, created_at)
  """,
  "idx_commercial_usage_reconciliation_shipments_lease": """
    CREATE INDEX IF NOT EXISTS idx_commercial_usage_reconciliation_shipments_lease
    ON commercial_usage_reconciliation_shipments(state, sending_lease_expires_at)
  """,
}
_RECONCILIATION_SHIPMENT_TRIGGERS = {
  "trg_commercial_usage_reconciliation_shipment_terminal_immutable": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_reconciliation_shipment_terminal_immutable
  BEFORE UPDATE ON commercial_usage_reconciliation_shipments
  WHEN OLD.state IN ('accepted', 'dead')
  BEGIN SELECT RAISE(ABORT, 'commercial usage reconciliation shipment is terminal'); END
  """,
  "trg_commercial_usage_reconciliation_shipment_no_delete": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_reconciliation_shipment_no_delete
  BEFORE DELETE ON commercial_usage_reconciliation_shipments
  BEGIN SELECT RAISE(ABORT, 'commercial usage reconciliation shipment is append-only'); END
  """,
}
_OUTBOX_GUARD_TRIGGERS = {
  "trg_commercial_usage_outbox_no_insert_legacy_acceptance": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_outbox_no_insert_legacy_acceptance
  BEFORE INSERT ON commercial_usage_outbox
  WHEN NEW.ingest_status = 'legacy_unknown'
  BEGIN SELECT RAISE(ABORT, 'legacy acceptance evidence is migration-only'); END
  """,
  "trg_commercial_usage_outbox_no_update_legacy_acceptance": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_outbox_no_update_legacy_acceptance
  BEFORE UPDATE OF ingest_status ON commercial_usage_outbox
  WHEN NEW.ingest_status = 'legacy_unknown'
       AND OLD.ingest_status IS NOT 'legacy_unknown'
  BEGIN SELECT RAISE(ABORT, 'legacy acceptance evidence is migration-only'); END
  """,
  "trg_commercial_usage_outbox_immutable_source": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_outbox_immutable_source
  BEFORE UPDATE ON commercial_usage_outbox
  WHEN NEW.event_id IS NOT OLD.event_id
    OR NEW.payload_sha256 IS NOT OLD.payload_sha256
    OR NEW.payload_json IS NOT OLD.payload_json
    OR NEW.environment IS NOT OLD.environment
    OR NEW.source_product IS NOT OLD.source_product
    OR NEW.occurred_at IS NOT OLD.occurred_at
    OR NEW.request_id IS NOT OLD.request_id
    OR NEW.session_id IS NOT OLD.session_id
    OR NEW.created_at IS NOT OLD.created_at
  BEGIN SELECT RAISE(ABORT, 'commercial usage source evidence is immutable'); END
  """,
  "trg_commercial_usage_outbox_terminal_immutable": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_outbox_terminal_immutable
  BEFORE UPDATE ON commercial_usage_outbox
  WHEN OLD.state IN ('accepted', 'dead')
  BEGIN SELECT RAISE(ABORT, 'commercial usage terminal evidence is immutable'); END
  """,
  "trg_commercial_usage_outbox_no_delete": """
  CREATE TRIGGER IF NOT EXISTS trg_commercial_usage_outbox_no_delete
  BEFORE DELETE ON commercial_usage_outbox
  BEGIN SELECT RAISE(ABORT, 'commercial usage source evidence is append-only'); END
  """,
}
OutboxState = Literal["pending", "sending", "accepted", "retryable", "dead"]
ReconciliationShipmentState = Literal[
  "held", "pending", "sending", "accepted", "retryable", "dead"
]


class CommercialUsageOutboxError(RuntimeError):
  """Base class for durable commercial outbox failures."""


class CommercialUsageOutboxConflict(CommercialUsageOutboxError):
  """An immutable source event ID was reused with different payload bytes."""


@dataclass(frozen=True)
class CommercialUsageOutboxRow:
  event_id: str
  payload_sha256: str
  payload_json: str
  environment: str
  source_product: str
  occurred_at: str
  request_id: str
  session_id: str
  state: OutboxState
  attempt_count: int
  next_attempt_at: str | None
  sending_lease_token: str | None
  sending_lease_expires_at: str | None
  ingest_status: str | None
  canonical_event_id: str | None
  ingest_reason_code: str | None
  ingest_decided_at: str | None
  accepted_at: str | None
  last_error: str | None
  created_at: str

  @property
  def payload(self) -> dict[str, Any]:
    value = json.loads(self.payload_json)
    if not isinstance(value, dict):
      raise CommercialUsageOutboxError("stored commercial payload is not an object")
    return value


@dataclass(frozen=True)
class CommercialUsageReconciliationRow:
  report_id: str
  environment: str
  source_product: str
  request_id: str
  session_id: str
  revision: int
  supersedes_report_sha256: str | None
  content_sha256: str
  report_sha256: str
  payload_json: str
  status: Literal["match", "mismatch", "incomplete"]
  recorded_at: str

  @property
  def payload(self) -> dict[str, Any]:
    value = json.loads(self.payload_json)
    if not isinstance(value, dict):
      raise CommercialUsageOutboxError("stored reconciliation report is not an object")
    return value


@dataclass(frozen=True)
class CommercialUsageReconciliationShipmentRow:
  report_id: str
  environment: str
  source_product: str
  request_id: str
  session_id: str
  revision: int
  report_sha256: str
  payload_json: str
  state: ReconciliationShipmentState
  attempt_count: int
  next_attempt_at: str | None
  sending_lease_token: str | None
  sending_lease_expires_at: str | None
  ingest_status: str | None
  ingest_reason_code: str | None
  ingest_decided_at: str | None
  accepted_at: str | None
  last_error: str | None
  created_at: str

  @property
  def envelope(self) -> dict[str, Any]:
    evidence = json.loads(self.payload_json)
    if not isinstance(evidence, dict):
      raise CommercialUsageOutboxError("stored reconciliation evidence is not an object")
    return {"report_sha256": self.report_sha256, "evidence": evidence}


def _utc_now() -> datetime:
  return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
  if value.tzinfo is None:
    raise ValueError("commercial outbox timestamps must be timezone-aware")
  return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _source_time_text(value: Any) -> str:
  try:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
  except (TypeError, ValueError) as exc:
    raise ValueError("commercial usage occurred_at is invalid") from exc
  if parsed.tzinfo is None:
    raise ValueError("commercial usage occurred_at must be timezone-aware")
  return _utc_text(parsed)


def _payload_json(payload: dict[str, Any]) -> str:
  return json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    allow_nan=False,
  )


def _normalized_ddl(sql: str | None) -> str:
  return " ".join((sql or "").lower().replace("if not exists ", "").split())


class CommercialUsageOutbox:
  """Atomic durable storage and fenced state transitions for usage delivery."""

  def __init__(
    self,
    path: str | Path,
    *,
    busy_timeout_ms: int = 5_000,
    synchronous: Literal["FULL", "EXTRA"] = "FULL",
    reconciliation_shipping_enabled: bool = False,
  ) -> None:
    if busy_timeout_ms <= 0:
      raise ValueError("commercial outbox busy_timeout_ms must be positive")
    if synchronous not in {"FULL", "EXTRA"}:
      raise ValueError("commercial outbox synchronous must be FULL or EXTRA")
    self.path = Path(path).expanduser()
    self._busy_timeout_ms = int(busy_timeout_ms)
    self._synchronous = synchronous
    self._reconciliation_shipping_enabled = bool(reconciliation_shipping_enabled)
    contracts_path = Path(__file__).parent / "contracts"
    self._reconciliation_validators = {
      version: Draft202012Validator(json.loads(
        (
          contracts_path
          / f"usage-reconciliation-v{version}"
          / "gateway-usage-reconciliation-evidence.schema.json"
        ).read_text(encoding="utf-8")
      ))
      for version in (1, 2)
    }
    self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    self._migrate()
    try:
      os.chmod(self.path, 0o600)
    except OSError:
      pass

  def _connect(self) -> sqlite3.Connection:
    connection = sqlite3.connect(
      self.path,
      timeout=self._busy_timeout_ms / 1_000,
      isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"PRAGMA synchronous={self._synchronous}")
    return connection

  def _migrate(self) -> None:
    with self._connect() as connection:
      version = int(connection.execute("PRAGMA user_version").fetchone()[0])
      if version > OUTBOX_SCHEMA_VERSION:
        raise CommercialUsageOutboxError(
          f"commercial outbox schema {version} is newer than supported {OUTBOX_SCHEMA_VERSION}"
        )
      connection.execute("BEGIN IMMEDIATE")
      try:
        stored = connection.execute(
          "SELECT 1 FROM sqlite_master WHERE type = 'table' "
          "AND name = 'commercial_usage_outbox'"
        ).fetchone()
        if version in {0, 1} and stored is not None:
          if not self._v1_schema_is_valid(
            connection, expected_version=version, allow_unversioned=version == 0
          ):
            raise CommercialUsageOutboxError(
              "commercial outbox table does not match version 1"
            )
          self._migrate_v1_to_v2(connection)
        elif version == 0:
          connection.execute(_OUTBOX_TABLE_SQL)
        elif version == 2:
          if not self._schema_is_valid(connection, expected_version=2):
            raise CommercialUsageOutboxError(
              "commercial outbox schema does not match version 2"
            )
        elif version == OUTBOX_SCHEMA_VERSION:
          if not self._schema_is_valid(
            connection, expected_version=OUTBOX_SCHEMA_VERSION
          ):
            raise CommercialUsageOutboxError(
              f"commercial outbox schema does not match version {OUTBOX_SCHEMA_VERSION}"
            )
          if self._reconciliation_shipping_enabled:
            connection.execute(
              "UPDATE commercial_usage_reconciliation_shipments "
              "SET state = 'pending' WHERE state = 'held'"
            )
          connection.execute("COMMIT")
          return
        elif version != OUTBOX_SCHEMA_VERSION:
          raise CommercialUsageOutboxError(
            f"commercial outbox schema {version} cannot be upgraded"
          )
        for index_sql in _OUTBOX_INDEX_SQL.values():
          connection.execute(index_sql)
        connection.execute(_REPORT_TABLE_SQL)
        for index_sql in _REPORT_INDEX_SQL.values():
          connection.execute(index_sql)
        for trigger_sql in _REPORT_IMMUTABLE_TRIGGERS.values():
          connection.execute(trigger_sql)
        for trigger_sql in _OUTBOX_GUARD_TRIGGERS.values():
          connection.execute(trigger_sql)
        connection.execute(_RECONCILIATION_SHIPMENT_TABLE_SQL)
        for index_sql in _RECONCILIATION_SHIPMENT_INDEX_SQL.values():
          connection.execute(index_sql)
        for trigger_sql in _RECONCILIATION_SHIPMENT_TRIGGERS.values():
          connection.execute(trigger_sql)
        connection.execute(
          """
          INSERT INTO commercial_usage_reconciliation_shipments (
            report_id, state, created_at
          )
          SELECT report_id, 'held', recorded_at
            FROM commercial_usage_reconciliation_reports
          """
        )
        if self._reconciliation_shipping_enabled:
          connection.execute(
            "UPDATE commercial_usage_reconciliation_shipments "
            "SET state = 'pending' WHERE state = 'held'"
          )
        connection.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION}")
        if not self._schema_is_valid(
          connection, expected_version=OUTBOX_SCHEMA_VERSION
        ):
          raise CommercialUsageOutboxError(
            f"commercial outbox schema does not match version {OUTBOX_SCHEMA_VERSION}"
          )
        connection.execute("COMMIT")
      except Exception:
        if connection.in_transaction:
          connection.execute("ROLLBACK")
        raise

  @staticmethod
  def _v1_schema_is_valid(
    connection: sqlite3.Connection,
    *,
    expected_version: int,
    allow_unversioned: bool,
  ) -> bool:
    actual_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if actual_version != expected_version and not (
      allow_unversioned and actual_version == 0
    ):
      return False
    table_info = tuple(
      (str(row["name"]), str(row["type"]), int(row["notnull"]),
       row["dflt_value"], int(row["pk"]))
      for row in connection.execute("PRAGMA table_info(commercial_usage_outbox)")
    )
    stored_table = connection.execute(
      "SELECT sql FROM sqlite_master WHERE type = 'table' "
      "AND name = 'commercial_usage_outbox'"
    ).fetchone()
    return (
      table_info == _V1_OUTBOX_TABLE_INFO
      and stored_table is not None
      and _normalized_ddl(stored_table["sql"]) == _normalized_ddl(_V1_OUTBOX_TABLE_SQL)
      and all(
        (
          (stored_index := connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (name,),
          ).fetchone()) is not None
          and _normalized_ddl(stored_index["sql"]) == _normalized_ddl(expected_sql)
        )
        for name, expected_sql in _V1_OUTBOX_INDEX_SQL.items()
      )
    )

  @staticmethod
  def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
      "SELECT * FROM commercial_usage_outbox ORDER BY created_at, event_id"
    ).fetchall()
    prepared: list[tuple[Any, ...]] = []
    for row in rows:
      try:
        payload = json.loads(str(row["payload_json"]))
      except (json.JSONDecodeError, UnicodeError) as exc:
        raise CommercialUsageOutboxError(
          "commercial outbox version 1 payload is corrupt"
        ) from exc
      if not isinstance(payload, dict):
        raise CommercialUsageOutboxError(
          "commercial outbox version 1 payload is not an object"
        )
      try:
        canonical_digest = canonical_usage_payload_sha256(payload)
      except (TypeError, ValueError) as exc:
        raise CommercialUsageOutboxError(
          "commercial outbox version 1 payload identity or digest is corrupt"
        ) from exc
      if (
        str(row["event_id"]) != str(payload.get("source_event_id") or "").strip()
        or str(row["payload_sha256"])
        != str(payload.get("source_payload_sha256") or "").strip()
        or str(row["payload_sha256"]) != canonical_digest
      ):
        raise CommercialUsageOutboxError(
          "commercial outbox version 1 payload identity or digest is corrupt"
        )
      facts = (
        str(payload.get("environment") or "").strip(),
        str(payload.get("source_product") or "").strip(),
        _source_time_text(payload.get("occurred_at")),
        str(payload.get("request_id") or "").strip(),
        str(payload.get("session_id") or "").strip(),
      )
      if not all(facts):
        raise CommercialUsageOutboxError(
          "commercial outbox version 1 payload lacks reconciliation identity"
        )
      accepted = str(row["state"]) == "accepted"
      prepared.append((
        row["event_id"], row["payload_sha256"], row["payload_json"], *facts,
        row["state"], row["attempt_count"], row["next_attempt_at"],
        row["sending_lease_token"], row["sending_lease_expires_at"],
        "legacy_unknown" if accepted else None, None, None,
        row["accepted_at"] if accepted else None, row["accepted_at"],
        row["last_error"], row["created_at"],
      ))
    connection.execute(
      "ALTER TABLE commercial_usage_outbox RENAME TO commercial_usage_outbox_v1"
    )
    connection.execute(_OUTBOX_TABLE_SQL)
    connection.executemany(
      """
      INSERT INTO commercial_usage_outbox (
        event_id, payload_sha256, payload_json, environment, source_product,
        occurred_at, request_id, session_id, state, attempt_count, next_attempt_at,
        sending_lease_token, sending_lease_expires_at, ingest_status,
        canonical_event_id, ingest_reason_code, ingest_decided_at, accepted_at,
        last_error, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """,
      prepared,
    )
    connection.execute("DROP TABLE commercial_usage_outbox_v1")

  @staticmethod
  def _schema_is_valid(
    connection: sqlite3.Connection,
    *,
    expected_version: int = OUTBOX_SCHEMA_VERSION,
    require_indexes: bool = True,
    allow_unversioned: bool = False,
  ) -> bool:
    actual_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if actual_version != expected_version and not (
      allow_unversioned and actual_version == 0
    ):
      return False
    table_info = tuple(
      (str(row["name"]), str(row["type"]), int(row["notnull"]), row["dflt_value"], int(row["pk"]))
      for row in connection.execute("PRAGMA table_info(commercial_usage_outbox)")
    )
    if table_info != _OUTBOX_TABLE_INFO:
      return False
    stored_table = connection.execute(
      "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'commercial_usage_outbox'"
    ).fetchone()
    if stored_table is None or _normalized_ddl(stored_table["sql"]) != _normalized_ddl(_OUTBOX_TABLE_SQL):
      return False
    if require_indexes:
      for name, expected_sql in _OUTBOX_INDEX_SQL.items():
        stored_index = connection.execute(
          "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)
        ).fetchone()
        if stored_index is None or _normalized_ddl(stored_index["sql"]) != _normalized_ddl(expected_sql):
          return False
      report_table = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'commercial_usage_reconciliation_reports'"
      ).fetchone()
      if (
        report_table is None
        or _normalized_ddl(report_table["sql"]) != _normalized_ddl(_REPORT_TABLE_SQL)
      ):
        return False
      for name, expected_sql in _REPORT_INDEX_SQL.items():
        report_index = connection.execute(
          "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)
        ).fetchone()
        if (
          report_index is None
          or _normalized_ddl(report_index["sql"]) != _normalized_ddl(expected_sql)
        ):
          return False
      for name, expected_sql in {
        **_REPORT_IMMUTABLE_TRIGGERS,
        **_OUTBOX_GUARD_TRIGGERS,
      }.items():
        stored_trigger = connection.execute(
          "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", (name,)
        ).fetchone()
        if (
          stored_trigger is None
          or _normalized_ddl(stored_trigger["sql"]) != _normalized_ddl(expected_sql)
        ):
          return False
      if expected_version >= 3:
        shipment_table = connection.execute(
          "SELECT sql FROM sqlite_master WHERE type = 'table' "
          "AND name = 'commercial_usage_reconciliation_shipments'"
        ).fetchone()
        if (
          shipment_table is None
          or _normalized_ddl(shipment_table["sql"])
          != _normalized_ddl(_RECONCILIATION_SHIPMENT_TABLE_SQL)
        ):
          return False
        for name, expected_sql in _RECONCILIATION_SHIPMENT_INDEX_SQL.items():
          stored_index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)
          ).fetchone()
          if (
            stored_index is None
            or _normalized_ddl(stored_index["sql"]) != _normalized_ddl(expected_sql)
          ):
            return False
        for name, expected_sql in _RECONCILIATION_SHIPMENT_TRIGGERS.items():
          stored_trigger = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", (name,)
          ).fetchone()
          if (
            stored_trigger is None
            or _normalized_ddl(stored_trigger["sql"]) != _normalized_ddl(expected_sql)
          ):
            return False
        coverage = connection.execute(
          """
          SELECT
            (SELECT COUNT(*) FROM commercial_usage_reconciliation_reports),
            (SELECT COUNT(*) FROM commercial_usage_reconciliation_shipments),
            (SELECT COUNT(*)
               FROM commercial_usage_reconciliation_shipments shipment
               LEFT JOIN commercial_usage_reconciliation_reports report
                 ON report.report_id = shipment.report_id
              WHERE report.report_id IS NULL)
          """
        ).fetchone()
        if int(coverage[0]) != int(coverage[1]) or int(coverage[2]) != 0:
          return False
      else:
        forbidden_names = {
          "commercial_usage_reconciliation_shipments",
          *_RECONCILIATION_SHIPMENT_INDEX_SQL,
          *_RECONCILIATION_SHIPMENT_TRIGGERS,
        }
        placeholders = ",".join("?" for _ in forbidden_names)
        if connection.execute(
          f"SELECT 1 FROM sqlite_master WHERE name IN ({placeholders}) LIMIT 1",
          tuple(sorted(forbidden_names)),
        ).fetchone() is not None:
          return False
    return True

  def enqueue_batch(
    self,
    payloads: list[dict[str, Any]],
    *,
    created_at: datetime | None = None,
  ) -> None:
    """Insert all derived economic deltas in one transaction."""
    if not payloads:
      raise ValueError("commercial outbox batch cannot be empty")
    prepared: list[tuple[str, str, str, str, str, str, str, str]] = []
    seen: dict[str, str] = {}
    for payload in payloads:
      if not isinstance(payload, dict):
        raise ValueError("commercial outbox payload must be an object")
      event_id = str(payload.get("source_event_id") or "").strip()
      digest = str(payload.get("source_payload_sha256") or "").strip()
      if not event_id or digest != canonical_usage_payload_sha256(payload):
        raise ValueError("commercial outbox payload identity or digest is invalid")
      facts = (
        str(payload.get("environment") or "").strip(),
        str(payload.get("source_product") or "").strip(),
        _source_time_text(payload.get("occurred_at")),
        str(payload.get("request_id") or "").strip(),
        str(payload.get("session_id") or "").strip(),
      )
      if not all(facts):
        raise ValueError("commercial outbox payload reconciliation identity is invalid")
      payload_json = _payload_json(payload)
      previous = seen.get(event_id)
      if previous is not None and previous != digest:
        raise CommercialUsageOutboxConflict(
          f"conflicting commercial usage event inside batch: {event_id}"
        )
      seen[event_id] = digest
      prepared.append((event_id, digest, payload_json, *facts))
    created_text = _utc_text(created_at or _utc_now())

    with self._connect() as connection:
      connection.execute("BEGIN IMMEDIATE")
      try:
        for (
          event_id, digest, payload_json, environment, source_product,
          occurred_at, request_id, session_id,
        ) in prepared:
          existing = connection.execute(
            "SELECT payload_sha256, payload_json FROM commercial_usage_outbox WHERE event_id = ?",
            (event_id,),
          ).fetchone()
          if existing is not None:
            if existing["payload_sha256"] != digest:
              raise CommercialUsageOutboxConflict(
                f"conflicting commercial usage event already stored: {event_id}"
              )
            continue
          connection.execute(
            """
            INSERT INTO commercial_usage_outbox (
              event_id, payload_sha256, payload_json, environment, source_product,
              occurred_at, request_id, session_id, state, attempt_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?)
            """,
            (
              event_id, digest, payload_json, environment, source_product,
              occurred_at, request_id, session_id, created_text,
            ),
          )
        connection.execute("COMMIT")
      except Exception:
        if connection.in_transaction:
          connection.execute("ROLLBACK")
        raise

  def producer(self, *, claim: Any, lineage: Any, enabled: bool = True) -> Any:
    """Construct only a disabled producer; enabled work requires resilient bootstrap."""
    from .commercial_usage import CommercialUsageProducer

    if enabled:
      raise CommercialUsageOutboxError(
        "enabled commercial usage must use CommercialUsageDurability.producer"
      )
    return CommercialUsageProducer(
      enabled=enabled,
      claim=claim,
      lineage=lineage,
      sink=self.enqueue_batch,
    )

  def lease_batch(
    self,
    *,
    limit: int,
    lease_for: timedelta,
    now: datetime | None = None,
  ) -> list[CommercialUsageOutboxRow]:
    if limit <= 0 or lease_for.total_seconds() <= 0:
      raise ValueError("commercial outbox lease limit and duration must be positive")
    current = now or _utc_now()
    current_text = _utc_text(current)
    expires_text = _utc_text(current + lease_for)
    lease_token = str(uuid4())
    with self._connect() as connection:
      connection.execute("BEGIN IMMEDIATE")
      try:
        connection.execute(
          """
          UPDATE commercial_usage_outbox
          SET state = 'retryable', sending_lease_token = NULL,
              sending_lease_expires_at = NULL, next_attempt_at = ?,
              last_error = 'sending lease expired'
          WHERE state = 'sending' AND sending_lease_expires_at <= ?
          """,
          (current_text, current_text),
        )
        rows = connection.execute(
          """
          SELECT event_id FROM commercial_usage_outbox
          WHERE state IN ('pending', 'retryable')
            AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
          ORDER BY created_at, event_id
          LIMIT ?
          """,
          (current_text, limit),
        ).fetchall()
        event_ids = [str(row["event_id"]) for row in rows]
        for event_id in event_ids:
          connection.execute(
            """
            UPDATE commercial_usage_outbox
            SET state = 'sending', attempt_count = attempt_count + 1,
                sending_lease_token = ?, sending_lease_expires_at = ?,
                next_attempt_at = NULL
            WHERE event_id = ? AND state IN ('pending', 'retryable')
            """,
            (lease_token, expires_text, event_id),
          )
        leased = self._select_rows(connection, event_ids)
        connection.execute("COMMIT")
        return leased
      except Exception:
        if connection.in_transaction:
          connection.execute("ROLLBACK")
        raise

  def mark_accepted(
    self,
    event_id: str,
    lease_token: str,
    *,
    ingest_status: Literal["accepted", "duplicate"],
    canonical_event_id: str,
    reason_code: str | None = None,
    accepted_at: datetime | None = None,
  ) -> bool:
    canonical_id = canonical_event_id.strip()
    if not canonical_id:
      raise ValueError("accepted usage requires canonical event identity")
    decided_at = _utc_text(accepted_at or _utc_now())
    return self._finish_sending(
      event_id,
      lease_token,
      state="accepted",
      ingest_status=ingest_status,
      canonical_event_id=canonical_id,
      ingest_reason_code=reason_code,
      ingest_decided_at=decided_at,
      accepted_at=decided_at,
    )

  def mark_retryable(
    self,
    event_id: str,
    lease_token: str,
    *,
    next_attempt_at: datetime,
    error: str,
  ) -> bool:
    return self._finish_sending(
      event_id,
      lease_token,
      state="retryable",
      next_attempt_at=_utc_text(next_attempt_at),
      error=error,
    )

  def mark_dead(
    self,
    event_id: str,
    lease_token: str,
    *,
    error: str,
    ingest_status: Literal["conflict", "rejected_terminal"] | None = None,
    canonical_event_id: str | None = None,
    reason_code: str | None = None,
    decided_at: datetime | None = None,
  ) -> bool:
    if ingest_status is not None and not str(reason_code or "").strip():
      raise ValueError("terminal ingest decision requires reason code")
    return self._finish_sending(
      event_id,
      lease_token,
      state="dead",
      error=error,
      ingest_status=ingest_status,
      canonical_event_id=(canonical_event_id or "").strip() or None,
      ingest_reason_code=(reason_code or "").strip() or None,
      ingest_decided_at=(
        _utc_text(decided_at or _utc_now()) if ingest_status is not None else None
      ),
    )

  def _finish_sending(
    self,
    event_id: str,
    lease_token: str,
    *,
    state: Literal["accepted", "retryable", "dead"],
    accepted_at: str | None = None,
    next_attempt_at: str | None = None,
    error: str | None = None,
    ingest_status: str | None = None,
    canonical_event_id: str | None = None,
    ingest_reason_code: str | None = None,
    ingest_decided_at: str | None = None,
  ) -> bool:
    normalized_error = error[:4_096] if error is not None else None
    with self._connect() as connection:
      cursor = connection.execute(
        """
        UPDATE commercial_usage_outbox
        SET state = ?, sending_lease_token = NULL, sending_lease_expires_at = NULL,
            ingest_status = ?, canonical_event_id = ?, ingest_reason_code = ?,
            ingest_decided_at = ?, accepted_at = ?, next_attempt_at = ?, last_error = ?
        WHERE event_id = ? AND state = 'sending' AND sending_lease_token = ?
        """,
        (
          state, ingest_status, canonical_event_id, ingest_reason_code,
          ingest_decided_at, accepted_at, next_attempt_at, normalized_error,
          event_id, lease_token,
        ),
      )
      return cursor.rowcount == 1

  def get(self, event_id: str) -> CommercialUsageOutboxRow | None:
    with self._connect() as connection:
      row = connection.execute(
        "SELECT * FROM commercial_usage_outbox WHERE event_id = ?", (event_id,)
      ).fetchone()
      return self._row(row) if row is not None else None

  def source_partition(
    self,
    *,
    environment: str,
    source_product: str,
    occurred_from: datetime,
    occurred_until: datetime,
  ) -> tuple[CommercialUsageOutboxRow, ...]:
    if (
      occurred_from.tzinfo is None
      or occurred_until.tzinfo is None
      or occurred_until <= occurred_from
    ):
      raise ValueError("commercial usage source partition is invalid")
    with self._connect() as connection:
      rows = connection.execute(
        """
        SELECT * FROM commercial_usage_outbox
         WHERE environment = ? AND source_product = ?
           AND occurred_at >= ? AND occurred_at < ?
         ORDER BY occurred_at, event_id
        """,
        (
          environment,
          source_product,
          _utc_text(occurred_from),
          _utc_text(occurred_until),
        ),
      ).fetchall()
      return tuple(self._row(row) for row in rows)

  @staticmethod
  def _validate_reconciliation_consistency(payload: dict[str, Any]) -> None:
    lines = payload.get("event_lines")
    observed_ids = payload.get("observed_source_event_ids")
    if (
      not isinstance(lines, list)
      or not isinstance(observed_ids, list)
      or any(not isinstance(line, dict) for line in lines)
    ):
      raise ValueError("commercial reconciliation report source manifest is invalid")
    line_ids = [str(line.get("source_event_id") or "") for line in lines]
    identity_list_fields = (
      "missing_source_event_ids", "conflicting_source_event_ids",
      "late_source_event_ids", "observed_source_event_ids",
      "summary_usage_event_ids",
    )
    if any(
      not isinstance(payload.get(field), list)
      or payload[field] != sorted(set(str(value) for value in payload[field]))
      for field in identity_list_fields
    ):
      raise ValueError("commercial reconciliation identity lists are not canonical")
    provider_lines = [line for line in lines if line.get("event_kind") == "provider_call"]
    durable_provider_ids = {
      str(line.get("source_event_id") or "")
      for line in provider_lines if line.get("durability") != "lost"
    }
    summary_ids = {str(value) for value in payload.get("summary_usage_event_ids") or []}
    missing_ids = sorted(summary_ids - durable_provider_ids)
    late_ids = sorted(
      str(line.get("source_event_id") or "") for line in lines if line.get("late") is True
    )
    conflicts = sorted(str(value) for value in payload.get("conflicting_source_event_ids") or [])
    expected_count = int(payload.get("expected_provider_call_count") or 0)
    provider_count = len(provider_lines)
    durable_provider_count = len(durable_provider_ids)
    expected_missing_count = max(
      len(missing_ids), expected_count - durable_provider_count, 0
    )

    def integer_total(field: str) -> int:
      return sum(int(line.get(field) or 0) for line in lines)

    def decimal_total(field: str) -> Decimal:
      return sum(
        (Decimal(str(line[field])) for line in lines if line.get(field) is not None),
        Decimal("0"),
      )

    commercial_input = integer_total("uncached_input_tokens")
    commercial_output = integer_total("billable_output_tokens")
    commercial_cache_read = integer_total("cache_read_tokens")
    commercial_cache_write = integer_total("cache_write_tokens")
    commercial_estimate = decimal_total("producer_estimated_cost_usd")
    summary_estimate = Decimal(str(payload.get("summary_estimate_usd") or 0))
    estimate_delta = commercial_estimate - summary_estimate
    expected_values = {
      "commercial_event_count": len(lines),
      "provider_call_event_count": provider_count,
      "durable_provider_call_event_count": durable_provider_count,
      "separate_unit_event_count": sum(
        line.get("event_kind") == "separate_unit" for line in lines
      ),
      "provider_call_count_delta": provider_count - expected_count,
      "missing_event_id_count": expected_missing_count,
      "emergency_spooled_event_count": sum(
        line.get("durability") == "emergency_spool" for line in lines
      ),
      "durability_lost_event_count": sum(
        line.get("durability") == "lost" for line in lines
      ),
      "conflicting_event_id_count": len(conflicts),
      "late_event_count": len(late_ids),
      "commercial_input_tokens": commercial_input,
      "input_token_delta": commercial_input - int(payload.get("summary_input_tokens") or 0),
      "commercial_output_tokens": commercial_output,
      "output_token_delta": commercial_output - int(payload.get("summary_output_tokens") or 0),
      "commercial_cache_read_tokens": commercial_cache_read,
      "cache_read_token_delta": (
        commercial_cache_read - int(payload.get("summary_cache_read_tokens") or 0)
      ),
      "commercial_cache_write_tokens": commercial_cache_write,
      "cache_write_token_delta": (
        commercial_cache_write - int(payload.get("summary_cache_write_tokens") or 0)
      ),
      "reasoning_tokens_observed": integer_total("reasoning_tokens_observed"),
    }
    if (
      not all(line_ids)
      or line_ids != sorted(set(line_ids))
      or line_ids != observed_ids
      or missing_ids != payload.get("missing_source_event_ids")
      or late_ids != payload.get("late_source_event_ids")
      or not set(conflicts).issubset(line_ids)
      or any(int(payload.get(field) or 0) != value for field, value in expected_values.items())
      or Decimal(str(payload.get("provider_units") or 0))
         != decimal_total("provider_units")
      or Decimal(str(payload.get("commercial_producer_estimate_usd") or 0))
         != commercial_estimate
      or Decimal(str(payload.get("estimate_delta_usd") or 0)) != estimate_delta
      or not math.isfinite(float(payload.get("summary_started_at") or 0))
      or not math.isfinite(float(payload.get("summary_ended_at") or 0))
      or float(payload.get("summary_started_at") or 0)
         > float(payload.get("summary_ended_at") or 0)
    ):
      raise ValueError("commercial reconciliation report source manifest is inconsistent")
    lineage_fields = {
      "workflow_attempt_group_id", "workflow_attempt_number",
      "retry_of_workflow_run_id", "workflow_attempt_kind",
      "work_authorization_id",
    }
    present_lineage = lineage_fields.intersection(payload)
    if present_lineage and present_lineage != lineage_fields:
      raise ValueError("commercial reconciliation attempt lineage is incomplete")
    if present_lineage:
      expected_lineage = tuple(payload.get(field) for field in sorted(lineage_fields))
      if any(
        line.get("source_schema_version") != 2
        or tuple(line.get(field) for field in sorted(lineage_fields))
           != expected_lineage
        for line in lines
      ):
        raise ValueError("commercial reconciliation attempt lineage is inconsistent")
    is_match = (
      not any(
        int(payload.get(field) or 0)
        for field in (
          "input_token_delta", "output_token_delta", "cache_read_token_delta",
          "cache_write_token_delta", "missing_event_id_count",
          "provider_call_count_delta", "conflicting_event_id_count",
          "durability_lost_event_count", "late_event_count",
        )
      )
      and abs(estimate_delta) <= Decimal("0.00000001")
      and durable_provider_ids == summary_ids
    )
    expected_status = (
      "incomplete"
      if not payload.get("drain_complete") or int(payload.get("in_flight_task_count") or 0)
      else "match" if is_match else "mismatch"
    )
    if payload.get("status") != expected_status:
      raise ValueError("commercial reconciliation report status is inconsistent")

  def record_reconciliation_report(
    self,
    report: Any,
    *,
    recorded_at: datetime | None = None,
  ) -> tuple[CommercialUsageReconciliationRow, bool]:
    raw_payload = report.as_dict() if hasattr(report, "as_dict") else report
    if not isinstance(raw_payload, dict):
      raise ValueError("commercial reconciliation report must be an object")
    payload = dict(raw_payload)
    envelope_fields = {
      "evidence_schema_version", "evidence_revision",
      "supersedes_report_sha256", "content_sha256",
    }
    supplied_envelope = envelope_fields.intersection(payload)
    supplied_content_sha256: str | None = None
    if supplied_envelope:
      if supplied_envelope != envelope_fields:
        raise ValueError("commercial reconciliation evidence envelope is incomplete")
      evidence_schema_version = payload.get("evidence_schema_version")
      validator = self._reconciliation_validators.get(evidence_schema_version)
      if validator is None:
        raise ValueError("commercial reconciliation evidence version is unsupported")
      errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: list(error.path),
      )
      if errors:
        raise ValueError(
          f"commercial reconciliation evidence is invalid: {errors[0].message}"
        )
      supplied_content_sha256 = str(payload["content_sha256"])
      for field in envelope_fields:
        payload.pop(field)
    lineage_fields = {
      "workflow_attempt_group_id", "workflow_attempt_number",
      "retry_of_workflow_run_id", "workflow_attempt_kind",
      "work_authorization_id",
    }
    present_lineage = lineage_fields.intersection(payload)
    if present_lineage and present_lineage != lineage_fields:
      raise ValueError("commercial reconciliation attempt lineage is incomplete")
    evidence_schema_version = 2 if present_lineage else 1
    request_id = str(payload.get("request_id") or "").strip()
    session_id = str(payload.get("session_id") or "").strip()
    environment = str(payload.get("environment") or "").strip()
    source_product = str(payload.get("source_product") or "").strip()
    status = str(payload.get("status") or "").strip()
    if (
        not environment
        or not source_product
        or not request_id
      or not session_id
      or status not in {"match", "mismatch", "incomplete"}
      or payload.get("summary_emitted_as_cost_event") is not False
    ):
      raise ValueError("commercial reconciliation report identity/status is invalid")
    self._validate_reconciliation_consistency(payload)
    content_json = _payload_json(payload)
    content_sha256 = "sha256:" + hashlib.sha256(content_json.encode("utf-8")).hexdigest()
    if (
      supplied_content_sha256 is not None
      and supplied_content_sha256 != content_sha256
    ):
      raise ValueError("commercial reconciliation evidence content digest is invalid")
    recorded_text = _utc_text(recorded_at or _utc_now())
    with self._connect() as connection:
      connection.execute("BEGIN IMMEDIATE")
      try:
        existing = connection.execute(
          """
            SELECT * FROM commercial_usage_reconciliation_reports
             WHERE environment = ? AND source_product = ?
               AND request_id = ? AND session_id = ? AND content_sha256 = ?
            """,
            (environment, source_product, request_id, session_id, content_sha256),
        ).fetchone()
        if existing is not None:
          connection.execute("COMMIT")
          return self._report_row(existing), True
        prior = connection.execute(
          """
          SELECT revision, report_sha256, payload_json
             FROM commercial_usage_reconciliation_reports
             WHERE environment = ? AND source_product = ?
               AND request_id = ? AND session_id = ?
             ORDER BY revision DESC LIMIT 1
            """,
            (environment, source_product, request_id, session_id),
        ).fetchone()
        revision = 1 if prior is None else int(prior["revision"]) + 1
        supersedes = None if prior is None else str(prior["report_sha256"])
        if (
          prior is not None
          and int(json.loads(str(prior["payload_json"]))["evidence_schema_version"])
              != evidence_schema_version
        ):
          raise ValueError(
            "commercial reconciliation evidence version cannot change within a revision chain"
          )
        if prior is not None and evidence_schema_version == 2:
          prior_payload = json.loads(str(prior["payload_json"]))
          attempt_fields = (
            "workflow_attempt_group_id", "workflow_attempt_number",
            "retry_of_workflow_run_id", "workflow_attempt_kind",
            "work_authorization_id",
          )
          if any(
            prior_payload.get(field) != payload.get(field)
            for field in attempt_fields
          ):
            raise ValueError(
              "commercial reconciliation attempt identity cannot change within a revision chain"
            )
        evidence = dict(payload)
        evidence.update({
          "evidence_schema_version": evidence_schema_version,
          "evidence_revision": revision,
          "supersedes_report_sha256": supersedes,
          "content_sha256": content_sha256,
        })
        errors = sorted(
          self._reconciliation_validators[evidence_schema_version].iter_errors(evidence),
          key=lambda error: list(error.path),
        )
        if errors:
          raise ValueError(
            f"commercial reconciliation evidence is invalid: {errors[0].message}"
          )
        payload_json = _payload_json(evidence)
        report_sha256 = "sha256:" + hashlib.sha256(
          payload_json.encode("utf-8")
        ).hexdigest()
        report_id = str(uuid5(
          NAMESPACE_URL,
          f"commercial-usage-reconciliation:{request_id}:{session_id}:{report_sha256}",
        ))
        connection.execute(
          """
          INSERT INTO commercial_usage_reconciliation_reports (
            report_id, environment, source_product, request_id, session_id, revision,
            supersedes_report_sha256, content_sha256, report_sha256,
            payload_json, status, recorded_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            report_id, environment, source_product, request_id, session_id,
            revision, supersedes,
            content_sha256, report_sha256, payload_json, status, recorded_text,
          ),
        )
        connection.execute(
          """
          INSERT INTO commercial_usage_reconciliation_shipments (
            report_id, state, created_at
          ) VALUES (?, ?, ?)
          """,
          (
            report_id,
            "pending" if self._reconciliation_shipping_enabled else "held",
            recorded_text,
          ),
        )
        row = connection.execute(
          "SELECT * FROM commercial_usage_reconciliation_reports WHERE report_id = ?",
          (report_id,),
        ).fetchone()
        connection.execute("COMMIT")
        return self._report_row(row), False
      except Exception:
        if connection.in_transaction:
          connection.execute("ROLLBACK")
        raise

  def current_reconciliation_report(
    self,
    *,
    environment: str,
    source_product: str,
    request_id: str,
    session_id: str,
  ) -> CommercialUsageReconciliationRow | None:
    with self._connect() as connection:
      row = connection.execute(
        """
        SELECT * FROM commercial_usage_reconciliation_reports
         WHERE environment = ? AND source_product = ?
           AND request_id = ? AND session_id = ?
         ORDER BY revision DESC LIMIT 1
        """,
        (environment, source_product, request_id, session_id),
      ).fetchone()
      return None if row is None else self._report_row(row)

  def reconciliation_reports(
    self,
    *,
    environment: str,
    source_product: str,
    recorded_from: datetime,
    recorded_until: datetime,
  ) -> tuple[CommercialUsageReconciliationRow, ...]:
    if (
      recorded_from.tzinfo is None
      or recorded_until.tzinfo is None
      or recorded_until <= recorded_from
    ):
      raise ValueError("commercial reconciliation report partition is invalid")
    with self._connect() as connection:
      rows = connection.execute(
        """
        SELECT * FROM commercial_usage_reconciliation_reports
         WHERE environment = ? AND source_product = ?
           AND recorded_at >= ? AND recorded_at < ?
         ORDER BY recorded_at, report_id
        """,
        (
          environment,
          source_product,
          _utc_text(recorded_from),
          _utc_text(recorded_until),
        ),
      ).fetchall()
      return tuple(self._report_row(row) for row in rows)

  def lease_reconciliation_batch(
    self,
    *,
    limit: int,
    lease_for: timedelta,
    now: datetime | None = None,
  ) -> list[CommercialUsageReconciliationShipmentRow]:
    if limit <= 0 or lease_for <= timedelta(0):
      raise ValueError("commercial reconciliation shipment lease is invalid")
    current = now or _utc_now()
    current_text = _utc_text(current)
    expires_text = _utc_text(current + lease_for)
    with self._connect() as connection:
      connection.execute("BEGIN IMMEDIATE")
      try:
        connection.execute(
          """
          UPDATE commercial_usage_reconciliation_shipments
             SET state = 'retryable', sending_lease_token = NULL,
                 sending_lease_expires_at = NULL, next_attempt_at = ?,
                 last_error = 'sending lease expired'
           WHERE state = 'sending' AND sending_lease_expires_at <= ?
          """,
          (current_text, current_text),
        )
        candidates = connection.execute(
          """
          SELECT shipment.report_id
            FROM commercial_usage_reconciliation_shipments shipment
            JOIN commercial_usage_reconciliation_reports report
              ON report.report_id = shipment.report_id
           WHERE shipment.state IN ('pending', 'retryable')
             AND (shipment.next_attempt_at IS NULL OR shipment.next_attempt_at <= ?)
             AND NOT EXISTS (
               SELECT 1
                 FROM commercial_usage_reconciliation_reports predecessor_report
                 JOIN commercial_usage_reconciliation_shipments predecessor
                   ON predecessor.report_id = predecessor_report.report_id
                WHERE predecessor_report.environment = report.environment
                  AND predecessor_report.source_product = report.source_product
                  AND predecessor_report.request_id = report.request_id
                  AND predecessor_report.session_id = report.session_id
                  AND predecessor_report.revision < report.revision
                  AND predecessor.state <> 'accepted'
             )
           ORDER BY shipment.created_at, shipment.report_id
           LIMIT ?
          """,
          (current_text, limit),
        ).fetchall()
        leased_ids = []
        for candidate in candidates:
          token = str(uuid4())
          cursor = connection.execute(
            """
            UPDATE commercial_usage_reconciliation_shipments
               SET state = 'sending', attempt_count = attempt_count + 1,
                   sending_lease_token = ?, sending_lease_expires_at = ?,
                   next_attempt_at = NULL
             WHERE report_id = ? AND state IN ('pending', 'retryable')
            """,
            (token, expires_text, candidate["report_id"]),
          )
          if cursor.rowcount == 1:
            leased_ids.append(str(candidate["report_id"]))
        rows = self._select_reconciliation_shipments(connection, leased_ids)
        connection.execute("COMMIT")
        return rows
      except Exception:
        if connection.in_transaction:
          connection.execute("ROLLBACK")
        raise

  def enable_reconciliation_shipping(self) -> int:
    """Explicitly release held evidence after a reconciliation shipper is configured."""
    self._reconciliation_shipping_enabled = True
    with self._connect() as connection:
      cursor = connection.execute(
        "UPDATE commercial_usage_reconciliation_shipments "
        "SET state = 'pending' WHERE state = 'held'"
      )
      return cursor.rowcount

  def mark_reconciliation_accepted(
    self,
    report_id: str,
    lease_token: str,
    *,
    ingest_status: Literal["accepted", "duplicate"],
    reason_code: str | None = None,
    accepted_at: datetime | None = None,
  ) -> bool:
    decided = _utc_text(accepted_at or _utc_now())
    return self._finish_reconciliation_shipment(
      report_id,
      lease_token,
      state="accepted",
      ingest_status=ingest_status,
      ingest_reason_code=(reason_code or "").strip() or None,
      ingest_decided_at=decided,
      accepted_at=decided,
    )

  def mark_reconciliation_retryable(
    self,
    report_id: str,
    lease_token: str,
    *,
    next_attempt_at: datetime,
    error: str,
  ) -> bool:
    return self._finish_reconciliation_shipment(
      report_id,
      lease_token,
      state="retryable",
      next_attempt_at=_utc_text(next_attempt_at),
      error=error,
    )

  def mark_reconciliation_dead(
    self,
    report_id: str,
    lease_token: str,
    *,
    error: str,
    ingest_status: Literal["conflict"] | None = None,
    reason_code: str | None = None,
    decided_at: datetime | None = None,
  ) -> bool:
    if ingest_status is not None and not str(reason_code or "").strip():
      raise ValueError("terminal reconciliation decision requires reason code")
    return self._finish_reconciliation_shipment(
      report_id,
      lease_token,
      state="dead",
      ingest_status=ingest_status,
      ingest_reason_code=(reason_code or "").strip() or None,
      ingest_decided_at=(
        _utc_text(decided_at or _utc_now()) if ingest_status is not None else None
      ),
      error=error,
    )

  def _finish_reconciliation_shipment(
    self,
    report_id: str,
    lease_token: str,
    *,
    state: Literal["accepted", "retryable", "dead"],
    ingest_status: str | None = None,
    ingest_reason_code: str | None = None,
    ingest_decided_at: str | None = None,
    accepted_at: str | None = None,
    next_attempt_at: str | None = None,
    error: str | None = None,
  ) -> bool:
    with self._connect() as connection:
      cursor = connection.execute(
        """
        UPDATE commercial_usage_reconciliation_shipments
           SET state = ?, sending_lease_token = NULL,
               sending_lease_expires_at = NULL, ingest_status = ?,
               ingest_reason_code = ?, ingest_decided_at = ?, accepted_at = ?,
               next_attempt_at = ?, last_error = ?
         WHERE report_id = ? AND state = 'sending' AND sending_lease_token = ?
        """,
        (
          state, ingest_status, ingest_reason_code, ingest_decided_at, accepted_at,
          next_attempt_at, error[:4_096] if error is not None else None,
          report_id, lease_token,
        ),
      )
      return cursor.rowcount == 1

  def health(self, *, now: datetime | None = None) -> dict[str, Any]:
    current = now or _utc_now()
    with self._connect() as connection:
      quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
      journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
      synchronous = int(connection.execute("PRAGMA synchronous").fetchone()[0])
      schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
      schema_valid = self._schema_is_valid(connection)
      counts: dict[str, int] = {}
      oldest = None
      if schema_valid:
        counts = {
          str(row["state"]): int(row["count"])
          for row in connection.execute(
            "SELECT state, COUNT(*) AS count FROM commercial_usage_outbox GROUP BY state"
          )
        }
        oldest = connection.execute(
          """
          SELECT MIN(created_at) AS oldest FROM commercial_usage_outbox
          WHERE state IN ('pending', 'sending', 'retryable')
          """
        ).fetchone()["oldest"]
        report_counts = {
          str(row["status"]): int(row["count"])
          for row in connection.execute(
            """
            SELECT status, COUNT(*) AS count
              FROM commercial_usage_reconciliation_reports GROUP BY status
            """
          )
        }
        reconciliation_shipment_counts = {
          str(row["state"]): int(row["count"])
          for row in connection.execute(
            """
            SELECT state, COUNT(*) AS count
              FROM commercial_usage_reconciliation_shipments GROUP BY state
            """
          )
        }
      else:
        report_counts = {}
        reconciliation_shipment_counts = {}
      oldest_age = None
      if oldest:
        parsed = datetime.fromisoformat(str(oldest).replace("Z", "+00:00"))
        oldest_age = max(0.0, (current.astimezone(timezone.utc) - parsed).total_seconds())
    database_bytes = self.path.stat().st_size if self.path.exists() else 0
    wal_path = Path(f"{self.path}-wal")
    wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0
    return {
      "ok": (
        quick_check == "ok"
        and journal_mode == "wal"
        and synchronous >= 2
        and schema_version == OUTBOX_SCHEMA_VERSION
        and schema_valid
      ),
      "schema_version": schema_version,
      "schema_valid": schema_valid,
      "journal_mode": journal_mode,
      "synchronous": synchronous,
      "quick_check": quick_check,
      "counts": counts,
      "backlog_count": sum(counts.get(state, 0) for state in ("pending", "sending", "retryable")),
      "reconciliation_report_counts": report_counts,
      "reconciliation_shipment_counts": reconciliation_shipment_counts,
      "reconciliation_shipment_backlog_count": sum(
        reconciliation_shipment_counts.get(state, 0)
        for state in ("pending", "sending", "retryable")
      ),
      "oldest_backlog_age_seconds": oldest_age,
      "database_bytes": database_bytes,
      "wal_bytes": wal_bytes,
      "storage_bytes": database_bytes + wal_bytes,
    }

  def _select_rows(
    self, connection: sqlite3.Connection, event_ids: Iterable[str]
  ) -> list[CommercialUsageOutboxRow]:
    ids = list(event_ids)
    if not ids:
      return []
    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
      f"SELECT * FROM commercial_usage_outbox WHERE event_id IN ({placeholders}) ORDER BY created_at, event_id",
      ids,
    ).fetchall()
    return [self._row(row) for row in rows]

  def _select_reconciliation_shipments(
    self, connection: sqlite3.Connection, report_ids: Iterable[str]
  ) -> list[CommercialUsageReconciliationShipmentRow]:
    ids = list(report_ids)
    if not ids:
      return []
    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
      f"""
      SELECT report.report_id, report.environment, report.source_product,
             report.request_id, report.session_id, report.revision,
             report.report_sha256, report.payload_json,
             shipment.state, shipment.attempt_count, shipment.next_attempt_at,
             shipment.sending_lease_token, shipment.sending_lease_expires_at,
             shipment.ingest_status, shipment.ingest_reason_code,
             shipment.ingest_decided_at, shipment.accepted_at,
             shipment.last_error, shipment.created_at
        FROM commercial_usage_reconciliation_shipments shipment
        JOIN commercial_usage_reconciliation_reports report
          ON report.report_id = shipment.report_id
       WHERE shipment.report_id IN ({placeholders})
       ORDER BY shipment.created_at, shipment.report_id
      """,
      ids,
    ).fetchall()
    return [self._reconciliation_shipment_row(row) for row in rows]

  @staticmethod
  def _row(row: sqlite3.Row) -> CommercialUsageOutboxRow:
    return CommercialUsageOutboxRow(**dict(row))

  @staticmethod
  def _report_row(row: sqlite3.Row) -> CommercialUsageReconciliationRow:
    return CommercialUsageReconciliationRow(**dict(row))

  @staticmethod
  def _reconciliation_shipment_row(
    row: sqlite3.Row,
  ) -> CommercialUsageReconciliationShipmentRow:
    return CommercialUsageReconciliationShipmentRow(**dict(row))
