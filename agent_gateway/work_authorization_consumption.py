"""Durable one-time attachment of verified commercial work authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import Callable, Literal
from uuid import UUID

from .commercial_work_authorization import VerifiedWorkAuthorization


_TABLE_SQL = """
  CREATE TABLE IF NOT EXISTS commercial_work_authorization_consumptions (
    authorization_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    token_sha256 TEXT NOT NULL UNIQUE CHECK (
      length(token_sha256) = 71
      AND substr(token_sha256, 1, 7) = 'sha256:'
      AND substr(token_sha256, 8) NOT GLOB '*[^0-9a-f]*'
    ),
    content_sha256 TEXT NOT NULL CHECK (
      length(content_sha256) = 71
      AND substr(content_sha256, 1, 7) = 'sha256:'
      AND substr(content_sha256, 8) NOT GLOB '*[^0-9a-f]*'
    ),
    key_id TEXT NOT NULL,
    environment TEXT NOT NULL CHECK (environment IN ('dev', 'staging', 'prod')),
    execution_context_id TEXT NOT NULL,
    workflow_run_id TEXT NOT NULL UNIQUE,
    workflow_attempt_group_id TEXT NOT NULL,
    workflow_attempt_number INTEGER NOT NULL CHECK (workflow_attempt_number > 0),
    retry_of_workflow_run_id TEXT,
    workflow_attempt_kind TEXT NOT NULL CHECK (
      workflow_attempt_kind IN ('initial', 'user_retry', 'automatic_retry')
    ),
    primary_inference_observability TEXT NOT NULL CHECK (
      primary_inference_observability IN ('hank_metered', 'hank_byok_observed')
    ),
    funding_route_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    billing_mode TEXT NOT NULL CHECK (billing_mode IN ('byok', 'metered')),
    reservation_id TEXT,
    operation TEXT NOT NULL,
    capability_id TEXT,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    issued_at INTEGER NOT NULL CHECK (issued_at > 0),
    expires_at INTEGER NOT NULL CHECK (expires_at > issued_at),
    attached_at TEXT NOT NULL,
    CHECK ((billing_mode = 'metered') = (reservation_id IS NOT NULL))
  )
"""
_TRIGGERS = (
  """
  CREATE TRIGGER IF NOT EXISTS trg_work_authorization_consumption_no_update
  BEFORE UPDATE ON commercial_work_authorization_consumptions
  BEGIN SELECT RAISE(ABORT, 'work authorization consumption is immutable'); END
  """,
  """
  CREATE TRIGGER IF NOT EXISTS trg_work_authorization_consumption_no_delete
  BEFORE DELETE ON commercial_work_authorization_consumptions
  BEGIN SELECT RAISE(ABORT, 'work authorization consumption is append-only'); END
  """,
)
_COLUMNS = (
  "authorization_id", "schema_version", "token_sha256", "content_sha256",
  "key_id", "environment",
  "execution_context_id", "workflow_run_id", "workflow_attempt_group_id",
  "workflow_attempt_number", "retry_of_workflow_run_id", "workflow_attempt_kind",
  "primary_inference_observability", "funding_route_id", "provider",
  "billing_mode", "reservation_id", "operation", "capability_id", "request_id",
  "session_id", "issued_at", "expires_at", "attached_at",
)


def _normalized_ddl(sql: str | None) -> str:
  return " ".join((sql or "").replace("IF NOT EXISTS ", "").split())


class WorkAuthorizationConsumptionError(RuntimeError):
  """Verified work authority cannot be safely attached."""


class WorkAuthorizationConsumptionConflict(WorkAuthorizationConsumptionError):
  """An immutable authorization, workflow, or token identity was reused."""


class WorkAuthorizationAlreadyAttached(WorkAuthorizationConsumptionError):
  def __init__(self, record: "WorkAuthorizationConsumptionRecord") -> None:
    self.record = record
    super().__init__("work authorization is already attached")


@dataclass(frozen=True)
class WorkAuthorizationConsumptionRecord:
  authorization_id: UUID
  schema_version: int
  token_sha256: str
  content_sha256: str
  key_id: str
  environment: Literal["dev", "staging", "prod"]
  execution_context_id: UUID
  workflow_run_id: UUID
  workflow_attempt_group_id: UUID
  workflow_attempt_number: int
  retry_of_workflow_run_id: UUID | None
  workflow_attempt_kind: str
  primary_inference_observability: str
  funding_route_id: UUID
  provider: str
  billing_mode: str
  reservation_id: UUID | None
  operation: str
  capability_id: str | None
  request_id: str
  session_id: str
  issued_at: int
  expires_at: int
  attached_at: str


class WorkAuthorizationConsumptionStore:
  """Persist a token-free, immutable attachment before provider work starts."""

  def __init__(
    self,
    path: str | Path,
    *,
    busy_timeout_ms: int = 5_000,
    synchronous: Literal["FULL", "EXTRA"] = "FULL",
    clock: Callable[[], datetime] | None = None,
  ) -> None:
    if busy_timeout_ms <= 0:
      raise ValueError("work authorization busy timeout must be positive")
    if synchronous not in {"FULL", "EXTRA"}:
      raise ValueError("work authorization synchronous mode must be FULL or EXTRA")
    self.path = Path(path).expanduser()
    self._busy_timeout_ms = busy_timeout_ms
    self._synchronous = synchronous
    self._clock = clock or (lambda: datetime.now(timezone.utc))
    self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    self._migrate()
    self._secure_database_files()

  def _connect(self) -> sqlite3.Connection:
    connection = sqlite3.connect(
      self.path,
      timeout=self._busy_timeout_ms / 1_000,
      isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
      connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
      connection.execute("PRAGMA foreign_keys=ON")
      journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
      connection.execute(f"PRAGMA synchronous={self._synchronous}")
      synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
      if journal_mode.lower() != "wal" or synchronous < 2:
        raise WorkAuthorizationConsumptionError(
          "work authorization storage durability mode is unavailable"
        )
      self._secure_database_files()
      return connection
    except BaseException:
      connection.close()
      raise

  def _secure_database_files(self) -> None:
    if os.name != "posix":
      return
    for path in (
      self.path,
      Path(str(self.path) + "-wal"),
      Path(str(self.path) + "-shm"),
    ):
      if not path.exists():
        continue
      try:
        os.chmod(path, 0o600)
        mode = stat.S_IMODE(path.stat().st_mode)
      except OSError as exc:
        raise WorkAuthorizationConsumptionError(
          "work authorization storage permissions cannot be secured"
        ) from exc
      if mode != 0o600:
        raise WorkAuthorizationConsumptionError(
          "work authorization storage permissions are insecure"
        )

  def _migrate(self) -> None:
    with self._connect() as connection:
      connection.execute("BEGIN IMMEDIATE")
      self._secure_database_files()
      connection.execute(_TABLE_SQL)
      for trigger in _TRIGGERS:
        connection.execute(trigger)
      columns = tuple(
        row[1]
        for row in connection.execute(
          "PRAGMA table_info(commercial_work_authorization_consumptions)"
        )
      )
      if columns != _COLUMNS:
        raise WorkAuthorizationConsumptionError(
          "work authorization consumption schema is incompatible"
        )
      table_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("commercial_work_authorization_consumptions",),
      ).fetchone()[0]
      if _normalized_ddl(table_sql) != _normalized_ddl(_TABLE_SQL):
        raise WorkAuthorizationConsumptionError(
          "work authorization consumption table constraints are incompatible"
        )
      for expected in _TRIGGERS:
        trigger_name = expected.split("TRIGGER IF NOT EXISTS ", 1)[1].split()[0]
        row = connection.execute(
          "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
          (trigger_name,),
        ).fetchone()
        if row is None or _normalized_ddl(row[0]) != _normalized_ddl(expected):
          raise WorkAuthorizationConsumptionError(
            "work authorization consumption trigger is incompatible"
          )
      connection.commit()

  def attach_once(
    self,
    authorization: VerifiedWorkAuthorization,
  ) -> WorkAuthorizationConsumptionRecord:
    if not isinstance(authorization, VerifiedWorkAuthorization):
      raise TypeError("attach_once requires verified work authority")
    content_sha256 = _content_sha256(authorization)
    with self._connect() as connection:
      connection.execute("BEGIN IMMEDIATE")
      epoch, attached_text = self._trusted_now()
      if epoch < authorization.issued_at or epoch >= authorization.expires_at:
        connection.rollback()
        raise WorkAuthorizationConsumptionError(
          "work authorization expired before durable attachment"
        )
      values = _values(authorization, content_sha256, attached_text)
      existing = connection.execute(
        """
        SELECT * FROM commercial_work_authorization_consumptions
         WHERE authorization_id = ? OR workflow_run_id = ? OR token_sha256 = ?
        """,
        (
          str(authorization.authorization_id),
          str(authorization.workflow_run_id),
          authorization.token_sha256,
        ),
      ).fetchall()
      if existing:
        exact = next(
          (
            row for row in existing
            if row["authorization_id"] == str(authorization.authorization_id)
            and row["workflow_run_id"] == str(authorization.workflow_run_id)
            and row["token_sha256"] == authorization.token_sha256
            and row["content_sha256"] == content_sha256
          ),
          None,
        )
        connection.rollback()
        if exact is not None and len(existing) == 1:
          raise WorkAuthorizationAlreadyAttached(_record(exact))
        raise WorkAuthorizationConsumptionConflict(
          "work authorization immutable identity conflict"
        )
      try:
        connection.execute(
          f"""
          INSERT INTO commercial_work_authorization_consumptions ({', '.join(_COLUMNS)})
          VALUES ({', '.join('?' for _ in _COLUMNS)})
          """,
          values,
        )
        connection.commit()
        self._secure_database_files()
      except BaseException:
        connection.rollback()
        raise
    record = self.get(authorization.authorization_id)
    if record is None:
      raise WorkAuthorizationConsumptionError(
        "work authorization attachment commit is not visible"
      )
    post_commit_epoch, _ = self._trusted_now()
    if (
      post_commit_epoch < authorization.issued_at
      or post_commit_epoch >= authorization.expires_at
    ):
      raise WorkAuthorizationConsumptionError(
        "work authorization expired during durable attachment; authority is consumed"
      )
    return record

  def _trusted_now(self) -> tuple[int, str]:
    now = self._clock()
    if now.tzinfo is None or now.utcoffset() is None:
      raise ValueError("work authorization attachment clock must be timezone-aware")
    now = now.astimezone(timezone.utc)
    return int(now.timestamp()), now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

  def get(self, authorization_id: UUID) -> WorkAuthorizationConsumptionRecord | None:
    with self._connect() as connection:
      row = connection.execute(
        """
        SELECT * FROM commercial_work_authorization_consumptions
         WHERE authorization_id = ?
        """,
        (str(authorization_id),),
      ).fetchone()
    return None if row is None else _record(row)

  def health(self) -> dict[str, object]:
    with self._connect() as connection:
      integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
      count = connection.execute(
        "SELECT COUNT(*) FROM commercial_work_authorization_consumptions"
      ).fetchone()[0]
      journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
      synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
      self._secure_database_files()
      secure_permissions = all(
        not path.exists() or stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in (
          self.path,
          Path(str(self.path) + "-wal"),
          Path(str(self.path) + "-shm"),
        )
      ) if os.name == "posix" else True
    return {
      "ok": (
        integrity == "ok"
        and journal_mode.lower() == "wal"
        and synchronous >= 2
        and secure_permissions
      ),
      "integrity": integrity,
      "journal_mode": journal_mode.lower(),
      "synchronous": synchronous,
      "secure_permissions": secure_permissions,
      "consumption_count": count,
    }


def _content_sha256(authorization: VerifiedWorkAuthorization) -> str:
  payload = asdict(authorization)
  payload.pop("token_sha256")
  encoded = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    default=str,
  ).encode("utf-8")
  return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _values(
  authorization: VerifiedWorkAuthorization,
  content_sha256: str,
  attached_at: str,
) -> tuple[object, ...]:
  return (
    str(authorization.authorization_id), authorization.schema_version,
    authorization.token_sha256, content_sha256,
    authorization.key_id, authorization.environment,
    str(authorization.execution_context_id), str(authorization.workflow_run_id),
    str(authorization.workflow_attempt_group_id),
    authorization.workflow_attempt_number,
    str(authorization.retry_of_workflow_run_id)
    if authorization.retry_of_workflow_run_id else None,
    authorization.workflow_attempt_kind,
    authorization.primary_inference_observability,
    str(authorization.funding_route_id), authorization.provider,
    authorization.billing_mode,
    str(authorization.reservation_id) if authorization.reservation_id else None,
    authorization.operation, authorization.capability_id,
    authorization.request_id, authorization.session_id,
    authorization.issued_at, authorization.expires_at, attached_at,
  )


def _record(row: sqlite3.Row) -> WorkAuthorizationConsumptionRecord:
  return WorkAuthorizationConsumptionRecord(
    authorization_id=UUID(row["authorization_id"]),
    schema_version=row["schema_version"],
    token_sha256=row["token_sha256"],
    content_sha256=row["content_sha256"],
    key_id=row["key_id"],
    environment=row["environment"],
    execution_context_id=UUID(row["execution_context_id"]),
    workflow_run_id=UUID(row["workflow_run_id"]),
    workflow_attempt_group_id=UUID(row["workflow_attempt_group_id"]),
    workflow_attempt_number=row["workflow_attempt_number"],
    retry_of_workflow_run_id=UUID(row["retry_of_workflow_run_id"])
    if row["retry_of_workflow_run_id"] else None,
    workflow_attempt_kind=row["workflow_attempt_kind"],
    primary_inference_observability=row["primary_inference_observability"],
    funding_route_id=UUID(row["funding_route_id"]),
    provider=row["provider"],
    billing_mode=row["billing_mode"],
    reservation_id=UUID(row["reservation_id"]) if row["reservation_id"] else None,
    operation=row["operation"],
    capability_id=row["capability_id"],
    request_id=row["request_id"],
    session_id=row["session_id"],
    issued_at=row["issued_at"],
    expires_at=row["expires_at"],
    attached_at=row["attached_at"],
  )


__all__ = [
  "WorkAuthorizationAlreadyAttached",
  "WorkAuthorizationConsumptionConflict",
  "WorkAuthorizationConsumptionError",
  "WorkAuthorizationConsumptionRecord",
  "WorkAuthorizationConsumptionStore",
]
