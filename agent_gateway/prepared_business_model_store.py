"""Durable prepare-once records for business-model approval and replay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import sqlite3


class PreparedBusinessModelError(RuntimeError):
  pass


class PreparedBusinessModelLifecycle(StrEnum):
  PENDING = "PENDING"
  AUTHORIZED = "AUTHORIZED"
  DENIED = "DENIED"
  EXPIRED = "EXPIRED"
  SUPERSEDED_PRECOMMIT = "SUPERSEDED_PRECOMMIT"
  CONSUMED = "CONSUMED"


_ALLOWED_TRANSITIONS = {
  PreparedBusinessModelLifecycle.PENDING: frozenset(
    {
      PreparedBusinessModelLifecycle.AUTHORIZED,
      PreparedBusinessModelLifecycle.DENIED,
      PreparedBusinessModelLifecycle.EXPIRED,
    }
  ),
  PreparedBusinessModelLifecycle.AUTHORIZED: frozenset(
    {
      PreparedBusinessModelLifecycle.CONSUMED,
      PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT,
    }
  ),
  PreparedBusinessModelLifecycle.DENIED: frozenset(),
  PreparedBusinessModelLifecycle.EXPIRED: frozenset(),
  PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT: frozenset(),
  PreparedBusinessModelLifecycle.CONSUMED: frozenset(),
}


@dataclass(frozen=True)
class PreparedBusinessModelChange:
  schema_version: str
  caller_kind: str
  user_scope: str
  idempotency_locator: str
  intent_digest: str
  prepared_payload: bytes
  change_set_id: str
  change_hash: str
  base_vector_hash: str
  prepared_payload_digest: str
  lifecycle: PreparedBusinessModelLifecycle
  created_at: str
  expires_at: str | None = None
  approval_id: str | None = None
  approval_chain_id: str | None = None
  execution_receipt: bytes | None = None
  execution_receipt_digest: str | None = None
  restoration_digest: str | None = None
  checkpoint_id: str | None = None
  consumed_at: str | None = None

  def __post_init__(self) -> None:
    if self.schema_version != "prepared-business-model-change.v1":
      raise PreparedBusinessModelError("unsupported prepared business-model schema")
    for label, value in (
      ("caller kind", self.caller_kind),
      ("user scope", self.user_scope),
      ("idempotency locator", self.idempotency_locator),
    ):
      if not isinstance(value, str) or not value.strip():
        raise PreparedBusinessModelError(f"{label} must be non-empty text")
    for label, value in (
      ("intent", self.intent_digest),
      ("change set", self.change_set_id),
      ("change", self.change_hash),
      ("base vector", self.base_vector_hash),
      ("prepared payload", self.prepared_payload_digest),
    ):
      _require_digest(value, label)
    if not isinstance(self.prepared_payload, bytes):
      raise PreparedBusinessModelError("prepared payload must be immutable bytes")
    if _canonical_json_bytes(_decode_json(self.prepared_payload)) != self.prepared_payload:
      raise PreparedBusinessModelError("prepared payload must be canonical JSON bytes")
    if hashlib.sha256(self.prepared_payload).hexdigest() != self.prepared_payload_digest:
      raise PreparedBusinessModelError("prepared payload digest does not match its bytes")
    if type(self.lifecycle) is not PreparedBusinessModelLifecycle:
      raise PreparedBusinessModelError("prepared lifecycle must be a closed enum")
    _parse_timestamp(self.created_at, "created_at")
    if self.expires_at is not None:
      _parse_timestamp(self.expires_at, "expires_at")
    if (self.approval_id is None) is not (self.approval_chain_id is None):
      raise PreparedBusinessModelError("approval id and chain id must be present together")
    if self.execution_receipt is not None:
      if self.execution_receipt_digest is None:
        raise PreparedBusinessModelError("execution receipt requires its digest")
      if _canonical_json_bytes(_decode_json(self.execution_receipt)) != self.execution_receipt:
        raise PreparedBusinessModelError("execution receipt must be canonical JSON bytes")
      if hashlib.sha256(self.execution_receipt).hexdigest() != self.execution_receipt_digest:
        raise PreparedBusinessModelError("execution receipt digest does not match")
    elif self.execution_receipt_digest is not None:
      raise PreparedBusinessModelError("execution receipt digest requires receipt bytes")
    for label, value in (
      ("restoration", self.restoration_digest),
      ("checkpoint", self.checkpoint_id),
    ):
      if value is not None:
        _require_digest(value, label)
    if self.consumed_at is not None:
      _parse_timestamp(self.consumed_at, "consumed_at")
    _validate_lifecycle_evidence(self)


def ensure_schema(conn: sqlite3.Connection) -> None:
  conn.executescript(
    """
    CREATE TABLE IF NOT EXISTS prepared_business_model_change (
      caller_kind TEXT NOT NULL,
      user_scope TEXT NOT NULL,
      idempotency_locator TEXT NOT NULL,
      schema_version TEXT NOT NULL,
      intent_digest TEXT NOT NULL,
      prepared_payload BLOB NOT NULL,
      change_set_id TEXT NOT NULL,
      change_hash TEXT NOT NULL,
      base_vector_hash TEXT NOT NULL,
      prepared_payload_digest TEXT NOT NULL,
      lifecycle TEXT NOT NULL,
      created_at TEXT NOT NULL,
      expires_at TEXT,
      approval_id TEXT,
      approval_chain_id TEXT,
      execution_receipt BLOB,
      execution_receipt_digest TEXT,
      restoration_digest TEXT,
      checkpoint_id TEXT,
      consumed_at TEXT,
      PRIMARY KEY (caller_kind, user_scope, idempotency_locator)
    );
    CREATE INDEX IF NOT EXISTS idx_prepared_bm_payload_digest
      ON prepared_business_model_change(prepared_payload_digest);
    CREATE INDEX IF NOT EXISTS idx_prepared_bm_lifecycle_expiry
      ON prepared_business_model_change(lifecycle, expires_at);
    """
  )


def insert_or_verify(
  conn: sqlite3.Connection,
  record: PreparedBusinessModelChange,
) -> tuple[PreparedBusinessModelChange, bool]:
  if record.lifecycle is not PreparedBusinessModelLifecycle.PENDING:
    raise PreparedBusinessModelError(
      "new prepared business-model records must start PENDING"
    )
  if record.expires_at is None:
    raise PreparedBusinessModelError(
      "new prepared business-model records require finite expiry"
    )
  row = _select(conn, record.caller_kind, record.user_scope, record.idempotency_locator)
  if row is not None:
    current = _from_row(row)
    if _immutable_identity(current) != _immutable_identity(record):
      raise PreparedBusinessModelError(
        "idempotency locator is already bound to contradictory intent"
      )
    return current, False
  conn.execute(
    """
    INSERT INTO prepared_business_model_change (
      caller_kind, user_scope, idempotency_locator, schema_version,
      intent_digest, prepared_payload, change_set_id, change_hash,
      base_vector_hash, prepared_payload_digest, lifecycle, created_at,
      expires_at, approval_id, approval_chain_id, execution_receipt,
      execution_receipt_digest, restoration_digest, checkpoint_id, consumed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    _to_values(record),
  )
  return record, True


def get(
  conn: sqlite3.Connection,
  *,
  caller_kind: str,
  user_scope: str,
  idempotency_locator: str,
) -> PreparedBusinessModelChange | None:
  row = _select(conn, caller_kind, user_scope, idempotency_locator)
  return None if row is None else _from_row(row)


def transition(
  conn: sqlite3.Connection,
  *,
  caller_kind: str,
  user_scope: str,
  idempotency_locator: str,
  expected: PreparedBusinessModelLifecycle,
  target: PreparedBusinessModelLifecycle,
  approval_id: str | None = None,
  approval_chain_id: str | None = None,
  execution_receipt: bytes | None = None,
  restoration_digest: str | None = None,
  checkpoint_id: str | None = None,
  consumed_at: str | None = None,
) -> PreparedBusinessModelChange:
  current = get(
    conn,
    caller_kind=caller_kind,
    user_scope=user_scope,
    idempotency_locator=idempotency_locator,
  )
  if current is None:
    raise PreparedBusinessModelError("prepared business-model record not found")
  if type(expected) is not PreparedBusinessModelLifecycle or type(
    target
  ) is not PreparedBusinessModelLifecycle:
    raise PreparedBusinessModelError("prepared lifecycle transition is not typed")
  if current.lifecycle is target:
    _verify_idempotent_transition(
      current,
      target=target,
      approval_id=approval_id,
      approval_chain_id=approval_chain_id,
      execution_receipt=execution_receipt,
      restoration_digest=restoration_digest,
      checkpoint_id=checkpoint_id,
      consumed_at=consumed_at,
    )
    return current
  if current.lifecycle is not expected:
    raise PreparedBusinessModelError(
      f"prepared lifecycle conflict: expected {expected.value}, found {current.lifecycle.value}"
    )
  if target not in _ALLOWED_TRANSITIONS[current.lifecycle]:
    raise PreparedBusinessModelError(
      f"invalid prepared lifecycle transition: {current.lifecycle.value}->{target.value}"
    )
  if approval_id is not None and approval_id != current.approval_id:
    raise PreparedBusinessModelError("prepared approval_id is immutable")
  if (
    approval_chain_id is not None
    and approval_chain_id != current.approval_chain_id
  ):
    raise PreparedBusinessModelError("prepared approval_chain_id is immutable")
  receipt_digest = (
    hashlib.sha256(execution_receipt).hexdigest()
    if execution_receipt is not None
    else current.execution_receipt_digest
  )
  updated = replace(
    current,
    lifecycle=target,
    approval_id=approval_id if approval_id is not None else current.approval_id,
    approval_chain_id=(
      approval_chain_id if approval_chain_id is not None else current.approval_chain_id
    ),
    execution_receipt=(
      execution_receipt if execution_receipt is not None else current.execution_receipt
    ),
    execution_receipt_digest=receipt_digest,
    restoration_digest=(
      restoration_digest if restoration_digest is not None else current.restoration_digest
    ),
    checkpoint_id=checkpoint_id if checkpoint_id is not None else current.checkpoint_id,
    consumed_at=consumed_at if consumed_at is not None else current.consumed_at,
  )
  cursor = conn.execute(
    """
    UPDATE prepared_business_model_change SET
      lifecycle = ?, approval_id = ?, approval_chain_id = ?,
      execution_receipt = ?, execution_receipt_digest = ?,
      restoration_digest = ?, checkpoint_id = ?, consumed_at = ?
    WHERE caller_kind = ? AND user_scope = ? AND idempotency_locator = ?
      AND lifecycle = ?
    """,
    (
      updated.lifecycle.value,
      updated.approval_id,
      updated.approval_chain_id,
      updated.execution_receipt,
      updated.execution_receipt_digest,
      updated.restoration_digest,
      updated.checkpoint_id,
      updated.consumed_at,
      caller_kind,
      user_scope,
      idempotency_locator,
      expected.value,
    ),
  )
  if cursor.rowcount != 1:
    raise PreparedBusinessModelError("prepared lifecycle compare-and-swap lost")
  return updated


def _validate_lifecycle_evidence(record: PreparedBusinessModelChange) -> None:
  has_approval = record.approval_id is not None
  has_receipt = record.execution_receipt is not None
  if not has_approval:
    raise PreparedBusinessModelError(
      f"{record.lifecycle.value} record requires approval lineage"
    )
  if record.lifecycle is PreparedBusinessModelLifecycle.PENDING:
    if has_receipt or record.restoration_digest or record.checkpoint_id or record.consumed_at:
      raise PreparedBusinessModelError("PENDING record cannot carry terminal evidence")
    return
  if record.lifecycle is PreparedBusinessModelLifecycle.AUTHORIZED:
    if has_receipt or record.restoration_digest or record.checkpoint_id or record.consumed_at:
      raise PreparedBusinessModelError("AUTHORIZED record cannot carry terminal evidence")
    return
  if record.lifecycle in {
    PreparedBusinessModelLifecycle.DENIED,
    PreparedBusinessModelLifecycle.EXPIRED,
  }:
    if has_receipt or record.restoration_digest or record.checkpoint_id or record.consumed_at:
      raise PreparedBusinessModelError(
        f"{record.lifecycle.value} record cannot carry execution evidence"
      )
    return
  if record.lifecycle is PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT:
    if not has_approval or not has_receipt or record.restoration_digest is None:
      raise PreparedBusinessModelError(
        "SUPERSEDED_PRECOMMIT requires approval, failure receipt, and restoration"
      )
    if record.checkpoint_id is not None or record.consumed_at is not None:
      raise PreparedBusinessModelError(
        "SUPERSEDED_PRECOMMIT cannot carry checkpoint consumption evidence"
      )
    return
  if record.lifecycle is PreparedBusinessModelLifecycle.CONSUMED:
    if (
      not has_approval
      or not has_receipt
      or record.checkpoint_id is None
      or record.consumed_at is None
    ):
      raise PreparedBusinessModelError(
        "CONSUMED requires approval, receipt, checkpoint, and consumed_at"
      )
    if record.restoration_digest is not None:
      raise PreparedBusinessModelError("CONSUMED cannot carry restoration evidence")


def _verify_idempotent_transition(
  current: PreparedBusinessModelChange,
  *,
  target: PreparedBusinessModelLifecycle,
  approval_id: str | None,
  approval_chain_id: str | None,
  execution_receipt: bytes | None,
  restoration_digest: str | None,
  checkpoint_id: str | None,
  consumed_at: str | None,
) -> None:
  required = {
    PreparedBusinessModelLifecycle.AUTHORIZED: ("approval_id", "approval_chain_id"),
    PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT: (
      "approval_id",
      "approval_chain_id",
      "execution_receipt",
      "restoration_digest",
    ),
    PreparedBusinessModelLifecycle.CONSUMED: (
      "approval_id",
      "approval_chain_id",
      "execution_receipt",
      "checkpoint_id",
      "consumed_at",
    ),
  }.get(target, ())
  supplied = {
    "approval_id": approval_id,
    "approval_chain_id": approval_chain_id,
    "execution_receipt": execution_receipt,
    "restoration_digest": restoration_digest,
    "checkpoint_id": checkpoint_id,
    "consumed_at": consumed_at,
  }
  if any(supplied[field] is None for field in required):
    raise PreparedBusinessModelError(
      f"idempotent {target.value} transition requires complete evidence"
    )
  for field, value in supplied.items():
    if value is not None and getattr(current, field) != value:
      raise PreparedBusinessModelError(
        f"idempotent {target.value} transition evidence conflicts: {field}"
      )


def expire_pending(conn: sqlite3.Connection, *, now: str) -> int:
  _parse_timestamp(now, "expiry time")
  cursor = conn.execute(
    """
    UPDATE prepared_business_model_change
    SET lifecycle = 'EXPIRED'
    WHERE lifecycle = 'PENDING'
      AND expires_at IS NOT NULL AND expires_at <= ?
    """,
    (now,),
  )
  return int(cursor.rowcount)


def _immutable_identity(record: PreparedBusinessModelChange) -> tuple[object, ...]:
  return (
    record.schema_version,
    record.caller_kind,
    record.user_scope,
    record.idempotency_locator,
    record.intent_digest,
    record.prepared_payload,
    record.change_set_id,
    record.change_hash,
    record.base_vector_hash,
    record.prepared_payload_digest,
    record.expires_at,
    record.approval_id,
    record.approval_chain_id,
  )


def _select(
  conn: sqlite3.Connection,
  caller_kind: str,
  user_scope: str,
  idempotency_locator: str,
) -> sqlite3.Row | None:
  return conn.execute(
    """
    SELECT * FROM prepared_business_model_change
    WHERE caller_kind = ? AND user_scope = ? AND idempotency_locator = ?
    """,
    (caller_kind, user_scope, idempotency_locator),
  ).fetchone()


def _to_values(record: PreparedBusinessModelChange) -> tuple[object, ...]:
  return (
    record.caller_kind,
    record.user_scope,
    record.idempotency_locator,
    record.schema_version,
    record.intent_digest,
    record.prepared_payload,
    record.change_set_id,
    record.change_hash,
    record.base_vector_hash,
    record.prepared_payload_digest,
    record.lifecycle.value,
    record.created_at,
    record.expires_at,
    record.approval_id,
    record.approval_chain_id,
    record.execution_receipt,
    record.execution_receipt_digest,
    record.restoration_digest,
    record.checkpoint_id,
    record.consumed_at,
  )


def _from_row(row: sqlite3.Row) -> PreparedBusinessModelChange:
  return PreparedBusinessModelChange(
    schema_version=str(row["schema_version"]),
    caller_kind=str(row["caller_kind"]),
    user_scope=str(row["user_scope"]),
    idempotency_locator=str(row["idempotency_locator"]),
    intent_digest=str(row["intent_digest"]),
    prepared_payload=bytes(row["prepared_payload"]),
    change_set_id=str(row["change_set_id"]),
    change_hash=str(row["change_hash"]),
    base_vector_hash=str(row["base_vector_hash"]),
    prepared_payload_digest=str(row["prepared_payload_digest"]),
    lifecycle=PreparedBusinessModelLifecycle(str(row["lifecycle"])),
    created_at=str(row["created_at"]),
    expires_at=row["expires_at"],
    approval_id=row["approval_id"],
    approval_chain_id=row["approval_chain_id"],
    execution_receipt=(
      bytes(row["execution_receipt"]) if row["execution_receipt"] is not None else None
    ),
    execution_receipt_digest=row["execution_receipt_digest"],
    restoration_digest=row["restoration_digest"],
    checkpoint_id=row["checkpoint_id"],
    consumed_at=row["consumed_at"],
  )


def _canonical_json_bytes(value: object) -> bytes:
  return json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
  ).encode("utf-8")


def _decode_json(value: bytes) -> object:
  try:
    return json.loads(value.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise PreparedBusinessModelError("record payload must contain JSON") from exc


def _require_digest(value: object, label: str) -> None:
  if not isinstance(value, str) or len(value) != 64:
    raise PreparedBusinessModelError(f"{label} must be a lowercase sha256 digest")
  try:
    int(value, 16)
  except ValueError as exc:
    raise PreparedBusinessModelError(f"{label} must be a lowercase sha256 digest") from exc
  if value.lower() != value:
    raise PreparedBusinessModelError(f"{label} must be a lowercase sha256 digest")


def _parse_timestamp(value: str, label: str) -> datetime:
  try:
    parsed = datetime.fromisoformat(value)
  except ValueError as exc:
    raise PreparedBusinessModelError(f"{label} must be ISO-8601") from exc
  if parsed.tzinfo is None:
    raise PreparedBusinessModelError(f"{label} must include a timezone")
  return parsed.astimezone(UTC)


__all__ = [
  "PreparedBusinessModelChange",
  "PreparedBusinessModelError",
  "PreparedBusinessModelLifecycle",
  "ensure_schema",
  "expire_pending",
  "get",
  "insert_or_verify",
  "transition",
]
