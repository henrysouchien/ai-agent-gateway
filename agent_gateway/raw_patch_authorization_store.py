"""Atomic single-call consumption for outward raw patch authorizations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import re
import sqlite3


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHANGE_SET_RE = re.compile(r"^reviewed-change:v1:[0-9a-f]{64}$")
_REFERENCE_PREFIX = "approval-ref:v1:"


class RawPatchAuthorizationError(RuntimeError):
  pass


@dataclass(frozen=True)
class RawPatchAuthorizationConsumption:
  approval_id: str
  change_set_id: str
  change_hash: str
  receipt: bytes
  receipt_digest: str
  consumed_at: str

  def __post_init__(self) -> None:
    if not isinstance(self.approval_id, str) or not self.approval_id.strip():
      raise RawPatchAuthorizationError("approval_id must be non-empty text")
    if not isinstance(self.change_set_id, str) or _CHANGE_SET_RE.fullmatch(
      self.change_set_id
    ) is None:
      raise RawPatchAuthorizationError("change_set_id is not canonical")
    _require_digest(self.change_hash, "change_hash")
    _require_digest(self.receipt_digest, "receipt_digest")
    if not isinstance(self.receipt, bytes):
      raise RawPatchAuthorizationError("receipt must be immutable bytes")
    if _canonical_json_bytes(_decode_json(self.receipt)) != self.receipt:
      raise RawPatchAuthorizationError("receipt must contain canonical JSON")
    receipt_value = _decode_json(self.receipt)
    if not isinstance(receipt_value, dict):
      raise RawPatchAuthorizationError("receipt must contain an object")
    supplied_digest = receipt_value.get("receipt_digest")
    receipt_body = dict(receipt_value)
    receipt_body.pop("receipt_digest", None)
    if supplied_digest != self.receipt_digest or _digest(
      _canonical_json_bytes(receipt_body)
    ) != self.receipt_digest:
      raise RawPatchAuthorizationError("receipt digest does not match its body")
    _parse_timestamp(self.consumed_at)


@dataclass(frozen=True)
class RawPatchPreparedAuthorization:
  approval_id: str
  prepared_payload: bytes
  prepared_payload_digest: str
  created_at: str

  def __post_init__(self) -> None:
    if not isinstance(self.approval_id, str) or not self.approval_id.strip():
      raise RawPatchAuthorizationError("approval_id must be non-empty text")
    if not isinstance(self.prepared_payload, bytes):
      raise RawPatchAuthorizationError("prepared payload must be immutable bytes")
    if _canonical_json_bytes(_decode_json(self.prepared_payload)) != (
      self.prepared_payload
    ):
      raise RawPatchAuthorizationError(
        "prepared payload must contain canonical JSON"
      )
    _require_digest(self.prepared_payload_digest, "prepared_payload_digest")
    if _digest(self.prepared_payload) != self.prepared_payload_digest:
      raise RawPatchAuthorizationError(
        "prepared payload digest does not match its bytes"
      )
    _parse_timestamp(self.created_at)


@dataclass(frozen=True)
class RawPatchAuthorizationClaim:
  approval_id: str
  claim_token: str
  change_set_id: str
  change_hash: str
  prepared_payload_digest: str
  claimed_at: str
  lease_expires_at: str

  def __post_init__(self) -> None:
    if not self.approval_id or not self.claim_token:
      raise RawPatchAuthorizationError("raw patch claim identity is incomplete")
    if _CHANGE_SET_RE.fullmatch(self.change_set_id) is None:
      raise RawPatchAuthorizationError("claim change_set_id is not canonical")
    _require_digest(self.change_hash, "claim change_hash")
    _require_digest(self.prepared_payload_digest, "claim prepared payload")
    _parse_timestamp(self.claimed_at)
    _parse_timestamp(self.lease_expires_at)
    if _timestamp(self.lease_expires_at) < _timestamp(self.claimed_at):
      raise RawPatchAuthorizationError("claim lease cannot expire before it starts")


def ensure_schema(conn: sqlite3.Connection) -> None:
  conn.executescript(
    """
    CREATE TABLE IF NOT EXISTS raw_patch_authorization_consumptions (
      approval_id TEXT PRIMARY KEY
        REFERENCES approval_requests(approval_id) ON DELETE CASCADE,
      change_set_id TEXT NOT NULL,
      change_hash TEXT NOT NULL,
      receipt BLOB NOT NULL,
      receipt_digest TEXT NOT NULL,
      consumed_at TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_patch_consumption_receipt
      ON raw_patch_authorization_consumptions(receipt_digest);
    CREATE TABLE IF NOT EXISTS raw_patch_prepared_authorizations (
      approval_id TEXT PRIMARY KEY
        REFERENCES approval_requests(approval_id) ON DELETE CASCADE,
      prepared_payload BLOB NOT NULL,
      prepared_payload_digest TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_raw_patch_prepared_digest
      ON raw_patch_prepared_authorizations(prepared_payload_digest);
    CREATE TABLE IF NOT EXISTS raw_patch_authorization_claims (
      approval_id TEXT PRIMARY KEY
        REFERENCES approval_requests(approval_id) ON DELETE CASCADE,
      claim_token TEXT NOT NULL UNIQUE,
      change_set_id TEXT NOT NULL,
      change_hash TEXT NOT NULL,
      prepared_payload_digest TEXT NOT NULL,
      claimed_at TEXT NOT NULL,
      lease_expires_at TEXT NOT NULL
    );
    """
  )
  columns = {
    str(row["name"])
    for row in conn.execute("PRAGMA table_info(raw_patch_authorization_claims)")
  }
  if "lease_expires_at" not in columns:
    conn.execute(
      "ALTER TABLE raw_patch_authorization_claims ADD COLUMN lease_expires_at TEXT"
    )


def claim(
  conn: sqlite3.Connection,
  *,
  approval_id: str,
  claim_token: str,
  change_set_id: str,
  change_hash: str,
  prepared_payload_digest: str,
  claimed_at: str | None = None,
  lease_seconds: float = 120.0,
) -> tuple[RawPatchAuthorizationClaim, bool]:
  started_at = claimed_at or datetime.now(UTC).isoformat()
  started = _timestamp(started_at)
  if lease_seconds <= 0:
    raise RawPatchAuthorizationError("claim lease must be positive")
  candidate = RawPatchAuthorizationClaim(
    approval_id=approval_id,
    claim_token=claim_token,
    change_set_id=change_set_id,
    change_hash=change_hash,
    prepared_payload_digest=prepared_payload_digest,
    claimed_at=started_at,
    lease_expires_at=(started + timedelta(seconds=lease_seconds)).isoformat(),
  )
  row = conn.execute(
    "SELECT * FROM raw_patch_authorization_claims WHERE approval_id = ?",
    (approval_id,),
  ).fetchone()
  if row is not None:
    current = _claim_from_row(row)
    if (
      current.change_set_id,
      current.change_hash,
      current.prepared_payload_digest,
    ) != (
      candidate.change_set_id,
      candidate.change_hash,
      candidate.prepared_payload_digest,
    ):
      raise RawPatchAuthorizationError(
        "authorization is already claimed by a different patch identity"
      )
    if current.lease_expires_at and _timestamp(current.lease_expires_at) > started:
      raise RawPatchAuthorizationError(
        "authorization already has an active execution claim"
      )
    cursor = conn.execute(
      """
      UPDATE raw_patch_authorization_claims SET
        claim_token = ?, claimed_at = ?, lease_expires_at = ?
      WHERE approval_id = ? AND claim_token = ?
      """,
      (
        candidate.claim_token,
        candidate.claimed_at,
        candidate.lease_expires_at,
        candidate.approval_id,
        current.claim_token,
      ),
    )
    if cursor.rowcount != 1:
      raise RawPatchAuthorizationError("raw patch claim takeover lost")
    return candidate, True
  conn.execute(
    """
    INSERT INTO raw_patch_authorization_claims (
      approval_id, claim_token, change_set_id, change_hash,
      prepared_payload_digest, claimed_at, lease_expires_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (
      candidate.approval_id,
      candidate.claim_token,
      candidate.change_set_id,
      candidate.change_hash,
      candidate.prepared_payload_digest,
      candidate.claimed_at,
      candidate.lease_expires_at,
    ),
  )
  return candidate, True


def get_claim(
  conn: sqlite3.Connection,
  *,
  approval_id: str,
) -> RawPatchAuthorizationClaim | None:
  row = conn.execute(
    "SELECT * FROM raw_patch_authorization_claims WHERE approval_id = ?",
    (approval_id,),
  ).fetchone()
  return None if row is None else _claim_from_row(row)


def heartbeat_claim(
  conn: sqlite3.Connection,
  *,
  approval_id: str,
  claim_token: str,
  lease_seconds: float = 120.0,
  now: str | None = None,
) -> RawPatchAuthorizationClaim:
  timestamp = now or datetime.now(UTC).isoformat()
  started = _timestamp(timestamp)
  if lease_seconds <= 0:
    raise RawPatchAuthorizationError("claim lease must be positive")
  lease_expires_at = (started + timedelta(seconds=lease_seconds)).isoformat()
  cursor = conn.execute(
    """
    UPDATE raw_patch_authorization_claims
    SET lease_expires_at = ?
    WHERE approval_id = ? AND claim_token = ?
    """,
    (lease_expires_at, approval_id, claim_token),
  )
  if cursor.rowcount != 1:
    raise RawPatchAuthorizationError("raw patch claim changed before heartbeat")
  claim_row = conn.execute(
    "SELECT * FROM raw_patch_authorization_claims WHERE approval_id = ?",
    (approval_id,),
  ).fetchone()
  if claim_row is None:
    raise RawPatchAuthorizationError("raw patch claim missing after heartbeat")
  return _claim_from_row(claim_row)


def finalize_claim(
  conn: sqlite3.Connection,
  *,
  approval_id: str,
  claim_token: str,
) -> None:
  cursor = conn.execute(
    "DELETE FROM raw_patch_authorization_claims WHERE approval_id = ? AND claim_token = ?",
    (approval_id, claim_token),
  )
  if cursor.rowcount != 1:
    raise RawPatchAuthorizationError("raw patch claim changed before finalization")


def release_claim(
  conn: sqlite3.Connection,
  *,
  approval_id: str,
  claim_token: str,
) -> None:
  """Release one exact active claim after execution or consumption failure."""

  cursor = conn.execute(
    "DELETE FROM raw_patch_authorization_claims WHERE approval_id = ? AND claim_token = ?",
    (approval_id, claim_token),
  )
  if cursor.rowcount != 1:
    raise RawPatchAuthorizationError("raw patch claim changed before release")


def encode_reference(tool_call_id: str) -> str:
  normalized = str(tool_call_id or "").strip()
  if not normalized or normalized != tool_call_id:
    raise RawPatchAuthorizationError("tool_call_id must be canonical text")
  return f"{_REFERENCE_PREFIX}{normalized}"


def decode_reference(reference: str) -> str:
  if not isinstance(reference, str) or not reference.startswith(_REFERENCE_PREFIX):
    raise RawPatchAuthorizationError("authorization reference is malformed")
  tool_call_id = reference[len(_REFERENCE_PREFIX):]
  if not tool_call_id or tool_call_id.strip() != tool_call_id:
    raise RawPatchAuthorizationError("authorization reference is malformed")
  return tool_call_id


def consume(
  conn: sqlite3.Connection,
  *,
  approval_id: str,
  change_set_id: str,
  change_hash: str,
  receipt: bytes,
  consumed_at: str | None = None,
) -> tuple[RawPatchAuthorizationConsumption, bool]:
  timestamp = consumed_at or datetime.now(UTC).isoformat()
  receipt_value = _decode_json(receipt)
  if not isinstance(receipt_value, dict):
    raise RawPatchAuthorizationError("receipt must contain an object")
  candidate = RawPatchAuthorizationConsumption(
    approval_id=approval_id,
    change_set_id=change_set_id,
    change_hash=change_hash,
    receipt=receipt,
    receipt_digest=str(receipt_value.get("receipt_digest") or ""),
    consumed_at=timestamp,
  )
  row = conn.execute(
    "SELECT * FROM raw_patch_authorization_consumptions WHERE approval_id = ?",
    (approval_id,),
  ).fetchone()
  if row is not None:
    current = _from_row(row)
    if (
      current.change_set_id,
      current.change_hash,
      current.receipt,
      current.receipt_digest,
    ) != (
      candidate.change_set_id,
      candidate.change_hash,
      candidate.receipt,
      candidate.receipt_digest,
    ):
      raise RawPatchAuthorizationError(
        "authorization was already consumed by a different patch receipt"
      )
    return current, False
  conn.execute(
    """
    INSERT INTO raw_patch_authorization_consumptions (
      approval_id, change_set_id, change_hash, receipt, receipt_digest, consumed_at
    ) VALUES (?, ?, ?, ?, ?, ?)
    """,
    (
      candidate.approval_id,
      candidate.change_set_id,
      candidate.change_hash,
      candidate.receipt,
      candidate.receipt_digest,
      candidate.consumed_at,
    ),
  )
  return candidate, True


def bind_prepared(
  conn: sqlite3.Connection,
  *,
  approval_id: str,
  prepared_payload: bytes,
  created_at: str | None = None,
) -> tuple[RawPatchPreparedAuthorization, bool]:
  candidate = RawPatchPreparedAuthorization(
    approval_id=approval_id,
    prepared_payload=prepared_payload,
    prepared_payload_digest=_digest(prepared_payload),
    created_at=created_at or datetime.now(UTC).isoformat(),
  )
  row = conn.execute(
    "SELECT * FROM raw_patch_prepared_authorizations WHERE approval_id = ?",
    (approval_id,),
  ).fetchone()
  if row is not None:
    current = _prepared_from_row(row)
    if current.prepared_payload != candidate.prepared_payload:
      raise RawPatchAuthorizationError(
        "authorization is already bound to different prepared patch bytes"
      )
    return current, False
  conn.execute(
    """
    INSERT INTO raw_patch_prepared_authorizations (
      approval_id, prepared_payload, prepared_payload_digest, created_at
    ) VALUES (?, ?, ?, ?)
    """,
    (
      candidate.approval_id,
      candidate.prepared_payload,
      candidate.prepared_payload_digest,
      candidate.created_at,
    ),
  )
  return candidate, True


def get_prepared(
  conn: sqlite3.Connection,
  *,
  approval_id: str,
) -> RawPatchPreparedAuthorization | None:
  row = conn.execute(
    "SELECT * FROM raw_patch_prepared_authorizations WHERE approval_id = ?",
    (approval_id,),
  ).fetchone()
  return None if row is None else _prepared_from_row(row)


def get(
  conn: sqlite3.Connection,
  *,
  approval_id: str,
) -> RawPatchAuthorizationConsumption | None:
  row = conn.execute(
    "SELECT * FROM raw_patch_authorization_consumptions WHERE approval_id = ?",
    (approval_id,),
  ).fetchone()
  return None if row is None else _from_row(row)


def _from_row(row: sqlite3.Row) -> RawPatchAuthorizationConsumption:
  return RawPatchAuthorizationConsumption(
    approval_id=str(row["approval_id"]),
    change_set_id=str(row["change_set_id"]),
    change_hash=str(row["change_hash"]),
    receipt=bytes(row["receipt"]),
    receipt_digest=str(row["receipt_digest"]),
    consumed_at=str(row["consumed_at"]),
  )


def _prepared_from_row(row: sqlite3.Row) -> RawPatchPreparedAuthorization:
  return RawPatchPreparedAuthorization(
    approval_id=str(row["approval_id"]),
    prepared_payload=bytes(row["prepared_payload"]),
    prepared_payload_digest=str(row["prepared_payload_digest"]),
    created_at=str(row["created_at"]),
  )


def _claim_from_row(row: sqlite3.Row) -> RawPatchAuthorizationClaim:
  return RawPatchAuthorizationClaim(
    approval_id=str(row["approval_id"]),
    claim_token=str(row["claim_token"]),
    change_set_id=str(row["change_set_id"]),
    change_hash=str(row["change_hash"]),
    prepared_payload_digest=str(row["prepared_payload_digest"]),
    claimed_at=str(row["claimed_at"]),
    lease_expires_at=str(row["lease_expires_at"] or row["claimed_at"]),
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
    raise RawPatchAuthorizationError("receipt must contain JSON") from exc


def _digest(value: bytes) -> str:
  return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _require_digest(value: str, label: str) -> None:
  if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
    raise RawPatchAuthorizationError(f"{label} is not canonical")


def _parse_timestamp(value: str) -> None:
  try:
    parsed = datetime.fromisoformat(value)
  except (TypeError, ValueError) as exc:
    raise RawPatchAuthorizationError("consumed_at is invalid") from exc
  if parsed.tzinfo is None:
    raise RawPatchAuthorizationError("consumed_at must include a timezone")


def _timestamp(value: str) -> datetime:
  try:
    parsed = datetime.fromisoformat(value)
  except (TypeError, ValueError) as exc:
    raise RawPatchAuthorizationError("timestamp is invalid") from exc
  if parsed.tzinfo is None:
    raise RawPatchAuthorizationError("timestamp must include a timezone")
  return parsed.astimezone(UTC)


__all__ = [
  "RawPatchAuthorizationConsumption",
  "RawPatchAuthorizationClaim",
  "RawPatchAuthorizationError",
  "RawPatchPreparedAuthorization",
  "bind_prepared",
  "claim",
  "consume",
  "decode_reference",
  "encode_reference",
  "ensure_schema",
  "finalize_claim",
  "get",
  "get_claim",
  "get_prepared",
  "heartbeat_claim",
  "release_claim",
]
