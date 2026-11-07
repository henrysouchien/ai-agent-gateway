"""Stored issue-inbox rows shared by readers, publication and legacy import.

The contract lives with the gateway's durable-state paths so legacy adoption
can parse rows without importing an embedding application's tool handlers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import sqlite3
from typing import Any, Mapping


ISSUE_INBOX_SCHEMA_VERSION = 1
DELIVERY_STATES = frozenset({
  "queued",
  "materialized",
  "historical_pre_fix",
  "delivery_failed",
})
_INTEGER_COLUMNS = frozenset({"schema_version", "occurrence_count", "delivery_attempts"})
_NULLABLE_COLUMNS = frozenset({
  "destination_path",
  "materialized_content_sha256",
  "delivery_error_code",
  "delivery_error_message",
})


class IssueInboxRowError(sqlite3.DatabaseError, ValueError):
  """A stored `issue_inbox` row is not a schema-1 row.

  A `sqlite3.DatabaseError` so intake and the drain's per-inbox containment
  already catch it; a `ValueError` so the drain CLI's JSON error path does too.
  """


@dataclass(frozen=True)
class StoredIssueV1:
  """One `issue_inbox` row, parsed once where it is read.

  Every column is declared; a row with another key set, a wrong-typed value,
  a `schema_version` other than `ISSUE_INBOX_SCHEMA_VERSION` or a status
  outside the delivery states is refused with `IssueInboxRowError`, never
  read as a schema-1 row.
  """

  issue_id: str
  schema_version: int
  repo: str
  severity: str
  title: str
  description: str
  signature: str
  payload_sha256: str
  status: str
  occurrence_count: int
  delivery_attempts: int
  created_at: str
  last_seen_at: str
  updated_at: str
  destination_path: str | None
  materialized_content_sha256: str | None
  delivery_error_code: str | None
  delivery_error_message: str | None

  @classmethod
  def parse(cls, row: Mapping[str, Any]) -> "StoredIssueV1":
    declared = {field.name for field in fields(cls)}
    if set(row) != declared:
      raise IssueInboxRowError(
        f"issue inbox row {row.get('issue_id')!r} columns do not match schema "
        f"{ISSUE_INBOX_SCHEMA_VERSION}: unexpected {sorted(set(row) - declared)}, "
        f"missing {sorted(declared - set(row))}"
      )
    for name in declared:
      value = row[name]
      expected = int if name in _INTEGER_COLUMNS else str
      if not (type(value) is expected or (value is None and name in _NULLABLE_COLUMNS)):
        raise IssueInboxRowError(
          f"issue inbox row {row['issue_id']!r} field {name} must be "
          f"{expected.__name__}{' or NULL' if name in _NULLABLE_COLUMNS else ''}, got {value!r}"
        )
    if row["schema_version"] != ISSUE_INBOX_SCHEMA_VERSION:
      raise IssueInboxRowError(
        f"issue inbox row {row['issue_id']!r} has unsupported schema_version "
        f"{row['schema_version']!r}; supported: {ISSUE_INBOX_SCHEMA_VERSION}"
      )
    if row["status"] not in DELIVERY_STATES:
      raise IssueInboxRowError(
        f"issue inbox row {row['issue_id']!r} has unsupported status {row['status']!r}"
      )
    return cls(**row)

  def as_record(self) -> dict[str, Any]:
    return asdict(self)
