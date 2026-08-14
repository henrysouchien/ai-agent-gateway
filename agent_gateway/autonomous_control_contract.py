from __future__ import annotations

import hashlib
import json
from typing import Any, BinaryIO


AUTONOMOUS_CONTROL_RECORD_VERSION = 1
AUTONOMOUS_CONTROL_RECORD_MAX_BYTES = 256 * 1024
AUTONOMOUS_OPERATOR_MESSAGE_LIMIT = 64
AUTONOMOUS_OPERATOR_AGGREGATE_BYTES_LIMIT = 1024 * 1024
AUTONOMOUS_APPROVAL_DECISION_LIMIT = 256
AUTONOMOUS_CONTROL_RECORDS_PER_DRAIN = 16

AUTONOMOUS_OPERATOR_RECORD_FIELDS = frozenset({
  "version",
  "kind",
  "task_id",
  "control_run_id",
  "session_id",
  "channel_id",
  "message_id",
  "text",
  "sent_at_ns",
})
AUTONOMOUS_APPROVAL_RECORD_FIELDS = frozenset({
  "version",
  "kind",
  "task_id",
  "control_run_id",
  "session_id",
  "channel_id",
  "approval_id",
  "tool_call_id",
  "nonce",
  "approved",
  "allow_tool_type",
  "decided_at_ns",
})


def _reject_duplicate_fields(
  pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(
        f"autonomous control record has duplicate field {key!r}"
      )
    result[key] = value
  return result


def _reject_non_json_constant(value: str) -> Any:
  raise ValueError(
    f"autonomous control record contains non-JSON constant {value}"
  )


def read_bounded_control_line(
  handle: BinaryIO,
  *,
  offset: int,
  max_bytes: int = AUTONOMOUS_CONTROL_RECORD_MAX_BYTES,
) -> tuple[bytes, int, bool] | None:
  if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
    raise ValueError("autonomous control record byte limit must be positive")
  handle.seek(offset)
  raw_line = handle.readline(max_bytes + 1)
  if len(raw_line) > max_bytes:
    raise RuntimeError(
      "autonomous control record exceeds the size limit"
    )
  if not raw_line:
    return None
  return raw_line, handle.tell(), raw_line.endswith(b"\n")


def decode_closed_control_record(
  raw_line: bytes,
  *,
  kind: str,
  fields: frozenset[str],
) -> tuple[dict[str, Any], bytes]:
  if not raw_line.endswith(b"\n"):
    raise RuntimeError(
      f"incomplete autonomous {kind} control record"
    )
  try:
    body = raw_line[:-1].strip().decode("utf-8", errors="strict")
  except UnicodeDecodeError as exc:
    raise RuntimeError(
      f"malformed autonomous {kind} control record"
    ) from exc
  if not body:
    raise RuntimeError(
      f"autonomous {kind} inbox contains an empty record"
    )
  try:
    record = json.loads(
      body,
      object_pairs_hook=_reject_duplicate_fields,
      parse_constant=_reject_non_json_constant,
    )
  except (TypeError, ValueError, json.JSONDecodeError) as exc:
    raise RuntimeError(
      f"malformed autonomous {kind} control record"
    ) from exc
  if not isinstance(record, dict) or set(record) != fields:
    raise RuntimeError(
      f"autonomous {kind} control record violates its closed contract"
    )
  if (
    type(record["version"]) is not int
    or record["version"] != AUTONOMOUS_CONTROL_RECORD_VERSION
    or type(record["kind"]) is not str
    or record["kind"] != kind
  ):
    raise RuntimeError(
      f"autonomous {kind} control record version or kind is invalid"
    )
  canonical = json.dumps(
    record,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  ).encode("utf-8")
  return record, hashlib.sha256(canonical).digest()


def encode_closed_control_record(payload: dict[str, Any]) -> bytes:
  try:
    encoded = (
      json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
      )
      + "\n"
    ).encode("utf-8")
  except (TypeError, ValueError) as exc:
    raise ValueError(
      "autonomous control record is not strict JSON"
    ) from exc
  if len(encoded) > AUTONOMOUS_CONTROL_RECORD_MAX_BYTES:
    raise ValueError(
      "autonomous control record exceeds its size bound"
    )
  return encoded


__all__ = [
  "AUTONOMOUS_APPROVAL_DECISION_LIMIT",
  "AUTONOMOUS_APPROVAL_RECORD_FIELDS",
  "AUTONOMOUS_CONTROL_RECORD_MAX_BYTES",
  "AUTONOMOUS_CONTROL_RECORD_VERSION",
  "AUTONOMOUS_CONTROL_RECORDS_PER_DRAIN",
  "AUTONOMOUS_OPERATOR_AGGREGATE_BYTES_LIMIT",
  "AUTONOMOUS_OPERATOR_MESSAGE_LIMIT",
  "AUTONOMOUS_OPERATOR_RECORD_FIELDS",
  "decode_closed_control_record",
  "encode_closed_control_record",
  "read_bounded_control_line",
]
