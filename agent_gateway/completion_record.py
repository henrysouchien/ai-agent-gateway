"""One durable row per one-shot provider call: what was sent and what came back.

Multi-turn agent loops record into `AgentSessionLog`; a prompt→completion call
has no turns, tools or session ref, so it writes one `completion_record` v1
JSON line instead. The packaged schema
(`contracts/completion-record-v1/completion_record.schema.json`) is the
contract; writers do not validate against it — the exporter does
(observe-and-allow). Capture never fails the call it records: a write error is
logged and swallowed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import base64
import binascii
import copy
import fcntl
import hashlib
from importlib.resources import files
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal
import uuid


log = logging.getLogger("agent_gateway.completion_record")

SCHEMA_VERSION = 1
CaptureLevel = Literal["wire", "stream", "text"]
OutcomeStatus = Literal["ok", "error", "canceled"]


def completion_record_schema_path() -> Path:
  return Path(
    str(files("agent_gateway") / "contracts" / "completion-record-v1" / "completion_record.schema.json")
  )


def _iso(moment: datetime) -> str:
  return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _blob_ref(block: dict[str, Any]) -> dict[str, Any]:
  source = block["source"]
  data = source.get("data")
  try:
    raw = base64.b64decode(data, validate=True)
  except (binascii.Error, ValueError, TypeError):
    raw = str(data).encode("utf-8")
  return {
    "type": block.get("type"),
    "source": {
      "type": "blob_ref",
      "sha256": hashlib.sha256(raw).hexdigest(),
      "media_type": source.get("media_type"),
      "bytes": len(raw),
    },
  }


def _strip_binary(value: Any) -> Any:
  """Replace base64 `source.data` blocks with a content-addressed reference.

  Text is kept verbatim; the binary original stays in its owning product store.
  """
  if isinstance(value, dict):
    source = value.get("source")
    if (
      isinstance(source, dict)
      and source.get("type") == "base64"
      and "data" in source
    ):
      return _blob_ref(value)
    return {key: _strip_binary(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_strip_binary(item) for item in value]
  return value


def build_completion_record(
  *,
  source_repo: str | None,
  call_site: str,
  provider: str,
  model: str,
  capture_level: CaptureLevel,
  request: dict[str, Any],
  response: dict[str, Any] | None,
  status: OutcomeStatus,
  started_at: datetime,
  recorded_at: datetime | None = None,
  duration_ms: int | None = None,
  caller: str | None = None,
  session_id: str | None = None,
  request_id: str | None = None,
  group_id: str | None = None,
  task_id: str | None = None,
  trace_id: str | None = None,
  user_id: str | None = None,
  subject: dict[str, Any] | None = None,
  provider_reported_model: str | None = None,
  capability_bind: dict[str, Any] | None = None,
  model_selection: dict[str, Any] | None = None,
  usage: dict[str, int | None] | None = None,
  error_type: str | None = None,
  error_message: str | None = None,
) -> dict[str, Any]:
  recorded = recorded_at or datetime.now(timezone.utc)
  usage = usage or {}
  return {
    "record_type": "completion",
    "schema_version": SCHEMA_VERSION,
    "completion_id": f"cmp_{uuid.uuid4().hex}",
    "recorded_at": _iso(recorded),
    "started_at": _iso(started_at),
    "duration_ms": duration_ms,
    "source": {"repo": source_repo, "call_site": call_site, "caller": caller},
    "join": {
      "session_id": session_id,
      "request_id": request_id,
      "group_id": group_id,
      "task_id": task_id,
      "trace_id": trace_id,
      "user_id": user_id,
      "subject": dict(subject or {}),
    },
    "model": {
      "provider": provider,
      "model": model,
      "provider_reported_model": provider_reported_model,
      "capability_bind": capability_bind,
      "model_selection": model_selection,
    },
    "capture_level": capture_level,
    "request": _strip_binary(copy.deepcopy(request)),
    "response": None if response is None else _strip_binary(copy.deepcopy(response)),
    "usage": {
      key: usage.get(key)
      for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
        "thinking_chars",
      )
    },
    "outcome": {
      "status": status,
      "error_type": error_type,
      "error_message": error_message,
    },
  }


def append_completion_record(root: Path, record: dict[str, Any]) -> None:
  """Append one record to `root/<YYYY-MM-DD>.jsonl`; never raises.

  Several worker processes share a day shard and lines run to hundreds of KB,
  so the whole line is one `os.write` on an `O_APPEND` fd under an exclusive
  `flock`.
  """
  try:
    day = str(record["recorded_at"])[:10]
    root.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    fd = os.open(root / f"{day}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
      fcntl.flock(fd, fcntl.LOCK_EX)
      try:
        os.write(fd, line)
      finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
      os.close(fd)
  except Exception as exc:  # capture, not a runtime prerequisite
    source = record.get("source") if isinstance(record, dict) else None
    log.error(
      "completion_record_write_failed call_site=%s error=%s",
      source.get("call_site") if isinstance(source, dict) else None,
      type(exc).__name__,
    )


__all__ = [
  "SCHEMA_VERSION",
  "append_completion_record",
  "build_completion_record",
  "completion_record_schema_path",
]
