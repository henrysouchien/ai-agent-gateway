from __future__ import annotations

import contextlib
from datetime import UTC, datetime
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal


EVENT_SCHEMA_VERSION = 1
_REVERSE_SCAN_CHUNK_SIZE = 64 * 1024
_SLUG_RE = re.compile(r"[^a-z0-9_-]")
_SEGMENT_FILE_RE = re.compile(r"^(?P<first>\d{12})-(?P<last>\d{12})-g(?P<generation>\d{6})\.jsonl$")
_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentSessionRef:
  """Identifier for a durable agent session log."""

  user_id: str
  agent_id: str
  agent_session_id: str


@dataclass(frozen=True)
class LogEntry:
  """Single durable session log entry."""

  seq: int
  timestamp: float
  event: dict[str, Any]


@dataclass(frozen=True)
class QueryCursor:
  """Opaque pagination cursor returned by query()."""

  after_seq: int
  direction: Literal["asc", "desc"]


@dataclass(frozen=True)
class _QuerySpec:
  event_types: set[str] | None
  tool_name: str | None
  tool_call_id: str | None
  sub_agent_id: str | None
  runner_id: str | None
  role: str | None
  after_seq: int | None
  before_seq: int | None
  after_ts: float | None
  before_ts: float | None
  contains_text: str | None
  has_error: bool | None


@dataclass(frozen=True)
class _Segment:
  path: Path
  first_seq: int
  last_seq: int
  active: bool = False


def slugify(value: str) -> str:
  slug = _SLUG_RE.sub("_", str(value).strip().lower())[:64]
  if not slug:
    raise ValueError("slugify() produced an empty slug")
  return slug


def resolve_agent_session_id(user_id: str, agent_id: str) -> str:
  return f"agentsess_{slugify(agent_id)}_{slugify(user_id)}"


def _now_iso() -> str:
  return datetime.now(UTC).isoformat()


def _fsync_parent_dir(path: Path) -> None:
  fd = os.open(str(path), os.O_RDONLY)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def _atomic_write_sidecar(meta_path: Path, meta: dict[str, Any]) -> None:
  """Write a sidecar atomically on local POSIX filesystems."""
  if meta_path.exists():
    return
  parent = meta_path.parent
  parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_name = tempfile.mkstemp(prefix=f"{meta_path.name}.", suffix=".tmp", dir=parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(meta, handle, separators=(",", ":"), ensure_ascii=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    if meta_path.exists():
      with contextlib.suppress(OSError):
        os.unlink(tmp_name)
      return
    os.replace(tmp_name, meta_path)
    _fsync_parent_dir(parent)
  except Exception:
    with contextlib.suppress(OSError):
      os.unlink(tmp_name)
    raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
  parent = path.parent
  parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(tmp_name, path)
    _fsync_parent_dir(parent)
  except Exception:
    with contextlib.suppress(OSError):
      os.unlink(tmp_name)
    raise


def _read_json_dict(path: Path) -> dict[str, Any] | None:
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return None
  return payload if isinstance(payload, dict) else None


def agent_session_logical_path_for_jsonl(path: str | Path) -> Path | None:
  """Return the logical active stream path for an AgentSessionLog JSONL file.

  Active v1/no-sidecar files map to themselves. Closed v2 segment files map to
  their sidecar's logical_stream_id. Segment files without a valid v2 sidecar
  fall back to their sibling active stream path when the rotated filename is
  well-formed, so readers can repair crash-window segment metadata.
  """
  jsonl_path = Path(path).expanduser()
  sidecar = _read_json_dict(jsonl_path.with_suffix(".meta.json"))

  if jsonl_path.parent.name.endswith(".segments"):
    if sidecar is None or sidecar.get("schema_version") != 2 or sidecar.get("file_role") != "segment":
      if _SEGMENT_FILE_RE.fullmatch(jsonl_path.name) is None:
        return None
      active_stem = jsonl_path.parent.name[: -len(".segments")]
      return (jsonl_path.parent.parent / f"{active_stem}.jsonl").resolve()
    logical_stream_id = sidecar.get("logical_stream_id")
    if not isinstance(logical_stream_id, str) or not logical_stream_id:
      if _SEGMENT_FILE_RE.fullmatch(jsonl_path.name) is None:
        return None
      active_stem = jsonl_path.parent.name[: -len(".segments")]
      return (jsonl_path.parent.parent / f"{active_stem}.jsonl").resolve()
    return Path(logical_stream_id).expanduser().resolve()

  if sidecar is not None and sidecar.get("schema_version") == 2:
    logical_stream_id = sidecar.get("logical_stream_id")
    if isinstance(logical_stream_id, str) and logical_stream_id:
      return Path(logical_stream_id).expanduser().resolve()

  return jsonl_path.resolve()


def _encode_entry(entry: LogEntry) -> bytes:
  payload = {
    "seq": entry.seq,
    "timestamp": entry.timestamp,
    "event": entry.event,
  }
  return (json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _parse_entry(
  raw: bytes,
  *,
  is_last_line: bool,
  path: Path,
  logger: logging.Logger,
) -> LogEntry | None:
  stripped = raw.strip()
  if not stripped:
    return None
  try:
    payload = json.loads(stripped.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError):
    if is_last_line:
      logger.warning("Skipping truncated trailing JSONL line in %s", path)
    else:
      logger.warning("Skipping malformed JSONL line in %s", path)
    return None
  if not isinstance(payload, dict):
    logger.warning("Skipping malformed JSONL line in %s", path)
    return None
  event = payload.get("event")
  if not isinstance(event, dict):
    logger.warning("Skipping malformed JSONL line in %s", path)
    return None
  try:
    seq = int(payload["seq"])
    timestamp = float(payload["timestamp"])
  except (KeyError, TypeError, ValueError):
    logger.warning("Skipping malformed JSONL line in %s", path)
    return None
  return LogEntry(seq=seq, timestamp=timestamp, event=event)


def _iter_lines_reverse(handle: Any, chunk_size: int = _REVERSE_SCAN_CHUNK_SIZE):
  handle.seek(0, os.SEEK_END)
  position = handle.tell()
  buffer = b""
  is_last_line = True

  while position > 0:
    read_size = min(chunk_size, position)
    position -= read_size
    handle.seek(position)
    chunk = handle.read(read_size)
    buffer = chunk + buffer
    parts = buffer.split(b"\n")
    buffer = parts[0]
    for line in reversed(parts[1:]):
      if not line:
        continue
      yield line, is_last_line
      is_last_line = False

  if buffer:
    yield buffer, is_last_line


def _matches_entry(
  entry: LogEntry,
  spec: _QuerySpec,
  *,
  contains_text: Callable[[Any, str], bool],
  event_has_error: Callable[[dict[str, Any]], bool],
) -> bool:
  event = entry.event
  event_type = str(event.get("type") or "")
  if spec.event_types is not None and event_type not in spec.event_types:
    return False
  if spec.tool_name is not None and event.get("tool_name") != spec.tool_name:
    return False
  if spec.tool_call_id is not None and event.get("tool_call_id") != spec.tool_call_id:
    return False
  if spec.sub_agent_id is not None and event.get("sub_agent_id") != spec.sub_agent_id:
    return False
  if spec.runner_id is not None and event.get("runner_id") != spec.runner_id:
    return False
  if spec.role is not None and event.get("role") != spec.role:
    return False
  if spec.after_ts is not None and entry.timestamp < spec.after_ts:
    return False
  if spec.before_ts is not None and entry.timestamp > spec.before_ts:
    return False
  if spec.contains_text is not None and not contains_text(event, spec.contains_text):
    return False
  if spec.has_error is not None and event_has_error(event) != spec.has_error:
    return False
  return True


def _contains_text(value: Any, needle: str) -> bool:
  if isinstance(value, str):
    return needle in value.lower()
  if isinstance(value, dict):
    return any(_contains_text(item, needle) for item in value.values())
  if isinstance(value, (list, tuple)):
    return any(_contains_text(item, needle) for item in value)
  return False


def _event_has_error(event: dict[str, Any]) -> bool:
  error = event.get("error")
  return error not in (None, "", {}, [])


__all__ = [
  "EVENT_SCHEMA_VERSION",
  "AgentSessionRef",
  "LogEntry",
  "QueryCursor",
  "_MANIFEST_SCHEMA_VERSION",
  "_QuerySpec",
  "_REVERSE_SCAN_CHUNK_SIZE",
  "_SEGMENT_FILE_RE",
  "_Segment",
  "_atomic_write_json",
  "_atomic_write_sidecar",
  "_contains_text",
  "_encode_entry",
  "_event_has_error",
  "_fsync_parent_dir",
  "_iter_lines_reverse",
  "_matches_entry",
  "_now_iso",
  "_parse_entry",
  "_read_json_dict",
  "agent_session_logical_path_for_jsonl",
  "resolve_agent_session_id",
  "slugify",
]
