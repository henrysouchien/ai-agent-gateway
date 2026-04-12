from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import fcntl


EVENT_SCHEMA_VERSION = 1
_REVERSE_SCAN_CHUNK_SIZE = 64 * 1024
_SLUG_RE = re.compile(r"[^a-z0-9_-]")
log = logging.getLogger("agent_gateway.agent_session_log")


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


def slugify(value: str) -> str:
  slug = _SLUG_RE.sub("_", str(value).strip().lower())[:64]
  if not slug:
    raise ValueError("slugify() produced an empty slug")
  return slug


def resolve_agent_session_id(user_id: str, agent_id: str) -> str:
  return f"agentsess_{slugify(agent_id)}_{slugify(user_id)}"


class AgentSessionLog:
  """Durable JSONL-backed event log for one `(user_id, agent_id)` pair."""

  def __init__(
    self,
    path: str | Path | None = None,
    *,
    session_ref: AgentSessionRef | None = None,
    base_dir: str | Path | None = None,
  ) -> None:
    if path is None:
      if session_ref is None or base_dir is None:
        raise ValueError("Provide either path or both session_ref and base_dir")
      path = self.path_for_session(base_dir, session_ref)
    elif session_ref is not None or base_dir is not None:
      raise ValueError("path cannot be combined with session_ref/base_dir")

    self.path = Path(path).expanduser()
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.path.touch(exist_ok=True)

    self.write_lease_path = self.path.with_name(f"{self.path.name}.write_lease")
    self.append_mutex_path = self.path.with_name(f"{self.path.name}.append_mutex")
    self.write_lease_meta_path = self.path.with_name(f"{self.path.name}.write_lease.meta")

    self._cache_lock = threading.Lock()
    self._seq_offsets: dict[int, int] = {}
    self._max_cached_seq = 0
    self._cache_complete = False

  @staticmethod
  def path_for_session(base_dir: str | Path, session_ref: AgentSessionRef) -> Path:
    expected_agent_session_id = resolve_agent_session_id(session_ref.user_id, session_ref.agent_id)
    if session_ref.agent_session_id != expected_agent_session_id:
      raise ValueError("agent_session_id does not match canonical resolution")
    agent_dir = Path(base_dir).expanduser() / slugify(session_ref.agent_id)
    return agent_dir / f"{expected_agent_session_id}.jsonl"

  async def append(self, event: dict[str, Any]) -> LogEntry:
    return await asyncio.to_thread(self._append_sync, dict(event))

  async def query(
    self,
    *,
    event_types: set[str] | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    sub_agent_id: str | None = None,
    runner_id: str | None = None,
    role: str | None = None,
    after_seq: int | None = None,
    before_seq: int | None = None,
    after_ts: float | None = None,
    before_ts: float | None = None,
    contains_text: str | None = None,
    has_error: bool | None = None,
    order: Literal["asc", "desc"] = "asc",
    limit: int | None = None,
    cursor: QueryCursor | None = None,
  ) -> tuple[list[LogEntry], QueryCursor | None]:
    spec = _QuerySpec(
      event_types=set(event_types) if event_types is not None else None,
      tool_name=tool_name,
      tool_call_id=tool_call_id,
      sub_agent_id=sub_agent_id,
      runner_id=runner_id,
      role=role,
      after_seq=after_seq,
      before_seq=before_seq,
      after_ts=after_ts,
      before_ts=before_ts,
      contains_text=contains_text.lower() if contains_text is not None else None,
      has_error=has_error,
    )
    return await asyncio.to_thread(self._query_sync, spec, order, limit, cursor)

  async def iter_from(self, after_seq: int = 0) -> AsyncIterator[LogEntry]:
    entries, _cursor = await self.query(after_seq=max(after_seq, 0) + 1, order="asc")
    for entry in entries:
      yield entry

  async def latest_seq(self) -> int:
    return await asyncio.to_thread(self._latest_seq_sync)

  def _append_sync(self, event: dict[str, Any]) -> LogEntry:
    event_payload = dict(event)
    event_payload["event_schema_version"] = EVENT_SCHEMA_VERSION

    with self.append_mutex_path.open("a+b") as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        with self.path.open("a+b") as handle:
          handle.seek(0, os.SEEK_END)
          file_size = handle.tell()
          latest_seq = self._latest_seq_from_handle(handle)
          needs_separator = False
          if file_size > 0:
            handle.seek(file_size - 1)
            needs_separator = handle.read(1) != b"\n"

          seq = latest_seq + 1
          timestamp = time.time()
          entry = LogEntry(seq=seq, timestamp=timestamp, event=event_payload)
          prefix = b"\n" if needs_separator else b""
          line_offset = file_size + len(prefix)
          payload = prefix + self._encode_entry(entry)

          handle.seek(0, os.SEEK_END)
          handle.write(payload)
          handle.flush()
      finally:
        fcntl.flock(mutex_file.fileno(), fcntl.LOCK_UN)

    self._update_cache(seq=entry.seq, offset=line_offset)
    return entry

  def _query_sync(
    self,
    spec: _QuerySpec,
    order: Literal["asc", "desc"],
    limit: int | None,
    cursor: QueryCursor | None,
  ) -> tuple[list[LogEntry], QueryCursor | None]:
    if order not in {"asc", "desc"}:
      raise ValueError("order must be 'asc' or 'desc'")
    if limit is not None and limit <= 0:
      raise ValueError("limit must be positive when provided")
    if cursor is not None and cursor.direction != order:
      raise ValueError("cursor direction does not match query order")
    if spec.after_seq is not None and spec.before_seq is not None and spec.after_seq > spec.before_seq:
      return [], None

    effective_spec = spec
    if cursor is not None:
      if order == "asc":
        cursor_after = cursor.after_seq + 1
        effective_spec = _QuerySpec(
          **{
            **spec.__dict__,
            "after_seq": cursor_after if spec.after_seq is None else max(spec.after_seq, cursor_after),
          }
        )
      else:
        cursor_before = cursor.after_seq - 1
        effective_spec = _QuerySpec(
          **{
            **spec.__dict__,
            "before_seq": cursor_before if spec.before_seq is None else min(spec.before_seq, cursor_before),
          }
        )
      if effective_spec.before_seq is not None and effective_spec.after_seq is not None:
        if effective_spec.after_seq > effective_spec.before_seq:
          return [], None

    fetch_limit = None if limit is None else limit + 1
    if order == "asc":
      entries = self._query_asc_sync(effective_spec, fetch_limit)
    else:
      entries = self._query_desc_sync(effective_spec, fetch_limit)

    next_cursor = None
    if limit is not None and len(entries) > limit:
      next_cursor = QueryCursor(after_seq=entries[limit - 1].seq, direction=order)
      entries = entries[:limit]
    return entries, next_cursor

  def _query_asc_sync(self, spec: _QuerySpec, limit: int | None) -> list[LogEntry]:
    results: list[LogEntry] = []
    start_offset = self._starting_offset_for_seq(spec.after_seq)
    should_mark_complete = False
    with self.path.open("rb") as handle:
      file_size = handle.seek(0, os.SEEK_END)
      handle.seek(start_offset)
      while True:
        offset = handle.tell()
        raw = handle.readline()
        if not raw:
          should_mark_complete = handle.tell() >= file_size
          break
        is_last_line = handle.tell() >= file_size
        entry = self._parse_entry(raw, is_last_line=is_last_line)
        if entry is None:
          continue
        self._update_cache(seq=entry.seq, offset=offset)
        if spec.after_seq is not None and entry.seq < spec.after_seq:
          continue
        if spec.before_seq is not None and entry.seq > spec.before_seq:
          break
        if not self._matches(entry, spec):
          continue
        results.append(entry)
        if limit is not None and len(results) >= limit:
          break
    if should_mark_complete:
      with self._cache_lock:
        self._cache_complete = True
    return results

  def _query_desc_sync(self, spec: _QuerySpec, limit: int | None) -> list[LogEntry]:
    results: list[LogEntry] = []
    with self.path.open("rb") as handle:
      for raw, is_last_line in self._iter_lines_reverse(handle):
        entry = self._parse_entry(raw, is_last_line=is_last_line)
        if entry is None:
          continue
        if spec.before_seq is not None and entry.seq > spec.before_seq:
          continue
        if spec.after_seq is not None and entry.seq < spec.after_seq:
          break
        if not self._matches(entry, spec):
          continue
        results.append(entry)
        if limit is not None and len(results) >= limit:
          break
    return results

  def _latest_seq_sync(self) -> int:
    with self.path.open("rb") as handle:
      return self._latest_seq_from_handle(handle)

  def _latest_seq_from_handle(self, handle: Any) -> int:
    for raw, is_last_line in self._iter_lines_reverse(handle):
      entry = self._parse_entry(raw, is_last_line=is_last_line)
      if entry is not None:
        return entry.seq
    return 0

  def _starting_offset_for_seq(self, after_seq: int | None) -> int:
    if after_seq is None or after_seq <= 1:
      return 0
    with self._cache_lock:
      if after_seq in self._seq_offsets:
        return self._seq_offsets[after_seq]
      if self._cache_complete and after_seq > self._max_cached_seq:
        return self.path.stat().st_size
      if self._max_cached_seq > 0 and after_seq > self._max_cached_seq:
        return self._seq_offsets.get(self._max_cached_seq, 0)
    return 0

  def _update_cache(self, *, seq: int, offset: int) -> None:
    with self._cache_lock:
      existing = self._seq_offsets.get(seq)
      if existing is None or offset < existing:
        self._seq_offsets[seq] = offset
      if seq > self._max_cached_seq:
        self._max_cached_seq = seq

  def _encode_entry(self, entry: LogEntry) -> bytes:
    payload = {
      "seq": entry.seq,
      "timestamp": entry.timestamp,
      "event": entry.event,
    }
    return (json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")

  def _parse_entry(self, raw: bytes, *, is_last_line: bool) -> LogEntry | None:
    stripped = raw.strip()
    if not stripped:
      return None
    try:
      payload = json.loads(stripped.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
      if is_last_line:
        log.warning("Skipping truncated trailing JSONL line in %s", self.path)
      else:
        log.warning("Skipping malformed JSONL line in %s", self.path)
      return None
    if not isinstance(payload, dict):
      log.warning("Skipping malformed JSONL line in %s", self.path)
      return None
    event = payload.get("event")
    if not isinstance(event, dict):
      log.warning("Skipping malformed JSONL line in %s", self.path)
      return None
    try:
      seq = int(payload["seq"])
      timestamp = float(payload["timestamp"])
    except (KeyError, TypeError, ValueError):
      log.warning("Skipping malformed JSONL line in %s", self.path)
      return None
    return LogEntry(seq=seq, timestamp=timestamp, event=event)

  def _iter_lines_reverse(self, handle: Any, chunk_size: int = _REVERSE_SCAN_CHUNK_SIZE):
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

  def _matches(self, entry: LogEntry, spec: _QuerySpec) -> bool:
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
    if spec.contains_text is not None and not self._contains_text(event, spec.contains_text):
      return False
    if spec.has_error is not None and self._event_has_error(event) != spec.has_error:
      return False
    return True

  def _contains_text(self, value: Any, needle: str) -> bool:
    if isinstance(value, str):
      return needle in value.lower()
    if isinstance(value, dict):
      return any(self._contains_text(item, needle) for item in value.values())
    if isinstance(value, (list, tuple)):
      return any(self._contains_text(item, needle) for item in value)
    return False

  def _event_has_error(self, event: dict[str, Any]) -> bool:
    error = event.get("error")
    return error not in (None, "", {}, [])


__all__ = [
  "AgentSessionLog",
  "AgentSessionRef",
  "LogEntry",
  "QueryCursor",
  "resolve_agent_session_id",
  "slugify",
]
