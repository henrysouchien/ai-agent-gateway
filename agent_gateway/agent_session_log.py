from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import fcntl


EVENT_SCHEMA_VERSION = 1
_REVERSE_SCAN_CHUNK_SIZE = 64 * 1024
_SLUG_RE = re.compile(r"[^a-z0-9_-]")
_SEGMENT_FILE_RE = re.compile(r"^(?P<first>\d{12})-(?P<last>\d{12})-g(?P<generation>\d{6})\.jsonl$")
_MANIFEST_SCHEMA_VERSION = 1
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
    if not self.path.exists():
      self.path.touch()
    self.segments_dir = self.path.with_name(f"{self.path.stem}.segments")
    self.manifest_path = self.segments_dir / "manifest.json"
    self._max_active_bytes = self._configured_max_active_bytes()

    self.write_lease_path = self.path.with_name(f"{self.path.name}.write_lease")
    self.append_mutex_path = self.path.with_name(f"{self.path.name}.append_mutex")
    self.write_lease_meta_path = self.path.with_name(f"{self.path.name}.write_lease.meta")

    self._cache_lock = threading.Lock()
    self._seq_offsets: dict[int, int] = {}
    self._max_cached_seq = 0
    self._cache_complete = False
    self._cache_file_identity: tuple[int, int, int, int] | None = None

    if session_ref is not None:
      self._write_meta_sidecar(session_ref)
    self.repair_manifest()

  @staticmethod
  def path_for_session(base_dir: str | Path, session_ref: AgentSessionRef) -> Path:
    expected_agent_session_id = resolve_agent_session_id(session_ref.user_id, session_ref.agent_id)
    if session_ref.agent_session_id != expected_agent_session_id:
      raise ValueError("agent_session_id does not match canonical resolution")
    agent_dir = Path(base_dir).expanduser() / slugify(session_ref.agent_id)
    return agent_dir / f"{expected_agent_session_id}.jsonl"

  def _write_meta_sidecar(self, session_ref: AgentSessionRef) -> None:
    meta_path = self.path.with_suffix(".meta.json")
    if meta_path.exists():
      return
    try:
      from .product_config import gateway_product_id

      _atomic_write_sidecar(
        meta_path,
        self._active_sidecar_payload(
          {
            "agent_session_id": session_ref.agent_session_id,
            "agent_id": session_ref.agent_id,
            "user_id": session_ref.user_id,
            "product_id": gateway_product_id() or None,
            "file_kind": "canonical",
            "channel": None,
            "profile": None,
            "created_at": _now_iso(),
          },
          active_generation=0,
        ),
      )
    except Exception:
      log.warning("Sidecar write failed for %s (telemetry-only)", meta_path, exc_info=True)

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

  def repair_manifest(self) -> None:
    with self.append_mutex_path.open("a+b") as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        self._repair_manifest_locked()
      finally:
        fcntl.flock(mutex_file.fileno(), fcntl.LOCK_UN)

  def _append_sync(self, event: dict[str, Any]) -> LogEntry:
    event_payload = dict(event)
    event_payload["event_schema_version"] = EVENT_SCHEMA_VERSION

    with self.append_mutex_path.open("a+b") as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        self._repair_manifest_locked()
        self._rotate_active_if_needed_locked()
        with self.path.open("a+b") as handle:
          handle.seek(0, os.SEEK_END)
          file_size = handle.tell()
          latest_seq = self._latest_seq_for_append(handle)
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
          os.fsync(handle.fileno())
        self._update_manifest_latest_seq_locked(seq)
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
    for segment in self._segment_view_sync():
      if spec.before_seq is not None and segment.first_seq > spec.before_seq:
        break
      if spec.after_seq is not None and segment.last_seq < spec.after_seq:
        continue
      remaining = None if limit is None else max(0, limit - len(results))
      if remaining == 0:
        break
      results.extend(self._query_asc_file_sync(segment, spec, remaining))
      if limit is not None and len(results) >= limit:
        break
    return results

  def _query_asc_file_sync(self, segment: _Segment, spec: _QuerySpec, limit: int | None) -> list[LogEntry]:
    results: list[LogEntry] = []
    active_identity = self._active_file_identity() if segment.active else None
    start_offset = self._starting_offset_for_seq(spec.after_seq, active_identity=active_identity) if segment.active else 0
    should_mark_complete = False
    with segment.path.open("rb") as handle:
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
        if segment.active:
          self._update_cache(seq=entry.seq, offset=offset, active_identity=active_identity)
        if spec.after_seq is not None and entry.seq < spec.after_seq:
          continue
        if spec.before_seq is not None and entry.seq > spec.before_seq:
          break
        if not self._matches(entry, spec):
          continue
        results.append(entry)
        if limit is not None and len(results) >= limit:
          break
    if segment.active and should_mark_complete:
      self._mark_active_cache_complete(active_identity=active_identity, file_size=file_size)
    return results

  def _query_desc_sync(self, spec: _QuerySpec, limit: int | None) -> list[LogEntry]:
    results: list[LogEntry] = []
    for segment in reversed(self._segment_view_sync()):
      if spec.before_seq is not None and segment.first_seq > spec.before_seq:
        continue
      if spec.after_seq is not None and segment.last_seq < spec.after_seq:
        break
      remaining = None if limit is None else max(0, limit - len(results))
      if remaining == 0:
        break
      results.extend(self._query_desc_file_sync(segment, spec, remaining))
      if limit is not None and len(results) >= limit:
        break
    return results

  def _query_desc_file_sync(self, segment: _Segment, spec: _QuerySpec, limit: int | None) -> list[LogEntry]:
    results: list[LogEntry] = []
    with segment.path.open("rb") as handle:
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
    manifest = self._load_manifest()
    manifest_latest = int(manifest.get("latest_seq") or 0) if manifest is not None else 0
    with self.path.open("rb") as handle:
      return max(manifest_latest, self._latest_seq_from_handle(handle))

  def _latest_seq_for_append(self, active_handle: Any) -> int:
    manifest = self._load_manifest()
    manifest_latest = int(manifest.get("latest_seq") or 0) if manifest is not None else 0
    return max(manifest_latest, self._latest_seq_from_handle(active_handle))

  def _latest_seq_from_handle(self, handle: Any) -> int:
    for raw, is_last_line in self._iter_lines_reverse(handle):
      entry = self._parse_entry(raw, is_last_line=is_last_line)
      if entry is not None:
        return entry.seq
    return 0

  def _active_file_identity(self) -> tuple[int, int, int, int] | None:
    try:
      stat = self.path.stat()
    except OSError:
      return None
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))

  def _refresh_active_cache_identity_locked(self, active_identity: tuple[int, int, int, int] | None) -> None:
    if self._cache_file_identity == active_identity:
      return
    self._seq_offsets.clear()
    self._max_cached_seq = 0
    self._cache_complete = False
    self._cache_file_identity = active_identity

  def _starting_offset_for_seq(
    self,
    after_seq: int | None,
    *,
    active_identity: tuple[int, int, int, int] | None = None,
  ) -> int:
    if after_seq is None or after_seq <= 1:
      return 0
    if active_identity is None:
      active_identity = self._active_file_identity()
    with self._cache_lock:
      self._refresh_active_cache_identity_locked(active_identity)
      if after_seq in self._seq_offsets:
        return self._seq_offsets[after_seq]
      if self._cache_complete and after_seq > self._max_cached_seq:
        return active_identity[2] if active_identity is not None else 0
      if self._max_cached_seq > 0 and after_seq > self._max_cached_seq:
        return self._seq_offsets.get(self._max_cached_seq, 0)
    return 0

  def _update_cache(
    self,
    *,
    seq: int,
    offset: int,
    active_identity: tuple[int, int, int, int] | None = None,
  ) -> None:
    if active_identity is None:
      active_identity = self._active_file_identity()
    with self._cache_lock:
      self._refresh_active_cache_identity_locked(active_identity)
      existing = self._seq_offsets.get(seq)
      if existing is None or offset < existing:
        self._seq_offsets[seq] = offset
      if seq > self._max_cached_seq:
        self._max_cached_seq = seq

  def _mark_active_cache_complete(
    self,
    *,
    active_identity: tuple[int, int, int, int] | None,
    file_size: int,
  ) -> None:
    current_identity = self._active_file_identity()
    if active_identity is None or current_identity != active_identity or current_identity[2] != file_size:
      return
    with self._cache_lock:
      self._refresh_active_cache_identity_locked(active_identity)
      self._cache_complete = True

  def _clear_active_cache(self) -> None:
    with self._cache_lock:
      self._seq_offsets.clear()
      self._max_cached_seq = 0
      self._cache_complete = False
      self._cache_file_identity = None

  def _configured_max_active_bytes(self) -> int | None:
    raw = os.getenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES")
    if raw is None or str(raw).strip() == "":
      return None
    try:
      value = int(raw)
    except ValueError:
      log.warning("Ignoring invalid AGENT_SESSION_LOG_MAX_ACTIVE_BYTES=%r", raw)
      return None
    return value if value > 0 else None

  def _logical_stream_id(self) -> str:
    return str(self.path.resolve())

  def _stream_hash(self) -> str:
    return hashlib.sha1(self._logical_stream_id().encode("utf-8")).hexdigest()[:16]

  def _telemetry_source_id(self, role: str, suffix: str) -> str:
    return f"agent_session_log:{self._stream_hash()}:{role}:{suffix}"

  def _active_sidecar_payload(self, base: dict[str, Any], *, active_generation: int) -> dict[str, Any]:
    payload = dict(base)
    payload.update(
      {
        "schema_version": 2,
        "file_role": "active",
        "logical_stream_id": self._logical_stream_id(),
        "telemetry_source_id": self._telemetry_source_id("active", f"{active_generation:06d}"),
        "active_generation": active_generation,
      }
    )
    return payload

  def _load_sidecar_payload(self) -> dict[str, Any] | None:
    meta_path = self.path.with_suffix(".meta.json")
    if not meta_path.exists():
      return None
    try:
      payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      return None
    return payload if isinstance(payload, dict) else None

  def _segment_sidecar_payload(
    self,
    base: dict[str, Any],
    *,
    segment_id: str,
    first_seq: int,
    last_seq: int,
    active_generation: int,
    rotated_from_file_identity: dict[str, int],
  ) -> dict[str, Any]:
    payload = {
      "schema_version": 2,
      "agent_session_id": str(base.get("agent_session_id") or self.path.stem),
      "agent_id": base.get("agent_id"),
      "user_id": base.get("user_id"),
      "product_id": base.get("product_id"),
      "file_kind": base.get("file_kind") or "canonical",
      "channel": base.get("channel"),
      "profile": base.get("profile"),
      "created_at": base.get("created_at") or _now_iso(),
      "file_role": "segment",
      "logical_stream_id": self._logical_stream_id(),
      "telemetry_source_id": self._telemetry_source_id("segment", segment_id),
      "active_generation": active_generation,
      "segment_id": segment_id,
      "first_seq": first_seq,
      "last_seq": last_seq,
      "rotated_from_source_id": self._telemetry_source_id("active", f"{active_generation:06d}"),
      "rotated_from_path": str(self.path),
      "rotated_from_file_identity": rotated_from_file_identity,
    }
    return payload

  def _fallback_sidecar_base(self) -> dict[str, Any]:
    user_id = "unknown"
    if self.path.stem.startswith("agentsess_"):
      remainder = self.path.stem[len("agentsess_") :]
      if "_" in remainder:
        user_id = remainder.rsplit("_", 1)[1] or "unknown"
    return {
      "agent_session_id": self.path.stem,
      "agent_id": self.path.parent.name or "unknown",
      "user_id": user_id,
      "product_id": None,
      "file_kind": "canonical",
      "channel": None,
      "profile": None,
      "created_at": _now_iso(),
    }

  def _sidecar_base_from_segment_meta(self, meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
      return None
    base = {
      key: meta.get(key)
      for key in ("agent_session_id", "agent_id", "user_id", "product_id", "file_kind", "channel", "profile", "created_at")
      if key in meta
    }
    return base or None

  def _sidecar_base_for_repair(self, segment_metas: list[dict[str, Any]]) -> dict[str, Any]:
    active_base = self._load_sidecar_payload()
    if active_base is not None:
      return active_base
    for meta in segment_metas:
      segment_base = self._sidecar_base_from_segment_meta(meta)
      if segment_base is not None:
        return segment_base
    return self._fallback_sidecar_base()

  def _file_identity(self, path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
      "st_dev": int(stat.st_dev),
      "st_ino": int(stat.st_ino),
      "size": int(stat.st_size),
      "mtime_ns": int(stat.st_mtime_ns),
    }

  def _new_manifest(self) -> dict[str, Any]:
    return {
      "schema_version": _MANIFEST_SCHEMA_VERSION,
      "logical_stream_id": self._logical_stream_id(),
      "active_path": f"../{self.path.name}",
      "active_generation": 0,
      "active_telemetry_source_id": self._telemetry_source_id("active", "000000"),
      "segments": [],
      "min_seq_available": 1,
      "latest_seq": 0,
    }

  def _load_manifest(self) -> dict[str, Any] | None:
    if not self.manifest_path.exists():
      return None
    try:
      payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      log.warning("Ignoring unreadable AgentSessionLog manifest: %s", self.manifest_path)
      return None
    if not isinstance(payload, dict):
      log.warning("Ignoring malformed AgentSessionLog manifest: %s", self.manifest_path)
      return None
    return payload

  def _write_manifest(self, manifest: dict[str, Any]) -> None:
    _atomic_write_json(self.manifest_path, manifest)

  def _segment_filename_parts(self, path: Path) -> tuple[str, int, int, int] | None:
    match = _SEGMENT_FILE_RE.fullmatch(path.name)
    if match is None:
      return None
    first_seq = int(match.group("first"))
    last_seq = int(match.group("last"))
    generation = int(match.group("generation"))
    segment_id = f"{first_seq:012d}-{last_seq:012d}-g{generation:06d}"
    return segment_id, first_seq, last_seq, generation

  def _segment_path_from_manifest(self, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else self.segments_dir / path

  def _seq_range_for_file(self, path: Path) -> tuple[int, int]:
    first = 0
    last = 0
    if not path.exists() or path.stat().st_size == 0:
      return 0, 0
    with path.open("rb") as handle:
      file_size = handle.seek(0, os.SEEK_END)
      handle.seek(0)
      while True:
        raw = handle.readline()
        if not raw:
          break
        entry = self._parse_entry(raw, is_last_line=handle.tell() >= file_size)
        if entry is None:
          continue
        if first == 0:
          first = entry.seq
        last = entry.seq
    return first, last

  def _seq_bounds_for_file(self, path: Path) -> tuple[int, int]:
    if not path.exists() or path.stat().st_size == 0:
      return 0, 0
    with path.open("rb") as handle:
      file_size = handle.seek(0, os.SEEK_END)
      handle.seek(0)
      first = 0
      while True:
        raw = handle.readline()
        if not raw:
          break
        entry = self._parse_entry(raw, is_last_line=handle.tell() >= file_size)
        if entry is not None:
          first = entry.seq
          break
      if first == 0:
        return 0, 0
      last = self._latest_seq_from_handle(handle)
    return first, last

  def _segment_descriptor_from_meta(
    self,
    segment_path: Path,
    meta: dict[str, Any],
    *,
    fallback_generation: int | None = None,
  ) -> dict[str, Any] | None:
    try:
      segment_id = str(meta["segment_id"])
      first_seq = int(meta["first_seq"])
      last_seq = int(meta["last_seq"])
    except (KeyError, TypeError, ValueError):
      filename_parts = self._segment_filename_parts(segment_path)
      if filename_parts is None:
        return None
      segment_id, first_seq, last_seq, fallback_generation = filename_parts
    if first_seq <= 0 or last_seq < first_seq:
      return None
    active_generation = meta.get("active_generation")
    if active_generation is None:
      active_generation = fallback_generation
    try:
      generation = int(active_generation)
    except (TypeError, ValueError):
      generation = 0
    stat = segment_path.stat()
    return {
      "segment_id": segment_id,
      "path": segment_path.name,
      "first_seq": first_seq,
      "last_seq": last_seq,
      "bytes": int(stat.st_size),
      "telemetry_source_id": str(meta.get("telemetry_source_id") or self._telemetry_source_id("segment", segment_id)),
      "rotated_from_source_id": str(meta.get("rotated_from_source_id") or self._telemetry_source_id("active", f"{generation:06d}")),
      "rotated_from_path": str(meta.get("rotated_from_path") or f"../{self.path.name}"),
      "rotated_from_file_identity": (
        meta.get("rotated_from_file_identity")
        if isinstance(meta.get("rotated_from_file_identity"), dict)
        else self._file_identity(segment_path)
      ),
      "created_at": str(meta.get("created_at") or _now_iso()),
      "closed_at": str(meta.get("closed_at") or _now_iso()),
    }

  def _segment_descriptor_from_file(self, segment_path: Path, base_meta: dict[str, Any]) -> dict[str, Any] | None:
    filename_parts = self._segment_filename_parts(segment_path)
    if filename_parts is None:
      return None
    segment_id, filename_first, filename_last, generation = filename_parts
    scanned_first, scanned_last = self._seq_range_for_file(segment_path)
    first_seq = scanned_first or filename_first
    last_seq = scanned_last or filename_last
    if first_seq <= 0 or last_seq < first_seq:
      return None
    identity = self._file_identity(segment_path)
    meta = self._segment_sidecar_payload(
      base_meta,
      segment_id=segment_id,
      first_seq=first_seq,
      last_seq=last_seq,
      active_generation=generation,
      rotated_from_file_identity=identity,
    )
    _atomic_write_json(segment_path.with_suffix(".meta.json"), meta)
    return self._segment_descriptor_from_meta(segment_path, meta, fallback_generation=generation)

  def _ensure_segment_sidecar_from_descriptor(
    self,
    segment_path: Path,
    descriptor: dict[str, Any],
    base_meta: dict[str, Any],
  ) -> dict[str, Any] | None:
    sidecar_path = segment_path.with_suffix(".meta.json")
    meta = _read_json_dict(sidecar_path)
    if isinstance(meta, dict) and meta.get("schema_version") == 2 and meta.get("file_role") == "segment":
      return meta
    try:
      segment_id = str(descriptor["segment_id"])
      first_seq = int(descriptor["first_seq"])
      last_seq = int(descriptor["last_seq"])
      generation = int(segment_id.rsplit("-g", 1)[1])
    except (KeyError, IndexError, TypeError, ValueError):
      filename_parts = self._segment_filename_parts(segment_path)
      if filename_parts is None:
        return None
      segment_id, first_seq, last_seq, generation = filename_parts
    identity = descriptor.get("rotated_from_file_identity")
    if not isinstance(identity, dict):
      identity = self._file_identity(segment_path)
    clean_identity: dict[str, int] = {}
    for key in ("st_dev", "st_ino", "size", "mtime_ns"):
      try:
        clean_identity[key] = int(identity[key])
      except (KeyError, TypeError, ValueError):
        clean_identity = self._file_identity(segment_path)
        break
    meta = self._segment_sidecar_payload(
      base_meta,
      segment_id=segment_id,
      first_seq=first_seq,
      last_seq=last_seq,
      active_generation=generation,
      rotated_from_file_identity=clean_identity,
    )
    _atomic_write_json(sidecar_path, meta)
    return meta

  def _repair_manifest_locked(self) -> None:
    segment_paths = sorted(
      path
      for path in self.segments_dir.glob("*.jsonl")
      if path.is_file() and self._segment_filename_parts(path) is not None
    )
    manifest = self._load_manifest()
    if manifest is None and not segment_paths:
      return

    segment_metas = [
      meta
      for path in segment_paths
      if isinstance(meta := _read_json_dict(path.with_suffix(".meta.json")), dict)
    ]
    base_meta = self._sidecar_base_for_repair(segment_metas)
    manifest = manifest or self._new_manifest()

    repaired_by_id: dict[str, dict[str, Any]] = {}
    for raw in manifest.get("segments", []):
      if not isinstance(raw, dict):
        continue
      try:
        segment_id = str(raw["segment_id"])
        segment_path = self._segment_path_from_manifest(raw["path"])
      except (KeyError, TypeError, ValueError):
        continue
      if not segment_path.exists():
        continue
      meta = self._ensure_segment_sidecar_from_descriptor(segment_path, raw, base_meta)
      descriptor = self._segment_descriptor_from_meta(segment_path, meta or raw)
      if descriptor is not None:
        repaired_by_id[segment_id] = descriptor

    for segment_path in segment_paths:
      meta = _read_json_dict(segment_path.with_suffix(".meta.json"))
      descriptor = None
      if isinstance(meta, dict) and meta.get("schema_version") == 2 and meta.get("file_role") == "segment":
        descriptor = self._segment_descriptor_from_meta(segment_path, meta)
      if descriptor is None:
        descriptor = self._segment_descriptor_from_file(segment_path, base_meta)
      if descriptor is not None:
        repaired_by_id[str(descriptor["segment_id"])] = descriptor

    segments = sorted(repaired_by_id.values(), key=lambda item: (int(item.get("first_seq") or 0), str(item.get("segment_id") or "")))
    active_first, active_last = self._seq_bounds_for_file(self.path)
    segment_latest = max((int(item.get("last_seq") or 0) for item in segments), default=0)
    segment_min = min((int(item.get("first_seq") or 0) for item in segments if int(item.get("first_seq") or 0) > 0), default=0)
    generations = [int(str(item["segment_id"]).rsplit("-g", 1)[1]) + 1 for item in segments]
    active_generation = max(int(manifest.get("active_generation") or 0), *generations) if generations else int(manifest.get("active_generation") or 0)

    manifest.update(
      {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "logical_stream_id": self._logical_stream_id(),
        "agent_session_id": str(base_meta.get("agent_session_id") or self.path.stem),
        "active_path": f"../{self.path.name}",
        "active_generation": active_generation,
        "active_telemetry_source_id": self._telemetry_source_id("active", f"{active_generation:06d}"),
        "segments": segments,
        "min_seq_available": segment_min or active_first or int(manifest.get("min_seq_available") or 1),
        "latest_seq": max(int(manifest.get("latest_seq") or 0), segment_latest, active_last),
      }
    )
    if segments or self.manifest_path.exists():
      self._write_manifest(manifest)

    if not self.path.exists():
      self.path.touch()
    active_meta_path = self.path.with_suffix(".meta.json")
    active_meta = _read_json_dict(active_meta_path)
    if (segments or self.manifest_path.exists()) and (
      not isinstance(active_meta, dict)
      or active_meta.get("schema_version") != 2
      or active_meta.get("file_role") != "active"
      or int(active_meta.get("active_generation") or -1) != active_generation
    ):
      _atomic_write_json(active_meta_path, self._active_sidecar_payload(base_meta, active_generation=active_generation))

  def _segment_view_sync(self) -> list[_Segment]:
    manifest = self._load_manifest()
    if manifest is None:
      first, last = self._seq_bounds_for_file(self.path)
      return [_Segment(self.path, first, last, active=True)] if last else []

    segments: list[_Segment] = []
    for item in manifest.get("segments", []):
      if not isinstance(item, dict):
        continue
      try:
        segment = _Segment(
          self._segment_path_from_manifest(item["path"]),
          int(item["first_seq"]),
          int(item["last_seq"]),
          active=False,
        )
      except (KeyError, TypeError, ValueError):
        continue
      if segment.path.exists():
        segments.append(segment)

    first, last = self._seq_bounds_for_file(self.path)
    if last:
      segments.append(_Segment(self.path, first, last, active=True))
    segments.sort(key=lambda item: item.first_seq)
    return segments

  def _rotate_active_if_needed_locked(self) -> None:
    if self._max_active_bytes is None:
      return
    try:
      active_size = self.path.stat().st_size
    except FileNotFoundError:
      self.path.touch(exist_ok=True)
      return
    if active_size == 0 or active_size <= self._max_active_bytes:
      return

    first_seq, last_seq = self._seq_bounds_for_file(self.path)
    if first_seq <= 0 or last_seq <= 0:
      return

    manifest = self._load_manifest() or self._new_manifest()
    active_generation = int(manifest.get("active_generation") or 0)
    segment_id = f"{first_seq:012d}-{last_seq:012d}-g{active_generation:06d}"
    self.segments_dir.mkdir(parents=True, exist_ok=True)
    segment_path = self.segments_dir / f"{segment_id}.jsonl"
    file_identity = self._file_identity(self.path)
    base_meta = self._load_sidecar_payload() or {}

    os.replace(self.path, segment_path)
    _fsync_parent_dir(self.segments_dir)
    segment_meta_path = segment_path.with_suffix(".meta.json")
    _atomic_write_json(
      segment_meta_path,
      self._segment_sidecar_payload(
        base_meta,
        segment_id=segment_id,
        first_seq=first_seq,
        last_seq=last_seq,
        active_generation=active_generation,
        rotated_from_file_identity=file_identity,
      ),
    )

    segments = [item for item in manifest.get("segments", []) if isinstance(item, dict)]
    segments.append(
      {
        "segment_id": segment_id,
        "path": segment_path.name,
        "first_seq": first_seq,
        "last_seq": last_seq,
        "bytes": active_size,
        "telemetry_source_id": self._telemetry_source_id("segment", segment_id),
        "rotated_from_source_id": self._telemetry_source_id("active", f"{active_generation:06d}"),
        "rotated_from_path": f"../{self.path.name}",
        "rotated_from_file_identity": file_identity,
        "created_at": _now_iso(),
        "closed_at": _now_iso(),
      }
    )
    next_generation = active_generation + 1
    manifest.update(
      {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "logical_stream_id": self._logical_stream_id(),
        "active_path": f"../{self.path.name}",
        "active_generation": next_generation,
        "active_telemetry_source_id": self._telemetry_source_id("active", f"{next_generation:06d}"),
        "segments": segments,
        "min_seq_available": int(manifest.get("min_seq_available") or 1),
        "latest_seq": max(int(manifest.get("latest_seq") or 0), last_seq),
      }
    )
    self._write_manifest(manifest)

    self.path.touch(exist_ok=True)
    if base_meta:
      _atomic_write_json(self.path.with_suffix(".meta.json"), self._active_sidecar_payload(base_meta, active_generation=next_generation))
    self._clear_active_cache()

  def _update_manifest_latest_seq_locked(self, seq: int) -> None:
    manifest = self._load_manifest()
    if manifest is None:
      return
    if seq <= int(manifest.get("latest_seq") or 0):
      return
    manifest["latest_seq"] = seq
    self._write_manifest(manifest)

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
  "agent_session_logical_path_for_jsonl",
  "_atomic_write_sidecar",
  "resolve_agent_session_id",
  "slugify",
]
