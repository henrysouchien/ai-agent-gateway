from __future__ import annotations

import asyncio
from copy import deepcopy
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import fcntl

from .agent_session_log_cache import ActiveFileIdentity, ActiveFileOffsetCache
from . import agent_session_log_sidecars as _sidecar_helpers
from . import agent_session_log_rotation as _rotation_helpers
from .agent_session_log_records import (
  EVENT_SCHEMA_VERSION,
  AgentSessionRef,
  LogEntry,
  QueryCursor,
  _MANIFEST_SCHEMA_VERSION,
  _QuerySpec,
  _REVERSE_SCAN_CHUNK_SIZE,
  _SEGMENT_FILE_RE,
  _Segment,
  _atomic_write_json,
  _atomic_write_sidecar,
  _contains_text,
  _encode_entry,
  _event_has_error,
  _iter_lines_reverse,
  _matches_entry,
  _now_iso,
  _parse_entry,
  _read_json_dict,
  agent_session_logical_path_for_jsonl,
  resolve_agent_session_id,
  slugify,
)
from .events import AgentCompletionEvent, event_from_dict, event_to_dict
from .openai_history_fence import (
  DURABLE_HISTORY_VERSION_KEY,
  RESPONSES_HISTORY_VERSION,
  contains_openai_responses_history,
)
from .session import GatewaySession
from .secret_boundary import sanitize_tool_event

log = logging.getLogger("agent_gateway.agent_session_log")


class ReplayStorageSecurityError(RuntimeError):
  """Replay storage could not be made private before opaque state persistence."""


class IdempotentEventConflictError(RuntimeError):
  """A durable semantic event identity was reused for another payload."""


def _canonical_idempotent_event(
  event: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
  """Return the typed, timestamp-free identity projection for one event."""

  if event.get("type") != "agent_completion":
    return None
  try:
    typed = event_from_dict(event)
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(
      "agent_completion must be a canonical typed durable event"
    ) from exc
  if not isinstance(typed, AgentCompletionEvent):  # pragma: no cover
    raise TypeError("agent_completion parsed as an unexpected event type")
  canonical = event_to_dict(typed)
  canonical.pop("ts", None)
  return typed.event_id, typed.fingerprint, canonical


class AgentSessionLog:
  """Durable JSONL-backed event log for one `(user_id, agent_id)` pair."""

  def __init__(
    self,
    path: str | Path | None = None,
    *,
    session_ref: AgentSessionRef | None = None,
    base_dir: str | Path | None = None,
    gateway_session: GatewaySession | None = None,
  ) -> None:
    if (
      gateway_session is not None
      and type(gateway_session) is not GatewaySession
    ):
      raise TypeError(
        "AgentSessionLog gateway_session must be exact GatewaySession"
      )
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
    self._pending_append_futures: set[asyncio.Future[LogEntry]] = set()
    self._gateway_session = gateway_session

    self.write_lease_path = self.path.with_name(f"{self.path.name}.write_lease")
    self.append_mutex_path = self.path.with_name(f"{self.path.name}.append_mutex")
    self.write_lease_meta_path = self.path.with_name(f"{self.path.name}.write_lease.meta")

    self._active_offset_cache = ActiveFileOffsetCache()

    if session_ref is not None:
      self._write_meta_sidecar(session_ref)
    self.repair_manifest()

  @property
  def gateway_session(self) -> GatewaySession | None:
    return self._gateway_session

  @staticmethod
  def path_for_session(base_dir: str | Path, session_ref: AgentSessionRef) -> Path:
    expected_agent_session_id = resolve_agent_session_id(session_ref.user_id, session_ref.agent_id)
    if session_ref.agent_session_id != expected_agent_session_id:
      raise ValueError("agent_session_id does not match canonical resolution")
    agent_dir = Path(base_dir).expanduser() / slugify(session_ref.agent_id)
    return agent_dir / f"{expected_agent_session_id}.jsonl"

  def _write_meta_sidecar(self, session_ref: AgentSessionRef) -> None:
    _sidecar_helpers.write_meta_sidecar(
      self.path,
      session_ref,
      active_sidecar_payload_fn=lambda base: self._active_sidecar_payload(base, active_generation=0),
      atomic_write_sidecar_fn=_atomic_write_sidecar,
      now_iso_fn=_now_iso,
      logger=log,
    )

  async def append(self, event: dict[str, Any]) -> LogEntry:
    loop = asyncio.get_running_loop()
    append_future = loop.run_in_executor(
      None,
      self._append_sync,
      dict(event),
    )
    self._pending_append_futures.add(append_future)
    append_future.add_done_callback(
      self._pending_append_futures.discard
    )
    return await asyncio.shield(append_future)

  def append_sync(self, event: dict[str, Any]) -> LogEntry:
    """Durably append without yielding, for terminal settlement barriers."""

    return self._append_sync(dict(event))

  @property
  def pending_append_futures(
    self,
  ) -> tuple[asyncio.Future[LogEntry], ...]:
    """Return unsettled synchronous appends for writer-lease fencing."""

    return tuple(
      future
      for future in self._pending_append_futures
      if not future.done()
    )

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

  def query_sync(
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
    """Synchronously read exact durable envelopes during fenced settlement."""

    spec = _QuerySpec(
      event_types=(
        set(event_types)
        if event_types is not None
        else None
      ),
      tool_name=tool_name,
      tool_call_id=tool_call_id,
      sub_agent_id=sub_agent_id,
      runner_id=runner_id,
      role=role,
      after_seq=after_seq,
      before_seq=before_seq,
      after_ts=after_ts,
      before_ts=before_ts,
      contains_text=(
        contains_text.lower()
        if contains_text is not None
        else None
      ),
      has_error=has_error,
    )
    entries, next_cursor = self._query_sync(
      spec,
      order,
      limit,
      cursor,
    )
    return (
      [
        LogEntry(
          seq=entry.seq,
          timestamp=entry.timestamp,
          event=deepcopy(entry.event),
        )
        for entry in entries
      ],
      next_cursor,
    )

  async def iter_from(self, after_seq: int = 0) -> AsyncIterator[LogEntry]:
    entries, _cursor = await self.query(after_seq=max(after_seq, 0) + 1, order="asc")
    for entry in entries:
      yield entry

  async def latest_seq(self) -> int:
    return await asyncio.to_thread(self._latest_seq_sync)

  async def ensure_private_replay_storage(self) -> None:
    await asyncio.to_thread(self._ensure_private_replay_storage_sync)

  @staticmethod
  def _secure_replay_path(path: Path, mode: int, *, directory: bool = False) -> None:
    try:
      info = path.lstat()
    except FileNotFoundError:
      if directory:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
      else:
        path.touch(mode=mode, exist_ok=True)
      info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
      raise ReplayStorageSecurityError(f"OpenAI replay storage path is a symlink: {path}")
    if info.st_uid != os.geteuid():
      raise ReplayStorageSecurityError(f"OpenAI replay storage is not owned by the service user: {path}")
    if directory and not stat.S_ISDIR(info.st_mode):
      raise ReplayStorageSecurityError(f"OpenAI replay storage directory is not a directory: {path}")
    if not directory and not stat.S_ISREG(info.st_mode):
      raise ReplayStorageSecurityError(f"OpenAI replay storage file is not regular: {path}")
    try:
      path.chmod(mode, follow_symlinks=False)
    except (NotImplementedError, OSError) as exc:
      raise ReplayStorageSecurityError(f"Unable to secure OpenAI replay storage path: {path}") from exc
    if stat.S_IMODE(path.lstat().st_mode) != mode:
      raise ReplayStorageSecurityError(f"OpenAI replay storage permissions remain unsafe: {path}")

  def _replay_storage_files(self) -> list[Path]:
    candidates = [
      self.path,
      self.path.with_suffix(".meta.json"),
      self.write_lease_path,
      self.append_mutex_path,
      self.write_lease_meta_path,
      self.manifest_path,
    ]
    candidates.extend(self.path.parent.glob(f".{self.path.name}*.tmp"))
    if self.segments_dir.exists():
      candidates.extend(path for path in self.segments_dir.iterdir())
    return [path for path in candidates if path.exists()]

  def _ensure_private_replay_storage_locked(self) -> None:
    self._secure_replay_path(self.path.parent, 0o700, directory=True)
    self._secure_replay_path(self.segments_dir, 0o700, directory=True)
    for path in self._replay_storage_files():
      self._secure_replay_path(path, 0o600)

  def _ensure_private_replay_storage_sync(self) -> None:
    if self.append_mutex_path.is_symlink():
      raise ReplayStorageSecurityError(
        f"OpenAI replay storage path is a symlink: {self.append_mutex_path}"
      )
    with self.append_mutex_path.open("a+b") as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        self._ensure_private_replay_storage_locked()
      finally:
        fcntl.flock(mutex_file.fileno(), fcntl.LOCK_UN)

  def repair_manifest(self) -> None:
    with self.append_mutex_path.open("a+b") as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        self._repair_manifest_locked()
      finally:
        fcntl.flock(mutex_file.fileno(), fcntl.LOCK_UN)

  def _append_sync(self, event: dict[str, Any]) -> LogEntry:
    event_payload = sanitize_tool_event(event, sink="session_log")
    event_payload["event_schema_version"] = EVENT_SCHEMA_VERSION
    idempotent = _canonical_idempotent_event(event_payload)
    contains_responses_state = contains_openai_responses_history(event_payload)
    if contains_responses_state:
      event_payload[DURABLE_HISTORY_VERSION_KEY] = RESPONSES_HISTORY_VERSION

    if contains_responses_state:
      self._ensure_private_replay_storage_sync()

    with self.append_mutex_path.open("a+b") as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        self._repair_manifest_locked()
        if idempotent is not None:
          event_id, fingerprint, canonical = idempotent
          candidates = self._query_asc_sync(
            _QuerySpec(
              event_types={"agent_completion"},
              tool_name=None,
              tool_call_id=None,
              sub_agent_id=None,
              runner_id=None,
              role=None,
              after_seq=None,
              before_seq=None,
              after_ts=None,
              before_ts=None,
              contains_text=event_id.lower(),
              has_error=None,
            ),
            None,
          )
          matches = [
            candidate
            for candidate in candidates
            if candidate.event.get("event_id") == event_id
          ]
          for candidate in matches:
            try:
              existing = _canonical_idempotent_event(candidate.event)
            except (TypeError, ValueError) as exc:
              raise IdempotentEventConflictError(
                "stored agent completion has malformed identity payload: "
                f"{event_id}"
              ) from exc
            if (
              existing is None
              or existing[1] != fingerprint
              or existing[2] != canonical
            ):
              raise IdempotentEventConflictError(
                "agent completion event identity conflicts with stored "
                f"payload: {event_id}"
              )
          if matches:
            return matches[0]
        self._rotate_active_if_needed_locked()
        if contains_responses_state:
          # Rotation creates a new active file and metadata. Verify and repair
          # every resulting path before ciphertext can be appended.
          self._ensure_private_replay_storage_locked()
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

  def _active_file_identity(self) -> ActiveFileIdentity | None:
    try:
      stat = self.path.stat()
    except OSError:
      return None
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))

  def _starting_offset_for_seq(
    self,
    after_seq: int | None,
    *,
    active_identity: ActiveFileIdentity | None = None,
  ) -> int:
    if active_identity is None and after_seq is not None and after_seq > 1:
      active_identity = self._active_file_identity()
    return self._active_offset_cache.starting_offset_for_seq(after_seq, active_identity=active_identity)

  def _update_cache(
    self,
    *,
    seq: int,
    offset: int,
    active_identity: ActiveFileIdentity | None = None,
  ) -> None:
    if active_identity is None:
      active_identity = self._active_file_identity()
    self._active_offset_cache.update(seq=seq, offset=offset, active_identity=active_identity)

  def _mark_active_cache_complete(
    self,
    *,
    active_identity: ActiveFileIdentity | None,
    file_size: int,
  ) -> None:
    self._active_offset_cache.mark_complete(
      active_identity=active_identity,
      current_identity=self._active_file_identity(),
      file_size=file_size,
    )

  def _clear_active_cache(self) -> None:
    self._active_offset_cache.clear()

  def _configured_max_active_bytes(self) -> int:
    raw = os.getenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES")
    if raw is None or str(raw).strip() == "":
      return 64 * 1024 * 1024
    try:
      value = int(raw)
    except ValueError:
      log.warning("Invalid AGENT_SESSION_LOG_MAX_ACTIVE_BYTES=%r; using 64 MiB", raw)
      return 64 * 1024 * 1024
    if value <= 0:
      log.warning("Non-positive AGENT_SESSION_LOG_MAX_ACTIVE_BYTES=%r; using 64 MiB", raw)
      return 64 * 1024 * 1024
    return value

  def _logical_stream_id(self) -> str:
    return _sidecar_helpers.logical_stream_id(self.path)

  def _stream_hash(self) -> str:
    return _sidecar_helpers.stream_hash(logical_stream_id_fn=self._logical_stream_id)

  def _telemetry_source_id(self, role: str, suffix: str) -> str:
    return _sidecar_helpers.telemetry_source_id(role, suffix, stream_hash_fn=self._stream_hash)

  def _active_sidecar_payload(self, base: dict[str, Any], *, active_generation: int) -> dict[str, Any]:
    return _sidecar_helpers.active_sidecar_payload(
      base,
      active_generation=active_generation,
      logical_stream_id_fn=self._logical_stream_id,
      telemetry_source_id_fn=self._telemetry_source_id,
    )

  def _load_sidecar_payload(self) -> dict[str, Any] | None:
    return _sidecar_helpers.load_sidecar_payload(self.path)

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
    return _sidecar_helpers.segment_sidecar_payload(
      self.path,
      base,
      segment_id=segment_id,
      first_seq=first_seq,
      last_seq=last_seq,
      active_generation=active_generation,
      rotated_from_file_identity=rotated_from_file_identity,
      logical_stream_id_fn=self._logical_stream_id,
      telemetry_source_id_fn=self._telemetry_source_id,
      now_iso_fn=_now_iso,
    )

  def _fallback_sidecar_base(self) -> dict[str, Any]:
    return _sidecar_helpers.fallback_sidecar_base(self.path, now_iso_fn=_now_iso)

  def _sidecar_base_from_segment_meta(self, meta: dict[str, Any] | None) -> dict[str, Any] | None:
    return _sidecar_helpers.sidecar_base_from_segment_meta(meta)

  def _sidecar_base_for_repair(self, segment_metas: list[dict[str, Any]]) -> dict[str, Any]:
    return _sidecar_helpers.sidecar_base_for_repair(
      segment_metas,
      load_sidecar_payload_fn=self._load_sidecar_payload,
      sidecar_base_from_segment_meta_fn=self._sidecar_base_from_segment_meta,
      fallback_sidecar_base_fn=self._fallback_sidecar_base,
    )

  def _file_identity(self, path: Path) -> dict[str, int]:
    return _sidecar_helpers.file_identity(path)

  def _new_manifest(self) -> dict[str, Any]:
    return _sidecar_helpers.new_manifest(
      self.path,
      manifest_schema_version=_MANIFEST_SCHEMA_VERSION,
      logical_stream_id_fn=self._logical_stream_id,
      telemetry_source_id_fn=self._telemetry_source_id,
    )

  def _load_manifest(self) -> dict[str, Any] | None:
    return _sidecar_helpers.load_manifest(self.manifest_path, log)

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
    _rotation_helpers.rotate_active_if_needed_locked(self)

  def _update_manifest_latest_seq_locked(self, seq: int) -> None:
    _rotation_helpers.update_manifest_latest_seq_locked(self, seq)

  def _encode_entry(self, entry: LogEntry) -> bytes:
    return _encode_entry(entry)

  def _parse_entry(self, raw: bytes, *, is_last_line: bool) -> LogEntry | None:
    return _parse_entry(raw, is_last_line=is_last_line, path=self.path, logger=log)

  def _iter_lines_reverse(self, handle: Any, chunk_size: int = _REVERSE_SCAN_CHUNK_SIZE):
    yield from _iter_lines_reverse(handle, chunk_size=chunk_size)

  def _matches(self, entry: LogEntry, spec: _QuerySpec) -> bool:
    return _matches_entry(entry, spec, contains_text=self._contains_text, event_has_error=self._event_has_error)

  def _contains_text(self, value: Any, needle: str) -> bool:
    return _contains_text(value, needle)

  def _event_has_error(self, event: dict[str, Any]) -> bool:
    return _event_has_error(event)


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
