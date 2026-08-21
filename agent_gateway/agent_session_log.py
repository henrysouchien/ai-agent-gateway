from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterator, Mapping, Set as AbstractSet
from copy import deepcopy
from dataclasses import dataclass
import errno
import json
import logging
import math
import os
import stat
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, Literal, overload

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
  _atomic_write_sidecar,
  _contains_text,
  _encode_entry,
  _event_has_error,
  _iter_lines_reverse,
  _matches_entry,
  _now_iso,
  _parse_entry,
  agent_session_logical_path_for_jsonl,
  resolve_agent_session_id,
  slugify,
)
from .events import AgentCompletionEvent, event_from_dict, event_to_dict
from .descriptor_paths import (
  DirectoryChainSecurityError,
  DirectoryIdentity,
  absolute_lexical_path as _absolute_lexical_path,
  open_directory_chain,
)
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


class AgentSessionLogEnumerationError(RuntimeError):
  """The source-owned durable session-log layout could not be enumerated."""


class AgentSessionLogStorageSecurityError(RuntimeError):
  """Durable session-log storage escaped or violated its owned layout."""


class AgentSessionLogCurrentIntegrityError(RuntimeError):
  """A source-owned current-log scan found an ambiguous durable record."""


@dataclass(frozen=True, slots=True)
class AgentSessionLogLocation(os.PathLike[str]):
  """One enumerated logical log bound to its observed storage identities."""

  path: Path
  parent_identity: DirectoryIdentity
  active_identity: tuple[int, int] | None
  segments_identity: tuple[int, int] | None

  def __fspath__(self) -> str:
    return os.fspath(self.path)


@dataclass(frozen=True, slots=True)
class AgentSessionLogSegmentRetirement:
  """Result of one storage-owner segment retirement."""

  segment_id: str
  telemetry_source_id: str
  removed_from_manifest: bool
  deleted_bytes: int


def _open_directory_chain(
  raw_path: str | Path,
  *,
  create: bool = False,
) -> tuple[int, DirectoryIdentity]:
  """Open a confined chain with the session-log owner's error contract."""

  try:
    return open_directory_chain(raw_path, create=create)
  except DirectoryChainSecurityError as exc:
    detail = {
      "descriptor-confined directory access is unavailable": (
        "durable session-log descriptor confinement is unavailable"
      ),
      "directory path is not absolute": (
        "durable session-log directory path is not absolute"
      ),
      "filesystem root is unsafe": (
        "durable session-log filesystem root is unsafe"
      ),
      "directory is missing": (
        "durable session-log directory is missing"
      ),
      "directory chain is unsafe": (
        "durable session-log directory chain is unsafe"
      ),
      "directory identity changed": (
        "durable session-log directory identity changed"
      ),
    }.get(
      str(exc),
      "durable session-log directory chain is unavailable",
    )
    raise AgentSessionLogStorageSecurityError(
      detail
    ) from exc


class AgentSessionLogWriteLeaseSet:
  """Exclusive writer-lease descriptors held for exact logical log paths."""

  def __init__(self, descriptors: tuple[int, ...]) -> None:
    self._descriptors = descriptors

  def release(self) -> None:
    descriptors, self._descriptors = self._descriptors, ()
    for descriptor in reversed(descriptors):
      try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
      finally:
        os.close(descriptor)

  def __enter__(self) -> AgentSessionLogWriteLeaseSet:
    return self

  def __exit__(self, *_args: object) -> None:
    self.release()


def try_acquire_agent_session_log_write_leases(
  log_paths: Iterable[str | Path | AgentSessionLogLocation],
) -> AgentSessionLogWriteLeaseSet | None:
  """Try exact existing writer leases, all-or-none, without blocking.

  The helper deliberately does not write admission fence records or construct
  an ``AgentSessionLog``.  It only holds the same advisory lease inode used by
  writers, allowing a caller to perform one fenced selection-and-append pass.
  """

  normalized_by_path: dict[Path, DirectoryIdentity | None] = {}
  for raw_path in log_paths:
    location = (
      raw_path
      if type(raw_path) is AgentSessionLogLocation
      else None
    )
    path = _normalized_log_path_for_lease(raw_path)
    expected = (
      location.parent_identity
      if location is not None
      else None
    )
    previous = normalized_by_path.get(path)
    if previous is not None and expected is not None and previous != expected:
      raise AgentSessionLogEnumerationError(
        "durable session-log lease locations disagree"
      )
    normalized_by_path[path] = previous or expected
  normalized = tuple(sorted(normalized_by_path.items()))
  descriptors: list[int] = []
  try:
    for log_path, expected_parent in normalized:
      descriptor = _open_agent_session_log_write_lease(
        log_path,
        expected_parent_identity=expected_parent,
      )
      try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
      except OSError as exc:
        os.close(descriptor)
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
          for acquired in reversed(descriptors):
            fcntl.flock(acquired, fcntl.LOCK_UN)
            os.close(acquired)
          descriptors = []
          return None
        raise
      descriptors.append(descriptor)
    result = AgentSessionLogWriteLeaseSet(tuple(descriptors))
    descriptors = []
    return result
  finally:
    for descriptor in reversed(descriptors):
      try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
      finally:
        os.close(descriptor)


def _normalized_log_path_for_lease(
  raw_path: str | Path | AgentSessionLogLocation,
) -> Path:
  expanded = Path(os.fspath(raw_path)).expanduser()
  if not expanded.is_absolute():
    raise AgentSessionLogEnumerationError(
      "durable session-log lease requires an exact absolute log path"
    )
  path = _absolute_lexical_path(raw_path)
  if path.name in {"", ".", ".."}:
    raise AgentSessionLogEnumerationError(
      "durable session-log lease requires an exact absolute log path"
    )
  if path.suffix != ".jsonl":
    raise AgentSessionLogEnumerationError(
      "durable session-log lease requires a JSONL logical path"
    )
  return path


def _open_agent_session_log_write_lease(
  log_path: Path,
  *,
  expected_parent_identity: DirectoryIdentity | None = None,
) -> int:
  try:
    directory_descriptor, identities = _open_directory_chain(
      log_path.parent
    )
  except AgentSessionLogStorageSecurityError as exc:
    raise AgentSessionLogEnumerationError(
      "durable session-log directory is unavailable"
    ) from exc
  if (
    expected_parent_identity is not None
    and identities != expected_parent_identity
  ):
    os.close(directory_descriptor)
    raise AgentSessionLogEnumerationError(
      "durable session-log directory chain was displaced"
    )
  descriptor = -1
  lease_name = f"{log_path.name}.write_lease"
  try:
    try:
      descriptor = os.open(
        lease_name,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_descriptor,
      )
    except OSError as exc:
      raise AgentSessionLogEnumerationError(
        "durable session-log writer lease is unavailable"
      ) from exc
    info = os.fstat(descriptor)
    if (
      not stat.S_ISREG(info.st_mode)
      or info.st_uid != os.geteuid()
      or info.st_nlink != 1
      or stat.S_IMODE(info.st_mode) != 0o600
    ):
      raise AgentSessionLogEnumerationError(
        "durable session-log writer lease is unsafe"
      )
    named = os.stat(
      lease_name,
      dir_fd=directory_descriptor,
      follow_symlinks=False,
    )
    if (
      stat.S_ISLNK(named.st_mode)
      or not stat.S_ISREG(named.st_mode)
      or named.st_uid != os.geteuid()
      or named.st_nlink != 1
      or stat.S_IMODE(named.st_mode) != 0o600
      or (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino)
    ):
      raise AgentSessionLogEnumerationError(
        "durable session-log writer lease identity changed"
      )
    result = descriptor
    descriptor = -1
    return result
  finally:
    if descriptor >= 0:
      os.close(descriptor)
    os.close(directory_descriptor)


def enumerate_agent_session_log_paths(
  base_dir: str | Path,
) -> tuple[AgentSessionLogLocation, ...]:
  """Return identity-bound logical logs in the owned storage layout.

  The filesystem layout is the only discovery authority.  Active logs are
  immediate ``*.jsonl`` children of immediate agent directories.  A sibling
  ``*.segments`` directory also names its logical active path, including the
  rotation crash window where that active file does not currently exist.
  Sidecars, manifests, and event content do not admit or exclude a stream.
  """

  root = _absolute_lexical_path(base_dir)
  root_info = _enumeration_lstat(root, missing_ok=True)
  if root_info is None:
    return ()
  _require_enumeration_directory(root, root_info)
  root_descriptor = -1
  try:
    root_descriptor, _root_identities = _open_directory_chain(root)
    opened_root = os.fstat(root_descriptor)
    if (
      opened_root.st_dev,
      opened_root.st_ino,
    ) != (root_info.st_dev, root_info.st_ino):
      raise AgentSessionLogEnumerationError(
        "durable session-log root identity changed"
      )
    agent_names = tuple(os.listdir(root_descriptor))
  except AgentSessionLogEnumerationError:
    if root_descriptor >= 0:
      os.close(root_descriptor)
      root_descriptor = -1
    raise
  except (AgentSessionLogStorageSecurityError, OSError) as exc:
    if root_descriptor >= 0:
      os.close(root_descriptor)
      root_descriptor = -1
    raise AgentSessionLogEnumerationError(
      "durable session-log layout is unavailable"
    ) from exc

  logical_locations: dict[
    Path,
    tuple[
      DirectoryIdentity,
      tuple[int, int] | None,
      tuple[int, int] | None,
    ],
  ] = {}
  try:
    for agent_name in agent_names:
      agent_info = os.stat(
        agent_name,
        dir_fd=root_descriptor,
        follow_symlinks=False,
      )
      if stat.S_ISLNK(agent_info.st_mode):
        raise AgentSessionLogEnumerationError(
          "durable session-log layout contains an unsafe agent directory"
        )
      if not stat.S_ISDIR(agent_info.st_mode):
        continue
      agent_descriptor = -1
      try:
        agent_descriptor = os.open(
          agent_name,
          os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
          | getattr(os, "O_CLOEXEC", 0),
          dir_fd=root_descriptor,
        )
        opened_agent = os.fstat(agent_descriptor)
        if (
          not stat.S_ISDIR(opened_agent.st_mode)
          or (opened_agent.st_dev, opened_agent.st_ino)
          != (agent_info.st_dev, agent_info.st_ino)
        ):
          raise AgentSessionLogEnumerationError(
            "durable session-log agent directory identity changed"
          )
        parent_identity = _root_identities + ((
          opened_agent.st_dev,
          opened_agent.st_ino,
        ),)
        child_names = tuple(os.listdir(agent_descriptor))
        for child_name in child_names:
          child_info = os.stat(
            child_name,
            dir_fd=agent_descriptor,
            follow_symlinks=False,
          )
          if child_name.endswith(".jsonl"):
            _require_enumeration_regular_info(child_info)
            logical_path = root / agent_name / child_name
            previous = logical_locations.get(logical_path)
            logical_locations[logical_path] = (
              parent_identity,
              (child_info.st_dev, child_info.st_ino),
              previous[2] if previous is not None else None,
            )
            continue
          if not child_name.endswith(".segments"):
            continue
          if (
            stat.S_ISLNK(child_info.st_mode)
            or not stat.S_ISDIR(child_info.st_mode)
          ):
            raise AgentSessionLogEnumerationError(
              "durable session-log segment directory is unsafe"
            )
          active_stem = child_name[: -len(".segments")]
          if not active_stem:
            raise AgentSessionLogEnumerationError(
              "durable session-log segment directory has no logical stream"
            )
          _validate_segment_files_at(agent_descriptor, child_name, child_info)
          active_name = f"{active_stem}.jsonl"
          try:
            active_info = os.stat(
              active_name,
              dir_fd=agent_descriptor,
              follow_symlinks=False,
            )
          except FileNotFoundError:
            active_identity = None
          else:
            _require_enumeration_regular_info(active_info)
            active_identity = (active_info.st_dev, active_info.st_ino)
          logical_path = root / agent_name / active_name
          previous = logical_locations.get(logical_path)
          logical_locations[logical_path] = (
            parent_identity,
            (
              previous[1]
              if previous is not None and previous[1] is not None
              else active_identity
            ),
            (child_info.st_dev, child_info.st_ino),
          )
      finally:
        if agent_descriptor >= 0:
          os.close(agent_descriptor)
  except AgentSessionLogEnumerationError:
    raise
  except OSError as exc:
    raise AgentSessionLogEnumerationError(
      "durable session-log layout changed during enumeration"
    ) from exc
  finally:
    if root_descriptor >= 0:
      os.close(root_descriptor)
  return tuple(
    AgentSessionLogLocation(
      path=path,
      parent_identity=identity[0],
      active_identity=identity[1],
      segments_identity=identity[2],
    )
    for path, identity in sorted(logical_locations.items())
  )


@overload
def _enumeration_lstat(
  path: Path,
  *,
  missing_ok: Literal[False] = False,
) -> os.stat_result:
  ...


@overload
def _enumeration_lstat(
  path: Path,
  *,
  missing_ok: Literal[True],
) -> os.stat_result | None:
  ...


def _enumeration_lstat(
  path: Path,
  *,
  missing_ok: bool = False,
) -> os.stat_result | None:
  try:
    return path.lstat()
  except FileNotFoundError:
    if missing_ok:
      return None
    raise AgentSessionLogEnumerationError(
      "durable session-log candidate changed during enumeration"
    ) from None
  except OSError as exc:
    raise AgentSessionLogEnumerationError(
      "durable session-log candidate is unavailable"
    ) from exc


def _require_enumeration_directory(
  path: Path,
  info: os.stat_result,
) -> None:
  if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
    raise AgentSessionLogEnumerationError(
      "durable session-log directory is unsafe"
    )


def _require_enumeration_regular_info(info: os.stat_result) -> None:
  if (
    stat.S_ISLNK(info.st_mode)
    or not stat.S_ISREG(info.st_mode)
    or info.st_nlink != 1
    or stat.S_IMODE(info.st_mode) & 0o022
  ):
    raise AgentSessionLogEnumerationError(
      "durable session-log file is unsafe"
    )


def _validate_segment_files_at(
  agent_descriptor: int,
  name: str,
  expected: os.stat_result,
) -> None:
  descriptor = -1
  try:
    descriptor = os.open(
      name,
      os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
      | getattr(os, "O_CLOEXEC", 0),
      dir_fd=agent_descriptor,
    )
    opened = os.fstat(descriptor)
    if (
      not stat.S_ISDIR(opened.st_mode)
      or (opened.st_dev, opened.st_ino)
      != (expected.st_dev, expected.st_ino)
    ):
      raise AgentSessionLogEnumerationError(
        "durable session-log segment directory identity changed"
      )
    for child_name in os.listdir(descriptor):
      if not child_name.endswith(".jsonl"):
        continue
      if _SEGMENT_FILE_RE.fullmatch(child_name) is None:
        raise AgentSessionLogEnumerationError(
          "durable session-log segment filename is invalid"
        )
      child_info = os.stat(
        child_name,
        dir_fd=descriptor,
        follow_symlinks=False,
      )
      _require_enumeration_regular_info(child_info)
  finally:
    if descriptor >= 0:
      os.close(descriptor)
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
    path: str | Path | AgentSessionLogLocation | None = None,
    *,
    session_ref: AgentSessionRef | None = None,
    base_dir: str | Path | None = None,
    gateway_session: GatewaySession | None = None,
    _create_active_on_open: bool = True,
    _repair_manifest_on_open: bool = True,
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

    enumerated_location = (
      path
      if type(path) is AgentSessionLogLocation
      else None
    )
    raw_path = (
      enumerated_location.path
      if enumerated_location is not None
      else path
    )
    assert raw_path is not None
    self.path = _absolute_lexical_path(raw_path)
    self.segments_dir = self.path.with_name(f"{self.path.stem}.segments")
    self.manifest_path = self.segments_dir / "manifest.json"
    self.write_lease_path = self.path.with_name(f"{self.path.name}.write_lease")
    self.append_mutex_path = self.path.with_name(f"{self.path.name}.append_mutex")
    self.write_lease_meta_path = self.path.with_name(f"{self.path.name}.write_lease.meta")
    self._parent_storage_identity: DirectoryIdentity | None = (
      enumerated_location.parent_identity
      if enumerated_location is not None
      else None
    )
    self._segments_storage_identity: tuple[int, int] | None = (
      enumerated_location.segments_identity
      if enumerated_location is not None
      else None
    )
    self._append_mutex_identity: tuple[int, int] | None = None
    self._enumerated_active_was_absent = (
      enumerated_location is not None
      and enumerated_location.active_identity is None
    )
    self._ensure_active_parent_directory()
    self._active_storage_identity: tuple[int, int] | None = (
      enumerated_location.active_identity
      if enumerated_location is not None
      else None
    )
    if _create_active_on_open:
      self._create_or_bind_active_storage()
    elif self._active_storage_identity is not None:
      opened_active = self._open_active_file()
      assert opened_active is not None
      active_handle, _active_info, _created = opened_active
      active_handle.close()
    self._max_active_bytes = self._configured_max_active_bytes()
    self._pending_append_futures: set[asyncio.Future[LogEntry]] = set()
    self._gateway_session = gateway_session

    self._active_offset_cache = ActiveFileOffsetCache()

    if session_ref is not None:
      self._write_meta_sidecar(session_ref)
    if _repair_manifest_on_open:
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
    try:
      from .product_config import gateway_product_id

      self._atomic_write_parent_json(
        self.path.with_suffix(".meta.json").name,
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
        only_if_missing=True,
      )
    except Exception:
      log.warning(
        "Sidecar write failed for %s (telemetry-only)",
        self.path.with_suffix(".meta.json"),
        exc_info=True,
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
    try:
      return await asyncio.shield(append_future)
    except asyncio.CancelledError:
      # The synchronous append owns one durability decision.  A cancelled
      # caller must not return while the executor may still fsync a new fact.
      # Repeated cancellation is absorbed until the exact append outcome is
      # known; the original cancellation is then propagated.
      while not append_future.done():
        try:
          await asyncio.shield(append_future)
        except asyncio.CancelledError:
          continue
        except Exception:
          break
      if append_future.done() and not append_future.cancelled():
        try:
          append_future.result()
        except Exception:
          pass
      raise

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

  def _open_writer_lease_file(self) -> Any:
    """Open this log's exact writer-lease inode beneath its bound parent."""

    try:
      descriptor = _open_agent_session_log_write_lease(
        self.path,
        expected_parent_identity=self._parent_storage_identity,
      )
    except AgentSessionLogEnumerationError as exc:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log writer lease is unavailable"
      ) from exc
    try:
      handle = os.fdopen(descriptor, "r+b")
    except Exception:
      os.close(descriptor)
      raise
    return handle

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

  async def query_current_strict(
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
    excluded_seqs: AbstractSet[int] | None = None,
    exclude_entry: Callable[[LogEntry], bool] | None = None,
  ) -> tuple[list[LogEntry], QueryCursor | None]:
    """Read a complete current projection, rejecting ambiguous source bytes.

    ``exclude_entry`` is an internal pure visibility predicate. It runs before
    caller filters and paging so a current consumer can project exact hidden
    coordinates without materializing the complete source.
    """

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
    entries, cursor = await asyncio.to_thread(
      self._query_current_strict_sync,
      spec,
      order,
      limit,
      cursor,
      frozenset(excluded_seqs or ()),
      exclude_entry,
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
      cursor,
    )

  def query_current_strict_sync(
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
    excluded_seqs: AbstractSet[int] | None = None,
    exclude_entry: Callable[[LogEntry], bool] | None = None,
  ) -> tuple[list[LogEntry], QueryCursor | None]:
    """Synchronously read the strict current view at a terminal barrier."""

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
      contains_text=(
        contains_text.lower()
        if contains_text is not None
        else None
      ),
      has_error=has_error,
    )
    entries, next_cursor = self._query_current_strict_sync(
      spec,
      order,
      limit,
      cursor,
      frozenset(excluded_seqs or ()),
      exclude_entry,
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

  async def latest_seq_current_strict(self) -> int:
    """Return one integrity-checked max sequence for a current snapshot."""

    return await asyncio.to_thread(self._latest_seq_current_strict_locked_sync)

  def latest_seq_current_strict_sync(self) -> int:
    """Synchronously bind a strict current max sequence at a terminal barrier."""

    return self._latest_seq_current_strict_locked_sync()

  async def ensure_private_replay_storage(self) -> None:
    await asyncio.to_thread(self._ensure_private_replay_storage_sync)

  def _ensure_private_replay_storage_locked(self) -> None:
    parent_descriptor = self._open_active_parent_directory()
    try:
      self._secure_replay_directory_descriptor(
        parent_descriptor,
        target="parent directory",
      )
      exact_parent_names = {
        self.path.name,
        self.path.with_suffix(".meta.json").name,
        self.write_lease_path.name,
        self.append_mutex_path.name,
        self.write_lease_meta_path.name,
      }
      exact_parent_names.update(
        name
        for name in os.listdir(parent_descriptor)
        if name.startswith(f".{self.path.name}") and name.endswith(".tmp")
      )
      for name in exact_parent_names:
        self._secure_replay_file_at(parent_descriptor, name)
    finally:
      os.close(parent_descriptor)

    self._ensure_segments_directory()
    segments_descriptor = self._open_segments_directory()
    assert segments_descriptor is not None
    try:
      self._secure_replay_directory_descriptor(
        segments_descriptor,
        target="segment directory",
      )
      for name in os.listdir(segments_descriptor):
        self._secure_replay_file_at(segments_descriptor, name)
    finally:
      os.close(segments_descriptor)

  @staticmethod
  def _secure_replay_directory_descriptor(
    descriptor: int,
    *,
    target: str,
  ) -> None:
    try:
      info = os.fstat(descriptor)
      if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise ReplayStorageSecurityError(
          f"OpenAI replay storage {target} is unsafe"
        )
      os.fchmod(descriptor, 0o700)
      if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
        raise ReplayStorageSecurityError(
          f"OpenAI replay storage {target} permissions remain unsafe"
        )
    except ReplayStorageSecurityError:
      raise
    except OSError as exc:
      raise ReplayStorageSecurityError(
        f"Unable to secure OpenAI replay storage {target}"
      ) from exc

  def _secure_replay_file_at(self, directory_descriptor: int, name: str) -> None:
    descriptor = -1
    try:
      try:
        named = os.stat(
          name,
          dir_fd=directory_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        return
      if (
        not stat.S_ISREG(named.st_mode)
        or named.st_uid != os.geteuid()
        or named.st_nlink != 1
      ):
        raise ReplayStorageSecurityError(
          "OpenAI replay storage file is unsafe"
        )
      descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_descriptor,
      )
      opened = os.fstat(descriptor)
      if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or opened.st_nlink != 1
        or (opened.st_dev, opened.st_ino)
        != (named.st_dev, named.st_ino)
      ):
        raise ReplayStorageSecurityError(
          "OpenAI replay storage file identity changed"
        )
      if (
        name == self.path.name
        and self._active_storage_identity is not None
        and (opened.st_dev, opened.st_ino)
        != self._active_storage_identity
      ):
        raise ReplayStorageSecurityError(
          "OpenAI replay storage active file was displaced"
        )
      os.fchmod(descriptor, 0o600)
      if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
        raise ReplayStorageSecurityError(
          "OpenAI replay storage file permissions remain unsafe"
        )
    except ReplayStorageSecurityError:
      raise
    except OSError as exc:
      raise ReplayStorageSecurityError(
        "Unable to secure OpenAI replay storage file"
      ) from exc
    finally:
      if descriptor >= 0:
        os.close(descriptor)

  def _ensure_private_replay_storage_sync(self) -> None:
    with self._open_append_mutex_file() as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        self._ensure_private_replay_storage_locked()
      finally:
        fcntl.flock(mutex_file.fileno(), fcntl.LOCK_UN)

  def repair_manifest(self) -> None:
    with self._open_append_mutex_file() as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        self._repair_manifest_locked()
      finally:
        fcntl.flock(mutex_file.fileno(), fcntl.LOCK_UN)

  @classmethod
  def retire_segment(
    cls,
    location: AgentSessionLogLocation,
    expected_descriptor: Mapping[str, Any],
    *,
    expected_manifest: Mapping[str, Any],
    expected_manifest_identity: tuple[int, int, int, int],
    expected_segment_identity: tuple[int, int, int, int] | None,
    expected_sidecar_identity: tuple[int, int, int, int] | None,
  ) -> AgentSessionLogSegmentRetirement | None:
    """Durably retire one exact rotated segment, or return ``None`` if busy.

    Retention callers must pass an identity-bound enumerated location. The
    owner reacquires the writer lease and append mutex before trusting the
    manifest, segment, or sidecar. A descriptor that is still present must
    match the complete expected descriptor. A minimal identity projection is
    accepted only after the manifest and storage already prove the segment is
    absent, allowing recovery after manifest publication but before a caller's
    independent watermark update.
    """

    if type(location) is not AgentSessionLogLocation:
      raise TypeError(
        "segment retirement requires an exact AgentSessionLogLocation"
      )
    if location.segments_identity is None:
      raise AgentSessionLogStorageSecurityError(
        "segment retirement requires bound segment storage"
      )
    if (
      not isinstance(expected_descriptor, Mapping)
      or not isinstance(expected_manifest, Mapping)
    ):
      raise TypeError("segment retirement expectations must be mappings")
    for identity in (
      expected_manifest_identity,
      expected_segment_identity,
      expected_sidecar_identity,
    ):
      if identity is not None and (
        type(identity) is not tuple
        or len(identity) != 4
        or any(type(value) is not int for value in identity)
      ):
        raise TypeError("segment retirement file identity is invalid")
    owner = cls(
      location,
      _create_active_on_open=False,
      _repair_manifest_on_open=False,
    )
    return owner._retire_segment_with_owner_lock(
      dict(expected_descriptor),
      expected_manifest=dict(expected_manifest),
      expected_manifest_identity=expected_manifest_identity,
      expected_segment_identity=expected_segment_identity,
      expected_sidecar_identity=expected_sidecar_identity,
    )

  def _retire_segment_with_owner_lock(
    self,
    expected_descriptor: dict[str, Any],
    *,
    expected_manifest: dict[str, Any],
    expected_manifest_identity: tuple[int, int, int, int],
    expected_segment_identity: tuple[int, int, int, int] | None,
    expected_sidecar_identity: tuple[int, int, int, int] | None,
  ) -> AgentSessionLogSegmentRetirement | None:
    with self._open_writer_lease_file() as writer_lease:
      try:
        fcntl.flock(
          writer_lease.fileno(),
          fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
      except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
          return None
        raise
      try:
        with self._open_append_mutex_file() as mutex_file:
          fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
          try:
            return self._retire_segment_locked(
              expected_descriptor,
              expected_manifest=expected_manifest,
              expected_manifest_identity=expected_manifest_identity,
              expected_segment_identity=expected_segment_identity,
              expected_sidecar_identity=expected_sidecar_identity,
            )
          finally:
            fcntl.flock(mutex_file.fileno(), fcntl.LOCK_UN)
      finally:
        fcntl.flock(writer_lease.fileno(), fcntl.LOCK_UN)

  def _append_sync(self, event: dict[str, Any]) -> LogEntry:
    event_payload = sanitize_tool_event(event, sink="session_log")
    event_payload["event_schema_version"] = EVENT_SCHEMA_VERSION
    idempotent = _canonical_idempotent_event(event_payload)
    contains_responses_state = contains_openai_responses_history(event_payload)
    if contains_responses_state:
      event_payload[DURABLE_HISTORY_VERSION_KEY] = RESPONSES_HISTORY_VERSION

    if contains_responses_state:
      self._ensure_private_replay_storage_sync()

    with self._open_append_mutex_file() as mutex_file:
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
        opened_active = self._open_active_file(writable=True)
        assert opened_active is not None
        handle, _active_info, _created = opened_active
        with handle:
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

  def _query_current_strict_sync(
    self,
    spec: _QuerySpec,
    order: Literal["asc", "desc"],
    limit: int | None,
    cursor: QueryCursor | None,
    excluded_seqs: AbstractSet[int],
    exclude_entry: Callable[[LogEntry], bool] | None,
  ) -> tuple[list[LogEntry], QueryCursor | None]:
    with self._open_append_mutex_file() as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        return self._query_sync(
          spec,
          order,
          limit,
          cursor,
          True,
          excluded_seqs,
          exclude_entry,
        )
      finally:
        fcntl.flock(mutex_file.fileno(), fcntl.LOCK_UN)

  def _query_sync(
    self,
    spec: _QuerySpec,
    order: Literal["asc", "desc"],
    limit: int | None,
    cursor: QueryCursor | None,
    strict_current_integrity: bool = False,
    excluded_seqs: AbstractSet[int] | None = None,
    exclude_entry: Callable[[LogEntry], bool] | None = None,
  ) -> tuple[list[LogEntry], QueryCursor | None]:
    if order not in {"asc", "desc"}:
      raise ValueError("order must be 'asc' or 'desc'")
    if limit is not None and limit <= 0:
      raise ValueError("limit must be positive when provided")
    if cursor is not None and cursor.direction != order:
      raise ValueError("cursor direction does not match query order")
    if (
      not strict_current_integrity
      and spec.after_seq is not None
      and spec.before_seq is not None
      and spec.after_seq > spec.before_seq
    ):
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
      if (
        not strict_current_integrity
        and effective_spec.before_seq is not None
        and effective_spec.after_seq is not None
      ):
        if effective_spec.after_seq > effective_spec.before_seq:
          return [], None

    fetch_limit = None if limit is None else limit + 1
    if strict_current_integrity:
      excluded = excluded_seqs or frozenset()
      retained: list[LogEntry] | deque[LogEntry]
      if order == "desc" and fetch_limit is not None:
        retained = deque(maxlen=fetch_limit)
      else:
        retained = []
      for entry in self._iter_current_entries_strict_sync():
        if (
          effective_spec.after_seq is not None
          and entry.seq < effective_spec.after_seq
        ):
          continue
        if (
          effective_spec.before_seq is not None
          and entry.seq > effective_spec.before_seq
        ):
          continue
        if entry.seq in excluded:
          continue
        if exclude_entry is not None and exclude_entry(entry):
          continue
        if not self._matches(entry, effective_spec):
          continue
        if (
          order == "asc"
          and fetch_limit is not None
          and len(retained) >= fetch_limit
        ):
          continue
        retained.append(entry)
      entries = list(retained)
      if order == "desc":
        entries.reverse()
    elif order == "asc":
      entries = self._query_asc_sync(
        effective_spec,
        fetch_limit,
      )
    else:
      entries = self._query_desc_sync(effective_spec, fetch_limit)

    next_cursor = None
    if limit is not None and len(entries) > limit:
      next_cursor = QueryCursor(after_seq=entries[limit - 1].seq, direction=order)
      entries = entries[:limit]
    return entries, next_cursor

  def _query_asc_sync(
    self,
    spec: _QuerySpec,
    limit: int | None,
  ) -> list[LogEntry]:
    results: list[LogEntry] = []
    segments = self._segment_view_sync()
    for segment in segments:
      if spec.before_seq is not None and segment.first_seq > spec.before_seq:
        break
      if spec.after_seq is not None and segment.last_seq < spec.after_seq:
        continue
      remaining = None if limit is None else max(0, limit - len(results))
      if remaining == 0:
        break
      results.extend(self._query_asc_file_sync(
        segment,
        spec,
        remaining,
      ))
      if limit is not None and len(results) >= limit:
        break
    return results

  def _iter_current_entries_strict_sync(self) -> Iterator[LogEntry]:
    """Validate every current source byte while streaming durable entries."""

    segments = [
      _Segment(path, 0, 0, active=False)
      for path in self._safe_segment_paths()
    ]
    segments.append(_Segment(self.path, 0, 0, active=True))
    previous_seq: int | None = None
    for segment in segments:
      active_identity: ActiveFileIdentity | None = None
      if segment.active:
        opened_active = self._open_active_file()
        assert opened_active is not None
        handle, active_info, _created = opened_active
        active_identity = (
          int(active_info.st_dev),
          int(active_info.st_ino),
          int(active_info.st_size),
          int(active_info.st_mtime_ns),
        )
      else:
        opened_segment = self._open_segment_file(segment.path)
        assert opened_segment is not None
        handle, _segment_info = opened_segment
      completed = False
      with handle:
        file_size = handle.seek(0, os.SEEK_END)
        handle.seek(0)
        while True:
          offset = handle.tell()
          raw = handle.readline()
          if not raw:
            completed = handle.tell() >= file_size
            break
          entry = self._parse_entry_current_strict(raw)
          if entry is None:
            continue
          if entry.seq <= 0 or (
            previous_seq is not None
            and entry.seq != previous_seq + 1
          ):
            raise AgentSessionLogCurrentIntegrityError(
              "durable session-log current sequence order is ambiguous"
            )
          previous_seq = entry.seq
          if segment.active:
            self._update_cache(
              seq=entry.seq,
              offset=offset,
              active_identity=active_identity,
            )
          yield entry
      if segment.active and completed:
        self._mark_active_cache_complete(
          active_identity=active_identity,
          file_size=file_size,
        )

  def _parse_entry_current_strict(self, raw: bytes) -> LogEntry | None:
    stripped = raw.strip()
    if not stripped:
      return None
    try:
      payload = json.loads(
        stripped.decode("utf-8"),
        object_pairs_hook=self._strict_json_object,
      )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise AgentSessionLogCurrentIntegrityError(
        "durable session-log current projection is ambiguous"
      ) from exc
    if not isinstance(payload, dict):
      raise AgentSessionLogCurrentIntegrityError(
        "durable session-log current envelope is not an object"
      )
    event = payload.get("event")
    seq = payload.get("seq")
    timestamp = payload.get("timestamp")
    if not isinstance(event, dict):
      raise AgentSessionLogCurrentIntegrityError(
        "durable session-log current event is not an object"
      )
    if type(seq) is not int or seq <= 0:
      raise AgentSessionLogCurrentIntegrityError(
        "durable session-log current sequence is invalid"
      )
    if (
      type(timestamp) not in {int, float}
      or not math.isfinite(timestamp)
    ):
      raise AgentSessionLogCurrentIntegrityError(
        "durable session-log current timestamp is invalid"
      )
    return LogEntry(
      seq=seq,
      timestamp=float(timestamp),
      event=event,
    )

  @staticmethod
  def _strict_json_object(
    pairs: list[tuple[str, Any]],
  ) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
      if key in payload:
        raise AgentSessionLogCurrentIntegrityError(
          "durable session-log current JSON contains duplicate keys"
        )
      payload[key] = value
    return payload

  def _query_asc_file_sync(
    self,
    segment: _Segment,
    spec: _QuerySpec,
    limit: int | None,
  ) -> list[LogEntry]:
    results: list[LogEntry] = []
    active_identity = self._active_file_identity() if segment.active else None
    start_offset = self._starting_offset_for_seq(spec.after_seq, active_identity=active_identity) if segment.active else 0
    should_mark_complete = False
    if segment.active:
      opened_active = self._open_active_file()
      assert opened_active is not None
      handle, _active_info, _created = opened_active
    else:
      opened_segment = self._open_segment_file(segment.path)
      assert opened_segment is not None
      handle, _segment_info = opened_segment
    with handle:
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
    if segment.active:
      opened_active = self._open_active_file()
      assert opened_active is not None
      handle, _active_info, _created = opened_active
    else:
      opened_segment = self._open_segment_file(segment.path)
      assert opened_segment is not None
      handle, _segment_info = opened_segment
    with handle:
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
    opened = self._open_active_file()
    assert opened is not None
    handle, _info, _created = opened
    with handle:
      return max(manifest_latest, self._latest_seq_from_handle(handle))

  def _latest_seq_current_strict_sync(self) -> int:
    latest = 0
    for entry in self._iter_current_entries_strict_sync():
      latest = max(latest, entry.seq)
    return latest

  def _latest_seq_current_strict_locked_sync(self) -> int:
    with self._open_append_mutex_file() as mutex_file:
      fcntl.flock(mutex_file.fileno(), fcntl.LOCK_EX)
      try:
        return self._latest_seq_current_strict_sync()
      finally:
        fcntl.flock(mutex_file.fileno(), fcntl.LOCK_UN)

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
    opened = self._open_active_file(missing_ok=True)
    if opened is None:
      return None
    handle, info, _created = opened
    handle.close()
    return (
      int(info.st_dev),
      int(info.st_ino),
      int(info.st_size),
      int(info.st_mtime_ns),
    )

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
    # ``self.path`` is already an absolute lexical path whose directory chain
    # was descriptor-bound at construction. Re-resolving it would re-enter
    # mutable ambient ancestors.
    return str(self.path)

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
    return self._read_parent_json(self.path.with_suffix(".meta.json").name)

  def _write_active_sidecar(self, payload: dict[str, Any]) -> None:
    self._atomic_write_parent_json(
      self.path.with_suffix(".meta.json").name,
      payload,
    )

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

  def _write_segment_sidecar(
    self,
    segment_path: Path,
    payload: dict[str, Any],
  ) -> None:
    if (
      segment_path.parent != self.segments_dir
      or _SEGMENT_FILE_RE.fullmatch(segment_path.name) is None
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log segment sidecar target is unsafe"
      )
    opened = self._open_segment_file(segment_path)
    assert opened is not None
    handle, _info = opened
    handle.close()
    self._atomic_write_segments_json(
      segment_path.with_suffix(".meta.json").name,
      payload,
    )

  def _rotate_active_to_segment(self, segment_path: Path) -> None:
    """Move the active inode into the exact descriptor-bound segment dir."""

    if (
      self.path.parent != self.segments_dir.parent
      or segment_path.parent != self.segments_dir
      or _SEGMENT_FILE_RE.fullmatch(segment_path.name) is None
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log rotation target is unsafe"
      )
    segments_descriptor = self._open_segments_directory()
    assert segments_descriptor is not None
    parent_descriptor = -1
    active_descriptor = -1
    try:
      parent_descriptor = self._open_active_parent_directory()
      active_named = os.stat(
        self.path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
      )
      self._require_owned_single_link_file(
        active_named,
        target="active log",
      )
      active_descriptor = os.open(
        self.path.name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_descriptor,
      )
      active_opened = os.fstat(active_descriptor)
      self._require_owned_single_link_file(
        active_opened,
        target="active log",
      )
      if (
        active_opened.st_dev,
        active_opened.st_ino,
      ) != (active_named.st_dev, active_named.st_ino):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log active identity changed"
        )
      try:
        os.stat(
          segment_path.name,
          dir_fd=segments_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        pass
      else:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log rotation target already exists"
        )
      os.replace(
        self.path.name,
        segment_path.name,
        src_dir_fd=parent_descriptor,
        dst_dir_fd=segments_descriptor,
      )
      rotated_named = os.stat(
        segment_path.name,
        dir_fd=segments_descriptor,
        follow_symlinks=False,
      )
      rotated_opened = os.fstat(active_descriptor)
      self._require_owned_single_link_file(
        rotated_named,
        target="rotated segment",
      )
      self._require_owned_single_link_file(
        rotated_opened,
        target="rotated segment",
      )
      if (
        rotated_named.st_dev,
        rotated_named.st_ino,
      ) != (rotated_opened.st_dev, rotated_opened.st_ino):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log rotated identity changed"
        )
      os.fsync(segments_descriptor)
      os.fsync(parent_descriptor)
    except OSError as exc:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log rotation is unavailable"
      ) from exc
    finally:
      if active_descriptor >= 0:
        os.close(active_descriptor)
      if parent_descriptor >= 0:
        os.close(parent_descriptor)
      os.close(segments_descriptor)

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

  def _ensure_active_parent_directory(self) -> None:
    descriptor, identity = _open_directory_chain(
      self.path.parent,
      create=True,
    )
    try:
      opened = os.fstat(descriptor)
      if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) & 0o022
      ):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log parent directory is unsafe"
        )
      expected = self._parent_storage_identity
      if expected is not None and identity != expected:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log directory chain was displaced"
        )
      self._parent_storage_identity = identity
    finally:
      os.close(descriptor)

  def _open_active_parent_directory(self) -> int:
    descriptor, identity = _open_directory_chain(self.path.parent)
    try:
      opened = os.fstat(descriptor)
      if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) & 0o022
      ):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log parent directory is unsafe"
        )
      expected = self._parent_storage_identity
      if expected is None or identity != expected:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log directory chain was displaced"
        )
    except Exception:
      os.close(descriptor)
      raise
    return descriptor

  def _open_append_mutex_file(self) -> Any:
    """Open and bind the exact append mutex relative to the owned parent."""

    parent_descriptor = self._open_active_parent_directory()
    descriptor = -1
    try:
      try:
        descriptor = os.open(
          self.append_mutex_path.name,
          os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
          | getattr(os, "O_CLOEXEC", 0),
          0o600,
          dir_fd=parent_descriptor,
        )
      except OSError as exc:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log append mutex is unavailable"
        ) from exc
      opened = os.fstat(descriptor)
      self._require_owned_single_link_file(opened, target="append mutex")
      named = os.stat(
        self.append_mutex_path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
      )
      self._require_owned_single_link_file(named, target="append mutex")
      identity = (opened.st_dev, opened.st_ino)
      if identity != (named.st_dev, named.st_ino):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log append mutex identity changed"
        )
      expected = self._append_mutex_identity
      if expected is not None and identity != expected:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log append mutex was displaced"
        )
      self._append_mutex_identity = identity
      handle = os.fdopen(descriptor, "r+b")
      descriptor = -1
      return handle
    finally:
      if descriptor >= 0:
        os.close(descriptor)
      os.close(parent_descriptor)

  def _require_parent_json_name(self, name: str) -> None:
    if name != self.path.with_suffix(".meta.json").name:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log active metadata name is unsafe"
      )

  def _read_parent_json(self, name: str) -> dict[str, Any] | None:
    self._require_parent_json_name(name)
    parent_descriptor = self._open_active_parent_directory()
    descriptor = -1
    try:
      try:
        named = os.stat(
          name,
          dir_fd=parent_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        return None
      self._require_owned_single_link_file(named, target="active metadata")
      descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_descriptor,
      )
      opened = os.fstat(descriptor)
      self._require_owned_single_link_file(opened, target="active metadata")
      if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log active metadata identity changed"
        )
      with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        descriptor = -1
        try:
          payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
          return None
      return payload if isinstance(payload, dict) else None
    except AgentSessionLogStorageSecurityError:
      raise
    except OSError as exc:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log active metadata is unavailable"
      ) from exc
    finally:
      if descriptor >= 0:
        os.close(descriptor)
      os.close(parent_descriptor)

  def _atomic_write_parent_json(
    self,
    name: str,
    payload: dict[str, Any],
    *,
    only_if_missing: bool = False,
  ) -> None:
    self._require_parent_json_name(name)
    parent_descriptor = self._open_active_parent_directory()
    temporary_name = f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor = -1
    try:
      try:
        existing = os.stat(
          name,
          dir_fd=parent_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        existing = None
      if existing is not None:
        self._require_owned_single_link_file(
          existing,
          target="active metadata",
        )
        if only_if_missing:
          return
      descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=parent_descriptor,
      )
      encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
      ).encode("utf-8")
      offset = 0
      while offset < len(encoded):
        written = os.write(descriptor, encoded[offset:])
        if written <= 0:
          raise OSError("durable session-log metadata write made no progress")
        offset += written
      os.fsync(descriptor)
      os.close(descriptor)
      descriptor = -1
      os.replace(
        temporary_name,
        name,
        src_dir_fd=parent_descriptor,
        dst_dir_fd=parent_descriptor,
      )
      persisted = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
      )
      self._require_owned_single_link_file(
        persisted,
        target="active metadata",
      )
      os.fsync(parent_descriptor)
    except AgentSessionLogStorageSecurityError:
      raise
    except OSError as exc:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log active metadata cannot be written"
      ) from exc
    finally:
      if descriptor >= 0:
        os.close(descriptor)
      try:
        os.unlink(temporary_name, dir_fd=parent_descriptor)
      except FileNotFoundError:
        pass
      os.close(parent_descriptor)

  def _open_active_file(
    self,
    *,
    writable: bool = False,
    create: bool = False,
    missing_ok: bool = False,
    allow_identity_change: bool = False,
  ) -> tuple[Any, os.stat_result, bool] | None:
    """Open the exact active inode relative to its verified parent."""

    parent_descriptor = self._open_active_parent_directory()
    descriptor = -1
    created = False
    try:
      try:
        named = os.stat(
          self.path.name,
          dir_fd=parent_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        if not create:
          if missing_ok:
            return None
          raise AgentSessionLogStorageSecurityError(
            "durable session-log active file is missing"
          ) from None
        try:
          descriptor = os.open(
            self.path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
          )
        except OSError as exc:
          raise AgentSessionLogStorageSecurityError(
            "durable session-log active file cannot be created"
          ) from exc
        created = True
        named = os.fstat(descriptor)
      self._require_owned_single_link_file(named, target="active file")
      if descriptor < 0:
        flags = os.O_RDWR if writable else os.O_RDONLY
        try:
          descriptor = os.open(
            self.path.name,
            flags | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
          )
        except OSError as exc:
          raise AgentSessionLogStorageSecurityError(
            "durable session-log active file is unavailable"
          ) from exc
      opened = os.fstat(descriptor)
      self._require_owned_single_link_file(opened, target="active file")
      if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log active identity changed during open"
        )
      if self._enumerated_active_was_absent and not created:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log active file appeared after enumeration"
        )
      expected = self._active_storage_identity
      if (
        expected is not None
        and (opened.st_dev, opened.st_ino) != expected
        and not allow_identity_change
      ):
        if not self._active_identity_was_safely_rotated(expected):
          raise AgentSessionLogStorageSecurityError(
            "durable session-log active identity was displaced"
          )
        self._active_storage_identity = (opened.st_dev, opened.st_ino)
        active_offset_cache = getattr(self, "_active_offset_cache", None)
        if active_offset_cache is not None:
          active_offset_cache.clear()
      handle = os.fdopen(descriptor, "r+b" if writable else "rb")
      descriptor = -1
      self._enumerated_active_was_absent = False
      return handle, opened, created
    finally:
      if descriptor >= 0:
        os.close(descriptor)
      os.close(parent_descriptor)

  def _active_identity_was_safely_rotated(
    self,
    expected: tuple[int, int],
  ) -> bool:
    """Accept a peer transition only when the prior inode is a safe segment."""

    manifest = self._load_manifest()
    if (
      not isinstance(manifest, dict)
      or manifest.get("active_path") != f"../{self.path.name}"
    ):
      return False
    raw_segments = manifest.get("segments")
    if not isinstance(raw_segments, list):
      return False
    for raw in raw_segments:
      if not isinstance(raw, dict):
        continue
      try:
        segment_path = self._segment_path_from_manifest(raw["path"])
        opened_segment = self._open_segment_file(
          segment_path,
          missing_ok=True,
        )
      except (AgentSessionLogStorageSecurityError, KeyError):
        continue
      if opened_segment is None:
        continue
      handle, info = opened_segment
      handle.close()
      if (info.st_dev, info.st_ino) == expected:
        return True
    return False

  def _create_or_bind_active_storage(
    self,
    *,
    allow_identity_change: bool = False,
  ) -> None:
    opened = self._open_active_file(
      writable=True,
      create=True,
      allow_identity_change=allow_identity_change,
    )
    assert opened is not None
    handle, info, created = opened
    if created:
      os.fsync(handle.fileno())
    handle.close()
    self._active_storage_identity = (info.st_dev, info.st_ino)
    if created:
      parent_descriptor = self._open_active_parent_directory()
      try:
        os.fsync(parent_descriptor)
      finally:
        os.close(parent_descriptor)

  def _active_file_size(self) -> int:
    opened = self._open_active_file(missing_ok=True)
    if opened is None:
      return 0
    handle, info, _created = opened
    handle.close()
    return int(info.st_size)

  def _file_identity(self, path: Path) -> dict[str, int]:
    if path == self.path:
      opened = self._open_active_file()
      assert opened is not None
      handle, info, _created = opened
      handle.close()
      return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
      }
    if path.parent == self.segments_dir:
      opened = self._open_segment_file(path)
      assert opened is not None
      handle, info = opened
      handle.close()
      return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
      }
    return _sidecar_helpers.file_identity(path)

  def _new_manifest(self) -> dict[str, Any]:
    return _sidecar_helpers.new_manifest(
      self.path,
      manifest_schema_version=_MANIFEST_SCHEMA_VERSION,
      logical_stream_id_fn=self._logical_stream_id,
      telemetry_source_id_fn=self._telemetry_source_id,
    )

  def _load_manifest(self) -> dict[str, Any] | None:
    return self._read_segments_json("manifest.json")

  def _write_manifest(self, manifest: dict[str, Any]) -> None:
    self._atomic_write_segments_json("manifest.json", manifest)

  def _open_segments_directory(
    self,
    *,
    missing_ok: bool = False,
  ) -> int | None:
    """Open the exact owned segment directory without following a link."""

    parent_descriptor = self._open_active_parent_directory()
    descriptor = -1
    try:
      try:
        named = os.stat(
          self.segments_dir.name,
          dir_fd=parent_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        if missing_ok:
          return None
        raise AgentSessionLogStorageSecurityError(
          "durable session-log segment directory is missing"
        ) from None
      if (
        stat.S_ISLNK(named.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or named.st_uid != os.geteuid()
        or stat.S_IMODE(named.st_mode) & 0o022
      ):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log segment directory is unsafe"
        )
      descriptor = os.open(
        self.segments_dir.name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_descriptor,
      )
      opened = os.fstat(descriptor)
      if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) & 0o022
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
      ):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log segment directory identity changed"
        )
      identity = (opened.st_dev, opened.st_ino)
      expected = self._segments_storage_identity
      if expected is not None and identity != expected:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log segment directory was displaced"
        )
      self._segments_storage_identity = identity
      result = descriptor
      descriptor = -1
      return result
    except AgentSessionLogStorageSecurityError:
      raise
    except OSError as exc:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log segment directory is unavailable"
      ) from exc
    finally:
      if descriptor >= 0:
        os.close(descriptor)
      os.close(parent_descriptor)

  def _ensure_segments_directory(self) -> None:
    parent_descriptor = self._open_active_parent_directory()
    try:
      try:
        os.mkdir(
          self.segments_dir.name,
          mode=0o700,
          dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
      except FileExistsError:
        pass
    except OSError as exc:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log segment directory cannot be created"
      ) from exc
    finally:
      os.close(parent_descriptor)
    descriptor = self._open_segments_directory()
    assert descriptor is not None
    os.close(descriptor)

  @staticmethod
  def _require_segments_json_name(name: str) -> None:
    if name == "manifest.json":
      return
    if name.endswith(".meta.json"):
      segment_name = f"{name[:-len('.meta.json')]}.jsonl"
      if _SEGMENT_FILE_RE.fullmatch(segment_name) is not None:
        return
    raise AgentSessionLogStorageSecurityError(
      "durable session-log metadata name is unsafe"
    )

  @staticmethod
  def _require_owned_single_link_file(
    info: os.stat_result,
    *,
    target: str,
  ) -> None:
    if (
      not stat.S_ISREG(info.st_mode)
      or info.st_uid != os.geteuid()
      or info.st_nlink != 1
      or stat.S_IMODE(info.st_mode) & 0o022
    ):
      raise AgentSessionLogStorageSecurityError(
        f"durable session-log {target} is unsafe"
      )

  def _open_segment_file(
    self,
    segment_path: Path,
    *,
    missing_ok: bool = False,
  ) -> tuple[Any, os.stat_result] | None:
    """Open one exact immediate segment and bind its named inode."""

    if (
      segment_path.parent != self.segments_dir
      or _SEGMENT_FILE_RE.fullmatch(segment_path.name) is None
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log segment path is unsafe"
      )
    directory_descriptor = self._open_segments_directory()
    assert directory_descriptor is not None
    descriptor = -1
    try:
      try:
        named = os.stat(
          segment_path.name,
          dir_fd=directory_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        if missing_ok:
          return None
        raise AgentSessionLogStorageSecurityError(
          "durable session-log segment is missing"
        ) from None
      self._require_owned_single_link_file(named, target="segment")
      try:
        descriptor = os.open(
          segment_path.name,
          os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
          dir_fd=directory_descriptor,
        )
      except OSError as exc:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log segment is unavailable"
        ) from exc
      opened = os.fstat(descriptor)
      self._require_owned_single_link_file(opened, target="segment")
      if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log segment identity changed"
        )
      handle = os.fdopen(descriptor, "rb")
      descriptor = -1
      return handle, opened
    finally:
      if descriptor >= 0:
        os.close(descriptor)
      os.close(directory_descriptor)

  def _safe_segment_paths(self) -> tuple[Path, ...]:
    directory_descriptor = self._open_segments_directory(missing_ok=True)
    if directory_descriptor is None:
      return ()
    try:
      names = tuple(os.listdir(directory_descriptor))
    except OSError as exc:
      os.close(directory_descriptor)
      raise AgentSessionLogStorageSecurityError(
        "durable session-log segment directory cannot be listed"
      ) from exc
    os.close(directory_descriptor)
    paths: list[Path] = []
    for name in names:
      if not name.endswith(".jsonl"):
        continue
      if _SEGMENT_FILE_RE.fullmatch(name) is None:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log segment filename is invalid"
        )
      path = self.segments_dir / name
      opened = self._open_segment_file(path)
      assert opened is not None
      handle, _info = opened
      handle.close()
      paths.append(path)
    return tuple(sorted(paths))

  def _open_retirement_file_at(
    self,
    directory_descriptor: int,
    name: str,
    *,
    target: str,
    missing_ok: bool = False,
  ) -> tuple[int, os.stat_result] | None:
    descriptor = -1
    try:
      try:
        named = os.stat(
          name,
          dir_fd=directory_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        if missing_ok:
          return None
        raise AgentSessionLogStorageSecurityError(
          f"durable session-log {target} is missing"
        ) from None
      self._require_owned_single_link_file(named, target=target)
      descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_descriptor,
      )
      opened = os.fstat(descriptor)
      self._require_owned_single_link_file(opened, target=target)
      if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise AgentSessionLogStorageSecurityError(
          f"durable session-log {target} identity changed"
        )
      result = descriptor, opened
      descriptor = -1
      return result
    except AgentSessionLogStorageSecurityError:
      raise
    except OSError as exc:
      raise AgentSessionLogStorageSecurityError(
        f"durable session-log {target} is unavailable"
      ) from exc
    finally:
      if descriptor >= 0:
        os.close(descriptor)

  @staticmethod
  def _read_retirement_json(
    descriptor: int,
    *,
    target: str,
  ) -> dict[str, Any]:
    def reject_duplicate_fields(
      pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
      payload: dict[str, Any] = {}
      for key, value in pairs:
        if key in payload:
          raise AgentSessionLogStorageSecurityError(
            f"durable session-log {target} contains duplicate fields"
          )
        payload[key] = value
      return payload

    try:
      os.lseek(descriptor, 0, os.SEEK_SET)
      chunks: list[bytes] = []
      while chunk := os.read(descriptor, 64 * 1024):
        chunks.append(chunk)
      payload = json.loads(
        b"".join(chunks).decode("utf-8"),
        object_pairs_hook=reject_duplicate_fields,
      )
    except AgentSessionLogStorageSecurityError:
      raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise AgentSessionLogStorageSecurityError(
        f"durable session-log {target} is invalid"
      ) from exc
    if not isinstance(payload, dict):
      raise AgentSessionLogStorageSecurityError(
        f"durable session-log {target} is invalid"
      )
    return payload

  @staticmethod
  def _json_values_match(left: Any, right: Any) -> bool:
    try:
      return json.dumps(
        left,
        sort_keys=True,
        separators=(",", ":"),
      ) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
      )
    except (TypeError, ValueError):
      return False

  def _validate_retirement_sidecar(
    self,
    *,
    manifest: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    segment_info: os.stat_result | None,
  ) -> None:
    path = self._segment_path_from_manifest(descriptor.get("path"))
    filename_parts = self._segment_filename_parts(path)
    if filename_parts is None:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log retirement descriptor is invalid"
      )
    segment_id, first_seq, last_seq, generation = filename_parts
    try:
      descriptor_bytes = int(descriptor["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log retirement descriptor is invalid"
      ) from exc
    if (
      descriptor.get("segment_id") != segment_id
      or type(descriptor.get("first_seq")) is not int
      or descriptor.get("first_seq") != first_seq
      or type(descriptor.get("last_seq")) is not int
      or descriptor.get("last_seq") != last_seq
      or descriptor_bytes < 0
      or not isinstance(descriptor.get("telemetry_source_id"), str)
      or not descriptor.get("telemetry_source_id")
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log retirement descriptor is invalid"
      )
    identity = descriptor.get("rotated_from_file_identity")
    identity_fields = {"st_dev", "st_ino", "size", "mtime_ns"}
    if (
      not isinstance(identity, Mapping)
      or set(identity) != identity_fields
      or any(type(identity.get(key)) is not int for key in identity_fields)
      or identity.get("size") != descriptor_bytes
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log retirement descriptor identity is invalid"
      )
    if segment_info is not None:
      observed_identity = {
        "st_dev": int(segment_info.st_dev),
        "st_ino": int(segment_info.st_ino),
        "size": int(segment_info.st_size),
        "mtime_ns": int(segment_info.st_mtime_ns),
      }
      if not self._json_values_match(identity, observed_identity):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log retirement segment identity changed"
        )
    comparable_fields = (
      "segment_id",
      "first_seq",
      "last_seq",
      "telemetry_source_id",
      "rotated_from_source_id",
      "rotated_from_file_identity",
    )
    if (
      sidecar.get("schema_version") != 2
      or sidecar.get("file_role") != "segment"
      or sidecar.get("logical_stream_id") != self._logical_stream_id()
      or sidecar.get("rotated_from_path") != self._logical_stream_id()
      or sidecar.get("active_generation") != generation
      or any(
        not self._json_values_match(sidecar.get(key), descriptor.get(key))
        for key in comparable_fields
      )
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log retirement sidecar contradicts its descriptor"
      )
    manifest_agent_session_id = manifest.get("agent_session_id")
    if (
      manifest_agent_session_id is not None
      and sidecar.get("agent_session_id") != manifest_agent_session_id
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log retirement sidecar contradicts its manifest"
      )
    for key in _sidecar_helpers.V2_STORAGE_IDENTITY_FIELDS:
      if key in manifest and sidecar.get(key) != manifest.get(key):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log retirement storage identity changed"
        )

  def _require_retirement_name_identity(
    self,
    directory_descriptor: int,
    name: str,
    opened: os.stat_result,
    *,
    target: str,
  ) -> None:
    try:
      named = os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
      )
    except FileNotFoundError:
      raise AgentSessionLogStorageSecurityError(
        f"durable session-log {target} changed before retirement"
      ) from None
    self._require_owned_single_link_file(named, target=target)
    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
      raise AgentSessionLogStorageSecurityError(
        f"durable session-log {target} changed before retirement"
      )

  def _retire_segment_locked(
    self,
    expected_descriptor: dict[str, Any],
    *,
    expected_manifest: dict[str, Any],
    expected_manifest_identity: tuple[int, int, int, int],
    expected_segment_identity: tuple[int, int, int, int] | None,
    expected_sidecar_identity: tuple[int, int, int, int] | None,
  ) -> AgentSessionLogSegmentRetirement:
    required_identity_fields = {
      "segment_id",
      "path",
      "telemetry_source_id",
    }
    if (
      any(
        not isinstance(expected_descriptor.get(key), str)
        or not expected_descriptor.get(key)
        for key in required_identity_fields
      )
      or _SEGMENT_FILE_RE.fullmatch(
        str(expected_descriptor.get("path") or "")
      ) is None
      or expected_descriptor.get("path")
      != f"{expected_descriptor.get('segment_id')}.jsonl"
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log retirement identity is invalid"
      )
    segment_id = str(expected_descriptor["segment_id"])
    source_id = str(expected_descriptor["telemetry_source_id"])
    segment_name = str(expected_descriptor["path"])
    sidecar_name = f"{segment_name[:-len('.jsonl')]}.meta.json"

    directory_descriptor = self._open_segments_directory()
    assert directory_descriptor is not None
    manifest_descriptor = segment_descriptor = sidecar_descriptor = -1
    try:
      opened_manifest = self._open_retirement_file_at(
        directory_descriptor,
        "manifest.json",
        target="retirement manifest",
      )
      assert opened_manifest is not None
      manifest_descriptor, manifest_info = opened_manifest
      manifest = self._read_retirement_json(
        manifest_descriptor,
        target="retirement manifest",
      )
      observed_manifest_identity = (
        int(manifest_info.st_dev),
        int(manifest_info.st_ino),
        int(manifest_info.st_size),
        int(manifest_info.st_mtime_ns),
      )
      if (
        observed_manifest_identity != expected_manifest_identity
        or not self._json_values_match(manifest, expected_manifest)
      ):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log retirement manifest changed after inventory"
        )
      if (
        manifest.get("logical_stream_id") not in {
          None,
          self._logical_stream_id(),
        }
        or manifest.get("active_path") != f"../{self.path.name}"
        or not isinstance(manifest.get("segments"), list)
      ):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log retirement manifest contradicts storage"
        )
      segments: list[dict[str, Any]] = []
      ids: set[str] = set()
      paths: set[str] = set()
      sources: set[str] = set()
      target_descriptor: dict[str, Any] | None = None
      for raw in manifest["segments"]:
        if not isinstance(raw, dict):
          raise AgentSessionLogStorageSecurityError(
            "durable session-log retirement manifest is invalid"
          )
        raw_id = raw.get("segment_id")
        raw_path = raw.get("path")
        raw_source = raw.get("telemetry_source_id")
        if (
          not isinstance(raw_id, str)
          or not isinstance(raw_path, str)
          or not isinstance(raw_source, str)
          or raw_id in ids
          or raw_path in paths
          or raw_source in sources
        ):
          raise AgentSessionLogStorageSecurityError(
            "durable session-log retirement manifest identities are invalid"
          )
        ids.add(raw_id)
        paths.add(raw_path)
        sources.add(raw_source)
        segments.append(raw)
        if raw_id == segment_id:
          target_descriptor = raw

      if target_descriptor is None:
        if (
          expected_segment_identity is not None
          or expected_sidecar_identity is not None
        ):
          raise AgentSessionLogStorageSecurityError(
            "durable session-log retired files changed after inventory"
          )
        if segment_name in paths or source_id in sources:
          raise AgentSessionLogStorageSecurityError(
            "durable session-log retirement identity was reused"
          )
        for name, target in (
          (segment_name, "retired segment"),
          (sidecar_name, "retired segment sidecar"),
        ):
          opened = self._open_retirement_file_at(
            directory_descriptor,
            name,
            target=target,
            missing_ok=True,
          )
          if opened is not None:
            descriptor, _info = opened
            os.close(descriptor)
            raise AgentSessionLogStorageSecurityError(
              f"durable session-log {target} remains outside its manifest"
            )
        return AgentSessionLogSegmentRetirement(
          segment_id=segment_id,
          telemetry_source_id=source_id,
          removed_from_manifest=False,
          deleted_bytes=0,
        )

      if not self._json_values_match(
        target_descriptor,
        expected_descriptor,
      ):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log retirement descriptor changed"
        )
      opened_segment = self._open_retirement_file_at(
        directory_descriptor,
        segment_name,
        target="retirement segment",
        missing_ok=True,
      )
      segment_info: os.stat_result | None = None
      if opened_segment is not None:
        segment_descriptor, segment_info = opened_segment
      opened_sidecar = self._open_retirement_file_at(
        directory_descriptor,
        sidecar_name,
        target="retirement segment sidecar",
        missing_ok=True,
      )
      sidecar_info: os.stat_result | None = None
      if opened_sidecar is not None:
        sidecar_descriptor, sidecar_info = opened_sidecar
      observed_segment_identity = (
        (
          int(segment_info.st_dev),
          int(segment_info.st_ino),
          int(segment_info.st_size),
          int(segment_info.st_mtime_ns),
        )
        if segment_info is not None
        else None
      )
      observed_sidecar_identity = (
        (
          int(sidecar_info.st_dev),
          int(sidecar_info.st_ino),
          int(sidecar_info.st_size),
          int(sidecar_info.st_mtime_ns),
        )
        if sidecar_info is not None
        else None
      )
      segment_first_crash_state = (
        observed_segment_identity is None
        and expected_segment_identity is None
        and observed_sidecar_identity is not None
        and expected_sidecar_identity is None
      )
      if (
        observed_segment_identity != expected_segment_identity
        or (
          observed_sidecar_identity != expected_sidecar_identity
          and not segment_first_crash_state
        )
      ):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log retired files changed after inventory"
        )
      if segment_info is not None and sidecar_info is None:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log retirement segment sidecar is missing"
        )
      if sidecar_info is not None:
        sidecar = self._read_retirement_json(
          sidecar_descriptor,
          target="retirement segment sidecar",
        )
        self._validate_retirement_sidecar(
          manifest=manifest,
          descriptor=target_descriptor,
          sidecar=sidecar,
          segment_info=segment_info,
        )
      elif segment_info is None:
        # Both files may already be absent after a crash between durable
        # unlink and manifest publication. The exact manifest descriptor is
        # still the recovery authority in this state.
        self._validate_retirement_sidecar_descriptor_only(target_descriptor)

      deleted_bytes = int(segment_info.st_size) if segment_info is not None else 0
      if segment_info is not None:
        self._require_retirement_name_identity(
          directory_descriptor,
          segment_name,
          segment_info,
          target="retirement segment",
        )
        os.unlink(segment_name, dir_fd=directory_descriptor)
        # Persist segment absence before sidecar removal. This ordering makes
        # segment-present/sidecar-absent impossible for every durable prefix.
        os.fsync(directory_descriptor)
      if sidecar_info is not None:
        self._require_retirement_name_identity(
          directory_descriptor,
          sidecar_name,
          sidecar_info,
          target="retirement segment sidecar",
        )
        os.unlink(sidecar_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)

      remaining = [
        item
        for item in segments
        if item.get("segment_id") != segment_id
      ]
      manifest["segments"] = remaining
      try:
        active_first = min(
          int(target_descriptor.get("last_seq") or 0) + 1,
          int(manifest.get("latest_seq") or 0) + 1,
        )
        manifest["min_seq_available"] = min(
          (
            int(item.get("first_seq") or active_first)
            for item in remaining
          ),
          default=active_first,
        )
      except (TypeError, ValueError) as exc:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log retirement manifest sequence is invalid"
        ) from exc
      self._atomic_write_segments_json_at(
        directory_descriptor,
        "manifest.json",
        manifest,
        expected_identity=(manifest_info.st_dev, manifest_info.st_ino),
      )
      return AgentSessionLogSegmentRetirement(
        segment_id=segment_id,
        telemetry_source_id=source_id,
        removed_from_manifest=True,
        deleted_bytes=deleted_bytes,
      )
    finally:
      for descriptor in (
        sidecar_descriptor,
        segment_descriptor,
        manifest_descriptor,
      ):
        if descriptor >= 0:
          os.close(descriptor)
      os.close(directory_descriptor)

  def _validate_retirement_sidecar_descriptor_only(
    self,
    descriptor: Mapping[str, Any],
  ) -> None:
    path = self._segment_path_from_manifest(descriptor.get("path"))
    filename_parts = self._segment_filename_parts(path)
    identity = descriptor.get("rotated_from_file_identity")
    if (
      filename_parts is None
      or descriptor.get("segment_id") != filename_parts[0]
      or descriptor.get("first_seq") != filename_parts[1]
      or descriptor.get("last_seq") != filename_parts[2]
      or type(descriptor.get("bytes")) is not int
      or descriptor.get("bytes") < 0
      or not isinstance(identity, Mapping)
      or set(identity) != {"st_dev", "st_ino", "size", "mtime_ns"}
      or any(
        type(identity.get(key)) is not int
        for key in ("st_dev", "st_ino", "size", "mtime_ns")
      )
      or identity.get("size") != descriptor.get("bytes")
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log retirement descriptor identity is invalid"
      )

  def _read_segments_json(self, name: str) -> dict[str, Any] | None:
    self._require_segments_json_name(name)
    directory_descriptor = self._open_segments_directory(missing_ok=True)
    if directory_descriptor is None:
      return None
    descriptor = -1
    try:
      try:
        named = os.stat(
          name,
          dir_fd=directory_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        return None
      self._require_owned_single_link_file(named, target="metadata")
      try:
        descriptor = os.open(
          name,
          os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
          dir_fd=directory_descriptor,
        )
      except OSError as exc:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log metadata is unavailable"
        ) from exc
      opened = os.fstat(descriptor)
      self._require_owned_single_link_file(opened, target="metadata")
      if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise AgentSessionLogStorageSecurityError(
          "durable session-log metadata identity changed"
        )
      with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        descriptor = -1
        try:
          payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
          return None
      return payload if isinstance(payload, dict) else None
    finally:
      if descriptor >= 0:
        os.close(descriptor)
      os.close(directory_descriptor)

  def _segments_json_exists(self, name: str) -> bool:
    self._require_segments_json_name(name)
    directory_descriptor = self._open_segments_directory(missing_ok=True)
    if directory_descriptor is None:
      return False
    try:
      try:
        info = os.stat(
          name,
          dir_fd=directory_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        return False
      self._require_owned_single_link_file(info, target="metadata")
      return True
    finally:
      os.close(directory_descriptor)

  def _atomic_write_segments_json(
    self,
    name: str,
    payload: dict[str, Any],
  ) -> None:
    self._require_segments_json_name(name)
    directory_descriptor = self._open_segments_directory()
    assert directory_descriptor is not None
    try:
      self._atomic_write_segments_json_at(
        directory_descriptor,
        name,
        payload,
      )
    finally:
      os.close(directory_descriptor)

  def _atomic_write_segments_json_at(
    self,
    directory_descriptor: int,
    name: str,
    payload: dict[str, Any],
    *,
    expected_identity: tuple[int, int] | None = None,
  ) -> None:
    """Atomically publish metadata beneath one already-bound directory."""

    self._require_segments_json_name(name)
    temporary_name = (
      f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    descriptor = -1
    try:
      descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_descriptor,
      )
      encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
      ).encode("utf-8")
      offset = 0
      while offset < len(encoded):
        written = os.write(descriptor, encoded[offset:])
        if written <= 0:
          raise OSError("durable session-log metadata write made no progress")
        offset += written
      os.fsync(descriptor)
      os.close(descriptor)
      descriptor = -1
      if expected_identity is not None:
        try:
          current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
          )
        except FileNotFoundError:
          raise AgentSessionLogStorageSecurityError(
            "durable session-log metadata changed before publication"
          ) from None
        self._require_owned_single_link_file(current, target="metadata")
        if (current.st_dev, current.st_ino) != expected_identity:
          raise AgentSessionLogStorageSecurityError(
            "durable session-log metadata changed before publication"
          )
      os.replace(
        temporary_name,
        name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
      )
      persisted = os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
      )
      self._require_owned_single_link_file(persisted, target="metadata")
      os.fsync(directory_descriptor)
    finally:
      if descriptor >= 0:
        os.close(descriptor)
      try:
        os.unlink(temporary_name, dir_fd=directory_descriptor)
      except FileNotFoundError:
        pass

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
    if type(raw_path) is not str or _SEGMENT_FILE_RE.fullmatch(raw_path) is None:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log manifest segment path is unsafe"
      )
    path = Path(raw_path)
    if path.is_absolute() or path.parts != (raw_path,):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log manifest segment path is unsafe"
      )
    return self.segments_dir / raw_path

  def _seq_range_for_file(self, path: Path) -> tuple[int, int]:
    first = 0
    last = 0
    if path == self.path:
      opened_active = self._open_active_file(missing_ok=True)
      if opened_active is None:
        return 0, 0
      handle, info, _created = opened_active
      if info.st_size == 0:
        handle.close()
        return 0, 0
    elif path.parent == self.segments_dir:
      opened = self._open_segment_file(path, missing_ok=True)
      if opened is None:
        return 0, 0
      handle, info = opened
      if info.st_size == 0:
        handle.close()
        return 0, 0
    else:
      if not path.exists() or path.stat().st_size == 0:
        return 0, 0
      handle = path.open("rb")
    with handle:
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
    if path == self.path:
      opened_active = self._open_active_file(missing_ok=True)
      if opened_active is None:
        return 0, 0
      handle, info, _created = opened_active
      if info.st_size == 0:
        handle.close()
        return 0, 0
    elif path.parent == self.segments_dir:
      opened = self._open_segment_file(path, missing_ok=True)
      if opened is None:
        return 0, 0
      handle, info = opened
      if info.st_size == 0:
        handle.close()
        return 0, 0
    else:
      if not path.exists() or path.stat().st_size == 0:
        return 0, 0
      handle = path.open("rb")
    with handle:
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
    filename_parts = self._segment_filename_parts(segment_path)
    if filename_parts is None:
      return None
    filename_segment_id, filename_first, filename_last, filename_generation = (
      filename_parts
    )
    try:
      segment_id = str(meta["segment_id"])
      first_seq = int(meta["first_seq"])
      last_seq = int(meta["last_seq"])
    except (KeyError, TypeError, ValueError):
      return None
    if (
      segment_id != filename_segment_id
      or first_seq != filename_first
      or last_seq != filename_last
      or first_seq <= 0
      or last_seq < first_seq
    ):
      return None
    active_generation = meta.get("active_generation")
    if active_generation is None:
      active_generation = (
        fallback_generation
        if fallback_generation is not None
        else filename_generation
      )
    try:
      generation = int(active_generation)
    except (TypeError, ValueError):
      return None
    identity = self._file_identity(segment_path)
    return {
      "segment_id": segment_id,
      "path": segment_path.name,
      "first_seq": first_seq,
      "last_seq": last_seq,
      "bytes": identity["size"],
      "telemetry_source_id": str(meta.get("telemetry_source_id") or self._telemetry_source_id("segment", segment_id)),
      "rotated_from_source_id": str(meta.get("rotated_from_source_id") or self._telemetry_source_id("active", f"{generation:06d}")),
      "rotated_from_path": str(meta.get("rotated_from_path") or f"../{self.path.name}"),
      "rotated_from_file_identity": (
        meta.get("rotated_from_file_identity")
        if isinstance(meta.get("rotated_from_file_identity"), dict)
        else identity
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
    if (
      (scanned_first and scanned_first != filename_first)
      or (scanned_last and scanned_last != filename_last)
    ):
      raise AgentSessionLogStorageSecurityError(
        "durable session-log segment sequence bounds do not match its name"
      )
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
    self._atomic_write_segments_json(
      segment_path.with_suffix(".meta.json").name,
      meta,
    )
    return self._segment_descriptor_from_meta(segment_path, meta, fallback_generation=generation)

  def _ensure_segment_sidecar_from_descriptor(
    self,
    segment_path: Path,
    descriptor: dict[str, Any],
    base_meta: dict[str, Any],
  ) -> dict[str, Any] | None:
    sidecar_path = segment_path.with_suffix(".meta.json")
    meta = self._read_segments_json(sidecar_path.name)
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
    self._atomic_write_segments_json(sidecar_path.name, meta)
    return meta

  def _repair_manifest_locked(self) -> None:
    segment_paths = list(self._safe_segment_paths())
    segment_path_set = set(segment_paths)
    manifest = self._load_manifest()
    if manifest is None and not segment_paths:
      return

    segment_metas = [
      meta
      for path in segment_paths
      if isinstance(
        meta := self._read_segments_json(
          path.with_suffix(".meta.json").name
        ),
        dict,
      )
    ]
    base_meta = self._sidecar_base_for_repair(segment_metas)
    manifest = manifest or self._new_manifest()

    repaired_by_id: dict[str, dict[str, Any]] = {}
    raw_segments = manifest.get("segments", [])
    manifest_requires_rebuild = not isinstance(raw_segments, list)
    if not isinstance(raw_segments, list):
      raw_segments = []
    manifest_rows: list[tuple[dict[str, Any], Path]] = []
    for raw in raw_segments:
      if not isinstance(raw, dict):
        manifest_requires_rebuild = True
        continue
      try:
        segment_path = self._segment_path_from_manifest(raw["path"])
      except (
        AgentSessionLogStorageSecurityError,
        KeyError,
        TypeError,
        ValueError,
      ):
        manifest_requires_rebuild = True
        continue
      if segment_path not in segment_path_set:
        manifest_requires_rebuild = True
        continue
      manifest_rows.append((raw, segment_path))
    if manifest_requires_rebuild and not segment_paths:
      raise AgentSessionLogStorageSecurityError(
        "durable session-log manifest has no safe local segment source"
      )
    if manifest_requires_rebuild:
      manifest_rows = []

    for raw, segment_path in manifest_rows:
      meta = self._ensure_segment_sidecar_from_descriptor(segment_path, raw, base_meta)
      descriptor = self._segment_descriptor_from_meta(segment_path, meta or raw)
      if descriptor is not None:
        repaired_by_id[str(descriptor["segment_id"])] = descriptor

    for segment_path in segment_paths:
      meta = self._read_segments_json(
        segment_path.with_suffix(".meta.json").name
      )
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
        **{
          key: base_meta.get(key)
          for key in _sidecar_helpers.V2_STORAGE_IDENTITY_FIELDS
          if key in base_meta
        },
      }
    )
    manifest_exists = self._segments_json_exists("manifest.json")
    if segments or manifest_exists:
      self._write_manifest(manifest)

    opened_active = self._open_active_file()
    assert opened_active is not None
    active_handle, _active_info, _created = opened_active
    active_handle.close()
    active_meta_name = self.path.with_suffix(".meta.json").name
    active_meta = self._read_parent_json(active_meta_name)
    if (segments or manifest_exists) and (
      not isinstance(active_meta, dict)
      or active_meta.get("schema_version") != 2
      or active_meta.get("file_role") != "active"
      or int(active_meta.get("active_generation") or -1) != active_generation
    ):
      self._atomic_write_parent_json(
        active_meta_name,
        self._active_sidecar_payload(
          base_meta,
          active_generation=active_generation,
        ),
      )

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
      except (KeyError, TypeError, ValueError) as exc:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log manifest segment is invalid"
        ) from exc
      opened = self._open_segment_file(segment.path, missing_ok=True)
      if opened is None:
        raise AgentSessionLogStorageSecurityError(
          "durable session-log manifest segment is missing"
        )
      handle, _info = opened
      handle.close()
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
  "AgentSessionLogCurrentIntegrityError",
  "AgentSessionLogEnumerationError",
  "AgentSessionLogLocation",
  "AgentSessionLogSegmentRetirement",
  "AgentSessionLogStorageSecurityError",
  "AgentSessionLogWriteLeaseSet",
  "AgentSessionRef",
  "LogEntry",
  "QueryCursor",
  "agent_session_logical_path_for_jsonl",
  "enumerate_agent_session_log_paths",
  "try_acquire_agent_session_log_write_leases",
  "_atomic_write_sidecar",
  "resolve_agent_session_id",
  "slugify",
]
