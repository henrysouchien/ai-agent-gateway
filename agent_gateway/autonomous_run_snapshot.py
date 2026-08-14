from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat


MAX_AUTONOMOUS_RUN_SNAPSHOT_FILES = 64
MAX_AUTONOMOUS_RUN_SNAPSHOT_FILE_BYTES = 8 * 1024 * 1024
MAX_AUTONOMOUS_RUN_SNAPSHOT_TOTAL_BYTES = 32 * 1024 * 1024

_READ_CHUNK_BYTES = 1024 * 1024
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODES = frozenset({0o400, 0o600})
_SNAPSHOT_DIGEST_DOMAIN = b"agent-gateway/autonomous-run-snapshot/v1\x00"
_MAX_ROOT_PATH_BYTES = 4096
_MAX_ROOT_COMPONENTS = 64
_MAX_RELATIVE_PATH_BYTES = 4096
_MAX_RELATIVE_PATH_COMPONENTS = 8
_MAX_COMPONENT_BYTES = 255


class AutonomousRunSnapshotError(RuntimeError):
  """A private autonomous run root could not be snapshotted safely."""

  def __init__(self, code: str, message: str) -> None:
    super().__init__(message)
    self.code = code


@dataclass(frozen=True, slots=True)
class AutonomousRunFileSnapshot:
  relative_path: str
  raw_bytes: bytes
  size_bytes: int
  sha256: str


@dataclass(frozen=True, slots=True)
class AutonomousRunSnapshot:
  files: tuple[AutonomousRunFileSnapshot, ...]
  total_bytes: int
  sha256: str


@dataclass(frozen=True, slots=True)
class _DirectoryBinding:
  parent_fd: int | None
  name: str
  fd: int
  initial: os.stat_result
  private: bool


@dataclass(frozen=True, slots=True)
class _CapturedFile:
  snapshot: AutonomousRunFileSnapshot
  parent_fd: int
  name: str
  initial: os.stat_result
  descriptor: int


def _required_open_flag(name: str) -> int:
  value = getattr(os, name, None)
  if type(value) is not int or value == 0:
    raise AutonomousRunSnapshotError(
      "platform_unsupported",
      f"required filesystem flag {name} is unavailable",
    )
  return value


def _directory_open_flags() -> int:
  return (
    os.O_RDONLY
    | _required_open_flag("O_DIRECTORY")
    | _required_open_flag("O_NOFOLLOW")
    | _required_open_flag("O_CLOEXEC")
  )


def _file_open_flags() -> int:
  return (
    os.O_RDONLY
    | _required_open_flag("O_NOFOLLOW")
    | _required_open_flag("O_CLOEXEC")
    | _required_open_flag("O_NONBLOCK")
  )


def _close_descriptors_once(
  descriptors: tuple[int, ...],
) -> OSError | None:
  first_error: OSError | None = None
  for descriptor in descriptors:
    try:
      os.close(descriptor)
    except OSError as exc:
      if first_error is None:
        first_error = exc
  return first_error


def _raise_cleanup_failure(error: OSError) -> None:
  raise AutonomousRunSnapshotError(
    "descriptor_cleanup_failed",
    "snapshot descriptor cleanup failed closed",
  ) from error


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
  return (
    left.st_dev == right.st_dev
    and left.st_ino == right.st_ino
  )


def _stable_directory_attributes(
  before: os.stat_result,
  after: os.stat_result,
) -> bool:
  return (
    _same_object(before, after)
    and before.st_mode == after.st_mode
    and before.st_uid == after.st_uid
    and before.st_nlink == after.st_nlink
  )


def _stable_file_attributes(
  before: os.stat_result,
  after: os.stat_result,
) -> bool:
  return (
    _same_object(before, after)
    and before.st_mode == after.st_mode
    and before.st_uid == after.st_uid
    and before.st_nlink == after.st_nlink
    and before.st_size == after.st_size
    and before.st_mtime_ns == after.st_mtime_ns
    and before.st_ctime_ns == after.st_ctime_ns
  )


def _require_visible_directory(
  info: os.stat_result,
  *,
  private: bool,
  owner_uid: int,
  root_device: int | None,
) -> None:
  if not stat.S_ISDIR(info.st_mode):
    raise AutonomousRunSnapshotError(
      "directory_untrusted",
      "snapshot path component is not a directory",
    )
  if info.st_nlink < 1:
    raise AutonomousRunSnapshotError(
      "directory_untrusted",
      "snapshot directory has an invalid link count",
    )
  if not private:
    return
  if info.st_uid != owner_uid:
    raise AutonomousRunSnapshotError(
      "directory_untrusted",
      "private snapshot directory has an unexpected owner",
    )
  if stat.S_IMODE(info.st_mode) != _PRIVATE_DIRECTORY_MODE:
    raise AutonomousRunSnapshotError(
      "directory_untrusted",
      "private snapshot directory must have mode 0700",
    )
  if root_device is not None and info.st_dev != root_device:
    raise AutonomousRunSnapshotError(
      "directory_untrusted",
      "private snapshot directory crosses a device boundary",
    )


def _require_regular_file(
  info: os.stat_result,
  *,
  owner_uid: int,
  root_device: int,
  max_file_bytes: int,
) -> None:
  if not stat.S_ISREG(info.st_mode):
    raise AutonomousRunSnapshotError(
      "file_untrusted",
      "approved snapshot entry is not a regular file",
    )
  if info.st_uid != owner_uid:
    raise AutonomousRunSnapshotError(
      "file_untrusted",
      "approved snapshot file has an unexpected owner",
    )
  if info.st_nlink != 1:
    raise AutonomousRunSnapshotError(
      "file_untrusted",
      "approved snapshot file must have exactly one hard link",
    )
  if stat.S_IMODE(info.st_mode) not in _PRIVATE_FILE_MODES:
    raise AutonomousRunSnapshotError(
      "file_untrusted",
      "approved snapshot file must have mode 0400 or 0600",
    )
  if info.st_dev != root_device:
    raise AutonomousRunSnapshotError(
      "file_untrusted",
      "approved snapshot file crosses a device boundary",
    )
  if info.st_size < 0 or info.st_size > max_file_bytes:
    raise AutonomousRunSnapshotError(
      "snapshot_bound_exceeded",
      "approved snapshot file exceeds the per-file byte bound",
    )


def _visible_stat(parent_fd: int, name: str) -> os.stat_result:
  try:
    return os.stat(
      name,
      dir_fd=parent_fd,
      follow_symlinks=False,
    )
  except OSError as exc:
    raise AutonomousRunSnapshotError(
      "path_changed",
      "snapshot path binding is unavailable",
    ) from exc


def _open_directory_binding(
  parent_fd: int,
  name: str,
  *,
  owner_uid: int,
  private: bool,
  root_device: int | None,
) -> _DirectoryBinding:
  visible = _visible_stat(parent_fd, name)
  _require_visible_directory(
    visible,
    private=private,
    owner_uid=owner_uid,
    root_device=root_device,
  )
  try:
    descriptor = os.open(
      name,
      _directory_open_flags(),
      dir_fd=parent_fd,
    )
  except OSError as exc:
    raise AutonomousRunSnapshotError(
      "directory_untrusted",
      "snapshot directory cannot be opened without following links",
    ) from exc
  try:
    held = os.fstat(descriptor)
    _require_visible_directory(
      held,
      private=private,
      owner_uid=owner_uid,
      root_device=root_device,
    )
    if not _same_object(visible, held):
      raise AutonomousRunSnapshotError(
        "path_changed",
        "snapshot directory changed while being opened",
      )
    return _DirectoryBinding(
      parent_fd=parent_fd,
      name=name,
      fd=descriptor,
      initial=held,
      private=private,
    )
  except BaseException:
    os.close(descriptor)
    raise


def _canonical_root_parts(run_root: str | Path) -> tuple[str, ...]:
  raw = os.fspath(run_root)
  if type(raw) is not str or not raw or "\x00" in raw:
    raise AutonomousRunSnapshotError(
      "root_invalid",
      "private run root must be a filesystem path",
    )
  encoded_raw = os.fsencode(raw)
  if (
    not os.path.isabs(raw)
    or raw.startswith(os.sep * 2)
    or raw == os.sep
    or os.path.normpath(raw) != raw
    or len(encoded_raw) > _MAX_ROOT_PATH_BYTES
  ):
    raise AutonomousRunSnapshotError(
      "root_invalid",
      "private run root must be a canonical absolute directory path",
    )
  parts = tuple(part for part in Path(raw).parts if part != os.sep)
  if (
    not parts
    or len(parts) > _MAX_ROOT_COMPONENTS
    or any(
      part in {"", ".", ".."}
      or len(os.fsencode(part)) > _MAX_COMPONENT_BYTES
      for part in parts
    )
  ):
    raise AutonomousRunSnapshotError(
      "root_invalid",
      "private run root contains an invalid path component",
    )
  return parts


def _canonical_approved_paths(
  approved_relative_paths: tuple[str, ...],
  *,
  max_files: int,
) -> tuple[str, ...]:
  if type(approved_relative_paths) is not tuple:
    raise AutonomousRunSnapshotError(
      "approved_paths_invalid",
      "approved snapshot paths must be one closed tuple",
    )
  values = approved_relative_paths
  if not values:
    raise AutonomousRunSnapshotError(
      "approved_paths_invalid",
      "approved snapshot paths cannot be empty",
    )
  if len(values) > max_files:
    raise AutonomousRunSnapshotError(
      "snapshot_bound_exceeded",
      "approved snapshot path count exceeds the file bound",
    )

  canonical: list[str] = []
  for value in values:
    if type(value) is not str or not value or "\x00" in value:
      raise AutonomousRunSnapshotError(
        "approved_paths_invalid",
        "each approved snapshot path must be a non-empty string",
      )
    encoded_value = value.encode("utf-8")
    if len(encoded_value) > _MAX_RELATIVE_PATH_BYTES:
      raise AutonomousRunSnapshotError(
        "approved_paths_invalid",
        "approved snapshot path exceeds the path byte bound",
      )
    parsed = PurePosixPath(value)
    parts = parsed.parts
    if (
      parsed.is_absolute()
      or str(parsed) != value
      or not parts
      or len(parts) > _MAX_RELATIVE_PATH_COMPONENTS
      or any(
        part in {"", ".", ".."}
        or len(part.encode("utf-8")) > _MAX_COMPONENT_BYTES
        for part in parts
      )
    ):
      raise AutonomousRunSnapshotError(
        "approved_paths_invalid",
        "approved snapshot paths must be canonical relative file paths",
      )
    canonical.append(value)

  if len(set(canonical)) != len(canonical):
    raise AutonomousRunSnapshotError(
      "approved_paths_invalid",
      "approved snapshot paths must be unique",
    )
  return tuple(sorted(canonical))


def _bounded_limit(name: str, value: int, hard_maximum: int) -> int:
  if type(value) is not int or value < 1 or value > hard_maximum:
    raise AutonomousRunSnapshotError(
      "snapshot_bound_invalid",
      f"{name} must be between 1 and {hard_maximum}",
    )
  return value


def _read_exact_file(
  descriptor: int,
  expected_size: int,
) -> bytes:
  chunks: list[bytes] = []
  remaining = expected_size
  while remaining:
    try:
      chunk = os.read(
        descriptor,
        min(_READ_CHUNK_BYTES, remaining),
      )
    except OSError as exc:
      raise AutonomousRunSnapshotError(
        "file_changed",
        "approved snapshot file could not be read",
      ) from exc
    if not chunk:
      raise AutonomousRunSnapshotError(
        "file_changed",
        "approved snapshot file truncated while being read",
      )
    chunks.append(chunk)
    remaining -= len(chunk)
  try:
    overflow = os.read(descriptor, 1)
  except OSError as exc:
    raise AutonomousRunSnapshotError(
      "file_changed",
      "approved snapshot file EOF could not be verified",
    ) from exc
  if overflow:
    raise AutonomousRunSnapshotError(
      "file_changed",
      "approved snapshot file grew while being read",
    )
  return b"".join(chunks)


def _read_verified_descriptor(
  descriptor: int,
  *,
  expected: os.stat_result,
  owner_uid: int,
  root_device: int,
  max_file_bytes: int,
) -> bytes:
  before = os.fstat(descriptor)
  _require_regular_file(
    before,
    owner_uid=owner_uid,
    root_device=root_device,
    max_file_bytes=max_file_bytes,
  )
  if not _stable_file_attributes(expected, before):
    raise AutonomousRunSnapshotError(
      "file_changed",
      "approved snapshot file identity changed before read",
    )
  raw = _read_exact_file(descriptor, before.st_size)
  after = os.fstat(descriptor)
  if (
    not _stable_file_attributes(before, after)
    or len(raw) != after.st_size
  ):
    raise AutonomousRunSnapshotError(
      "file_changed",
      "approved snapshot file changed while being read",
    )
  return raw


def _open_and_capture_file(
  *,
  parent_fd: int,
  name: str,
  relative_path: str,
  owner_uid: int,
  root_device: int,
  max_file_bytes: int,
  remaining_total_bytes: int,
) -> _CapturedFile:
  visible_before = _visible_stat(parent_fd, name)
  _require_regular_file(
    visible_before,
    owner_uid=owner_uid,
    root_device=root_device,
    max_file_bytes=max_file_bytes,
  )
  if visible_before.st_size > remaining_total_bytes:
    raise AutonomousRunSnapshotError(
      "snapshot_bound_exceeded",
      "approved snapshot files exceed the aggregate byte bound",
    )
  try:
    descriptor = os.open(
      name,
      _file_open_flags(),
      dir_fd=parent_fd,
    )
  except OSError as exc:
    raise AutonomousRunSnapshotError(
      "file_untrusted",
      "approved snapshot file cannot be opened without following links",
    ) from exc
  reopened = -1
  try:
    raw = _read_verified_descriptor(
      descriptor,
      expected=visible_before,
      owner_uid=owner_uid,
      root_device=root_device,
      max_file_bytes=max_file_bytes,
    )
    held_after = os.fstat(descriptor)
    visible_after = _visible_stat(parent_fd, name)
    if (
      not _stable_file_attributes(visible_before, held_after)
      or not _stable_file_attributes(held_after, visible_after)
    ):
      raise AutonomousRunSnapshotError(
        "path_changed",
        "approved snapshot file path changed during capture",
      )
    try:
      reopened = os.open(
        name,
        _file_open_flags(),
        dir_fd=parent_fd,
      )
    except OSError as exc:
      raise AutonomousRunSnapshotError(
        "path_changed",
        "approved snapshot file cannot be reopened safely",
      ) from exc
    reopened_raw = _read_verified_descriptor(
      reopened,
      expected=held_after,
      owner_uid=owner_uid,
      root_device=root_device,
      max_file_bytes=max_file_bytes,
    )
    if reopened_raw != raw:
      raise AutonomousRunSnapshotError(
        "file_changed",
        "approved snapshot file bytes changed on reopen",
      )
    reopened_after = os.fstat(reopened)
    visible_reopened = _visible_stat(parent_fd, name)
    if (
      not _stable_file_attributes(held_after, reopened_after)
      or not _stable_file_attributes(
        reopened_after,
        visible_reopened,
      )
    ):
      raise AutonomousRunSnapshotError(
        "path_changed",
        "approved snapshot file reopen identity changed",
      )
    captured = _CapturedFile(
      snapshot=AutonomousRunFileSnapshot(
        relative_path=relative_path,
        raw_bytes=raw,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
      ),
      parent_fd=parent_fd,
      name=name,
      initial=held_after,
      descriptor=descriptor,
    )
  except BaseException:
    cleanup_error = _close_descriptors_once(
      tuple(
        local_descriptor
        for local_descriptor in (reopened, descriptor)
        if local_descriptor >= 0
      )
    )
    if cleanup_error is not None:
      _raise_cleanup_failure(cleanup_error)
    raise

  reopened_cleanup_error = _close_descriptors_once((reopened,))
  if reopened_cleanup_error is not None:
    _close_descriptors_once((descriptor,))
    _raise_cleanup_failure(reopened_cleanup_error)
  return captured


def _revalidate_directory_binding(
  binding: _DirectoryBinding,
  *,
  owner_uid: int,
  root_device: int,
) -> None:
  held = os.fstat(binding.fd)
  _require_visible_directory(
    held,
    private=binding.private,
    owner_uid=owner_uid,
    root_device=root_device if binding.private else None,
  )
  if not _stable_directory_attributes(binding.initial, held):
    raise AutonomousRunSnapshotError(
      "path_changed",
      "snapshot directory attributes changed during capture",
    )
  if binding.parent_fd is None:
    visible = os.stat(os.sep, follow_symlinks=False)
  else:
    visible = _visible_stat(binding.parent_fd, binding.name)
  if not _same_object(held, visible):
    raise AutonomousRunSnapshotError(
      "path_changed",
      "snapshot directory path identity changed during capture",
    )
  try:
    if binding.parent_fd is None:
      reopened_fd = os.open(os.sep, _directory_open_flags())
    else:
      reopened_fd = os.open(
        binding.name,
        _directory_open_flags(),
        dir_fd=binding.parent_fd,
      )
  except OSError as exc:
    raise AutonomousRunSnapshotError(
      "path_changed",
      "snapshot directory cannot be reopened safely",
    ) from exc
  try:
    reopened = os.fstat(reopened_fd)
    if not _stable_directory_attributes(held, reopened):
      raise AutonomousRunSnapshotError(
        "path_changed",
        "snapshot directory reopen identity changed",
      )
  finally:
    os.close(reopened_fd)


def _revalidate_captured_file(
  captured: _CapturedFile,
  *,
  owner_uid: int,
  root_device: int,
  max_file_bytes: int,
) -> None:
  visible = _visible_stat(captured.parent_fd, captured.name)
  _require_regular_file(
    visible,
    owner_uid=owner_uid,
    root_device=root_device,
    max_file_bytes=max_file_bytes,
  )
  if not _stable_file_attributes(captured.initial, visible):
    raise AutonomousRunSnapshotError(
      "path_changed",
      "approved snapshot file binding changed before completion",
    )
  try:
    offset = os.lseek(captured.descriptor, 0, os.SEEK_SET)
  except OSError as exc:
    raise AutonomousRunSnapshotError(
      "file_changed",
      "pinned snapshot file cannot be rewound",
    ) from exc
  if offset != 0:
    raise AutonomousRunSnapshotError(
      "file_changed",
      "pinned snapshot file rewind did not reach the start",
    )
  pinned_raw = _read_verified_descriptor(
    captured.descriptor,
    expected=captured.initial,
    owner_uid=owner_uid,
    root_device=root_device,
    max_file_bytes=max_file_bytes,
  )
  if pinned_raw != captured.snapshot.raw_bytes:
    raise AutonomousRunSnapshotError(
      "file_changed",
      "pinned snapshot file bytes changed before completion",
    )
  try:
    descriptor = os.open(
      captured.name,
      _file_open_flags(),
      dir_fd=captured.parent_fd,
    )
  except OSError as exc:
    raise AutonomousRunSnapshotError(
      "path_changed",
      "approved snapshot file cannot be reopened before completion",
    ) from exc
  try:
    raw = _read_verified_descriptor(
      descriptor,
      expected=captured.initial,
      owner_uid=owner_uid,
      root_device=root_device,
      max_file_bytes=max_file_bytes,
    )
    if raw != captured.snapshot.raw_bytes:
      raise AutonomousRunSnapshotError(
        "file_changed",
        "approved snapshot file bytes changed before completion",
      )
    held_after = os.fstat(descriptor)
    visible_after = _visible_stat(
      captured.parent_fd,
      captured.name,
    )
    if (
      not _stable_file_attributes(captured.initial, held_after)
      or not _stable_file_attributes(held_after, visible_after)
    ):
      raise AutonomousRunSnapshotError(
        "path_changed",
        "approved snapshot file changed during final reopen",
      )
  finally:
    os.close(descriptor)


def _snapshot_digest(
  files: tuple[AutonomousRunFileSnapshot, ...],
) -> str:
  digest = hashlib.sha256()
  digest.update(_SNAPSHOT_DIGEST_DOMAIN)
  digest.update(len(files).to_bytes(8, "big"))
  for file_snapshot in files:
    encoded_path = file_snapshot.relative_path.encode("utf-8")
    digest.update(len(encoded_path).to_bytes(8, "big"))
    digest.update(encoded_path)
    digest.update(file_snapshot.size_bytes.to_bytes(8, "big"))
    digest.update(file_snapshot.raw_bytes)
  return digest.hexdigest()


def snapshot_autonomous_run_root(
  run_root: str | Path,
  approved_relative_paths: tuple[str, ...],
  *,
  max_files: int = MAX_AUTONOMOUS_RUN_SNAPSHOT_FILES,
  max_file_bytes: int = MAX_AUTONOMOUS_RUN_SNAPSHOT_FILE_BYTES,
  max_total_bytes: int = MAX_AUTONOMOUS_RUN_SNAPSHOT_TOTAL_BYTES,
) -> AutonomousRunSnapshot:
  """Freeze an exact bounded set of private run-root files without path fallback."""

  max_files = _bounded_limit(
    "max_files",
    max_files,
    MAX_AUTONOMOUS_RUN_SNAPSHOT_FILES,
  )
  max_file_bytes = _bounded_limit(
    "max_file_bytes",
    max_file_bytes,
    MAX_AUTONOMOUS_RUN_SNAPSHOT_FILE_BYTES,
  )
  max_total_bytes = _bounded_limit(
    "max_total_bytes",
    max_total_bytes,
    MAX_AUTONOMOUS_RUN_SNAPSHOT_TOTAL_BYTES,
  )
  root_parts = _canonical_root_parts(run_root)
  approved_paths = _canonical_approved_paths(
    approved_relative_paths,
    max_files=max_files,
  )
  owner_uid = os.geteuid()
  bindings: list[_DirectoryBinding] = []
  owned_descriptors: list[int] = []
  captured_files: list[_CapturedFile] = []
  directory_by_parts: dict[tuple[str, ...], _DirectoryBinding] = {}
  aggregate_size = 0

  try:
    try:
      anchor_fd = os.open(os.sep, _directory_open_flags())
    except OSError as exc:
      raise AutonomousRunSnapshotError(
        "root_untrusted",
        "filesystem root cannot be opened safely",
      ) from exc
    owned_descriptors.append(anchor_fd)
    anchor_info = os.fstat(anchor_fd)
    _require_visible_directory(
      anchor_info,
      private=False,
      owner_uid=owner_uid,
      root_device=None,
    )
    anchor = _DirectoryBinding(
      parent_fd=None,
      name=os.sep,
      fd=anchor_fd,
      initial=anchor_info,
      private=False,
    )
    bindings.append(anchor)

    current = anchor
    for index, part in enumerate(root_parts):
      child = _open_directory_binding(
        current.fd,
        part,
        owner_uid=owner_uid,
        private=index == len(root_parts) - 1,
        root_device=None,
      )
      owned_descriptors.append(child.fd)
      bindings.append(child)
      current = child
    root_binding = current
    root_device = root_binding.initial.st_dev
    directory_by_parts[()] = root_binding

    for relative_path in approved_paths:
      parts = PurePosixPath(relative_path).parts
      parent_parts: tuple[str, ...] = ()
      parent = root_binding
      for component in parts[:-1]:
        candidate_parts = (*parent_parts, component)
        child = directory_by_parts.get(candidate_parts)
        if child is None:
          child = _open_directory_binding(
            parent.fd,
            component,
            owner_uid=owner_uid,
            private=True,
            root_device=root_device,
          )
          owned_descriptors.append(child.fd)
          bindings.append(child)
          directory_by_parts[candidate_parts] = child
        parent = child
        parent_parts = candidate_parts

      captured = _open_and_capture_file(
        parent_fd=parent.fd,
        name=parts[-1],
        relative_path=relative_path,
        owner_uid=owner_uid,
        root_device=root_device,
        max_file_bytes=max_file_bytes,
        remaining_total_bytes=max_total_bytes - aggregate_size,
      )
      owned_descriptors.append(captured.descriptor)
      aggregate_size += captured.snapshot.size_bytes
      captured_files.append(captured)

    for binding in bindings:
      _revalidate_directory_binding(
        binding,
        owner_uid=owner_uid,
        root_device=root_device,
      )
    for captured in captured_files:
      _revalidate_captured_file(
        captured,
        owner_uid=owner_uid,
        root_device=root_device,
        max_file_bytes=max_file_bytes,
      )
    for binding in bindings:
      _revalidate_directory_binding(
        binding,
        owner_uid=owner_uid,
        root_device=root_device,
      )
    if os.geteuid() != owner_uid:
      raise AutonomousRunSnapshotError(
        "root_untrusted",
        "effective user changed during snapshot capture",
      )

    immutable_files = tuple(
      captured.snapshot
      for captured in captured_files
    )
    total_bytes = sum(
      file_snapshot.size_bytes
      for file_snapshot in immutable_files
    )
    return AutonomousRunSnapshot(
      files=immutable_files,
      total_bytes=total_bytes,
      sha256=_snapshot_digest(immutable_files),
    )
  except AutonomousRunSnapshotError:
    raise
  except OSError as exc:
    raise AutonomousRunSnapshotError(
      "snapshot_failed",
      "private autonomous run snapshot failed closed",
    ) from exc
  finally:
    cleanup_error = _close_descriptors_once(
      tuple(reversed(owned_descriptors))
    )
    if cleanup_error is not None:
      _raise_cleanup_failure(cleanup_error)


__all__ = [
  "AutonomousRunFileSnapshot",
  "AutonomousRunSnapshot",
  "AutonomousRunSnapshotError",
  "MAX_AUTONOMOUS_RUN_SNAPSHOT_FILES",
  "MAX_AUTONOMOUS_RUN_SNAPSHOT_FILE_BYTES",
  "MAX_AUTONOMOUS_RUN_SNAPSHOT_TOTAL_BYTES",
  "snapshot_autonomous_run_root",
]
