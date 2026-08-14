from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import time
from typing import Any, Iterator


_LOCK_FILE_MODE = 0o600
_LOCK_NAME_DOMAIN = b"agent-gateway/canonical-json-target-lock/v1\x00"
_LOCK_NAME_PREFIX = ".canonical-json-target-lock."
_LOCK_NAME_SUFFIX = ".lock"
_LOCK_RETRY_INITIAL_SECONDS = 0.005
_LOCK_RETRY_MAX_SECONDS = 0.05
CANONICAL_JSON_TARGET_LOCK_TIMEOUT_SECONDS = 30.0
MAX_CANONICAL_JSON_TARGET_BYTES = 4 * 1024 * 1024
_MAX_CANONICAL_JSON_TARGET_FILE_BYTES = (
  MAX_CANONICAL_JSON_TARGET_BYTES + 1
)
_PARENT_DIRECTORY_MODE = 0o700
_TARGET_FILE_MODE = 0o600


class CanonicalJsonTargetLockError(RuntimeError):
  """The stable lock for a canonical JSON target cannot be trusted."""


class CanonicalJsonTargetLockTimeout(
  CanonicalJsonTargetLockError
):
  """The bounded wait for a canonical JSON target lock expired."""


@dataclass(frozen=True, slots=True)
class LockedCanonicalJsonTarget:
  parent_fd: int
  target_name: str
  lock_name: str


@dataclass(frozen=True, slots=True)
class CanonicalJsonTargetSnapshot:
  exists: bool
  value: Any | None
  raw: bytes | None
  parse_error: str | None


def canonical_json_target_lock_name(target_name: str) -> str:
  normalized = _require_target_name(target_name)
  digest = hashlib.sha256(
    _LOCK_NAME_DOMAIN + os.fsencode(normalized)
  ).hexdigest()
  return f"{_LOCK_NAME_PREFIX}{digest}{_LOCK_NAME_SUFFIX}"


def _require_target_name(target_name: str) -> str:
  if (
    type(target_name) is not str
    or not target_name
    or target_name in {".", ".."}
    or os.path.basename(target_name) != target_name
  ):
    raise ValueError("canonical JSON target name must be one basename")
  return target_name


def _require_platform_flag(name: str) -> int:
  value = getattr(os, name, None)
  if type(value) is not int or value == 0:
    raise CanonicalJsonTargetLockError(
      f"platform does not provide required {name}"
    )
  return value


def _require_verified_parent_fd(parent_fd: int) -> os.stat_result:
  info = os.fstat(parent_fd)
  if not stat.S_ISDIR(info.st_mode):
    raise CanonicalJsonTargetLockError(
      "canonical JSON target parent is not a directory"
    )
  if info.st_uid != os.geteuid():
    raise CanonicalJsonTargetLockError(
      "canonical JSON target parent is not owned by the service user"
    )
  if info.st_nlink < 1:
    raise CanonicalJsonTargetLockError(
      "canonical JSON target parent has an invalid link count"
    )
  if stat.S_IMODE(info.st_mode) & 0o022:
    raise CanonicalJsonTargetLockError(
      "canonical JSON target parent is group/world writable"
    )
  return info


def _require_traversal_directory_stat(
  info: os.stat_result,
  *,
  path_label: str,
) -> None:
  if not stat.S_ISDIR(info.st_mode):
    raise CanonicalJsonTargetLockError(
      f"canonical JSON path component {path_label!r} is not a directory"
    )
  if info.st_uid not in {0, os.geteuid()}:
    raise CanonicalJsonTargetLockError(
      f"canonical JSON path component {path_label!r} has an untrusted owner"
    )
  if info.st_nlink < 1:
    raise CanonicalJsonTargetLockError(
      f"canonical JSON path component {path_label!r} has an invalid link count"
    )
  if stat.S_IMODE(info.st_mode) & 0o022:
    raise CanonicalJsonTargetLockError(
      f"canonical JSON path component {path_label!r} is group/world writable"
    )


def _require_lock_stat(
  info: os.stat_result,
  *,
  lock_name: str,
) -> None:
  if not stat.S_ISREG(info.st_mode):
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target lock {lock_name!r} is not a regular file"
    )
  if info.st_uid != os.geteuid():
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target lock {lock_name!r} has the wrong owner"
    )
  if info.st_nlink != 1:
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target lock {lock_name!r} must have one hard link"
    )
  mode = stat.S_IMODE(info.st_mode)
  if mode != _LOCK_FILE_MODE:
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target lock {lock_name!r} must have mode "
      f"{_LOCK_FILE_MODE:04o}, got {mode:04o}"
    )


def _require_target_stat(
  info: os.stat_result,
  *,
  target_name: str,
) -> None:
  if not stat.S_ISREG(info.st_mode):
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target {target_name!r} is not a regular file"
    )
  if info.st_uid != os.geteuid():
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target {target_name!r} has the wrong owner"
    )
  if info.st_nlink != 1:
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target {target_name!r} must have one hard link"
    )
  if stat.S_IMODE(info.st_mode) & 0o022:
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target {target_name!r} is group/world writable"
    )


def _require_lock_binding(
  *,
  parent_fd: int,
  lock_fd: int,
  lock_name: str,
) -> None:
  descriptor_info = os.fstat(lock_fd)
  _require_lock_stat(descriptor_info, lock_name=lock_name)
  try:
    path_info = os.stat(
      lock_name,
      dir_fd=parent_fd,
      follow_symlinks=False,
    )
  except FileNotFoundError as exc:
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target lock {lock_name!r} lost its binding"
    ) from exc
  _require_lock_stat(path_info, lock_name=lock_name)
  if (
    descriptor_info.st_dev != path_info.st_dev
    or descriptor_info.st_ino != path_info.st_ino
  ):
    raise CanonicalJsonTargetLockError(
      f"canonical JSON target lock {lock_name!r} binding changed"
    )


def _open_lock_file(parent_fd: int, lock_name: str) -> int:
  nofollow = _require_platform_flag("O_NOFOLLOW")
  cloexec = _require_platform_flag("O_CLOEXEC")
  common_flags = os.O_RDWR | cloexec | nofollow
  created = False
  try:
    lock_fd = os.open(
      lock_name,
      common_flags | os.O_CREAT | os.O_EXCL,
      _LOCK_FILE_MODE,
      dir_fd=parent_fd,
    )
    created = True
  except FileExistsError:
    try:
      lock_fd = os.open(
        lock_name,
        common_flags,
        dir_fd=parent_fd,
      )
    except OSError as exc:
      raise CanonicalJsonTargetLockError(
        f"cannot open canonical JSON target lock {lock_name!r}"
      ) from exc
  except OSError as exc:
    raise CanonicalJsonTargetLockError(
      f"cannot create canonical JSON target lock {lock_name!r}"
    ) from exc
  os.set_inheritable(lock_fd, False)
  if created:
    try:
      os.fchmod(lock_fd, _LOCK_FILE_MODE)
      _require_lock_stat(
        os.fstat(lock_fd),
        lock_name=lock_name,
      )
      os.fsync(lock_fd)
      os.fsync(parent_fd)
    except BaseException:
      os.close(lock_fd)
      raise
  return lock_fd


def _acquire_lock(lock_fd: int, *, lock_name: str) -> None:
  timeout = CANONICAL_JSON_TARGET_LOCK_TIMEOUT_SECONDS
  if (
    type(timeout) not in {int, float}
    or timeout < 0
  ):
    raise CanonicalJsonTargetLockError(
      "canonical JSON target lock timeout is invalid"
    )
  deadline = time.monotonic() + float(timeout)
  delay = _LOCK_RETRY_INITIAL_SECONDS
  while True:
    try:
      fcntl.flock(
        lock_fd,
        fcntl.LOCK_EX | fcntl.LOCK_NB,
      )
      return
    except OSError as exc:
      if exc.errno not in {errno.EACCES, errno.EAGAIN}:
        raise CanonicalJsonTargetLockError(
          f"cannot acquire canonical JSON target lock {lock_name!r}"
        ) from exc
    remaining = deadline - time.monotonic()
    if remaining <= 0:
      raise CanonicalJsonTargetLockTimeout(
        f"timed out acquiring canonical JSON target lock "
        f"{lock_name!r}"
      )
    time.sleep(min(delay, remaining))
    delay = min(delay * 2, _LOCK_RETRY_MAX_SECONDS)


def _directory_open_flags() -> int:
  directory = _require_platform_flag("O_DIRECTORY")
  nofollow = _require_platform_flag("O_NOFOLLOW")
  cloexec = _require_platform_flag("O_CLOEXEC")
  return os.O_RDONLY | directory | nofollow | cloexec


def _open_traversal_directory(
  path: str,
  *,
  parent_fd: int | None = None,
) -> int:
  try:
    directory_fd = os.open(
      path,
      _directory_open_flags(),
      dir_fd=parent_fd,
    )
  except OSError as exc:
    raise CanonicalJsonTargetLockError(
      f"cannot open canonical JSON path component {path!r}"
    ) from exc
  os.set_inheritable(directory_fd, False)
  try:
    _require_traversal_directory_stat(
      os.fstat(directory_fd),
      path_label=path,
    )
  except BaseException:
    os.close(directory_fd)
    raise
  return directory_fd


def _open_or_create_traversal_directory(
  parent_fd: int,
  name: str,
  *,
  create: bool,
) -> int:
  try:
    return _open_traversal_directory(
      name,
      parent_fd=parent_fd,
    )
  except CanonicalJsonTargetLockError as open_error:
    cause = open_error.__cause__
    if not create or not isinstance(cause, FileNotFoundError):
      raise
  _require_verified_parent_fd(parent_fd)
  try:
    os.mkdir(
      name,
      _PARENT_DIRECTORY_MODE,
      dir_fd=parent_fd,
    )
  except FileExistsError:
    pass
  except OSError as exc:
    raise CanonicalJsonTargetLockError(
      f"cannot create canonical JSON path component {name!r}"
    ) from exc
  else:
    os.fsync(parent_fd)
  return _open_traversal_directory(
    name,
    parent_fd=parent_fd,
  )


def _reject_ambiguous_path_parts(
  path_text: str,
  *,
  absolute: bool,
) -> None:
  if not path_text or "\x00" in path_text:
    raise ValueError("canonical JSON path must be non-empty")
  if os.path.altsep and os.path.altsep in path_text:
    raise ValueError("canonical JSON path uses an alternate separator")
  parts = path_text.split(os.sep)
  if absolute:
    if not path_text.startswith(os.sep):
      raise ValueError("canonical JSON anchor must be absolute")
    parts = parts[1:]
  if any(part in {"", ".", ".."} for part in parts):
    raise ValueError(
      "canonical JSON path must not contain empty, dot, or parent components"
    )


def _normalized_absolute_path(path: str | Path) -> Path:
  path_text = os.path.expanduser(os.fspath(path))
  if os.path.isabs(path_text):
    _reject_ambiguous_path_parts(
      path_text,
      absolute=True,
    )
    return Path(path_text)
  _reject_ambiguous_path_parts(
    path_text,
    absolute=False,
  )
  absolute = os.path.join(os.getcwd(), path_text)
  _reject_ambiguous_path_parts(
    absolute,
    absolute=True,
  )
  return Path(absolute)


def _open_absolute_directory(
  path: str | Path,
  *,
  create: bool,
) -> int:
  absolute = _normalized_absolute_path(path)
  directory_fd = _open_traversal_directory(os.sep)
  try:
    for component in absolute.parts[1:]:
      next_fd = _open_or_create_traversal_directory(
        directory_fd,
        component,
        create=create,
      )
      os.close(directory_fd)
      directory_fd = next_fd
    return directory_fd
  except BaseException:
    os.close(directory_fd)
    raise


def _relative_target_parts(path: str | Path) -> tuple[str, ...]:
  path_text = os.fspath(path)
  if os.path.isabs(path_text):
    raise ValueError(
      "canonical JSON relative target must not be absolute"
    )
  _reject_ambiguous_path_parts(
    path_text,
    absolute=False,
  )
  parts = tuple(path_text.split(os.sep))
  _require_target_name(parts[-1])
  return parts


def _open_relative_target_parent(
  root_fd: int,
  relative_target: str | Path,
  *,
  create_parents: bool,
) -> tuple[int, str]:
  parts = _relative_target_parts(relative_target)
  parent_fd = os.dup(root_fd)
  os.set_inheritable(parent_fd, False)
  try:
    for component in parts[:-1]:
      next_fd = _open_or_create_traversal_directory(
        parent_fd,
        component,
        create=create_parents,
      )
      os.close(parent_fd)
      parent_fd = next_fd
    _require_verified_parent_fd(parent_fd)
    return parent_fd, parts[-1]
  except BaseException:
    os.close(parent_fd)
    raise


def _read_all_bounded(
  fd: int,
  *,
  target_name: str,
) -> bytes:
  chunks: list[bytes] = []
  total = 0
  while True:
    chunk = os.read(
      fd,
      min(
        64 * 1024,
        _MAX_CANONICAL_JSON_TARGET_FILE_BYTES + 1 - total,
      ),
    )
    if not chunk:
      return b"".join(chunks)
    chunks.append(chunk)
    total += len(chunk)
    if total > _MAX_CANONICAL_JSON_TARGET_FILE_BYTES:
      raise CanonicalJsonTargetLockError(
        f"canonical JSON target {target_name!r} exceeds "
        "its bounded size"
      )


def read_locked_canonical_json_target(
  target: LockedCanonicalJsonTarget,
) -> CanonicalJsonTargetSnapshot:
  """Read one locked target relative to its verified parent."""

  nofollow = _require_platform_flag("O_NOFOLLOW")
  cloexec = _require_platform_flag("O_CLOEXEC")
  try:
    fd = os.open(
      target.target_name,
      os.O_RDONLY | nofollow | cloexec,
      dir_fd=target.parent_fd,
    )
  except FileNotFoundError:
    return CanonicalJsonTargetSnapshot(
      exists=False,
      value=None,
      raw=None,
      parse_error=None,
    )
  except OSError as exc:
    raise CanonicalJsonTargetLockError(
      f"cannot open canonical JSON target {target.target_name!r}"
    ) from exc
  try:
    os.set_inheritable(fd, False)
    _require_target_stat(
      os.fstat(fd),
      target_name=target.target_name,
    )
    raw = _read_all_bounded(
      fd,
      target_name=target.target_name,
    )
  finally:
    os.close(fd)
  try:
    value = json.loads(raw.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    return CanonicalJsonTargetSnapshot(
      exists=True,
      value=None,
      raw=raw,
      parse_error=str(exc),
    )
  return CanonicalJsonTargetSnapshot(
    exists=True,
    value=value,
    raw=raw,
    parse_error=None,
  )


def _canonical_json_bytes(value: Any) -> bytes:
  try:
    return json.dumps(
      value,
      allow_nan=False,
      ensure_ascii=True,
      separators=(",", ":"),
      sort_keys=True,
    ).encode("utf-8")
  except (TypeError, ValueError) as exc:
    raise CanonicalJsonTargetLockError(
      "canonical JSON target replacement is not JSON-compatible"
    ) from exc


def _write_all(fd: int, payload: bytes) -> None:
  offset = 0
  while offset < len(payload):
    written = os.write(fd, payload[offset:])
    if written <= 0:
      raise OSError(
        "short write while persisting canonical JSON target"
      )
    offset += written


def write_locked_canonical_json_target(
  target: LockedCanonicalJsonTarget,
  value: Any,
) -> CanonicalJsonTargetSnapshot:
  """Durably replace and exactly read back one locked JSON target."""

  canonical_payload = _canonical_json_bytes(value)
  if len(canonical_payload) > MAX_CANONICAL_JSON_TARGET_BYTES:
    raise CanonicalJsonTargetLockError(
      "canonical JSON target replacement exceeds its bounded size"
    )
  payload = canonical_payload + b"\n"
  nofollow = _require_platform_flag("O_NOFOLLOW")
  cloexec = _require_platform_flag("O_CLOEXEC")
  temp_name = (
    f".{target.target_name}.canonical-json."
    f"{secrets.token_hex(16)}.tmp"
  )
  flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | nofollow
    | cloexec
  )
  temp_fd: int | None = None
  try:
    temp_fd = os.open(
      temp_name,
      flags,
      _TARGET_FILE_MODE,
      dir_fd=target.parent_fd,
    )
    os.set_inheritable(temp_fd, False)
    os.fchmod(temp_fd, _TARGET_FILE_MODE)
    _require_target_stat(
      os.fstat(temp_fd),
      target_name=temp_name,
    )
    _write_all(temp_fd, payload)
    os.fsync(temp_fd)
    os.close(temp_fd)
    temp_fd = None
    os.rename(
      temp_name,
      target.target_name,
      src_dir_fd=target.parent_fd,
      dst_dir_fd=target.parent_fd,
    )
    os.fsync(target.parent_fd)
    readback = read_locked_canonical_json_target(target)
    if (
      not readback.exists
      or readback.parse_error is not None
      or readback.raw != payload
    ):
      raise CanonicalJsonTargetLockError(
        "canonical JSON target exact readback mismatch"
      )
    return readback
  except BaseException:
    if temp_fd is not None:
      os.close(temp_fd)
    try:
      os.unlink(temp_name, dir_fd=target.parent_fd)
    except FileNotFoundError:
      pass
    raise


@contextmanager
def lock_canonical_json_target(
  parent_fd: int,
  target_name: str,
) -> Iterator[LockedCanonicalJsonTarget]:
  """Hold the stable cross-process lock for one target basename."""

  _require_verified_parent_fd(parent_fd)
  normalized = _require_target_name(target_name)
  lock_name = canonical_json_target_lock_name(normalized)
  lock_fd = _open_lock_file(parent_fd, lock_name)
  locked = False
  try:
    _require_lock_binding(
      parent_fd=parent_fd,
      lock_fd=lock_fd,
      lock_name=lock_name,
    )
    _acquire_lock(lock_fd, lock_name=lock_name)
    locked = True
    _require_lock_binding(
      parent_fd=parent_fd,
      lock_fd=lock_fd,
      lock_name=lock_name,
    )
    try:
      yield LockedCanonicalJsonTarget(
        parent_fd=parent_fd,
        target_name=normalized,
        lock_name=lock_name,
      )
    finally:
      _require_lock_binding(
        parent_fd=parent_fd,
        lock_fd=lock_fd,
        lock_name=lock_name,
      )
  finally:
    try:
      if locked:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
      os.close(lock_fd)


@contextmanager
def lock_canonical_json_path(
  target_path: str | Path,
  *,
  create_parents: bool = False,
) -> Iterator[LockedCanonicalJsonTarget]:
  """Traverse a path without following links and lock its JSON target."""

  path = _normalized_absolute_path(target_path)
  _require_target_name(path.name)
  parent_fd = _open_absolute_directory(
    path.parent,
    create=create_parents,
  )
  try:
    with lock_canonical_json_target(parent_fd, path.name) as target:
      yield target
  finally:
    os.close(parent_fd)


@contextmanager
def lock_canonical_json_relative_path(
  root_path: str | Path,
  relative_target: str | Path,
  *,
  create_parents: bool = False,
) -> Iterator[LockedCanonicalJsonTarget]:
  """Lock a target reached without links beneath one verified root."""

  root_fd = _open_absolute_directory(
    root_path,
    create=create_parents,
  )
  try:
    _require_verified_parent_fd(root_fd)
    parent_fd, target_name = _open_relative_target_parent(
      root_fd,
      relative_target,
      create_parents=create_parents,
    )
  finally:
    os.close(root_fd)
  try:
    with lock_canonical_json_target(
      parent_fd,
      target_name,
    ) as target:
      yield target
  finally:
    os.close(parent_fd)


__all__ = [
  "CanonicalJsonTargetLockError",
  "CanonicalJsonTargetLockTimeout",
  "CANONICAL_JSON_TARGET_LOCK_TIMEOUT_SECONDS",
  "CanonicalJsonTargetSnapshot",
  "LockedCanonicalJsonTarget",
  "MAX_CANONICAL_JSON_TARGET_BYTES",
  "canonical_json_target_lock_name",
  "lock_canonical_json_path",
  "lock_canonical_json_relative_path",
  "lock_canonical_json_target",
  "read_locked_canonical_json_target",
  "write_locked_canonical_json_target",
]
