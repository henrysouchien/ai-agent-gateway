from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import os
from pathlib import Path
import stat
from typing import Any

from .autonomous_control_contract import (
  AUTONOMOUS_CONTROL_RECORD_MAX_BYTES,
  decode_closed_control_record,
  encode_closed_control_record,
  read_bounded_control_line,
)


class AutonomousControlAppendError(RuntimeError):
  """A control append failed after mutation may have begun."""

  def __init__(
    self,
    message: str,
    *,
    stream_recovered: bool,
  ) -> None:
    super().__init__(message)
    self.stream_recovered = stream_recovered


def secure_create_owned_file(path: Path) -> tuple[int, os.stat_result]:
  flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
  )
  fd = os.open(path, flags, 0o600)
  try:
    os.fchmod(fd, 0o600)
    file_stat = os.fstat(fd)
    if (
      not stat.S_ISREG(file_stat.st_mode)
      or stat.S_IMODE(file_stat.st_mode) != 0o600
      or file_stat.st_nlink != 1
      or file_stat.st_uid != os.geteuid()
    ):
      raise RuntimeError(
        f"autonomous control file has unsafe identity: {path}"
      )
    return fd, file_stat
  except BaseException:
    os.close(fd)
    raise


def fsync_owned_file_directory(path: Path) -> None:
  directory_fd = os.open(
    path.parent,
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0),
  )
  try:
    os.fsync(directory_fd)
  finally:
    os.close(directory_fd)


def require_existing_owned_file_identity(
  path: Path,
  *,
  field_name: str,
) -> os.stat_result:
  try:
    file_stat = os.lstat(path)
  except OSError as exc:
    raise RuntimeError(
      f"autonomous {field_name} must be a preexisting regular file"
    ) from exc
  if (
    not stat.S_ISREG(file_stat.st_mode)
    or stat.S_IMODE(file_stat.st_mode) != 0o600
    or file_stat.st_nlink != 1
    or file_stat.st_uid != os.geteuid()
  ):
    raise RuntimeError(
      f"autonomous {field_name} has unsafe file identity"
    )
  return file_stat


def unlink_created_owned_file(
  path: Path,
  *,
  device: int,
  inode: int,
) -> None:
  try:
    file_stat = os.lstat(path)
  except FileNotFoundError:
    return
  if (
    stat.S_ISREG(file_stat.st_mode)
    and file_stat.st_dev == device
    and file_stat.st_ino == inode
  ):
    os.unlink(path)


def _open_verified(
  path: Path,
  *,
  flags: int,
  expected_device: int,
  expected_inode: int,
) -> tuple[int, os.stat_result]:
  fd = os.open(
    path,
    flags
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0),
  )
  try:
    file_stat = os.fstat(fd)
    if (
      not stat.S_ISREG(file_stat.st_mode)
      or stat.S_IMODE(file_stat.st_mode) != 0o600
      or file_stat.st_nlink != 1
      or file_stat.st_uid != os.geteuid()
      or file_stat.st_dev != expected_device
      or file_stat.st_ino != expected_inode
    ):
      raise RuntimeError(
        f"signed autonomous control file identity changed: {path}"
      )
    return fd, file_stat
  except BaseException:
    os.close(fd)
    raise


def append_closed_json_record(
  path: Path,
  *,
  expected_device: int,
  expected_inode: int,
  payload: dict[str, Any],
) -> None:
  encoded = encode_closed_control_record(payload)
  fd, file_stat = _open_verified(
    path,
    flags=os.O_RDWR | os.O_APPEND,
    expected_device=expected_device,
    expected_inode=expected_inode,
  )
  original_size = file_stat.st_size
  try:
    try:
      written = os.write(fd, encoded)
      if written != len(encoded):
        raise OSError(
          "autonomous control record append was incomplete"
        )
      os.fsync(fd)
    except BaseException as append_error:
      stream_recovered = False
      try:
        os.ftruncate(fd, original_size)
        os.fsync(fd)
        stream_recovered = (
          os.fstat(fd).st_size == original_size
        )
      except BaseException:
        stream_recovered = False
      raise AutonomousControlAppendError(
        "autonomous control append failed after mutation began",
        stream_recovered=stream_recovered,
      ) from append_error
  finally:
    os.close(fd)


def adopt_open_json_record_file(path: Path) -> tuple[os.stat_result, bool]:
  """Adopt an open-schema JSONL file for identity-pinned appends."""
  flags = (
    os.O_RDWR
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
  )
  created = False
  try:
    fd = os.open(path, flags)
  except FileNotFoundError:
    fd, file_stat = secure_create_owned_file(path)
    created = True
  else:
    file_stat = os.fstat(fd)
  try:
    if (
      not stat.S_ISREG(file_stat.st_mode)
      or file_stat.st_nlink != 1
      or file_stat.st_uid != os.geteuid()
    ):
      raise RuntimeError(
        f"autonomous open record file has unsafe identity: {path}"
      )
    os.fchmod(fd, 0o600)
    file_stat = os.fstat(fd)
    if stat.S_IMODE(file_stat.st_mode) != 0o600:
      raise RuntimeError(
        f"autonomous open record file permissions are unsafe: {path}"
      )
    if file_stat.st_size:
      os.lseek(fd, -1, os.SEEK_END)
      if os.read(fd, 1) != b"\n":
        cursor = file_stat.st_size
        last_newline = -1
        while cursor > 0 and last_newline < 0:
          chunk_start = max(0, cursor - 64 * 1024)
          os.lseek(fd, chunk_start, os.SEEK_SET)
          chunk = os.read(fd, cursor - chunk_start)
          last_newline = chunk.rfind(b"\n")
          if last_newline >= 0:
            last_newline += chunk_start
            break
          cursor = chunk_start
        os.ftruncate(fd, last_newline + 1)
        os.fsync(fd)
        file_stat = os.fstat(fd)
    return file_stat, created
  finally:
    os.close(fd)


def append_open_json_record(
  path: Path,
  *,
  expected_device: int,
  expected_inode: int,
  payload: dict[str, Any],
) -> None:
  """Append one heterogeneous JSON object without imposing a schema or cap."""
  encoded = (
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  fd, file_stat = _open_verified(
    path,
    flags=os.O_RDWR | os.O_APPEND,
    expected_device=expected_device,
    expected_inode=expected_inode,
  )
  original_size = file_stat.st_size
  try:
    try:
      written = os.write(fd, encoded)
      if written != len(encoded):
        raise OSError("autonomous open record append was incomplete")
      os.fsync(fd)
    except BaseException as append_error:
      stream_recovered = False
      try:
        os.ftruncate(fd, original_size)
        os.fsync(fd)
        stream_recovered = os.fstat(fd).st_size == original_size
      except BaseException:
        stream_recovered = False
      raise AutonomousControlAppendError(
        "autonomous open record append failed after mutation began",
        stream_recovered=stream_recovered,
      ) from append_error
  finally:
    os.close(fd)


def require_appendable_owned_file(
  path: Path,
  *,
  expected_device: int,
  expected_inode: int,
) -> None:
  fd, _file_stat = _open_verified(
    path,
    flags=os.O_WRONLY | os.O_APPEND,
    expected_device=expected_device,
    expected_inode=expected_inode,
  )
  os.close(fd)


def iter_closed_json_records(
  path: Path,
  *,
  expected_device: int,
  expected_inode: int,
  kind: str,
  fields: frozenset[str],
) -> Iterator[dict[str, Any]]:
  fd, _file_stat = _open_verified(
    path,
    flags=os.O_RDONLY,
    expected_device=expected_device,
    expected_inode=expected_inode,
  )
  try:
    with os.fdopen(fd, "rb") as handle:
      fd = -1
      offset = 0
      while True:
        result = read_bounded_control_line(handle, offset=offset)
        if result is None:
          break
        raw_line, line_end, complete = result
        if not complete:
          raise RuntimeError(
            f"incomplete autonomous {kind} control record"
          )
        payload, _digest = decode_closed_control_record(
          raw_line,
          kind=kind,
          fields=fields,
        )
        yield payload
        offset = line_end
  finally:
    if fd >= 0:
      os.close(fd)


def find_closed_json_record(
  path: Path,
  *,
  expected_device: int,
  expected_inode: int,
  kind: str,
  fields: frozenset[str],
  predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
  for payload in iter_closed_json_records(
    path,
    expected_device=expected_device,
    expected_inode=expected_inode,
    kind=kind,
    fields=fields,
  ):
    if predicate(payload):
      return payload
  return None


__all__ = [
  "AUTONOMOUS_CONTROL_RECORD_MAX_BYTES",
  "AutonomousControlAppendError",
  "adopt_open_json_record_file",
  "append_closed_json_record",
  "append_open_json_record",
  "find_closed_json_record",
  "fsync_owned_file_directory",
  "iter_closed_json_records",
  "require_appendable_owned_file",
  "require_existing_owned_file_identity",
  "secure_create_owned_file",
  "unlink_created_owned_file",
]
