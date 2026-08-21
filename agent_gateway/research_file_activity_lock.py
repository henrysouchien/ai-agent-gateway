from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
import stat
from typing import Iterable


class ResearchFileActivityLease:
  """One process-safe operational lease for a research-file writer boundary."""

  def __init__(self, descriptor: int) -> None:
    self._descriptor = descriptor

  def release(self) -> None:
    descriptor = self._descriptor
    if descriptor < 0:
      return
    self._descriptor = -1
    try:
      fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
      os.close(descriptor)

  def __enter__(self) -> ResearchFileActivityLease:
    return self

  def __exit__(self, *_args: object) -> None:
    self.release()


class ResearchFileActivityLeaseSet:
  """Sorted exact-file leases released in reverse acquisition order."""

  def __init__(self, leases: tuple[ResearchFileActivityLease, ...]) -> None:
    self._leases = leases

  def release(self) -> None:
    leases, self._leases = self._leases, ()
    for lease in reversed(leases):
      lease.release()

  def __enter__(self) -> ResearchFileActivityLeaseSet:
    return self

  def __exit__(self, *_args: object) -> None:
    self.release()


def try_acquire_research_file_activity(
  workspace_dir: Path,
  *,
  research_file_id: int,
  exclusive: bool,
) -> ResearchFileActivityLease | None:
  """Try one exact advisory lease without waiting or changing product state."""

  normalized_id = _research_file_id(research_file_id)
  _workspace, workspace_descriptor = _open_workspace(Path(workspace_dir))
  directory_descriptors: list[int] = []
  descriptor = -1
  try:
    owner_uid = os.fstat(workspace_descriptor).st_uid
    locks_descriptor = _open_or_create_owned_directory(
      workspace_descriptor,
      ".locks",
      owner_uid=owner_uid,
      private=False,
    )
    directory_descriptors.append(locks_descriptor)
    activity_descriptor = _open_or_create_owned_directory(
      locks_descriptor,
      "research_file_activity",
      owner_uid=owner_uid,
      private=True,
    )
    directory_descriptors.append(activity_descriptor)
    lock_name = f"{normalized_id}.lock"
    descriptor = _open_or_create_lock_file(
      activity_descriptor,
      lock_name,
      owner_uid=owner_uid,
    )
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
      fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
    except OSError as exc:
      if exc.errno not in {errno.EACCES, errno.EAGAIN}:
        raise
      os.close(descriptor)
      descriptor = -1
      return None
    _require_lock_entry_identity(
      activity_descriptor,
      lock_name,
      descriptor,
      owner_uid=owner_uid,
    )
    lease = ResearchFileActivityLease(descriptor)
    descriptor = -1
    return lease
  finally:
    if descriptor >= 0:
      os.close(descriptor)
    for directory_descriptor in reversed(directory_descriptors):
      os.close(directory_descriptor)
    os.close(workspace_descriptor)


def try_acquire_research_file_activity_set(
  workspace_dir: Path,
  *,
  research_file_ids: Iterable[int],
  exclusive: bool,
) -> ResearchFileActivityLeaseSet | None:
  normalized_ids = tuple(sorted({_research_file_id(value) for value in research_file_ids}))
  leases: list[ResearchFileActivityLease] = []
  try:
    for research_file_id in normalized_ids:
      lease = try_acquire_research_file_activity(
        workspace_dir,
        research_file_id=research_file_id,
        exclusive=exclusive,
      )
      if lease is None:
        for acquired in reversed(leases):
          acquired.release()
        return None
      leases.append(lease)
    result = ResearchFileActivityLeaseSet(tuple(leases))
    leases = []
    return result
  finally:
    for lease in reversed(leases):
      lease.release()


def _open_workspace(raw_workspace: Path) -> tuple[Path, int]:
  expanded = raw_workspace.expanduser()
  if expanded.is_symlink():
    raise ValueError("research file activity workspace cannot be a symlink")
  workspace = expanded.resolve()
  workspace.mkdir(parents=True, exist_ok=True)
  path_info = workspace.lstat()
  if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISDIR(path_info.st_mode):
    raise ValueError("research file activity workspace is unsafe")
  flags = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
  )
  try:
    descriptor = os.open(workspace, flags)
  except OSError as exc:
    raise ValueError("research file activity workspace is unsafe") from exc
  descriptor_info = os.fstat(descriptor)
  if (
    not stat.S_ISDIR(descriptor_info.st_mode)
    or (descriptor_info.st_dev, descriptor_info.st_ino)
    != (path_info.st_dev, path_info.st_ino)
    or descriptor_info.st_uid != os.geteuid()
  ):
    os.close(descriptor)
    raise ValueError("research file activity workspace changed during open")
  return workspace, descriptor


def _open_or_create_owned_directory(
  parent_descriptor: int,
  name: str,
  *,
  owner_uid: int,
  private: bool,
) -> int:
  try:
    os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
  except FileExistsError:
    pass
  try:
    path_info = os.stat(
      name,
      dir_fd=parent_descriptor,
      follow_symlinks=False,
    )
  except OSError as exc:
    raise ValueError("research file activity lock directory is unsafe") from exc
  if not _owned_directory_info(
    path_info,
    owner_uid=owner_uid,
    private=private,
  ):
    raise ValueError("research file activity lock directory is unsafe")
  flags = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
  )
  try:
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
  except OSError as exc:
    raise ValueError("research file activity lock directory is unsafe") from exc
  descriptor_info = os.fstat(descriptor)
  if (
    not _owned_directory_info(
      descriptor_info,
      owner_uid=owner_uid,
      private=private,
    )
    or (descriptor_info.st_dev, descriptor_info.st_ino)
    != (path_info.st_dev, path_info.st_ino)
  ):
    os.close(descriptor)
    raise ValueError("research file activity lock directory changed during open")
  return descriptor


def _open_or_create_lock_file(
  directory_descriptor: int,
  name: str,
  *,
  owner_uid: int,
) -> int:
  flags = (
    os.O_RDWR
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
  )
  try:
    descriptor = os.open(
      name,
      flags | os.O_CREAT | os.O_EXCL,
      0o600,
      dir_fd=directory_descriptor,
    )
  except FileExistsError:
    try:
      path_info = os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
      )
    except OSError as exc:
      raise ValueError("research file activity lock is unsafe") from exc
    if not _owned_lock_info(path_info, owner_uid=owner_uid):
      raise ValueError("research file activity lock is unsafe")
    try:
      descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
      raise ValueError("research file activity lock is unsafe") from exc
    descriptor_info = os.fstat(descriptor)
    if (
      not _owned_lock_info(descriptor_info, owner_uid=owner_uid)
      or (descriptor_info.st_dev, descriptor_info.st_ino)
      != (path_info.st_dev, path_info.st_ino)
    ):
      os.close(descriptor)
      raise ValueError("research file activity lock changed during open")
    return descriptor
  descriptor_info = os.fstat(descriptor)
  if not _owned_lock_info(descriptor_info, owner_uid=owner_uid):
    os.close(descriptor)
    raise ValueError("research file activity lock is unsafe")
  try:
    _require_lock_entry_identity(
      directory_descriptor,
      name,
      descriptor,
      owner_uid=owner_uid,
    )
  except BaseException:
    os.close(descriptor)
    raise
  return descriptor


def _require_lock_entry_identity(
  directory_descriptor: int,
  name: str,
  descriptor: int,
  *,
  owner_uid: int,
) -> None:
  try:
    path_info = os.stat(
      name,
      dir_fd=directory_descriptor,
      follow_symlinks=False,
    )
  except OSError as exc:
    raise ValueError("research file activity lock changed during open") from exc
  descriptor_info = os.fstat(descriptor)
  if (
    not _owned_lock_info(path_info, owner_uid=owner_uid)
    or not _owned_lock_info(descriptor_info, owner_uid=owner_uid)
    or (path_info.st_dev, path_info.st_ino)
    != (descriptor_info.st_dev, descriptor_info.st_ino)
  ):
    raise ValueError("research file activity lock changed during open")


def _owned_directory_info(
  info: os.stat_result,
  *,
  owner_uid: int,
  private: bool,
) -> bool:
  mode = stat.S_IMODE(info.st_mode)
  return (
    stat.S_ISDIR(info.st_mode)
    and info.st_uid == owner_uid
    and mode & 0o700 == 0o700
    and mode & (0o077 if private else 0o022) == 0
  )


def _owned_lock_info(info: os.stat_result, *, owner_uid: int) -> bool:
  return (
    stat.S_ISREG(info.st_mode)
    and info.st_uid == owner_uid
    and info.st_nlink == 1
    and stat.S_IMODE(info.st_mode) == 0o600
  )


def _research_file_id(value: object) -> int:
  if (
    isinstance(value, bool)
    or not isinstance(value, int)
    or not 0 < value < (1 << 63)
  ):
    raise ValueError("research_file_id must be a positive signed 64-bit integer")
  return value


__all__ = [
  "ResearchFileActivityLease",
  "ResearchFileActivityLeaseSet",
  "try_acquire_research_file_activity",
  "try_acquire_research_file_activity_set",
]
