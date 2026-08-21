from __future__ import annotations

import os
from pathlib import Path
import stat


DirectoryIdentity = tuple[tuple[int, int], ...]


class DirectoryChainSecurityError(RuntimeError):
  """A descriptor-relative directory walk left its trusted namespace."""


def absolute_lexical_path(raw_path: str | Path) -> Path:
  """Return an absolute path without following any filesystem component."""

  return Path(os.path.abspath(os.path.expanduser(os.fspath(raw_path))))


def open_directory_chain(
  raw_path: str | Path,
  *,
  create: bool = False,
) -> tuple[int, DirectoryIdentity]:
  """Open every absolute directory component relative to its verified parent."""

  if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
    raise DirectoryChainSecurityError(
      "descriptor-confined directory access is unavailable"
    )
  path = absolute_lexical_path(raw_path)
  if not path.is_absolute():  # pragma: no cover - abspath guarantees this
    raise DirectoryChainSecurityError("directory path is not absolute")
  flags = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
    | getattr(os, "O_CLOEXEC", 0)
  )
  current_descriptor = -1
  try:
    current_descriptor = os.open(os.sep, flags)
    identities: list[tuple[int, int]] = []
    root_info = os.fstat(current_descriptor)
    if not stat.S_ISDIR(root_info.st_mode):  # pragma: no cover - POSIX root
      raise DirectoryChainSecurityError("filesystem root is unsafe")
    identities.append((root_info.st_dev, root_info.st_ino))
    for component in path.parts[1:]:
      try:
        named = os.stat(
          component,
          dir_fd=current_descriptor,
          follow_symlinks=False,
        )
      except FileNotFoundError:
        if not create:
          raise DirectoryChainSecurityError(
            "directory is missing"
          ) from None
        try:
          os.mkdir(component, mode=0o700, dir_fd=current_descriptor)
        except FileExistsError:
          pass
        named = os.stat(
          component,
          dir_fd=current_descriptor,
          follow_symlinks=False,
        )
      if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
        raise DirectoryChainSecurityError("directory chain is unsafe")
      next_descriptor = -1
      try:
        next_descriptor = os.open(
          component,
          flags,
          dir_fd=current_descriptor,
        )
        opened = os.fstat(next_descriptor)
        if (
          not stat.S_ISDIR(opened.st_mode)
          or (opened.st_dev, opened.st_ino)
          != (named.st_dev, named.st_ino)
        ):
          raise DirectoryChainSecurityError(
            "directory identity changed"
          )
      except Exception:
        if next_descriptor >= 0:
          os.close(next_descriptor)
        raise
      os.close(current_descriptor)
      current_descriptor = next_descriptor
      identities.append((opened.st_dev, opened.st_ino))
    result = current_descriptor
    current_descriptor = -1
    return result, tuple(identities)
  except DirectoryChainSecurityError:
    raise
  except OSError as exc:
    raise DirectoryChainSecurityError(
      "directory chain is unavailable"
    ) from exc
  finally:
    if current_descriptor >= 0:
      os.close(current_descriptor)
