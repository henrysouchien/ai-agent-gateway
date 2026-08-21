from __future__ import annotations

# ruff: noqa: E402

import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import descriptor_paths


def test_open_directory_chain_returns_readable_exact_directory(
  tmp_path: Path,
) -> None:
  target = tmp_path / "one" / "two"
  target.mkdir(parents=True)
  (target / "entry").write_text("ok", encoding="utf-8")

  descriptor, identity = descriptor_paths.open_directory_chain(
    target
  )
  try:
    assert os.listdir(descriptor) == ["entry"]
    opened = os.fstat(descriptor)
    assert identity[-1] == (opened.st_dev, opened.st_ino)
    assert len(identity) == len(target.resolve().parts)
  finally:
    os.close(descriptor)


def test_linux_path_walk_uses_lookup_only_ancestors_and_readable_target(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  target = tmp_path / "one" / "two"
  target.mkdir(parents=True)
  fake_o_path = 1 << 29
  real_open = os.open
  observed: list[tuple[str, int]] = []

  def tracked_open(
    path: str | bytes | int,
    flags: int,
    mode: int = 0o777,
    *,
    dir_fd: int | None = None,
  ) -> int:
    observed.append((os.fsdecode(path), flags))
    return real_open(
      path,
      flags & ~fake_o_path,
      mode,
      dir_fd=dir_fd,
    )

  monkeypatch.setattr(
    descriptor_paths.os,
    "O_PATH",
    fake_o_path,
    raising=False,
  )
  monkeypatch.setattr(descriptor_paths.os, "open", tracked_open)

  descriptor, _identity = descriptor_paths.open_directory_chain(
    target
  )
  os.close(descriptor)

  readable_flags = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
    | getattr(os, "O_CLOEXEC", 0)
  )
  ancestor_flags = readable_flags | fake_o_path
  assert len(observed) == len(target.resolve().parts)
  assert all(
    flags == ancestor_flags for _path, flags in observed[:-1]
  )
  assert observed[-1][1] == readable_flags


def test_platform_fallback_uses_readable_descriptors_for_entire_chain(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  target = tmp_path / "one" / "two"
  target.mkdir(parents=True)
  real_open = os.open
  observed: list[int] = []

  def tracked_open(
    path: str | bytes | int,
    flags: int,
    mode: int = 0o777,
    *,
    dir_fd: int | None = None,
  ) -> int:
    observed.append(flags)
    return real_open(path, flags, mode, dir_fd=dir_fd)

  monkeypatch.setattr(
    descriptor_paths.os,
    "O_PATH",
    None,
    raising=False,
  )
  monkeypatch.setattr(descriptor_paths.os, "open", tracked_open)

  descriptor, _identity = descriptor_paths.open_directory_chain(
    target
  )
  os.close(descriptor)

  expected_flags = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
    | getattr(os, "O_CLOEXEC", 0)
  )
  assert observed and all(flags == expected_flags for flags in observed)


def test_create_preserves_final_directory_identity(
  tmp_path: Path,
) -> None:
  target = tmp_path / "created" / "nested"

  descriptor, identity = descriptor_paths.open_directory_chain(
    target,
    create=True,
  )
  try:
    assert target.is_dir()
    opened = os.fstat(descriptor)
    assert identity[-1] == (opened.st_dev, opened.st_ino)
  finally:
    os.close(descriptor)


def test_symlinked_ancestor_remains_rejected(tmp_path: Path) -> None:
  outside = tmp_path / "outside"
  outside.mkdir()
  linked = tmp_path / "linked"
  linked.symlink_to(outside, target_is_directory=True)

  with pytest.raises(
    descriptor_paths.DirectoryChainSecurityError,
    match="directory chain is unsafe",
  ):
    descriptor_paths.open_directory_chain(linked)
