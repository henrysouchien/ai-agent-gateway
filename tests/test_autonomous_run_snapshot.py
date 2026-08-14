from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
import errno
import os
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.autonomous_run_snapshot as snapshot_module  # noqa: E402
from agent_gateway.autonomous_run_snapshot import (  # noqa: E402
  AutonomousRunSnapshotError,
  snapshot_autonomous_run_root,
)


def _private_directory(path: Path) -> Path:
  path.mkdir(parents=True)
  path.chmod(0o700)
  return path


def _private_file(path: Path, payload: bytes) -> Path:
  path.write_bytes(payload)
  path.chmod(0o600)
  return path


def _run_root(tmp_path: Path) -> Path:
  return _private_directory(tmp_path / "registry" / "run")


def test_snapshot_is_deterministic_sorted_and_deeply_immutable(
  tmp_path: Path,
) -> None:
  root = _run_root(tmp_path)
  nested = _private_directory(root / "nested")
  _private_file(root / "z.raw", b"zeta")
  _private_file(nested / "a.raw", b"alpha")

  first = snapshot_autonomous_run_root(
    root,
    ("z.raw", "nested/a.raw"),
  )
  second = snapshot_autonomous_run_root(
    root,
    ("nested/a.raw", "z.raw"),
  )

  assert first == second
  assert tuple(item.relative_path for item in first.files) == (
    "nested/a.raw",
    "z.raw",
  )
  assert tuple(item.raw_bytes for item in first.files) == (
    b"alpha",
    b"zeta",
  )
  assert first.total_bytes == 9
  assert len(first.sha256) == 64
  assert all(len(item.sha256) == 64 for item in first.files)
  with pytest.raises(FrozenInstanceError):
    setattr(first, "total_bytes", 0)
  with pytest.raises(FrozenInstanceError):
    setattr(first.files[0], "raw_bytes", b"changed")


@pytest.mark.parametrize(
  "approved",
  (
    ["event.raw"],
    (),
    ("/event.raw",),
    ("../event.raw",),
    ("nested/../event.raw",),
    ("nested//event.raw",),
    ("event.raw/",),
    ("event.raw", "event.raw"),
  ),
)
def test_snapshot_rejects_non_closed_or_noncanonical_paths(
  tmp_path: Path,
  approved: object,
) -> None:
  root = _run_root(tmp_path)
  _private_file(root / "event.raw", b"event")

  with pytest.raises(
    AutonomousRunSnapshotError,
    match="approved snapshot path",
  ):
    snapshot_autonomous_run_root(root, approved)


def test_snapshot_enforces_file_count_and_byte_bounds(
  tmp_path: Path,
) -> None:
  root = _run_root(tmp_path)
  _private_file(root / "one.raw", b"1234")
  _private_file(root / "two.raw", b"5678")

  with pytest.raises(
    AutonomousRunSnapshotError,
    match="file bound",
  ):
    snapshot_autonomous_run_root(
      root,
      ("one.raw", "two.raw"),
      max_files=1,
    )
  with pytest.raises(
    AutonomousRunSnapshotError,
    match="per-file",
  ):
    snapshot_autonomous_run_root(
      root,
      ("one.raw",),
      max_file_bytes=3,
    )
  with pytest.raises(
    AutonomousRunSnapshotError,
    match="aggregate",
  ):
    snapshot_autonomous_run_root(
      root,
      ("one.raw", "two.raw"),
      max_total_bytes=7,
    )
  with pytest.raises(
    AutonomousRunSnapshotError,
    match="must be between",
  ):
    snapshot_autonomous_run_root(
      root,
      ("one.raw",),
      max_files=True,
    )


@pytest.mark.parametrize("mode", (0o755, 0o750, 0o500))
def test_snapshot_rejects_non_private_run_root_mode(
  tmp_path: Path,
  mode: int,
) -> None:
  root = _run_root(tmp_path)
  _private_file(root / "event.raw", b"event")
  root.chmod(mode)

  with pytest.raises(
    AutonomousRunSnapshotError,
    match="mode 0700",
  ):
    snapshot_autonomous_run_root(root, ("event.raw",))


def test_snapshot_rejects_symlinked_run_root(
  tmp_path: Path,
) -> None:
  real_root = _run_root(tmp_path)
  _private_file(real_root / "event.raw", b"event")
  linked_root = tmp_path / "linked-run"
  linked_root.symlink_to(real_root, target_is_directory=True)

  with pytest.raises(AutonomousRunSnapshotError):
    snapshot_autonomous_run_root(linked_root, ("event.raw",))


def test_snapshot_rejects_ambiguous_or_foreign_owned_root(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = _run_root(tmp_path)
  _private_file(root / "event.raw", b"event")

  with pytest.raises(AutonomousRunSnapshotError):
    snapshot_autonomous_run_root(
      f"//{str(root).lstrip('/')}",
      ("event.raw",),
    )

  actual_euid = os.geteuid()
  monkeypatch.setattr(
    snapshot_module.os,
    "geteuid",
    lambda: actual_euid + 1,
  )
  with pytest.raises(
    AutonomousRunSnapshotError,
    match="unexpected owner",
  ):
    snapshot_autonomous_run_root(root, ("event.raw",))


def test_snapshot_accepts_owner_read_only_file(
  tmp_path: Path,
) -> None:
  root = _run_root(tmp_path)
  target = _private_file(root / "event.raw", b"event")
  target.chmod(0o400)

  snapshot = snapshot_autonomous_run_root(
    root,
    ("event.raw",),
  )

  assert snapshot.files[0].raw_bytes == b"event"


@pytest.mark.parametrize(
  "attack",
  ("symlink", "hardlink", "fifo", "mode"),
)
def test_snapshot_rejects_untrusted_file_objects(
  tmp_path: Path,
  attack: str,
) -> None:
  root = _run_root(tmp_path)
  target = root / "event.raw"
  decoy = _private_file(root / "decoy.raw", b"decoy")
  if attack == "symlink":
    target.symlink_to(decoy)
  elif attack == "hardlink":
    os.link(decoy, target)
  elif attack == "fifo":
    os.mkfifo(target, 0o600)
  else:
    _private_file(target, b"event")
    target.chmod(0o644)

  with pytest.raises(AutonomousRunSnapshotError):
    snapshot_autonomous_run_root(root, ("event.raw",))


def test_snapshot_rejects_symlinked_nested_directory(
  tmp_path: Path,
) -> None:
  root = _run_root(tmp_path)
  outside = _private_directory(tmp_path / "outside")
  _private_file(outside / "event.raw", b"event")
  (root / "nested").symlink_to(outside, target_is_directory=True)

  with pytest.raises(AutonomousRunSnapshotError):
    snapshot_autonomous_run_root(
      root,
      ("nested/event.raw",),
    )


def test_snapshot_rejects_regular_file_swapped_to_fifo_without_blocking(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = _run_root(tmp_path)
  target = _private_file(root / "event.raw", b"event")
  real_open = snapshot_module.os.open
  swapped = False

  def _swap_before_file_open(
    path: str,
    flags: int,
    *args: Any,
    **kwargs: Any,
  ) -> int:
    nonlocal swapped
    if path == "event.raw" and not swapped:
      swapped = True
      target.unlink()
      os.mkfifo(target, 0o600)
      assert flags & os.O_NONBLOCK
    return real_open(path, flags, *args, **kwargs)

  monkeypatch.setattr(
    snapshot_module.os,
    "open",
    _swap_before_file_open,
  )

  with pytest.raises(
    AutonomousRunSnapshotError,
    match="regular file",
  ):
    snapshot_autonomous_run_root(root, ("event.raw",))

  assert swapped is True


@pytest.mark.parametrize("attack", ("grow", "truncate", "mode"))
def test_snapshot_rejects_file_drift_during_capture(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  attack: str,
) -> None:
  root = _run_root(tmp_path)
  target = _private_file(root / "event.raw", b"abcdefgh")
  real_read = snapshot_module._read_exact_file
  attacked = False

  def _attacking_read(descriptor: int, expected_size: int) -> bytes:
    nonlocal attacked
    raw = real_read(descriptor, expected_size)
    if not attacked:
      attacked = True
      if attack == "grow":
        with target.open("ab") as handle:
          handle.write(b"x")
      elif attack == "truncate":
        os.truncate(target, 3)
      else:
        target.chmod(0o400)
    return raw

  monkeypatch.setattr(
    snapshot_module,
    "_read_exact_file",
    _attacking_read,
  )

  with pytest.raises(
    AutonomousRunSnapshotError,
    match="changed",
  ):
    snapshot_autonomous_run_root(root, ("event.raw",))


def test_snapshot_rejects_file_replacement_during_capture(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = _run_root(tmp_path)
  target = _private_file(root / "event.raw", b"original")
  replacement = _private_file(root / "replacement.raw", b"replaced")
  real_read = snapshot_module._read_exact_file
  attacked = False

  def _replacing_read(descriptor: int, expected_size: int) -> bytes:
    nonlocal attacked
    raw = real_read(descriptor, expected_size)
    if not attacked:
      attacked = True
      os.replace(replacement, target)
    return raw

  monkeypatch.setattr(
    snapshot_module,
    "_read_exact_file",
    _replacing_read,
  )

  with pytest.raises(
    AutonomousRunSnapshotError,
    match="changed",
  ):
    snapshot_autonomous_run_root(root, ("event.raw",))


def test_snapshot_rejects_file_replacement_during_final_reopen(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = _run_root(tmp_path)
  target = _private_file(root / "event.raw", b"original")
  replacement = _private_file(root / "replacement.raw", b"replaced")
  real_read = snapshot_module._read_exact_file
  read_count = 0

  def _replacing_final_read(
    descriptor: int,
    expected_size: int,
  ) -> bytes:
    nonlocal read_count
    raw = real_read(descriptor, expected_size)
    read_count += 1
    if read_count == 3:
      os.replace(replacement, target)
    return raw

  monkeypatch.setattr(
    snapshot_module,
    "_read_exact_file",
    _replacing_final_read,
  )

  with pytest.raises(
    AutonomousRunSnapshotError,
    match="changed",
  ):
    snapshot_autonomous_run_root(root, ("event.raw",))


@pytest.mark.parametrize("attack", ("root", "ancestor"))
def test_snapshot_rejects_root_or_ancestor_replacement(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  attack: str,
) -> None:
  registry = _private_directory(tmp_path / "registry")
  root = _private_directory(registry / "run")
  _private_file(root / "event.raw", b"event")
  real_read = snapshot_module._read_exact_file
  attacked = False

  def _replacing_read(descriptor: int, expected_size: int) -> bytes:
    nonlocal attacked
    raw = real_read(descriptor, expected_size)
    if attacked:
      return raw
    attacked = True
    if attack == "root":
      root.rename(registry / "displaced-run")
      replacement_root = _private_directory(registry / "run")
      _private_file(replacement_root / "event.raw", b"event")
    else:
      registry.rename(tmp_path / "displaced-registry")
      replacement_registry = _private_directory(
        tmp_path / "registry",
      )
      replacement_root = _private_directory(
        replacement_registry / "run",
      )
      _private_file(replacement_root / "event.raw", b"event")
    return raw

  monkeypatch.setattr(
    snapshot_module,
    "_read_exact_file",
    _replacing_read,
  )

  with pytest.raises(
    AutonomousRunSnapshotError,
    match="changed",
  ):
    snapshot_autonomous_run_root(root, ("event.raw",))


def test_snapshot_rejects_nested_directory_mode_drift(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = _run_root(tmp_path)
  nested = _private_directory(root / "nested")
  _private_file(nested / "event.raw", b"event")
  real_read = snapshot_module._read_exact_file
  attacked = False

  def _changing_mode(descriptor: int, expected_size: int) -> bytes:
    nonlocal attacked
    raw = real_read(descriptor, expected_size)
    if not attacked:
      attacked = True
      nested.chmod(0o750)
    return raw

  monkeypatch.setattr(
    snapshot_module,
    "_read_exact_file",
    _changing_mode,
  )

  with pytest.raises(AutonomousRunSnapshotError):
    snapshot_autonomous_run_root(
      root,
      ("nested/event.raw",),
    )


@pytest.mark.parametrize("required_flag", ("O_NOFOLLOW", "O_NONBLOCK"))
def test_snapshot_requires_platform_file_open_support(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  required_flag: str,
) -> None:
  root = _run_root(tmp_path)
  _private_file(root / "event.raw", b"event")
  monkeypatch.delattr(snapshot_module.os, required_flag)

  with pytest.raises(
    AutonomousRunSnapshotError,
    match=required_flag,
  ):
    snapshot_autonomous_run_root(root, ("event.raw",))


def test_snapshot_closes_every_owned_descriptor_on_failure(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = _run_root(tmp_path)
  _private_file(root / "event.raw", b"event")
  real_open = snapshot_module.os.open
  real_close = snapshot_module.os.close
  opened: list[int] = []
  closed: list[int] = []

  def _tracking_open(*args: Any, **kwargs: Any) -> int:
    descriptor = real_open(*args, **kwargs)
    opened.append(descriptor)
    return descriptor

  def _tracking_close(descriptor: int) -> None:
    closed.append(descriptor)
    real_close(descriptor)

  monkeypatch.setattr(snapshot_module.os, "open", _tracking_open)
  monkeypatch.setattr(snapshot_module.os, "close", _tracking_close)

  with pytest.raises(AutonomousRunSnapshotError):
    snapshot_autonomous_run_root(
      root,
      ("event.raw", "missing.raw"),
    )

  assert Counter(opened) == Counter(closed)


def test_snapshot_cleanup_continues_after_eintr_close(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = _run_root(tmp_path)
  real_open = snapshot_module.os.open
  real_close = snapshot_module.os.close
  opened: list[int] = []
  attempted: list[int] = []
  live: set[int] = set()
  injected = False

  def _tracking_open(*args: Any, **kwargs: Any) -> int:
    descriptor = real_open(*args, **kwargs)
    opened.append(descriptor)
    live.add(descriptor)
    return descriptor

  def _eintr_after_real_close(descriptor: int) -> None:
    nonlocal injected
    attempted.append(descriptor)
    real_close(descriptor)
    live.remove(descriptor)
    if not injected:
      injected = True
      raise InterruptedError(
        errno.EINTR,
        "injected EINTR after real close",
      )

  monkeypatch.setattr(snapshot_module.os, "open", _tracking_open)
  monkeypatch.setattr(
    snapshot_module.os,
    "close",
    _eintr_after_real_close,
  )

  with pytest.raises(
    AutonomousRunSnapshotError,
    match="cleanup failed closed",
  ):
    snapshot_autonomous_run_root(root, ("missing.raw",))

  assert injected is True
  assert Counter(attempted) == Counter(opened)
  assert all(count == 1 for count in Counter(attempted).values())
  assert live == set()
