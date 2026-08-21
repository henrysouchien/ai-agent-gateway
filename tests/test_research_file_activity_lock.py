from __future__ import annotations

import multiprocessing
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.research_file_activity_lock import (  # noqa: E402
  try_acquire_research_file_activity,
  try_acquire_research_file_activity_set,
)
from agent_gateway import research_file_activity_lock as lock_module  # noqa: E402


def _hold_shared_lease(
  workspace: str,
  ready: Any,
  release: Any,
) -> None:
  lease = try_acquire_research_file_activity(
    Path(workspace),
    research_file_id=41,
    exclusive=False,
  )
  ready.put(lease is not None)
  if lease is None:
    return
  try:
    release.get(timeout=10)
  finally:
    lease.release()


def _open_descriptor_count() -> int:
  return len(tuple(Path("/dev/fd").iterdir()))


def test_same_process_shared_leases_exclude_exact_file_erase(
  tmp_path: Path,
) -> None:
  first = try_acquire_research_file_activity(
    tmp_path,
    research_file_id=41,
    exclusive=False,
  )
  second = try_acquire_research_file_activity(
    tmp_path,
    research_file_id=41,
    exclusive=False,
  )
  assert first is not None
  assert second is not None
  try:
    assert try_acquire_research_file_activity(
      tmp_path,
      research_file_id=41,
      exclusive=True,
    ) is None
    unrelated = try_acquire_research_file_activity(
      tmp_path,
      research_file_id=42,
      exclusive=True,
    )
    assert unrelated is not None
    unrelated.release()
  finally:
    second.release()
    first.release()

  exclusive = try_acquire_research_file_activity(
    tmp_path,
    research_file_id=41,
    exclusive=True,
  )
  assert exclusive is not None
  try:
    assert try_acquire_research_file_activity(
      tmp_path,
      research_file_id=41,
      exclusive=False,
    ) is None
  finally:
    exclusive.release()


def test_multi_file_acquisition_is_all_or_none(tmp_path: Path) -> None:
  held = try_acquire_research_file_activity(
    tmp_path,
    research_file_id=42,
    exclusive=False,
  )
  assert held is not None
  try:
    assert try_acquire_research_file_activity_set(
      tmp_path,
      research_file_ids=(41, 42),
      exclusive=True,
    ) is None
    # The failed set released the earlier sorted file lease.
    available = try_acquire_research_file_activity(
      tmp_path,
      research_file_id=41,
      exclusive=True,
    )
    assert available is not None
    available.release()
  finally:
    held.release()


def test_lock_path_rejects_symlinked_directory(tmp_path: Path) -> None:
  outside = tmp_path / "outside"
  outside.mkdir()
  locks = tmp_path / "workspace" / ".locks"
  locks.parent.mkdir()
  locks.symlink_to(outside, target_is_directory=True)

  with pytest.raises(ValueError, match="unsafe"):
    try_acquire_research_file_activity(
      locks.parent,
      research_file_id=41,
      exclusive=False,
    )

  assert list(outside.iterdir()) == []


def test_lock_path_rejects_symlinked_leaf(tmp_path: Path) -> None:
  workspace = tmp_path / "workspace"
  directory = workspace / ".locks" / "research_file_activity"
  directory.mkdir(parents=True, mode=0o700)
  outside = tmp_path / "outside.lock"
  outside.write_text("outside", encoding="utf-8")
  (directory / "41.lock").symlink_to(outside)

  with pytest.raises(ValueError, match="unsafe"):
    try_acquire_research_file_activity(
      workspace,
      research_file_id=41,
      exclusive=False,
    )

  assert outside.read_text(encoding="utf-8") == "outside"


def test_locks_ancestor_swap_cannot_redirect_namespace(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = tmp_path / "workspace"
  outside = tmp_path / "outside-locks"
  outside.mkdir()
  real_open = lock_module.os.open
  swapped = False

  def swap_before_open(path: Any, *args: Any, **kwargs: Any) -> int:
    nonlocal swapped
    if path == ".locks" and kwargs.get("dir_fd") is not None and not swapped:
      swapped = True
      locks_path = workspace / ".locks"
      locks_path.rename(workspace / ".locks-before-swap")
      locks_path.symlink_to(outside, target_is_directory=True)
    return real_open(path, *args, **kwargs)

  monkeypatch.setattr(lock_module.os, "open", swap_before_open)

  with pytest.raises(ValueError, match="unsafe"):
    try_acquire_research_file_activity(
      workspace,
      research_file_id=41,
      exclusive=False,
    )

  assert swapped is True
  assert list(outside.iterdir()) == []
  assert not (outside / "research_file_activity" / "41.lock").exists()


def test_cross_process_shared_and_exclusive_compatibility(
  tmp_path: Path,
) -> None:
  context = multiprocessing.get_context("spawn")
  ready = context.Queue()
  release = context.Queue()
  process = context.Process(
    target=_hold_shared_lease,
    args=(str(tmp_path), ready, release),
  )
  process.start()
  try:
    assert ready.get(timeout=10) is True
    assert try_acquire_research_file_activity(
      tmp_path,
      research_file_id=41,
      exclusive=True,
    ) is None
    unrelated = try_acquire_research_file_activity(
      tmp_path,
      research_file_id=42,
      exclusive=True,
    )
    assert unrelated is not None
    unrelated.release()
    release.put(True)
    process.join(timeout=10)
    assert process.exitcode == 0

    exclusive = try_acquire_research_file_activity(
      tmp_path,
      research_file_id=41,
      exclusive=True,
    )
    assert exclusive is not None
    exclusive.release()
  finally:
    if process.is_alive():
      release.put(True)
      process.join(timeout=2)
    if process.is_alive():
      process.terminate()
      process.join(timeout=5)


def test_post_create_identity_failure_does_not_leak_descriptor(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  descriptor_count = _open_descriptor_count()

  def fail_identity(*_args: Any, **_kwargs: Any) -> None:
    raise ValueError("injected identity failure")

  monkeypatch.setattr(
    lock_module,
    "_require_lock_entry_identity",
    fail_identity,
  )
  for research_file_id in range(1, 21):
    with pytest.raises(ValueError, match="injected identity failure"):
      try_acquire_research_file_activity(
        tmp_path,
        research_file_id=research_file_id,
        exclusive=False,
      )

  assert _open_descriptor_count() == descriptor_count
