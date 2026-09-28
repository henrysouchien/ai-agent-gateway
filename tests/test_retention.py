from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.autonomous_run_lock import AutonomousRunMutationLock
from agent_gateway.retention import (
  FileAgeAdapter,
  KeepForeverAdapter,
  RetentionCatalog,
  RetentionCatalogEntry,
  RetentionLockBusy,
  RetentionPolicy,
  RetentionSafetyError,
  RetentionSweeper,
  resolve_contained_path,
  resolve_safe_root,
  sweep_lock,
)
from agent_gateway.server import _cleanup_old_transcripts
from agent_gateway.server_chat_helpers import _cleanup_sessions_loop
from agent_gateway.server_models import GatewayServerConfig


def _touch(path: Path, *, mtime: float) -> None:
  path.write_text("{}", encoding="utf-8")
  os.utime(path, (mtime, mtime))


def test_transcript_retention_sweeps_old_jsonl_and_meta_files(tmp_path: Path) -> None:
  now = 1_000_000.0
  old_jsonl = tmp_path / "old-session.jsonl"
  old_meta = tmp_path / "old-session.meta.json"
  recent_jsonl = tmp_path / "recent-session.jsonl"
  ignored_file = tmp_path / "notes.txt"
  _touch(old_jsonl, mtime=now - (8 * 86400))
  _touch(old_meta, mtime=now - (8 * 86400))
  _touch(recent_jsonl, mtime=now - (2 * 86400))
  _touch(ignored_file, mtime=now - (30 * 86400))

  removed = _cleanup_old_transcripts(tmp_path, 7, now=now)

  assert removed == 2
  assert not old_jsonl.exists()
  assert not old_meta.exists()
  assert recent_jsonl.exists()
  assert ignored_file.exists()


def test_transcript_retention_uses_configurable_horizon(tmp_path: Path) -> None:
  now = 1_000_000.0
  two_days_old = tmp_path / "two-days-old.jsonl"
  half_day_old = tmp_path / "half-day-old.jsonl"
  _touch(two_days_old, mtime=now - (2 * 86400))
  _touch(half_day_old, mtime=now - (12 * 3600))

  removed = _cleanup_old_transcripts(tmp_path, 1, now=now)

  assert removed == 1
  assert not two_days_old.exists()
  assert half_day_old.exists()


def test_transcript_retention_keeps_stale_meta_for_fresh_transcript(tmp_path: Path) -> None:
  now = 1_000_000.0
  transcript = tmp_path / "active-session.jsonl"
  stale_meta = tmp_path / "active-session.meta.json"
  _touch(transcript, mtime=now - (2 * 86400))
  _touch(stale_meta, mtime=now - (8 * 86400))

  removed = _cleanup_old_transcripts(tmp_path, 7, now=now)

  assert removed == 0
  assert transcript.exists()
  assert stale_meta.exists()


def test_transcript_retention_noops_without_directory_or_positive_horizon(tmp_path: Path) -> None:
  transcript = tmp_path / "session.jsonl"
  _touch(transcript, mtime=1.0)

  assert _cleanup_old_transcripts(None, 7, now=1_000_000.0) == 0
  assert _cleanup_old_transcripts(tmp_path / "missing", 7, now=1_000_000.0) == 0
  assert _cleanup_old_transcripts(tmp_path, 0, now=1_000_000.0) == 0
  assert transcript.exists()


def test_configured_transcript_dir_sweeps_nothing_at_the_library_default(tmp_path: Path) -> None:
  now = 1_000_000.0
  transcript = tmp_path / "old-session.jsonl"
  meta = tmp_path / "old-session.meta.json"
  _touch(transcript, mtime=now - (8 * 86400))
  _touch(meta, mtime=now - (8 * 86400))

  config = GatewayServerConfig(transcript_dir=tmp_path)

  assert config.transcript_retention_days == 0
  assert (
    _cleanup_old_transcripts(
      config.transcript_dir, config.transcript_retention_days, now=now
    )
    == 0
  )
  assert transcript.exists()
  assert meta.exists()

  swept = GatewayServerConfig(transcript_dir=tmp_path, transcript_retention_days=7)

  assert (
    _cleanup_old_transcripts(
      swept.transcript_dir, swept.transcript_retention_days, now=now
    )
    == 2
  )
  assert not transcript.exists()
  assert not meta.exists()


def test_cleanup_sessions_loop_carries_the_same_off_default() -> None:
  parameter = inspect.signature(_cleanup_sessions_loop).parameters[
    "transcript_retention_days"
  ]

  assert parameter.default == GatewayServerConfig().transcript_retention_days == 0


class _FailingAdapter:
  def sweep(self, context):
    raise RuntimeError("isolated failure")


def test_policy_inversion_and_explicit_keep_forever(caplog: pytest.LogCaptureFixture) -> None:
  with pytest.raises(TypeError):
    RetentionPolicy()  # type: ignore[call-arg]
  with caplog.at_level(logging.INFO):
    policy = RetentionPolicy.keep_forever("legal ruling pending", "platform")
    RetentionCatalogEntry("audit", policy, KeepForeverAdapter("audit"), (Path("/tmp/audit"),))
  assert policy.is_keep_forever
  assert "keep_forever registered" in caplog.text
  assert "legal ruling pending" in caplog.text


def test_registration_accepts_keep_forever_for_an_error_triage_input() -> None:
  """No floor on deletion can reject a policy that deletes nothing."""

  forever = RetentionPolicy.keep_forever("a scheduled lane's own run log", "henry")
  entry = RetentionCatalogEntry(
    "session", forever, KeepForeverAdapter("session"), (Path("/tmp/session"),)
  )
  assert entry.policy.is_keep_forever
  short = RetentionPolicy("age", "platform", "scratch", max_age_days=2)
  RetentionCatalogEntry("unrelated", short, KeepForeverAdapter("unrelated"), (Path("/tmp/unrelated"),))


def test_safety_refuses_symlink_escape_and_broad_roots(tmp_path: Path) -> None:
  root = tmp_path / "safe"
  root.mkdir()
  outside = tmp_path / "outside"
  outside.write_text("x")
  link = root / "link"
  link.symlink_to(outside)
  with pytest.raises(RetentionSafetyError):
    resolve_contained_path(link, root)
  with pytest.raises(RetentionSafetyError):
    resolve_contained_path(outside, root)
  with pytest.raises(RetentionSafetyError):
    resolve_safe_root(Path("/"))
  with pytest.raises(RetentionSafetyError):
    resolve_safe_root(Path.home())
  with pytest.raises(RetentionSafetyError):
    resolve_safe_root(tmp_path, repo_root=tmp_path)


def test_driver_isolates_failure_and_file_age_is_idempotent(tmp_path: Path) -> None:
  state = tmp_path / "state"
  root = tmp_path / "files"
  state.mkdir()
  root.mkdir()
  old = root / "old.log"
  old.write_bytes(b"old")
  old_ts = time.time() - 40 * 86_400
  os.utime(old, (old_ts, old_ts))
  policy = RetentionPolicy("age", "platform", "test", max_age_days=30)
  catalog = RetentionCatalog(
    (
      RetentionCatalogEntry("broken", policy, _FailingAdapter(), (root,)),
      RetentionCatalogEntry("files", policy, FileAgeAdapter("files", root), (root,)),
    ),
    state,
  )
  reports = RetentionSweeper(catalog).sweep("dry_run", now=datetime.now(timezone.utc))
  assert reports[0].errors == ("RuntimeError: isolated failure",)
  assert reports[1].would_delete_count == 1
  assert old.exists()
  enforced = RetentionSweeper(catalog).sweep("enforce", now=datetime.now(timezone.utc))
  assert enforced[1].deleted_count == 1
  assert not old.exists()
  repeated = RetentionSweeper(catalog).sweep("enforce", now=datetime.now(timezone.utc))
  assert repeated[1].would_delete_count == 0


def test_enforce_allowlist_uses_per_entry_effective_modes(tmp_path: Path) -> None:
  state = tmp_path / "state"
  root = tmp_path / "files"
  state.mkdir()
  root.mkdir()
  old = root / "old.log"
  old.write_text("old", encoding="utf-8")
  stale = time.time() - 40 * 86_400
  os.utime(old, (stale, stale))
  policy = RetentionPolicy("age", "platform", "test", max_age_days=30)
  catalog = RetentionCatalog(
    (
      RetentionCatalogEntry(
        "pending",
        policy,
        FileAgeAdapter("pending", root),
        (root,),
      ),
    ),
    state,
  )
  pending = RetentionSweeper(catalog).sweep(
    "enforce",
    enforce_allowlist=frozenset(),
  )[0]
  assert pending.mode == "dry_run"
  assert pending.deleted_count == 0
  assert pending.details == ("pending migration re-acceptance",)
  assert old.exists()
  authorized = RetentionSweeper(catalog).sweep(
    "enforce",
    enforce_allowlist=frozenset({"pending"}),
  )[0]
  assert authorized.mode == "enforce"
  assert authorized.deleted_count == 1
  assert not old.exists()

  old.write_text("old", encoding="utf-8")
  os.utime(old, (stale, stale))
  preserved = RetentionSweeper(catalog).sweep("enforce", enforce_allowlist=None)[0]
  assert preserved.mode == "enforce"
  assert preserved.deleted_count == 1


def _hold_lock(path: str, ready: multiprocessing.Queue) -> None:
  with sweep_lock(Path(path), blocking=True):
    ready.put(True)
    time.sleep(2)


def test_cross_process_sweep_lock_prevents_double_sweep(tmp_path: Path) -> None:
  ready: multiprocessing.Queue = multiprocessing.Queue()
  process = multiprocessing.Process(target=_hold_lock, args=(str(tmp_path), ready))
  process.start()
  assert ready.get(timeout=5) is True
  try:
    with pytest.raises(RetentionLockBusy):
      with sweep_lock(tmp_path):
        pass
  finally:
    process.terminate()
    process.join(timeout=5)


@pytest.mark.asyncio
async def test_run_mutation_lock_serializes_resume_publication_window(tmp_path: Path) -> None:
  first = AutonomousRunMutationLock(tmp_path)
  second = AutonomousRunMutationLock(tmp_path)
  second_entered = asyncio.Event()

  async def successor_and_backlink() -> None:
    async with second:
      second_entered.set()

  async with first:
    task = asyncio.create_task(successor_and_backlink())
    await asyncio.sleep(0.05)
    assert not second_entered.is_set()
  await task
  assert second_entered.is_set()
