from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.server import _cleanup_old_transcripts


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
