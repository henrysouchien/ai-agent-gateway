from __future__ import annotations

import asyncio
import fcntl
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import threading
import time

import pytest

import agent_gateway.agent_session_log as session_log_module
from agent_gateway.agent_session_log import (
  AgentSessionLog,
  AgentSessionLogEnumerationError,
  AgentSessionLogStorageSecurityError,
  enumerate_agent_session_log_paths,
  try_acquire_agent_session_log_write_leases,
)


def _entry(*, seq: int, event_type: str = "workflow_run_started") -> bytes:
  return (
    json.dumps({
      "seq": seq,
      "timestamp": float(seq),
      "event": {"type": event_type, "event_schema_version": 2},
    })
    + "\n"
  ).encode("utf-8")


def _manifest(segment_path: str, *, seq: int = 1) -> dict[str, object]:
  segment_id = "000000000001-000000000001-g000000"
  return {
    "schema_version": 1,
    "segments": [{
      "segment_id": segment_id,
      "path": segment_path,
      "first_seq": seq,
      "last_seq": seq,
    }],
    "latest_seq": seq,
    "active_generation": 0,
    "min_seq_available": 1,
  }


def test_agent_session_log_has_a_clean_standalone_import() -> None:
  package_root = Path(__file__).resolve().parents[1]
  env = dict(os.environ)
  existing_pythonpath = env.get("PYTHONPATH")
  env["PYTHONPATH"] = os.pathsep.join(
    item
    for item in (
      os.fspath(package_root),
      existing_pythonpath,
    )
    if item
  )

  completed = subprocess.run(
    [
      sys.executable,
      "-c",
      (
        "from agent_gateway.agent_session_log import AgentSessionLog; "
        "assert AgentSessionLog.__name__ == 'AgentSessionLog'"
      ),
    ],
    check=False,
    capture_output=True,
    env=env,
    text=True,
  )

  assert completed.returncode == 0, completed.stderr


def test_enumerates_legacy_active_log_without_sidecar(tmp_path: Path) -> None:
  base = tmp_path / "sessions"
  agent_dir = base / "web_advisor"
  agent_dir.mkdir(parents=True)
  active = agent_dir / "agentsess_session_user.jsonl"
  active.write_bytes(_entry(seq=1))

  assert tuple(
    location.path for location in enumerate_agent_session_log_paths(base)
  ) == (active.resolve(),)


def test_constructor_rejects_group_writable_active_log_without_effect(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  active.parent.mkdir(parents=True)
  original = _entry(seq=1)
  active.write_bytes(original)
  active.chmod(0o622)

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="active file is unsafe",
  ):
    AgentSessionLog(active)

  assert active.read_bytes() == original
  assert stat.S_IMODE(active.stat().st_mode) == 0o622
  assert tuple(active.parent.iterdir()) == (active,)


def test_current_query_rejects_active_log_that_becomes_group_writable(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  log.append_sync({"type": "workflow_run_started"})
  original = active.read_bytes()
  active.chmod(0o622)

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="active file is unsafe",
  ):
    asyncio.run(log.query_current_strict())

  assert active.read_bytes() == original
  assert stat.S_IMODE(active.stat().st_mode) == 0o622


def test_enumeration_rejects_group_writable_active_log_without_effect(
  tmp_path: Path,
) -> None:
  base = tmp_path / "sessions"
  active = base / "web_advisor" / "session.jsonl"
  active.parent.mkdir(parents=True)
  original = _entry(seq=1)
  active.write_bytes(original)
  active.chmod(0o622)

  with pytest.raises(
    AgentSessionLogEnumerationError,
    match="file is unsafe",
  ):
    enumerate_agent_session_log_paths(base)

  assert active.read_bytes() == original
  assert stat.S_IMODE(active.stat().st_mode) == 0o622
  assert tuple(active.parent.iterdir()) == (active,)


def test_writer_lease_mode_policy_is_identical_across_call_sites(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  lease_path = active.with_name(f"{active.name}.write_lease")
  lease_path.write_bytes(b"unchanged")
  lease_path.chmod(0o622)

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="writer lease is unavailable",
  ):
    log._open_writer_lease_file()
  with pytest.raises(
    AgentSessionLogEnumerationError,
    match="writer lease is unsafe",
  ):
    try_acquire_agent_session_log_write_leases((active,))

  assert lease_path.read_bytes() == b"unchanged"
  assert stat.S_IMODE(lease_path.stat().st_mode) == 0o622


def test_rotated_only_crash_window_returns_usable_logical_active_path(
  tmp_path: Path,
) -> None:
  base = tmp_path / "sessions"
  agent_dir = base / "web_advisor"
  segments_dir = agent_dir / "agentsess_session_user.segments"
  segments_dir.mkdir(parents=True)
  segment = segments_dir / "000000000001-000000000001-g000000.jsonl"
  segment.write_bytes(_entry(seq=1))
  active = agent_dir / "agentsess_session_user.jsonl"

  assert tuple(
    location.path for location in enumerate_agent_session_log_paths(base)
  ) == (active.resolve(),)
  log = AgentSessionLog(active)
  entries, _ = log.query_sync(event_types={"workflow_run_started"})
  assert [entry.seq for entry in entries] == [1]


def test_segment_sidecar_cannot_redirect_discovery_outside_owned_layout(
  tmp_path: Path,
) -> None:
  base = tmp_path / "sessions"
  agent_dir = base / "web_advisor"
  segments_dir = agent_dir / "agentsess_session_user.segments"
  segments_dir.mkdir(parents=True)
  segment = segments_dir / "000000000001-000000000001-g000000.jsonl"
  segment.write_bytes(_entry(seq=1))
  segment.with_suffix(".meta.json").write_text(json.dumps({
    "schema_version": 2,
    "file_role": "segment",
    "logical_stream_id": str(tmp_path / "outside.jsonl"),
  }))
  expected = agent_dir / "agentsess_session_user.jsonl"

  assert tuple(
    location.path for location in enumerate_agent_session_log_paths(base)
  ) == (expected.resolve(),)


def test_active_and_segment_layout_deduplicates_one_logical_path(
  tmp_path: Path,
) -> None:
  base = tmp_path / "sessions"
  agent_dir = base / "web_advisor"
  segments_dir = agent_dir / "agentsess_session_user.segments"
  segments_dir.mkdir(parents=True)
  active = agent_dir / "agentsess_session_user.jsonl"
  active.write_bytes(_entry(seq=2))
  (segments_dir / "000000000001-000000000001-g000000.jsonl").write_bytes(
    _entry(seq=1)
  )

  assert tuple(
    location.path for location in enumerate_agent_session_log_paths(base)
  ) == (active.resolve(),)


@pytest.mark.parametrize("unsafe_kind", ["active_symlink", "segment_symlink"])
def test_symlink_candidate_fails_closed(
  tmp_path: Path,
  unsafe_kind: str,
) -> None:
  base = tmp_path / "sessions"
  agent_dir = base / "web_advisor"
  agent_dir.mkdir(parents=True)
  outside = tmp_path / "outside.jsonl"
  outside.write_bytes(_entry(seq=1))
  if unsafe_kind == "active_symlink":
    (agent_dir / "agentsess_session_user.jsonl").symlink_to(outside)
  else:
    outside_dir = tmp_path / "outside.segments"
    outside_dir.mkdir()
    (agent_dir / "agentsess_session_user.segments").symlink_to(
      outside_dir,
      target_is_directory=True,
    )

  with pytest.raises(AgentSessionLogEnumerationError):
    enumerate_agent_session_log_paths(base)


def test_malformed_or_non_regular_segment_fails_closed(tmp_path: Path) -> None:
  base = tmp_path / "sessions"
  agent_dir = base / "web_advisor"
  segments_dir = agent_dir / "agentsess_session_user.segments"
  segments_dir.mkdir(parents=True)
  malformed = segments_dir / "unexpected.jsonl"
  malformed.write_bytes(_entry(seq=1))

  with pytest.raises(AgentSessionLogEnumerationError):
    enumerate_agent_session_log_paths(base)

  malformed.unlink()
  fifo = segments_dir / "000000000001-000000000001-g000000.jsonl"
  os.mkfifo(fifo)
  with pytest.raises(AgentSessionLogEnumerationError):
    enumerate_agent_session_log_paths(base)


def test_write_lease_set_rejects_live_writer_then_holds_until_release(
  tmp_path: Path,
) -> None:
  log_path = (tmp_path / "sessions" / "web_advisor" / "session.jsonl").resolve()
  log_path.parent.mkdir(parents=True)
  log_path.touch()
  lease_path = log_path.with_name(f"{log_path.name}.write_lease")
  lease_path.touch(mode=0o600)
  live_writer = lease_path.open("a+b")
  fcntl.flock(live_writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

  assert try_acquire_agent_session_log_write_leases((log_path,)) is None
  fcntl.flock(live_writer.fileno(), fcntl.LOCK_UN)
  live_writer.close()

  lease_set = try_acquire_agent_session_log_write_leases((log_path,))
  assert lease_set is not None
  competing_writer = lease_path.open("a+b")
  with pytest.raises(BlockingIOError):
    fcntl.flock(
      competing_writer.fileno(),
      fcntl.LOCK_EX | fcntl.LOCK_NB,
    )
  lease_set.release()
  fcntl.flock(
    competing_writer.fileno(),
    fcntl.LOCK_EX | fcntl.LOCK_NB,
  )
  competing_writer.close()


def test_write_lease_set_rolls_back_prior_locks_on_later_contention(
  tmp_path: Path,
) -> None:
  parent = (tmp_path / "sessions" / "web_advisor").resolve()
  parent.mkdir(parents=True)
  first = parent / "a.jsonl"
  second = parent / "b.jsonl"
  first.touch()
  second.touch()
  second_lease_path = second.with_name(f"{second.name}.write_lease")
  second_lease_path.touch(mode=0o600)
  held_second = second_lease_path.open("a+b")
  fcntl.flock(held_second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

  assert try_acquire_agent_session_log_write_leases((second, first)) is None
  first_lease_path = first.with_name(f"{first.name}.write_lease")
  probe_first = first_lease_path.open("a+b")
  fcntl.flock(probe_first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  probe_first.close()
  held_second.close()


def test_write_lease_rejects_symlink_target(tmp_path: Path) -> None:
  parent = (tmp_path / "sessions" / "web_advisor").resolve()
  parent.mkdir(parents=True)
  log_path = parent / "session.jsonl"
  log_path.touch()
  outside = tmp_path / "outside.lease"
  outside.touch()
  log_path.with_name(f"{log_path.name}.write_lease").symlink_to(outside)

  with pytest.raises(AgentSessionLogEnumerationError):
    try_acquire_agent_session_log_write_leases((log_path,))


def test_write_lease_rejects_swapped_ancestor_without_outside_creation(
  tmp_path: Path,
) -> None:
  base = tmp_path / "sessions"
  log = AgentSessionLog(base / "web_advisor" / "session.jsonl")
  selected_location = enumerate_agent_session_log_paths(base)[0]
  displaced = tmp_path / "sessions.displaced"
  base.rename(displaced)
  outside_base = tmp_path / "outside-sessions"
  outside_parent = outside_base / "web_advisor"
  outside_parent.mkdir(parents=True)
  outside_log = outside_parent / "session.jsonl"
  outside_log.write_bytes(_entry(seq=1, event_type="foreign"))
  outside_before = outside_log.read_bytes()
  base.symlink_to(outside_base, target_is_directory=True)

  with pytest.raises(AgentSessionLogEnumerationError):
    try_acquire_agent_session_log_write_leases((selected_location,))

  assert outside_log.read_bytes() == outside_before
  assert not outside_log.with_name(f"{outside_log.name}.write_lease").exists()
  assert log.path.name == selected_location.path.name


@pytest.mark.parametrize(
  "unsafe_reference",
  ("absolute", "parent", "nested", "invalid_name"),
)
def test_manifest_segment_reference_cannot_escape_local_layout(
  tmp_path: Path,
  unsafe_reference: str,
) -> None:
  base = tmp_path / "sessions"
  agent_dir = base / "web_advisor"
  agent_dir.mkdir(parents=True)
  active = agent_dir / "session.jsonl"
  active.touch()
  segments_dir = agent_dir / "session.segments"
  segments_dir.mkdir()
  outside_dir = tmp_path / "outside"
  outside_dir.mkdir()
  segment_name = "000000000001-000000000001-g000000.jsonl"
  outside = outside_dir / segment_name
  outside.write_bytes(_entry(seq=1))
  outside_meta = outside.with_suffix(".meta.json")
  references = {
    "absolute": str(outside),
    "parent": f"../{segment_name}",
    "nested": f"nested/{segment_name}",
    "invalid_name": "segment.jsonl",
  }
  (segments_dir / "manifest.json").write_text(
    json.dumps(_manifest(references[unsafe_reference])),
    encoding="utf-8",
  )

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="no safe local segment source",
  ):
    AgentSessionLog(active)

  assert not outside_meta.exists()


def test_invalid_manifest_reference_rebuilds_from_safe_local_segment(
  tmp_path: Path,
) -> None:
  agent_dir = tmp_path / "sessions" / "web_advisor"
  agent_dir.mkdir(parents=True)
  active = agent_dir / "session.jsonl"
  active.touch()
  segments_dir = agent_dir / "session.segments"
  segments_dir.mkdir()
  segment_name = "000000000001-000000000001-g000000.jsonl"
  local_segment = segments_dir / segment_name
  local_segment.write_bytes(_entry(seq=1))
  outside = tmp_path / segment_name
  outside.write_bytes(_entry(seq=1))
  outside_meta = outside.with_suffix(".meta.json")
  (segments_dir / "manifest.json").write_text(
    json.dumps(_manifest(str(outside))),
    encoding="utf-8",
  )

  log = AgentSessionLog(active)
  entries, _cursor = log.query_sync(
    event_types={"workflow_run_started"},
  )

  assert [entry.seq for entry in entries] == [1]
  assert not outside_meta.exists()
  repaired = json.loads(
    (segments_dir / "manifest.json").read_text(encoding="utf-8")
  )
  assert [row["path"] for row in repaired["segments"]] == [segment_name]


@pytest.mark.parametrize("unsafe_kind", ("symlink", "hardlink"))
def test_segment_file_must_be_owned_regular_and_single_link(
  tmp_path: Path,
  unsafe_kind: str,
) -> None:
  agent_dir = tmp_path / "sessions" / "web_advisor"
  agent_dir.mkdir(parents=True)
  active = agent_dir / "session.jsonl"
  active.touch()
  segments_dir = agent_dir / "session.segments"
  segments_dir.mkdir()
  segment_name = "000000000001-000000000001-g000000.jsonl"
  outside = tmp_path / segment_name
  outside.write_bytes(_entry(seq=1))
  local_segment = segments_dir / segment_name
  if unsafe_kind == "symlink":
    local_segment.symlink_to(outside)
  else:
    os.link(outside, local_segment)

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="segment is unsafe",
  ):
    AgentSessionLog(active)


def test_segment_swap_between_name_check_and_open_fails_closed(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  agent_dir = tmp_path / "sessions" / "web_advisor"
  agent_dir.mkdir(parents=True)
  active = agent_dir / "session.jsonl"
  active.touch()
  segments_dir = agent_dir / "session.segments"
  segments_dir.mkdir()
  segment_name = "000000000001-000000000001-g000000.jsonl"
  local_segment = segments_dir / segment_name
  local_segment.write_bytes(_entry(seq=1))
  outside = tmp_path / "outside.jsonl"
  outside.write_bytes(_entry(seq=1))
  real_open = session_log_module.os.open
  swapped = False

  def swap_then_open(path: object, flags: int, *args: object, **kwargs: object):
    nonlocal swapped
    if path == segment_name and kwargs.get("dir_fd") is not None and not swapped:
      swapped = True
      local_segment.unlink()
      local_segment.symlink_to(outside)
    return real_open(path, flags, *args, **kwargs)

  monkeypatch.setattr(session_log_module.os, "open", swap_then_open)

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="segment is unavailable",
  ):
    AgentSessionLog(active)

  assert swapped is True


def test_foreign_owned_segment_file_fails_closed(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  agent_dir = tmp_path / "sessions" / "web_advisor"
  agent_dir.mkdir(parents=True)
  active = agent_dir / "session.jsonl"
  active.touch()
  segments_dir = agent_dir / "session.segments"
  segments_dir.mkdir()
  segment_name = "000000000001-000000000001-g000000.jsonl"
  (segments_dir / segment_name).write_bytes(_entry(seq=1))
  real_stat = session_log_module.os.stat

  def foreign_owned_stat(
    path: object,
    *args: object,
    **kwargs: object,
  ) -> os.stat_result:
    result = real_stat(path, *args, **kwargs)
    if path == segment_name and kwargs.get("dir_fd") is not None:
      values = list(result)
      values[4] = os.geteuid() + 1
      return os.stat_result(values)
    return result

  monkeypatch.setattr(session_log_module.os, "stat", foreign_owned_stat)

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="segment is unsafe",
  ):
    AgentSessionLog(active)


def test_normal_rotation_and_repair_remain_descriptor_safe(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = AgentSessionLog(tmp_path / "sessions" / "web_advisor" / "session.jsonl")

  first = log.append_sync({"type": "first"})
  second = log.append_sync({"type": "second"})
  reconstructed = AgentSessionLog(log.path)
  entries, _cursor = reconstructed.query_sync(order="asc")

  assert [entry.seq for entry in entries] == [first.seq, second.seq]
  assert len(tuple(reconstructed.segments_dir.glob("*.jsonl"))) == 1
  assert reconstructed.manifest_path.exists()


def test_strict_current_scan_holds_rotation_snapshot_mutex(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  writer = AgentSessionLog(
    tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  )
  writer.append_sync({"type": "target"})
  reader = AgentSessionLog(writer.path)
  entered = threading.Event()
  release = threading.Event()
  original_parse = reader._parse_entry_current_strict

  def pause_during_scan(raw: bytes):
    entered.set()
    assert release.wait(timeout=5)
    return original_parse(raw)

  monkeypatch.setattr(reader, "_parse_entry_current_strict", pause_during_scan)
  query_result: list[object] = []
  query_thread = threading.Thread(
    target=lambda: query_result.extend(asyncio.run(
      reader.query_current_strict()
    )[0]),
  )
  query_thread.start()
  assert entered.wait(timeout=5)

  append_result: list[object] = []
  append_thread = threading.Thread(
    target=lambda: append_result.append(
      writer.append_sync({"type": "foreign_after_rotation"})
    ),
  )
  append_thread.start()
  time.sleep(0.05)
  assert append_thread.is_alive()

  release.set()
  query_thread.join(timeout=5)
  append_thread.join(timeout=5)
  assert not query_thread.is_alive()
  assert not append_thread.is_alive()
  assert [entry.event["type"] for entry in query_result] == ["target"]
  assert len(append_result) == 1
  entries, _cursor = asyncio.run(reader.query_current_strict())
  assert [entry.event["type"] for entry in entries] == [
    "target",
    "foreign_after_rotation",
  ]


def test_rotation_directory_swap_cannot_move_active_log_outside(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = AgentSessionLog(tmp_path / "sessions" / "web_advisor" / "session.jsonl")
  first = log.append_sync({"type": "private", "secret": "do-not-move"})
  outside = tmp_path / "outside"
  outside.mkdir()
  displaced = log.segments_dir.with_name("session.displaced-segments")
  real_replace = session_log_module.os.replace
  swapped = False

  def swap_directory_then_replace(
    source: object,
    destination: object,
    *args: object,
    **kwargs: object,
  ) -> None:
    nonlocal swapped
    if (
      source == log.path.name
      and kwargs.get("src_dir_fd") is not None
      and kwargs.get("dst_dir_fd") is not None
      and not swapped
    ):
      swapped = True
      log.segments_dir.rename(displaced)
      log.segments_dir.symlink_to(outside, target_is_directory=True)
    real_replace(source, destination, *args, **kwargs)

  monkeypatch.setattr(session_log_module.os, "replace", swap_directory_then_replace)

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="segment directory is unsafe",
  ):
    log.append_sync({"type": "second"})

  assert swapped is True
  assert list(outside.iterdir()) == []
  safe_segments = tuple(displaced.glob("*.jsonl"))
  assert len(safe_segments) == 1
  assert b"do-not-move" in safe_segments[0].read_bytes()
  assert first.seq == 1


@pytest.mark.parametrize("unsafe_kind", ("symlink", "hardlink"))
def test_active_log_constructor_rejects_linked_outside_file(
  tmp_path: Path,
  unsafe_kind: str,
) -> None:
  parent = tmp_path / "sessions" / "web_advisor"
  parent.mkdir(parents=True)
  active = parent / "session.jsonl"
  outside = tmp_path / "outside.jsonl"
  outside.write_bytes(_entry(seq=1))
  original = outside.read_bytes()
  if unsafe_kind == "symlink":
    active.symlink_to(outside)
  else:
    os.link(outside, active)

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="active file is unsafe",
  ):
    AgentSessionLog(active)

  assert outside.read_bytes() == original


def test_active_log_displacement_after_binding_fails_closed(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  log.append_sync({"type": "private", "secret": "bound-secret"})
  displaced = active.with_name("session.displaced.jsonl")
  active.rename(displaced)
  outside = tmp_path / "outside.jsonl"
  outside.write_bytes(_entry(seq=7, event_type="foreign"))
  active.symlink_to(outside)
  outside_before = outside.read_bytes()

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="active file is unsafe",
  ):
    log.query_sync()
  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="active file is unsafe",
  ):
    log.append_sync({"type": "late-private"})

  assert outside.read_bytes() == outside_before
  assert b"bound-secret" in displaced.read_bytes()


def test_active_log_regular_inode_swap_after_binding_is_rejected(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  log.append_sync({"type": "private"})
  displaced = active.with_name("session.displaced.jsonl")
  active.rename(displaced)
  active.write_bytes(_entry(seq=1, event_type="replacement"))
  replacement_before = active.read_bytes()

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="active identity was displaced",
  ):
    log.query_sync()
  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="active identity was displaced",
  ):
    log.append_sync({"type": "late-private"})

  assert active.read_bytes() == replacement_before


def test_active_log_swap_after_descriptor_open_fails_closed(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  log.append_sync({"type": "private", "secret": "bound-secret"})
  outside = tmp_path / "outside.jsonl"
  outside.write_bytes(_entry(seq=9, event_type="foreign"))
  displaced = active.with_name("session.displaced.jsonl")
  real_open = session_log_module.os.open
  swapped = False

  def swap_after_open(path: object, flags: int, *args: object, **kwargs: object):
    nonlocal swapped
    descriptor = real_open(path, flags, *args, **kwargs)
    if (
      path == active.name
      and flags & os.O_ACCMODE == os.O_RDONLY
      and kwargs.get("dir_fd") is not None
      and not swapped
    ):
      swapped = True
      active.rename(displaced)
      active.symlink_to(outside)
    return descriptor

  monkeypatch.setattr(session_log_module.os, "open", swap_after_open)

  outside_before = outside.read_bytes()
  with pytest.raises(AgentSessionLogStorageSecurityError):
    log.query_sync()

  assert swapped is True
  assert outside.read_bytes() == outside_before
  assert b"bound-secret" in displaced.read_bytes()


def test_ancestor_root_swap_after_enumeration_fails_without_outside_access(
  tmp_path: Path,
) -> None:
  base = tmp_path / "sessions"
  active = base / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  log.append_sync({"type": "private", "secret": "bound-secret"})
  enumerated = enumerate_agent_session_log_paths(base)
  assert tuple(location.path for location in enumerated) == (active,)

  displaced = tmp_path / "sessions.displaced"
  base.rename(displaced)
  outside_base = tmp_path / "outside-sessions"
  outside_active = outside_base / "web_advisor" / "session.jsonl"
  outside_active.parent.mkdir(parents=True)
  outside_active.write_bytes(_entry(seq=9, event_type="foreign"))
  outside_before = outside_active.read_bytes()
  base.symlink_to(outside_base, target_is_directory=True)

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="directory chain is unsafe",
  ):
    AgentSessionLog(enumerated[0])
  with pytest.raises(AgentSessionLogStorageSecurityError):
    log.query_sync()
  with pytest.raises(AgentSessionLogStorageSecurityError):
    log.append_sync({"type": "late-private"})

  assert outside_active.read_bytes() == outside_before
  assert tuple(outside_active.parent.iterdir()) == (outside_active,)
  assert b"bound-secret" in (
    displaced / "web_advisor" / "session.jsonl"
  ).read_bytes()


def test_regular_ancestor_replacement_after_enumeration_is_rejected(
  tmp_path: Path,
) -> None:
  base = tmp_path / "sessions"
  active = base / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  log.append_sync({"type": "private", "secret": "bound-secret"})
  location = enumerate_agent_session_log_paths(base)[0]

  displaced = tmp_path / "sessions.displaced"
  base.rename(displaced)
  replacement_active = base / "web_advisor" / "session.jsonl"
  replacement_active.parent.mkdir(parents=True)
  replacement_active.write_bytes(_entry(seq=9, event_type="foreign"))
  replacement_before = replacement_active.read_bytes()

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="directory chain was displaced",
  ):
    AgentSessionLog(location)
  with pytest.raises(
    AgentSessionLogEnumerationError,
    match="directory chain was displaced",
  ):
    try_acquire_agent_session_log_write_leases((location,))

  assert replacement_active.read_bytes() == replacement_before
  assert tuple(replacement_active.parent.iterdir()) == (replacement_active,)


def test_bound_writer_lease_open_rejects_ancestor_swap(
  tmp_path: Path,
) -> None:
  base = tmp_path / "sessions"
  log = AgentSessionLog(base / "web_advisor" / "session.jsonl")
  displaced = tmp_path / "sessions.displaced"
  base.rename(displaced)
  outside_base = tmp_path / "outside-sessions"
  outside_parent = outside_base / "web_advisor"
  outside_parent.mkdir(parents=True)
  base.symlink_to(outside_base, target_is_directory=True)

  with pytest.raises(AgentSessionLogStorageSecurityError):
    log._open_writer_lease_file()

  assert tuple(outside_parent.iterdir()) == ()


def test_directory_descriptor_fstat_failures_do_not_leak(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fd_dir = Path("/dev/fd")
  if not fd_dir.is_dir():
    pytest.skip("descriptor accounting requires /dev/fd")
  log = AgentSessionLog(
    tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  )
  real_open = session_log_module.os.open
  real_fstat = session_log_module.os.fstat
  target_descriptors: set[int] = set()

  def track_final_parent(
    path: object,
    flags: int,
    *args: object,
    **kwargs: object,
  ) -> int:
    descriptor = real_open(path, flags, *args, **kwargs)
    if (
      path == log.path.parent.name
      and kwargs.get("dir_fd") is not None
      and flags & os.O_DIRECTORY
    ):
      target_descriptors.add(descriptor)
    return descriptor

  def fail_for_final_parent(descriptor: int):
    if descriptor in target_descriptors:
      target_descriptors.remove(descriptor)
      raise OSError("injected post-open fstat failure")
    return real_fstat(descriptor)

  before = set(os.listdir(fd_dir))
  monkeypatch.setattr(session_log_module.os, "open", track_final_parent)
  monkeypatch.setattr(session_log_module.os, "fstat", fail_for_final_parent)
  for _ in range(20):
    with pytest.raises(
      AgentSessionLogStorageSecurityError,
      match="directory chain is unavailable",
    ):
      log._open_active_parent_directory()
  after = set(os.listdir(fd_dir))

  assert after == before


def test_strict_current_query_rejects_ambiguous_unterminated_active_tail(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  log.append_sync({"type": "workflow_run_started", "owner_id": "owner"})
  assert asyncio.run(log.latest_seq_current_strict()) == 1
  with active.open("ab") as handle:
    handle.write(b'{"seq":2,"timestamp":')

  with pytest.raises(session_log_module.AgentSessionLogCurrentIntegrityError):
    asyncio.run(log.latest_seq_current_strict())
  with pytest.raises(session_log_module.AgentSessionLogCurrentIntegrityError):
    asyncio.run(
      log.query_current_strict(event_types={"workflow_run_started"})
    )
  entries, _cursor = asyncio.run(
    log.query(event_types={"workflow_run_started"})
  )
  assert [entry.seq for entry in entries] == [1]


@pytest.mark.parametrize("order", ("asc", "desc"))
def test_strict_current_query_streams_full_large_log_with_bounded_page(
  tmp_path: Path,
  order: str,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  active.write_bytes(b"".join(
    _entry(
      seq=seq,
      event_type="visible" if seq % 3 else "other",
    )
    for seq in range(1, 2_001)
  ))
  excluded = frozenset(range(10, 2_001, 10))
  visited = 0

  def exclude_entry(entry) -> bool:
    nonlocal visited
    visited += 1
    return entry.seq % 7 == 0

  entries, cursor = asyncio.run(log.query_current_strict(
    event_types={"visible"},
    order=order,
    limit=11,
    excluded_seqs=excluded,
    exclude_entry=exclude_entry,
  ))
  oracle, _cursor = asyncio.run(log.query(event_types={"visible"}, order=order))
  expected = [
    entry
    for entry in oracle
    if entry.seq not in excluded and entry.seq % 7 != 0
  ]

  assert visited == 2_000 - len(excluded)
  assert [entry.seq for entry in entries] == [
    entry.seq for entry in expected[:11]
  ]
  assert cursor is not None
  assert cursor.after_seq == entries[-1].seq


def test_strict_current_query_validates_after_bounded_page_is_full(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  rows = [
    _entry(seq=seq, event_type="visible")
    for seq in range(1, 101)
  ]
  rows[90] = b'{"seq":91,"timestamp":91,"event": BROKEN\n'
  active.write_bytes(b"".join(rows))

  with pytest.raises(session_log_module.AgentSessionLogCurrentIntegrityError):
    asyncio.run(log.query_current_strict(
      event_types={"visible"},
      order="asc",
      limit=1,
    ))


def test_strict_current_query_rejects_duplicate_sequence_coordinates(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  active.write_bytes(
    _entry(seq=1, event_type="first")
    + _entry(seq=1, event_type="replacement")
  )

  with pytest.raises(
    session_log_module.AgentSessionLogCurrentIntegrityError,
    match="sequence order is ambiguous",
  ):
    asyncio.run(log.query_current_strict(limit=1))
  entries, _cursor = asyncio.run(log.query(order="asc"))
  assert [entry.event["type"] for entry in entries] == [
    "first",
    "replacement",
  ]


def test_strict_current_query_accepts_arbitrary_contiguous_retained_sequence(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  active.write_bytes(
    _entry(seq=41, event_type="first-retained")
    + _entry(seq=42, event_type="next-retained")
  )

  entries, cursor = asyncio.run(log.query_current_strict(order="asc"))

  assert cursor is None
  assert [entry.seq for entry in entries] == [41, 42]


def test_strict_current_query_rejects_sequence_gap(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  active.write_bytes(
    _entry(seq=41, event_type="before-gap")
    + _entry(seq=43, event_type="after-gap")
  )

  with pytest.raises(
    session_log_module.AgentSessionLogCurrentIntegrityError,
    match="sequence order is ambiguous",
  ):
    asyncio.run(log.query_current_strict(limit=1))


@pytest.mark.parametrize(
  ("seq", "timestamp"),
  (
    ("1", 1.0),
    (1.9, 1.0),
    (True, 1.0),
    (1, "1.0"),
    (1, True),
    (1, float("nan")),
    (1, float("inf")),
  ),
)
def test_strict_current_query_rejects_coerced_envelope_coordinates(
  tmp_path: Path,
  seq: object,
  timestamp: object,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  active.write_text(json.dumps({
    "seq": seq,
    "timestamp": timestamp,
    "event": {"type": "workflow_run_started"},
  }) + "\n")

  with pytest.raises(session_log_module.AgentSessionLogCurrentIntegrityError):
    asyncio.run(log.query_current_strict())
  entries, _cursor = asyncio.run(log.query())
  assert len(entries) == 1


def test_strict_current_query_rejects_duplicate_json_keys(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "web_advisor" / "session.jsonl"
  log = AgentSessionLog(active)
  active.write_bytes(
    b'{"seq":1,"seq":2,"timestamp":1.0,'
    b'"event":{"type":"workflow_run_started"}}\n'
  )

  with pytest.raises(
    session_log_module.AgentSessionLogCurrentIntegrityError,
    match="duplicate keys",
  ):
    asyncio.run(log.query_current_strict())
