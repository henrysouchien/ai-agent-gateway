from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent_gateway.agent_session_log import (
  AgentSessionLog,
  AgentSessionLogLocation,
  AgentSessionLogStorageSecurityError,
  enumerate_agent_session_log_paths,
)
from agent_gateway.agent_session_log_records import (
  AgentSessionRef,
  resolve_agent_session_id,
)


def _rotated_log(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> tuple[
  AgentSessionLog,
  AgentSessionLogLocation,
  dict[str, Any],
  Path,
]:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  session_ref = AgentSessionRef(
    user_id="alice",
    agent_id="analyst",
    agent_session_id=resolve_agent_session_id("alice", "analyst"),
  )
  root = tmp_path / "sessions"
  log = AgentSessionLog(session_ref=session_ref, base_dir=root)
  log.append_sync({"type": "assistant_message", "text": "first"})
  log.append_sync({"type": "assistant_message", "text": "second"})
  manifest = json.loads(log.manifest_path.read_text(encoding="utf-8"))
  descriptor = dict(manifest["segments"][0])
  segment = log.segments_dir / descriptor["path"]
  locations = enumerate_agent_session_log_paths(root)
  assert len(locations) == 1
  return log, locations[0], descriptor, segment


def _retirement_expectations(
  log: AgentSessionLog,
  segment: Path,
) -> dict[str, Any]:
  def identity(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return (
      int(info.st_dev),
      int(info.st_ino),
      int(info.st_size),
      int(info.st_mtime_ns),
    )

  sidecar = segment.with_suffix(".meta.json")
  return {
    "expected_manifest": json.loads(
      log.manifest_path.read_text(encoding="utf-8")
    ),
    "expected_manifest_identity": identity(log.manifest_path),
    "expected_segment_identity": identity(segment) if segment.exists() else None,
    "expected_sidecar_identity": identity(sidecar) if sidecar.exists() else None,
  }


def test_retire_segment_is_durable_and_idempotent_after_unlink_crash(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log, location, descriptor, segment = _rotated_log(tmp_path, monkeypatch)
  original_publish = AgentSessionLog._atomic_write_segments_json_at
  crashed = False

  def crash_before_manifest(
    self: AgentSessionLog,
    directory_descriptor: int,
    name: str,
    payload: dict[str, Any],
    *,
    expected_identity: tuple[int, int] | None = None,
  ) -> None:
    nonlocal crashed
    if not crashed and name == "manifest.json":
      crashed = True
      raise RuntimeError("crash after durable unlink")
    original_publish(
      self,
      directory_descriptor,
      name,
      payload,
      expected_identity=expected_identity,
    )

  monkeypatch.setattr(
    AgentSessionLog,
    "_atomic_write_segments_json_at",
    crash_before_manifest,
  )
  expectations = _retirement_expectations(log, segment)
  with pytest.raises(RuntimeError, match="durable unlink"):
    AgentSessionLog.retire_segment(location, descriptor, **expectations)

  assert not segment.exists()
  assert not segment.with_suffix(".meta.json").exists()
  assert json.loads(log.manifest_path.read_text())["segments"] == [descriptor]

  recovered = AgentSessionLog.retire_segment(
    location,
    descriptor,
    **_retirement_expectations(log, segment),
  )
  assert recovered is not None
  assert recovered.removed_from_manifest is True
  assert recovered.deleted_bytes == 0
  assert json.loads(log.manifest_path.read_text())["segments"] == []

  again = AgentSessionLog.retire_segment(
    location,
    {
      "segment_id": descriptor["segment_id"],
      "path": descriptor["path"],
      "telemetry_source_id": descriptor["telemetry_source_id"],
    },
    **_retirement_expectations(log, segment),
  )
  assert again is not None
  assert again.removed_from_manifest is False


def test_retire_segment_fails_closed_for_inverse_half_deleted_state(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log, location, descriptor, segment = _rotated_log(tmp_path, monkeypatch)
  sidecar = segment.with_suffix(".meta.json")
  sidecar.unlink()

  with pytest.raises(
    AgentSessionLogStorageSecurityError,
    match="segment sidecar is missing",
  ):
    AgentSessionLog.retire_segment(
      location,
      descriptor,
      **_retirement_expectations(log, segment),
    )

  assert segment.exists()
  assert json.loads(log.manifest_path.read_text())["segments"] == [descriptor]


@pytest.mark.parametrize("mutation", ["descriptor", "segment", "sidecar"])
def test_retire_segment_rejects_identity_mismatch_before_deletion(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  mutation: str,
) -> None:
  log, location, descriptor, segment = _rotated_log(tmp_path, monkeypatch)
  expectations = _retirement_expectations(log, segment)
  expected = dict(descriptor)
  if mutation == "descriptor":
    expected["bytes"] = int(expected["bytes"]) + 1
  elif mutation == "segment":
    with segment.open("ab") as handle:
      handle.write(b"changed")
  else:
    sidecar_path = segment.with_suffix(".meta.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["telemetry_source_id"] = "changed-source"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

  with pytest.raises(AgentSessionLogStorageSecurityError):
    AgentSessionLog.retire_segment(
      location,
      expected,
      **expectations,
    )

  assert segment.exists()
  assert segment.with_suffix(".meta.json").exists()
  assert json.loads(log.manifest_path.read_text())["segments"] == [descriptor]


def test_retire_segment_rejects_symlinked_segment_without_touching_target(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log, location, descriptor, segment = _rotated_log(tmp_path, monkeypatch)
  expectations = _retirement_expectations(log, segment)
  outside = tmp_path / "outside.jsonl"
  segment.rename(outside)
  segment.symlink_to(outside)

  with pytest.raises(AgentSessionLogStorageSecurityError, match="unsafe"):
    AgentSessionLog.retire_segment(location, descriptor, **expectations)

  assert outside.exists()
  assert outside.read_bytes()
  assert json.loads(log.manifest_path.read_text())["segments"] == [descriptor]


def test_retire_segment_rejects_segments_directory_swap(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log, location, descriptor, segment = _rotated_log(tmp_path, monkeypatch)
  expectations = _retirement_expectations(log, segment)
  displaced = log.segments_dir.with_name(f"{log.segments_dir.name}.displaced")
  log.segments_dir.rename(displaced)
  log.segments_dir.mkdir()

  with pytest.raises(AgentSessionLogStorageSecurityError, match="displaced"):
    AgentSessionLog.retire_segment(location, descriptor, **expectations)

  assert (displaced / segment.name).exists()
  assert json.loads((displaced / "manifest.json").read_text())["segments"] == [
    descriptor
  ]


@pytest.mark.parametrize("target", ["manifest", "sidecar"])
def test_retire_segment_rejects_duplicate_json_fields(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  target: str,
) -> None:
  log, location, descriptor, segment = _rotated_log(tmp_path, monkeypatch)
  path = (
    log.manifest_path
    if target == "manifest"
    else segment.with_suffix(".meta.json")
  )
  payload = path.read_text(encoding="utf-8").rstrip()
  path.write_text(
    payload[:-1] + ',"schema_version":1}',
    encoding="utf-8",
  )
  expectations = _retirement_expectations(log, segment)

  with pytest.raises(AgentSessionLogStorageSecurityError, match="duplicate"):
    AgentSessionLog.retire_segment(location, descriptor, **expectations)

  assert segment.exists()
  assert json.loads(log.manifest_path.read_text())["segments"] == [descriptor]


@pytest.mark.parametrize("target", ["manifest", "sidecar"])
def test_retire_segment_rejects_same_content_inode_replacement_after_inventory(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  target: str,
) -> None:
  log, location, descriptor, segment = _rotated_log(tmp_path, monkeypatch)
  expectations = _retirement_expectations(log, segment)
  path = (
    log.manifest_path
    if target == "manifest"
    else segment.with_suffix(".meta.json")
  )
  replacement = path.with_name(f"{path.name}.replacement")
  replacement.write_bytes(path.read_bytes())
  os.replace(replacement, path)

  with pytest.raises(AgentSessionLogStorageSecurityError, match="inventory"):
    AgentSessionLog.retire_segment(location, descriptor, **expectations)

  assert segment.exists()


@pytest.mark.parametrize("target", ["parent", "segments"])
def test_retire_segment_rejects_writable_directory_mode_change(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  target: str,
) -> None:
  log, location, descriptor, segment = _rotated_log(tmp_path, monkeypatch)
  expectations = _retirement_expectations(log, segment)
  path = log.path.parent if target == "parent" else log.segments_dir
  original_mode = path.stat().st_mode & 0o777
  path.chmod(0o777)
  try:
    with pytest.raises(AgentSessionLogStorageSecurityError, match="unsafe"):
      AgentSessionLog.retire_segment(location, descriptor, **expectations)
  finally:
    path.chmod(original_mode)

  assert segment.exists()


def test_retire_segment_supports_rotation_only_crash_window_without_recreating_active(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log, _location, descriptor, _segment = _rotated_log(tmp_path, monkeypatch)
  log.path.unlink()
  location = enumerate_agent_session_log_paths(tmp_path / "sessions")[0]
  assert location.active_identity is None

  result = AgentSessionLog.retire_segment(
    location,
    descriptor,
    **_retirement_expectations(log, log.segments_dir / descriptor["path"]),
  )

  assert result is not None
  assert result.removed_from_manifest is True
  assert not log.path.exists()
  assert json.loads(log.manifest_path.read_text())["segments"] == []
