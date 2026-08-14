from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

from agent_gateway.artifact_sidecar_index import (
  INDEX_VERSION,
  artifact_sidecar_index_path,
  get_artifact_sidecar_index_row,
  reconcile_ui_blocks_index,
  register_ui_blocks_payload_sidecar,
)
from agent_gateway.retention import (
  RetentionPolicy,
  RetentionSweepContext,
  UiBlocksEnvelopeAgeAdapter,
)
from agent_gateway.ui_blocks_metrics import record, snapshot
from agent_gateway.ui_blocks_store import read_ui_blocks_payload, write_ui_blocks_payload
import agent_gateway.ui_blocks_store as store_module

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from test_artifact_api import (  # noqa: E402, F401
  ArtifactApiFixture,
  USER_ID,
  _signed_headers,
  artifact_api,
)


UI_BLOCKS_ID = "ub_0123456789abcdef"


def _envelope(ui_blocks_id: str = UI_BLOCKS_ID, *, ts: float = 1_700_000_000.25) -> dict:
  return {
    "ui_blocks_id": ui_blocks_id,
    "session_id": "session-1",
    "turn_key": "turn-1",
    "emission_index": 2,
    "skill_run_id": None,
    "contract_version": 1,
    "payload": {"lead_text": "Overview", "blocks": []},
    "text_fallback": "Overview",
    "ts": ts,
  }


def _workspace(data_dir: Path, user_id: str = USER_ID) -> Path:
  return data_dir / "users" / user_id / "workspace"


def _configure_workspace(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  *,
  user_id: str = USER_ID,
) -> Path:
  data_dir = tmp_path / "data"
  monkeypatch.setenv("USER_DATA_DIR", str(data_dir))
  return _workspace(data_dir, user_id)


def test_atomic_rename_commit_has_no_temp_residue(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = _configure_workspace(tmp_path, monkeypatch)
  outcome = write_ui_blocks_payload(workspace, _envelope(), user_id=USER_ID)
  assert outcome.path.is_file()
  assert outcome.index_lag is False
  assert json.loads(outcome.path.read_text(encoding="utf-8")) == _envelope()
  assert not outcome.path.with_suffix(".json.tmp").exists()


def test_atomic_rename_failure_leaves_no_partial_or_temp_file(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = _configure_workspace(tmp_path, monkeypatch)
  original_replace = Path.replace

  def fail_replace(self: Path, target: Path) -> Path:
    if self.name == f"{UI_BLOCKS_ID}.json.tmp":
      raise OSError("injected rename failure")
    return original_replace(self, target)

  monkeypatch.setattr(Path, "replace", fail_replace)
  with pytest.raises(OSError, match="injected rename failure"):
    write_ui_blocks_payload(workspace, _envelope(), user_id=USER_ID)
  directory = workspace / "artifacts" / "_ui_blocks"
  assert not (directory / f"{UI_BLOCKS_ID}.json").exists()
  assert not (directory / f"{UI_BLOCKS_ID}.json.tmp").exists()


def test_read_helper_fetches_committed_file_without_index(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = _configure_workspace(tmp_path, monkeypatch)
  directory = workspace / "artifacts" / "_ui_blocks"
  directory.mkdir(parents=True)
  path = directory / f"{UI_BLOCKS_ID}.json"
  path.write_text(json.dumps(_envelope()), encoding="utf-8")
  assert not artifact_sidecar_index_path(workspace).exists()
  assert read_ui_blocks_payload(workspace, UI_BLOCKS_ID) == _envelope()
  with pytest.raises(ValueError, match="invalid ui_blocks_id"):
    read_ui_blocks_payload(workspace, "../escape")


def test_index_registration_failure_reports_lag_after_commit(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = _configure_workspace(tmp_path, monkeypatch)

  def fail_registration(**_kwargs) -> None:
    raise RuntimeError("index offline")

  monkeypatch.setattr(store_module, "register_ui_blocks_payload_sidecar", fail_registration)
  outcome = write_ui_blocks_payload(workspace, _envelope(), user_id=USER_ID)
  assert outcome.index_lag is True
  assert outcome.path.is_file()
  assert read_ui_blocks_payload(workspace, UI_BLOCKS_ID) == _envelope()


def test_registration_writes_exact_index_row_mapping(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = _configure_workspace(tmp_path, monkeypatch)
  outcome = write_ui_blocks_payload(workspace, _envelope(), user_id=USER_ID)
  row = get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="ui_blocks",
    artifact_id=UI_BLOCKS_ID,
    user_id=USER_ID,
  )
  assert row is not None
  assert set(row) == {
    "user_id", "artifact_kind", "artifact_id", "artifact_ref", "payload_ref",
    "scope", "scope_label", "ticker", "skill", "contract_name", "purpose",
    "research_file_id", "control_run_id", "origin_kind", "visibility", "origin_ref",
    "classification_source", "created_ts", "updated_ts", "sidecar_mtime_ns",
    "content_hash", "index_version", "last_seen_ts", "stale_ts", "last_error",
  }
  expected_ts = datetime.fromtimestamp(_envelope()["ts"], tz=timezone.utc).isoformat()
  assert row == {
    "user_id": USER_ID,
    "artifact_kind": "ui_blocks",
    "artifact_id": UI_BLOCKS_ID,
    "artifact_ref": UI_BLOCKS_ID,
    "payload_ref": f"artifacts/_ui_blocks/{UI_BLOCKS_ID}.json",
    "scope": None,
    "scope_label": None,
    "ticker": None,
    "skill": None,
    "contract_name": "hank_ui_blocks.v1",
    "purpose": None,
    "research_file_id": None,
    "control_run_id": None,
    "origin_kind": "chat",
    "visibility": "default",
    "origin_ref": json.dumps(
      {"emission_index": 2, "session_id": "session-1", "turn_key": "turn-1"},
      sort_keys=True,
      separators=(",", ":"),
    ),
    "classification_source": "ui_blocks_store",
    "created_ts": expected_ts,
    "updated_ts": expected_ts,
    "sidecar_mtime_ns": outcome.path.stat().st_mtime_ns,
    "content_hash": hashlib.sha256(outcome.path.read_bytes()).hexdigest(),
    "index_version": INDEX_VERSION,
    "last_seen_ts": row["last_seen_ts"],
    "stale_ts": None,
    "last_error": None,
  }
  assert isinstance(row["last_seen_ts"], str)
  assert json.loads(row["origin_ref"]) == {
    "session_id": "session-1",
    "turn_key": "turn-1",
    "emission_index": 2,
  }


def test_reconcile_removes_missing_rows_registers_files_and_marks_corrupt_orphan(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = _configure_workspace(tmp_path, monkeypatch)
  users_root = workspace.parents[1]
  missing = write_ui_blocks_payload(workspace, _envelope(), user_id=USER_ID)
  missing.path.unlink()

  orphan_id = "ub_1111111111111111"
  directory = workspace / "artifacts" / "_ui_blocks"
  orphan_path = directory / f"{orphan_id}.json"
  orphan_path.write_text(json.dumps(_envelope(orphan_id)), encoding="utf-8")
  corrupt_id = "ub_2222222222222222"
  corrupt_path = directory / f"{corrupt_id}.json"
  corrupt_path.write_text("{not-json", encoding="utf-8")
  existing_corrupt_id = "ub_5555555555555555"
  existing_corrupt = write_ui_blocks_payload(
    workspace,
    _envelope(existing_corrupt_id),
    user_id=USER_ID,
  )
  existing_corrupt.path.write_text("{also-not-json", encoding="utf-8")

  reconcile_ui_blocks_index(users_root)
  assert get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="ui_blocks",
    artifact_id=UI_BLOCKS_ID,
    user_id=USER_ID,
  ) is None
  orphan_row = get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="ui_blocks",
    artifact_id=orphan_id,
    user_id=USER_ID,
  )
  assert orphan_row is not None
  assert orphan_row["last_error"] is None
  corrupt_row = get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="ui_blocks",
    artifact_id=corrupt_id,
    user_id=USER_ID,
  )
  assert corrupt_row is not None
  assert corrupt_row["artifact_ref"] == corrupt_id
  assert corrupt_row["payload_ref"] == f"artifacts/_ui_blocks/{corrupt_id}.json"
  assert corrupt_row["created_ts"] == corrupt_row["updated_ts"]
  assert corrupt_row["sidecar_mtime_ns"] == corrupt_path.stat().st_mtime_ns
  assert corrupt_row["content_hash"] == hashlib.sha256(corrupt_path.read_bytes()).hexdigest()
  assert corrupt_row["index_version"] == INDEX_VERSION
  assert corrupt_row["stale_ts"] is not None
  assert corrupt_row["last_error"] == "corrupt_envelope"
  for column in (
    "scope", "scope_label", "ticker", "skill", "contract_name", "purpose",
    "research_file_id", "control_run_id", "origin_kind", "visibility", "origin_ref",
    "classification_source", "last_seen_ts",
  ):
    assert corrupt_row[column] is None
  existing_corrupt_row = get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="ui_blocks",
    artifact_id=existing_corrupt_id,
    user_id=USER_ID,
  )
  assert existing_corrupt_row is not None
  assert existing_corrupt_row["stale_ts"] is not None
  assert existing_corrupt_row["last_error"] == "corrupt_envelope"


def test_ui_blocks_route_auth_validation_isolation_and_payload_shape(
  artifact_api: ArtifactApiFixture,
) -> None:
  alice_workspace = _workspace(artifact_api.data_dir, USER_ID)
  bob_workspace = _workspace(artifact_api.data_dir, "bob")
  alice_envelope = _envelope()
  bob_id = "ub_bbbbbbbbbbbbbbbb"
  write_ui_blocks_payload(alice_workspace, alice_envelope, user_id=USER_ID)
  write_ui_blocks_payload(bob_workspace, _envelope(bob_id), user_id="bob")

  with TestClient(artifact_api.app) as client:
    unsigned = client.get(f"/api/ui-blocks/{UI_BLOCKS_ID}")
    invalid = client.get("/api/ui-blocks/not-valid", headers=_signed_headers())
    wrong_user = client.get(f"/api/ui-blocks/{bob_id}", headers=_signed_headers())
    response = client.get(f"/api/ui-blocks/{UI_BLOCKS_ID}", headers=_signed_headers())

  assert unsigned.status_code == 401
  assert invalid.status_code == 400
  assert wrong_user.status_code == 404
  assert response.status_code == 200
  assert response.json() == alice_envelope
  assert response.json()["payload"] == alice_envelope["payload"]
  assert response.headers["cache-control"].startswith("private")
  assert response.headers["x-content-type-options"] == "nosniff"


def test_metrics_snapshot_and_health_additive_counters(artifact_api: ArtifactApiFixture) -> None:
  name = "wave4_test_counter"
  before = snapshot().get(name, 0)
  record(name)
  assert snapshot()[name] == before + 1
  with TestClient(artifact_api.app) as client:
    response = client.get("/api/health")
  assert response.status_code == 200
  assert response.json()["status"] == "ok"
  assert "package" in response.json()
  assert response.json()["counters"][name] == before + 1


def test_ui_blocks_envelope_age_uses_ts_falls_back_to_mtime_and_reconciles(
  tmp_path: Path,
) -> None:
  users_root = tmp_path / "users"
  workspace = users_root / USER_ID / "workspace"
  directory = workspace / "artifacts" / "_ui_blocks"
  directory.mkdir(parents=True)
  now = 2_000_000_000.0
  old_id = UI_BLOCKS_ID
  fresh_id = "ub_3333333333333333"
  corrupt_id = "ub_4444444444444444"
  old_path = directory / f"{old_id}.json"
  fresh_path = directory / f"{fresh_id}.json"
  corrupt_path = directory / f"{corrupt_id}.json"
  old_path.write_text(json.dumps(_envelope(old_id, ts=now - 10 * 86_400)), encoding="utf-8")
  fresh_path.write_text(json.dumps(_envelope(fresh_id, ts=now)), encoding="utf-8")
  corrupt_path.write_text("broken", encoding="utf-8")
  os.utime(old_path, (now, now))
  os.utime(fresh_path, (now - 30 * 86_400, now - 30 * 86_400))
  os.utime(corrupt_path, (now - 10 * 86_400, now - 10 * 86_400))
  register_ui_blocks_payload_sidecar(
    workspace_dir=workspace,
    user_id=USER_ID,
    ui_blocks_id=old_id,
    path=old_path,
    session_id="session-1",
    turn_key="turn-1",
    emission_index=2,
    ts=now - 10 * 86_400,
  )
  policy = RetentionPolicy(
    "age",
    "platform",
    "chat render cache",
    max_age_days=7,
  )
  adapter = UiBlocksEnvelopeAgeAdapter("ui_blocks_payloads", users_root)
  dry_context = RetentionSweepContext(
    mode="dry_run",
    now=datetime.fromtimestamp(now, tz=timezone.utc),
    policy=policy,
    authorized_roots=(users_root,),
  )
  dry_report = adapter.sweep(dry_context)
  assert dry_report.would_delete_count == 2
  assert old_path.exists() and fresh_path.exists() and corrupt_path.exists()

  enforce_context = RetentionSweepContext(
    mode="enforce",
    now=datetime.fromtimestamp(now, tz=timezone.utc),
    policy=policy,
    authorized_roots=(users_root,),
  )
  report = adapter.sweep(enforce_context)
  assert report.deleted_count == 2
  assert not old_path.exists()
  assert not corrupt_path.exists()
  assert fresh_path.exists()
  assert get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="ui_blocks",
    artifact_id=old_id,
    user_id=USER_ID,
  ) is None
