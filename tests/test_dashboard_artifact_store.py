from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.dashboard_artifact_store import (
  list_dashboard_artifacts,
  read_dashboard_artifact_payload,
  read_dashboard_artifact_sidecar,
  write_dashboard_artifact,
)
from schema.dashboard_artifact import DashboardArtifact


def test_dashboard_artifact_store_round_trips_sidecar_and_payload(tmp_path: Path) -> None:
  artifact = _artifact("dashboard-artifact-1", ticker="PCTY")
  payload = _payload("PCTY Dashboard")

  write_dashboard_artifact(
    workspace_dir=tmp_path,
    artifact=artifact,
    payload_json=payload,
  )

  assert read_dashboard_artifact_sidecar(tmp_path, "dashboard-artifact-1") == artifact
  assert read_dashboard_artifact_payload(tmp_path, "dashboard-artifact-1") == payload


def test_dashboard_artifact_store_lists_newest_first_with_filters(tmp_path: Path) -> None:
  old_artifact = _artifact("old-artifact", ticker="PCTY")
  new_artifact = _artifact("new-artifact", ticker=None)
  other_artifact = _artifact("other-artifact", ticker="MSFT")

  for artifact in [old_artifact, new_artifact, other_artifact]:
    write_dashboard_artifact(
      workspace_dir=tmp_path,
      artifact=artifact,
      payload_json=_payload(artifact.artifact_id),
    )

  _set_sidecar_mtime(tmp_path, old_artifact.artifact_id, 100)
  _set_sidecar_mtime(tmp_path, new_artifact.artifact_id, 300)
  _set_sidecar_mtime(tmp_path, other_artifact.artifact_id, 200)

  assert [artifact.artifact_id for artifact in list_dashboard_artifacts(tmp_path)] == [
    "new-artifact",
    "other-artifact",
    "old-artifact",
  ]
  assert [artifact.artifact_id for artifact in list_dashboard_artifacts(tmp_path, ticker="PCTY")] == [
    "old-artifact"
  ]
  assert [artifact.artifact_id for artifact in list_dashboard_artifacts(tmp_path, since=250)] == [
    "new-artifact"
  ]
  assert [artifact.artifact_id for artifact in list_dashboard_artifacts(tmp_path, limit=1)] == [
    "new-artifact"
  ]


def test_dashboard_artifact_store_rejects_unsafe_artifact_ids_before_writing(tmp_path: Path) -> None:
  artifact = _artifact("../escape", ticker="PCTY")

  try:
    write_dashboard_artifact(workspace_dir=tmp_path, artifact=artifact, payload_json=_payload("bad"))
  except ValueError as exc:
    assert "invalid dashboard artifact_id" in str(exc)
  else:
    raise AssertionError("unsafe artifact id should raise")

  assert not (tmp_path / "artifacts").exists()


def test_dashboard_artifact_store_rejects_symlink_escape(tmp_path: Path) -> None:
  workspace = tmp_path / "workspace"
  outside = tmp_path / "outside"
  (workspace / "artifacts").mkdir(parents=True)
  outside.mkdir()
  (workspace / "artifacts" / "_dashboards").symlink_to(outside, target_is_directory=True)

  try:
    write_dashboard_artifact(
      workspace_dir=workspace,
      artifact=_artifact("symlink-escape", ticker="PCTY"),
      payload_json=_payload("safe"),
    )
  except ValueError as exc:
    assert "escapes workspace" in str(exc)
  else:
    raise AssertionError("symlinked dashboard artifact directory should raise")

  assert list(outside.iterdir()) == []


def test_dashboard_artifact_store_list_ignores_symlinked_sidecar_escape(tmp_path: Path) -> None:
  workspace = tmp_path / "workspace"
  outside = tmp_path / "outside"
  outside.mkdir()
  write_dashboard_artifact(
    workspace_dir=workspace,
    artifact=_artifact("safe-artifact", ticker="PCTY"),
    payload_json=_payload("safe"),
  )
  outside_sidecar = outside / "leaked.json"
  outside_sidecar.write_text(_artifact("leaked", ticker="MSFT").model_dump_json(), encoding="utf-8")
  (workspace / "artifacts" / "_dashboards" / "leaked.json").symlink_to(outside_sidecar)

  assert [artifact.artifact_id for artifact in list_dashboard_artifacts(workspace)] == ["safe-artifact"]


def test_dashboard_artifact_store_missing_payload_is_soft_absent(tmp_path: Path) -> None:
  artifact = _artifact("sidecar-only", ticker=None)
  directory = tmp_path / "artifacts" / "_dashboards"
  directory.mkdir(parents=True)
  (directory / "sidecar-only.json").write_text(artifact.model_dump_json(), encoding="utf-8")

  assert read_dashboard_artifact_sidecar(tmp_path, "sidecar-only") == artifact
  assert read_dashboard_artifact_payload(tmp_path, "sidecar-only") is None


def test_dashboard_artifact_store_ignores_partial_temp_writes(tmp_path: Path) -> None:
  directory = tmp_path / "artifacts" / "_dashboards"
  directory.mkdir(parents=True)
  (directory / "partial.payload.json.tmp").write_text('{"partial":true}', encoding="utf-8")

  assert list_dashboard_artifacts(tmp_path) == []


def _artifact(
  artifact_id: str,
  *,
  ticker: str | None,
) -> DashboardArtifact:
  return DashboardArtifact(
    artifact_id=artifact_id,
    title=f"{artifact_id} title",
    summary=f"{artifact_id} summary",
    ticker=ticker,
    scope_label=None,
    source_skill="fixture-dashboard-artifact",
    readiness_posture="decision_ready",
    profile="production",
    payload_ref=f"{artifact_id}.payload.json",
    ts="2026-06-01T12:00:00+00:00",
  )


def _payload(title: str) -> dict[str, Any]:
  return {
    "kind": "hank_dashboard.v1",
    "title": title,
    "sections": [],
  }


def _set_sidecar_mtime(workspace_dir: Path, artifact_id: str, mtime: int) -> None:
  path = workspace_dir / "artifacts" / "_dashboards" / f"{artifact_id}.json"
  os.utime(path, (mtime, mtime))
