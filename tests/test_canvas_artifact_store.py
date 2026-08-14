from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.canvas_artifact_store as canvas_store
from agent_gateway.artifact_sidecar_index import get_artifact_sidecar_index_row
from agent_gateway.canvas_artifact_store import (
  list_canvas_artifacts,
  read_canvas_artifact_bundle,
  read_canvas_artifact_sidecar,
  read_canvas_artifact_source,
  write_canvas_artifact,
)
from schema.canvas_artifact import CanvasArtifact, StaticExports


SOURCE = "export default function Artifact() { return <div>PCTY</div>; }\n"
BUNDLE = b"(() => { globalThis.canvasArtifact = 'PCTY'; })();\n"


def test_canvas_artifact_store_round_trips_all_three_files(tmp_path: Path) -> None:
  artifact = _artifact("canvas-artifact-1", ticker="PCTY")
  write_canvas_artifact(workspace_dir=tmp_path, artifact=artifact, source=SOURCE, bundle=BUNDLE)
  assert read_canvas_artifact_sidecar(tmp_path, artifact.artifact_id) == artifact
  assert read_canvas_artifact_source(tmp_path, artifact.artifact_id) == SOURCE
  assert read_canvas_artifact_bundle(tmp_path, artifact.artifact_id) == BUNDLE


def test_canvas_artifact_store_registers_canvas_index_row(tmp_path: Path) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  artifact = _artifact("canvas-artifact-1", ticker="PCTY", purpose="comparison")
  write_canvas_artifact(workspace_dir=workspace, artifact=artifact, source=SOURCE, bundle=BUNDLE)
  row = get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="canvas",
    artifact_id=artifact.artifact_id,
    user_id="alice",
  )
  assert row is not None
  assert row["artifact_ref"] == "artifacts/_canvas/canvas-artifact-1.json"
  assert row["payload_ref"] == "artifacts/_canvas/canvas-artifact-1.bundle.js"
  assert row["contract_name"] == "CanvasArtifact"
  assert row["purpose"] == "comparison"
  assert row["ticker"] == "PCTY"


def test_canvas_artifact_store_sidecar_is_last_and_failure_removes_orphans(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  artifact = _artifact("interrupted", ticker=None)
  original_replace = Path.replace
  replacements: list[str] = []

  def _replace(path: Path, target: Path) -> Path:
    replacements.append(target.name)
    if target.name == "interrupted.json":
      raise OSError("simulated sidecar failure")
    return original_replace(path, target)

  monkeypatch.setattr(Path, "replace", _replace)
  with pytest.raises(OSError, match="simulated sidecar failure"):
    write_canvas_artifact(workspace_dir=tmp_path, artifact=artifact, source=SOURCE, bundle=BUNDLE)

  assert replacements == ["interrupted.tsx", "interrupted.bundle.js", "interrupted.json"]
  directory = tmp_path / "artifacts" / "_canvas"
  assert list(directory.iterdir()) == []
  assert list_canvas_artifacts(tmp_path) == []


def test_canvas_artifact_store_list_ignores_source_and_bundle_orphans(tmp_path: Path) -> None:
  directory = tmp_path / "artifacts" / "_canvas"
  directory.mkdir(parents=True)
  (directory / "orphan.tsx").write_text(SOURCE, encoding="utf-8")
  (directory / "orphan.bundle.js").write_bytes(BUNDLE)
  assert list_canvas_artifacts(tmp_path) == []


def test_canvas_artifact_store_lists_newest_first_with_filters(tmp_path: Path) -> None:
  old = _artifact("old", ticker="PCTY", purpose="exploration")
  new = _artifact("new", ticker=None, purpose="comparison")
  other = _artifact("other", ticker="MSFT", purpose="comparison")
  for artifact in (old, new, other):
    write_canvas_artifact(workspace_dir=tmp_path, artifact=artifact, source=SOURCE, bundle=BUNDLE)
  _set_sidecar_mtime(tmp_path, "old", 100)
  _set_sidecar_mtime(tmp_path, "new", 300)
  _set_sidecar_mtime(tmp_path, "other", 200)
  assert [item.artifact_id for item in list_canvas_artifacts(tmp_path)] == ["new", "other", "old"]
  assert [item.artifact_id for item in list_canvas_artifacts(tmp_path, ticker="PCTY")] == ["old"]
  assert [item.artifact_id for item in list_canvas_artifacts(tmp_path, purpose="comparison")] == ["new", "other"]
  assert [item.artifact_id for item in list_canvas_artifacts(tmp_path, since=250)] == ["new"]


def test_canvas_artifact_store_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="invalid canvas artifact_id"):
    write_canvas_artifact(
      workspace_dir=tmp_path,
      artifact=_artifact("../escape", ticker="PCTY"),
      source=SOURCE,
      bundle=BUNDLE,
    )
  assert not (tmp_path / "artifacts").exists()

  workspace = tmp_path / "workspace"
  outside = tmp_path / "outside"
  (workspace / "artifacts").mkdir(parents=True)
  outside.mkdir()
  (workspace / "artifacts" / "_canvas").symlink_to(outside, target_is_directory=True)
  with pytest.raises(ValueError, match="escapes workspace"):
    write_canvas_artifact(
      workspace_dir=workspace,
      artifact=_artifact("symlink-escape", ticker="PCTY"),
      source=SOURCE,
      bundle=BUNDLE,
    )
  assert list(outside.iterdir()) == []


def test_canvas_artifact_store_user_workspaces_are_isolated(tmp_path: Path) -> None:
  alice = tmp_path / "users" / "alice" / "workspace"
  bob = tmp_path / "users" / "bob" / "workspace"
  write_canvas_artifact(
    workspace_dir=bob,
    artifact=_artifact("bob-only", ticker="PCTY"),
    source=SOURCE,
    bundle=BUNDLE,
  )
  assert read_canvas_artifact_sidecar(alice, "bob-only") is None
  assert read_canvas_artifact_sidecar(bob, "bob-only") is not None


def test_canvas_artifact_store_validates_source_and_bundle_digests(tmp_path: Path) -> None:
  artifact = _artifact("bad-digest", ticker=None)
  with pytest.raises(ValueError, match="source digest"):
    write_canvas_artifact(workspace_dir=tmp_path, artifact=artifact, source=SOURCE + "x", bundle=BUNDLE)
  with pytest.raises(ValueError, match="bundle digest"):
    write_canvas_artifact(workspace_dir=tmp_path, artifact=artifact, source=SOURCE, bundle=BUNDLE + b"x")
  assert not (tmp_path / "artifacts").exists()


def test_canvas_artifact_store_keeps_committed_files_when_index_registration_fails(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
) -> None:
  artifact = _artifact("index-failure", ticker="PCTY")
  monkeypatch.setattr(canvas_store, "register_canvas_artifact_sidecar", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("down")))
  with caplog.at_level(logging.WARNING, logger="agent_gateway.canvas_artifact_store"):
    write_canvas_artifact(workspace_dir=tmp_path, artifact=artifact, source=SOURCE, bundle=BUNDLE)
  assert read_canvas_artifact_sidecar(tmp_path, artifact.artifact_id) == artifact
  assert read_canvas_artifact_source(tmp_path, artifact.artifact_id) == SOURCE
  assert read_canvas_artifact_bundle(tmp_path, artifact.artifact_id) == BUNDLE
  assert "artifact_index_failure" in caplog.messages


def _artifact(
  artifact_id: str,
  *,
  ticker: str | None,
  purpose: str = "exploration",
) -> CanvasArtifact:
  return CanvasArtifact(
    artifact_id=artifact_id,
    title=f"{artifact_id} title",
    purpose=purpose,
    source_ref=f"{artifact_id}.tsx",
    source_digest=hashlib.sha256(SOURCE.encode("utf-8")).hexdigest(),
    bundle_ref=f"{artifact_id}.bundle.js",
    bundle_digest=hashlib.sha256(BUNDLE).hexdigest(),
    toolchain_version="node/24.4.1 tsc/5.8.3 esbuild/0.25.6",
    kit_contract_version=1,
    summary=f"{artifact_id} summary",
    ticker=ticker,
    session_id=None,
    source_skill="scenario-comparison",
    sources=[],
    exports=StaticExports(
      copy_as_prompt="Prompt export",
      copy_as_markdown=f"## {artifact_id}",
      copy_as_json={"artifact_id": artifact_id},
    ),
    ts="2026-07-21T12:00:00+00:00",
  )


def _set_sidecar_mtime(workspace_dir: Path, artifact_id: str, mtime: int) -> None:
  os.utime(workspace_dir / "artifacts" / "_canvas" / f"{artifact_id}.json", (mtime, mtime))
