from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.artifact_sidecar_index import (  # noqa: E402
  delete_artifact_sidecar_index_rows,
  get_artifact_sidecar_index_row,
  register_skill_artifact_sidecar,
  register_ui_blocks_payload_sidecar,
)
from agent_gateway.canvas_artifact_store import (  # noqa: E402
  read_canvas_artifact_sidecar,
  write_canvas_artifact,
)
from agent_gateway.dashboard_artifact_store import (  # noqa: E402
  read_dashboard_artifact_sidecar,
  write_dashboard_artifact,
)
from agent_gateway.document_artifact_erase import (  # noqa: E402
  purge_research_file_artifacts,
  purge_ui_blocks_payloads,
)
from agent_gateway import document_artifact_erase as erase_module  # noqa: E402
from agent_gateway.html_artifact_store import (  # noqa: E402
  read_html_artifact_sidecar,
  read_html_artifact_content,
  write_html_artifact,
)
from schema.canvas_artifact import (  # noqa: E402
  CanvasArtifact,
  StaticExports as CanvasStaticExports,
)
from schema.dashboard_artifact import DashboardArtifact  # noqa: E402
from schema.html_artifact import (  # noqa: E402
  HtmlArtifact,
  StaticExports as HtmlStaticExports,
)


SOURCE = "export default function Artifact() { return <div>PCTY</div>; }\n"
BUNDLE = b"(() => { globalThis.canvasArtifact = 'PCTY'; })();\n"


def canvas_artifact(artifact_id: str, *, ticker: str | None) -> CanvasArtifact:
  return CanvasArtifact(
    artifact_id=artifact_id,
    title=f"{artifact_id} title",
    purpose="exploration",
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
    exports=CanvasStaticExports(
      copy_as_prompt="Prompt export",
      copy_as_markdown=f"## {artifact_id}",
      copy_as_json={"artifact_id": artifact_id},
    ),
    ts="2026-07-21T12:00:00+00:00",
  )


def dashboard_artifact(
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


def dashboard_payload(title: str) -> dict[str, Any]:
  return {"kind": "hank_dashboard.v1", "title": title, "sections": []}


def html_artifact(artifact_id: str, *, ticker: str | None) -> HtmlArtifact:
  return HtmlArtifact(
    artifact_id=artifact_id,
    title=f"{artifact_id} title",
    purpose="exploration",
    content_ref=f"{artifact_id}.html",
    summary=f"{artifact_id} summary",
    ticker=ticker,
    session_id=None,
    source_skill="historical-coincidences",
    sources=[],
    exports=HtmlStaticExports(
      copy_as_prompt="Prompt export",
      copy_as_markdown=None,
      copy_as_json={"artifact_id": artifact_id},
    ),
    ts="2026-06-01T12:00:00+00:00",
  )


def _with_file_id(artifact, research_file_id: int):
  return artifact.model_copy(update={"research_file_id": research_file_id})


def _open_descriptor_count() -> int:
  return len(tuple(Path("/dev/fd").iterdir()))


def _write_skill_artifact(
  workspace: Path,
  *,
  artifact_id: str,
  research_file_id: int,
) -> tuple[Path, Path]:
  sidecar = (
    workspace / "artifacts" / "PCTY" / "ir-composer" /
    f"{artifact_id}.json"
  )
  docx_ref = f"letters/PCTY/{artifact_id}.docx"
  docx = workspace / docx_ref
  sidecar.parent.mkdir(parents=True, exist_ok=True)
  docx.parent.mkdir(parents=True, exist_ok=True)
  docx.write_bytes(b"private document evidence")
  sidecar.write_text(
    json.dumps({
      "artifact_id": artifact_id,
      "ticker": "PCTY",
      "skill": "ir-composer",
      "research_file_id": research_file_id,
      "binary_artifact_path": docx_ref,
    }),
    encoding="utf-8",
  )
  register_skill_artifact_sidecar(
    workspace_dir=workspace,
    sidecar_path=sidecar,
    user_id="alice",
  )
  return sidecar, docx


def test_purge_removes_typed_file_artifacts_and_preserves_unrelated(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  bob_workspace = tmp_path / "users" / "bob" / "workspace"
  target_canvas = _with_file_id(
    canvas_artifact("target-canvas", ticker="PCTY"),
    41,
  )
  unrelated_canvas = _with_file_id(
    canvas_artifact("unrelated-canvas", ticker="PCTY"),
    42,
  )
  bob_canvas = _with_file_id(
    canvas_artifact("bob-canvas", ticker="PCTY"),
    41,
  )
  unanchored_same_ticker = canvas_artifact(
    "legacy-same-ticker",
    ticker="PCTY",
  )
  unanchored_other_ticker = canvas_artifact(
    "legacy-other-ticker",
    ticker="MSFT",
  )
  bob_unanchored_same_ticker = canvas_artifact(
    "bob-legacy-same-ticker",
    ticker="PCTY",
  )
  for target_workspace, artifact in (
    (workspace, target_canvas),
    (workspace, unrelated_canvas),
    (workspace, unanchored_same_ticker),
    (workspace, unanchored_other_ticker),
    (bob_workspace, bob_canvas),
    (bob_workspace, bob_unanchored_same_ticker),
  ):
    write_canvas_artifact(
      workspace_dir=target_workspace,
      artifact=artifact,
      source=SOURCE,
      bundle=BUNDLE,
    )

  target_dashboard = _with_file_id(
    dashboard_artifact("target-dashboard", ticker="PCTY"),
    41,
  )
  write_dashboard_artifact(
    workspace_dir=workspace,
    artifact=target_dashboard,
    payload_json=dashboard_payload("private"),
  )
  target_html = _with_file_id(
    html_artifact("target-html", ticker="PCTY"),
    41,
  )
  write_html_artifact(
    workspace_dir=workspace,
    artifact=target_html,
    html_content="<p>private</p>",
  )
  skill_sidecar, skill_docx = _write_skill_artifact(
    workspace,
    artifact_id="target-letter",
    research_file_id=41,
  )

  purge_research_file_artifacts(
    workspace,
    user_id="alice",
    research_file_ids=(41,),
    research_file_tickers=("PCTY",),
  )

  assert read_canvas_artifact_sidecar(workspace, "target-canvas") is None
  assert read_dashboard_artifact_sidecar(workspace, "target-dashboard") is None
  assert read_html_artifact_sidecar(workspace, "target-html") is None
  assert skill_sidecar.exists() is False
  assert skill_docx.exists() is False
  assert read_canvas_artifact_sidecar(workspace, "unrelated-canvas") is not None
  assert read_canvas_artifact_sidecar(workspace, "legacy-same-ticker") is None
  assert read_canvas_artifact_sidecar(workspace, "legacy-other-ticker") is not None
  assert read_canvas_artifact_sidecar(bob_workspace, "bob-canvas") is not None
  assert read_canvas_artifact_sidecar(
    bob_workspace,
    "bob-legacy-same-ticker",
  ) is not None
  for artifact_kind, artifact_id in (
    ("canvas", "target-canvas"),
    ("dashboard", "target-dashboard"),
    ("html", "target-html"),
    ("skill_artifact", "target-letter"),
  ):
    assert get_artifact_sidecar_index_row(
      workspace_dir=workspace,
      artifact_kind=artifact_kind,
      artifact_id=artifact_id,
      user_id="alice",
    ) is None

  # Retry is a value-semantic no-op.
  purge_research_file_artifacts(
    workspace,
    user_id="alice",
    research_file_ids=(41,),
    research_file_tickers=("PCTY",),
  )


def test_purge_uses_typed_sidecars_when_index_is_lagging(tmp_path: Path) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  artifact = _with_file_id(
    canvas_artifact("lagging-canvas", ticker="PCTY"),
    41,
  )
  write_canvas_artifact(
    workspace_dir=workspace,
    artifact=artifact,
    source=SOURCE,
    bundle=BUNDLE,
  )
  delete_artifact_sidecar_index_rows(
    workspace_dir=workspace,
    user_id="alice",
    keys=(("canvas", "artifacts/_canvas/lagging-canvas.json"),),
  )

  purge_research_file_artifacts(
    workspace,
    user_id="alice",
    research_file_ids=(41,),
  )

  assert read_canvas_artifact_sidecar(workspace, "lagging-canvas") is None
  assert not list((workspace / "artifacts" / "_canvas").iterdir())


def test_ui_blocks_purge_uses_exact_anchor_with_and_without_index(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  directory = workspace / "artifacts" / "_ui_blocks"
  directory.mkdir(parents=True)
  indexed_id = "ub_1111111111111111"
  lagging_id = "ub_2222222222222222"
  for ui_blocks_id in (indexed_id, lagging_id):
    path = directory / f"{ui_blocks_id}.json"
    path.write_text(
      json.dumps({
        "ui_blocks_id": ui_blocks_id,
        "session_id": "session-1",
        "turn_key": "turn-1",
        "emission_index": 0,
        "contract_version": 1,
        "payload": {"blocks": []},
        "text_fallback": "private",
        "ts": 1.0,
      }),
      encoding="utf-8",
    )
  register_ui_blocks_payload_sidecar(
    workspace_dir=workspace,
    user_id="alice",
    ui_blocks_id=indexed_id,
    path=directory / f"{indexed_id}.json",
    session_id="session-1",
    turn_key="turn-1",
    emission_index=0,
    ts=1.0,
  )

  purge_ui_blocks_payloads(
    workspace,
    user_id="alice",
    ui_blocks_ids=(indexed_id, lagging_id),
  )

  assert not (directory / f"{indexed_id}.json").exists()
  assert not (directory / f"{lagging_id}.json").exists()
  assert get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="ui_blocks",
    artifact_id=indexed_id,
    user_id="alice",
  ) is None


def test_purge_validates_all_targets_before_effects_and_rejects_symlinks(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  canvas = _with_file_id(
    canvas_artifact("safe-until-validation", ticker="PCTY"),
    41,
  )
  write_canvas_artifact(
    workspace_dir=workspace,
    artifact=canvas,
    source=SOURCE,
    bundle=BUNDLE,
  )
  bad_sidecar = (
    workspace / "artifacts" / "PCTY" / "ir-composer" / "bad-letter.json"
  )
  bad_sidecar.parent.mkdir(parents=True, exist_ok=True)
  bad_sidecar.write_text(
    json.dumps({
      "artifact_id": "bad-letter",
      "ticker": "PCTY",
      "skill": "ir-composer",
      "research_file_id": 41,
      "binary_artifact_path": "../../outside.docx",
    }),
    encoding="utf-8",
  )

  with pytest.raises(ValueError, match="canonical DOCX"):
    purge_research_file_artifacts(
      workspace,
      user_id="alice",
      research_file_ids=(41,),
    )
  assert read_canvas_artifact_sidecar(
    workspace,
    "safe-until-validation",
  ) is not None

  outside = tmp_path / "outside.json"
  outside.write_text(json.dumps({"ui_blocks_id": "ub_3333333333333333"}), encoding="utf-8")
  ui_directory = workspace / "artifacts" / "_ui_blocks"
  ui_directory.mkdir(parents=True, exist_ok=True)
  (ui_directory / "ub_3333333333333333.json").symlink_to(outside)
  with pytest.raises(ValueError, match="unsafe"):
    purge_ui_blocks_payloads(
      workspace,
      user_id="alice",
      ui_blocks_ids=("ub_3333333333333333",),
    )
  assert outside.exists()


def test_purge_rejects_html_payload_hardlink_before_any_effects(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  target_canvas = _with_file_id(
    canvas_artifact("safe-before-hardlink", ticker="PCTY"),
    41,
  )
  write_canvas_artifact(
    workspace_dir=workspace,
    artifact=target_canvas,
    source=SOURCE,
    bundle=BUNDLE,
  )
  write_html_artifact(
    workspace_dir=workspace,
    artifact=_with_file_id(html_artifact("target-html", ticker="PCTY"), 41),
    html_content="<p>private</p>",
  )
  write_html_artifact(
    workspace_dir=workspace,
    artifact=_with_file_id(html_artifact("unrelated-html", ticker="PCTY"), 42),
    html_content="<p>unrelated</p>",
  )
  target_payload = workspace / "artifacts" / "_html" / "target-html.html"
  unrelated_payload = workspace / "artifacts" / "_html" / "unrelated-html.html"
  unrelated_payload.unlink()
  os.link(target_payload, unrelated_payload)

  with pytest.raises(ValueError, match="unsafe"):
    purge_research_file_artifacts(
      workspace,
      user_id="alice",
      research_file_ids=(41,),
    )

  assert read_canvas_artifact_sidecar(
    workspace,
    "safe-before-hardlink",
  ) is not None
  assert read_html_artifact_content(workspace, "target-html") == "<p>private</p>"
  assert read_html_artifact_content(workspace, "unrelated-html") == "<p>private</p>"


def test_purge_rejects_hidden_payload_hardlink(tmp_path: Path) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  write_html_artifact(
    workspace_dir=workspace,
    artifact=_with_file_id(html_artifact("target-html", ticker="PCTY"), 41),
    html_content="<p>private</p>",
  )
  target_payload = workspace / "artifacts" / "_html" / "target-html.html"
  hidden_payload = workspace / ".hidden-private-copy.html"
  os.link(target_payload, hidden_payload)

  with pytest.raises(ValueError, match="unsafe"):
    purge_research_file_artifacts(
      workspace,
      user_id="alice",
      research_file_ids=(41,),
    )

  assert target_payload.exists()
  assert hidden_payload.read_text(encoding="utf-8") == "<p>private</p>"


def test_purge_rejects_sidecar_hardlink_before_read(tmp_path: Path) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  write_html_artifact(
    workspace_dir=workspace,
    artifact=_with_file_id(html_artifact("target-html", ticker="PCTY"), 41),
    html_content="<p>private</p>",
  )
  target_sidecar = workspace / "artifacts" / "_html" / "target-html.json"
  hidden_sidecar = workspace / ".hidden-private-sidecar.json"
  os.link(target_sidecar, hidden_sidecar)

  with pytest.raises(ValueError, match="unsafe"):
    purge_research_file_artifacts(
      workspace,
      user_id="alice",
      research_file_ids=(41,),
    )

  assert target_sidecar.exists()
  assert hidden_sidecar.exists()
  assert read_html_artifact_content(workspace, "target-html") == "<p>private</p>"


def test_ancestor_swap_fails_before_delete_and_preserves_other_owner(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  bob_workspace = tmp_path / "users" / "bob" / "workspace"
  for target_workspace, content in (
    (workspace, "<p>alice</p>"),
    (bob_workspace, "<p>bob</p>"),
  ):
    write_html_artifact(
      workspace_dir=target_workspace,
      artifact=_with_file_id(html_artifact("same-id", ticker="PCTY"), 41),
      html_content=content,
    )

  original_open_target = erase_module._open_delete_target
  call_count = 0
  saved_artifacts = workspace / "artifacts-before-swap"

  def swap_before_delete(path: Path, *, workspace: Path, required: bool):
    nonlocal call_count
    call_count += 1
    if call_count == 2:
      (workspace / "artifacts").rename(saved_artifacts)
      (workspace / "artifacts").symlink_to(
        bob_workspace / "artifacts",
        target_is_directory=True,
      )
    return original_open_target(
      path,
      workspace=workspace,
      required=required,
    )

  monkeypatch.setattr(erase_module, "_open_delete_target", swap_before_delete)

  with pytest.raises(ValueError, match="unsafe"):
    purge_research_file_artifacts(
      workspace,
      user_id="alice",
      research_file_ids=(41,),
    )

  assert (saved_artifacts / "_html" / "same-id.html").read_text(
    encoding="utf-8"
  ) == "<p>alice</p>"
  assert read_html_artifact_content(bob_workspace, "same-id") == "<p>bob</p>"


def test_json_fdopen_failure_does_not_leak_descriptor(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  write_html_artifact(
    workspace_dir=workspace,
    artifact=_with_file_id(html_artifact("target-html", ticker="PCTY"), 41),
    html_content="<p>private</p>",
  )
  sidecar = workspace / "artifacts" / "_html" / "target-html.json"
  descriptor_count = _open_descriptor_count()

  def fail_fdopen(*_args: Any, **_kwargs: Any) -> None:
    raise OSError("injected fdopen failure")

  monkeypatch.setattr(erase_module.os, "fdopen", fail_fdopen)
  for _index in range(20):
    with pytest.raises(OSError, match="injected fdopen failure"):
      erase_module._read_json_object(sidecar, workspace=workspace)

  assert _open_descriptor_count() == descriptor_count
