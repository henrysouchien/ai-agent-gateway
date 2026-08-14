from __future__ import annotations

import logging

# ruff: noqa: F811

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

from fastapi.testclient import TestClient

from agent_gateway.canvas_artifact_store import write_canvas_artifact
from agent_gateway.control_plane.canvas_artifacts import (
  CANVAS_RENDER_FAILURE_REPORTS_PER_RENDER,
  CANVAS_RENDER_FAILURE_REQUEST_MAX_BYTES,
  CANVAS_RENDER_FAILURE_STORED_CAP,
  _append_render_failure,
)
from schema.canvas_artifact import CanvasArtifact, StaticExports

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from test_artifact_api import ArtifactApiFixture, USER_ID, _signed_headers, artifact_api  # noqa: F401


SOURCE = "export default function Artifact() { return <div>PCTY</div>; }\n"
BUNDLE = b"(() => { globalThis.canvasArtifact = 'PCTY'; })();\n"


def test_canvas_artifacts_list_sidecar_source_and_bundle_routes(artifact_api: ArtifactApiFixture) -> None:
  old = _artifact("old", ticker="PCTY", purpose="exploration")
  new = _artifact("new", ticker=None, purpose="comparison")
  for artifact in (old, new):
    _write_canvas_artifact(artifact_api, artifact)
  _set_sidecar_mtime(artifact_api, "old", 100)
  _set_sidecar_mtime(artifact_api, "new", 300)

  with TestClient(artifact_api.app) as client:
    response = client.get("/api/canvas-artifacts", headers=_signed_headers())
    ticker_response = client.get("/api/canvas-artifacts?ticker=PCTY", headers=_signed_headers())
    purpose_response = client.get("/api/canvas-artifacts?purpose=comparison", headers=_signed_headers())
    sidecar = client.get("/api/canvas-artifacts/old", headers=_signed_headers())
    source = client.get("/api/canvas-artifacts/old/source", headers=_signed_headers())
    bundle = client.get(
      f"/api/canvas-artifacts/old/bundle/{old.bundle_digest}.js",
      headers=_signed_headers(),
    )

  assert response.status_code == 200
  assert [item["artifact_id"] for item in response.json()] == ["new", "old"]
  assert [item["artifact_id"] for item in ticker_response.json()] == ["old"]
  assert [item["artifact_id"] for item in purpose_response.json()] == ["new"]
  assert sidecar.status_code == 200
  assert sidecar.json()["contract_name"] == "CanvasArtifact"
  assert source.status_code == 200
  assert source.text == SOURCE
  assert bundle.status_code == 200
  assert bundle.content == BUNDLE
  assert bundle.headers["content-type"] == "application/javascript"
  assert bundle.headers["x-content-type-options"] == "nosniff"
  assert bundle.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_canvas_bundle_route_refuses_wrong_digest_and_tampered_bytes(
  artifact_api: ArtifactApiFixture,
) -> None:
  artifact = _artifact("tampered", ticker="PCTY")
  _write_canvas_artifact(artifact_api, artifact)
  bundle_path = _canvas_dir(artifact_api) / "tampered.bundle.js"
  bundle_path.write_bytes(b"tampered bundle")

  with TestClient(artifact_api.app) as client:
    wrong = client.get(
      f"/api/canvas-artifacts/tampered/bundle/{'f' * 64}.js",
      headers=_signed_headers(),
    )
    uppercase = client.get(
      f"/api/canvas-artifacts/tampered/bundle/{artifact.bundle_digest.upper()}.js",
      headers=_signed_headers(),
    )
    tampered = client.get(
      f"/api/canvas-artifacts/tampered/bundle/{artifact.bundle_digest}.js",
      headers=_signed_headers(),
    )
    no_undigested_route = client.get(
      "/api/canvas-artifacts/tampered/bundle",
      headers=_signed_headers(),
    )

  assert wrong.status_code == 404
  assert uppercase.status_code == 404
  assert tampered.status_code == 404
  assert no_undigested_route.status_code == 404


def test_canvas_artifact_auth_traversal_and_user_isolation(artifact_api: ArtifactApiFixture) -> None:
  _write_canvas_artifact(artifact_api, _artifact("bob-only", ticker="PCTY"), user_id="bob")
  with TestClient(artifact_api.app) as client:
    unsigned = client.get("/api/canvas-artifacts")
    invalid = client.get("/api/canvas-artifacts/bad.id", headers=_signed_headers())
    alice_missing = client.get("/api/canvas-artifacts/bob-only", headers=_signed_headers())
    bob = client.get(
      "/api/canvas-artifacts/bob-only",
      headers=_signed_headers(user_id="bob", user_email="bob@example.com"),
    )
  assert unsigned.status_code == 401
  assert invalid.status_code == 400
  assert alice_missing.status_code == 404
  assert bob.status_code == 200


def test_render_failure_rejects_malformed_and_overlength_without_write(
  artifact_api: ArtifactApiFixture,
) -> None:
  artifact = _artifact("failure-target", ticker="PCTY")
  _write_canvas_artifact(artifact_api, artifact)
  report_path = _canvas_dir(artifact_api) / "failure-target.render_failures.json"
  with TestClient(artifact_api.app) as client:
    malformed = client.post(
      "/api/canvas-artifacts/failure-target/render-failures",
      headers={**_signed_headers(), "Content-Type": "application/json"},
      content=b"{not json",
    )
    over_message = client.post(
      "/api/canvas-artifacts/failure-target/render-failures",
      headers=_signed_headers(),
      json=_report("failure-target", nonce="render-a", message="x" * 2001),
    )
    wrong_artifact = client.post(
      "/api/canvas-artifacts/failure-target/render-failures",
      headers=_signed_headers(),
      json=_report("different", nonce="render-a"),
    )
  assert malformed.status_code == 400
  assert over_message.status_code == 400
  assert wrong_artifact.status_code == 400
  assert not report_path.exists()


def test_render_failure_rejection_is_logged_with_offending_fields(
  artifact_api: ArtifactApiFixture,
  caplog,
) -> None:
  """A rejected report must leave a server-side record.

  Renderers swallow the 400 client-side (console.warn), so a schema drift between the
  connectors CanvasRenderFailure type and this model silently discards every render
  failure. Found live 2026-07-22: the connectors type omitted nonce + artifact_id, so
  every report 400'd and nothing was recorded on either side.
  """
  artifact = _artifact("log-target", ticker="PCTY")
  _write_canvas_artifact(artifact_api, artifact)
  with TestClient(artifact_api.app) as client:
    with caplog.at_level(logging.WARNING, logger="agent_gateway.control_plane.canvas_artifacts"):
      missing_fields = client.post(
        "/api/canvas-artifacts/log-target/render-failures",
        headers=_signed_headers(),
        json={"kind": "canvas_render_error", "message": "boom", "component_stack": "at Canvas"},
      )
  assert missing_fields.status_code == 400
  rejected = [record.getMessage() for record in caplog.records if "rejected" in record.getMessage()]
  assert rejected, "rejection must be logged server-side"
  assert "artifact_id=log-target" in rejected[0]
  # the offending fields are named, so drift is diagnosable...
  assert "nonce" in rejected[0]
  # ...but the untrusted body is never echoed
  assert "boom" not in rejected[0]
  assert "at Canvas" not in rejected[0]


def test_render_failure_unknown_length_oversize_is_rejected_at_raw_cap(
  artifact_api: ArtifactApiFixture,
) -> None:
  artifact = _artifact("oversize", ticker=None)
  _write_canvas_artifact(artifact_api, artifact)
  report_path = _canvas_dir(artifact_api) / "oversize.render_failures.json"

  def chunks():
    yield b"{" + b" " * 16_000
    yield b" " * (CANVAS_RENDER_FAILURE_REQUEST_MAX_BYTES + 1)

  with TestClient(artifact_api.app) as client:
    response = client.post(
      "/api/canvas-artifacts/oversize/render-failures",
      headers={**_signed_headers(), "Content-Type": "application/json"},
      content=chunks(),
    )
  assert response.status_code == 413
  assert not report_path.exists()


def test_render_failure_two_concurrent_writers_lose_no_report(
  artifact_api: ArtifactApiFixture,
) -> None:
  artifact = _artifact("concurrent", ticker=None)
  _write_canvas_artifact(artifact_api, artifact)
  workspace = _workspace(artifact_api)

  with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
      executor.submit(_append_render_failure, workspace, "concurrent", _report("concurrent", nonce=f"render-{i}"))
      for i in range(2)
    ]
    for future in futures:
      future.result()

  reports = _read_reports(artifact_api, "concurrent")
  assert len(reports) == 2
  assert {report["nonce"] for report in reports} == {"render-0", "render-1"}


def test_render_failure_stale_pre_replace_inode_cannot_clobber_new_log(
  artifact_api: ArtifactApiFixture,
) -> None:
  artifact = _artifact("stale-inode", ticker=None)
  _write_canvas_artifact(artifact_api, artifact)
  workspace = _workspace(artifact_api)
  report_path = _canvas_dir(artifact_api) / "stale-inode.render_failures.json"
  _append_render_failure(workspace, "stale-inode", _report("stale-inode", nonce="render-0"))

  with report_path.open("r+", encoding="utf-8") as stale_handle:
    _append_render_failure(workspace, "stale-inode", _report("stale-inode", nonce="render-1"))
    stale_handle.seek(0)
    stale_handle.truncate()
    stale_handle.write("[]")
    stale_handle.flush()
    os.fsync(stale_handle.fileno())

  _append_render_failure(workspace, "stale-inode", _report("stale-inode", nonce="render-2"))
  assert [report["nonce"] for report in _read_reports(artifact_api, "stale-inode")] == [
    "render-0",
    "render-1",
    "render-2",
  ]


def test_render_failure_caps_stored_reports_and_reports_per_render(
  artifact_api: ArtifactApiFixture,
) -> None:
  artifact = _artifact("caps", ticker=None)
  _write_canvas_artifact(artifact_api, artifact)
  workspace = _workspace(artifact_api)
  assert CANVAS_RENDER_FAILURE_REPORTS_PER_RENDER == 3
  assert CANVAS_RENDER_FAILURE_STORED_CAP == 20

  for index in range(4):
    _append_render_failure(
      workspace,
      "caps",
      _report("caps", nonce="same-render", message=f"same-{index}"),
    )
  assert [report["message"] for report in _read_reports(artifact_api, "caps")] == [
    "same-0",
    "same-1",
    "same-2",
  ]

  for index in range(25):
    _append_render_failure(
      workspace,
      "caps",
      _report("caps", nonce=f"render-{index}", message=f"report-{index}"),
    )
  reports = _read_reports(artifact_api, "caps")
  assert len(reports) == 20
  assert [report["message"] for report in reports] == [f"report-{index}" for index in range(5, 25)]


def test_render_failure_valid_post_is_persisted_and_user_isolated(
  artifact_api: ArtifactApiFixture,
) -> None:
  artifact = _artifact("post-target", ticker=None)
  _write_canvas_artifact(artifact_api, artifact, user_id="bob")
  with TestClient(artifact_api.app) as client:
    alice = client.post(
      "/api/canvas-artifacts/post-target/render-failures",
      headers=_signed_headers(),
      json=_report("post-target", nonce="alice-render"),
    )
    bob = client.post(
      "/api/canvas-artifacts/post-target/render-failures",
      headers=_signed_headers(user_id="bob", user_email="bob@example.com"),
      json=_report("post-target", nonce="bob-render"),
    )
  assert alice.status_code == 404
  assert bob.status_code == 204
  reports = json.loads(
    (_canvas_dir(artifact_api, user_id="bob") / "post-target.render_failures.json").read_text(encoding="utf-8")
  )
  assert reports == [_report("post-target", nonce="bob-render")]


def _write_canvas_artifact(
  fixture: ArtifactApiFixture,
  artifact: CanvasArtifact,
  *,
  user_id: str = USER_ID,
) -> None:
  write_canvas_artifact(
    workspace_dir=_workspace(fixture, user_id=user_id),
    artifact=artifact,
    source=SOURCE,
    bundle=BUNDLE,
  )


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


def _report(
  artifact_id: str,
  *,
  nonce: str,
  message: str = "Component failed",
) -> dict[str, str]:
  return {
    "kind": "canvas_render_error",
    "nonce": nonce,
    "artifact_id": artifact_id,
    "message": message,
    "component_stack": "at ScenarioGrid",
  }


def _read_reports(fixture: ArtifactApiFixture, artifact_id: str) -> list[dict[str, str]]:
  return json.loads((_canvas_dir(fixture) / f"{artifact_id}.render_failures.json").read_text(encoding="utf-8"))


def _set_sidecar_mtime(fixture: ArtifactApiFixture, artifact_id: str, mtime: int) -> None:
  os.utime(_canvas_dir(fixture) / f"{artifact_id}.json", (mtime, mtime))


def _canvas_dir(fixture: ArtifactApiFixture, *, user_id: str = USER_ID) -> Path:
  return _workspace(fixture, user_id=user_id) / "artifacts" / "_canvas"


def _workspace(fixture: ArtifactApiFixture, *, user_id: str = USER_ID) -> Path:
  return fixture.data_dir / "users" / user_id / "workspace"
