from __future__ import annotations

import os
from pathlib import Path
import sys

from fastapi.testclient import TestClient

from agent_gateway.html_artifact_store import write_html_artifact
from schema.html_artifact import HtmlArtifact, StaticExports

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from test_artifact_api import ArtifactApiFixture, USER_ID, _signed_headers, artifact_api


def test_html_artifacts_list_returns_bare_array_newest_first_with_filters(
  artifact_api: ArtifactApiFixture,
) -> None:
  old_artifact = _artifact("old-artifact", ticker="PCTY", purpose="exploration")
  new_artifact = _artifact("new-artifact", ticker=None, purpose="report")
  other_artifact = _artifact("other-artifact", ticker="MSFT", purpose="report")

  for artifact in [old_artifact, new_artifact, other_artifact]:
    _write_html_artifact(artifact_api, artifact)

  _set_sidecar_mtime(artifact_api, old_artifact.artifact_id, 100)
  _set_sidecar_mtime(artifact_api, new_artifact.artifact_id, 300)
  _set_sidecar_mtime(artifact_api, other_artifact.artifact_id, 200)

  with TestClient(artifact_api.app) as client:
    response = client.get("/api/html-artifacts", headers=_signed_headers())
    ticker_response = client.get("/api/html-artifacts?ticker=PCTY", headers=_signed_headers())
    purpose_response = client.get("/api/html-artifacts?purpose=report", headers=_signed_headers())
    since_response = client.get("/api/html-artifacts?since=250", headers=_signed_headers())
    limit_response = client.get("/api/html-artifacts?limit=1", headers=_signed_headers())

  assert response.status_code == 200
  assert [item["artifact_id"] for item in response.json()] == [
    "new-artifact",
    "other-artifact",
    "old-artifact",
  ]
  assert response.json()[0] == {
    "artifact_id": "new-artifact",
    "title": "new-artifact title",
    "purpose": "report",
    "summary": "new-artifact summary",
    "ticker": None,
    "session_id": None,
    "source_skill": "historical-coincidences",
    "ts": "2026-06-01T12:00:00+00:00",
  }
  assert [item["artifact_id"] for item in ticker_response.json()] == ["old-artifact"]
  assert [item["artifact_id"] for item in purpose_response.json()] == ["new-artifact", "other-artifact"]
  assert [item["artifact_id"] for item in since_response.json()] == ["new-artifact"]
  assert [item["artifact_id"] for item in limit_response.json()] == ["new-artifact"]


def test_html_artifact_sidecar_and_content_endpoints(artifact_api: ArtifactApiFixture) -> None:
  artifact = _artifact("html-artifact-1", ticker="PCTY")
  _write_html_artifact(artifact_api, artifact, html="<section><h1>Risk bridge</h1></section>")

  with TestClient(artifact_api.app) as client:
    sidecar_response = client.get("/api/html-artifacts/html-artifact-1", headers=_signed_headers())
    content_response = client.get("/api/html-artifacts/html-artifact-1/content", headers=_signed_headers())

  assert sidecar_response.status_code == 200
  assert sidecar_response.json()["artifact_id"] == "html-artifact-1"
  assert sidecar_response.json()["contract_name"] == "HtmlArtifact"
  assert content_response.status_code == 200
  assert content_response.text == "<section><h1>Risk bridge</h1></section>"
  assert content_response.headers["content-type"] == "text/html; charset=utf-8"
  assert content_response.headers["content-disposition"] == "inline"
  assert "sandbox" in content_response.headers["content-security-policy"]
  assert content_response.headers["x-content-type-options"] == "nosniff"


def test_html_artifact_missing_sidecar_or_content_returns_404(
  artifact_api: ArtifactApiFixture,
) -> None:
  artifact = _artifact("sidecar-only", ticker=None)
  _write_html_artifact(artifact_api, artifact)
  (_workspace(artifact_api) / "artifacts" / "_html" / "sidecar-only.html").unlink()

  with TestClient(artifact_api.app) as client:
    missing_sidecar = client.get("/api/html-artifacts/missing", headers=_signed_headers())
    missing_content = client.get("/api/html-artifacts/sidecar-only/content", headers=_signed_headers())

  assert missing_sidecar.status_code == 404
  assert missing_content.status_code == 404


def test_html_artifact_auth_invalid_id_and_user_isolation(
  artifact_api: ArtifactApiFixture,
) -> None:
  _write_html_artifact(artifact_api, _artifact("bob-artifact", ticker="PCTY"), user_id="bob")

  with TestClient(artifact_api.app) as client:
    unsigned = client.get("/api/html-artifacts")
    invalid = client.get("/api/html-artifacts/bad.id", headers=_signed_headers())
    alice_missing = client.get("/api/html-artifacts/bob-artifact", headers=_signed_headers(user_id=USER_ID))
    bob_response = client.get(
      "/api/html-artifacts/bob-artifact",
      headers=_signed_headers(user_id="bob", user_email="bob@example.com"),
    )

  assert unsigned.status_code == 401
  assert invalid.status_code == 400
  assert alice_missing.status_code == 404
  assert bob_response.status_code == 200


def _write_html_artifact(
  fixture: ArtifactApiFixture,
  artifact: HtmlArtifact,
  *,
  html: str | None = None,
  user_id: str = USER_ID,
) -> None:
  write_html_artifact(
    workspace_dir=_workspace(fixture, user_id=user_id),
    artifact=artifact,
    html_content=html or f"<p>{artifact.artifact_id}</p>",
  )


def _artifact(
  artifact_id: str,
  *,
  ticker: str | None,
  purpose: str = "exploration",
) -> HtmlArtifact:
  return HtmlArtifact(
    artifact_id=artifact_id,
    title=f"{artifact_id} title",
    purpose=purpose,
    content_ref=f"{artifact_id}.html",
    summary=f"{artifact_id} summary",
    ticker=ticker,
    session_id=None,
    source_skill="historical-coincidences",
    sources=[],
    exports=StaticExports(
      copy_as_prompt="Prompt export",
      copy_as_markdown=None,
      copy_as_json={"artifact_id": artifact_id},
    ),
    ts="2026-06-01T12:00:00+00:00",
  )


def _set_sidecar_mtime(fixture: ArtifactApiFixture, artifact_id: str, mtime: int) -> None:
  path = _workspace(fixture) / "artifacts" / "_html" / f"{artifact_id}.json"
  os.utime(path, (mtime, mtime))


def _workspace(fixture: ArtifactApiFixture, *, user_id: str = USER_ID) -> Path:
  return fixture.data_dir / "users" / user_id / "workspace"
