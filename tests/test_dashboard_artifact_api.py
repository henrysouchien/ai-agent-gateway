from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from fastapi.testclient import TestClient

from agent_gateway.dashboard_artifact_store import write_dashboard_artifact
from schema.dashboard_artifact import DashboardArtifact

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from test_artifact_api import ArtifactApiFixture, USER_ID, _signed_headers, artifact_api


def test_dashboard_artifacts_list_returns_bare_array_newest_first_with_filters(
  artifact_api: ArtifactApiFixture,
) -> None:
  old_artifact = _artifact("old-artifact", ticker="PCTY")
  new_artifact = _artifact("new-artifact", ticker=None)
  other_artifact = _artifact("other-artifact", ticker="MSFT")

  for artifact in [old_artifact, new_artifact, other_artifact]:
    _write_dashboard_artifact(artifact_api, artifact)

  _set_sidecar_mtime(artifact_api, old_artifact.artifact_id, 100)
  _set_sidecar_mtime(artifact_api, new_artifact.artifact_id, 300)
  _set_sidecar_mtime(artifact_api, other_artifact.artifact_id, 200)

  with TestClient(artifact_api.app) as client:
    response = client.get("/api/dashboard-artifacts", headers=_signed_headers())
    ticker_response = client.get("/api/dashboard-artifacts?ticker=PCTY", headers=_signed_headers())
    since_response = client.get("/api/dashboard-artifacts?since=250", headers=_signed_headers())
    limit_response = client.get("/api/dashboard-artifacts?limit=1", headers=_signed_headers())

  assert response.status_code == 200
  assert [item["artifact_id"] for item in response.json()] == [
    "new-artifact",
    "other-artifact",
    "old-artifact",
  ]
  assert response.json()[0] == {
    "artifact_id": "new-artifact",
    "title": "new-artifact title",
    "summary": "new-artifact summary",
    "ticker": None,
    "scope_label": None,
    "source_skill": "fixture-dashboard-artifact",
    "readiness_posture": "decision_ready",
    "profile": "production",
    "ts": "2026-06-01T12:00:00+00:00",
  }
  assert [item["artifact_id"] for item in ticker_response.json()] == ["old-artifact"]
  assert [item["artifact_id"] for item in since_response.json()] == ["new-artifact"]
  assert [item["artifact_id"] for item in limit_response.json()] == ["new-artifact"]


def test_dashboard_artifact_sidecar_and_payload_endpoints(artifact_api: ArtifactApiFixture) -> None:
  artifact = _artifact("dashboard-artifact-1", ticker="PCTY")
  payload = _payload("Risk bridge")
  _write_dashboard_artifact(artifact_api, artifact, payload=payload)

  with TestClient(artifact_api.app) as client:
    sidecar_response = client.get("/api/dashboard-artifacts/dashboard-artifact-1", headers=_signed_headers())
    payload_response = client.get(
      "/api/dashboard-artifacts/dashboard-artifact-1/payload",
      headers=_signed_headers(),
    )

  assert sidecar_response.status_code == 200
  assert sidecar_response.json()["artifact_id"] == "dashboard-artifact-1"
  assert sidecar_response.json()["contract_name"] == "DashboardArtifact"
  assert payload_response.status_code == 200
  assert payload_response.json() == payload
  assert payload_response.headers["content-type"] == "application/json"
  assert payload_response.headers["x-content-type-options"] == "nosniff"
  assert "content-security-policy" not in payload_response.headers


def test_dashboard_artifact_missing_sidecar_or_payload_returns_404(
  artifact_api: ArtifactApiFixture,
) -> None:
  artifact = _artifact("sidecar-only", ticker=None)
  _write_dashboard_artifact(artifact_api, artifact)
  (_workspace(artifact_api) / "artifacts" / "_dashboards" / "sidecar-only.payload.json").unlink()

  with TestClient(artifact_api.app) as client:
    missing_sidecar = client.get("/api/dashboard-artifacts/missing", headers=_signed_headers())
    missing_payload = client.get("/api/dashboard-artifacts/sidecar-only/payload", headers=_signed_headers())

  assert missing_sidecar.status_code == 404
  assert missing_payload.status_code == 404


def test_dashboard_artifact_auth_invalid_id_and_user_isolation(
  artifact_api: ArtifactApiFixture,
) -> None:
  _write_dashboard_artifact(artifact_api, _artifact("bob-artifact", ticker="PCTY"), user_id="bob")

  with TestClient(artifact_api.app) as client:
    unsigned = client.get("/api/dashboard-artifacts")
    invalid = client.get("/api/dashboard-artifacts/bad.id", headers=_signed_headers())
    alice_missing = client.get("/api/dashboard-artifacts/bob-artifact", headers=_signed_headers(user_id=USER_ID))
    bob_response = client.get(
      "/api/dashboard-artifacts/bob-artifact",
      headers=_signed_headers(user_id="bob", user_email="bob@example.com"),
    )

  assert unsigned.status_code == 401
  assert invalid.status_code == 400
  assert alice_missing.status_code == 404
  assert bob_response.status_code == 200


def _write_dashboard_artifact(
  fixture: ArtifactApiFixture,
  artifact: DashboardArtifact,
  *,
  payload: dict[str, Any] | None = None,
  user_id: str = USER_ID,
) -> None:
  write_dashboard_artifact(
    workspace_dir=_workspace(fixture, user_id=user_id),
    artifact=artifact,
    payload_json=payload or _payload(artifact.artifact_id),
  )


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


def _set_sidecar_mtime(fixture: ArtifactApiFixture, artifact_id: str, mtime: int) -> None:
  path = _workspace(fixture) / "artifacts" / "_dashboards" / f"{artifact_id}.json"
  os.utime(path, (mtime, mtime))


def _workspace(fixture: ArtifactApiFixture, *, user_id: str = USER_ID) -> Path:
  return fixture.data_dir / "users" / user_id / "workspace"
