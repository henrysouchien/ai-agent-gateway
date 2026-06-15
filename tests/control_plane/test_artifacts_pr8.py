from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


API_KEY = "artifacts-pr8-key"
HMAC_KEY = "artifact-api-test-secret"
USER_ID = "alice"
USER_EMAIL = "alice@example.com"
CLAIM_HEADERS = {
  "audience": "X-Agent-Claim-Audience",
  "issued_at": "X-Agent-Claim-Issued-At",
  "expiry": "X-Agent-Claim-Expiry",
  "user_id": "X-Agent-Claim-User-Id",
  "user_email": "X-Agent-Claim-User-Email",
  "nonce": "X-Agent-Claim-Nonce",
  "signature": "X-Agent-Claim-Signature",
}


@dataclass(frozen=True)
class ArtifactPr8Fixture:
  app: Any
  data_dir: Path


@dataclass(frozen=True)
class _FakeAutonomousTask:
  user_id: str
  control_run_id: str
  event_lines: list[dict[str, Any]]
  started_at: float | None = None


class _FakeAutonomousRegistry:
  def __init__(self, tasks: list[_FakeAutonomousTask]) -> None:
    self._tasks = {task.control_run_id: task for task in tasks}

  async def shutdown(self) -> None:
    return None


@pytest.fixture
def artifact_pr8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ArtifactPr8Fixture:
  data_dir = tmp_path / "data"
  monkeypatch.setenv("USER_DATA_DIR", str(data_dir))
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  monkeypatch.setenv("AGENT_API_CLAIM_MAX_TTL_SECONDS", "600")

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="artifacts-pr8-test-secret-0123456789",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )
  return ArtifactPr8Fixture(app=app, data_dir=data_dir)


def test_control_artifacts_lists_recent_50_sorted_and_scoped(artifact_pr8: ArtifactPr8Fixture) -> None:
  artifact_ids: list[str] = []
  for index in range(55):
    artifact_id = f"2026-05-20T120000.{index:03d}-run-{index:02d}"
    artifact_ids.append(artifact_id)
    _write_artifact(
      artifact_pr8.data_dir,
      USER_ID,
      "PCTY",
      "earnings-scenarios",
      artifact_id,
      mtime=1_800_000_000 + index,
      include_artifact_path=index != 54,
    )
  bob_id = "2026-05-20T130000.000-run-bob"
  _write_artifact(
    artifact_pr8.data_dir,
    "bob",
    "PCTY",
    "earnings-scenarios",
    bob_id,
    mtime=1_900_000_000,
  )

  with TestClient(artifact_pr8.app) as client:
    response = client.get("/api/control/artifacts", headers=_bearer_headers(client, USER_ID))

  assert response.status_code == 200
  artifacts = response.json()["artifacts"]
  assert len(artifacts) == 50
  assert [artifact["artifact_id"] for artifact in artifacts] == list(reversed(artifact_ids[5:]))
  assert all(artifact["ticker"] == "PCTY" for artifact in artifacts)
  assert all(artifact["skill"] == "earnings-scenarios" for artifact in artifacts)
  assert bob_id not in {artifact["artifact_id"] for artifact in artifacts}
  assert artifacts[0]["artifact_path"] == f"artifacts/PCTY/earnings-scenarios/{artifact_ids[-1]}.json"
  assert set(artifacts[0]) == {
    "ticker",
    "skill",
    "artifact_id",
    "artifact_path",
    "binary_artifact_path",
    "contract_name",
    "data_source",
    "created_at",
    "skill_run_id",
    "run_id",
  }
  assert artifacts[0]["run_id"] is None


def test_control_artifacts_list_includes_html_artifact_sidecars(
  artifact_pr8: ArtifactPr8Fixture,
) -> None:
  json_artifact_id = "2026-05-20T120000.000-run-json"
  html_artifact_id = "20260605T142435-cabbb60fece34f28"
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    json_artifact_id,
    mtime=1_800_000_001,
  )
  _write_html_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    html_artifact_id,
    mtime=1_800_000_002,
  )

  with TestClient(artifact_pr8.app) as client:
    headers = _bearer_headers(client, USER_ID)
    response = client.get("/api/control/artifacts", headers=headers)
    skill_response = client.get("/api/control/artifacts?skill=fixture-html-artifact", headers=headers)

  assert response.status_code == 200
  artifacts = response.json()["artifacts"]
  assert [artifact["artifact_id"] for artifact in artifacts] == [html_artifact_id, json_artifact_id]
  html_artifact = artifacts[0]
  assert html_artifact == {
    "ticker": "PCTY",
    "skill": "fixture-html-artifact",
    "artifact_id": html_artifact_id,
    "artifact_path": f"artifacts/_html/{html_artifact_id}.json",
    "binary_artifact_path": f"artifacts/_html/{html_artifact_id}.html",
    "contract_name": "HtmlArtifact",
    "data_source": "live",
    "created_at": "2026-06-05T14:24:35.683368+00:00",
    "skill_run_id": "fixture-session",
    "run_id": "fixture-session",
  }
  assert skill_response.status_code == 200
  assert [artifact["artifact_id"] for artifact in skill_response.json()["artifacts"]] == [html_artifact_id]


def test_control_artifacts_list_includes_dashboard_artifact_sidecars(
  artifact_pr8: ArtifactPr8Fixture,
) -> None:
  json_artifact_id = "2026-05-20T120000.000-run-json"
  dashboard_artifact_id = "20260605T142435-cabbb60fece34f28"
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    json_artifact_id,
    mtime=1_800_000_001,
  )
  _write_dashboard_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    dashboard_artifact_id,
    mtime=1_800_000_002,
  )

  with TestClient(artifact_pr8.app) as client:
    headers = _bearer_headers(client, USER_ID)
    response = client.get("/api/control/artifacts", headers=headers)
    skill_response = client.get("/api/control/artifacts?skill=fixture-dashboard-artifact", headers=headers)

  assert response.status_code == 200
  artifacts = response.json()["artifacts"]
  assert [artifact["artifact_id"] for artifact in artifacts] == [dashboard_artifact_id, json_artifact_id]
  dashboard_artifact = artifacts[0]
  assert dashboard_artifact == {
    "ticker": "PCTY",
    "skill": "fixture-dashboard-artifact",
    "artifact_id": dashboard_artifact_id,
    "artifact_path": f"artifacts/_dashboards/{dashboard_artifact_id}.json",
    "binary_artifact_path": f"artifacts/_dashboards/{dashboard_artifact_id}.payload.json",
    "contract_name": "DashboardArtifact",
    "data_source": "live",
    "created_at": "2026-06-05T14:24:35.683368+00:00",
    "skill_run_id": dashboard_artifact_id,
    "run_id": None,
  }
  assert skill_response.status_code == 200
  assert [artifact["artifact_id"] for artifact in skill_response.json()["artifacts"]] == [dashboard_artifact_id]


def test_control_artifacts_filters_by_control_run_id_from_autonomous_events(
  artifact_pr8: ArtifactPr8Fixture,
) -> None:
  json_artifact_id = "2026-05-20T120000.000-run-json"
  html_artifact_id = "20260605T142435-cabbb60fece34f28"
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    json_artifact_id,
    mtime=1_800_000_001,
  )
  _write_html_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    html_artifact_id,
    mtime=1_800_000_002,
    session_id=None,
  )

  artifact_pr8.app.state.subprocess_registry = _FakeAutonomousRegistry([
    _FakeAutonomousTask(
      user_id=USER_ID,
      control_run_id="bg_1",
      event_lines=[
        {
          "type": "artifact_ready",
          "artifact_id": json_artifact_id,
          "artifact_path": f"artifacts/PCTY/earnings-scenarios/{json_artifact_id}.json",
          "skill_run_id": "skill-json",
        },
        {
          "type": "artifact_ready",
          "artifact_id": html_artifact_id,
          "artifact_path": f"artifacts/_html/{html_artifact_id}.json",
          "skill_run_id": "skill-html",
        },
      ],
    ),
    _FakeAutonomousTask(
      user_id="bob",
      control_run_id="bg_bob",
      event_lines=[
        {
          "type": "artifact_ready",
          "artifact_id": html_artifact_id,
          "artifact_path": f"artifacts/_html/{html_artifact_id}.json",
          "skill_run_id": "skill-bob",
        },
      ],
    ),
  ])

  with TestClient(artifact_pr8.app) as client:
    headers = _bearer_headers(client, USER_ID)
    run_response = client.get("/api/control/artifacts?run_id=bg_1", headers=headers)
    other_run_response = client.get("/api/control/artifacts?run_id=bg_missing", headers=headers)

  assert run_response.status_code == 200
  artifacts = run_response.json()["artifacts"]
  assert [artifact["artifact_id"] for artifact in artifacts] == [html_artifact_id, json_artifact_id]
  assert all(artifact["run_id"] == "bg_1" for artifact in artifacts)
  assert [artifact["skill_run_id"] for artifact in artifacts] == ["skill-html", "skill-json"]
  assert other_run_response.status_code == 200
  assert other_run_response.json()["artifacts"] == []


def test_control_artifacts_prefers_registry_run_context_over_sidecar_context(
  artifact_pr8: ArtifactPr8Fixture,
) -> None:
  json_artifact_id = "2026-05-20T120000.000-run-json"
  html_artifact_id = "20260605T142435-cabbb60fece34f28"
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    json_artifact_id,
    mtime=1_800_000_001,
    run_id="bg_stale",
  )
  _write_html_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    html_artifact_id,
    mtime=1_800_000_002,
    session_id="bg_stale",
  )
  artifact_pr8.app.state.subprocess_registry = _FakeAutonomousRegistry([
    _FakeAutonomousTask(
      user_id=USER_ID,
      control_run_id="bg_1",
      event_lines=[
        {
          "type": "artifact_ready",
          "artifact_id": json_artifact_id,
          "artifact_path": f"artifacts/PCTY/earnings-scenarios/{json_artifact_id}.json",
          "skill_run_id": "skill-json",
        },
        {
          "type": "artifact_ready",
          "artifact_id": html_artifact_id,
          "artifact_path": f"artifacts/_html/{html_artifact_id}.json",
          "skill_run_id": "skill-html",
        },
      ],
    ),
  ])

  with TestClient(artifact_pr8.app) as client:
    headers = _bearer_headers(client, USER_ID)
    run_response = client.get("/api/control/artifacts?run_id=bg_1", headers=headers)
    stale_response = client.get("/api/control/artifacts?run_id=bg_stale", headers=headers)

  assert run_response.status_code == 200
  artifacts = run_response.json()["artifacts"]
  assert [artifact["artifact_id"] for artifact in artifacts] == [html_artifact_id, json_artifact_id]
  assert all(artifact["run_id"] == "bg_1" for artifact in artifacts)
  assert [artifact["skill_run_id"] for artifact in artifacts] == ["skill-html", "skill-json"]
  assert stale_response.status_code == 200
  assert stale_response.json()["artifacts"] == []


def test_control_artifacts_run_filter_rejects_stale_reused_run_id_sidecars(
  artifact_pr8: ArtifactPr8Fixture,
) -> None:
  fresh_id = "20260605T142435-cabbb60fece34f28"
  stale_id = "20260605T135000-cabbb60fece34f28"
  _write_html_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    fresh_id,
    mtime=1_800_000_002,
    session_id="bg_1",
    ts="2026-06-05T14:24:35.683368+00:00",
  )
  _write_html_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    stale_id,
    mtime=1_800_000_001,
    session_id="bg_1",
    ts="2026-06-05T13:50:00+00:00",
  )
  artifact_pr8.app.state.subprocess_registry = _FakeAutonomousRegistry([
    _FakeAutonomousTask(
      user_id=USER_ID,
      control_run_id="bg_1",
      event_lines=[],
      started_at=datetime.fromisoformat("2026-06-05T14:00:00+00:00").timestamp(),
    ),
  ])

  with TestClient(artifact_pr8.app) as client:
    response = client.get("/api/control/artifacts?run_id=bg_1", headers=_bearer_headers(client, USER_ID))

  assert response.status_code == 200
  assert [artifact["artifact_id"] for artifact in response.json()["artifacts"]] == [fresh_id]


def test_control_artifacts_run_filter_rejects_artifacts_before_reused_run_start(
  artifact_pr8: ArtifactPr8Fixture,
) -> None:
  html_artifact_id = "20260605T142435-cabbb60fece34f28"
  _write_html_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    html_artifact_id,
    mtime=1_800_000_002,
    session_id="bg_reused",
  )

  artifact_pr8.app.state.subprocess_registry = _FakeAutonomousRegistry([
    _FakeAutonomousTask(
      user_id=USER_ID,
      control_run_id="bg_reused",
      started_at=1_900_000_000,
      event_lines=[],
    ),
  ])

  with TestClient(artifact_pr8.app) as client:
    headers = _bearer_headers(client, USER_ID)
    run_response = client.get("/api/control/artifacts?run_id=bg_reused", headers=headers)
    global_response = client.get("/api/control/artifacts", headers=headers)

  assert run_response.status_code == 200
  assert run_response.json()["artifacts"] == []
  assert global_response.status_code == 200
  assert [artifact["artifact_id"] for artifact in global_response.json()["artifacts"]] == [html_artifact_id]


def test_control_artifacts_filters_by_ticker_and_skill(artifact_pr8: ArtifactPr8Fixture) -> None:
  pcty_earnings = "2026-05-20T120000.000-run-pcty-earnings"
  msft_earnings = "2026-05-20T121000.000-run-msft-earnings"
  pcty_model = "2026-05-20T122000.000-run-pcty-model"
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    pcty_earnings,
    mtime=1_800_000_001,
  )
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "MSFT",
    "earnings-scenarios",
    msft_earnings,
    mtime=1_800_000_002,
  )
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "model-review",
    pcty_model,
    mtime=1_800_000_003,
  )

  with TestClient(artifact_pr8.app) as client:
    headers = _bearer_headers(client, USER_ID)
    ticker_response = client.get("/api/control/artifacts?ticker=PCTY", headers=headers)
    skill_response = client.get("/api/control/artifacts?skill=earnings-scenarios", headers=headers)
    combined_response = client.get(
      "/api/control/artifacts?ticker=PCTY&skill=earnings-scenarios",
      headers=headers,
    )

  assert ticker_response.status_code == 200
  assert [artifact["artifact_id"] for artifact in ticker_response.json()["artifacts"]] == [
    pcty_model,
    pcty_earnings,
  ]
  assert all(artifact["ticker"] == "PCTY" for artifact in ticker_response.json()["artifacts"])

  assert skill_response.status_code == 200
  assert [artifact["artifact_id"] for artifact in skill_response.json()["artifacts"]] == [
    msft_earnings,
    pcty_earnings,
  ]
  assert all(artifact["skill"] == "earnings-scenarios" for artifact in skill_response.json()["artifacts"])

  assert combined_response.status_code == 200
  assert [artifact["artifact_id"] for artifact in combined_response.json()["artifacts"]] == [pcty_earnings]


def test_control_artifacts_read_latest_and_specific_sidecar(artifact_pr8: ArtifactPr8Fixture) -> None:
  old_id = "2026-05-20T120000.000-run-old"
  latest_id = "2026-05-20T130000.000-run-latest"
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    old_id,
    mtime=1_800_000_001,
  )
  latest_payload = _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    latest_id,
    mtime=1_800_000_002,
  )

  with TestClient(artifact_pr8.app) as client:
    headers = _bearer_headers(client, USER_ID)
    latest_response = client.get("/api/control/artifacts/PCTY/earnings-scenarios/latest", headers=headers)
    specific_response = client.get(
      f"/api/control/artifacts/PCTY/earnings-scenarios/{old_id}",
      headers=headers,
    )
    missing_response = client.get(
      "/api/control/artifacts/PCTY/earnings-scenarios/2026-05-20T140000.000-run-missing",
      headers=headers,
    )

  assert latest_response.status_code == 200
  assert latest_response.json() == latest_payload
  assert latest_response.headers["etag"]
  assert latest_response.headers["cache-control"] == "private, max-age=0"

  assert specific_response.status_code == 200
  assert specific_response.json()["artifact_id"] == old_id

  assert missing_response.status_code == 404


@pytest.mark.parametrize("auth_mode", ["bearer", "signed"])
def test_artifact_and_letter_endpoints_accept_bearer_and_signed_claim(
  artifact_pr8: ArtifactPr8Fixture,
  auth_mode: str,
) -> None:
  artifact_id = "2026-05-20T120000.000-run-a"
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    artifact_id,
    mtime=1_800_000_000,
  )
  _write_letter(artifact_pr8.data_dir, USER_ID, "PCTY", artifact_id, b"docx bytes")

  with TestClient(artifact_pr8.app) as client:
    headers = _headers_for_mode(client, auth_mode, USER_ID)
    for path, expected_status in _artifact_endpoint_cases(artifact_id):
      response = client.get(path, headers=headers)
      assert response.status_code == expected_status, f"{auth_mode} {path}: {response.text}"


@pytest.mark.parametrize(
  "path",
  [
    "/api/artifacts/PCTY/earnings-scenarios/latest",
    "/api/artifacts/PCTY/earnings-scenarios/2026-05-20T120000.000-run-a",
    "/api/artifacts/PCTY",
    "/api/letters/PCTY/2026-05-20T120000.000-run-a",
    "/api/artifacts/PCTY/earnings-scenarios/2026-05-20T120000.000-run-a/extra",
    "/api/letters/PCTY/2026-05-20T120000.000-run-a/extra",
    "/api/control/artifacts",
    "/api/control/artifacts/PCTY/earnings-scenarios/latest",
    "/api/control/artifacts/PCTY/earnings-scenarios/2026-05-20T120000.000-run-a",
  ],
)
def test_invalid_bearer_never_falls_back_to_signed_claim(
  artifact_pr8: ArtifactPr8Fixture,
  path: str,
) -> None:
  artifact_id = "2026-05-20T120000.000-run-a"
  _write_artifact(
    artifact_pr8.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    artifact_id,
    mtime=1_800_000_000,
  )
  _write_letter(artifact_pr8.data_dir, USER_ID, "PCTY", artifact_id, b"docx bytes")
  headers = _signed_headers(user_id=USER_ID)
  headers["Authorization"] = "Bearer not-a-valid-jwt"

  with TestClient(artifact_pr8.app) as client:
    response = client.get(path, headers=headers)

  assert response.status_code == 401


def _artifact_endpoint_cases(artifact_id: str) -> list[tuple[str, int]]:
  return [
    ("/api/artifacts/PCTY/earnings-scenarios/latest", 200),
    (f"/api/artifacts/PCTY/earnings-scenarios/{artifact_id}", 200),
    ("/api/artifacts/PCTY", 200),
    (f"/api/letters/PCTY/{artifact_id}", 200),
    (f"/api/artifacts/PCTY/earnings-scenarios/{artifact_id}/extra", 404),
    (f"/api/letters/PCTY/{artifact_id}/extra", 404),
    ("/api/control/artifacts", 200),
    ("/api/control/artifacts/PCTY/earnings-scenarios/latest", 200),
    (f"/api/control/artifacts/PCTY/earnings-scenarios/{artifact_id}", 200),
  ]


def _headers_for_mode(client: TestClient, auth_mode: str, user_id: str) -> dict[str, str]:
  if auth_mode == "bearer":
    return _bearer_headers(client, user_id)
  return _signed_headers(user_id=user_id)


def _bearer_headers(client: TestClient, user_id: str) -> dict[str, str]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": "tui"}},
  )
  assert response.status_code == 200, response.text
  return {"Authorization": f"Bearer {response.json()['session_token']}"}


def _signed_headers(
  *,
  user_id: str = USER_ID,
  user_email: str = USER_EMAIL,
  issued_at: int | None = None,
  expiry: int | None = None,
  key: str = HMAC_KEY,
) -> dict[str, str]:
  issued = int(time.time()) if issued_at is None else issued_at
  expires = issued + 300 if expiry is None else expiry
  nonce = "0123456789abcdef0123456789abcdef"
  canonical = f"agent_api_v1\n{issued}\n{expires}\n{user_id}\n{user_email}\n{nonce}".encode("utf-8")
  signature = hmac.new(key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
  return {
    CLAIM_HEADERS["audience"]: "agent_api_v1",
    CLAIM_HEADERS["issued_at"]: str(issued),
    CLAIM_HEADERS["expiry"]: str(expires),
    CLAIM_HEADERS["user_id"]: user_id,
    CLAIM_HEADERS["user_email"]: user_email,
    CLAIM_HEADERS["nonce"]: nonce,
    CLAIM_HEADERS["signature"]: signature,
  }


def _write_artifact(
  data_dir: Path,
  user_id: str,
  ticker: str,
  skill: str,
  artifact_id: str,
  *,
  mtime: int,
  include_artifact_path: bool = True,
  run_id: str | None = None,
) -> dict[str, Any]:
  payload = {
    "artifact_id": artifact_id,
    "binary_artifact_path": None,
    "contract_name": "EarningsScenarios" if skill == "earnings-scenarios" else skill,
    "created_at": "2026-05-20T12:00:00.000Z",
    "data_source": "fixture",
    "skill": skill,
    "skill_run_id": artifact_id.split("-", 3)[-1],
    "ticker": ticker,
  }
  if run_id is not None:
    payload["run_id"] = run_id
  if include_artifact_path:
    payload["artifact_path"] = f"artifacts/{ticker}/{skill}/{artifact_id}.json"

  path = _workspace(data_dir, user_id) / "artifacts" / ticker / skill / f"{artifact_id}.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload), encoding="utf-8")
  os.utime(path, (mtime, mtime))
  return payload


def _write_letter(data_dir: Path, user_id: str, ticker: str, artifact_id: str, content: bytes) -> Path:
  path = _workspace(data_dir, user_id) / "letters" / ticker / f"{artifact_id}.docx"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(content)
  return path


def _write_html_artifact(
  data_dir: Path,
  user_id: str,
  artifact_id: str,
  *,
  mtime: int,
  session_id: str | None = "fixture-session",
  ts: str = "2026-06-05T14:24:35.683368+00:00",
) -> None:
  path = _workspace(data_dir, user_id) / "artifacts" / "_html" / f"{artifact_id}.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(
      {
        "artifact_id": artifact_id,
        "title": "Fixture HTML Artifact",
        "purpose": "report",
        "content_ref": f"artifacts/_html/{artifact_id}.html",
        "summary": "Deterministic dev-only HTML artifact fixture for Hank web live QA.",
        "ticker": "PCTY",
        "session_id": session_id,
        "source_skill": "fixture-html-artifact",
        "sources": [],
        "exports": {
          "copy_as_prompt": None,
          "copy_as_markdown": None,
          "copy_as_json": None,
        },
        "ts": ts,
        "contract_name": "HtmlArtifact",
      }
    ),
    encoding="utf-8",
  )
  html_path = path.with_suffix(".html")
  html_path.write_text("<main><h1>Fixture HTML Artifact</h1></main>", encoding="utf-8")
  os.utime(path, (mtime, mtime))
  os.utime(html_path, (mtime, mtime))


def _write_dashboard_artifact(
  data_dir: Path,
  user_id: str,
  artifact_id: str,
  *,
  mtime: int,
  ts: str = "2026-06-05T14:24:35.683368+00:00",
) -> None:
  path = _workspace(data_dir, user_id) / "artifacts" / "_dashboards" / f"{artifact_id}.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(
      {
        "artifact_id": artifact_id,
        "title": "Fixture Dashboard Artifact",
        "summary": "Deterministic dev-only DashboardArtifact fixture for Hank web live QA.",
        "ticker": "PCTY",
        "scope_label": None,
        "source_skill": "fixture-dashboard-artifact",
        "readiness_posture": "decision_ready",
        "profile": "production",
        "payload_ref": f"{artifact_id}.payload.json",
        "ts": ts,
        "contract_name": "DashboardArtifact",
      }
    ),
    encoding="utf-8",
  )
  payload_path = path.with_suffix(".payload.json")
  payload_path.write_text(
    json.dumps({"kind": "hank_dashboard.v1", "title": "Fixture Dashboard Artifact"}),
    encoding="utf-8",
  )
  os.utime(path, (mtime, mtime))
  os.utime(payload_path, (mtime, mtime))


def _workspace(data_dir: Path, user_id: str) -> Path:
  return data_dir / "users" / user_id / "workspace"
