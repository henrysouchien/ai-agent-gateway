from __future__ import annotations

from dataclasses import dataclass
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
  }


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


def _workspace(data_dir: Path, user_id: str) -> Path:
  return data_dir / "users" / user_id / "workspace"
