from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


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
class ArtifactApiFixture:
  app: Any
  data_dir: Path


@pytest.fixture
def artifact_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ArtifactApiFixture:
  data_dir = tmp_path / "data"
  monkeypatch.setenv("USER_DATA_DIR", str(data_dir))
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  monkeypatch.setenv("AGENT_API_CLAIM_MAX_TTL_SECONDS", "600")

  async def _build_chat_runtime(_session, _request, _channel, _auth_manager):
    return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)

  app = create_gateway_app(
    GatewayServerConfig(
      auth_config={"model": "claude-sonnet-4-6"},
      build_chat_runtime=_build_chat_runtime,
    )
  )
  return ArtifactApiFixture(app=app, data_dir=data_dir)


def test_latest_artifact_returns_latest_json_sidecar(artifact_api: ArtifactApiFixture) -> None:
  old_id = "2026-05-20T120000.000-run-a"
  latest_id = "2026-05-20T130000.000-run-b"
  _write_artifact(artifact_api.data_dir, USER_ID, "PCTY", "earnings-scenarios", old_id)
  latest_payload = _write_artifact(
    artifact_api.data_dir,
    USER_ID,
    "PCTY",
    "earnings-scenarios",
    latest_id,
    payload={"artifact_id": latest_id, "ticker": "PCTY", "skill": "earnings-scenarios", "latest": True},
  )

  with TestClient(artifact_api.app) as client:
    response = client.get(
      "/api/artifacts/PCTY/earnings-scenarios/latest",
      headers=_signed_headers(),
    )

  assert response.status_code == 200
  assert response.json() == latest_payload
  assert response.headers["etag"]


def test_artifact_by_id_returns_specific_json_sidecar(artifact_api: ArtifactApiFixture) -> None:
  artifact_id = "2026-05-20T120000.000-run-a"
  payload = _write_artifact(
    artifact_api.data_dir,
    USER_ID,
    "PCTY",
    "critical-factors",
    artifact_id,
    payload={"artifact_id": artifact_id, "ticker": "PCTY", "skill": "critical-factors"},
  )
  _write_artifact(artifact_api.data_dir, USER_ID, "PCTY", "critical-factors", "2026-05-20T130000.000-run-b")

  with TestClient(artifact_api.app) as client:
    response = client.get(
      f"/api/artifacts/PCTY/critical-factors/{artifact_id}",
      headers=_signed_headers(),
    )

  assert response.status_code == 200
  assert response.json() == payload


def test_artifact_index_returns_latest_per_skill_and_empty_list(artifact_api: ArtifactApiFixture) -> None:
  _write_artifact(artifact_api.data_dir, USER_ID, "PCTY", "earnings-scenarios", "2026-05-20T120000.000-run-a")
  _write_artifact(artifact_api.data_dir, USER_ID, "PCTY", "earnings-scenarios", "2026-05-20T130000.000-run-b")
  _write_artifact(artifact_api.data_dir, USER_ID, "PCTY", "critical-factors", "2026-05-20T121500.000-run-c")

  with TestClient(artifact_api.app) as client:
    response = client.get("/api/artifacts/PCTY", headers=_signed_headers())
    empty_response = client.get("/api/artifacts/MSCI", headers=_signed_headers())

  assert response.status_code == 200
  assert response.json() == [
    {"skill": "critical-factors", "latest_artifact_id": "2026-05-20T121500.000-run-c"},
    {"skill": "earnings-scenarios", "latest_artifact_id": "2026-05-20T130000.000-run-b"},
  ]
  assert empty_response.status_code == 200
  assert empty_response.json() == []


def test_letter_endpoint_returns_docx_blob(artifact_api: ArtifactApiFixture) -> None:
  artifact_id = "2026-05-20T130000.000-run-b"
  content = b"docx bytes"
  _write_letter(artifact_api.data_dir, USER_ID, "PCTY", artifact_id, content)

  with TestClient(artifact_api.app) as client:
    response = client.get(f"/api/letters/PCTY/{artifact_id}", headers=_signed_headers())

  assert response.status_code == 200
  assert response.content == content
  assert response.headers["content-type"] == (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  )
  assert response.headers["content-disposition"] == (
    'attachment; filename="LP-letter-PCTY-2026-05-20.docx"'
  )


def test_missing_artifact_returns_404(artifact_api: ArtifactApiFixture) -> None:
  with TestClient(artifact_api.app) as client:
    response = client.get(
      "/api/artifacts/PCTY/earnings-scenarios/2026-05-20T120000.000-run-a",
      headers=_signed_headers(),
    )
    latest_response = client.get(
      "/api/artifacts/PCTY/earnings-scenarios/latest",
      headers=_signed_headers(),
    )
    letter_response = client.get(
      "/api/letters/PCTY/2026-05-20T120000.000-run-a",
      headers=_signed_headers(),
    )

  assert response.status_code == 404
  assert latest_response.status_code == 404
  assert letter_response.status_code == 404


def test_unsigned_request_returns_401(artifact_api: ArtifactApiFixture) -> None:
  with TestClient(artifact_api.app) as client:
    response = client.get("/api/artifacts/PCTY/earnings-scenarios/latest")

  assert response.status_code == 401


def test_expired_claim_returns_401(artifact_api: ArtifactApiFixture) -> None:
  now = int(time.time())
  headers = _signed_headers(issued_at=now - 900, expiry=now - 300)

  with TestClient(artifact_api.app) as client:
    response = client.get("/api/artifacts/PCTY/earnings-scenarios/latest", headers=headers)

  assert response.status_code == 401


def test_raw_path_traversal_rejected(artifact_api: ArtifactApiFixture) -> None:
  status, _headers, _body = _asgi_get(
    artifact_api.app,
    "/api/artifacts/../etc/passwd/latest",
    headers=_signed_headers(),
  )

  assert status == 400


def test_encoded_path_traversal_rejected(artifact_api: ArtifactApiFixture) -> None:
  with TestClient(artifact_api.app) as client:
    response = client.get(
      "/api/artifacts/%2e%2e/etc/passwd/latest",
      headers=_signed_headers(),
    )
    component_response = client.get(
      "/api/artifacts/PCTY/earnings-scenarios/%2e.",
      headers=_signed_headers(),
    )

  assert response.status_code == 400
  assert component_response.status_code == 400


def test_symlink_escape_rejected(artifact_api: ArtifactApiFixture, tmp_path: Path) -> None:
  workspace = _workspace(artifact_api.data_dir, USER_ID)
  artifacts_dir = workspace / "artifacts"
  artifacts_dir.mkdir(parents=True)
  outside = tmp_path / "outside"
  outside.mkdir()
  (artifacts_dir / "PCTY").symlink_to(outside, target_is_directory=True)

  with TestClient(artifact_api.app) as client:
    response = client.get(
      "/api/artifacts/PCTY/earnings-scenarios/latest",
      headers=_signed_headers(),
    )

  assert response.status_code == 400


def test_user_isolation_returns_404_not_403(artifact_api: ArtifactApiFixture) -> None:
  artifact_id = "2026-05-20T130000.000-run-b"
  _write_artifact(artifact_api.data_dir, "bob", "PCTY", "earnings-scenarios", artifact_id)

  with TestClient(artifact_api.app) as client:
    response = client.get(
      f"/api/artifacts/PCTY/earnings-scenarios/{artifact_id}",
      headers=_signed_headers(user_id=USER_ID),
    )
    bob_response = client.get(
      f"/api/artifacts/PCTY/earnings-scenarios/{artifact_id}",
      headers=_signed_headers(user_id="bob", user_email="bob@example.com"),
    )

  assert response.status_code == 404
  assert bob_response.status_code == 200


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
  payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
  resolved_payload = payload or {"artifact_id": artifact_id, "ticker": ticker, "skill": skill}
  path = _workspace(data_dir, user_id) / "artifacts" / ticker / skill / f"{artifact_id}.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(resolved_payload), encoding="utf-8")
  return resolved_payload


def _write_letter(data_dir: Path, user_id: str, ticker: str, artifact_id: str, content: bytes) -> Path:
  path = _workspace(data_dir, user_id) / "letters" / ticker / f"{artifact_id}.docx"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(content)
  return path


def _workspace(data_dir: Path, user_id: str) -> Path:
  return data_dir / "users" / user_id / "workspace"


def _asgi_get(app: Any, path: str, *, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
  async def _run() -> tuple[int, dict[str, str], bytes]:
    messages: list[dict[str, Any]] = []
    received = False

    async def receive() -> dict[str, Any]:
      nonlocal received
      if received:
        return {"type": "http.disconnect"}
      received = True
      return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
      messages.append(message)

    encoded_headers = [(b"host", b"testserver")]
    encoded_headers.extend(
      (name.lower().encode("latin-1"), value.encode("latin-1"))
      for name, value in headers.items()
    )
    scope = {
      "type": "http",
      "asgi": {"version": "3.0"},
      "http_version": "1.1",
      "method": "GET",
      "scheme": "http",
      "path": path,
      "raw_path": path.encode("ascii"),
      "query_string": b"",
      "headers": encoded_headers,
      "client": ("testclient", 50000),
      "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
      message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    response_headers = {
      name.decode("latin-1"): value.decode("latin-1")
      for name, value in start.get("headers", [])
    }
    return int(start["status"]), response_headers, body

  return asyncio.run(_run())
