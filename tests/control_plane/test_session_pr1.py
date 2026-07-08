from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agent_gateway import session as session_module
from agent_gateway.control_plane.middleware import CONTROL_PLANE_VERSION_HEADER
from agent_gateway.control_plane import session as control_session_module
from agent_gateway.control_plane.session import CONTROL_SESSION_TTL_SECONDS
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


def _gateway_user_keys(*, key: str, slug: str, email: str, risk_user_id: int, channel: str) -> str:
  return json.dumps([
    {
      "key": key,
      "channel": channel,
      "slug": slug,
      "email": email,
      "risk_user_id": risk_user_id,
      "role": "owner",
    }
  ])


def test_control_session_lifecycle_create_use_expire_recreate(
  monkeypatch: pytest.MonkeyPatch,
  client: TestClient,
  control_plane_app,
  control_session_url: str,
  test_api_key: str,
  test_user_id: str,
  test_channel: str,
) -> None:
  fake_now = [1_700_000_000]
  monkeypatch.setattr(session_module.time, "time", lambda: fake_now[0])
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    _gateway_user_keys(
      key=test_api_key,
      slug=test_user_id,
      email="tui@example.com",
      risk_user_id=101,
      channel=test_channel,
    ),
  )

  first = client.post(
    control_session_url,
    json={"api_key": test_api_key, "context": {"channel": test_channel}},
  )
  assert first.status_code == 200
  assert first.headers[CONTROL_PLANE_VERSION_HEADER] == "1"
  payload = first.json()
  assert payload["kind"] == "control"
  assert payload["user_id"] == "101"
  assert payload["risk_user_id"] == 101
  assert payload["user_slug"] == test_user_id
  assert payload["user_email"] == "tui@example.com"
  assert payload["channel"] == test_channel
  assert payload["identity"] == {
    "owner_user_id": "101",
    "user_slug": test_user_id,
    "aliases": ["101", test_user_id, "tui@example.com"],
    "identity_status": "risk_user_id_authoritative",
  }
  assert payload["expires_at"] == fake_now[0] + CONTROL_SESSION_TTL_SECONDS

  session = control_plane_app.state.auth.session_store.get_session(payload["session_id"])
  assert session is not None
  assert session.kind == "control"
  assert session.channel == test_channel
  assert session.user_id == test_user_id
  assert session.owner_user_id == "101"
  assert session.user_slug == test_user_id
  assert session.risk_user_id == 101
  assert session.user_aliases == ("101", test_user_id, "tui@example.com")
  assert session.expires_at == payload["expires_at"]

  verified_session, claims = control_plane_app.state.auth.verify_token_with_payload(payload["session_token"])
  assert verified_session is session
  assert claims["session_id"] == session.session_id
  assert claims["risk_user_id"] == 101
  assert "kind" not in claims

  fake_now[0] += CONTROL_SESSION_TTL_SECONDS + 1
  with pytest.raises(HTTPException) as exc_info:
    control_plane_app.state.auth.verify_token(payload["session_token"])
  assert exc_info.value.status_code == 401
  assert exc_info.value.detail == "Session expired"
  assert control_plane_app.state.auth.session_store.get_session(payload["session_id"]) is None

  second = client.post(
    control_session_url,
    json={"api_key": test_api_key, "context": {"channel": test_channel}},
  )
  assert second.status_code == 200
  assert second.json()["session_id"] != payload["session_id"]
  assert second.json()["user_id"] == "101"
  assert second.json()["channel"] == test_channel
  assert second.json()["expires_at"] == fake_now[0] + CONTROL_SESSION_TTL_SECONDS


def test_control_session_stores_identity_derived_numeric_risk_user_id_without_resolver() -> None:
  async def _build_chat_runtime(_session, _request, _channel, _auth_manager):
    return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="control-plane-test-secret-0123456789",
      valid_api_keys={"legacy-key"},
      build_chat_runtime=_build_chat_runtime,
    )
  )

  with TestClient(app) as client:
    response = client.post(
      "/api/control/session",
      json={"api_key": "legacy-key", "user_id": "101", "context": {"channel": "tui"}},
    )

  assert response.status_code == 200
  payload = response.json()
  assert payload["user_id"] == "101"
  assert payload["risk_user_id"] == 101
  assert payload["identity"]["identity_status"] == "numeric_user_id"
  session = app.state.auth.session_store.get_session(payload["session_id"])
  assert session is not None
  assert session.owner_user_id == "101"
  assert session.raw_user_id == "101"
  assert session.risk_user_id == 101
  _verified_session, claims = app.state.auth.verify_token_with_payload(payload["session_token"])
  assert claims["risk_user_id"] == 101


def test_control_session_stores_identity_mapped_email_without_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    _gateway_user_keys(
      key="mapped-key",
      slug="henry",
      email="henry@example.com",
      risk_user_id=1,
      channel="mcp",
    ),
  )

  async def _build_chat_runtime(_session, _request, _channel, _auth_manager):
    return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="control-plane-test-secret-0123456789",
      valid_api_keys={"legacy-key"},
      build_chat_runtime=_build_chat_runtime,
    )
  )

  with TestClient(app) as client:
    response = client.post(
      "/api/control/session",
      json={"api_key": "legacy-key", "user_id": "henry", "context": {"channel": "tui"}},
    )

  assert response.status_code == 200
  payload = response.json()
  assert payload["user_email"] == "henry@example.com"
  assert payload["identity"]["owner_user_id"] == "1"
  session = app.state.auth.session_store.get_session(payload["session_id"])
  assert session is not None
  assert session.owner_user_id == "1"
  assert session.user_email == "henry@example.com"
  assert session.user_aliases == ("1", "henry", "henry@example.com")
  _verified_session, claims = app.state.auth.verify_token_with_payload(payload["session_token"])
  assert claims["user_email"] == "henry@example.com"


def test_control_session_channel_mismatch_returns_401(
  client: TestClient,
  control_session_url: str,
  test_api_key: str,
  test_user_id: str,
) -> None:
  response = client.post(
    control_session_url,
    json={"api_key": test_api_key, "context": {"channel": "excel"}},
  )

  assert response.status_code == 401
  assert response.headers[CONTROL_PLANE_VERSION_HEADER] == "1"
  assert response.json()["error"] == "channel_mismatch"
  assert response.json()["user_id"] == test_user_id


def test_control_identity_falls_back_when_imported_helper_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(sys.modules, "user_identity", SimpleNamespace())

  identity = control_session_module._resolve_control_identity(
    user_id="henry",
    risk_user_id=1,
    user_email="henry@example.com",
    role="owner",
    channel="mcp",
  )

  assert identity.owner_user_id == "1"
  assert identity.user_slug == "henry"
  assert identity.aliases == ("1", "henry", "henry@example.com")
  assert identity.identity_status == "fallback_canonical"


def test_chat_init_still_creates_chat_session(
  client: TestClient,
  control_plane_app,
  test_api_key: str,
  test_user_id: str,
  test_channel: str,
) -> None:
  response = client.post(
    "/api/chat/init",
    json={"api_key": test_api_key, "context": {"channel": test_channel}},
  )

  assert response.status_code == 200
  payload = response.json()
  session = control_plane_app.state.auth.session_store.get_session(payload["session_id"])
  assert session is not None
  assert session.kind == "chat"
  assert session.user_id == test_user_id
  assert session.channel == test_channel


def test_control_session_token_cannot_dispatch_chat(
  client: TestClient,
  test_control_session: dict,
  test_user_id: str,
) -> None:
  response = client.post(
    "/api/chat",
    headers={"Authorization": f"Bearer {test_control_session['session_token']}"},
    json={"messages": [{"role": "user", "content": "hi"}], "user_id": test_user_id},
  )

  assert response.status_code == 400
  assert response.json()["error"] == "invalid_session_kind"
