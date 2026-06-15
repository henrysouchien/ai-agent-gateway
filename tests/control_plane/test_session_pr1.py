from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agent_gateway import session as session_module
from agent_gateway.control_plane.middleware import CONTROL_PLANE_VERSION_HEADER
from agent_gateway.control_plane.session import CONTROL_SESSION_TTL_SECONDS


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

  first = client.post(
    control_session_url,
    json={"api_key": test_api_key, "context": {"channel": test_channel}},
  )
  assert first.status_code == 200
  assert first.headers[CONTROL_PLANE_VERSION_HEADER] == "1"
  payload = first.json()
  assert payload["kind"] == "control"
  assert payload["user_id"] == test_user_id
  assert payload["channel"] == test_channel
  assert payload["expires_at"] == fake_now[0] + CONTROL_SESSION_TTL_SECONDS

  session = control_plane_app.state.auth.session_store.get_session(payload["session_id"])
  assert session is not None
  assert session.kind == "control"
  assert session.channel == test_channel
  assert session.user_id == test_user_id
  assert session.expires_at == payload["expires_at"]

  verified_session, claims = control_plane_app.state.auth.verify_token_with_payload(payload["session_token"])
  assert verified_session is session
  assert claims["session_id"] == session.session_id
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
  assert second.json()["channel"] == test_channel
  assert second.json()["expires_at"] == fake_now[0] + CONTROL_SESSION_TTL_SECONDS


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
