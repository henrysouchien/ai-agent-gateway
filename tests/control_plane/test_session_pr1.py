from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agent_gateway import session as session_module
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.control_plane.middleware import CONTROL_PLANE_VERSION_HEADER
from agent_gateway.event_log import EventLog
from agent_gateway.runner import AgentRunner
from agent_gateway.control_plane import session as control_session_module
from agent_gateway.control_plane.session import CONTROL_SESSION_TTL_SECONDS
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app

from gateway_test_support.control_plane_identity import fake_identity_resolver


def _unused_runner(
  _event_log: EventLog,
  _session_id: str,
  _started_at: float,
) -> AgentRunner:
  raise AssertionError("control session tests never run a chat turn")


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
  assert session.tenant_id == "test-product"
  assert session.session_credential_handle is not None
  assert session.session_credential_handle.provider == "anthropic"
  assert session.session_credential_handle.principal == "service"
  assert session.session_credential_handle.actor_id is None
  assert session.allow_service_for_interactive is True

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


def test_control_session_stores_numeric_identity_without_credentials_resolver() -> None:
  async def _build_chat_runtime(_session, _request, _channel, _auth_manager, *, storage_root: Path | None = None):
    return ChatRuntime(
      system_prompt="test",
      build_runner=_unused_runner,
      capability_execution=_request.capability_execution,
    )

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="control-plane-test-secret-0123456789",
      valid_api_keys={"legacy-key"},
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      identity_resolver=fake_identity_resolver,
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


def test_control_session_stores_email_from_supplied_identity_resolver() -> None:
  def resolve_identity(user_id, **_kwargs):
    return fake_identity_resolver(
      user_id,
      risk_user_id=1,
      user_email="henry@example.com",
    )

  async def _build_chat_runtime(_session, _request, _channel, _auth_manager, *, storage_root: Path | None = None):
    return ChatRuntime(
      system_prompt="test",
      build_runner=_unused_runner,
      capability_execution=_request.capability_execution,
    )

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="control-plane-test-secret-0123456789",
      valid_api_keys={"legacy-key"},
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      identity_resolver=resolve_identity,
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


def test_control_identity_uses_supplied_metadata_without_resolver() -> None:
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
  assert session.tenant_id == "test-product"
  assert session.session_credential_handle is not None
  assert session.session_credential_handle.principal == "service"


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
