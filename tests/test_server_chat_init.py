from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


def _auth_config() -> AuthConfig:
  return AuthConfig.from_dict(
    {
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": "operator-key",
      "model": "claude-sonnet-4-6",
    }
  )


def _resolver_result(channel: str = "cli") -> ResolverResult:
  return ResolverResult(
    user_id="alice",
    channel=channel,
    auth_config=_auth_config(),
    risk_user_id=101,
    role="owner",
    user_email="alice@example.com",
  )


def _make_app(credentials_resolver):
  async def _build_chat_runtime(_session, _request, _channel, _auth_manager):
    return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)

  return create_gateway_app(
    GatewayServerConfig(
      auth_config={"model": "claude-sonnet-4-6"},
      credentials_resolver=credentials_resolver,
      build_chat_runtime=_build_chat_runtime,
    )
  )


def test_channel_mismatch_at_init_returns_400() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "context": {"channel": "excel"}},
    )

  assert response.status_code == 400
  assert response.json()["error"] == "channel_mismatch"
  assert response.json()["user_id"] == "alice"


def test_channel_match_at_init_succeeds() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "context": {"channel": "cli"}},
    )

  assert response.status_code == 200
  session = app.state.auth.session_store.get_session(response.json()["session_id"])
  assert session.channel == "cli"


def test_channel_omitted_at_init_succeeds() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="mcp")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "mcp-key"})

  assert response.status_code == 200
  session = app.state.auth.session_store.get_session(response.json()["session_id"])
  assert session.channel == "mcp"


def test_chat_init_response_includes_resolved_user_id() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "user_id": "claimed-user"},
    )

  assert response.status_code == 200
  assert response.json()["user_id"] == "alice"


def test_httpexception_from_resolver_passes_through_status() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    raise HTTPException(status_code=401, detail="API key is not mapped to a user identity")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "legacy-key", "user_id": "alice"})

  assert response.status_code == 401
  assert response.json() == {
    "error": "auth_failed",
    "message": "API key is not mapped to a user identity",
    "user_id": "alice",
  }
