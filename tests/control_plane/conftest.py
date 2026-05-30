from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[4]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from memory import get_skills_root


@pytest.fixture
def test_api_key() -> str:
  return "test-tui-key"


@pytest.fixture
def test_user_id() -> str:
  return "tui-user"


@pytest.fixture
def test_channel() -> str:
  return "tui"


@pytest.fixture
def control_session_url() -> str:
  return "/api/control/session"


@pytest.fixture
def control_health_url() -> str:
  return "/api/control/health"


@pytest.fixture
def auth_config() -> AuthConfig:
  return AuthConfig.from_dict(
    {
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": "operator-key",
      "model": "claude-sonnet-4-6",
    }
  )


@pytest.fixture
def credentials_resolver(test_api_key: str, test_user_id: str, test_channel: str, auth_config: AuthConfig):
  async def _resolver(api_key: str, _init_request: Any) -> ResolverResult:
    assert api_key == test_api_key
    return ResolverResult(
      user_id=test_user_id,
      channel=test_channel,
      auth_config=auth_config,
      risk_user_id=101,
      role="owner",
      user_email="tui@example.com",
    )

  return _resolver


@pytest.fixture
def control_plane_app(test_api_key: str, credentials_resolver):
  async def _build_chat_runtime(_session, _request, _channel, _auth_manager):
    return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="control-plane-test-secret-0123456789",
      valid_api_keys={test_api_key},
      credentials_resolver=credentials_resolver,
      build_chat_runtime=_build_chat_runtime,
      control_skills_dir=get_skills_root(),
    )
  )


@pytest.fixture
def client(control_plane_app):
  with TestClient(control_plane_app) as test_client:
    yield test_client


@pytest.fixture
def test_control_session(client: TestClient, control_session_url: str, test_api_key: str):
  response = client.post(
    control_session_url,
    json={"api_key": test_api_key, "context": {"channel": "tui"}},
  )
  assert response.status_code == 200
  return response.json()
