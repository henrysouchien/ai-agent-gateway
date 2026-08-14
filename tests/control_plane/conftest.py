from __future__ import annotations

# ruff: noqa: E402

import json
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
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from memory import get_skills_root


class _ControlTestRunner:
  def __init__(self, event_log, capability_execution) -> None:
    self._event_log = event_log
    self.capability_execution = capability_execution

  async def run(self, **_kwargs: Any) -> None:
    self._event_log.append({"type": "stream_complete", "usage": {}})


@pytest.fixture(autouse=True)
def _canonical_gateway_state_root(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  state_root = tmp_path / "gateway-state"
  (state_root / "gateway").mkdir(parents=True)
  monkeypatch.setenv("USER_DATA_DIR", str(state_root))
  monkeypatch.delenv(
    "GATEWAY_APPROVAL_DB_PATH",
    raising=False,
  )
  # Autonomous spawn narrows GATEWAY_USER_KEYS to the admitted user's mcp
  # entry and refuses when none matches. Cover every identity these suites
  # dispatch as, by slug match, with emails equal to the dispatch emails (the
  # matcher refuses on id-match + email-mismatch). risk_user_id is
  # deliberately OMITTED so canonical identity resolution skips these entries
  # (owners stay legacy ids like "alice", which the suites assert on); real
  # deployments can't do this — gateway boot auth-validates full entries.
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([
      {
        "key": f"control-test-mcp-key-{slug}",
        "channel": "mcp",
        "slug": slug,
        "email": email,
        "role": "owner",
      }
      for slug, email in (
        ("alice", "alice@example.com"),
        ("bob", "bob@example.com"),
        ("carol", "carol@example.com"),
        ("owner-1", "owner-1@example.com"),
        ("tui-user", "tui@example.com"),
        ("1", "one@example.com"),
        ("101", "one-oh-one@example.com"),
      )
    ]),
  )


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
      credential_principal="service",
      allow_service_for_interactive=True,
      risk_user_id=101,
      role="owner",
      user_email="tui@example.com",
      model_entitled_capabilities=frozenset(
        INITIAL_MODEL_SELECTION_POLICY.capabilities
      ),
      model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
    )

  return _resolver


@pytest.fixture
def control_plane_app(test_api_key: str, credentials_resolver):
  async def _build_chat_runtime(_session, _request, _channel, _auth_manager):
    capability_execution = _request.capability_execution
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda event_log, _sid, _started_at: _ControlTestRunner(
        event_log,
        capability_execution,
      ),
      capability_execution=capability_execution,
    )

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="control-plane-test-secret-0123456789",
      valid_api_keys={test_api_key},
      tenant_id="test-product",
      credentials_resolver=credentials_resolver,
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
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
