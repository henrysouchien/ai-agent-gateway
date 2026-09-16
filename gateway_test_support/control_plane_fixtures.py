from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.event_log import EventLog
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.runner import AgentRunner
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.session import GatewaySession, session_owner_user_id
from agent_gateway.tool_dispatcher import ToolDispatcher

from .control_plane_identity import fake_identity_resolver, fake_mcp_user_key_lookup


class _ControlTestRunner(AgentRunner):
  def __init__(
    self,
    event_log: EventLog,
    session_id: str,
    started_at: float,
    capability_execution: BoundCapabilityExecution,
    *,
    gateway_session: GatewaySession,
    billing_mode: str,
    channel: str | None,
  ) -> None:
    super().__init__(
      event_log=event_log,
      dispatcher=ToolDispatcher(
        mcp_client=McpClientManager(config_path=None),
        local_tool_handlers={},
        event_log=event_log,
        session_id=session_id,
      ),
      session_id=session_id,
      capability_execution=capability_execution,
      started_at=started_at,
      gateway_session=gateway_session,
      user_id=session_owner_user_id(gateway_session),
      rate_table_version="test",
      billing_mode=billing_mode,
      channel=channel,
    )
    self._event_log = event_log

  async def run(
    self,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None = None,
    max_turns: int | None = None,
    *,
    resume_initial_messages: list[dict[str, Any]] | None = None,
    **_kwargs: Any,
  ) -> None:
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
def control_plane_config(
  test_api_key: str,
  auth_config: AuthConfig,
  credentials_resolver,
):
  async def _build_chat_runtime(session, _request, channel, _auth_manager, *, storage_root: Path | None = None):
    capability_execution = _request.capability_execution
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda event_log, session_id, started_at: _ControlTestRunner(
        event_log,
        session_id,
        started_at,
        capability_execution,
        gateway_session=session,
        billing_mode=auth_config.billing_mode,
        channel=channel,
      ),
      capability_execution=capability_execution,
    )

  return GatewayServerConfig(
    jwt_secret="control-plane-test-secret-0123456789",
    valid_api_keys={test_api_key},
    tenant_id="test-product",
    identity_resolver=fake_identity_resolver,
    mcp_user_key_lookup=fake_mcp_user_key_lookup,
    credentials_resolver=credentials_resolver,
    model_registry=INITIAL_MODEL_REGISTRY,
    model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    build_chat_runtime=_build_chat_runtime,
  )


@pytest.fixture
def control_plane_app(control_plane_config):
  return create_gateway_app(control_plane_config)


@pytest.fixture
def client(control_plane_app) -> Iterator[TestClient]:
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
