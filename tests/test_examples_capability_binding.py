from __future__ import annotations

import asyncio
import runpy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_gateway.event_log import EventLog
from agent_gateway.server import ChatTurnInputs
from agent_gateway.server_chat_helpers import prepare_session_driver_turn
from agent_gateway.server_models import ChatMessage


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATHS = (
  ROOT / "packages" / "agent-gateway" / "examples" / "06-tool-approval" / "agent.py",
  ROOT / "packages" / "agent-gateway" / "examples" / "07-full-production" / "agent.py",
)


@pytest.mark.parametrize("example_path", EXAMPLE_PATHS, ids=lambda path: path.parent.name)
def test_custom_gateway_examples_consume_exact_session_driver_bind(
  example_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
  module = runpy.run_path(
    str(example_path),
    run_name=f"_test_{example_path.parent.name.replace('-', '_')}",
  )
  app = module["app"]
  config = app.state.gateway_config

  init_response = TestClient(app).post(
    "/api/chat/init",
    json={"api_key": "demo-key", "user_id": "demo-user"},
  )
  assert init_response.status_code == 200
  assert init_response.json()["user_id"] == "demo-user"

  policy = config.model_selection_policy.capabilities["session.driver"]
  assert policy.default.model_key == "anthropic.claude-opus-5"
  assert policy.default.effort == module["DEFAULT_EFFORT"]
  entry = config.model_registry.require(policy.default.model_key)
  assert entry.provider == "anthropic"
  assert entry.upstream_model == module["DEFAULT_MODEL"]
  assert entry.adapter == "anthropic.messages"
  assert config.allow_service_credentials_for_interactive is True

  service_handle = config.service_provider_handles["anthropic"]
  materialized = config.service_auth_config_resolver(service_handle)
  assert materialized.handle is service_handle
  assert materialized.auth_config["api_key"] == "test-anthropic-key"
  assert "model" not in materialized.auth_config
  assert "effort" not in materialized.auth_config

  session = app.state.auth.session_store.get_session(
    init_response.json()["session_id"]
  )
  assert session is not None
  prepared = prepare_session_driver_turn(
    session,
    ChatTurnInputs(
      messages=[ChatMessage(role="user", content="hello")],
      request_id="example-request",
      context={"channel": "web"},
      metadata={},
      model_key=None,
    ),
    build_chat_runtime=app.state.gateway_build_chat_runtime,
  )
  request = prepared.request
  bind = request.capability_bind
  assert bind is not None
  assert bind.receipt() == {
    "schema_version": "1.0",
    "capability_id": "session.driver",
    "model_key": "anthropic.claude-opus-5",
    "provider": "anthropic",
    "upstream_model": module["DEFAULT_MODEL"],
    "adapter": "anthropic.messages",
    "protocol_profile": "messages.adaptive",
    "route": "anthropic.public",
    "effort": module["DEFAULT_EFFORT"],
    "credential_principal": "service",
    "credential_ref": module["SERVICE_CREDENTIAL_HANDLE"].handle_id,
    "run_mode": "interactive",
    "registry_revision": config.model_registry.revision,
    "policy_revision": config.model_selection_policy.revision,
    "selection_source": "capability_default",
  }
  assert request.bound_auth_config == {
    "provider": "anthropic",
    "auth_mode": "api",
    "api_key": "test-anthropic-key",
    "auth_token": "",
    "max_tokens": 16_000,
    "billing_mode": "byok",
    "rate_table_version": module["rate_table"].version,
  }

  runtime = asyncio.run(
    config.build_chat_runtime(
      session,
      request,
      prepared.channel,
      app.state.auth,
    )
  )
  assert runtime.capability_bind is bind
  assert runtime.capability_execution is request.capability_execution
  assert runtime.provider is request.bound_provider
  assert runtime.resolved_provider_name == bind.provider

  runner = runtime.build_runner(
    EventLog(session_id=session.session_id),
    session.session_id,
    float(session.created_at),
  )
  assert runner._provider is request.bound_provider
  assert runner._auth_config == dict(request.bound_auth_config)
  assert runner._capability_execution is request.capability_execution
