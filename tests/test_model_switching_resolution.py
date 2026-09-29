from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agent_gateway import AgentRunner, EventLog, McpClientManager, ToolDispatcher
from agent_gateway.auth import AuthConfig, ResolverResult
from model_authority.binding import CredentialHandle
from agent_gateway.capability_execution import (
  BoundCapabilityExecution,
  MaterializedCredential,
)
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from gateway_test_support.model_defaults import SESSION_DRIVER


class _ExactProvider(ModelProvider):
  def __init__(self, name: str) -> None:
    self.name = name

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=64_000,
      supports_thinking=True,
    )


class _CompleteRunner(AgentRunner):
  def __init__(
    self,
    event_log: EventLog,
    calls: list[dict[str, Any]],
    execution: BoundCapabilityExecution,
    session_id: str,
  ) -> None:
    mcp_client = McpClientManager(config_path=None)
    super().__init__(
      event_log=event_log,
      dispatcher=ToolDispatcher(
        mcp_client=mcp_client,
        local_tool_handlers={},
        event_log=event_log,
        session_id=session_id,
      ),
      session_id=session_id,
      capability_execution=execution,
      mcp_client=mcp_client,
      get_tool_definitions=lambda: [],
      user_id="alice",
      billing_mode="byok",
      rate_table_version="test",
    )
    self._calls = calls

  async def run(
    self,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None = None,
    max_turns: int | None = None,
    *,
    resume_initial_messages: list[dict[str, Any]] | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns, resume_initial_messages
    self._calls.append(self.capability_execution.bind.to_json())
    self._log.append({"type": "stream_complete", "usage": {}})


def _make_app():
  calls: list[dict[str, Any]] = []
  providers = {
    family: _ExactProvider(family)
    for family in {"anthropic", "codex", "openai", "xai"}
  }
  service_handles = {
    family: CredentialHandle(
      handle_id=f"service:model-resolution:{family}",
      provider=family,
      principal="service",
      tenant_id="model-resolution-test",
      actor_id=None,
    )
    for family in providers
  }

  async def _credentials(_api_key: str, payload: Any) -> ResolverResult:
    return ResolverResult(
      user_id=str(payload.user_id or "alice"),
      channel="web",
      auth_config=AuthConfig.from_dict({
        "provider": "anthropic",
        "api_key": "session-test-key",
        "billing_mode": "byok",
      }),
      credential_principal="service",
      allow_service_for_interactive=True,
      risk_user_id=1,
      role="owner",
      model_entitled_capabilities=CAPABILITY_IDS,
      model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
    )

  def _materialize(handle: CredentialHandle) -> MaterializedCredential:
    return MaterializedCredential(
      handle=handle,
      auth_config={
        "provider": handle.provider,
        "api_key": "service-test-key",
        "auth_mode": "api",
        "billing_mode": "byok",
        "rate_table_version": "test",
      },
    )

  async def _build_chat_runtime(session, request, channel, auth_manager, *, storage_root: Path | None = None):
    _ = session, channel, auth_manager

    def _build_runner(
      event_log: EventLog,
      session_id: str,
      started_at: float,
    ) -> AgentRunner:
      _ = started_at
      return _CompleteRunner(
        event_log,
        calls,
        request.capability_execution,
        session_id,
      )

    return ChatRuntime(
      system_prompt="system",
      build_runner=_build_runner,
      capability_execution=request.capability_execution,
    )

  app = create_gateway_app(
    GatewayServerConfig(
      tenant_id="model-resolution-test",
      credentials_resolver=_credentials,
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      service_provider_handles=service_handles,
      service_auth_config_resolver=_materialize,
      capability_adapter_resolver=lambda adapter: providers[
        INITIAL_MODEL_REGISTRY.models[
          next(
            key
            for key, entry in INITIAL_MODEL_REGISTRY.models.items()
            if entry.adapter == adapter
          )
        ].provider
      ],
      build_chat_runtime=_build_chat_runtime,
    )
  )
  return app, calls


def _init(client: TestClient) -> str:
  response = client.post(
    "/api/chat/init",
    json={"api_key": "gateway-key", "user_id": "alice"},
  )
  assert response.status_code == 200, response.text
  return response.json()["session_token"]


def _run(client: TestClient, token: str, payload: dict[str, Any]) -> Any:
  with client.stream(
    "POST",
    "/api/chat",
    headers={"Authorization": f"Bearer {token}"},
    json={
      "messages": [{"role": "user", "content": "hi"}],
      "user_id": "alice",
      **payload,
    },
  ) as response:
    list(response.iter_lines())
    return response


def test_server_omission_uses_registry_default_complete_binding() -> None:
  app, calls = _make_app()
  with TestClient(app) as client:
    token = _init(client)
    response = _run(client, token, {})

  assert response.status_code == 200
  assert calls[0]["model_key"] == SESSION_DRIVER.model_key
  assert calls[0]["upstream_model"] == SESSION_DRIVER.upstream_model
  assert calls[0]["selection_source"] == "capability_default"


def test_server_accepts_only_stable_key_selection() -> None:
  app, calls = _make_app()
  with TestClient(app) as client:
    token = _init(client)
    response = _run(
      client,
      token,
      {"model_key": "openai.gpt-5-6", "effort": "xhigh"},
    )

  assert response.status_code == 200
  assert calls[0]["model_key"] == "openai.gpt-5-6"
  assert calls[0]["provider"] == "openai"
  assert calls[0]["upstream_model"] == "gpt-5.6"
  assert calls[0]["effort"] == "xhigh"


def test_server_rejects_unknown_stable_key_before_runtime_dispatch() -> None:
  app, calls = _make_app()
  with TestClient(app) as client:
    token = _init(client)
    response = client.post(
      "/api/chat",
      headers={"Authorization": f"Bearer {token}"},
      json={
        "messages": [{"role": "user", "content": "hi"}],
        "user_id": "alice",
        "model_key": "openai:gpt-5.6",
      },
    )

  assert response.status_code == 400
  assert response.json()["error_code"] == "capability_model_unavailable"
  assert response.json()["model_key"] == "openai:gpt-5.6"
  assert calls == []
