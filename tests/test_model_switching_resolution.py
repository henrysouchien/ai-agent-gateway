from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway.capability_binding import CredentialHandle
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


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


class _CompleteRunner:
  def __init__(self, event_log: Any, calls: list[dict[str, Any]], execution: Any) -> None:
    self._event_log = event_log
    self._calls = calls
    self.capability_execution = execution

  async def run(self, **_: Any) -> None:
    self._calls.append(self.capability_execution.bind.receipt())
    self._event_log.append({"type": "stream_complete", "usage": {}})


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

  async def _build_chat_runtime(session, request, channel, auth_manager):
    _ = session, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, *_args: _CompleteRunner(
        event_log,
        calls,
        request.capability_execution,
      ),
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
  assert calls[0]["model_key"] == "anthropic.claude-opus-5"
  assert calls[0]["upstream_model"] == "claude-opus-5"
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
