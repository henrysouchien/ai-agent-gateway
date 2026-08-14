from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway.capability_binding import (
  CAPABILITY_IDS,
  CredentialHandle,
)
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.server import (
  ChatRuntime,
  GatewayServerConfig,
  MaterializedCredential,
  create_gateway_app,
)


_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:schema-version-tests:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="schema-version-tests",
  actor_id=None,
)


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


_PROVIDERS = {
  family: _ExactProvider(family)
  for family in {"anthropic", "codex", "openai", "xai"}
}


async def _resolve_credentials(
  _api_key: str,
  payload: Any,
) -> ResolverResult:
  return ResolverResult(
    user_id=str(payload.user_id or "alice"),
    channel="web",
    auth_config=AuthConfig.from_dict({
      "provider": "anthropic",
      "api_key": "test-key",
      "billing_mode": "byok",
    }),
    credential_principal="service",
    allow_service_for_interactive=True,
    risk_user_id=1,
    role="owner",
    model_entitled_capabilities=CAPABILITY_IDS,
    model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
  )


def _materialize_service_credential(
  handle: CredentialHandle,
) -> MaterializedCredential:
  assert handle is _SERVICE_HANDLE
  return MaterializedCredential(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "api_key": "test-key",
      "billing_mode": "byok",
      "rate_table_version": "test-v1",
    },
  )


class _StreamingRunner:
  def __init__(self, event_log: Any, capability_execution: Any) -> None:
    self._event_log = event_log
    self.capability_execution = capability_execution

  async def run(self, **_: Any) -> None:
    self._event_log.append({"type": "text_delta", "text": "hello", "future_only": "strip-me"})
    self._event_log.append({"type": "future_event", "value": 1})
    self._event_log.append(
      {
        "type": "tool_output_chunk",
        "tool_call_id": "toolu_1",
        "tool_name": "code_execute",
        "stream": "stdout",
        "text": "line",
        "seq": 5,
      }
    )
    self._event_log.append({"type": "stream_complete", "usage": {}})


def _make_app():
  async def _build_chat_runtime(session, request, channel, auth_manager):
    _ = (session, channel, auth_manager)
    capability_bind = request.capability_bind
    assert capability_bind is not None
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda event_log, *_args: _StreamingRunner(
        event_log,
        request.capability_execution,
      ),
      capability_execution=request.capability_execution,
    )

  return create_gateway_app(
    GatewayServerConfig(
      tenant_id=_SERVICE_HANDLE.tenant_id,
      allow_service_credentials_for_interactive=True,
      credentials_resolver=_resolve_credentials,
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      service_provider_handles={"anthropic": _SERVICE_HANDLE},
      service_auth_config_resolver=_materialize_service_credential,
      capability_adapter_resolver=lambda adapter: _PROVIDERS[
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


def _init(client: TestClient, payload: dict[str, Any] | None = None) -> dict[str, Any]:
  response = client.post(
    "/api/chat/init",
    json={"api_key": "gateway-key", "user_id": "alice", **(payload or {})},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _sse_payloads(response_text: str) -> list[dict[str, Any]]:
  payloads: list[dict[str, Any]] = []
  for line in response_text.splitlines():
    if line.startswith("data: "):
      payloads.append(json.loads(line[6:]))
  return payloads


def _without_product_id(event: dict[str, Any]) -> dict[str, Any]:
  return {key: value for key, value in event.items() if key != "product_id"}


def test_chat_init_defaults_schema_version_and_echoes_response() -> None:
  app = _make_app()

  with TestClient(app) as client:
    init = _init(client)

  session = app.state.auth.session_store.get_session(init["session_id"])
  assert init["schema_version"] == 1
  assert session.schema_version == 1


def test_chat_init_accepts_explicit_v1() -> None:
  app = _make_app()

  with TestClient(app) as client:
    init = _init(client, {"schema_version": 1})

  session = app.state.auth.session_store.get_session(init["session_id"])
  assert init["schema_version"] == 1
  assert session.schema_version == 1


def test_chat_init_rejects_unsupported_schema_version() -> None:
  app = _make_app()

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "gateway-key", "user_id": "alice", "schema_version": 99},
    )

  assert response.status_code == 400
  assert response.json()["error"] == "unsupported_schema_version"
  assert "supported: [1]" in response.json()["message"]


def test_subscribe_rejects_schema_version_query_param() -> None:
  app = _make_app()

  with TestClient(app) as client:
    init = _init(client)
    response = client.get(
      f"/api/chat/subscribe?session_id={init['session_id']}&schema_version=1",
      headers={"Authorization": f"Bearer {init['session_token']}"},
    )

  assert response.status_code == 400
  assert "schema_version is set at session init" in response.json()["detail"]


def test_stream_envelopes_echo_session_schema_and_apply_v1_projection() -> None:
  app = _make_app()

  with TestClient(app) as client:
    init = _init(client, {"schema_version": 1})
    response = client.post(
      "/api/chat",
      json={
        "messages": [{"role": "user", "content": "hello"}],
        "user_id": "alice",
        "context": {},
      },
      headers={"Authorization": f"Bearer {init['session_token']}"},
    )

  assert response.status_code == 200, response.text
  payloads = _sse_payloads(response.text)
  assert [payload["schema_version"] for payload in payloads] == [1, 1, 1]
  assert [payload["seq"] for payload in payloads] == [1, 3, 4]
  assert _without_product_id(payloads[0]["event"]) == {"type": "text_delta", "text": "hello"}
  assert _without_product_id(payloads[1]["event"]) == {
    "type": "tool_output_chunk",
    "tool_call_id": "toolu_1",
    "tool_name": "code_execute",
    "stream": "stdout",
    "text": "line",
    "seq": 5,
  }
  assert _without_product_id(payloads[2]["event"]) == {"type": "stream_complete", "usage": {}}
