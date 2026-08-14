from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway.capability_binding import ModelSelectionIntent
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.model_preferences import ModelPreferenceStore
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.session import bind_session_credentials


_SELECTABLE_CAPABILITIES = frozenset({
  "session.driver",
  "plan.author",
})


def _auth_config() -> AuthConfig:
  return AuthConfig.from_dict(
    {
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": "operator-key",
    }
  )


def _resolver_result(channel: str = "cli") -> ResolverResult:
  return ResolverResult(
    user_id="alice",
    channel=channel,
    auth_config=_auth_config(),
    credential_principal="service",
    allow_service_for_interactive=True,
    risk_user_id=101,
    role="owner",
    user_email="alice@example.com",
    model_entitled_capabilities=CAPABILITY_IDS,
    model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
  )


def _make_app(
  credentials_resolver,
  *,
  on_session_created=None,
  model_preference_store: ModelPreferenceStore | None = None,
):
  async def _build_chat_runtime(_session, request, _channel, _auth_manager):
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda *_args: None,
      capability_execution=request.capability_execution,
    )

  return create_gateway_app(
    GatewayServerConfig(
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      credentials_resolver=credentials_resolver,
      model_preference_store=model_preference_store,
      build_chat_runtime=_build_chat_runtime,
      on_session_created=on_session_created,
    )
  )


def test_delete_model_preference_uses_authenticated_resolver_identity(
  tmp_path: Path,
) -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  store = ModelPreferenceStore(tmp_path / "model-preferences.sqlite3")
  app = _make_app(_resolver, model_preference_store=store)

  with TestClient(app) as client:
    init_response = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "context": {"channel": "cli"}},
    )
    assert init_response.status_code == 200, init_response.text
    session = app.state.auth.session_store.get_session(
      init_response.json()["session_id"]
    )
    assert session is not None
    session.tenant_id = None
    store.put(
      tenant_id="test-product",
      actor_id="101",
      capability_id="session.driver",
      model_key="anthropic.claude-sonnet-5",
      effort="high",
    )

    response = client.delete(
      "/api/model-preferences/session.driver",
      headers={
        "Authorization": f"Bearer {init_response.json()['session_token']}"
      },
    )

  assert response.status_code == 200, response.text
  assert response.json() == {
    "capability": "session.driver",
    "model_key": None,
    "effort": None,
  }
  assert store.get(
    tenant_id="test-product",
    actor_id="101",
    capability_id="session.driver",
  ) is None


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
  assert response.headers["cache-control"] == "private, no-store"
  session = app.state.auth.session_store.get_session(response.json()["session_id"])
  assert session.channel == "cli"
  assert session.tenant_id == "test-product"
  assert session.session_credential_handle is not None
  assert session.session_credential_handle.provider == "anthropic"
  assert session.session_credential_handle.principal == "service"
  assert session.session_credential_handle.actor_id is None
  assert session.allow_service_for_interactive is True


def test_channel_omitted_at_init_succeeds() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="mcp")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "mcp-key"})

  assert response.status_code == 200
  session = app.state.auth.session_store.get_session(response.json()["session_id"])
  assert session.channel == "mcp"


def test_session_created_hook_selects_initial_user_handle_before_resolver_bind() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  def _bind_user_credentials(session, _api_key: str, _payload: Any) -> None:
    session.auth_config = {
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": "user-key",
    }
    bind_session_credentials(
      session,
      tenant_id="test-product",
      credential_principal="user",
    )

  app = _make_app(_resolver, on_session_created=_bind_user_credentials)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "cli-key"})

  assert response.status_code == 200
  session = app.state.auth.session_store.get_session(response.json()["session_id"])
  assert session is not None
  assert session.session_credential_handle is not None
  assert session.session_credential_handle.principal == "user"
  assert session.session_credential_handle.actor_id == session.owner_user_id


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


def test_chat_init_returns_only_session_executable_stable_key_choices() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "cli-key"})

  assert response.status_code == 200
  choices = response.json()["capability_choices"]
  assert set(choices) == _SELECTABLE_CAPABILITIES
  assert "node.fork" not in choices
  driver = choices["session.driver"]
  assert driver["catalog_revision"] == INITIAL_MODEL_REGISTRY.revision
  assert driver["policy_revision"] == INITIAL_MODEL_SELECTION_POLICY.revision
  assert driver["selected"] == {
    "model_key": "anthropic.claude-opus-5",
    "label": "Opus 5",
    "effort": "high",
    "reason": "capability_default",
  }
  assert {choice["model_key"] for choice in driver["choices"]} == {
    "anthropic.claude-fable-5",
    "anthropic.claude-haiku-4-5",
    "anthropic.claude-mythos-5",
    "anthropic.claude-opus-5",
    "anthropic.claude-sonnet-5",
  }
  author = choices["plan.author"]
  assert author["selected"] is None
  assert [notice["code"] for notice in author["notices"]] == [
    "inherits_parent"
  ]
  assert "operator-key" not in response.text


def test_chat_init_normalizes_and_freezes_capability_selections() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={
        "api_key": "cli-key",
        "capability_selections": {
          "plan.author": {
            "model_key": "anthropic.claude-sonnet-5",
            "effort": " HIGH ",
          },
        },
      },
    )

  assert response.status_code == 200
  session = app.state.auth.session_store.get_session(response.json()["session_id"])
  assert session is not None
  assert session.capability_run_overrides == {
    "plan.author": ModelSelectionIntent(
      model_key="anthropic.claude-sonnet-5",
      effort="high",
      source="explicit_user",
    ),
  }
  with pytest.raises(TypeError):
    session.capability_run_overrides["node.verify"] = ModelSelectionIntent(  # type: ignore[index]
      model_key="anthropic.claude-opus-5",
      effort="none",
      source="explicit_user",
    )


@pytest.mark.parametrize(
  ("capability_id", "error_code"),
  [
    ("unknown.role", "unknown_capability"),
    ("session.driver", "capability_model_not_allowed"),
    ("node.fork", "capability_model_not_allowed"),
    ("node.mutate", "capability_model_not_allowed"),
  ],
)
def test_chat_init_rejects_unknown_or_unselectable_capability(
  capability_id: str,
  error_code: str,
) -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={
        "api_key": "cli-key",
        "capability_selections": {
          capability_id: {"model_key": "anthropic.claude-opus-5"},
        },
      },
    )

  assert response.status_code == 422
  assert response.json()["error_code"] == error_code
  assert response.json()["capability_id"] == capability_id
  assert app.state.auth.session_store.sessions == {}
  assert "operator-key" not in response.text


def test_chat_init_rejects_unavailable_model_before_issuing_token() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={
        "api_key": "cli-key",
        "capability_selections": {
          "plan.author": {"model_key": "anthropic.does-not-exist"},
        },
      },
    )

  assert response.status_code == 422
  assert response.json()["error_code"] == "capability_model_unavailable"
  assert "session_token" not in response.json()
  assert app.state.auth.session_store.sessions == {}


def test_chat_init_rejects_unsupported_effort_before_issuing_token() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={
        "api_key": "cli-key",
        "capability_selections": {
          "plan.author": {
            "model_key": "anthropic.claude-fable-5",
            "effort": "none",
          },
        },
      },
    )

  assert response.status_code == 422
  assert response.json()["error_code"] == "capability_effort_unsupported"
  assert "session_token" not in response.json()
  assert app.state.auth.session_store.sessions == {}


@pytest.mark.parametrize(
  "selection",
  [
    {},
    {"model_key": 123},
    {"model_key": "anthropic.claude-opus-5", "unexpected": "value"},
  ],
)
def test_chat_init_rejects_malformed_capability_selection(
  selection: dict[str, Any],
) -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={
        "api_key": "cli-key",
        "capability_selections": {"plan.author": selection},
      },
    )

  assert response.status_code == 422
  assert response.json()["error_code"] == "capability_selection_invalid"
  assert app.state.auth.session_store.sessions == {}
  assert "operator-key" not in response.text


def test_chat_init_rejects_selection_without_eligible_credential() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={
        "api_key": "cli-key",
        "capability_selections": {
          "plan.author": {
            "model_key": "openai.gpt-5-6",
            "effort": "low",
          },
        },
      },
    )

  assert response.status_code == 422
  assert response.json()["error_code"] == "credential_unavailable"
  assert response.json()["provider"] == "openai"
  assert "session_token" not in response.json()
  assert app.state.auth.session_store.sessions == {}


def test_chat_init_session_preserves_canonical_identity_metadata() -> None:
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
  assert session is not None
  assert session.user_id == "alice"
  assert session.owner_user_id == "101"
  assert session.raw_user_id == "alice"
  assert session.user_slug == "alice"
  assert session.risk_user_id == 101
  assert session.user_email == "alice@example.com"
  assert session.user_aliases == ("101", "alice", "alice@example.com")
  assert session.identity_status == "risk_user_id_authoritative"


def test_chat_init_session_stores_identity_derived_numeric_risk_user_id() -> None:
  app = _make_app(None)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "legacy-key", "user_id": "101"})

  assert response.status_code == 200
  session = app.state.auth.session_store.get_session(response.json()["session_id"])
  assert session is not None
  assert session.user_id == "101"
  assert session.owner_user_id == "101"
  assert session.raw_user_id == "101"
  assert session.risk_user_id == 101
  assert session.identity_status == "numeric_user_id"
  _verified_session, claims = app.state.auth.verify_token_with_payload(response.json()["session_token"])
  assert claims["risk_user_id"] == 101


def test_chat_init_session_stores_identity_mapped_email(monkeypatch) -> None:
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    '[{"key":"mapped-key","channel":"mcp","slug":"henry","email":"henry@example.com","risk_user_id":1,"role":"owner"}]',
  )
  app = _make_app(None)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "legacy-key", "user_id": "henry"})

  assert response.status_code == 200
  session = app.state.auth.session_store.get_session(response.json()["session_id"])
  assert session is not None
  assert session.owner_user_id == "1"
  assert session.user_email == "henry@example.com"
  assert session.user_aliases == ("1", "henry", "henry@example.com")
  _verified_session, claims = app.state.auth.verify_token_with_payload(response.json()["session_token"])
  assert claims["user_email"] == "henry@example.com"


def test_chat_init_identity_config_error_returns_http_error(monkeypatch) -> None:
  monkeypatch.setenv("GATEWAY_USER_KEYS", "{not-json")
  app = _make_app(None)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "legacy-key", "user_id": "alice"})

  assert response.status_code == 400
  assert response.json()["error"] == "credential_resolver_invalid"


def test_httpexception_from_resolver_passes_through_status() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    raise HTTPException(status_code=401, detail="API key is not mapped to a user identity")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "legacy-key", "user_id": "alice"})

  assert response.status_code == 401
  assert response.json() == {
    "error": "auth_failed",
    "message": "Authentication failed",
    "user_id": "alice",
  }


def test_generic_resolver_exception_is_value_free_in_chat_init_response() -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-chat-init-error-8f21d7"

  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    raise RuntimeError(f"resolver failed with {secret}")

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "legacy-key", "user_id": "alice"},
    )

  assert response.status_code == 500
  assert response.json() == {
    "error": "credentials_unavailable",
    "message": "Credential resolver unavailable",
    "reason": "credential_resolver_failed",
    "user_id": "alice",
  }
  assert secret not in response.text
