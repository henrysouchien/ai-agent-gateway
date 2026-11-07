from __future__ import annotations
from agent_gateway.registry_scope import RegistryScope
from gateway_test_support.run_context_test_support import run_context_for_test

# ruff: noqa: E402

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from dataclasses import replace
from model_authority.selection import TierModelsDefault
from agent_gateway import AgentRunner, McpClientManager, ToolDispatcher

from agent_gateway.auth import AuthConfig, ResolverResult
from model_authority.binding import ModelSelectionIntent
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
from model_authority.registry import ProductModelRegistry
from model_authority.schema import SCHEMA
from model_authority.selection import ProductModelSelectionPolicy
from agent_gateway.model_preferences import ModelPreferenceStore
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.session import bind_session_credentials
from gateway_test_support.control_plane_identity import fake_identity_resolver
from gateway_test_support.model_defaults import SESSION_DRIVER


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
  identity_resolver=fake_identity_resolver,
  on_session_created=None,
  model_preference_store: ModelPreferenceStore | None = None,
  model_registry=None,
  model_selection_policy=None,
):
  async def _build_chat_runtime(_session, request, _channel, _auth_manager, *, storage_root: Path | None = None):
    def _build_runner(event_log, session_id, started_at):
      return AgentRunner(
        event_log,
        ToolDispatcher(capability_bind=request.capability_execution.bind, mcp_client=McpClientManager(config_path=None), run_context=run_context_for_test(registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store'))),
        session_id,
        capability_execution=request.capability_execution,
        started_at=started_at,
      )

    return ChatRuntime(
      system_prompt="test",
      build_runner=_build_runner,
      capability_execution=request.capability_execution,
    )

  return create_gateway_app(
    GatewayServerConfig(
      identity_resolver=identity_resolver,
      tenant_id="test-product",
      model_registry=model_registry or INITIAL_MODEL_REGISTRY,
      model_selection_policy=model_selection_policy or INITIAL_MODEL_SELECTION_POLICY,
      credentials_resolver=credentials_resolver,
      model_preference_store=model_preference_store,
      build_chat_runtime=_build_chat_runtime,
      on_session_created=on_session_created,
    )
  )


@pytest.mark.parametrize("source", ["", " alice ", "042", "0", "../alice", "Alice"])
def test_chat_init_fallback_storage_owner_finding_returns_400_without_desk(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  (tmp_path / "gateway").mkdir(mode=0o700)
  app = _make_app(None, identity_resolver=None)
  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "legacy-key", "user_id": source})
  assert response.status_code == 400
  assert not app.state.auth.session_store.sessions
  assert not (tmp_path / "users").exists()


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


def _put_preference(
  tmp_path: Path,
  body: dict[str, Any],
) -> tuple[Any, ModelPreferenceStore]:
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
    response = client.put(
      "/api/model-preferences/session.driver",
      json=body,
      headers={
        "Authorization": f"Bearer {init_response.json()['session_token']}"
      },
    )
  return response, store


def test_put_preference_with_stale_revision_and_ineligible_key_names_stale_catalog(
  tmp_path: Path,
) -> None:
  response, store = _put_preference(
    tmp_path,
    {
      # No openai credential is eligible in this session; under a stale
      # observed revision the typed answer is the stale-catalog refresh, not
      # a bare unavailable-key error.
      "model_key": "openai.gpt-6-1-sol",
      "catalog_revision": "1999-01-01.0",
    },
  )

  assert response.status_code == 422, response.text
  payload = response.json()
  assert payload["error_code"] == "capability_catalog_stale"
  assert payload["catalog_revision"] == INITIAL_MODEL_REGISTRY.revision
  assert "anthropic.claude-sonnet-5-5" in payload["eligible_model_keys"]
  assert store.get(
    tenant_id="test-product",
    actor_id="101",
    capability_id="session.driver",
  ) is None


def test_put_preference_with_stale_revision_and_still_eligible_key_is_accepted(
  tmp_path: Path,
) -> None:
  response, store = _put_preference(
    tmp_path,
    {
      "model_key": "anthropic.claude-sonnet-5-5",
      "effort": "high",
      "catalog_revision": "1999-01-01.0",
    },
  )

  assert response.status_code == 200, response.text
  assert response.json() == {
    "capability": "session.driver",
    "model_key": "anthropic.claude-sonnet-5-5",
    "effort": "high",
  }
  stored = store.get(
    tenant_id="test-product",
    actor_id="101",
    capability_id="session.driver",
  )
  assert stored is not None
  assert stored.model_key == "anthropic.claude-sonnet-5-5"


def test_put_preference_with_current_revision_keeps_specific_refusal(
  tmp_path: Path,
) -> None:
  response, _store = _put_preference(
    tmp_path,
    {
      "model_key": "openai.gpt-6-1-sol",
      "catalog_revision": INITIAL_MODEL_REGISTRY.revision,
    },
  )

  assert response.status_code == 422, response.text
  assert response.json()["error_code"] == "capability_model_unavailable"


def test_put_preference_with_stale_revision_and_unsupported_effort_names_stale_catalog(
  tmp_path: Path,
) -> None:
  response, store = _put_preference(
    tmp_path,
    {
      # opus-5-5 does not support "none"; under a stale observed revision the
      # supported-effort set may have changed, so the answer is the typed
      # stale-catalog refresh.
      "model_key": "anthropic.claude-opus-5-5",
      "effort": "none",
      "catalog_revision": "1999-01-01.0",
    },
  )

  assert response.status_code == 422, response.text
  assert response.json()["error_code"] == "capability_catalog_stale"
  assert store.get(
    tenant_id="test-product",
    actor_id="101",
    capability_id="session.driver",
  ) is None


def _stale_preference_deployment(
  lifecycle: str,
) -> tuple[ProductModelRegistry, ProductModelSelectionPolicy]:
  """A later deployment where xai.grok-4-7 left the active lifecycle."""

  entries = dict(INITIAL_MODEL_REGISTRY.models)
  entries["xai.grok-4-7"] = replace(
    entries["xai.grok-4-7"],
    lifecycle=lifecycle,
    capabilities={
      capability_id: "internal"
      for capability_id in entries["xai.grok-4-7"].capabilities
    },
  )
  registry = ProductModelRegistry(
    schema=SCHEMA,
    revision=f"post-{lifecycle}",
    models=entries,
  )
  capabilities = {
    capability_id: (
      replace(
        policy,
        allowed_model_keys=policy.allowed_model_keys - {"xai.grok-4-7"},
      )
      if "xai.grok-4-7" in policy.allowed_model_keys
      else policy
    )
    for capability_id, policy in INITIAL_MODEL_SELECTION_POLICY.capabilities.items()
  }
  # A deployment that retires a model drops it from every role chain and tier
  # ceiling too: load admission refuses an unavailable fallback entry.
  tiers = {
    name: replace(
      tier,
      allowed_model_keys=(
        None if tier.allowed_model_keys is None
        else tier.allowed_model_keys - {"xai.grok-4-7"}
      ),
      fallback=(
        None if tier.fallback is None
        else tuple(entry for entry in tier.fallback if entry.model != "xai.grok-4-7")
      ),
    )
    for name, tier in INITIAL_MODEL_SELECTION_POLICY.tiers.items()
  }
  presets = {}
  for preset_name, role_map in INITIAL_MODEL_SELECTION_POLICY.presets.items():
    filtered: dict[str, Any] = {}
    for role_name, default in role_map.items():
      if isinstance(default, TierModelsDefault):
        filtered[role_name] = replace(
          default,
          models=tuple(key for key in default.models if key != "xai.grok-4-7"),
          fallback=(
            None if default.fallback is None
            else tuple(entry for entry in default.fallback if entry.model != "xai.grok-4-7")
          ),
        )
      else:
        filtered[role_name] = default
    presets[preset_name] = filtered
  selection_policy = replace(
    INITIAL_MODEL_SELECTION_POLICY,
    revision=f"post-{lifecycle}",
    capabilities=capabilities,
    tiers=tiers,
    presets=presets,
  )
  selection_policy.admit_registry(registry)
  return registry, selection_policy


@pytest.mark.parametrize(
  ("lifecycle", "reason"),
  [
    ("deprecated", "model_deprecated"),
    ("revoked", "model_revoked"),
    ("hidden", "model_hidden"),
  ],
)
def test_init_with_stale_saved_preference_binds_default_and_names_reason(
  tmp_path: Path,
  lifecycle: str,
  reason: str,
) -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  store = ModelPreferenceStore(tmp_path / "model-preferences.sqlite3")
  stored = store.put(
    tenant_id="test-product",
    actor_id="101",
    capability_id="session.driver",
    model_key="xai.grok-4-7",
    effort="medium",
  )
  registry, selection_policy = _stale_preference_deployment(lifecycle)
  app = _make_app(
    _resolver,
    model_preference_store=store,
    model_registry=registry,
    model_selection_policy=selection_policy,
  )

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "context": {"channel": "cli"}},
    )

  assert response.status_code == 200, response.text
  driver = response.json()["capability_choices"]["session.driver"]
  assert driver["selected"] is not None
  assert driver["selected"]["model_key"] == SESSION_DRIVER.model_key
  assert driver["selected"]["reason"] == "capability_default"
  notices = {notice["code"]: notice for notice in driver["notices"]}
  notice = notices["saved_preference_not_applied"]
  assert notice["model_key"] == "xai.grok-4-7"
  assert notice["reason"] == reason
  assert "it remains saved until replaced or cleared" in notice["message"]
  assert "xai.grok-4-7" not in {
    choice["model_key"] for choice in driver["choices"]
  }
  # The stored preference row is retained, never repaired or deleted.
  assert store.get(
    tenant_id="test-product",
    actor_id="101",
    capability_id="session.driver",
  ) == stored


def test_init_with_unsupported_effort_preference_binds_default(
  tmp_path: Path,
) -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  store = ModelPreferenceStore(tmp_path / "model-preferences.sqlite3")
  stored = store.put(
    tenant_id="test-product",
    actor_id="101",
    capability_id="session.driver",
    model_key="anthropic.claude-fable-5-1",
    effort="none",
  )
  app = _make_app(_resolver, model_preference_store=store)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "context": {"channel": "cli"}},
    )

  assert response.status_code == 200, response.text
  driver = response.json()["capability_choices"]["session.driver"]
  assert driver["selected"] is not None
  assert driver["selected"]["model_key"] == SESSION_DRIVER.model_key
  notices = {notice["code"]: notice for notice in driver["notices"]}
  notice = notices["saved_preference_not_applied"]
  assert notice["model_key"] == "anthropic.claude-fable-5-1"
  assert notice["reason"] == "effort_unsupported"
  assert store.get(
    tenant_id="test-product",
    actor_id="101",
    capability_id="session.driver",
  ) == stored


def test_init_with_mistyped_stored_driver_preference_refuses_with_a_usable_next_call(
  tmp_path: Path,
) -> None:
  import sqlite3

  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  store = ModelPreferenceStore(tmp_path / "model-preferences.sqlite3")
  store.put(
    tenant_id="test-product",
    actor_id="101",
    capability_id="session.driver",
    model_key="anthropic.claude-sonnet-5",
    effort="high",
  )
  with sqlite3.connect(store.path) as connection:
    connection.execute("UPDATE model_preferences SET model_key=x'3432'")
  app = _make_app(_resolver, model_preference_store=store)

  with TestClient(app) as client:
    refused = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "context": {"channel": "cli"}},
    )
    assert refused.status_code == 409, refused.text
    body = refused.json()
    assert body["error_code"] == "stored_model_preference_invalid"
    assert body["scope"] == "preference"
    assert body["column"] == "model_key"
    assert body["message"] == "stored preference column model_key must be text, got bytes"
    assert body["next_call"].startswith("DELETE /api/model-preferences/session.driver ")
    assert "b'42'" not in refused.text
    assert refused.headers["Cache-Control"] == "private, no-store"

    # The refusal's own session token is the credential its next_call needs.
    repaired = client.delete(
      "/api/model-preferences/session.driver",
      headers={"Authorization": f"Bearer {body['session_token']}"},
    )
    assert repaired.status_code == 200, repaired.text
    retry = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "context": {"channel": "cli"}},
    )
  assert retry.status_code == 200, retry.text


def test_init_with_mistyped_stored_user_preset_names_the_preset_repair(
  tmp_path: Path,
) -> None:
  import sqlite3

  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  store = ModelPreferenceStore(tmp_path / "model-preferences.sqlite3")
  store.put_user_preset(
    tenant_id="test-product", actor_id="101", name="economy", policy_revision="candidate-1",
  )
  with sqlite3.connect(store.path) as connection:
    connection.execute("UPDATE user_model_presets SET policy_revision=x'3432'")
  app = _make_app(_resolver, model_preference_store=store)

  with TestClient(app) as client:
    refused = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "context": {"channel": "cli"}},
    )

  assert refused.status_code == 409, refused.text
  body = refused.json()
  assert body["scope"] == "user_preset"
  assert body["column"] == "policy_revision"
  assert body["next_call"].startswith("DELETE /api/model-presets/self ")


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


def test_chat_init_reports_unknown_model_entitlements_the_resolver_supplied() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    result = _resolver_result(channel="cli")
    return replace(
      result,
      model_entitled_capabilities=result.model_entitled_capabilities
      | {"not.a.capability"},
    )

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "cli-key"})

  assert response.status_code == 200, response.text
  assert response.json()["unknown_model_entitled_capabilities"] == ["not.a.capability"]


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
    "model_key": SESSION_DRIVER.model_key,
    "label": INITIAL_MODEL_REGISTRY.require(SESSION_DRIVER.model_key).label,
    "effort": SESSION_DRIVER.effort,
    "reason": "capability_default",
  }
  # Only active session.driver keys this anthropic-only session can select.
  driver_allowed = INITIAL_MODEL_SELECTION_POLICY.capabilities[
    "session.driver"
  ].allowed_model_keys
  anthropic_keys = {
    key for key in driver_allowed
    if INITIAL_MODEL_REGISTRY.require(key).provider == "anthropic"
    and INITIAL_MODEL_REGISTRY.require(key).lifecycle == "active"
  }
  assert anthropic_keys and anthropic_keys != driver_allowed
  assert {choice["model_key"] for choice in driver["choices"]} == anthropic_keys
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
            "model_key": "anthropic.claude-sonnet-5-5",
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
      model_key="anthropic.claude-sonnet-5-5",
      effort="high",
      source="explicit_user",
    ),
  }
  with pytest.raises(TypeError):
    session.capability_run_overrides["node.verify"] = ModelSelectionIntent(
      model_key="anthropic.claude-opus-5-5",
      effort="high",
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
          capability_id: {"model_key": "anthropic.claude-opus-5-5"},
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
            "model_key": "anthropic.claude-fable-5-1",
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
    {"model_key": "anthropic.claude-opus-5-5", "unexpected": "value"},
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
            "model_key": "openai.gpt-6-1-sol",
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


_RUN_PRESET_POLICY = replace(
  INITIAL_MODEL_SELECTION_POLICY,
  presets={
    "eval_economy": {
      "default": TierModelsDefault(("anthropic.claude-sonnet-5-5",), "low"),
    },
  },
)


def test_chat_init_model_preset_freezes_and_catalogue_selects_it() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver, model_selection_policy=_RUN_PRESET_POLICY)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "model_preset": "eval_economy"},
    )

  assert response.status_code == 200
  session = app.state.auth.session_store.get_session(response.json()["session_id"])
  assert session is not None
  assert session.model_preset == "eval_economy"
  selected = response.json()["capability_choices"]["session.driver"]["selected"]
  assert (selected["model_key"], selected["effort"], selected["reason"]) == (
    "anthropic.claude-sonnet-5-5",
    "low",
    "internal_policy",
  )


def test_chat_init_refuses_unpublished_model_preset_before_issuing_token() -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver, model_selection_policy=_RUN_PRESET_POLICY)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "model_preset": "not_published"},
    )

  assert response.status_code == 422
  assert response.json()["error_code"] == "capability_selection_invalid"
  assert "session_token" not in response.json()
  assert app.state.auth.session_store.sessions == {}


@pytest.mark.parametrize("model_preset", ["", "   ", 7])
def test_chat_init_rejects_blank_or_non_text_model_preset(model_preset: Any) -> None:
  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result(channel="cli")

  app = _make_app(_resolver, model_selection_policy=_RUN_PRESET_POLICY)

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "cli-key", "model_preset": model_preset},
    )

  assert response.status_code == 422
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


@pytest.mark.parametrize(
  ("reason", "expected_reason", "next_call_prefix"),
  [
    (
      "not_configured",
      "web_subject_verification_not_configured",
      "set GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS",
    ),
    (
      "replay_cache_saturated",
      "web_subject_verification_replay_cache_saturated",
      "retry POST /api/chat/init",
    ),
  ],
)
def test_subject_verification_unavailable_reaches_the_caller_with_reason_and_next_call(
  reason: str,
  expected_reason: str,
  next_call_prefix: str,
) -> None:
  from agent_gateway.subject_assertion import SubjectVerificationUnavailable

  async def _resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    raise SubjectVerificationUnavailable(reason)  # type: ignore[arg-type]

  app = _make_app(_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "web-key", "user_id": "alice"})

  assert response.status_code == 503
  body = response.json()
  assert body["error"] == "credentials_unavailable"
  assert body["reason"] == expected_reason
  assert body["user_id"] == "alice"
  assert body["next_call"].startswith(next_call_prefix)


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


def test_capability_choices_refresh_keeps_the_authenticated_session(tmp_path: Path) -> None:
  async def resolver(_api_key: str, _init_request: Any) -> ResolverResult:
    return _resolver_result()

  store = ModelPreferenceStore(tmp_path / "model-preferences.sqlite3")
  app = _make_app(resolver, model_preference_store=store)
  with TestClient(app) as client:
    initial = client.post("/api/chat/init", json={"api_key": "cli-key", "context": {"channel": "cli"}})
    assert initial.status_code == 200
    original = app.state.auth.session_store.get_session(initial.json()["session_id"])
    assert original is not None
    changed_choice = next(
      choice for choice in initial.json()["capability_choices"]["session.driver"]["choices"]
      if choice["model_key"] != SESSION_DRIVER.model_key
    )
    headers = {"Authorization": f"Bearer {initial.json()['session_token']}"}
    selected = client.put(
      "/api/model-preferences/session.driver",
      headers=headers, json={"model_key": changed_choice["model_key"], "effort": changed_choice["default_effort"]},
    )
    assert selected.status_code == 200
    refreshed = client.get("/api/capability-choices/session.driver", headers=headers)
    assert refreshed.status_code == 200
    assert refreshed.json()["selected"]["reason"] == "saved_preference"
    assert refreshed.json()["selected"]["model_key"] == changed_choice["model_key"]
    assert refreshed.headers["cache-control"] == "private, no-store"
    assert tuple(app.state.auth.session_store.sessions) == (original.session_id,)
    assert app.state.auth.session_store.get_session(original.session_id) is original
    assert client.get("/api/capability-choices/session.driver").status_code == 401
    from agent_gateway.server_chat_helpers import prepare_session_driver_turn
    from agent_gateway.server_models import ChatMessage, ChatTurnInputs

    prepared = prepare_session_driver_turn(
      original,
      ChatTurnInputs(
        messages=[ChatMessage(role="user", content="next turn")],
        request_id=None, context={}, metadata={}, model_key=None,
      ),
      build_chat_runtime=app.state.gateway_build_chat_runtime,
    )
    next_turn = prepared.request
    assert next_turn.capability_execution is not None
    assert next_turn.capability_execution.bind.model_key == changed_choice["model_key"]
    assert next_turn.capability_execution.bind.effort == changed_choice["default_effort"]


@pytest.mark.parametrize("provider", ["xai", "google", "kimi_code", "muse", "ollama_cloud"])
def test_r4_init_exposes_key_service_outage_as_unavailable_not_unenrolled(provider, tmp_path):
  from agent_gateway.server_chat_helpers import (
    KeyServiceUnavailable, _capability_execution_resolver_for_session, prepare_session_driver_turn,
  )
  from agent_gateway.server_models import ChatMessage, ChatTurnInputs
  from model_authority.selection import CapabilityDefault

  slot = INITIAL_MODEL_SELECTION_POLICY.capabilities["session.driver"]
  entry = next(
    entry for entry in INITIAL_MODEL_REGISTRY.models.values()
    if entry.provider == provider and entry.key in slot.allowed_model_keys
  )
  registry = replace(INITIAL_MODEL_REGISTRY, models={
    **INITIAL_MODEL_REGISTRY.models,
    entry.key: replace(entry, lifecycle="active"),
  })
  policy = replace(INITIAL_MODEL_SELECTION_POLICY, capabilities={
    **INITIAL_MODEL_SELECTION_POLICY.capabilities,
    "session.driver": replace(
      slot, default=CapabilityDefault(kind="model", model_key=entry.key, effort=entry.default_effort),
    ),
  })

  async def resolver(_api_key, _request):
    return replace(
      _resolver_result(), auth_config=AuthConfig.from_dict({
        "provider": provider, "api_key": "r4-key", "billing_mode": "metered",
      }),
      unavailable_user_key_providers=frozenset({"anthropic", "openai"}),
    )

  app = _make_app(
    resolver, model_registry=registry, model_selection_policy=policy,
    model_preference_store=ModelPreferenceStore(tmp_path / "preferences.sqlite"),
  )
  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "cli-key"})
    assert response.status_code == 200, response.text
    payload = response.json()
    driver = payload["capability_choices"]["session.driver"]
    assert driver["selected"]["model_key"] == entry.key
    assert all(not choice["model_key"].startswith(("anthropic.", "openai.")) for choice in driver["choices"])
    unavailable = {
      notice["model_key"]: notice
      for notice in driver["notices"] if notice["code"] == "key_service_unavailable"
    }
    assert {"anthropic.claude-sonnet-5-5", "openai.gpt-6-1-sol"} <= unavailable.keys()
    assert all(notice["reason"] == "key_service_unavailable" for notice in unavailable.values())
    [deep] = [role for role in driver["roles"] if role["role"] == "deep"]
    assert deep["unavailable"] == "key_service_unavailable"
    session = app.state.auth.session_store.get_session(payload["session_id"])
    assert session is not None
    assert session.unavailable_user_key_providers == frozenset({"anthropic", "openai"})
    bound_resolver = _capability_execution_resolver_for_session(
      session=session, build_chat_runtime=app.state.gateway_build_chat_runtime,
    )
    assert not {"anthropic", "openai"} & bound_resolver.auth_context.user_provider_handles.keys()
    for model_key in ("anthropic.claude-sonnet-5-5", "openai.gpt-6-1-sol"):
      selected = client.post("/api/chat/init", json={
        "api_key": "cli-key",
        "capability_selections": {"plan.author": {"model_key": model_key}},
      })
      assert selected.status_code == 422, selected.text
      assert selected.json()["reason"] == "key_service_unavailable"
      saved = client.put(
        "/api/model-preferences/session.driver", json={"model_key": model_key},
        headers={"Authorization": f"Bearer {payload['session_token']}"},
      )
      assert saved.status_code == 422, saved.text
      assert saved.json()["reason"] == "key_service_unavailable"
      with pytest.raises(KeyServiceUnavailable) as caught:
        prepare_session_driver_turn(
          session, ChatTurnInputs(
            messages=[ChatMessage(role="user", content="next")], request_id=None,
            context={}, metadata={}, model_key=model_key,
          ),
          build_chat_runtime=app.state.gateway_build_chat_runtime,
        )
      assert caught.value.reason == "key_service_unavailable"
