# ruff: noqa: E402

import asyncio
import json
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
import agent_gateway.sub_agent as sub_agent_module
from agent_gateway import EventLog, McpClientManager, ToolResultContext, create_agent
from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway._provider_utils import _resolve_provider
from agent_gateway.capability_binding import CapabilityResolutionError
from agent_gateway.commercial_authority_cache import CommercialAuthorityStateCache
from agent_gateway.commercial_authority_subscriber import CommercialAuthoritySubscriber
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
)
from agent_gateway.providers import AnthropicProvider, CodexProvider, OpenAIProvider, XAIProvider
from agent_gateway.server import ChatRequest, ChatTurnInputs
from agent_gateway.server_chat_helpers import prepare_session_driver_turn

DEFAULT_MODEL_KEY = "anthropic.claude-opus-5"
DEFAULT_ANTHROPIC_MODEL = INITIAL_MODEL_REGISTRY.require(
  DEFAULT_MODEL_KEY
).upstream_model


def _run(coro):
  return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_credential_env(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_MODE", raising=False)
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)
  monkeypatch.delenv("XAI_API_KEY", raising=False)


def _create_session(
  app,
  *,
  api_key_hash: str = "hash",
  user_id: str = "alice",
  channel: str | None = None,
  model_entitled_capabilities: frozenset[str] = frozenset({"session.driver"}),
):
  config = app.state.gateway_config
  session = app.state.auth.session_store.create_session(
    api_key_hash=api_key_hash,
    user_id=user_id,
    tenant_id=config.tenant_id,
    allow_service_for_interactive=(
      config.allow_service_credentials_for_interactive
    ),
    model_entitled_capabilities=model_entitled_capabilities,
    model_entitled_keys=config.model_selection_policy.capabilities[
      "session.driver"
    ].allowed_model_keys,
  )
  session.channel = channel
  return session


def _prepare_runtime_for_session(
  app,
  *,
  session,
  request: ChatRequest,
):
  prepared = prepare_session_driver_turn(
    session,
    ChatTurnInputs(
      messages=list(request.messages),
      request_id=request.request_id,
      context=dict(request.context),
      metadata=dict(request.metadata),
      model_key=request.model_key,
      effort=request.effort,
      catalog_revision=request.catalog_revision,
      ui_blocks_contract=request.ui_blocks_contract,
      commercial_work_start=request.commercial_work_start,
    ),
    build_chat_runtime=app.state.gateway_build_chat_runtime,
  )
  prepared.request._bind_commercial_work_start(request.commercial_work_start)
  runtime = _run(
    app.state.gateway_build_chat_runtime(
      session=session,
      request=prepared.request,
      channel=prepared.channel,
      auth_manager=app.state.auth,
    )
  )
  return runtime, prepared.request


def _build_runtime(
  app,
  *,
  request_model: str | None = None,
  model_entitled_capabilities: frozenset[str] = frozenset({"session.driver"}),
):
  session = _create_session(
    app,
    model_entitled_capabilities=model_entitled_capabilities,
  )
  runtime, _request = _prepare_runtime_for_session(
    app,
    session=session,
    request=ChatRequest(
      messages=[{"role": "user", "content": "hello"}],
      context={},
      model_key=request_model,
    ),
  )
  return session, runtime


def _build_runtime_with_request(app, *, request: ChatRequest, channel: str | None):
  session = _create_session(app, channel=channel)
  runtime, prepared_request = _prepare_runtime_for_session(
    app,
    session=session,
    request=request,
  )
  return session, runtime, prepared_request


def _service_auth_config(app) -> dict[str, object]:
  config = app.state.gateway_config
  assert config.service_auth_config_resolver is not None
  [handle] = config.service_provider_handles.values()
  materialized = config.service_auth_config_resolver(handle)
  resolved = dict(materialized.auth_config)
  for field in ("provider", "billing_mode", "rate_table_version"):
    resolved.pop(field, None)
  return resolved


def _capability_models(app) -> set[str]:
  config = app.state.gateway_config
  assert config.model_registry is not None
  assert config.model_selection_policy is not None
  default = config.model_selection_policy.capabilities["session.driver"].default
  assert default.model_key is not None
  return {config.model_registry.require(default.model_key).upstream_model}


def test_commercial_usage_producer_factory_is_request_scoped() -> None:
  produced = []

  def factory(session, request, channel):
    producer = object()
    produced.append((session.user_id, request.request_id, channel, producer))
    return producer

  app = create_agent(
    "test",
    api_key="test-key",
    commercial_usage_producer_factory=factory,
  )
  first_session = _create_session(
    app,
    api_key_hash="a",
    user_id="alice",
    channel="mcp",
  )
  second_session = _create_session(
    app,
    api_key_hash="b",
    user_id="bob",
    channel="mcp",
  )
  runtimes = []
  for session, request_id in ((first_session, "req-a"), (second_session, "req-b")):
    request = ChatRequest(
      messages=[{"role": "user", "content": "hello"}],
      context={},
      request_id=request_id,
    )
    runtime, _prepared_request = _prepare_runtime_for_session(
      app,
      session=session,
      request=request,
    )
    runtimes.append(runtime)

  runners = [
    runtime.build_runner(EventLog(), session.session_id)
    for runtime, session in zip(
      runtimes,
      (first_session, second_session),
    )
  ]
  assert [(item[0], item[1], item[2]) for item in produced] == [
    ("alice", "req-a", "mcp"),
    ("bob", "req-b", "mcp"),
  ]
  assert runners[0]._commercial_usage_producer is produced[0][3]
  assert runners[1]._commercial_usage_producer is produced[1][3]
  assert produced[0][3] is not produced[1][3]


def test_commercial_usage_factory_receives_verified_work_start_context() -> None:
  produced = []

  def factory(session, request, channel, commercial_work_start):
    produced.append((session, request, channel, commercial_work_start))
    return object()

  app = create_agent(
    "test",
    api_key="test-key",
    commercial_usage_producer_factory=factory,
  )
  request = ChatRequest(
    messages=[{"role": "user", "content": "hello"}],
    request_id="request-1",
  )
  verified_context = object()
  request._bind_commercial_work_start(verified_context)
  session, _, prepared_request = _build_runtime_with_request(
    app,
    request=request,
    channel="mcp",
  )

  assert produced == [(session, prepared_request, "mcp", verified_context)]


def test_commercial_usage_factory_preserves_legacy_positional_args_with_kwargs() -> None:
  produced = []

  def factory(a, b, c, **kwargs):
    produced.append((a, b, c, kwargs))
    return object()

  app = create_agent(
    "test",
    api_key="test-key",
    commercial_usage_producer_factory=factory,
  )
  request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
  session, _, prepared_request = _build_runtime_with_request(
    app,
    request=request,
    channel="mcp",
  )

  assert produced == [(
    session,
    prepared_request,
    "mcp",
    {"commercial_work_start": None},
  )]


def test_commercial_usage_factory_preserves_legacy_varargs_arity() -> None:
  produced = []

  def factory(*args):
    produced.append(args)
    return object()

  app = create_agent(
    "test",
    api_key="test-key",
    commercial_usage_producer_factory=factory,
  )
  request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
  session, _, prepared_request = _build_runtime_with_request(
    app,
    request=request,
    channel="mcp",
  )

  assert produced == [(session, prepared_request, "mcp")]


def test_default_off_chat_request_accepts_large_benign_context() -> None:
  request = ChatRequest(
    messages=[{"role": "user", "content": "hello"}],
    context={"cells": [{"value": index} for index in range(10_001)]},
  )

  assert len(request.context["cells"]) == 10_001


def test_commercial_usage_shipper_runs_for_app_lifecycle() -> None:
  started = threading.Event()
  stopped = threading.Event()

  class Shipper:
    async def run_forever(self, stop):
      started.set()
      await stop.wait()
      stopped.set()

  app = create_agent("test", commercial_usage_shipper=Shipper())
  with TestClient(app):
    assert started.wait(timeout=1)
    assert not stopped.is_set()
  assert stopped.wait(timeout=1)


def test_commercial_reconciliation_shipper_runs_for_app_lifecycle() -> None:
  started = threading.Event()
  stopped = threading.Event()

  class Shipper:
    async def run_forever(self, stop):
      started.set()
      await stop.wait()
      stopped.set()

  app = create_agent(
    "test", commercial_usage_reconciliation_shipper=Shipper()
  )
  with TestClient(app):
    assert started.wait(timeout=1)
    assert not stopped.is_set()
  assert stopped.wait(timeout=1)


def test_commercial_authority_additive_v1_catches_up_before_easy_startup(
  tmp_path: Path,
) -> None:
  class Client:
    environment = "dev"

    def fetch(self, cursor):
      if cursor == 0:
        return {
          "schema_version": 1,
          "events": [{
            "sequence_id": 1,
            "environment": "dev",
            "kind": "entitlement",
            "commercial_account_id": 7,
            "entitlement_revision": 2,
            "context_id": None,
            "token_id": None,
            "occurred_at": "2026-08-19T12:00:00Z",
            "producer_extension": "compatible",
          }],
          "next_sequence": 1,
          "high_water_sequence": 1,
          "envelope_extension": "compatible",
        }
      return {
        "schema_version": 1,
        "events": [],
        "next_sequence": cursor,
        "high_water_sequence": cursor,
      }

  cursor_path = tmp_path / "authority.cursor.json"
  subscriber = CommercialAuthoritySubscriber(
    client=Client(),
    cache=CommercialAuthorityStateCache(
      lambda _: pytest.fail("loader must not run")
    ),
    cursor_path=cursor_path,
  )
  app = create_agent(
    "test",
    commercial_authority_subscriber=subscriber,
  )

  with TestClient(app):
    assert json.loads(cursor_path.read_text()) == {"sequence": 1}
    assert subscriber.health(max_staleness_seconds=30)["ok"] is True


def _collect_sse_events(response) -> list[dict]:
  events = []
  for line in response.iter_lines():
    if line.startswith("data: "):
      events.append(_unwrap_sse_payload(json.loads(line[6:])))
  return events


def _unwrap_sse_payload(payload: dict) -> dict:
  candidate = payload.get("event")
  if isinstance(payload.get("seq"), int) and isinstance(candidate, dict) and isinstance(candidate.get("type"), str):
    return candidate
  return payload


def _write_skill(skills_dir: Path, name: str, body: str) -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  (skills_dir / f"{name}.md").write_text(body, encoding="utf-8")


def test_create_agent_exposes_default_routes_and_open_defaults() -> None:
  app = create_agent("test")

  route_paths = {route.path for route in app.routes}
  assert "/api/chat/init" in route_paths
  assert "/api/chat" in route_paths
  assert "/api/health" in route_paths

  config = app.state.gateway_config
  auth_config = _service_auth_config(app)
  assert auth_config["auth_mode"] == "api"
  assert auth_config["api_key"] == ""
  assert auth_config["auth_token"] == ""
  assert "model" not in auth_config
  assert "effort" not in auth_config
  assert config.cors_origins == ["*"]
  assert _capability_models(app) == {DEFAULT_ANTHROPIC_MODEL}
  assert config.tenant_id == "agent-gateway.easy"
  assert config.allow_service_credentials_for_interactive is True
  assert config.model_selection_policy is not None
  driver_policy = config.model_selection_policy.capabilities[
    "session.driver"
  ]
  assert driver_policy.default.model_key == DEFAULT_MODEL_KEY
  assert driver_policy.allowed_model_keys == frozenset({DEFAULT_MODEL_KEY})
  assert len(app.state.auth._secret) == 64


def test_create_agent_uses_explicit_api_key() -> None:
  app = create_agent("test", api_key="sk-123")
  assert _service_auth_config(app) == {
    "auth_mode": "api",
    "api_key": "sk-123",
    "auth_token": "",
    "max_tokens": 16000,
  }


@pytest.mark.parametrize("value", ["", "   "])
def test_create_agent_blank_api_key_falls_back_to_env(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  app = create_agent("test", api_key=value)
  assert _service_auth_config(app)["auth_mode"] == "api"
  assert _service_auth_config(app)["api_key"] == "env-key"


def test_create_agent_uses_explicit_auth_token() -> None:
  app = create_agent("test", auth_token="tok-123")
  assert _service_auth_config(app) == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok-123",
    "max_tokens": 16000,
  }


def test_create_agent_with_credentials_resolver_skips_env_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
  async def _resolver(_api_key: str, _init_request):
    return ResolverResult(
      user_id="alice",
      channel="excel",
      auth_config=AuthConfig.from_dict(
        {
          "provider": "anthropic",
          "billing_mode": "byok",
          "api_key": "resolver-key",
          "model": DEFAULT_ANTHROPIC_MODEL,
          "max_tokens": 16000,
        }
      ),
      credential_principal="service",
      allow_service_for_interactive=True,
      risk_user_id=101,
      role="owner",
    )

  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  app = create_agent("test", credentials_resolver=_resolver)

  assert app.state.gateway_config.credentials_resolver is _resolver
  assert _service_auth_config(app) == {
    "auth_mode": "api",
    "api_key": "",
    "auth_token": "",
    "max_tokens": 16000,
  }


def test_create_agent_accepts_model_free_credential_material(monkeypatch: pytest.MonkeyPatch) -> None:
  def _fake_resolve_provider(*args, **kwargs):
    _ = args, kwargs
    return AnthropicProvider(), "anthropic", {
      "auth_mode": "api",
      "api_key": "resolver-key",
      "max_tokens": 16000,
    }

  monkeypatch.setattr("agent_gateway.easy._resolve_provider", _fake_resolve_provider)

  app = create_agent("test")
  assert _service_auth_config(app) == {
    "auth_mode": "api",
    "api_key": "resolver-key",
    "max_tokens": 16000,
  }
  _session, runtime = _build_runtime(app)
  assert runtime.capability_bind.model_key == DEFAULT_MODEL_KEY


def test_create_agent_budget_thinking_alias_binds_exact_high_effort() -> None:
  app = create_agent(
    "test",
    model_key="anthropic.claude-opus-5",
    effort="high",
    api_key="test-key",
  )

  _session, runtime = _build_runtime(app)
  policy = app.state.gateway_config.model_selection_policy.capabilities[
    "session.driver"
  ]
  entry = app.state.gateway_config.model_registry.require(
    policy.default.model_key
  )

  assert "high" in entry.supported_efforts
  assert policy.default.effort == "high"
  assert runtime.capability_bind is not None
  assert runtime.capability_bind.effort == "high"


def test_easy_policy_carries_base_revision_not_caller_choice() -> None:
  import re as _re

  from agent_gateway.easy import _easy_model_selection_policy
  from agent_gateway.model_registry import INITIAL_MODEL_SELECTION_POLICY

  entry, policy = _easy_model_selection_policy(
    model_key="anthropic.claude-haiku-4-5",
    effort="none",
  )

  # Revisions are configuration provenance: the derived policy carries the
  # base revision verbatim and never encodes the caller's model/effort.
  assert policy.revision == INITIAL_MODEL_SELECTION_POLICY.revision
  assert entry.key not in policy.revision
  assert _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", policy.revision)

  # The caller's choice is expressed as the derived session.driver default,
  # which resolution surfaces as selection_source="capability_default".
  driver = policy.capabilities["session.driver"]
  assert driver.default.kind == "model"
  assert driver.default.model_key == entry.key
  assert driver.default.effort == "none"
  assert driver.allowed_model_keys == frozenset({entry.key})


def test_create_agent_refuses_unsupported_configured_effort() -> None:
  with pytest.raises(ValueError, match="effort 'high' is not supported"):
    create_agent(
      "test",
      model_key="anthropic.claude-haiku-4-5",
      effort="high",
      api_key="test-key",
      max_tokens=1024,
    )


def test_create_agent_threads_request_metadata_to_runner() -> None:
  app = create_agent("test")
  request = ChatRequest(
    messages=[{"role": "user", "content": "hello"}],
    request_id="req-123",
    context={"channel": "web"},
  )
  config = app.state.gateway_config
  session = app.state.auth.session_store.create_session(
    api_key_hash="hash",
    user_id="alice",
    auth_config={
      "provider": "anthropic",
      "api_key": "k",
      "billing_mode": "metered",
    },
    tenant_id=config.tenant_id,
    credential_principal="user",
    allow_service_for_interactive=(
      config.allow_service_credentials_for_interactive
    ),
    model_entitled_capabilities=frozenset({"session.driver"}),
    model_entitled_keys=config.model_selection_policy.capabilities[
      "session.driver"
    ].allowed_model_keys,
  )
  session.channel = "web"
  runtime, _prepared_request = _prepare_runtime_for_session(
    app,
    session=session,
    request=request,
  )

  runner = runtime.build_runner(EventLog(), session.session_id)

  assert runner._request_id == "req-123"
  assert runner._usage_user_id == "alice"
  assert runner._billing_mode == "metered"
  assert runner._channel == "web"


def test_create_agent_openai_provider_string_uses_openai_defaults() -> None:
  app = create_agent(
    "test",
    provider="openai",
    model_key="openai.gpt-5-6",
  )

  config = app.state.gateway_config
  assert isinstance(config.default_provider, OpenAIProvider)
  assert _service_auth_config(app) == {
    "max_tokens": 16000,
  }
  assert _capability_models(app) == {"gpt-5.6"}


def test_create_agent_codex_provider_string_uses_codex_defaults() -> None:
  app = create_agent(
    "test",
    provider="codex",
    model_key="codex.gpt-5-6-terra",
  )

  config = app.state.gateway_config
  assert isinstance(config.default_provider, CodexProvider)
  assert _service_auth_config(app) == {
    "max_tokens": 16000,
  }
  assert _capability_models(app) == {"gpt-5.6-terra"}


def test_create_agent_xai_provider_string_uses_xai_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
  app = create_agent(
    "test",
    provider="xai",
    model_key="xai.grok-4-5",
  )

  config = app.state.gateway_config
  assert isinstance(config.default_provider, XAIProvider)
  auth_config = _service_auth_config(app)
  assert config.default_provider.has_active_credential(auth_config)
  assert auth_config == {
    "auth_mode": "api",
    "api_key": "xai-test-key",
    "auth_token": "",
    "max_tokens": 16000,
  }
  assert _capability_models(app) == {"grok-4.5"}


def test_create_agent_accepts_provider_instance_when_model_is_explicit() -> None:
  provider = OpenAIProvider()
  app = create_agent(
    "test",
    provider=provider,
    model_key="openai.gpt-5-6",
  )

  config = app.state.gateway_config
  assert config.default_provider is provider
  assert _service_auth_config(app) == {
    "max_tokens": 16000,
  }
  assert _capability_models(app) == {"gpt-5.6"}


def test_create_agent_provider_instance_must_match_model_key() -> None:
  with pytest.raises(ValueError, match="does not match model key"):
    create_agent("test", provider=OpenAIProvider())


def test_create_agent_unknown_provider_raises_value_error() -> None:
  with pytest.raises(ValueError, match="does not match model key"):
    create_agent("test", provider="unknown")


def test_create_agent_invalid_provider_type_raises_type_error() -> None:
  with pytest.raises(ValueError, match="does not match model key"):
    create_agent("test", provider=123)  # type: ignore[arg-type]


def test_create_agent_openai_snapshots_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
  app = create_agent(
    "test",
    provider="openai",
    model_key="openai.gpt-5-6",
  )
  assert _service_auth_config(app) == {
    "auth_mode": "api",
    "api_key": "env-openai-key",
    "auth_token": "",
    "max_tokens": 16000,
  }


def test_create_agent_openai_uses_explicit_api_key_and_provider_config() -> None:
  app = create_agent(
    "test",
    provider="openai",
    model_key="openai.gpt-5-6",
    api_key="sk-openai",
    auth_token="ignored",
    provider_config={
      "base_url": "https://custom.example/v1",
      "compat": {"streaming": True},
    },
  )
  assert _service_auth_config(app) == {
    "auth_mode": "api",
    "api_key": "sk-openai",
    "auth_token": "",
    "max_tokens": 16000,
    "base_url": "https://custom.example/v1",
    "compat": {"streaming": True},
  }


def test_create_agent_openai_uses_explicit_auth_token() -> None:
  app = create_agent(
    "test",
    provider="openai",
    model_key="openai.gpt-5-6",
    auth_token="oat-xxx",
  )

  assert _service_auth_config(app) == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "oat-xxx",
    "max_tokens": 16000,
  }


def test_create_agent_codex_uses_explicit_auth_token() -> None:
  app = create_agent(
    "test",
    provider="codex",
    model_key="codex.gpt-5-6-terra",
    auth_token="oat-codex",
  )

  assert _service_auth_config(app) == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "oat-codex",
    "max_tokens": 16000,
  }


def test_resolve_provider_openai_auth_config_with_oauth() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "openai",
    "gpt-5.6",
    None,
    None,
    None,
    auth_config={"auth_mode": "oauth", "auth_token": "tok"},
  )

  assert config == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
    "max_tokens": 16000,
  }


def test_resolve_provider_openai_auth_config_infers_mode() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "openai",
    "gpt-5.6",
    None,
    None,
    None,
    auth_config={"auth_token": "tok"},
  )

  assert config == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
    "max_tokens": 16000,
  }


def test_resolve_provider_openai_auth_config_plain_passthrough() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "openai",
    "gpt-5.6",
    None,
    None,
    None,
    auth_config={"api_key": "sk-x"},
  )

  assert config == {
    "auth_mode": "api",
    "api_key": "sk-x",
    "auth_token": "",
    "max_tokens": 16000,
  }


def test_resolve_provider_codex_auth_config_with_oauth() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "codex",
    "gpt-5.6-terra",
    None,
    None,
    None,
    auth_config={"auth_mode": "oauth", "auth_token": "tok"},
  )

  assert config == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
    "max_tokens": 16000,
  }


def test_create_agent_blank_auth_token_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  app = create_agent("test", auth_token="   ")
  assert _service_auth_config(app)["auth_mode"] == "oauth"
  assert _service_auth_config(app)["auth_token"] == "env-token"


def test_create_agent_no_credentials_refuses_before_stub_response() -> None:
  app = create_agent("test")

  with TestClient(app) as client:
    init_response = client.post("/api/chat/init", json={"api_key": "any-key", "user_id": "test-user"})
    assert init_response.status_code == 200
    token = init_response.json()["session_token"]

    with client.stream(
      "POST",
      "/api/chat",
      headers={"Authorization": f"Bearer {token}"},
      json={
        "messages": [{"role": "user", "content": "Hello gateway"}],
        "context": {},
      },
    ) as response:
      response.read()
      assert response.status_code == 400
      assert response.json()["error_code"] == "default_not_eligible"


def test_create_agent_valid_api_keys_and_jwt_secret() -> None:
  jwt_secret = "easy-test-secret-with-at-least-32-bytes"
  app = create_agent("test", valid_api_keys={"k1"}, jwt_secret=jwt_secret)
  assert app.state.auth._secret == jwt_secret

  with TestClient(app) as client:
    assert client.post("/api/chat/init", json={"api_key": "bad"}).status_code == 401
    assert client.post("/api/chat/init", json={"api_key": "k1", "user_id": "test-user"}).status_code == 200


def test_create_agent_inline_mcp_uses_inline_only_config_and_builtin_filtering() -> None:
  async def _tool(_tool_input, **_kwargs):
    return {"ok": True}, None

  app = create_agent(
    "test",
    mcp_servers={"filesystem": {"command": "npx", "args": ["-y", "server"]}},
    tool_handlers={"local_tool": _tool},
    code_execution=True,
  )

  mcp_client = app.state.gateway_config.mcp_client
  assert isinstance(mcp_client, McpClientManager)
  assert mcp_client._config_path is None
  assert mcp_client._inline_servers == {"filesystem": {"command": "npx", "args": ["-y", "server"]}}
  assert mcp_client._builtin_tool_names >= {"local_tool", "code_execute", "code_execute_status"}


def test_create_agent_skills_dir_registers_run_agent_handler_tool_def_and_builtin_name(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", "Research deeply.")

  app = create_agent(
    "test",
    api_key="test-key",
    skills_dir=skills_dir,
    mcp_servers={"filesystem": {"command": "npx", "args": ["-y", "server"]}},
  )

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)
  tool_defs = runtime.get_tool_definitions()

  assert runner._gateway_session is session
  assert "run_agent" in runner._dispatcher._local
  assert "get_background_result" in runner._dispatcher._local
  assert "send_message" in runner._dispatcher._local
  assert {tool["name"] for tool in tool_defs} >= {"run_agent", "get_background_result", "send_message"}
  assert app.state.gateway_config.mcp_client._builtin_tool_names >= {"run_agent", "get_background_result", "send_message"}


def test_create_agent_forwards_outputs_dir(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  skills_dir = tmp_path / "skills"
  outputs_dir = tmp_path / "outputs"
  skills_dir.mkdir(parents=True, exist_ok=True)
  captured: dict[str, object] = {}

  async def _fake_run_agent(_tool_input, **_kwargs):
    return {"response": "ok"}, None

  def _fake_make_run_agent_handler(*args, **kwargs):
    _ = args
    captured["kwargs"] = kwargs
    return _fake_run_agent

  monkeypatch.setattr(sub_agent_module, "make_run_agent_handler", _fake_make_run_agent_handler)

  app = create_agent(
    "test",
    api_key="test-key",
    skills_dir=skills_dir,
    outputs_dir=outputs_dir,
  )

  _build_runtime(app)

  assert captured["kwargs"]["outputs_dir"] == outputs_dir


def test_create_agent_forwards_skill_state_file(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  skills_dir = tmp_path / "skills"
  state_file = tmp_path / "skill_state.json"
  skills_dir.mkdir(parents=True, exist_ok=True)
  captured: dict[str, object] = {}

  async def _fake_run_agent(_tool_input, **_kwargs):
    return {"response": "ok"}, None

  def _fake_make_run_agent_handler(*args, **kwargs):
    _ = args
    captured["kwargs"] = kwargs
    return _fake_run_agent

  monkeypatch.setattr(sub_agent_module, "make_run_agent_handler", _fake_make_run_agent_handler)

  app = create_agent(
    "test",
    api_key="test-key",
    skills_dir=skills_dir,
    skill_state_file=state_file,
  )

  _build_runtime(app)

  store = captured["kwargs"]["skill_state_store"]
  assert store.state_file == state_file


def test_create_agent_forwards_needs_approval_and_cache_denylist() -> None:
  async def _tool(_tool_input, **_kwargs):
    return {"ok": True}, None

  app = create_agent(
    "test",
    api_key="test-key",
    tool_handlers={"execute_trade": _tool},
    needs_approval=lambda name, _tool_input=None, _qualifier="": name == "execute_trade",
    session_cache_denied_tools=frozenset({"execute_trade"}),
  )

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)
  dispatcher = runner._dispatcher

  assert dispatcher._needs_approval("execute_trade", {}, "") is True
  assert dispatcher._session_cache_denied == frozenset({"execute_trade"})


def test_create_agent_skills_dir_does_not_duplicate_user_run_agent_tool_definition(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", "Research deeply.")
  tool_def = {
    "name": "run_agent",
    "description": "custom",
    "input_schema": {"type": "object", "properties": {}},
  }

  app = create_agent(
    "test",
    api_key="test-key",
    skills_dir=skills_dir,
    tool_definitions=[tool_def],
  )

  _session, runtime = _build_runtime(app)

  tool_defs = runtime.get_tool_definitions()
  assert [tool for tool in tool_defs if tool["name"] == "run_agent"] == [tool_def]
  assert any(tool["name"] == "get_background_result" for tool in tool_defs)
  assert any(tool["name"] == "send_message" for tool in tool_defs)


def test_create_agent_skills_dir_respects_custom_run_agent_handler_override(tmp_path: Path) -> None:
  async def _custom_run_agent(_tool_input, **_kwargs):
    return {"override": True}, None

  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", "Research deeply.")
  app = create_agent(
    "test",
    api_key="test-key",
    skills_dir=skills_dir,
    tool_handlers={"run_agent": _custom_run_agent},
  )

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)

  assert runner._dispatcher._local["run_agent"] is _custom_run_agent
  assert runtime.get_tool_definitions() == []


def test_create_agent_skills_dir_dispatches_bound_child_execution(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(
    skills_dir,
    "deep-research",
    """---
name: deep-research
version: '1.0'
agent_callable: true
agent_description: Exercise exact child execution binding.
mutation_mode: read_only
allowed_tools: []
mcp_tools: {}
semantic_metadata:
  allowed_effects: []
  tool_refs: []
  capability_requirements: []
---
Research deeply.
""",
  )

  async def _file_read(_tool_input, **_kwargs):
    return {"content": "evidence"}, None

  app = create_agent(
    "test",
    model_key="anthropic.claude-opus-5",
    api_key="test-key",
    skills_dir=skills_dir,
    session_log_base_dir=tmp_path / "session-logs",
    # `explore` declares the open-web domain as a required capability (B-8),
    # so a parent offering only `file_read` could not authorize it.
    tool_handlers={"file_read": _file_read, "web_search": _file_read},
    tool_definitions=[
      {
        "name": "file_read",
        "description": "Read exact test evidence.",
        "input_schema": {
          "type": "object",
          "properties": {},
          "additionalProperties": False,
        },
      },
      {
        "name": "web_search",
        "description": "Search exact test evidence.",
        "input_schema": {
          "type": "object",
          "properties": {},
          "additionalProperties": False,
        },
      },
    ],
  )

  session, runtime = _build_runtime(
    app,
      model_entitled_capabilities=frozenset({
        "session.driver",
        "node.explore",
    }),
  )
  runner = runtime.build_runner(EventLog(), session.session_id)
  captured: dict[str, object] = {}

  async def _fake_spawn_sub_agent(task: str, **kwargs):
    captured["task"] = task
    captured.update(kwargs)
    return {"response": "ok"}, None

  runner.spawn_sub_agent = _fake_spawn_sub_agent  # type: ignore[method-assign]

  result, error = _run(
    runner._dispatcher._local["run_agent"]({
      "background": False,
      "objective": "Collect",
    })
  )

  assert error is None
  assert result == {"response": "ok"}
  assert captured["task"] == "Collect"
  execution = captured["capability_execution"]
  assert execution.bind.upstream_model == "claude-opus-5"  # type: ignore[union-attr]


def test_create_agent_rejects_request_model_outside_driver_policy(
  tmp_path: Path,
) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", "Research deeply.")
  app = create_agent(
    "test",
    model_key="anthropic.claude-haiku-4-5",
    skills_dir=skills_dir,
  )

  with pytest.raises(CapabilityResolutionError) as exc_info:
    _build_runtime(app, request_model="anthropic.claude-opus-5")
  assert exc_info.value.receipt() == {
    "error_code": "capability_model_not_allowed",
    "capability_id": "session.driver",
    "model_key": "anthropic.claude-opus-5",
    "provider": "anthropic",
    "upstream_model": "claude-opus-5",
    # Typed refusals carry the current eligible stable keys so the caller can
    # recover explicitly (design § Failure and fallback behavior).
    "eligible_model_keys": ["anthropic.claude-haiku-4-5"],
  }


def test_create_agent_mcp_config_path_expands_and_merges_inline_servers(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("HOME", str(tmp_path))
  config_path = tmp_path / ".claude.json"
  config_path.write_text(
    json.dumps(
      {
        "mcpServers": {
          "from_config": {"command": "cfg"},
          "shared": {"command": "cfg-shared"},
        }
      }
    ),
    encoding="utf-8",
  )

  app = create_agent(
    "test",
    mcp_servers={
      "shared": {"command": "inline"},
      "from_inline": {"command": "inline-only"},
    },
    mcp_config_path="~/.claude.json",
  )
  mcp_client = app.state.gateway_config.mcp_client
  assert isinstance(mcp_client, McpClientManager)
  assert mcp_client._config_path == config_path

  seen: dict[str, dict] = {}

  async def _fake_connect_or_warn(self, name: str, config: dict):
    seen[name] = dict(config)
    return None

  monkeypatch.setattr(mcp_client_module, "MCP_IMPORT_ERROR", None)
  monkeypatch.setattr(McpClientManager, "_connect_or_warn", _fake_connect_or_warn)

  _run(mcp_client.startup())

  assert set(seen) == {"from_config", "shared", "from_inline"}
  assert seen["shared"]["command"] == "inline"


def test_create_agent_registers_local_tool_handlers_and_definitions() -> None:
  async def _tool(_tool_input, **_kwargs):
    return {"ok": True}, None

  tool_def = {
    "name": "t",
    "description": "test tool",
    "input_schema": {"type": "object", "properties": {}},
  }
  app = create_agent(
    "test",
    api_key="test-key",
    tool_handlers={"t": _tool},
    tool_definitions=[tool_def],
  )

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)

  assert runner._dispatcher._local["t"] is _tool
  assert runtime.get_tool_definitions() == [tool_def]
  assert runner._dispatcher._request_approval is not None


def test_create_agent_code_execution_wires_hooks_approval_and_expiry_cleanup(tmp_path: Path) -> None:
  async def _user_on_tool_result(_ctx):
    return [{"type": "text", "text": "extra"}]

  app = create_agent(
    "test",
    api_key="test-key",
    code_execution=True,
    on_tool_result=_user_on_tool_result,
  )
  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)

  assert {"code_execute", "code_execute_status"} <= set(runner._dispatcher._local)
  assert {tool["name"] for tool in runtime.get_tool_definitions()} == {
    "code_execute",
    "code_execute_status",
  }
  assert runner._dispatcher._request_approval is not None
  assert runner._dispatcher._approved_tool_types is session.approved_tool_types
  valid_code_execute_input = {"code": "print(1)", "host": "subprocess"}
  assert runner._dispatcher._should_request_approval("code_execute", valid_code_execute_input, "subprocess") is True

  session.approved_tool_types.add("code_execute:subprocess")
  assert runner._dispatcher._should_request_approval("code_execute", valid_code_execute_input, "subprocess") is False

  ctx = ToolResultContext(
    tool_name="code_execute",
    tool_input={},
    result={"images": [{"filename": "plot.png", "data_base64": "abc"}]},
    error=None,
    duration_ms=5,
    tool_call_id="tool_1",
    session_id=session.session_id,
    server=None,
    result_entry={"content": json.dumps({"images": [{"filename": "plot.png", "data_base64": "abc"}]})},
  )
  extra_blocks = _run(runner._on_tool_result(ctx))
  payload = json.loads(ctx.result_entry["content"])

  assert extra_blocks == [{"type": "text", "text": "extra"}]
  assert payload["images"][0]["data_base64"] == "[image: plot.png]"

  work_dir = tmp_path / "expiry-cleanup"
  work_dir.mkdir()
  session.code_execution_work_dir = str(work_dir)
  _run(app.state.auth.session_store.expire_session_async(session.session_id))
  assert not work_dir.exists()


def test_create_agent_code_execution_cleans_up_active_sessions_on_shutdown(tmp_path: Path) -> None:
  app = create_agent("test", code_execution=True)
  work_dir = tmp_path / "shutdown-cleanup"

  with TestClient(app):
    session = app.state.auth.session_store.create_session(api_key_hash="hash", user_id="alice")
    work_dir.mkdir()
    session.code_execution_work_dir = str(work_dir)

  assert not work_dir.exists()


def test_create_agent_model_and_cors_configuration() -> None:
  app = create_agent(
    "test",
    model_key="anthropic.claude-haiku-4-5",
    cors_origins=["https://x.com"],
  )
  assert _capability_models(app) == {"claude-haiku-4-5"}
  assert app.state.gateway_config.cors_origins == ["https://x.com"]

  empty_cors_app = create_agent("test", cors_origins=[])
  assert empty_cors_app.state.gateway_config.cors_origins == []
