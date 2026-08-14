# ruff: noqa: E402

import asyncio
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.capability_binding import (
  CapabilityBind,
  CapabilityResolutionError,
)
from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.multi_user.billing import UsageEvent
from agent_gateway.providers import (
  ModelInfo,
  ModelProvider,
  StreamEvent,
  ThinkingLevel,
)
from tests.capability_execution_test_support import stub_bound_capability_execution
from agent_gateway.send_prompt import _call_usage_callback

send_prompt_module = importlib.import_module("agent_gateway.send_prompt")


def test_send_prompt_sync_requires_bound_execution_at_its_public_boundary() -> None:
  parameter = inspect.signature(
    send_prompt_module.send_prompt_sync
  ).parameters["capability_execution"]

  assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
  assert parameter.default is inspect.Parameter.empty


def _resolved_bind() -> CapabilityBind:
  return stub_bound_capability_execution(
    provider=_PreflightProvider(),
    model="claude-sonnet-4-6",
    effort="none",
    credential_principal="user",
  ).bind


def _bound_auth_config(
  bind: CapabilityBind,
  **changes: Any,
) -> dict[str, Any]:
  config = {
    "provider": bind.provider,
    "max_tokens": 4096,
    "auth_mode": "api",
    "api_key": "test-key",
  }
  config.update(changes)
  return config


def _bound_execution(
  *,
  bind: CapabilityBind | None = None,
  provider_adapter: ModelProvider | None = None,
  **config_changes: Any,
) -> BoundCapabilityExecution:
  resolved_bind = bind or _resolved_bind()
  resolved_provider = provider_adapter or _PreflightProvider()
  if resolved_provider.name != resolved_bind.provider:
    admitted = stub_bound_capability_execution(
      provider=_PreflightProvider(name=resolved_bind.provider),
      model=resolved_bind.upstream_model,
      effort=resolved_bind.effort,
      credential_principal=resolved_bind.credential_principal,
      auth_config=_bound_auth_config(resolved_bind, **config_changes),
    )
    return BoundCapabilityExecution(
      bind=admitted.bind,
      registry=admitted.registry,
      adapter=resolved_provider,
      auth_config=admitted.auth_config,
    )
  return stub_bound_capability_execution(
    provider=resolved_provider,
    model=resolved_bind.upstream_model,
    effort=resolved_bind.effort,
    credential_principal=resolved_bind.credential_principal,
    auth_config=_bound_auth_config(resolved_bind, **config_changes),
  )


class _PreflightProvider(ModelProvider):
  def __init__(self, *, name: str = "anthropic") -> None:
    self.name = name
    self.create_client_calls = 0

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      supports_thinking=True,
    )

  def create_client(
    self,
    config: dict[str, Any],
    *,
    timeout: float | None = None,
  ) -> object:
    _ = config, timeout
    self.create_client_calls += 1
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout


def _event() -> UsageEvent:
  bind = _resolved_bind()
  return UsageEvent(
    user_id="alice",
    session_id="sess",
    request_id="req",
    parent_turn_id=None,
    timestamp=1.0,
    model="claude-sonnet-4-6",
    input_tokens=10,
    output_tokens=20,
    cache_read_tokens=3,
    cache_creation_tokens=4,
    cost_usd=0.0,
    rate_table_version="unknown",
    billing_mode="byok",
    channel=None,
    provider=bind.provider,
    capability_bind=bind.receipt(),
    provider_reported_model=None,
  )


def test_usage_callback_receives_usage_event() -> None:
  calls: list[UsageEvent] = []

  def cb(event: UsageEvent) -> None:
    calls.append(event)

  _call_usage_callback(cb, _event())
  assert len(calls) == 1
  assert calls[0].input_tokens == 10


def test_send_prompt_public_signature_is_execution_only() -> None:
  parameters = inspect.signature(send_prompt_module.send_prompt).parameters

  assert "capability_execution" in parameters
  assert {
    "capability_bind",
    "provider",
    "bound_auth_config",
    "execution_transport",
    "model",
    "max_tokens",
    "thinking",
    "effort",
    "auth_config",
  }.isdisjoint(parameters)


def test_send_prompt_usage_event_sets_provider() -> None:
  class _FakeProvider(_PreflightProvider):
    def __init__(self) -> None:
      super().__init__()
      self.request_params: dict[str, Any] | None = None

    def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
      self.request_params = kwargs
      return {}

    async def stream(self, client: Any, params: dict[str, Any]):
      yield StreamEvent(type="message_start", input_tokens=10, cache_read_tokens=2, cache_creation_tokens=3)
      yield StreamEvent(type="text_delta", text="ok")
      yield StreamEvent(type="usage_update", output_tokens=4)
      yield StreamEvent(type="message_end", stop_reason="end_turn")

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int, **kwargs: Any):
      return type("Cost", (), {"total": 0.0042})()

  bind = _resolved_bind()
  provider = _FakeProvider()
  events: list[UsageEvent] = []

  result = asyncio.run(
    send_prompt_module.send_prompt(
      "hello",
      capability_execution=_bound_execution(bind=bind, provider_adapter=provider),
      user_id="alice",
      on_usage=events.append,
    )
  )

  assert result == "ok"
  assert len(events) == 1
  assert events[0].model == "claude-sonnet-4-6"
  assert events[0].provider == "anthropic"
  assert events[0].cost_usd == 0.0042
  assert provider.request_params is not None
  assert provider.request_params["model"] == bind.upstream_model
  assert provider.request_params["max_tokens"] == 4096
  assert provider.request_params["thinking_level"] is ThinkingLevel.NONE
  assert provider.request_params["effort_resolution"].effective is ThinkingLevel.NONE


def test_send_prompt_clamps_max_tokens_to_model_max_output() -> None:
  class _FakeProvider(_PreflightProvider):
    def __init__(self) -> None:
      super().__init__()
      self.request_params: dict[str, Any] | None = None

    def get_model_info(self, model: str) -> ModelInfo:
      return ModelInfo(
        id=model,
        provider=self.name,
        max_output_tokens=128_000,
        supports_thinking=True,
      )

    def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
      self.request_params = kwargs
      return {}

    async def stream(self, client: Any, params: dict[str, Any]):
      yield StreamEvent(type="text_delta", text="ok")
      yield StreamEvent(type="message_end", stop_reason="end_turn")

  provider = _FakeProvider()
  bind = stub_bound_capability_execution(
    provider=provider,
    model="claude-opus-5",
    effort="none",
    credential_principal="user",
  ).bind

  result = asyncio.run(
    send_prompt_module.send_prompt(
      "hello",
      capability_execution=_bound_execution(
        bind=bind,
        provider_adapter=provider,
        max_tokens=200_000,
      ),
      user_id="alice",
    )
  )

  assert result == "ok"
  assert provider.request_params is not None
  assert provider.request_params["max_tokens"] == 128_000


@pytest.mark.parametrize(
  ("failure", "expected_state"),
  [(RuntimeError("failed"), "failed_billable"), (asyncio.CancelledError(), "canceled")],
)
def test_send_prompt_emits_partial_commercial_usage_on_terminal_stream_paths(
  failure, expected_state
) -> None:
  class _FailingProvider(_PreflightProvider):
    def build_request_params(self, **kwargs):
      return {}

    async def stream(self, client, params):
      yield StreamEvent(
        type="message_start", input_tokens=10,
        cache_read_tokens=2, cache_creation_tokens=3,
      )
      yield StreamEvent(type="usage_update", output_tokens=4, reasoning_tokens=1)
      raise failure

    def estimate_cost(self, model, input_tokens, output_tokens, **kwargs):
      assert (input_tokens, output_tokens) == (10, 4)
      assert kwargs == {"cache_read_tokens": 2, "cache_creation_tokens": 3}
      return type("Cost", (), {"total": 0.0042})()

  class Producer:
    def __init__(self):
      self.items = []

    async def emit(self, event, *, usage_state="succeeded"):
      self.items.append((event, usage_state))

  producer = Producer()
  bind = _resolved_bind()
  provider = _FailingProvider()
  with pytest.raises(type(failure)):
    asyncio.run(send_prompt_module.send_prompt(
      "hello",
      capability_execution=_bound_execution(bind=bind, provider_adapter=provider),
      user_id="alice",
      commercial_usage_producer=producer,
      request_id="req-1", rate_table_version="v1", billing_mode="metered",
      channel="mcp",
    ))

  assert len(producer.items) == 1
  event, state = producer.items[0]
  assert state == expected_state
  assert (event.input_tokens, event.output_tokens, event.reasoning_tokens_observed) == (10, 4, 1)
  assert event.cost_usd == 0.0042


def test_send_prompt_rejects_blank_commercial_identity_before_provider_call() -> None:
  bind = _resolved_bind()
  provider = _PreflightProvider()
  with pytest.raises(ValueError, match="commercial send_prompt requires"):
    asyncio.run(send_prompt_module.send_prompt(
      "hello",
      capability_execution=_bound_execution(bind=bind, provider_adapter=provider),
      user_id="alice",
      commercial_usage_producer=object(), request_id=" ",
      rate_table_version="v1", billing_mode="metered", channel=" ",
    ))
  assert provider.create_client_calls == 0


@pytest.mark.parametrize(
  ("config_changes", "provider_name", "message"),
  [
    ({}, "openai", "does not match binding provider"),
    ({"provider": "openai"}, "anthropic", "provider"),
    ({"model": "claude-other"}, "anthropic", "model-selection fields"),
    (
      {"effort": "high", "thinking_enabled_requested": True},
      "anthropic",
      "model-selection fields",
    ),
    ({"thinking": False}, "anthropic", "model-selection fields"),
  ],
)
def test_send_prompt_rejects_bound_execution_disagreement_before_client_creation(
  config_changes: dict[str, Any],
  provider_name: str,
  message: str,
) -> None:
  bind = _resolved_bind()
  provider = _PreflightProvider(name=provider_name)

  with pytest.raises((ValueError, CapabilityResolutionError), match=message):
    if "provider" in config_changes:
      baseline = _bound_execution(bind=bind, provider_adapter=provider)
      execution = BoundCapabilityExecution(
        bind=baseline.bind,
        registry=baseline.registry,
        adapter=baseline.adapter,
        auth_config=_bound_auth_config(bind, **config_changes),
      )
    else:
      execution = _bound_execution(
        bind=bind,
        provider_adapter=provider,
        **config_changes,
      )
    asyncio.run(
      send_prompt_module.send_prompt(
        "hello",
        capability_execution=execution,
        user_id="alice",
      )
    )

  assert provider.create_client_calls == 0


def test_send_prompt_does_not_fall_back_to_process_credentials(
  monkeypatch,
) -> None:
  monkeypatch.setenv("ANTHROPIC_API_KEY", "process-key")
  bind = _resolved_bind()
  provider = _PreflightProvider()
  config = _bound_auth_config(bind)
  config.pop("api_key")

  with pytest.raises(CapabilityResolutionError) as refused:
    execution = BoundCapabilityExecution(
      bind=bind,
      registry=_bound_execution(bind=bind, provider_adapter=provider).registry,
      adapter=provider,
      auth_config=config,
    )
    asyncio.run(send_prompt_module.send_prompt(
      "hello",
      capability_execution=execution,
      user_id="alice",
    ))

  assert refused.value.code == "credential_unavailable"
  assert provider.create_client_calls == 0
