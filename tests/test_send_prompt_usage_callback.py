# ruff: noqa: E402

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.multi_user.billing import UsageEvent
from agent_gateway.providers import StreamEvent
from agent_gateway.send_prompt import _call_usage_callback

send_prompt_module = importlib.import_module("agent_gateway.send_prompt")


def _event() -> UsageEvent:
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
  )


def test_usage_callback_receives_usage_event() -> None:
  calls: list[UsageEvent] = []

  def cb(event: UsageEvent) -> None:
    calls.append(event)

  _call_usage_callback(cb, _event())
  assert len(calls) == 1
  assert calls[0].input_tokens == 10


def test_send_prompt_usage_event_sets_provider(monkeypatch) -> None:
  class _FakeProvider:
    name = "anthropic"

    def has_active_credential(self, config: dict[str, Any]) -> bool:
      return True

    def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
      return object()

    def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
      return {}

    async def stream(self, client: Any, params: dict[str, Any]):
      yield StreamEvent(type="message_start", input_tokens=10, cache_read_tokens=2, cache_creation_tokens=3)
      yield StreamEvent(type="text_delta", text="ok")
      yield StreamEvent(type="usage_update", output_tokens=4)
      yield StreamEvent(type="message_end", stop_reason="end_turn")

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int, **kwargs: Any):
      return type("Cost", (), {"total": 0.0042})()

    async def close_client(self, client: Any, timeout: float = 2.0) -> None:
      return None

  events: list[UsageEvent] = []
  monkeypatch.setattr(send_prompt_module, "AnthropicProvider", _FakeProvider)

  result = asyncio.run(
    send_prompt_module.send_prompt(
      "hello",
      model="claude-sonnet-4-6",
      user_id="alice",
      auth_config={"api_key": "k"},
      on_usage=events.append,
    )
  )

  assert result == "ok"
  assert len(events) == 1
  assert events[0].model == "claude-sonnet-4-6"
  assert events[0].provider == "anthropic"
  assert events[0].cost_usd == 0.0042


@pytest.mark.parametrize(
  ("failure", "expected_state"),
  [(RuntimeError("failed"), "failed_billable"), (asyncio.CancelledError(), "canceled")],
)
def test_send_prompt_emits_partial_commercial_usage_on_terminal_stream_paths(
  monkeypatch, failure, expected_state
) -> None:
  class _FailingProvider:
    name = "anthropic"

    def has_active_credential(self, config):
      return True

    def create_client(self, config, *, timeout=None):
      return object()

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

    async def close_client(self, client, timeout=2.0):
      return None

  class Producer:
    def __init__(self):
      self.items = []

    async def emit(self, event, *, usage_state="succeeded"):
      self.items.append((event, usage_state))

  producer = Producer()
  monkeypatch.setattr(send_prompt_module, "AnthropicProvider", _FailingProvider)
  with pytest.raises(type(failure)):
    asyncio.run(send_prompt_module.send_prompt(
      "hello", model="claude-sonnet-4-6", user_id="alice",
      auth_config={"api_key": "k"}, commercial_usage_producer=producer,
      request_id="req-1", rate_table_version="v1", billing_mode="metered",
      channel="mcp",
    ))

  assert len(producer.items) == 1
  event, state = producer.items[0]
  assert state == expected_state
  assert (event.input_tokens, event.output_tokens, event.reasoning_tokens_observed) == (10, 4, 1)
  assert event.cost_usd == 0.0042


def test_send_prompt_rejects_blank_commercial_identity_before_provider_call(monkeypatch) -> None:
  class ShouldNotConstruct:
    def __init__(self):
      raise AssertionError("provider construction must follow commercial preflight")

  monkeypatch.setattr(send_prompt_module, "AnthropicProvider", ShouldNotConstruct)
  with pytest.raises(ValueError, match="commercial send_prompt requires"):
    asyncio.run(send_prompt_module.send_prompt(
      "hello", model="claude-sonnet-4-6", user_id="alice",
      commercial_usage_producer=object(), request_id=" ",
      rate_table_version="v1", billing_mode="metered", channel=" ",
    ))
