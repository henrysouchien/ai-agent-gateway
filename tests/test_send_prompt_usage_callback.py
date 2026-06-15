import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

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
