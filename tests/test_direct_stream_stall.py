# ruff: noqa: E402
from __future__ import annotations

import asyncio
import importlib
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.multi_user.billing import UsageEvent
from agent_gateway.provider_summarize import ProviderSummarizeResult, provider_summarize
from agent_gateway.providers import CostEstimate, ModelInfo, ModelProvider, StreamEvent
from agent_gateway.rate_limit import AnthropicTokenBucket
from agent_gateway import runner_streaming
from agent_gateway.send_prompt import send_prompt
from model_authority.binding import CapabilityEffort
from gateway_test_support.capability_execution_test_support import stub_bound_capability_execution

summary_module = importlib.import_module("agent_gateway.provider_summarize")


class _DirectStreamProvider(ModelProvider):
  name = "anthropic"

  def __init__(self, scenario: str) -> None:
    self.scenario = scenario
    self.client = object()
    self.closed_clients: list[Any] = []
    self.stream_closed = False
    self.stream_elapsed = 0.0
    self.heartbeat_count = 0
    self.provider_timeout = asyncio.TimeoutError("provider transport timeout")
    # A depleted local bucket. 300 rpm refills one request every 0.2s, longer
    # than the 0.05s stall timeout the tests install; 2000 rpm refills one every
    # 0.03s, shorter than it.
    self.bucket = AnthropicTokenBucket(
      rpm=2000 if scenario == "queued_then_delayed" else 300,
    )
    self.bucket._rpm_tokens = 0.0
    self.waited_seconds = 0.0

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=4096,
      supports_thinking=True,
    )

  def create_client(
    self, config: dict[str, Any], *, timeout: float | None = None,
  ) -> object:
    return self.client

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    assert self.stream_closed
    self.closed_clients.append(client)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return kwargs

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    return CostEstimate(total=0.0042)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    started_at = time.monotonic()
    try:
      if self.scenario in {"queued", "queued_then_silent", "queued_then_delayed"}:
        self.waited_seconds = await self.bucket.acquire()
        if self.scenario == "queued_then_silent":
          await asyncio.Event().wait()
          return
        if self.scenario == "queued_then_delayed":
          # Peer silence after the queue (0.04s) is under the stall timeout,
          # but queue plus silence (~0.07s) is over it.
          await asyncio.sleep(0.04)
      if self.scenario == "silent":
        self.heartbeat_count += 1
        yield StreamEvent(type="heartbeat")
        await asyncio.Event().wait()
        return
      if self.scenario == "provider_timeout":
        raise self.provider_timeout
      yield StreamEvent(
        type="message_start",
        input_tokens=10,
        cache_read_tokens=2,
        cache_creation_tokens=3,
        provider_wait_ms=17,
        provider_reported_model=params["model"],
      )
      if self.scenario in {"heartbeats", "ready_heartbeats"}:
        while True:
          if self.scenario == "heartbeats":
            await asyncio.sleep(0.005)
          self.heartbeat_count += 1
          yield StreamEvent(type="heartbeat")
      for text in "abcdefghij":
        await asyncio.sleep(0.01)
        yield StreamEvent(type="text_delta", text=text)
      yield StreamEvent(type="usage_update", output_tokens=4, reasoning_tokens=1)
      yield StreamEvent(type="message_end", stop_reason="end_turn")
    finally:
      self.stream_elapsed = time.monotonic() - started_at
      self.stream_closed = True


def _execution(
  provider: _DirectStreamProvider, effort: CapabilityEffort = "none",
) -> BoundCapabilityExecution:
  return stub_bound_capability_execution(
    provider=provider,
    model="claude-opus-5",
    effort=effort,
    auth_config={"max_tokens": 4096},
  )


async def _call_direct(
  consumer: str,
  execution: BoundCapabilityExecution,
  usage_events: list[UsageEvent],
  liveness: list[dict[str, Any]],
) -> ProviderSummarizeResult | str:
  if consumer == "summary":
    return await provider_summarize(
      capability_execution=execution,
      messages=[{"role": "user", "content": "summarize"}],
      system_prompt=None,
      max_tokens=4096,
      on_event=liveness.append,
    )
  return await send_prompt(
    "hello",
    capability_execution=execution,
    user_id="alice",
    on_usage=usage_events.append,
  )


@pytest.fixture
def small_stall_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(runner_streaming, "STREAM_STALL_TIMEOUT", 0.05)
  monkeypatch.setattr(runner_streaming, "STREAM_THINKING_STALL_TIMEOUT", 0.09)


@pytest.mark.parametrize("consumer", ["summary", "prompt"])
@pytest.mark.parametrize("effort, stall_timeout", [("none", 0.05), ("high", 0.09)])
def test_direct_stream_silent_peer_bounds_blocked_next_event(
  consumer: str,
  effort: CapabilityEffort,
  stall_timeout: float,
  small_stall_timeouts: None,
) -> None:
  provider = _DirectStreamProvider("silent")
  usage_events: list[UsageEvent] = []
  liveness: list[dict[str, Any]] = []

  async def run() -> None:
    with pytest.raises(runner_streaming.StreamStallError) as caught:
      await asyncio.wait_for(
        _call_direct(consumer, _execution(provider, effort), usage_events, liveness),
        timeout=0.4,
      )
    assert "stall" in str(caught.value).lower()
    assert caught.value.stall_seconds >= stall_timeout

  asyncio.run(run())

  assert provider.heartbeat_count == 1
  assert provider.closed_clients == [provider.client]
  assert usage_events == []
  assert liveness == []


@pytest.mark.parametrize("consumer", ["summary", "prompt"])
def test_direct_stream_sustained_progress_has_no_total_duration_cap(
  consumer: str,
  small_stall_timeouts: None,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(summary_module, "STREAM_PROGRESS_LOG_INTERVAL", 0)
  provider = _DirectStreamProvider("progress")
  execution = _execution(provider)
  usage_events: list[UsageEvent] = []
  liveness: list[dict[str, Any]] = []

  async def run() -> ProviderSummarizeResult | str:
    return await asyncio.wait_for(
      _call_direct(consumer, execution, usage_events, liveness), timeout=0.4,
    )

  result = asyncio.run(run())

  assert provider.stream_elapsed > runner_streaming.STREAM_STALL_TIMEOUT
  assert provider.closed_clients == [provider.client]
  if isinstance(result, ProviderSummarizeResult):
    assert result.text == "abcdefghij"
    assert result.usage["input_tokens"] == 10
    assert result.usage["output_tokens"] == 4
    assert result.usage["reasoning_tokens_observed"] == 1
    assert result.usage["cache_read_input_tokens"] == 2
    assert result.usage["cache_creation_input_tokens"] == 3
    assert result.usage["provider_wait_ms"] == 17
    assert result.usage["provider_reported_model"] == execution.bind.upstream_model
    assert result.usage["capability_bind"] == execution.bind.to_json()
    assert [event["events"] for event in liveness] == list(range(1, 14))
    assert all(event["type"] == "heartbeat" for event in liveness)
    assert usage_events == []
  else:
    assert result == "abcdefghij"
    assert len(usage_events) == 1
    usage = usage_events[0]
    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.reasoning_tokens_observed == 1
    assert usage.cache_read_tokens == 2
    assert usage.cache_creation_tokens == 3
    assert usage.cost_usd == 0.0042
    assert usage.provider_reported_model == execution.bind.upstream_model
    assert usage.capability_bind == execution.bind.to_json()


@pytest.mark.parametrize("consumer", ["summary", "prompt"])
@pytest.mark.parametrize("scenario", ["heartbeats", "ready_heartbeats"])
def test_direct_stream_heartbeats_after_start_do_not_reset_progress(
  consumer: str,
  scenario: str,
  small_stall_timeouts: None,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(summary_module, "STREAM_PROGRESS_LOG_INTERVAL", 0)
  provider = _DirectStreamProvider(scenario)
  usage_events: list[UsageEvent] = []
  liveness: list[dict[str, Any]] = []

  async def run() -> None:
    with pytest.raises(runner_streaming.StreamStallError) as caught:
      await asyncio.wait_for(
        _call_direct(consumer, _execution(provider), usage_events, liveness),
        timeout=0.4,
      )
    assert "stall" in str(caught.value).lower()

  asyncio.run(run())

  assert provider.heartbeat_count >= 2
  assert provider.closed_clients == [provider.client]
  if consumer == "summary":
    assert len(liveness) == 1
    assert liveness[0]["events"] == 1
  else:
    assert len(usage_events) == 1
    assert usage_events[0].input_tokens == 10
    assert usage_events[0].output_tokens == 0


@pytest.mark.parametrize("consumer", ["summary", "prompt"])
def test_direct_stream_preserves_provider_timeout_error(
  consumer: str,
  small_stall_timeouts: None,
) -> None:
  provider = _DirectStreamProvider("provider_timeout")

  async def run() -> None:
    with pytest.raises(asyncio.TimeoutError) as caught:
      await _call_direct(consumer, _execution(provider), [], [])
    assert caught.value is provider.provider_timeout

  asyncio.run(run())

  assert provider.closed_clients == [provider.client]


@pytest.mark.parametrize("consumer", ["summary", "prompt"])
def test_direct_stream_local_bucket_wait_is_not_a_stall(
  consumer: str,
  small_stall_timeouts: None,
) -> None:
  provider = _DirectStreamProvider("queued")
  usage_events: list[UsageEvent] = []

  async def run() -> ProviderSummarizeResult | str:
    return await asyncio.wait_for(
      _call_direct(consumer, _execution(provider), usage_events, []), timeout=1.0,
    )

  result = asyncio.run(run())

  assert provider.waited_seconds > runner_streaming.STREAM_STALL_TIMEOUT
  text = result.text if isinstance(result, ProviderSummarizeResult) else result
  assert text == "abcdefghij"
  assert provider.closed_clients == [provider.client]


@pytest.mark.parametrize("consumer", ["summary", "prompt"])
def test_direct_stream_stall_clock_restarts_after_local_wait(
  consumer: str,
  small_stall_timeouts: None,
) -> None:
  provider = _DirectStreamProvider("queued_then_silent")

  async def run() -> None:
    with pytest.raises(runner_streaming.StreamStallError):
      await asyncio.wait_for(
        _call_direct(consumer, _execution(provider), [], []), timeout=1.0,
      )

  asyncio.run(run())

  assert provider.waited_seconds > runner_streaming.STREAM_STALL_TIMEOUT
  assert provider.closed_clients == [provider.client]


@pytest.mark.parametrize("consumer", ["summary", "prompt"])
def test_direct_stream_peer_silence_counts_only_after_local_queue(
  consumer: str,
  small_stall_timeouts: None,
) -> None:
  provider = _DirectStreamProvider("queued_then_delayed")

  async def run() -> ProviderSummarizeResult | str:
    return await asyncio.wait_for(
      _call_direct(consumer, _execution(provider), [], []), timeout=1.0,
    )

  result = asyncio.run(run())

  assert provider.waited_seconds + 0.04 > runner_streaming.STREAM_STALL_TIMEOUT
  text = result.text if isinstance(result, ProviderSummarizeResult) else result
  assert text == "abcdefghij"
