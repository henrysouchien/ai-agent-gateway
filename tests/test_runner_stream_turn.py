from agent_gateway.registry_scope import RegistryScope
from gateway_test_support.run_context_test_support import run_context_for_test
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import dataclasses
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, ToolDispatcher  # noqa: E402
from agent_gateway.agent_session_log import AgentSessionLog  # noqa: E402
from hank_model_transport.auth import ProviderCredentialFailure  # noqa: E402
from hank_model_transport.capability_execution import (  # noqa: E402
  CapabilityExecutionResolver,
  ModelFallbackHop,
)
from agent_gateway.event_log import EventLog  # noqa: E402
from agent_gateway.containment_settings import ContainmentSettings  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from hank_model_transport.multi_user.billing import UsageTotals  # noqa: E402
from hank_model_transport.providers import ModelInfo, ModelProvider
from model_authority.thinking import ThinkingLevel
from hank_model_transport.providers.base import StreamEvent  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
import agent_gateway.runner_stream_turn as runner_stream_turn  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from model_authority.thinking import EffortResolution
from model_authority.key_pool import KeyPool
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_bound_capability_execution,
  stub_runner_capability_execution,
)


def _set_retry_policy(monkeypatch, **changes) -> None:
  monkeypatch.setattr(
    runner_stream_turn,
    "TRANSPORT_RETRY_POLICY",
    dataclasses.replace(runner_stream_turn.TRANSPORT_RETRY_POLICY, **changes),
  )


@pytest.mark.parametrize(
  ("last_prompt", "expected", "prompt_totals"),
  [
    ({"input_tokens": 20, "cache_read_input_tokens": 40, "cache_creation_input_tokens": 5}, 65, (120, 240, 35)),
    ({"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}, 0, (100, 200, 30)),
    ({}, None, (100, 200, 30)),
  ],
)
def test_anthropic_stream_request_measurement_wait_and_retry_isolation(
  monkeypatch, last_prompt, expected, prompt_totals,
) -> None:
  import time
  import anthropic
  import httpx2
  from hank_model_transport.providers import AnthropicProvider
  import hank_model_transport.providers.anthropic as adapter

  first = {
    "input_tokens": 100, "cache_read_input_tokens": 200,
    "cache_creation_input_tokens": 30, "output_tokens": 10,
  }

  def sdk_events(prompt):
    snapshot = SimpleNamespace(
      **first,
      iterations=[SimpleNamespace(**first), SimpleNamespace(**prompt, output_tokens=7)],
    )
    return [
      SimpleNamespace(type="message_start", message=SimpleNamespace(
        usage=SimpleNamespace(**{**first, "output_tokens": 0}),
      )),
      *[
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"), usage=snapshot)
        for _ in range(2)
      ],
    ]

  class SDKStream:
    def __init__(self, events, fail=False):
      self.events, self.fail = events, fail

    async def __aenter__(self):
      return self

    async def __aexit__(self, *_args):
      return False

    def __aiter__(self):
      return self.iterate()

    async def iterate(self):
      for event in self.events:
        yield event
      if self.fail:
        raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))

  scripts = iter([
    SDKStream(sdk_events({"input_tokens": 900}), fail=True),
    SDKStream(sdk_events(last_prompt)),
    SDKStream(sdk_events({})),
  ])
  messages = SimpleNamespace(stream=lambda **_kwargs: next(scripts))
  client = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))
  acquired = []

  async def acquire(*, estimated_tokens):
    acquired.append(estimated_tokens)
    return 1.234

  monkeypatch.setattr(adapter, "get_global_token_bucket", lambda: SimpleNamespace(acquire=acquire))
  _set_retry_policy(monkeypatch, base_delay_seconds=0.0)
  provider = AnthropicProvider()
  # Only SDK client construction is replaced; request building and stream normalization are real.
  monkeypatch.setattr(provider, "create_client", lambda *_args, **_kwargs: client)
  runner = _refusal_runner(provider, [])
  billed = []
  runner._on_usage = billed.append
  totals: UsageTotals = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

  async def run():
    results = []
    for turn in (1, 2):
      returned = await runner._stream_turn(
        client=client,
        config={"model": "claude-opus-5", "effort": "none", "auth_mode": "api_key", "api_key": "k"},
        model_info=provider.get_model_info("claude-opus-5"),
        system_prompt=None, current_messages=[{"role": "user", "content": "hello"}],
        base_kwargs={"tools": []}, max_tokens=4096,
        turn_count=turn, turn_t0=time.time(), turn_t0_mono=time.monotonic(),
        system_chars=0, tools_chars=0, usage_totals=totals,
      )
      assert isinstance(returned, tuple)
      results.append(returned[1])
      if turn == 1:
        assert results[-1].request_input_tokens == expected
        assert tuple(totals[key] for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")) == prompt_totals
        assert totals["output_tokens"] == 17
        assert totals["provider_wait_ms"] == 1234
    return results

  results = asyncio.run(run())
  assert acquired == [4096, 4096, 4096]
  assert results[0].request_input_tokens == expected
  assert results[1].request_input_tokens is None
  assert totals["provider_wait_ms"] == 2468
  assert totals["output_tokens"] == 34
  assert len(billed) == 1
  assert billed[0].input_tokens == 1000
  assert billed[0].cache_read_tokens == 200
  assert billed[0].cache_creation_tokens == 30
  assert billed[0].output_tokens == 17
  expected_cost = provider.estimate_cost(
    "claude-opus-5", 1000, 17, cache_read_tokens=200, cache_creation_tokens=30,
  )
  assert billed[0].cost_usd == pytest.approx(expected_cost.total)
  assert billed[0].cost_usd > 0




def test_runner_stream_turn_reexports_streaming_helpers() -> None:
  assert gateway_runner.AgentRunner._thinking_level(False) is ThinkingLevel.NONE
  assert gateway_runner.STREAM_STALL_TIMEOUT > 0
  assert gateway_runner.STREAM_THINKING_STALL_TIMEOUT > gateway_runner.STREAM_STALL_TIMEOUT


def test_tool_history_preserves_a_semantic_copy_without_changing_execution_input() -> None:
  raw_input = {
    "credential": "raw-secret",
    "payload": {"value": 7},
  }
  raw_block = {
    "type": "tool_use",
    "id": "call-1",
    "name": "registered_write",
    "input": raw_input,
    "provider_extension": {"signature": "signed"},
  }

  class _Provider(ModelProvider):
    name = "stub"

    def has_active_credential(self, config):
      return bool(config.get("api_key"))
    def get_model_info(self, model):
      return ModelInfo(max_output_tokens=16_384, id=model, provider=self.name, supports_thinking=True)


    def resolve_effort(self, **kwargs):
      requested = kwargs["requested"]
      return EffortResolution(
        requested=requested,
        effective=requested,
        thinking_enabled_effective=False,
        payload_fragments={},
      )

    def normalize_messages(self, messages, model_info):
      _ = model_info
      return messages

    def build_request_params(self, **_kwargs):
      return {}

    async def stream(self, client, params):
      _ = client, params
      yield StreamEvent(
        type="tool_use_end",
        tool_id="call-1",
        tool_name="registered_write",
        tool_input=raw_input,
        raw_block=raw_block,
      )

  class _Dispatcher(ToolDispatcher):
    def __init__(self, *, capability_bind = TEST_CAPABILITY_BIND) -> None:
      super().__init__(
        McpClientManager(config_path=None),
        capability_bind=capability_bind,
        run_context=run_context_for_test(registry_scope=RegistryScope("run", "test-run", "/tmp/test-store")),
      )

    def redact_raw_tool_input_for_history(self, tool_name, tool_input):
      _ = self, tool_name, tool_input
      raise AssertionError(
        "stream normalization must not redact model semantic history"
      )

    def prepare_tool_call(self, *_args, **_kwargs):
      raise AssertionError("assistant history must not prepare executable input")

  provider = _Provider()
  runner = object.__new__(AgentRunner)
  runner._containment = ContainmentSettings()
  runner._provider = provider
  runner._dispatcher = _Dispatcher(capability_bind=TEST_CAPABILITY_BIND)
  runner._capability_execution = stub_runner_capability_execution(
    provider=provider,
    model="model",
    effort="none",
  )
  runner._stream_stall_timeout = 60.0
  runner._compaction_trigger = None
  runner._compaction_instructions = None
  runner._disconnected = False
  runner._billing_mode = "byok"
  runner._sid = "history-redaction"
  runner._append = lambda _event: None  # type: ignore[method-assign]

  returned = asyncio.run(runner._stream_turn(
    client=object(),
    config={
      "model": "model",
      "effort": "none",
      "auth_mode": "api_key",
    },
    model_info=ModelInfo(max_output_tokens=16_384, id="model", provider="stub"),
    system_prompt=None,
    current_messages=[],
    base_kwargs={"tools": []},
    max_tokens=128,
    turn_count=1,
    turn_t0=0.0,
    turn_t0_mono=0.0,
    system_chars=0,
    tools_chars=0,
    usage_totals={
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_creation_tokens": 0,
      "cache_read_tokens": 0,
    },
  ))

  assert returned is not None
  assert isinstance(returned, tuple)
  _, result = returned
  assert result.tool_uses == [("call-1", "registered_write", raw_input)]
  assert result.tool_uses[0][2] is not raw_input
  assert result.content_blocks == [{
    "type": "tool_use",
    "id": "call-1",
    "name": "registered_write",
    "input": {
      "credential": "raw-secret",
      "payload": {"value": 7},
    },
    "provider_extension": {"signature": "signed"},
  }]
  assert result.content_blocks[0] is not raw_block
  assert raw_block["input"] is raw_input
  assert raw_input["credential"] == "raw-secret"


def test_stream_turn_cancellation_preserves_primary_when_internal_close_fails(
  monkeypatch,
) -> None:
  class _Provider(ModelProvider):
    name = "stub"

    def has_active_credential(self, config):
      return bool(config.get("api_key"))
    def get_model_info(self, model):
      return ModelInfo(max_output_tokens=16_384, id=model, provider=self.name, supports_thinking=True)


    def resolve_effort(self, **kwargs):
      requested = kwargs["requested"]
      return EffortResolution(
        requested=requested,
        effective=requested,
        thinking_enabled_effective=False,
        payload_fragments={},
      )

    def normalize_messages(self, messages, model_info):
      _ = model_info
      return messages

    def build_request_params(self, **_kwargs):
      return {}

  class _Task:
    def cancel(self) -> None:
      pass

  def _create_task(coro):
    coro.close()
    return _Task()

  async def _cancel_wait(*_args, **_kwargs):
    raise asyncio.CancelledError("primary cancellation")

  fake_asyncio = SimpleNamespace(
    CancelledError=asyncio.CancelledError,
    FIRST_COMPLETED=asyncio.FIRST_COMPLETED,
    create_task=_create_task,
    wait=_cancel_wait,
  )
  monkeypatch.setattr(gateway_runner, "asyncio", fake_asyncio)

  provider = _Provider()
  runner = object.__new__(AgentRunner)
  runner._containment = ContainmentSettings()
  runner._provider = provider
  runner._capability_execution = stub_runner_capability_execution(
    provider=provider,
    model="model",
    effort="none",
  )
  runner._stream_stall_timeout = 60.0
  runner._compaction_trigger = None
  runner._compaction_instructions = None
  runner._disconnected = False
  runner._billing_mode = "byok"
  runner._sid = "stream-cleanup"
  events: list[dict[str, object]] = []
  runner._append = events.append  # type: ignore[method-assign]

  async def _force_close(*_args, **_kwargs):
    raise RuntimeError("provider close exploded")

  runner.force_close = _force_close

  with pytest.raises(asyncio.CancelledError) as exc_info:
    asyncio.run(
      runner._stream_turn(
        client=object(),
        config={
          "model": "model",
          "effort": "none",
          "auth_mode": "api_key",
        },
        model_info=ModelInfo(max_output_tokens=16_384, id="model", provider="stub"),
        system_prompt=None,
        current_messages=[],
        base_kwargs={"tools": []},
        max_tokens=128,
        turn_count=1,
        turn_t0=0.0,
        turn_t0_mono=0.0,
        system_chars=0,
        tools_chars=0,
        usage_totals={
          "input_tokens": 0,
          "output_tokens": 0,
          "cache_creation_tokens": 0,
          "cache_read_tokens": 0,
        },
      )
    )

  assert str(exc_info.value) == "primary cancellation"
  assert exc_info.value.__notes__ == [
    "Child cleanup failed: RuntimeError: provider close exploded"
  ]
  assert events == [{
    "type": "run_error",
    "phase": "stream_cancellation_cleanup",
    "error_type": "RuntimeError",
    "error": "Child cleanup failed: RuntimeError: provider close exploded",
    "message": "Child cleanup failed: RuntimeError: provider close exploded",
  }]


def test_stream_guard_appends_heartbeat_while_tool_input_streams(
  monkeypatch,
) -> None:
  import time

  import agent_gateway.runner_stream_turn as stream_turn_module

  monkeypatch.setattr(stream_turn_module, "STREAM_PROGRESS_LOG_INTERVAL", 0.02)
  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", 0.02)
  # A stall would fail the turn outright instead of retrying behind a delay.
  _set_retry_policy(monkeypatch, max_retries=0)

  tool_input = {"body": "x" * 20}
  tool_block = {"type": "tool_use", "id": "call-1", "name": "write_memo"}

  class _Provider(ModelProvider):
    name = "stub"

    def has_active_credential(self, config):
      return bool(config.get("api_key"))

    def get_model_info(self, model):
      return ModelInfo(max_output_tokens=16_384, id=model, provider=self.name, supports_thinking=True)

    def resolve_effort(self, **kwargs):
      requested = kwargs["requested"]
      return EffortResolution(
        requested=requested,
        effective=requested,
        thinking_enabled_effective=False,
        payload_fragments={},
      )

    def normalize_messages(self, messages, model_info):
      _ = model_info
      return messages

    def build_request_params(self, **_kwargs):
      return {}

    async def stream(self, client, params):
      _ = client, params
      yield StreamEvent(
        type="tool_use_start",
        tool_id="call-1",
        tool_name="write_memo",
        raw_block=dict(tool_block),
      )
      # A large tool input streams as input_json deltas only: nothing the
      # runner appends until tool_use_end, for longer than the stall timeout.
      deadline = time.monotonic() + 0.25
      while time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        yield StreamEvent(type="tool_use_delta", tool_input_json="x")
      yield StreamEvent(
        type="tool_use_end",
        tool_id="call-1",
        tool_name="write_memo",
        tool_input=tool_input,
        raw_block={**tool_block, "input": tool_input},
      )
      yield StreamEvent(type="message_end", stop_reason="tool_use")

  provider = _Provider()
  runner = object.__new__(AgentRunner)
  runner._containment = ContainmentSettings()
  runner._provider = provider
  runner._capability_execution = stub_runner_capability_execution(
    provider=provider,
    model="model",
    effort="none",
  )
  # Shorter than the stream: a guard that did not count deltas would fire.
  runner._stream_stall_timeout = 0.15
  runner._compaction_trigger = None
  runner._compaction_instructions = None
  runner._disconnected = False
  runner._billing_mode = "byok"
  runner._sid = "tool-input-heartbeat"
  events: list[dict[str, object]] = []
  runner._append = events.append  # type: ignore[method-assign]

  returned = asyncio.run(runner._stream_turn(
    client=object(),
    config={
      "model": "model",
      "effort": "none",
      "auth_mode": "api_key",
    },
    model_info=ModelInfo(max_output_tokens=16_384, id="model", provider="stub"),
    system_prompt=None,
    current_messages=[],
    base_kwargs={"tools": []},
    max_tokens=128,
    turn_count=1,
    turn_t0=time.time(),
    turn_t0_mono=time.monotonic(),
    system_chars=0,
    tools_chars=0,
    usage_totals={
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_creation_tokens": 0,
      "cache_read_tokens": 0,
    },
  ))

  heartbeats = [event for event in events if event["type"] == "heartbeat"]
  assert heartbeats
  for heartbeat in heartbeats:
    assert set(heartbeat) == {"type", "elapsed_s", "last_progress_s", "events"}
    assert all(
      type(heartbeat[key]) is int
      for key in ("elapsed_s", "last_progress_s", "events")
    )
  assert any(heartbeat["events"] > 1 for heartbeat in heartbeats if isinstance(heartbeat["events"], int))
  assert [event for event in events if event["type"] != "heartbeat"] == []

  assert isinstance(returned, tuple)
  _, result = returned
  assert result.stop_reason == "tool_use"
  assert result.tool_uses == [("call-1", "write_memo", tool_input)]


_REFUSAL_STOP_DETAILS = {
  "type": "refusal",
  "category": "reasoning_extraction",
  "explanation": "The request asks the model to reproduce its internal reasoning.",
}


class _RefusingProvider(ModelProvider):
  """Delivers some text per request, ending each turn with the next scripted stop.

  A ``refusal`` stop carries the provider's ``stop_details``; any other stop
  carries none.
  """

  name = "stub"

  def __init__(self, stop_reasons: tuple[str, ...] = ("refusal",)) -> None:
    self.requests = 0
    self._stop_reasons = stop_reasons

  def has_active_credential(self, config):
    return True

  def create_client(self, config, *, timeout=None):
    _ = config, timeout
    return object()

  async def close_client(self, client, timeout=2.0):
    _ = client, timeout

  def get_model_info(self, model):
    return ModelInfo(max_output_tokens=16_384, id=model, provider=self.name)

  def build_request_params(self, **_kwargs):
    self.requests += 1
    return {}

  async def stream(self, client, params):
    _ = client, params
    stop_reason = self._stop_reasons[self.requests - 1]
    text = "Here is the first half" if stop_reason == "refusal" else "Final answer."
    yield StreamEvent(type="message_start", input_tokens=10)
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
    yield StreamEvent(type="usage_update", output_tokens=5)
    yield StreamEvent(
      type="message_end",
      stop_reason=stop_reason,
      stop_details=dict(_REFUSAL_STOP_DETAILS) if stop_reason == "refusal" else None,
    )


def _refusal_runner(
  provider: ModelProvider,
  durable_events: list[dict[str, object]],
) -> AgentRunner:
  from agent_gateway.event_log import EventLog
  from gateway_test_support.capability_execution_test_support import (
    stub_bound_capability_execution,
  )

  event_log = EventLog()
  _dispatcher_execution_631 = stub_bound_capability_execution(
        provider=provider,
        model="claude-opus-5",
        effort="none",
        auth_config={"api_key": "k"},
      )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      capability_bind=_dispatcher_execution_631.bind,
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={},
      event_log=event_log,
      session_id="sess-refusal",
      get_tool_definitions=None,
      run_context=run_context_for_test(registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store')),
    ),
    session_id="sess-refusal",
    capability_execution=_dispatcher_execution_631,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  async def _append_durable_event(event):
    durable_events.append(dict(event))
    return SimpleNamespace(seq=len(durable_events))

  runner._append_durable_event = _append_durable_event
  return runner



def test_provider_defaults_survive_initial_auth_and_stream_retry(monkeypatch) -> None:
  from hank_model_transport.auth import ProviderCredentialFailure

  _set_retry_policy(monkeypatch, max_retries=2, base_delay_seconds=0.0)

  class Provider(_RefusingProvider):
    def __init__(self):
      super().__init__(("end_turn",))
      self.clients = []
      self.attempts = 0

    def create_client(self, config, **kwargs):
      assert kwargs == {}
      client = object()
      self.clients.append(client)
      return client

    def classify_credential_failure(self, exc):
      if str(exc) == "expired":
        return ProviderCredentialFailure(provider="stub", kind="auth")
      return None

    def next_credential(self, config, failure):
      return {"api_key": "fresh"}

    def is_retryable_error(self, exc):
      return str(exc) == "transient"

    async def stream(self, client, params):
      assert client is self.clients[-1]
      self.attempts += 1
      if self.attempts < 3:
        raise RuntimeError("expired" if self.attempts == 1 else "transient")
      yield StreamEvent(type="text_delta", text="done")
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "done"})
      yield StreamEvent(type="message_end", stop_reason="end_turn")

  provider = Provider()
  events = []
  runner = _refusal_runner(provider, events)
  asyncio.run(runner.run(messages=[{"role": "user", "content": "finish"}]))
  assert provider.attempts == 3
  assert len(provider.clients) == 3
  assert len({id(client) for client in provider.clients}) == 3
  [assistant_message] = [
    event for event in events if event.get("type") == "assistant_message"
  ]
  assert assistant_message["stop_reason"] == "end_turn"
  assert assistant_message["content_blocks"] == [{"type": "text", "text": "done"}]
  [stream_complete] = [
    entry.event for entry in runner._log.entries
    if entry.event.get("type") == "stream_complete"
  ]
  assert stream_complete["terminal_disposition"] == "completed"


@pytest.mark.parametrize("between_polls", [False, True])
@pytest.mark.parametrize("silent_after_queue", [False, True])
def test_stream_guard_excludes_local_queue_but_bounds_peer_silence(
  monkeypatch, between_polls, silent_after_queue,
) -> None:
  import agent_gateway.runner_stream_turn as stream_module

  clock = [0.0]
  polls = [0]
  consumer: list[asyncio.Task[object] | None] = [None]
  queued = [False]
  left_at: list[float | None] = [None]
  release = asyncio.Event()
  heartbeat_ready = asyncio.Event()
  real_sleep = asyncio.sleep

  async def guard_sleep(delay):
    polls[0] += 1
    clock[0] = polls[0] * 10.0
    if polls[0] == 1:
      queued[0] = not between_polls
      left_at[0] = 9.0 if between_polls else None
    else:
      queued[0] = False
      left_at[0] = 19.0 if not between_polls else 9.0
      release.set()
      if polls[0] == 3:
        heartbeat_ready.set()
    await real_sleep(0)
    await real_sleep(0)

  def is_queued(task):
    assert task is consumer[0]
    return queued[0]

  monkeypatch.setattr(gateway_runner, "time", SimpleNamespace(
    monotonic=lambda: clock[0], time=lambda: clock[0],
  ))
  monkeypatch.setattr(gateway_runner, "asyncio", SimpleNamespace(
    **{**vars(asyncio), "sleep": guard_sleep},
  ))
  _set_retry_policy(monkeypatch, max_retries=0)
  monkeypatch.setattr(stream_module, "locally_queued", is_queued)
  monkeypatch.setattr(stream_module, "local_queue_left_at", lambda task: left_at[0])

  class Provider(_RefusingProvider):
    async def stream(self, client, params):
      consumer[0] = asyncio.current_task()
      await release.wait()
      yield StreamEvent(type="text_delta", text="progress")
      if silent_after_queue:
        await heartbeat_ready.wait()
        yield StreamEvent(type="heartbeat")
        await asyncio.Event().wait()
      yield StreamEvent(type="message_end", stop_reason="end_turn")

  runner = _refusal_runner(Provider(), [])
  runner._stream_stall_timeout = 5.0
  result = asyncio.run(runner._stream_turn(
    client=object(),
    config={"model": "claude-opus-5", "effort": "none", "auth_mode": "api_key", "api_key": "k"},
    model_info=ModelInfo(max_output_tokens=16_384, id="claude-opus-5", provider="stub"),
    system_prompt=None, current_messages=[], base_kwargs={"tools": []},
    max_tokens=128, turn_count=1, turn_t0=0.0, turn_t0_mono=0.0,
    system_chars=0, tools_chars=0,
    usage_totals={"input_tokens": 0, "output_tokens": 0,
                  "cache_creation_tokens": 0, "cache_read_tokens": 0},
  ))
  if silent_after_queue:
    assert isinstance(result, tuple)
    partial = result[1]
    assert partial.full_text == "progress"
    assert partial.stop_reason == "partial_stream_failure"
    assert partial.stop_details is not None
    assert partial.stop_details["retained_text_chars"] == len("progress")
    assert partial.stop_details["retained_reasoning_chars"] == 0
    assert "no stream progress for 10s" in partial.stop_details["error"]
    [retry] = [event for event in _logged(runner) if event["type"] == "stream_retry"]
    _assert_retry_settlement(retry, text_chars=len("progress"))
    consumer_task = consumer[0]
    assert consumer_task is not None
    assert consumer_task.cancelled()
    assert runner._terminal_closure_event is None
    assert polls[0] == 3
  else:
    assert isinstance(result, tuple)
    assert result[1].full_text == "progress"
    assert result[1].stop_reason == "end_turn"


def _set_thinking_without_progress_timeout(
  monkeypatch, runner: AgentRunner, seconds: float,
  role_calls: list[str | None] | None = None,
) -> None:
  resolver = runner._capability_execution.resolver
  assert resolver is not None
  def projected(_policy, _capability_id, *, role):
    if role_calls is not None:
      role_calls.append(role)
    return seconds
  monkeypatch.setattr(
    type(resolver.selection_policy),
    "thinking_without_progress_seconds",
    projected,
  )


def _thinking_stream_kwargs() -> dict[str, Any]:
  return dict(
    client=object(),
    config={"model": "claude-opus-5", "effort": "none", "auth_mode": "api_key", "api_key": "k"},
    model_info=ModelInfo(max_output_tokens=16_384, id="claude-opus-5", provider="stub"),
    system_prompt=None, current_messages=[], base_kwargs={"tools": []},
    max_tokens=128, turn_count=1, turn_t0=0.0, turn_t0_mono=0.0,
    system_chars=0, tools_chars=0,
    usage_totals={"input_tokens": 0, "output_tokens": 0,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
  )


def test_thinking_only_100ms_ends_typed_without_retry_hop_or_partial(monkeypatch) -> None:
  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", 0.005)
  _set_retry_policy(monkeypatch, max_retries=2, base_delay_seconds=0.0)

  class Provider(_RefusingProvider):
    async def stream(self, client, params):
      yield StreamEvent(type="message_start", input_tokens=1)
      while True:
        await asyncio.sleep(0.005)
        yield StreamEvent(type="thinking_delta", thinking_text="still reasoning")

  provider = Provider()
  runner = _refusal_runner(provider, [])
  capture_spawning_turn = runner._dispatcher.capture_spawning_turn
  monkeypatch.setattr(
    runner._dispatcher,
    "capture_spawning_turn",
    lambda: SimpleNamespace(
      context=SimpleNamespace(selection_snapshot=object(), selection_provenance=None),
      root_turn=capture_spawning_turn().root_turn,
    ),
  )
  monkeypatch.setattr(
    CapabilityExecutionResolver,
    "rebind_model_fallback",
    staticmethod(lambda execution, snapshot, **kwargs: ModelFallbackHop(
      execution=execution, role="deep", chain_index=0, skipped=(),
      primary_model_key=execution.bind.model_key,
    )),
  )
  runner._stream_stall_timeout = 5.0
  _set_thinking_without_progress_timeout(monkeypatch, runner, 0.1)
  result = asyncio.run(asyncio.wait_for(
    runner._stream_turn(**_thinking_stream_kwargs()), timeout=1.0,
  ))
  assert result is None
  assert provider.requests == 1
  events = _logged(runner)
  assert not [event for event in events if event["type"] in {"stream_retry", "model_fallback"}]
  [terminal_error] = [event for event in events if event["type"] == "error"]
  assert "thinking_without_progress" in str(terminal_error["error"])
  assert "session_id=sess-refusal" in str(terminal_error["error"])
  assert "follow or retry the turn" in str(terminal_error["error"])


def test_visible_text_permanently_disarms_thinking_only_bound(monkeypatch) -> None:
  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", 0.005)
  _set_retry_policy(monkeypatch, max_retries=0)

  class Provider(_RefusingProvider):
    async def stream(self, client, params):
      yield StreamEvent(type="message_start", input_tokens=1)
      yield StreamEvent(type="thinking_delta", thinking_text="first thought")
      await asyncio.sleep(0.06)
      yield StreamEvent(type="thinking_delta", thinking_text="second thought")
      await asyncio.sleep(0.02)
      yield StreamEvent(type="text_delta", text="answer")
      for _ in range(30):
        await asyncio.sleep(0.01)
        yield StreamEvent(type="thinking_delta", thinking_text="later thought")
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "answer"})
      yield StreamEvent(type="message_end", stop_reason="end_turn")

  provider = Provider()
  runner = _refusal_runner(provider, [])
  runner._stream_stall_timeout = 5.0
  _set_thinking_without_progress_timeout(monkeypatch, runner, 0.2)
  result = asyncio.run(asyncio.wait_for(
    runner._stream_turn(**_thinking_stream_kwargs()), timeout=1.0,
  ))

  assert isinstance(result, tuple)
  assert result[1].full_text == "answer"
  assert result[1].stop_reason == "end_turn"
  assert provider.requests == 1
  assert not [event for event in _logged(runner) if event["type"] == "error"]

@pytest.mark.parametrize("role_kind", ["failover", "provenance"])
def test_thinking_timeout_uses_the_bound_selection_role(monkeypatch, role_kind: str) -> None:
  runner = _refusal_runner(_RefusingProvider(("end_turn",)), [])
  expected_role = "deep" if role_kind == "failover" else "fast"
  if role_kind == "failover":
    from model_authority.bind import FailoverCapabilityBind
    bind = runner._capability_execution.bind
    runner._capability_execution = dataclasses.replace(
      runner._capability_execution,
      bind=FailoverCapabilityBind.model_validate({
        **bind.to_json(),
        "failed_over_from": "primary",
        "requested_role": expected_role,
      }),
    )
  else:
    capture_spawning_turn = runner._dispatcher.capture_spawning_turn
    monkeypatch.setattr(
      runner._dispatcher,
      "capture_spawning_turn",
      lambda: SimpleNamespace(
        context=SimpleNamespace(
          selection_snapshot=object(),
          selection_provenance=SimpleNamespace(role=expected_role),
        ),
        root_turn=capture_spawning_turn().root_turn,
      ),
    )
  roles: list[str | None] = []
  _set_thinking_without_progress_timeout(monkeypatch, runner, 0.2, role_calls=roles)

  result = asyncio.run(asyncio.wait_for(
    runner._stream_turn(**_thinking_stream_kwargs()), timeout=1.0,
  ))

  assert isinstance(result, tuple)
  assert result[1].stop_reason == "end_turn"
  assert roles == [expected_role]

def test_ollama_read_error_mid_thinking_rotates_to_sibling_credential(monkeypatch) -> None:
  from hank_model_transport.providers.ollama_cloud import OllamaCloudProvider

  _set_retry_policy(monkeypatch, max_retries=1, base_delay_seconds=0.0)

  class Provider(OllamaCloudProvider):
    def __init__(self) -> None:
      super().__init__()
      self.keys: list[str] = []

    def get_model_info(self, model):
      return ModelInfo(max_output_tokens=16_384, id=model, provider=self.name, supports_thinking=True)

    def resolve_effort(self, **kwargs):
      requested = kwargs["requested"]
      return EffortResolution(
        requested=requested, effective=requested,
        thinking_enabled_effective=requested != ThinkingLevel.NONE,
        payload_fragments={},
      )

    def normalize_messages(self, messages, model_info):
      return messages

    def build_request_params(self, **kwargs):
      return {}

    def create_client(self, config, *, timeout=None):
      return str(config["api_key"])

    async def close_client(self, client, timeout=2.0):
      return None

    def next_credential(self, config, failure):
      assert failure.kind == "transport"
      return {**config, "api_key": "fresh"}

    async def stream(self, client, params):
      self.keys.append(str(client))
      yield StreamEvent(type="message_start", input_tokens=1)
      if len(self.keys) == 1:
        yield StreamEvent(type="thinking_delta", thinking_text="interrupted thought")
        raise httpx.ReadError("broken mid-stream")
      yield StreamEvent(type="text_delta", text="answer")
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "answer"})
      yield StreamEvent(type="message_end", stop_reason="end_turn")

  provider = Provider()
  runner = _refusal_runner(provider, [])
  result = asyncio.run(asyncio.wait_for(
    runner._stream_turn(**_thinking_stream_kwargs()), timeout=1.0,
  ))

  assert isinstance(result, tuple), (provider.keys, _logged(runner))
  assert result[1].full_text == "answer"
  assert result[1].stop_reason == "end_turn"
  assert len(provider.keys) == 2
  assert provider.keys[-1] == "fresh"
  events = _logged(runner)
  assert not [event for event in events if event["type"] == "model_fallback"]
  [rotated] = [event for event in events if event["type"] == "credential_refreshed"]
  assert rotated["kind"] == "transport"

@pytest.mark.parametrize("silent_after_queue", [False, True])
def test_queued_child_heartbeats_preserve_parent_liveness_and_peer_stall(
  monkeypatch, silent_after_queue,
) -> None:
  import time

  from hank_model_transport.rate_limit import AnthropicTokenBucket, locally_queued
  import agent_gateway.runner_sub_agents as children
  import agent_gateway.runner_stream_turn as stream_turn_module

  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", 0.01)
  monkeypatch.setattr(stream_turn_module, "STREAM_PROGRESS_LOG_INTERVAL", 0.03)
  _set_retry_policy(monkeypatch, max_retries=0)
  monkeypatch.setattr(children, "STREAM_GUARD_POLL_INTERVAL", 0.01)

  async def case():
    bucket = AnthropicTokenBucket(rpm=120)
    for _ in range(120):
      await bucket.acquire()
    activity = children._ChildActivity()
    queued_heartbeats: list[dict[str, object]] = []
    consumer: list[asyncio.Task | None] = [None]
    provider_wait: list[float] = []
    left_queue: list[float] = []

    class Provider(_RefusingProvider):
      async def stream(self, client, params):
        consumer[0] = asyncio.current_task()
        provider_wait.append(await bucket.acquire())
        left_queue.append(time.monotonic())
        if silent_after_queue:
          await asyncio.Event().wait()
        yield StreamEvent(type="text_delta", text="finished")
        yield StreamEvent(type="message_end", stop_reason="end_turn")

    runner = _refusal_runner(Provider(), [])
    runner._stream_stall_timeout = 0.10
    original_append = runner._append

    def append(event):
      activity.observe(event)
      if event["type"] == "heartbeat" and locally_queued(consumer[0]):
        queued_heartbeats.append(dict(event))
      return original_append(event)

    runner._append = append
    results = []

    async def child_run():
      started = time.monotonic()
      results.append(await runner._stream_turn(
        client=object(),
        config={"model": "claude-opus-5", "effort": "none", "auth_mode": "api_key", "api_key": "k"},
        model_info=ModelInfo(max_output_tokens=16_384, id="claude-opus-5", provider="stub"),
        system_prompt=None, current_messages=[], base_kwargs={"tools": []},
        max_tokens=128, turn_count=1, turn_t0=time.time(),
        turn_t0_mono=started, system_chars=0, tools_chars=0,
        usage_totals={"input_tokens": 0, "output_tokens": 0,
                      "cache_creation_tokens": 0, "cache_read_tokens": 0},
      ))

    parent_stall = await children._await_child_under_activity_guard(
      child_run(), activity, silence_timeout=0.15,
    )
    assert parent_stall is None
    assert provider_wait[0] > 0.35
    assert queued_heartbeats
    assert all(event["events"] == 0 for event in queued_heartbeats)
    assert all(event["last_progress_s"] == 0 for event in queued_heartbeats)
    if silent_after_queue:
      assert results == [None]
      assert time.monotonic() - left_queue[0] >= 0.10
      error = runner._terminal_closure_event
      assert error is not None
      assert "no stream progress" in error["error"]
      assert consumer[0] is not None and consumer[0].cancelled()
    else:
      assert results[0] is not None
      assert results[0][1].full_text == "finished"

  asyncio.run(case())


def test_refusal_stop_completes_the_turn_with_a_refusal_notice() -> None:
  from agent_gateway.runner_session_events import REFUSAL_GUIDANCE

  provider = _RefusingProvider()
  durable_events: list[dict[str, object]] = []
  runner = _refusal_runner(provider, durable_events)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "Explain your reasoning first."}]))

  # Nothing is retried or reset: one provider request.
  assert provider.requests == 1
  [assistant_message] = [
    event for event in durable_events if event.get("type") == "assistant_message"
  ]
  assert assistant_message["stop_reason"] == "refusal"
  assert assistant_message["content_blocks"] == [
    {"type": "text", "text": "Here is the first half"}
  ]
  # The provider's reason is durable beside the stop, not only on the live notice.
  assert assistant_message["stop_details"] == _REFUSAL_STOP_DETAILS
  events = [entry.event for entry in runner._log.entries]
  types = [event["type"] for event in events]
  refusal = next(event for event in events if event["type"] == "refusal")
  assert refusal == {
    "type": "refusal",
    "category": "reasoning_extraction",
    "explanation": "The request asks the model to reproduce its internal reasoning.",
    "guidance": REFUSAL_GUIDANCE,
  }
  assert types.index("text_delta") < types.index("refusal") < types.index("stream_complete")
  [stream_complete] = [event for event in events if event["type"] == "stream_complete"]
  assert stream_complete["terminal_disposition"] == "completed"
  assert not {"error", "run_error", "interrupted"} & set(types)
  assert not any(event.get("type") in {"error", "run_error"} for event in durable_events)


@pytest.mark.parametrize(
  ("content", "expected"),
  [
    ("first", (0, 0)),
    ([{"type": "text", "text": "a"}, {"type": "thinking", "thinking": "x"}], (0, 0)),
    ([{"type": "thinking", "thinking": "x"}], None),
  ],
)
def test_markerless_provider_records_the_request_prefix_boundary(
  content, expected,
) -> None:
  """A provider that renders no cache_control still gives the fork its boundary."""

  provider = _RefusingProvider(("end_turn",))
  runner = _refusal_runner(provider, [])

  asyncio.run(runner.run(messages=[{"role": "user", "content": content}]))

  assert runner._last_request_message_marker_position == expected


def test_markerless_request_prefix_boundary_excludes_the_request_tail_reminder() -> None:
  """The reminder rides this request only, so the fork's boundary is the history's."""

  provider = _RefusingProvider(("end_turn",))
  runner = _refusal_runner(provider, [])
  runner._record_reminder = lambda: "[Run record — calls: 1 ok]"

  asyncio.run(runner.run(messages=[
    {"role": "user", "content": [{"type": "text", "text": "first"}]},
  ]))

  assert runner._last_request_message_marker_position == (0, 0)


def test_end_turn_assistant_message_carries_no_stop_details() -> None:
  provider = _RefusingProvider(stop_reasons=("end_turn",))
  durable_events: list[dict[str, object]] = []
  runner = _refusal_runner(provider, durable_events)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "Answer plainly."}]))

  [assistant_message] = [
    event for event in durable_events if event.get("type") == "assistant_message"
  ]
  assert assistant_message["stop_reason"] == "end_turn"
  assert "stop_details" not in assistant_message


class _DiagnosticsProvider(_RefusingProvider):
  """One turn with provider ids, rate-limit headers, two thinking deltas and a redacted block."""

  async def stream(self, client, params):
    _ = client, params
    yield StreamEvent(
      type="message_start",
      input_tokens=10,
      provider_response_id="msg_abc",
      provider_request_id="req_123",
      rate_limit={"anthropic-ratelimit-requests-remaining": "41", "retry-after": "2"},
    )
    yield StreamEvent(type="thinking_delta", thinking_text="step one, ")
    yield StreamEvent(type="thinking_delta", thinking_text="step two")
    yield StreamEvent(
      type="thinking_end",
      thinking_text="step one, step two",
      raw_block={"type": "thinking", "thinking": "step one, step two", "signature": "sig"},
    )
    # A provider that sends only the final block: counted from the raw block.
    yield StreamEvent(
      type="thinking_end",
      thinking_text="",
      raw_block={"type": "redacted_thinking", "data": "opaque"},
    )
    yield StreamEvent(
      type="thinking_end",
      thinking_text="final only",
      raw_block={"type": "thinking", "thinking": "final only", "signature": "sig2"},
    )
    yield StreamEvent(type="text_delta", text="ok")
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "ok"})
    yield StreamEvent(type="usage_update", output_tokens=5, reasoning_tokens=3)
    yield StreamEvent(type="message_end", stop_reason="end_turn", service_tier="standard")


def test_turn_usage_carries_provider_diagnostics_and_thinking_counts() -> None:
  provider = _DiagnosticsProvider(stop_reasons=("end_turn",))
  durable_events: list[dict[str, object]] = []
  runner = _refusal_runner(provider, durable_events)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "Think."}]))

  [assistant_message] = [
    event for event in durable_events if event.get("type") == "assistant_message"
  ]
  expected = {
    "provider_response_id": "msg_abc",
    "provider_request_id": "req_123",
    "service_tier": "standard",
    "rate_limit": {"anthropic-ratelimit-requests-remaining": "41", "retry-after": "2"},
    "thinking_chars": len("step one, step two") + len("final only"),
    "redacted_thinking_blocks": 1,
  }
  usage = assistant_message["usage"]
  content_blocks = assistant_message["content_blocks"]
  assert isinstance(usage, dict) and isinstance(content_blocks, list)
  assert {key: usage[key] for key in expected} == expected
  # The redacted block is replayed intact beside the readable thinking blocks.
  assert [block["type"] for block in content_blocks] == [
    "thinking", "redacted_thinking", "thinking", "text",
  ]
  events = [entry.event for entry in runner._log.entries]
  [turn_complete] = [event for event in events if event["type"] == "turn_complete"]
  assert {key: turn_complete["usage"][key] for key in expected} == expected
  [stream_complete] = [event for event in events if event["type"] == "stream_complete"]
  assert stream_complete["usage"]["reasoning_tokens_observed"] == 3
  assert stream_complete["usage"]["provider_wait_ms"] == 0
  assert stream_complete["usage"]["thinking_chars"] == expected["thinking_chars"]


def test_refusal_with_running_background_child_waits_for_delivery_like_end_turn() -> None:
  from agent_gateway.task_registry import TaskState

  provider = _RefusingProvider(stop_reasons=("refusal", "end_turn"))
  durable_events: list[dict[str, object]] = []
  runner = _refusal_runner(provider, durable_events)

  async def case() -> None:
    entry = runner._task_registry.register("background_agent", agent_name="reviewer")
    wait_started = asyncio.Event()

    async def child() -> None:
      await wait_started.wait()
      runner._task_registry.transition(
        entry.task_id,
        TaskState.COMPLETED,
        result={"kind": "report", "report": {"summary": "review complete"}},
      )

    child_task = asyncio.create_task(child())
    entry.asyncio_task = child_task
    runner._task_registry.transition(entry.task_id, TaskState.RUNNING)
    assert runner._task_registry.admission_count > 0

    original_wait = runner._wait_for_background_notification
    wait_calls = 0

    async def tracked_wait() -> bool:
      nonlocal wait_calls
      wait_calls += 1
      wait_started.set()
      return await original_wait()

    runner._wait_for_background_notification = tracked_wait
    await runner.run(
      messages=[{"role": "user", "content": "Explain your reasoning first."}],
    )

    # The refused turn held the run open for the child exactly as end_turn
    # does, then delivered its notification on a follow-up turn.
    assert wait_calls == 1
    assert provider.requests == 2
    assert entry.state is TaskState.COMPLETED
    assert runner._notification_queue.pending_count == 0

  asyncio.run(case())

  events = [entry.event for entry in runner._log.entries]
  types = [event["type"] for event in events]
  assert types.count("refusal") == 1
  [stream_complete] = [event for event in events if event["type"] == "stream_complete"]
  assert stream_complete["terminal_disposition"] == "completed"
  assert not {"error", "run_error", "interrupted"} & set(types)
  assert not any(event.get("type") in {"error", "run_error"} for event in durable_events)
  assert [
    event["stop_reason"] for event in durable_events if event.get("type") == "assistant_message"
  ] == ["refusal", "end_turn"]


# --- Credential preparation at episode boundaries ----------------------------


class _Quota429(RuntimeError):
  """A 429 that carries quota evidence: the account is parked until its reset."""


class _Concurrency429(RuntimeError):
  """A 429 with no quota evidence: transient, the account is not parked."""


class _Transient(RuntimeError):
  pass


def _answer(text: str = "done") -> list[StreamEvent]:
  return [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="text_delta", text=text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    StreamEvent(type="usage_update", output_tokens=2),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]


def _tool_turn(tool_id: str, tool_name: str) -> list[StreamEvent]:
  return [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="tool_use_start", tool_id=tool_id, tool_name=tool_name),
    StreamEvent(
      type="tool_use_end",
      tool_id=tool_id,
      tool_name=tool_name,
      tool_input={},
      raw_block={"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
    ),
    StreamEvent(type="message_end", stop_reason="tool_use"),
  ]


class _AccountProvider(_RefusingProvider):
  """An account pool double: ``prepare_credential`` ranks the unparked accounts.

  Each request consumes the next scripted outcome: an exception to raise, a
  list of events to stream, or ``(events, exception)`` for a stream that fails
  after delivering ``events``. The client is the account key it was built for.
  """

  def __init__(self, accounts: tuple[str, ...], script: list[object], *, name: str = "stub") -> None:
    super().__init__(("end_turn",))
    self.name = name
    self.accounts = accounts
    self.script = script
    self.parked: dict[str, float] = {}
    # (account key in the config handed to prepare, model, session id)
    self.prepared: list[tuple[str, str, str | None]] = []
    self.clients: list[str] = []
    self.streamed: list[str] = []

  def _first_usable(self) -> str | None:
    return next((key for key in self.accounts if key not in self.parked), None)

  async def prepare_credential(self, config, *, model, session_id) -> dict[str, object] | None:
    self.prepared.append((str(config["api_key"]), model, session_id))
    usable = self._first_usable() or self.accounts[0]
    return {
      "api_key": usable,
      "credential_source": {"generation_ref": f"anthropic-generation-{usable}"},
    }

  def create_client(self, config, *, timeout=None):
    _ = timeout
    self.clients.append(str(config["api_key"]))
    return str(config["api_key"])

  def classify_credential_failure(self, exc):
    if isinstance(exc, _Quota429):
      return ProviderCredentialFailure(
        provider=self.name, kind="rate_limit", status_code=429, reset_at=time.time() + 3600,
      )
    return None

  def next_credential(self, config, failure):
    self.parked[str(config["api_key"])] = failure.reset_at or 0.0
    usable = self._first_usable()
    return None if usable is None else {
      "api_key": usable,
      "credential_source": {"generation_ref": f"anthropic-generation-{usable}"},
    }

  def is_retryable_error(self, exc):
    return isinstance(exc, (_Quota429, _Concurrency429, _Transient, httpx.ReadError))

  async def stream(self, client, params):
    _ = params
    self.streamed.append(client)
    outcome = self.script.pop(0)
    if isinstance(outcome, BaseException):
      raise outcome
    if isinstance(outcome, tuple):
      events, failure = outcome
      for event in events:
        yield event
      raise failure
    for event in outcome:  # type: ignore[attr-defined]
      yield event


class _NoPrepareProvider(_AccountProvider):
  async def prepare_credential(self, config, *, model, session_id):
    self.prepared.append((str(config["api_key"]), model, session_id))
    return None


def _account_execution(provider: ModelProvider, api_key: str, model: str = "claude-opus-5"):
  return stub_bound_capability_execution(
    provider=provider, model=model, effort="none", auth_config={"api_key": api_key},
  )


def _account_runner(
  execution,
  *,
  session_id: str = "sess-pool",
  gateway_session: GatewaySession | None = None,
  agent_session_log: AgentSessionLog | None = None,
  handlers: dict | None = None,
  tool_names: tuple[str, ...] = (),
  identity=None,
) -> AgentRunner:
  event_log = EventLog()
  return AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      capability_bind=execution.bind,
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers=handlers or {},
      event_log=event_log,
      session_id=session_id or "sess-pool",
      role="owner",
      identity=identity,
      run_context=run_context_for_test(registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store')),
    ),
    session_id=session_id,
    gateway_session=gateway_session,
    agent_session_log=agent_session_log,
    capability_execution=execution,
    get_tool_definitions=lambda: [
      {"name": name, "description": "", "input_schema": {"type": "object", "properties": {}}}
      for name in tool_names
    ],
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def _logged(runner: AgentRunner) -> list[dict[str, object]]:
  return [entry.event for entry in runner._log.entries]


def _types(events: list[dict[str, object]], *wanted: str) -> list[object]:
  return [event["type"] for event in events if event["type"] in wanted]


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch) -> None:
  _set_retry_policy(monkeypatch, base_delay_seconds=0.0)


@pytest.mark.parametrize(
  ("session_id", "expected_session_id"),
  [("sess-pool", "sess-pool"), ("", None)],
)
def test_prepare_runs_before_the_first_client_but_not_same_model_retries(
  session_id, expected_session_id,
) -> None:
  provider = _AccountProvider(("A1",), [_Transient("transient"), _answer()])
  runner = _account_runner(_account_execution(provider, "A0"), session_id=session_id)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  # A same-model retry keeps the first preparation's binding instead of
  # reranking the pool at each client build.
  assert provider.prepared == [
    ("A0", "claude-opus-5", expected_session_id),
  ]
  assert provider.clients == ["A1", "A1"]
  assert runner._auth_config["api_key"] == "A1"
  assert _types(_logged(runner), "error") == []
  [retry] = [event for event in _logged(runner) if event["type"] == "stream_retry"]
  _assert_retry_settlement(retry)


def test_resumed_runner_prepares_with_the_durable_log_identity(tmp_path: Path) -> None:
  log = AgentSessionLog(tmp_path / "original-conversation.jsonl")
  provider = _AccountProvider(("A",), [])
  for runtime_id in ("original-conversation", "resumed-runtime"):
    session = GatewaySession(
      session_id=runtime_id, api_key_hash="hash", created_at=1, expires_at=2,
      user_id="alice", owner_user_id="alice", tenant_id="test-tenant",
      resume_log_path=tmp_path / "stale-session-log.jsonl",
    )
    runner = _account_runner(
      _account_execution(provider, "A"), session_id=runtime_id,
      gateway_session=session, agent_session_log=log,
    )
    asyncio.run(runner._prepare_credential(dict(runner._auth_config)))

  assert [call[2] for call in provider.prepared] == [
    "original-conversation", "original-conversation",
  ]


def test_sub_agent_prepares_with_its_own_identity_despite_sharing_the_parent_log(
  tmp_path: Path,
) -> None:
  provider = _AccountProvider(("A",), [])
  runner = _account_runner(
    _account_execution(provider, "A"), session_id="sub1:parent-conversation",
    agent_session_log=AgentSessionLog(tmp_path / "parent-conversation.jsonl"),
  )

  asyncio.run(runner._prepare_credential(dict(runner._auth_config)))

  assert [call[2] for call in provider.prepared] == ["sub1:parent-conversation"]


def test_compaction_and_prepare_share_the_resumed_conversation_identity(
  monkeypatch, tmp_path: Path,
) -> None:
  provider = _AccountProvider(("A",), [_answer()])
  session = GatewaySession(
    session_id="resumed-runtime", api_key_hash="hash", created_at=1, expires_at=2,
    user_id="alice", owner_user_id="alice", tenant_id="test-tenant",
    resume_log_path=tmp_path / "original-conversation.jsonl",
  )
  runner = _account_runner(
    _account_execution(provider, "A"), session_id=session.session_id,
    gateway_session=session,
  )
  runner._compaction_trigger = 1
  compacted_sessions: list[str | None] = []

  async def compact_messages(messages, *_args, session_id=None, **_kwargs):
    compacted_sessions.append(session_id)
    return SimpleNamespace(
      applied=False, messages=messages, reason="below_trigger", summarize_usage=None,
    )

  monkeypatch.setattr(
    "agent_gateway.runner_run_loop.maybe_compact_current_messages", compact_messages,
  )

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert compacted_sessions == [call[2] for call in provider.prepared]
  assert compacted_sessions == ["original-conversation"]



def test_a_none_preparation_keeps_the_bound_config_and_execution() -> None:
  provider = _NoPrepareProvider(("A1",), [_Transient("transient"), _answer()])
  execution = _account_execution(provider, "A0")
  runner = _account_runner(execution)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert [call[0] for call in provider.prepared] == ["A0"]
  assert provider.clients == ["A0", "A0"]
  assert runner._auth_config["api_key"] == "A0"
  assert runner.capability_execution is execution


def test_a_disconnected_runner_prepares_nothing() -> None:
  provider = _AccountProvider(("A1",), [])
  runner = _account_runner(_account_execution(provider, "A0"))
  runner._disconnected = True
  config = dict(runner._auth_config)

  assert asyncio.run(runner._prepare_credential(config)) is False

  assert provider.prepared == []
  assert config == dict(runner._auth_config)


def test_model_fallback_rebuild_prepares_once_on_the_fallback_provider(monkeypatch) -> None:
  primary = _AccountProvider(("A1",), [_Quota429("usage limit")], name="stub")
  fallback = _AccountProvider(("F1",), [_answer("from fallback")], name="stub-fallback")
  runner = _account_runner(_account_execution(primary, "A0"))
  fallback_execution = _account_execution(fallback, "F0", model="fallback-model")

  def one_hop(execution, snapshot, **_kwargs):
    _ = execution, snapshot
    return ModelFallbackHop(
      execution=fallback_execution, role="deep", chain_index=0, skipped=(),
      primary_model_key=execution.bind.model_key,
    )

  monkeypatch.setattr(CapabilityExecutionResolver, "rebind_model_fallback", staticmethod(one_hop))
  capture_spawning_turn = runner._dispatcher.capture_spawning_turn
  monkeypatch.setattr(
    runner._dispatcher,
    "capture_spawning_turn",
    lambda: SimpleNamespace(
      context=SimpleNamespace(selection_snapshot=object(), selection_provenance=None),
      root_turn=capture_spawning_turn().root_turn,
    ),
  )

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert primary.prepared == [("A0", "claude-opus-5", "sess-pool")]
  assert primary.clients == ["A1"]
  assert fallback.prepared == [("F0", "fallback-model", "sess-pool")]
  assert fallback.clients == ["F1"]
  assert runner._auth_config["api_key"] == "F1"
  events = _logged(runner)
  assert _types(events, "model_fallback") == ["model_fallback"]
  assert _types(events, "error") == []


def test_quota_429_on_one_account_rotates_to_the_next_and_succeeds() -> None:
  provider = _AccountProvider(("A", "B"), [_Quota429("usage limit"), _answer()])
  runner = _account_runner(_account_execution(provider, "A"))

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert provider.streamed == ["A", "B"]
  assert provider.parked.keys() == {"A"}
  assert [call[0] for call in provider.prepared] == ["A"]
  assert runner._auth_config["api_key"] == "B"
  events = _logged(runner)
  assert _types(events, "credential_refreshed") == ["credential_refreshed"]
  assert _types(events, "error") == []



def _exhausted_anthropic_pool():
  from model_authority.key_pool import PooledKey
  key = PooledKey(key="sk-ant-seat-pool", fingerprint="abcdef12")
  pool = KeyPool("anthropic", (key,), state_path=None, repo="ai-excel-addin")
  pool.mark_exhausted(key.fingerprint, "out_of_credit")
  return pool


class _EnrolledProbe:
  def resolve(self, *, risk_user_id, provider, context):
    import hashlib
    from hank_model_transport.user_provider_keys import ResolvedKey, ResolvedUserKeys
    key = "sk-ant-user-fixture"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return ResolvedUserKeys(provider, risk_user_id, (ResolvedKey(digest, "mine", key),))

  def report(self, **kwargs):
    raise AssertionError("enrollment probe must not report")


class _OutageProbe:
  def resolve(self, *, risk_user_id, provider, context):
    from hank_model_transport.user_provider_keys import UserKeyResolverUnavailable
    raise UserKeyResolverUnavailable(provider, "resolver_response_invalid")

  def report(self, **kwargs):
    raise AssertionError("outage probe must not report")


class _UnenrolledProbe:
  def resolve(self, *, risk_user_id, provider, context):
    from hank_model_transport.user_provider_keys import UserKeyMissing
    raise UserKeyMissing(provider, risk_user_id)

  def report(self, **kwargs):
    raise AssertionError("unenrolled probe must not report")


def test_enrolled_credit_exhaustion_rebinds_runner_to_user_key() -> None:
  from hank_model_transport.capability_execution import MaterializedCredential
  from model_authority.binding import AuthContext, CredentialHandle, ModelSelectionIntent
  from model_authority.capabilities import CAPABILITY_IDS
  from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
  from agent_gateway.execution_identity import (
    DispatchIdentity, ExecutionIdentity, execution_identity_from_session, resolved_execution_identity,
  )
  from agent_gateway.fork_request_handoff import fork_credential_identity_available

  class SeatProvider(_AccountProvider):
    def __init__(self, script):
      super().__init__(("seat",), script, name="anthropic")
      self.key_pool: KeyPool | None = None

    def credential_key_pool(self, config) -> KeyPool | None:
      return self.key_pool

    def get_model_info(self, model):
      return ModelInfo(
        max_output_tokens=16384, id=model, provider=self.name, supports_thinking=True,
      )

    async def prepare_credential(self, config, *, model, session_id):
      return None

    def create_client(self, config, *, timeout=None):
      key = str(config.get("auth_token") or config["api_key"])
      self.clients.append(key)
      return key

    def next_credential(self, config, failure):
      assert failure.reset_at is not None
      self.parked[str(config["auth_token"])] = failure.reset_at
      return None

  provider = SeatProvider([_Quota429("usage limit"), _answer()])
  provider.key_pool = _exhausted_anthropic_pool()
  models = dict(INITIAL_MODEL_REGISTRY.models)
  seat_key = "anthropic.claude-fable-5-1"
  api_key = "anthropic.claude-sonnet-5-5"
  seat_entry = models[seat_key]
  models[seat_key] = dataclasses.replace(seat_entry, route="anthropic.oauth")
  models[api_key] = dataclasses.replace(
    models[api_key], upstream_model=seat_entry.upstream_model,
    reported_identities=seat_entry.reported_identities,
  )
  registry = dataclasses.replace(INITIAL_MODEL_REGISTRY, models=models)
  seat = CredentialHandle("seat:owner", "anthropic", "service", "tenant", None)
  user = CredentialHandle("user-key:101:anthropic", "anthropic", "user", "tenant", "alice")
  offered = []

  def materialize(handle):
    offered.append(handle)
    config = (
      {"provider": "anthropic", "api_key": "user-key", "auth_mode": "api",
       "auth_mechanism": "api_key", "entitlement_kind": "commercial_byok"}
      if handle == user else
      {"provider": "anthropic", "auth_token": "seat", "auth_mode": "oauth",
       "auth_mechanism": "oauth_bearer", "entitlement_kind": "operator_personal_subscription"}
    )
    return MaterializedCredential(handle, config)

  resolver = CapabilityExecutionResolver(
    registry=registry, selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth_context=AuthContext(
      run_mode="interactive", actor_id="alice", tenant_id="tenant",
      user_provider_handles={}, service_provider_handles={"anthropic": seat},
      entitled_capabilities=CAPABILITY_IDS, entitled_model_keys=frozenset(models),
      allow_service_for_interactive=True,
    ),
    credential_materializer=materialize, adapter_resolver=lambda _: provider,
    user_key_handles={"anthropic": user}, risk_user_id=101,
  )
  execution = resolver.resolve(
    "session.driver", explicit_intent=ModelSelectionIntent(seat_key, "high", "explicit_user"),
  )
  identity = DispatchIdentity(
    context=None, user_id="alice", risk_user_id=101,
    execution=ExecutionIdentity(tenant_id="tenant", credential_handle_id=seat.handle_id),
  )
  runner = _account_runner(execution, identity=identity)
  from hank_model_transport.user_provider_keys import UserKeyPools, install_user_key_pools
  install_user_key_pools(UserKeyPools(_EnrolledProbe()))
  try:
    asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))
  finally:
    install_user_key_pools(None)

  assert provider.streamed == ["seat", "user-key"]
  assert provider.parked.keys() == {"seat"}
  rebound = runner._capability_execution
  assert rebound.bind.route == "anthropic.public"
  assert rebound.bind.model_key == api_key
  assert rebound.bind.credential_principal == "user"
  assert rebound.bind.credential_ref == user.handle_id
  assert rebound.bind.effort == "high"
  assert runner._auth_config["api_key"] == "user-key"
  assert "auth_token" not in runner._auth_config
  assert isinstance(runner._dispatcher, ToolDispatcher)
  assert runner._dispatcher._capability_bind == rebound.bind
  assert all(handle in (seat, user) for handle in offered)
  events = _logged(runner)
  assert _types(events, "credential_refreshed") == ["credential_refreshed"]
  assert _types(events, "model_fallback", "error") == []
  # A plain bind still cannot change principal, nor can a materialized hop
  # cross the dispatcher's admitted user or tenant.
  dispatcher = _account_runner(execution, identity=identity)._dispatcher
  with pytest.raises(ValueError, match="changes execution authority"):
    dispatcher.replace_capability_bind(rebound.bind)
  for other in (
    dataclasses.replace(identity, risk_user_id=202),
    dataclasses.replace(identity, user_id="mallory"),
    dataclasses.replace(identity, execution=ExecutionIdentity(tenant_id="other")),
  ):
    dispatcher = _account_runner(execution, identity=other)._dispatcher
    with pytest.raises(ValueError, match="changes execution authority"):
      dispatcher.replace_capability_bind(rebound)
  session = SimpleNamespace(session_credential_handle=seat, risk_user_id=101, tenant_id="tenant")
  parent = SimpleNamespace(_gateway_session=session, _capability_execution=rebound)
  expected_identity = ExecutionIdentity(tenant_id="tenant", credential_handle_id=user.handle_id)
  assert resolved_execution_identity(parent) == expected_identity
  assert execution_identity_from_session(session, bind=rebound.bind) == expected_identity
  assert fork_credential_identity_available(parent)
  assert session.session_credential_handle is seat  # A reset seat is still the next session's first choice.
  for other_user in (None, 202):
    other_session = SimpleNamespace(
      session_credential_handle=seat, risk_user_id=other_user, tenant_id="tenant",
    )
    assert execution_identity_from_session(other_session, bind=rebound.bind) is None
  assert execution_identity_from_session(
    session, bind=rebound.bind.model_copy(update={"credential_principal": "service"}),
  ) is None
  assert execution_identity_from_session(
    session, bind=rebound.bind.model_copy(update={
      "provider": "openai", "credential_ref": "user-key:101:openai",
    }),
  ) is None



def test_autonomous_enrolled_exhausted_pool_still_leases_through_parent() -> None:
  """Positive control: an enrolled autonomous child still fails over to its key.

  Every usable seat-pool key is out of credit. The child source is the parent
  user-provider-keys route, and the lease is that POST, not a fabricated key.
  """
  import hashlib
  from dataclasses import asdict

  from agent_gateway.autonomous_user_keys import (
    AUTONOMOUS_USER_KEYS_PATH, AutonomousUserKeySource,
  )
  from agent_gateway.execution_identity import DispatchIdentity, ExecutionIdentity
  from hank_model_transport.capability_execution import MaterializedCredential
  from hank_model_transport.user_provider_keys import (
    NO_API_KEY_CONSUMER, ResolveContext, ResolvedKey, ResolvedUserKeys,
    UserKeyPools, install_user_key_pools, installed_user_key_pools,
  )
  from model_authority.binding import AuthContext, CredentialHandle, ModelSelectionIntent
  from model_authority.capabilities import CAPABILITY_IDS
  from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY

  secret = "sk-ant-user-fixture"
  digest = hashlib.sha256(secret.encode()).hexdigest()[:12]
  body = asdict(ResolvedUserKeys(
    "anthropic", 101, (ResolvedKey(digest, "mine", secret),), identity_role="owner",
  ))
  posts: list[str] = []

  def post(url, *, json, headers):
    _ = json, headers
    posts.append(url)

    class _Reply:
      status_code = 200

      def json(self):
        return body

    return _Reply()

  class SeatProvider(_AccountProvider):
    def __init__(self):
      super().__init__(("seat",), [_Quota429("usage limit"), _answer("from user key")], name="anthropic")
      self.key_pool = _exhausted_anthropic_pool()

    def credential_key_pool(self, config):
      return self.key_pool

    def get_model_info(self, model):
      return ModelInfo(
        max_output_tokens=16384, id=model, provider=self.name, supports_thinking=True,
      )

    async def prepare_credential(self, config, *, model, session_id):
      return None

    def create_client(self, config, *, timeout=None):
      key = str(config.get("auth_token") or config["api_key"])
      self.clients.append(key)
      return key

    def next_credential(self, config, failure):
      self.parked[str(config["auth_token"])] = failure.reset_at
      return None

  provider = SeatProvider()
  models = dict(INITIAL_MODEL_REGISTRY.models)
  seat_key = "anthropic.claude-fable-5-1"
  api_model = "anthropic.claude-sonnet-5-5"
  seat_entry = models[seat_key]
  models[seat_key] = dataclasses.replace(seat_entry, route="anthropic.oauth")
  models[api_model] = dataclasses.replace(
    models[api_model], upstream_model=seat_entry.upstream_model,
    reported_identities=seat_entry.reported_identities,
  )
  registry = dataclasses.replace(INITIAL_MODEL_REGISTRY, models=models)
  seat = CredentialHandle("seat:owner", "anthropic", "service", "tenant", None)
  user = CredentialHandle("user-key:101:anthropic", "anthropic", "user", "tenant", "alice")
  leases: list[str] = []

  def materialize(handle):
    if handle == user:
      leases.append(handle.handle_id)
      lease = installed_user_key_pools().lease(
        risk_user_id=101, provider="anthropic",
        context=ResolveContext(
          consumer_key_hash=NO_API_KEY_CONSUMER, channel="autonomous",
        ),
      )
      return MaterializedCredential(handle, lease.auth_config())
    return MaterializedCredential(handle, {
      "provider": "anthropic", "auth_token": "seat", "auth_mode": "oauth",
      "auth_mechanism": "oauth_bearer", "entitlement_kind": "operator_personal_subscription",
    })

  resolver = CapabilityExecutionResolver(
    registry=registry, selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth_context=AuthContext(
      run_mode="autonomous", actor_id="alice", tenant_id="tenant",
      user_provider_handles={}, service_provider_handles={"anthropic": seat},
      entitled_capabilities=CAPABILITY_IDS, entitled_model_keys=frozenset(models),
    ),
    credential_materializer=materialize, adapter_resolver=lambda _: provider,
    user_key_handles={"anthropic": user}, risk_user_id=101,
  )
  execution = resolver.resolve(
    "session.driver", explicit_intent=ModelSelectionIntent(seat_key, "high", "explicit_user"),
  )
  identity = DispatchIdentity(
    context=None, user_id="alice", risk_user_id=101,
    execution=ExecutionIdentity(tenant_id="tenant", credential_handle_id=seat.handle_id),
  )
  runner = _account_runner(execution, identity=identity)
  now = int(time.time())
  install_user_key_pools(UserKeyPools(AutonomousUserKeySource(
    session=GatewaySession(
      session_id="child-1", api_key_hash="run:autonomous",
      created_at=now, expires_at=now + 3600, user_id="alice",
      risk_user_id=101, role="owner", tenant_id="tenant",
      session_token="child-token",
    ),
    base_url="http://parent.test", post=post,
  )))
  try:
    asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))
  finally:
    install_user_key_pools(None)

  assert leases == [user.handle_id]
  assert posts and all(url.endswith(AUTONOMOUS_USER_KEYS_PATH) for url in posts)
  assert provider.streamed == ["seat", secret]
  rebound = runner._capability_execution
  assert rebound.bind.credential_principal == "user"
  assert rebound.bind.credential_ref == user.handle_id
  assert runner._auth_config["api_key"] == secret
  assert _types(_logged(runner), "model_fallback", "error") == []



def _install_probe(probe):
  from hank_model_transport.user_provider_keys import UserKeyPools, install_user_key_pools
  install_user_key_pools(UserKeyPools(probe))


def test_unenrolled_or_outage_429_falls_back_without_a_lease(monkeypatch) -> None:
  _assert_seat_429_uses_preset_fallback(monkeypatch, probe=None)
  _assert_seat_429_uses_preset_fallback(monkeypatch, probe=_UnenrolledProbe())
  _assert_seat_429_uses_preset_fallback(monkeypatch, probe=_OutageProbe())


def test_enrolled_rate_limit_falls_back_without_user_key_failover(monkeypatch) -> None:
  _assert_seat_429_uses_preset_fallback(monkeypatch, probe=_EnrolledProbe(), rate_limited_pool=True)


def _assert_seat_429_uses_preset_fallback(monkeypatch, *, probe, rate_limited_pool: bool = False) -> None:
  from hank_model_transport.capability_execution import MaterializedCredential
  from hank_model_transport.user_provider_keys import install_user_key_pools
  from model_authority.binding import AuthContext, CredentialHandle, ModelSelectionIntent
  from model_authority.capabilities import CAPABILITY_IDS
  from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
  from model_authority.key_pool import PooledKey

  class SeatProvider(_AccountProvider):
    def __init__(self):
      super().__init__(("seat",), [_Quota429("usage limit")], name="anthropic")
      self.key_pool: KeyPool | None = None

    def credential_key_pool(self, config) -> KeyPool | None:
      return self.key_pool

    def get_model_info(self, model):
      return ModelInfo(max_output_tokens=16384, id=model, provider=self.name, supports_thinking=True)

    async def prepare_credential(self, config, *, model, session_id):
      return None

    def create_client(self, config, *, timeout=None):
      key = str(config.get("auth_token") or config["api_key"])
      self.clients.append(key)
      return key

    def next_credential(self, config, failure):
      self.parked[str(config["auth_token"])] = failure.reset_at
      return None

  provider = SeatProvider()
  if rate_limited_pool:
    key = PooledKey(key="sk-ant-seat-pool", fingerprint="abcdef12")
    provider.key_pool = KeyPool("anthropic", (key,), state_path=None, repo="ai-excel-addin")
    provider.key_pool.mark_exhausted(key.fingerprint, "rate_limit")
  offered = []
  models = dict(INITIAL_MODEL_REGISTRY.models)
  seat_key = "anthropic.claude-fable-5-1"
  seat_entry = models[seat_key]
  models[seat_key] = dataclasses.replace(seat_entry, route="anthropic.oauth")
  registry = dataclasses.replace(INITIAL_MODEL_REGISTRY, models=models)
  seat = CredentialHandle("seat:owner", "anthropic", "service", "tenant", None)
  user = CredentialHandle("user-key:101:anthropic", "anthropic", "user", "tenant", "alice")

  def materialize(handle):
    offered.append(handle.handle_id)
    config = (
      {"provider": "anthropic", "api_key": "user-key", "auth_mode": "api",
       "auth_mechanism": "api_key", "entitlement_kind": "commercial_byok"}
      if handle == user else
      {"provider": "anthropic", "auth_token": "seat", "auth_mode": "oauth",
       "auth_mechanism": "oauth_bearer", "entitlement_kind": "operator_personal_subscription"}
    )
    return MaterializedCredential(handle, config)

  resolver = CapabilityExecutionResolver(
    registry=registry, selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth_context=AuthContext(
      run_mode="interactive", actor_id="alice", tenant_id="tenant",
      user_provider_handles={}, service_provider_handles={"anthropic": seat},
      entitled_capabilities=CAPABILITY_IDS, entitled_model_keys=frozenset(models),
      allow_service_for_interactive=True,
    ),
    credential_materializer=materialize, adapter_resolver=lambda _: provider,
    user_key_handles={"anthropic": user}, risk_user_id=101,
  )
  execution = resolver.resolve(
    "session.driver", explicit_intent=ModelSelectionIntent(seat_key, "high", "explicit_user"),
  )
  fallback = _AccountProvider(("F1",), [_answer("from fallback")], name="stub-fallback")
  fallback_execution = _account_execution(fallback, "F0", model="fallback-model")

  def one_hop(primary, snapshot, **_kwargs):
    return ModelFallbackHop(
      execution=fallback_execution, role="deep", chain_index=0, skipped=(),
      primary_model_key=primary.bind.model_key,
    )

  monkeypatch.setattr(CapabilityExecutionResolver, "rebind_model_fallback", staticmethod(one_hop))
  runner = _account_runner(execution)
  capture_spawning_turn = runner._dispatcher.capture_spawning_turn
  monkeypatch.setattr(
    runner._dispatcher, "capture_spawning_turn",
    lambda: SimpleNamespace(
      context=SimpleNamespace(selection_snapshot=object(), selection_provenance=None),
      root_turn=capture_spawning_turn().root_turn,
    ),
  )
  if probe is None:
    asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))
  else:
    _install_probe(probe)
    try:
      asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))
    finally:
      install_user_key_pools(None)
  assert offered == ["seat:owner"]
  assert provider.streamed == ["seat"]
  assert fallback.clients == ["F1"]
  events = _logged(runner)
  assert _types(events, "model_fallback") == ["model_fallback"]
  assert _types(events, "error") == []



def test_every_account_limited_backs_off_on_the_last_account_without_cycling(monkeypatch) -> None:
  _set_retry_policy(monkeypatch, max_retries=2)
  attempts = 5
  provider = _AccountProvider(
    ("A", "B", "C"), [_Quota429("usage limit") for _ in range(attempts)],
  )
  runner = _account_runner(_account_execution(provider, "A"))

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  # Each sibling is visited once, then the original bounded retry budget
  # stays on C even though preparation would rerank an all-blocked pool to A.
  expected = ["A", "B", "C", "C", "C"]
  assert provider.streamed == expected
  assert provider.clients == expected
  assert [call[0] for call in provider.prepared] == ["A"]
  assert provider.parked.keys() == {"A", "B", "C"}
  events = _logged(runner)
  retries = [event for event in events if event["type"] == "stream_retry"]
  assert [event["backoff_s"] for event in retries] == [0.0] * (attempts - 1)
  for retry in retries:
    _assert_retry_settlement(retry)
  assert len(_types(events, "credential_refreshed")) == 2
  assert _types(events, "model_fallback") == []
  assert _types(events, "error") == ["error"]


def test_concurrency_429_backs_off_on_the_same_account_without_rotating(monkeypatch) -> None:
  _set_retry_policy(monkeypatch, base_delay_seconds=0.25)
  slept: list[float] = []

  async def record_sleep(delay: float, result=None):
    if delay != gateway_runner.STREAM_GUARD_POLL_INTERVAL:
      slept.append(delay)
    return await asyncio.sleep(0, result)

  monkeypatch.setattr(
    gateway_runner, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": record_sleep}),
  )
  provider = _AccountProvider(("A", "B"), [_Concurrency429("concurrent"), _answer()])
  runner = _account_runner(_account_execution(provider, "A"))

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert provider.streamed == ["A", "A"]
  assert provider.parked == {}
  assert slept == [0.25]
  events = _logged(runner)
  assert _types(events, "credential_refreshed") == []
  [retry] = [event for event in events if event["type"] == "stream_retry"]
  assert retry["backoff_s"] == 0.25
  _assert_retry_settlement(retry)
  assert _types(events, "error") == []


def _assert_retry_settlement(
  event: dict[str, object], *, text_chars: int = 0, reasoning_chars: int = 0,
  preserve_partial: bool = False,
  discarded_text_chars: int = 0, discarded_reasoning_chars: int = 0,
) -> None:
  partial = preserve_partial or bool(text_chars or reasoning_chars)
  assert event["settlement"] == ("preserve_partial" if partial else "retry")
  assert event["retained_text_chars"] == text_chars
  assert event["retained_reasoning_chars"] == reasoning_chars
  assert event["discarded_text_chars"] == discarded_text_chars
  assert event["discarded_reasoning_chars"] == discarded_reasoning_chars
  for key in (
    "resumed_text_chars", "resumed_reasoning_chars",
  ):
    assert event[key] == 0
  if partial:
    assert event["backoff_s"] is None


@pytest.mark.parametrize("credential_failure", [False, True])
def test_partial_stream_failure_retains_output_without_regeneration(
  monkeypatch, credential_failure,
) -> None:
  from agent_gateway.autonomous_output import collect_run_output

  text = "Revenue grew 18%; operating margins expanded.\n" * 128
  reasoning = "Compare reported growth with constant-currency growth. " * 96
  partial = [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="thinking_delta", thinking_text=reasoning[:1300]),
    StreamEvent(type="thinking_delta", thinking_text=reasoning[1300:]),
  ]
  if text:
    partial.extend([
      StreamEvent(
        type="thinking_end", thinking_text=reasoning,
        raw_block={"type": "thinking", "thinking": reasoning, "signature": "signed"},
      ),
      StreamEvent(type="text_delta", text=text[:2000]),
      StreamEvent(type="text_delta", text=text[2000:]),
      StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    ])
  partial.append(StreamEvent(type="usage_update", output_tokens=123))
  failure = _Quota429("usage limit") if credential_failure else httpx.ReadError("peer closed stream")
  provider = _AccountProvider(("A", "B"), [(partial, failure), _answer("must not regenerate")])
  durable_events: list[dict[str, object]] = []
  runner = _refusal_runner(provider, durable_events)
  stream_turn = runner._stream_turn
  results = []
  billed = []
  runner._on_usage = billed.append

  async def capture_turn(**kwargs):
    returned = await stream_turn(**kwargs)
    results.append(returned)
    return returned

  monkeypatch.setattr(runner, "_stream_turn", capture_turn)
  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert provider.requests == 1
  assert provider.streamed == ["A"]
  assert provider.clients == ["A"]
  assert len(provider.script) == 1
  [returned] = results
  assert isinstance(returned, tuple)
  client, result = returned
  assert client == "A"
  assert result.full_text == text
  expected_blocks = [
    {"type": "thinking", "thinking": reasoning, "signature": "signed"},
    {"type": "text", "text": text},
  ] if text else []
  assert result.content_blocks == expected_blocks
  assert result.stop_reason == "partial_stream_failure"
  events = _logged(runner)
  [retry] = [event for event in events if event["type"] == "stream_retry"]
  _assert_retry_settlement(retry, text_chars=len(text), reasoning_chars=len(reasoning))
  assert retry["attempt"] == 0
  assert str(failure) in str(retry["error"])
  assert result.stop_details == {
    "error": retry["error"],
    "retained_text_chars": len(text),
    "retained_reasoning_chars": len(reasoning),
  }
  [assistant] = [event for event in durable_events if event["type"] == "assistant_message"]
  assert assistant["stop_reason"] == result.stop_reason
  assert assistant["stop_details"] == result.stop_details
  assert assistant["content_blocks"] == expected_blocks
  assert _types(events, "credential_refreshed", "model_fallback") == []
  [terminal_error] = [event for event in events if event["type"] == "error"]
  expected_terminal_error = f"partial_stream_failure: {retry['error']}"
  assert terminal_error["error"] == expected_terminal_error
  assert len(billed) == 1
  assert billed[0].output_tokens == 123
  output = collect_run_output(runner._log, timed_out=False)
  assert output.response == text.strip()
  assert output.error == expected_terminal_error


@pytest.mark.parametrize("visible_text", ["", "Partial visible answer"])
def test_reasoning_only_stream_failure_retries_and_reports_discarded_reasoning_chars(
  monkeypatch, visible_text,
) -> None:
  _set_retry_policy(monkeypatch, base_delay_seconds=0.0)
  reasoning = "partial chain of thought"
  partial = [
    StreamEvent(type="message_start", input_tokens=10),
    *([StreamEvent(type="text_delta", text=visible_text)] if visible_text else []),
    StreamEvent(type="thinking_delta", thinking_text=reasoning),
  ]
  provider = _AccountProvider(
    ("A",), [(partial, httpx.ReadError("peer closed stream")), _answer("Recovered answer.")],
  )
  durable_events: list[dict[str, object]] = []
  runner = _refusal_runner(provider, durable_events)
  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert provider.requests == (1 if visible_text else 2)
  events = _logged(runner)
  [retry] = [event for event in events if event["type"] == "stream_retry"]
  _assert_retry_settlement(
    retry, text_chars=len(visible_text), discarded_reasoning_chars=len(reasoning),
  )
  [assistant] = [event for event in durable_events if event["type"] == "assistant_message"]
  assert assistant["content_blocks"] == [{
    "type": "text", "text": visible_text or "Recovered answer.",
  }]
  assert assistant["stop_reason"] == ("partial_stream_failure" if visible_text else "end_turn")


@pytest.mark.parametrize("completed_chars", [0, 15, 29])
@pytest.mark.parametrize("suppress_tool_turn_text", [False, True])
def test_partial_text_finalization_preserves_missing_suffix_without_duplicate_blocks(
  monkeypatch, completed_chars, suppress_tool_turn_text,
) -> None:
  from agent_gateway.autonomous_output import collect_run_output

  text = "First analysis. Final answer."
  assert len(text) == 29
  partial = [StreamEvent(type="message_start", input_tokens=10)]
  expected_blocks = []
  if completed_chars:
    completed = text[:completed_chars]
    partial.extend([
      StreamEvent(type="text_delta", text=completed),
      StreamEvent(type="text_end", raw_block={"type": "text", "text": completed}),
    ])
    expected_blocks.append({"type": "text", "text": completed})
  suffix = text[completed_chars:]
  if suffix:
    partial.append(StreamEvent(type="text_delta", text=suffix))
    expected_blocks.append({"type": "text", "text": suffix})
  provider = _AccountProvider(
    ("A",), [(partial, httpx.ReadError("peer closed stream")), _answer("must not regenerate")],
  )
  durable_events: list[dict[str, object]] = []
  runner = _refusal_runner(provider, durable_events)
  stream_turn = runner._stream_turn
  results = []

  async def capture_turn(**kwargs):
    returned = await stream_turn(**kwargs)
    results.append(returned)
    return returned

  monkeypatch.setattr(runner, "_stream_turn", capture_turn)
  prompt = "Tool-call messages are tool-only." if suppress_tool_turn_text else "go"
  asyncio.run(runner.run(messages=[{"role": "user", "content": prompt}]))

  assert provider.requests == 1
  assert provider.streamed == ["A"]
  [returned] = results
  assert isinstance(returned, tuple)
  assert returned[1].full_text == text
  assert returned[1].content_blocks == expected_blocks
  assert returned[1].stop_reason == "partial_stream_failure"
  [assistant] = [event for event in durable_events if event["type"] == "assistant_message"]
  assert assistant["content_blocks"] == expected_blocks
  events = _logged(runner)
  assert "".join(str(event["text"]) for event in events if event["type"] == "text_delta") == text
  [retry] = [event for event in events if event["type"] == "stream_retry"]
  _assert_retry_settlement(retry, text_chars=len(text))
  assert collect_run_output(runner._log, timed_out=False).response == text


def test_text_transform_holds_the_turns_deltas_and_emits_the_transformed_text_once(monkeypatch) -> None:
  # A repair turn's text is not streamed as produced: the deltas are held, the
  # caller's transform runs once on the complete text, and that text is what
  # the client, the content block and the validation read.
  from agent_gateway.autonomous_output import collect_run_output

  provider = _AccountProvider(("A",), [[
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="text_delta", text="1. revenue "),
    StreamEvent(type="text_delta", text="was $5M [S1]"),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": "1. revenue was $5M [S1]"}),
    StreamEvent(type="usage_update", output_tokens=2),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]])
  durable_events: list[dict[str, object]] = []
  runner = _refusal_runner(provider, durable_events)
  stream_turn = runner._stream_turn
  seen: list[str] = []
  results = []

  def splice(reply: str) -> str:
    seen.append(reply)
    return f"Draft head. {reply[3:]} Draft tail."

  async def capture_turn(**kwargs):
    returned = await stream_turn(**{**kwargs, "text_transform": splice})
    results.append(returned)
    return returned

  monkeypatch.setattr(runner, "_stream_turn", capture_turn)
  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert seen == ["1. revenue was $5M [S1]"]
  [returned] = results
  assert isinstance(returned, tuple)
  assert returned[1].full_text == "Draft head. revenue was $5M [S1] Draft tail."
  assert returned[1].content_blocks == [{"type": "text", "text": "Draft head. revenue was $5M [S1] Draft tail."}]
  deltas = [event["text"] for event in _logged(runner) if event["type"] == "text_delta"]
  assert deltas == ["Draft head. revenue was $5M [S1] Draft tail."]
  assert collect_run_output(runner._log, timed_out=False).response == "Draft head. revenue was $5M [S1] Draft tail."


def test_text_transform_is_not_applied_to_a_reply_cut_at_max_tokens(monkeypatch) -> None:
  # The transform reads a whole reply; a cut-off one is returned raw with
  # nothing emitted, for the run loop to hold until its continuation
  # (audit 1b7646d5b287).
  provider = _AccountProvider(("A",), [[
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="text_delta", text="1. revenue was"),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": "1. revenue was"}),
    StreamEvent(type="usage_update", output_tokens=2),
    StreamEvent(type="message_end", stop_reason="max_tokens"),
  ], [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="text_delta", text="done"),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": "done"}),
    StreamEvent(type="usage_update", output_tokens=2),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]])
  durable_events: list[dict[str, object]] = []
  runner = _refusal_runner(provider, durable_events)
  stream_turn = runner._stream_turn
  seen: list[str] = []
  results = []

  async def capture_turn(**kwargs):
    returned = await stream_turn(**{**kwargs, "text_transform": seen.append if not results else None})
    results.append(returned)
    return returned

  monkeypatch.setattr(runner, "_stream_turn", capture_turn)
  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert seen == []
  first = results[0]
  assert isinstance(first, tuple)
  assert (first[1].stop_reason, first[1].full_text) == ("max_tokens", "1. revenue was")
  deltas = [event["text"] for event in _logged(runner) if event["type"] == "text_delta"]
  assert deltas == ["done"]



def test_partial_completed_tool_turn_keeps_tool_only_text_suppression(monkeypatch) -> None:
  from agent_gateway.autonomous_output import collect_run_output

  lookups: list[dict] = []

  async def lookup(tool_input, *, call_index=0, tool_ctx=None):
    lookups.append(dict(tool_input))
    return {"status": "success"}, None

  hidden_text = " Let me check that tool first."
  partial = [
    StreamEvent(type="text_delta", text=hidden_text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": hidden_text}),
    *_tool_turn("tool_1", "lookup")[:-1],
  ]
  provider = _AccountProvider(
    ("A",), [(partial, httpx.ReadError("peer closed stream")), _answer("Final answer.")],
  )
  runner = _account_runner(
    _account_execution(provider, "A"), handlers={"lookup": lookup}, tool_names=("lookup",),
  )
  stream_turn = runner._stream_turn
  results = []

  async def capture_turn(**kwargs):
    returned = await stream_turn(**kwargs)
    results.append(returned)
    return returned

  monkeypatch.setattr(runner, "_stream_turn", capture_turn)
  asyncio.run(runner.run(messages=[{
    "role": "user", "content": "Tool-call messages are tool-only.",
  }]))

  assert provider.streamed == ["A", "A"]
  assert provider.requests == 2
  assert lookups == [{}]
  first = results[0]
  assert isinstance(first, tuple)
  turn = first[1]
  assert turn.full_text == ""
  assert turn.stop_reason == "partial_stream_failure"
  assert turn.content_blocks == [
    {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {}},
  ]
  assert turn.stop_details is not None
  assert turn.stop_details["retained_text_chars"] == 0
  events = _logged(runner)
  [retry] = [event for event in events if event["type"] == "stream_retry"]
  _assert_retry_settlement(
    retry, preserve_partial=True,
    discarded_text_chars=len(" Let me check that tool first."),
  )
  assert [event["text"] for event in events if event["type"] == "text_delta"] == ["Final answer."]
  output = collect_run_output(runner._log, timed_out=False)
  assert output.response == "Final answer."
  assert output.tools_used == ["lookup"]
  assert output.error is None


@pytest.mark.parametrize("settlement", [None, "retry", "preserve_partial"])
def test_autonomous_retry_settlement_retains_only_preserved_output(settlement) -> None:
  from agent_gateway.autonomous_output import collect_run_output

  retry = {"type": "stream_retry"}
  if settlement is not None:
    retry["settlement"] = settlement
  log = SimpleNamespace(entries=[
    SimpleNamespace(event=event) for event in [
      {"type": "text_delta", "text": "retained analysis"},
      {"type": "tool_call_start", "tool_name": "lookup"},
      retry,
      {"type": "stream_complete", "terminal_disposition": "completed", "usage": {}},
    ]
  ])
  output = collect_run_output(log, timed_out=False)
  assert output.response == ("retained analysis" if settlement == "preserve_partial" else "")
  assert output.tools_used == (["lookup"] if settlement == "preserve_partial" else [])
  assert output.error is None


def test_a_completed_tool_call_runs_once_when_the_next_request_rotates_accounts() -> None:
  lookups: list[dict] = []

  async def lookup(tool_input, *, call_index=0, tool_ctx=None):
    _ = call_index, tool_ctx
    lookups.append(dict(tool_input))
    return {"status": "success"}, None

  provider = _AccountProvider(
    ("A", "B"), [_tool_turn("tool_1", "lookup"), _Quota429("usage limit"), _answer()],
  )
  runner = _account_runner(
    _account_execution(provider, "A"), handlers={"lookup": lookup}, tool_names=("lookup",),
  )

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert provider.streamed == ["A", "A", "B"]
  assert lookups == [{}]
  events = _logged(runner)
  assert _types(events, "credential_refreshed") == ["credential_refreshed"]
  assert _types(events, "error") == []


def test_user_credit_failover_rebuilds_stream_and_attributes_answer(tmp_path, monkeypatch):
  from hank_model_transport.capability_execution import MaterializedCredential
  from hank_model_transport.user_provider_keys import (
    ResolveContext, ResolvedKey, ResolvedUserKeys, UserKeyPools, UserKeysExhausted,
    install_user_key_pools, installed_user_key_pools,
    consumer_key_hash_for,
  )
  from model_authority.binding import AuthContext, CredentialHandle, ModelSelectionIntent
  from model_authority.capabilities import CAPABILITY_IDS
  from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
  from model_authority.spend_ledger import fingerprint_credential

  monkeypatch.setenv("HANK_SPEND_LEDGER_PATH", str(tmp_path / "spend.sqlite"))
  monkeypatch.setenv("HANK_SPEND_SETTINGS_PATH", str(tmp_path / "settings.json"))

  class Source:
    def resolve(self, *, risk_user_id, provider, context):
      secret = f"fixture-{provider}"
      return ResolvedUserKeys(provider, risk_user_id, (
        ResolvedKey(fingerprint_credential(secret), "mine", secret),
      ))

    def report(self, **kwargs):
      raise AssertionError("the scripted adapter owns this credit outcome")

  class CreditProvider(_RefusingProvider):
    def get_model_info(self, model):
      return ModelInfo(id=model, provider=self.name, max_output_tokens=64000, supports_thinking=True)

    async def stream(self, client, params):
      if self.name == "anthropic":
        raise RuntimeError("primary credit denied")
      async for event in super().stream(client, params):
        yield event

    def classify_credential_failure(self, exc):
      return ProviderCredentialFailure(provider=self.name, kind="billing")

    def next_credential(self, config, failure):
      raise UserKeysExhausted(self.name)

  primary, target = CreditProvider(("end_turn",)), CreditProvider(("end_turn",))
  primary.name, target.name = "anthropic", "openai"
  handles = {
    provider: CredentialHandle(f"user-key:101:{provider}", provider, "user", "tenant", "alice")
    for provider in ("anthropic", "openai")
  }

  def materialize(handle):
    return MaterializedCredential(handle, installed_user_key_pools().lease(
      risk_user_id=101, provider=handle.provider,
      context=ResolveContext(consumer_key_hash=consumer_key_hash_for("risk-api-key")),
    ).auth_config())

  install_user_key_pools(UserKeyPools(Source()))
  try:
    resolver = CapabilityExecutionResolver(
      registry=INITIAL_MODEL_REGISTRY, selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      auth_context=AuthContext(
        run_mode="interactive", actor_id="alice", tenant_id="tenant",
        user_provider_handles=handles, service_provider_handles={},
        entitled_capabilities=CAPABILITY_IDS, entitled_model_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
      ),
      credential_materializer=materialize,
      adapter_resolver=lambda adapter: target if adapter.startswith("openai") else primary,
      risk_user_id=101,
    )
    execution = resolver.resolve(
      "session.driver", role="fast",
      explicit_intent=ModelSelectionIntent("anthropic.claude-haiku-5-5", "none", "explicit_user"),
    )
    runner = _account_runner(execution)
    asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))
    events = _logged(runner)
    assert not [event for event in events if event["type"] == "error"]
    [failover] = [event for event in events if event["type"] == "model_failover"]
    assert failover["tier"] == "failover" and failover["failover_reason"] == "out_of_credit"
    assert failover["failed_over_from"] == execution.bind.model_key
    assert runner._capability_execution.bind.model_key == "openai.gpt-6-1-sol"
    assert primary.requests == 1 and target.requests == 1
    [answer] = [event for event in events if event["type"] == "turn_complete"]
    usage = answer["usage"]
    assert isinstance(usage, dict) and usage["tier"] == "failover"
  finally:
    install_user_key_pools(None)


@pytest.mark.parametrize("key_count", [1, 2])
def test_openai_user_429_rotates_or_raises_typed_cooldown(key_count, tmp_path, monkeypatch):
  import threading
  from hank_model_transport.capability_execution import MaterializedCredential
  from hank_model_transport.providers.openai import OpenAIProvider
  from hank_model_transport.user_provider_keys import (
    ResolveContext, ResolvedKey, ResolvedUserKeys, UserKeyPools, UserKeysRateLimited,
    consumer_key_hash_for, install_user_key_pools,
  )
  from model_authority.binding import AuthContext, CredentialHandle, ModelSelectionIntent
  from model_authority.capabilities import CAPABILITY_IDS
  from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
  from model_authority.spend_ledger import fingerprint_credential

  monkeypatch.setenv("HANK_SPEND_LEDGER_PATH", str(tmp_path / "spend.sqlite"))
  monkeypatch.setenv("HANK_SPEND_SETTINGS_PATH", str(tmp_path / "settings.json"))
  event_loop_thread = threading.get_ident()
  secrets = tuple(f"sk-openai-user-{index}" for index in range(key_count))
  resolves = []

  class Source:
    def resolve(self, *, risk_user_id, provider, context):
      resolves.append(threading.get_ident())
      return ResolvedUserKeys(provider, risk_user_id, tuple(
        ResolvedKey(fingerprint_credential(secret), "mine", secret) for secret in secrets
      ))

    def report(self, **kwargs):
      raise AssertionError("a transient 429 must not suspend a key at Risk")

  class RateLimitError(RuntimeError):
    status_code = 429
    response = SimpleNamespace(headers={"retry-after": "60"})

  error = RateLimitError("rate_limit_exceeded")

  class UserOpenAI(_NoPrepareProvider):
    def __init__(self):
      super().__init__(secrets, [error, _answer()], name="openai")
      self.adapter = OpenAIProvider()

    def get_model_info(self, model):
      return ModelInfo(id=model, provider=self.name, max_output_tokens=64000, supports_thinking=True)

    def classify_credential_failure(self, exc):
      return self.adapter.classify_credential_failure(exc)

    def next_credential(self, config, failure):
      assert threading.get_ident() != event_loop_thread
      return self.adapter.next_credential(config, failure)

  provider = UserOpenAI()
  pools = UserKeyPools(Source())
  handle = CredentialHandle("user-key:101:openai", "openai", "user", "tenant", "alice")

  def materialize(selected):
    return MaterializedCredential(selected, pools.lease(
      risk_user_id=101, provider="openai",
      context=ResolveContext(consumer_key_hash=consumer_key_hash_for("risk-api-key")),
    ).auth_config())

  install_user_key_pools(pools)
  try:
    resolver = CapabilityExecutionResolver(
      registry=INITIAL_MODEL_REGISTRY, selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      auth_context=AuthContext(
        run_mode="interactive", actor_id="alice", tenant_id="tenant",
        user_provider_handles={"openai": handle}, service_provider_handles={},
        entitled_capabilities=CAPABILITY_IDS, entitled_model_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
      ),
      credential_materializer=materialize, adapter_resolver=lambda _: provider, risk_user_id=101,
    )
    execution = resolver.resolve(
      "session.driver", explicit_intent=ModelSelectionIntent("openai.gpt-6-1-sol", "low", "explicit_user"),
    )
    runner = _account_runner(execution)
    before = time.time()
    if key_count == 1:
      with pytest.raises(UserKeysRateLimited) as caught:
        asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))
      assert before + 60 <= caught.value.retry_at <= time.time() + 60
      assert provider.streamed == [secrets[0]]
      assert len(resolves) == 2 and resolves[-1] != event_loop_thread
    else:
      asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))
      assert provider.streamed == list(secrets)
      assert runner._auth_config["api_key"] == secrets[1]
      assert _types(_logged(runner), "credential_refreshed") == ["credential_refreshed"]
      assert _types(_logged(runner), "error", "model_failover", "model_fallback") == []
      assert len(resolves) == 2 and resolves[-1] != event_loop_thread
  finally:
    install_user_key_pools(None)
