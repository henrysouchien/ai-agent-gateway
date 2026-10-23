from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, ToolDispatcher  # noqa: E402
from agent_gateway.agent_session_log import AgentSessionLog  # noqa: E402
from agent_gateway.auth import ProviderCredentialFailure  # noqa: E402
from agent_gateway.capability_execution import (  # noqa: E402
  CapabilityExecutionResolver,
  ModelFallbackHop,
)
from agent_gateway.event_log import EventLog  # noqa: E402
from agent_gateway.containment_settings import ContainmentSettings  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from agent_gateway.multi_user.billing import UsageTotals  # noqa: E402
from agent_gateway.providers import ModelInfo, ModelProvider
from model_authority.thinking import ThinkingLevel
from agent_gateway.providers.base import StreamEvent  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_stream_turn import RunnerStreamTurnMixin  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from model_authority.thinking import EffortResolution
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_bound_capability_execution,
  stub_runner_capability_execution,
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
  from agent_gateway.providers import AnthropicProvider
  import agent_gateway.providers.anthropic as adapter

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
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0)
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


def test_runner_stream_turn_methods_are_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerStreamTurnMixin)
  assert gateway_runner.RunnerStreamTurnMixin is RunnerStreamTurnMixin

  for method_name in (
    "_thinking_level",
    "_effective_stream_stall_timeout",
    "_classify_guard_outcome",
    "_stream_turn",
  ):
    assert getattr(AgentRunner, method_name) is getattr(RunnerStreamTurnMixin, method_name)


def test_stream_turn_helpers_resolve_parent_module_aliases(monkeypatch) -> None:
  runner = object.__new__(AgentRunner)
  runner._stream_stall_timeout = None
  calls: dict[str, object] = {}

  monkeypatch.setattr(gateway_runner, "STREAM_STALL_TIMEOUT", 12.0)
  monkeypatch.setattr(gateway_runner, "STREAM_THINKING_STALL_TIMEOUT", 34.0)
  monkeypatch.setattr(gateway_runner, "thinking_level", lambda enabled: f"patched-{enabled}")

  def _effective_stream_stall_timeout(override, **kwargs):
    calls["override"] = override
    calls["kwargs"] = kwargs
    return 56.0

  def _classify_guard_outcome(guard_reason, attempt, max_attempts):
    calls["guard"] = (guard_reason, attempt, max_attempts)
    return ("patched", "guard", "kind")

  monkeypatch.setattr(gateway_runner, "effective_stream_stall_timeout", _effective_stream_stall_timeout)
  monkeypatch.setattr(gateway_runner, "classify_guard_outcome", _classify_guard_outcome)

  model_info = ModelInfo(id="model", provider="stub", supports_thinking=True)

  assert AgentRunner._thinking_level(True) == "patched-True"
  assert (
    runner._effective_stream_stall_timeout(
      config={"thinking": True},
      model_info=model_info,
      max_tokens=128,
    )
    == 56.0
  )
  assert AgentRunner._classify_guard_outcome(("stall", "quiet"), 1, 3) == ("patched", "guard", "kind")

  assert calls["kwargs"] == {
    "config": {"thinking": True},
    "model_info": model_info,
    "max_tokens": 128,
    "observed_thinking": False,
    "stream_stall_timeout_default": 12.0,
    "stream_thinking_stall_timeout_default": 34.0,
    "effort_resolution": None,
  }
  assert calls["guard"] == (("stall", "quiet"), 1, 3)


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
      return ModelInfo(id=model, provider=self.name, supports_thinking=True)


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
      super().__init__(McpClientManager(config_path=None), capability_bind=capability_bind)

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
    model_info=ModelInfo(id="model", provider="stub"),
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
      return ModelInfo(id=model, provider=self.name, supports_thinking=True)


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
        model_info=ModelInfo(id="model", provider="stub"),
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
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 0)

  tool_input = {"body": "x" * 20}
  tool_block = {"type": "tool_use", "id": "call-1", "name": "write_memo"}

  class _Provider(ModelProvider):
    name = "stub"

    def has_active_credential(self, config):
      return bool(config.get("api_key"))

    def get_model_info(self, model):
      return ModelInfo(id=model, provider=self.name, supports_thinking=True)

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
    model_info=ModelInfo(id="model", provider="stub"),
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
    return ModelInfo(id=model, provider=self.name)

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
  from agent_gateway.auth import ProviderCredentialFailure

  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 2)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0)

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
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 0)
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
    model_info=ModelInfo(id="claude-opus-5", provider="stub"),
    system_prompt=None, current_messages=[], base_kwargs={"tools": []},
    max_tokens=128, turn_count=1, turn_t0=0.0, turn_t0_mono=0.0,
    system_chars=0, tools_chars=0,
    usage_totals={"input_tokens": 0, "output_tokens": 0,
                  "cache_creation_tokens": 0, "cache_read_tokens": 0},
  ))
  if silent_after_queue:
    assert result is None
    consumer_task = consumer[0]
    assert consumer_task is not None
    assert consumer_task.cancelled()
    error = runner._terminal_closure_event
    assert error is not None
    assert error["type"] == "error"
    assert "no stream progress for 10s" in error["error"]
    assert polls[0] == 3
  else:
    assert isinstance(result, tuple)
    assert result[1].full_text == "progress"
    assert result[1].stop_reason == "end_turn"

@pytest.mark.parametrize("silent_after_queue", [False, True])
def test_queued_child_heartbeats_preserve_parent_liveness_and_peer_stall(
  monkeypatch, silent_after_queue,
) -> None:
  import time

  from agent_gateway.rate_limit import AnthropicTokenBucket, locally_queued
  import agent_gateway.runner_sub_agents as children
  import agent_gateway.runner_stream_turn as stream_turn_module

  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", 0.01)
  monkeypatch.setattr(stream_turn_module, "STREAM_PROGRESS_LOG_INTERVAL", 0.03)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 0)
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
        model_info=ModelInfo(id="claude-opus-5", provider="stub"),
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
    return isinstance(exc, (_Quota429, _Concurrency429, _Transient))

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
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0)


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


def test_resumed_runner_prepares_with_the_durable_log_identity(tmp_path: Path) -> None:
  log = AgentSessionLog(tmp_path / "original-conversation.jsonl")
  provider = _AccountProvider(("A",), [])
  for runtime_id in ("original-conversation", "resumed-runtime"):
    session = GatewaySession(
      session_id=runtime_id, api_key_hash="hash", created_at=1, expires_at=2,
      user_id="alice", tenant_id="test-tenant",
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
    user_id="alice", tenant_id="test-tenant",
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


def test_every_account_limited_backs_off_on_the_last_account_without_cycling() -> None:
  attempts = 3 + gateway_runner.STREAM_RETRY_MAX
  provider = _AccountProvider(
    ("A", "B", "C"), [_Quota429("usage limit") for _ in range(attempts)],
  )
  runner = _account_runner(_account_execution(provider, "A"))

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  # Each sibling is visited once, then the original bounded retry budget
  # stays on C even though preparation would rerank an all-blocked pool to A.
  expected = ["A", "B"] + ["C"] * (1 + gateway_runner.STREAM_RETRY_MAX)
  assert provider.streamed == expected
  assert provider.clients == expected
  assert [call[0] for call in provider.prepared] == ["A"]
  assert provider.parked.keys() == {"A", "B", "C"}
  events = _logged(runner)
  retries = [event for event in events if event["type"] == "stream_retry"]
  assert [event["backoff_s"] for event in retries] == [0.0] * (attempts - 1)
  assert len(_types(events, "credential_refreshed")) == 2
  assert _types(events, "model_fallback") == []
  assert _types(events, "error") == ["error"]


def test_concurrency_429_backs_off_on_the_same_account_without_rotating(monkeypatch) -> None:
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.25)
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
  assert _types(events, "error") == []


def test_partial_stream_retry_events_are_unchanged() -> None:
  partial = [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="text_delta", text="par"),
  ]
  provider = _AccountProvider(("A",), [(partial, _Transient("transient")), _answer()])
  runner = _account_runner(_account_execution(provider, "A"))

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  events = _logged(runner)
  assert _types(events, "text_delta", "stream_retry", "credential_refreshed") == [
    "text_delta", "stream_retry", "text_delta",
  ]
  [retry] = [event for event in events if event["type"] == "stream_retry"]
  assert set(retry) >= {"attempt", "error", "attempt_elapsed_s", "backoff_s"}
  assert retry["attempt"] == 0
  assert "transient" in str(retry["error"])
  assert retry["backoff_s"] == 0
  assert provider.streamed == ["A", "A"]
  assert _types(events, "error") == []


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
