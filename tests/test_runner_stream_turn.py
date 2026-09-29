import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, ToolDispatcher  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from agent_gateway.providers import ModelInfo, ModelProvider
from model_authority.thinking import ThinkingLevel
from agent_gateway.providers.base import StreamEvent  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_stream_turn import RunnerStreamTurnMixin  # noqa: E402
from model_authority.thinking import EffortResolution
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_runner_capability_execution,
)


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
    def __init__(self) -> None:
      super().__init__(McpClientManager(config_path=None))

    def redact_raw_tool_input_for_history(self, tool_name, tool_input):
      _ = self, tool_name, tool_input
      raise AssertionError(
        "stream normalization must not redact model semantic history"
      )

    def prepare_tool_call(self, *_args, **_kwargs):
      raise AssertionError("assistant history must not prepare executable input")

  provider = _Provider()
  runner = object.__new__(AgentRunner)
  runner._provider = provider
  runner._dispatcher = _Dispatcher()
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
  provider: _RefusingProvider,
  durable_events: list[dict[str, object]],
) -> AgentRunner:
  from agent_gateway.event_log import EventLog
  from gateway_test_support.capability_execution_test_support import (
    stub_bound_capability_execution,
  )

  event_log = EventLog()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={},
      event_log=event_log,
      session_id="sess-refusal",
      get_tool_definitions=None,
    ),
    session_id="sess-refusal",
    capability_execution=stub_bound_capability_execution(
      provider=provider,
      model="claude-opus-5",
      effort="none",
      auth_config={"api_key": "k"},
    ),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  async def _append_durable_event(event):
    durable_events.append(dict(event))
    return SimpleNamespace(seq=len(durable_events))

  runner._append_durable_event = _append_durable_event
  return runner


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
      max_turns=3,
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
