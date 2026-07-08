import asyncio
import json
import logging
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog, GatewaySession, ModelInfo, ModelProvider, ToolDispatcher  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.multi_user.billing import SessionUsageSummary, UsageEvent  # noqa: E402
from agent_gateway.providers import CostEstimate, StreamEvent  # noqa: E402
from agent_gateway.runner_usage import (  # noqa: E402
  apply_message_start_usage,
  apply_usage_update,
  build_usage_event,
  call_late_usage_event_hook,
  call_session_summary_hook,
  call_usage_event_hook,
  empty_usage_totals,
  estimate_usage_cost,
  turn_usage_payload,
  usage_delta,
  usage_delta_state,
  usage_has_tokens,
)


def test_runner_usage_wrappers_resolve_parent_module_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(gateway_runner, "_usage_has_tokens", lambda usage_totals: usage_totals.get("patched") == 1)
  monkeypatch.setattr(gateway_runner, "_usage_delta", lambda before, after: {"patched": after["x"] - before["x"]})

  assert AgentRunner._usage_has_tokens({"patched": 1}) is True
  assert AgentRunner._usage_delta({"x": 2}, {"x": 5}) == {"patched": 3}

  runner = object.__new__(AgentRunner)
  runner._usage_user_id = "alice"
  runner._full_session_id = "sess"
  runner._request_id = "req"
  runner._parent_turn_id = "turn"
  runner._provider = SimpleNamespace(name="stub")
  runner._rate_table_version = "v1"
  runner._billing_mode = "metered"
  runner._channel = "web"
  runner._estimate_usage_cost = lambda _model, _usage_totals: CostEstimate(total=0.5)  # type: ignore[method-assign]
  monkeypatch.setattr(gateway_runner, "time", SimpleNamespace(time=lambda: 42.0))
  monkeypatch.setattr(
    gateway_runner,
    "_build_usage_event",
    lambda **kwargs: {"patched": kwargs},
  )

  event = AgentRunner._build_usage_event(runner, model="model", usage_totals={"input_tokens": 1})

  assert event["patched"]["timestamp"] == 42.0
  assert event["patched"]["cost_total"] == 0.5
  assert event["patched"]["provider_name"] == "stub"


def _run(coro):
  return asyncio.run(coro)


def _usage_event() -> UsageEvent:
  return UsageEvent(
    user_id="alice",
    session_id="sess-parent",
    request_id="req-123",
    parent_turn_id=None,
    timestamp=123.0,
    model="claude-sonnet-4-6",
    provider="stub",
    input_tokens=10,
    output_tokens=5,
    cache_read_tokens=1,
    cache_creation_tokens=2,
    cost_usd=0.01,
    rate_table_version="2026-04-08",
    billing_mode="metered",
    channel="web",
  )


def _session_summary() -> SessionUsageSummary:
  return SessionUsageSummary(
    user_id="alice",
    session_id="sess-parent",
    request_id="req-123",
    input_tokens=10,
    output_tokens=5,
    cache_read_tokens=1,
    cache_creation_tokens=2,
    cost=0.01,
    turns=1,
    channel="web",
    started_at=100.0,
    ended_at=123.0,
    model="claude-sonnet-4-6",
    provider="stub",
    rate_table_version="2026-04-08",
    billing_mode="metered",
  )


class _RecordingUsageAggregator:
  def __init__(self, *, recorded: bool = True) -> None:
    self.recorded = recorded
    self.events: list[UsageEvent] = []

  async def record(self, event: UsageEvent) -> bool:
    self.events.append(event)
    return self.recorded


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _UsageProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      input_cost_per_mtok=1.0,
      output_cost_per_mtok=2.0,
      cache_read_cost_per_mtok=0.5,
      cache_write_cost_per_mtok=0.75,
    )

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(type="message_start", input_tokens=100, cache_read_tokens=10, cache_creation_tokens=5)
    yield StreamEvent(type="text_delta", text="hello ")
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "hello "})
    yield StreamEvent(type="usage_update", output_tokens=50)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def test_usage_helper_functions_match_runner_usage_contract() -> None:
  assert empty_usage_totals() == {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
  }

  before = {
    "input_tokens": 20,
    "output_tokens": 10,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }
  after = {
    "input_tokens": "30",
    "output_tokens": 9,
    "cache_read_input_tokens": 8,
    "cache_creation_input_tokens": 7,
  }

  delta = usage_delta(before, after)

  assert delta == {
    "input_tokens": 10,
    "output_tokens": 0,
    "cache_read_input_tokens": 3,
    "cache_creation_input_tokens": 5,
  }
  assert usage_has_tokens(delta) is True
  assert usage_has_tokens({key: 0 for key in delta}) is False
  assert AgentRunner._usage_delta(before, after) == delta
  assert AgentRunner._usage_has_tokens(delta) is True


def test_usage_delta_state_returns_delta_and_token_flag() -> None:
  before = {
    "input_tokens": 20,
    "output_tokens": 10,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }
  after = {
    "input_tokens": "25",
    "output_tokens": 10,
    "cache_read_input_tokens": 3,
    "cache_creation_input_tokens": 2,
  }

  state = usage_delta_state(before, after)

  assert state.usage == {
    "input_tokens": 5,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }
  assert state.has_tokens is True


def test_usage_delta_state_reports_false_for_empty_delta() -> None:
  usage = {
    "input_tokens": 20,
    "output_tokens": 10,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }

  state = usage_delta_state(usage, dict(usage))

  assert state.usage == {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }
  assert state.has_tokens is False


def test_apply_message_start_usage_mutates_existing_totals() -> None:
  usage = {
    "input_tokens": 20,
    "output_tokens": 4,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }

  result = apply_message_start_usage(
    usage,
    input_tokens=30,
    cache_read_tokens=8,
    cache_creation_tokens=7,
  )

  assert result is usage
  assert usage == {
    "input_tokens": 50,
    "output_tokens": 4,
    "cache_read_input_tokens": 13,
    "cache_creation_input_tokens": 9,
  }


def test_apply_usage_update_mutates_output_tokens_only() -> None:
  usage = {
    "input_tokens": 20,
    "output_tokens": 4,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }

  result = apply_usage_update(usage, output_tokens=11)

  assert result is usage
  assert usage == {
    "input_tokens": 20,
    "output_tokens": 15,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }


def test_turn_usage_payload_copies_usage_and_rounds_cost() -> None:
  usage = {
    "input_tokens": 10,
    "output_tokens": 5,
    "cache_read_input_tokens": 3,
    "cache_creation_input_tokens": 2,
  }

  payload = turn_usage_payload(usage, estimated_cost=0.123456)
  usage["input_tokens"] = 99

  assert payload == {
    "input_tokens": 10,
    "output_tokens": 5,
    "cache_read_input_tokens": 3,
    "cache_creation_input_tokens": 2,
    "estimated_cost": 0.1235,
  }


def test_turn_usage_payload_omits_cost_when_not_provided() -> None:
  assert turn_usage_payload({
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }) == {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }


def test_estimate_usage_cost_passes_uncached_and_cache_token_counts() -> None:
  calls: list[tuple[Any, ...]] = []

  class _CostProvider:
    def estimate_cost(
      self,
      model: str,
      input_tokens: int,
      output_tokens: int,
      *,
      cache_read_tokens: int = 0,
      cache_creation_tokens: int = 0,
    ) -> CostEstimate:
      calls.append((model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens))
      return CostEstimate(total=0.42)

  cost = estimate_usage_cost(
    _CostProvider(),
    "claude-sonnet-4-6",
    {
      "input_tokens": 100,
      "output_tokens": 50,
      "cache_read_input_tokens": 10,
      "cache_creation_input_tokens": 5,
    },
  )

  assert cost.total == 0.42
  assert calls == [("claude-sonnet-4-6", 85, 50, 10, 5)]


def test_build_usage_event_helper_sets_billing_fields() -> None:
  event = build_usage_event(
    user_id="alice",
    session_id="sess-parent",
    request_id="req-123",
    parent_turn_id="tool-run-agent-1",
    timestamp=123.5,
    model="claude-sonnet-4-6",
    provider_name="stub",
    usage_totals={
      "input_tokens": 100,
      "output_tokens": 50,
      "cache_read_input_tokens": 10,
      "cache_creation_input_tokens": 5,
    },
    cost_total=0.25,
    rate_table_version="2026-04-08",
    billing_mode="metered",
    channel="web",
  )

  assert event.user_id == "alice"
  assert event.session_id == "sess-parent"
  assert event.request_id == "req-123"
  assert event.parent_turn_id == "tool-run-agent-1"
  assert event.timestamp == 123.5
  assert event.model == "claude-sonnet-4-6"
  assert event.provider == "stub"
  assert event.input_tokens == 100
  assert event.output_tokens == 50
  assert event.cache_read_tokens == 10
  assert event.cache_creation_tokens == 5
  assert event.cost_usd == 0.25
  assert event.rate_table_version == "2026-04-08"
  assert event.billing_mode == "metered"
  assert event.channel == "web"


def test_call_usage_event_hook_records_and_invokes_async_callback() -> None:
  event = _usage_event()
  aggregator = _RecordingUsageAggregator()
  received: list[UsageEvent] = []
  metrics: list[tuple[str, int]] = []

  async def _on_usage(usage_event: UsageEvent) -> None:
    received.append(usage_event)

  _run(
    call_usage_event_hook(
      aggregator,
      event,
      is_summary_emitted=lambda: False,
      on_usage=_on_usage,
      on_late_usage_event=None,
      emit_metric=lambda name, value: metrics.append((name, value)),
      dlq_path=None,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  assert aggregator.events == [event]
  assert received == [event]
  assert metrics == []


@pytest.mark.parametrize("recorded, summary_emitted", [(False, False), (True, True)])
def test_call_usage_event_hook_routes_late_events(recorded: bool, summary_emitted: bool) -> None:
  event = _usage_event()
  aggregator = _RecordingUsageAggregator(recorded=recorded)
  late_events: list[UsageEvent] = []

  def _unexpected_on_usage(_usage_event: UsageEvent) -> None:
    raise AssertionError("on_usage should not run for late usage events")

  _run(
    call_usage_event_hook(
      aggregator,
      event,
      is_summary_emitted=lambda: summary_emitted,
      on_usage=_unexpected_on_usage,
      on_late_usage_event=late_events.append,
      emit_metric=lambda _name, _value: None,
      dlq_path=None,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  assert aggregator.events == [event]
  assert late_events == [event]


def test_call_usage_event_hook_checks_summary_flag_after_record() -> None:
  event = _usage_event()
  summary_emitted = False
  late_events: list[UsageEvent] = []

  class _FlippingUsageAggregator:
    async def record(self, usage_event: UsageEvent) -> bool:
      nonlocal summary_emitted
      assert usage_event == event
      summary_emitted = True
      return True

  def _unexpected_on_usage(_usage_event: UsageEvent) -> None:
    raise AssertionError("on_usage should not run after summary emission")

  _run(
    call_usage_event_hook(
      _FlippingUsageAggregator(),
      event,
      is_summary_emitted=lambda: summary_emitted,
      on_usage=_unexpected_on_usage,
      on_late_usage_event=late_events.append,
      emit_metric=lambda _name, _value: None,
      dlq_path=None,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  assert late_events == [event]


def test_call_usage_event_hook_failure_records_metric_and_dlq(tmp_path: Path) -> None:
  event = _usage_event()
  metrics: list[tuple[str, int]] = []
  dlq_path = tmp_path / "usage_dlq.jsonl"

  def _failing_on_usage(_usage_event: UsageEvent) -> None:
    raise RuntimeError("ledger offline")

  _run(
    call_usage_event_hook(
      _RecordingUsageAggregator(),
      event,
      is_summary_emitted=lambda: False,
      on_usage=_failing_on_usage,
      on_late_usage_event=None,
      emit_metric=lambda name, value: metrics.append((name, value)),
      dlq_path=dlq_path,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  payload = json.loads(dlq_path.read_text(encoding="utf-8").strip())
  assert metrics == [("gateway.usage_event_dropped", 1)]
  assert payload["event_id"] == event.event_id
  assert payload["user_id"] == "alice"


def test_late_usage_and_summary_helpers_support_async_callbacks() -> None:
  event = _usage_event()
  summary = _session_summary()
  late_events: list[UsageEvent] = []
  summaries: list[SessionUsageSummary] = []

  async def _on_late_usage_event(usage_event: UsageEvent) -> None:
    late_events.append(usage_event)

  async def _on_session_summary(session_summary: SessionUsageSummary) -> None:
    summaries.append(session_summary)

  _run(
    call_late_usage_event_hook(
      _on_late_usage_event,
      event,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )
  _run(
    call_session_summary_hook(
      _on_session_summary,
      summary,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  assert late_events == [event]
  assert summaries == [summary]


class _TwoTurnTextProvider(_UsageProvider):
  def __init__(self) -> None:
    self.requests: list[list[dict[str, Any]]] = []

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, system_prompt, tools, max_tokens, kwargs
    self.requests.append([dict(message) for message in messages])
    return {"call_index": len(self.requests)}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client
    yield StreamEvent(type="message_start", input_tokens=10)
    text = "rough 25 bps" if params["call_index"] == 1 else "verified 26.1 bps"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
    yield StreamEvent(type="usage_update", output_tokens=5)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _FailingAfterUsageProvider(_UsageProvider):
  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(type="message_start", input_tokens=40, cache_read_tokens=4, cache_creation_tokens=3)
    yield StreamEvent(type="usage_update", output_tokens=7)
    raise RuntimeError("stream exploded")
    yield  # pragma: no cover


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-parent",
  )


def test_runner_tool_timing_forwards_tool_call_and_request_ids() -> None:
  timing_calls: list[dict[str, Any]] = []

  def on_tool_timing(
    session_id,
    tool_name,
    server,
    duration_ms,
    is_error,
    result_bytes,
    *,
    user_id=None,
    tool_call_id=None,
    request_id=None,
  ):
    timing_calls.append(
      {
        "session_id": session_id,
        "tool_name": tool_name,
        "server": server,
        "duration_ms": duration_ms,
        "is_error": is_error,
        "result_bytes": result_bytes,
        "user_id": user_id,
        "tool_call_id": tool_call_id,
        "request_id": request_id,
      }
    )

  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    on_tool_timing=on_tool_timing,
    user_id="alice",
    request_id="req-123",
    billing_mode="metered",
    rate_table_version="2026-04-08",
    channel="web",
  )

  runner._call_on_tool_timing(
    tool_name="documents_search",
    server="portfolio-reads-mcp",
    duration_ms=12,
    is_error=False,
    result_bytes=34,
    tool_call_id="toolu-123",
    request_id=runner._request_id,
  )

  assert timing_calls == [
    {
      "session_id": "sess-parent",
      "tool_name": "documents_search",
      "server": "portfolio-reads-mcp",
      "duration_ms": 12,
      "is_error": False,
      "result_bytes": 34,
      "user_id": "alice",
      "tool_call_id": "toolu-123",
      "request_id": "req-123",
    }
  ]


def test_runner_build_usage_event_preserves_timestamp_and_cost_delegates(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    user_id="alice",
    request_id="req-123",
    billing_mode="metered",
    rate_table_version="2026-04-08",
    channel="web",
  )
  monkeypatch.setattr(gateway_runner.time, "time", lambda: 456.75)
  runner._estimate_usage_cost = lambda _model, _usage_totals: CostEstimate(total=0.125)  # type: ignore[method-assign]

  event = runner._build_usage_event(
    model="claude-sonnet-4-6",
    usage_totals={
      "input_tokens": 100,
      "output_tokens": 50,
      "cache_read_input_tokens": 10,
      "cache_creation_input_tokens": 5,
    },
  )

  assert event.timestamp == 456.75
  assert event.cost_usd == 0.125
  assert event.provider == "stub"


def test_final_answer_guard_can_inject_follow_up_turn() -> None:
  event_log = EventLog()
  provider = _TwoTurnTextProvider()
  durable_events: list[dict[str, Any]] = []

  def guard(messages, answer_text, tools_used, tool_definitions, turn_count):
    _ = messages, answer_text, tools_used, tool_definitions
    return "verify with tools before final" if turn_count == 1 else None

  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    provider=provider,
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    final_answer_guard=guard,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  async def _append_durable_event(event: dict[str, Any]):
    durable_events.append(dict(event))
    return SimpleNamespace(seq=len(durable_events))

  runner._append_durable_event = _append_durable_event  # type: ignore[method-assign]

  _run(runner.run(messages=[{"role": "user", "content": "compare margin bps"}]))

  assert len(provider.requests) == 2
  assert provider.requests[1][-1] == {"role": "user", "content": "verify with tools before final"}
  assert any(entry.event.get("type") == "runtime_guard" for entry in event_log.entries)
  assistant_messages = [
    event
    for event in durable_events
    if event.get("type") == "assistant_message"
  ]
  runtime_guard_events = [
    event
    for event in durable_events
    if event.get("type") == "runtime_guard"
  ]
  assert len(assistant_messages) == 1
  assert assistant_messages[0]["content_blocks"] == [{"type": "text", "text": "verified 26.1 bps"}]
  assert len(runtime_guard_events) == 1
  assert runtime_guard_events[0]["draft_content_blocks"] == [{"type": "text", "text": "rough 25 bps"}]
  assert runtime_guard_events[0]["draft_usage"] == {
    "input_tokens": 10,
    "output_tokens": 5,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }


def test_final_answer_guard_records_draft_before_budget_stop() -> None:
  event_log = EventLog()
  provider = _TwoTurnTextProvider()
  durable_events: list[dict[str, Any]] = []

  def guard(messages, answer_text, tools_used, tool_definitions, turn_count):
    _ = messages, answer_text, tools_used, tool_definitions
    return "verify with tools before final" if turn_count == 1 else None

  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    provider=provider,
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    final_answer_guard=guard,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    max_budget_usd=0.000001,
  )

  async def _append_durable_event(event: dict[str, Any]):
    durable_events.append(dict(event))
    runner._last_durable_seq = len(durable_events)
    return SimpleNamespace(seq=len(durable_events))

  runner._append_durable_event = _append_durable_event  # type: ignore[method-assign]

  _run(runner.run(messages=[{"role": "user", "content": "compare margin bps"}]))

  assert len(provider.requests) == 1
  assert any(entry.event.get("type") == "budget_exceeded" for entry in event_log.entries)
  assert [
    event for event in durable_events if event.get("type") == "assistant_message"
  ] == []
  runtime_guard_events = [
    event
    for event in durable_events
    if event.get("type") == "runtime_guard"
  ]
  assert len(runtime_guard_events) == 1
  assert runtime_guard_events[0]["draft_content_blocks"] == [{"type": "text", "text": "rough 25 bps"}]


def test_final_answer_guard_records_draft_before_operator_pause() -> None:
  event_log = EventLog()
  provider = _TwoTurnTextProvider()
  durable_events: list[dict[str, Any]] = []
  pause_event = asyncio.Event()

  def guard(messages, answer_text, tools_used, tool_definitions, turn_count):
    _ = messages, answer_text, tools_used, tool_definitions
    if turn_count == 1:
      pause_event.set()
      return "verify with tools before final"
    return None

  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    provider=provider,
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    final_answer_guard=guard,
    operator_pause_event=pause_event,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  async def _append_durable_event(event: dict[str, Any]):
    durable_events.append(dict(event))
    runner._last_durable_seq = len(durable_events)
    return SimpleNamespace(seq=len(durable_events))

  runner._append_durable_event = _append_durable_event  # type: ignore[method-assign]

  _run(runner.run(messages=[{"role": "user", "content": "compare margin bps"}]))

  assert len(provider.requests) == 1
  assert any(entry.event.get("type") == "operator_pause" for entry in event_log.entries)
  assert [
    event for event in durable_events if event.get("type") == "assistant_message"
  ] == []
  runtime_guard_events = [
    event
    for event in durable_events
    if event.get("type") == "runtime_guard"
  ]
  assert len(runtime_guard_events) == 1
  assert runtime_guard_events[0]["draft_content_blocks"] == [{"type": "text", "text": "rough 25 bps"}]


def test_on_usage_fires_once_per_turn_with_usage_event_fields() -> None:
  events: list[UsageEvent] = []
  event_log = EventLog()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    on_usage=events.append,
    user_id="alice",
    request_id="req-123",
    billing_mode="metered",
    rate_table_version="2026-04-08",
    channel="web",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert len(events) == 1
  event = events[0]
  assert event.user_id == "alice"
  assert event.session_id == "sess-parent"
  assert event.request_id == "req-123"
  assert event.parent_turn_id is None
  assert event.model == "claude-sonnet-4-6"
  assert event.input_tokens == 100
  assert event.output_tokens == 50
  assert event.cache_read_tokens == 10
  assert event.cache_creation_tokens == 5
  assert event.cost_usd == pytest.approx(0.00019375)
  assert event.rate_table_version == "2026-04-08"
  assert event.billing_mode == "metered"
  assert event.channel == "web"
  assert event.provider == "stub"


def test_stream_turn_failure_emits_partial_usage_and_rolls_back_totals() -> None:
  events: list[UsageEvent] = []
  event_log = EventLog()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    provider=_FailingAfterUsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    on_usage=events.append,
    user_id="alice",
    request_id="req-123",
    billing_mode="metered",
    rate_table_version="2026-04-08",
    channel="web",
  )
  usage_totals = {
    "input_tokens": 5,
    "output_tokens": 2,
    "cache_read_input_tokens": 1,
    "cache_creation_input_tokens": 0,
  }
  initial_usage_totals = dict(usage_totals)

  result = _run(
    runner._stream_turn(
      client=object(),
      config={"model": "claude-sonnet-4-6", "thinking": False, "auth_mode": "api"},
      model_info=runner._provider.get_model_info("claude-sonnet-4-6"),
      system_prompt=None,
      current_messages=[{"role": "user", "content": "hello"}],
      base_kwargs={"tools": []},
      max_tokens=1024,
      turn_count=1,
      turn_t0=gateway_runner.time.time(),
      turn_t0_mono=gateway_runner.time.monotonic(),
      system_chars=0,
      tools_chars=0,
      usage_totals=usage_totals,
    )
  )

  assert result is None
  assert usage_totals == initial_usage_totals
  assert len(events) == 1
  event = events[0]
  assert event.input_tokens == 40
  assert event.output_tokens == 7
  assert event.cache_read_tokens == 4
  assert event.cache_creation_tokens == 3
  assert event.cost_usd == pytest.approx(0.00005125)
  error_events = [entry.event for entry in event_log.entries if entry.event["type"] == "error"]
  assert len(error_events) == 1
  assert "stream exploded" in error_events[0]["error"]


def test_on_usage_failure_does_not_block_chat_response(tmp_path: Path) -> None:
  event_log = EventLog()

  def _failing_on_usage(_event: UsageEvent) -> None:
    raise RuntimeError("ledger offline")

  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    on_usage=_failing_on_usage,
    user_id="alice",
    request_id="req-123",
    usage_ledger_dlq_path=tmp_path / "usage_dlq.jsonl",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  event_types = [entry.event["type"] for entry in event_log.entries]
  assert "stream_complete" in event_types


def test_on_usage_failure_writes_to_dlq_spool(tmp_path: Path) -> None:
  spool_path = tmp_path / "usage_dlq.jsonl"

  async def _failing_on_usage(_event: UsageEvent) -> None:
    raise RuntimeError("db unavailable")

  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    on_usage=_failing_on_usage,
    user_id="alice",
    request_id="req-123",
    usage_ledger_dlq_path=spool_path,
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  payload = json.loads(spool_path.read_text(encoding="utf-8").strip())
  assert payload["user_id"] == "alice"
  assert payload["request_id"] == "req-123"
  assert payload["session_id"] == "sess-parent"
  assert payload["input_tokens"] == 100
  assert payload["output_tokens"] == 50


def test_spawn_sub_agent_emits_usage_with_parent_turn_id() -> None:
  events: list[UsageEvent] = []
  parent_runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    on_usage=events.append,
    user_id="alice",
    request_id="req-123",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  sub_session = GatewaySession(
    session_id="sub0:sess-parent",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="alice",
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
  )

  result, error = _run(
    parent_runner.spawn_sub_agent(
      "Collect usage",
      dispatcher=_make_dispatcher(),
      sub_session=sub_session,
      max_turns=1,
      timeout=5.0,
      parent_turn_id="tool-run-agent-1",
    )
  )

  assert error is None
  assert result is not None
  assert len(events) == 1
  assert events[0].session_id == "sub0:sess-parent"
  assert events[0].request_id == "req-123"
  assert events[0].parent_turn_id == "tool-run-agent-1"
  assert events[0].provider == "stub"


def test_run_appends_turn_complete_event_to_event_log() -> None:
  event_log = EventLog()
  durable_events: list[dict[str, Any]] = []
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  async def _append_durable_event(event: dict[str, Any]):
    durable_events.append(dict(event))
    return SimpleNamespace(seq=len(durable_events))

  runner._append_durable_event = _append_durable_event  # type: ignore[method-assign]

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  turn_complete = [entry.event for entry in event_log.entries if entry.event.get("type") == "turn_complete"]
  assistant_messages = [event for event in durable_events if event.get("type") == "assistant_message"]
  assert len(turn_complete) == 1
  assert len(assistant_messages) == 1
  assert turn_complete[0]["turn"] == 1
  assert turn_complete[0]["usage"] == {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": 10,
    "cache_creation_input_tokens": 5,
    "estimated_cost": 0.0002,
  }
  assert assistant_messages[0]["model"] == "claude-sonnet-4-6"
  assert assistant_messages[0]["provider"] == "stub"


def test_runner_emits_session_summary_once_after_run() -> None:
  summaries: list[SessionUsageSummary] = []
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    user_id="alice",
    request_id="req-summary",
    channel="web",
    on_session_summary=summaries.append,
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert len(summaries) == 1
  summary = summaries[0]
  assert summary.user_id == "alice"
  assert summary.session_id == "sess-parent"
  assert summary.request_id == "req-summary"
  assert summary.input_tokens == 100
  assert summary.output_tokens == 50
  assert summary.cache_read_tokens == 10
  assert summary.cache_creation_tokens == 5
  assert summary.cost == pytest.approx(0.00019375)
  assert summary.turns == 1
  assert summary.channel == "web"
  assert summary.rate_table_version == "unknown"
  assert summary.billing_mode == "byok"
  assert summary.drain_complete is True
  assert summary.in_flight_task_count == 0
  assert summary.model == "claude-sonnet-4-6"
  assert summary.provider == "stub"


def test_runner_session_summary_reports_failed_drain_and_in_flight_tasks() -> None:
  summaries: list[SessionUsageSummary] = []
  pending_task = SimpleNamespace(done=lambda: False)
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    user_id="alice",
    request_id="req-summary-drain",
    on_session_summary=summaries.append,
    billing_mode="byok",
    rate_table_version="unknown",
  )

  async def _raise_shutdown(was_cancelled: bool) -> None:
    _ = was_cancelled
    raise RuntimeError("drain failed")

  def _list_running_tasks(*, state: Any = None) -> list[Any]:
    _ = state
    return [SimpleNamespace(asyncio_task=pending_task)]

  runner._shutdown_background_tasks = _raise_shutdown  # type: ignore[method-assign]
  runner._task_registry = SimpleNamespace(list_tasks=_list_running_tasks)

  with pytest.raises(RuntimeError, match="drain failed"):
    _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert len(summaries) == 1
  assert summaries[0].drain_complete is False
  assert summaries[0].in_flight_task_count == 1


def test_runner_is_single_use() -> None:
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  with pytest.raises(RuntimeError, match="single-use"):
    _run(runner.run(messages=[{"role": "user", "content": "hello again"}]))


def test_sub_runner_with_parent_aggregator_does_not_emit_own_summary() -> None:
  parent_summaries: list[SessionUsageSummary] = []
  child_summaries: list[SessionUsageSummary] = []
  parent_runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    user_id="alice",
    request_id="req-parent",
    on_session_summary=parent_summaries.append,
    billing_mode="byok",
    rate_table_version="unknown",
  )
  child_runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sub0:sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    user_id="alice",
    request_id="req-parent",
    on_session_summary=child_summaries.append,
    _parent_aggregator=parent_runner._aggregator,
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(child_runner.run(messages=[{"role": "user", "content": "child"}]))
  parent_summary = _run(parent_runner._aggregator.snapshot())

  assert child_summaries == []
  assert parent_summary.input_tokens == 100
  assert parent_summary.turns == 1
  assert parent_summary.model == "claude-sonnet-4-6"
  assert parent_summary.provider == "stub"
  assert parent_summary.rate_table_version == "unknown"
  assert parent_summary.billing_mode == "byok"


@pytest.mark.parametrize("timeout", [0, None, -1])
def test_spawn_sub_agent_no_wall_clock(
  monkeypatch: pytest.MonkeyPatch,
  timeout: float | None,
) -> None:
  async def _unexpected_wait_for(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("asyncio.wait_for should not wrap non-positive timeouts")

  monkeypatch.setattr(gateway_runner.asyncio, "wait_for", _unexpected_wait_for)
  parent_runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_UsageProvider(),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  result, error = _run(
    parent_runner.spawn_sub_agent(
      "Collect usage",
      dispatcher=_make_dispatcher(),
      max_turns=1,
      timeout=timeout,
    )
  )

  assert error is None
  assert result is not None
  assert result["response"] == "hello"
