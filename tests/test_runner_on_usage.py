import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog, GatewaySession, ModelInfo, ModelProvider, ToolDispatcher
import agent_gateway.runner as gateway_runner
from agent_gateway.multi_user.billing import SessionUsageSummary, UsageEvent
from agent_gateway.providers import StreamEvent


def _run(coro):
  return asyncio.run(coro)


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


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-parent",
  )


def test_final_answer_guard_can_inject_follow_up_turn() -> None:
  event_log = EventLog()
  provider = _TwoTurnTextProvider()

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

  _run(runner.run(messages=[{"role": "user", "content": "compare margin bps"}]))

  assert len(provider.requests) == 2
  assert provider.requests[1][-1] == {"role": "user", "content": "verify with tools before final"}
  assert any(entry.event.get("type") == "runtime_guard" for entry in event_log.entries)


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


def test_run_appends_turn_complete_event_to_event_log() -> None:
  event_log = EventLog()
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

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  turn_complete = [entry.event for entry in event_log.entries if entry.event.get("type") == "turn_complete"]
  assert len(turn_complete) == 1
  assert turn_complete[0]["turn"] == 1
  assert turn_complete[0]["usage"] == {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": 10,
    "cache_creation_input_tokens": 5,
  }


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
  assert summary.drain_complete is True
  assert summary.in_flight_task_count == 0


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
