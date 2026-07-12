# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentSDKConfig, AgentSDKRunner, EventLog
from agent_gateway.multi_user.billing import SessionUsageSummary, UsageEvent


def _run(coro):
  return asyncio.run(coro)


def test_sdk_runner_usage_event_threads_identity() -> None:
  events: list[UsageEvent] = []
  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk",
    sdk_config=AgentSDKConfig(
      api_key="k",
      model="claude-sonnet-4-6",
      user_id="alice",
      channel="web",
      rate_table_version="v1",
      billing_mode="metered",
      request_id="req-sdk",
    ),
    system_prompt="test",
    on_usage=events.append,
  )
  runner._update_usage(
    {
      "input_tokens": 10,
      "output_tokens": 5,
      "cache_read_input_tokens": 1,
      "cache_creation_input_tokens": 2,
      "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1},
    },
    total_cost_usd=0.03,
    num_turns=2,
  )

  _run(runner._emit_usage_hook())

  assert len(events) == 1
  event = events[0]
  assert event.user_id == "alice"
  assert event.channel == "web"
  assert event.rate_table_version == "v1"
  assert event.billing_mode == "metered"
  assert event.request_id == "req-sdk"
  assert event.model == "claude-sonnet-4-6"
  assert event.provider == "agent-sdk"
  assert event.input_tokens == 10
  assert event.provider_unit_deltas == {"web_fetch": 1, "web_search": 2}
  assert event.cost_usd == pytest.approx(0.03)


def test_sdk_runner_requires_explicit_usage_identity() -> None:
  with pytest.raises(ValueError, match="user_id"):
    AgentSDKRunner(
      event_log=EventLog(),
      session_id="sess-sdk",
      sdk_config=AgentSDKConfig(api_key="k", model="claude-sonnet-4-6"),
      system_prompt="test",
    )


def test_sdk_stream_emits_one_commercial_delta_per_provider_call() -> None:
  emitted = []

  class Producer:
    async def emit(self, event, *, usage_state="succeeded"):
      emitted.append((event, usage_state))

  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk",
    sdk_config=AgentSDKConfig(
      api_key="k", model="claude-sonnet-4-6", user_id="alice",
      channel="web", rate_table_version="v1", billing_mode="metered",
      request_id="req-sdk",
    ),
    system_prompt="test",
    commercial_usage_producer=Producer(),
  )
  for model, input_tokens, cached, output_tokens in (
    ("claude-sonnet-4-6", 10, 2, 4),
    ("claude-sonnet-4-6", 20, 5, 8),
  ):
    runner._handle_stream_event({
      "type": "message_start",
      "message": {"model": model, "usage": {
        "input_tokens": input_tokens, "cache_read_input_tokens": cached,
        "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 1},
      }},
    })
    runner._handle_stream_event({
      "type": "message_delta", "usage": {"output_tokens": output_tokens},
    })
  while runner._pending_sdk_usage_deltas:
    _run(runner._emit_sdk_provider_call_usage(runner._pending_sdk_usage_deltas.pop(0)))

  assert [(event.input_tokens, event.cache_read_tokens, event.output_tokens, state)
          for event, state in emitted] == [
    (10, 2, 4, "succeeded"),
    (20, 5, 8, "succeeded"),
  ]
  assert {event.provider for event, _ in emitted} == {"anthropic"}
  assert [event.provider_unit_deltas for event, _ in emitted] == [
    {"web_fetch": 1, "web_search": 1},
    {"web_fetch": 1, "web_search": 1},
  ]


def test_sdk_commercial_default_off_preserves_cumulative_legacy_grain() -> None:
  events = []
  runner = AgentSDKRunner(
    event_log=EventLog(), session_id="sess-sdk",
    sdk_config=AgentSDKConfig(
      api_key="k", model="claude-sonnet-4-6", user_id="alice",
      request_id="req-sdk", rate_table_version="v1", billing_mode="byok",
    ),
    system_prompt="test", on_usage=events.append,
  )
  runner._handle_stream_event({
    "type": "message_start",
    "message": {"usage": {"input_tokens": 10}},
  })
  runner._handle_stream_event({
    "type": "message_delta", "usage": {"output_tokens": 4},
  })
  assert runner._pending_sdk_usage_deltas == []

  runner._update_usage({"input_tokens": 30, "output_tokens": 12}, total_cost_usd=0.02)
  _run(runner._emit_usage_hook())
  assert len(events) == 1
  assert (events[0].input_tokens, events[0].output_tokens) == (30, 12)


def test_sdk_reconciliation_failure_does_not_block_summary_callback() -> None:
  summaries = []

  class Producer:
    async def reconcile(self, summary):
      raise RuntimeError("comparison failed")

  runner = AgentSDKRunner(
    event_log=EventLog(), session_id="sess-sdk",
    sdk_config=AgentSDKConfig(
      api_key="k", model="claude-sonnet-4-6", user_id="alice",
      request_id="req-sdk", rate_table_version="v1", billing_mode="byok",
    ),
    system_prompt="test", commercial_usage_producer=Producer(),
    on_session_summary=summaries.append,
  )
  summary = SessionUsageSummary(
    user_id="alice", session_id="sess-sdk", request_id="req-sdk",
    input_tokens=0, output_tokens=0, cache_read_tokens=0,
    cache_creation_tokens=0, cost=0, turns=0, channel=None,
    started_at=1, ended_at=2,
  )

  _run(runner._call_on_session_summary(summary))
  assert summaries == [summary]


def test_sdk_runner_explicit_identity_summary() -> None:
  summaries: list[SessionUsageSummary] = []
  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk",
    sdk_config=AgentSDKConfig(
      api_key="k",
      model="claude-sonnet-4-6",
      user_id="alice",
      rate_table_version="v1",
      billing_mode="byok",
    ),
    system_prompt="test",
    on_session_summary=summaries.append,
  )
  runner._update_usage({"input_tokens": 3, "output_tokens": 2}, total_cost_usd=0.01, num_turns=1)
  _run(runner._emit_usage_hook())

  async def close_and_emit() -> None:
    await runner._aggregator.close()
    summary = await runner._aggregator.snapshot(ended_at=2.0)
    await runner._call_on_session_summary(summary)

  _run(close_and_emit())

  assert summaries
  assert summaries[0].user_id == "alice"
  assert summaries[0].request_id
  assert summaries[0].input_tokens == 3
  assert summaries[0].turns == 1
  assert summaries[0].model == "claude-sonnet-4-6"
  assert summaries[0].provider == "agent-sdk"
  assert summaries[0].rate_table_version == "v1"
  assert summaries[0].billing_mode == "byok"
