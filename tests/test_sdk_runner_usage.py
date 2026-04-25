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
  assert event.input_tokens == 10
  assert event.cost_usd == pytest.approx(0.03)


def test_sdk_runner_identity_defaults_and_summary() -> None:
  summaries: list[SessionUsageSummary] = []
  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk",
    sdk_config=AgentSDKConfig(api_key="k", model="claude-sonnet-4-6"),
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
  assert summaries[0].user_id == "_default"
  assert summaries[0].request_id
  assert summaries[0].input_tokens == 3
  assert summaries[0].turns == 1

