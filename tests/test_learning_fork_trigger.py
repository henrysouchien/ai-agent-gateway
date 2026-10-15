from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway.approval_policy import RunContext
from agent_gateway.fork_ledger import ForkLedger
from agent_gateway.event_log import EventLog
from agent_gateway.fork_task_registry import ForkTaskRegistry
from agent_gateway.learning_fork_trigger import (
  claim_learning_receipts,
  evaluate_learning_fork_trigger,
  owner_operated_interactive_analyst,
  flush_learning_fork_events,
  settle_learning_receipts,
  submit_learning_fork_after_turn,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.runner_fork_agents import cross_check_learning_memory_writes
from agent_gateway.runner_fork_agents import _publish_fork_measurements
from agent_gateway.session_recap import emit_recap_then_terminal
from agent_gateway.runner_notifications import build_notification_reminder
from agent_gateway.task_registry import NotificationQueue
from agent_gateway.tool_dispatcher import ToolDispatcher

ROOT = Path(__file__).resolve().parents[3]





def _clock() -> int:
  return int(
    datetime(2026, 7, 27, 12, tzinfo=timezone.utc).timestamp()
    * 1_000_000_000
  )


def _ledger(tmp_path: Path) -> ForkLedger:
  return ForkLedger(
    tmp_path / "fork-ledger.sqlite3",
    process_instance_id="test-process",
    clock_ns=_clock,
  )




def test_hermes_counters_increment_reset_disable_and_trip_combined() -> None:
  first = evaluate_learning_fork_trigger(
    memory_turns=0,
    skill_iters=0,
    tool_calling_iters=1,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
    enabled=True,
    memory_threshold=2,
    skill_threshold=2,
  )
  assert (first.memory_turns, first.skill_iters) == (1, 1)
  assert not first.should_submit

  second = evaluate_learning_fork_trigger(
    memory_turns=first.memory_turns,
    skill_iters=first.skill_iters,
    tool_calling_iters=1,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
    enabled=True,
    memory_threshold=2,
    skill_threshold=2,
  )
  assert second.should_submit
  assert second.reason == "tripped"

  memory_reset = evaluate_learning_fork_trigger(
    memory_turns=9,
    skill_iters=7,
    tool_calling_iters=2,
    foreground_memory_write=True,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
    enabled=True,
    memory_threshold=10,
    skill_threshold=10,
  )
  assert memory_reset.memory_turns == 0
  assert memory_reset.skill_iters == 9

  disabled = evaluate_learning_fork_trigger(
    memory_turns=20,
    skill_iters=20,
    tool_calling_iters=1,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
    enabled=True,
    memory_threshold=0,
    skill_threshold=10,
  )
  assert not disabled.should_submit
  assert disabled.reason == "counter_disabled"


@pytest.mark.parametrize("terminal_flag", ("errored", "aborted", "cancelled"))
def test_no_fire_or_counter_advance_for_unsuccessful_turns(
  terminal_flag: str,
) -> None:
  flags = {"errored": False, "aborted": False, "cancelled": False}
  flags[terminal_flag] = True
  decision = evaluate_learning_fork_trigger(
    memory_turns=10,
    skill_iters=10,
    tool_calling_iters=3,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    enabled=True,
    memory_threshold=10,
    skill_threshold=10,
    **flags,
  )
  assert not decision.should_submit
  assert decision.reason == terminal_flag
  assert (decision.memory_turns, decision.skill_iters) == (10, 10)








async def _unused_spawn(_fork_id, _handoff):
  raise AssertionError("no fork is launched here")




def test_post_turn_launch_and_fork_counts_land_in_second_turn_recap(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def case() -> None:
    ledger = _ledger(tmp_path)
    config = SimpleNamespace(_pending_events={})
    session = SimpleNamespace(
      session_id="session-1", role="owner", owner_user_id="owner-1",
      user_id="owner-1", learning_fork_config=config,
      learning_fork_ledger=ledger,
      learn_memory_nudge_turns=0, learn_skill_nudge_iters=0,
    )
    first_log = EventLog(session_id=session.session_id)
    emit_recap_then_terminal(
      first_log, {"type": "stream_complete"},
      session_id=session.session_id, started_at=1.0,
    )
    runner = SimpleNamespace(
      _gateway_session=session, _log=first_log, _billing_mode="platform",
      _fork_mode=False,
      _capability_execution=SimpleNamespace(bind=SimpleNamespace(
        credential_principal="service", run_mode="interactive",
      )),
      _dispatcher=SimpleNamespace(run_context=SimpleNamespace(profile="analyst")),
    )
    finished = asyncio.Event()

    async def spawn(fork_id, work_item):
      child_log = EventLog(session_id=fork_id)
      child_log.append({"type": "turn_complete"})
      child_log.append({"type": "turn_complete"})
      child_log.append({"type": "stream_complete"})
      _publish_fork_measurements(
        work_item.parent, child_log, SimpleNamespace(_fork_suffix_tokens=321),
        sub_session_id=fork_id, independent_accounting=True,
      )
      finished.set()
      # This boundary test observes the measurements independently of the
      # registry's TaskResult settlement, then shuts down its worker.
      await asyncio.Event().wait()
      raise AssertionError("worker must be cancelled during shutdown")

    registry = ForkTaskRegistry(
      ledger, spawn_fork=spawn, shutdown_timeout_seconds=0.01,
    )
    session.learning_fork_registry = registry
    monkeypatch.setenv("HANK_LEARN_MEMORY_NUDGE_TURNS", "1")
    monkeypatch.setenv("HANK_LEARN_SKILL_NUDGE_ITERS", "1")
    launch = submit_learning_fork_after_turn(
      runner, handoff=SimpleNamespace(_fork_retained_payload={"message": "ready"}),
      tool_calling_iters=1, foreground_memory_write=False,
      completed=True, real_final_response=True,
      errored=False, aborted=False, cancelled=False,
    )
    assert launch is not None and launch.launched
    await finished.wait()
    assert first_log.has_terminal
    assert not any(
      entry.event["type"] in {"learning_fork_launched", "fork_measurements"}
      for entry in first_log.entries
    )

    second_log = EventLog(session_id=session.session_id)
    second_runner = SimpleNamespace(_gateway_session=session, _log=second_log)
    flush_learning_fork_events(second_runner)
    emit_recap_then_terminal(
      second_log, {"type": "stream_complete"},
      session_id=session.session_id, started_at=2.0,
    )
    recap = next(
      entry.event for entry in second_log.entries
      if entry.event["type"] == "session_recap"
    )
    assert recap["tool_calls_summary"]["measurements"] == {
      "in_flight": launch.in_flight,
      "day_invocations": launch.day_invocations,
      "day_charged_microusd": launch.day_charged_microusd,
      "retained_bytes": launch.retained_bytes,
      "turns": 2,
      "suffix_tokens": 321,
    }
    assert launch.in_flight == 1
    assert launch.day_invocations == 1
    assert launch.day_charged_microusd == 2_000_000
    assert launch.retained_bytes is not None and launch.retained_bytes > 0
    assert config._pending_events == {}
    await registry.shutdown()

  asyncio.run(case())


def test_measurement_delivery_keeps_closed_logs_and_other_sessions_pending() -> None:
  config = SimpleNamespace(_pending_events={
    "session-1": [{"type": "fork_measurements", "turns": 3, "suffix_tokens": 22}],
    "session-2": [{"type": "fork_measurements", "turns": 5, "suffix_tokens": 44}],
  })
  session = SimpleNamespace(session_id="session-1", learning_fork_config=config)
  closed_log = EventLog(session_id="session-1")
  closed_log.append({"type": "stream_complete"})
  flush_learning_fork_events(SimpleNamespace(_gateway_session=session, _log=closed_log))
  assert set(config._pending_events) == {"session-1", "session-2"}
  open_log = EventLog(session_id="session-1")
  flush_learning_fork_events(SimpleNamespace(_gateway_session=session, _log=open_log))
  assert [entry.event for entry in open_log.entries] == [
    {"type": "fork_measurements", "turns": 3, "suffix_tokens": 22},
  ]
  assert set(config._pending_events) == {"session-2"}
