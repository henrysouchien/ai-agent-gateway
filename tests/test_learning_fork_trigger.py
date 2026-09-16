from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway.approval_policy import RunContext
from agent_gateway.fork_ledger import ForkLedger
from agent_gateway.fork_task_registry import ForkTaskRegistry
from agent_gateway.learning_fork_trigger import (
  claim_learning_receipts,
  evaluate_learning_fork_trigger,
  owner_operated_interactive_analyst,
  settle_learning_receipts,
  submit_learning_fork_after_turn,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.runner_fork_agents import (
  LEARNING_FORK_ALLOWED_TOOLS,
  build_learning_fork_tool_decisions,
  cross_check_learning_memory_writes,
)
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


def test_learning_policy_is_closed_and_fail_closed() -> None:
  wire = [
    {"name": name}
    for name in sorted({
      *LEARNING_FORK_ALLOWED_TOOLS,
      "memory_store",
      "memory_delete",
      "memory_sync",
      "invoke_skill",
      "run_agent",
      "unclassified_future_tool",
    })
  ]
  decisions = {
    item.tool: item.decision
    for item in build_learning_fork_tool_decisions(wire)
  }
  assert {
    name for name, decision in decisions.items() if decision == "allow"
  } == LEARNING_FORK_ALLOWED_TOOLS
  assert all(
    decisions[name] == "deny"
    for name in {
      "memory_store",
      "memory_delete",
      "memory_sync",
      "invoke_skill",
      "run_agent",
      "unclassified_future_tool",
    }
  )










async def _unused_spawn(_fork_id, _handoff):
  raise AssertionError("no fork is launched here")


