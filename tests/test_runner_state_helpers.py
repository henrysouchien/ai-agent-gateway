import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import SubAgentConfig as PackageSubAgentConfig, ToolResultContext as PackageToolResultContext  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_state import (  # noqa: E402
  BackgroundTask,
  budget_cost_progress,
  budget_exceeded_state,
  ChildCostAccumulator,
  CostAccumulator,
  StreamTurnResult,
  SubAgentConfig,
  ToolResultContext,
  budget_reason_suffix,
)


def test_runner_preserves_state_helper_aliases() -> None:
  assert gateway_runner.ToolResultContext is ToolResultContext
  assert PackageToolResultContext is ToolResultContext
  assert gateway_runner.SubAgentConfig is SubAgentConfig
  assert PackageSubAgentConfig is SubAgentConfig
  assert gateway_runner.StreamTurnResult is StreamTurnResult
  assert gateway_runner.BackgroundTask is BackgroundTask
  assert gateway_runner.CostAccumulator is CostAccumulator
  assert gateway_runner.ChildCostAccumulator is ChildCostAccumulator
  assert gateway_runner._budget_cost_progress is budget_cost_progress
  assert gateway_runner._budget_exceeded_state is budget_exceeded_state
  assert gateway_runner._budget_reason_suffix is budget_reason_suffix


def test_tool_result_context_defaults_are_optional() -> None:
  ctx = ToolResultContext(
    tool_name="lookup",
    tool_input={"symbol": "MSFT"},
    result={"ok": True},
    error=None,
    duration_ms=12,
    tool_call_id="call-1",
    session_id="session-1",
    server=None,
    result_entry=None,
  )

  assert ctx.skill_run_id is None
  assert ctx.workspace_dir is None
  assert ctx.tool_input == {"symbol": "MSFT"}


def test_stream_turn_result_uses_independent_collections() -> None:
  first = StreamTurnResult()
  second = StreamTurnResult()

  first.tool_uses.append(("call-1", "lookup", {"symbol": "MSFT"}))
  first.content_blocks.append({"type": "text", "text": "done"})

  assert second.tool_uses == []
  assert second.content_blocks == []


def test_background_task_defaults_are_incomplete_without_payloads() -> None:
  task = BackgroundTask(task_id="task-1", agent_name=None, asyncio_task=None, started_at=12.5)

  assert task.result is None
  assert task.error is None
  assert task.completed is False
  assert task.completed_at is None


def test_child_cost_accumulator_forwards_parent_spend_and_reports_child_limit() -> None:
  parent = CostAccumulator(10.0)
  child = ChildCostAccumulator(parent, 3.0)

  child.add(3.5)

  assert child.total == 3.5
  assert parent.total == 3.5
  assert child.exceeded is True
  assert child.exceeded_reason == "child_budget"
  assert child.effective_total == 3.5
  assert child.effective_budget == 3.0
  assert budget_reason_suffix(child.exceeded_reason) == " (child budget)"
  state = budget_exceeded_state(child)
  assert state is not None
  assert state.total_cost == 3.5
  assert state.budget == 3.0
  assert state.reason == "child_budget"
  assert state.reason_suffix == " (child budget)"


def test_child_cost_accumulator_reports_parent_limit() -> None:
  parent = CostAccumulator(4.0)
  child = ChildCostAccumulator(parent, 10.0)

  child.add(4.5)

  assert parent.exceeded is True
  assert child.exceeded is True
  assert child.exceeded_reason == "parent_budget"
  assert child.effective_total == 4.5
  assert child.effective_budget == 4.0
  assert budget_reason_suffix(child.exceeded_reason) == " (parent budget)"
  assert budget_reason_suffix(None) == ""
  state = budget_exceeded_state(child)
  assert state is not None
  assert state.total_cost == 4.5
  assert state.budget == 4.0
  assert state.reason == "parent_budget"
  assert state.reason_suffix == " (parent budget)"


def test_budget_exceeded_state_handles_plain_accumulator_and_non_exceeded_state() -> None:
  accumulator = CostAccumulator(1.0)

  assert budget_exceeded_state(accumulator) is None

  accumulator.add(1.25)
  state = budget_exceeded_state(accumulator)

  assert state is not None
  assert state.total_cost == 1.25
  assert state.budget == 1.0
  assert state.reason is None
  assert state.reason_suffix == ""


def test_budget_cost_progress_adds_positive_delta_and_reports_exceeded_state() -> None:
  accumulator = CostAccumulator(1.0)

  progress = budget_cost_progress(
    accumulator,
    running_total=1.25,
    last_reported_cost=0.50,
  )

  assert progress.incremental_cost == 0.75
  assert progress.last_reported_cost == 1.25
  assert accumulator.total == 0.75
  assert progress.exceeded_state is None

  progress = budget_cost_progress(
    accumulator,
    running_total=1.50,
    last_reported_cost=1.25,
  )

  assert progress.incremental_cost == 0.25
  assert progress.last_reported_cost == 1.50
  assert accumulator.total == 1.0
  assert progress.exceeded_state is not None
  assert progress.exceeded_state.total_cost == 1.0
  assert progress.exceeded_state.budget == 1.0


def test_budget_cost_progress_ignores_non_positive_delta_but_updates_last_reported() -> None:
  accumulator = CostAccumulator(1.0)
  accumulator.add(0.80)

  progress = budget_cost_progress(
    accumulator,
    running_total=0.40,
    last_reported_cost=0.80,
  )

  assert progress.incremental_cost == 0.0
  assert progress.last_reported_cost == 0.40
  assert accumulator.total == 0.80
  assert progress.exceeded_state is None

  progress = budget_cost_progress(
    accumulator,
    running_total=0.40,
    last_reported_cost=0.40,
  )

  assert progress.incremental_cost == 0.0
  assert progress.last_reported_cost == 0.40
  assert accumulator.total == 0.80
  assert progress.exceeded_state is None
