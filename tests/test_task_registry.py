import asyncio
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.task_registry import TaskEntry, TaskRegistry, TaskState, _TERMINAL_STATES, make_progress_tracker


def _run(coro):
  return asyncio.run(coro)


class _RecordingListener:
  def __init__(self) -> None:
    self.transitions: list[tuple[str, TaskState, TaskState]] = []

  def on_transition(self, entry: TaskEntry, old_state: TaskState, new_state: TaskState) -> None:
    self.transitions.append((entry.task_id, old_state, new_state))


def test_task_registry_register_transition_get_and_list_tasks() -> None:
  registry = TaskRegistry()
  first = registry.register("background_agent", agent_name="writer", owner="alice")
  second = registry.register("background_agent", agent_name="reviewer")
  first.started_at = 20.0
  second.started_at = 10.0

  registry.transition(first.task_id, TaskState.RUNNING)
  registry.transition(second.task_id, TaskState.COMPLETED, result={"summary": "done"})

  assert first.metadata == {"owner": "alice"}
  assert registry.get(first.task_id) is first
  assert registry.list_tasks() == [second, first]
  assert registry.list_tasks(state=TaskState.COMPLETED) == [second]
  assert second.result == {"summary": "done"}
  assert second.completed_at is not None


def test_transition_terminal_guard_is_no_op() -> None:
  registry = TaskRegistry()
  listener = _RecordingListener()
  registry.add_listener(listener)
  entry = registry.register("background_agent")

  registry.transition(entry.task_id, TaskState.COMPLETED, result={"ok": True})
  completed_at = entry.completed_at
  result = registry.transition(entry.task_id, TaskState.FAILED, error={"code": "late"})

  assert result is entry
  assert entry.state == TaskState.COMPLETED
  assert entry.result == {"ok": True}
  assert entry.error is None
  assert entry.completed_at == completed_at
  assert listener.transitions == [(entry.task_id, TaskState.PENDING, TaskState.COMPLETED)]


def test_kill_cancels_asyncio_task_and_transitions_to_killed() -> None:
  async def _case() -> None:
    registry = TaskRegistry()
    entry = registry.register("background_agent")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _worker() -> None:
      started.set()
      try:
        await asyncio.sleep(60)
      except asyncio.CancelledError:
        cancelled.set()
        raise

    task = asyncio.create_task(_worker())
    await started.wait()
    entry.asyncio_task = task
    registry.transition(entry.task_id, TaskState.RUNNING)

    assert registry.kill(entry.task_id) is True
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    await asyncio.gather(task, return_exceptions=True)
    assert entry.state == TaskState.KILLED
    assert entry.completed is True
    assert entry.completed_at is not None

  _run(_case())


def test_kill_returns_false_for_terminal_task() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent")
  registry.transition(entry.task_id, TaskState.COMPLETED)

  assert registry.kill(entry.task_id) is False


def test_inflight_count_only_counts_running_tasks() -> None:
  registry = TaskRegistry()
  pending = registry.register("background_agent")
  running = registry.register("background_agent")
  completed = registry.register("background_agent")
  failed = registry.register("background_agent")

  registry.transition(running.task_id, TaskState.RUNNING)
  registry.transition(completed.task_id, TaskState.COMPLETED)
  registry.transition(failed.task_id, TaskState.FAILED, error={"code": "boom"})

  assert pending.state == TaskState.PENDING
  assert registry.inflight_count == 1


def test_max_inflight_enforced_only_for_running_tasks() -> None:
  registry = TaskRegistry(max_inflight=1)
  first = registry.register("background_agent")
  second = registry.register("background_agent")

  registry.transition(first.task_id, TaskState.RUNNING)
  with pytest.raises(RuntimeError, match="Task inflight limit reached"):
    registry.transition(second.task_id, TaskState.RUNNING)

  registry.transition(first.task_id, TaskState.COMPLETED)
  registry.transition(second.task_id, TaskState.RUNNING)
  assert registry.inflight_count == 1


def test_max_retained_auto_evicts_oldest_completed_entries() -> None:
  registry = TaskRegistry(max_retained=2)
  first = registry.register("background_agent")
  first.started_at = 1.0
  registry.transition(first.task_id, TaskState.COMPLETED)

  second = registry.register("background_agent")
  second.started_at = 2.0
  registry.transition(second.task_id, TaskState.COMPLETED)

  third = registry.register("background_agent")
  third.started_at = 3.0

  assert registry.get(first.task_id) is None
  assert registry.list_tasks() == [second, third]


def test_evict_completed_removes_old_terminal_entries() -> None:
  registry = TaskRegistry()
  old_entry = registry.register("background_agent")
  recent_entry = registry.register("background_agent")
  running_entry = registry.register("background_agent")

  registry.transition(old_entry.task_id, TaskState.COMPLETED)
  registry.transition(recent_entry.task_id, TaskState.FAILED, error={"code": "failed"})
  registry.transition(running_entry.task_id, TaskState.RUNNING)

  old_entry.completed_at = time.time() - 600
  recent_entry.completed_at = time.time() - 60

  assert registry.evict_completed(max_age_seconds=300) == 1
  assert registry.get(old_entry.task_id) is None
  assert registry.get(recent_entry.task_id) is recent_entry
  assert registry.get(running_entry.task_id) is running_entry


def test_listener_fires_synchronously_on_transition() -> None:
  registry = TaskRegistry()
  listener = _RecordingListener()
  registry.add_listener(listener)
  entry = registry.register("background_agent")

  registry.transition(entry.task_id, TaskState.RUNNING)

  assert listener.transitions == [(entry.task_id, TaskState.PENDING, TaskState.RUNNING)]


def test_task_entry_completed_property() -> None:
  assert TaskEntry(task_id="t0", task_type="background_agent", state=TaskState.PENDING).completed is False
  assert TaskEntry(task_id="t1", task_type="background_agent", state=TaskState.RUNNING).completed is False
  assert TaskEntry(task_id="t2", task_type="background_agent", state=TaskState.COMPLETED).completed is True
  assert TaskEntry(task_id="t3", task_type="background_agent", state=TaskState.FAILED).completed is True
  assert TaskEntry(task_id="t4", task_type="background_agent", state=TaskState.KILLED).completed is True
  assert TaskEntry(task_id="t5", task_type="background_agent", state=TaskState.INTERRUPTED).completed is True


def test_interrupted_is_terminal_for_send_kill_and_completion() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent")
  entry.state = TaskState.INTERRUPTED
  entry.completed_at = time.time()

  assert TaskState.INTERRUPTED in _TERMINAL_STATES
  assert entry.completed is True
  assert registry.kill(entry.task_id) is False


def test_task_entry_message_inbox_exists_at_creation() -> None:
  entry = TaskRegistry().register("background_agent")

  assert isinstance(entry.message_inbox, asyncio.Queue)
  assert entry.message_inbox.empty()


def test_load_from_events_reconstructs_tasks_restores_seq_and_bypasses_listeners() -> None:
  registry = TaskRegistry()
  listener = _RecordingListener()
  registry.add_listener(listener)

  registry.load_from_events(
    [
      {
        "type": "task_completed",
        "task_id": "bg_7",
        "final_state": "completed",
        "completed_at": 125.0,
        "result": {"response": "done"},
        "owner_runner_id": "runner_old",
        "owner_role": "writer",
        "sub_agent_id": "sub7:sess-parent",
        "parent_turn_id": "turn-1",
        "call_index": 7,
        "task_type": "background",
        "provider_name": "anthropic",
        "model": "claude-sonnet-4-6",
      },
      {
        "type": "task_registered",
        "task_id": "bg_7",
        "agent_name": "writer",
        "started_at": 100.0,
        "metadata": {"custom": "value"},
        "owner_runner_id": "runner_old",
        "owner_role": "writer",
        "sub_agent_id": "sub7:sess-parent",
        "parent_turn_id": "turn-1",
        "call_index": 7,
        "task_type": "background",
        "provider_name": "anthropic",
        "model": "claude-sonnet-4-6",
      },
      {
        "type": "task_registered",
        "task_id": "bg_9",
        "agent_name": "reviewer",
        "started_at": 200.0,
        "owner_runner_id": "runner_old",
        "owner_role": "writer",
        "sub_agent_id": "sub9:sess-parent",
        "parent_turn_id": "turn-2",
        "call_index": 9,
        "task_type": "background",
        "provider_name": "openai",
        "model": "gpt-5.2",
      },
      {
        "type": "parent_message_sent",
        "task_id": "bg_9",
        "message_id": "msg-1",
        "message": "status",
      },
    ]
  )

  completed = registry.get("bg_7")
  interrupted = registry.get("bg_9")
  assert completed is not None
  assert interrupted is not None
  assert completed.state == TaskState.COMPLETED
  assert completed.result == {"response": "done"}
  assert completed.completed_at == 125.0
  assert completed.metadata["custom"] == "value"
  assert interrupted.state == TaskState.INTERRUPTED
  assert interrupted.completed_at == 200.0
  assert interrupted.reconstructed_from_log is True
  assert interrupted.metadata["parent_messages"][0]["message_id"] == "msg-1"
  assert registry.register("background_agent").task_id == "bg_10"
  assert listener.transitions == []


def test_make_progress_tracker_updates_tool_usage_fields() -> None:
  entry = TaskEntry(task_id="bg_0", task_type="background_agent")
  track = make_progress_tracker(entry)

  before = time.time()
  track({"type": "tool_call_start", "tool_name": "search_query"}, "sub0:sess")

  assert entry.progress.tool_use_count == 1
  assert entry.progress.last_tool_name == "search_query"
  assert entry.progress.recent_tools == ["search_query"]
  assert entry.progress.last_activity_at >= before


def test_make_progress_tracker_updates_turn_counts_and_tokens() -> None:
  entry = TaskEntry(task_id="bg_0", task_type="background_agent")
  track = make_progress_tracker(entry)

  track({"type": "turn_complete", "usage": {"input_tokens": 11, "output_tokens": 7}}, "sub0:sess")
  track({"type": "turn_complete", "usage": {"input_tokens": 3}}, "sub0:sess")

  assert entry.progress.turn_count == 2
  assert entry.progress.input_tokens == 14
  assert entry.progress.output_tokens == 7
  assert entry.progress.last_activity_at > 0


def test_make_progress_tracker_caps_recent_tools_at_ten() -> None:
  entry = TaskEntry(task_id="bg_0", task_type="background_agent")
  track = make_progress_tracker(entry)

  for index in range(12):
    track({"type": "tool_call_start", "tool_name": f"tool_{index}"}, "sub0:sess")

  assert entry.progress.tool_use_count == 12
  assert entry.progress.last_tool_name == "tool_11"
  assert len(entry.progress.recent_tools) == 10
  assert entry.progress.recent_tools == [f"tool_{index}" for index in range(2, 12)]
