import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (
  AgentRunner,
  EventLog,
  NotificationQueue,
  TaskNotification,
  TaskRegistry,
  TaskState,
  ToolDispatcher,
)
import agent_gateway.runner as gateway_runner


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _StubProvider:
  name = "stub"


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-notifications",
  )


def _make_runner(task_registry: TaskRegistry | None = None) -> AgentRunner:
  event_log = EventLog()
  return AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-notifications",
    provider=_StubProvider(),
    auth_config={"api_key": "k", "model": "stub-model"},
    task_registry=task_registry,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def _notification(
  task_id: str,
  *,
  agent_name: str | None = "writer",
  event: str = "completed",
  summary: str = "done",
  payload: dict[str, Any] | None = None,
) -> TaskNotification:
  return TaskNotification(
    task_id=task_id,
    agent_name=agent_name,
    event=event,
    summary=summary,
    timestamp=time.time(),
    payload=payload or {},
  )


def test_notification_queue_push_peek_and_pending_count() -> None:
  queue = NotificationQueue()
  queue.push(_notification("bg_0"))
  queue.push(_notification("bg_1"))

  assert queue.pending_count == 2
  assert [item.task_id for item in queue.peek()] == ["bg_0", "bg_1"]


def test_notification_queue_peek_is_non_destructive() -> None:
  queue = NotificationQueue()
  queue.push(_notification("bg_0"))
  queue.push(_notification("bg_1"))

  peeked = queue.peek(max_count=1)

  assert [item.task_id for item in peeked] == ["bg_0"]
  assert queue.pending_count == 2
  assert [item.task_id for item in queue.peek()] == ["bg_0", "bg_1"]


def test_notification_queue_drain_respects_max_count_and_order() -> None:
  queue = NotificationQueue()
  queue.push(_notification("bg_0"))
  queue.push(_notification("bg_1"))
  queue.push(_notification("bg_2"))

  drained = queue.drain(max_count=2)

  assert [item.task_id for item in drained] == ["bg_0", "bg_1"]
  assert queue.pending_count == 1
  assert [item.task_id for item in queue.peek()] == ["bg_2"]


def test_notification_queue_honors_max_pending_cap() -> None:
  queue = NotificationQueue(max_pending=2)

  queue.push(_notification("bg_0"))
  queue.push(_notification("bg_1"))
  queue.push(_notification("bg_2"))

  assert queue.pending_count == 2
  assert [item.task_id for item in queue.peek()] == ["bg_0", "bg_1"]


def test_task_notification_format_xml_sanitizes_and_truncates() -> None:
  xml = TaskNotification(
    task_id="bg_1",
    agent_name='writer & <reviewer>',
    event="completed",
    summary="<done & ready>",
    timestamp=1.0,
    payload={},
  ).format_xml()
  long_xml = TaskNotification(
    task_id="bg_2",
    agent_name=None,
    event="failed",
    summary="x" * 2100,
    timestamp=1.0,
    payload={},
  ).format_xml()

  assert xml == "\n".join([
    '<task-notification task_id="bg_1">',
    "  <status>completed</status>",
    "  <agent>writer &amp; &lt;reviewer&gt;</agent>",
    "  <summary>&lt;done &amp; ready&gt;</summary>",
    "</task-notification>",
  ])
  assert f"  <summary>{'x' * 2000}</summary>" in long_xml
  assert "x" * 2001 not in long_xml
  assert "<agent>" not in long_xml


@pytest.mark.parametrize(
  ("new_state", "transition_kwargs", "expected_event", "expected_summary", "expected_payload"),
  [
    (
      TaskState.COMPLETED,
      {"result": {"response": "done", "tool": "search"}},
      "completed",
      "done",
      {"response": "done", "tool": "search"},
    ),
    (
      TaskState.FAILED,
      {"error": {"code": "boom", "message": "failed hard"}},
      "failed",
      "failed hard",
      {"code": "boom", "message": "failed hard"},
    ),
  ],
)
def test_task_registry_terminal_transitions_push_notifications(
  new_state: TaskState,
  transition_kwargs: dict[str, Any],
  expected_event: str,
  expected_summary: str,
  expected_payload: dict[str, Any],
) -> None:
  runner = _make_runner()
  entry = runner._task_registry.register("background_agent", agent_name="writer")

  runner._task_registry.transition(entry.task_id, TaskState.RUNNING)

  assert runner._notification_queue.pending_count == 0

  runner._task_registry.transition(entry.task_id, new_state, **transition_kwargs)

  notifications = runner._notification_queue.peek()
  assert len(notifications) == 1
  assert notifications[0].task_id == entry.task_id
  assert notifications[0].agent_name == "writer"
  assert notifications[0].event == expected_event
  assert notifications[0].summary == expected_summary
  assert notifications[0].payload == expected_payload


def test_task_registry_kill_pushes_notification() -> None:
  runner = _make_runner()
  entry = runner._task_registry.register("background_agent", agent_name="writer")
  runner._task_registry.transition(entry.task_id, TaskState.RUNNING)

  assert runner._task_registry.kill(entry.task_id) is True

  notifications = runner._notification_queue.peek()
  assert len(notifications) == 1
  assert notifications[0].task_id == entry.task_id
  assert notifications[0].event == "killed"
  assert notifications[0].summary == ""
  assert notifications[0].payload == {}


def test_build_notification_reminder_returns_empty_string_without_notifications() -> None:
  runner = _make_runner()

  assert runner._build_notification_reminder() == ""


def test_background_task_reminder_only_includes_running_tasks() -> None:
  runner = _make_runner()
  running = runner._task_registry.register("background_agent", agent_name="writer")
  completed = runner._task_registry.register("background_agent", agent_name="reviewer")

  runner._task_registry.transition(running.task_id, TaskState.RUNNING)
  runner._task_registry.transition(completed.task_id, TaskState.COMPLETED, result={"response": "done"})

  reminder = runner._background_task_reminder_text()

  assert running.task_id in reminder
  assert completed.task_id not in reminder


def test_background_task_payload_includes_progress_for_running_tasks_with_tool_activity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runner = _make_runner()
  running = runner._task_registry.register("background_agent", agent_name="writer")
  runner._task_registry.transition(running.task_id, TaskState.RUNNING)
  running.progress.tool_use_count = 3
  running.progress.turn_count = 2
  running.progress.last_tool_name = "search_query"
  running.progress.last_activity_at = 470.4
  running.progress.output_tokens = 55

  monkeypatch.setattr(gateway_runner.time, "time", lambda: 500.0)

  payload = runner._background_task_payload(running)

  assert payload["progress"] == {
    "tools_used": 3,
    "turns": 2,
    "last_tool": "search_query",
    "idle_seconds": 29,
    "output_tokens": 55,
  }


def test_background_task_reminder_includes_progress_info() -> None:
  runner = _make_runner()
  running = runner._task_registry.register("background_agent", agent_name="writer")
  runner._task_registry.transition(running.task_id, TaskState.RUNNING)
  running.progress.tool_use_count = 3
  running.progress.last_tool_name = "search_query"

  reminder = runner._background_task_reminder_text()

  assert running.task_id in reminder
  assert "3 tools" in reminder
  assert "last: search_query" in reminder


def test_register_background_task_wraps_handler_with_task_entry() -> None:
  async def _case() -> None:
    runner = _make_runner()
    seen: dict[str, Any] = {}

    async def _handler(tool_input: dict[str, Any], **kwargs: Any):
      _ = tool_input
      seen["task_entry"] = kwargs.get("task_entry")
      seen["call_index"] = kwargs.get("call_index")
      return {"response": "done"}, None

    result, error = await runner._register_background_task(
      tool_input={"task": "Collect"},
      handler=_handler,
      agent_name="writer",
    )

    assert error is None
    assert result is not None
    task_entry = runner._task_registry.get(result["task_id"])
    assert task_entry is not None
    assert task_entry.asyncio_task is not None

    await asyncio.wait_for(task_entry.asyncio_task, timeout=1.0)

    assert seen["task_entry"] is task_entry
    assert seen["call_index"] == 0

  asyncio.run(_case())


def test_build_notification_reminder_includes_pending_suffix_when_over_cap() -> None:
  runner = _make_runner()
  for index in range(7):
    runner._notification_queue.push(_notification(f"bg_{index}", summary=f"done {index}"))

  reminder = runner._build_notification_reminder()

  assert reminder.count("<task-notification") == 5
  assert "bg_0" in reminder
  assert "bg_4" in reminder
  assert "bg_5" not in reminder
  assert "[2 more task notification(s) pending]" in reminder
  assert runner._notification_queue.pending_count == 7


def test_consume_notifications_drains_requested_count() -> None:
  runner = _make_runner()
  runner._notification_queue.push(_notification("bg_0"))
  runner._notification_queue.push(_notification("bg_1"))
  runner._notification_queue.push(_notification("bg_2"))

  consumed = runner._consume_notifications(max_count=2)

  assert consumed == 2
  assert runner._notification_queue.pending_count == 1
  assert [item.task_id for item in runner._notification_queue.peek()] == ["bg_2"]
