import asyncio
import math
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (  # noqa: E402
  AgentRunner,
  NotificationQueue,
  TaskNotification,
  TaskRegistry,
  WorkflowTaskMetadata,
)
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_background_lifecycle import RunnerBackgroundLifecycleMixin  # noqa: E402
from agent_gateway.runner_background_tasks import (  # noqa: E402
  BackgroundResultRequest,
  PlanNotificationProducer,
  PlanProgressSnapshot,
  background_asyncio_tasks,
  background_task_call_index,
  background_elapsed_seconds,
  background_task_payload,
  background_task_ids,
  background_task_ids_for_asyncio_tasks,
  background_task_limit_error,
  background_task_registration_metadata,
  background_task_reminder_text,
  background_result_task,
  background_task_started_result,
  background_timeout_value,
  background_wait_tasks,
  call_before_background_task_start_hook,
  drain_cancelled_background_tasks,
  drain_still_pending_background_tasks,
  entry_aware_background_handler,
  ensure_sub_agent_semaphore,
  kill_background_tasks,
  kill_background_tasks_for_asyncio_tasks,
  parse_background_result_request,
  parse_cost_observation_threshold_usd,
  prepare_background_task_registration,
  resume_chain_depth,
  resume_root_task_id,
  resume_root_task_id_from_registry,
  resume_task_id_override,
  resumed_task_ids,
  resumed_task_ids_from_registry,
  task_completed_event_payload,
  task_correlation_payload,
  task_registered_event_payload,
  wait_for_background_tasks,
  workflow_owns_terminal_notification,
  WORKFLOW_TASK_METADATA_KEY,
)
from agent_gateway.task_registry import TaskState  # noqa: E402
from agent_workflow_contracts import (  # noqa: E402
  AgentOperationRef,
  OutcomeRequirement,
  ResultRequirement,
)


def _progress(**overrides: Any) -> SimpleNamespace:
  values = {
    "tool_use_count": 0,
    "turn_count": 0,
    "last_tool_name": None,
    "last_activity_at": None,
    "output_tokens": 0,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def _bind_receipt(**overrides: str) -> dict[str, str]:
  values = {
    "capability_id": "node.implement",
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "effort": "high",
    "policy_id": "test-policy",
    "policy_version": "1",
    "credential_principal": "user",
    "run_mode": "interactive",
  }
  values.update(overrides)
  return values


def _report_submission(
  summary: str,
  *,
  submission_id: str,
) -> dict[str, Any]:
  return {
    "submission_id": submission_id,
    "projection": {
      "summary": summary,
      "findings": [],
      "artifacts": [],
      "caveats": [],
    },
  }


def _task(**overrides: Any) -> SimpleNamespace:
  values = {
    "task_id": "bg_1",
    "agent_name": None,
    "started_at": 100.0,
    "completed_at": None,
    "completed": False,
    "result": None,
    "error": None,
    "state": TaskState.RUNNING,
    "progress": _progress(),
    "metadata": {},
    "task_type": "background_agent",
    "capability_bind_receipt": _bind_receipt(),
    "original_task_id": None,
    "asyncio_task": None,
    "termination_intent": None,
    "pending_final_state": None,
    "completion_persistence_state": "not_started",
    "completion_persistence_error": None,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def _entry(event: dict[str, Any]) -> SimpleNamespace:
  return SimpleNamespace(event=event)


def test_runner_background_lifecycle_methods_are_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerBackgroundLifecycleMixin)
  assert gateway_runner.RunnerBackgroundLifecycleMixin is RunnerBackgroundLifecycleMixin

  for method_name in (
    "_ensure_sub_agent_semaphore",
    "_background_timeout_value",
    "_background_elapsed_seconds",
    "_task_entry_for_chain",
    "_resume_chain_depth",
    "_resume_root_task_id",
    "_resumed_task_ids",
    "_resume_root_task_id_from_registry",
    "_resumed_task_ids_from_registry",
    "_background_task_payload",
    "_background_task_reminder_text",
    "_build_notification_reminder",
    "_consume_notifications",
    "_wait_for_background_notification",
    "_inject_system_prompt_reminder",
    "_run_background_agent",
    "_task_correlation_payload",
    "_append_task_completed_event",
    "_register_background_task",
    "cancel_background_task",
    "get_background_result",
    "_shutdown_background_tasks",
  ):
    assert getattr(AgentRunner, method_name) is getattr(RunnerBackgroundLifecycleMixin, method_name)


def test_runner_background_lifecycle_resolves_parent_notification_helpers(monkeypatch: Any) -> None:
  runner = object.__new__(AgentRunner)
  runner._notification_queue = object()
  captured: dict[str, Any] = {}

  def _build_notification_reminder(queue: Any, *, max_count: int) -> str:
    captured["queue"] = queue
    captured["max_count"] = max_count
    return "patched reminder"

  monkeypatch.setattr(gateway_runner, "_build_notification_reminder", _build_notification_reminder)
  monkeypatch.setattr(gateway_runner, "_MAX_NOTIFICATIONS_PER_TURN", 9)
  monkeypatch.setattr(gateway_runner, "_consume_notifications", lambda queue, *, max_count: max_count + 1)

  assert AgentRunner._build_notification_reminder(runner) == "patched reminder"
  assert captured == {"queue": runner._notification_queue, "max_count": 9}
  assert AgentRunner._consume_notifications(runner, 3) == 4


def test_wait_for_background_notification_wakes_before_child_callback_returns() -> None:
  async def case() -> None:
    runner = object.__new__(AgentRunner)
    runner._sid = "sess_wait_notification"
    runner._notification_queue = NotificationQueue()
    runner._task_registry = TaskRegistry()
    runner._operator_pause_event = asyncio.Event()

    release_child = asyncio.Event()

    async def child() -> None:
      await release_child.wait()

    entry = runner._task_registry.register("background_agent")
    child_task = asyncio.create_task(child())
    entry.asyncio_task = child_task
    runner._task_registry.transition(entry.task_id, TaskState.RUNNING)

    wait_task = asyncio.create_task(
      AgentRunner._wait_for_background_notification(runner)
    )
    await asyncio.sleep(0)
    runner._notification_queue.push(
      TaskNotification(
        task_id=entry.task_id,
        agent_name=None,
        event="completed",
        summary="durable completion committed",
        timestamp=1.0,
        payload={"kind": "report"},
      )
    )

    assert await asyncio.wait_for(wait_task, timeout=0.1) is True
    assert not child_task.done()

    release_child.set()
    await child_task

  asyncio.run(case())


def test_wait_for_background_notification_rescans_after_interruption() -> None:
  async def case() -> None:
    runner = object.__new__(AgentRunner)
    runner._sid = "sess_wait_rescan"
    runner._notification_queue = NotificationQueue()
    runner._task_registry = TaskRegistry()
    runner._operator_pause_event = asyncio.Event()

    begin = asyncio.Event()
    release_second = asyncio.Event()
    first_entry = runner._task_registry.register("background_agent")
    second_entry = runner._task_registry.register("background_agent")

    async def interrupt_first() -> None:
      await begin.wait()
      runner._task_registry.transition(
        first_entry.task_id,
        TaskState.INTERRUPTED,
      )

    async def complete_second() -> None:
      await begin.wait()
      await release_second.wait()
      runner._task_registry.transition(
        second_entry.task_id,
        TaskState.COMPLETED,
        result={"kind": "report", "report": {"summary": "second done"}},
      )
      runner._notification_queue.push(
        TaskNotification(
          task_id=second_entry.task_id,
          agent_name=None,
          event="completed",
          summary="second done",
          timestamp=1.0,
          payload={"kind": "report"},
        )
      )

    first_task = asyncio.create_task(interrupt_first())
    second_task = asyncio.create_task(complete_second())
    first_entry.asyncio_task = first_task
    second_entry.asyncio_task = second_task
    runner._task_registry.transition(first_entry.task_id, TaskState.RUNNING)
    runner._task_registry.transition(second_entry.task_id, TaskState.RUNNING)

    wait_task = asyncio.create_task(
      AgentRunner._wait_for_background_notification(runner)
    )
    await asyncio.sleep(0)
    begin.set()
    await first_task
    await asyncio.sleep(0)
    assert not wait_task.done()

    release_second.set()
    assert await asyncio.wait_for(wait_task, timeout=0.1) is True
    await second_task

  asyncio.run(case())


def test_wait_for_background_notification_ignores_done_handles_and_honors_pause() -> None:
  async def case() -> None:
    runner = object.__new__(AgentRunner)
    runner._sid = "sess_wait_pause"
    runner._notification_queue = NotificationQueue()
    runner._task_registry = TaskRegistry()
    runner._operator_pause_event = asyncio.Event()

    async def already_done() -> None:
      return

    done_entry = runner._task_registry.register("background_agent")
    done_task = asyncio.create_task(already_done())
    await done_task
    done_entry.asyncio_task = done_task
    runner._task_registry.transition(done_entry.task_id, TaskState.RUNNING)
    assert (
      await asyncio.wait_for(
        AgentRunner._wait_for_background_notification(runner),
        timeout=0.1,
      )
      is False
    )

    runner._task_registry.transition(
      done_entry.task_id,
      TaskState.INTERRUPTED,
    )
    release_live = asyncio.Event()

    async def live_child() -> None:
      await release_live.wait()

    live_entry = runner._task_registry.register("background_agent")
    live_task = asyncio.create_task(live_child())
    live_entry.asyncio_task = live_task
    runner._task_registry.transition(live_entry.task_id, TaskState.RUNNING)

    pause_wait = asyncio.create_task(
      AgentRunner._wait_for_background_notification(runner)
    )
    await asyncio.sleep(0)
    assert not pause_wait.done()
    runner._operator_pause_event.set()
    assert await asyncio.wait_for(pause_wait, timeout=0.1) is False
    assert not live_task.done()

    release_live.set()
    await live_task

  asyncio.run(case())


def test_runner_background_registration_resolves_parent_module_helpers(monkeypatch: Any) -> None:
  transitions: list[tuple[str, TaskState]] = []
  registered_types: list[str] = []
  workflow_metadata_seen: list[WorkflowTaskMetadata | None] = []

  class Registry:
    inflight_count = 0
    admission_count = 0
    pending_notification_retrieval_count = 0
    notification_retrieval_retention_limit = 10

    def notification_retrieval_capacity_available(
      self,
      *,
      admission_count: int | None = None,
    ) -> bool:
      _ = admission_count
      return True

    def register(
      self,
      task_type: str,
      **kwargs: Any,
    ) -> SimpleNamespace:
      registered_types.append(task_type)
      return _task(
        task_id=kwargs.get("task_id") or "bg_5",
        task_type=task_type,
        metadata={},
      )

    def transition(self, task_id: str, state: TaskState, **_kwargs: Any) -> None:
      transitions.append((task_id, state))

  runner = object.__new__(AgentRunner)
  runner._task_registry = Registry()
  runner._max_background_tasks = 3
  runner._max_resume_chain_depth = 3
  runner._provider = SimpleNamespace(name="stub-provider")
  runner._auth_config = {"model": "stub-model"}
  runner._runner_id = "runner-1"
  runner._role = "writer"
  runner._gateway_session_id = "sess"
  runner._parent_turn_id = "turn-1"
  runner._sid = "sess"

  durable_events: list[dict[str, Any]] = []

  async def _append_durable_event(event: dict[str, Any]) -> None:
    durable_events.append(event)

  runner._append_durable_event = _append_durable_event  # type: ignore[method-assign]

  def _prepare_background_task_registration(entry: Any, **kwargs: Any) -> int:
    workflow_metadata_seen.append(kwargs["workflow_task_metadata"])
    entry.metadata["patched_sub_agent_id"] = kwargs["sub_agent_id_for_call_index"](5)
    entry.capability_bind_receipt = kwargs["capability_bind_receipt"]
    return 5

  created_tasks: list[Any] = []

  def _create_task(coro: Any, *, name: str | None = None) -> SimpleNamespace:
    if hasattr(coro, "close"):
      coro.close()
    task = SimpleNamespace(name=name)
    created_tasks.append(task)
    return task

  monkeypatch.setattr(gateway_runner, "_prepare_background_task_registration", _prepare_background_task_registration)
  monkeypatch.setattr(gateway_runner, "_derive_sub_agent_id", lambda session_id, call_index: f"patched-{session_id}-{call_index}")
  monkeypatch.setattr(gateway_runner, "_task_registered_event_payload", lambda entry, **_kwargs: {"type": "patched_registered", "sub": entry.metadata["patched_sub_agent_id"]})
  monkeypatch.setattr(gateway_runner, "_background_task_started_result", lambda entry, *, agent_name: {"patched": entry.task_id, "agent": agent_name})
  monkeypatch.setattr(gateway_runner, "_entry_aware_background_handler", lambda handler, _entry: handler)
  monkeypatch.setattr(gateway_runner, "asyncio", SimpleNamespace(create_task=_create_task))

  async def handler(_tool_input: dict[str, Any], **_kwargs: Any) -> tuple[None, None]:
    return None, None

  workflow_task_metadata = _workflow_task_metadata()
  result, error = asyncio.run(
    AgentRunner._register_background_task(
      runner,
      tool_input={},
      handler=handler,
      task_type="plan_run",
      agent_name="Analyst",
      workflow_task_metadata=workflow_task_metadata,
    )
  )

  assert error is None
  assert result == {"patched": "bg_5", "agent": "Analyst"}
  assert durable_events == [{"type": "patched_registered", "sub": "patched-sess-5"}]
  assert created_tasks[0].name == "bg_5"
  assert registered_types == ["plan_run"]
  assert workflow_metadata_seen == [workflow_task_metadata]
  assert transitions == [("bg_5", TaskState.RUNNING)]


def test_runner_shutdown_background_tasks_resolves_parent_module_helpers(monkeypatch: Any) -> None:
  calls: list[Any] = []
  pending = [object()]
  entry = _task(task_id="bg_1")

  class Registry:
    def list_tasks(self, *, state: TaskState) -> list[SimpleNamespace]:
      calls.append(("list", state))
      return [entry] if state == TaskState.RUNNING else []

    def kill(self, task_id: str, **_kwargs: Any) -> None:
      calls.append(("kill", task_id))

  runner = object.__new__(AgentRunner)
  runner._task_registry = Registry()

  async def _drain_cancelled_background_tasks(running_entries: Any, selected_pending: Any, **kwargs: Any) -> None:
    calls.append(("drain_cancelled", list(running_entries)[0].task_id, selected_pending is pending, kwargs["timeout"]))
    entry.state = TaskState.KILLED

  monkeypatch.setattr(gateway_runner, "_background_asyncio_tasks", lambda _entries: pending)
  monkeypatch.setattr(gateway_runner, "_drain_cancelled_background_tasks", _drain_cancelled_background_tasks)
  monkeypatch.setattr(gateway_runner, "asyncio", SimpleNamespace(wait=object()))

  asyncio.run(AgentRunner._shutdown_background_tasks(runner, was_cancelled=True))

  assert calls == [
    ("list", TaskState.PENDING),
    ("list", TaskState.RUNNING),
    ("drain_cancelled", "bg_1", True, 5.0),
  ]


def test_background_timeout_and_elapsed_helpers_bound_values() -> None:
  assert background_timeout_value(None) == 60.0
  assert background_timeout_value(-5) == 0.0
  assert background_timeout_value(300) == 120.0
  assert background_timeout_value("45") == 45.0
  assert background_elapsed_seconds(_task(started_at=100.4), now=110.9) == 10
  assert background_elapsed_seconds(_task(started_at=120.0), now=110.0) == 0
  assert background_elapsed_seconds(_task(started_at=100.0, completed_at=105.9), now=110.0) == 5


def test_ensure_sub_agent_semaphore_creates_only_when_limit_exists() -> None:
  created: list[int] = []

  def factory(limit: int) -> dict[str, int]:
    created.append(limit)
    return {"limit": limit}

  existing = {"limit": 2}

  assert ensure_sub_agent_semaphore(None, None, semaphore_factory=factory) is None
  assert created == []
  assert ensure_sub_agent_semaphore(existing, 5, semaphore_factory=factory) is existing
  assert created == []
  assert ensure_sub_agent_semaphore(None, 3, semaphore_factory=factory) == {"limit": 3}
  assert created == [3]


def test_parse_background_result_request_normalizes_valid_inputs() -> None:
  request, error = parse_background_result_request({
    "task_id": " bg_1 ",
    "wait": True,
    "timeout": 300,
    "cursor": " next-page ",
  })

  assert error is None
  assert request == BackgroundResultRequest(
    task_id="bg_1",
    wait=True,
    timeout=120.0,
    cursor="next-page",
  )

  request, error = parse_background_result_request({"task_id": "*"})

  assert error is None
  assert request == BackgroundResultRequest(
    task_id="*",
    wait=False,
    timeout=60.0,
    cursor=None,
  )


def test_parse_background_result_request_preserves_legacy_errors() -> None:
  assert parse_background_result_request({}) == (
    None,
    {"code": "invalid_input", "message": "task_id is required"},
  )
  assert parse_background_result_request({"task_id": "bg_1", "wait": "yes"}) == (
    None,
    {"code": "invalid_input", "message": "wait must be a boolean"},
  )
  assert parse_background_result_request({"task_id": "bg_1", "timeout": "5"}) == (
    None,
    {"code": "invalid_input", "message": "timeout must be a number"},
  )
  assert parse_background_result_request({"task_id": "bg_1", "timeout": True}) == (
    None,
    {"code": "invalid_input", "message": "timeout must be a number"},
  )
  assert parse_background_result_request({"task_id": "bg_1", "timeout": math.inf}) == (
    None,
    {"code": "invalid_input", "message": "timeout must be finite"},
  )
  assert parse_background_result_request({"task_id": "bg_1", "timeout": -math.inf}) == (
    None,
    {"code": "invalid_input", "message": "timeout must be finite"},
  )
  assert parse_background_result_request({"task_id": "bg_1", "timeout": math.nan}) == (
    None,
    {"code": "invalid_input", "message": "timeout must be finite"},
  )
  assert parse_background_result_request({"task_id": "bg_1", "cursor": 3}) == (
    None,
    {
      "code": "invalid_input",
      "message": "cursor must be a non-empty string",
    },
  )
  assert parse_background_result_request({"task_id": "*", "cursor": "next"}) == (
    None,
    {
      "code": "invalid_input",
      "message": "cursor requires one exact task_id",
    },
  )


def test_background_shutdown_selection_helpers_filter_task_handles_and_ids() -> None:
  task_a = object()
  task_b = object()
  task_c = object()
  entries = [
    _task(task_id="bg_1", asyncio_task=task_a),
    _task(task_id="bg_2", asyncio_task=None),
    _task(task_id="bg_3", asyncio_task=task_b),
    _task(task_id="bg_4", asyncio_task=task_c, completed=True),
  ]

  assert background_asyncio_tasks(entries) == [task_a, task_b, task_c]
  assert background_wait_tasks(entries) == [task_a, task_b]
  assert background_task_ids(entries) == ["bg_1", "bg_2", "bg_3", "bg_4"]
  assert background_task_ids_for_asyncio_tasks(entries, [task_b]) == ["bg_3"]


def test_background_shutdown_kill_helpers_preserve_selection_order() -> None:
  task_a = object()
  task_b = object()
  task_c = object()
  entries = [
    _task(task_id="bg_1", asyncio_task=task_a),
    _task(task_id="bg_2", asyncio_task=None),
    _task(task_id="bg_3", asyncio_task=task_b),
    _task(task_id="bg_4", asyncio_task=task_c),
  ]
  killed: list[str] = []

  kill_background_tasks(entries, kill_task=killed.append)

  assert gateway_runner.kill_background_tasks is kill_background_tasks
  assert gateway_runner._kill_background_tasks is kill_background_tasks
  assert killed == ["bg_1", "bg_2", "bg_3", "bg_4"]

  killed.clear()
  kill_background_tasks_for_asyncio_tasks(entries, [task_c, task_a], kill_task=killed.append)

  assert gateway_runner.kill_background_tasks_for_asyncio_tasks is kill_background_tasks_for_asyncio_tasks
  assert gateway_runner._kill_background_tasks_for_asyncio_tasks is kill_background_tasks_for_asyncio_tasks
  assert killed == ["bg_1", "bg_4"]


def test_drain_cancelled_background_tasks_kills_then_drains_pending() -> None:
  task_a = object()
  task_b = object()
  entries = [
    _task(task_id="bg_1", asyncio_task=task_a),
    _task(task_id="bg_2", asyncio_task=task_b),
  ]
  events: list[str] = []

  def kill_task(task_id: str) -> None:
    events.append(f"kill:{task_id}")

  async def wait_fn(
    pending: list[object],
    *,
    timeout: float,
  ) -> tuple[list[object], list[object]]:
    assert pending == [task_a, task_b]
    events.append(f"wait:{timeout}")
    return [task_a], [task_b]

  remaining = asyncio.run(
    drain_cancelled_background_tasks(
      entries,
      [task_a, task_b],
      kill_task=kill_task,
      wait_fn=wait_fn,
      timeout=5.0,
    )
  )

  assert gateway_runner._drain_cancelled_background_tasks is drain_cancelled_background_tasks
  assert events == ["kill:bg_1", "kill:bg_2", "wait:5.0"]
  assert remaining == {task_b}


def test_drain_still_pending_background_tasks_returns_when_all_done() -> None:
  task_a = object()
  task_b = object()
  entries = [
    _task(task_id="bg_1", asyncio_task=task_a),
    _task(task_id="bg_2", asyncio_task=task_b),
  ]
  events: list[str] = []

  def kill_task(task_id: str) -> None:
    events.append(f"kill:{task_id}")

  async def wait_fn(pending: list[object], *, timeout: float) -> tuple[list[object], list[object]]:
    assert pending == [task_a, task_b]
    events.append(f"wait:{timeout}")
    return [task_a, task_b], []

  remaining = asyncio.run(
    drain_still_pending_background_tasks(
      entries,
      [task_a, task_b],
      kill_task=kill_task,
      wait_fn=wait_fn,
      wait_timeout=30.0,
      drain_timeout=5.0,
    )
  )

  assert gateway_runner._drain_still_pending_background_tasks is drain_still_pending_background_tasks
  assert events == ["wait:30.0"]
  assert remaining == set()


def test_drain_still_pending_background_tasks_kills_then_drains_selected_pending() -> None:
  task_a = object()
  task_b = object()
  task_c = object()
  entries = [
    _task(task_id="bg_1", asyncio_task=task_a),
    _task(task_id="bg_2", asyncio_task=task_b),
    _task(task_id="bg_3", asyncio_task=task_c),
  ]
  still_pending = [task_c, task_a]
  events: list[str] = []
  wait_count = 0

  def kill_task(task_id: str) -> None:
    events.append(f"kill:{task_id}")

  async def wait_fn(pending: list[object], *, timeout: float) -> tuple[list[object], list[object]]:
    nonlocal wait_count
    wait_count += 1
    events.append(f"wait:{timeout}")
    if wait_count == 1:
      assert pending == [task_a, task_b, task_c]
      return [task_b], still_pending
    assert set(pending) == {task_a, task_c}
    return [], list(pending)

  remaining = asyncio.run(
    drain_still_pending_background_tasks(
      entries,
      [task_a, task_b, task_c],
      kill_task=kill_task,
      wait_fn=wait_fn,
      wait_timeout=30.0,
      drain_timeout=5.0,
    )
  )

  assert gateway_runner._drain_still_pending_background_tasks is drain_still_pending_background_tasks
  assert events == ["wait:30.0", "kill:bg_1", "kill:bg_3", "wait:5.0"]
  assert remaining == {task_a, task_c}


def test_cancelled_drain_has_hard_elapsed_bound_for_stubborn_task() -> None:
  async def _case() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _stubborn_worker() -> None:
      started.set()
      while not release.is_set():
        try:
          await release.wait()
        except asyncio.CancelledError:
          continue

    task = asyncio.create_task(_stubborn_worker())
    await started.wait()
    entry = _task(task_id="bg_stubborn", asyncio_task=task)

    started_at = asyncio.get_running_loop().time()
    remaining = await drain_cancelled_background_tasks(
      [entry],
      [task],
      kill_task=lambda _task_id: task.cancel(),
      wait_fn=asyncio.wait,
      timeout=0.01,
    )
    elapsed = asyncio.get_running_loop().time() - started_at

    assert task in remaining
    assert elapsed < 0.2

    release.set()
    await asyncio.wait_for(task, timeout=1.0)

  asyncio.run(_case())


def test_grace_then_kill_drain_has_hard_total_bound_for_stubborn_task() -> None:
  async def _case() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _stubborn_worker() -> None:
      started.set()
      while not release.is_set():
        try:
          await release.wait()
        except asyncio.CancelledError:
          continue

    task = asyncio.create_task(_stubborn_worker())
    await started.wait()
    entry = _task(task_id="bg_stubborn", asyncio_task=task)

    started_at = asyncio.get_running_loop().time()
    remaining = await drain_still_pending_background_tasks(
      [entry],
      [task],
      kill_task=lambda _task_id: task.cancel(),
      wait_fn=asyncio.wait,
      wait_timeout=0.01,
      drain_timeout=0.01,
    )
    elapsed = asyncio.get_running_loop().time() - started_at

    assert task in remaining
    assert elapsed < 0.2

    release.set()
    await asyncio.wait_for(task, timeout=1.0)

  asyncio.run(_case())


def test_wait_for_background_tasks_short_circuits_or_waits_for_pending_handles() -> None:
  task_a = object()
  task_b = object()
  entries = [
    _task(task_id="bg_1", asyncio_task=task_a),
    _task(task_id="bg_2", asyncio_task=None),
    _task(task_id="bg_3", asyncio_task=task_b, completed=True),
  ]
  calls: list[tuple[list[Any], float]] = []

  async def wait_fn(pending: list[Any], *, timeout: float) -> None:
    calls.append((pending, timeout))

  asyncio.run(wait_for_background_tasks(entries, wait=False, timeout=1.5, wait_fn=wait_fn))
  assert calls == []

  asyncio.run(
    wait_for_background_tasks(
      [_task(task_id="bg_4", asyncio_task=None)],
      wait=True,
      timeout=2.5,
      wait_fn=wait_fn,
    )
  )
  assert calls == []

  asyncio.run(wait_for_background_tasks(entries, wait=True, timeout=3.5, wait_fn=wait_fn))
  assert calls == [([task_a], 3.5)]


def test_background_result_task_uses_registry_before_log_fallback() -> None:
  registry_task = _task(task_id="bg_registry")
  log_task = _task(task_id="bg_log")
  log_lookups: list[str] = []

  async def log_lookup(task_id: str) -> SimpleNamespace | None:
    log_lookups.append(task_id)
    if task_id == "bg_log":
      return log_task
    return None

  task, error = asyncio.run(
    background_result_task(
      "bg_registry",
      registry_lookup={"bg_registry": registry_task}.get,
      log_lookup=log_lookup,
    )
  )

  assert task is registry_task
  assert error is None
  assert log_lookups == []

  task, error = asyncio.run(
    background_result_task(
      "bg_log",
      registry_lookup={}.get,
      log_lookup=log_lookup,
    )
  )

  assert task is log_task
  assert error is None
  assert log_lookups == ["bg_log"]


def test_background_result_task_returns_legacy_not_found_error() -> None:
  log_lookups: list[str] = []

  async def log_lookup(task_id: str) -> None:
    log_lookups.append(task_id)
    return None

  task, error = asyncio.run(
    background_result_task(
      "missing",
      registry_lookup={}.get,
      log_lookup=log_lookup,
    )
  )

  assert task is None
  assert error == {"code": "not_found", "message": "Unknown background task: missing"}
  assert log_lookups == ["missing"]


def test_background_task_limit_error_preserves_legacy_message() -> None:
  assert background_task_limit_error(admission_count=1, max_background_tasks=2) is None
  assert background_task_limit_error(admission_count=2, max_background_tasks=2) == {
    "code": "max_background_tasks",
    "message": (
      "Background task limit reached (2). "
      "Wait for an existing background task to finish before launching another."
    ),
  }
  assert background_task_limit_error(admission_count=3, max_background_tasks=2)["code"] == "max_background_tasks"


def test_call_before_background_task_start_hook_handles_none_success_and_failure() -> None:
  class Logger:
    def __init__(self) -> None:
      self.warnings: list[tuple[str, tuple[Any, ...]]] = []

    def warning(self, message: str, *args: Any) -> None:
      self.warnings.append((message, args))

  logger = Logger()
  calls: list[str] = []
  resolved_sessions: list[str] = []

  def session_id() -> str:
    resolved_sessions.append("resolved")
    return "sid-2"

  call_before_background_task_start_hook(None, log_session_id=session_id, logger=logger)
  call_before_background_task_start_hook(lambda: calls.append("started"), log_session_id=session_id, logger=logger)

  def failing_hook() -> None:
    calls.append("failing")
    raise RuntimeError("boom")

  call_before_background_task_start_hook(failing_hook, log_session_id=session_id, logger=logger)

  assert calls == ["started", "failing"]
  assert resolved_sessions == ["resolved"]
  assert len(logger.warnings) == 1
  message, args = logger.warnings[0]
  assert message == "[%s] on_before_start hook failed (non-fatal): %s"
  assert args[0] == "sid-2"
  assert isinstance(args[1], RuntimeError)
  assert str(args[1]) == "boom"


def test_entry_aware_background_handler_injects_entry_and_preserves_kwargs() -> None:
  entry = _task(task_id="bg_7")
  seen: dict[str, Any] = {}

  async def handler(tool_input: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], None]:
    seen["tool_input"] = tool_input
    seen["kwargs"] = kwargs
    return {"response": "done"}, None

  wrapped = entry_aware_background_handler(handler, entry)
  result, error = asyncio.run(
    wrapped(
      {"task": "Collect"},
      call_index=7,
      task_entry="stale-entry",
    )
  )

  assert gateway_runner._entry_aware_background_handler is entry_aware_background_handler
  assert result == {"response": "done"}
  assert error is None
  assert seen == {
    "tool_input": {"task": "Collect"},
    "kwargs": {
      "call_index": 7,
      "task_entry": entry,
    },
  }


def test_resume_chain_depth_counts_parents_and_breaks_cycles() -> None:
  tasks = {
    "bg_root": _task(task_id="bg_root"),
    "bg_r1": _task(task_id="bg_r1", original_task_id="bg_root"),
    "bg_r2": _task(task_id="bg_r2", original_task_id="bg_r1"),
  }

  async def lookup(task_id: str) -> SimpleNamespace | None:
    return tasks.get(task_id)

  assert asyncio.run(resume_chain_depth("bg_r2", task_lookup=lookup)) == 2
  assert asyncio.run(resume_chain_depth("missing", task_lookup=lookup)) == 0

  cycle_tasks = {
    "a": _task(task_id="a", original_task_id="b"),
    "b": _task(task_id="b", original_task_id="a"),
  }

  async def cycle_lookup(task_id: str) -> SimpleNamespace | None:
    return cycle_tasks.get(task_id)

  assert asyncio.run(resume_chain_depth("a", task_lookup=cycle_lookup)) == 2


def test_resume_root_task_id_follows_parents_and_breaks_cycles() -> None:
  tasks = {
    "bg_root": _task(task_id="bg_root"),
    "bg_r1": _task(task_id="bg_r1", original_task_id="bg_root"),
    "bg_r2": _task(task_id="bg_r2", original_task_id="bg_r1"),
  }

  async def lookup(task_id: str) -> SimpleNamespace | None:
    return tasks.get(task_id)

  assert asyncio.run(resume_root_task_id("bg_r2", task_lookup=lookup)) == "bg_root"
  assert asyncio.run(resume_root_task_id("missing", task_lookup=lookup)) == "missing"

  cycle_tasks = {
    "a": _task(task_id="a", original_task_id="b"),
    "b": _task(task_id="b", original_task_id="a"),
  }

  async def cycle_lookup(task_id: str) -> SimpleNamespace | None:
    return cycle_tasks.get(task_id)

  assert asyncio.run(resume_root_task_id("a", task_lookup=cycle_lookup)) == "a"


def test_resumed_task_ids_returns_descendants_in_entry_order() -> None:
  tasks = {
    "bg_root": _task(task_id="bg_root"),
    "bg_r1": _task(task_id="bg_r1", original_task_id="bg_root"),
    "bg_other": _task(task_id="bg_other"),
    "bg_r2": _task(task_id="bg_r2", original_task_id="bg_r1"),
  }

  async def lookup(task_id: str) -> SimpleNamespace | None:
    return tasks.get(task_id)

  async def root(task_id: str) -> str:
    return await resume_root_task_id(task_id, task_lookup=lookup)

  assert asyncio.run(
    resumed_task_ids(
      "bg_r2",
      task_entries=tasks.values,
      resume_root=root,
    )
  ) == ["bg_r1", "bg_r2"]


def test_resumed_task_ids_lists_entries_after_root_resolution() -> None:
  events: list[str] = []

  async def root(task_id: str) -> str:
    events.append(f"root:{task_id}")
    return "bg_root"

  def entries() -> list[SimpleNamespace]:
    events.append("list")
    return [
      _task(task_id="bg_root"),
      _task(task_id="bg_r1", original_task_id="bg_root"),
    ]

  assert asyncio.run(
    resumed_task_ids(
      "bg_r1",
      task_entries=entries,
      resume_root=root,
    )
  ) == ["bg_r1"]
  assert events[:2] == ["root:bg_r1", "list"]


def test_resume_task_id_override_handles_absent_original_without_callbacks() -> None:
  calls: list[str] = []

  async def depth(task_id: str) -> int:
    calls.append(f"depth:{task_id}")
    return 0

  async def root(task_id: str) -> str:
    calls.append(f"root:{task_id}")
    return task_id

  assert asyncio.run(
    resume_task_id_override(
      None,
      max_resume_chain_depth=3,
      resume_chain_depth=depth,
      resume_root=root,
    )
  ) == (None, None)
  assert calls == []


def test_resume_task_id_override_builds_next_id_or_depth_error() -> None:
  calls: list[str] = []

  async def depth(task_id: str) -> int:
    calls.append(f"depth:{task_id}")
    return 2

  async def root(task_id: str) -> str:
    calls.append(f"root:{task_id}")
    return "bg_root"

  assert asyncio.run(
    resume_task_id_override(
      "bg_root_r2",
      max_resume_chain_depth=3,
      resume_chain_depth=depth,
      resume_root=root,
    )
  ) == ("bg_root_r3", None)
  assert calls == ["depth:bg_root_r2", "root:bg_root_r2"]

  calls.clear()

  assert asyncio.run(
    resume_task_id_override(
      "bg_root_r2",
      max_resume_chain_depth=2,
      resume_chain_depth=depth,
      resume_root=root,
    )
  ) == (
    None,
    {
      "code": "max_resume_chain_depth",
      "message": "Resume chain depth limit reached (2) for bg_root_r2",
    },
  )
  assert calls == ["depth:bg_root_r2"]


def test_resume_root_task_id_from_registry_follows_parents_and_breaks_cycles() -> None:
  tasks = {
    "bg_root": _task(task_id="bg_root"),
    "bg_r1": _task(task_id="bg_r1", original_task_id="bg_root"),
    "bg_r2": _task(task_id="bg_r2", original_task_id="bg_r1"),
  }

  assert resume_root_task_id_from_registry("bg_r2", task_lookup=tasks.get) == "bg_root"
  assert resume_root_task_id_from_registry("missing", task_lookup=tasks.get) == "missing"

  cycle_tasks = {
    "a": _task(task_id="a", original_task_id="b"),
    "b": _task(task_id="b", original_task_id="a"),
  }

  assert resume_root_task_id_from_registry("a", task_lookup=cycle_tasks.get) == "a"


def test_resumed_task_ids_from_registry_returns_descendants_in_entry_order() -> None:
  tasks = {
    "bg_root": _task(task_id="bg_root"),
    "bg_r1": _task(task_id="bg_r1", original_task_id="bg_root"),
    "bg_other": _task(task_id="bg_other"),
    "bg_r2": _task(task_id="bg_r2", original_task_id="bg_r1"),
  }

  assert resumed_task_ids_from_registry(
    "bg_r2",
    task_entries=tasks.values(),
    task_lookup=tasks.get,
  ) == ["bg_r1", "bg_r2"]


def test_background_task_payload_formats_running_completed_and_failed_states() -> None:
  running = _task(
    agent_name="Analyst",
    progress=_progress(
      tool_use_count=2,
      turn_count=3,
      last_tool_name="lookup",
      last_activity_at=115.0,
      output_tokens=42,
    ),
  )
  completed = _task(
    completed=True,
    state=TaskState.COMPLETED,
    result={"task_id": "ignored", "status": "ignored", "agent": "ignored", "response": "done"},
  )
  failed = _task(completed=True, state=TaskState.FAILED, error={"code": "boom"})
  abandoned = _task(
    completed=True,
    state=TaskState.FAILED,
    result={
      "kind": "unstructured",
      "version": "1",
      "response": "Resume chain cannot continue",
      "reason": "resume_abandoned",
      "error_detail": "contract drift",
      "fms_results": [],
      "artifact_events": [],
      "usage": {},
      "tools_used": [],
      "warning": "resume_abandoned_code=contract_drift",
    },
  )

  assert background_task_payload(running, elapsed_seconds=25, now=120.0) == {
    "task_id": "bg_1",
    "status": "running",
    "agent": "Analyst",
    "elapsed_seconds": 25,
    "progress": {
      "tools_used": 2,
      "turns": 3,
      "last_tool": "lookup",
      "idle_seconds": 5,
      "output_tokens": 42,
    },
  }
  assert background_task_payload(completed, elapsed_seconds=8, now=120.0) == {
    "task_id": "bg_1",
    "status": "completed",
    "elapsed_seconds": 8,
    "response": "done",
  }
  assert background_task_payload(failed, elapsed_seconds=8, now=120.0) == {
      "task_id": "bg_1",
      "status": "error",
      "elapsed_seconds": 8,
      "error": {"code": "boom", "message": "boom"},
    }
  assert background_task_payload(
    abandoned,
    elapsed_seconds=9,
    now=120.0,
  ) == {
    "task_id": "bg_1",
    "status": "error",
    "elapsed_seconds": 9,
    "kind": "unstructured",
    "version": "1",
    "response": "Resume chain cannot continue",
    "reason": "resume_abandoned",
    "error_detail": "contract drift",
    "fms_results": [],
    "artifact_events": [],
    "usage": {},
    "tools_used": [],
    "warning": "resume_abandoned_code=contract_drift",
  }


def test_background_task_payload_formats_interrupted_and_killed_states() -> None:
  interrupted = _task(
    state=TaskState.INTERRUPTED,
    error={
      "code": "background_completion_persistence_uncertain",
      "message": "completion durability is uncertain",
    },
    metadata={
      "owner_runner_id": "runner-1",
      "owner_role": "writer",
      "sub_agent_id": "sub-1",
      "parent_turn_id": "turn-1",
      "call_index": 2,
      "task_type": "sub_agent",
      "capability_bind": _bind_receipt(model="claude"),
      "resumable": True,
    },
    original_task_id="bg_root",
  )
  killed = _task(
    state=TaskState.KILLED,
    result={
      "kind": "unstructured",
      "version": "1",
      "response": "",
      "reason": "killed",
      "error_detail": "Background task was killed before completion",
      "fms_results": None,
      "artifact_events": None,
      "usage": {},
      "tools_used": [],
      "warning": None,
    },
  )

  interrupted_payload = background_task_payload(
    interrupted,
    elapsed_seconds=12,
    resumed_task_ids=["bg_root_r1"],
    now=120.0,
  )

  assert interrupted_payload["status"] == "interrupted"
  assert interrupted_payload["completed"] is True
  assert interrupted_payload["owner_runner_id"] == "runner-1"
  assert interrupted_payload["resumable"] is True
  assert interrupted_payload["resumed_as"] == ["bg_root_r1"]
  assert interrupted_payload["latest_resume_task_id"] == "bg_root_r1"
  assert interrupted_payload["message"] == "completion durability is uncertain"
  assert interrupted_payload["error"] == {
    "code": "background_completion_persistence_uncertain",
    "message": "completion durability is uncertain",
  }
  assert background_task_payload(killed, elapsed_seconds=6, now=120.0) == {
    "task_id": "bg_1",
    "status": "killed",
    "elapsed_seconds": 6,
    "kind": "unstructured",
    "version": "1",
    "response": "",
    "reason": "killed",
    "error_detail": "Background task was killed before completion",
    "fms_results": None,
    "artifact_events": None,
    "usage": {},
    "tools_used": [],
    "warning": None,
  }


def test_background_task_reminder_text_formats_running_tasks() -> None:
  assert background_task_reminder_text([], elapsed_seconds=lambda _task: 0) == ""

  reminder = background_task_reminder_text(
    [
      _task(task_id="bg_1", agent_name="Analyst", progress=_progress(tool_use_count=1, last_tool_name="lookup")),
      _task(task_id="bg_2", agent_name=None, progress=_progress()),
    ],
    elapsed_seconds=lambda task: 7 if task.task_id == "bg_1" else 3,
  )

  assert reminder == "[Background tasks active: bg_1 (Analyst, running, 7s, 1 tools, last: lookup), bg_2 (running, 3s)]"


def test_runner_background_task_payload_delegate_injects_resumed_ids(monkeypatch) -> None:
  monkeypatch.setattr(gateway_runner.time, "time", lambda: 120.0)
  runner = object.__new__(AgentRunner)
  runner._resumed_task_ids_from_registry = lambda _task_id: ["bg_root_r1"]  # type: ignore[method-assign]
  interrupted = _task(
    task_id="bg_root",
    state=TaskState.INTERRUPTED,
    started_at=100.0,
    metadata={"resumable": True},
  )

  payload = AgentRunner._background_task_payload(runner, interrupted)

  assert payload["elapsed_seconds"] == 20
  assert payload["resumed_as"] == ["bg_root_r1"]
  assert payload["latest_resume_task_id"] == "bg_root_r1"


def test_runner_ensure_sub_agent_semaphore_delegates_and_reuses_existing(monkeypatch) -> None:
  created: list[int] = []

  def fake_semaphore(limit: int) -> dict[str, int]:
    created.append(limit)
    return {"limit": limit}

  monkeypatch.setattr(gateway_runner.asyncio, "Semaphore", fake_semaphore)

  runner = object.__new__(AgentRunner)
  runner._sub_agent_semaphore = None
  runner._max_concurrent_sub_agents = 2

  semaphore = runner._ensure_sub_agent_semaphore()

  assert semaphore is runner._sub_agent_semaphore
  assert semaphore == {"limit": 2}
  assert created == [2]
  assert runner._ensure_sub_agent_semaphore() is semaphore
  assert created == [2]

  runner_without_limit = object.__new__(AgentRunner)
  runner_without_limit._sub_agent_semaphore = None
  runner_without_limit._max_concurrent_sub_agents = None

  assert runner_without_limit._ensure_sub_agent_semaphore() is None


def test_runner_resume_chain_depth_delegates_to_background_helper() -> None:
  tasks = {
    "bg_root": _task(task_id="bg_root"),
    "bg_r1": _task(task_id="bg_r1", original_task_id="bg_root"),
    "bg_r2": _task(task_id="bg_r2", original_task_id="bg_r1"),
  }

  async def lookup(task_id: str) -> SimpleNamespace | None:
    return tasks.get(task_id)

  runner = object.__new__(AgentRunner)
  runner._task_entry_for_chain = lookup  # type: ignore[method-assign]

  assert asyncio.run(runner._resume_chain_depth("bg_r2")) == 2


def test_runner_resume_root_task_id_delegates_to_background_helper() -> None:
  tasks = {
    "bg_root": _task(task_id="bg_root"),
    "bg_r1": _task(task_id="bg_r1", original_task_id="bg_root"),
    "bg_r2": _task(task_id="bg_r2", original_task_id="bg_r1"),
  }

  async def lookup(task_id: str) -> SimpleNamespace | None:
    return tasks.get(task_id)

  runner = object.__new__(AgentRunner)
  runner._task_entry_for_chain = lookup  # type: ignore[method-assign]

  assert asyncio.run(runner._resume_root_task_id("bg_r2")) == "bg_root"


def test_runner_resumed_task_ids_delegates_after_registry_rebuild() -> None:
  class Registry:
    def __init__(self, entries: list[SimpleNamespace]) -> None:
      self._entries = {entry.task_id: entry for entry in entries}

    def get(self, task_id: str) -> SimpleNamespace | None:
      return self._entries.get(task_id)

    def list_tasks(self) -> list[SimpleNamespace]:
      return list(self._entries.values())

  rebuilt: list[bool] = []

  async def rebuild() -> None:
    rebuilt.append(True)

  runner = object.__new__(AgentRunner)
  runner._task_registry = Registry(
    [
      _task(task_id="bg_root"),
      _task(task_id="bg_r1", original_task_id="bg_root"),
      _task(task_id="bg_other"),
      _task(task_id="bg_r2", original_task_id="bg_r1"),
    ]
  )
  runner._rebuild_task_registry_from_log = rebuild  # type: ignore[method-assign]

  async def lookup(task_id: str) -> SimpleNamespace | None:
    return runner._task_registry.get(task_id)

  runner._task_entry_for_chain = lookup  # type: ignore[method-assign]

  assert asyncio.run(runner._resumed_task_ids("bg_root")) == ["bg_r1", "bg_r2"]
  assert rebuilt == [True]


def test_runner_register_background_task_returns_limit_error_before_hooks() -> None:
  class Registry:
    admission_count = 2

  runner = object.__new__(AgentRunner)
  runner._task_registry = Registry()
  runner._max_background_tasks = 2
  started: list[str] = []

  async def handler(_tool_input: dict[str, Any], **_: Any) -> tuple[None, None]:
    started.append("handler")
    return None, None

  result, error = asyncio.run(
    runner._register_background_task(
      tool_input={},
      handler=handler,
      on_before_start=lambda: started.append("hook"),
    )
  )

  assert result is None
  assert error == {
    "code": "max_background_tasks",
    "message": (
      "Background task limit reached (2). "
      "Wait for an existing background task to finish before launching another."
    ),
  }
  assert started == []


def test_runner_register_background_task_returns_resume_depth_error_before_registration() -> None:
  class Registry:
    inflight_count = 0

    def register(self, *_args: Any, **_kwargs: Any) -> None:
      started.append("register")

  started: list[str] = []

  async def depth(task_id: str) -> int:
    started.append(f"depth:{task_id}")
    return 2

  async def root(task_id: str) -> str:
    started.append(f"root:{task_id}")
    return "bg_root"

  async def handler(_tool_input: dict[str, Any], **_: Any) -> tuple[None, None]:
    started.append("handler")
    return None, None

  runner = object.__new__(AgentRunner)
  runner._task_registry = Registry()
  runner._max_background_tasks = 2
  runner._max_resume_chain_depth = 2
  runner._resume_chain_depth = depth  # type: ignore[method-assign]
  runner._resume_root_task_id = root  # type: ignore[method-assign]

  result, error = asyncio.run(
    runner._register_background_task(
      tool_input={},
      handler=handler,
      on_before_start=lambda: started.append("hook"),
      original_task_id="bg_root_r2",
    )
  )

  assert result is None
  assert error == {
    "code": "max_resume_chain_depth",
    "message": "Resume chain depth limit reached (2) for bg_root_r2",
  }
  assert started == ["depth:bg_root_r2"]


def test_runner_resume_registry_helpers_delegate_to_background_helpers() -> None:
  class Registry:
    def __init__(self, entries: list[SimpleNamespace]) -> None:
      self._entries = {entry.task_id: entry for entry in entries}

    def get(self, task_id: str) -> SimpleNamespace | None:
      return self._entries.get(task_id)

    def list_tasks(self) -> list[SimpleNamespace]:
      return list(self._entries.values())

  runner = object.__new__(AgentRunner)
  runner._task_registry = Registry(
    [
      _task(task_id="bg_root"),
      _task(task_id="bg_r1", original_task_id="bg_root"),
      _task(task_id="bg_r2", original_task_id="bg_r1"),
    ]
  )

  assert runner._resume_root_task_id_from_registry("bg_r2") == "bg_root"
  assert runner._resumed_task_ids_from_registry("bg_root") == ["bg_r1", "bg_r2"]


def test_task_correlation_payload_prefers_metadata_and_preserves_original_id() -> None:
  entry = _task(
    task_id="bg_2",
    capability_bind_receipt=_bind_receipt(provider="openai", model="gpt-5.2"),
    original_task_id="bg_root",
    metadata={
      "owner_runner_id": "runner-meta",
      "owner_role": "sub_agent",
      "sub_agent_id": "sub-2:sess",
      "parent_turn_id": "turn-1",
      "call_index": 2,
      "task_type": "resume",
      "capability_bind": _bind_receipt(provider="xai", model="grok-4"),
    },
  )

  assert task_correlation_payload(entry, runner_id="runner-fallback", role="writer") == {
    "task_id": "bg_2",
    "owner_runner_id": "runner-meta",
    "owner_role": "sub_agent",
    "sub_agent_id": "sub-2:sess",
    "parent_turn_id": "turn-1",
    "call_index": 2,
    "task_type": "resume",
    "capability_bind": _bind_receipt(provider="openai", model="gpt-5.2"),
    "original_task_id": "bg_root",
  }


def _workflow_task_metadata(**overrides: Any) -> WorkflowTaskMetadata:
  values: dict[str, Any] = {
    "workflow_run_id": "wfr_123",
    "plan_id": "plan_456",
    "phase_number": 1,
    "revision": 2,
    "node_id": "research-competitors",
    "attempt_number": 3,
    "attempt_id": "attempt-3",
    "operation": AgentOperationRef(
      namespace="agent-operation",
      name="research-competitors",
      version="v1",
      digest="sha256:" + "1" * 64,
    ),
    "admitted_plan_digest": "sha256:" + "2" * 64,
    "admitted_task_digest": "sha256:" + "3" * 64,
    "model_bind_digest": "sha256:" + "4" * 64,
    "capability_binding_digest": "sha256:" + "5" * 64,
    "tool_grant_digest": "sha256:" + "6" * 64,
    "result_requirement": ResultRequirement(
      mode="narrative",
      terminal_narrative="required",
      outcome=OutcomeRequirement(
        required=False,
        source="none",
      ),
    ),
    "item_key": "competitor:ACME",
  }
  values.update(overrides)
  return WorkflowTaskMetadata(**values)


def test_workflow_terminal_delivery_is_coalesced_at_workflow_boundary() -> None:
  workflow_entry = _task(
    metadata={
      WORKFLOW_TASK_METADATA_KEY: _workflow_task_metadata().payload(),
    },
  )
  ordinary_entry = _task(metadata={"task_type": "background"})

  assert workflow_owns_terminal_notification(workflow_entry) is True
  assert workflow_owns_terminal_notification(ordinary_entry) is False


def test_workflow_task_metadata_is_frozen_closed_and_canonical() -> None:
  metadata = _workflow_task_metadata()
  assert metadata.payload() == {
    "workflow_run_id": "wfr_123",
    "plan_id": "plan_456",
    "phase_number": 1,
    "revision": 2,
    "node_id": "research-competitors",
    "attempt_number": 3,
    "attempt_id": "attempt-3",
    "operation": metadata.operation.model_dump(mode="json"),
    "admitted_plan_digest": "sha256:" + "2" * 64,
    "admitted_task_digest": "sha256:" + "3" * 64,
    "model_bind_digest": "sha256:" + "4" * 64,
    "capability_binding_digest": "sha256:" + "5" * 64,
    "tool_grant_digest": "sha256:" + "6" * 64,
    "result_requirement": metadata.result_requirement.model_dump(mode="json"),
    "item_key": "competitor:ACME",
  }
  assert WorkflowTaskMetadata.from_payload(metadata.payload()) == metadata
  assert not hasattr(metadata, "__dict__")
  with pytest.raises(FrozenInstanceError):
    metadata.node_id = "forged"  # type: ignore[misc]
  with pytest.raises(TypeError):
    WorkflowTaskMetadata(  # type: ignore[call-arg]
      **metadata.payload(),
      unexpected="open-metadata",
    )
  with pytest.raises(ValueError, match="exact schema"):
    WorkflowTaskMetadata.from_payload({
      **metadata.payload(),
      "unexpected": "open-metadata",
    })


@pytest.mark.parametrize(
  ("field_name", "value", "message"),
  [
    ("workflow_run_id", " wfr_123", "workflow_run_id"),
    ("plan_id", "", "plan_id"),
    ("node_id", "bad\nnode", "control characters"),
    ("item_key", " ", "item_key"),
    ("phase_number", 0, "positive integer"),
    ("revision", True, "positive integer"),
    ("attempt_number", -1, "positive integer"),
    ("attempt_id", " attempt-3", "attempt_id"),
    ("admitted_task_digest", "sha256:not-a-digest", "canonical sha256"),
  ],
)
def test_workflow_task_metadata_rejects_noncanonical_values(
  field_name: str,
  value: Any,
  message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    _workflow_task_metadata(**{field_name: value})


def test_workflow_task_metadata_is_server_owned_and_correlates_both_events() -> None:
  trusted = _workflow_task_metadata()
  forged = _workflow_task_metadata(
    workflow_run_id="wfr_forged",
    plan_id="plan_forged",
    node_id="forged-node",
  ).payload()
  tool_input = {
    WORKFLOW_TASK_METADATA_KEY: forged,
    "workflow_run_id": "wfr_forged",
    "plan_id": "plan_forged",
    "phase_number": 99,
    "revision": 99,
    "node_id": "forged-node",
    "attempt_number": 99,
    "item_key": "forged:item",
  }
  registration_metadata = background_task_registration_metadata(
    owner_runner_id="runner-1",
    owner_role="writer",
    sub_agent_id="sub3:sess",
    parent_turn_id="turn-1",
    call_index=3,
    capability_bind_receipt=_bind_receipt(),
    cost_observation_threshold_usd=2.0,
    original_task_id=None,
    tool_input=tool_input,
    workflow_task_metadata=trusted,
  )
  assert registration_metadata[WORKFLOW_TASK_METADATA_KEY] == trusted.payload()

  entry = _task(metadata=registration_metadata)
  correlation = task_correlation_payload(
    entry,
    runner_id="runner-fallback",
    role="writer",
  )
  for field_name, expected in trusted.payload().items():
    assert correlation[field_name] == expected

  registered = task_registered_event_payload(
    entry,
    correlation_payload=correlation,
    agent_name="Analyst",
    parent_session_id="sess-parent",
  )
  entry.metadata[WORKFLOW_TASK_METADATA_KEY]["node_id"] = "mutated-later"
  completed = task_completed_event_payload(
    entry,
    TaskState.COMPLETED,
    correlation_payload=correlation,
    completed_at=123.5,
  )
  assert registered["metadata"][WORKFLOW_TASK_METADATA_KEY] == trusted.payload()
  for event in (registered, completed):
    for field_name, expected in trusted.payload().items():
      assert event[field_name] == expected

  untrusted_only = background_task_registration_metadata(
    owner_runner_id="runner-1",
    owner_role="writer",
    sub_agent_id="sub3:sess",
    parent_turn_id="turn-1",
    call_index=3,
    capability_bind_receipt=_bind_receipt(),
    cost_observation_threshold_usd=2.0,
    original_task_id=None,
    tool_input=tool_input,
  )
  assert WORKFLOW_TASK_METADATA_KEY not in untrusted_only
  assert "workflow_run_id" not in task_correlation_payload(
    _task(metadata=untrusted_only),
    runner_id="runner-fallback",
    role="writer",
  )


def test_task_completed_event_payload_extends_correlation_without_mutating_it() -> None:
  entry = _task(
    result={"response": "done"},
    error=None,
  )
  correlation = {
    "task_id": "bg_1",
    "owner_runner_id": "runner-1",
    "owner_role": "writer",
  }

  payload = task_completed_event_payload(
    entry,
    TaskState.COMPLETED,
    correlation_payload=correlation,
    completed_at=123.5,
  )

  assert payload == {
    "task_id": "bg_1",
    "owner_runner_id": "runner-1",
    "owner_role": "writer",
    "type": "task_completed",
    "final_state": "completed",
    "completed_at": 123.5,
    "result": {"response": "done"},
    "error": None,
  }
  assert correlation == {
    "task_id": "bg_1",
    "owner_runner_id": "runner-1",
    "owner_role": "writer",
  }


def test_task_registered_payload_and_started_result_shape_outputs() -> None:
  entry = _task(
    task_id="bg_4",
    started_at=42.5,
    metadata={"sub_agent_id": "sub-4"},
  )
  correlation = {
    "task_id": "bg_4",
    "owner_runner_id": "runner-1",
    "owner_role": "writer",
  }

  payload = task_registered_event_payload(
    entry,
    correlation_payload=correlation,
    agent_name="Analyst",
    parent_session_id="sess-parent",
  )
  entry.metadata["sub_agent_id"] = "mutated"

  assert payload == {
    "type": "task_registered",
    "task_id": "bg_4",
    "owner_runner_id": "runner-1",
    "owner_role": "writer",
    "task_type": "background_agent",
    "agent_name": "Analyst",
    "parent_session_id": "sess-parent",
    "metadata": {"sub_agent_id": "sub-4"},
    "started_at": 42.5,
  }
  assert background_task_started_result(entry, agent_name="Analyst") == {
    "task_id": "bg_4",
    "status": "running",
    "agent": "Analyst",
  }
  assert background_task_started_result(entry, agent_name=None) == {
    "task_id": "bg_4",
    "status": "running",
  }


def test_plan_notification_producer_bounds_cadence_and_coalesces() -> None:
  clock = [100.0]
  queue = NotificationQueue(max_pending=3)
  entry = _task(
    task_id="plan_0",
    task_type="plan_run",
    notification_generation=0,
  )
  producer = PlanNotificationProducer(
    queue=queue,
    entry=entry,
    monotonic=lambda: clock[0],
    wall_clock=lambda: 123.0,
  )

  def snapshot(items_complete: int, *, status: str = "running") -> PlanProgressSnapshot:
    return PlanProgressSnapshot(
      plan_id="plan:ACME",
      phase="research",
      nodes_total=2,
      nodes_complete=0,
      items_total=20,
      items_complete=items_complete,
      current_node="review",
      status=status,
    )

  assert producer.emit_transition(snapshot(0)) is True
  assert producer.emit_item_completion(snapshot(4)) is False
  assert producer.emit_item_completion(snapshot(5)) is True
  assert queue.pending_count == 1
  assert queue.peek()[0].payload == snapshot(5).payload()
  assert queue.peek()[0].notification_generation == 2

  clock[0] += 15.0
  assert producer.emit_item_completion(snapshot(6)) is True
  assert producer.flush(snapshot(6, status="node_complete")) is True
  assert queue.pending_count == 1
  notification = queue.peek()[0]
  assert notification.payload == snapshot(
    6,
    status="node_complete",
  ).payload()
  assert notification.inline_payload()[0] is not None
  assert notification.notification_generation == 4
  assert entry.notification_generation == 4


def test_plan_notification_producer_never_coalesces_approval() -> None:
  queue = NotificationQueue()
  entry = _task(
    task_id="plan_0",
    task_type="plan_run",
    notification_generation=0,
  )
  producer = PlanNotificationProducer(queue=queue, entry=entry)
  snapshot = PlanProgressSnapshot(
    plan_id="plan:ACME",
    phase="approval",
    nodes_total=2,
    nodes_complete=1,
    items_total=5,
    items_complete=5,
    current_node="approval",
    status="approval_pending",
  )

  assert producer.emit_approval_pending(snapshot) is True
  assert producer.emit_approval_pending(snapshot) is True
  assert [item.event for item in queue.peek()] == [
    "plan_approval_pending",
    "plan_approval_pending",
  ]
  assert [item.notification_generation for item in queue.peek()] == [1, 2]


def test_plan_progress_snapshot_rejects_invalid_counts() -> None:
  with pytest.raises(ValueError, match="items_complete"):
    PlanProgressSnapshot(
      plan_id="plan:ACME",
      phase="research",
      nodes_total=1,
      nodes_complete=0,
      items_total=1,
      items_complete=2,
      current_node=None,
      status="running",
    )


def test_runner_task_correlation_delegate_supplies_runner_defaults() -> None:
  runner = object.__new__(AgentRunner)
  runner._runner_id = "runner-default"
  runner._role = "writer"

  assert AgentRunner._task_correlation_payload(runner, _task(task_id="bg_3", metadata={})) == {
    "task_id": "bg_3",
    "owner_runner_id": "runner-default",
    "owner_role": "writer",
    "sub_agent_id": None,
    "parent_turn_id": None,
    "call_index": None,
    "task_type": "background",
    "capability_bind": _bind_receipt(),
  }


def test_background_registration_helpers_parse_observation_threshold_and_call_index() -> None:
  assert parse_cost_observation_threshold_usd("1.25") == 1.25
  assert parse_cost_observation_threshold_usd(0) is None
  assert parse_cost_observation_threshold_usd(-1) is None
  assert parse_cost_observation_threshold_usd(True) is None
  assert parse_cost_observation_threshold_usd("bad") is None
  assert parse_cost_observation_threshold_usd(math.inf) is None

  assert background_task_call_index("bg_3") == 3
  assert background_task_call_index("bg_3_r2") == 3
  assert background_task_call_index("custom") == 0


def test_background_registration_metadata_includes_optional_fields() -> None:
  bind_receipt = _bind_receipt()
  metadata = background_task_registration_metadata(
    owner_runner_id="runner-1",
    owner_role="writer",
    sub_agent_id="sub3:sess",
    parent_turn_id="turn-1",
    call_index=3,
    capability_bind_receipt=bind_receipt,
    cost_observation_threshold_usd=1.25,
    original_task_id="bg_root",
    tool_input={
      "resumable": "yes",
    },
  )

  assert metadata == {
    "owner_runner_id": "runner-1",
    "owner_role": "writer",
    "sub_agent_id": "sub3:sess",
    "parent_turn_id": "turn-1",
    "call_index": 3,
    "task_type": "background",
    "capability_bind": bind_receipt,
    "cost_observation_threshold_usd": 1.25,
    "original_task_id": "bg_root",
    "resumable": True,
  }
  correlation = task_correlation_payload(
    _task(metadata=metadata),
    runner_id="runner-fallback",
    role="writer",
  )
  assert correlation["cost_observation_threshold_usd"] == 1.25


def test_prepare_background_task_registration_mutates_entry_and_returns_call_index() -> None:
  entry = _task(
    task_id="bg_3_r2",
    metadata={"existing": "kept"},
  )

  call_indexes: list[int] = []

  def sub_agent_id(call_index: int) -> str:
    call_indexes.append(call_index)
    return f"sub-{call_index}:sess"

  call_index = prepare_background_task_registration(
    entry,
    tool_input={
      "provider": " anthropic ",
      "model": " claude ",
      "cost_observation_threshold_usd": "1.5",
      "resumable": True,
    },
    capability_bind_receipt=_bind_receipt(),
    owner_runner_id="runner-1",
    owner_role="writer",
    sub_agent_id_for_call_index=sub_agent_id,
    parent_turn_id="turn-1",
    original_task_id="bg_3",
    workflow_task_metadata=_workflow_task_metadata(),
  )

  assert gateway_runner._prepare_background_task_registration is prepare_background_task_registration
  assert call_index == 3
  assert call_indexes == [3]
  assert entry.capability_bind_receipt == _bind_receipt()
  assert entry.metadata == {
    "existing": "kept",
    "owner_runner_id": "runner-1",
    "owner_role": "writer",
    "sub_agent_id": "sub-3:sess",
    "parent_turn_id": "turn-1",
    "call_index": 3,
    "task_type": "background",
    "capability_bind": _bind_receipt(),
    "cost_observation_threshold_usd": 1.5,
    "original_task_id": "bg_3",
    "resumable": True,
    WORKFLOW_TASK_METADATA_KEY: _workflow_task_metadata().payload(),
  }


def test_register_background_task_rejects_untyped_workflow_metadata_before_registration() -> None:
  class Registry:
    admission_count = 0

    def register(self, *_args: Any, **_kwargs: Any) -> None:
      pytest.fail("invalid workflow metadata must reject before registration")

  runner = object.__new__(AgentRunner)
  runner._task_registry = Registry()
  runner._max_background_tasks = 2
  runner._max_resume_chain_depth = 2

  async def handler(_tool_input: dict[str, Any], **_: Any) -> tuple[None, None]:
    pytest.fail("invalid workflow metadata must reject before dispatch")

  with pytest.raises(TypeError, match="exact WorkflowTaskMetadata"):
    asyncio.run(
      runner._register_background_task(
        tool_input={},
        handler=handler,
        workflow_task_metadata=_workflow_task_metadata().payload(),  # type: ignore[arg-type]
      )
    )


def test_runner_cancel_background_task_requests_idempotent_cancellation() -> None:
  class CancellationProbe:
    calls = 0

    def cancel(self) -> None:
      self.calls += 1

  registry = TaskRegistry()
  entry = registry.register("background_agent")
  probe = CancellationProbe()
  entry.asyncio_task = probe  # type: ignore[assignment]
  runner = object.__new__(AgentRunner)
  runner._task_registry = registry

  assert runner.cancel_background_task(entry.task_id) is True
  assert entry.termination_intent == "cancelled"
  assert probe.calls == 1
  assert runner.cancel_background_task(entry.task_id) is False
  assert runner.cancel_background_task("bg_missing") is False
  with pytest.raises(ValueError, match="canonical non-empty"):
    runner.cancel_background_task(" bg_1")
