import asyncio
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_background_lifecycle import RunnerBackgroundLifecycleMixin  # noqa: E402
from agent_gateway.runner_background_tasks import (  # noqa: E402
  BackgroundResultRequest,
  background_asyncio_tasks,
  background_task_call_index,
  background_elapsed_seconds,
  background_task_model,
  background_task_payload,
  background_task_provider_name,
  background_task_ids,
  background_task_ids_for_asyncio_tasks,
  background_task_limit_error,
  background_task_registration_metadata,
  background_task_reminder_text,
  background_result_task,
  background_result_tasks,
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
  parse_child_budget_usd,
  prepare_background_task_registration,
  resume_chain_depth,
  resume_root_task_id,
  resume_root_task_id_from_registry,
  resume_task_id_override,
  resumed_task_ids,
  resumed_task_ids_from_registry,
  sub_agent_result_from_log_entries,
  task_completed_event_payload,
  task_correlation_payload,
  task_registered_event_payload,
  wait_for_background_tasks,
)
from agent_gateway.task_registry import TaskState  # noqa: E402


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
    "provider_name": "stub",
    "model": "stub-model",
    "original_task_id": None,
    "asyncio_task": None,
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
    "_inject_system_prompt_reminder",
    "_run_background_agent",
    "_task_correlation_payload",
    "_append_task_completed_event",
    "_register_background_task",
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


def test_runner_background_registration_resolves_parent_module_helpers(monkeypatch: Any) -> None:
  transitions: list[tuple[str, TaskState]] = []

  class Registry:
    inflight_count = 0

    def register(self, *_args: Any, **kwargs: Any) -> SimpleNamespace:
      return _task(task_id=kwargs.get("task_id") or "bg_5", metadata={})

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
    entry.metadata["patched_sub_agent_id"] = kwargs["sub_agent_id_for_call_index"](5)
    entry.provider_name = kwargs["default_provider_name"]
    entry.model = kwargs["auth_model"]
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

  result, error = asyncio.run(
    AgentRunner._register_background_task(
      runner,
      tool_input={},
      handler=handler,
      agent_name="Analyst",
    )
  )

  assert error is None
  assert result == {"patched": "bg_5", "agent": "Analyst"}
  assert durable_events == [{"type": "patched_registered", "sub": "patched-sess-5"}]
  assert created_tasks[0].name == "bg_5"
  assert transitions == [("bg_5", TaskState.RUNNING)]


def test_runner_shutdown_background_tasks_resolves_parent_module_helpers(monkeypatch: Any) -> None:
  calls: list[Any] = []
  pending = [object()]

  class Registry:
    def list_tasks(self, *, state: TaskState) -> list[SimpleNamespace]:
      calls.append(("list", state))
      return [_task(task_id="bg_1")]

    def kill(self, task_id: str) -> None:
      calls.append(("kill", task_id))

  runner = object.__new__(AgentRunner)
  runner._task_registry = Registry()

  async def _drain_cancelled_background_tasks(running_entries: Any, selected_pending: Any, **kwargs: Any) -> None:
    calls.append(("drain_cancelled", list(running_entries)[0].task_id, selected_pending is pending, kwargs["timeout"]))

  monkeypatch.setattr(gateway_runner, "_background_asyncio_tasks", lambda _entries: pending)
  monkeypatch.setattr(gateway_runner, "_drain_cancelled_background_tasks", _drain_cancelled_background_tasks)
  monkeypatch.setattr(gateway_runner, "asyncio", SimpleNamespace(gather=object(), wait_for=object(), wait=object(), TimeoutError=asyncio.TimeoutError))

  asyncio.run(AgentRunner._shutdown_background_tasks(runner, was_cancelled=True))

  assert calls == [
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
  request, error = parse_background_result_request({"task_id": " bg_1 ", "wait": True, "timeout": 300})

  assert error is None
  assert request == BackgroundResultRequest(task_id="bg_1", wait=True, timeout=120.0)

  request, error = parse_background_result_request({"task_id": "*"})

  assert error is None
  assert request == BackgroundResultRequest(task_id="*", wait=False, timeout=60.0)


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
  assert gateway_runner._background_wait_tasks is background_wait_tasks
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
  gather_result = object()
  entries = [
    _task(task_id="bg_1", asyncio_task=task_a),
    _task(task_id="bg_2", asyncio_task=task_b),
  ]
  events: list[str] = []

  def kill_task(task_id: str) -> None:
    events.append(f"kill:{task_id}")

  def gather_fn(*pending: object, return_exceptions: bool) -> object:
    assert return_exceptions is True
    assert pending == (task_a, task_b)
    events.append("gather")
    return gather_result

  async def wait_for_fn(awaitable: object, *, timeout: float) -> None:
    assert awaitable is gather_result
    events.append(f"wait:{timeout}")

  asyncio.run(
    drain_cancelled_background_tasks(
      entries,
      [task_a, task_b],
      kill_task=kill_task,
      gather_fn=gather_fn,
      wait_for_fn=wait_for_fn,
      timeout=5.0,
    )
  )

  assert gateway_runner._drain_cancelled_background_tasks is drain_cancelled_background_tasks
  assert events == ["kill:bg_1", "kill:bg_2", "gather", "wait:5.0"]


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

  def gather_fn(*pending: object, return_exceptions: bool) -> object:
    events.append("gather")
    return object()

  async def wait_for_fn(awaitable: object, *, timeout: float) -> None:
    events.append(f"drain:{timeout}")

  asyncio.run(
    drain_still_pending_background_tasks(
      entries,
      [task_a, task_b],
      kill_task=kill_task,
      wait_fn=wait_fn,
      gather_fn=gather_fn,
      wait_for_fn=wait_for_fn,
      wait_timeout=30.0,
      drain_timeout=5.0,
    )
  )

  assert gateway_runner._drain_still_pending_background_tasks is drain_still_pending_background_tasks
  assert events == ["wait:30.0"]


def test_drain_still_pending_background_tasks_kills_then_drains_selected_pending() -> None:
  task_a = object()
  task_b = object()
  task_c = object()
  gather_result = object()
  entries = [
    _task(task_id="bg_1", asyncio_task=task_a),
    _task(task_id="bg_2", asyncio_task=task_b),
    _task(task_id="bg_3", asyncio_task=task_c),
  ]
  still_pending = [task_c, task_a]
  events: list[str] = []

  def kill_task(task_id: str) -> None:
    events.append(f"kill:{task_id}")

  async def wait_fn(pending: list[object], *, timeout: float) -> tuple[list[object], list[object]]:
    assert pending == [task_a, task_b, task_c]
    events.append(f"wait:{timeout}")
    return [task_b], still_pending

  def gather_fn(*pending: object, return_exceptions: bool) -> object:
    assert return_exceptions is True
    assert pending == (task_c, task_a)
    events.append("gather")
    return gather_result

  async def wait_for_fn(awaitable: object, *, timeout: float) -> None:
    assert awaitable is gather_result
    events.append(f"drain:{timeout}")

  asyncio.run(
    drain_still_pending_background_tasks(
      entries,
      [task_a, task_b, task_c],
      kill_task=kill_task,
      wait_fn=wait_fn,
      gather_fn=gather_fn,
      wait_for_fn=wait_for_fn,
      wait_timeout=30.0,
      drain_timeout=5.0,
    )
  )

  assert gateway_runner._drain_still_pending_background_tasks is drain_still_pending_background_tasks
  assert events == ["wait:30.0", "kill:bg_1", "kill:bg_3", "gather", "drain:5.0"]


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
  assert gateway_runner._wait_for_background_tasks is wait_for_background_tasks
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

  assert gateway_runner._background_result_task is background_result_task
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


def test_background_result_tasks_lists_waits_then_builds_payloads() -> None:
  task_a = _task(task_id="bg_1", asyncio_task=object())
  task_b = _task(task_id="bg_2", asyncio_task=object(), completed=True)
  events: list[str] = []

  def entries() -> list[SimpleNamespace]:
    events.append("list")
    return [task_a, task_b]

  async def wait_fn(pending: list[Any], *, timeout: float) -> None:
    events.append(f"wait:{len(pending)}:{timeout}")
    assert pending == [task_a.asyncio_task]

  def payload(bg_task: SimpleNamespace) -> dict[str, str]:
    events.append(f"payload:{bg_task.task_id}")
    return {"task_id": bg_task.task_id}

  result = asyncio.run(
    background_result_tasks(
      task_entries=entries,
      wait=True,
      timeout=4.5,
      wait_fn=wait_fn,
      payload=payload,
    )
  )

  assert gateway_runner._background_result_tasks is background_result_tasks
  assert result == {"tasks": [{"task_id": "bg_1"}, {"task_id": "bg_2"}]}
  assert events == ["list", "wait:1:4.5", "payload:bg_1", "payload:bg_2"]


def test_background_result_tasks_skips_wait_but_preserves_payloads() -> None:
  task = _task(task_id="bg_1", asyncio_task=object())
  wait_calls: list[list[Any]] = []

  async def wait_fn(pending: list[Any], *, timeout: float) -> None:
    wait_calls.append(pending)

  result = asyncio.run(
    background_result_tasks(
      task_entries=lambda: [task],
      wait=False,
      timeout=1.0,
      wait_fn=wait_fn,
      payload=lambda bg_task: {"task_id": bg_task.task_id},
    )
  )

  assert result == {"tasks": [{"task_id": "bg_1"}]}
  assert wait_calls == []


def test_background_task_limit_error_preserves_legacy_message() -> None:
  assert background_task_limit_error(inflight_count=1, max_background_tasks=2) is None
  assert background_task_limit_error(inflight_count=2, max_background_tasks=2) == {
    "code": "max_background_tasks",
    "message": (
      "Background task limit reached (2). "
      "Wait for an existing background task to finish before launching another."
    ),
  }
  assert background_task_limit_error(inflight_count=3, max_background_tasks=2)["code"] == "max_background_tasks"


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

  assert gateway_runner._call_before_background_task_start_hook is call_before_background_task_start_hook
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
    "error": {"code": "boom"},
  }


def test_background_task_payload_formats_interrupted_and_killed_states() -> None:
  interrupted = _task(
    state=TaskState.INTERRUPTED,
    metadata={
      "owner_runner_id": "runner-1",
      "owner_role": "writer",
      "sub_agent_id": "sub-1",
      "parent_turn_id": "turn-1",
      "call_index": 2,
      "task_type": "sub_agent",
      "provider_name": "anthropic",
      "model": "claude",
      "resumable": True,
    },
    original_task_id="bg_root",
  )
  killed = _task(state=TaskState.KILLED)

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
  assert interrupted_payload["message"] == "Background task was interrupted by a gateway restart before completion."
  assert background_task_payload(killed, elapsed_seconds=6, now=120.0) == {
    "task_id": "bg_1",
    "status": "killed",
    "elapsed_seconds": 6,
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


def test_sub_agent_result_from_log_entries_summarizes_events_and_warnings() -> None:
  result = sub_agent_result_from_log_entries(
    [
      _entry({"type": "text_delta", "text": "discarded"}),
      _entry({"type": "tool_call_start", "tool_name": "discarded_tool"}),
      _entry({"type": "stream_retry"}),
      _entry({"type": "text_delta", "text": " Alpha"}),
      _entry({"type": "text_delta", "text": " Beta "}),
      _entry({"type": "tool_call_start", "tool_name": "lookup"}),
      _entry({"type": "stream_complete", "usage": {"input_tokens": 2}}),
      _entry({"type": "error", "error": "late failure"}),
      _entry({"type": "budget_exceeded"}),
      _entry({"type": "max_turns_reached"}),
    ],
    timed_out=False,
    timeout=30.0,
    budget_exceeded_reason="parent_budget",
    original_task_id="bg_root",
  )

  assert result == {
    "response": "Alpha Beta",
    "tools_used": ["lookup"],
    "usage": {"input_tokens": 2},
    "original_task_id": "bg_root",
    "warning": (
      "Sub-agent error: late failure; "
      "Sub-agent stopped: budget limit reached (parent budget); "
      "Sub-agent stopped: max turns reached — partial results"
    ),
  }


def test_sub_agent_result_from_log_entries_timeout_takes_error_precedence() -> None:
  result = sub_agent_result_from_log_entries(
    [
      _entry({"type": "text_delta", "text": "Partial output"}),
      _entry({"type": "error", "error": "stalled stream"}),
      _entry({"type": "budget_exceeded", "reason": "child_budget"}),
    ],
    timed_out=True,
    timeout=0.01,
    budget_exceeded_reason="parent_budget",
  )

  assert result == {
    "response": "Partial output",
    "tools_used": [],
    "usage": {},
    "warning": (
      "Sub-agent timed out after 0.01s — partial results returned; "
      "Sub-agent stopped: budget limit reached (child budget)"
    ),
  }


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

  assert gateway_runner._ensure_sub_agent_semaphore is ensure_sub_agent_semaphore

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

  assert gateway_runner._resume_chain_depth is resume_chain_depth
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

  assert gateway_runner._resume_root_task_id is resume_root_task_id
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

  assert gateway_runner._resumed_task_ids is resumed_task_ids
  assert asyncio.run(runner._resumed_task_ids("bg_root")) == ["bg_r1", "bg_r2"]
  assert rebuilt == [True]


def test_runner_register_background_task_returns_limit_error_before_hooks() -> None:
  class Registry:
    inflight_count = 2

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

  assert gateway_runner._background_task_limit_error is background_task_limit_error
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

  assert gateway_runner._resume_task_id_override is resume_task_id_override
  assert result is None
  assert error == {
    "code": "max_resume_chain_depth",
    "message": "Resume chain depth limit reached (2) for bg_root_r2",
  }
  assert started == ["hook", "depth:bg_root_r2"]


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

  assert gateway_runner._resume_root_task_id_from_registry is resume_root_task_id_from_registry
  assert gateway_runner._resumed_task_ids_from_registry is resumed_task_ids_from_registry
  assert runner._resume_root_task_id_from_registry("bg_r2") == "bg_root"
  assert runner._resumed_task_ids_from_registry("bg_root") == ["bg_r1", "bg_r2"]


def test_task_correlation_payload_prefers_metadata_and_preserves_original_id() -> None:
  entry = _task(
    task_id="bg_2",
    provider_name="entry-provider",
    model="entry-model",
    original_task_id="bg_root",
    metadata={
      "owner_runner_id": "runner-meta",
      "owner_role": "sub_agent",
      "sub_agent_id": "sub-2:sess",
      "parent_turn_id": "turn-1",
      "call_index": 2,
      "task_type": "resume",
      "provider_name": "meta-provider",
      "model": "meta-model",
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
    "provider_name": "meta-provider",
    "model": "meta-model",
    "original_task_id": "bg_root",
  }


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
    "provider_name": "stub",
    "model": "stub-model",
  }


def test_background_registration_helpers_parse_provider_model_budget_and_call_index() -> None:
  assert background_task_provider_name({"provider_name": " anthropic "}, default_provider_name="stub") == "anthropic"
  assert background_task_provider_name({"provider": " openai "}, default_provider_name="stub") == "openai"
  assert background_task_provider_name({"provider": "   "}, default_provider_name="stub") == "stub"

  assert background_task_model({"model": " claude "}, auth_model="auth-model") == "claude"
  assert background_task_model({"model": "   "}, auth_model=" auth-model ") == "auth-model"
  assert background_task_model({}, auth_model=None) is None

  assert parse_child_budget_usd("1.25") == 1.25
  assert parse_child_budget_usd(0) is None
  assert parse_child_budget_usd(-1) is None
  assert parse_child_budget_usd(True) is None
  assert parse_child_budget_usd("bad") is None

  assert background_task_call_index("bg_3") == 3
  assert background_task_call_index("bg_3_r2") == 3
  assert background_task_call_index("custom") == 0


def test_background_registration_metadata_includes_optional_fields() -> None:
  metadata = background_task_registration_metadata(
    owner_runner_id="runner-1",
    owner_role="writer",
    sub_agent_id="sub3:sess",
    parent_turn_id="turn-1",
    call_index=3,
    provider_name="anthropic",
    model="claude",
    child_budget_usd=1.25,
    original_task_id="bg_root",
    tool_input={"resumable": "yes"},
  )

  assert metadata == {
    "owner_runner_id": "runner-1",
    "owner_role": "writer",
    "sub_agent_id": "sub3:sess",
    "parent_turn_id": "turn-1",
    "call_index": 3,
    "task_type": "background",
    "provider_name": "anthropic",
    "model": "claude",
    "child_budget_usd": 1.25,
    "original_task_id": "bg_root",
    "resumable": True,
  }


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
      "child_budget_usd": "1.5",
      "resumable": True,
    },
    default_provider_name="fallback-provider",
    auth_model="fallback-model",
    owner_runner_id="runner-1",
    owner_role="writer",
    sub_agent_id_for_call_index=sub_agent_id,
    parent_turn_id="turn-1",
    original_task_id="bg_3",
  )

  assert gateway_runner._prepare_background_task_registration is prepare_background_task_registration
  assert call_index == 3
  assert call_indexes == [3]
  assert entry.provider_name == "anthropic"
  assert entry.model == "claude"
  assert entry.metadata == {
    "existing": "kept",
    "owner_runner_id": "runner-1",
    "owner_role": "writer",
    "sub_agent_id": "sub-3:sess",
    "parent_turn_id": "turn-1",
    "call_index": 3,
    "task_type": "background",
    "provider_name": "anthropic",
    "model": "claude",
    "child_budget_usd": 1.5,
    "original_task_id": "bg_3",
    "resumable": True,
  }
