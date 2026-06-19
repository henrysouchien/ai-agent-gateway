from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from .runner_background_tasks import (
  background_asyncio_tasks as _background_asyncio_tasks,
  background_elapsed_seconds as _background_elapsed_seconds,
  background_task_limit_error as _background_task_limit_error,
  background_task_payload as _background_task_payload,
  background_task_reminder_text as _background_task_reminder_text,
  background_result_task as _background_result_task,
  background_result_tasks as _background_result_tasks,
  background_task_started_result as _background_task_started_result,
  background_timeout_value as _background_timeout_value,
  call_before_background_task_start_hook as _call_before_background_task_start_hook,
  drain_cancelled_background_tasks as _drain_cancelled_background_tasks,
  drain_still_pending_background_tasks as _drain_still_pending_background_tasks,
  entry_aware_background_handler as _entry_aware_background_handler,
  ensure_sub_agent_semaphore as _ensure_sub_agent_semaphore,
  parse_background_result_request as _parse_background_result_request,
  prepare_background_task_registration as _prepare_background_task_registration,
  resume_chain_depth as _resume_chain_depth,
  resume_root_task_id as _resume_root_task_id,
  resume_root_task_id_from_registry as _resume_root_task_id_from_registry,
  resume_task_id_override as _resume_task_id_override,
  resumed_task_ids as _resumed_task_ids,
  resumed_task_ids_from_registry as _resumed_task_ids_from_registry,
  task_completed_event_payload as _task_completed_event_payload,
  task_correlation_payload as _task_correlation_payload,
  task_registered_event_payload as _task_registered_event_payload,
  wait_for_background_tasks as _wait_for_background_tasks,
)
from .runner_introspection import derive_sub_agent_id as _derive_sub_agent_id
from .runner_notifications import (
  build_notification_reminder as _build_notification_reminder,
  consume_notifications as _consume_notifications,
  inject_system_prompt_reminder as _inject_system_prompt_reminder,
)
from .runner_session_lifecycle import _runner_attr
from .runner_state import BackgroundTask
from .task_registry import TaskEntry, TaskState


BackgroundTaskHandler = Callable[..., Awaitable[Tuple[Optional[Any], Optional[Dict[str, Any]]]]]
BackgroundTaskCallback = Callable[[BackgroundTask | TaskEntry], Awaitable[None] | None]

log = logging.getLogger("agent_gateway.runner")
_MAX_NOTIFICATIONS_PER_TURN = 5


def _runner_module_attr(name: str, fallback: Any) -> Any:
  module = sys.modules.get("agent_gateway.runner")
  if module is None:
    return fallback
  return getattr(module, name, fallback)


class RunnerBackgroundLifecycleMixin:
  def _ensure_sub_agent_semaphore(self) -> asyncio.Semaphore | None:
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    self._sub_agent_semaphore = _runner_attr(self, "_ensure_sub_agent_semaphore", _ensure_sub_agent_semaphore)(
      self._sub_agent_semaphore,
      self._max_concurrent_sub_agents,
      semaphore_factory=asyncio_module.Semaphore,
    )
    return self._sub_agent_semaphore

  @staticmethod
  def _background_timeout_value(raw_timeout: Any) -> float:
    return _runner_module_attr("_background_timeout_value", _background_timeout_value)(raw_timeout)

  def _background_elapsed_seconds(self, bg_task: BackgroundTask | TaskEntry) -> int:
    return _runner_attr(self, "_background_elapsed_seconds", _background_elapsed_seconds)(
      bg_task,
      now=_runner_attr(self, "time", time).time(),
    )

  async def _task_entry_for_chain(self, task_id: str) -> TaskEntry | None:
    entry = self._task_registry.get(task_id)
    if entry is not None:
      return entry
    return await self._lookup_task_in_log(task_id)

  async def _resume_chain_depth(self, task_id: str) -> int:
    return await _runner_attr(self, "_resume_chain_depth", _resume_chain_depth)(
      task_id,
      task_lookup=self._task_entry_for_chain,
    )

  async def _resume_root_task_id(self, task_id: str) -> str:
    return await _runner_attr(self, "_resume_root_task_id", _resume_root_task_id)(
      task_id,
      task_lookup=self._task_entry_for_chain,
    )

  async def _resumed_task_ids(self, task_id: str) -> list[str]:
    await self._rebuild_task_registry_from_log()
    return await _runner_attr(self, "_resumed_task_ids", _resumed_task_ids)(
      task_id,
      task_entries=self._task_registry.list_tasks,
      resume_root=self._resume_root_task_id,
    )

  def _resume_root_task_id_from_registry(self, task_id: str) -> str:
    return _runner_attr(self, "_resume_root_task_id_from_registry", _resume_root_task_id_from_registry)(
      task_id,
      task_lookup=self._task_registry.get,
    )

  def _resumed_task_ids_from_registry(self, task_id: str) -> list[str]:
    return _runner_attr(self, "_resumed_task_ids_from_registry", _resumed_task_ids_from_registry)(
      task_id,
      task_entries=self._task_registry.list_tasks(),
      task_lookup=self._task_registry.get,
    )

  def _background_task_payload(self, bg_task: BackgroundTask | TaskEntry) -> Dict[str, Any]:
    now = _runner_attr(self, "time", time).time()
    resumed_task_ids = (
      self._resumed_task_ids_from_registry(bg_task.task_id)
      if getattr(bg_task, "state", None) == TaskState.INTERRUPTED
      else None
    )
    return _runner_attr(self, "_background_task_payload", _background_task_payload)(
      bg_task,
      elapsed_seconds=_runner_attr(self, "_background_elapsed_seconds", _background_elapsed_seconds)(bg_task, now=now),
      resumed_task_ids=resumed_task_ids,
      now=now,
    )

  def _background_task_reminder_text(self) -> str:
    return _runner_attr(self, "_background_task_reminder_text", _background_task_reminder_text)(
      self._task_registry.list_tasks(state=TaskState.RUNNING),
      elapsed_seconds=self._background_elapsed_seconds,
    )

  def _build_notification_reminder(self) -> str:
    return _runner_attr(self, "_build_notification_reminder", _build_notification_reminder)(
      self._notification_queue,
      max_count=_runner_attr(self, "_MAX_NOTIFICATIONS_PER_TURN", _MAX_NOTIFICATIONS_PER_TURN),
    )

  def _consume_notifications(self, max_count: int) -> int:
    return _runner_attr(self, "_consume_notifications", _consume_notifications)(
      self._notification_queue,
      max_count=max_count,
    )

  @staticmethod
  def _inject_system_prompt_reminder(
    system_prompt: Optional[Union[str, List[Tuple[str, bool]]]],
    reminder: str,
  ) -> Optional[Union[str, List[Tuple[str, bool]]]]:
    return _runner_module_attr("_inject_system_prompt_reminder", _inject_system_prompt_reminder)(system_prompt, reminder)

  async def _run_background_agent(
    self,
    bg_task: TaskEntry,
    handler: BackgroundTaskHandler,
    tool_input: Dict[str, Any],
    call_index: int,
    on_complete: BackgroundTaskCallback | None = None,
  ) -> None:
    try:
      asyncio_module = _runner_attr(self, "asyncio", asyncio)
      semaphore = self._ensure_sub_agent_semaphore()
      if semaphore is not None:
        async with semaphore:
          result, error = await handler(tool_input, call_index=call_index)
      else:
        result, error = await handler(tool_input, call_index=call_index)
      bg_task.result = result if isinstance(result, dict) else ({"result": result} if result is not None else None)
      bg_task.error = error
      if bg_task.error is not None:
        await self._append_task_completed_event(bg_task, TaskState.FAILED)
        self._task_registry.transition(bg_task.task_id, TaskState.FAILED, error=bg_task.error)
      else:
        await self._append_task_completed_event(bg_task, TaskState.COMPLETED)
        self._task_registry.transition(bg_task.task_id, TaskState.COMPLETED, result=bg_task.result)
    except asyncio.CancelledError:
      bg_task.error = {"code": "cancelled", "message": "Background task was cancelled"}
      await self._append_task_completed_event(bg_task, TaskState.FAILED)
      self._task_registry.transition(bg_task.task_id, TaskState.FAILED, error=bg_task.error)
    except Exception as exc:
      bg_task.error = {"code": "background_error", "message": str(exc)}
      await self._append_task_completed_event(bg_task, TaskState.FAILED)
      self._task_registry.transition(bg_task.task_id, TaskState.FAILED, error=bg_task.error)
    finally:
      if on_complete is not None:
        try:
          maybe_awaitable = on_complete(bg_task)
          if asyncio_module.iscoroutine(maybe_awaitable):
            await maybe_awaitable
        except Exception:
          pass

  def _task_correlation_payload(self, entry: TaskEntry) -> Dict[str, Any]:
    return _runner_attr(self, "_task_correlation_payload", _task_correlation_payload)(
      entry,
      runner_id=self._runner_id,
      role=self._role,
    )

  async def _append_task_completed_event(self, entry: TaskEntry, final_state: TaskState) -> None:
    await self._append_durable_event(
      _runner_attr(self, "_task_completed_event_payload", _task_completed_event_payload)(
        entry,
        final_state,
        correlation_payload=self._task_correlation_payload(entry),
        completed_at=_runner_attr(self, "time", time).time(),
      )
    )

  async def _register_background_task(
    self,
    *,
    tool_input: Dict[str, Any],
    handler: BackgroundTaskHandler,
    agent_name: str | None = None,
    parent_turn_id: str | None = None,
    on_complete: BackgroundTaskCallback | None = None,
    on_before_start: Callable[[], None] | None = None,
    original_task_id: str | None = None,
  ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    limit_error = _runner_attr(self, "_background_task_limit_error", _background_task_limit_error)(
      inflight_count=self._task_registry.inflight_count,
      max_background_tasks=self._max_background_tasks,
    )
    if limit_error is not None:
      return None, limit_error

    _runner_attr(self, "_call_before_background_task_start_hook", _call_before_background_task_start_hook)(
      on_before_start,
      log_session_id=lambda: self._sid,
      logger=log,
    )

    task_id_override, resume_error = await _runner_attr(self, "_resume_task_id_override", _resume_task_id_override)(
      original_task_id,
      max_resume_chain_depth=self._max_resume_chain_depth,
      resume_chain_depth=self._resume_chain_depth,
      resume_root=self._resume_root_task_id,
    )
    if resume_error is not None:
      return None, resume_error

    entry = self._task_registry.register(
      "background_agent",
      agent_name=agent_name,
      task_id=task_id_override,
      original_task_id=original_task_id,
    )
    call_index = _runner_attr(
      self,
      "_prepare_background_task_registration",
      _prepare_background_task_registration,
    )(
      entry,
      tool_input=tool_input,
      default_provider_name=getattr(self._provider, "name", None),
      auth_model=self._auth_config.get("model"),
      owner_runner_id=self._runner_id,
      owner_role=self._role,
      sub_agent_id_for_call_index=lambda call_index: _runner_attr(
        self,
        "_derive_sub_agent_id",
        _derive_sub_agent_id,
      )(self._gateway_session_id, call_index),
      parent_turn_id=parent_turn_id if parent_turn_id is not None else self._parent_turn_id,
      original_task_id=original_task_id,
    )
    await self._append_durable_event(
      _runner_attr(self, "_task_registered_event_payload", _task_registered_event_payload)(
        entry,
        correlation_payload=self._task_correlation_payload(entry),
        agent_name=agent_name,
        parent_session_id=self._gateway_session_id,
      )
    )

    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    entry.asyncio_task = asyncio_module.create_task(
      self._run_background_agent(
        entry,
        _runner_attr(self, "_entry_aware_background_handler", _entry_aware_background_handler)(handler, entry),
        dict(tool_input),
        call_index,
        on_complete=on_complete,
      ),
      name=entry.task_id,
    )
    self._task_registry.transition(entry.task_id, TaskState.RUNNING)

    return _runner_attr(self, "_background_task_started_result", _background_task_started_result)(
      entry,
      agent_name=agent_name,
    ), None

  async def get_background_result(
    self,
    tool_input: Dict[str, Any],
    **_: Any,
  ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    request, error = _runner_attr(self, "_parse_background_result_request", _parse_background_result_request)(tool_input)
    if error is not None or request is None:
      return None, error
    task_id = request.task_id
    wait = request.wait
    timeout = request.timeout

    await self._rebuild_task_registry_from_log()

    if task_id == "*":
      return await _runner_attr(self, "_background_result_tasks", _background_result_tasks)(
        task_entries=self._task_registry.list_tasks,
        wait=wait,
        timeout=timeout,
        wait_fn=_runner_attr(self, "asyncio", asyncio).wait,
        payload=self._background_task_payload,
      ), None

    bg_task, error = await _runner_attr(self, "_background_result_task", _background_result_task)(
      task_id,
      registry_lookup=self._task_registry.get,
      log_lookup=self._lookup_task_in_log,
    )
    if error is not None or bg_task is None:
      return None, error

    await _runner_attr(self, "_wait_for_background_tasks", _wait_for_background_tasks)(
      [bg_task],
      wait=wait,
      timeout=timeout,
      wait_fn=_runner_attr(self, "asyncio", asyncio).wait,
    )
    return self._background_task_payload(bg_task), None

  async def _shutdown_background_tasks(self, was_cancelled: bool) -> None:
    running_entries = self._task_registry.list_tasks(state=TaskState.RUNNING)
    pending = _runner_attr(self, "_background_asyncio_tasks", _background_asyncio_tasks)(running_entries)
    if not pending:
      return

    try:
      asyncio_module = _runner_attr(self, "asyncio", asyncio)
      if was_cancelled:
        await _runner_attr(self, "_drain_cancelled_background_tasks", _drain_cancelled_background_tasks)(
          running_entries,
          pending,
          kill_task=self._task_registry.kill,
          gather_fn=asyncio_module.gather,
          wait_for_fn=asyncio_module.wait_for,
          timeout=5.0,
        )
        return

      await _runner_attr(self, "_drain_still_pending_background_tasks", _drain_still_pending_background_tasks)(
        running_entries,
        pending,
        kill_task=self._task_registry.kill,
        wait_fn=asyncio_module.wait,
        gather_fn=asyncio_module.gather,
        wait_for_fn=asyncio_module.wait_for,
        wait_timeout=30.0,
        drain_timeout=5.0,
      )
    except _runner_attr(self, "asyncio", asyncio).TimeoutError:
      pass
