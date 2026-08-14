from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import secrets
import sys
import time
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, Union

from agent_workflow_contracts import (
  AdmittedTask,
  ContentHandle,
  ContentReadGrant,
  ParentResultPolicy,
  TaskResult,
  UsageObservation,
  sha256_digest,
  terminal_task_result,
)

from .runner_background_tasks import (
  _BACKGROUND_RESULT_ACK_RESULT_KEY,
  REQUIRED_SKILL_LIFECYCLE_METADATA_KEY,
  background_asyncio_tasks as _background_asyncio_tasks,
  background_elapsed_seconds as _background_elapsed_seconds,
  background_task_limit_error as _background_task_limit_error,
  background_task_payload as _background_task_payload,
  background_task_reminder_text as _background_task_reminder_text,
  background_result_task as _background_result_task,
  background_task_started_result as _background_task_started_result,
  background_timeout_value as _background_timeout_value,
  call_before_background_task_start_hook as _call_before_background_task_start_hook,
  drain_cancelled_background_tasks as _drain_cancelled_background_tasks,
  drain_still_pending_background_tasks as _drain_still_pending_background_tasks,
  entry_aware_background_handler as _entry_aware_background_handler,
  ensure_sub_agent_semaphore as _ensure_sub_agent_semaphore,
  agent_completion_message_id as _agent_completion_message_id,
  build_agent_completion_envelope as _build_agent_completion_envelope,
  normalize_required_skill_lifecycle as _normalize_required_skill_lifecycle,
  normalize_workflow_task_metadata as _normalize_workflow_task_metadata,
  ordinary_parent_result_policy as _ordinary_parent_result_policy,
  parse_background_result_request as _parse_background_result_request,
  prepare_background_task_registration as _prepare_background_task_registration,
  require_capability_bind_receipt as _require_capability_bind_receipt,
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
  WorkflowTaskMetadata,
  workflow_owns_terminal_notification as _workflow_owns_terminal_notification,
)
from .runner_introspection import (
  derive_sub_agent_id as _derive_sub_agent_id,
  mark_exception_traceback_logged,
)
from .runner_notifications import (
  build_notification_reminder as _build_notification_reminder,
  consume_notifications as _consume_notifications,
  inject_system_prompt_reminder as _inject_system_prompt_reminder,
)
from .runner_run_loop_defaults import MAX_NOTIFICATIONS_PER_TURN as _MAX_NOTIFICATIONS_PER_TURN
from .runner_session_events import build_agent_completion_event as _build_agent_completion_event
from .runner_session_lifecycle import _exact_value_match, _runner_attr
from .events import AgentCompletionEvent, event_from_dict
from .runner_state import BackgroundTask
from .skill_result_events import build_skill_result_captured_event
from .skill_lifecycle import TopLevelSkillLifecycleMetadata
from .sub_agent_narrative_result import (
  read_task_result_terminal_narrative,
)
from .tool_result_compaction import model_tool_result_max_chars
from .task_registry import (
  RequiredSkillResultMarkerValidationError,
  ResumeSuccessorConflictError,
  TaskEntry,
  TaskState,
  TerminationIntent,
  resolve_required_skill_result_marker,
  task_state_for_result,
)


BackgroundTaskHandler = Callable[..., Awaitable[Tuple[Optional[Any], Optional[Dict[str, Any]]]]]
BackgroundTaskCallback = Callable[[BackgroundTask | TaskEntry], Awaitable[None] | None]
RequiredSkillResultEventFactory = Callable[
  [
    TaskEntry,
    dict[str, Any] | None,
    dict[str, Any] | None,
  ],
  dict[str, Any],
]
RequiredSkillResultProjector = Callable[
  [dict[str, Any]],
  Any,
]

log = logging.getLogger("agent_gateway.runner")

_BACKGROUND_COMPLETION_PERSIST_TIMEOUT_SECONDS = 5.0
_BACKGROUND_CANCEL_DRAIN_TIMEOUT_SECONDS = 5.0
_BACKGROUND_GRACE_WAIT_TIMEOUT_SECONDS = 30.0
_BACKGROUND_KILL_DRAIN_TIMEOUT_SECONDS = 5.0
_BACKGROUND_RESULT_PAGE_CONTENT_MAX_CHARS = 20_000
_BACKGROUND_RESULT_SERIALIZED_MAX_CHARS = 32_768 + 4_096


def _workflow_metadata_for_admitted_task(
  admitted_task: AdmittedTask,
) -> WorkflowTaskMetadata | None:
  identity = admitted_task.workflow_identity
  if identity is None:
    return None
  admitted_plan_digest = admitted_task.admitted_plan_digest
  if admitted_plan_digest is None:
    raise ValueError("workflow admitted task requires admitted_plan_digest")
  return WorkflowTaskMetadata(
    workflow_run_id=identity.workflow_run_id,
    plan_id=identity.plan_id,
    phase_number=identity.phase_number,
    revision=identity.revision,
    node_id=identity.node_id,
    item_key=identity.item_key,
    attempt_number=admitted_task.attempt.attempt_number,
    attempt_id=admitted_task.attempt.attempt_id,
    operation=admitted_task.operation.operation,
    admitted_plan_digest=admitted_plan_digest,
    admitted_task_digest=admitted_task.admitted_task_digest,
    model_bind_digest=admitted_task.model_bind_digest,
    capability_binding_digest=admitted_task.capability_binding_digest,
    tool_grant_digest=admitted_task.tool_grant_digest,
    result_requirement=admitted_task.result_requirement,
  )


def _background_result_page_content_chars() -> int:
  model_cap = model_tool_result_max_chars()
  if model_cap <= 0:
    return _BACKGROUND_RESULT_PAGE_CONTENT_MAX_CHARS
  return max(
    1_000,
    min(
      _BACKGROUND_RESULT_PAGE_CONTENT_MAX_CHARS,
      max(0, model_cap - 2_000) // 2,
    ),
  )


def _resume_source_admission_error(
  source: TaskEntry | None,
  *,
  original_task_id: str,
) -> dict[str, Any] | None:
  if source is None:
    return {
      "code": "not_found",
      "message": f"Unknown background task: {original_task_id}",
    }
  if (
    source.completion_finalizer_detached
    or source.completion_persistence_state in {"in_flight", "uncertain"}
  ):
    return {
      "code": "completion_pending",
      "message": (
        f"Task {original_task_id} has an unresolved completion; retry "
        "after its durable outcome settles"
      ),
    }
  if source.state != TaskState.INTERRUPTED:
    return {
      "code": "not_interrupted",
      "message": f"Task {original_task_id} is not interrupted",
    }
  return None


def _task_state_from_task_result(result: TaskResult) -> TaskState:
  return task_state_for_result(result)


def _terminal_task_result(
  entry: TaskEntry,
  *,
  status: Literal["failed", "interrupted", "cancelled"],
  reason: str,
) -> TaskResult | None:
  admitted = entry.admitted_task
  if admitted is None:
    return None
  progress = entry.progress
  return terminal_task_result(
    admitted,
    status=status,
    reason=reason,
    tools_used=tuple(progress.recent_tools),
    usage=UsageObservation(
      input_tokens=progress.input_tokens,
      output_tokens=progress.output_tokens,
      tool_calls=progress.tool_use_count,
    ),
  )


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
    peek = getattr(self._notification_queue, "peek", None)
    notifications = (
      list(peek(max_count=max_count))
      if callable(peek)
      else []
    )
    consumed = _runner_attr(
      self,
      "_consume_notifications",
      _consume_notifications,
    )(
      self._notification_queue,
      max_count=max_count,
    )
    registry = getattr(self, "_task_registry", None)
    registry_get = getattr(registry, "get", None)
    if callable(registry_get):
      for notification in notifications[:consumed]:
        entry = registry_get(notification.task_id)
        if (
          entry is not None
          and entry.notification_delivery_state == "queued"
          and notification.notification_generation
          == entry.notification_generation
        ):
          entry.notification_delivery_state = "delivered"
    return consumed

  async def _wait_for_background_notification(self) -> bool:
    """Wait event-first for a terminal child notification or pause request."""
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    first_completed = getattr(
      asyncio_module,
      "FIRST_COMPLETED",
      asyncio.FIRST_COMPLETED,
    )
    is_future = getattr(asyncio_module, "isfuture", asyncio.isfuture)
    runner_label = getattr(self, "_sid", "runner")

    while (
      self._notification_queue.pending_count == 0
      and not self._operator_pause_requested()
    ):
      live_tasks: set[asyncio.Task[Any]] = set()
      for entry in self._task_registry.list_tasks(state=TaskState.RUNNING):
        task = entry.asyncio_task
        done = getattr(task, "done", None)
        if (
          task is not None
          and is_future(task)
          and callable(done)
          and not done()
        ):
          live_tasks.add(task)
      if not live_tasks:
        return False

      notification_waiter = asyncio_module.create_task(
        self._notification_queue.wait_until_available(),
        name=f"{runner_label}:wait-background-notification",
      )
      temporary_waiters = [notification_waiter]
      pause_event = getattr(self, "_operator_pause_event", None)
      if pause_event is not None:
        temporary_waiters.append(
          asyncio_module.create_task(
            pause_event.wait(),
            name=f"{runner_label}:wait-operator-pause",
          )
        )
      try:
        await asyncio_module.wait(
          [*live_tasks, *temporary_waiters],
          return_when=first_completed,
        )
      finally:
        for waiter in temporary_waiters:
          if not waiter.done():
            waiter.cancel()
        await asyncio_module.gather(
          *temporary_waiters,
          return_exceptions=True,
        )

    return self._notification_queue.pending_count > 0

  @staticmethod
  def _inject_system_prompt_reminder(
    system_prompt: Optional[Union[str, List[Tuple[str, bool]]]],
    reminder: str,
  ) -> Optional[Union[str, List[Tuple[str, bool]]]]:
    return _runner_module_attr("_inject_system_prompt_reminder", _inject_system_prompt_reminder)(system_prompt, reminder)

  def _background_completion_persist_timeout(self) -> float:
    raw_timeout = getattr(
      self,
      "_background_completion_persist_timeout_seconds",
      _BACKGROUND_COMPLETION_PERSIST_TIMEOUT_SECONDS,
    )
    return max(0.0, float(raw_timeout))

  @staticmethod
  def _consume_background_task_exception(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
      return
    try:
      task.exception()
    except BaseException:
      pass

  def _track_background_completion_append(
    self,
    bg_task: TaskEntry,
    append_task: asyncio.Task[Any],
    *,
    final_state: TaskState,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
  ) -> None:
    pending_appends = getattr(
      self,
      "_pending_background_completion_appends",
      None,
    )
    if pending_appends is None:
      pending_appends = set()
      self._pending_background_completion_appends = pending_appends
    pending_appends.add(append_task)

    def _completion_append_done(task: asyncio.Task[Any]) -> None:
      pending_appends.discard(task)
      try:
        task.result()
      except BaseException as exc:
        if task.cancelled():
          self._writer_lease_poisoned = True
        self._record_completion_persistence_uncertainty(bg_task, exc)
        if bg_task.completion_finalizer_detached:
          self._schedule_detached_completion_reconciliation(
            bg_task
          )
      else:
        bg_task.completion_persistence_state = "committed"
        bg_task.completion_persistence_error = None
        if bg_task.completion_finalizer_detached:
          late_finalizer = asyncio.create_task(
            self._commit_detached_background_completion(
              bg_task,
              final_state=final_state,
              result=result,
              error=error,
            ),
            name=f"{bg_task.task_id}:commit-detached-completion",
          )
          self._track_background_initialization(late_finalizer)
      if (
        getattr(self, "_deferred_write_lease_release", False)
        and not pending_appends
      ):
        try:
          self._release_write_lease()
        except BaseException:
          log.exception(
            "[%s] failed to release deferred writer lease",
            self._sid,
          )

    append_task.add_done_callback(_completion_append_done)

  async def _invoke_background_completion_callback(
    self,
    bg_task: TaskEntry,
  ) -> None:
    callback = bg_task.completion_callback
    if callback is None or bg_task.completion_callback_invoked:
      return
    bg_task.completion_callback_invoked = True
    try:
      maybe_awaitable = callback(bg_task)
      if asyncio.iscoroutine(maybe_awaitable):
        await maybe_awaitable
    except Exception:
      pass

  @staticmethod
  def _required_skill_lifecycle(
    bg_task: TaskEntry,
  ) -> dict[str, Any] | None:
    metadata = (
      bg_task.metadata
      if isinstance(bg_task.metadata, dict)
      else {}
    )
    raw_lifecycle = metadata.get(
      REQUIRED_SKILL_LIFECYCLE_METADATA_KEY
    )
    try:
      normalized = _normalize_required_skill_lifecycle(raw_lifecycle)
      if normalized is not None:
        resolve_required_skill_result_marker(
          lifecycle=dict(raw_lifecycle),
          task_id=bg_task.task_id,
          registration_seq=0,
          completion_seq=0,
          marker_candidates=(),
        )
      return normalized
    except (
      RequiredSkillResultMarkerValidationError,
      TypeError,
      ValueError,
    ) as exc:
      raise RuntimeError(
        f"Task {bg_task.task_id} has invalid required skill lifecycle "
        f"metadata: {exc}"
      ) from exc

  def _attach_required_skill_result_settlement(
    self,
    bg_task: TaskEntry,
    *,
    requested_lifecycle: dict[str, Any] | None,
    event_factory: RequiredSkillResultEventFactory | None,
    projector: RequiredSkillResultProjector | None,
  ) -> None:
    persisted_lifecycle = self._required_skill_lifecycle(bg_task)
    if persisted_lifecycle is None or requested_lifecycle is None:
      return
    try:
      normalized_requested = _normalize_required_skill_lifecycle(
        requested_lifecycle
      )
    except ValueError as exc:
      raise RuntimeError(
        f"Task {bg_task.task_id} received invalid required skill lifecycle "
        f"metadata: {exc}"
      ) from exc
    if normalized_requested != persisted_lifecycle:
      return
    if (
      event_factory is not None
      and bg_task.required_skill_result_event_factory is None
    ):
      bg_task.required_skill_result_event_factory = event_factory
    if (
      projector is not None
      and bg_task.required_skill_result_projector is None
    ):
      bg_task.required_skill_result_projector = projector

  async def _durable_background_task_bounds(
    self,
    bg_task: TaskEntry,
  ) -> tuple[int, int]:
    durable_log = getattr(self, "_agent_session_log", None)
    if durable_log is None:
      raise RuntimeError(
        f"Task {bg_task.task_id} requires durable skill settlement "
        "but no durable session log is configured"
      )
    task_entries, _ = await durable_log.query(
      event_types={"task_registered", "task_completed"},
      contains_text=bg_task.task_id,
      order="asc",
    )
    registrations = [
      entry
      for entry in task_entries
      if (
        entry.event.get("type") == "task_registered"
        and entry.event.get("task_id") == bg_task.task_id
      )
    ]
    completions = [
      entry
      for entry in task_entries
      if (
        entry.event.get("type") == "task_completed"
        and entry.event.get("task_id") == bg_task.task_id
      )
    ]
    if len(registrations) != 1 or len(completions) != 1:
      raise RuntimeError(
        f"Task {bg_task.task_id} requires exactly one durable "
        "registration and completion before skill result settlement"
      )
    registered_seq = registrations[0].seq
    completed_seq = completions[0].seq
    if registered_seq >= completed_seq:
      raise RuntimeError(
        f"Task {bg_task.task_id} durable completion does not follow "
        "registration"
      )
    return registered_seq, completed_seq

  async def _durable_required_skill_result_event(
    self,
    bg_task: TaskEntry,
    lifecycle: dict[str, Any],
  ) -> dict[str, Any] | None:
    durable_log = getattr(self, "_agent_session_log", None)
    if durable_log is None:
      raise RuntimeError(
        f"Task {bg_task.task_id} requires durable skill settlement "
        "but no durable session log is configured"
      )
    registration_seq, completion_seq = (
      await self._durable_background_task_bounds(bg_task)
    )

    skill_run_id = lifecycle["skill_run_id"]
    task_marker_entries, _ = await durable_log.query(
      event_types={"skill_result_captured"},
      contains_text=bg_task.task_id,
      order="asc",
    )
    lifecycle_marker_entries, _ = await durable_log.query(
      event_types={"skill_result_captured"},
      contains_text=skill_run_id,
      order="asc",
    )
    marker_candidates = {
      entry.seq: (entry.seq, entry.event)
      for entry in [
        *task_marker_entries,
        *lifecycle_marker_entries,
      ]
    }
    try:
      return resolve_required_skill_result_marker(
        lifecycle=lifecycle,
        task_id=bg_task.task_id,
        registration_seq=registration_seq,
        completion_seq=completion_seq,
        marker_candidates=marker_candidates.values(),
      )
    except RequiredSkillResultMarkerValidationError as exc:
      raise RuntimeError(str(exc)) from exc

  async def _build_required_skill_result_event(
    self,
    bg_task: TaskEntry,
    lifecycle: dict[str, Any],
  ) -> dict[str, Any]:
    if bg_task.pending_final_state is not None:
      result = bg_task.pending_result
      error = bg_task.pending_error
    else:
      result = bg_task.result
      error = bg_task.error
    event_factory = bg_task.required_skill_result_event_factory
    if event_factory is not None:
      event = event_factory(bg_task, result, error)
      if not isinstance(event, dict):
        raise RuntimeError(
          "required skill result event factory must return an object"
        )
      event = dict(event)
    else:
      durable_log = getattr(self, "_agent_session_log", None)
      if durable_log is None:
        raise RuntimeError(
          f"Task {bg_task.task_id} requires durable skill settlement "
          "but no durable session log is configured"
        )
      sub_agent_id = str(
        bg_task.metadata.get("sub_agent_id") or ""
      ).strip()
      entries = []
      if sub_agent_id:
        registered_seq, completed_seq = (
          await self._durable_background_task_bounds(bg_task)
        )
        if registered_seq + 1 <= completed_seq - 1:
          entries, _ = await durable_log.query(
            sub_agent_id=sub_agent_id,
            after_seq=registered_seq + 1,
            before_seq=completed_seq - 1,
            order="asc",
          )
      event = build_skill_result_captured_event(
        skill_run_id=lifecycle["skill_run_id"],
        skill=lifecycle["skill"],
        ticker=lifecycle["ticker"],
        scope=lifecycle["scope"],
        portfolio_id=lifecycle["portfolio_id"],
        entries=entries,
        result=result,
        error=error,
        canonical_result_evidence_authoritative=True,
      )

    event.update({
      "type": "skill_result_captured",
      "skill_run_id": lifecycle["skill_run_id"],
      "skill": lifecycle["skill"],
      "scope": lifecycle["scope"],
      "ticker": lifecycle["ticker"],
      "portfolio_id": lifecycle["portfolio_id"],
    })
    lifecycle_identity = TopLevelSkillLifecycleMetadata(
      skill_run_id=lifecycle["skill_run_id"],
      skill=lifecycle["skill"],
      scope=lifecycle["scope"],
      ticker=lifecycle["ticker"],
      portfolio_id=lifecycle["portfolio_id"],
    )
    event = lifecycle_identity.normalize_result_event(event)
    event.update(self._task_correlation_payload(bg_task))
    event.update({
      "type": "skill_result_captured",
      "skill_run_id": lifecycle["skill_run_id"],
      "skill": lifecycle["skill"],
      "task_id": bg_task.task_id,
      "scope": lifecycle["scope"],
      "ticker": lifecycle["ticker"],
      "portfolio_id": lifecycle["portfolio_id"],
    })
    return event

  async def _project_required_skill_result_event(
    self,
    bg_task: TaskEntry,
    event: dict[str, Any],
  ) -> None:
    if bg_task.required_skill_result_projected:
      return
    projector = bg_task.required_skill_result_projector
    if projector is None:
      return
    try:
      maybe_awaitable = projector(dict(event))
      if inspect.isawaitable(maybe_awaitable):
        await maybe_awaitable
    except Exception as exc:
      log.warning(
        "[%s] durable skill result projection failed for %s",
        self._sid,
        bg_task.task_id,
        exc_info=True,
      )
      mark_exception_traceback_logged(exc)
      raise
    bg_task.required_skill_result_projected = True

  async def _persist_required_skill_result_once(
    self,
    bg_task: TaskEntry,
  ) -> bool:
    lifecycle = self._required_skill_lifecycle(bg_task)
    if lifecycle is None:
      return True
    existing = await self._durable_required_skill_result_event(
      bg_task,
      lifecycle,
    )
    if existing is None:
      event = await self._build_required_skill_result_event(
        bg_task,
        lifecycle,
      )
      append_failure: BaseException | None = None
      try:
        durable_entry = await self._append_durable_event(event)
        if durable_entry is None:
          append_failure = RuntimeError(
            "durable skill result append returned no entry"
          )
      except BaseException as exc:
        append_failure = exc
      exact_confirmation = await self._confirm_durable_skill_event(
        event
      )
      existing = await self._durable_required_skill_result_event(
        bg_task,
        lifecycle,
      )
      if exact_confirmation is None or existing is None:
        if append_failure is not None:
          raise append_failure
        raise RuntimeError(
          f"Task {bg_task.task_id} skill result append was not durably "
          "confirmed"
        )
      if not _exact_value_match(exact_confirmation, existing):
        raise RuntimeError(
          f"Task {bg_task.task_id} skill result durable confirmation "
          "does not match its validated task marker"
        )
    bg_task.required_skill_result_settled = True
    await self._project_required_skill_result_event(
      bg_task,
      existing,
    )
    return True

  async def _ensure_required_skill_result_settled(
    self,
    bg_task: TaskEntry,
  ) -> bool:
    if self._required_skill_lifecycle(bg_task) is None:
      return True
    if (
      bg_task.completion_persistence_state != "committed"
      or (
        bg_task.state not in {
          TaskState.COMPLETED,
          TaskState.FAILED,
          TaskState.KILLED,
        }
        and bg_task.pending_final_state not in {
          TaskState.COMPLETED,
          TaskState.FAILED,
          TaskState.KILLED,
        }
      )
    ):
      raise RuntimeError(
        f"Task {bg_task.task_id} cannot settle its required skill result "
        "before terminal completion is durably committed"
      )
    if (
      bg_task.required_skill_result_settled
      and (
        bg_task.required_skill_result_projector is None
        or bg_task.required_skill_result_projected
      )
    ):
      return True

    lifecycle = self._required_skill_lifecycle(bg_task)
    if lifecycle is None:
      return True
    settlement_key = (
      bg_task.task_id,
      lifecycle["skill_run_id"],
    )
    settlement_tasks = getattr(
      self,
      "_required_skill_result_tasks",
      None,
    )
    if settlement_tasks is None:
      settlement_tasks = {}
      self._required_skill_result_tasks = settlement_tasks
    settlement_task = settlement_tasks.get(settlement_key)
    if settlement_task is not None and settlement_task.done():
      try:
        settled = settlement_task.result()
      except BaseException:
        settlement_task = None
      else:
        bg_task.required_skill_result_settled = settled
        bg_task.required_skill_result_task = settlement_task
        return settled
    if settlement_task is None:
      settlement_task = asyncio.create_task(
        self._persist_required_skill_result_once(bg_task),
        name=f"{bg_task.task_id}:settle-skill-result",
      )
      settlement_tasks[settlement_key] = settlement_task

      def _release_settlement_task(
        completed_task: asyncio.Task[bool],
      ) -> None:
        if settlement_tasks.get(settlement_key) is completed_task:
          settlement_tasks.pop(settlement_key, None)

      settlement_task.add_done_callback(_release_settlement_task)
      self._track_background_initialization(settlement_task)
    bg_task.required_skill_result_task = settlement_task
    try:
      done, _pending = await asyncio.wait(
        {settlement_task},
        timeout=self._background_completion_persist_timeout(),
      )
    except asyncio.CancelledError:
      while True:
        try:
          await asyncio.shield(settlement_task)
          break
        except asyncio.CancelledError:
          if settlement_task.done():
            settlement_task.result()
            break
      raise
    if settlement_task not in done:
      timeout = self._background_completion_persist_timeout()
      raise asyncio.TimeoutError(
        f"required skill result settlement exceeded {timeout:.3f}s"
      )
    settled = settlement_task.result()
    bg_task.required_skill_result_settled = settled
    return settled

  async def _commit_detached_background_completion(
    self,
    bg_task: TaskEntry,
    *,
    final_state: TaskState,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
  ) -> None:
    async with bg_task.finalization_lock:
      if (
        not bg_task.completion_finalizer_detached
        or bg_task.completion_persistence_state != "committed"
      ):
        return
      await self._settle_and_commit_background_completion_live(
        bg_task,
        final_state=final_state,
        result=result,
        error=error,
      )
    await self._invoke_background_completion_callback(bg_task)

  def _schedule_detached_completion_reconciliation(
    self,
    bg_task: TaskEntry,
  ) -> None:
    reconciliation_task = asyncio.create_task(
      self._reconcile_detached_background_completion(bg_task),
      name=f"{bg_task.task_id}:reconcile-detached-completion",
    )
    self._track_background_initialization(reconciliation_task)

  async def _reconcile_detached_background_completion(
    self,
    bg_task: TaskEntry,
  ) -> None:
    committed = False
    async with bg_task.finalization_lock:
      if not bg_task.completion_finalizer_detached:
        return
      try:
        durable_entry = (
          await self._confirmed_durable_background_completion(bg_task)
        )
      except asyncio.CancelledError:
        raise
      except BaseException:
        self._writer_lease_poisoned = True
        log.exception(
          "[%s] failed to reconcile detached background completion for %s",
          self._sid,
          bg_task.task_id,
        )
        return
      if durable_entry is None:
        return
      bg_task.completion_persistence_state = "committed"
      bg_task.completion_persistence_error = None
      await self._settle_and_commit_background_completion_live(
        bg_task,
        final_state=durable_entry.state,
        result=(
          dict(durable_entry.result)
          if isinstance(durable_entry.result, dict)
          else None
        ),
        error=(
          dict(durable_entry.error)
          if isinstance(durable_entry.error, dict)
          else None
        ),
      )
      committed = True
    if committed:
      await self._invoke_background_completion_callback(bg_task)

  def _track_background_initialization(
    self,
    initialization_task: asyncio.Task[Any],
  ) -> None:
    pending_initializations = getattr(
      self,
      "_pending_background_initializations",
      None,
    )
    if pending_initializations is None:
      pending_initializations = set()
      self._pending_background_initializations = pending_initializations
    pending_initializations.add(initialization_task)

    def _initialization_done(task: asyncio.Task[Any]) -> None:
      pending_initializations.discard(task)
      if task.cancelled():
        self._writer_lease_poisoned = True
      self._consume_background_task_exception(task)
      if (
        getattr(self, "_deferred_write_lease_release", False)
        and not pending_initializations
        and not getattr(
          self,
          "_pending_background_completion_appends",
          set(),
        )
      ):
        try:
          self._release_write_lease()
        except BaseException:
          log.exception(
            "[%s] failed to release writer lease after background "
            "initialization",
            self._sid,
          )

    initialization_task.add_done_callback(_initialization_done)

  @staticmethod
  def _completion_persistence_error_detail(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
      error_type = type(error).__name__
      message = str(error).strip() or error_type
      return f"{error_type}: {message}"
    return str(error).strip() or "completion persistence outcome is unknown"

  def _record_completion_persistence_uncertainty(
    self,
    bg_task: TaskEntry,
    error: BaseException | str,
  ) -> None:
    if getattr(bg_task, "completion_persistence_state", "not_started") == "committed":
      return
    detail = self._completion_persistence_error_detail(error)
    bg_task.completion_persistence_state = "uncertain"
    if getattr(bg_task, "completion_persistence_error", None) is None:
      bg_task.completion_persistence_error = detail
      log.warning(
        "[%s] background task %s completion persistence is uncertain: %s",
        self._sid,
        bg_task.task_id,
        detail,
      )

  async def _persist_background_completion_once(
    self,
    bg_task: TaskEntry,
    final_state: TaskState,
    *,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
  ) -> None:
    bg_task.completion_persistence_state = "in_flight"
    append_task = asyncio.create_task(
      self._append_task_completed_event(
        bg_task,
        final_state,
        result=result,
        error=error,
      ),
      name=f"{bg_task.task_id}:persist-completion",
    )
    self._track_background_completion_append(
      bg_task,
      append_task,
      final_state=final_state,
      result=result,
      error=error,
    )
    try:
      done, _pending = await asyncio.wait(
        {append_task},
        timeout=self._background_completion_persist_timeout(),
      )
    except BaseException as exc:
      self._record_completion_persistence_uncertainty(bg_task, exc)
      raise

    if append_task not in done:
      timeout = self._background_completion_persist_timeout()
      error = asyncio.TimeoutError(
        f"task completion persistence exceeded {timeout:.3f}s"
      )
      self._record_completion_persistence_uncertainty(bg_task, error)
      raise error

    try:
      append_task.result()
    except BaseException as exc:
      self._record_completion_persistence_uncertainty(bg_task, exc)
      raise
    bg_task.completion_persistence_state = "committed"
    bg_task.completion_persistence_error = None

  async def _durable_agent_completion(
    self,
    task_id: str,
  ) -> Any | None:
    durable_log = getattr(self, "_agent_session_log", None)
    if durable_log is None:
      return None
    entries, _ = await durable_log.query(
      event_types={"agent_completion"},
      contains_text=task_id,
      order="asc",
    )
    payloads = [
      entry.event
      for entry in entries
      if (
        entry.event.get("type") == "agent_completion"
        and entry.event.get("task_id") == task_id
      )
    ]
    if not payloads:
      return None
    try:
      typed = [event_from_dict(payload) for payload in payloads]
    except (KeyError, TypeError, ValueError) as exc:
      raise RuntimeError(
        f"Task {task_id} has a malformed durable agent completion"
      ) from exc
    if any(not isinstance(event, AgentCompletionEvent) for event in typed):
      raise RuntimeError(
        f"Task {task_id} has an invalid durable agent completion type"
      )
    first = typed[0]
    if any(
      event.event_id != first.event_id
      or event.fingerprint != first.fingerprint
      or event.envelope != first.envelope
      for event in typed[1:]
    ):
      raise RuntimeError(
        f"Task {task_id} has conflicting durable agent completions"
      )
    return first.envelope

  async def _ensure_agent_completion_published(
    self,
    bg_task: TaskEntry,
  ) -> None:
    task_result = bg_task.task_result
    if task_result is None or _workflow_owns_terminal_notification(bg_task):
      return
    if not (
      task_result.values.terminal_narrative
      or task_result.values.projection
      or task_result.values.artifacts
    ):
      return
    policy = bg_task.parent_result_policy or _ordinary_parent_result_policy(
      task_result
    )

    def _read_terminal(result: TaskResult) -> str:
      return read_task_result_terminal_narrative(
        result,
        workspace_dir=self._workspace_dir,
      )

    def _read_grant(source: ContentHandle) -> ContentReadGrant:
      identity = sha256_digest({
        "task_id": bg_task.task_id,
        "content_id": source.content_id,
        "scope": "direct_parent",
        "principal_id": self._runner_id,
      }).removeprefix("sha256:")
      return ContentReadGrant(
        grant_id=f"content-read:{identity}",
        content_id=source.content_id,
        scope="direct_parent",
        principal_id=str(self._runner_id),
      )

    envelope = _build_agent_completion_envelope(
      task_result,
      policy=policy,
      terminal_narrative_reader=_read_terminal,
      read_grant_factory=_read_grant,
      message_id=_agent_completion_message_id(task_result),
    )
    durable = await self._durable_agent_completion(bg_task.task_id)
    if durable is not None:
      if durable != envelope:
        raise RuntimeError(
          f"Task {bg_task.task_id} completion replay conflicts with durable bytes"
        )
      bg_task.completion_envelope = durable
      return
    await self._append_durable_event(
      _build_agent_completion_event(
        task_id=bg_task.task_id,
        envelope=envelope,
        ts=_runner_attr(self, "time", time).time(),
      )
    )
    bg_task.completion_envelope = envelope

  async def _run_background_agent(
    self,
    bg_task: TaskEntry,
    handler: BackgroundTaskHandler,
    tool_input: Dict[str, Any],
    call_index: int,
    on_complete: BackgroundTaskCallback | None = None,
  ) -> None:
    if (
      on_complete is not None
      and bg_task.completion_callback is None
    ):
      bg_task.completion_callback = on_complete
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    final_state = TaskState.FAILED
    try:
      semaphore = self._ensure_sub_agent_semaphore()
      if semaphore is not None:
        async with semaphore:
          handler_result, handler_error = await handler(tool_input, call_index=call_index)
      else:
        handler_result, handler_error = await handler(tool_input, call_index=call_index)
      error = handler_error
      if isinstance(handler_result, TaskResult):
        if error is not None:
          raise ValueError("canonical TaskResult cannot be paired with handler error")
        if handler_result.attempt.physical_task_id != bg_task.task_id:
          raise ValueError("TaskResult physical task identity mismatch")
        if (
          bg_task.admitted_task is not None
          and (
            handler_result.logical_task != bg_task.admitted_task.logical_task
            or handler_result.attempt != bg_task.admitted_task.attempt
            or handler_result.provenance.admitted_task_digest
            != bg_task.admitted_task.admitted_task_digest
          )
        ):
          raise ValueError("TaskResult does not settle the exact admitted task")
        bg_task.task_result = handler_result
        result = handler_result.model_dump(mode="json")
        final_state = _task_state_from_task_result(handler_result)
      else:
        if bg_task.admitted_task is not None:
          raise TypeError("admitted agent handler must return TaskResult")
        result = (
          handler_result
          if isinstance(handler_result, dict)
          else ({"result": handler_result} if handler_result is not None else None)
        )
        final_state = (
          TaskState.COMPLETED if error is None else TaskState.FAILED
        )
    except asyncio.CancelledError:
      termination_intent = bg_task.termination_intent or "cancelled"
      canonical = _terminal_task_result(
        bg_task,
        status="cancelled",
        reason=termination_intent,
      )
      if canonical is not None:
        bg_task.task_result = canonical
        result = canonical.model_dump(mode="json")
      else:
        result = self._background_termination_result(bg_task, termination_intent)
      error = None
      final_state = (
        _task_state_from_task_result(canonical)
        if canonical is not None
        else (
          TaskState.KILLED
          if termination_intent in {"cancelled", "killed"}
          else TaskState.INTERRUPTED
        )
      )
    except Exception as exc:
      error = {"code": "background_error", "message": str(exc)}
      canonical = _terminal_task_result(
        bg_task,
        status="failed",
        reason=error["message"],
      )
      if canonical is not None:
        bg_task.task_result = canonical
        result = canonical.model_dump(mode="json")
      final_state = TaskState.FAILED

    finalizer_task = asyncio.create_task(
      self._finalize_background_agent(
        bg_task,
        final_state=final_state,
        result=result,
        error=error,
      ),
      name=f"{bg_task.task_id}:finalize",
    )
    try:
      await asyncio.shield(finalizer_task)
    except asyncio.CancelledError:
      # A termination request that races with an already accepted handler
      # result must not cancel the sole durable/live finalizer.
      try:
        done, _pending = await asyncio.wait(
          {finalizer_task},
          timeout=self._background_completion_persist_timeout(),
        )
      except asyncio.CancelledError as exc:
        self._record_completion_persistence_uncertainty(bg_task, exc)
        finalizer_task.cancel()
        finalizer_task.add_done_callback(self._consume_background_task_exception)
        await self._reconcile_background_agent(
          bg_task,
          default_intent=bg_task.termination_intent or "cancelled",
        )
        raise
      if finalizer_task in done:
        finalizer_task.result()
      else:
        timeout = self._background_completion_persist_timeout()
        error = asyncio.TimeoutError(
          f"background finalizer exceeded {timeout:.3f}s after cancellation"
        )
        self._record_completion_persistence_uncertainty(bg_task, error)
        finalizer_task.cancel()
        finalizer_task.add_done_callback(self._consume_background_task_exception)
        await self._reconcile_background_agent(
          bg_task,
          default_intent=bg_task.termination_intent or "cancelled",
        )
        raise error
    finally:
      if (
        bg_task.completion_persistence_state == "committed"
        and bg_task.state in {
          TaskState.COMPLETED,
          TaskState.FAILED,
          TaskState.KILLED,
        }
      ):
        await self._invoke_background_completion_callback(bg_task)

  @staticmethod
  def _background_termination_result(
    bg_task: TaskEntry,
    termination_intent: TerminationIntent,
  ) -> dict[str, Any]:
    if bg_task.task_result is not None:
      return bg_task.task_result.model_dump(mode="json")
    canonical = _terminal_task_result(
      bg_task,
      status=(
        "cancelled" if termination_intent == "cancelled" else "interrupted"
      ),
      reason=termination_intent,
    )
    if canonical is not None:
      bg_task.task_result = canonical
      return canonical.model_dump(mode="json")
    progress = bg_task.progress
    detail = (
      "Background task was cancelled with its parent run"
      if termination_intent == "cancelled"
      else "Background task was killed before completion"
    )
    return {
      "status": "interrupted",
      "reason": termination_intent,
      "detail": detail,
      "usage": {
        "input_tokens": progress.input_tokens,
        "output_tokens": progress.output_tokens,
        "turn_count": progress.turn_count,
      },
      "tools_used": list(progress.recent_tools),
    }

  def _background_persistence_failure_result(
    self,
    bg_task: TaskEntry,
    failure: BaseException,
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    detail = (
      "Background work finished, but its result could not be durably "
      f"committed: {self._completion_persistence_error_detail(failure)}"
    )
    progress = bg_task.progress
    canonical = _terminal_task_result(
      bg_task,
      status="failed",
      reason=f"runtime_error: {detail}",
    )
    if canonical is not None:
      bg_task.task_result = canonical
      result = canonical.model_dump(mode="json")
    else:
      result = {
        "status": "failed",
        "reason": "runtime_error",
        "detail": detail,
        "usage": {
        "input_tokens": progress.input_tokens,
        "output_tokens": progress.output_tokens,
        "turn_count": progress.turn_count,
        },
        "tools_used": list(progress.recent_tools),
      }
    return result, {
      "code": "background_completion_persistence_failed",
      "message": detail,
    }

  async def _confirmed_durable_background_completion(
    self,
    bg_task: TaskEntry,
  ) -> TaskEntry | None:
    if getattr(self, "_agent_session_log", None) is None:
      return None
    durable_entry = await self._lookup_task_in_log(bg_task.task_id)
    if durable_entry is None or durable_entry.state not in {
      TaskState.COMPLETED,
      TaskState.FAILED,
      TaskState.KILLED,
    }:
      return None
    return durable_entry

  def _commit_background_completion_live(
    self,
    bg_task: TaskEntry,
    *,
    final_state: TaskState,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
  ) -> None:
    bg_task.completion_persistence_state = "committed"
    bg_task.completion_persistence_error = None
    bg_task.pending_final_state = final_state
    bg_task.pending_result = result
    bg_task.pending_error = error
    bg_task.completion_finalizer_detached = False
    if bg_task.state == TaskState.INTERRUPTED:
      self._task_registry.finalize_interrupted(
        bg_task.task_id,
        final_state,
        result=result,
        error=error,
      )
    else:
      self._task_registry.transition(
        bg_task.task_id,
        final_state,
        result=result,
        error=error,
      )

  async def _settle_and_commit_background_completion_live(
    self,
    bg_task: TaskEntry,
    *,
    final_state: TaskState,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
  ) -> None:
    if bg_task.completion_persistence_state != "committed":
      raise RuntimeError(
        f"Task {bg_task.task_id} cannot publish terminal completion "
        "before its durable completion is confirmed"
      )
    bg_task.pending_final_state = final_state
    bg_task.pending_result = result
    bg_task.pending_error = error
    await self._ensure_required_skill_result_settled(bg_task)
    await self._ensure_agent_completion_published(bg_task)
    self._commit_background_completion_live(
      bg_task,
      final_state=final_state,
      result=result,
      error=error,
    )

  def _interrupt_uncertain_background_completion(
    self,
    bg_task: TaskEntry,
    failure: BaseException,
  ) -> None:
    detail = self._completion_persistence_error_detail(failure)
    bg_task.completion_finalizer_detached = True
    bg_task.result = None
    bg_task.error = {
      "code": "background_completion_persistence_uncertain",
      "message": detail,
    }
    if bg_task.state not in {
      TaskState.COMPLETED,
      TaskState.FAILED,
      TaskState.KILLED,
      TaskState.INTERRUPTED,
    }:
      self._task_registry.transition(
        bg_task.task_id,
        TaskState.INTERRUPTED,
        error=bg_task.error,
      )
    self._schedule_detached_completion_reconciliation(bg_task)

  async def _finalize_background_agent(
    self,
    bg_task: TaskEntry,
    *,
    final_state: TaskState,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
  ) -> None:
    async with bg_task.finalization_lock:
      if bg_task.state in {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.KILLED,
      }:
        await self._ensure_required_skill_result_settled(bg_task)
        return
      bg_task.pending_final_state = final_state
      bg_task.pending_result = result
      bg_task.pending_error = error
      try:
        await self._persist_background_completion_once(
          bg_task,
          final_state,
          result=result,
          error=error,
        )
      except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        if bg_task.completion_persistence_state == "committed":
          await self._settle_and_commit_background_completion_live(
            bg_task,
            final_state=final_state,
            result=result,
            error=error,
          )
        else:
          self._interrupt_uncertain_background_completion(
            bg_task,
            exc,
          )
        raise
      except BaseException as primary_failure:
        durable_entry: TaskEntry | None = None
        try:
          durable_entry = (
            await self._confirmed_durable_background_completion(bg_task)
          )
        except asyncio.CancelledError as lookup_failure:
          self._writer_lease_poisoned = True
          self._interrupt_uncertain_background_completion(
            bg_task,
            lookup_failure,
          )
          raise
        except BaseException as lookup_failure:
          self._writer_lease_poisoned = True
          self._interrupt_uncertain_background_completion(
            bg_task,
            lookup_failure,
          )
          raise
        if durable_entry is not None:
          bg_task.completion_persistence_state = "committed"
          bg_task.completion_persistence_error = None
          await self._settle_and_commit_background_completion_live(
            bg_task,
            final_state=durable_entry.state,
            result=(
              dict(durable_entry.result)
              if isinstance(durable_entry.result, dict)
              else None
            ),
            error=(
              dict(durable_entry.error)
              if isinstance(durable_entry.error, dict)
              else None
            ),
          )
          return

        fallback_result, fallback_error = (
          self._background_persistence_failure_result(
            bg_task,
            primary_failure,
          )
        )
        fallback_state = TaskState.FAILED
        bg_task.pending_final_state = fallback_state
        bg_task.pending_result = fallback_result
        bg_task.pending_error = fallback_error
        try:
          await self._persist_background_completion_once(
            bg_task,
            fallback_state,
            result=fallback_result,
            error=fallback_error,
          )
        except BaseException as fallback_failure:
          try:
            durable_entry = (
              await self._confirmed_durable_background_completion(
                bg_task,
              )
            )
          except asyncio.CancelledError as lookup_failure:
            self._writer_lease_poisoned = True
            self._interrupt_uncertain_background_completion(
              bg_task,
              lookup_failure,
            )
            raise
          except BaseException as lookup_failure:
            self._writer_lease_poisoned = True
            self._interrupt_uncertain_background_completion(
              bg_task,
              lookup_failure,
            )
            raise
          if durable_entry is not None:
            bg_task.completion_persistence_state = "committed"
            bg_task.completion_persistence_error = None
            await self._settle_and_commit_background_completion_live(
              bg_task,
              final_state=durable_entry.state,
              result=(
                dict(durable_entry.result)
                if isinstance(durable_entry.result, dict)
                else None
              ),
              error=(
                dict(durable_entry.error)
                if isinstance(durable_entry.error, dict)
                else None
              ),
            )
            return
          self._interrupt_uncertain_background_completion(
            bg_task,
            fallback_failure,
          )
          raise
        await self._settle_and_commit_background_completion_live(
          bg_task,
          final_state=fallback_state,
          result=fallback_result,
          error=fallback_error,
        )
        return
      await self._settle_and_commit_background_completion_live(
        bg_task,
        final_state=final_state,
        result=result,
        error=error,
      )

  async def _reconcile_background_agent(
    self,
    bg_task: TaskEntry,
    *,
    default_intent: TerminationIntent,
  ) -> None:
    if bg_task.state != TaskState.RUNNING:
      if (
        bg_task.completion_persistence_state == "committed"
        and bg_task.state in {
          TaskState.COMPLETED,
          TaskState.FAILED,
          TaskState.KILLED,
        }
      ):
        await self._ensure_required_skill_result_settled(bg_task)
      return

    pending_final_state = getattr(bg_task, "pending_final_state", None)
    persistence_state = getattr(
      bg_task,
      "completion_persistence_state",
      "not_started",
    )
    if pending_final_state is not None:
      if persistence_state == "committed":
        await self._settle_and_commit_background_completion_live(
          bg_task,
          final_state=pending_final_state,
          result=bg_task.pending_result,
          error=bg_task.pending_error,
        )
      else:
        failure = RuntimeError(
          "live shutdown reconciliation preceded durable completion "
          "confirmation"
        )
        self._record_completion_persistence_uncertainty(
          bg_task,
          failure,
        )
        self._interrupt_uncertain_background_completion(
          bg_task,
          failure,
        )
      return

    asyncio_task = bg_task.asyncio_task
    task_done = (
      asyncio_task is not None
      and callable(getattr(asyncio_task, "done", None))
      and asyncio_task.done()
    )
    if task_done:
      task_error: BaseException | None = None
      cancelled = getattr(asyncio_task, "cancelled", None)
      task_cancelled = bool(cancelled()) if callable(cancelled) else False
      try:
        if not task_cancelled:
          task_error = asyncio_task.exception()
      except BaseException as exc:
        task_error = exc
      if task_error is not None:
        error = {
          "code": "background_error",
          "message": str(task_error).strip() or type(task_error).__name__,
        }
        canonical = _terminal_task_result(
          bg_task,
          status="failed",
          reason=f"runtime_error: {error['message']}",
        )
        if canonical is not None:
          bg_task.task_result = canonical
          result = canonical.model_dump(mode="json")
        else:
          result = {
            "status": "failed",
            "reason": "runtime_error",
            "detail": error["message"],
          }
        await self._finalize_background_agent(
          bg_task,
          final_state=TaskState.FAILED,
          result=result,
          error=error,
        )
        return
      if not task_cancelled:
        detail = "Background task ended without staging a durable completion"
        canonical = _terminal_task_result(
          bg_task,
          status="failed",
          reason=f"runtime_error: {detail}",
        )
        if canonical is not None:
          bg_task.task_result = canonical
          result = canonical.model_dump(mode="json")
        else:
          result = {
            "status": "failed",
            "reason": "runtime_error",
            "detail": detail,
          }
        await self._finalize_background_agent(
          bg_task,
          final_state=TaskState.FAILED,
          result=result,
          error={
            "code": "background_completion_missing",
            "message": (
              "Background task ended without staging a durable completion"
            ),
          },
        )
        return

    self._task_registry.kill(
      bg_task.task_id,
      termination_intent=default_intent,
    )
    effective_intent = bg_task.termination_intent or default_intent
    result = self._background_termination_result(
      bg_task,
      effective_intent,
    )
    final_state = (
      _task_state_from_task_result(bg_task.task_result)
      if bg_task.task_result is not None
      else TaskState.KILLED
    )
    try:
      await self._finalize_background_agent(
        bg_task,
        final_state=final_state,
        result=result,
        error=None,
      )
    except BaseException as exc:
      if bg_task.state != TaskState.INTERRUPTED:
        self._record_completion_persistence_uncertainty(
          bg_task,
          exc,
        )
        self._interrupt_uncertain_background_completion(
          bg_task,
          exc,
        )

  async def _reconcile_background_agents(
    self,
    running_entries: List[TaskEntry],
    *,
    default_intent: TerminationIntent,
  ) -> None:
    for bg_task in running_entries:
      await self._reconcile_background_agent(
        bg_task,
        default_intent=default_intent,
      )

  def _task_correlation_payload(self, entry: TaskEntry) -> Dict[str, Any]:
    return _runner_attr(self, "_task_correlation_payload", _task_correlation_payload)(
      entry,
      runner_id=self._runner_id,
      role=self._role,
    )

  async def _append_task_completed_event(
    self,
    entry: TaskEntry,
    final_state: TaskState,
    *,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
  ) -> None:
    payload = _runner_attr(
      self,
      "_task_completed_event_payload",
      _task_completed_event_payload,
    )(
        entry,
        final_state,
        correlation_payload=self._task_correlation_payload(entry),
        completed_at=_runner_attr(self, "time", time).time(),
    )
    payload["result"] = result
    payload["error"] = error
    await self._append_durable_event(payload)

  async def _prepare_and_start_background_task(
    self,
    *,
    entry: TaskEntry,
    tool_input: Dict[str, Any],
    handler: BackgroundTaskHandler,
    agent_name: str | None,
    parent_turn_id: str | None,
    on_complete: BackgroundTaskCallback | None,
    on_before_start: Callable[[], None] | None,
    required_skill_lifecycle: dict[str, Any] | None,
    workflow_task_metadata: WorkflowTaskMetadata | None,
    required_skill_result_event_factory: (
      RequiredSkillResultEventFactory | None
    ),
    required_skill_result_projector: (
      RequiredSkillResultProjector | None
    ),
    original_task_id: str | None,
    capability_bind_receipt: dict[str, str],
    admitted_task: AdmittedTask | None,
    parent_result_policy: ParentResultPolicy | None,
    reconcile_registration: bool,
    validate_resume_source: bool,
    resume_source_entry: TaskEntry | None,
  ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if admitted_task is not None:
      if not isinstance(admitted_task, AdmittedTask):
        raise TypeError("admitted_task must be AdmittedTask")
      if admitted_task.attempt.physical_task_id != entry.task_id:
        raise ValueError(
          "admitted task physical_task_id must equal reserved task ID"
        )
      if entry.admitted_task is not None and entry.admitted_task != admitted_task:
        raise RuntimeError(
          "reserved task identity was reused with conflicting admission"
        )
      entry.admitted_task = admitted_task
      entry.metadata["admitted_task"] = admitted_task.model_dump(mode="json")
    if parent_result_policy is not None:
      if not isinstance(parent_result_policy, ParentResultPolicy):
        raise TypeError("parent_result_policy must be ParentResultPolicy")
      entry.parent_result_policy = parent_result_policy
      entry.metadata["parent_result_policy"] = parent_result_policy.model_dump(
        mode="json"
      )
    call_index = _runner_attr(
      self,
      "_prepare_background_task_registration",
      _prepare_background_task_registration,
    )(
      entry,
      tool_input=tool_input,
      capability_bind_receipt=capability_bind_receipt,
      owner_runner_id=self._runner_id,
      owner_role=self._role,
      sub_agent_id_for_call_index=lambda call_index: _runner_attr(
        self,
        "_derive_sub_agent_id",
        _derive_sub_agent_id,
      )(self._gateway_session_id, call_index),
      parent_turn_id=(
        parent_turn_id
        if parent_turn_id is not None
        else self._parent_turn_id
      ),
      original_task_id=original_task_id,
      required_skill_lifecycle=required_skill_lifecycle,
      workflow_task_metadata=workflow_task_metadata,
    )
    self._attach_required_skill_result_settlement(
      entry,
      requested_lifecycle=required_skill_lifecycle,
      event_factory=required_skill_result_event_factory,
      projector=required_skill_result_projector,
    )
    if on_complete is not None and entry.completion_callback is None:
      entry.completion_callback = on_complete

    durable_existing: TaskEntry | None = None
    registration_state = getattr(
      entry,
      "registration_persistence_state",
      "not_started",
    )
    registration_committed = registration_state == "committed"

    async def _reject_if_resume_source_changed() -> tuple[
      dict[str, Any] | None,
      dict[str, Any] | None,
    ] | None:
      if not validate_resume_source or original_task_id is None:
        return None
      source_error = _resume_source_admission_error(
        (
          self._task_registry.get(original_task_id)
          or resume_source_entry
        ),
        original_task_id=original_task_id,
      )
      if source_error is None:
        return None
      if not registration_committed:
        self._task_registry.discard_unstarted(
          entry.task_id,
          expected=entry,
        )
        return None, source_error
      detail = (
        f"{source_error['code']}: {source_error['message']}"
      )[:2000]
      canonical = _terminal_task_result(
        entry,
        status="failed",
        reason=f"resume_abandoned: {detail}",
      )
      if canonical is not None:
        entry.task_result = canonical
        rejected_result = canonical.model_dump(mode="json")
      else:
        rejected_result = {
          "status": "failed",
          "reason": "resume_abandoned",
          "detail": detail,
        }
      await self._finalize_background_agent(
        entry,
        final_state=TaskState.FAILED,
        result=rejected_result,
        error=None,
      )
      return None, source_error

    if (
      reconcile_registration
      and registration_state == "uncertain"
    ):
      durable_existing = await self._lookup_task_in_log(entry.task_id)
      if durable_existing is None:
        entry.registration_persistence_state = "not_started"
      else:
        registration_committed = True

    if durable_existing is not None and original_task_id is not None:
      if durable_existing.original_task_id != original_task_id:
        conflict = ResumeSuccessorConflictError(
          task_id=entry.task_id,
          original_task_id=original_task_id,
          existing=durable_existing,
        )
        return None, {
          "code": "resume_successor_conflict",
          "message": str(conflict),
        }
    if durable_existing is not None and original_task_id is not None:
      if durable_existing.state in {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.KILLED,
      }:
        entry.registration_persistence_state = "committed"
        entry.registration_persistence_error = None
        entry.completion_persistence_state = "committed"
        entry.completion_persistence_error = None
        await self._settle_and_commit_background_completion_live(
          entry,
          final_state=durable_existing.state,
          result=durable_existing.result,
          error=durable_existing.error,
        )
        return self._background_task_payload(entry), None

    source_rejection = await _reject_if_resume_source_changed()
    if source_rejection is not None:
      return source_rejection

    if not registration_committed:
      entry.registration_persistence_state = "in_flight"
      try:
        await self._append_durable_event(
          _runner_attr(
            self,
            "_task_registered_event_payload",
            _task_registered_event_payload,
          )(
            entry,
            correlation_payload=self._task_correlation_payload(entry),
            agent_name=agent_name,
            parent_session_id=self._gateway_session_id,
          )
        )
      except BaseException as exc:
        entry.registration_persistence_state = "uncertain"
        entry.registration_persistence_error = (
          f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"
        )
        if not reconcile_registration:
          raise
        try:
          durable_existing = await self._lookup_task_in_log(entry.task_id)
        except BaseException as lookup_exc:
          raise exc from lookup_exc
        if durable_existing is None:
          if original_task_id is None:
            self._task_registry.discard_unstarted(
              entry.task_id,
              expected=entry,
            )
            return None, {
              "code": "background_registration_failed",
              "message": (
                f"Could not durably register background task "
                f"{entry.task_id}: {exc}"
              ),
            }
          raise
        if (
          original_task_id is not None
          and durable_existing.original_task_id != original_task_id
        ):
          conflict = ResumeSuccessorConflictError(
            task_id=entry.task_id,
            original_task_id=original_task_id,
            existing=durable_existing,
          )
          return None, {
            "code": "resume_successor_conflict",
            "message": str(conflict),
          }
      entry.registration_persistence_state = "committed"
      entry.registration_persistence_error = None
      registration_committed = True

    if durable_existing is not None and durable_existing.state in {
      TaskState.COMPLETED,
      TaskState.FAILED,
      TaskState.KILLED,
    }:
      entry.completion_persistence_state = "committed"
      entry.completion_persistence_error = None
      await self._settle_and_commit_background_completion_live(
        entry,
        final_state=durable_existing.state,
        result=durable_existing.result,
        error=durable_existing.error,
      )
      return self._background_task_payload(entry), None

    source_rejection = await _reject_if_resume_source_changed()
    if source_rejection is not None:
      return source_rejection

    if entry.termination_intent is not None:
      result = self._background_termination_result(
        entry,
        entry.termination_intent,
      )
      final_state = (
        _task_state_from_task_result(entry.task_result)
        if entry.task_result is not None
        else TaskState.KILLED
      )
      await self._finalize_background_agent(
        entry,
        final_state=final_state,
        result=result,
        error=None,
      )
      return self._background_task_payload(entry), None

    if not getattr(entry, "start_hook_invoked", False):
      entry.start_hook_invoked = True
      _runner_attr(
        self,
        "_call_before_background_task_start_hook",
        _call_before_background_task_start_hook,
      )(
        on_before_start,
        log_session_id=lambda: self._sid,
        logger=log,
      )

    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    worker_coro = self._run_background_agent(
      entry,
      _runner_attr(
        self,
        "_entry_aware_background_handler",
        _entry_aware_background_handler,
      )(handler, entry),
      dict(tool_input),
      call_index,
      on_complete=on_complete,
    )
    try:
      worker_task = asyncio_module.create_task(
        worker_coro,
        name=entry.task_id,
      )
    except BaseException:
      worker_coro.close()
      raise
    entry.asyncio_task = worker_task
    try:
      self._task_registry.transition(entry.task_id, TaskState.RUNNING)
    except BaseException:
      entry.asyncio_task = None
      worker_task.cancel()
      worker_task.add_done_callback(
        self._consume_background_task_exception
      )
      raise

    return _runner_attr(
      self,
      "_background_task_started_result",
      _background_task_started_result,
    )(
      entry,
      agent_name=agent_name,
    ), None

  async def _register_background_task(
    self,
    *,
    tool_input: Dict[str, Any],
    handler: BackgroundTaskHandler,
    task_type: str = "background_agent",
    agent_name: str | None = None,
    parent_turn_id: str | None = None,
    on_complete: BackgroundTaskCallback | None = None,
    on_before_start: Callable[[], None] | None = None,
    required_skill_lifecycle: dict[str, Any] | None = None,
    workflow_task_metadata: WorkflowTaskMetadata | None = None,
    required_skill_result_event_factory: (
      RequiredSkillResultEventFactory | None
    ) = None,
    required_skill_result_projector: (
      RequiredSkillResultProjector | None
    ) = None,
    original_task_id: str | None = None,
    capability_bind_receipt: dict[str, str],
    admitted_task: AdmittedTask | None = None,
    parent_result_policy: ParentResultPolicy | None = None,
    task_id_override: str | None = None,
    terminal_replay: bool = False,
    validate_resume_source: bool = False,
    resume_source_entry: TaskEntry | None = None,
  ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if (
      not isinstance(task_type, str)
      or not task_type
      or task_type.strip() != task_type
    ):
      raise ValueError("task_type must be canonical non-empty text")
    capability_bind_receipt = _require_capability_bind_receipt(
      capability_bind_receipt
    )
    try:
      required_skill_lifecycle = (
        _normalize_required_skill_lifecycle(
          required_skill_lifecycle
        )
      )
    except ValueError as exc:
      raise ValueError(
        f"invalid required skill lifecycle: {exc}"
      ) from exc
    workflow_task_metadata = _normalize_workflow_task_metadata(
      workflow_task_metadata
    )
    if admitted_task is not None:
      if not isinstance(admitted_task, AdmittedTask):
        raise TypeError("admitted_task must be AdmittedTask")
      admitted_metadata = _workflow_metadata_for_admitted_task(admitted_task)
      if workflow_task_metadata is None:
        workflow_task_metadata = admitted_metadata
      elif admitted_metadata != workflow_task_metadata:
        raise ValueError(
          "workflow_task_metadata must exactly match admitted_task"
        )
    if getattr(self, "_background_delivery_grace_active", False):
      return None, {
        "code": "background_delivery_grace_active",
        "message": (
          "The run is past its normal turn limit and is completing bounded "
          "background-result delivery. New background tasks and resumes are "
          "not admitted during this delivery-only phase."
        ),
      }

    def _source_error() -> dict[str, Any] | None:
      if not validate_resume_source or original_task_id is None:
        return None
      return _resume_source_admission_error(
        (
          self._task_registry.get(original_task_id)
          or resume_source_entry
        ),
        original_task_id=original_task_id,
      )

    async def _replay_existing(
      entry: TaskEntry,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
      self._attach_required_skill_result_settlement(
        entry,
        requested_lifecycle=required_skill_lifecycle,
        event_factory=required_skill_result_event_factory,
        projector=required_skill_result_projector,
      )
      if (
        entry.completion_persistence_state == "committed"
        and entry.state in {
          TaskState.COMPLETED,
          TaskState.FAILED,
          TaskState.KILLED,
        }
      ):
        await self._ensure_required_skill_result_settled(entry)
        await self._ensure_agent_completion_published(entry)
      return self._background_task_payload(entry), None

    resolved_task_id = task_id_override
    if original_task_id is not None:
      source_error = _source_error()
      if source_error is not None:
        return None, source_error
      resume_task_id, resume_error = await _runner_attr(self, "_resume_task_id_override", _resume_task_id_override)(
        original_task_id,
        max_resume_chain_depth=(
          sys.maxsize
          if terminal_replay
          else self._max_resume_chain_depth
        ),
        resume_chain_depth=self._resume_chain_depth,
        resume_root=self._resume_root_task_id,
      )
      source_error = _source_error()
      if source_error is not None:
        return None, source_error
      if resume_error is not None:
        return None, resume_error
      if resolved_task_id is not None and resolved_task_id != resume_task_id:
        return None, {
          "code": "resume_successor_conflict",
          "message": (
            "Caller task_id_override does not match the deterministic "
            "resume successor identity"
          ),
        }
      resolved_task_id = resume_task_id

    if resolved_task_id is not None and original_task_id is not None:
      existing = self._task_registry.get(resolved_task_id)
      if existing is not None:
        source_error = _source_error()
        if source_error is not None:
          return None, source_error
        try:
          entry, _created = self._task_registry.claim_resume_successor(
            task_type,
            agent_name=agent_name,
            task_id=resolved_task_id,
            original_task_id=original_task_id,
          )
        except ResumeSuccessorConflictError as exc:
          return None, {
            "code": "resume_successor_conflict",
            "message": str(exc),
          }
        if entry.state != TaskState.PENDING:
          return await _replay_existing(entry)
        initialization_task = entry.initialization_task
        if (
          initialization_task is not None
          and not initialization_task.done()
        ):
          return await asyncio.shield(initialization_task)
      if (
        existing is None
        and getattr(self, "_agent_session_log", None) is not None
      ):
        durable_existing = await self._lookup_task_in_log(resolved_task_id)
        source_error = _source_error()
        if source_error is not None:
          return None, source_error
        if durable_existing is not None:
          if durable_existing.original_task_id != original_task_id:
            exc = ResumeSuccessorConflictError(
              task_id=resolved_task_id,
              original_task_id=original_task_id,
              existing=durable_existing,
            )
            return None, {
              "code": "resume_successor_conflict",
              "message": str(exc),
            }
          return await _replay_existing(durable_existing)

    admission_count = self._task_registry.admission_count
    if getattr(self, "_background_delivery_grace_active", False):
      return None, {
        "code": "background_delivery_grace_active",
        "message": (
          "The run is past its normal turn limit and is completing bounded "
          "background-result delivery. New background tasks and resumes are "
          "not admitted during this delivery-only phase."
        ),
      }
    if resolved_task_id is not None:
      reserved_successor = self._task_registry.get(resolved_task_id)
      if (
        reserved_successor is not None
        and reserved_successor.state == TaskState.PENDING
      ):
        admission_count -= 1
    limit_error = _runner_attr(self, "_background_task_limit_error", _background_task_limit_error)(
      admission_count=admission_count,
      max_background_tasks=self._max_background_tasks,
    )
    if limit_error is not None:
      return None, limit_error
    if not self._task_registry.notification_retrieval_capacity_available(
      admission_count=admission_count,
    ):
      return None, {
        "code": "background_notification_retrieval_required",
        "message": (
          "Omitted background results have filled bounded retrieval "
          "retention. Call get_background_result with task_id='*' to "
          "list eligible task IDs, then retrieve each exact ID before "
          "starting more background tasks."
        ),
        "pending_omitted_results": (
          self._task_registry.pending_notification_retrieval_count
        ),
        "retention_limit": (
          self._task_registry.notification_retrieval_retention_limit
        ),
      }

    if resolved_task_id is not None and original_task_id is not None:
      source_error = _source_error()
      if source_error is not None:
        return None, source_error
      try:
        entry, created = self._task_registry.claim_resume_successor(
          task_type,
          agent_name=agent_name,
          task_id=resolved_task_id,
          original_task_id=original_task_id,
        )
      except ResumeSuccessorConflictError as exc:
        return None, {
          "code": "resume_successor_conflict",
          "message": str(exc),
        }
      if not created and entry.state != TaskState.PENDING:
        return await _replay_existing(entry)
      initialization_task = entry.initialization_task
      if (
        initialization_task is not None
        and not initialization_task.done()
      ):
        return await asyncio.shield(initialization_task)

      initialization_task = asyncio.create_task(
        self._prepare_and_start_background_task(
          entry=entry,
          tool_input=tool_input,
          handler=handler,
          agent_name=agent_name,
          parent_turn_id=parent_turn_id,
          on_complete=on_complete,
          on_before_start=on_before_start,
          required_skill_lifecycle=required_skill_lifecycle,
          workflow_task_metadata=workflow_task_metadata,
          required_skill_result_event_factory=(
            required_skill_result_event_factory
          ),
          required_skill_result_projector=(
            required_skill_result_projector
          ),
          original_task_id=original_task_id,
          capability_bind_receipt=capability_bind_receipt,
          admitted_task=admitted_task,
          parent_result_policy=parent_result_policy,
          reconcile_registration=True,
          validate_resume_source=validate_resume_source,
          resume_source_entry=resume_source_entry,
        ),
        name=f"{entry.task_id}:initialize",
      )
      entry.initialization_task = initialization_task
      self._track_background_initialization(initialization_task)
      return await asyncio.shield(initialization_task)
    else:
      if resolved_task_id is not None:
        existing = self._task_registry.get(resolved_task_id)
        if existing is not None:
          if existing.state != TaskState.PENDING:
            return await _replay_existing(existing)
          initialization_task = existing.initialization_task
          if initialization_task is not None and not initialization_task.done():
            return await asyncio.shield(initialization_task)
          entry = existing
        else:
          durable_existing = await self._lookup_task_in_log(resolved_task_id)
          if durable_existing is not None:
            return await _replay_existing(durable_existing)
          entry = self._task_registry.register(
            task_type,
            agent_name=agent_name,
            task_id=resolved_task_id,
          )
      else:
        entry = self._task_registry.register(
          task_type,
          agent_name=agent_name,
        )

    initialization_task = asyncio.create_task(
      self._prepare_and_start_background_task(
        entry=entry,
        tool_input=tool_input,
        handler=handler,
        agent_name=agent_name,
        parent_turn_id=parent_turn_id,
        on_complete=on_complete,
        on_before_start=on_before_start,
        required_skill_lifecycle=required_skill_lifecycle,
        workflow_task_metadata=workflow_task_metadata,
        required_skill_result_event_factory=(
          required_skill_result_event_factory
        ),
        required_skill_result_projector=(
          required_skill_result_projector
        ),
        original_task_id=original_task_id,
        capability_bind_receipt=capability_bind_receipt,
        admitted_task=admitted_task,
        parent_result_policy=parent_result_policy,
        reconcile_registration=True,
        validate_resume_source=validate_resume_source,
        resume_source_entry=resume_source_entry,
      ),
      name=f"{entry.task_id}:initialize",
    )
    entry.initialization_task = initialization_task
    self._track_background_initialization(initialization_task)
    return await asyncio.shield(initialization_task)

  def cancel_background_task(self, task_id: str) -> bool:
    """Request cancellation of one live child task owned by this runner."""

    if (
      type(task_id) is not str
      or not task_id
      or task_id != task_id.strip()
    ):
      raise ValueError("task_id must be canonical non-empty text")
    return self._task_registry.kill(
      task_id,
      termination_intent="cancelled",
    )

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
    cursor = request.cursor

    def _auto_notify_error(task_ids: list[str]) -> dict[str, Any]:
      return {
        "code": "background_result_auto_notify",
        "message": (
          "Automatic notifications own current-run background delivery. "
          "Wait for and use the task notification; get_background_result is "
          "limited to historical or explicitly omitted payloads while "
          "auto-notify is active."
        ),
        "task_ids": task_ids,
      }

    await self._rebuild_task_registry_from_log()
    current_runner_id = getattr(self, "_runner_id", None)

    def _is_current_run_reconstruction(entry: TaskEntry) -> bool:
      metadata = (
        entry.metadata
        if isinstance(getattr(entry, "metadata", None), dict)
        else {}
      )
      return (
        bool(entry.reconstructed_from_log)
        and current_runner_id is not None
        and metadata.get("owner_runner_id") == current_runner_id
      )

    selected_wildcard_entries: list[TaskEntry] | None = None
    if self._background_notifications_enabled:
      candidate_entries = (
        self._task_registry.list_tasks()
        if task_id == "*"
        else [self._task_registry.get(task_id)]
      )
      if task_id == "*":
        selected_wildcard_entries = [
          entry
          for entry in candidate_entries
          if (
            entry.notification_delivery_state in {
              "payload_omitted",
              "queue_omitted",
            }
            or (
              entry.reconstructed_from_log
              and not _is_current_run_reconstruction(entry)
            )
          )
        ]
      blocked_task_ids = [
        entry.task_id
        for entry in candidate_entries
        if (
          entry is not None
          and (
            _is_current_run_reconstruction(entry)
            or
            entry.notification_delivery_state in {
              "queued",
              "delivered",
            }
            or (
              not entry.reconstructed_from_log
              and entry.notification_delivery_state not in {
                "payload_omitted",
                "queue_omitted",
              }
            )
          )
        )
      ]
      if blocked_task_ids and not selected_wildcard_entries:
        return None, _auto_notify_error(blocked_task_ids)

    if task_id == "*":
      selected_entries = (
        selected_wildcard_entries
        if selected_wildcard_entries is not None
        else self._task_registry.list_tasks()
      )
      await _runner_attr(
        self,
        "_wait_for_background_tasks",
        _wait_for_background_tasks,
      )(
        selected_entries,
        wait=wait,
        timeout=timeout,
        wait_fn=_runner_attr(self, "asyncio", asyncio).wait,
      )
      return {
        "tasks": [
          {
            "task_id": entry.task_id,
            "status": entry.state.value,
            "retrieval": (
              entry.notification_delivery_state
              if not entry.reconstructed_from_log
              else "historical"
            ),
          }
          for entry in selected_entries
        ],
        "eligible_count": len(selected_entries),
        "message": (
          "Retrieve one payload at a time with its exact task_id. "
          "If a result is paged, pass each opaque cursor back unchanged."
        ),
      }, None

    bg_task, error = await _runner_attr(self, "_background_result_task", _background_result_task)(
      task_id,
      registry_lookup=self._task_registry.get,
      log_lookup=self._lookup_task_in_log,
    )
    if error is not None or bg_task is None:
      return None, error
    if (
      self._background_notifications_enabled
      and _is_current_run_reconstruction(bg_task)
    ):
      return None, _auto_notify_error([task_id])

    await _runner_attr(self, "_wait_for_background_tasks", _wait_for_background_tasks)(
      [bg_task],
      wait=wait,
      timeout=timeout,
      wait_fn=_runner_attr(self, "asyncio", asyncio).wait,
    )
    page_chars = _background_result_page_content_chars()
    page_state = getattr(self, "_background_result_page_state", None)
    result: dict[str, Any] | None = None
    if cursor is not None:
      if (
        not isinstance(page_state, dict)
        or page_state.get("task_id") != task_id
        or page_state.get("cursor") != cursor
        or not isinstance(page_state.get("serialized_result"), str)
      ):
        return None, {
          "code": "invalid_background_result_cursor",
          "message": (
            "The opaque cursor is stale or belongs to another result. "
            "Omit cursor to restart this exact task from the first page."
          ),
        }
      if (
        page_state.get("notification_generation")
        != bg_task.notification_generation
      ):
        self._background_result_page_state = None
        return None, {
          "code": "background_result_changed",
          "message": (
            "The retained result changed during paging. Omit cursor to restart "
            "from the new canonical payload."
          ),
        }
      serialized_result = page_state["serialized_result"]
      offset = int(page_state.get("next_offset", 0))
    else:
      self._background_result_page_state = None
      result = self._background_task_payload(bg_task)
      try:
        serialized_result = json.dumps(
          result,
          ensure_ascii=True,
          allow_nan=False,
          separators=(",", ":"),
          sort_keys=True,
        )
      except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        return None, {
          "code": "background_result_unserializable",
          "message": (
            f"Background result {task_id} is not canonical JSON and remains "
            "retained for operator recovery."
          ),
        }
      if len(serialized_result) > _BACKGROUND_RESULT_SERIALIZED_MAX_CHARS:
        return None, {
          "code": "background_result_contract_too_large",
          "message": (
            f"Background result {task_id} exceeds the bounded delivery "
            "contract and remains retained for operator recovery. Large child "
            "evidence must be moved to a durable artifact and represented by "
            "bounded receipts or references."
          ),
          "max_chars": _BACKGROUND_RESULT_SERIALIZED_MAX_CHARS,
          "actual_chars": len(serialized_result),
        }
      if len(serialized_result) <= page_chars:
        response = dict(result)
        response.pop(_BACKGROUND_RESULT_ACK_RESULT_KEY, None)
        if bg_task.notification_delivery_state in {
          "payload_omitted",
          "queue_omitted",
        }:
          response[_BACKGROUND_RESULT_ACK_RESULT_KEY] = {
            "task_id": task_id,
            "notification_generation": bg_task.notification_generation,
          }
        return response, None
      offset = 0

    end = min(len(serialized_result), offset + page_chars)
    complete = end >= len(serialized_result)
    next_cursor: str | None = None
    if complete:
      self._background_result_page_state = None
    else:
      next_cursor = secrets.token_urlsafe(24)
      self._background_result_page_state = {
        "task_id": task_id,
        "cursor": next_cursor,
        "next_offset": end,
        "serialized_result": serialized_result,
        "notification_generation": bg_task.notification_generation,
      }
    response = {
      "task_id": task_id,
      "status": "result_page",
      "encoding": "json",
      "cursor_offset": offset,
      "next_cursor": next_cursor,
      "complete": complete,
      "total_chars": len(serialized_result),
      "content": serialized_result[offset:end],
      "message": (
        "Concatenate content pages in cursor_offset order and parse the "
        "complete JSON text. Pass next_cursor back unchanged with this exact "
        "task_id; omit cursor to restart from page one."
      ),
    }
    if (
      complete
      and bg_task.notification_delivery_state in {
        "payload_omitted",
        "queue_omitted",
      }
    ):
      response[_BACKGROUND_RESULT_ACK_RESULT_KEY] = {
        "task_id": task_id,
        "notification_generation": bg_task.notification_generation,
      }
    return response, None

  def _background_delivery_grace_obligations(
    self,
  ) -> dict[tuple[str, int, str], int]:
    """Return bounded credits for each currently live delivery obligation."""

    page_chars = _background_result_page_content_chars()
    max_pages_per_result = max(
      1,
      math.ceil(
        _BACKGROUND_RESULT_SERIALIZED_MAX_CHARS / page_chars
      ),
    )
    obligations: dict[tuple[str, int, str], int] = {}
    peek = getattr(self._notification_queue, "peek", None)
    queued_notifications = (
      list(peek())
      if callable(peek)
      else []
    )
    keyed_notification_count = 0
    for notification in queued_notifications:
      queue_token = getattr(notification, "_queue_token", None)
      if isinstance(queue_token, int) and not isinstance(queue_token, bool):
        keyed_notification_count += 1
        obligations[
          (str(notification.task_id), queue_token, "notification")
        ] = 2
    unkeyed_notifications = max(
      0,
      int(self._notification_queue.pending_count)
      - keyed_notification_count,
    )
    if unkeyed_notifications:
      obligations[("notification_queue", 0, "unkeyed")] = math.ceil(
        unkeyed_notifications / _MAX_NOTIFICATIONS_PER_TURN
      )

    pending_acknowledgements = getattr(
      self,
      "_pending_background_result_acks",
      {},
    )
    acknowledgement_keys = {
      (acknowledgement[0], acknowledgement[1])
      for acknowledgement in (
        pending_acknowledgements.values()
        if isinstance(pending_acknowledgements, dict)
        else ()
      )
      if (
        isinstance(acknowledgement, tuple)
        and len(acknowledgement) == 2
        and isinstance(acknowledgement[0], str)
        and isinstance(acknowledgement[1], int)
        and not isinstance(acknowledgement[1], bool)
      )
    }
    recovery_pairs = {
      pair
      for pair in getattr(
        self,
        "_background_ack_recovery_pairs",
        set(),
      )
      if (
        isinstance(pair, tuple)
        and len(pair) == 2
        and isinstance(pair[0], str)
        and isinstance(pair[1], int)
        and not isinstance(pair[1], bool)
      )
    }
    for task_id, notification_generation in acknowledgement_keys:
      obligations[
        (
          task_id,
          notification_generation,
          (
            "recovery_acknowledgement"
            if (task_id, notification_generation)
            in recovery_pairs
            else "acknowledgement"
          ),
        )
      ] = 1

    for entry in self._task_registry.list_tasks():
      if entry.notification_delivery_state not in {
        "payload_omitted",
        "queue_omitted",
      }:
        continue
      obligation_identity = (
        entry.task_id,
        entry.notification_generation,
      )
      if obligation_identity in acknowledgement_keys:
        continue
      # One directory/notification turn, bounded exact pages, and one
      # synthesis turn. The run loop grants this key only once.
      obligations[
        (
          entry.task_id,
          entry.notification_generation,
          (
            "ack_recovery"
            if obligation_identity in recovery_pairs
            else "exact_retrieval"
          ),
        )
      ] = max_pages_per_result + 2
    return obligations

  def _background_delivery_grace_credit_limit(self) -> int:
    """Return one fixed ceiling for the frozen delivery-only epoch.

    One retained task may publish an interrupted generation followed by its
    terminal generation. Each generation can require an inline notification,
    exact paged retrieval, a primary acknowledgement, and one recovery cycle
    when compaction makes that acknowledgement unavailable. The ceiling covers
    both generations for every retained task; phase-keyed credits still ensure
    each individual obligation is granted only once.
    """

    page_chars = _background_result_page_content_chars()
    max_pages_per_result = max(
      1,
      math.ceil(
        _BACKGROUND_RESULT_SERIALIZED_MAX_CHARS / page_chars
      ),
    )
    retention_limit = max(
      1,
      int(
        getattr(
          self._task_registry,
          "notification_retrieval_retention_limit",
          1,
        )
      ),
    )
    queue_limit = max(
      1,
      int(getattr(self._notification_queue, "_max_pending", 1)),
    )
    per_generation_credits = (
      2  # Inline notification plus one committal synthesis turn.
      + (max_pages_per_result + 2)  # Exact retrieval and synthesis.
      + 1  # Primary acknowledgement.
      + (max_pages_per_result + 2)  # One recovery retrieval cycle.
      + 1  # Recovery acknowledgement.
    )
    return (
      2 * retention_limit * per_generation_credits
      + 2 * queue_limit
    )

  async def _shutdown_background_tasks(self, was_cancelled: bool) -> None:
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    termination_intent: TerminationIntent = (
      "cancelled" if was_cancelled else "killed"
    )
    initializing_entries = [
      entry
      for entry in self._task_registry.list_tasks(
        state=TaskState.PENDING,
      )
      if (
        getattr(entry, "initialization_task", None) is not None
        and not entry.initialization_task.done()
      )
    ]
    for entry in initializing_entries:
      self._task_registry.kill(
        entry.task_id,
        termination_intent=termination_intent,
      )
    initialization_tasks = {
      entry.initialization_task
      for entry in initializing_entries
      if entry.initialization_task is not None
    }
    if initialization_tasks:
      await asyncio_module.wait(
        initialization_tasks,
        timeout=max(
          0.0,
          float(
            getattr(
              self,
              "_background_cancel_drain_timeout_seconds",
              _BACKGROUND_CANCEL_DRAIN_TIMEOUT_SECONDS,
            )
          ),
        ),
      )

    running_entries = self._task_registry.list_tasks(state=TaskState.RUNNING)
    if not running_entries:
      return
    pending = _runner_attr(self, "_background_asyncio_tasks", _background_asyncio_tasks)(running_entries)

    try:
      if was_cancelled and pending:
        def _cancel_with_parent(task_id: str) -> bool:
          return self._task_registry.kill(
            task_id,
            termination_intent="cancelled",
          )

        await _runner_attr(self, "_drain_cancelled_background_tasks", _drain_cancelled_background_tasks)(
          running_entries,
          pending,
          kill_task=_cancel_with_parent,
          wait_fn=asyncio_module.wait,
          timeout=max(
            0.0,
            float(
              getattr(
                self,
                "_background_cancel_drain_timeout_seconds",
                _BACKGROUND_CANCEL_DRAIN_TIMEOUT_SECONDS,
              )
            ),
          ),
        )
        return

      if pending:
        await _runner_attr(self, "_drain_still_pending_background_tasks", _drain_still_pending_background_tasks)(
          running_entries,
          pending,
          kill_task=self._task_registry.kill,
          wait_fn=asyncio_module.wait,
          wait_timeout=max(
            0.0,
            float(
              getattr(
                self,
                "_background_grace_wait_timeout_seconds",
                _BACKGROUND_GRACE_WAIT_TIMEOUT_SECONDS,
              )
            ),
          ),
          drain_timeout=max(
            0.0,
            float(
              getattr(
                self,
                "_background_kill_drain_timeout_seconds",
                _BACKGROUND_KILL_DRAIN_TIMEOUT_SECONDS,
              )
            ),
          ),
        )
    finally:
      await self._reconcile_background_agents(
        running_entries,
        default_intent=termination_intent,
      )
