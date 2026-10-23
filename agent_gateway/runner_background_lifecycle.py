from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import secrets
import sys
import time
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, TypeGuard, TypeVar, Union, cast

from agent_workflow_contracts import (
  AdmittedTask,
  ParentResultPolicy,
  TaskResult,
  TaskResultProvenance,
  TaskResultRef,
  UsageObservation,
  terminal_task_result,
)

from .runner_background_tasks import (
  _BACKGROUND_RESULT_ACK_RESULT_KEY,
  REQUIRED_SKILL_LIFECYCLE_METADATA_KEY,
  background_asyncio_tasks as _background_asyncio_tasks,
  background_elapsed_seconds as _background_elapsed_seconds,
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
  normalize_required_skill_lifecycle as _normalize_required_skill_lifecycle,
  normalize_workflow_task_metadata as _normalize_workflow_task_metadata,
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
  workflow_obstruction_blocks_settlement,
  workflow_owns_terminal_notification as _workflow_owns_terminal_notification,
)
from .child_completion import materialize_child_completion_envelope
from .runner_introspection import (
  derive_sub_agent_id as _derive_sub_agent_id,
  mark_exception_traceback_logged,
)
from .runner_notifications import (
  build_notification_reminder as _build_notification_reminder,
  consume_notifications as _consume_notifications,
  inject_system_prompt_reminder as _inject_system_prompt_reminder,
  notification_delivery_set as _notification_delivery_set,
)
from .runner_run_loop_defaults import MAX_NOTIFICATIONS_PER_TURN as _MAX_NOTIFICATIONS_PER_TURN
from .runner_session_events import build_agent_completion_event as _build_agent_completion_event
from .runner_session_lifecycle import _exact_value_match, _runner_attr
from .session import GatewaySession
from .research_file_activity_lock import try_acquire_research_file_activity
from .events import AgentCompletionEvent, TypedEvent, event_from_dict
from .skill_result_events import build_skill_result_captured_event
from .runner_state import BackgroundTask
from .skill_lifecycle import TopLevelSkillLifecycleMetadata
from .sub_agent_narrative_result import (
  final_child_visible_text,
  task_result_from_execution,
)
from .sub_agent_result_contract import _usage_observation
from .sub_agent_result_evidence import (
  SubAgentResultEvidence,
  collect_sub_agent_result_evidence,
  merge_sub_agent_result_evidence,
)
from .sub_agent_skill_state import (
  latest_successful_declared_terminal_tool_result,
  persist_expectation_doors,
  terminal_result_doors,
)
from .tool_result_compaction import model_tool_result_max_chars
from .task_registry import (
  InterruptionReason,
  NotificationQueue,
  RequiredSkillResultMarkerValidationError,
  ResumeSuccessorConflictError,
  TaskEntry,
  TaskRegistry,
  TaskState,
  TerminationIntent,
  resolve_required_skill_result_marker,
  task_state_for_result,
)

if TYPE_CHECKING:
  from .agent_session_log import AgentSessionLog
  from .containment_settings import ContainmentSettings
  from .event_log import EventLog
  from .tool_dispatcher_approval_lifecycle import ApprovalPark


class _RunResearchFileActivityLease:
  """One run's exact-file leases, retained by its existing teardown owner."""

  def __init__(self) -> None:
    self._leases: dict[int | None, Any] = {}

  def bind(self, lease: Any, research_file_id: int | None) -> None:
    self._leases[research_file_id] = lease

  def ensure(self, workspace_dir: Path, research_file_id: int) -> bool:
    if research_file_id in self._leases:
      return True
    lease = try_acquire_research_file_activity(
      workspace_dir,
      research_file_id=research_file_id,
      exclusive=False,
    )
    if lease is None:
      return False
    self._leases[research_file_id] = lease
    return True

  def release(self) -> None:
    leases, self._leases = self._leases, {}
    for lease in reversed(tuple(leases.values())):
      lease.release()


BackgroundTaskHandler = Callable[..., Awaitable[Tuple[Optional[Any], Optional[Dict[str, Any]]]]]
BackgroundTaskCallback = Callable[[BackgroundTask | TaskEntry], Awaitable[None] | None]
RequiredSkillResultEventFactory = Callable[
  [
    TaskEntry,
    dict[str, Any] | None,
    dict[str, Any] | None,
  ],
  dict[str, Any] | None,
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
# Typical serialized-result size; sizes the delivery-grace page credits only.
# A larger result is delivered in more pages, never refused.
_BACKGROUND_RESULT_SERIALIZED_MAX_CHARS = 32_768 + 4_096


class StrictBackgroundTaskDrainUnavailable(RuntimeError):
  """A deletion-specific drain could not prove every child writer stopped."""


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
  if source.state not in {TaskState.INTERRUPTED, TaskState.PARKED}:
    return {
      "code": "not_interrupted",
      "message": f"Task {original_task_id} is not parked or interrupted",
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

def _all_agent_completion_events(
  events: list[TypedEvent],
) -> TypeGuard[list[AgentCompletionEvent]]:
  return all(
    isinstance(event, AgentCompletionEvent)
    for event in events
  )


def _runner_module_attr(name: str, fallback: Any) -> Any:
  module = sys.modules.get("agent_gateway.runner")
  if module is None:
    return fallback
  return getattr(module, name, fallback)


_TaskCallback = TypeVar(
  "_TaskCallback", bound=Callable[[TaskEntry], Awaitable[None] | None],
)


class _TaskCallbackFanout:
  """Run every attached owner's callback once, in attachment order.

  One owner's failure never withholds a later owner's callback; the first
  failure is raised once every member has run, to the caller that already
  owns callback-failure handling.
  """

  def __init__(
    self,
    callbacks: tuple[Callable[[TaskEntry], Awaitable[None] | None], ...],
  ) -> None:
    self.callbacks = callbacks

  async def __call__(self, entry: TaskEntry) -> None:
    first_failure: Exception | None = None
    for callback in self.callbacks:
      try:
        outcome = callback(entry)
        if inspect.isawaitable(outcome):
          await outcome
      except Exception as exc:
        if first_failure is None:
          first_failure = exc
    if first_failure is not None:
      raise first_failure


def _attach_task_callback(
  existing: _TaskCallback | _TaskCallbackFanout | None,
  added: _TaskCallback | _TaskCallbackFanout | None,
) -> _TaskCallback | _TaskCallbackFanout | None:
  """Union attached owners by equality, preserving their attachment order."""
  callbacks: list[Callable[[TaskEntry], Awaitable[None] | None]] = []

  def collect(callback: Callable[[TaskEntry], Awaitable[None] | None]) -> None:
    if isinstance(callback, _TaskCallbackFanout):
      for member in callback.callbacks:
        collect(member)
    elif not any(member == callback for member in callbacks):
      callbacks.append(callback)

  for callback in (existing, added):
    if callback is not None:
      collect(callback)
  if not callbacks:
    return None
  if len(callbacks) == 1:
    if existing is not None and not isinstance(existing, _TaskCallbackFanout):
      return existing
    if added is not None and not isinstance(added, _TaskCallbackFanout):
      return added
  if isinstance(existing, _TaskCallbackFanout) and tuple(callbacks) == existing.callbacks:
    return existing
  if isinstance(added, _TaskCallbackFanout) and tuple(callbacks) == added.callbacks:
    return added
  return _TaskCallbackFanout(tuple(callbacks))


class RunnerBackgroundLifecycleMixin:
  # The runner's own admitted job, when this run is an independent child.
  _admitted_self_task: TaskEntry | None = None
  _background_grace_wait_timeout_seconds: float = _BACKGROUND_GRACE_WAIT_TIMEOUT_SECONDS
  _background_kill_drain_timeout_seconds: float = _BACKGROUND_KILL_DRAIN_TIMEOUT_SECONDS

  if TYPE_CHECKING:
    _agent_session_log: AgentSessionLog | None
    _approval_park: ApprovalPark | None
    _full_session_id: str
    _log: EventLog
    _sub_agent_id: str | None
    _terminal_closure_event: Dict[str, Any] | None
    _background_notifications_enabled: bool
    _gateway_session_id: str
    _notification_queue: NotificationQueue
    _parent_turn_id: str | None
    _role: str
    _runner_id: str | None
    _sid: str
    _task_registry: TaskRegistry
    _containment: ContainmentSettings
    _workspace_dir: str | None

    def _append(self, event: Dict[str, Any]) -> Any | None: ...

    async def _append_durable_event(
      self,
      event: Dict[str, Any],
    ) -> Any | None: ...

    async def _append_task_parked(
      self,
      task_id: str,
      reason: str,
    ) -> Any | None: ...

    async def _confirm_durable_skill_event(
      self,
      event: Dict[str, Any],
    ) -> Dict[str, Any] | None: ...

    async def _lookup_task_in_log(
      self,
      task_id: str,
    ) -> TaskEntry | None: ...

    def _lookup_task_in_log_current_sync(
      self,
      task_id: str,
    ) -> TaskEntry | None: ...

    def _operator_pause_requested(self) -> bool: ...

    async def _rebuild_task_registry_from_log(self) -> None: ...

    def _release_write_lease(self) -> None: ...

  def _ensure_sub_agent_semaphore(self) -> asyncio.Semaphore:
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    semaphore: asyncio.Semaphore = _runner_attr(self, "_ensure_sub_agent_semaphore", _ensure_sub_agent_semaphore)(
      self._sub_agent_semaphore,
      self._containment.max_concurrent_children,
      semaphore_factory=asyncio_module.Semaphore,
    )
    self._sub_agent_semaphore = semaphore
    return semaphore

  @staticmethod
  def _background_timeout_value(raw_timeout: Any) -> float:
    return _runner_module_attr("_background_timeout_value", _background_timeout_value)(raw_timeout)

  def _background_elapsed_seconds(self, bg_task: BackgroundTask | TaskEntry) -> int:
    return _runner_attr(self, "_background_elapsed_seconds", _background_elapsed_seconds)(
      bg_task,
      now=_runner_attr(self, "time", time).time(),
    )

  async def attach_background_task(
    self,
    task_id: str,
    *,
    on_complete: Callable[[TaskEntry], Awaitable[None] | None],
    on_park: Callable[[TaskEntry], Awaitable[None]] | None,
  ) -> TaskEntry | None:
    """Attach lifecycle handling without starting or resetting a task.

    Callbacks already on the entry (for example the activity release a
    resume installed) keep running; the attaching owner's callbacks run after
    them, once each, however many times the same owner attaches.
    """
    entry = self._task_registry.get(task_id)
    if entry is None:
      reconstructed = await self._lookup_task_in_log(task_id)
      if reconstructed is None:
        return None
      entry = self._task_registry.adopt_reconstructed(reconstructed)
    entry.completion_callback = _attach_task_callback(
      entry.completion_callback, on_complete,
    )
    entry.park_callback = _attach_task_callback(entry.park_callback, on_park)
    return entry

  async def _task_entry_for_chain(self, task_id: str) -> TaskEntry | None:
    entry = self._task_registry.get(task_id)
    if entry is not None:
      return entry
    return await self._lookup_task_in_log(task_id)

  async def _is_resume_successor(
    self,
    successor: AdmittedTask,
    original_task_id: str,
  ) -> bool:
    """Authorize only this admitted attempt's exact same-tenant resume ancestry."""
    snapshot = successor.execution_snapshot
    context = snapshot.admitted_context if snapshot is not None else None
    tenant = context.tenant if context is not None else None
    if tenant is None or not tenant.storage_id:
      return False
    successor_task_id = successor.attempt.physical_task_id
    if not original_task_id or original_task_id == successor_task_id:
      return False
    current_task_id = successor_task_id
    seen: set[str] = set()
    while current_task_id not in seen:
      seen.add(current_task_id)
      entry = await self._task_entry_for_chain(current_task_id)
      if entry is None or entry.admitted_task is None:
        return False
      admitted = entry.admitted_task
      snapshot = admitted.execution_snapshot
      context = snapshot.admitted_context if snapshot is not None else None
      current_tenant = context.tenant if context is not None else None
      predecessor = admitted.attempt.resume_of_task_id
      if (
        entry.task_id != current_task_id
        or admitted.attempt.physical_task_id != current_task_id
        or admitted.logical_task != successor.logical_task
        or current_tenant is None
        or current_tenant.storage_id != tenant.storage_id
        or entry.original_task_id != predecessor
        or predecessor in seen
        or (current_task_id == successor_task_id and admitted != successor)
      ):
        return False
      if current_task_id == original_task_id:
        return True
      if predecessor is None or admitted.attempt.restart_of_attempt_id is not None:
        return False
      current_task_id = predecessor
    return False

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
      if getattr(bg_task, "state", None) in {TaskState.INTERRUPTED, TaskState.PARKED}
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

  def _notification_delivery_set(self) -> Tuple[Any, ...]:
    """The notification objects this request's reminder renders (A-M7)."""
    return _runner_attr(
      self,
      "_notification_delivery_set",
      _notification_delivery_set,
    )(
      self._notification_queue,
      max_count=_runner_attr(
        self,
        "_MAX_NOTIFICATIONS_PER_TURN",
        _MAX_NOTIFICATIONS_PER_TURN,
      ),
    )

  def _settle_consumed_notifications(
    self,
    notifications: List[Any],
  ) -> None:
    """Mark consumed notifications delivered and release their boundaries.

    Runs only under the parent's ack path, so it is the §5.3 guard's
    "acked" fact: a workflow boundary notification carries a
    delivered-callback that releases its settlement obstruction
    (T2-I04, D-T2-4a). Never called at reminder-build time — a
    rendered-but-unanswered boundary must keep obstructing (T3-I02).
    """
    registry = getattr(self, "_task_registry", None)
    registry_get = getattr(registry, "get", None)
    if callable(registry_get):
      for notification in notifications:
        entry = self._task_registry.get(notification.task_id)
        if (
          entry is not None
          and entry.notification_delivery_state == "queued"
          and notification.notification_generation
          == entry.notification_generation
        ):
          entry.notification_delivery_state = "delivered"
    for notification in notifications:
      boundary_delivered = getattr(
        notification,
        "workflow_boundary_delivered",
        None,
      )
      if callable(boundary_delivered):
        try:
          boundary_delivered()
        except Exception:
          pass

  def _consume_notifications(self, max_count: int) -> int:
    peek = getattr(self._notification_queue, "peek", None)
    notifications = (
      list(self._notification_queue.peek(max_count=max_count))
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
    self._settle_consumed_notifications(notifications[:consumed])
    return consumed

  def _ack_delivered_notifications(self, delivered: Any) -> int:
    """Ack the notifications the delivering provider request rendered.

    Delivery is **at-least-once**, deliberately. The delivered set is
    in-memory state of one request; the durable carrier of a child's
    result is its ``agent_completion`` event, not the queue. A crash
    between rendering the reminder and acking it therefore redelivers the
    notification on the next run rather than losing it, and redelivery is
    deduplicated at build time: a child produces one queue entry per
    ``(task_id, notification_generation)``, and a workflow boundary is
    deduped by ``(run, phase, revision)`` in ``_boundary_notifications``.

    Acking here — not at build — is what keeps the §5.3 settlement guard
    honest: a boundary that was rendered but never answered still
    obstructs (T3-I02).
    """
    notifications = list(delivered or ())
    if not notifications:
      return 0
    drain_delivered = getattr(
      self._notification_queue,
      "drain_delivered",
      None,
    )
    if callable(drain_delivered):
      consumed = list(self._notification_queue.drain_delivered(notifications))
    else:
      # Duck-typed queues without identity draining keep the historical
      # front-slice contract.
      count = _runner_attr(
        self,
        "_consume_notifications",
        _consume_notifications,
      )(
        self._notification_queue,
        max_count=len(notifications),
      )
      consumed = notifications[:count]
    self._settle_consumed_notifications(consumed)
    return len(consumed)

  def _workflow_settlement_obstructions(self) -> tuple[Any, ...]:
    """Read the session's unsettled workflow obligations, failing open."""

    session: GatewaySession | None = getattr(
      self,
      "_gateway_session",
      None,
    )
    query = (
      session.workflow_settlement_obstruction
      if session is not None
      else None
    )
    if query is None:
      return ()
    try:
      return tuple(query())
    except Exception:
      return ()

  def _workflow_settlement_obstructed(self) -> bool:
    """Return whether any workflow obligation blocks clean settlement."""

    return any(
      workflow_obstruction_blocks_settlement(row)
      for row in self._workflow_settlement_obstructions()
    )

  def _unsettled_workflow_obstruction_count(self) -> int:
    return sum(
      1
      for row in self._workflow_settlement_obstructions()
      if workflow_obstruction_blocks_settlement(row)
    )

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
      # An unsettled workflow obligation holds the session open even with
      # no RUNNING registry task (§5.3, T2-I04): the authoring window and
      # a parked-but-unacked boundary have no running children.  A row's
      # wake handle (the in-process drive) joins the wait set so a run
      # settling terminally releases the hold without a notification.
      workflow_obstructed = False
      for row in self._workflow_settlement_obstructions():
        if not workflow_obstruction_blocks_settlement(row):
          continue
        workflow_obstructed = True
        wake = getattr(row, "wake", None)
        done = getattr(wake, "done", None)
        if (
          wake is not None
          and is_future(wake)
          and callable(done)
          and not done()
        ):
          live_tasks.add(wake)
      if not live_tasks and not workflow_obstructed:
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
    # Callers run this after releasing the finalization lock: a committed park
    # notifies its park callback here, so callback owners may take their own
    # locks without inverting against this task's finalization lock.
    if bg_task.state == TaskState.AWAITING_APPROVAL:
      await self._invoke_background_park_callback(bg_task)
      return
    callback = bg_task.completion_callback
    if (
      bg_task.completion_persistence_state != "committed"
      or bg_task.state not in {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.KILLED,
      }
    ):
      return
    if callback is None or bg_task.completion_callback_invoked:
      return
    bg_task.completion_callback_invoked = True
    try:
      maybe_awaitable = callback(bg_task)
      if asyncio.iscoroutine(maybe_awaitable):
        await maybe_awaitable
    except Exception:
      pass

  async def _invoke_background_park_callback(
    self,
    bg_task: TaskEntry,
  ) -> None:
    callback = bg_task.park_callback
    canonical = bg_task.task_result
    admitted = bg_task.admitted_task
    if (
      callback is None
      or bg_task.park_callback_invoked
      or bg_task.completion_persistence_state != "committed"
      or bg_task.state != TaskState.AWAITING_APPROVAL
      or canonical is None
      or canonical.execution.status != "awaiting_approval"
      or admitted is None
      or canonical.attempt.physical_task_id != bg_task.task_id
      or canonical.logical_task != admitted.logical_task
      or canonical.attempt != admitted.attempt
      or canonical.provenance.admitted_task_digest != admitted.admitted_task_digest
      or bg_task.error is not None
      or bg_task.result != canonical.model_dump(mode="json")
    ):
      return
    bg_task.park_callback_invoked = True
    try:
      await callback(bg_task)
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
          lifecycle=normalized,
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
  ) -> dict[str, Any] | None:
    if bg_task.pending_final_state is not None:
      result = bg_task.pending_result
      error = bg_task.pending_error
    else:
      result = bg_task.result
      error = bg_task.error
    event_factory = bg_task.required_skill_result_event_factory
    if event_factory is not None:
      event = event_factory(bg_task, result, error)
      if event is not None and not isinstance(event, dict):
        raise RuntimeError(
          "required skill result event factory must return an object"
        )
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
        declared_terminal_doors=persist_expectation_doors(
          bg_task.admitted_task
        ),
      )
    if event is None:
      return None
    event = dict(event)

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
      if event is None:
        return False
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
          TaskState.INTERRUPTED,
        }
        and bg_task.pending_final_state not in {
          TaskState.COMPLETED,
          TaskState.FAILED,
          TaskState.KILLED,
          TaskState.INTERRUPTED,
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

    async def _append_and_settle_detached_completion() -> None:
      # A late append's acknowledgement includes its live projection. A done
      # callback cannot order that projection before awaiters of this task.
      try:
        await self._append_task_completed_event(
          bg_task,
          final_state,
          result=result,
          error=error,
        )
      except BaseException as exc:
        self._record_completion_persistence_uncertainty(bg_task, exc)
        if bg_task.completion_finalizer_detached:
          await self._reconcile_detached_background_completion(bg_task)
        raise
      bg_task.completion_persistence_state = "committed"
      bg_task.completion_persistence_error = None
      if bg_task.completion_finalizer_detached:
        await self._commit_detached_background_completion(
          bg_task,
          final_state=final_state,
          result=result,
          error=error,
        )

    append_task = asyncio.create_task(
      _append_and_settle_detached_completion(),
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
      timeout_error = asyncio.TimeoutError(
        f"task completion persistence exceeded {timeout:.3f}s"
      )
      self._record_completion_persistence_uncertainty(
        bg_task,
        timeout_error,
      )
      raise timeout_error

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
    if not _all_agent_completion_events(typed):
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
    durable = await self._durable_agent_completion(bg_task.task_id)
    if durable is not None:
      if durable.task_result_ref != TaskResultRef.from_result(task_result):
        raise RuntimeError(
          f"Task {bg_task.task_id} completion replay conflicts with durable result identity"
        )
      # Publication owns its bytes. Re-materializing under today's policy
      # would rewrite history (and needlessly require the source content).
      bg_task.completion_envelope = durable
      return
    workspace_dir = self._workspace_dir
    envelope = materialize_child_completion_envelope(
      task_result,
      workspace_dir="" if workspace_dir is None else str(workspace_dir),
      principal_id=str(self._runner_id),
      policy=bg_task.parent_result_policy,
    )
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
      async with self._ensure_sub_agent_semaphore():
        handler_result, handler_error = await handler(tool_input, call_index=call_index)
      if bg_task.interruption_reason is not None and bg_task.termination_intent is None:
        # An accepted interrupt also wins when a handler consumes cancellation
        # or returns before the queued cancellation callback runs.
        raise asyncio.CancelledError(bg_task.interruption_reason)
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
      termination_intent = bg_task.termination_intent
      interruption_reason = bg_task.interruption_reason
      reason = termination_intent or interruption_reason or "cancelled"
      interrupted = termination_intent is None and interruption_reason is not None
      canonical = _terminal_task_result(
        bg_task,
        status="interrupted" if interrupted else "cancelled",
        reason=reason,
      )
      if canonical is not None:
        bg_task.task_result = canonical
        result = canonical.model_dump(mode="json")
      else:
        result = self._background_termination_result(
          bg_task, reason,
        )
      error = None
      final_state = (
        _task_state_from_task_result(canonical)
        if canonical is not None
        else TaskState.INTERRUPTED if interrupted else TaskState.KILLED
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

    if bg_task.state not in {
      TaskState.COMPLETED,
      TaskState.FAILED,
      TaskState.KILLED,
    }:
      # Recorded before the finalizer starts so a cancellation that races it
      # reconciles this outcome; a task already settled by shutdown or kill
      # keeps that settled result, which the finalizer also preserves.
      bg_task.pending_final_state = final_state
      bg_task.pending_result = result
      bg_task.pending_error = error
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
        timeout_error = asyncio.TimeoutError(
          f"background finalizer exceeded {timeout:.3f}s after cancellation"
        )
        self._record_completion_persistence_uncertainty(
          bg_task,
          timeout_error,
        )
        finalizer_task.cancel()
        finalizer_task.add_done_callback(self._consume_background_task_exception)
        await self._reconcile_background_agent(
          bg_task,
          default_intent=bg_task.termination_intent or "cancelled",
        )
        raise timeout_error
    finally:
      if (
        bg_task.completion_persistence_state == "committed"
        and bg_task.state in {
          TaskState.COMPLETED,
          TaskState.FAILED,
          TaskState.KILLED,
          TaskState.AWAITING_APPROVAL,
        }
      ):
        await self._invoke_background_completion_callback(bg_task)

  @staticmethod
  def _background_termination_result(
    bg_task: TaskEntry,
    termination_intent: TerminationIntent | InterruptionReason,
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
      "Background task was interrupted and can be resumed"
      if termination_intent in {"user_interrupted", "parent_interrupted"}
      else (
        "Background task was cancelled with its parent run"
        if termination_intent == "cancelled"
        else "Background task was killed before completion"
      )
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
    lookup_override = vars(self).get("_lookup_task_in_log")
    if callable(lookup_override):
      durable_entry = await self._lookup_task_in_log(bg_task.task_id)
    else:
      lookup_sync = getattr(
        self,
        "_lookup_task_in_log_current_sync",
        None,
      )
      durable_entry = (
        self._lookup_task_in_log_current_sync(bg_task.task_id)
        if callable(lookup_sync)
        else await self._lookup_task_in_log(bg_task.task_id)
      )
    if (
      durable_entry is not None
      and durable_entry.state == TaskState.AWAITING_APPROVAL
      and durable_entry.task_result is not None
      and durable_entry.task_result.execution.status == "awaiting_approval"
      and durable_entry.task_id == bg_task.task_id
      and durable_entry.admitted_task == bg_task.admitted_task
      and durable_entry.task_result == bg_task.task_result
      and durable_entry.error is None
    ):
      return durable_entry
    # A park is confirmable only off the canonical TaskResult row; a
    # registration-only rebuild defaults to PARKED without one.
    if (
      durable_entry is not None
      and durable_entry.state == TaskState.PARKED
      and durable_entry.task_result is not None
    ):
      return durable_entry
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
    if (
      bg_task.state == TaskState.INTERRUPTED
      and final_state == TaskState.AWAITING_APPROVAL
    ):
      self._task_registry.restore_committed_awaiting(
        bg_task.task_id,
        result=result,
      )
    elif (
      bg_task.state == TaskState.INTERRUPTED
      and final_state == TaskState.PARKED
    ):
      self._task_registry.restore_committed_park(
        bg_task.task_id,
        result=result,
      )
    elif final_state == TaskState.PARKED:
      self._task_registry.park(bg_task.task_id, result=result)
    elif bg_task.state == TaskState.INTERRUPTED:
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
    if (
      final_state == TaskState.AWAITING_APPROVAL
      and bg_task.task_result is not None
      and bg_task.task_result.execution.status == "awaiting_approval"
    ):
      self._commit_background_completion_live(
        bg_task,
        final_state=final_state,
        result=result,
        error=error,
      )
      return
    if final_state == TaskState.PARKED:
      # A parked job owes no completion: no skill result, no parent
      # publication. The journal records the park; ``continue_task`` resumes it.
      self._commit_background_completion_live(
        bg_task,
        final_state=final_state,
        result=result,
        error=error,
      )
      # PARKED is only ever derived from this task's canonical result
      # (``task_state_for_result``), so the result is present.
      parked_result = cast(TaskResult, bg_task.task_result)
      await self._append_task_parked(
        bg_task.task_id,
        reason=str(parked_result.execution.terminal_reason),
      )
      return
    await self._ensure_required_skill_result_settled(bg_task)
    if bg_task is not self._admitted_self_task:
      # A run's own job settles here; only a child is a parent's obligation.
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

  async def _finalize_admitted_self_task(
    self,
    entry: TaskEntry,
    *,
    run_error: BaseException | None,
    was_cancelled: bool,
    prior_evidence: SubAgentResultEvidence | None,
  ) -> None:
    """Settle this run's own admitted job from its committed run closure.

    The result comes from the owners an ordinary child's settlement uses,
    over this run's own events and journal, with the restored attempt's
    evidence merged so usage stays cumulative. It commits through the
    registered-task finalizer while the writer lease is still held.
    """
    from .runner_sub_agents import _runtime_exception_detail

    admitted = cast(AdmittedTask, entry.admitted_task)
    events = self._log.entries
    approval_park = self._approval_park
    terminal = self._terminal_closure_event
    cancelled = was_cancelled or isinstance(run_error, asyncio.CancelledError)
    if (
      approval_park is not None
      and terminal is not None
      and terminal.get("terminal_disposition") == "awaiting_approval"
      and run_error is None
      and not cancelled
    ):
      evidence = merge_sub_agent_result_evidence(
        prior_evidence,
        collect_sub_agent_result_evidence(events, durable=False),
      )
      result = terminal_task_result(
        admitted,
        status="awaiting_approval",
        reason=(
          f"approval:{approval_park.approval_id} "
          f"call:{approval_park.original_tool_call_id}"
        ),
        tools_used=evidence.tools_used,
        usage=_usage_observation(evidence.usage),
      )
    else:
      final_narrative = None
      if (
        latest_successful_declared_terminal_tool_result(
          events,
          declared_terminal_doors=terminal_result_doors(admitted),
        ) is None
        and admitted.result_requirement.terminal_narrative != "forbidden"
        and self._workspace_dir is not None
      ):
        narrative = await final_child_visible_text(
          self._agent_session_log,
          sub_session_id=self._full_session_id,
          role=cast(Literal["writer", "sub_agent"], self._role),
          sub_agent_id=self._sub_agent_id,
          workspace_dir=self._workspace_dir,
          runner_id=self._runner_id,
        )
        final_narrative = narrative.final_narrative
      result = task_result_from_execution(
        events,
        logical_task=admitted.logical_task,
        attempt=admitted.attempt,
        requirement=admitted.result_requirement,
        provenance=TaskResultProvenance(
          admitted_task_digest=admitted.admitted_task_digest,
          model_bind_digest=admitted.model_bind_digest,
          capability_binding_digest=admitted.capability_binding_digest,
          tool_grant_digest=admitted.tool_grant_digest,
        ),
        final_narrative=final_narrative,
        stalled=False,
        runtime_error_detail=(
          _runtime_exception_detail(run_error)
          if isinstance(run_error, Exception)
          else None
        ),
        external_terminal_signals=(
          [entry.termination_intent or "cancelled"] if cancelled else []
        ),
        prior_evidence=prior_evidence,
        admitted_task=admitted,
      )
    entry.task_result = result
    await self._finalize_background_agent(
      entry,
      final_state=_task_state_from_task_result(result),
      result=result.model_dump(mode="json"),
      error=None,
    )
    await self._invoke_background_completion_callback(entry)

  async def terminate_parked_background_task(
    self,
    task_id: str,
    *,
    persist_request: Callable[[], Awaitable[None]] | None = None,
  ) -> bool:
    """Durably STOP a parked attempt, or cancel its actual resume successor.

    Successor registration and this decision share the original entry's
    finalization_lock. ``persist_request`` is the trusted caller's durable
    STOP record; it runs inside that lock only when the STOP is actionable (a
    started successor, a committed park, or a running attempt whose completion
    is not already committing), so no successor can install between the
    request and the decision, and a settled task records no request. Only a
    started successor is routed;
    an unstarted reservation stays behind and its own source check sees the
    stopped original. The journal append/readback and parked-control commit
    stay inside the lock; the final callback runs outside it.
    """
    if (
      type(task_id) is not str
      or not task_id
      or task_id != task_id.strip()
    ):
      raise ValueError("task_id must be canonical non-empty text")
    entry = self._task_registry.get(task_id)
    if entry is None:
      return False
    successor_id: str | None = None
    changed = False
    cancelled_waiter: asyncio.CancelledError | None = None
    async with entry.finalization_lock:
      successors = [
        successor
        for successor in self._task_registry.list_tasks()
        if successor.original_task_id == entry.task_id
        and (
          successor.state != TaskState.PENDING
          or successor.asyncio_task is not None
        )
      ]
      if successors:
        if persist_request is not None:
          await persist_request()
        successor_id = successors[-1].task_id
      elif entry.state == TaskState.AWAITING_APPROVAL:
        if persist_request is not None:
          await persist_request()
        if getattr(self, "_agent_session_log", None) is None:
          raise RuntimeError("parked STOP requires the task's durable event log")
        parked = entry.task_result
        admitted = entry.admitted_task
        if parked is None or admitted is None:
          raise RuntimeError("parked STOP requires the original admitted TaskResult")
        cancelled = terminal_task_result(
          admitted,
          status="cancelled",
          reason="cancelled",
          tools_used=parked.evidence.tools_used,
          usage=parked.observation.usage,
        )

        async def _confirmed() -> TaskEntry | None:
          try:
            durable = await self._confirmed_durable_background_completion(entry)
          except BaseException:
            self._writer_lease_poisoned = True
            raise
          if durable is None:
            raise RuntimeError("parked STOP could not confirm the durable park")
          if durable.admitted_task != admitted or durable.error is not None:
            raise RuntimeError("parked STOP readback conflicts with original admission")
          if durable.state == TaskState.KILLED and durable.task_result == cancelled:
            return durable
          if (
            durable.state != TaskState.AWAITING_APPROVAL
            or durable.task_result != parked
            or durable.result != parked.model_dump(mode="json")
          ):
            raise RuntimeError("parked STOP readback conflicts with the canonical park")
          return None

        confirmed = await _confirmed()
        completed_at = _runner_attr(self, "time", time).time()
        if confirmed is None:
          payload = self._task_correlation_payload(entry)
          payload.update({
            "type": "task_terminated",
            "final_state": TaskState.KILLED.value,
            "completed_at": completed_at,
            "result": cancelled.model_dump(mode="json"),
            "error": None,
          })
          try:
            await self._append_durable_event(payload)
          except BaseException as failure:
            # append owns its fsync decision even if the caller is cancelled.
            # Read back that decision before publishing or permitting a retry.
            confirmed = await _confirmed()
            if confirmed is None:
              raise
            if isinstance(failure, asyncio.CancelledError):
              cancelled_waiter = failure
          else:
            confirmed = await _confirmed()
            if confirmed is None:
              raise RuntimeError("parked STOP append did not commit its termination")
        if confirmed is not None:
          if confirmed.completed_at is None:
            raise RuntimeError("parked STOP readback has no completion time")
          completed_at = confirmed.completed_at
        self._task_registry.commit_parked_termination(
          entry.task_id,
          result=cancelled,
          completed_at=completed_at,
        )
        changed = True
      elif entry.state not in {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.KILLED,
        TaskState.INTERRUPTED,
      } and entry.pending_final_state is None and entry.termination_intent is None:
        # A running attempt whose completion is not already committing.
        if persist_request is not None:
          await persist_request()
        return self.cancel_background_task(entry.task_id)
    if successor_id is not None:
      return await self.terminate_parked_background_task(successor_id)
    await self._invoke_background_completion_callback(entry)
    if cancelled_waiter is not None:
      raise cancelled_waiter
    return changed

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
    if task_done and asyncio_task is not None:
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
    on_park: Callable[[TaskEntry], Awaitable[None]] | None,
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
    # The attempt that installs lifecycle callbacks owns them only if it
    # starts or settles the registration; an unstarted reservation retained
    # for retry gets the retrying attempt's callbacks.
    completion_installed = (
      on_complete is not None and entry.completion_callback is None
    )
    if completion_installed:
      entry.completion_callback = on_complete
    park_installed = on_park is not None and entry.park_callback is None
    if park_installed:
      entry.park_callback = on_park
    try:
      return await self._reconcile_and_start_background_task(
        entry=entry,
        tool_input=tool_input,
        handler=handler,
        agent_name=agent_name,
        on_complete=on_complete,
        on_before_start=on_before_start,
        call_index=call_index,
        original_task_id=original_task_id,
        reconcile_registration=reconcile_registration,
        validate_resume_source=validate_resume_source,
        resume_source_entry=resume_source_entry,
      )
    finally:
      if entry.asyncio_task is None and entry.state == TaskState.PENDING:
        if completion_installed and entry.completion_callback is on_complete:
          entry.completion_callback = None
        if park_installed and entry.park_callback is on_park:
          entry.park_callback = None

  async def _reconcile_and_start_background_task(
    self,
    *,
    entry: TaskEntry,
    tool_input: Dict[str, Any],
    handler: BackgroundTaskHandler,
    agent_name: str | None,
    on_complete: BackgroundTaskCallback | None,
    on_before_start: Callable[[], None] | None,
    call_index: int,
    original_task_id: str | None,
    reconcile_registration: bool,
    validate_resume_source: bool,
    resume_source_entry: TaskEntry | None,
  ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    durable_existing: TaskEntry | None = None
    registration_state = getattr(
      entry,
      "registration_persistence_state",
      "not_started",
    )
    registration_committed = registration_state == "committed"

    async def _reject_if_resume_source_changed(
      registration_committed: bool,
    ) -> tuple[
      dict[str, Any] | None,
      dict[str, Any] | None,
    ] | None:
      # ``registration_committed`` is an argument, not a closed-over local:
      # the durable append now happens in
      # ``_append_task_registered_then_start`` and only that frame knows
      # whether a durable record exists yet. Pre-append the reservation is
      # discarded; post-append it must be settled with a terminal.
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

    source_rejection = await _reject_if_resume_source_changed(
      registration_committed,
    )
    if source_rejection is not None:
      return source_rejection

    return await self._append_task_registered_then_start(
      entry=entry,
      tool_input=tool_input,
      handler=handler,
      agent_name=agent_name,
      on_before_start=on_before_start,
      on_complete=on_complete,
      call_index=call_index,
      original_task_id=original_task_id,
      reconcile_registration=reconcile_registration,
      registration_committed=registration_committed,
      durable_existing=durable_existing,
      reject_if_resume_source_changed=_reject_if_resume_source_changed,
    )

  async def _append_task_registered_then_start(
    self,
    *,
    entry: TaskEntry,
    tool_input: Dict[str, Any],
    handler: BackgroundTaskHandler,
    agent_name: str | None,
    on_before_start: Callable[[], None] | None,
    on_complete: BackgroundTaskCallback | None,
    call_index: int,
    original_task_id: str | None,
    reconcile_registration: bool,
    registration_committed: bool,
    durable_existing: TaskEntry | None,
    reject_if_resume_source_changed: Callable[
      [bool],
      Awaitable[
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]] | None
      ],
    ],
  ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Append ``task_registered`` and bracket every exit that follows it.

    A-M8 / T3-I03. Admission (``TaskRegistry.admit``) already reserved the
    slot synchronously, so the durable append below is the next ``await`` on
    the registration path. The rule this method encodes:

    * **pre-append** failure -> ``discard_unstarted``: no durable record
      exists, so the reservation is simply released. On the resume lane the
      reservation is deliberately retained for the retry and is caller-owned
      (D-A8-2).
    * **post-append** failure -> a compensating durable
      ``task_completed(final_state="failed", reason="registration_aborted:…")``
      before the exception leaves, shielded on cancellation. Without it a
      PENDING ghost holds ``admission_count > 0`` forever, the run loop
      refuses clean success with ``background_delivery_incomplete``, and
      replay rebuilds the task as PARKED instead of FAILED
      (CUR-E2E-03, seam-map Hole 2).
    """

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

    # ---- Post-append bracket (D-A8-2/3) ---------------------------------
    # A durable ``task_registered`` now exists for this task. Every exit
    # below must leave a durable terminal behind it.
    try:
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

      source_rejection = await reject_if_resume_source_changed(
        registration_committed,
      )
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
    except BaseException as exc:
      await self._compensate_aborted_registration(entry, exc)
      raise

  async def _compensate_aborted_registration(
    self,
    entry: TaskEntry,
    exc: BaseException,
  ) -> None:
    """Settle a durably registered task that never reached its worker.

    The compensating terminal is what makes registration transactional: the
    durable ``task_registered`` is answered by a durable
    ``task_completed(final_state="failed")`` so replay rebuilds FAILED rather
    than a ghost PARKED entry, and ``admission_count`` returns to zero
    so the session-completion guard stops reporting
    ``pending_or_running_tasks`` (D-A8-2/3; the notification the FAILED
    transition emits is the accepted D-A8-4 cost for direct dispatch —
    workflow children stay exempt via the existing predicate).
    """

    detail = (
      f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"
    )[:2000]
    canonical = _terminal_task_result(
      entry,
      status="failed",
      reason=f"registration_aborted: {detail}",
    )
    if canonical is not None:
      entry.task_result = canonical
      aborted_result = canonical.model_dump(mode="json")
    else:
      aborted_result = {
        "status": "failed",
        "reason": "registration_aborted",
        "detail": detail,
      }
    finalize = self._finalize_background_agent(
      entry,
      final_state=TaskState.FAILED,
      result=aborted_result,
      error=None,
    )
    if not isinstance(exc, asyncio.CancelledError):
      await finalize
      return
    # The terminal must survive the cancellation that caused it, or the
    # ghost simply reappears.
    terminal_task = asyncio.ensure_future(finalize)
    terminal_task.add_done_callback(self._consume_background_task_exception)
    self._track_background_initialization(terminal_task)
    try:
      await asyncio.shield(terminal_task)
    except asyncio.CancelledError:
      # Our waiter was cancelled again; the shielded append continues and is
      # tracked. The original cancellation is re-raised by the caller.
      pass

  async def _started_background_task(self, task_id: str) -> TaskEntry | None:
    """Replay a started worker, never an unstarted registration reservation."""
    entry = self._task_registry.get(task_id)
    if entry is None:
      return None
    initialization = entry.initialization_task
    if initialization is not None and not initialization.done():
      await asyncio.shield(initialization)
    if entry.asyncio_task is not None or (
      entry.registration_persistence_state == "committed"
      and entry.state != TaskState.PENDING
    ):
      return entry
    return None

  async def _register_background_task(
    self,
    *,
    tool_input: Dict[str, Any],
    handler: BackgroundTaskHandler,
    task_type: str = "background_agent",
    agent_name: str | None = None,
    parent_turn_id: str | None = None,
    on_complete: BackgroundTaskCallback | None = None,
    on_park: Callable[[TaskEntry], Awaitable[None]] | None = None,
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
      resume_task_id = await _runner_attr(self, "_resume_task_id_override", _resume_task_id_override)(
        original_task_id,
        resume_chain_depth=self._resume_chain_depth,
        resume_root=self._resume_root_task_id,
      )
      source_error = _source_error()
      if source_error is not None:
        return None, source_error
      if resolved_task_id is not None and resolved_task_id != resume_task_id:
        return None, {
          "code": "resume_successor_conflict",
          "message": (
            "Caller task_id_override does not match the deterministic "
            "resume successor identity"
          ),
        }
      resolved_task_id = resume_task_id

    # ---- Pre-decision replay/reuse checks (A-M8 / T3-I03) -----------------
    # Every ``await`` on the registration path lives ABOVE the admission
    # point. Both durable replay lookups used to sit between the capacity
    # verdict and the reservation, so N concurrent registrations could all
    # read the same stale ``admission_count`` and all register afterwards
    # (CUR-E2E-03, Hole 1). They are pure pre-checks now.
    if resolved_task_id is not None:
      existing = self._task_registry.get(resolved_task_id)
      if existing is not None:
        if original_task_id is not None:
          source_error = _source_error()
          if source_error is not None:
            return None, source_error
          try:
            existing, _created = self._task_registry.claim_resume_successor(
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
        if existing.state != TaskState.PENDING:
          return await _replay_existing(existing)
        initialization_task = existing.initialization_task
        if (
          initialization_task is not None
          and not initialization_task.done()
        ):
          return await asyncio.shield(initialization_task)
      elif (
        original_task_id is None
        or getattr(self, "_agent_session_log", None) is not None
      ):
        durable_existing = await self._lookup_task_in_log(resolved_task_id)
        if original_task_id is not None:
          source_error = _source_error()
          if source_error is not None:
            return None, source_error
        if durable_existing is not None:
          if (
            original_task_id is not None
            and durable_existing.original_task_id != original_task_id
          ):
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

    if getattr(self, "_background_delivery_grace_active", False):
      return None, {
        "code": "background_delivery_grace_active",
        "message": (
          "The run is past its normal turn limit and is completing bounded "
          "background-result delivery. New background tasks and resumes are "
          "not admitted during this delivery-only phase."
        ),
      }
    if original_task_id is not None:
      source_error = _source_error()
      if source_error is not None:
        return None, source_error

    # ---- Atomic admission -------------------------------------------------
    # One synchronous critical section folds the notification-retrieval
    # ceiling and the reservation. A rejection here produces no durable
    # ``task_registered`` at all (D-A8-3). The in-flight count is measured
    # below, never refused: an admitted child waits on the sub-agent
    # semaphore for an execution slot.
    entry, rejection = self._task_registry.admit(
      task_type,
      agent_name=agent_name,
      task_id=resolved_task_id,
      original_task_id=original_task_id,
      reject_retrieval_backpressure=(
        lambda pending_omitted_results, retention_limit: {
          "code": "background_notification_retrieval_required",
          "message": (
            "Omitted background results have filled bounded retrieval "
            "retention. Call get_background_result with task_id='*' to "
            "list eligible task IDs, then retrieve each exact ID before "
            "starting more background tasks."
          ),
          "pending_omitted_results": pending_omitted_results,
          "retention_limit": retention_limit,
        }
      ),
    )
    if rejection is not None:
      return None, rejection
    if entry is None:
      raise RuntimeError("task admission returned neither entry nor rejection")
    self._append({
      "type": "interceptor_measurement",
      "measurement": {"concurrent_sub_agents": self._task_registry.admission_count},
    })

    # ``admit`` may hand back a reservation another caller took while this
    # one was in its pre-decision awaits. Re-read the reservation's live
    # state synchronously (no slot is at stake — it is the same entry).
    if entry.state != TaskState.PENDING:
      return await _replay_existing(entry)
    initialization_task = entry.initialization_task
    if initialization_task is not None and not initialization_task.done():
      return await asyncio.shield(initialization_task)

    initialization_task = asyncio.create_task(
      self._prepare_and_start_background_task(
        entry=entry,
        tool_input=tool_input,
        handler=handler,
        agent_name=agent_name,
        parent_turn_id=parent_turn_id,
        on_complete=on_complete,
        on_park=on_park,
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

  def interrupt_background_task(
    self,
    task_id: str,
    *,
    reason: InterruptionReason = "user_interrupted",
  ) -> bool:
    """Pause one owned live child without cancelling this runner or siblings.

    Controls use ``user_interrupted``; a parent-agent tool uses
    ``parent_interrupted``. The returned bool acknowledges the request, not
    settlement; await the child's worker before resuming its durable transcript.
    """
    return self._task_registry.interrupt(task_id, reason=reason)

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

    await self._rebuild_task_registry_from_log()

    if task_id == "*":
      selected_entries = self._task_registry.list_tasks()
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
      list(self._notification_queue.peek())
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

  async def cancel_and_require_background_tasks_drained(
    self,
    *,
    timeout: float | None = None,
  ) -> None:
    """Strict deletion-only cancellation for every live child writer.

    Normal parent shutdown retains its bounded best-effort behavior. This
    method is called only after the parent turn has been cancelled, and it
    examines underlying asyncio tasks even when normal reconciliation has
    already terminalized their task-registry projections.
    """

    selected_tasks: set[Any] = set()
    for entry in self._task_registry.list_tasks():
      for task in (
        getattr(entry, "initialization_task", None),
        getattr(entry, "asyncio_task", None),
      ):
        if (
          task is not None
          and callable(getattr(task, "done", None))
          and not task.done()
        ):
          selected_tasks.add(task)
    if not selected_tasks:
      return

    for task in selected_tasks:
      task.cancel()
    drain_timeout = (
      max(0.0, float(timeout))
      if timeout is not None
      else max(
        0.0,
        float(
          getattr(
            self,
            "_background_cancel_drain_timeout_seconds",
            _BACKGROUND_CANCEL_DRAIN_TIMEOUT_SECONDS,
          )
        ),
      )
    )
    done, pending = await asyncio.wait(
      selected_tasks,
      timeout=drain_timeout,
    )
    if done:
      await asyncio.gather(*done, return_exceptions=True)
    if pending:
      raise StrictBackgroundTaskDrainUnavailable(
        "document_erase_background_drain_unavailable"
      )

  def bind_research_file_activity_lease(self, lease: Any) -> None:
    """Transfer pre-bound activity to the run's exact-file lease owner."""

    if lease is None or not callable(getattr(lease, "release", None)):
      raise TypeError("research file activity lease is invalid")
    if getattr(self, "_research_file_activity_lease", None) is not None:
      raise RuntimeError("research file activity lease is already bound")
    activity = _RunResearchFileActivityLease()
    activity.bind(lease, getattr(self, "_context_research_file_id", None))
    self._research_file_activity_lease = activity

  def ensure_research_file_activity(
    self, workspace_dir: Path, research_file_id: int
  ) -> bool:
    """Fence lazy bindings until this run and its tracked children settle."""

    activity = getattr(self, "_research_file_activity_lease", None)
    if activity is None:
      activity = _RunResearchFileActivityLease()
      self._research_file_activity_lease = activity
    return activity.ensure(workspace_dir, research_file_id)

  def release_research_file_activity_lease_if_owned(self) -> None:
    """Release an unstarted transfer, retaining it for any tracked children."""

    RunnerBackgroundLifecycleMixin._release_activity_lease_after_children(
      self,
      lease_attribute="_research_file_activity_lease",
      hold_attribute="_research_file_activity_release_hold",
    )

  def bind_selected_content_activity_lease(self, lease: Any) -> None:
    """Transfer exact selected-content activity to the runner owner."""

    if lease is None or not callable(getattr(lease, "release", None)):
      raise TypeError("selected content activity lease is invalid")
    if getattr(self, "_selected_content_activity_lease", None) is not None:
      raise RuntimeError("selected content activity lease is already bound")
    self._selected_content_activity_lease = lease

  def release_selected_content_activity_lease_if_owned(self) -> None:
    """Release an unstarted transfer, retaining it for any tracked children."""

    RunnerBackgroundLifecycleMixin._release_activity_lease_after_children(
      self,
      lease_attribute="_selected_content_activity_lease",
      hold_attribute="_selected_content_activity_release_hold",
    )

  def _release_research_file_activity_after_children(self) -> None:
    RunnerBackgroundLifecycleMixin.release_research_file_activity_lease_if_owned(
      self
    )

  def _release_selected_content_activity_after_children(self) -> None:
    RunnerBackgroundLifecycleMixin.release_selected_content_activity_lease_if_owned(
      self
    )

  def _release_activity_lease_after_children(
    self,
    *,
    lease_attribute: str,
    hold_attribute: str,
  ) -> None:
    lease = getattr(self, lease_attribute, None)
    if lease is None:
      return
    setattr(self, lease_attribute, None)
    pending = {
      task
      for entry in self._task_registry.list_tasks()
      for task in (
        getattr(entry, "initialization_task", None),
        getattr(entry, "asyncio_task", None),
      )
      if (
        task is not None
        and callable(getattr(task, "done", None))
        and not task.done()
      )
    }
    if not pending:
      lease.release()
      return

    hold = {"lease": lease, "pending": pending}
    setattr(self, hold_attribute, hold)

    def _release_when_last_child_finishes(task: Any) -> None:
      hold["pending"].discard(task)
      if hold["pending"]:
        return
      hold["lease"].release()
      if getattr(self, hold_attribute, None) is hold:
        setattr(self, hold_attribute, None)

    for task in pending:
      task.add_done_callback(_release_when_last_child_finishes)

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
        (initialization_task := entry.initialization_task)
        is not None
        and not initialization_task.done()
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
          wait_timeout=max(0.0, self._background_grace_wait_timeout_seconds),
          drain_timeout=max(0.0, self._background_kill_drain_timeout_seconds),
        )
    finally:
      await self._reconcile_background_agents(
        running_entries,
        default_intent=termination_intent,
      )
