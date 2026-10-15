from __future__ import annotations

import asyncio
import logging
import time
from typing import (
  TYPE_CHECKING,
  Any,
  Awaitable,
  Callable,
  Dict,
  List,
  Literal,
  Mapping,
  Optional,
  Set,
  Tuple,
  cast,
)

from agent_workflow_contracts import (
  AdmittedTask,
  AttemptRef,
  LogicalTaskRef,
  ResultRequirement,
  TaskResult,
  TaskResultProvenance,
  terminal_task_result,
)

from .admission import current_parent_context
from .capability_execution import BoundCapabilityExecution
from .event_log import EventLog
from .providers.anthropic import _is_cacheable_message_block
from .runner_introspection import derive_sub_agent_id as _derive_sub_agent_id
from .runner_session_lifecycle import _runner_attr
from .runner_cleanup import cleanup_failure_notes
from .runner_budget import CostAccumulator
from .runner_stream_turn import STREAM_GUARD_POLL_INTERVAL
from .runner_state import user_turn_message as _user_turn_message
from .skill_lifecycle import drain_owned_lifecycle_task
from .session import session_owner_user_id
from .sub_agent_result_contract import _usage_observation
from .sub_agent_result_evidence import (
  SubAgentResultEvidence,
  collect_sub_agent_result_evidence,
  merge_sub_agent_result_evidence,
)
from .sub_agent_narrative_result import (
  final_child_visible_text,
  task_result_from_execution,
)
from .sub_agent_skill_state import (
  latest_successful_declared_terminal_tool_result,
  terminal_result_doors,
)
from .task_registry import (
  ParentMessage,
  RegistryOwner,
  TaskEntry,
  TaskRegistry,
  make_progress_tracker,
)
from .tool_result_spill import job_spill_sink
from .tool_dispatcher import ToolDispatcher
from .tool_dispatcher_approval_lifecycle import ApprovalPark

if TYPE_CHECKING:
  from pathlib import Path
  from .sub_agent import ApprovedCallRef

  from .agent_session_log import AgentSessionLog
  from .containment_settings import ContainmentSettings
  from .mcp_activation import McpActivationFold
  from .mcp_client import McpClientManager
  from .multi_user.billing import SessionUsageSummary, UsageEvent, _UsageAggregator
  from .runner_state import SubAgentConfig, ToolResultContext
  from .tool_result_spill import SpillSink

log = logging.getLogger("agent_gateway.runner")
_MISSING_SUB_AGENT_ID = object()
# The parent's log carries a child's approval requests, which the parent
# answers, its interceptor measurements (the child shares the run's counters,
# and session_recap reports them from the parent's log) and its model
# fallback hops, which the user sees on every path (design §6.5 R9). The
# parent's append is these events' one delivery.
_PARENT_LOG_EVENT_TYPES = frozenset({"tool_approval_request", "interceptor_measurement", "model_fallback"})


class _ChildActivity:
  """Parent-side liveness of one child run: its last event and open tools."""

  __slots__ = ("last_at", "tools_in_flight")

  def __init__(self) -> None:
    self.last_at = time.monotonic()
    self.tools_in_flight: set[str] = set()

  def observe(self, event: Mapping[str, Any]) -> None:
    self.last_at = time.monotonic()
    event_type = event.get("type")
    tool_call_id = str(event.get("tool_call_id") or "")
    if event_type == "tool_call_start":
      self.tools_in_flight.add(tool_call_id)
    elif event_type in {"tool_call_complete", "tool_call_interrupted"}:
      self.tools_in_flight.discard(tool_call_id)


async def _await_child_under_activity_guard(
  child_run: Awaitable[Any],
  activity: _ChildActivity,
  *,
  silence_timeout: float,
) -> float | None:
  """Await one child run; cancel it once it is silent with no tool in flight.

  A child silent for ``silence_timeout`` seconds with no tool open is wedged
  (ACUI-1: an await outside both stream and tool);
  ``ContainmentSettings.child_silence_timeout_seconds`` supplies it, kept
  above the stream stall guard so the child's own guard fires first on stream
  silence, and ``0`` never cancels.

  Returns the silence in seconds when the guard cancelled the child, else
  None. A child exception or cancellation propagates unchanged, and a parent
  cancellation is delivered into the child before it propagates.
  """

  child = asyncio.ensure_future(child_run)
  try:
    while True:
      done, _pending = await asyncio.wait(
        {child},
        timeout=STREAM_GUARD_POLL_INTERVAL,
      )
      if done:
        child.result()
        return None
      silence = time.monotonic() - activity.last_at
      if (
        silence_timeout > 0
        and not activity.tools_in_flight
        and silence > silence_timeout
      ):
        child.cancel()
        await asyncio.wait({child})
        return silence
  except asyncio.CancelledError:
    if not child.done():
      child.cancel()
      await asyncio.wait({child})
    raise


def _validate_child_result_requirement(
  requirement: ResultRequirement,
  *,
  skill_name: str,
) -> None:
  if not isinstance(requirement, ResultRequirement):
    raise TypeError(
      "sub-agent result requirement must be an exact ResultRequirement"
    )
  if not isinstance(skill_name, str) or not skill_name.strip():
    raise ValueError("sub-agent execution requires a non-empty skill_name")


def _build_child_event_log(
  *,
  parent_log: EventLog,
  event_log_cls: Callable[..., EventLog],
  sub_session_id: str,
  activity: _ChildActivity,
  progress_cb: Callable[[Dict[str, Any], str], None] | None,
  on_sub_event: Callable[[Dict[str, Any], str], None] | None,
) -> EventLog:
  """Inherit strict parent delivery while tagging every child event."""

  original_prepare_event = getattr(
    parent_log,
    "_prepare_event",
    None,
  )
  original_on_event = getattr(parent_log, "_on_event", None)
  original_on_event_error = getattr(
    parent_log,
    "_on_event_error",
    "ignore",
  )


  def _composed_on_event(
    event: Dict[str, Any],
    session_id: str,
  ) -> None:
    activity.observe(event)
    event_copy = dict(event)
    event_copy["sub_agent_id"] = session_id
    if progress_cb is not None:
      try:
        progress_cb(event_copy, session_id)
      except Exception:
        pass
    if event_copy.get("type") in _PARENT_LOG_EVENT_TYPES:
      # With a parent prepare, _composed_prepare_event already appended it.
      if original_prepare_event is None:
        parent_log.append(event_copy)
    elif original_on_event is not None:
      try:
        original_on_event(event_copy, session_id)
      except Exception:
        if original_on_event_error == "raise":
          raise
    if on_sub_event is not None:
      try:
        on_sub_event(event_copy, session_id)
      except Exception:
        pass

  if original_prepare_event is not None:
    def _composed_prepare_event(
      event: Dict[str, Any],
    ) -> Dict[str, Any]:
      if type(event) is not dict:
        raise TypeError("sub-agent event must be an exact dictionary")
      prior_sub_agent_id = event.get(
        "sub_agent_id",
        _MISSING_SUB_AGENT_ID,
      )
      event["sub_agent_id"] = sub_session_id
      try:
        parent_entry = (
          parent_log.append(event)
          if event.get("type") in _PARENT_LOG_EVENT_TYPES
          else None
        )
        # A closed parent log records nothing; the event is still delivered.
        prepared = (
          original_prepare_event(event)
          if parent_entry is None
          else parent_entry.event
        )
      finally:
        if prior_sub_agent_id is _MISSING_SUB_AGENT_ID:
          event.pop("sub_agent_id", None)
        else:
          event["sub_agent_id"] = prior_sub_agent_id
      return prepared

    return event_log_cls(
      prepare_event=_composed_prepare_event,
      on_event=_composed_on_event,
      on_event_error=original_on_event_error,
      session_id=sub_session_id,
    )
  return event_log_cls(
    on_event=_composed_on_event,
    session_id=sub_session_id,
  )


def _authoritative_child_tool_getter(
  dispatcher: ToolDispatcher,
  *,
  operation: str,
) -> Callable[[], List[Dict[str, Any]]]:
  get_tool_definitions: Callable[[], List[Dict[str, Any]]] | None = getattr(
    dispatcher,
    "get_tool_definitions",
    None,
  )
  if not callable(get_tool_definitions):
    raise TypeError(
      f"{operation} requires a dispatcher with an authoritative tool catalog"
    )
  return get_tool_definitions


class RunnerSubAgentMixin:
  if TYPE_CHECKING:
    _agent_session_log: AgentSessionLog | None
    _aggregator: _UsageAggregator
    _billing_mode: Literal["byok", "metered"]
    _channel: str | None
    _compaction_trigger: int | None
    _context_surfaces_provider: (
      Callable[[], list[dict[str, Any]]] | None
    )
    _context_surfaces_static: list[dict[str, Any]]
    _full_session_id: str
    _log: EventLog
    _containment: ContainmentSettings
    _max_resume_chain_depth: int
    _mcp_activation_fold: McpActivationFold
    _mcp_client: McpClientManager | None
    _on_late_usage_event: (
      Callable[[UsageEvent], Awaitable[None] | None] | None
    )
    _on_metric: Callable[[str, int], None] | None
    _on_session_summary: (
      Callable[[SessionUsageSummary], Awaitable[None] | None] | None
    )
    _on_tool_result: (
      Callable[
        [ToolResultContext],
        Awaitable[List[Dict[str, Any]] | None] | List[Dict[str, Any]] | None,
      ]
      | None
    )
    _on_tool_timing: Callable[..., None] | None
    _on_usage: Callable[[UsageEvent], Awaitable[None] | None] | None
    _rate_table_version: str
    _request_id: str
    _skill_run_id: str | None
    _spill_dir_provider: SpillSink | None
    _task_registry: TaskRegistry
    _stream_stall_timeout: float | None
    _sub_agent_config: SubAgentConfig | None
    _tool_call_timeout: float | None
    _usage_ledger_dlq_path: Path
    _usage_user_id: str
    _workspace_dir: str | None

  async def spawn_sub_agent(
    self,
    task: str,
    *,
    capability_execution: BoundCapabilityExecution,
    skill_name: str,
    logical_task: LogicalTaskRef,
    attempt: AttemptRef,
    result_requirement: ResultRequirement,
    result_provenance: TaskResultProvenance,
    system_prompt: str | None = None,
    dispatcher: ToolDispatcher,
    sub_session: Any | None = None,
    excluded_tools: Set[str] | None = None,
    max_tokens: int = 64000,
    call_index: int = 0,
    parent_turn_id: str | None = None,
    task_entry: TaskEntry | None = None,
    cost_observation_threshold_usd: float | None = None,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
    skill_run_id: str | None = None,
    admitted_task: AdmittedTask | None = None,
    bind_job_runner: Callable[[Any], None],
  ) -> Tuple[Optional[TaskResult], Optional[Dict[str, Any]]]:
    """Run a focused sub-agent task and return its canonical result.

    This method is used by the built-in `run_agent` tool. The sub-agent shares
    usage aggregation with the parent. An unbudgeted generic child uses a
    non-enforcing observation accumulator; a named budgeted child uses a
    child-local hard accumulator over that observer. It gets a fresh `EventLog`,
    turn limit, and its own dispatcher. Provider, credential, model, and effort
    are already frozen in ``capability_execution`` and are never inherited
    from the parent.
    """
    if not isinstance(capability_execution, BoundCapabilityExecution):
      raise TypeError(
        "spawn_sub_agent requires a BoundCapabilityExecution"
      )
    capability_execution.revalidate()
    _validate_child_result_requirement(
      result_requirement,
      skill_name=skill_name,
    )
    entry_admitted_task = (
      task_entry.admitted_task
      if task_entry is not None
      and isinstance(task_entry.admitted_task, AdmittedTask)
      else None
    )
    if (
      admitted_task is not None
      and entry_admitted_task is not None
      and admitted_task != entry_admitted_task
    ):
      raise ValueError("spawn_sub_agent received conflicting admitted tasks")
    resolved_admitted_task = admitted_task or entry_admitted_task
    if resolved_admitted_task is not None:
      admitted = resolved_admitted_task
      if (
        admitted.logical_task != logical_task
        or admitted.attempt != attempt
        or admitted.admitted_task_digest
        != result_provenance.admitted_task_digest
      ):
        raise ValueError("spawn_sub_agent identity differs from admitted task")
    if self._agent_session_log is None:
      raise ValueError(
        "narrative child execution requires a durable session log"
      )
    child_get_tool_definitions = _authoritative_child_tool_getter(
      dispatcher,
      operation="spawn_sub_agent",
    )

    if self._sub_agent_config is not None:
      if system_prompt is None:
        system_prompt = self._sub_agent_config.system_prompt
      if excluded_tools is None:
        excluded_tools = set(self._sub_agent_config.excluded_tools)

    derive_sub_agent_id = _runner_attr(self, "_derive_sub_agent_id", _derive_sub_agent_id)
    sub_session_id = str(getattr(sub_session, "session_id", "") or derive_sub_agent_id(self._full_session_id, call_index))
    progress_tracker_factory = _runner_attr(self, "make_progress_tracker", make_progress_tracker)
    progress_cb = progress_tracker_factory(task_entry) if task_entry else None

    event_log_cls = _runner_attr(self, "EventLog", EventLog)
    activity = _ChildActivity()
    sub_log = _build_child_event_log(
      parent_log=self._log,
      event_log_cls=event_log_cls,
      sub_session_id=sub_session_id,
      activity=activity,
      progress_cb=progress_cb,
      on_sub_event=on_sub_event,
    )
    dispatcher._event_log = sub_log
    child_cost_accumulator = CostAccumulator(cost_observation_threshold_usd)
    runner_cls = _runner_attr(self, "AgentRunner", type(self))
    child_spill_sink = _child_spill_sink(self._spill_dir_provider, attempt)
    sub_runner = runner_cls(
      event_log=sub_log,
      dispatcher=dispatcher,
      session_id=sub_session_id,
      gateway_session=sub_session,
      context_research_file_id=(
        current_parent_context(sub_session).research_binding.research_file_id
        if sub_session is not None
        else None
      ),
      capability_execution=capability_execution,
      max_tokens_override=max_tokens,
      stream_stall_timeout=self._stream_stall_timeout,
      mcp_client=self._mcp_client,
      mcp_activation_fold=self._mcp_activation_fold,
      excluded_tools=excluded_tools or set(),
      get_tool_definitions=child_get_tool_definitions,
      on_tool_result=self._on_tool_result,
      on_usage=self._on_usage,
      on_session_summary=self._on_session_summary,
      on_late_usage_event=self._on_late_usage_event,
      on_tool_timing=self._on_tool_timing,
      user_id=(
        session_owner_user_id(sub_session)
        if sub_session is not None
        else self._usage_user_id
      ),
      request_id=self._request_id,
      parent_turn_id=parent_turn_id,
      billing_mode=self._billing_mode,
      rate_table_version=self._rate_table_version,
      channel=self._channel,
      usage_ledger_dlq_path=self._usage_ledger_dlq_path,
      on_metric=self._on_metric,
      sub_agent_config=self._sub_agent_config,
      compaction_trigger=self._compaction_trigger,
      compaction_instructions=None,
      tool_call_timeout=self._tool_call_timeout,
      _cost_accumulator=child_cost_accumulator,
      _usage_parent=self._aggregator,
      containment=self._containment,
      result_requirement=result_requirement,
      agent_session_log=self._agent_session_log,
      message_inbox=task_entry.message_inbox if task_entry else None,
      max_resume_chain_depth=self._max_resume_chain_depth,
      emit_session_recap=False,
      code_execution_spill_dir_provider=child_spill_sink,
      commercial_usage_producer=getattr(self, "_commercial_usage_producer", None),
      skill_run_id=skill_run_id or self._skill_run_id,
      workspace_dir=self._workspace_dir,
      batch_id=getattr(self, "_batch_id", None),
      context_surfaces=self._context_surfaces_provider or self._context_surfaces_static,
      terminal_tool_result_ids=set(terminal_result_doors(resolved_admitted_task)),
    )
    _bind_child_registry(
      sub_runner,
      parent_registry=self._task_registry,
      attempt=attempt,
      sub_session=sub_session,
      task_entry=task_entry,
      spill_sink=child_spill_sink,
      bind_job_runner=bind_job_runner,
    )
    dispatcher.bind_research_file_activity_owner(sub_runner.ensure_research_file_activity)
    stalled = False
    runtime_exception_detail: str | None = None
    cancelled_error: asyncio.CancelledError | None = None
    cancellation_signal: str | None = None
    cleanup_warnings: list[str] = []
    user_turn_message = _runner_attr(self, "_user_turn_message", _user_turn_message)
    coro = sub_runner.run(
      messages=[user_turn_message(task)],
      system_prompt=system_prompt,
    )
    try:
      silence = await _await_child_under_activity_guard(
        coro, activity,
        silence_timeout=self._containment.child_silence_timeout_seconds,
      )
      if silence is not None:
        stalled = True
        detail = f"Sub-agent stalled: no activity for {silence:.0f}s"
        _runner_attr(self, "log", log).warning("[%s] %s", sub_session_id, detail)
        sub_log.append({"type": "error", "error": detail, "error_sub_code": "stalled"})
    except asyncio.CancelledError as exc:
      cancelled_error = exc
      cleanup_warnings.extend(cleanup_failure_notes(exc))
      cancellation_signal = (
        task_entry.termination_intent
        if task_entry is not None
        and task_entry.termination_intent is not None
        else "cancelled"
      )
      _runner_attr(self, "log", log).warning("[%s] Sub-agent cancelled (parent disconnect or shutdown)", sub_session_id)
      sub_log.append({"type": "error", "error": "Sub-agent cancelled"})
    except ApprovalPark:
      raise
    except Exception as exc:
      runtime_exception_detail = _runtime_exception_detail(exc)
      cleanup_warnings.extend(cleanup_failure_notes(exc))
      _runner_attr(self, "log", log).warning(
        "[%s] Sub-agent failed: %s",
        sub_session_id,
        runtime_exception_detail,
      )
      sub_log.append(
        {"type": "error", "error": runtime_exception_detail}
      )
    finally:
      approval_park = sub_runner._approval_park
      (
        cancelled_error,
        cancellation_signal,
        runtime_exception_detail,
        cleanup_warnings,
      ) = await _close_sub_runner(
        sub_runner,
        sub_log,
        stalled=stalled,
        approval_park=approval_park,
        cancelled_error=cancelled_error,
        cancellation_signal=cancellation_signal,
        runtime_exception_detail=runtime_exception_detail,
        task_entry=task_entry,
        cleanup_warnings=cleanup_warnings,
        sub_session=sub_session,
      )

    if (
      approval_park is not None
      and sub_runner._terminal_closure_event is not None
      and sub_runner._terminal_closure_event.get("terminal_disposition") == "awaiting_approval"
      and cancelled_error is None
      and runtime_exception_detail is None
      and not stalled
    ):
      evidence = collect_sub_agent_result_evidence(sub_log.entries, durable=False)
      return terminal_task_result(
        cast(AdmittedTask, resolved_admitted_task),
        status="awaiting_approval",
        reason=(
          f"approval:{approval_park.approval_id} "
          f"call:{approval_park.original_tool_call_id}"
        ),
        tools_used=evidence.tools_used,
        usage=_usage_observation(evidence.usage),
      ), None

    terminal_tool_result = latest_successful_declared_terminal_tool_result(
      sub_log.entries,
      declared_terminal_doors=terminal_result_doors(resolved_admitted_task),
    )
    final_narrative = None
    if (
      terminal_tool_result is None
      and result_requirement.terminal_narrative != "forbidden"
      and self._workspace_dir is not None
    ):
      sub_runner_id = getattr(sub_runner, "_runner_id", None)
      if not isinstance(sub_runner_id, str) or not sub_runner_id:
        raise RuntimeError(
          "narrative child completion requires its exact durable runner_id"
        )
      narrative_text = await final_child_visible_text(
        self._agent_session_log,
        sub_session_id=sub_session_id,
        role="sub_agent",
        sub_agent_id=sub_session_id,
        workspace_dir=self._workspace_dir,
        runner_id=sub_runner_id,
      )
      final_narrative = narrative_text.final_narrative
    result = task_result_from_execution(
      sub_log.entries,
      logical_task=logical_task,
      attempt=attempt,
      requirement=result_requirement,
      provenance=result_provenance,
      final_narrative=final_narrative,
      stalled=stalled,
      runtime_error_detail=runtime_exception_detail,
      external_terminal_signals=(
        [cancellation_signal]
        if cancellation_signal is not None
        else []
      ),
      # B-3: the authority frozen at admission, never the ambient catalog.
      admitted_task=resolved_admitted_task,
    )
    if cancelled_error is not None and terminal_tool_result is None:
      if task_entry is not None:
        task_entry.task_result = result
        task_entry.result = result.model_dump(mode="json")
      raise cancelled_error
    return result, None

  async def resume_sub_agent(
    self,
    *,
    original_task_id: str,
    reconstructed_messages: List[Dict[str, Any]],
    parent_messages: list[ParentMessage],
    capability_execution: BoundCapabilityExecution,
    skill_name: str,
    logical_task: LogicalTaskRef,
    attempt: AttemptRef,
    result_requirement: ResultRequirement,
    result_provenance: TaskResultProvenance,
    prior_evidence: SubAgentResultEvidence | None = None,
    system_prompt: str | None = None,
    dispatcher: ToolDispatcher,
    sub_session: Any | None = None,
    excluded_tools: Set[str] | None = None,
    max_tokens: int = 64000,
    call_index: int = 0,
    parent_turn_id: str | None = None,
    task_entry: TaskEntry | None = None,
    cost_observation_threshold_usd: float | None = None,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
    skill_run_id: str | None = None,
    admitted_task: AdmittedTask | None = None,
    prior_terminal_tool_result: Mapping[str, Any] | None = None,
    bind_job_runner: Callable[[Any], None],
    usage_aggregator: _UsageAggregator | None,
    approved_call: ApprovedCallRef | None = None,
    bind_research_file_activity_lease_func: (
      Callable[[Any], None] | None
    ) = None,
  ) -> Tuple[Optional[TaskResult], Optional[Dict[str, Any]]]:
    if not isinstance(capability_execution, BoundCapabilityExecution):
      raise TypeError(
        "resume_sub_agent requires a BoundCapabilityExecution"
      )
    capability_execution.revalidate()
    _validate_child_result_requirement(
      result_requirement,
      skill_name=skill_name,
    )
    entry_admitted_task = (
      task_entry.admitted_task
      if task_entry is not None
      and isinstance(task_entry.admitted_task, AdmittedTask)
      else None
    )
    if (
      admitted_task is not None
      and entry_admitted_task is not None
      and admitted_task != entry_admitted_task
    ):
      raise ValueError("resume_sub_agent received conflicting admitted tasks")
    resolved_admitted_task = admitted_task or entry_admitted_task
    if resolved_admitted_task is not None:
      admitted = resolved_admitted_task
      if (
        admitted.logical_task != logical_task
        or admitted.attempt != attempt
        or admitted.admitted_task_digest
        != result_provenance.admitted_task_digest
      ):
        raise ValueError("resume_sub_agent identity differs from admitted task")
    child_get_tool_definitions = _authoritative_child_tool_getter(
      dispatcher,
      operation="resume_sub_agent",
    )

    if prior_terminal_tool_result is not None:
      return task_result_from_execution(
        (),
        logical_task=logical_task,
        attempt=attempt,
        requirement=result_requirement,
        provenance=result_provenance,
        final_narrative=None,
        stalled=False,
        prior_evidence=prior_evidence,
        admitted_task=resolved_admitted_task,
        prior_terminal_tool_result=prior_terminal_tool_result,
      ), None

    if task_entry is not None:
      task_entry.delivered_messages.update(message.message_id for message in parent_messages)
      task_entry.accepted_parent_messages.update({
        message.message_id: message
        for message in parent_messages
      })

    if self._sub_agent_config is not None:
      if system_prompt is None:
        system_prompt = self._sub_agent_config.system_prompt
      if excluded_tools is None:
        excluded_tools = set(self._sub_agent_config.excluded_tools)

    derive_sub_agent_id = _runner_attr(self, "_derive_sub_agent_id", _derive_sub_agent_id)
    sub_session_id = str(getattr(sub_session, "session_id", "") or derive_sub_agent_id(self._full_session_id, call_index))
    progress_tracker_factory = _runner_attr(self, "make_progress_tracker", make_progress_tracker)
    progress_cb = progress_tracker_factory(task_entry) if task_entry else None

    event_log_cls = _runner_attr(self, "EventLog", EventLog)
    activity = _ChildActivity()
    sub_log = _build_child_event_log(
      parent_log=self._log,
      event_log_cls=event_log_cls,
      sub_session_id=sub_session_id,
      activity=activity,
      progress_cb=progress_cb,
      on_sub_event=on_sub_event,
    )
    dispatcher._event_log = sub_log
    child_cost_accumulator = CostAccumulator(cost_observation_threshold_usd)
    # The parent's own aggregator is ordinary delegation: the resumed child
    # keeps its own row under the parent's unit. Any other aggregator is
    # shared as given; None keeps the child's accounting independent.
    delegated_child = (
      usage_aggregator is not None and usage_aggregator is self._aggregator
    )
    runner_cls = _runner_attr(self, "AgentRunner", type(self))
    child_spill_sink = _child_spill_sink(self._spill_dir_provider, attempt)
    sub_runner = runner_cls(
      event_log=sub_log,
      dispatcher=dispatcher,
      session_id=sub_session_id,
      gateway_session=sub_session,
      context_research_file_id=(
        current_parent_context(sub_session).research_binding.research_file_id
        if sub_session is not None
        else None
      ),
      capability_execution=capability_execution,
      max_tokens_override=max_tokens,
      stream_stall_timeout=self._stream_stall_timeout,
      mcp_client=self._mcp_client,
      mcp_activation_fold=self._mcp_activation_fold,
      excluded_tools=excluded_tools or set(),
      get_tool_definitions=child_get_tool_definitions,
      on_tool_result=self._on_tool_result,
      on_usage=self._on_usage,
      on_session_summary=(
        self._on_session_summary if delegated_child else None
      ),
      on_late_usage_event=self._on_late_usage_event,
      on_tool_timing=self._on_tool_timing,
      user_id=(
        session_owner_user_id(sub_session)
        if sub_session is not None
        else self._usage_user_id
      ),
      request_id=self._request_id,
      parent_turn_id=parent_turn_id,
      billing_mode=self._billing_mode,
      rate_table_version=self._rate_table_version,
      channel=self._channel,
      usage_ledger_dlq_path=self._usage_ledger_dlq_path,
      on_metric=self._on_metric,
      sub_agent_config=self._sub_agent_config,
      compaction_trigger=self._compaction_trigger,
      compaction_instructions=None,
      tool_call_timeout=self._tool_call_timeout,
      _cost_accumulator=child_cost_accumulator,
      _parent_aggregator=None if delegated_child else usage_aggregator,
      _usage_parent=self._aggregator if delegated_child else None,
      containment=self._containment,
      result_requirement=result_requirement,
      agent_session_log=self._agent_session_log,
      message_inbox=task_entry.message_inbox if task_entry else None,
      max_resume_chain_depth=self._max_resume_chain_depth,
      emit_session_recap=False,
      code_execution_spill_dir_provider=child_spill_sink,
      commercial_usage_producer=getattr(self, "_commercial_usage_producer", None),
      skill_run_id=skill_run_id or self._skill_run_id,
      workspace_dir=self._workspace_dir,
      batch_id=getattr(self, "_batch_id", None),
      context_surfaces=self._context_surfaces_provider or self._context_surfaces_static,
      terminal_tool_result_ids=set(terminal_result_doors(resolved_admitted_task)),
    )
    if (
      resolved_admitted_task is not None
      and resolved_admitted_task.operation.execution_class == "node.fork"
    ):
      sub_runner._fork_mode = True
      sub_runner._soft_budget_kind = "fork"
      sub_runner._fork_suffix_tokens = 0
      # The continued suffix starts after its own recovered child history,
      # never after a fresh snapshot of the parent's current conversation.
      # A child interrupted before its first message pins the continuation.
      sub_runner._fork_marker_position = next(
        (message_index, block_index)
        for message_index in (
          *range(len(reconstructed_messages) - 2, -1, -1),
          len(reconstructed_messages) - 1,
        )
        for content in [reconstructed_messages[message_index].get("content")]
        for blocks in [
          [{"type": "text", "text": content}]
          if isinstance(content, str)
          else content if isinstance(content, list) else []
        ]
        for block_index in range(len(blocks) - 1, -1, -1)
        if _is_cacheable_message_block(blocks[block_index])
      )
    _bind_child_registry(
      sub_runner,
      parent_registry=self._task_registry,
      attempt=attempt,
      sub_session=sub_session,
      task_entry=task_entry,
      spill_sink=child_spill_sink,
      bind_job_runner=bind_job_runner,
    )
    if approved_call is not None:
      sub_runner.bind_resume_approved_call(
        approved_call, dispatcher.execute_approved_binding,
      )
    dispatcher.bind_research_file_activity_owner(sub_runner.ensure_research_file_activity)
    if bind_research_file_activity_lease_func is not None:
      bind_research_file_activity_lease_func(sub_runner)
    sub_runner._resume_parent_messages_for_ack = tuple(parent_messages)
    stalled = False
    runtime_exception_detail: str | None = None
    cancelled_error: asyncio.CancelledError | None = None
    cancellation_signal: str | None = None
    cleanup_warnings: list[str] = []
    user_turn_message = _runner_attr(self, "_user_turn_message", _user_turn_message)
    coro = sub_runner.run(
      messages=reconstructed_messages[-1:] or [user_turn_message("")],
      system_prompt=system_prompt,
      resume_initial_messages=reconstructed_messages,
    )
    try:
      silence = await _await_child_under_activity_guard(
        coro, activity,
        silence_timeout=self._containment.child_silence_timeout_seconds,
      )
      if silence is not None:
        stalled = True
        detail = f"Sub-agent stalled: no activity for {silence:.0f}s"
        _runner_attr(self, "log", log).warning("[%s] %s", sub_session_id, detail)
        sub_log.append({"type": "error", "error": detail, "error_sub_code": "stalled"})
    except asyncio.CancelledError as exc:
      cancelled_error = exc
      cleanup_warnings.extend(cleanup_failure_notes(exc))
      cancellation_signal = (
        task_entry.termination_intent
        if task_entry is not None
        and task_entry.termination_intent is not None
        else "cancelled"
      )
      _runner_attr(self, "log", log).warning("[%s] Resumed sub-agent cancelled (parent disconnect or shutdown)", sub_session_id)
      sub_log.append({"type": "error", "error": "Sub-agent cancelled"})
    except ApprovalPark:
      raise
    except Exception as exc:
      runtime_exception_detail = _runtime_exception_detail(exc)
      cleanup_warnings.extend(cleanup_failure_notes(exc))
      _runner_attr(self, "log", log).warning(
        "[%s] Resumed sub-agent failed: %s",
        sub_session_id,
        runtime_exception_detail,
      )
      sub_log.append(
        {"type": "error", "error": runtime_exception_detail}
      )
    finally:
      approval_park = sub_runner._approval_park
      (
        cancelled_error,
        cancellation_signal,
        runtime_exception_detail,
        cleanup_warnings,
      ) = await _close_sub_runner(
        sub_runner,
        sub_log,
        stalled=stalled,
        approval_park=approval_park,
        cancelled_error=cancelled_error,
        cancellation_signal=cancellation_signal,
        runtime_exception_detail=runtime_exception_detail,
        task_entry=task_entry,
        cleanup_warnings=cleanup_warnings,
        sub_session=sub_session,
      )

    if (
      approval_park is not None
      and sub_runner._terminal_closure_event is not None
      and sub_runner._terminal_closure_event.get("terminal_disposition") == "awaiting_approval"
      and cancelled_error is None
      and runtime_exception_detail is None
      and not stalled
    ):
      evidence = merge_sub_agent_result_evidence(
        prior_evidence,
        collect_sub_agent_result_evidence(sub_log.entries, durable=False),
      )
      return terminal_task_result(
        cast(AdmittedTask, resolved_admitted_task),
        status="awaiting_approval",
        reason=(
          f"approval:{approval_park.approval_id} "
          f"call:{approval_park.original_tool_call_id}"
        ),
        tools_used=evidence.tools_used,
        usage=_usage_observation(evidence.usage),
      ), None

    terminal_tool_result = latest_successful_declared_terminal_tool_result(
      sub_log.entries,
      declared_terminal_doors=terminal_result_doors(resolved_admitted_task),
    )
    final_narrative = None
    if (
      terminal_tool_result is None
      and result_requirement.terminal_narrative != "forbidden"
      and self._workspace_dir is not None
    ):
      sub_runner_id = getattr(sub_runner, "_runner_id", None)
      if not isinstance(sub_runner_id, str) or not sub_runner_id:
        raise RuntimeError(
          "narrative child completion requires its exact durable runner_id"
        )
      narrative_text = await final_child_visible_text(
        self._agent_session_log,
        sub_session_id=sub_session_id,
        role="sub_agent",
        sub_agent_id=sub_session_id,
        workspace_dir=self._workspace_dir,
        runner_id=sub_runner_id,
      )
      final_narrative = narrative_text.final_narrative
    result = task_result_from_execution(
      sub_log.entries,
      logical_task=logical_task,
      attempt=attempt,
      requirement=result_requirement,
      provenance=result_provenance,
      final_narrative=final_narrative,
      stalled=stalled,
      runtime_error_detail=runtime_exception_detail,
      external_terminal_signals=(
        [cancellation_signal]
        if cancellation_signal is not None
        else []
      ),
      prior_evidence=prior_evidence,
      # B-3: the authority frozen at admission, never the ambient catalog.
      admitted_task=resolved_admitted_task,
    )
    if cancelled_error is not None and terminal_tool_result is None:
      if task_entry is not None:
        task_entry.task_result = result
        task_entry.result = result.model_dump(mode="json")
      raise cancelled_error
    return result, None


def _runtime_exception_detail(exc: Exception) -> str:
  message = str(exc).strip()
  if not message:
    return type(exc).__name__
  return f"{type(exc).__name__}: {message}"


def _child_spill_sink(
  parent_sink: SpillSink | None,
  attempt: AttemptRef,
) -> SpillSink | None:
  """The child job's own spill sink, under its parent's spill root."""

  if parent_sink is None:
    return None
  return job_spill_sink(parent_sink, attempt.physical_task_id)


def _bind_child_registry(
  sub_runner: Any,
  *,
  parent_registry: TaskRegistry,
  attempt: AttemptRef,
  sub_session: Any | None,
  task_entry: TaskEntry | None,
  spill_sink: SpillSink | None,
  bind_job_runner: Callable[[Any], None],
) -> None:
  """Name the child job its registry belongs to and link it into the tree.

  The child's own children register in ``sub_runner``'s registry; the link to
  the parent's registry is what puts them on the tree roster. A background
  child's mailbox is its task entry's inbox; a foreground child has none.
  """

  sub_runner._task_registry.bind_owner(RegistryOwner(
    job_id=attempt.physical_task_id,
    tenant=lambda: (
      current_parent_context(sub_session).tenant
      if sub_session is not None
      else None
    ),
    inbox=task_entry.message_inbox if task_entry is not None else None,
    parent=parent_registry,
    spill_sink=spill_sink,
  ))
  bind_job_runner(sub_runner)


def _termination_signal(task_entry: TaskEntry | None) -> str:
  if (
    task_entry is not None
    and task_entry.termination_intent is not None
  ):
    return task_entry.termination_intent
  return "cancelled"


async def _close_sub_runner(
  sub_runner: Any,
  sub_log: Any,
  *,
  stalled: bool,
  approval_park: ApprovalPark | None,
  cancelled_error: asyncio.CancelledError | None,
  cancellation_signal: str | None,
  runtime_exception_detail: str | None,
  task_entry: TaskEntry | None,
  cleanup_warnings: list[str],
  sub_session: Any | None = None,
) -> tuple[
  asyncio.CancelledError | None,
  str | None,
  str | None,
  list[str],
]:
  try:
    await sub_runner.force_close(timeout=2.0)
  except asyncio.CancelledError as exc:
    detail = "Sub-agent cleanup was cancelled"
    if detail not in cleanup_warnings:
      cleanup_warnings.append(detail)
    sub_log.append({
      "type": "run_error",
      "phase": "force_close",
      "error_type": type(exc).__name__,
      "error": detail,
      "message": detail,
    })
    if cancelled_error is None and not stalled:
      cancelled_error = exc
      cancellation_signal = _termination_signal(task_entry)
  except Exception as exc:
    detail = f"Sub-agent cleanup failed: {_runtime_exception_detail(exc)}"
    if detail not in cleanup_warnings:
      cleanup_warnings.append(detail)
    sub_log.append({
      "type": "run_error",
      "phase": "force_close",
      "error_type": type(exc).__name__,
      "error": detail,
      "message": detail,
    })
    if (
      cancelled_error is None
      and not stalled
      and runtime_exception_detail is None
    ):
      runtime_exception_detail = detail
      cleanup_warnings.remove(detail)
  finally:
    session_cleanup_completed = False
    try:
      if sub_session is not None:
        from .code_execution import cleanup_code_execution

        cleanup_task = asyncio.create_task(
          cleanup_code_execution(sub_session),
          name=f"sub-agent-session:{sub_session.session_id}:cleanup",
        )
        try:
          await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
          # The waiter's cancellation and the cleanup's own re-raised
          # cancellation are one outcome: drain the release, record it once.
          try:
            await drain_owned_lifecycle_task(cleanup_task)
          except asyncio.CancelledError:
            pass
          if cancelled_error is None and not stalled:
            cancelled_error = exc
            cancellation_signal = _termination_signal(task_entry)
      session_cleanup_completed = True
    finally:
      if (
        approval_park is None
        or sub_runner._terminal_closure_event is None
        or sub_runner._terminal_closure_event.get("terminal_disposition") != "awaiting_approval"
        or cancelled_error is not None
        or runtime_exception_detail is not None
        or stalled
        or not session_cleanup_completed
      ):
        sub_runner._task_registry.complete_owner()
  return (
    cancelled_error,
    cancellation_signal,
    runtime_exception_detail,
    cleanup_warnings,
  )
