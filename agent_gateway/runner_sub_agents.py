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
)

from agent_workflow_contracts import (
  AdmittedTask,
  AttemptRef,
  LogicalTaskRef,
  ResultRequirement,
  TaskResult,
  TaskResultProvenance,
)

from .capability_execution import BoundCapabilityExecution
from .event_log import EventLog
from .runner_introspection import derive_sub_agent_id as _derive_sub_agent_id
from .runner_session_lifecycle import _runner_attr
from .runner_cleanup import cleanup_failure_notes
from .runner_budget import (
  ChildCostAccumulator,
  ObservationOnlyCostAccumulator,
)
from .runner_stream_turn import STREAM_GUARD_POLL_INTERVAL
from .runner_streaming import STREAM_THINKING_STALL_TIMEOUT
from .runner_state import user_turn_message as _user_turn_message
from .sub_agent_result_evidence import SubAgentResultEvidence
from .sub_agent_narrative_result import (
  final_child_visible_text,
  task_result_from_execution,
)
from .sub_agent_skill_state import (
  declared_terminal_doors_from_grant,
  latest_successful_declared_terminal_tool_result,
)
from .task_registry import ParentMessage, TaskEntry, make_progress_tracker
from .tool_dispatcher import ToolDispatcher

if TYPE_CHECKING:
  from pathlib import Path

  from .agent_session_log import AgentSessionLog
  from .mcp_activation import McpActivationFold
  from .mcp_client import McpClientManager
  from .multi_user.billing import UsageEvent, _UsageAggregator
  from .runner_state import SubAgentConfig, ToolResultContext
  from .tool_result_spill import SpillSink

log = logging.getLogger("agent_gateway.runner")
_MISSING_SUB_AGENT_ID = object()
# A child that emits nothing for this long with no tool in flight is wedged
# (ACUI-1: an await outside both stream and tool). Derived from the stream
# guard so the child's own stall guard always fires first on stream silence.
SUB_AGENT_ACTIVITY_GAP = 2 * STREAM_THINKING_STALL_TIMEOUT


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
) -> float | None:
  """Await one child run; cancel it once it is silent with no tool in flight.

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
      if not activity.tools_in_flight and silence > SUB_AGENT_ACTIVITY_GAP:
        child.cancel()
        await asyncio.wait({child})
        return silence
  except asyncio.CancelledError:
    if not child.done():
      child.cancel()
      await asyncio.wait({child})
    raise


def _child_cost_accumulator(
  runner: Any,
  *,
  cost_observation_threshold_usd: float | None,
  max_budget_usd: float | None,
) -> Any:
  """Create child hard-budget authority over observational telemetry."""

  observation_accumulator_cls = _runner_attr(
    runner,
    "ObservationOnlyCostAccumulator",
    ObservationOnlyCostAccumulator,
  )
  observation_accumulator = observation_accumulator_cls(
    cost_observation_threshold_usd
  )
  if max_budget_usd is None:
    return observation_accumulator
  child_accumulator_cls = _runner_attr(
    runner,
    "ChildCostAccumulator",
    ChildCostAccumulator,
  )
  return child_accumulator_cls(observation_accumulator, max_budget_usd)


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
  if requirement.mode != "narrative":
    raise ValueError("agent execution accepts terminal-message results only")


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
    if event_copy.get("type") == "tool_approval_request":
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
        prepared = original_prepare_event(event)
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
    _max_concurrent_sub_agents: int | None
    _max_resume_chain_depth: int
    _mcp_activation_fold: McpActivationFold
    _mcp_client: McpClientManager | None
    _on_late_usage_event: (
      Callable[[UsageEvent], Awaitable[None] | None] | None
    )
    _on_max_turns: (
      Callable[[List[Dict[str, Any]], int], Awaitable[str | None]]
      | None
    )
    _on_metric: Callable[[str, int], None] | None
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
    max_turns: int | None,
    client_timeout: float = 90,
    max_tokens: int = 64000,
    call_index: int = 0,
    parent_turn_id: str | None = None,
    task_entry: TaskEntry | None = None,
    cost_observation_threshold_usd: float | None = None,
    max_budget_usd: float | None = None,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
    skill_run_id: str | None = None,
    admitted_task: AdmittedTask | None = None,
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
    if not capability_execution.bind.capability_id.startswith("node."):
      raise ValueError("spawn_sub_agent requires a node.* capability bind")
    capability_execution.validate()
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
    child_cost_accumulator = _child_cost_accumulator(
      self,
      cost_observation_threshold_usd=cost_observation_threshold_usd,
      max_budget_usd=max_budget_usd,
    )
    runner_cls = _runner_attr(self, "AgentRunner", type(self))
    sub_runner = runner_cls(
      event_log=sub_log,
      dispatcher=dispatcher,
      session_id=sub_session_id,
      capability_execution=capability_execution,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      stream_stall_timeout=self._stream_stall_timeout,
      mcp_client=self._mcp_client,
      mcp_activation_fold=self._mcp_activation_fold,
      excluded_tools=excluded_tools or set(),
      get_tool_definitions=child_get_tool_definitions,
      on_tool_result=self._on_tool_result,
      on_usage=self._on_usage,
      on_session_summary=None,
      on_late_usage_event=self._on_late_usage_event,
      on_tool_timing=self._on_tool_timing,
      user_id=getattr(sub_session, "user_id", None) or self._usage_user_id,
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
      on_max_turns=self._on_max_turns,
      max_budget_usd=max_budget_usd,
      _cost_accumulator=child_cost_accumulator,
      _parent_aggregator=self._aggregator,
      max_concurrent_sub_agents=self._max_concurrent_sub_agents,
      result_requirement=result_requirement,
      agent_session_log=self._agent_session_log,
      message_inbox=task_entry.message_inbox if task_entry else None,
      max_resume_chain_depth=self._max_resume_chain_depth,
      emit_session_recap=False,
      code_execution_spill_dir_provider=self._spill_dir_provider,
      commercial_usage_producer=getattr(self, "_commercial_usage_producer", None),
      skill_run_id=skill_run_id or self._skill_run_id,
      workspace_dir=self._workspace_dir,
      batch_id=getattr(self, "_batch_id", None),
      context_surfaces=self._context_surfaces_provider or self._context_surfaces_static,
      terminal_tool_result_ids=set(
        declared_terminal_doors_from_grant(
          resolved_admitted_task.tool_grant
          if resolved_admitted_task is not None
          else None
        )
      ),
    )
    stalled = False
    runtime_exception_detail: str | None = None
    cancelled_error: asyncio.CancelledError | None = None
    cancellation_signal: str | None = None
    cleanup_warnings: list[str] = []
    user_turn_message = _runner_attr(self, "_user_turn_message", _user_turn_message)
    coro = sub_runner.run(
      messages=[user_turn_message(task)],
      system_prompt=system_prompt,
      max_turns=max_turns,
    )
    try:
      silence = await _await_child_under_activity_guard(coro, activity)
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
      (
        cancelled_error,
        cancellation_signal,
        runtime_exception_detail,
        cleanup_warnings,
      ) = await _close_sub_runner(
        sub_runner,
        sub_log,
        stalled=stalled,
        cancelled_error=cancelled_error,
        cancellation_signal=cancellation_signal,
        runtime_exception_detail=runtime_exception_detail,
        task_entry=task_entry,
        cleanup_warnings=cleanup_warnings,
      )

    terminal_tool_result = latest_successful_declared_terminal_tool_result(
      sub_log.entries,
      declared_terminal_doors=declared_terminal_doors_from_grant(
        resolved_admitted_task.tool_grant
        if resolved_admitted_task is not None
        else None
      ),
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
    max_turns: int | None,
    client_timeout: float = 90,
    max_tokens: int = 64000,
    call_index: int = 0,
    parent_turn_id: str | None = None,
    task_entry: TaskEntry | None = None,
    cost_observation_threshold_usd: float | None = None,
    max_budget_usd: float | None = None,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
    skill_run_id: str | None = None,
    admitted_task: AdmittedTask | None = None,
    prior_terminal_tool_result: Mapping[str, Any] | None = None,
    bind_research_file_activity_lease_func: (
      Callable[[Any], None] | None
    ) = None,
  ) -> Tuple[Optional[TaskResult], Optional[Dict[str, Any]]]:
    if not isinstance(capability_execution, BoundCapabilityExecution):
      raise TypeError(
        "resume_sub_agent requires a BoundCapabilityExecution"
      )
    if not capability_execution.bind.capability_id.startswith("node."):
      raise ValueError("resume_sub_agent requires a node.* capability bind")
    capability_execution.validate()
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
    child_cost_accumulator = _child_cost_accumulator(
      self,
      cost_observation_threshold_usd=cost_observation_threshold_usd,
      max_budget_usd=max_budget_usd,
    )
    runner_cls = _runner_attr(self, "AgentRunner", type(self))
    sub_runner = runner_cls(
      event_log=sub_log,
      dispatcher=dispatcher,
      session_id=sub_session_id,
      capability_execution=capability_execution,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      stream_stall_timeout=self._stream_stall_timeout,
      mcp_client=self._mcp_client,
      mcp_activation_fold=self._mcp_activation_fold,
      excluded_tools=excluded_tools or set(),
      get_tool_definitions=child_get_tool_definitions,
      on_tool_result=self._on_tool_result,
      on_usage=self._on_usage,
      on_session_summary=None,
      on_late_usage_event=self._on_late_usage_event,
      on_tool_timing=self._on_tool_timing,
      user_id=getattr(sub_session, "user_id", None) or self._usage_user_id,
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
      on_max_turns=self._on_max_turns,
      max_budget_usd=max_budget_usd,
      _cost_accumulator=child_cost_accumulator,
      _parent_aggregator=self._aggregator,
      max_concurrent_sub_agents=self._max_concurrent_sub_agents,
      result_requirement=result_requirement,
      agent_session_log=self._agent_session_log,
      message_inbox=task_entry.message_inbox if task_entry else None,
      max_resume_chain_depth=self._max_resume_chain_depth,
      emit_session_recap=False,
      code_execution_spill_dir_provider=self._spill_dir_provider,
      commercial_usage_producer=getattr(self, "_commercial_usage_producer", None),
      skill_run_id=skill_run_id or self._skill_run_id,
      workspace_dir=self._workspace_dir,
      batch_id=getattr(self, "_batch_id", None),
      context_surfaces=self._context_surfaces_provider or self._context_surfaces_static,
      terminal_tool_result_ids=set(
        declared_terminal_doors_from_grant(
          resolved_admitted_task.tool_grant
          if resolved_admitted_task is not None
          else None
        )
      ),
    )
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
      max_turns=max_turns,
      resume_initial_messages=reconstructed_messages,
    )
    try:
      silence = await _await_child_under_activity_guard(coro, activity)
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
      (
        cancelled_error,
        cancellation_signal,
        runtime_exception_detail,
        cleanup_warnings,
      ) = await _close_sub_runner(
        sub_runner,
        sub_log,
        stalled=stalled,
        cancelled_error=cancelled_error,
        cancellation_signal=cancellation_signal,
        runtime_exception_detail=runtime_exception_detail,
        task_entry=task_entry,
        cleanup_warnings=cleanup_warnings,
      )

    terminal_tool_result = latest_successful_declared_terminal_tool_result(
      sub_log.entries,
      declared_terminal_doors=declared_terminal_doors_from_grant(
        resolved_admitted_task.tool_grant
        if resolved_admitted_task is not None
        else None
      ),
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
  cancelled_error: asyncio.CancelledError | None,
  cancellation_signal: str | None,
  runtime_exception_detail: str | None,
  task_entry: TaskEntry | None,
  cleanup_warnings: list[str],
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
  return (
    cancelled_error,
    cancellation_signal,
    runtime_exception_detail,
    cleanup_warnings,
  )
