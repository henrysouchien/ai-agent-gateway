from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agent_workflow_contracts import (
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
  ObservationOnlyCostAccumulator,
)
from .runner_state import user_turn_message as _user_turn_message
from .sub_agent_result_evidence import SubAgentResultEvidence
from .sub_agent_narrative_result import (
  final_child_visible_text,
  task_result_from_execution,
)
from .task_registry import ParentMessage, TaskEntry, make_progress_tracker
from .tool_dispatcher import ToolDispatcher


log = logging.getLogger("agent_gateway.runner")
_MISSING_SUB_AGENT_ID = object()


def _child_cost_observation_accumulator(
  runner: Any,
  *,
  cost_observation_threshold_usd: float | None,
) -> Any:
  """Create non-authoritative child cost measurement state."""

  accumulator_cls = _runner_attr(
    runner,
    "ObservationOnlyCostAccumulator",
    ObservationOnlyCostAccumulator,
  )
  return accumulator_cls(cost_observation_threshold_usd)


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

  def _composed_on_event(
    event: Dict[str, Any],
    session_id: str,
  ) -> None:
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
  get_tool_definitions = getattr(
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
    timeout: float | None,
    client_timeout: float = 90,
    per_turn_timeout: float | None = None,
    max_tokens: int = 64000,
    call_index: int = 0,
    parent_turn_id: str | None = None,
    task_entry: TaskEntry | None = None,
    cost_observation_threshold_usd: float | None = None,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
    skill_run_id: str | None = None,
  ) -> Tuple[Optional[TaskResult], Optional[Dict[str, Any]]]:
    """Run a focused sub-agent task and return its canonical result.

    This method is used by the built-in `run_agent` tool. The sub-agent shares
    usage aggregation with the parent but records cost in a non-enforcing
    observation accumulator. It gets a fresh `EventLog`, its own structural
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
    if task_entry is not None and task_entry.admitted_task is not None:
      admitted = task_entry.admitted_task
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
    sub_log = _build_child_event_log(
      parent_log=self._log,
      event_log_cls=event_log_cls,
      sub_session_id=sub_session_id,
      progress_cb=progress_cb,
      on_sub_event=on_sub_event,
    )
    dispatcher._event_log = sub_log
    child_cost_accumulator = _child_cost_observation_accumulator(
      self,
      cost_observation_threshold_usd=cost_observation_threshold_usd,
    )
    runner_cls = _runner_attr(self, "AgentRunner", type(self))
    sub_runner = runner_cls(
      event_log=sub_log,
      dispatcher=dispatcher,
      session_id=sub_session_id,
      capability_execution=capability_execution,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      per_turn_timeout=per_turn_timeout if per_turn_timeout is not None else self._per_turn_timeout,
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
      max_budget_usd=None,
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
    )
    timed_out = False
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
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    timeout_error = getattr(asyncio_module, "TimeoutError", asyncio.TimeoutError)
    try:
      if timeout is not None and timeout > 0:
        await asyncio_module.wait_for(coro, timeout=timeout)
      else:
        await coro
    except timeout_error as exc:
      timed_out = True
      cleanup_warnings.extend(cleanup_failure_notes(exc))
      sub_log.append({"type": "error", "error": f"Sub-agent timed out after {timeout}s"})
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
        timed_out=timed_out,
        cancelled_error=cancelled_error,
        cancellation_signal=cancellation_signal,
        runtime_exception_detail=runtime_exception_detail,
        task_entry=task_entry,
        cleanup_warnings=cleanup_warnings,
      )

    final_narrative = None
    if result_requirement.terminal_narrative != "forbidden":
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
      timed_out=timed_out,
      timeout=timeout,
      runtime_error_detail=runtime_exception_detail,
      external_terminal_signals=(
        [cancellation_signal]
        if cancellation_signal is not None
        else []
      ),
      # B-3: the authority frozen at admission, never the ambient catalog.
      admitted_task=(
        task_entry.admitted_task if task_entry is not None else None
      ),
    )
    if cancelled_error is not None:
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
    timeout: float | None,
    client_timeout: float = 90,
    per_turn_timeout: float | None = None,
    max_tokens: int = 64000,
    call_index: int = 0,
    parent_turn_id: str | None = None,
    task_entry: TaskEntry | None = None,
    cost_observation_threshold_usd: float | None = None,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
    skill_run_id: str | None = None,
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
    if task_entry is not None and task_entry.admitted_task is not None:
      admitted = task_entry.admitted_task
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
    sub_log = _build_child_event_log(
      parent_log=self._log,
      event_log_cls=event_log_cls,
      sub_session_id=sub_session_id,
      progress_cb=progress_cb,
      on_sub_event=on_sub_event,
    )
    dispatcher._event_log = sub_log
    child_cost_accumulator = _child_cost_observation_accumulator(
      self,
      cost_observation_threshold_usd=cost_observation_threshold_usd,
    )
    runner_cls = _runner_attr(self, "AgentRunner", type(self))
    sub_runner = runner_cls(
      event_log=sub_log,
      dispatcher=dispatcher,
      session_id=sub_session_id,
      capability_execution=capability_execution,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      per_turn_timeout=per_turn_timeout if per_turn_timeout is not None else self._per_turn_timeout,
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
      max_budget_usd=None,
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
    )
    sub_runner._resume_parent_messages_for_ack = tuple(parent_messages)
    timed_out = False
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
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    timeout_error = getattr(asyncio_module, "TimeoutError", asyncio.TimeoutError)
    try:
      if timeout is not None and timeout > 0:
        await asyncio_module.wait_for(coro, timeout=timeout)
      else:
        await coro
    except timeout_error as exc:
      timed_out = True
      cleanup_warnings.extend(cleanup_failure_notes(exc))
      sub_log.append({"type": "error", "error": f"Sub-agent timed out after {timeout}s"})
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
        timed_out=timed_out,
        cancelled_error=cancelled_error,
        cancellation_signal=cancellation_signal,
        runtime_exception_detail=runtime_exception_detail,
        task_entry=task_entry,
        cleanup_warnings=cleanup_warnings,
      )

    final_narrative = None
    if result_requirement.terminal_narrative != "forbidden":
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
      timed_out=timed_out,
      timeout=timeout,
      runtime_error_detail=runtime_exception_detail,
      external_terminal_signals=(
        [cancellation_signal]
        if cancellation_signal is not None
        else []
      ),
      prior_evidence=prior_evidence,
      # B-3: the authority frozen at admission, never the ambient catalog.
      admitted_task=(
        task_entry.admitted_task if task_entry is not None else None
      ),
    )
    if cancelled_error is not None:
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
  timed_out: bool,
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
    if cancelled_error is None and not timed_out:
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
      and not timed_out
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
