from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .event_log import EventLog
from .providers import ModelProvider
from .runner_background_tasks import sub_agent_result_from_log_entries as _sub_agent_result_from_log_entries
from .runner_introspection import derive_sub_agent_id as _derive_sub_agent_id
from .runner_session_lifecycle import _runner_attr
from .runner_state import ChildCostAccumulator, user_turn_message as _user_turn_message
from .task_registry import ParentMessage, TaskEntry, make_progress_tracker
from .tool_dispatcher import ToolDispatcher


log = logging.getLogger("agent_gateway.runner")


class RunnerSubAgentMixin:
  async def spawn_sub_agent(
    self,
    task: str,
    *,
    provider: ModelProvider | None = None,
    auth_config: Dict[str, Any] | None = None,
    model: str | None = None,
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
    max_budget_usd: float | None = None,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
  ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Run a focused sub-agent task and return its summarized result.

    This method is used by the built-in `run_agent` tool. The sub-agent shares
    the same budget accounting as the parent runner, but it gets a fresh
    `EventLog`, its own turn budget, and its own dispatcher. By default the
    sub-agent inherits the parent's provider, but callers may override it with
    an explicit `provider` + `auth_config` pair.
    """
    if self._sub_agent_config is not None:
      if model is None:
        model = self._sub_agent_config.model
      if system_prompt is None:
        system_prompt = self._sub_agent_config.system_prompt
      if excluded_tools is None:
        excluded_tools = set(self._sub_agent_config.excluded_tools)

    effective_provider = provider or self._provider
    if provider is not None:
      if auth_config is None:
        return None, {"code": "invalid_input", "message": "auth_config required when overriding provider"}
      effective_auth = dict(auth_config)
    else:
      effective_auth = getattr(sub_session, "auth_config", None) or self._auth_config

    derive_sub_agent_id = _runner_attr(self, "_derive_sub_agent_id", _derive_sub_agent_id)
    sub_session_id = str(getattr(sub_session, "session_id", "") or derive_sub_agent_id(self._full_session_id, call_index))
    original_on_event = getattr(self._log, "_on_event", None)
    progress_tracker_factory = _runner_attr(self, "make_progress_tracker", make_progress_tracker)
    progress_cb = progress_tracker_factory(task_entry) if task_entry else None

    def _composed_on_event(event: Dict[str, Any], session_id: str) -> None:
      event_copy = dict(event)
      event_copy.setdefault("sub_agent_id", session_id)
      if progress_cb is not None:
        try:
          progress_cb(event_copy, session_id)
        except Exception:
          pass
      if original_on_event is not None:
        try:
          original_on_event(event_copy, session_id)
        except Exception:
          pass
      if on_sub_event is not None:
        try:
          on_sub_event(event_copy, session_id)
        except Exception:
          pass

    event_log_cls = _runner_attr(self, "EventLog", EventLog)
    sub_log = event_log_cls(
      on_event=_composed_on_event,
      session_id=sub_session_id,
    )
    child_max_budget_usd = self._max_budget_usd
    child_cost_accumulator = self._cost_accumulator
    if max_budget_usd is not None:
      child_max_budget_usd = max_budget_usd
      child_accumulator_cls = _runner_attr(self, "ChildCostAccumulator", ChildCostAccumulator)
      child_cost_accumulator = child_accumulator_cls(self._cost_accumulator, max_budget_usd)
    runner_cls = _runner_attr(self, "AgentRunner", type(self))
    sub_runner = runner_cls(
      event_log=sub_log,
      dispatcher=dispatcher,
      session_id=sub_session_id,
      provider=effective_provider,
      auth_config=effective_auth,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      per_turn_timeout=per_turn_timeout if per_turn_timeout is not None else self._per_turn_timeout,
      stream_stall_timeout=self._stream_stall_timeout,
      mcp_client=self._mcp_client,
      loaded_mcp_servers=self._loaded_mcp_servers,
      excluded_tools=excluded_tools or set(),
      get_tool_definitions=self._get_tool_definitions,
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
      max_budget_usd=child_max_budget_usd,
      _cost_accumulator=child_cost_accumulator,
      _parent_aggregator=self._aggregator,
      max_concurrent_sub_agents=self._max_concurrent_sub_agents,
      agent_session_log=self._agent_session_log,
      message_inbox=task_entry.message_inbox if task_entry else None,
      max_resume_chain_depth=self._max_resume_chain_depth,
      emit_session_recap=False,
      code_execution_spill_dir_provider=self._spill_dir_provider,
      skill_run_id=self._skill_run_id,
      workspace_dir=self._workspace_dir,
      context_surfaces=self._context_surfaces_provider or self._context_surfaces_static,
    )

    timed_out = False
    user_turn_message = _runner_attr(self, "_user_turn_message", _user_turn_message)
    coro = sub_runner.run(
      messages=[user_turn_message(task)],
      system_prompt=system_prompt,
      model_override=model,
      max_turns=max_turns,
    )
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    timeout_error = getattr(asyncio_module, "TimeoutError", asyncio.TimeoutError)
    try:
      if timeout is not None and timeout > 0:
        await asyncio_module.wait_for(coro, timeout=timeout)
      else:
        await coro
    except timeout_error:
      timed_out = True
      sub_log.append({"type": "error", "error": f"Sub-agent timed out after {timeout}s"})
    except asyncio.CancelledError:
      _runner_attr(self, "log", log).warning("[%s] Sub-agent cancelled (parent disconnect or shutdown)", sub_session_id)
      sub_log.append({"type": "error", "error": "Sub-agent cancelled"})
      raise
    finally:
      await sub_runner.force_close(timeout=2.0)

    result_from_log_entries = _runner_attr(
      self,
      "_sub_agent_result_from_log_entries",
      _sub_agent_result_from_log_entries,
    )
    return result_from_log_entries(
      sub_log.entries,
      timed_out=timed_out,
      timeout=timeout,
      budget_exceeded_reason=getattr(child_cost_accumulator, "exceeded_reason", None),
    ), None

  async def resume_sub_agent(
    self,
    *,
    original_task_id: str,
    reconstructed_messages: List[Dict[str, Any]],
    parent_messages: list[ParentMessage],
    provider: ModelProvider | None = None,
    auth_config: Dict[str, Any] | None = None,
    model: str | None = None,
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
    max_budget_usd: float | None = None,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
  ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if task_entry is not None:
      task_entry.delivered_messages.update(message.message_id for message in parent_messages)

    if self._sub_agent_config is not None:
      if model is None:
        model = self._sub_agent_config.model
      if system_prompt is None:
        system_prompt = self._sub_agent_config.system_prompt
      if excluded_tools is None:
        excluded_tools = set(self._sub_agent_config.excluded_tools)

    effective_provider = provider or self._provider
    if provider is not None:
      if auth_config is None:
        return None, {"code": "invalid_input", "message": "auth_config required when overriding provider"}
      effective_auth = dict(auth_config)
    else:
      effective_auth = getattr(sub_session, "auth_config", None) or self._auth_config

    derive_sub_agent_id = _runner_attr(self, "_derive_sub_agent_id", _derive_sub_agent_id)
    sub_session_id = str(getattr(sub_session, "session_id", "") or derive_sub_agent_id(self._full_session_id, call_index))
    original_on_event = getattr(self._log, "_on_event", None)
    progress_tracker_factory = _runner_attr(self, "make_progress_tracker", make_progress_tracker)
    progress_cb = progress_tracker_factory(task_entry) if task_entry else None

    def _composed_on_event(event: Dict[str, Any], session_id: str) -> None:
      event_copy = dict(event)
      event_copy.setdefault("sub_agent_id", session_id)
      if progress_cb is not None:
        try:
          progress_cb(event_copy, session_id)
        except Exception:
          pass
      if original_on_event is not None:
        try:
          original_on_event(event_copy, session_id)
        except Exception:
          pass
      if on_sub_event is not None:
        try:
          on_sub_event(event_copy, session_id)
        except Exception:
          pass

    event_log_cls = _runner_attr(self, "EventLog", EventLog)
    sub_log = event_log_cls(on_event=_composed_on_event, session_id=sub_session_id)
    child_max_budget_usd = self._max_budget_usd
    child_cost_accumulator = self._cost_accumulator
    if max_budget_usd is not None:
      child_max_budget_usd = max_budget_usd
      child_accumulator_cls = _runner_attr(self, "ChildCostAccumulator", ChildCostAccumulator)
      child_cost_accumulator = child_accumulator_cls(self._cost_accumulator, max_budget_usd)
    runner_cls = _runner_attr(self, "AgentRunner", type(self))
    sub_runner = runner_cls(
      event_log=sub_log,
      dispatcher=dispatcher,
      session_id=sub_session_id,
      provider=effective_provider,
      auth_config=effective_auth,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      per_turn_timeout=per_turn_timeout if per_turn_timeout is not None else self._per_turn_timeout,
      stream_stall_timeout=self._stream_stall_timeout,
      mcp_client=self._mcp_client,
      loaded_mcp_servers=self._loaded_mcp_servers,
      excluded_tools=excluded_tools or set(),
      get_tool_definitions=self._get_tool_definitions,
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
      max_budget_usd=child_max_budget_usd,
      _cost_accumulator=child_cost_accumulator,
      _parent_aggregator=self._aggregator,
      max_concurrent_sub_agents=self._max_concurrent_sub_agents,
      agent_session_log=self._agent_session_log,
      message_inbox=task_entry.message_inbox if task_entry else None,
      max_resume_chain_depth=self._max_resume_chain_depth,
      emit_session_recap=False,
      code_execution_spill_dir_provider=self._spill_dir_provider,
      skill_run_id=self._skill_run_id,
      workspace_dir=self._workspace_dir,
      context_surfaces=self._context_surfaces_provider or self._context_surfaces_static,
    )

    timed_out = False
    user_turn_message = _runner_attr(self, "_user_turn_message", _user_turn_message)
    coro = sub_runner.run(
      messages=reconstructed_messages[-1:] or [user_turn_message("")],
      system_prompt=system_prompt,
      model_override=model,
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
    except timeout_error:
      timed_out = True
      sub_log.append({"type": "error", "error": f"Sub-agent timed out after {timeout}s"})
    except asyncio.CancelledError:
      _runner_attr(self, "log", log).warning("[%s] Resumed sub-agent cancelled (parent disconnect or shutdown)", sub_session_id)
      sub_log.append({"type": "error", "error": "Sub-agent cancelled"})
      raise
    finally:
      await sub_runner.force_close(timeout=2.0)

    result_from_log_entries = _runner_attr(
      self,
      "_sub_agent_result_from_log_entries",
      _sub_agent_result_from_log_entries,
    )
    return result_from_log_entries(
      sub_log.entries,
      timed_out=timed_out,
      timeout=timeout,
      budget_exceeded_reason=getattr(child_cost_accumulator, "exceeded_reason", None),
      original_task_id=original_task_id,
    ), None
