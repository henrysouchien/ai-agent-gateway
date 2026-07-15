from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from ._provider_utils import _get_default_model_for_provider
from .runner_limits import (
  CONTEXT_WARNING_PCT,
  effective_compaction_trigger as _effective_compaction_trigger,
  model_context_window as _model_context_window,
  system_prompt_estimate_text as _system_prompt_estimate_text,
  token_estimate_snapshot as _token_estimate_snapshot,
)
from .runner_prompt_rules import prepend_system_prompt_preamble as _prepend_system_prompt_preamble
from .runner_run_loop_defaults import (
  MAX_NOTIFICATIONS_PER_TURN as _MAX_NOTIFICATIONS_PER_TURN,
  MAX_TOKENS_CONTINUATIONS as _MAX_TOKENS_CONTINUATIONS,
  MAX_TOKENS_NUDGE as _MAX_TOKENS_NUDGE,
)
from .runner_session_lifecycle import _runner_attr
from .runner_session_events import (
  build_budget_exceeded_event as _build_budget_exceeded_event,
  build_budget_exceeded_text_event as _build_budget_exceeded_text_event,
  build_chat_done_log_data as _build_chat_done_log_data,
  build_context_warning_log_data as _build_context_warning_log_data,
  build_max_turns_reached_event as _build_max_turns_reached_event,
  build_max_turns_text_event as _build_max_turns_text_event,
  build_runtime_guard_event as _build_runtime_guard_event,
  build_stream_complete_event as _build_stream_complete_event,
  build_token_estimate_log_data as _build_token_estimate_log_data,
  build_turn_complete_event as _build_turn_complete_event,
  build_turn_complete_log_data as _build_turn_complete_log_data,
  run_detach_reason as _run_detach_reason,
  run_interrupted_reason as _run_interrupted_reason,
)
from .runner_state import (
  assistant_turn_message as _assistant_turn_message,
  background_tasks_completed_user_message as _background_tasks_completed_user_message,
  budget_cost_progress as _budget_cost_progress,
  budget_exceeded_state as _budget_exceeded_state,
  execute_tool_use_loop as _execute_tool_use_loop,
  model_visible_extra_blocks as _model_visible_extra_blocks,
  no_tool_use_turn_outcome as _no_tool_use_turn_outcome,
  normalized_run_config as _normalized_run_config,
  select_run_max_tokens as _select_run_max_tokens,
  session_drain_state as _session_drain_state,
  stream_turn_log_summary as _stream_turn_log_summary,
  sub_agent_batch_error as _sub_agent_batch_error,
  turn_reminder_state as _turn_reminder_state,
  usage_cache_status as _usage_cache_status,
  user_turn_message as _user_turn_message,
)
from .runner_usage import (
  empty_usage_totals as _empty_usage_totals,
  turn_usage_payload as _turn_usage_payload,
  usage_delta_state as _usage_delta_state,
)
from .task_registry import (
  COORDINATOR_DEFAULT_PREAMBLE,
  ParentMessage,
  TaskState,
  format_parent_messages_for_model,
)


log = logging.getLogger("agent_gateway.runner")


class RunnerRunLoopMixin:
  async def run(
    self,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[Union[str, List[Tuple[str, bool]]]] = None,
    model_override: Optional[str] = None,
    max_turns: Optional[int] = None,
    *,
    resume_initial_messages: List[Dict[str, Any]] | None = None,
  ) -> None:
    """Execute the full chat loop and stream events into `EventLog`."""
    if self._summary_emitted:
      raise RuntimeError("AgentRunner is single-use; construct a new runner for subsequent runs")
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    time_module = _runner_attr(self, "time", time)
    uuid_module = _runner_attr(self, "uuid", uuid)
    logger = _runner_attr(self, "log", log)
    queue_empty = getattr(asyncio_module, "QueueEmpty", asyncio.QueueEmpty)
    cancelled_error = getattr(asyncio_module, "CancelledError", asyncio.CancelledError)
    gather_fn = getattr(asyncio_module, "gather", asyncio.gather)
    context_warning_pct = _runner_attr(self, "CONTEXT_WARNING_PCT", CONTEXT_WARNING_PCT)
    coordinator_default_preamble = _runner_attr(
      self, "COORDINATOR_DEFAULT_PREAMBLE", COORDINATOR_DEFAULT_PREAMBLE
    )
    get_default_model_for_provider = _runner_attr(
      self, "_get_default_model_for_provider", _get_default_model_for_provider
    )
    prepend_system_prompt_preamble = _runner_attr(
      self, "_prepend_system_prompt_preamble", _prepend_system_prompt_preamble
    )
    normalized_run_config = _runner_attr(self, "_normalized_run_config", _normalized_run_config)
    select_run_max_tokens = _runner_attr(self, "_select_run_max_tokens", _select_run_max_tokens)
    effective_compaction_trigger_fn = _runner_attr(
      self, "_effective_compaction_trigger", _effective_compaction_trigger
    )
    model_context_window = _runner_attr(self, "_model_context_window", _model_context_window)
    system_prompt_estimate_text = _runner_attr(
      self, "_system_prompt_estimate_text", _system_prompt_estimate_text
    )
    token_estimate_snapshot = _runner_attr(self, "_token_estimate_snapshot", _token_estimate_snapshot)
    build_context_warning_log_data = _runner_attr(
      self, "_build_context_warning_log_data", _build_context_warning_log_data
    )
    build_token_estimate_log_data = _runner_attr(
      self, "_build_token_estimate_log_data", _build_token_estimate_log_data
    )
    empty_usage_totals = _runner_attr(self, "_empty_usage_totals", _empty_usage_totals)
    build_max_turns_reached_event = _runner_attr(
      self, "_build_max_turns_reached_event", _build_max_turns_reached_event
    )
    build_max_turns_text_event = _runner_attr(
      self, "_build_max_turns_text_event", _build_max_turns_text_event
    )
    turn_reminder_state = _runner_attr(self, "_turn_reminder_state", _turn_reminder_state)
    max_notifications_per_turn = _runner_attr(
      self, "_MAX_NOTIFICATIONS_PER_TURN", _MAX_NOTIFICATIONS_PER_TURN
    )
    user_turn_message = _runner_attr(self, "_user_turn_message", _user_turn_message)
    format_parent_messages = _runner_attr(
      self, "format_parent_messages_for_model", format_parent_messages_for_model
    )
    usage_delta_state = _runner_attr(self, "_usage_delta_state", _usage_delta_state)
    turn_usage_payload_fn = _runner_attr(self, "_turn_usage_payload", _turn_usage_payload)
    build_turn_complete_event = _runner_attr(
      self, "_build_turn_complete_event", _build_turn_complete_event
    )
    stream_turn_log_summary = _runner_attr(self, "_stream_turn_log_summary", _stream_turn_log_summary)
    build_turn_complete_log_data = _runner_attr(
      self, "_build_turn_complete_log_data", _build_turn_complete_log_data
    )
    budget_cost_progress = _runner_attr(self, "_budget_cost_progress", _budget_cost_progress)
    budget_exceeded_state = _runner_attr(self, "_budget_exceeded_state", _budget_exceeded_state)
    build_budget_exceeded_event = _runner_attr(
      self, "_build_budget_exceeded_event", _build_budget_exceeded_event
    )
    build_budget_exceeded_text_event = _runner_attr(
      self, "_build_budget_exceeded_text_event", _build_budget_exceeded_text_event
    )
    build_runtime_guard_event = _runner_attr(
      self, "_build_runtime_guard_event", _build_runtime_guard_event
    )
    assistant_turn_message = _runner_attr(self, "_assistant_turn_message", _assistant_turn_message)
    no_tool_use_turn_outcome = _runner_attr(
      self, "_no_tool_use_turn_outcome", _no_tool_use_turn_outcome
    )
    max_tokens_continuations_limit = _runner_attr(
      self, "_MAX_TOKENS_CONTINUATIONS", _MAX_TOKENS_CONTINUATIONS
    )
    max_tokens_nudge = _runner_attr(self, "_MAX_TOKENS_NUDGE", _MAX_TOKENS_NUDGE)
    execute_tool_use_loop = _runner_attr(self, "_execute_tool_use_loop", _execute_tool_use_loop)
    sub_agent_batch_error = _runner_attr(self, "_sub_agent_batch_error", _sub_agent_batch_error)
    model_visible_extra_blocks = _runner_attr(
      self, "_model_visible_extra_blocks", _model_visible_extra_blocks
    )
    background_tasks_completed_user_message = _runner_attr(
      self, "_background_tasks_completed_user_message", _background_tasks_completed_user_message
    )
    usage_cache_status = _runner_attr(self, "_usage_cache_status", _usage_cache_status)
    build_chat_done_log_data = _runner_attr(
      self, "_build_chat_done_log_data", _build_chat_done_log_data
    )
    build_stream_complete_event = _runner_attr(
      self, "_build_stream_complete_event", _build_stream_complete_event
    )
    task_state_cls = _runner_attr(self, "TaskState", TaskState)
    session_drain_state = _runner_attr(self, "_session_drain_state", _session_drain_state)
    run_interrupted_reason = _runner_attr(self, "_run_interrupted_reason", _run_interrupted_reason)
    run_detach_reason = _runner_attr(self, "_run_detach_reason", _run_detach_reason)
    was_cancelled = False
    run_error: BaseException | None = None
    clean_detach_reason = "completed"
    try:
      if self._agent_session_log is None:
        self._runner_id = None
        self._last_assistant_message_seq = None
        self._durable_attach_emitted = False
      if self._agent_session_log is not None:
        self._runner_id = f"runner_{uuid_module.uuid4().hex}"
        self._last_durable_seq = 0
        self._last_assistant_message_seq = None
        self._durable_attach_emitted = False
        if self._role == "writer":
          await self._acquire_writer_lease_and_recover()
        await self._emit_attach_event()
        if self._role == "writer":
          self._write_lease_metadata()
        await self._rebuild_task_registry_from_log()
        if resume_initial_messages is not None:
          messages = [dict(message) for message in resume_initial_messages]
        elif self._context_builder is not None:
          # Autonomous / server-authoritative path: context_builder is the source
          # of truth; the incoming `messages` carries only the new user turn.
          prior_messages = await self._context_builder.build()
          new_user_input = self._extract_last_user_message(messages)
          if new_user_input is not None:
            await self._append_user_message_event(new_user_input)
          messages = prior_messages + ([new_user_input] if new_user_input is not None else [])
        else:
          # Durable-log path without a replay policy: preserve caller-provided
          # messages without injecting prior durable history.
          new_user_input = self._extract_last_user_message(messages)
          if new_user_input is not None:
            await self._append_user_message_event(new_user_input)
          messages = [dict(message) for message in messages]

      if self._coordinator is not None and self._coordinator.enabled:
        preamble = self._coordinator.preamble or coordinator_default_preamble
        system_prompt = prepend_system_prompt_preamble(system_prompt, preamble)

      default_model = get_default_model_for_provider(getattr(self._provider, "name", None))
      config = normalized_run_config(
        self._auth_config,
        default_model=default_model,
        model_override=model_override,
      )

      if not self._provider.has_active_credential(config):
        await self._emit_stub_response(messages)
        return

      try:
        client = self._provider.create_client(config, timeout=self._client_timeout)
        self._set_client(client)
      except Exception:
        await self._emit_stub_response(messages)
        return

      try:
        model_info = self._provider.get_model_info(config["model"])
      except Exception as exc:
        await self._emit_error_event(str(exc))
        await self._close_client(client, timeout=5.0)
        return

      cached_tools = self._filter_excluded_tool_definitions(self._default_tool_definitions())

      max_tokens_selection = select_run_max_tokens(
        config,
        max_tokens_override=self._max_tokens_override,
        model_info=model_info,
      )
      max_tokens = max_tokens_selection.value
      if max_tokens_selection.clamped:
        logger.info(
          "[%s] max_tokens %d clamped to model max_output_tokens %d",
          self._sid,
          max_tokens_selection.requested,
          max_tokens_selection.model_max_output,
        )

      base_kwargs: Dict[str, Any] = {
        "tools": cached_tools,
      }
      logger.info("[%s] Effort requested | effort=%s", self._sid, config["effort"])

      effective_compaction_trigger = effective_compaction_trigger_fn(self._compaction_trigger, model_info)
      if effective_compaction_trigger is not None:
        logger.info("[%s] Compaction enabled | trigger=%d tokens", self._sid, effective_compaction_trigger)
      context_limit = model_context_window(model_info)

      logger.info("[%s] Chat start | model=%s max_tokens=%d messages=%d", self._sid, config["model"], max_tokens, len(messages))

      chat_t0 = time_module.time()
      system_text = system_prompt_estimate_text(system_prompt)
      initial_estimate = token_estimate_snapshot(
        system_text=system_text,
        messages=messages,
        tools=cached_tools,
      )
      system_chars = initial_estimate.system_chars
      if initial_estimate.est_total_tokens > context_limit * context_warning_pct / 100:
        logger.warning(
          "[%s] Context usage high | est=%d tokens (%.0f%% of %dk limit)",
          self._sid,
          initial_estimate.est_total_tokens,
          initial_estimate.est_total_tokens / context_limit * 100,
          context_limit // 1000,
          extra={
            "data": build_context_warning_log_data(
              session_id=self._sid,
              est_tokens=initial_estimate.est_total_tokens,
              context_limit=context_limit,
            )
          },
        )
      logger.info(
        "[%s] Pre-request estimate | system=%d msgs=%d tools=%d total=%d tokens (est)",
        self._sid,
        initial_estimate.est_system_tokens,
        initial_estimate.est_messages_tokens,
        initial_estimate.est_tools_tokens,
        initial_estimate.est_total_tokens,
        extra={
          "data": build_token_estimate_log_data(
            session_id=self._sid,
            est_system_tokens=initial_estimate.est_system_tokens,
            est_messages_tokens=initial_estimate.est_messages_tokens,
            est_tools_tokens=initial_estimate.est_tools_tokens,
            est_total_tokens=initial_estimate.est_total_tokens,
            message_count=initial_estimate.message_count,
            tool_count=initial_estimate.tool_count,
          )
        },
      )
      tools_chars = initial_estimate.tools_chars
      usage_totals = empty_usage_totals()
      self._last_reported_cost = 0.0
      self._ensure_sub_agent_semaphore()
      turn_count = 0
      tools_used: List[str] = []
      final_answer_guard_fired = False
      max_tokens_continuations = 0
      current_messages = list(messages)

      async def emit_budget_exceeded_stop(exceeded_state: Any) -> None:
        logger.warning(
          "[%s] Budget exceeded: $%.4f >= $%.4f%s — stopping",
          self._sid,
          exceeded_state.total_cost,
          exceeded_state.budget,
          exceeded_state.reason_suffix,
        )
        self._append(
          build_budget_exceeded_event(
            total_cost=exceeded_state.total_cost,
            budget=exceeded_state.budget,
            reason=exceeded_state.reason,
          )
        )
        self._append(
          build_budget_exceeded_text_event(
            total_cost=exceeded_state.total_cost,
            budget=exceeded_state.budget,
            reason_suffix=exceeded_state.reason_suffix,
          )
        )
        await self._emit_interrupted_event("budget_exceeded")

      while True:
        if self._operator_pause_requested():
          logger.info("[%s] Operator pause requested before next turn; stopping at safe boundary", self._sid)
          clean_detach_reason = "operator_pause"
          await self._emit_operator_pause_event("before_turn")
          break

        turn_count += 1
        if max_turns is not None and turn_count > max_turns:
          logger.warning("[%s] Max turns (%d) reached, stopping", self._sid, max_turns)
          self._append(build_max_turns_reached_event(turn_count=turn_count, max_turns=max_turns))
          await self._emit_interrupted_event("max_turns_reached")
          summary_text = None
          if self._on_max_turns is not None:
            try:
              summary_text = await self._on_max_turns(current_messages, turn_count)
            except Exception as exc:
              logger.warning("[%s] on_max_turns callback failed: %s", self._sid, exc)
          self._append(build_max_turns_text_event(summary_text))
          break
        turn_t0 = time_module.time()
        turn_t0_mono = time_module.monotonic()

        if turn_count > 1:
          current_tools = base_kwargs.get("tools") or []
          turn_estimate = token_estimate_snapshot(
            system_text=system_text,
            messages=current_messages,
            tools=current_tools,
          )
          if turn_estimate.est_total_tokens > context_limit * context_warning_pct / 100:
            logger.warning(
              "[%s] Context usage high | est=%d tokens (%.0f%% of %dk limit)",
              self._sid,
              turn_estimate.est_total_tokens,
              turn_estimate.est_total_tokens / context_limit * 100,
              context_limit // 1000,
              extra={
                "data": build_context_warning_log_data(
                  session_id=self._sid,
                  turn=turn_count,
                  est_tokens=turn_estimate.est_total_tokens,
                  context_limit=context_limit,
                )
              },
            )
          logger.info(
            "[%s] Turn %d pre-request | est=%d tokens",
            self._sid,
            turn_count,
            turn_estimate.est_total_tokens,
            extra={
              "data": build_token_estimate_log_data(
                session_id=self._sid,
                turn=turn_count,
                est_system_tokens=turn_estimate.est_system_tokens,
                est_messages_tokens=turn_estimate.est_messages_tokens,
                est_tools_tokens=turn_estimate.est_tools_tokens,
                est_total_tokens=turn_estimate.est_total_tokens,
                message_count=turn_estimate.message_count,
                tool_count=turn_estimate.tool_count,
              )
            },
          )

        bg_reminder = self._background_task_reminder_text()
        notif_reminder = self._build_notification_reminder()
        turn_reminder = turn_reminder_state(
          bg_reminder,
          notif_reminder,
          pending_notification_count=self._notification_queue.pending_count,
          max_notifications_per_turn=max_notifications_per_turn,
        )
        peeked_notification_count = turn_reminder.peeked_notification_count
        if turn_reminder.text:
          turn_system_prompt = self._inject_system_prompt_reminder(system_prompt, turn_reminder.text)
        else:
          turn_system_prompt = system_prompt
        if self._message_inbox is not None:
          parent_messages: list[ParentMessage] = []
          while not self._message_inbox.empty():
            try:
              parent_messages.append(self._message_inbox.get_nowait())
            except queue_empty:
              break
          if parent_messages:
            current_messages.append(user_turn_message(format_parent_messages(parent_messages)))
        turn_usage_before = dict(usage_totals)
        turn_result = await self._stream_turn(
          client=client,
          config=config,
          model_info=model_info,
          system_prompt=turn_system_prompt,
          current_messages=current_messages,
          base_kwargs=base_kwargs,
          max_tokens=max_tokens,
          turn_count=turn_count,
          turn_t0=turn_t0,
          turn_t0_mono=turn_t0_mono,
          system_chars=system_chars,
          tools_chars=tools_chars,
          usage_totals=usage_totals,
        )
        if turn_result is None:
          return
        client, turn = turn_result
        turn_usage_state = usage_delta_state(turn_usage_before, usage_totals)
        turn_usage = turn_usage_state.usage
        turn_usage_payload = turn_usage_payload_fn(turn_usage)
        if turn_usage_state.has_tokens:
          turn_cost = self._estimate_usage_cost(config["model"], turn_usage)
          turn_usage_payload = turn_usage_payload_fn(turn_usage, estimated_cost=turn_cost.total)
          await self._call_on_usage(self._build_usage_event(model=config["model"], usage_totals=turn_usage))
        assistant_message_persisted = False

        async def persist_assistant_message_once() -> None:
          nonlocal assistant_message_persisted
          if assistant_message_persisted:
            return
          await self._append_assistant_message_event(
            content_blocks=turn.content_blocks,
            stop_reason=turn.stop_reason,
            model=config["model"],
            usage=turn_usage,
          )
          assistant_message_persisted = True

        self._append(build_turn_complete_event(turn=turn_count, usage=turn_usage_payload))

        turn_log_summary = stream_turn_log_summary(
          turn,
          turn_started_at=turn_t0,
          now=time_module.time(),
        )

        logger.info(
          "[%s] Turn %d complete | %.1fs | TTFT=%.2fs | text=%d chars | tools=%s | stop=%s | response=%s",
          self._sid,
          turn_count,
          turn_log_summary.elapsed_s,
          turn_log_summary.ttft_s if turn_log_summary.ttft_s is not None else -1,
          turn_log_summary.text_chars,
          turn_log_summary.tool_names or "none",
          turn.stop_reason,
          turn_log_summary.text_preview or "(none)",
          extra={
            "data": build_turn_complete_log_data(
              session_id=self._sid,
              turn=turn_count,
              elapsed_s=turn_log_summary.elapsed_s,
              ttft_s=turn_log_summary.ttft_s,
              text_chars=turn_log_summary.text_chars,
              tools=turn_log_summary.tool_names,
              stop_reason=turn.stop_reason,
            )
          },
        )

        if self._cost_accumulator is not None:
          running_cost = self._estimate_usage_cost(config["model"], usage_totals)
          budget_progress = budget_cost_progress(
            self._cost_accumulator,
            running_total=running_cost.total,
            last_reported_cost=self._last_reported_cost,
          )
          self._last_reported_cost = budget_progress.last_reported_cost
          exceeded_state = budget_progress.exceeded_state
        else:
          exceeded_state = None

        if turn.tool_uses and self._operator_pause_requested():
          await persist_assistant_message_once()
          logger.info("[%s] Operator pause requested after turn; stopping before tool dispatch", self._sid)
          clean_detach_reason = "operator_pause"
          await self._emit_operator_pause_event("after_turn_before_tools")
          break

        if not turn.tool_uses:
          final_answer_guard_allowed = turn.stop_reason not in {"max_tokens", "pause_turn", "compaction"}
          if (
            final_answer_guard_allowed
            and not final_answer_guard_fired
            and self._final_answer_guard is not None
          ):
            guard_message = self._final_answer_guard(
              current_messages,
              turn.full_text,
              list(tools_used),
              list(base_kwargs.get("tools") or []),
              turn_count,
            )
            if guard_message:
              final_answer_guard_fired = True
              logger.info("[%s] Final-answer guard injected follow-up before completing turn", self._sid)
              runtime_guard_event = build_runtime_guard_event(guard="final_answer", message=guard_message)
              self._append(runtime_guard_event)
              durable_runtime_guard_event = dict(runtime_guard_event)
              durable_runtime_guard_event.update(
                {
                  "draft_content_blocks": list(turn.content_blocks),
                  "draft_stop_reason": turn.stop_reason,
                  "draft_model": config["model"],
                  "draft_provider": self._provider.name,
                  "draft_usage": dict(turn_usage),
                }
              )
              await self._append_durable_event(durable_runtime_guard_event)
              current_messages.append(
                assistant_turn_message(
                  turn.content_blocks,
                  provider=self._provider.name,
                  model=config["model"],
                  stop_reason=turn.stop_reason,
                )
              )
              current_messages.append(user_turn_message(guard_message))
              if exceeded_state is not None:
                await emit_budget_exceeded_stop(exceeded_state)
                break
              continue
          await persist_assistant_message_once()
          if exceeded_state is not None:
            await emit_budget_exceeded_stop(exceeded_state)
            break
          if self._operator_pause_requested():
            logger.info("[%s] Operator pause requested after turn; stopping before tool dispatch", self._sid)
            clean_detach_reason = "operator_pause"
            await self._emit_operator_pause_event("after_turn_before_tools")
            break
          no_tool_outcome = no_tool_use_turn_outcome(
            content_blocks=turn.content_blocks,
            provider=self._provider.name,
            model=config["model"],
            stop_reason=turn.stop_reason,
            pending_notification_count=self._notification_queue.pending_count,
            max_tokens_continuations=max_tokens_continuations,
            max_tokens_max_attempts=max_tokens_continuations_limit,
            max_tokens_nudge=max_tokens_nudge,
          )
          max_tokens_continuations = no_tool_outcome.max_tokens_continuations
          if turn.stop_reason == "pause_turn":
            logger.info("[%s] Pause turn — continuing", self._sid)
          elif turn.stop_reason == "compaction":
            logger.info("[%s] Compaction pause — continuing", self._sid)
          elif no_tool_outcome.reason == "max_tokens_continue":
            logger.warning(
              "[%s] Turn %d hit max_tokens with no usable tool call; continuing with truncation nudge (%d/%d)",
              self._sid,
              turn_count,
              max_tokens_continuations,
              max_tokens_continuations_limit,
            )
          elif no_tool_outcome.reason == "max_tokens_terminal":
            logger.error(
              "[%s] Turn %d hit max_tokens with no usable tool call after %d continuation attempts; ending run",
              self._sid,
              turn_count,
              max_tokens_continuations_limit,
            )
          if no_tool_outcome.runtime_guard is not None:
            guard, message = no_tool_outcome.runtime_guard
            self._append(build_runtime_guard_event(guard=guard, message=message))
          current_messages.extend(no_tool_outcome.messages)
          if no_tool_outcome.action == "continue":
            continue
          break

        await persist_assistant_message_once()
        current_messages.append(
          assistant_turn_message(
            turn.content_blocks,
            provider=self._provider.name,
            model=config["model"],
            stop_reason=turn.stop_reason,
          )
        )

        tool_loop = await execute_tool_use_loop(
          turn.tool_uses,
          base_kwargs=base_kwargs,
          sub_agent_semaphore=self._sub_agent_semaphore,
          effective_excluded_tools=self._effective_excluded_tools,
          execute_single_tool=self._execute_single_tool,
          gather_fn=gather_fn,
          batch_error=sub_agent_batch_error,
          make_error_result=self._make_error_result,
          model_visible_extra_blocks=model_visible_extra_blocks,
          on_exception=lambda exc: logger.warning("[%s] run_agent gather exception: %s", self._sid, exc),
        )
        tools_used.extend(tool_loop.tools_used)
        current_messages.append(user_turn_message(tool_loop.tool_results_content))
        if peeked_notification_count > 0:
          self._consume_notifications(max_count=peeked_notification_count)

        if self._cost_accumulator is not None:
          exceeded_state = budget_exceeded_state(self._cost_accumulator) or exceeded_state
        if exceeded_state is not None:
          await emit_budget_exceeded_stop(exceeded_state)
          break

        stop_after_tool_results_reason = getattr(self, "_stop_after_tool_results_reason", None)
        if stop_after_tool_results_reason:
          stop_after_tool_results_tool_name = getattr(self, "_stop_after_tool_results_tool_name", None)
          logger.info(
            "[%s] Stop-after-tool-results requested after %s (%s); ending run without follow-up model turn",
            self._sid,
            stop_after_tool_results_tool_name or "tool result",
            stop_after_tool_results_reason,
          )
          break

        if turn.stop_reason == "end_turn":
          if self._notification_queue.pending_count > 0:
            current_messages.append(
              background_tasks_completed_user_message()
            )
            continue
          break

      total_elapsed = time_module.time() - chat_t0
      cache_status = usage_cache_status(usage_totals)

      cost = self._estimate_usage_cost(config["model"], usage_totals)

      logger.info(
        "[%s] Chat done | %.1fs total | %d turns | tools=%s | tokens in=%d out=%d | cache=%s | cost=$%.4f",
        self._sid,
        total_elapsed,
        turn_count,
        tools_used or "none",
        usage_totals["input_tokens"],
        usage_totals["output_tokens"],
        cache_status,
        cost.total,
        extra={
          "data": build_chat_done_log_data(
            session_id=self._sid,
            elapsed_s=total_elapsed,
            turns=turn_count,
            tools=tools_used,
            usage_totals=usage_totals,
            cost=cost.total,
          )
        },
      )

      terminal_event = build_stream_complete_event(
        usage_totals=usage_totals,
        estimated_cost=cost.total,
      )
      await self._call_on_before_stream_complete(terminal_event)

      self._append(terminal_event)

      try:
        await self._close_client(client, timeout=5.0)
      except Exception as exc:
        logger.warning(
          "[%s] client close after completed stream failed (non-fatal): %s",
          self._sid,
          exc,
        )
        await self._emit_run_error_event(exc, phase="client_close_after_stream_complete")
    except cancelled_error as exc:
      was_cancelled = True
      run_error = exc
    except BaseException as exc:
      run_error = exc
    finally:
      finalizer_error: BaseException | None = None
      shutdown_failed = False
      try:
        from .skill_context import clear_current_skill

        clear_current_skill()
      except Exception:
        pass
      self._active_skill_deny.clear()
      self._active_skill_report_doors.clear()
      try:
        await self._shutdown_background_tasks(was_cancelled)
      except BaseException as exc:
        finalizer_error = exc
        shutdown_failed = True
      finally:
        running_entries = self._task_registry.list_tasks(state=task_state_cls.RUNNING)
        drain_state = session_drain_state(
          running_entries,
          shutdown_failed=shutdown_failed,
        )
        if self._parent_aggregator is None:
          try:
            await self._aggregator.close()
            summary = await self._aggregator.snapshot(
              ended_at=time_module.time(),
              drain_complete=drain_state.drain_complete,
              in_flight_task_count=drain_state.in_flight_task_count,
              context_surfaces=self._context_surface_records(),
            )
            self._summary_emitted = True
            await self._call_on_session_summary(summary)
          except Exception as exc:
            logger.error("[%s] session summary emission failed (non-fatal): %s", self._sid, exc)
        else:
          self._summary_emitted = True
      try:
        if self._agent_session_log is not None and self._durable_attach_emitted:
          if run_error is not None:
            await self._emit_run_error_event(run_error)
            shutdown_reason, shutdown_extra_fields = self._shutdown_interrupted_reason()
            reason, extra_fields = run_interrupted_reason(
              run_error=run_error,
              role=self._role,
              shutdown_reason=shutdown_reason,
              shutdown_extra_fields=shutdown_extra_fields,
            )
            await self._emit_interrupted_event(reason, extra_fields=extra_fields or None)
          await self._emit_detach_event(
            run_detach_reason(
              clean_detach_reason=clean_detach_reason,
              run_error=run_error,
            )
          )
      except BaseException as exc:
        if finalizer_error is None:
          finalizer_error = exc
      finally:
        try:
          self._release_write_lease()
        finally:
          await self.force_close()
      if run_error is not None:
        if finalizer_error is not None:
          raise finalizer_error from run_error
        raise run_error
      if finalizer_error is not None:
        raise finalizer_error
