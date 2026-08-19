from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from agent_workflow_contracts import AgentCompletionEnvelope

from .context_capture import build_context_manifest_event, canonical_manifest_digest
from .fork_request_handoff import (
  build_mid_turn_handoff,
  build_post_turn_handoff,
  fork_credential_identity_available,
)
from .learning_fork_trigger import (
  claim_learning_receipts,
  settle_learning_receipts,
  submit_learning_fork_after_turn,
  successful_memory_write_from_events,
)
from .runner_fork_agents import (
  FORK_SUFFIX_WRAP_UP_REMINDER,
  fork_suffix_messages,
)
from .runner_limits import (
  CONTEXT_PRESSURE_REMINDER_PCT,
  CONTEXT_PRESSURE_REMINDER_STEP_PCT,
  CONTEXT_WARNING_PCT,
  conservative_request_input_token_bound_for_request as _conservative_request_input_token_bound_for_request,
  context_pressure_reminder_decision as _context_pressure_reminder_decision,
  effective_compaction_trigger as _effective_compaction_trigger,
  model_context_window as _model_context_window,
  system_prompt_estimate_text as _system_prompt_estimate_text,
  token_estimate_snapshot as _token_estimate_snapshot,
)
from .server_compaction import (
  CONTINUATION_NUDGE,
  MIN_COMPACT_GAIN_TOKENS,
  maybe_compact_current_messages,
  portable_compaction_enabled,
)
from .runner_prompt_rules import (
  last_user_message,
  prepend_system_prompt_preamble as _prepend_system_prompt_preamble,
)
from .runner_background_tasks import dispatch_objective_echo
from .runner_cleanup import attach_cleanup_failure
from .runner_run_loop_defaults import (
  MAX_NOTIFICATIONS_PER_TURN as _MAX_NOTIFICATIONS_PER_TURN,
  MAX_TOKENS_CONTINUATIONS as _MAX_TOKENS_CONTINUATIONS,
  MAX_TOKENS_NUDGE as _MAX_TOKENS_NUDGE,
)
from .runner_session_lifecycle import _runner_attr
from .secret_boundary import sanitize_tool_event
from .skill_completion_wal import TopLevelSkillCompletionEffectPlan
from .runner_session_events import (
  build_budget_exceeded_event as _build_budget_exceeded_event,
  build_budget_exceeded_text_event as _build_budget_exceeded_text_event,
  build_chat_done_log_data as _build_chat_done_log_data,
  build_context_pressure_reminder as _build_context_pressure_reminder,
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
  ProviderRequestBudgetError,
  admit_provider_request_budget as _admit_provider_request_budget,
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
  StreamTurnFailure,
)
from .runner_usage import (
  empty_usage_totals as _empty_usage_totals,
  turn_usage_payload as _turn_usage_payload,
  usage_delta_state as _usage_delta_state,
  usage_snapshot,
)
from .task_registry import (
  COORDINATOR_DEFAULT_PREAMBLE,
  ParentMessage,
  TaskState,
  format_parent_messages_for_model,
)
from .workflow_evidence_provenance import (
  guard_visible_tools_used as _guard_visible_tools_used,
)


log = logging.getLogger("agent_gateway.runner")


def _tool_result_ids(messages: list[dict[str, Any]]) -> set[str]:
  return {
    str(block["tool_use_id"])
    for message in messages
    for block in (
      message.get("content")
      if isinstance(message.get("content"), list)
      else []
    )
    if (
      isinstance(block, dict)
      and block.get("type") == "tool_result"
      and isinstance(block.get("tool_use_id"), str)
      and block["tool_use_id"]
    )
  }


def _agent_completion_contract_error(
  *,
  tool_uses: list[tuple[str, str, dict[str, Any]]],
  tool_result_blocks: list[dict[str, Any]],
) -> str | None:
  """Validate newly normalized foreground run-agent deliveries.

  Existing non-envelope results are left to the executor cutover.  Once a
  result identifies itself as an AgentCompletionEnvelope, however, it must
  remain fully parseable after model-context materialization.  This detects a
  generic tool-result compactor turning that envelope into a clipped prefix.
  """

  foreground_ids = {
    tool_use_id
    for tool_use_id, tool_name, tool_input in tool_uses
    if (
      tool_name == "run_agent"
      and isinstance(tool_input, dict)
      and tool_input.get("background", True) is False
    )
  }
  if not foreground_ids:
    return None
  for block in tool_result_blocks:
    if (
      not isinstance(block, dict)
      or block.get("tool_use_id") not in foreground_ids
      or block.get("is_error") is True
    ):
      continue
    content = block.get("content")
    if not isinstance(content, str):
      continue
    identifies_completion = (
      "agent-completion:" in content
      or '"task_result_ref"' in content
      or '"parent_materialization"' in content
    )
    if not identifies_completion:
      continue
    try:
      payload = json.loads(content)
      AgentCompletionEnvelope.model_validate(payload)
    except Exception as exc:
      return (
        "agent_completion_contract_invalid: normalized run_agent result "
        "was not a complete AgentCompletionEnvelope after context "
        f"materialization ({type(exc).__name__})."
      )
  return None


def _acknowledge_consumed_background_results(
  runner: Any,
  *,
  consumed_tool_result_ids: set[str],
) -> None:
  pending = getattr(
    runner,
    "_pending_background_result_acks",
    None,
  )
  if not isinstance(pending, dict):
    return
  for tool_use_id in consumed_tool_result_ids:
    acknowledgement = pending.pop(tool_use_id, None)
    if (
      not isinstance(acknowledgement, tuple)
      or len(acknowledgement) != 2
    ):
      continue
    task_id, notification_generation = acknowledgement
    if (
      isinstance(task_id, str)
      and isinstance(notification_generation, int)
      and not isinstance(notification_generation, bool)
    ):
      runner._task_registry.mark_notification_payload_retrieved(
        task_id,
        notification_generation=notification_generation,
      )
      acknowledgement_key = (task_id, notification_generation)
      for pending_tool_use_id, pending_acknowledgement in list(
        pending.items()
      ):
        if pending_acknowledgement == acknowledgement_key:
          pending.pop(pending_tool_use_id, None)


def _discard_unavailable_background_result_acks(
  runner: Any,
  *,
  available_tool_result_ids: set[str],
) -> None:
  pending = getattr(
    runner,
    "_pending_background_result_acks",
    None,
  )
  recovery_pairs = getattr(
    runner,
    "_background_ack_recovery_pairs",
    None,
  )
  if not isinstance(pending, dict) or not isinstance(
    recovery_pairs,
    set,
  ):
    return
  for tool_use_id, acknowledgement in list(pending.items()):
    if tool_use_id in available_tool_result_ids:
      continue
    pending.pop(tool_use_id, None)
    if (
      isinstance(acknowledgement, tuple)
      and len(acknowledgement) == 2
      and isinstance(acknowledgement[0], str)
      and isinstance(acknowledgement[1], int)
      and not isinstance(acknowledgement[1], bool)
    ):
      recovery_pairs.add(acknowledgement)


def _unread_settled_handle_entries(runner: Any) -> list[Any]:
  """Settled tasks whose delivered result handle was never read (CUR-E2E-08).

  These are NOT delivery obligations and NOT success blockers: the
  notification was rendered and acked, the settlement is real, and finish
  stays allowed after one ignored reminder. The predicate is deliberately
  narrow: a terminal entry, an envelope whose materialization is a bare
  ``result_handle`` (an authored summary already delivers usable inline
  text), a notification the model has already consumed (``delivered`` —
  queued/omitted states are owned by the existing delivery machinery), an
  unexercised read grant, and settlement this writer owns.
  """
  entries: list[Any] = []
  for entry in runner._task_registry.list_tasks():
    if not getattr(entry, "completed", False):
      continue
    if getattr(entry, "result_content_read", False):
      continue
    if getattr(entry, "notification_delivery_state", None) != "delivered":
      continue
    envelope = getattr(entry, "completion_envelope", None)
    materialization = getattr(envelope, "parent_materialization", None)
    if getattr(materialization, "kind", None) != "result_handle":
      continue
    settlement_owned = (
      not bool(getattr(entry, "reconstructed_from_log", False))
      or getattr(runner, "_role", None) == "writer"
    )
    if not settlement_owned:
      continue
    entries.append(entry)
  return entries


#: Tasks named per reminder turn. The arms slice to this bound BEFORE adding
#: to the reminded set, so an over-cap fan-out gets its remaining tasks named
#: at the next stop boundary instead of being silently marked reminded.
_UNREAD_HANDLE_NUDGE_MAX_TASKS = 10


def _unread_result_handle_nudge(entries: list[Any]) -> str:
  lines: list[str] = []
  for entry in entries[:_UNREAD_HANDLE_NUDGE_MAX_TASKS]:
    materialization = getattr(
      getattr(entry, "completion_envelope", None),
      "parent_materialization",
      None,
    )
    content_id = getattr(
      getattr(materialization, "source", None), "content_id", ""
    )
    grant_id = getattr(
      getattr(materialization, "read_grant", None), "grant_id", ""
    )
    echo = dispatch_objective_echo(entry) or "objective unavailable"
    lines.append(
      f"- task {getattr(entry, 'task_id', '')} ({echo}): "
      f"get_agent_result_content(content_id=\"{content_id}\", "
      f"read_grant_id=\"{grant_id}\")"
    )
  listing = "\n".join(lines)
  return (
    "[System: The following background task result(s) were delivered as "
    "content handles and have never been read:\n"
    f"{listing}\n"
    "Each of these tasks is SETTLED — completed, not running, not "
    "outstanding. Before finishing, either read each result now and "
    "integrate it, or state explicitly in your answer that it was not "
    "integrated and why. Do not describe any of these tasks as still "
    "running or outstanding.]"
  )


def _pending_omitted_background_task_ids(runner: Any) -> list[str]:
  return [
    entry.task_id
    for entry in runner._task_registry.list_tasks()
    if entry.notification_delivery_state in {
      "payload_omitted",
      "queue_omitted",
    }
  ]


def _async_task_fingerprint(task: Any) -> tuple[Any, ...]:
  if task is None:
    return ("none",)
  done = getattr(task, "done", None)
  if not callable(done):
    return ("malformed", id(task), type(task).__name__)
  try:
    is_done = bool(done())
  except Exception as exc:
    return (
      "inspection_error",
      id(task),
      type(task).__name__,
      type(exc).__name__,
    )
  return ("task", id(task), is_done)


def _task_collection_snapshot(
  values: Any,
) -> tuple[bool, tuple[tuple[Any, ...], ...]]:
  try:
    tasks = tuple(values or ())
  except Exception:
    return True, (("malformed_collection", id(values)),)
  fingerprints = tuple(
    sorted(
      (_async_task_fingerprint(task) for task in tasks),
      key=repr,
    )
  )
  unfinished = any(
    fingerprint[0] != "none"
    and (
      fingerprint[0] != "task"
      or fingerprint[2] is not True
    )
    for fingerprint in fingerprints
  )
  return unfinished, fingerprints


def _background_success_snapshot(
  runner: Any,
) -> tuple[tuple[str, ...], tuple[Any, ...]]:
  """Return blockers and an immutable terminal-settlement fingerprint."""

  blockers: list[str] = []
  registry = runner._task_registry
  entries = registry.list_tasks()
  if registry.admission_count > 0:
    blockers.append("pending_or_running_tasks")

  initialization_unfinished, initialization_fingerprint = (
    _task_collection_snapshot(
      getattr(runner, "_pending_background_initializations", ())
    )
  )
  if initialization_unfinished:
    blockers.append("background_initialization_or_reconciliation")
  completion_unfinished, completion_fingerprint = (
    _task_collection_snapshot(
      getattr(runner, "_pending_background_completion_appends", ())
    )
  )
  if completion_unfinished:
    blockers.append("background_completion_persistence")

  entry_fingerprints: list[tuple[Any, ...]] = []
  for entry in entries:
    initialization_task = _async_task_fingerprint(
      getattr(entry, "initialization_task", None)
    )
    asyncio_task = _async_task_fingerprint(
      getattr(entry, "asyncio_task", None)
    )
    required_skill_result_task = _async_task_fingerprint(
      getattr(entry, "required_skill_result_task", None)
    )
    metadata = getattr(entry, "metadata", {})
    required_skill_lifecycle = (
      metadata.get("required_skill_lifecycle")
      if isinstance(metadata, dict)
      else None
    )
    requires_skill_result = (
      isinstance(metadata, dict)
      and "required_skill_lifecycle" in metadata
    )
    required_skill_result_settled = bool(
      getattr(entry, "required_skill_result_settled", False)
    )
    state = getattr(entry, "state", None)
    state_value = getattr(state, "value", str(state))
    pending_final_state = getattr(entry, "pending_final_state", None)
    pending_final_state_value = getattr(
      pending_final_state,
      "value",
      str(pending_final_state),
    )
    completion_persistence_state = str(
      getattr(entry, "completion_persistence_state", "")
    )
    terminal_state_values = {
      TaskState.COMPLETED.value,
      TaskState.FAILED.value,
      TaskState.KILLED.value,
    }
    terminal_committed = (
      completion_persistence_state == "committed"
      and (
        state_value in terminal_state_values
        or pending_final_state_value in terminal_state_values
      )
    )
    settlement_owned = (
      not bool(getattr(entry, "reconstructed_from_log", False))
      or getattr(runner, "_role", None) == "writer"
    )
    entry_fingerprints.append((
      str(getattr(entry, "task_id", "")),
      str(state_value),
      int(getattr(entry, "notification_generation", 0)),
      str(getattr(entry, "notification_delivery_state", "")),
      str(getattr(entry, "registration_persistence_state", "")),
      completion_persistence_state,
      bool(getattr(entry, "completion_finalizer_detached", False)),
      pending_final_state_value,
      bool(getattr(entry, "start_hook_invoked", False)),
      bool(getattr(entry, "completion_callback_invoked", False)),
      requires_skill_result,
      repr(required_skill_lifecycle),
      required_skill_result_settled,
      terminal_committed,
      settlement_owned,
      required_skill_result_task,
      initialization_task,
      asyncio_task,
    ))
    if (
      requires_skill_result
      and terminal_committed
      and settlement_owned
      and not required_skill_result_settled
    ):
      blockers.append("background_required_skill_result_settlement")
    if (
      bool(getattr(entry, "completion_finalizer_detached", False))
      or completion_persistence_state in {"in_flight", "uncertain"}
      or initialization_task[0] not in {"none", "task"}
      or (
        initialization_task[0] == "task"
        and initialization_task[2] is not True
      )
      or asyncio_task[0] not in {"none", "task"}
      or (
        asyncio_task[0] == "task"
        and asyncio_task[2] is not True
      )
    ):
      blockers.append("background_finalization")

  pending_delivery_states = {
    "queued",
    "payload_omitted",
    "queue_omitted",
  }
  if any(
    getattr(entry, "notification_delivery_state", None)
    in pending_delivery_states
    for entry in entries
  ):
    blockers.append("background_result_delivery")

  queue = runner._notification_queue
  try:
    queued_notifications = tuple(queue.peek())
    queue_pending_count = int(queue.pending_count)
  except Exception as exc:
    queued_notifications = ()
    queue_pending_count = -1
    blockers.append("notification_queue_uninspectable")
    queue_fingerprint: tuple[Any, ...] = (
      "inspection_error",
      type(exc).__name__,
    )
  else:
    queue_fingerprint = tuple(
      (
        getattr(notification, "_queue_token", None),
        str(getattr(notification, "task_id", "")),
        getattr(notification, "notification_generation", None),
        str(getattr(notification, "event", "")),
        getattr(notification, "omission_reason_override", None),
      )
      for notification in queued_notifications
    )
    if queue_pending_count > 0:
      blockers.append("background_result_delivery")

  pending_acknowledgements = getattr(
    runner,
    "_pending_background_result_acks",
    None,
  )
  if not isinstance(pending_acknowledgements, dict):
    blockers.append("background_result_acknowledgement_malformed")
    acknowledgement_fingerprint: tuple[Any, ...] = (
      "malformed",
      type(pending_acknowledgements).__name__,
      id(pending_acknowledgements),
    )
  else:
    if pending_acknowledgements:
      blockers.append("background_result_acknowledgement")
    acknowledgement_fingerprint = tuple(
      sorted(
        (
          str(tool_use_id),
          (
            "valid",
            acknowledgement[0],
            acknowledgement[1],
          )
          if (
            isinstance(acknowledgement, tuple)
            and len(acknowledgement) == 2
            and isinstance(acknowledgement[0], str)
            and isinstance(acknowledgement[1], int)
            and not isinstance(acknowledgement[1], bool)
          )
          else (
            "malformed",
            type(acknowledgement).__name__,
            id(acknowledgement),
          ),
        )
        for tool_use_id, acknowledgement
        in pending_acknowledgements.items()
      )
    )

  try:
    delivery_obligations = runner._background_delivery_grace_obligations()
  except Exception as exc:
    blockers.append("background_delivery_obligations_uninspectable")
    obligation_fingerprint: tuple[Any, ...] = (
      "inspection_error",
      type(exc).__name__,
    )
  else:
    obligation_fingerprint = tuple(
      sorted(
        (
          str(task_id),
          int(generation),
          str(phase),
          int(credits),
        )
        for (task_id, generation, phase), credits
        in delivery_obligations.items()
      )
    )
    if delivery_obligations:
      blockers.append("background_result_delivery")

  fingerprint = (
    int(getattr(registry, "_seq", 0)),
    tuple(sorted(entry_fingerprints)),
    initialization_fingerprint,
    completion_fingerprint,
    int(getattr(queue, "_next_queue_token", 0)),
    queue_pending_count,
    queue_fingerprint,
    acknowledgement_fingerprint,
    obligation_fingerprint,
  )
  return tuple(dict.fromkeys(blockers)), fingerprint


def _omitted_background_result_nudge(task_ids: list[str]) -> str:
  shown = task_ids[:20]
  remaining = len(task_ids) - len(shown)
  suffix = (
    f" and {remaining} more listed by task_id='*'"
    if remaining > 0
    else ""
  )
  return (
    "[System: Background result payload delivery is still required for exact "
    f"task IDs {', '.join(shown)}{suffix}. Retrieve each exact task ID now, "
    "following opaque cursors through the final page. Do not finish while "
    "these explicitly omitted results remain retained.]"
  )


def _merge_usage_totals(usage_totals: Dict[str, Any], usage: dict[str, Any] | None) -> None:
  if not usage:
    return
  for key in (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens_observed",
    "provider_units",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
  ):
    usage_totals[key] = int(usage_totals.get(key, 0) or 0) + int(usage.get(key, 0) or 0)
  accumulated_units = usage_totals.setdefault("provider_unit_deltas", {})
  if not isinstance(accumulated_units, dict):
    accumulated_units = {}
    usage_totals["provider_unit_deltas"] = accumulated_units
  for operation, count in dict(usage.get("provider_unit_deltas") or {}).items():
    accumulated_units[str(operation)] = int(accumulated_units.get(str(operation), 0) or 0) + int(count or 0)
  for key in (
    "capability_bind",
    "provider_reported_model",
  ):
    value = usage.get(key)
    if value is not None:
      usage_totals[key] = value


def _is_anchor_only_compaction_message(message: dict[str, Any] | None) -> bool:
  if not isinstance(message, dict) or message.get("role") != "assistant":
    return False
  if str(message.get("stop_reason") or "") != "compaction":
    return False
  content = message.get("content")
  return (
    isinstance(content, list)
    and len(content) == 1
    and isinstance(content[0], dict)
    and content[0].get("type") == "compaction"
  )


class RunnerRunLoopMixin:
  async def run(
    self,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[Union[str, List[Tuple[str, bool]]]] = None,
    max_turns: Optional[int] = None,
    *,
    resume_initial_messages: List[Dict[str, Any]] | None = None,
  ) -> None:
    """Execute the full chat loop and stream events into `EventLog`."""
    if self._summary_emitted:
      raise RuntimeError("AgentRunner is single-use; construct a new runner for subsequent runs")
    self._background_delivery_grace_active = False
    self._background_ack_recovery_pairs: set[
      tuple[str, int]
    ] = set()
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
    build_context_pressure_reminder = _runner_attr(
      self,
      "_build_context_pressure_reminder",
      _build_context_pressure_reminder,
    )
    context_pressure_reminder_decision = _runner_attr(
      self,
      "_context_pressure_reminder_decision",
      _context_pressure_reminder_decision,
    )
    conservative_request_input_token_bound_for_request = _runner_attr(
      self,
      "_conservative_request_input_token_bound_for_request",
      _conservative_request_input_token_bound_for_request,
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
    admit_provider_request_budget = _runner_attr(
      self,
      "_admit_provider_request_budget",
      _admit_provider_request_budget,
    )
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
    learning_receipt_delivery = claim_learning_receipts(self)
    learning_tool_calling_iters = 0
    learning_real_final_response = False
    terminal_success_reason: str | None = None
    selected_content_committed = not bool(
      getattr(self, "_selected_content_bindings", ())
    )
    top_level_user_input = self._preflight_top_level_skill_run(
      messages,
      resume_initial_messages,
    )
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
        if self._role == "sub_agent":
          await self._materialize_parent_message_consumption_audits()
        await self._rebuild_task_registry_from_log()
        if resume_initial_messages is not None:
          messages = [dict(message) for message in resume_initial_messages]
        elif self._context_builder is not None:
          # Autonomous / server-authoritative path: context_builder is the source
          # of truth; the incoming `messages` carries only the new user turn.
          prior_messages = await self._context_builder.build()
          new_user_input = (
            top_level_user_input
            if top_level_user_input is not None
            else last_user_message(messages)
          )
          if new_user_input is not None:
            user_entry = await self._append_user_message_event(
              new_user_input
            )
            selected_content_committed = user_entry is not None
            if (
              top_level_user_input is not None
              and user_entry is None
            ):
              raise RuntimeError(
                "Top-level skill user_message was not durably "
                "persisted"
              )
          messages = prior_messages + ([new_user_input] if new_user_input is not None else [])
        else:
          # Durable-log path without a replay policy: preserve caller-provided
          # messages without injecting prior durable history.
          new_user_input = (
            top_level_user_input
            if top_level_user_input is not None
            else last_user_message(messages)
          )
          if new_user_input is not None:
            user_entry = await self._append_user_message_event(
              new_user_input
            )
            selected_content_committed = user_entry is not None
            if (
              top_level_user_input is not None
              and user_entry is None
            ):
              raise RuntimeError(
                "Top-level skill user_message was not durably "
                "persisted"
              )
          messages = [dict(message) for message in messages]
        if top_level_user_input is not None:
          await self._emit_top_level_skill_started()
          system_prompt = (
            await self._prepare_top_level_skill_system_prompt(
              system_prompt
            )
          )

      if not selected_content_committed:
        raise RuntimeError(
          "Selected content was not durably committed before model work."
        )

      if self._coordinator is not None and self._coordinator.enabled:
        preamble = self._coordinator.preamble or coordinator_default_preamble
        system_prompt = prepend_system_prompt_preamble(system_prompt, preamble)

      self._capability_execution.validate()
      upstream_model = self._capability_execution.bind.upstream_model
      bound_effort = self._capability_execution.bind.effort
      config = normalized_run_config(
        self._auth_config,
        upstream_model=upstream_model,
        effort=bound_effort,
      )

      if not self._provider.has_active_credential(config):
        if getattr(self, "_allow_stub_response", False):
          await self._emit_stub_response(messages)
          return
        provider_name = str(getattr(self._provider, "name", "unknown") or "unknown")
        message = (
          "Provider startup failed: no active credential configured "
          f"for provider={provider_name}."
        )
        logger.error("[%s] %s", self._sid, message)
        clean_detach_reason = "error"
        await self._emit_error_event(message)
        return

      try:
        client = self._provider.create_client(config, timeout=self._client_timeout)
        self._set_client(client)
      except Exception as exc:
        if getattr(self, "_allow_stub_response", False):
          await self._emit_stub_response(messages)
          return
        provider_name = str(getattr(self._provider, "name", "unknown") or "unknown")
        message = f"Provider startup failed: could not create client for provider={provider_name}."
        logger.error(
          "[%s] %s | exception_type=%s",
          self._sid,
          message,
          type(exc).__name__,
        )
        clean_detach_reason = "error"
        await self._emit_error_event(message)
        return

      try:
        model_info = self._provider.get_model_info(upstream_model)
      except Exception as exc:
        provider_name = str(getattr(self._provider, "name", "unknown") or "unknown")
        message = (
          "Provider startup failed: model metadata unavailable "
          f"for provider={provider_name}."
        )
        logger.error(
          "[%s] %s | exception_type=%s",
          self._sid,
          message,
          type(exc).__name__,
        )
        clean_detach_reason = "error"
        await self._emit_error_event(message)
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
      logger.info("[%s] Effort requested | effort=%s", self._sid, bound_effort)

      effective_compaction_trigger = effective_compaction_trigger_fn(self._compaction_trigger, model_info)
      native_compaction = model_info.supports_native_compaction
      if effective_compaction_trigger is not None:
        compaction_mode = (
          "anthropic_native"
          if native_compaction
          else ("portable" if portable_compaction_enabled() else "disabled")
        )
        logger.info(
          "[%s] Compaction enabled | mode=%s trigger=%d tokens",
          self._sid,
          compaction_mode,
          effective_compaction_trigger,
        )
      else:
        logger.info("[%s] Compaction enabled | mode=disabled", self._sid)
      context_limit = model_context_window(model_info)

      logger.info("[%s] Chat start | model=%s max_tokens=%d messages=%d", self._sid, upstream_model, max_tokens, len(messages))

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
      delivery_grace_credits: dict[
        tuple[str, int, str],
        int,
      ] = {}
      delivery_grace_credit_limit: int | None = None
      delivery_grace_credits_granted = 0
      delivery_epoch_active = False
      delivery_epoch_from_max = False
      delivery_turn_compelled = False
      unread_handle_reminded_task_ids: set[str] = set()
      tools_used: List[str] = []
      final_answer_guard_fired = False
      max_tokens_continuations = 0
      max_tokens_continuation_pending = False
      is_child_logical_response = self._role == "sub_agent"
      logical_response_id: str | None = None
      logical_response_segment_ordinal = 0
      logical_response_previous_event_seq: int | None = None
      terminal_failure: tuple[str, str] | None = None
      terminal_interruption_reason: str | None = None
      pending_model_visible_tool_result_ids: set[str] = set()
      pending_parent_message_acks: dict[
        tuple[str, int],
        ParentMessage,
      ] = {
        (message.task_id, message.sent_seq): message
        for message in getattr(
          self,
          "_resume_parent_messages_for_ack",
          (),
        )
        if (
          isinstance(message, ParentMessage)
          and message.task_id is not None
          and message.sent_seq is not None
        )
      }
      self._resume_parent_messages_for_ack = ()
      model_visible_tool_result_generation = 0
      model_visible_synthesis_credits_granted = 0
      current_messages = list(messages)
      last_real_stream_input_tokens: int | None = None

      def reset_logical_response_lineage() -> None:
        nonlocal logical_response_id
        nonlocal logical_response_segment_ordinal
        nonlocal logical_response_previous_event_seq
        logical_response_id = None
        logical_response_segment_ordinal = 0
        logical_response_previous_event_seq = None

      def ensure_logical_response_id() -> str:
        nonlocal logical_response_id
        if logical_response_id is None:
          logical_response_id = (
            f"logical_response_{uuid_module.uuid4().hex}"
          )
        return logical_response_id

      def mark_terminal_interruption(reason: str) -> None:
        nonlocal clean_detach_reason
        nonlocal terminal_interruption_reason
        clean_detach_reason = reason
        terminal_interruption_reason = reason

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
        mark_terminal_interruption("budget_exceeded")
        await self._emit_interrupted_event("budget_exceeded")

      async def emit_terminal_failure(
        reason: str,
        message: str,
      ) -> None:
        nonlocal clean_detach_reason
        clean_detach_reason = "error"
        if (
          self._top_level_skill_lifecycle is not None
          and not self._top_level_skill_result_committed
          and self._top_level_skill_result_failure is None
          and self._prepared_top_level_terminal_event is not None
        ):
          self._top_level_skill_result_failure = RuntimeError(
            message
          )
          self._top_level_skill_result_failure_code = reason
        logger.error("[%s] %s", self._sid, message)
        await self._emit_error_event(message)
        await self._emit_interrupted_event(reason)
        await self._close_client(client, timeout=5.0)

      while True:
        if self._operator_pause_requested():
          logger.info("[%s] Operator pause requested before next turn; stopping at safe boundary", self._sid)
          mark_terminal_interruption("operator_pause")
          await self._emit_operator_pause_event("before_turn")
          break

        parent_messages: list[ParentMessage] = []
        if self._message_inbox is not None:
          while not self._message_inbox.empty():
            try:
              parent_messages.append(
                self._message_inbox.get_nowait()
              )
            except queue_empty:
              break
        if parent_messages:
          if (
            is_child_logical_response
            and max_tokens_continuation_pending
          ):
            # Parent/operator guidance changes the task semantics. The next
            # provider response is not a continuation of the pre-update
            # partial, even though that partial ended at max_tokens.
            reset_logical_response_lineage()
            max_tokens_continuation_pending = False
            max_tokens_continuations = 0
          current_messages.append(user_turn_message(
            format_parent_messages(parent_messages)
          ))
          pending_parent_message_acks.update({
            (message.task_id, message.sent_seq): message
            for message in parent_messages
            if (
              message.task_id is not None
              and message.sent_seq is not None
            )
          })

        continuing_provider_segment = (
          is_child_logical_response
          and max_tokens_continuation_pending
        )
        max_tokens_continuation_pending = False
        if not continuing_provider_segment:
          turn_count += 1
        active_delivery_obligation: tuple[str, int, str] | None = None
        crossed_max_turns = (
          max_turns is not None
          and turn_count > max_turns
        )
        if crossed_max_turns and not delivery_epoch_active:
          delivery_epoch_active = True
          delivery_epoch_from_max = True
          self._background_delivery_grace_active = True
        if delivery_epoch_active:
          if delivery_grace_credit_limit is None:
            delivery_grace_credit_limit = (
              self._background_delivery_grace_credit_limit()
            )
          pending_acknowledgements = getattr(
            self,
            "_pending_background_result_acks",
            {},
          )
          if isinstance(pending_acknowledgements, dict):
            for tool_use_id, acknowledgement in list(
              pending_acknowledgements.items()
            ):
              if (
                not isinstance(acknowledgement, tuple)
                or len(acknowledgement) != 2
                or not isinstance(acknowledgement[0], str)
                or not isinstance(acknowledgement[1], int)
                or isinstance(acknowledgement[1], bool)
              ):
                continue
              task_id, notification_generation = acknowledgement
              if acknowledgement in self._background_ack_recovery_pairs:
                phase = "recovery_acknowledgement"
              else:
                phase = "acknowledgement"
              acknowledgement_obligation = (
                task_id,
                notification_generation,
                phase,
              )
              if (
                phase == "acknowledgement"
                and acknowledgement_obligation
                in delivery_grace_credits
                and delivery_grace_credits[
                  acknowledgement_obligation
                ] <= 0
              ):
                self._background_ack_recovery_pairs.add(
                  acknowledgement
                )
                for duplicate_id, duplicate in list(
                  pending_acknowledgements.items()
                ):
                  if duplicate == acknowledgement:
                    pending_acknowledgements.pop(
                      duplicate_id,
                      None,
                    )
          obligations = self._background_delivery_grace_obligations()
          if (
            (
              pending_model_visible_tool_result_ids
              # B-3: at turn exhaustion the synthesis obligation is minted
              # unconditionally — this is the completion reserve that elicits a
              # terminal narrative, and without it the honest-partial remap has
              # no narrative to settle on. Still credit-capped below by the one
              # synthesis credit and by delivery_grace_credit_limit.
              or delivery_epoch_from_max
            )
            and not obligations
          ):
            obligations[
              (
                "model_visible_tool_results",
                model_visible_tool_result_generation,
                "synthesis",
              )
            ] = 1
          for obligation, requested_credits in obligations.items():
            if obligation in delivery_grace_credits:
              continue
            available_credits = max(
              0,
              delivery_grace_credit_limit
              - delivery_grace_credits_granted,
            )
            if obligation[2] == "synthesis":
              available_credits = min(
                available_credits,
                max(
                  0,
                  1 - model_visible_synthesis_credits_granted,
                ),
              )
            granted_credits = min(
              max(0, int(requested_credits)),
              available_credits,
            )
            delivery_grace_credits[obligation] = granted_credits
            delivery_grace_credits_granted += granted_credits
            if obligation[2] == "synthesis":
              model_visible_synthesis_credits_granted += (
                granted_credits
              )
          # Credits count the turns that *cannot* settle a delivery: the
          # compelled ones, where the loop re-invokes the provider itself
          # (a continuation the model never completed, or a delivery nudge
          # after it tried to stop). A completed turn acks what was rendered
          # to it, so it makes progress on its own and is not charged --
          # otherwise a parent that legitimately keeps working between
          # deliveries, which is exactly what a staged fan-out does, spends a
          # budget sized for draining a queue and dies mid-flight. The frozen
          # post-`max_turns` epoch stays metered unconditionally: there the
          # turn budget is already spent and every turn is a delivery turn.
          metered_delivery_turn = (
            delivery_epoch_from_max or delivery_turn_compelled
          )
          delivery_turn_compelled = False
          if metered_delivery_turn:
            active_delivery_obligation = next(
              (
                obligation
                for obligation in obligations
                if delivery_grace_credits.get(obligation, 0) > 0
              ),
              None,
            )
          if not metered_delivery_turn:
            active_delivery_obligation = None
          elif active_delivery_obligation is not None:
            delivery_grace_credits[
              active_delivery_obligation
            ] -= 1
            logger.info(
              "[%s] Bounded background delivery turn "
              "(obligation=%s generation=%d phase=%s credits=%d/%d)",
              self._sid,
              active_delivery_obligation[0],
              active_delivery_obligation[1],
              active_delivery_obligation[2],
              delivery_grace_credits[
                active_delivery_obligation
              ],
              delivery_grace_credit_limit,
            )
          else:
            if not delivery_epoch_from_max and not obligations:
              terminal_success_reason = "background_delivery_settled"
              break
            if delivery_epoch_from_max:
              logger.warning(
                "[%s] Max turns (%d) reached, stopping",
                self._sid,
                max_turns,
              )
              self._append(
                build_max_turns_reached_event(
                  turn_count=turn_count,
                  max_turns=max_turns,
                )
              )
              mark_terminal_interruption("max_turns_reached")
              await self._emit_interrupted_event(
                "max_turns_reached"
              )
              summary_text = None
              if self._on_max_turns is not None:
                try:
                  summary_text = await self._on_max_turns(
                    current_messages,
                    turn_count,
                  )
                except Exception as exc:
                  logger.warning(
                    "[%s] on_max_turns callback failed: %s",
                    self._sid,
                    exc,
                  )
              self._append(
                build_max_turns_text_event(summary_text)
              )
            else:
              logger.warning(
                "[%s] Bounded background delivery credits exhausted",
                self._sid,
              )
              clean_detach_reason = "error"
              await self._emit_error_event(
                "background_delivery_exhausted: bounded background-result "
                "delivery credits were exhausted before all retained "
                "notifications were acknowledged."
              )
              await self._close_client(client, timeout=5.0)
              return
            break
        turn_t0 = time_module.time()
        turn_t0_mono = time_module.monotonic()

        estimate_snapshot = initial_estimate
        if turn_count > 1:
          current_tools = base_kwargs.get("tools") or []
          turn_estimate = token_estimate_snapshot(
            system_text=system_text,
            messages=current_messages,
            tools=current_tools,
          )
          estimate_snapshot = turn_estimate
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
          delivered=self._notification_delivery_set(),
        )
        # The delivered set for this provider request. Recorded once per
        # outer turn (the inner request-preparation loop re-sends the same
        # reminder unchanged) and acked at the response boundary that saw
        # it (A-M7, T3-I01).
        delivered_notifications: list[Any] = list(turn_reminder.delivered)
        delivered_notifications_recorded = False
        pending_notifications_at_render = (
          self._notification_queue.pending_count
        )
        est_for_compaction = int(estimate_snapshot.est_total_tokens)
        if last_real_stream_input_tokens is not None:
          est_for_compaction = max(est_for_compaction, last_real_stream_input_tokens)

        turn_usage_before = usage_snapshot(usage_totals)
        pending_compaction_block: dict[str, Any] | None = None
        pending_compaction_usage: dict[str, Any] | None = None
        provisional_anchor_message: dict[str, Any] | None = None
        request_scoped_nudge_message: dict[str, Any] | None = None
        delivery_request_nudge_message: dict[str, Any] | None = None
        delivery_request_nudge_recorded = False
        fork_suffix_reminder_text: str | None = None

        def remove_current_message(target: dict[str, Any] | None) -> None:
          if target is None:
            return
          for index, message in enumerate(current_messages):
            if message is target:
              current_messages.pop(index)
              return
          for index, message in enumerate(current_messages):
            if message == target:
              current_messages.pop(index)
              return

        async def persist_anchor_only_and_return() -> None:
          if pending_compaction_block is None:
            return
          await self._append_assistant_message_event(
            content_blocks=[pending_compaction_block],
            stop_reason="compaction",
            model=upstream_model,
            usage=dict(pending_compaction_usage or {}),
            logical_response_id=(
              f"logical_response_{uuid_module.uuid4().hex}"
            ),
            logical_response_segment_ordinal=0,
          )

        async def apply_portable_compaction(*, force: bool = False) -> bool:
          nonlocal current_messages
          nonlocal pending_compaction_block
          nonlocal pending_compaction_usage
          nonlocal provisional_anchor_message
          compact_result = await maybe_compact_current_messages(
            current_messages,
            self._compaction_instructions,
            capability_execution=self._capability_execution,
            trigger=effective_compaction_trigger,
            est_total_tokens=est_for_compaction,
            turn_count=turn_count,
            last_compact_turn=getattr(self, "_portable_compaction_last_turn", None),
            failed_est_at=None if force else getattr(self, "_portable_compaction_failed_est_at", None),
            tools=base_kwargs.get("tools") or [],
            force=force,
          )
          if compact_result.summarize_usage is not None:
            _merge_usage_totals(usage_totals, compact_result.summarize_usage)
          if compact_result.applied:
            current_messages = compact_result.messages
            _discard_unavailable_background_result_acks(
              self,
              available_tool_result_ids=_tool_result_ids(
                current_messages
              ),
            )
            pending_compaction_block = compact_result.anchor_block
            pending_compaction_usage = compact_result.summarize_usage
            provisional_anchor_message = current_messages[0] if current_messages else None
            self._portable_compaction_last_turn = turn_count
            self._portable_compaction_failed_est_at = None
            self._append({
              "type": "compaction",
              "chars": compact_result.summary_chars,
            })
            logger.info(
              "[%s] Portable compaction applied | turn=%d force=%s est_before=%d est_after=%d summary_chars=%d real=%s",
              self._sid,
              turn_count,
              force,
              compact_result.est_before,
              compact_result.est_after,
              compact_result.summary_chars,
              last_real_stream_input_tokens if last_real_stream_input_tokens is not None else "n/a",
            )
            return True
          if compact_result.reason == "summary_too_short":
            self._portable_compaction_failed_est_at = compact_result.est_before
          if (
            compact_result.reason == "insufficient_gain"
            and est_for_compaction >= int(effective_compaction_trigger or 0)
            and not getattr(self, "_portable_compaction_floor_warned", False)
          ):
            self._portable_compaction_floor_warned = True
            logger.warning(
              "[%s] portable_compaction_floor | est=%d trigger=%d compressible_below=%d",
              self._sid,
              est_for_compaction,
              int(effective_compaction_trigger or 0),
              MIN_COMPACT_GAIN_TOKENS,
            )
          if compact_result.reason not in {"below_trigger", "native_model", "disabled", "disabled_by_env"}:
            logger.info(
              "[%s] Portable compaction skipped | turn=%d force=%s reason=%s est=%d real=%s",
              self._sid,
              turn_count,
              force,
              compact_result.reason,
              compact_result.est_before,
              last_real_stream_input_tokens if last_real_stream_input_tokens is not None else "n/a",
            )
          return False

        if effective_compaction_trigger is not None and not native_compaction:
          await apply_portable_compaction()

        if (
          pending_compaction_block is None
          and effective_compaction_trigger is not None
          and not native_compaction
          and current_messages
          and _is_anchor_only_compaction_message(current_messages[-1])
        ):
          request_scoped_nudge_message = user_turn_message(CONTINUATION_NUDGE)
          current_messages.append(request_scoped_nudge_message)

        reactive_retry_attempted = False
        request_tool_result_ids: set[str] = set()
        while True:
          usage_before_stream = usage_snapshot(usage_totals)
          if (
            bool(getattr(self, "_fork_mode", False))
            and not bool(
              getattr(self, "_fork_suffix_ceiling_triggered", False)
            )
          ):
            suffix_messages = fork_suffix_messages(
              current_messages,
              getattr(self, "_fork_marker_position"),
            )
            suffix_estimate = token_estimate_snapshot(
              system_text="",
              messages=suffix_messages,
              tools=[],
            )
            if suffix_estimate.est_messages_tokens > int(
              getattr(self, "_fork_suffix_max_tokens")
            ):
              self._fork_suffix_ceiling_triggered = True
              fork_suffix_reminder_text = (
                FORK_SUFFIX_WRAP_UP_REMINDER
              )
              max_turns = (
                turn_count
                if max_turns is None
                else min(max_turns, turn_count)
              )
          if (
            active_delivery_obligation is not None
            and active_delivery_obligation[2]
            in {"exact_retrieval", "ack_recovery"}
          ):
            omitted_nudge_text = _omitted_background_result_nudge(
              [active_delivery_obligation[0]]
            )
            delivery_request_nudge_message = user_turn_message(
              omitted_nudge_text
            )
            current_messages.append(
              delivery_request_nudge_message
            )

          combined_reminder = "\n\n".join(
            part
            for part in (
              turn_reminder.text,
              fork_suffix_reminder_text,
            )
            if part
          )
          if combined_reminder:
            provisional_system_prompt = (
              self._inject_system_prompt_reminder(
                system_prompt,
                combined_reminder,
              )
            )
          else:
            provisional_system_prompt = system_prompt
          request_estimate = token_estimate_snapshot(
            system_text=system_prompt_estimate_text(
              provisional_system_prompt
            ),
            messages=current_messages,
            tools=base_kwargs.get("tools") or [],
          )
          context_pressure = context_pressure_reminder_decision(
            est_tokens=request_estimate.est_total_tokens,
            context_limit=context_limit,
            next_threshold_pct=getattr(
              self,
              "_context_pressure_next_reminder_pct",
              CONTEXT_PRESSURE_REMINDER_PCT,
            ),
            initial_threshold_pct=CONTEXT_PRESSURE_REMINDER_PCT,
            step_pct=CONTEXT_PRESSURE_REMINDER_STEP_PCT,
          )
          self._context_pressure_next_reminder_pct = (
            context_pressure.next_threshold_pct
          )
          context_pressure_reminder = (
            build_context_pressure_reminder(
              pct=context_pressure.reminder_pct,
            )
            if context_pressure.reminder_pct is not None
            else ""
          )
          reminder_text = "\n\n".join(
            filter(
              None,
              [
                context_pressure_reminder,
                turn_reminder.text,
                fork_suffix_reminder_text,
              ],
            )
          )
          if reminder_text:
            turn_system_prompt = self._inject_system_prompt_reminder(
              system_prompt,
              reminder_text,
            )
          else:
            turn_system_prompt = system_prompt
          try:
            request_budget_admission = admit_provider_request_budget(
              self._cost_accumulator,
              provider=self._provider,
              model=upstream_model,
              estimated_input_tokens=(
                conservative_request_input_token_bound_for_request(
                  system_text=system_prompt_estimate_text(
                    turn_system_prompt
                  ),
                  messages=current_messages,
                  tools=base_kwargs.get("tools") or [],
                )
              ),
              requested_max_output_tokens=max_tokens,
            )
          except ProviderRequestBudgetError:
            await emit_terminal_failure(
              "budget_authority_invalid",
              "Provider request budget authority could not be verified.",
            )
            return
          if request_budget_admission.denied_state is not None:
            await emit_budget_exceeded_stop(
              request_budget_admission.denied_state
            )
            return
          request_max_tokens = request_budget_admission.max_output_tokens
          if request_max_tokens is None:  # pragma: no cover - denied above
            raise RuntimeError("provider request budget admission is invalid")
          if request_max_tokens < max_tokens:
            logger.info(
              "[%s] Provider max_tokens capped by remaining cost authority | requested=%d capped=%d",
              self._sid,
              max_tokens,
              request_max_tokens,
            )
          if self._context_capture is not None:
            surfaces = self._context_surface_records()
            try:
              system_prompt_hash = await asyncio.to_thread(
                self._context_capture.persist,
                surfaces=surfaces,
                rendered_system_prompt=turn_system_prompt,
              )
              digest = canonical_manifest_digest(
                surfaces,
                system_prompt_hash,
              )
              if digest != self._last_context_manifest_digest:
                manifest_event = build_context_manifest_event(
                  surfaces=surfaces,
                  system_prompt_hash=system_prompt_hash,
                  session_id=self._full_session_id,
                  request_id=self._request_id,
                  turn=turn_count,
                )
                await self._append_durable_event(manifest_event)
                self._append(manifest_event)
                self._last_context_manifest_digest = digest
            except Exception as exc:
              logger.warning(
                "[%s] context capture failed; manifest suppressed | exception_type=%s",
                self._sid,
                type(exc).__name__,
              )
          if delivered_notifications and not delivered_notifications_recorded:
            # Durable render record (CUR-E2E-08 observability): which
            # request boundary showed each queued notification to the
            # model. Identifiers only — the notification's summary/payload
            # is untrusted child content and stays out of unsanitized
            # event fields. Durable-only, like context_manifest: never on
            # the client wire, ignored by every fold. Sits after budget
            # admission so the record means "carried by an issued request";
            # a request the provider failed to process can still leave a
            # record whose notifications were never acked, so forensics
            # dedupe by (task_id, notification_generation). Observability
            # never vetoes work: append failures are logged and suppressed.
            delivered_notifications_recorded = True
            try:
              await self._append_durable_event({
                "type": "background_notifications_rendered",
                "turn": turn_count,
                "request_id": self._request_id,
                "pending_count": pending_notifications_at_render,
                "notifications": [
                  {
                    "task_id": str(getattr(notification, "task_id", "")),
                    "notification_generation": getattr(
                      notification, "notification_generation", None
                    ),
                    "event": str(getattr(notification, "event", "")),
                  }
                  for notification in delivered_notifications
                ],
              })
            except Exception as exc:
              logger.warning(
                "[%s] notification render record suppressed | exception_type=%s",
                self._sid,
                type(exc).__name__,
              )
          if (
            delivery_request_nudge_message is not None
            and not delivery_request_nudge_recorded
          ):
            # The matching durable record for the request-scoped delivery
            # nudge — emitted here, after budget admission, so it attests a
            # nudge an issued request actually carried.
            delivery_request_nudge_recorded = True
            try:
              await self._append_durable_event(
                build_runtime_guard_event(
                  guard="omitted_background_result_nudge",
                  message=str(
                    (delivery_request_nudge_message.get("content") or "")
                  ),
                )
              )
            except Exception as exc:
              logger.warning(
                "[%s] delivery nudge record suppressed | exception_type=%s",
                self._sid,
                type(exc).__name__,
              )
          request_tool_result_ids = _tool_result_ids(current_messages)
          turn_result = await self._stream_turn(
            client=client,
            config=config,
            model_info=model_info,
            system_prompt=turn_system_prompt,
            current_messages=current_messages,
            base_kwargs=base_kwargs,
            max_tokens=request_max_tokens,
            turn_count=turn_count,
            turn_t0=turn_t0,
            turn_t0_mono=turn_t0_mono,
            system_chars=system_chars,
            tools_chars=tools_chars,
            usage_totals=usage_totals,
          )
          if isinstance(turn_result, StreamTurnFailure):
            remove_current_message(request_scoped_nudge_message)
            request_scoped_nudge_message = None
            remove_current_message(
              delivery_request_nudge_message
            )
            delivery_request_nudge_message = None
            can_reactive_compact = (
              turn_result.is_context_length
              and not reactive_retry_attempted
              and pending_compaction_block is None
              and effective_compaction_trigger is not None
              and not native_compaction
            )
            if can_reactive_compact:
              reactive_retry_attempted = True
              if await apply_portable_compaction(force=True):
                try:
                  await self._close_client(client, timeout=2.0)
                except Exception:
                  pass
                client = self._provider.create_client(config, timeout=self._client_timeout)
                self._set_client(client)
                continue
            if pending_compaction_block is not None:
              await persist_anchor_only_and_return()
            await self._emit_error_event(turn_result.formatted_error)
            await self._close_client(client, timeout=5.0)
            return
          if turn_result is None:
            remove_current_message(request_scoped_nudge_message)
            remove_current_message(
              delivery_request_nudge_message
            )
            if pending_compaction_block is not None:
              await persist_anchor_only_and_return()
            return
          remove_current_message(request_scoped_nudge_message)
          request_scoped_nudge_message = None
          remove_current_message(
            delivery_request_nudge_message
          )
          delivery_request_nudge_message = None
          break

        client, turn = turn_result
        assistant_projection = sanitize_tool_event(
          {
            "type": "assistant_message",
            "content_blocks": turn.content_blocks,
          },
          sink="model_history",
          boundary=getattr(self, "_secret_boundary", None),
        )
        turn.content_blocks = list(
          assistant_projection.get("content_blocks") or []
        )
        base_kwargs["_request_advertised_tool_names"] = (
          getattr(turn, "advertised_tool_names", None)
        )
        if pending_compaction_block is not None:
          remove_current_message(provisional_anchor_message)
          turn.content_blocks.insert(0, pending_compaction_block)
          pending_compaction_block = None

        turn_usage_state = usage_delta_state(turn_usage_before, usage_totals)
        turn_usage = turn_usage_state.usage
        stream_usage_state = usage_delta_state(usage_before_stream, usage_totals)
        stream_usage = stream_usage_state.usage
        real_stream_input = (
          int(stream_usage.get("input_tokens", 0) or 0)
          + int(stream_usage.get("cache_read_input_tokens", 0) or 0)
          + int(stream_usage.get("cache_creation_input_tokens", 0) or 0)
        )
        if real_stream_input > 0:
          last_real_stream_input_tokens = real_stream_input
        turn_usage_payload = turn_usage_payload_fn(turn_usage)
        if turn_usage_state.has_tokens:
          turn_cost = self._estimate_usage_cost(upstream_model, turn_usage)
          turn_usage_payload = turn_usage_payload_fn(turn_usage, estimated_cost=turn_cost.total)
          await self._call_on_usage(self._build_usage_event(model=upstream_model, usage_totals=turn_usage))
        assistant_message_persisted = False

        async def persist_assistant_message_once() -> None:
          nonlocal assistant_message_persisted
          nonlocal logical_response_segment_ordinal
          nonlocal logical_response_previous_event_seq
          if assistant_message_persisted:
            return
          workflow_output_attachments = (
            list(self._pending_workflow_output_attachments.values())
            if (
              not turn.tool_uses
              and turn.stop_reason == "end_turn"
            )
            else None
          )
          entry = await self._append_assistant_message_event(
            content_blocks=turn.content_blocks,
            stop_reason=turn.stop_reason,
            model=upstream_model,
            usage=turn_usage_payload,
            parent_messages=list(pending_parent_message_acks.values()),
            consumer_turn=turn_count,
            logical_response_id=ensure_logical_response_id(),
            logical_response_segment_ordinal=(
              logical_response_segment_ordinal
            ),
            continued_from_assistant_message_seq=(
              logical_response_previous_event_seq
            ),
            workflow_output_attachments=workflow_output_attachments,
          )
          assistant_message_persisted = True
          if (
            turn.stop_reason == "max_tokens"
            and not bool(turn.tool_uses)
            and entry is not None
          ):
            logical_response_previous_event_seq = entry.seq
            logical_response_segment_ordinal += 1
          else:
            reset_logical_response_lineage()
          if pending_parent_message_acks:
            await self._acknowledge_parent_messages_consumed(
              list(pending_parent_message_acks.values()),
              consumer_turn=turn_count,
            )
            pending_parent_message_acks.clear()
          if (
            turn.stop_reason == "end_turn"
            or bool(turn.tool_uses)
          ):
            pending_model_visible_tool_result_ids.difference_update(
              request_tool_result_ids
            )
            _acknowledge_consumed_background_results(
              self,
              consumed_tool_result_ids=request_tool_result_ids,
            )

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
          running_cost = self._estimate_usage_cost(upstream_model, usage_totals)
          budget_progress = budget_cost_progress(
            self._cost_accumulator,
            running_total=running_cost.total,
            last_reported_cost=self._last_reported_cost,
          )
          self._last_reported_cost = budget_progress.last_reported_cost
          exceeded_state = budget_progress.exceeded_state
        else:
          exceeded_state = None

        if turn.stop_reason in {"error", "aborted"}:
          clean_detach_reason = "error"
          await self._emit_error_event(
            "provider_turn_error: provider response ended with "
            f"stop_reason={turn.stop_reason!r} before completing the turn."
          )
          await self._close_client(client, timeout=5.0)
          return

        if turn.tool_uses and self._operator_pause_requested():
          await persist_assistant_message_once()
          logger.info("[%s] Operator pause requested after turn; stopping before tool dispatch", self._sid)
          mark_terminal_interruption("operator_pause")
          await self._emit_operator_pause_event("after_turn_before_tools")
          break

        if not turn.tool_uses:
          if turn.stop_reason == "end_turn" and bool(turn.full_text.strip()):
            learning_real_final_response = True
          final_answer_guard_allowed = turn.stop_reason not in {"max_tokens", "pause_turn", "compaction"}
          if (
            final_answer_guard_allowed
            and not final_answer_guard_fired
            and self._final_answer_guard is not None
          ):
            guard_message = self._final_answer_guard(
              current_messages,
              turn.full_text,
              # Registered workflow-child evidence is verified provenance:
              # merge it into the guard's tools view so already-performed
              # workflow retrieval is not re-demanded from the parent.
              _guard_visible_tools_used(
                tools_used,
                getattr(
                  self,
                  "_workflow_evidence_provenance",
                  {},
                ).values(),
              ),
              list(base_kwargs.get("tools") or []),
              turn_count,
            )
            if guard_message:
              final_answer_guard_fired = True
              # The guard rejects this draft and starts a revised logical
              # response. Never splice a prior max_tokens prefix across that
              # semantic boundary.
              reset_logical_response_lineage()
              logger.info("[%s] Final-answer guard injected follow-up before completing turn", self._sid)
              runtime_guard_event = build_runtime_guard_event(guard="final_answer", message=guard_message)
              self._append(runtime_guard_event)
              durable_runtime_guard_event = dict(runtime_guard_event)
              durable_runtime_guard_event.update(
                {
                  "draft_content_blocks": list(turn.content_blocks),
                  "draft_stop_reason": turn.stop_reason,
                  "draft_model": upstream_model,
                  "draft_provider": self._provider.name,
                  "draft_usage": dict(turn_usage_payload),
                }
              )
              await self._append_durable_event(durable_runtime_guard_event)
              current_messages.append(
                assistant_turn_message(
                  turn.content_blocks,
                  provider=self._provider.name,
                  model=upstream_model,
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
            mark_terminal_interruption("operator_pause")
            await self._emit_operator_pause_event("after_turn_before_tools")
            break
          no_tool_outcome = no_tool_use_turn_outcome(
            content_blocks=turn.content_blocks,
            provider=self._provider.name,
            model=upstream_model,
            stop_reason=turn.stop_reason,
            pending_notification_count=0,
            max_tokens_continuations=max_tokens_continuations,
            max_tokens_max_attempts=max_tokens_continuations_limit,
            max_tokens_nudge=max_tokens_nudge,
            unbounded_max_tokens_continuations=(
              is_child_logical_response
            ),
          )
          max_tokens_continuations = no_tool_outcome.max_tokens_continuations
          if turn.stop_reason == "pause_turn":
            logger.info("[%s] Pause turn — continuing", self._sid)
          elif turn.stop_reason == "compaction":
            logger.info("[%s] Compaction pause — continuing", self._sid)
          elif no_tool_outcome.reason == "max_tokens_continue":
            if is_child_logical_response:
              logger.warning(
                "[%s] Turn %d logical response segment %d hit "
                "max_tokens; continuing provider transport",
                self._sid,
                turn_count,
                logical_response_segment_ordinal,
              )
              max_tokens_continuation_pending = True
            else:
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
            terminal_failure = (
              "max_tokens_exhausted",
              "provider_turn_error: provider exhausted the bounded "
              "max_tokens continuation attempts before completing the turn.",
            )
          if no_tool_outcome.runtime_guard is not None:
            guard, message = no_tool_outcome.runtime_guard
            self._append(build_runtime_guard_event(guard=guard, message=message))
          if no_tool_outcome.action == "continue":
            current_messages.extend(no_tool_outcome.messages)
            # The pause/compaction/max_tokens arm cannot ack (D-A7-1): it
            # re-invokes the provider on a turn the model never completed, so
            # nothing discharges an outstanding delivery. These are the turns
            # the bounded credits exist to count.
            delivery_turn_compelled = True
            continue
          if terminal_failure is not None:
            break
          delivery_follow_up_allowed = turn.stop_reason in {
            "end_turn",
            "tool_use",
          }
          # Ack follows delivery: this response consumed the reminder that
          # rendered `delivered_notifications`, so the model has seen them
          # (A-M7, T3-I01). Reached only on a completed model turn — the
          # `error` arm broke earlier at `terminal_failure`, and the
          # pause/compaction/max_tokens continuation arm `continue`d above
          # (D-A7-1).
          if delivered_notifications:
            self._ack_delivered_notifications(delivered_notifications)
            delivered_notifications = []
          if (
            turn.stop_reason == "end_turn"
            and self._background_notifications_enabled
            and self._notification_queue.pending_count == 0
            and (
              self._task_registry.list_tasks(
                state=task_state_cls.RUNNING
              )
              # An unsettled workflow obligation holds the session open
              # (§5.3, T2-I04): a parked or authoring run has no RUNNING
              # registry task.
              or self._workflow_settlement_obstructed()
            )
            and (max_turns is None or turn_count < max_turns)
          ):
            await self._wait_for_background_notification()
            if self._cost_accumulator is not None:
              exceeded_state = (
                budget_exceeded_state(self._cost_accumulator)
                or exceeded_state
              )
            if exceeded_state is not None:
              await emit_budget_exceeded_stop(exceeded_state)
              break
            if self._operator_pause_requested():
              logger.info(
                "[%s] Operator pause requested while awaiting background completion",
                self._sid,
              )
              mark_terminal_interruption("operator_pause")
              await self._emit_operator_pause_event(
                "awaiting_background_completion"
              )
              break
          if (
            delivery_follow_up_allowed
            and self._notification_queue.pending_count > 0
          ):
            current_messages.append(
              assistant_turn_message(
                turn.content_blocks,
                provider=self._provider.name,
                model=upstream_model,
                stop_reason=turn.stop_reason,
              )
            )
            current_messages.append(
              background_tasks_completed_user_message()
            )
            # Metering the delivery turns is all this arm does. It must NOT
            # latch `_background_delivery_grace_active`: that flag closes
            # admission of new background tasks and says so in words the run
            # would be lying about here ("past its normal turn limit" — this
            # arm runs precisely when `max_turns is None`). Latching it on the
            # first arriving notification made staged fan-out impossible: a
            # parent that dispatches its next track once a slot frees had that
            # dispatch rejected, and did the work inline instead until the
            # delivery credits ran out. Only the post-`max_turns` epoch and the
            # end of the loop close admission.
            if max_turns is None:
              delivery_epoch_active = True
              delivery_turn_compelled = True
            continue
          pending_omitted_task_ids = (
            _pending_omitted_background_task_ids(self)
          )
          if (
            delivery_follow_up_allowed
            and pending_omitted_task_ids
          ):
            current_messages.append(
              assistant_turn_message(
                turn.content_blocks,
                provider=self._provider.name,
                model=upstream_model,
                stop_reason=turn.stop_reason,
              )
            )
            omitted_nudge_text = _omitted_background_result_nudge(
              pending_omitted_task_ids
            )
            current_messages.append(
              user_turn_message(omitted_nudge_text)
            )
            try:
              await self._append_durable_event(
                build_runtime_guard_event(
                  guard="omitted_background_result_nudge",
                  message=omitted_nudge_text,
                )
              )
            except Exception as exc:
              logger.warning(
                "[%s] delivery nudge record suppressed | exception_type=%s",
                self._sid,
                type(exc).__name__,
              )
            if max_turns is None:
              delivery_epoch_active = True
              delivery_turn_compelled = True
            continue
          unread_handle_entries = [
            entry
            for entry in _unread_settled_handle_entries(self)
            if entry.task_id not in unread_handle_reminded_task_ids
          ][:_UNREAD_HANDLE_NUDGE_MAX_TASKS]
          if (
            delivery_follow_up_allowed
            and unread_handle_entries
            and not delivery_epoch_from_max
            # The reminder must never consume the run's LAST budgeted turn:
            # crossing `max_turns` here would latch the from-max epoch and
            # `_background_delivery_grace_active`, spend the synthesis
            # credit, and turn an obeying model's read into a
            # `max_turns_reached` interruption. On a bounded run whose
            # finish lands on the limit, the reminder yields (fail-open).
            and (max_turns is None or turn_count < max_turns)
          ):
            # CUR-E2E-08: the delivery contract ended at "rendered once and
            # acked", and a handle-shaped result whose notification landed
            # while the model was firefighting another child was never read
            # — the run then closed claiming the settled task was still
            # outstanding. One reminder turn per task, keyed by the
            # unexercised read grant (typed ground truth, not claim
            # parsing). Deliberately UNMETERED and epoch-free: this is not a
            # delivery obligation, so it must not spend delivery grace
            # credits (a5fe21207) and must not latch
            # `_background_delivery_grace_active` (e6b071335). The
            # reminded-once set is the liveness bound: a model that ignores
            # the reminder finishes on its next stop.
            unread_handle_reminded_task_ids.update(
              entry.task_id for entry in unread_handle_entries
            )
            # Every delivery obligation is discharged here (no queued
            # notifications, no omitted payloads), so the live delivery
            # epoch is over — close it. Left latched, the model's answer to
            # this reminder could stop at max_tokens/pause, and that
            # compelled continuation would hit a metered turn with zero
            # obligations and commit `background_delivery_settled` success
            # mid-sentence, truncating the very integration the reminder
            # asked for. A later arriving notification re-latches the epoch
            # exactly as the first one did.
            if (
              delivery_epoch_active
              and not delivery_epoch_from_max
              and not self._background_delivery_grace_obligations()
            ):
              delivery_epoch_active = False
            current_messages.append(
              assistant_turn_message(
                turn.content_blocks,
                provider=self._provider.name,
                model=upstream_model,
                stop_reason=turn.stop_reason,
              )
            )
            unread_handle_nudge_text = _unread_result_handle_nudge(
              unread_handle_entries
            )
            current_messages.append(
              user_turn_message(unread_handle_nudge_text)
            )
            try:
              await self._append_durable_event(
                build_runtime_guard_event(
                  guard="unread_result_handle_nudge",
                  message=unread_handle_nudge_text,
                )
              )
            except Exception as exc:
              logger.warning(
                "[%s] reminder record suppressed | exception_type=%s",
                self._sid,
                type(exc).__name__,
              )
            continue
          if turn.stop_reason == "end_turn":
            terminal_success_reason = "end_turn"
          else:
            terminal_failure = (
              "terminal_outcome_unproven",
              "terminal_outcome_unproven: provider turn ended without a "
              "recognized successful or interrupted terminal disposition.",
            )
          break

        await persist_assistant_message_once()
        # Same boundary that acknowledges consumed background results: the
        # model answered the request that rendered these notifications, so
        # ack before the tool batch runs. A batch ending in
        # `stop_after_tool_results_reason` (or a budget stop) can then no
        # longer strand what the model already saw (A-M7, CUR-E2E-04).
        if delivered_notifications:
          self._ack_delivered_notifications(delivered_notifications)
          delivered_notifications = []
        learning_tool_calling_iters += 1
        current_messages.append(
          assistant_turn_message(
            turn.content_blocks,
            provider=self._provider.name,
            model=upstream_model,
            stop_reason=turn.stop_reason,
          )
        )
        if any(
          tool_name == "run_agent"
          and isinstance(tool_input, dict)
          and tool_input.get("fork") is True
          for _, tool_name, tool_input in turn.tool_uses
        ):
          self._mid_turn_fork_handoff = None
          if not fork_credential_identity_available(self):
            logger.warning(
              "[%s] Mid-turn fork capture unavailable: no visible "
              "credential identity",
              self._sid,
            )
          else:
            try:
              self._mid_turn_fork_handoff = build_mid_turn_handoff(
                self,
                current_messages,
              )
            except Exception:
              logger.warning(
                "[%s] Mid-turn fork capture failed | failure=true",
                self._sid,
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
          on_exception=lambda exc: logger.warning(
            "[%s] run_agent gather exception | exception_type=%s",
            self._sid,
            type(exc).__name__,
          ),
          should_stop_after_tool_result=lambda: bool(
            getattr(self, "_stop_after_tool_results_reason", None)
          ),
        )
        tools_used.extend(tool_loop.tools_used)
        current_messages.append(user_turn_message(tool_loop.tool_results_content))
        completion_contract_error = _agent_completion_contract_error(
          tool_uses=turn.tool_uses,
          tool_result_blocks=tool_loop.tool_results_content,
        )
        if completion_contract_error is not None:
          terminal_failure = (
            "agent_completion_contract_invalid",
            completion_contract_error,
          )
          break
        model_visible_tool_result_ids = {
          str(block["tool_use_id"])
          for block in tool_loop.tool_results_content
          if (
            isinstance(block, dict)
            and isinstance(block.get("tool_use_id"), str)
            and block["tool_use_id"]
          )
        }
        if model_visible_tool_result_ids:
          model_visible_tool_result_generation += 1
          pending_model_visible_tool_result_ids.update(
            model_visible_tool_result_ids
          )

        stop_after_tool_results_reason = getattr(self, "_stop_after_tool_results_reason", None)
        # A terminal tool settlement has already incurred its model-turn cost;
        # preserve that explicit settlement at this boundary. Other forced-stop
        # reasons still yield to the budget guard.
        if stop_after_tool_results_reason not in {
          "terminal_tool_result",
          "terminal_tool_failure",
          "child_report_accepted",
        }:
          if self._cost_accumulator is not None:
            exceeded_state = budget_exceeded_state(self._cost_accumulator) or exceeded_state
          if exceeded_state is not None:
            await emit_budget_exceeded_stop(exceeded_state)
            break

        if stop_after_tool_results_reason:
          stop_after_tool_results_tool_name = getattr(self, "_stop_after_tool_results_tool_name", None)
          logger.info(
            "[%s] Stop-after-tool-results requested after %s (%s); ending run without follow-up model turn",
            self._sid,
            stop_after_tool_results_tool_name or "tool result",
            stop_after_tool_results_reason,
          )
          if stop_after_tool_results_reason == "terminal_tool_failure":
            terminal_failure = (
              "terminal_tool_failure",
              "terminal_tool_failure: terminal tool returned an "
              "unrecoverable failure"
              + (
                " with status "
                f"{getattr(self, '_stop_after_tool_results_status', None)!r}"
                if getattr(
                  self,
                  "_stop_after_tool_results_status",
                  None,
                )
                is not None
                else ""
              )
              + ".",
            )
          elif stop_after_tool_results_reason in {
            "terminal_tool_result",
            "child_report_accepted",
            "accepted_ui_blocks",
          }:
            pending_model_visible_tool_result_ids.difference_update(
              model_visible_tool_result_ids
            )
            terminal_success_reason = (
              f"tool:{stop_after_tool_results_reason}"
            )
          else:
            terminal_failure = (
              str(stop_after_tool_results_reason),
              "terminal_outcome_unproven: stop-after-tool-results ended "
              "without an accepted terminal result "
              f"({stop_after_tool_results_reason}).",
            )
          break

        if turn.stop_reason == "end_turn":
          if (
            self._background_notifications_enabled
            and self._notification_queue.pending_count == 0
            and (
              self._task_registry.list_tasks(
                state=task_state_cls.RUNNING
              )
              # An unsettled workflow obligation holds the session open
              # (§5.3, T2-I04).
              or self._workflow_settlement_obstructed()
            )
            and (max_turns is None or turn_count < max_turns)
          ):
            await self._wait_for_background_notification()
            if self._cost_accumulator is not None:
              exceeded_state = (
                budget_exceeded_state(self._cost_accumulator)
                or exceeded_state
              )
            if exceeded_state is not None:
              await emit_budget_exceeded_stop(exceeded_state)
              break
            if self._operator_pause_requested():
              logger.info(
                "[%s] Operator pause requested while awaiting background completion",
                self._sid,
              )
              mark_terminal_interruption("operator_pause")
              await self._emit_operator_pause_event(
                "awaiting_background_completion"
              )
              break
          if self._notification_queue.pending_count > 0:
            current_messages.append(
              background_tasks_completed_user_message()
            )
            if max_turns is None:
              delivery_epoch_active = True
              delivery_turn_compelled = True
            continue
          if (
            getattr(self, "_pending_background_result_acks", {})
            or _pending_omitted_background_task_ids(self)
          ):
            if max_turns is None:
              delivery_epoch_active = True
              delivery_turn_compelled = True
            continue
          if pending_model_visible_tool_result_ids:
            continue
          unread_handle_entries = [
            entry
            for entry in _unread_settled_handle_entries(self)
            if entry.task_id not in unread_handle_reminded_task_ids
          ][:_UNREAD_HANDLE_NUDGE_MAX_TASKS]
          if (
            unread_handle_entries
            and not delivery_epoch_from_max
            and (max_turns is None or turn_count < max_turns)
          ):
            # CUR-E2E-08, tools-then-end_turn stop boundary: same one
            # reminder per task, same unmetered/epoch-free contract, same
            # turn-budget guard and epoch close-out as the no-tool arm (the
            # assistant turn is already in current_messages here).
            unread_handle_reminded_task_ids.update(
              entry.task_id for entry in unread_handle_entries
            )
            if (
              delivery_epoch_active
              and not delivery_epoch_from_max
              and not self._background_delivery_grace_obligations()
            ):
              delivery_epoch_active = False
            unread_handle_nudge_text = _unread_result_handle_nudge(
              unread_handle_entries
            )
            current_messages.append(
              user_turn_message(unread_handle_nudge_text)
            )
            try:
              await self._append_durable_event(
                build_runtime_guard_event(
                  guard="unread_result_handle_nudge",
                  message=unread_handle_nudge_text,
                )
              )
            except Exception as exc:
              logger.warning(
                "[%s] reminder record suppressed | exception_type=%s",
                self._sid,
                type(exc).__name__,
              )
            continue
          terminal_success_reason = "end_turn_after_tools"
          break

      self._background_delivery_grace_active = True
      if terminal_failure is not None:
        await emit_terminal_failure(*terminal_failure)
        return
      if (
        (terminal_success_reason is None)
        == (terminal_interruption_reason is None)
      ):
        await emit_terminal_failure(
          "terminal_outcome_unproven",
          "terminal_outcome_unproven: run loop reached terminal settlement "
          "without exactly one successful or interrupted disposition.",
        )
        return
      if (
        terminal_success_reason is not None
        and pending_model_visible_tool_result_ids
      ):
        await emit_terminal_failure(
          "model_visible_tool_results_uncommitted",
          "model_visible_tool_results_uncommitted: successful completion "
          "was refused before the provider committed the latest tool "
          "results.",
        )
        return
      terminal_snapshot_before: tuple[Any, ...] | None = None
      staged_terminal_events: list[dict[str, Any]] = []
      top_level_result_event: dict[str, Any] | None = None
      if terminal_success_reason is not None:
        (
          terminal_blockers,
          terminal_snapshot_before,
        ) = _background_success_snapshot(self)
        if terminal_blockers:
          await emit_terminal_failure(
            "background_delivery_incomplete",
            "background_delivery_incomplete: successful completion was "
            "refused while background work or result delivery remained "
            f"unsettled ({', '.join(terminal_blockers)}).",
          )
          return

      total_elapsed = time_module.time() - chat_t0
      cache_status = usage_cache_status(usage_totals)
      cost = self._estimate_usage_cost(upstream_model, usage_totals)
      completion_event = build_stream_complete_event(
        usage_totals=usage_totals,
        estimated_cost=cost.total,
      )
      terminal_event = completion_event
      if terminal_interruption_reason is not None:
        terminal_event["terminal_disposition"] = "interrupted"
        terminal_event["reason"] = terminal_interruption_reason
      if terminal_success_reason is not None:
        # Success hooks may derive durable receipts from the proposed terminal
        # event. Stage those receipts until the post-hook settlement snapshot
        # proves that the proposal is still current.
        self._terminal_success_staged_events = staged_terminal_events
      terminal_hook_error: Exception | None = None
      terminal_result_mismatch: str | None = None
      try:
        await self._call_on_before_stream_complete(terminal_event)
        if terminal_success_reason is not None:
          top_level_result_event = (
            await self._prepare_top_level_skill_result(
              terminal_event
            )
          )
          if (
            top_level_result_event is not None
            and (
              top_level_result_event.get("exit_code") != 0
              or top_level_result_event.get("outcome") != "success"
            )
          ):
            terminal_result_mismatch = (
              "top_level_skill_result_terminal_mismatch: successful "
              "terminal proposal produced a non-success canonical "
              "skill result "
              f"(exit_code={top_level_result_event.get('exit_code')!r}, "
              f"outcome={top_level_result_event.get('outcome')!r})."
            )
      except Exception as exc:
        terminal_hook_error = exc
      finally:
        if terminal_success_reason is not None:
          self._terminal_success_staged_events = None

      # Operator ruling 2026-08-03: session-log bookkeeping never vetoes
      # completed work. The durable record is a JSON log behind a lock; a
      # settlement-hook or projection problem is a diary problem — warn and
      # deliver the run's real result. (Previously these two branches
      # rewrote the terminal as a failure and discarded the staged events,
      # which converted the first-ever completed interactive pipeline drain
      # into a reported failure and destroyed the original record.)
      if terminal_hook_error is not None:
        log.warning(
          "terminal settlement hook failed (%s: %s) — proceeding with the "
          "run's real result; bookkeeping never vetoes completed work",
          type(terminal_hook_error).__name__,
          terminal_hook_error,
        )
      if terminal_result_mismatch is not None:
        log.warning(
          "%s — proceeding with the run's real result; bookkeeping never "
          "vetoes completed work",
          terminal_result_mismatch,
        )

      if terminal_success_reason is not None:
        # The hook above is the last yielding operation before committing
        # success. Require both quiescence and identity equality: a generation
        # that was published and drained during the hook is still a change.
        (
          terminal_blockers,
          terminal_snapshot_after,
        ) = _background_success_snapshot(self)
        if (
          terminal_blockers
          or terminal_snapshot_after != terminal_snapshot_before
        ):
          staged_terminal_events.clear()
          changed_suffix = (
            "state changed"
            if terminal_snapshot_after != terminal_snapshot_before
            else "state remained unsettled"
          )
          await emit_terminal_failure(
            "background_delivery_incomplete",
            "background_delivery_incomplete: successful completion was "
            "refused because background work or result delivery "
            f"{changed_suffix} at the terminal boundary"
            + (
              f" ({', '.join(terminal_blockers)})."
              if terminal_blockers
              else "."
            ),
          )
          return

        if (
          getattr(
            self,
            "_last_request_message_marker_position",
            None,
          )
          is not None
        ):
          self._post_turn_fork_handoff = None
          if not fork_credential_identity_available(self):
            if not getattr(
              self,
              "_post_turn_fork_identity_unavailable_logged",
              False,
            ):
              logger.info(
                "[%s] Post-turn fork capture unavailable: no visible "
                "credential identity (session_id=%s)",
                self._sid,
                self._full_session_id,
              )
              self._post_turn_fork_identity_unavailable_logged = True
          else:
            try:
              self._post_turn_fork_handoff = build_post_turn_handoff(
                self,
                current_messages,
                assistant_turn_message(
                  turn.content_blocks,
                  provider=self._provider.name,
                  model=upstream_model,
                  stop_reason=turn.stop_reason,
                ),
              )
            except Exception:
              logger.warning(
                "[%s] Post-turn fork capture failed | failure=true",
                self._sid,
              )

      if len(staged_terminal_events) > 1:
        staged_terminal_events.clear()
        await emit_terminal_failure(
          "terminal_receipt_cardinality_invalid",
          "terminal_receipt_cardinality_invalid: terminal settlement "
          "produced more than one staged success receipt.",
        )
        return
      if any(
        event.get("type") in {
          "skill_run_started",
          "skill_result_captured",
        }
        for event in staged_terminal_events
      ):
        staged_terminal_events.clear()
        await emit_terminal_failure(
          "terminal_receipt_ownership_invalid",
          "terminal_receipt_ownership_invalid: a generic terminal "
          "hook attempted to emit an AgentRunner-owned skill "
          "lifecycle marker.",
        )
        return
      for staged_event in staged_terminal_events:
        try:
          durable_entry = self._append_durable_event_sync(
            staged_event
          )
          if durable_entry is None:
            raise RuntimeError(
              "durable terminal receipt target is unavailable"
            )
        except Exception as exc:
          staged_terminal_events.clear()
          await emit_terminal_failure(
            "terminal_receipt_persistence_failed",
            "terminal_receipt_persistence_failed: successful completion "
            "was refused because its required receipt could not be "
            f"persisted ({type(exc).__name__}: {exc}).",
          )
          return
      for staged_event in staged_terminal_events:
        self._append(staged_event)
      if top_level_result_event is not None:
        lifecycle = self._top_level_skill_lifecycle
        effect = self._top_level_skill_completion_effect_plan
        if lifecycle is None or not isinstance(
          effect,
          TopLevelSkillCompletionEffectPlan,
        ):
          raise RuntimeError(
            "Top-level completion plan was not retained exactly"
          )
        bound_terminal = self._bind_top_level_terminal_identity(
          terminal_event,
          lifecycle=lifecycle,
        )
        self._defer_top_level_terminal_event(bound_terminal)
        self._commit_top_level_skill_plan_sync(
          lifecycle=lifecycle,
          result=top_level_result_event,
          terminal=bound_terminal,
          effect=effect,
          project_live=True,
        )

      if terminal_success_reason is not None:
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
      else:
        logger.info(
          "[%s] Chat interrupted | %.1fs total | %d turns | reason=%s",
          self._sid,
          total_elapsed,
          turn_count,
          terminal_interruption_reason,
        )
      if (
        top_level_result_event is None
        and not self._defer_prepared_top_level_terminal_event(
          terminal_event
        )
      ):
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
      finalizer_errors: list[BaseException] = []
      shutdown_failed = False
      try:
        from .skill_context import clear_current_skill

        clear_current_skill()
      except Exception:
        pass
      self._active_skill_allow.clear()
      self._active_skill_deny.clear()
      self._active_skill_report_doors.clear()
      try:
        await self._shutdown_background_tasks(was_cancelled)
      except BaseException as exc:
        finalizer_errors.append(exc)
        shutdown_failed = True
      finally:
        running_entries = self._task_registry.list_tasks(state=task_state_cls.RUNNING)
        drain_state = session_drain_state(
          running_entries,
          shutdown_failed=shutdown_failed,
          unsettled_workflow_obstructions=(
            self._unsettled_workflow_obstruction_count()
          ),
        )
        try:
          await self._finalize_top_level_skill_result(
            run_error=run_error,
            clean_detach_reason=clean_detach_reason,
            terminal_event=getattr(
              self,
              "_deferred_top_level_terminal_event",
              None,
            ),
          )
        except BaseException as exc:
          finalizer_errors.append(exc)
        try:
          await self._flush_deferred_top_level_terminal_event()
        except BaseException as exc:
          finalizer_errors.append(exc)
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
            logger.error(
              "[%s] session summary emission failed (non-fatal) | exception_type=%s",
              self._sid,
              type(exc).__name__,
            )
        else:
          self._summary_emitted = True
      try:
        if self._agent_session_log is not None and self._durable_attach_emitted:
          server_terminal_cause = (
            self._top_level_server_terminal_cause()
          )
          if run_error is not None:
            await self._emit_run_error_event(
              run_error,
              server_terminal_cause=server_terminal_cause,
            )
            if server_terminal_cause is not None:
              reason = server_terminal_cause
              extra_fields = {
                "server_terminal_cause": (
                  server_terminal_cause
                )
              }
            else:
              (
                shutdown_reason,
                shutdown_extra_fields,
              ) = self._shutdown_interrupted_reason()
              reason, extra_fields = run_interrupted_reason(
                run_error=run_error,
                role=self._role,
                shutdown_reason=shutdown_reason,
                shutdown_extra_fields=shutdown_extra_fields,
              )
            await self._emit_interrupted_event(
              reason,
              extra_fields=extra_fields or None,
            )
          if self._top_level_skill_lifecycle is not None:
            await (
              self._flush_deferred_top_level_interrupted_event()
            )
          await self._emit_detach_event(
            (
              server_terminal_cause
              if server_terminal_cause is not None
              else run_detach_reason(
                clean_detach_reason=clean_detach_reason,
                run_error=run_error,
              )
            )
          )
      except BaseException as exc:
        finalizer_errors.append(exc)
      try:
        if self._top_level_skill_lifecycle is not None:
          await self._await_write_lease_handoff()
        else:
          self._release_write_lease()
      except BaseException as exc:
        finalizer_errors.append(exc)
      try:
        await self.force_close()
      except BaseException as exc:
        finalizer_errors.append(exc)
      self._background_delivery_grace_active = False
      self._background_ack_recovery_pairs.clear()
      if self._top_level_skill_lifecycle is not None:
        settlement_error = (
          finalizer_errors[0]
          if finalizer_errors
          else (
            run_error
            if not self._top_level_skill_started_committed
            else None
          )
        )
        try:
          self._publish_top_level_skill_settlement(
            settlement_error
          )
        except BaseException as exc:
          finalizer_errors.append(exc)
      learning_turn_success = (
        run_error is None
        and not finalizer_errors
        and terminal_success_reason is not None
      )
      settle_learning_receipts(
        self,
        learning_receipt_delivery,
        success=learning_turn_success,
      )
      if learning_turn_success:
        submit_learning_fork_after_turn(
          self,
          handoff=getattr(self, "_post_turn_fork_handoff", None),
          tool_calling_iters=learning_tool_calling_iters,
          foreground_memory_write=successful_memory_write_from_events(
            getattr(self._log, "entries", ()),
            fork=False,
          ),
          completed=True,
          real_final_response=learning_real_final_response,
          errored=False,
          aborted=False,
          cancelled=False,
        )
      if run_error is not None:
        if finalizer_errors:
          for finalizer_error in finalizer_errors:
            cleanup_detail = attach_cleanup_failure(
              run_error,
              finalizer_error,
            )
            try:
              self._append({
                "type": "run_error",
                "phase": "run_finalizer",
                "error_type": type(finalizer_error).__name__,
                "error": cleanup_detail,
                "message": cleanup_detail,
              })
            except Exception:
              pass
          raise run_error from finalizer_errors[0]
        raise run_error
      if finalizer_errors:
        primary_finalizer_error = finalizer_errors[0]
        for finalizer_error in finalizer_errors[1:]:
          attach_cleanup_failure(
            primary_finalizer_error,
            finalizer_error,
          )
        raise primary_finalizer_error
