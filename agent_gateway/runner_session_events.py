from __future__ import annotations

from dataclasses import dataclass
import asyncio
from collections.abc import Mapping
from copy import deepcopy
import json
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal

from agent_workflow_contracts import AgentCompletionEnvelope

from .events import AgentCompletionEvent, event_to_dict
from .event_adapter import TerminalDisposition
from .skill_completion_wal import TopLevelSkillCompletionEffectPlan
from .skill_lifecycle import TopLevelSkillLifecycleMetadata
from .tool_dispatch_classification import OUTCOME_OK
from .workflow_output_attachment import WorkflowOutputAttachment

if TYPE_CHECKING:
  from .tool_dispatcher_approval_lifecycle import ApprovalPark


def durable_event_payload(
  event: Dict[str, Any],
  *,
  runner_id: str,
  role: str,
  sub_agent_id: str | None,
  product_id: str | None,
) -> Dict[str, Any]:
  payload = dict(event)
  payload.setdefault("runner_id", runner_id)
  payload.setdefault("role", role)
  if sub_agent_id is not None:
    payload.setdefault("sub_agent_id", sub_agent_id)
  if product_id is not None:
    payload["product_id"] = product_id
  return payload


def build_attach_event(
  *,
  gateway_session_id: str,
  started_at: float,
  client_kind: str,
  hostname: str,
  user_id: str | None,
  channel: str | None,
  api_key_hash: str | None,
) -> Dict[str, Any]:
  """Build the attach event; its owner fields are the log's ownership record.

  `user_id`, `channel` and `api_key_hash` come from the writer's gateway
  session. Init-time session resume admits a log only when the first attach
  carries all three and they equal the resolving caller.
  """

  return {
    "type": "attach",
    "gateway_session_id": gateway_session_id,
    "started_at": started_at,
    "client_kind": client_kind,
    "hostname": hostname,
    "user_id": user_id,
    "channel": channel,
    "api_key_hash": api_key_hash,
  }


def build_write_lease_metadata(
  *,
  runner_id: str,
  gateway_session_id: str,
  started_at: float,
  hostname: str,
) -> Dict[str, Any]:
  return {
    "runner_id": runner_id,
    "gateway_session_id": gateway_session_id,
    "started_at": started_at,
    "hostname": hostname,
  }


def write_lease_metadata(
  agent_session_log: Any,
  *,
  role: str,
  runner_id: str | None,
  gateway_session_id: str,
  started_at: float,
  hostname: str,
) -> bool:
  if agent_session_log is None or role != "writer" or runner_id is None:
    return False
  payload = build_write_lease_metadata(
    runner_id=runner_id,
    gateway_session_id=gateway_session_id,
    started_at=started_at,
    hostname=hostname,
  )
  agent_session_log.write_lease_meta_path.write_text(
    json.dumps(payload, sort_keys=True),
    encoding="utf-8",
  )
  return True


def release_write_lease(write_lease_file: Any, *, clear_write_lease_file: Callable[[], None]) -> bool:
  if write_lease_file is None:
    return False
  try:
    write_lease_file.close()
  finally:
    clear_write_lease_file()
  return True


def build_user_message_event(
  *,
  content: Any,
  client_kind: str,
  received_at: float,
  selected_content: tuple[dict[str, object], ...],
  operator_messages: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> Dict[str, Any]:
  payload: Dict[str, Any] = {
    "type": "user_message",
    "content": content,
    "client_kind": client_kind,
    "received_at": received_at,
    "selected_content": list(selected_content),
  }
  if operator_messages is not None:
    payload["operator_messages"] = list(operator_messages)
  return payload


def build_agent_completion_event(
  *,
  task_id: str,
  envelope: AgentCompletionEnvelope,
  ts: float,
) -> Dict[str, Any]:
  """Build one durable, typed direct-parent result publication."""

  return event_to_dict(AgentCompletionEvent(
    task_id=task_id,
    envelope=envelope,
    ts=float(ts),
  ))


def build_skill_run_started_event(
  lifecycle: TopLevelSkillLifecycleMetadata,
  *,
  started_at: float,
) -> Dict[str, Any]:
  return {
    "type": "skill_run_started",
    **lifecycle.identity_fields(),
    "ts": started_at,
  }


def build_skill_result_failure_event(
  lifecycle: TopLevelSkillLifecycleMetadata,
  *,
  error: str,
) -> Dict[str, Any]:
  return {
    "type": "skill_result_captured",
    **lifecycle.identity_fields(),
    "exit_code": 1,
    "outcome": "error",
    "status": "error",
    "gate_code": None,
    "artifact_refs": [],
    "proposal_ids": [],
    "verdict_echo": None,
    "fms_results": [],
    "last_failed_door_answer": None,
    "artifact_events": [],
    "output_memory_file": None,
    "cost_usd": None,
    "duration_s": None,
    "compaction_count": 0,
    "error": error,
    "warnings": [],
    "approval_outcome": None,
    "approval_id": None,
    "approval_tool_name": None,
  }


def build_assistant_message_event(
  *,
  content_blocks: List[Dict[str, Any]],
  stop_reason: str | None,
  stop_details: Dict[str, Any] | None = None,
  model: str | None,
  provider: str | None,
  usage: Dict[str, Any],
  parent_message_consumptions: List[Dict[str, Any]] | None = None,
  logical_response_id: str | None = None,
  logical_response_segment_ordinal: int | None = None,
  continued_from_assistant_message_seq: int | None = None,
  workflow_output_attachments: Iterable[
    WorkflowOutputAttachment
  ] | None = None,
) -> Dict[str, Any]:
  event = {
    "type": "assistant_message",
    "content_blocks": list(content_blocks),
    "stop_reason": stop_reason,
    "model": model,
    "provider": provider,
    "usage": dict(usage),
  }
  # The provider's reason for the stop (a refusal's category/explanation),
  # absent when the provider sent none.
  if stop_details is not None:
    event["stop_details"] = dict(stop_details)
  attachments = [
    attachment.to_dict()
    for attachment in (workflow_output_attachments or ())
  ]
  if attachments:
    event["workflow_output_attachments"] = attachments
  if parent_message_consumptions:
    event["parent_message_consumptions"] = [
      dict(binding) for binding in parent_message_consumptions
    ]
  if logical_response_id is not None:
    normalized_response_id = str(logical_response_id).strip()
    if not normalized_response_id:
      raise ValueError("logical_response_id must be non-empty")
    if (
      type(logical_response_segment_ordinal) is not int
      or logical_response_segment_ordinal < 0
    ):
      raise ValueError(
        "logical_response_segment_ordinal must be a non-negative integer"
      )
    if logical_response_segment_ordinal == 0:
      if continued_from_assistant_message_seq is not None:
        raise ValueError(
          "the first logical response segment cannot continue another "
          "assistant message"
        )
    elif (
      type(continued_from_assistant_message_seq) is not int
      or continued_from_assistant_message_seq <= 0
    ):
      raise ValueError(
        "continued_from_assistant_message_seq must identify the prior "
        "durable assistant message"
      )
    event.update({
      "logical_response_id": normalized_response_id,
      "logical_response_segment_ordinal": (
        logical_response_segment_ordinal
      ),
    })
    if continued_from_assistant_message_seq is not None:
      event["continued_from_assistant_message_seq"] = (
        continued_from_assistant_message_seq
      )
  elif (
    logical_response_segment_ordinal is not None
    or continued_from_assistant_message_seq is not None
  ):
    raise ValueError(
      "logical response segment metadata requires logical_response_id"
    )
  return event


def build_workflow_output_attached_event(
  *,
  attachment: WorkflowOutputAttachment,
  assistant_message_seq: int | None,
) -> Dict[str, Any]:
  """Build the reader-visible projection of one final-message attachment."""

  return {
    "type": "workflow_output_attached",
    "assistant_message_seq": assistant_message_seq,
    **attachment.to_dict(),
  }


def build_tool_call_start_event(
  *,
  tool_call_id: str,
  tool_name: str,
  tool_input: Dict[str, Any],
  call_index: int,
  server: str | None,
  started_at: float,
  parent_assistant_message_seq: int | None,
) -> Dict[str, Any]:
  return {
    "type": "tool_call_start",
    "tool_call_id": tool_call_id,
    "tool_name": tool_name,
    "tool_input": tool_input,
    "execution_location": "backend",
    "call_index": call_index,
    "server": server,
    "started_at": started_at,
    "parent_assistant_message_seq": parent_assistant_message_seq,
  }


def build_tool_call_complete_event(
  *,
  tool_call_id: str,
  tool_name: str,
  result: Any,
  error: Dict[str, Any] | None,
  duration_ms: int,
  server: str | None,
  dispatch: Mapping[str, Any],
  semantic_error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
  """Build the settled tool event.

  ``dispatch`` is required and emitted unconditionally: every tool call in
  every runtime settles exactly one dispatch record (outcome, attempts,
  route_id, plural sources).  Both producers — this choke point and the SDK
  runner stream — go through here so no runtime silently degrades the
  evidence fold to the block-reading fallback (D-B1-1).
  """

  settled_dispatch = _normalized_dispatch_record(dispatch)
  payload: Dict[str, Any] = {
    "type": "tool_call_complete",
    "tool_call_id": tool_call_id,
    "tool_name": tool_name,
    "result": dict(result) if isinstance(result, dict) else result,
    "error": error,
    "duration_ms": duration_ms,
    "server": server,
    "is_error": settled_dispatch["outcome"] != OUTCOME_OK,
  }
  if semantic_error is not None:
    payload["semantic_error"] = dict(semantic_error)
  payload["dispatch"] = settled_dispatch
  return payload


def _normalized_dispatch_record(dispatch: Mapping[str, Any]) -> Dict[str, Any]:
  if not isinstance(dispatch, Mapping):
    raise TypeError("tool_call_complete requires a dispatch record mapping")
  record = dict(dispatch)
  raw_sources = record.get("sources")
  record["sources"] = [
    dict(source)
    for source in (raw_sources if isinstance(raw_sources, (list, tuple)) else ())
    if isinstance(source, Mapping)
  ]
  return record


def build_turn_complete_event(*, turn: int, usage: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "type": "turn_complete",
    "turn": turn,
    "usage": dict(usage),
  }


def build_tool_calls_planned_event(*, turn: int, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
  return {
    "type": "tool_calls_planned",
    "turn": turn,
    "calls": [dict(call) for call in calls],
  }


def _soft_budget_event(
  event_type: str,
  *,
  kind: str,
  observed: float,
  threshold: float,
) -> Dict[str, Any]:
  return {
    "type": event_type,
    "kind": kind,
    "observed": observed,
    "threshold": threshold,
  }


def build_soft_budget_steer_event(
  kind: str,
  observed: float,
  threshold: float,
) -> Dict[str, Any]:
  """The soft-budget ladder steered the run toward wrapping up."""

  return _soft_budget_event(
    "soft_budget_steer",
    kind=kind,
    observed=observed,
    threshold=threshold,
  )


def build_soft_budget_final_event(
  kind: str,
  observed: float,
  threshold: float,
) -> Dict[str, Any]:
  """The soft-budget ladder opened the forced-final report turn."""

  return _soft_budget_event(
    "soft_budget_final",
    kind=kind,
    observed=observed,
    threshold=threshold,
  )


def build_runtime_guard_event(*, guard: str, message: str) -> Dict[str, Any]:
  return {
    "type": "runtime_guard",
    "guard": guard,
    "message": message,
  }


def build_todo_snapshot_event(
  checklist: Any,
  *,
  actor: Literal["agent", "analyst"],
  op: str,
  gateway_session_id: str,
) -> Dict[str, Any]:
  """One accepted checklist change; the fold of these is the checklist (§5)."""
  from .todo_checklist import build_todo_snapshot_payload

  return build_todo_snapshot_payload(
    checklist, actor=actor, op=op, gateway_session_id=gateway_session_id
  )



REFUSAL_GUIDANCE = (
  "Anthropic recommends starting a fresh conversation before continuing; "
  "further turns in this context may be refused again."
)


def build_refusal_event(stop_details: Dict[str, Any] | None) -> Dict[str, Any]:
  """The client-visible notice for a turn the provider ended with ``refusal``.

  ``category`` and ``explanation`` are the provider's ``stop_details`` verbatim
  (``None`` when absent); the delivered text stays the turn's output.
  """
  details = stop_details if isinstance(stop_details, dict) else {}
  return {
    "type": "refusal",
    "category": details.get("category"),
    "explanation": details.get("explanation"),
    "guidance": REFUSAL_GUIDANCE,
  }


def build_stream_complete_event(
  *,
  usage_totals: Dict[str, int],
  estimated_cost: float,
  est_system_tokens: int,
  est_tools_tokens: int,
  todo_measurements: Dict[str, int] | None = None,
  terminal_disposition: TerminalDisposition = "completed",
) -> Dict[str, Any]:
  payload: Dict[str, Any] = {
    "type": "stream_complete",
    "terminal_disposition": terminal_disposition,
    "usage": {
      "input_tokens": usage_totals["input_tokens"],
      "output_tokens": usage_totals["output_tokens"],
      "cache_creation_input_tokens": usage_totals["cache_creation_input_tokens"],
      "cache_read_input_tokens": usage_totals["cache_read_input_tokens"],
      "reasoning_tokens_observed": usage_totals.get("reasoning_tokens_observed", 0),
      "provider_wait_ms": usage_totals.get("provider_wait_ms", 0),
      "thinking_chars": usage_totals.get("thinking_chars", 0),
      "estimated_cost": round(estimated_cost, 4),
      "est_system_tokens": est_system_tokens,
      "est_tools_tokens": est_tools_tokens,
    },
  }
  if todo_measurements is not None:
    payload.update(todo_measurements)
  return payload


def build_context_warning_log_data(
  *,
  session_id: str,
  est_tokens: int,
  context_limit: int,
  turn: int | None = None,
) -> Dict[str, Any]:
  payload: Dict[str, Any] = {
    "event": "context_warning",
    "session_id": session_id,
  }
  if turn is not None:
    payload["turn"] = turn
  payload.update({
    "est_tokens": est_tokens,
    "limit": context_limit,
    "pct": round(est_tokens / context_limit * 100, 1),
  })
  return payload


def build_context_pressure_reminder(*, pct: int) -> str:
  return (
    f"Context at {max(0, int(pct))}% — prefer delegating further reading; "
    "large results will spill."
  )


def build_token_estimate_log_data(
  *,
  session_id: str,
  est_system_tokens: int,
  est_messages_tokens: int,
  est_tools_tokens: int,
  est_total_tokens: int,
  message_count: int,
  tool_count: int,
  turn: int | None = None,
) -> Dict[str, Any]:
  payload: Dict[str, Any] = {
    "event": "token_estimate",
    "session_id": session_id,
  }
  if turn is not None:
    payload["turn"] = turn
  payload.update({
    "est_system_tokens": est_system_tokens,
    "est_messages_tokens": est_messages_tokens,
    "est_tools_tokens": est_tools_tokens,
    "est_total_tokens": est_total_tokens,
    "message_count": message_count,
    "tool_count": tool_count,
  })
  return payload


def build_turn_complete_log_data(
  *,
  session_id: str,
  turn: int,
  elapsed_s: float,
  ttft_s: float | None,
  text_chars: int,
  tools: Iterable[str],
  stop_reason: str | None,
) -> Dict[str, Any]:
  return {
    "event": "turn_complete",
    "session_id": session_id,
    "turn": turn,
    "elapsed_s": round(elapsed_s, 1),
    "ttft_s": round(ttft_s, 2) if ttft_s is not None else None,
    "text_chars": text_chars,
    "tools": list(tools),
    "stop_reason": stop_reason,
  }


def build_chat_done_log_data(
  *,
  session_id: str,
  elapsed_s: float,
  turns: int,
  tools: Iterable[str],
  usage_totals: Dict[str, int],
  cost: float,
) -> Dict[str, Any]:
  return {
    "event": "chat_done",
    "session_id": session_id,
    "elapsed_s": round(elapsed_s, 1),
    "turns": turns,
    "tools": list(tools),
    "tokens_in": usage_totals["input_tokens"],
    "tokens_out": usage_totals["output_tokens"],
    "cache_read": usage_totals["cache_read_input_tokens"],
    "cache_write": usage_totals["cache_creation_input_tokens"],
    "cost": round(cost, 4),
  }


_PROVIDER_ERROR_PROJECTION_FIELDS = (
  "status_code",
  "provider_request_id",
  "rate_limit_representative_claim",
  "rate_limit_5h_status",
  "rate_limit_5h_utilization",
  "rate_limit_5h_reset",
  "rate_limit_7d_status",
  "rate_limit_7d_utilization",
  "rate_limit_7d_reset",
  "rate_limit_unified_status",
  "rate_limit_unified_reset",
  "retry_after",
)


class _ErrorWithProviderProjection(str):
  _provider_error_projection: dict[str, str | int | float]


def error_with_provider_projection(error: str, exc: BaseException) -> str:
  projection = {
    name: value
    for name in _PROVIDER_ERROR_PROJECTION_FIELDS
    if (value := getattr(exc, name, None)) is not None
    and not isinstance(value, bool)
    and isinstance(value, (str, int, float))
  }
  if not projection:
    return error
  projected_error = _ErrorWithProviderProjection(error)
  projected_error._provider_error_projection = projection
  return projected_error


def _provider_error_event_fields(error: str) -> Dict[str, Any]:
  projection = getattr(error, "_provider_error_projection", None)
  return dict(projection) if isinstance(projection, Mapping) else {}


def build_stream_retry_event(
  *,
  attempt: int,
  error: str,
  attempt_elapsed_s: float,
  backoff_s: float | None,
) -> Dict[str, Any]:
  """``attempt_elapsed_s``: monotonic seconds from the failed attempt's start
  (after its own backoff sleep) to this retry. ``backoff_s``: the delay the
  next attempt sleeps first; None when no attempt follows."""
  return {
    "type": "stream_retry",
    "attempt": attempt,
    "error": str(error),
    "attempt_elapsed_s": attempt_elapsed_s,
    "backoff_s": backoff_s,
    **_provider_error_event_fields(error),
  }


def build_error_event(error: str) -> Dict[str, Any]:
  return {
    "type": "error",
    "error": str(error),
    **_provider_error_event_fields(error),
  }


def build_run_error_event(
  *,
  phase: str,
  error_type: str,
  error: str,
) -> Dict[str, Any]:
  return {
    "type": "run_error",
    "phase": phase,
    "error_type": error_type,
    "error": error,
  }


def build_interrupted_event(
  *,
  reason: str,
  runner_id: str | None,
  role: str,
  last_completed_seq: int,
  recovered_by_runner_id: str | None = None,
  recovered_at: float | None = None,
  extra_fields: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
  payload: Dict[str, Any] = {
    "type": "interrupted",
    "reason": reason,
    "runner_id": runner_id,
    "role": role,
    "last_completed_seq": last_completed_seq,
  }
  if recovered_by_runner_id is not None:
    payload["recovered_by_runner_id"] = recovered_by_runner_id
  if recovered_at is not None:
    payload["recovered_at"] = recovered_at
  if extra_fields:
    payload.update(extra_fields)
  return payload


def build_orphan_tool_call_interrupted_events(
  orphan_entries: Iterable[Any],
  *,
  discovered_at: float,
  tool_risk_for_tool: Callable[[str], str],
) -> List[Dict[str, Any]]:
  starts: Dict[str, Dict[str, Any]] = {}
  resolved_tool_ids: set[str] = set()
  for entry in orphan_entries:
    event = entry.event
    event_type = str(event.get("type") or "")
    if event_type == "assistant_message":
      for block in event.get("content_blocks") or []:
        if block.get("type") == "tool_use" and block.get("id"):
          starts.setdefault(str(block["id"]), {
            "tool_name": block.get("name"),
            "tool_input": block.get("input"),
            "runner_id": event.get("runner_id"),
            "role": event.get("role", "writer"),
            **({"sub_agent_id": event["sub_agent_id"]} if "sub_agent_id" in event else {}),
          })
      continue
    tool_call_id = str(event.get("tool_call_id") or "")
    if not tool_call_id:
      continue
    if event_type == "tool_call_start":
      if tool_call_id not in starts or "started_at" not in starts[tool_call_id]:
        starts[tool_call_id] = event
    elif event_type in {"tool_call_complete", "tool_call_interrupted"}:
      resolved_tool_ids.add(tool_call_id)

  synthetic_events: List[Dict[str, Any]] = []
  for tool_call_id, start_event in starts.items():
    if tool_call_id in resolved_tool_ids:
      continue
    synthetic_event: Dict[str, Any] = {
      "type": "tool_call_interrupted",
      "tool_call_id": tool_call_id,
      "tool_name": start_event.get("tool_name"),
      "tool_input": start_event.get("tool_input"),
      "original_started_at": start_event.get("started_at"),
      "discovered_at": discovered_at,
      "tool_risk": tool_risk_for_tool(str(start_event.get("tool_name") or "")),
      "runner_id": start_event.get("runner_id"),
      "role": start_event.get("role", "writer"),
      "final_tool_result_blocks": [{
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "is_error": True,
        "content": json.dumps({
          "error": {
            "code": "tool_interrupted",
            "message": "The run ended without a tool result; execution may not have completed.",
          },
        }),
      }],
    }
    if start_event.get("sub_agent_id") is not None:
      synthetic_event["sub_agent_id"] = start_event.get("sub_agent_id")
    synthetic_events.append(synthetic_event)
  return synthetic_events


def shutdown_interrupted_reason(signal_payload: Any) -> tuple[str, Dict[str, Any]]:
  if not isinstance(signal_payload, dict) or not signal_payload:
    return "graceful_shutdown", {}

  payload = dict(signal_payload)
  signal_name = str(payload.get("signal_name") or "").strip()
  signal_number = payload.get("signal")
  suffix = signal_name
  if not suffix and signal_number is not None:
    suffix = f"SIG{signal_number}"
  if not suffix:
    suffix = "unknown"
  return f"signal_{suffix}", {"shutdown": payload}


def build_detach_event(*, reason: str, ended_at: float) -> Dict[str, Any]:
  return {
    "type": "detach",
    "reason": reason,
    "ended_at": ended_at,
  }


@dataclass(frozen=True)
class NamedSkillClosure:
  result: Dict[str, Any]
  terminal: Dict[str, Any]
  effect: TopLevelSkillCompletionEffectPlan


@dataclass(frozen=True)
class TerminalClosureDecision:
  disposition: Literal["success", "error", "cancelled", "awaiting_approval"]
  reason: str
  cause: BaseException | None = None
  terminal: Dict[str, Any] | None = None
  skill_result: Dict[str, Any] | None = None
  skill_effect: TopLevelSkillCompletionEffectPlan | None = None
  interruption: tuple[str, Dict[str, Any] | None] | None = None

  def require_named_skill_closure(self) -> NamedSkillClosure:
    """Return the canonical named-skill pair and its completion effect."""
    if (
      self.skill_result is None
      or self.terminal is None
      or self.skill_effect is None
    ):
      raise RuntimeError(
        "Named-skill closure requires a prepared result, terminal, and effect plan"
      )
    return NamedSkillClosure(self.skill_result, self.terminal, self.skill_effect)


def run_terminal_for_disposition(
  terminal_event: Dict[str, Any] | None,
  *,
  disposition: str,
  reason: str,
  cause: BaseException | None,
  server_terminal_cause: str | None,
) -> Dict[str, Any] | None:
  """Project the run's own terminal onto a decided disposition.

  Never synthesizes a terminal: a run that produced none keeps ``None``, so
  the owner that closes it (a named skill's closure) builds its own.
  """
  if (
    terminal_event is None
    or terminal_event.get("type") != "stream_complete"
    or terminal_event.get("terminal_disposition") == "interrupted"
    or disposition in {"success", "awaiting_approval"}
  ):
    return terminal_event
  terminal = (
    {**terminal_event, "terminal_disposition": "interrupted", "reason": reason}
    if disposition == "cancelled"
    else build_error_event(str(cause or reason))
  )
  if server_terminal_cause is not None:
    terminal["server_terminal_cause"] = server_terminal_cause
  return terminal


def terminal_closure_decision(
  *,
  clean_detach_reason: str,
  run_error: BaseException | None,
  terminal_event: Dict[str, Any] | None,
  server_terminal_cause: str | None,
  persistence_error: Exception | None = None,
  persistence_failure_code: str = "terminal_persistence_failed",
  skill_lifecycle: TopLevelSkillLifecycleMetadata | None = None,
  skill_result: Dict[str, Any] | None = None,
  skill_effect: TopLevelSkillCompletionEffectPlan | None = None,
  approval_park: ApprovalPark | None = None,
  role: str = "",
  shutdown_reason: str | None = None,
  shutdown_extra_fields: Dict[str, Any] | None = None,
  staged_interruption: tuple[str, Dict[str, Any] | None] | None = None,
) -> TerminalClosureDecision:
  """Decide the canonical closure, including an unprepared escaping exception."""
  cause = run_error
  disposition: Literal["success", "error", "cancelled", "awaiting_approval"]
  if persistence_error is not None:
    disposition, reason, cause = "error", "persistence", persistence_error
  elif isinstance(run_error, asyncio.CancelledError):
    disposition, reason = "cancelled", server_terminal_cause or "cancelled"
  elif run_error is not None:
    disposition, reason = "error", "error"
  elif terminal_event is not None and terminal_event.get("type") == "error":
    disposition, reason = "error", "error"
    cause = RuntimeError(str(terminal_event.get("error", "")))
  elif clean_detach_reason == "error":
    disposition, reason = "error", "error"
  elif server_terminal_cause is not None:
    disposition, reason = "cancelled", server_terminal_cause
  elif approval_park is not None and clean_detach_reason == "awaiting_approval":
    disposition, reason = "awaiting_approval", "awaiting_approval"
  elif clean_detach_reason != "completed":
    disposition, reason = "cancelled", clean_detach_reason
  else:
    disposition, reason = "success", "completed"

  terminal = terminal_event
  if persistence_error is not None:
    terminal = {
      **build_error_event(
        f"{persistence_failure_code}: terminal persistence failed "
        f"({type(persistence_error).__name__}: {persistence_error})."
      ),
      "reason": reason,
    }
    if server_terminal_cause is not None:
      terminal["server_terminal_cause"] = server_terminal_cause
  elif terminal is None and isinstance(run_error, Exception):
    terminal = build_error_event(str(run_error))
  else:
    terminal = run_terminal_for_disposition(
      terminal,
      disposition=disposition,
      reason=reason,
      cause=cause,
      server_terminal_cause=server_terminal_cause,
    )

  if disposition == "awaiting_approval":
    skill_result = None
    skill_effect = None
    if terminal is not None and approval_park is not None:
      terminal = {
        **terminal,
        "terminal_disposition": "awaiting_approval",
        "job_id": approval_park.job_id,
        "approval_id": approval_park.approval_id,
        "original_tool_call_id": approval_park.original_tool_call_id,
      }
      if approval_park.root_turn_id is not None:
        terminal["root_turn_id"] = approval_park.root_turn_id
  elif skill_lifecycle is not None:
    if (
      disposition == "success"
      and terminal is not None
      and terminal.get("type") == "stream_complete"
      and skill_result is not None
      and (
        skill_result.get("exit_code", 0) != 0
        or skill_result.get("outcome") in {"error", "blocked"}
      )
    ):
      # The stream completed, but the named skill's own result failed: the
      # run's terminal states the skill's verdict, never a bare ``completed``.
      terminal = {
        **build_error_event(
          str(skill_result.get("error") or skill_result.get("outcome") or "error")
        ),
        "reason": str(skill_result.get("outcome") or "error"),
      }
    if persistence_error is not None or (
      disposition != "success"
      and skill_result is not None
      and skill_result.get("outcome") == "success"
    ):
      error = str(
        (terminal.get("error") or terminal.get("reason") if terminal is not None else None)
        or reason
      )
      if skill_result is None:
        skill_result = build_skill_result_failure_event(skill_lifecycle, error=error)
      else:
        skill_result = {
          **deepcopy(skill_result),
          "exit_code": 1, "outcome": "error", "status": "error", "error": error,
        }
      skill_effect = TopLevelSkillCompletionEffectPlan.noop()
    if terminal is not None:
      terminal = {**terminal, **skill_lifecycle.identity_fields()}
  if persistence_error is not None:
    interruption: tuple[str, Dict[str, Any] | None] | None = (
      reason,
      staged_interruption[1] if staged_interruption is not None else None,
    )
  else:
    built = staged_interruption
    if run_error is not None and shutdown_reason is not None:
      built = (
        (
          server_terminal_cause,
          {"server_terminal_cause": server_terminal_cause},
        )
        if server_terminal_cause is not None
        else run_interrupted_reason(
          run_error=run_error,
          role=role,
          shutdown_reason=shutdown_reason,
          shutdown_extra_fields=shutdown_extra_fields or {},
        )
      )
    elif disposition == "error" and built is None:
      built = (reason, None)
    if built is None:
      interruption = None
    elif disposition == "error":
      interruption = (reason, built[1])
    else:
      interruption = (built[0], built[1])
  return TerminalClosureDecision(
    disposition, reason, cause, terminal, skill_result, skill_effect,
    interruption,
  )




def run_interrupted_reason(
  *,
  run_error: BaseException,
  role: str,
  shutdown_reason: str,
  shutdown_extra_fields: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
  if isinstance(run_error, asyncio.CancelledError) and role == "sub_agent":
    return "sub_agent_cancelled", {}
  return shutdown_reason, shutdown_extra_fields


def build_operator_pause_event(safe_boundary: str) -> Dict[str, Any]:
  return {
    "type": "operator_pause",
    "reason": "operator_pause",
    "safe_boundary": safe_boundary,
  }


def build_stub_response_events(
  messages: List[Dict[str, Any]],
  *,
  provider_name: str,
) -> List[Dict[str, Any]]:
  last_user = next((msg for msg in reversed(messages) if msg.get("role") == "user"), {})
  prompt = last_user.get("content") or "your request"
  response = f"Stub response (no {provider_name.title()} credential configured). You asked: {prompt}"
  events: List[Dict[str, Any]] = [
    {"type": "text_delta", "text": token + " "}
    for token in response.split()
  ]
  events.append({
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {},
  })
  return events
