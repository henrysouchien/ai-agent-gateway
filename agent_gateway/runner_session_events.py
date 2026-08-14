from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, Iterable, List

from agent_workflow_contracts import AgentCompletionEnvelope

from .events import AgentCompletionEvent, event_to_dict
from .skill_lifecycle import TopLevelSkillLifecycleMetadata
from .workflow_output_attachment import WorkflowOutputAttachment


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
) -> Dict[str, Any]:
  return {
    "type": "attach",
    "gateway_session_id": gateway_session_id,
    "started_at": started_at,
    "client_kind": client_kind,
    "hostname": hostname,
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
) -> Dict[str, Any]:
  return {
    "type": "user_message",
    "content": content,
    "client_kind": client_kind,
    "received_at": received_at,
  }


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
  model: str,
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
  semantic_error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
  payload: Dict[str, Any] = {
    "type": "tool_call_complete",
    "tool_call_id": tool_call_id,
    "tool_name": tool_name,
    "result": dict(result) if isinstance(result, dict) else result,
    "error": error,
    "duration_ms": duration_ms,
    "server": server,
    "is_error": error is not None or semantic_error is not None,
  }
  if semantic_error is not None:
    payload["semantic_error"] = dict(semantic_error)
  return payload


def build_turn_complete_event(*, turn: int, usage: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "type": "turn_complete",
    "turn": turn,
    "usage": dict(usage),
  }


def build_max_turns_reached_event(*, turn_count: int, max_turns: int) -> Dict[str, Any]:
  return {"type": "max_turns_reached", "turn_count": turn_count, "max_turns": max_turns}


def build_max_turns_text_event(summary_text: str | None) -> Dict[str, Any]:
  if summary_text:
    return {"type": "text_delta", "text": f"\n\n[Max turns reached]\n{summary_text}"}
  return {"type": "text_delta", "text": "\n\n[Sub-agent reached maximum turn limit]"}


def build_runtime_guard_event(*, guard: str, message: str) -> Dict[str, Any]:
  return {
    "type": "runtime_guard",
    "guard": guard,
    "message": message,
  }


def build_budget_exceeded_event(
  *,
  total_cost: float,
  budget: float,
  reason: Any,
) -> Dict[str, Any]:
  return {
    "type": "budget_exceeded",
    "total_cost": round(total_cost, 4),
    "budget": budget,
    "reason": reason,
  }


def build_budget_exceeded_text_event(
  *,
  total_cost: float,
  budget: float,
  reason_suffix: str,
) -> Dict[str, Any]:
  return {
    "type": "text_delta",
    "text": (
      "\n\n"
      f"[Budget limit reached: ${total_cost:.4f} >= "
      f"${budget:.4f}{reason_suffix}]"
    ),
  }


def build_stream_complete_event(
  *,
  usage_totals: Dict[str, int],
  estimated_cost: float,
) -> Dict[str, Any]:
  return {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {
      "input_tokens": usage_totals["input_tokens"],
      "output_tokens": usage_totals["output_tokens"],
      "cache_creation_input_tokens": usage_totals["cache_creation_input_tokens"],
      "cache_read_input_tokens": usage_totals["cache_read_input_tokens"],
      "estimated_cost": round(estimated_cost, 4),
    },
  }


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


def build_stream_retry_event(*, attempt: int, error: str) -> Dict[str, Any]:
  return {"type": "stream_retry", "attempt": attempt, "error": error}


def build_error_event(error: str) -> Dict[str, Any]:
  return {"type": "error", "error": error}


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
    tool_call_id = str(event.get("tool_call_id") or "")
    if not tool_call_id:
      continue
    event_type = str(event.get("type") or "")
    if event_type == "tool_call_start":
      starts.setdefault(tool_call_id, event)
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


def run_detach_reason(*, clean_detach_reason: str, run_error: BaseException | None) -> str:
  if isinstance(run_error, asyncio.CancelledError):
    return "cancelled"
  if run_error is not None:
    return "error"
  return clean_detach_reason


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
  events = [{"type": "text_delta", "text": token + " "} for token in response.split()]
  events.append({
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {},
  })
  return events
