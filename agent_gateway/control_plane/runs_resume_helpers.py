"""Build autonomous resume context from durable run state and evidence files.

Durable task state lives in the autonomous registry manifest and its associated
evidence files. Model scrollback can be compacted or lost, so resume must
reconstruct from the manifest, event log, operator inbox, run log, and completed
tool results rather than relying on prior chat context. Every source is rendered
in full; nothing is clipped here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_gateway.autonomous_runner import AutonomousTask

from .runs_helpers import (
  AutonomousResumeRequest,
  _autonomous_events,
  _autonomous_state,
)

def _file_lines(path: Path) -> list[str]:
  try:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
      return [line.rstrip("\n") for line in handle]
  except FileNotFoundError:
    return []


def _operator_messages(record: AutonomousTask) -> list[dict[str, Any]]:
  path = record.operator_inbox_path
  if path is None or not path.exists():
    return []
  messages: list[dict[str, Any]] = []
  for line in _file_lines(path):
    try:
      payload = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(payload, dict):
      messages.append(payload)
  return messages


def _resume_json_text(value: Any) -> str:
  try:
    return json.dumps(value, default=str, separators=(",", ":"))
  except (TypeError, ValueError):
    return str(value)


def _completed_tool_results(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
  starts_by_id: dict[str, dict[str, Any]] = {}
  completed: list[dict[str, Any]] = []
  for event in events:
    event_type = event.get("type")
    tool_call_id = event.get("tool_call_id")
    if event_type == "tool_call_start" and isinstance(tool_call_id, str):
      starts_by_id[tool_call_id] = event
      continue
    if event_type != "tool_call_complete":
      continue

    summary: dict[str, Any] = {
      "tool_name": str(event.get("tool_name") or ""),
    }
    if isinstance(tool_call_id, str) and tool_call_id:
      summary["tool_call_id"] = tool_call_id
      start = starts_by_id.get(tool_call_id)
      if isinstance(start, dict) and "tool_input" in start:
        summary["tool_input"] = _resume_json_text(start.get("tool_input"))
    for field_name in ("server", "duration_ms", "is_error"):
      if field_name in event:
        summary[field_name] = event.get(field_name)

    if event.get("error") is not None:
      summary["error"] = _resume_json_text(event.get("error"))
    elif "result" in event:
      summary["result"] = _resume_json_text(event.get("result"))
    elif "final_tool_result_blocks" in event:
      summary["final_tool_result_blocks"] = _resume_json_text(event.get("final_tool_result_blocks"))
    completed.append(summary)

  return completed


def _render_newest_first(items: list[dict[str, Any]]) -> str:
  return json.dumps(list(reversed(items)), sort_keys=True, default=str, separators=(",", ":"))


def _build_autonomous_resume_context(record: AutonomousTask, payload: AutonomousResumeRequest) -> str:
  parts: list[str] = []
  evidence_status = getattr(record, "events_evidence_status", "missing")
  if evidence_status != "complete":
    parts.append(
      f"durable event evidence for this run is {evidence_status}; "
      "treat all durable writes as possibly already applied and inspect current "
      "state before writing"
    )
  parts.extend([
    (
      f"Resume autonomous control run {record.control_run_id}. "
      f"The prior top-level process ended with state={_autonomous_state(record.state)} "
      f"and exit_code={record.exit_code}."
    ),
    (
      "Continue from the latest safe point. Do not repeat durable writes that already succeeded; "
      "inspect current state before writing again, and summarize what was resumed."
    ),
  ])
  operator_note = payload.message or payload.context
  if isinstance(operator_note, str) and operator_note.strip():
    parts.append("Operator resume instruction:\n" + operator_note.strip())

  operator_messages = _operator_messages(record)
  if operator_messages:
    parts.append(
      "Recent operator messages:\n"
      + json.dumps(operator_messages, sort_keys=True, default=str, indent=2)
    )

  all_events = _autonomous_events(record)
  tail_parts: list[str] = []

  if all_events:
    tail_parts.append("Control events (newest first):\n" + _render_newest_first(all_events))

  if record.log_path.exists():
    log_lines = _file_lines(record.log_path)
    if log_lines:
      tail_parts.append("Run log:\n" + "\n".join(log_lines))

  if record.context:
    tail_parts.append("Original context:\n" + record.context)

  completed_tools = _completed_tool_results(all_events)
  if completed_tools:
    parts.append(
      "Prior completed tool results for recovery (newest first):\n"
      + _render_newest_first(completed_tools)
    )

  parts.extend(tail_parts)
  return "\n\n".join(parts)
