"""Build autonomous resume context from durable run state and evidence files.

Durable task state lives in the autonomous registry manifest and its associated
evidence files. Model scrollback can be compacted or lost, so resume must
reconstruct from the manifest, event log, operator inbox, log tail, and completed
tool-result tail rather than relying on prior chat context.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

from agent_gateway.autonomous_runner import AutonomousTask

from .runs_helpers import (
  AutonomousResumeRequest,
  _AUTONOMOUS_RESUME_CONTEXT_MAX_CHARS,
  _AUTONOMOUS_RESUME_EVENT_BLOCK_MAX_CHARS,
  _AUTONOMOUS_RESUME_EVENT_TAIL,
  _AUTONOMOUS_RESUME_LOG_BLOCK_MAX_CHARS,
  _AUTONOMOUS_RESUME_LOG_TAIL,
  _AUTONOMOUS_RESUME_OPERATOR_BLOCK_MAX_CHARS,
  _AUTONOMOUS_RESUME_OPERATOR_MESSAGE_TAIL,
  _AUTONOMOUS_RESUME_ORIGINAL_CONTEXT_BLOCK_MAX_CHARS,
  _AUTONOMOUS_RESUME_TOOL_RESULT_BLOCK_MAX_CHARS,
  _AUTONOMOUS_RESUME_TOOL_RESULT_DICT_HEAD_ITEMS,
  _AUTONOMOUS_RESUME_TOOL_RESULT_DICT_TAIL_ITEMS,
  _AUTONOMOUS_RESUME_TOOL_RESULT_LIST_MAX_ITEMS,
  _AUTONOMOUS_RESUME_TOOL_RESULT_STRING_MAX_CHARS,
  _AUTONOMOUS_RESUME_TOOL_RESULT_TAIL,
  _AUTONOMOUS_RESUME_TOOL_RESULT_VALUE_MAX_CHARS,
  _autonomous_events,
  _autonomous_state,
)

def _clip_resume_text(text: str, *, limit: int) -> str:
  if len(text) <= limit:
    return text
  return text[: max(0, limit - 120)].rstrip() + "\n[truncated for autonomous resume context]"


def _clip_resume_value_text(text: str, *, limit: int) -> str:
  if len(text) <= limit:
    return text
  marker = "...[truncated]"
  if limit <= len(marker):
    return text[:limit]
  return text[: limit - len(marker)].rstrip() + marker


def _tail_text_lines(path: Path, line_count: int) -> list[str]:
  if line_count <= 0:
    return []
  recent = deque(maxlen=line_count)
  try:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
      for line in handle:
        recent.append(line.rstrip("\n"))
  except FileNotFoundError:
    return []
  return list(recent)


def _operator_message_tail(record: AutonomousTask) -> list[dict[str, Any]]:
  path = record.operator_inbox_path
  if path is None or not path.exists():
    return []
  messages: list[dict[str, Any]] = []
  for line in _tail_text_lines(path, _AUTONOMOUS_RESUME_OPERATOR_MESSAGE_TAIL):
    try:
      payload = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(payload, dict):
      messages.append(payload)
  return messages


def _file_tail_lines(path: Path, line_count: int) -> list[str]:
  return _tail_text_lines(path, line_count)


def _resume_json_key(value: Any) -> str:
  text = str(value)
  if len(text) <= 48:
    return text
  digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:8]
  return text[:35].rstrip() + f"...#{digest}"


def _resume_json_value(
  value: Any,
  *,
  depth: int = 4,
  dict_head_items: int = _AUTONOMOUS_RESUME_TOOL_RESULT_DICT_HEAD_ITEMS,
  dict_tail_items: int = _AUTONOMOUS_RESUME_TOOL_RESULT_DICT_TAIL_ITEMS,
  list_max_items: int = _AUTONOMOUS_RESUME_TOOL_RESULT_LIST_MAX_ITEMS,
  string_limit: int = _AUTONOMOUS_RESUME_TOOL_RESULT_STRING_MAX_CHARS,
) -> Any:
  if depth <= 0:
    return _clip_resume_value_text(str(value), limit=string_limit)
  if isinstance(value, str):
    return _clip_resume_value_text(value, limit=string_limit)
  if value is None or isinstance(value, (bool, int, float)):
    return value
  if isinstance(value, dict):
    items = list(value.items())
    selected = items
    truncated_count = 0
    if len(items) > dict_head_items + dict_tail_items:
      selected = [*items[:dict_head_items], *items[-dict_tail_items:]]
      truncated_count = len(items) - len(selected)
    compacted = {
      _resume_json_key(key): _resume_json_value(
        child,
        depth=depth - 1,
        dict_head_items=dict_head_items,
        dict_tail_items=dict_tail_items,
        list_max_items=list_max_items,
        string_limit=string_limit,
      )
      for key, child in selected
    }
    if truncated_count:
      compacted["_truncated_keys"] = f"{truncated_count} keys truncated"
    return compacted
  if isinstance(value, list):
    clipped = [
      _resume_json_value(
        child,
        depth=depth - 1,
        dict_head_items=dict_head_items,
        dict_tail_items=dict_tail_items,
        list_max_items=list_max_items,
        string_limit=string_limit,
      )
      for child in value[:list_max_items]
    ]
    if len(value) > list_max_items:
      clipped.append(f"[{len(value) - list_max_items} items truncated]")
    return clipped
  return _clip_resume_value_text(str(value), limit=string_limit)


def _resume_json_text(value: Any, *, limit: int = _AUTONOMOUS_RESUME_TOOL_RESULT_VALUE_MAX_CHARS) -> str:
  rendered = ""
  compact_profiles = (
    (240, 20, 8, 6),
    (180, 12, 6, 4),
    (120, 8, 4, 3),
    (80, 4, 3, 2),
    (40, 2, 2, 2),
  )
  try:
    for string_limit, list_max_items, dict_head_items, dict_tail_items in compact_profiles:
      rendered = json.dumps(
        _resume_json_value(
          value,
          string_limit=string_limit,
          list_max_items=list_max_items,
          dict_head_items=dict_head_items,
          dict_tail_items=dict_tail_items,
        ),
        default=str,
        separators=(",", ":"),
      )
      if len(rendered) <= limit:
        return rendered
  except (TypeError, ValueError):
    rendered = str(value)
  return _clip_resume_text(rendered, limit=limit)


def _completed_tool_result_tail(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

  return completed[-_AUTONOMOUS_RESUME_TOOL_RESULT_TAIL:]


def _bounded_tool_summary(summary: dict[str, Any]) -> dict[str, Any]:
  bounded: dict[str, Any] = {}
  for key, value in summary.items():
    bounded_key = _resume_json_key(key)
    if isinstance(value, str):
      bounded[bounded_key] = _clip_resume_text(value, limit=_AUTONOMOUS_RESUME_TOOL_RESULT_VALUE_MAX_CHARS)
    elif value is None or isinstance(value, (bool, int, float)):
      bounded[bounded_key] = value
    else:
      bounded[bounded_key] = _resume_json_text(value)
  return bounded


def _minimal_tool_summary(summary: dict[str, Any]) -> dict[str, Any]:
  minimal = {"tool_name": _clip_resume_value_text(str(summary.get("tool_name") or ""), limit=120)}
  if summary.get("tool_call_id"):
    minimal["tool_call_id"] = _clip_resume_value_text(str(summary.get("tool_call_id")), limit=120)
  if summary.get("is_error") is not None:
    minimal["is_error"] = summary.get("is_error")
  for field_name in ("error", "result", "final_tool_result_blocks"):
    if summary.get(field_name):
      minimal[field_name] = _clip_resume_value_text(str(summary.get(field_name)), limit=120)
      break
  return minimal


def _render_completed_tool_result_tail(
  completed_tools: list[dict[str, Any]],
  *,
  max_chars: int = _AUTONOMOUS_RESUME_TOOL_RESULT_BLOCK_MAX_CHARS,
) -> str:
  selected_newest_first: list[dict[str, Any]] = []
  rendered = ""
  for tool in reversed(completed_tools):
    candidate = [*selected_newest_first, _bounded_tool_summary(tool)]
    candidate_text = json.dumps(candidate, sort_keys=True, default=str, separators=(",", ":"))
    if len(candidate_text) > max_chars and selected_newest_first:
      break
    selected_newest_first = candidate
    rendered = candidate_text
    if len(rendered) >= max_chars:
      break
  if len(rendered) > max_chars:
    minimal_text = json.dumps([_minimal_tool_summary(completed_tools[-1])], sort_keys=True, default=str, separators=(",", ":"))
    if len(minimal_text) <= max_chars:
      return minimal_text
    return "[]"
  return rendered


def _render_json_newest_first_tail(items: list[dict[str, Any]], *, max_chars: int) -> str:
  selected_newest_first: list[dict[str, Any]] = []
  rendered = ""
  for item in reversed(items):
    candidate = [*selected_newest_first, item]
    candidate_text = json.dumps(candidate, sort_keys=True, default=str, separators=(",", ":"))
    if len(candidate_text) > max_chars and selected_newest_first:
      break
    selected_newest_first = candidate
    rendered = candidate_text
    if len(rendered) >= max_chars:
      break
  if len(rendered) > max_chars:
    compact = {
      key: _clip_resume_value_text(str(value), limit=120) if isinstance(value, str) else value
      for key, value in items[-1].items()
      if key in {"type", "run_id", "task_id", "tool_name", "event_id", "text"}
    }
    minimal_text = json.dumps([compact], sort_keys=True, default=str, separators=(",", ":"))
    if len(minimal_text) <= max_chars:
      return minimal_text
    return "[]"
  return rendered


def _resume_section(header: str, body: str, *, limit: int) -> str:
  return header + "\n" + _clip_resume_text(body, limit=limit)


def _resume_joined_len(parts: list[str]) -> int:
  return len("\n\n".join(parts))


def _build_autonomous_resume_context(record: AutonomousTask, payload: AutonomousResumeRequest) -> str:
  parts: list[str] = [
    (
      f"Resume autonomous control run {record.control_run_id}. "
      f"The prior top-level process ended with state={_autonomous_state(record.state)} "
      f"and exit_code={record.exit_code}."
    ),
    (
      "Continue from the latest safe point. Do not repeat durable writes that already succeeded; "
      "inspect current state before writing again, and summarize what was resumed."
    ),
  ]
  operator_note = payload.message or payload.context
  if isinstance(operator_note, str) and operator_note.strip():
    parts.append(
      _resume_section(
        "Operator resume instruction:",
        operator_note.strip(),
        limit=_AUTONOMOUS_RESUME_OPERATOR_BLOCK_MAX_CHARS,
      )
    )

  operator_messages = _operator_message_tail(record)
  if operator_messages:
    parts.append(
      _resume_section(
        "Recent operator messages:",
        json.dumps(operator_messages, sort_keys=True, default=str, indent=2),
        limit=_AUTONOMOUS_RESUME_OPERATOR_BLOCK_MAX_CHARS,
      )
    )

  all_events = _autonomous_events(record)
  tail_parts: list[str] = []

  events = all_events[-_AUTONOMOUS_RESUME_EVENT_TAIL:]
  if events:
    event_block = _render_json_newest_first_tail(events, max_chars=_AUTONOMOUS_RESUME_EVENT_BLOCK_MAX_CHARS)
    if event_block != "[]":
      tail_parts.append("Recent control events (newest first):\n" + event_block)

  if record.log_path.exists():
    log_lines = _file_tail_lines(record.log_path, _AUTONOMOUS_RESUME_LOG_TAIL)
    if log_lines:
      tail_parts.append(
        _resume_section(
          "Recent log tail:",
          "\n".join(log_lines),
          limit=_AUTONOMOUS_RESUME_LOG_BLOCK_MAX_CHARS,
        )
      )

  if record.context:
    tail_parts.append(
      _resume_section(
        "Original context:",
        record.context,
        limit=_AUTONOMOUS_RESUME_ORIGINAL_CONTEXT_BLOCK_MAX_CHARS,
      )
    )

  completed_tools = _completed_tool_result_tail(all_events)
  if completed_tools:
    recovery_header = "Prior completed tool results for recovery (newest first):\n"
    base_parts = [*parts, *tail_parts]
    recovery_budget = _AUTONOMOUS_RESUME_CONTEXT_MAX_CHARS - _resume_joined_len(base_parts)
    if base_parts:
      recovery_budget -= len("\n\n")
    recovery_budget -= len(recovery_header)
    recovery_budget = min(_AUTONOMOUS_RESUME_TOOL_RESULT_BLOCK_MAX_CHARS, recovery_budget)
    if recovery_budget >= 300:
      recovery_block = _render_completed_tool_result_tail(completed_tools, max_chars=recovery_budget)
      if recovery_block != "[]":
        parts.append(recovery_header + recovery_block)

  parts.extend(tail_parts)
  return _clip_resume_text("\n\n".join(parts), limit=_AUTONOMOUS_RESUME_CONTEXT_MAX_CHARS)
