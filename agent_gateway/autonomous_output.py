from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Pattern, Sequence

from ._io import _atomic_write_json, _read_json_object

_STATE_JSON_MARKER = "## STATE_UPDATE_JSON"
_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class RunOutput:
  response: str
  tools_used: list[str]
  usage: dict[str, Any]
  error: str | None
  timed_out: bool
  budget_exceeded: bool = False
  max_turns_reached: bool = False
  operator_paused: bool = False
  max_tokens_reached: bool = False
  exit_reason: str | None = None
  post_run_guard: dict[str, Any] | None = None


def _extract_summary(text: str, limit: int = 1500) -> str:
  stripped = str(text or "").strip()
  if len(stripped) <= limit:
    return stripped
  return stripped[: limit - 3].rstrip() + "..."


def _ensure_string_list(value: Any) -> list[str]:
  if not isinstance(value, list):
    return []
  items: list[str] = []
  for item in value:
    if item is None:
      continue
    text = str(item).strip()
    if text:
      items.append(text)
  return items


def collect_run_output(
  event_log: Any,
  timed_out: bool,
  *,
  run_output_cls: type[RunOutput] = RunOutput,
) -> RunOutput:
  text_parts: list[str] = []
  tool_calls: list[str] = []
  usage: dict[str, Any] = {}
  error_msg: str | None = None
  budget_exceeded = False
  max_turns_reached = False
  operator_paused = False
  max_tokens_reached = False
  terminal_seen = False

  for entry in event_log.entries:
    event = entry.event
    event_type = event.get("type")
    if event_type == "stream_retry":
      text_parts.clear()
      tool_calls.clear()
    elif event_type == "text_delta":
      text_parts.append(str(event.get("text", "")))
    elif event_type == "tool_call_start":
      tool_calls.append(str(event.get("tool_name", "")))
    elif event_type == "stream_complete":
      terminal_seen = True
      event_usage = event.get("usage")
      if isinstance(event_usage, dict):
        usage = event_usage
      disposition = event.get("terminal_disposition")
      if disposition not in {"completed", "interrupted"}:
        error_msg = f"invalid_terminal_disposition: {disposition!r}"
      elif disposition == "interrupted":
        reason = str(event.get("reason") or "unspecified")
        if reason == "budget_exceeded":
          budget_exceeded = True
        elif reason == "max_turns_reached":
          max_turns_reached = True
        elif reason == "operator_pause":
          operator_paused = True
        elif reason == "timeout":
          pass
        else:
          error_msg = f"Autonomous run interrupted: {reason}"
      break
    elif event_type == "budget_exceeded":
      budget_exceeded = True
    elif event_type == "max_turns_reached":
      max_turns_reached = True
    elif event_type == "max_tokens_reached":
      max_tokens_reached = True
    elif event_type == "assistant_message" and event.get("stop_reason") == "max_tokens":
      max_tokens_reached = True
    elif event_type == "operator_pause":
      operator_paused = True
    elif event_type == "interrupted" and event.get("reason") == "operator_pause":
      operator_paused = True
    elif event_type == "error":
      error_msg = str(event.get("error", "Autonomous run encountered an error"))
      terminal_seen = True
      break

  if not terminal_seen:
    error_msg = "missing_terminal_event"

  return run_output_cls(
    response="".join(text_parts).strip(),
    tools_used=tool_calls,
    usage=usage,
    error=error_msg,
    timed_out=timed_out,
    budget_exceeded=budget_exceeded,
    max_turns_reached=max_turns_reached,
    operator_paused=operator_paused,
    max_tokens_reached=max_tokens_reached,
  )


def run_output_exit_code(run_output: RunOutput) -> int:
  if run_output.timed_out:
    return 124
  if run_output.exit_reason == "post_run_guard_failed":
    return 1
  if run_output.error:
    return 1
  if run_output.budget_exceeded:
    return 2
  if run_output.max_turns_reached:
    return 3
  if run_output.max_tokens_reached:
    return 4
  return 0


def run_output_outcome(run_output: RunOutput) -> str:
  if run_output.timed_out:
    return "timeout"
  if run_output.exit_reason:
    return run_output.exit_reason
  if run_output.error:
    return "error"
  if run_output.budget_exceeded:
    return "budget_exceeded"
  if run_output.max_turns_reached:
    return "max_turns"
  if run_output.max_tokens_reached:
    return "max_tokens"
  if run_output.operator_paused:
    return "operator_pause"
  return "success"


def mark_post_run_guard_failure(
  run_output: RunOutput,
  *,
  guard: str,
  message: str,
  details: dict[str, Any] | None = None,
) -> None:
  payload = {
    "guard": guard,
    "message": message,
  }
  if details:
    payload.update(details)
  run_output.error = message
  run_output.exit_reason = "post_run_guard_failed"
  run_output.post_run_guard = payload


def load_state(
  state_dir: str | Path,
  state_file: str = "state.json",
) -> dict[str, Any]:
  return _read_json_object(Path(state_dir) / state_file)


def save_state(
  state_dir: str | Path,
  state: dict[str, Any],
  state_file: str = "state.json",
) -> None:
  _atomic_write_json(Path(state_dir) / state_file, dict(state))


def extract_state_update(
  text: str,
  *,
  state_json_marker: str = _STATE_JSON_MARKER,
  json_fence_re: Pattern[str] = _JSON_FENCE_RE,
) -> dict[str, Any]:
  if not str(text or "").strip():
    return {}

  section = text
  marker_idx = text.rfind(state_json_marker)
  if marker_idx >= 0:
    section = text[marker_idx:]

  matches = list(json_fence_re.finditer(section))
  for match in reversed(matches):
    candidate = match.group(1)
    try:
      payload = json.loads(candidate)
    except json.JSONDecodeError:
      continue
    if isinstance(payload, dict):
      return payload
  return {}


def build_state_payload(
  previous_state: dict[str, Any],
  model_state: dict[str, Any],
  run_output: RunOutput,
  model_name: str = "",
  briefing_file: str = "",
  connected_servers: Sequence[str] | None = None,
  active_servers: Sequence[str] | None = None,
  extract_summary_fn: Callable[[str], str] | None = None,
  ensure_string_list_fn: Callable[[Any], list[str]] | None = None,
) -> dict[str, Any]:
  state: dict[str, Any] = {}
  if isinstance(previous_state, dict):
    state.update(previous_state)
  outcome = run_output_outcome(run_output)
  if outcome == "success" and isinstance(model_state, dict):
    state.update(model_state)

  summary_fn = extract_summary_fn or _extract_summary
  connected_server_names = sorted({str(name) for name in (connected_servers or []) if str(name).strip()})
  active_server_names = sorted({str(name) for name in (active_servers or []) if str(name).strip()})

  state["last_run"] = datetime.now(tz=timezone.utc).isoformat()
  state["model"] = model_name
  state["briefing_file"] = briefing_file
  state["timed_out"] = run_output.timed_out
  state["budget_exceeded"] = run_output.budget_exceeded
  state["max_turns_reached"] = run_output.max_turns_reached
  state["max_tokens_reached"] = run_output.max_tokens_reached
  state["operator_paused"] = run_output.operator_paused
  state["last_outcome"] = outcome
  state["connected_servers"] = connected_server_names
  state["active_servers"] = active_server_names
  state["tools_used"] = sorted({name for name in run_output.tools_used if name})
  state["usage"] = run_output.usage
  state["last_summary"] = summary_fn(run_output.response)

  if run_output.error:
    state["error"] = run_output.error
  else:
    state.pop("error", None)

  string_list_fn = ensure_string_list_fn or _ensure_string_list
  alerts = string_list_fn(state.get("alerts"))
  next_session = string_list_fn(state.get("next_session"))
  if alerts:
    state["alerts"] = alerts
  if next_session:
    state["next_session"] = next_session

  return state


def format_run_summary(
  run_output: RunOutput,
  label: str | None = None,
  state: dict[str, Any] | None = None,
  format_state_fn: Callable[[dict[str, Any]], str] | None = None,
  extract_summary_fn: Callable[..., str] | None = None,
) -> str:
  status = "timed out" if run_output.timed_out else "completed"
  if run_output.exit_reason == "post_run_guard_failed" and not run_output.timed_out:
    status = "post-run guard failed"
  elif run_output.error and not run_output.timed_out:
    status = "failed"
  elif run_output.budget_exceeded:
    status = "budget exceeded"
  elif run_output.max_turns_reached:
    status = "max turns reached"
  elif run_output.max_tokens_reached:
    status = "max tokens reached"
  elif run_output.operator_paused:
    status = "operator paused"

  usage = run_output.usage if isinstance(run_output.usage, dict) else {}
  in_tokens = usage.get("input_tokens", "?")
  out_tokens = usage.get("output_tokens", "?")
  est_cost = usage.get("estimated_cost", "?")
  tools_used = sorted({name for name in run_output.tools_used if name})
  tools_preview = ", ".join(tools_used[:8]) if tools_used else "none"
  if len(tools_used) > 8:
    tools_preview += f", ... (+{len(tools_used) - 8})"

  lines = [
    label or "Autonomous run",
    f"Status: {status}",
  ]
  if isinstance(state, dict) and state.get("briefing_file"):
    lines.append(f"Briefing: {state['briefing_file']}")
  lines.extend([
    f"Usage: in={in_tokens} out={out_tokens} est_cost={est_cost}",
    f"Tools: {tools_preview}",
  ])

  if state and format_state_fn is not None:
    formatted_state = str(format_state_fn(state) or "").strip()
    if formatted_state:
      lines.extend(["", *formatted_state.splitlines()])

  summary_fn = extract_summary_fn or _extract_summary
  summary = summary_fn(run_output.response, limit=1200)
  if summary:
    lines.extend(["", "Summary:", summary])
  if run_output.exit_reason:
    lines.extend(["", f"Exit reason: {run_output.exit_reason}"])
  if run_output.post_run_guard:
    guard_name = run_output.post_run_guard.get("guard")
    if guard_name:
      lines.append(f"Post-run guard: {guard_name}")
  if run_output.error:
    lines.extend(["", f"Error: {run_output.error}"])

  message = "\n".join(lines)
  if len(message) > 3900:
    message = message[:3897] + "..."
  return message


__all__ = [
  "RunOutput",
  "build_state_payload",
  "collect_run_output",
  "extract_state_update",
  "format_run_summary",
  "load_state",
  "mark_post_run_guard_failure",
  "run_output_exit_code",
  "run_output_outcome",
  "save_state",
]
