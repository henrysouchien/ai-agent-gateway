from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable, Dict

from .tool_result_spill import (
  SPILL_LANE_CODE_EXECUTE,
  SPILL_LANE_FILE_TOOLS,
  SpillPublication,
  SpillSink,
  normalize_spill_sink,
  write_spill_set,
)
from .tool_result_semantics import status_error_has_detail

MODEL_TOOL_RESULT_MAX_CHARS = 60_000
MODEL_TOOL_RESULT_MAX_CHARS_ENV = "AGENT_GATEWAY_MAX_MODEL_TOOL_RESULT_CHARS"
MODEL_TOOL_RESULT_MIN_CHARS = 4_000
SPILL_TRUNCATED_TOOL_RESULTS_ENV = "AGENT_GATEWAY_SPILL_TRUNCATED_TOOL_RESULTS"


def model_tool_result_max_chars() -> int:
  raw = os.getenv(MODEL_TOOL_RESULT_MAX_CHARS_ENV)
  if raw is None or not raw.strip():
    return MODEL_TOOL_RESULT_MAX_CHARS
  try:
    value = int(raw)
  except ValueError:
    return MODEL_TOOL_RESULT_MAX_CHARS
  if value <= 0:
    return 0
  return max(MODEL_TOOL_RESULT_MIN_CHARS, value)


def spill_truncated_tool_results_enabled() -> bool:
  raw = os.getenv(SPILL_TRUNCATED_TOOL_RESULTS_ENV)
  if raw is None:
    return True
  return raw.strip().lower() not in {"0", "false", "no"}


def scalar_preview_fields(value: Any) -> Dict[str, Any]:
  if not isinstance(value, dict):
    return {}
  preview: Dict[str, Any] = {}
  for key, item in value.items():
    if isinstance(item, str):
      preview[str(key)] = item if len(item) <= 500 else f"{item[:500]}... <truncated chars={len(item)}>"
    elif isinstance(item, (int, float, bool)) or item is None:
      preview[str(key)] = item
    if len(preview) >= 24:
      break
  return preview


def annotate_result(result: Any, tool_name: str = "") -> Any:
  """Add _runner_warning to generic results with detectable anomalies."""
  if not isinstance(result, dict):
    return result

  warnings: list[str] = []
  interceptor_warnings = result.pop("_interceptor_warnings", None)
  if isinstance(interceptor_warnings, list):
    for warning in interceptor_warnings:
      warnings.append(f"Policy warning: {warning}")

  low_match = result.get("low_match_warning")
  if low_match:
    warnings.append(f"Low match rate detected: {low_match}")

  if tool_name == "run_agent":
    sub_warning = result.get("warning")
    if sub_warning:
      warnings.append(f"Sub-agent warning: {sub_warning}")

  if _status_error_without_detail(result):
    tool_label = f"Tool {tool_name}" if tool_name else "Tool"
    warnings.append(
      f"{tool_label} returned status=error without error detail; "
      "do not retry unchanged input unless required context changed or there is new evidence the failure was transient."
    )

  if not warnings:
    return result

  enriched = dict(result)
  enriched["_runner_warning"] = " | ".join(warnings)
  if low_match:
    enriched["_runner_warning_detail"] = str(low_match)
  return enriched


def _status_error_without_detail(result: dict[str, Any]) -> bool:
  status = result.get("status")
  return isinstance(status, str) and status.strip().lower() == "error" and not status_error_has_detail(result)


def make_error_result(
  tool_use_id: str,
  code: str,
  message: str,
  sub_code: str = "",
  data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
  error_dict = {"code": code, "message": message}
  if sub_code:
    error_dict["sub_code"] = sub_code
  if data is not None:
    error_dict["data"] = data
  return {
    "type": "tool_result",
    "tool_use_id": tool_use_id,
    "content": json.dumps({"error": error_dict}),
    "is_error": True,
  }


def is_error_tool_result_entry(result_entry: Dict[str, Any], content: str) -> bool:
  if result_entry.get("is_error") or "error" in result_entry:
    return True
  try:
    parsed = json.loads(content)
  except Exception:
    return False
  # Only a *truthy* top-level "error" marks a genuine error result. A successful
  # data payload that merely carries `"error": null` (or empty) must still spill;
  # the authoritative tool-error signal is the is_error flag checked above.
  return isinstance(parsed, dict) and bool(parsed.get("error"))


def write_tool_result_spill(
  *,
  work_dir: str,
  tool_name: str,
  tool_use_id: Any,
  content: str,
  uuid_factory=None,
) -> tuple[str, str]:
  sink = normalize_spill_sink(lambda: work_dir)
  assert sink is not None
  publication = write_spill_set(
    sink=sink,
    tool_name=tool_name,
    tool_use_id=tool_use_id,
    content=content,
    model_max_chars=model_tool_result_max_chars(),
    uuid_factory=uuid_factory or uuid.uuid4,
  )
  return publication.filename, publication.abspath


def truncate_model_tool_result_content(
  content: str,
  *,
  tool_name: str,
  max_chars: int,
  spill_filename: str | None = None,
  spill_abspath: str | None = None,
  spill_hint: str | None = None,
  spill_summary: dict[str, Any] | None = None,
) -> tuple[str, bool]:
  if max_chars <= 0 or len(content) <= max_chars:
    return content, False

  parsed: Any | None = None
  try:
    parsed = json.loads(content)
  except Exception:
    parsed = None

  payload: Dict[str, Any] = {
    "_runner_truncated": True,
    "tool_name": tool_name,
    "original_chars": len(content),
    "message": (
      "The full tool result was retained in the gateway event log, but this "
      "model-bound preview was truncated to stay within the provider context "
      "window. Narrow the tool query (filters, pagination, fewer fields) if you "
      "need more detail. If a spill file is listed below, read it for the full result."
    ),
  }
  if spill_filename is not None:
    payload["spill_file"] = spill_filename
    payload["spill_abspath"] = spill_abspath
    payload["spill_hint"] = spill_hint or _legacy_spill_hint(spill_filename)
    if spill_summary:
      payload["spill_summary"] = spill_summary
  if isinstance(parsed, dict):
    payload["top_level_keys"] = list(parsed.keys())[:50]
    scalar_fields = scalar_preview_fields(parsed)
    if scalar_fields:
      payload["scalar_fields"] = scalar_fields
  elif isinstance(parsed, list):
    payload["top_level_type"] = "list"
    payload["top_level_items"] = len(parsed)

  prefix_budget = max(0, max_chars - len(json.dumps(payload, default=str)) - 200)
  payload["content_prefix"] = content[:prefix_budget]
  payload["retained_prefix_chars"] = len(payload["content_prefix"])

  truncated = json.dumps(payload, default=str)
  while len(truncated) > max_chars and prefix_budget > 0:
    prefix_budget = max(0, prefix_budget - (len(truncated) - max_chars) - 100)
    payload["content_prefix"] = content[:prefix_budget]
    payload["retained_prefix_chars"] = len(payload["content_prefix"])
    truncated = json.dumps(payload, default=str)
  if len(truncated) <= max_chars:
    return truncated, True

  fallback_payload = {
    "_runner_truncated": True,
    "tool_name": tool_name,
    "original_chars": len(content),
    "message": "Tool result omitted from model context because it exceeded the configured payload limit.",
  }
  if spill_filename is not None:
    fallback_payload["spill_file"] = spill_filename
    fallback_payload["spill_abspath"] = spill_abspath
    fallback_payload["spill_hint"] = spill_hint or (
      "The FULL, untruncated result was written to this file in your code_execute working directory."
    )
    if spill_summary:
      fallback_payload["spill_summary"] = spill_summary
  return json.dumps(fallback_payload, default=str), True


def _legacy_spill_hint(spill_filename: str) -> str:
  return (
    "The FULL, untruncated result was written to this file in your code_execute "
    "working directory. Read it there instead of relying on this preview - e.g. "
    f"in code_execute: `import pandas as pd; df = pd.read_json('{spill_filename}')` "
    f"(or `json.load(open('{spill_filename}'))`). In run_bash/file_read, use the "
    "absolute path (spill_abspath) instead of the bare name."
  )


def _publication_summary(publication: SpillPublication) -> dict[str, Any]:
  return {
    "member_count": publication.member_count,
    "total_chars": publication.total_chars,
    "largest_members": list(publication.largest_members),
  }


def _spill_hint(publication: SpillPublication, sink: SpillSink) -> str:
  if publication.lane == SPILL_LANE_CODE_EXECUTE:
    return _legacy_spill_hint(publication.filename)
  if publication.lane == SPILL_LANE_FILE_TOOLS:
    operations: list[str] = []
    if sink.capabilities.file_read:
      operations.append(
        "page with `file_read(file_path=spill_abspath, offset=0, limit=3)` "
        "(`limit=1` for `.chunks.txt` members)"
      )
    if sink.capabilities.file_grep:
      operations.append(
        "search an exact member with `file_grep(pattern=..., path=..., max_results=1, context_lines=1)`"
      )
    action = "; ".join(operations)
    return (
      "The full result is committed as the spill set described by this manifest. "
      f"Use the manifest's bounded member list to {action}. Do not read a whole large file in one call; "
      "that result will be truncated again."
    )
  return (
    "The full result was retained as a run-scoped audit spill, but this run has no file/code reader. "
    "Narrow or paginate the original tool call; an operator can inspect the audit artifact if needed."
  )


def compact_model_tool_result_entry(
  result_entry: Dict[str, Any],
  *,
  tool_name: str,
  spill_sink: Callable[[], str] | SpillSink | None = None,
  spill_dir_provider: Callable[[], str] | SpillSink | None = None,
  log_session_id: str,
  logger: Any,
  uuid_factory: Callable[[], Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
  content = result_entry.get("content")
  if not isinstance(content, str):
    return result_entry, result_entry

  max_chars = model_tool_result_max_chars()
  plain_compacted_content, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name=tool_name,
    max_chars=max_chars,
  )
  if not was_truncated:
    return result_entry, result_entry

  plain_compacted_entry = dict(result_entry)
  plain_compacted_entry["content"] = plain_compacted_content
  live_entry = plain_compacted_entry
  durable_entry = plain_compacted_entry
  if spill_sink is not None and spill_dir_provider is not None:
    raise ValueError("pass spill_sink or spill_dir_provider, not both")
  configured_sink = normalize_spill_sink(spill_sink if spill_sink is not None else spill_dir_provider)
  if (
    configured_sink is not None
    and spill_truncated_tool_results_enabled()
    and not is_error_tool_result_entry(result_entry, content)
  ):
    try:
      publication = write_spill_set(
        sink=configured_sink,
        tool_name=tool_name,
        tool_use_id=result_entry.get("tool_use_id"),
        content=content,
        model_max_chars=max_chars,
        uuid_factory=uuid_factory or uuid.uuid4,
      )
      live_compacted_content, _ = truncate_model_tool_result_content(
        content,
        tool_name=tool_name,
        max_chars=max_chars,
        spill_filename=publication.filename,
        spill_abspath=publication.abspath,
        spill_hint=_spill_hint(publication, configured_sink),
        spill_summary=(
          _publication_summary(publication)
          if publication.lane == SPILL_LANE_FILE_TOOLS
          else None
        ),
      )
      live_entry = dict(result_entry)
      live_entry["content"] = live_compacted_content
    except Exception as exc:
      logger.warning(
        "[%s] Tool %s result spill failed; using compacted preview only: %s",
        log_session_id,
        tool_name,
        exc,
        exc_info=True,
      )
  logger.info(
    "[%s] Tool %s result compacted for model context | original_chars=%d compacted_chars=%d limit=%d",
    log_session_id,
    tool_name,
    len(content),
    len(str(live_entry.get("content", ""))),
    max_chars,
    extra={
      "data": {
        "event": "tool_result_compacted",
        "session_id": log_session_id,
        "tool": tool_name,
        "original_chars": len(content),
        "compacted_chars": len(str(live_entry.get("content", ""))),
        "limit": max_chars,
      }
    },
  )
  return live_entry, durable_entry
