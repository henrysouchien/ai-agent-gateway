from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Callable, Dict

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

  if not warnings:
    return result

  enriched = dict(result)
  enriched["_runner_warning"] = " | ".join(warnings)
  if low_match:
    enriched["_runner_warning_detail"] = str(low_match)
  return enriched


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
  if uuid_factory is None:
    uuid_factory = uuid.uuid4
  raw = f"{tool_name}_{tool_use_id or uuid_factory().hex}"
  safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw)[:120]
  try:
    json.loads(content)
    ext = "json"
  except Exception:
    ext = "txt"

  filename = f"{safe}.{ext}"
  for attempt in range(2):
    if attempt:
      filename = f"{safe}_{uuid_factory().hex[:8]}.{ext}"
    spill_abspath = os.path.join(work_dir, filename)
    try:
      with open(spill_abspath, "x", encoding="utf-8") as handle:
        handle.write(content)
      return filename, spill_abspath
    except FileExistsError:
      if attempt == 0:
        continue
      raise
  raise FileExistsError(filename)


def truncate_model_tool_result_content(
  content: str,
  *,
  tool_name: str,
  max_chars: int,
  spill_filename: str | None = None,
  spill_abspath: str | None = None,
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
    payload["spill_hint"] = (
      "The FULL, untruncated result was written to this file in your code_execute "
      "working directory. Read it there instead of relying on this preview - e.g. "
      f"in code_execute: `import pandas as pd; df = pd.read_json('{spill_filename}')` "
      f"(or `json.load(open('{spill_filename}'))`). In run_bash/file_read, use the "
      "absolute path (spill_abspath) instead of the bare name."
    )
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
    fallback_payload["spill_hint"] = (
      "The FULL, untruncated result was written to this file in your code_execute working directory."
    )
  return json.dumps(fallback_payload, default=str), True


def compact_model_tool_result_entry(
  result_entry: Dict[str, Any],
  *,
  tool_name: str,
  spill_dir_provider: Callable[[], str] | None,
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
  if (
    spill_dir_provider is not None
    and spill_truncated_tool_results_enabled()
    and not is_error_tool_result_entry(result_entry, content)
  ):
    try:
      work_dir = spill_dir_provider()
      filename, spill_abspath = write_tool_result_spill(
        work_dir=work_dir,
        tool_name=tool_name,
        tool_use_id=result_entry.get("tool_use_id"),
        content=content,
        uuid_factory=uuid_factory,
      )
      live_compacted_content, _ = truncate_model_tool_result_content(
        content,
        tool_name=tool_name,
        max_chars=max_chars,
        spill_filename=filename,
        spill_abspath=spill_abspath,
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
