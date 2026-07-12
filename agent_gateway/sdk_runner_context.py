from __future__ import annotations

import inspect
import json
import sys
import time
from typing import Any, Sequence

from . import sdk_runner_helpers as _sdk_runner_helpers
from .runner import ToolResultContext
from .tool_result_semantics import classify_semantic_tool_error as _classify_semantic_tool_error


_PARENT_MODULE = "agent_gateway.sdk_runner"


def _compat(name: str, fallback: Any) -> Any:
  parent = sys.modules.get(_PARENT_MODULE)
  if parent is not None and hasattr(parent, name):
    return getattr(parent, name)
  return fallback


def normalize_context_surfaces(surfaces: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
  return [
    dict(surface)
    for surface in (surfaces or [])
    if isinstance(surface, dict)
  ]


def context_surface_records(runner: Any, *, logger: Any) -> list[dict[str, Any]]:
  if runner._context_surfaces_provider is None:
    return runner._normalize_context_surfaces(runner._context_surfaces_static)
  try:
    return runner._normalize_context_surfaces(runner._context_surfaces_provider())
  except Exception as exc:
    logger.warning("[%s] context surface provider failed (non-fatal): %s", runner._sid, exc)
    return runner._normalize_context_surfaces(runner._context_surfaces_static)


async def call_on_usage(
  runner: Any,
  usage_event: Any,
  *,
  logger: Any,
  usage_state: str = "succeeded",
  emit_commercial: bool = True,
) -> None:
  producer = getattr(runner, "_commercial_usage_producer", None)
  if emit_commercial and producer is not None:
    await producer.emit(usage_event, usage_state=usage_state)
  recorded = await runner._aggregator.record(usage_event)
  await runner._aggregator.set_turns(runner._num_turns)
  if not recorded or runner._summary_emitted:
    if producer is not None:
      mark_late = getattr(producer, "mark_late", None)
      if callable(mark_late):
        try:
          late_result = mark_late(usage_event.event_id)
          if inspect.isawaitable(late_result):
            await late_result
        except Exception as exc:
          logger.warning("[%s] late commercial reconciliation failed: %s", runner._sid, exc)
    logger.warning("[%s] Usage event arrived after session summary emission: %s", runner._sid, usage_event.event_id)
    await runner._call_on_late_usage_event(usage_event)
    return
  if runner._on_usage is None:
    return
  try:
    result = runner._on_usage(usage_event)
    if inspect.isawaitable(result):
      await result
  except Exception as exc:
    logger.warning("[%s] on_usage hook failed (non-fatal): %s", runner._sid, exc)


async def call_on_late_usage_event(runner: Any, usage_event: Any, *, logger: Any) -> None:
  if runner._on_late_usage_event is None:
    return
  try:
    result = runner._on_late_usage_event(usage_event)
    if inspect.isawaitable(result):
      await result
  except Exception as exc:
    logger.warning("[%s] on_late_usage_event hook failed (non-fatal): %s", runner._sid, exc)


async def call_on_session_summary(runner: Any, summary: Any, *, logger: Any) -> None:
  producer = getattr(runner, "_commercial_usage_producer", None)
  if producer is not None:
    reconcile = getattr(producer, "reconcile", None)
    if callable(reconcile):
      try:
        await reconcile(summary)
      except Exception as exc:
        logger.warning("[%s] commercial usage reconciliation failed: %s", runner._sid, exc)
  if runner._on_session_summary is None:
    return
  try:
    result = runner._on_session_summary(summary)
    if inspect.isawaitable(result):
      await result
  except Exception as exc:
    logger.warning("[%s] on_session_summary hook failed (non-fatal): %s", runner._sid, exc)


def call_on_tool_timing(
  runner: Any,
  *,
  tool_name: str,
  server: str | None,
  duration_ms: int,
  is_error: bool,
  result_bytes: int,
  logger: Any,
  tool_call_id: str | None = None,
  request_id: str | None = None,
) -> None:
  if runner._on_tool_timing is None:
    return
  try:
    kwargs: dict[str, Any] = {}
    if runner._on_tool_timing_accepts_context_surfaces:
      kwargs["context_surfaces"] = runner._context_surface_records()
    if getattr(runner, "_on_tool_timing_accepts_tool_call_id", False):
      kwargs["tool_call_id"] = tool_call_id
    if getattr(runner, "_on_tool_timing_accepts_request_id", False):
      kwargs["request_id"] = request_id
    if runner._on_tool_timing_accepts_user_id:
      kwargs["user_id"] = runner._usage_user_id
    runner._on_tool_timing(
      runner._session_id,
      tool_name,
      server,
      duration_ms,
      is_error,
      result_bytes,
      **kwargs,
    )
  except Exception as exc:
    logger.warning("[%s] on_tool_timing hook failed (non-fatal): %s", runner._sid, exc)


async def call_on_tool_result(runner: Any, ctx: ToolResultContext, *, logger: Any) -> list[dict[str, Any]]:
  if runner._on_tool_result is None:
    return []
  try:
    extra_blocks = await runner._on_tool_result(ctx)
  except Exception as exc:
    logger.warning("[%s] on_tool_result hook failed (non-fatal): %s", runner._sid, exc)
    return []
  if not extra_blocks:
    return []
  if isinstance(extra_blocks, list):
    return [block for block in extra_blocks if isinstance(block, dict)]
  return []


def build_prompt(messages: list[dict[str, Any]]) -> str:
  normalized: list[tuple[str, str]] = []
  for message in messages:
    role = str(message.get("role") or "user").strip().lower() or "user"
    content = str(message.get("content") or "")
    if not content:
      continue
    normalized.append((role, content))

  if not normalized:
    return ""
  if len(normalized) == 1 and normalized[0][0] == "user":
    return normalized[0][1]

  transcript = "\n\n".join(f"{role.upper()}: {content}" for role, content in normalized[:-1])
  last_role, last_content = normalized[-1]
  if last_role == "user":
    if transcript:
      return (
        "You are continuing a stateless conversation. Use the transcript below as prior context.\n\n"
        f"{transcript}\n\n"
        "Respond to the latest user message below.\n\n"
        f"USER: {last_content}"
      )
    return last_content

  combined = "\n\n".join(f"{role.upper()}: {content}" for role, content in normalized)
  return (
    "You are continuing a stateless conversation. Use the transcript below as prior context and continue appropriately.\n\n"
    f"{combined}"
  )


def make_result_entry(
  tool_call_id: str,
  result: Any | None,
  error: dict[str, Any] | None,
) -> dict[str, Any]:
  if error is not None:
    return {
      "type": "tool_result",
      "tool_use_id": tool_call_id,
      "content": json.dumps({"error": error}, default=str),
      "is_error": True,
    }
  entry = {
    "type": "tool_result",
    "tool_use_id": tool_call_id,
    "content": json.dumps(result, default=str),
  }
  classify = _compat("classify_semantic_tool_error", _classify_semantic_tool_error)
  if classify(result) is not None:
    entry["is_error"] = True
  return entry


def format_additional_context(
  *,
  tool_name: str,
  result_entry: dict[str, Any],
  extra_blocks: Sequence[dict[str, Any]],
) -> str | None:
  parts: list[str] = []
  parse_result_payload = _compat("_parse_result_payload", _sdk_runner_helpers.parse_result_payload)
  summarize_error_payload = _compat("_summarize_error_payload", _sdk_runner_helpers.summarize_error_payload)
  parsed = parse_result_payload(result_entry.get("content"))
  if isinstance(parsed, dict):
    warning = parsed.get("_runner_warning") or parsed.get("warning")
    if isinstance(warning, str) and warning.strip():
      parts.append(f"WARNING: {warning.strip()}")

    if result_entry.get("is_error") is True:
      summary = summarize_error_payload(parsed)
      parts.append(
        f"ERROR: The previous tool call ({tool_name}) returned a structured error: {summary}. "
        "Treat this result as a failure."
      )

  for block in extra_blocks:
    if block.get("_event_only"):
      continue
    block_type = str(block.get("type") or "")
    if block_type == "text":
      text = str(block.get("text") or "").strip()
      if text:
        parts.append(text)
      continue
    parts.append(json.dumps(block, default=str))

  if not parts:
    return None
  return "\n\n".join(part for part in parts if part)


async def build_hook_additional_context(
  runner: Any,
  *,
  tool_call_id: str,
  tool_name: str,
  tool_input: dict[str, Any],
  result: Any | None,
  error: dict[str, Any] | None,
  logger: Any,
) -> str | None:
  parent_time = _compat("time", time)
  pending = runner._pending_tool_calls.get(tool_call_id)
  duration_ms = int((parent_time.time() - pending.started_at) * 1000) if pending is not None else 0
  result_entry = runner._make_result_entry(tool_call_id, result, error)
  server_for_tool = _compat("_server_for_tool", _sdk_runner_helpers.server_for_tool)
  extra_blocks = await runner._call_on_tool_result(
    ToolResultContext(
      tool_name=tool_name,
      tool_input=dict(tool_input),
      result=result,
      error=error,
      duration_ms=duration_ms,
      tool_call_id=tool_call_id,
      session_id=runner._session_id,
      server=server_for_tool(tool_name),
      result_entry=result_entry,
      skill_run_id=runner._skill_run_id,
      workspace_dir=runner._workspace_dir,
      batch_id=getattr(runner, "_batch_id", None),
    )
  )
  additional_context = runner._format_additional_context(
    tool_name=tool_name,
    result_entry=result_entry,
    extra_blocks=extra_blocks,
  )
  if additional_context:
    logger.info("[%s] Injecting additionalContext for %s", runner._sid, tool_name)
  return additional_context


__all__ = [
  "build_hook_additional_context",
  "build_prompt",
  "call_on_late_usage_event",
  "call_on_session_summary",
  "call_on_tool_result",
  "call_on_tool_timing",
  "call_on_usage",
  "context_surface_records",
  "format_additional_context",
  "make_result_entry",
  "normalize_context_surfaces",
]
