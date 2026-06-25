from __future__ import annotations

import inspect
from typing import Any, Callable


def call_tool_timing_hook(
  on_tool_timing: Callable[..., None] | None,
  *,
  accepts_user_id: bool,
  accepts_context_surfaces: bool,
  session_id: str,
  log_session_id: str,
  user_id: str,
  context_surfaces: list[dict[str, Any]] | None,
  tool_name: str,
  server: str | None,
  duration_ms: int,
  is_error: bool,
  result_bytes: int,
  logger: Any,
  accepts_tool_call_id: bool = False,
  accepts_request_id: bool = False,
  tool_call_id: str | None = None,
  request_id: str | None = None,
) -> None:
  if on_tool_timing is None:
    return
  try:
    kwargs: dict[str, Any] = {}
    if accepts_context_surfaces:
      kwargs["context_surfaces"] = context_surfaces or []
    if accepts_tool_call_id:
      kwargs["tool_call_id"] = tool_call_id
    if accepts_request_id:
      kwargs["request_id"] = request_id
    if accepts_user_id:
      on_tool_timing(
        session_id,
        tool_name,
        server,
        duration_ms,
        is_error,
        result_bytes,
        user_id=user_id,
        **kwargs,
      )
    else:
      on_tool_timing(
        session_id,
        tool_name,
        server,
        duration_ms,
        is_error,
        result_bytes,
        **kwargs,
      )
  except Exception as exc:
    logger.warning("[%s] on_tool_timing hook failed (non-fatal): %s", log_session_id, exc)


def call_metric_hook(
  on_metric: Callable[[str, int], None] | None,
  *,
  name: str,
  value: int,
  log_session_id: str,
  logger: Any,
) -> None:
  if on_metric is None:
    return
  try:
    on_metric(name, value)
  except Exception as exc:
    logger.warning("[%s] metric hook failed (non-fatal): %s", log_session_id, exc)


async def call_tool_result_hook(
  on_tool_result: Callable[[Any], Any] | None,
  ctx: Any,
  *,
  log_session_id: str,
  logger: Any,
) -> list[dict[str, Any]]:
  if on_tool_result is None:
    return []
  try:
    extra_blocks = on_tool_result(ctx)
    if inspect.isawaitable(extra_blocks):
      extra_blocks = await extra_blocks
  except Exception as exc:
    logger.warning("[%s] on_tool_result hook failed (non-fatal): %s", log_session_id, exc)
    return []
  if not extra_blocks:
    return []
  if isinstance(extra_blocks, list):
    return [block for block in extra_blocks if isinstance(block, dict)]
  return []


async def call_before_stream_complete_hook(
  on_before_stream_complete: Callable[..., Any] | None,
  event_log: Any,
  terminal_event: dict[str, Any] | None,
  *,
  log_session_id: str,
  logger: Any,
) -> None:
  if on_before_stream_complete is None:
    return
  try:
    params = inspect.signature(on_before_stream_complete).parameters
    accepts_terminal_event = (
      any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values())
      or "terminal_event" in params
      or len([
        param
        for param in params.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
      ]) >= 2
    )
    if isinstance(terminal_event, dict) and terminal_event.get("type") != "stream_complete" and not accepts_terminal_event:
      return
    if accepts_terminal_event:
      result = on_before_stream_complete(event_log, terminal_event)
    else:
      result = on_before_stream_complete(event_log)
    if inspect.isawaitable(result):
      await result
  except Exception as exc:
    logger.warning("[%s] on_before_stream_complete hook failed (non-fatal): %s", log_session_id, exc)
