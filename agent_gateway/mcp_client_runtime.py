from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


_DEFAULT_LOGGER_NAME = "mcp.os.posix.utilities"
_DEFAULT_PG_FALLBACK_PREFIX = "Process group termination failed for PID "
_DEFAULT_PG_FALLBACK_MARKER = "falling back to simple terminate"
_DEFAULT_TERMINATE_FAILED_PREFIX = "Process termination failed for PID "
_DEFAULT_TERMINATE_FAILED_MARKER = "attempting force kill"
_DEFAULT_KILL_FAILED_PREFIX = "Failed to kill process "


class McpStdioTerminationFallbackFilter(logging.Filter):
  def __init__(
    self,
    *,
    logger_name: str = _DEFAULT_LOGGER_NAME,
    pg_fallback_prefix: str = _DEFAULT_PG_FALLBACK_PREFIX,
    pg_fallback_marker: str = _DEFAULT_PG_FALLBACK_MARKER,
    terminate_failed_prefix: str = _DEFAULT_TERMINATE_FAILED_PREFIX,
    terminate_failed_marker: str = _DEFAULT_TERMINATE_FAILED_MARKER,
    kill_failed_prefix: str = _DEFAULT_KILL_FAILED_PREFIX,
  ) -> None:
    super().__init__()
    self._logger_name = logger_name
    self._pg_fallback_prefix = pg_fallback_prefix
    self._pg_fallback_marker = pg_fallback_marker
    self._terminate_failed_prefix = terminate_failed_prefix
    self._terminate_failed_marker = terminate_failed_marker
    self._kill_failed_prefix = kill_failed_prefix

  def filter(self, record: logging.LogRecord) -> bool:
    if record.name != self._logger_name:
      return True
    message = record.getMessage()
    if (
      message.startswith(self._pg_fallback_prefix)
      and self._pg_fallback_marker in message
    ):
      return False
    if (
      message.startswith(self._terminate_failed_prefix)
      and self._terminate_failed_marker in message
    ):
      return False
    if message.startswith(self._kill_failed_prefix):
      exc_info = record.exc_info
      if exc_info is not None and isinstance(exc_info[1], ProcessLookupError):
        return False
    return True


@contextmanager
def suppress_mcp_stdio_termination_fallback_warnings(
  logger_factory: Callable[[str], logging.Logger] = logging.getLogger,
  filter_factory: Callable[..., logging.Filter] = McpStdioTerminationFallbackFilter,
  *,
  logger_name: str = _DEFAULT_LOGGER_NAME,
  pg_fallback_prefix: str = _DEFAULT_PG_FALLBACK_PREFIX,
  pg_fallback_marker: str = _DEFAULT_PG_FALLBACK_MARKER,
  terminate_failed_prefix: str = _DEFAULT_TERMINATE_FAILED_PREFIX,
  terminate_failed_marker: str = _DEFAULT_TERMINATE_FAILED_MARKER,
  kill_failed_prefix: str = _DEFAULT_KILL_FAILED_PREFIX,
):
  upstream_logger = logger_factory(logger_name)
  if (
    logger_name,
    pg_fallback_prefix,
    pg_fallback_marker,
    terminate_failed_prefix,
    terminate_failed_marker,
    kill_failed_prefix,
  ) == (
    _DEFAULT_LOGGER_NAME,
    _DEFAULT_PG_FALLBACK_PREFIX,
    _DEFAULT_PG_FALLBACK_MARKER,
    _DEFAULT_TERMINATE_FAILED_PREFIX,
    _DEFAULT_TERMINATE_FAILED_MARKER,
    _DEFAULT_KILL_FAILED_PREFIX,
  ):
    log_filter = filter_factory()
  else:
    log_filter = filter_factory(
      logger_name=logger_name,
      pg_fallback_prefix=pg_fallback_prefix,
      pg_fallback_marker=pg_fallback_marker,
      terminate_failed_prefix=terminate_failed_prefix,
      terminate_failed_marker=terminate_failed_marker,
      kill_failed_prefix=kill_failed_prefix,
    )
  upstream_logger.addFilter(log_filter)
  try:
    yield
  finally:
    upstream_logger.removeFilter(log_filter)


def consume_mcp_tool_call_result(task: asyncio.Task[Any], *, logger: logging.Logger) -> None:
  try:
    task.result()
  except asyncio.CancelledError:
    pass
  except Exception as exc:
    logger.debug("Late MCP tool call task finished after cancellation: %s", exc)


async def cancel_mcp_tool_call(
  task: asyncio.Task[Any],
  *,
  tool_name: str,
  reason: str,
  grace_seconds: float,
  consume_result: Callable[[asyncio.Task[Any]], None],
  current_task: Callable[[], asyncio.Task[Any] | None],
  logger: logging.Logger,
  shield: Callable[[asyncio.Future[Any]], asyncio.Future[Any]],
  wait_for: Callable[..., Any],
) -> None:
  if task.done():
    return
  task.cancel()
  try:
    await wait_for(
      shield(task),
      timeout=grace_seconds,
    )
  except asyncio.TimeoutError:
    logger.debug(
      "MCP tool %s cancellation still pending after %.1fs during %s; continuing",
      tool_name,
      grace_seconds,
      reason,
    )
    task.add_done_callback(consume_result)
  except asyncio.CancelledError:
    caller_task = current_task()
    if caller_task is not None and caller_task.cancelling():
      if not task.done():
        task.add_done_callback(consume_result)
      raise
    pass
  except Exception as exc:
    logger.debug("MCP tool %s cancellation raised during %s: %s", tool_name, reason, exc)


def extract_text(content: Any) -> str:
  if not content:
    return ""

  chunks: list[str] = []
  for item in content:
    if isinstance(item, dict):
      text = item.get("text")
    else:
      text = getattr(item, "text", None)
    if isinstance(text, str) and text.strip():
      chunks.append(text)
  return "\n".join(chunks).strip()


async def close_contexts(
  contexts: list[Any],
  *,
  close_timeout_seconds: float,
  logger: logging.Logger,
  suppress_warnings: Callable[[], Any],
  wait_for: Callable[..., Any],
) -> None:
  while contexts:
    ctx = contexts.pop()
    try:
      with suppress_warnings():
        await wait_for(
          ctx.__aexit__(None, None, None),
          timeout=close_timeout_seconds,
        )
    except asyncio.TimeoutError:
      logger.warning(
        "MCP context close timed out after %.1fs; continuing shutdown",
        close_timeout_seconds,
      )
    except asyncio.CancelledError as exc:
      logger.debug("MCP context close cancelled during cleanup: %s", exc)
    except Exception as exc:
      logger.debug("MCP context close failed: %s", exc)


def read_claude_config(
  config_path: Path | None,
  *,
  json_load: Callable[[Any], Any],
  logger: logging.Logger,
) -> dict[str, Any]:
  if config_path is None or not config_path.exists():
    return {}

  try:
    with open(config_path, "r", encoding="utf-8") as f:
      data = json_load(f)
  except Exception as exc:
    logger.warning("Failed to read %s: %s", config_path, exc)
    return {}

  return data if isinstance(data, dict) else {}


__all__ = [
  "McpStdioTerminationFallbackFilter",
  "cancel_mcp_tool_call",
  "close_contexts",
  "consume_mcp_tool_call_result",
  "extract_text",
  "read_claude_config",
  "suppress_mcp_stdio_termination_fallback_warnings",
]
