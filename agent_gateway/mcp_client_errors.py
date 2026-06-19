from __future__ import annotations

import asyncio
from typing import Any, Callable


def classify_exception(exc: Exception, msg: str) -> str:
  lower = msg.lower()
  if isinstance(exc, asyncio.TimeoutError) or "timeout" in lower or "timed out" in lower:
    return "timeout"
  if "connection" in lower or "refused" in lower:
    return "connection_error"
  return "unknown"


def classify_mcp_error(message: str) -> str:
  lower = message.lower()
  if "not found" in lower or "no filing" in lower or "no data" in lower:
    return "not_found"
  if "parse" in lower or "invalid" in lower or "malformed" in lower:
    return "parse_error"
  if "timeout" in lower or "timed out" in lower:
    return "timeout"
  return "unknown"


def startup_failure_from_exception(
  exc: BaseException,
  *,
  is_retryable_stdio_connect_error: Callable[[BaseException], bool],
) -> dict[str, Any]:
  message = str(exc).strip() or type(exc).__name__
  error_type = type(exc).__name__
  lower = message.lower()

  if isinstance(exc, asyncio.TimeoutError):
    return {
      "category": "transient_timeout",
      "retryable": True,
      "message": message,
      "error_type": error_type,
    }
  if is_retryable_stdio_connect_error(exc):
    return {
      "category": "transient_transport",
      "retryable": True,
      "message": message,
      "error_type": error_type,
    }
  if isinstance(exc, ValueError) and (
    "missing command" in lower
    or "missing url" in lower
    or "unsupported type" in lower
  ):
    return {
      "category": "config_error",
      "retryable": False,
      "message": message,
      "error_type": error_type,
    }
  return {
    "category": "startup_error",
    "retryable": False,
    "message": message,
    "error_type": error_type,
  }


__all__ = [
  "classify_exception",
  "classify_mcp_error",
  "startup_failure_from_exception",
]
