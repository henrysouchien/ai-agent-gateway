from __future__ import annotations

import asyncio
import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from .autonomous import RunOutput


log = logging.getLogger("agent_gateway.retry")

Outcome = Literal["success", "transient", "permanent"]
RetryHook = Callable[[int, str, float], Awaitable[None] | None]

_DEFAULT_BACKOFF_STEPS = [30.0, 60.0, 120.0, 300.0, 600.0]
_PERMANENT_STATUS_CODES = {400, 401, 403, 404}
_TRANSIENT_EXCEPTION_NAMES = {
  "APIConnectionError",
  "APITimeoutError",
  "ConnectionError",
  "StreamError",
  "TimeoutError",
  "TransportError",
}
_PERMANENT_EXCEPTION_NAMES = {
  "AttributeError",
  "FileNotFoundError",
  "KeyError",
  "TypeError",
  "ValueError",
}
_STATUS_CODE_RE = re.compile(r"\b(400|401|403|404|429|5\d\d)\b")
_TRANSIENT_ERROR_PATTERNS = (
  re.compile(r"\b(?:429|5\d\d)\b", re.IGNORECASE),
  re.compile(r"\btoo many requests\b", re.IGNORECASE),
  re.compile(r"\brate(?:\s+limit(?:ed|ing)?|\s+limited)\b", re.IGNORECASE),
  re.compile(r"\boverload(?:ed)?\b", re.IGNORECASE),
  re.compile(r"\bservice unavailable\b", re.IGNORECASE),
  re.compile(r"\bserver error\b", re.IGNORECASE),
  re.compile(r"\binternal server error\b", re.IGNORECASE),
  re.compile(r"\b(?:api)?connectionerror\b", re.IGNORECASE),
  re.compile(r"\bapitimeouterror\b", re.IGNORECASE),
  re.compile(r"\btransporterror\b", re.IGNORECASE),
  re.compile(r"\bstreamerror\b", re.IGNORECASE),
  re.compile(r"\btimeout(?:error)?\b", re.IGNORECASE),
  re.compile(r"\btimed out\b", re.IGNORECASE),
  re.compile(r"\bconnection (?:reset|aborted|refused|closed)\b", re.IGNORECASE),
)
_PERMANENT_ERROR_PATTERNS = (
  re.compile(r"\bunauthorized\b", re.IGNORECASE),
  re.compile(r"\bforbidden\b", re.IGNORECASE),
  re.compile(r"\bauth(?:entication|orization)?\b", re.IGNORECASE),
  re.compile(r"\binvalid api key\b", re.IGNORECASE),
  re.compile(r"\bbad request\b", re.IGNORECASE),
  re.compile(r"\bnot found\b", re.IGNORECASE),
  re.compile(r"\b(?:ValueError|TypeError|FileNotFoundError|KeyError|AttributeError)\b", re.IGNORECASE),
)


@dataclass
class RetryConfig:
  max_retries: int = 3
  backoff_steps: list[float] = field(default_factory=lambda: list(_DEFAULT_BACKOFF_STEPS))
  retry_on_timeout: bool = True
  on_retry: RetryHook | None = None

  def __post_init__(self) -> None:
    self.max_retries = max(0, int(self.max_retries))
    self.backoff_steps = [float(step) for step in self.backoff_steps]
    self.retry_on_timeout = bool(self.retry_on_timeout)


def _status_code_from_exception(exc: Exception) -> int | None:
  status_code = getattr(exc, "status_code", None)
  if status_code is None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
  if isinstance(status_code, int):
    return status_code
  if isinstance(status_code, str) and status_code.isdigit():
    return int(status_code)
  return None


def _classify_status_code(status_code: int | None) -> Outcome | None:
  if status_code in _PERMANENT_STATUS_CODES:
    return "permanent"
  if status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600):
    return "transient"
  return None


def _status_code_from_text(text: str | None) -> int | None:
  if not text:
    return None
  match = _STATUS_CODE_RE.search(text)
  if match is None:
    return None
  return int(match.group(1))


def _classify_error_text(text: str | None) -> Outcome | None:
  if not text:
    return None

  status_outcome = _classify_status_code(_status_code_from_text(text))
  if status_outcome is not None:
    return status_outcome

  for pattern in _PERMANENT_ERROR_PATTERNS:
    if pattern.search(text):
      return "permanent"

  for pattern in _TRANSIENT_ERROR_PATTERNS:
    if pattern.search(text):
      return "transient"

  return None


def classify_outcome(exc: Exception | None, output: RunOutput | None) -> Outcome:
  if output is not None:
    if output.operator_paused:
      return "permanent"
    if not output.error and not output.timed_out and not output.budget_exceeded and not output.max_turns_reached:
      return "success"
    if output.budget_exceeded or output.max_turns_reached:
      return "permanent"
    if output.timed_out:
      return "transient"

  if exc is not None:
    status_outcome = _classify_status_code(_status_code_from_exception(exc))
    if status_outcome is not None:
      return status_outcome

    exc_name = type(exc).__name__
    if exc_name in _TRANSIENT_EXCEPTION_NAMES:
      return "transient"
    if exc_name in _PERMANENT_EXCEPTION_NAMES:
      return "permanent"

  error_outcome = _classify_error_text(output.error if output is not None else None)
  if error_outcome is not None:
    return error_outcome

  return "transient"


def _failure_description(exc: Exception | None, output: RunOutput | None) -> str:
  if output is not None:
    if output.error:
      return output.error
    if output.budget_exceeded:
      return "Run budget exceeded"
    if output.max_turns_reached:
      return "Run reached max turns"
    if output.operator_paused:
      return "Run paused by operator"
    if output.timed_out:
      return "Run timed out"
  if exc is not None:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
  return "Unknown autonomous run failure"


def _exception_to_run_output(exc: Exception | None) -> RunOutput:
  return RunOutput(
    response="",
    tools_used=[],
    usage={},
    error=_failure_description(exc, None),
    timed_out=bool(exc is not None and type(exc).__name__ == "TimeoutError"),
  )


def _retry_delay(backoff_steps: list[float], retry_number: int) -> float:
  if not backoff_steps:
    return 0.0
  idx = min(max(retry_number - 1, 0), len(backoff_steps) - 1)
  return float(backoff_steps[idx])


async def _invoke_retry_hook(
  callback: RetryHook | None,
  attempt_number: int,
  error_description: str,
  delay: float,
) -> None:
  if callback is None:
    return
  try:
    result = callback(attempt_number, error_description, delay)
    if inspect.isawaitable(result):
      await result
  except Exception as exc:
    log.warning("Retry callback failed (non-fatal): %s", exc)


async def run_autonomous_with_retry(
  run_fn: Callable[..., Awaitable[RunOutput]],
  *args: Any,
  retry_config: RetryConfig | None = None,
  **kwargs: Any,
) -> RunOutput:
  config = retry_config or RetryConfig()

  for attempt_idx in range(config.max_retries + 1):
    exc: Exception | None = None
    output: RunOutput | None = None

    try:
      output = await run_fn(*args, **kwargs)
    except Exception as caught:
      exc = caught

    outcome = classify_outcome(exc, output)
    result = output if output is not None else _exception_to_run_output(exc)

    if outcome == "success":
      return result

    if outcome == "permanent":
      return result

    if output is not None and output.timed_out and not config.retry_on_timeout:
      return result

    if attempt_idx >= config.max_retries:
      return result

    retry_number = attempt_idx + 1
    next_attempt_number = attempt_idx + 2
    delay = _retry_delay(config.backoff_steps, retry_number)
    await _invoke_retry_hook(config.on_retry, next_attempt_number, _failure_description(exc, output), delay)
    if delay > 0:
      await asyncio.sleep(delay)

  raise RuntimeError("Retry loop exhausted without returning a RunOutput")


__all__ = ["RetryConfig", "classify_outcome", "run_autonomous_with_retry"]
