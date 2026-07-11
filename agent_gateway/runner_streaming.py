from __future__ import annotations

from typing import Any, Dict, Iterable

from .providers import ModelInfo, ThinkingLevel

STREAM_STALL_TIMEOUT = 60  # max seconds between stream progress events before watchdog cancels
STREAM_THINKING_STALL_TIMEOUT = 300  # extended-thinking turns can be quiet before first visible output


def thinking_level(enabled: bool) -> ThinkingLevel:
  return ThinkingLevel.HIGH if enabled else ThinkingLevel.NONE


def effective_stream_stall_timeout(
  stream_stall_timeout: float | None,
  *,
  config: Dict[str, Any],
  model_info: ModelInfo,
  max_tokens: int,
  observed_thinking: bool = False,
  stream_stall_timeout_default: float = STREAM_STALL_TIMEOUT,
  stream_thinking_stall_timeout_default: float = STREAM_THINKING_STALL_TIMEOUT,
) -> float:
  if stream_stall_timeout is not None:
    return float(stream_stall_timeout)
  if (
    (bool(config.get("thinking", True)) or observed_thinking)
    and max_tokens >= 2048
    and model_info.supports_thinking
  ):
    return float(stream_thinking_stall_timeout_default)
  return float(stream_stall_timeout_default)


def observed_thinking_in_messages(
  messages: Iterable[Dict[str, Any]],
  *,
  model_info: ModelInfo,
) -> bool:
  for message in messages:
    if str(message.get("role") or "").strip().lower() != "assistant":
      continue
    provider = str(message.get("provider") or "").strip().lower()
    if provider and provider != str(model_info.provider or "").strip().lower():
      continue
    model = str(message.get("model") or "").strip()
    if model and model != str(model_info.id or "").strip():
      continue
    content = message.get("content")
    if not isinstance(content, list):
      continue
    if any(
      isinstance(block, dict) and str(block.get("type") or "") in {"thinking", "redacted_thinking"}
      for block in content
    ):
      return True
  return False


def classify_guard_outcome(
  guard_reason: tuple[str, str] | None,
  attempt: int,
  max_attempts: int,
) -> tuple[str, str, str]:
  """Classify guard outcome into (action, guard_error, guard_kind)."""
  if not guard_reason:
    return ("not_guard", "", "")
  guard_kind, guard_message = guard_reason
  guard_error = f"Stream watchdog: {guard_message}"
  if guard_kind == "stall" and attempt < max_attempts:
    return ("retry", guard_error, guard_kind)
  return ("abort", guard_error, guard_kind)
