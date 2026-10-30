from __future__ import annotations

from typing import Any, Dict, Iterable

from hank_model_transport.providers import ModelInfo
from model_authority.thinking import ThinkingLevel
from hank_model_transport.streaming import (
  STREAM_STALL_TIMEOUT,
  STREAM_THINKING_STALL_TIMEOUT,
  LOCAL_PREFILL_TOKENS_PER_SECOND,
  StreamStallError,
  effective_stream_stall_timeout,
  _stream_with_stall_timeout,
)



def thinking_level(enabled: bool) -> ThinkingLevel:
  return ThinkingLevel.HIGH if enabled else ThinkingLevel.NONE




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
  return ("abort", f"{guard_error} (attempt {attempt + 1} of {max_attempts + 1})", guard_kind)
