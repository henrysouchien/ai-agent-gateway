from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .providers.base import truncate_to_last_compaction

MODEL_CONTEXT_LIMIT = 200_000
CONTEXT_WARNING_PCT = 80
# Server-side compaction fires when submitted input crosses the trigger. A fixed
# trigger calibrated to the old 200k-window era fires on every turn once the
# static system+tool surface alone exceeds it on a large-window model (e.g. 1M),
# so the API re-compacts each request for no benefit. Scale the trigger and the
# context-usage warning to the model's real window instead.
COMPACTION_TRIGGER_PCT = 80


def estimate_tokens(text: str) -> int:
  """Rough token estimate: ~4 chars per token for English text + JSON overhead."""
  return max(1, len(text) // 4)


@dataclass(frozen=True)
class TokenEstimateSnapshot:
  system_text: str
  messages_text: str
  tools_text: str
  system_chars: int
  tools_chars: int
  est_system_tokens: int
  est_messages_tokens: int
  est_tools_tokens: int
  est_total_tokens: int
  message_count: int
  tool_count: int


@dataclass(frozen=True)
class TokenBreakdownSnapshot:
  input_tokens: int
  est_system_tokens: int
  est_tools_tokens: int
  est_messages_tokens: int
  pct_system: int
  pct_tools: int
  pct_messages: int


def system_prompt_estimate_text(system_prompt: Any) -> str:
  if isinstance(system_prompt, list):
    return "\n\n".join(text for text, _should_cache in system_prompt if text)
  return system_prompt or ""


def token_estimate_snapshot(
  *,
  system_text: str,
  messages: list[dict[str, Any]],
  tools: list[dict[str, Any]],
) -> TokenEstimateSnapshot:
  messages_text = json.dumps(truncate_to_last_compaction(messages), default=str)
  tools_text = json.dumps(tools, default=str) if tools else ""
  est_system = estimate_tokens(system_text)
  est_messages = estimate_tokens(messages_text)
  est_tools = estimate_tokens(tools_text) if tools_text else 0
  return TokenEstimateSnapshot(
    system_text=system_text,
    messages_text=messages_text,
    tools_text=tools_text,
    system_chars=len(system_text),
    tools_chars=len(tools_text),
    est_system_tokens=est_system,
    est_messages_tokens=est_messages,
    est_tools_tokens=est_tools,
    est_total_tokens=est_system + est_messages + est_tools,
    message_count=len(messages),
    tool_count=len(tools),
  )


def token_breakdown_snapshot(
  *,
  input_tokens: int,
  system_chars: int,
  tools_chars: int,
  messages: list[dict[str, Any]],
) -> TokenBreakdownSnapshot | None:
  if input_tokens <= 0:
    return None
  messages_chars = len(json.dumps(messages, default=str))
  total_chars = system_chars + tools_chars + messages_chars
  if total_chars <= 0:
    return None
  pct_system = round(system_chars / total_chars * 100)
  pct_tools = round(tools_chars / total_chars * 100)
  pct_messages = round(messages_chars / total_chars * 100)
  tok_system = round(input_tokens * system_chars / total_chars)
  tok_tools = round(input_tokens * tools_chars / total_chars)
  return TokenBreakdownSnapshot(
    input_tokens=input_tokens,
    est_system_tokens=tok_system,
    est_tools_tokens=tok_tools,
    est_messages_tokens=input_tokens - tok_system - tok_tools,
    pct_system=pct_system,
    pct_tools=pct_tools,
    pct_messages=pct_messages,
  )


def model_context_window(model_info: Any) -> int:
  window = getattr(model_info, "context_window", None)
  return int(window) if window else MODEL_CONTEXT_LIMIT


def effective_compaction_trigger(compaction_trigger: int | None, model_info: Any) -> int | None:
  """Scale a fixed compaction trigger up to the model's real context window.

  Keeps the configured value as a floor (never compacts later than legacy on a
  small window) but raises it to ``COMPACTION_TRIGGER_PCT`` of a larger window so
  it only fires near genuine capacity instead of on every turn once the static
  system+tool surface alone already exceeds the legacy trigger.
  """
  if compaction_trigger is None:
    return None
  window_trigger = int(model_context_window(model_info) * COMPACTION_TRIGGER_PCT / 100)
  return max(window_trigger, int(compaction_trigger))
