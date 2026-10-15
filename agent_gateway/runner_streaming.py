from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Iterable

from .providers import ModelInfo, StreamEvent
from .rate_limit import local_queue_left_at, locally_queued
from model_authority.thinking import EffortResolution, ThinkingLevel

# The stream watchdog is the containment layer's hung-peer bound on a provider
# stream (rule 8 of docs/engineering/fail-closed-local-guards.md). No profile varies
# it; only facts about the request do (thinking, and a transport whose first event
# follows a full local prefill). It fires only on *no progress* — every
# non-heartbeat provider event resets the clock — never on a turn's duration,
# so it cannot cut valid work. The turn runner retries a stalled stream up to
# STREAM_RETRY_MAX; direct stream consumers raise StreamStallError instead.
# The soft budget cannot end a hung request, because its ladder counts completed
# requests. Each turn's elapsed and last-progress seconds ride the `heartbeat`
# event, every retry rides `stream_retry`, and the terminal error names attempts.
STREAM_STALL_TIMEOUT = 60  # max seconds between stream progress events before watchdog cancels
STREAM_THINKING_STALL_TIMEOUT = 300  # extended-thinking turns can be quiet before first visible output
# A transport that emits nothing until its host has prefilled the whole prompt
# (ModelInfo.first_progress_after_prefill: local Ollama) gets a bound scaled from
# the entry's context_window, the largest prompt history_fits admits, at this floor
# rate. Prefill slows as the prompt grows and as the host is loaded. Measured
# 2026-10-10 on this Mac, qwen3.8:27b Q4 (Ollama server.log `prompt eval time`):
# 394 tok/s at 25.5k tokens; at 61k, 301 tok/s idle, 262 with one merge gate
# running and 190 with two. Per-token cost grows linearly with position, so the
# fit gives about 130 tok/s at the full 262144-token window idle and about 80
# under the heaviest load seen. 60 keeps margin. The cost is that a wedged local
# daemon is detected late (qwen: about 73 minutes per attempt).
LOCAL_PREFILL_TOKENS_PER_SECOND = 60


def thinking_level(enabled: bool) -> ThinkingLevel:
  return ThinkingLevel.HIGH if enabled else ThinkingLevel.NONE


def effective_stream_stall_timeout(
  stream_stall_timeout: float | None,
  *,
  config: Dict[str, Any],
  model_info: ModelInfo,
  max_tokens: int,
  effort_resolution: EffortResolution | None = None,
  observed_thinking: bool = False,
  stream_stall_timeout_default: float = STREAM_STALL_TIMEOUT,
  stream_thinking_stall_timeout_default: float = STREAM_THINKING_STALL_TIMEOUT,
) -> float:
  if stream_stall_timeout is not None:
    return float(stream_stall_timeout)
  legacy_effective = (
    effort_resolution is None
    and bool(config.get("thinking", str(config.get("effort", "high")) != "none"))
    and max_tokens >= 2048
    and model_info.supports_thinking
  )
  if (
    (effort_resolution is not None and effort_resolution.thinking_enabled_effective)
    or legacy_effective
    or observed_thinking
  ):
    bound = float(stream_thinking_stall_timeout_default)
  else:
    bound = float(stream_stall_timeout_default)
  if model_info.first_progress_after_prefill:
    bound = max(bound, model_info.context_window / LOCAL_PREFILL_TOKENS_PER_SECOND)
  return bound


class StreamStallError(RuntimeError):
  """A provider stream exceeded its no-progress containment deadline."""

  def __init__(self, stall_seconds: float) -> None:
    self.stall_seconds = stall_seconds
    super().__init__(
      f"Provider stream made no progress for {stall_seconds:g}s (stream stall)"
    )


async def _stream_with_stall_timeout(
  stream: AsyncIterator[StreamEvent],
  *,
  config: Dict[str, Any],
  model_info: ModelInfo,
  max_tokens: int,
  effort_resolution: EffortResolution | None = None,
) -> AsyncIterator[StreamEvent]:
  """Bound a blocked next event without imposing a total stream duration cap."""
  stall_timeout = effective_stream_stall_timeout(
    None,
    config=config,
    model_info=model_info,
    max_tokens=max_tokens,
    effort_resolution=effort_resolution,
    stream_stall_timeout_default=STREAM_STALL_TIMEOUT,
    stream_thinking_stall_timeout_default=STREAM_THINKING_STALL_TIMEOUT,
  )
  last_progress_at = time.monotonic()
  next_event: asyncio.Future[StreamEvent] | None = None
  try:
    while True:
      if next_event is None:
        next_event = asyncio.ensure_future(stream.__anext__())
      remaining = stall_timeout - (time.monotonic() - last_progress_at)
      done = set()
      if remaining > 0:
        done, _ = await asyncio.wait({next_event}, timeout=remaining)
      if not done:
        # The deadline is checked here before any ready event is consumed, so
        # back-to-back heartbeats (which are not progress) cannot outrun it.
        if locally_queued(next_event):
          # Queued for local rate-limit capacity: the request has not reached
          # the peer yet, so the hung-peer clock does not run.
          last_progress_at = time.monotonic()
          continue
        left_queue_at = local_queue_left_at(next_event)
        if left_queue_at is not None and left_queue_at > last_progress_at:
          # The queue ended after the last progress mark: the peer has been
          # silent only since the request left the queue.
          last_progress_at = left_queue_at
          continue
        next_event.cancel()
        raise StreamStallError(time.monotonic() - last_progress_at)
      try:
        event = next_event.result()
      except StopAsyncIteration:
        return
      finally:
        next_event = None
      if event.type != "heartbeat":
        last_progress_at = time.monotonic()
      yield event
  finally:
    if next_event is not None and not next_event.done():
      # The generator cannot be closed while its __anext__ is still running.
      next_event.cancel()
      await asyncio.gather(next_event, return_exceptions=True)
    close = getattr(stream, "aclose", None)
    if close is not None:
      await close()


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
