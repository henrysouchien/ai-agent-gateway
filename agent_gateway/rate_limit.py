from __future__ import annotations

import asyncio
import os
import time
import weakref


# Tasks currently waiting in a local bucket, and when each last left it. A stream
# consumer's no-progress watchdog bounds a hung provider peer; time a request
# spends queued here, before it is sent, is the gateway's own pacing, so the
# watchdog reads these rather than counting the queue as a stall.
_LOCALLY_QUEUED: set[asyncio.Task] = set()
_LOCAL_QUEUE_LEFT_AT: weakref.WeakKeyDictionary[asyncio.Task, float] = (
  weakref.WeakKeyDictionary()
)


def locally_queued(task: asyncio.Future | None) -> bool:
  """Whether ``task`` is waiting for local rate-limit capacity right now."""
  return task is not None and task in _LOCALLY_QUEUED


def local_queue_left_at(task: asyncio.Future | None) -> float | None:
  """Monotonic time ``task`` last left the local rate-limit queue, if ever."""
  if not isinstance(task, asyncio.Task):
    return None
  return _LOCAL_QUEUE_LEFT_AT.get(task)


class AnthropicTokenBucket:
  def __init__(self, *, rpm: float | None = None, tpm: float | None = None) -> None:
    self._rpm_capacity = _positive_float_or_none(rpm)
    self._tpm_capacity = _positive_float_or_none(tpm)
    self._rpm_tokens = self._rpm_capacity
    self._tpm_tokens = self._tpm_capacity
    self._last_refill = time.monotonic()
    self._lock = asyncio.Lock()

  async def acquire(self, *, estimated_tokens: int = 0) -> float:
    """Wait until the operator's rpm/tpm allow the call; return seconds waited.

    The bucket never refuses. The wait is a measurement: the Anthropic
    provider reports it as ``provider_wait_ms`` on the call's usage.
    """
    started = time.monotonic()
    request_tokens = 1.0 if self._rpm_capacity is not None else 0.0
    token_tokens = (
      min(float(max(0, int(estimated_tokens or 0))), self._tpm_capacity)
      if self._tpm_capacity is not None
      else 0.0
    )

    task = asyncio.current_task()
    if task is not None:
      _LOCALLY_QUEUED.add(task)
    try:
      while True:
        async with self._lock:
          self._refill_locked()
          rpm_wait = self._wait_seconds(
            needed=request_tokens,
            available=self._rpm_tokens,
            capacity=self._rpm_capacity,
          )
          tpm_wait = self._wait_seconds(
            needed=token_tokens,
            available=self._tpm_tokens,
            capacity=self._tpm_capacity,
          )
          wait_seconds = max(rpm_wait, tpm_wait)
          if wait_seconds <= 0:
            if self._rpm_tokens is not None:
              self._rpm_tokens -= request_tokens
            if self._tpm_tokens is not None:
              self._tpm_tokens -= token_tokens
            return time.monotonic() - started
        await asyncio.sleep(wait_seconds)
    finally:
      if task is not None:
        _LOCALLY_QUEUED.discard(task)
        _LOCAL_QUEUE_LEFT_AT[task] = time.monotonic()

  def _refill_locked(self) -> None:
    now = time.monotonic()
    elapsed = max(0.0, now - self._last_refill)
    self._last_refill = now
    if self._rpm_capacity is not None and self._rpm_tokens is not None:
      self._rpm_tokens = min(
        self._rpm_capacity,
        self._rpm_tokens + elapsed * (self._rpm_capacity / 60.0),
      )
    if self._tpm_capacity is not None and self._tpm_tokens is not None:
      self._tpm_tokens = min(
        self._tpm_capacity,
        self._tpm_tokens + elapsed * (self._tpm_capacity / 60.0),
      )

  @staticmethod
  def _wait_seconds(
    *,
    needed: float,
    available: float | None,
    capacity: float | None,
  ) -> float:
    if capacity is None or available is None or needed <= available:
      return 0.0
    return (needed - available) / (capacity / 60.0)


_GLOBAL_BUCKET: AnthropicTokenBucket | None = None
_GLOBAL_BUCKET_KEY: tuple[str, str] | None = None


def get_global_token_bucket() -> AnthropicTokenBucket | None:
  global _GLOBAL_BUCKET, _GLOBAL_BUCKET_KEY

  rpm_raw = os.environ.get("ANTHROPIC_GLOBAL_RPM", "").strip()
  tpm_raw = os.environ.get("ANTHROPIC_GLOBAL_TPM", "").strip()
  key = (rpm_raw, tpm_raw)
  if not rpm_raw and not tpm_raw:
    _GLOBAL_BUCKET = None
    _GLOBAL_BUCKET_KEY = key
    return None

  if _GLOBAL_BUCKET is None or _GLOBAL_BUCKET_KEY != key:
    _GLOBAL_BUCKET = AnthropicTokenBucket(
      rpm=_parse_limit(rpm_raw, "ANTHROPIC_GLOBAL_RPM") if rpm_raw else None,
      tpm=_parse_limit(tpm_raw, "ANTHROPIC_GLOBAL_TPM") if tpm_raw else None,
    )
    _GLOBAL_BUCKET_KEY = key
  return _GLOBAL_BUCKET


def _parse_limit(raw: str, env_name: str) -> float:
  try:
    value = float(raw)
  except ValueError as exc:
    raise ValueError(f"{env_name} must be a positive number") from exc
  if value <= 0:
    raise ValueError(f"{env_name} must be positive")
  return value


def _positive_float_or_none(value: float | None) -> float | None:
  if value is None:
    return None
  parsed = float(value)
  if parsed <= 0:
    raise ValueError("token bucket limits must be positive")
  return parsed


__all__ = [
  "AnthropicTokenBucket",
  "get_global_token_bucket",
  "local_queue_left_at",
  "locally_queued",
]
