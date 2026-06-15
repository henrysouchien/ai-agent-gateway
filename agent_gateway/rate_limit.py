from __future__ import annotations

import asyncio
import os
import time


class AnthropicTokenBucket:
  def __init__(self, *, rpm: float | None = None, tpm: float | None = None) -> None:
    self._rpm_capacity = _positive_float_or_none(rpm)
    self._tpm_capacity = _positive_float_or_none(tpm)
    self._rpm_tokens = self._rpm_capacity
    self._tpm_tokens = self._tpm_capacity
    self._last_refill = time.monotonic()
    self._lock = asyncio.Lock()

  async def acquire(self, *, estimated_tokens: int = 0) -> None:
    request_tokens = 1.0 if self._rpm_capacity is not None else 0.0
    token_tokens = (
      min(float(max(0, int(estimated_tokens or 0))), self._tpm_capacity)
      if self._tpm_capacity is not None
      else 0.0
    )

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
          return
      await asyncio.sleep(wait_seconds)

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


__all__ = ["AnthropicTokenBucket", "get_global_token_bucket"]
