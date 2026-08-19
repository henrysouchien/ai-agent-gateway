"""Resumable invalidation subscriber with an atomically persisted cursor."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Callable

from .commercial_authority_cache import (
  CommercialAuthorityStateCache,
)
from .commercial_authority_feed import decode_commercial_authority_feed_v1


class CommercialAuthoritySubscriber:
  def __init__(
    self,
    *,
    client,
    cache: CommercialAuthorityStateCache,
    cursor_path: Path,
    monotonic: Callable[[], float] = time.monotonic,
  ) -> None:
    self._client = client
    self._cache = cache
    self._path = cursor_path
    self._monotonic = monotonic
    self._caught_up = False
    self._last_success_at: float | None = None
    self._last_error_type: str | None = None
    self._consecutive_failures = 0

  @property
  def cache(self) -> CommercialAuthorityStateCache:
    return self._cache

  async def run_forever(self, stop: asyncio.Event) -> None:
    cursor = self._load_cursor()
    while not stop.is_set():
      try:
        next_sequence, high_water = await self._consume_page(cursor)
        if next_sequence != cursor:
          cursor = next_sequence
        self._caught_up = next_sequence >= high_water
        self._record_success()
        if self._caught_up:
          await self._wait(stop, 0.25)
      except Exception as exc:
        self._record_failure(exc)
        await self._wait(stop, 1.0)

  async def catch_up(self) -> None:
    try:
      cursor = self._load_cursor()
      while True:
        cursor, high_water = await self._consume_page(cursor)
        if cursor >= high_water:
          self._caught_up = True
          self._record_success()
          return
    except Exception as exc:
      self._record_failure(exc)
      raise

  def health(self, *, max_staleness_seconds: float) -> dict[str, object]:
    if max_staleness_seconds <= 0:
      raise ValueError("commercial authority max staleness must be positive")
    last_success_at = self._last_success_at
    age = None if last_success_at is None else max(0.0, self._monotonic() - last_success_at)
    ok = bool(
      self._caught_up
      and age is not None
      and age <= max_staleness_seconds
      and self._last_error_type is None
    )
    return {
      "ok": ok,
      "caught_up": self._caught_up,
      "last_success_age_seconds": None if age is None else round(age, 3),
      "max_staleness_seconds": max_staleness_seconds,
      "consecutive_failures": self._consecutive_failures,
      "last_error_type": self._last_error_type,
    }

  def _record_success(self) -> None:
    self._last_success_at = self._monotonic()
    self._last_error_type = None
    self._consecutive_failures = 0

  def _record_failure(self, exc: Exception) -> None:
    self._last_error_type = type(exc).__name__
    self._consecutive_failures += 1

  async def _consume_page(self, cursor: int) -> tuple[int, int]:
    page = await asyncio.to_thread(self._client.fetch, cursor)
    decoded = decode_commercial_authority_feed_v1(
      page,
      cursor=cursor,
      expected_environment=getattr(self._client, "environment", None),
    )
    for invalidation in decoded.invalidations:
      self._cache.apply_invalidation(invalidation)
    if decoded.next_sequence != cursor:
      self._store_cursor(decoded.next_sequence)
    return decoded.next_sequence, decoded.high_water_sequence

  def _load_cursor(self) -> int:
    if not self._path.exists():
      return 0
    value = json.loads(self._path.read_text())
    if (
      not isinstance(value, dict)
      or set(value) != {"sequence"}
      or type(value.get("sequence")) is not int
      or value["sequence"] < 0
    ):
      raise ValueError("commercial invalidation cursor file is invalid")
    return value["sequence"]

  def _store_cursor(self, sequence: int) -> None:
    self._path.parent.mkdir(parents=True, exist_ok=True)
    temporary = self._path.with_suffix(self._path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
      handle.write(json.dumps({"sequence": sequence}, separators=(",", ":")))
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, self._path)

  @staticmethod
  async def _wait(stop: asyncio.Event, seconds: float) -> None:
    try:
      await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
      pass


__all__ = ["CommercialAuthoritySubscriber"]
