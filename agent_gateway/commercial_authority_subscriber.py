"""Resumable invalidation subscriber with an atomically persisted cursor."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import time
from typing import Callable
from uuid import UUID

from .commercial_authority_cache import (
  CommercialAuthorityInvalidation,
  CommercialAuthorityStateCache,
)


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
    events = page.get("events") if isinstance(page, dict) else None
    next_sequence = page.get("next_sequence") if isinstance(page, dict) else None
    high_water = page.get("high_water_sequence") if isinstance(page, dict) else None
    if (
      not isinstance(events, list)
      or type(next_sequence) is not int
      or type(high_water) is not int
      or next_sequence < cursor
      or high_water < next_sequence
    ):
      raise ValueError("commercial invalidation page is invalid")
    previous = cursor
    for raw in events:
      if not isinstance(raw, dict):
        raise ValueError("commercial invalidation event is invalid")
      if set(raw) != {
        "sequence_id", "environment", "kind", "commercial_account_id",
        "entitlement_revision", "context_id", "token_id", "occurred_at",
      } or raw.get("environment") not in {"dev", "staging", "prod"}:
        raise ValueError("commercial invalidation event schema is invalid")
      expected_environment = getattr(self._client, "environment", None)
      if expected_environment is not None and raw["environment"] != expected_environment:
        raise ValueError("commercial invalidation environment is invalid")
      if not isinstance(raw["occurred_at"], str) or not raw["occurred_at"]:
        raise ValueError("commercial invalidation occurrence time is invalid")
      try:
        occurred_at = datetime.fromisoformat(raw["occurred_at"].replace("Z", "+00:00"))
      except ValueError as error:
        raise ValueError("commercial invalidation occurrence time is invalid") from error
      if occurred_at.tzinfo is None:
        raise ValueError("commercial invalidation occurrence time is invalid")
      sequence = raw.get("sequence_id")
      if type(sequence) is not int or sequence <= previous:
        raise ValueError("commercial invalidation sequence is invalid")
      try:
        event = CommercialAuthorityInvalidation(
          kind=raw["kind"],
          commercial_account_id=raw["commercial_account_id"],
          entitlement_revision=raw["entitlement_revision"],
          context_id=UUID(raw["context_id"]) if raw.get("context_id") else None,
          token_id=UUID(raw["token_id"]) if raw.get("token_id") else None,
        )
      except (KeyError, TypeError, ValueError) as error:
        raise ValueError("commercial invalidation event is invalid") from error
      self._cache.apply_invalidation(event)
      previous = sequence
    if (events and previous != next_sequence) or (not events and next_sequence != cursor):
      raise ValueError("commercial invalidation page cursor mismatch")
    if not events and cursor < high_water:
      raise ValueError("commercial invalidation feed made no catch-up progress")
    if next_sequence != cursor:
      self._store_cursor(next_sequence)
    return next_sequence, high_water

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
