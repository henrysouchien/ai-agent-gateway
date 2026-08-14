"""Gateway-lifetime ownership and admission bounds for learning fork tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import json
import logging
import math
import os
from typing import Any, Literal

from .fork_ledger import (
  ForkAdmissionBudgetExceeded,
  ForkAdmissionQuotaExceeded,
  ForkLedger,
)


log = logging.getLogger("agent_gateway.fork_task_registry")

DEFAULT_LEARN_FORK_BUDGET_USD = Decimal("2.00")
DEFAULT_LEARN_FORK_DAILY_BUDGET_USD = Decimal("10.00")
DEFAULT_LEARN_FORK_DAILY_INVOCATION_QUOTA = 12
DEFAULT_LEARN_FORK_GLOBAL_CONCURRENCY = 2
DEFAULT_FORK_HANDOFF_MAX_BYTES = 33_554_432
DEFAULT_FORK_SHUTDOWN_TIMEOUT_SECONDS = 120.0

ForkSkipReason = Literal[
  "disabled",
  "session_cap",
  "global_concurrency",
  "handoff_too_large",
  "daily_invocation_quota",
  "daily_budget",
  "shutting_down",
]
ForkSpawn = Callable[[str, Any], Awaitable[Decimal | int | float | str]]
TelemetrySink = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class ForkLaunchDecision:
  launched: bool
  fork_id: str
  skip_reason: ForkSkipReason | None = None
  retained_bytes: int | None = None


def _positive_env_int(name: str, default: int) -> int:
  raw = os.getenv(name)
  if raw is None:
    return default
  try:
    value = int(raw)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{name} must be a positive integer") from exc
  if value <= 0:
    raise ValueError(f"{name} must be a positive integer")
  return value


def _positive_env_decimal(name: str, default: Decimal) -> Decimal:
  raw = os.getenv(name)
  if raw is None:
    return default
  try:
    value = Decimal(raw)
  except Exception as exc:
    raise ValueError(f"{name} must be finite and positive") from exc
  if not value.is_finite() or value <= 0:
    raise ValueError(f"{name} must be finite and positive")
  return value


def learn_fork_enabled(*, owner_operated_interactive: bool = False) -> bool:
  raw = os.getenv("HANK_LEARN_FORK_ENABLED")
  if raw is None:
    configured = True
  else:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
      configured = True
    elif normalized in {"0", "false", "no", "off"}:
      configured = False
    else:
      raise ValueError(
        "HANK_LEARN_FORK_ENABLED must be a boolean "
        "(1/0, true/false, yes/no, or on/off)"
      )
  return owner_operated_interactive and configured


def learn_fork_budget_usd() -> Decimal:
  return _positive_env_decimal(
    "HANK_LEARN_FORK_BUDGET_USD",
    DEFAULT_LEARN_FORK_BUDGET_USD,
  )


def learn_fork_daily_budget_usd() -> Decimal:
  return _positive_env_decimal(
    "HANK_LEARN_FORK_DAILY_BUDGET_USD",
    DEFAULT_LEARN_FORK_DAILY_BUDGET_USD,
  )


def learn_fork_daily_invocation_quota() -> int:
  return _positive_env_int(
    "HANK_LEARN_FORK_DAILY_INVOCATION_QUOTA",
    DEFAULT_LEARN_FORK_DAILY_INVOCATION_QUOTA,
  )


def learn_fork_global_concurrency() -> int:
  return _positive_env_int(
    "HANK_LEARN_FORK_GLOBAL_CONCURRENCY",
    DEFAULT_LEARN_FORK_GLOBAL_CONCURRENCY,
  )


def fork_handoff_max_bytes() -> int:
  return _positive_env_int(
    "HANK_FORK_HANDOFF_MAX_BYTES",
    DEFAULT_FORK_HANDOFF_MAX_BYTES,
  )


def _retained_size(value: Any, limit: int) -> int:
  """Return deterministic JSON UTF-8 bytes, stopping once the guard is crossed."""

  retained_payload = getattr(value, "_fork_retained_payload", None)
  if retained_payload is not None:
    value = retained_payload
  remaining = limit
  active: set[int] = set()
  stack: list[tuple[Any, bool]] = [(value, False)]
  used = 0
  while stack:
    item, leaving = stack.pop()
    if leaving:
      active.discard(id(item))
      continue
    if item is None:
      size = 4
    elif isinstance(item, bool):
      size = 5
    elif isinstance(item, str):
      size = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
    elif isinstance(item, int):
      size = len(str(item))
    elif isinstance(item, float):
      if not math.isfinite(item):
        raise ValueError("fork handoff contains a non-finite number")
      size = len(repr(item))
    elif isinstance(item, Mapping):
      identity = id(item)
      if identity in active:
        raise ValueError("fork handoff contains a cycle")
      active.add(identity)
      size = 2 + max(0, len(item) - 1)
      stack.append((item, True))
      for key, child in reversed(tuple(item.items())):
        stack.append((child, False))
        stack.append((str(key), False))
    elif isinstance(item, (Sequence, set, frozenset)):
      identity = id(item)
      if identity in active:
        raise ValueError("fork handoff contains a cycle")
      active.add(identity)
      size = 2 + max(0, len(item) - 1)
      stack.append((item, True))
      stack.extend((child, False) for child in reversed(tuple(item)))
    elif hasattr(item, "__dataclass_fields__"):
      values = {
        name: getattr(item, name)
        for name in item.__dataclass_fields__
      }
      stack.append((values, False))
      continue
    else:
      size = len(repr(item).encode("utf-8"))
    used += size
    remaining -= size
    if remaining < 0:
      return used
  return used


class ForkTaskRegistry:
  """Own paid fork tasks independently of request-task cancellation."""

  def __init__(
    self,
    ledger: ForkLedger,
    *,
    spawn_fork: ForkSpawn,
    telemetry: TelemetrySink | None = None,
    enabled: bool | None = None,
    owner_operated_interactive: bool = False,
    per_fork_budget_usd: Decimal | int | float | str | None = None,
    daily_budget_usd: Decimal | int | float | str | None = None,
    daily_invocation_quota: int | None = None,
    global_concurrency: int | None = None,
    handoff_max_bytes: int | None = None,
    shutdown_timeout_seconds: float = DEFAULT_FORK_SHUTDOWN_TIMEOUT_SECONDS,
  ) -> None:
    if not isinstance(ledger, ForkLedger):
      raise TypeError("fork task registry requires a ForkLedger")
    if not callable(spawn_fork):
      raise TypeError("spawn_fork must be callable")
    if telemetry is not None and not callable(telemetry):
      raise TypeError("telemetry must be callable")
    self._ledger = ledger
    self._spawn_fork = spawn_fork
    self._telemetry = telemetry
    self._enabled = (
      learn_fork_enabled(
        owner_operated_interactive=owner_operated_interactive,
      )
      if enabled is None
      else bool(enabled)
    )
    self._per_fork_budget = (
      learn_fork_budget_usd()
      if per_fork_budget_usd is None
      else Decimal(str(per_fork_budget_usd))
    )
    self._daily_budget = (
      learn_fork_daily_budget_usd()
      if daily_budget_usd is None
      else Decimal(str(daily_budget_usd))
    )
    self._daily_quota = (
      learn_fork_daily_invocation_quota()
      if daily_invocation_quota is None
      else daily_invocation_quota
    )
    self._global_cap = (
      learn_fork_global_concurrency()
      if global_concurrency is None
      else global_concurrency
    )
    self._handoff_max_bytes = (
      fork_handoff_max_bytes()
      if handoff_max_bytes is None
      else handoff_max_bytes
    )
    if (
      not self._per_fork_budget.is_finite()
      or self._per_fork_budget <= 0
      or not self._daily_budget.is_finite()
      or self._daily_budget <= 0
    ):
      raise ValueError("fork registry budgets must be finite and positive")
    for value, name in (
      (self._daily_quota, "daily invocation quota"),
      (self._global_cap, "global concurrency"),
      (self._handoff_max_bytes, "handoff maximum bytes"),
    ):
      if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if (
      isinstance(shutdown_timeout_seconds, bool)
      or not isinstance(shutdown_timeout_seconds, (int, float))
      or not math.isfinite(shutdown_timeout_seconds)
      or shutdown_timeout_seconds <= 0
    ):
      raise ValueError("shutdown timeout must be finite and positive")
    self._shutdown_timeout = float(shutdown_timeout_seconds)
    self._tasks_by_session: dict[str, asyncio.Task[None]] = {}
    self._handoffs: dict[str, Any] = {}
    self._shutting_down = False

  @property
  def active_count(self) -> int:
    return len(self._tasks_by_session)

  @property
  def retained_handoff_count(self) -> int:
    return len(self._handoffs)

  def _emit(self, event: str, **fields: Any) -> None:
    if self._telemetry is not None:
      try:
        self._telemetry(event, fields)
      except Exception:
        log.warning("Fork telemetry sink failed", exc_info=True)

  def _skip(
    self,
    fork_id: str,
    reason: ForkSkipReason,
    *,
    retained_bytes: int | None = None,
  ) -> ForkLaunchDecision:
    self._emit(
      "learning_fork_skipped",
      fork_id=fork_id,
      reason=reason,
      retained_bytes=retained_bytes,
    )
    return ForkLaunchDecision(False, fork_id, reason, retained_bytes)

  def submit(
    self,
    *,
    fork_id: str,
    session_id: str,
    owner: str,
    handoff: Any,
  ) -> ForkLaunchDecision:
    """Admit and own a fork without awaiting the new task."""

    if not self._enabled:
      return self._skip(fork_id, "disabled")
    if self._shutting_down:
      return self._skip(fork_id, "shutting_down")
    if session_id in self._tasks_by_session:
      return self._skip(fork_id, "session_cap")
    if len(self._tasks_by_session) >= self._global_cap:
      return self._skip(fork_id, "global_concurrency")
    retained_bytes = _retained_size(handoff, self._handoff_max_bytes)
    if retained_bytes > self._handoff_max_bytes:
      return self._skip(
        fork_id,
        "handoff_too_large",
        retained_bytes=retained_bytes,
      )
    loop = asyncio.get_running_loop()
    try:
      self._ledger.reserve_admission(
        fork_id=fork_id,
        owner=owner,
        max_reserved_usd=self._per_fork_budget,
        daily_budget_usd=self._daily_budget,
        daily_invocation_quota=self._daily_quota,
      )
    except ForkAdmissionQuotaExceeded:
      return self._skip(fork_id, "daily_invocation_quota")
    except ForkAdmissionBudgetExceeded:
      return self._skip(fork_id, "daily_budget")

    try:
      self._handoffs[fork_id] = handoff
      task = loop.create_task(
        self._run(fork_id=fork_id, session_id=session_id),
        name=f"learning-fork:{fork_id}",
      )
      self._tasks_by_session[session_id] = task
    except BaseException:
      self._handoffs.pop(fork_id, None)
      self._ledger.abandon_admission(fork_id=fork_id)
      raise
    self._emit(
      "learning_fork_launched",
      fork_id=fork_id,
      session_id=session_id,
      retained_bytes=retained_bytes,
    )
    return ForkLaunchDecision(True, fork_id, retained_bytes=retained_bytes)

  async def _run(self, *, fork_id: str, session_id: str) -> None:
    settled = False
    try:
      if not self._ledger.mark_admission_started(fork_id=fork_id):
        raise RuntimeError("fork admission could not transition to started")
      handoff = self._handoffs[fork_id]
      actual_cost = await self._spawn_fork(fork_id, handoff)
      if not self._ledger.settle_admission(
        fork_id=fork_id,
        actual_cost_usd=actual_cost,
      ):
        raise RuntimeError("fork admission could not be settled")
      settled = True
      on_settled = getattr(handoff, "on_admission_settled", None)
      if callable(on_settled):
        on_settled(fork_id)
      self._emit("learning_fork_completed", fork_id=fork_id)
    except asyncio.CancelledError:
      self._emit("learning_fork_cancelled", fork_id=fork_id)
      raise
    except Exception as exc:
      self._emit(
        "learning_fork_failed",
        fork_id=fork_id,
        error_type=type(exc).__name__,
      )
      log.warning("Learning fork %s failed", fork_id, exc_info=True)
    finally:
      if not settled:
        try:
          self._ledger.abandon_admission(fork_id=fork_id)
        except Exception:
          log.exception("Failed to abandon fork admission %s", fork_id)
      self._handoffs.pop(fork_id, None)
      current = asyncio.current_task()
      if self._tasks_by_session.get(session_id) is current:
        self._tasks_by_session.pop(session_id, None)

  async def shutdown(self) -> None:
    """Stop admission, drain to the bound, then cancel and join survivors."""

    self._shutting_down = True
    tasks = tuple(self._tasks_by_session.values())
    if not tasks:
      return
    done, pending = await asyncio.wait(
      tasks,
      timeout=self._shutdown_timeout,
    )
    del done
    if pending:
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)


__all__ = [
  "ForkLaunchDecision",
  "ForkTaskRegistry",
  "fork_handoff_max_bytes",
  "learn_fork_budget_usd",
  "learn_fork_daily_budget_usd",
  "learn_fork_daily_invocation_quota",
  "learn_fork_enabled",
  "learn_fork_global_concurrency",
]
