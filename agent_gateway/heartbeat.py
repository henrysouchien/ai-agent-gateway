from __future__ import annotations

import asyncio
import inspect
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ._io import _atomic_write_json, _read_json_object
from .autonomous import RunOutput


log = logging.getLogger("agent_gateway.heartbeat")

_DEFAULT_BACKOFF_STEPS = [30.0, 60.0, 300.0, 900.0, 3600.0]
_HEARTBEAT_STATE_FILE = "heartbeat_state.json"
_LEADING_OK_RE = re.compile(r"^\s*HEARTBEAT_OK\s*(\n|$)")
_TRAILING_OK_RE = re.compile(r"(\n|^)\s*HEARTBEAT_OK\s*$")


@dataclass
class HeartbeatConfig:
  interval_seconds: float = 1800
  active_hours: tuple[int, int] | None = None
  timezone: str = "UTC"
  quiet_threshold: int = 20
  backoff_steps: list[float] | None = None
  checklist_path: str | Path | None = None
  state_dir: str | Path | None = None
  max_ticks: int | None = None

  def __post_init__(self) -> None:
    self.interval_seconds = float(self.interval_seconds)
    self.quiet_threshold = int(self.quiet_threshold)
    if self.backoff_steps is None:
      self.backoff_steps = list(_DEFAULT_BACKOFF_STEPS)
    else:
      self.backoff_steps = [float(step) for step in self.backoff_steps]
    if self.checklist_path is not None:
      self.checklist_path = Path(self.checklist_path)
    if self.state_dir is not None:
      self.state_dir = Path(self.state_dir)
    if self.max_ticks is not None:
      self.max_ticks = int(self.max_ticks)


@dataclass
class TickResult:
  output: RunOutput | None
  skipped: bool
  skip_reason: str | None
  alert: bool
  error: str | None
  stripped_response: str
  tick_number: int
  started_at: str
  duration_seconds: float


def strip_heartbeat_ok(text: str) -> tuple[str, bool]:
  """Strip HEARTBEAT_OK from start/end. Returns (stripped_text, had_token)."""
  stripped = str(text or "")
  found = False

  match = _LEADING_OK_RE.match(stripped)
  if match:
    stripped = stripped[match.end() :]
    found = True

  match = _TRAILING_OK_RE.search(stripped)
  if match:
    stripped = stripped[: match.start()]
    found = True

  return stripped, found


def is_checklist_empty(path: str | Path | None) -> bool:
  if path is None:
    return False

  checklist_path = Path(path)
  if not checklist_path.exists():
    return True

  try:
    content = checklist_path.read_text(encoding="utf-8")
  except Exception as exc:
    log.warning("Failed to read heartbeat checklist %s: %s", checklist_path, exc)
    return True

  for line in content.splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    if stripped.startswith("#"):
      continue
    return False
  return True


def is_within_active_hours(
  active_hours: tuple[int, int] | None,
  timezone_name: str,
  *,
  now: datetime | None = None,
) -> bool:
  if active_hours is None:
    return True

  start_hour, end_hour = active_hours
  if start_hour == end_hour:
    return False

  try:
    tzinfo = ZoneInfo(timezone_name or "UTC")
  except ZoneInfoNotFoundError:
    log.warning("Invalid heartbeat timezone %r; falling back to UTC", timezone_name)
    tzinfo = timezone.utc

  current_time = now.astimezone(tzinfo) if now is not None else datetime.now(tzinfo)
  current_hour = current_time.hour

  if start_hour < end_hour:
    return start_hour <= current_hour < end_hour
  return current_hour >= start_hour or current_hour < end_hour


class HeartbeatLoop:
  """Run a persistent agent periodically and suppress quiet HEARTBEAT_OK responses.

  `run_fn` should usually be a `functools.partial(run_autonomous, ..., delivery=None)`.
  If callbacks need the agent state written by `run_autonomous(state_dir=...)`, load it
  inside the callback or capture that state path in a closure.
  """

  def __init__(
    self,
    run_fn: Callable[[], Awaitable[RunOutput]],
    config: HeartbeatConfig | None = None,
    *,
    on_alert: Callable[[RunOutput, dict[str, Any]], Awaitable[None] | None] | None = None,
    on_quiet: Callable[[RunOutput, dict[str, Any]], Awaitable[None] | None] | None = None,
    on_error: Callable[[Exception | RunOutput, dict[str, Any]], Awaitable[None] | None] | None = None,
    on_tick: Callable[[TickResult, dict[str, Any]], Awaitable[None] | None] | None = None,
  ) -> None:
    self._run_fn = run_fn
    self.config = config or HeartbeatConfig()
    self._on_alert = on_alert
    self._on_quiet = on_quiet
    self._on_error = on_error
    self._on_tick = on_tick
    self._running = False
    self._loop: asyncio.AbstractEventLoop | None = None
    self._stop_event: asyncio.Event | None = None
    self._state = self._normalize_state({})

  @property
  def running(self) -> bool:
    return self._running

  @property
  def tick_count(self) -> int:
    return int(self._state.get("tick_count", 0))

  @property
  def state(self) -> dict[str, Any]:
    return dict(self._state)

  async def start(self) -> None:
    if self._running:
      raise RuntimeError("HeartbeatLoop is already running")

    self._stop_event = asyncio.Event()
    self._loop = asyncio.get_running_loop()
    self._running = True
    self._state = self._load_state()
    ticks_this_run = 0

    try:
      if self.config.max_ticks is not None and self.config.max_ticks <= 0:
        return

      while not self._stop_event.is_set():
        tick_result = await self._run_tick()
        ticks_this_run += 1
        await self._invoke_callback(self._on_tick, "Tick", tick_result, self.state)

        if self.config.max_ticks is not None and ticks_this_run >= self.config.max_ticks:
          break

        delay = self._current_delay_seconds()
        self._state["last_delay_seconds"] = delay
        self._persist_state()

        try:
          await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
          break
        except asyncio.TimeoutError:
          continue
    finally:
      self._running = False
      self._loop = None
      self._stop_event = None

  def stop(self) -> None:
    stop_event = self._stop_event
    if stop_event is None:
      return

    try:
      running_loop = asyncio.get_running_loop()
    except RuntimeError:
      running_loop = None

    if self._loop is not None and self._loop.is_running() and running_loop is not self._loop:
      try:
        self._loop.call_soon_threadsafe(stop_event.set)
        return
      except RuntimeError:
        pass

    stop_event.set()

  async def _run_tick(self) -> TickResult:
    loop = asyncio.get_running_loop()
    tick_number = self.tick_count + 1
    started_at = datetime.now(tz=timezone.utc).isoformat()
    started_monotonic = loop.time()

    self._state["tick_count"] = tick_number
    self._state["last_tick_started_at"] = started_at

    if not is_within_active_hours(self.config.active_hours, self.config.timezone):
      return await self._finalize_tick(
        output=None,
        skipped=True,
        skip_reason="outside_active_hours",
        alert=False,
        error=None,
        stripped_response="",
        tick_number=tick_number,
        started_at=started_at,
        started_monotonic=started_monotonic,
        outcome="skipped",
      )

    if self.config.checklist_path is not None and is_checklist_empty(self.config.checklist_path):
      return await self._finalize_tick(
        output=None,
        skipped=True,
        skip_reason="empty_checklist",
        alert=False,
        error=None,
        stripped_response="",
        tick_number=tick_number,
        started_at=started_at,
        started_monotonic=started_monotonic,
        outcome="skipped",
      )

    try:
      output = await self._run_fn()
    except Exception as exc:
      self._state["consecutive_errors"] = int(self._state.get("consecutive_errors", 0)) + 1
      return await self._finalize_tick(
        output=None,
        skipped=False,
        skip_reason=None,
        alert=False,
        error=f"{type(exc).__name__}: {exc}",
        stripped_response="",
        tick_number=tick_number,
        started_at=started_at,
        started_monotonic=started_monotonic,
        outcome="error",
        callback=(self._on_error, "Error", exc),
      )

    stripped_response, had_token = strip_heartbeat_ok(output.response)
    error_message = output.error or ("Heartbeat run timed out" if output.timed_out else None)
    if error_message is not None:
      self._state["consecutive_errors"] = int(self._state.get("consecutive_errors", 0)) + 1
      return await self._finalize_tick(
        output=output,
        skipped=False,
        skip_reason=None,
        alert=False,
        error=error_message,
        stripped_response=stripped_response,
        tick_number=tick_number,
        started_at=started_at,
        started_monotonic=started_monotonic,
        outcome="error",
        callback=(self._on_error, "Error", output),
      )

    self._state["consecutive_errors"] = 0
    is_quiet = False
    if not output.budget_exceeded and not output.max_turns_reached:
      is_quiet = had_token and len(stripped_response.strip()) <= self.config.quiet_threshold

    if is_quiet:
      return await self._finalize_tick(
        output=output,
        skipped=False,
        skip_reason=None,
        alert=False,
        error=None,
        stripped_response=stripped_response,
        tick_number=tick_number,
        started_at=started_at,
        started_monotonic=started_monotonic,
        outcome="quiet",
        callback=(self._on_quiet, "Quiet", output),
      )

    outcome = "alert"
    if output.budget_exceeded:
      outcome = "budget_exceeded"
    elif output.max_turns_reached:
      outcome = "max_turns_reached"
    return await self._finalize_tick(
      output=output,
      skipped=False,
      skip_reason=None,
      alert=True,
      error=None,
      stripped_response=stripped_response,
      tick_number=tick_number,
      started_at=started_at,
      started_monotonic=started_monotonic,
      outcome=outcome,
      callback=(self._on_alert, "Alert", output),
    )

  async def _finalize_tick(
    self,
    *,
    output: RunOutput | None,
    skipped: bool,
    skip_reason: str | None,
    alert: bool,
    error: str | None,
    stripped_response: str,
    tick_number: int,
    started_at: str,
    started_monotonic: float,
    outcome: str,
    callback: tuple[Callable[..., Awaitable[None] | None] | None, str, Any] | None = None,
  ) -> TickResult:
    duration_seconds = asyncio.get_running_loop().time() - started_monotonic
    self._state["last_outcome"] = outcome
    self._state["last_error"] = error
    self._state["last_skip_reason"] = skip_reason
    self._state["last_alert"] = alert
    self._state["last_tick_finished_at"] = datetime.now(tz=timezone.utc).isoformat()
    self._state["last_duration_seconds"] = duration_seconds
    self._state["last_delay_seconds"] = None

    result = TickResult(
      output=output,
      skipped=skipped,
      skip_reason=skip_reason,
      alert=alert,
      error=error,
      stripped_response=stripped_response,
      tick_number=tick_number,
      started_at=started_at,
      duration_seconds=duration_seconds,
    )

    if callback is not None:
      callback_fn, label, callback_value = callback
      await self._invoke_callback(callback_fn, label, callback_value, self.state)

    self._persist_state()
    return result

  async def _invoke_callback(
    self,
    callback: Callable[..., Awaitable[None] | None] | None,
    label: str,
    *args: Any,
  ) -> None:
    if callback is None:
      return

    try:
      result = callback(*args)
      if inspect.isawaitable(result):
        await result
    except Exception as exc:
      log.warning("%s callback failed (non-fatal): %s", label, exc)

  def _current_delay_seconds(self) -> float:
    consecutive_errors = int(self._state.get("consecutive_errors", 0))
    steps = self.config.backoff_steps or []
    if consecutive_errors > 0 and steps:
      idx = min(consecutive_errors - 1, len(steps) - 1)
      return steps[idx]
    return self.config.interval_seconds

  def _load_state(self) -> dict[str, Any]:
    if self.config.state_dir is None:
      return self._normalize_state(self._state)
    return self._normalize_state(_read_json_object(Path(self.config.state_dir) / _HEARTBEAT_STATE_FILE))

  def _persist_state(self) -> None:
    if self.config.state_dir is None:
      return
    _atomic_write_json(Path(self.config.state_dir) / _HEARTBEAT_STATE_FILE, dict(self._state))

  def _normalize_state(self, payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(payload or {})
    return {
      "tick_count": _coerce_int(state.get("tick_count"), default=0),
      "consecutive_errors": _coerce_int(state.get("consecutive_errors"), default=0),
      "last_outcome": _coerce_str(state.get("last_outcome")),
      "last_error": _coerce_str(state.get("last_error")),
      "last_skip_reason": _coerce_str(state.get("last_skip_reason")),
      "last_alert": bool(state.get("last_alert", False)),
      "last_tick_started_at": _coerce_str(state.get("last_tick_started_at")),
      "last_tick_finished_at": _coerce_str(state.get("last_tick_finished_at")),
      "last_duration_seconds": _coerce_float(state.get("last_duration_seconds"), default=0.0),
      "last_delay_seconds": _coerce_optional_float(state.get("last_delay_seconds")),
    }


def _coerce_float(value: Any, *, default: float) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _coerce_int(value: Any, *, default: int) -> int:
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _coerce_optional_float(value: Any) -> float | None:
  if value is None:
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _coerce_str(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value).strip()
  return text or None


__all__ = [
  "HeartbeatConfig",
  "HeartbeatLoop",
  "TickResult",
  "is_checklist_empty",
  "is_within_active_hours",
  "strip_heartbeat_ok",
]
