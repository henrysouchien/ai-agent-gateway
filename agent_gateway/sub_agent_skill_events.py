from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Callable

from .events import SkillRunStartedEvent, event_to_dict
from .skill_result_events import build_skill_result_captured_event


class SkillRunEventEmitter:
  def __init__(
    self,
    *,
    skill_run_id: str,
    profile: Any,
    context_ticker: str,
    event_log_getter: Callable[[], Any],
    tool_ctx: Any,
    ticker_fn: Callable[[Any, str], str | None],
    scope_fn: Callable[[Any, str], str],
    time_fn: Callable[[], float] = time.time,
  ) -> None:
    self._skill_run_id = skill_run_id
    self._profile = profile
    self._context_ticker = context_ticker
    self._event_log_getter = event_log_getter
    self._tool_ctx = tool_ctx
    self._ticker_fn = ticker_fn
    self._scope_fn = scope_fn
    self._time_fn = time_fn
    self._started_emitted = False

  def emit_parent_event(self, event: dict[str, Any]) -> None:
    try:
      self._event_log_getter().append(event)
    except NameError:
      pass
    emit = getattr(self._tool_ctx, "emit", None)
    if callable(emit):
      emit(event)

  def emit_started(self) -> None:
    if self._started_emitted:
      return
    self.emit_parent_event(
      event_to_dict(
        SkillRunStartedEvent(
          skill_run_id=self._skill_run_id,
          skill=self._profile.name,
          ticker=self._ticker_fn(self._profile, self._context_ticker),
          ts=self._time_fn(),
          scope=self._scope_fn(self._profile, self._context_ticker),
        )
      )
    )
    self._started_emitted = True

  def emit_result_captured(self, result: Any | None, error: dict[str, Any] | None) -> None:
    self.emit_started()
    self.emit_parent_event(
      build_skill_result_captured_event(
        skill_run_id=self._skill_run_id,
        skill=self._profile.name,
        ticker=self._ticker_fn(self._profile, self._context_ticker),
        entries=self._event_log_entries(),
        result=result,
        error=error,
      )
    )

  def _event_log_entries(self) -> Iterable[Any]:
    try:
      return self._event_log_getter().entries
    except NameError:
      return ()


__all__ = [
  "SkillRunEventEmitter",
]
