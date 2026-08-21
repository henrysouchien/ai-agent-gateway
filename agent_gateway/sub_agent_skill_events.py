from __future__ import annotations

import asyncio
from copy import deepcopy
import time
from collections.abc import Awaitable, Iterable
from typing import Any, Callable

from .events import SkillRunStartedEvent, event_to_dict
from .skill_lifecycle import (
  TopLevelSkillLifecycleMetadata,
  resolve_skill_lifecycle_artifact_identity,
)
from .skill_result_events import build_skill_result_captured_event


class DurableSkillEventPersistenceError(RuntimeError):
  """A required named-skill lifecycle event was not durably confirmed."""


def _exact_value_match(expected: Any, actual: Any) -> bool:
  if type(actual) is not type(expected):
    return False
  if type(expected) is dict:
    return (
      actual.keys() == expected.keys()
      and all(
        _exact_value_match(value, actual[field_name])
        for field_name, value in expected.items()
      )
    )
  if type(expected) is list:
    return (
      len(actual) == len(expected)
      and all(
        _exact_value_match(expected_item, actual_item)
        for expected_item, actual_item in zip(expected, actual)
      )
    )
  return bool(actual == expected)


async def _drain_shielded_task(task: asyncio.Task[bool]) -> bool:
  """Await an owned lifecycle write despite repeated waiter cancellation."""

  while True:
    try:
      return await asyncio.shield(task)
    except asyncio.CancelledError:
      if task.done():
        return task.result()


class SkillRunEventEmitter:
  def __init__(
    self,
    *,
    skill_run_id: str,
    profile: Any | None = None,
    skill_name: str | None = None,
    semantic_scope: str | None,
    context_ticker: str | None,
    portfolio_id: str | None,
    event_log_getter: Callable[[], Any],
    tool_ctx: Any,
    durable_appender: Callable[[dict[str, Any]], Awaitable[Any | None]],
    durable_confirmer: Callable[
      [dict[str, Any]],
      Awaitable[dict[str, Any] | None],
    ],
    time_fn: Callable[[], float] = time.time,
  ) -> None:
    self._event_log_getter = event_log_getter
    self._tool_ctx = tool_ctx
    resolved_skill_name = (
      skill_name
      if skill_name is not None
      else getattr(profile, "name", None)
    )
    if (
      type(resolved_skill_name) is not str
      or not resolved_skill_name
      or resolved_skill_name != resolved_skill_name.strip()
    ):
      raise ValueError("skill lifecycle requires a canonical skill name")
    artifact_identity = resolve_skill_lifecycle_artifact_identity(
      semantic_scope=semantic_scope,
      context_ticker=context_ticker,
      portfolio_id=portfolio_id,
    )
    self._lifecycle = TopLevelSkillLifecycleMetadata(
      skill_run_id=skill_run_id,
      skill=resolved_skill_name,
      **artifact_identity.identity_fields(),
    )
    self._durable_appender = durable_appender
    self._durable_confirmer = durable_confirmer
    self._time_fn = time_fn
    self._started_task: asyncio.Task[bool] | None = None
    self._started_event: dict[str, Any] | None = None
    self._result_task: asyncio.Task[bool] | None = None
    self._result_event: dict[str, Any] | None = None
    self._result_projected = False

  def required_lifecycle_metadata(self) -> dict[str, Any]:
    return {
      "schema_version": 2,
      **self._lifecycle.identity_fields(),
    }

  def _live_parent_event_log(self) -> Any:
    """Return the owning live stream, falling back for legacy callers/tests."""

    tool_event_log = getattr(self._tool_ctx, "event_log", None)
    if tool_event_log is not None:
      return tool_event_log
    try:
      return self._event_log_getter()
    except NameError:
      return None

  def emit_parent_event(self, event: dict[str, Any]) -> None:
    event_log = self._live_parent_event_log()
    tool_event_log = getattr(self._tool_ctx, "event_log", None)
    if event_log is not None:
      lifecycle_event = (
        event.get("type")
        in {"skill_run_started", "skill_result_captured"}
        and event.get("skill_run_id")
        == self._lifecycle.skill_run_id
      )
      if lifecycle_event:
        matches = [
          entry.event
          for entry in getattr(event_log, "entries", ())
          if (
            isinstance(getattr(entry, "event", None), dict)
            and entry.event.get("type") == event["type"]
            and entry.event.get("skill_run_id")
            == self._lifecycle.skill_run_id
          )
        ]
        if len(matches) > 1:
          raise RuntimeError(
            "Duplicate live named-skill lifecycle projection"
          )
        if matches:
          if not _exact_value_match(event, matches[0]):
            raise RuntimeError(
              "Conflicting live named-skill lifecycle projection"
            )
        else:
          appended_entry = event_log.append(deepcopy(event))
          if (
            appended_entry is None
            or not _exact_value_match(
              event,
              getattr(appended_entry, "event", None),
            )
          ):
            raise RuntimeError(
              "Live named-skill lifecycle projection was not exact"
            )
          matches = [
            entry.event
            for entry in getattr(event_log, "entries", ())
            if (
              isinstance(getattr(entry, "event", None), dict)
              and entry.event.get("type") == event["type"]
              and entry.event.get("skill_run_id")
              == self._lifecycle.skill_run_id
            )
          ]
          if (
            len(matches) != 1
            or not _exact_value_match(event, matches[0])
          ):
            raise RuntimeError(
              "Live named-skill lifecycle projection was not exact"
            )
      else:
        event_log.append(deepcopy(event))
    emit = getattr(self._tool_ctx, "emit", None)
    if callable(emit) and tool_event_log is not event_log:
      emit(deepcopy(event))

  async def _emit_durable_parent_event(
    self,
    event: dict[str, Any],
    *,
    projector: Callable[[dict[str, Any]], Any] | None = None,
  ) -> bool:
    confirmed_event = await self._durable_confirmer(
      deepcopy(event)
    )
    if confirmed_event is None:
      append_failure: BaseException | None = None
      try:
        await self._durable_appender(deepcopy(event))
      except BaseException as exc:
        append_failure = exc
      confirmed_event = await self._durable_confirmer(
        deepcopy(event)
      )
      if confirmed_event is None:
        if append_failure is not None:
          raise append_failure
        raise DurableSkillEventPersistenceError(
          "Required named-skill lifecycle event was not durably confirmed"
        )
    if type(confirmed_event) is not dict:
      raise DurableSkillEventPersistenceError(
        "Durable named-skill confirmation did not return an exact envelope"
      )
    (projector or self.emit_parent_event)(
      deepcopy(confirmed_event)
    )
    return True

  async def _await_owned_task(self, task: asyncio.Task[bool]) -> bool:
    try:
      return await asyncio.shield(task)
    except asyncio.CancelledError:
      await _drain_shielded_task(task)
      raise

  async def emit_started(self) -> bool:
    if self._started_task is not None and self._started_task.done():
      try:
        return self._started_task.result()
      except BaseException:
        self._started_task = None
    if self._started_task is None:
      if self._started_event is None:
        self._started_event = event_to_dict(
          SkillRunStartedEvent(
            ts=self._time_fn(),
            **self._lifecycle.identity_fields(),
          )
        )
      self._started_task = asyncio.create_task(
        self._emit_durable_parent_event(
          deepcopy(self._started_event)
        ),
        name=f"skill-run:{self._lifecycle.skill_run_id}:started",
      )
    return await self._await_owned_task(self._started_task)

  async def _emit_result_captured(
    self,
    result: Any | None,
    error: dict[str, Any] | None,
  ) -> bool:
    if not await self.emit_started():
      return False
    if self._result_event is None:
      self._result_event = self.build_result_captured_event(
        result,
        error,
      )
    return await self._emit_durable_parent_event(
      deepcopy(self._result_event),
      projector=self.project_result_captured,
    )

  def build_result_captured_event(
    self,
    result: Any | None,
    error: dict[str, Any] | None,
  ) -> dict[str, Any]:
    return build_skill_result_captured_event(
      **self._lifecycle.identity_fields(),
      entries=self._event_log_entries(),
      result=result,
      error=error,
    )

  def project_result_captured(self, event: dict[str, Any]) -> bool:
    if self._result_projected:
      return False
    self.emit_parent_event(event)
    self._result_projected = True
    return True

  async def emit_result_captured(
    self,
    result: Any | None,
    error: dict[str, Any] | None,
  ) -> bool:
    if self._result_task is not None and self._result_task.done():
      try:
        return self._result_task.result()
      except BaseException:
        self._result_task = None
    if self._result_task is None:
      self._result_task = asyncio.create_task(
        self._emit_result_captured(result, error),
        name=f"skill-run:{self._lifecycle.skill_run_id}:result",
      )
    return await self._await_owned_task(self._result_task)

  def _event_log_entries(self) -> Iterable[Any]:
    try:
      return self._event_log_getter().entries
    except NameError:
      return ()


__all__ = [
  "DurableSkillEventPersistenceError",
  "SkillRunEventEmitter",
]
