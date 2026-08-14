from __future__ import annotations

import inspect
from typing import Any

import pytest

from agent_gateway.event_log import EventLog
from agent_gateway.runner_sub_agents import (
  RunnerSubAgentMixin,
  _build_child_event_log,
)


def test_child_log_inherits_parent_prepare_and_forces_sub_agent_identity() -> None:
  prepared_events: list[dict[str, Any]] = []
  source = {
    "type": "text_delta",
    "text": "hello",
    "sub_agent_id": None,
  }

  def _prepare(event: dict[str, Any]) -> dict[str, Any]:
    prepared_events.append(dict(event))
    return dict(event)

  parent_log = EventLog(prepare_event=_prepare)
  child_log = _build_child_event_log(
    parent_log=parent_log,
    event_log_cls=EventLog,
    sub_session_id="sub0:session-1",
    progress_cb=None,
    on_sub_event=None,
  )

  entry = child_log.append(source)

  assert entry is not None
  assert prepared_events == [{
    "type": "text_delta",
    "text": "hello",
    "sub_agent_id": "sub0:session-1",
  }]
  assert entry.event == prepared_events[0]
  assert source["sub_agent_id"] is None


def test_child_log_propagates_strict_prepare_failure_without_callbacks() -> None:
  progress_events: list[dict[str, Any]] = []
  observed_events: list[dict[str, Any]] = []

  def _reject(_event: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("strict channel delivery failed")

  child_log = _build_child_event_log(
    parent_log=EventLog(prepare_event=_reject),
    event_log_cls=EventLog,
    sub_session_id="sub1:session-1",
    progress_cb=lambda event, _session_id: progress_events.append(event),
    on_sub_event=lambda event, _session_id: observed_events.append(event),
  )

  with pytest.raises(
    RuntimeError,
    match="strict channel delivery failed",
  ):
    child_log.append({
      "type": "text_delta",
      "payload": ["hostile"] * 100_000,
    })

  assert child_log.closed is True
  assert child_log.entries == []
  assert progress_events == []
  assert observed_events == []


def test_spawn_and_resume_both_use_strict_child_log_builder() -> None:
  spawn_source = inspect.getsource(RunnerSubAgentMixin.spawn_sub_agent)
  resume_source = inspect.getsource(RunnerSubAgentMixin.resume_sub_agent)

  assert "_build_child_event_log(" in spawn_source
  assert "_build_child_event_log(" in resume_source
