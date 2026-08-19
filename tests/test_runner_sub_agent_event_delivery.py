from __future__ import annotations

import asyncio
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


def test_foreground_child_approval_is_replayable_from_parent_log_once() -> None:
  delivered_events: list[dict[str, Any]] = []
  captured_events: list[dict[str, Any]] = []
  parent_log = EventLog(
    on_event=lambda event, _session_id: delivered_events.append(event),
    session_id="parent-session",
  )
  child_log = _build_child_event_log(
    parent_log=parent_log,
    event_log_cls=EventLog,
    sub_session_id="sub0:parent-session",
    progress_cb=None,
    on_sub_event=lambda event, _session_id: captured_events.append(event),
  )
  approval = {
    "type": "tool_approval_request",
    "approval_id": "approval-start-quant",
    "tool_call_id": "call-start-quant",
    "nonce": "nonce-start-quant",
    "tool_name": "start_quant_research",
    "tool_input": {"request": {"research_file_id": 42}},
    "allow_persistent_approval": False,
  }

  child_log.append(approval)
  parent_log.append({"type": "text_delta", "text": "after approval"})

  expected = {
    **approval,
    "sub_agent_id": "sub0:parent-session",
  }
  assert [entry.seq for entry in parent_log.entries] == [1, 2]
  assert parent_log.entries[0].event == expected
  assert delivered_events == [
    expected,
    {"type": "text_delta", "text": "after approval"},
  ]
  assert captured_events == [expected]
  assert child_log.entries[0].event == approval

  async def _replay() -> dict[str, Any]:
    replay = parent_log.iter_from(after_seq=0)
    return (await anext(replay)).event

  assert asyncio.run(_replay()) == expected


def test_non_approval_child_events_remain_isolated_from_parent_log() -> None:
  delivered_events: list[dict[str, Any]] = []
  parent_log = EventLog(
    on_event=lambda event, _session_id: delivered_events.append(event),
    session_id="parent-session",
  )
  child_log = _build_child_event_log(
    parent_log=parent_log,
    event_log_cls=EventLog,
    sub_session_id="sub0:parent-session",
    progress_cb=None,
    on_sub_event=None,
  )

  child_log.append({"type": "text_delta", "text": "child-only"})

  assert parent_log.entries == []
  assert delivered_events == [{
    "type": "text_delta",
    "text": "child-only",
    "sub_agent_id": "sub0:parent-session",
  }]
