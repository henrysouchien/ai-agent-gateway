from __future__ import annotations

import pytest

from agent_gateway.session_event_history import SessionEventHistory


def test_session_event_history_retains_last_5000_events() -> None:
  history = SessionEventHistory()

  for index in range(6000):
    history.append({"type": "event", "index": index})

  snapshot = history.snapshot()
  assert len(history) == 5000
  assert len(snapshot) == 5000
  assert snapshot[0]["index"] == 1000
  assert snapshot[-1]["index"] == 5999


def test_session_event_history_never_closes_on_terminal_events() -> None:
  history = SessionEventHistory(max_events=3)

  history.append({"type": "stream_complete"})
  history.append({"type": "error", "error": "first"})
  history.append({"type": "text_delta", "text": "still accepted"})
  history.append({"type": "error", "error": "second"})

  assert history.snapshot() == [
    {"type": "error", "error": "first"},
    {"type": "text_delta", "text": "still accepted"},
    {"type": "error", "error": "second"},
  ]


def test_session_event_history_tail_and_copy_semantics() -> None:
  history = SessionEventHistory(max_events=5)
  event = {"type": "text_delta", "nested": {"value": 1}}

  history.append(event)
  event["type"] = "mutated"

  assert history.snapshot(tail=0) == []
  assert history.snapshot(tail=1) == [{"type": "text_delta", "nested": {"value": 1}}]
  assert history.snapshot()[0] is not event


def test_session_event_history_rejects_non_positive_bound() -> None:
  with pytest.raises(ValueError, match="max_events must be positive"):
    SessionEventHistory(max_events=0)
