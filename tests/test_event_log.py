from __future__ import annotations

import asyncio

import pytest

import agent_gateway.event_log as event_log_module
from agent_gateway.event_log import EventLog


def test_append_isolates_nested_entry_from_input_and_callback() -> None:
  callback_events: list[dict[str, object]] = []

  def _mutate_callback(
    event: dict[str, object],
    _session_id: str,
  ) -> None:
    callback_events.append(event)
    nested = event["nested"]
    assert isinstance(nested, dict)
    nested["items"] = ["callback-mutated"]

  event_log = EventLog(on_event=_mutate_callback)
  source = {"type": "custom", "nested": {"items": ["original"]}}

  entry = event_log.append(source)
  assert entry is not None
  source["nested"]["items"].append("caller-mutated")

  assert callback_events[0]["nested"] == {
    "items": ["callback-mutated"],
  }
  assert entry.event["nested"] == {"items": ["original"]}
  assert event_log.entries[0].event["nested"] == {
    "items": ["original"],
  }

  entry.event["nested"]["items"] = ["append-return-mutated"]
  entries = event_log.entries
  entries[0].event["nested"]["items"] = ["entries-mutated"]

  assert event_log.entries[0].event["nested"] == {
    "items": ["original"],
  }


def test_iter_from_yields_defensive_nested_entry_copy() -> None:
  async def _case() -> None:
    event_log = EventLog()
    event_log.append({
      "type": "custom",
      "nested": {"items": ["original"]},
    })
    event_log.close()

    yielded = [
      entry
      async for entry in event_log.iter_from()
    ]
    yielded[0].event["nested"]["items"] = ["iter-mutated"]

    assert event_log.entries[0].event["nested"] == {
      "items": ["original"],
    }

  asyncio.run(_case())


def test_callback_error_policy_defaults_to_best_effort() -> None:
  def _fail_callback(
    _event: dict[str, object],
    _session_id: str,
  ) -> None:
    raise RuntimeError("delivery failed")

  event_log = EventLog(on_event=_fail_callback)

  entry = event_log.append({"type": "custom"})

  assert entry is not None
  assert event_log.closed is False
  assert [item.event for item in event_log.entries] == [
    {"type": "custom"},
  ]


def test_strict_callback_error_propagates_and_fail_closes_log() -> None:
  def _fail_callback(
    _event: dict[str, object],
    _session_id: str,
  ) -> None:
    raise RuntimeError("delivery failed")

  event_log = EventLog(
    on_event=_fail_callback,
    on_event_error="raise",
  )

  with pytest.raises(RuntimeError, match="delivery failed"):
    event_log.append({"type": "custom"})

  assert event_log.closed is True
  assert event_log.has_terminal is False
  assert event_log.next_seq == 2
  assert [item.event for item in event_log.entries] == [
    {"type": "custom"},
  ]
  assert event_log.append({"type": "after_failure"}) is None


def test_callback_error_policy_is_closed() -> None:
  with pytest.raises(
    ValueError,
    match="on_event_error must be either 'ignore' or 'raise'",
  ):
    EventLog(on_event_error="drop")  # type: ignore[arg-type]


def test_prepare_event_failure_precedes_any_deepcopy_and_fail_closes(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ordering: list[str] = []

  def _reject_before_copy(
    _event: dict[str, object],
  ) -> dict[str, object]:
    ordering.append("prepare")
    raise RuntimeError("bounded preflight rejected event")

  def _unexpected_deepcopy(_value: object) -> object:
    ordering.append("deepcopy")
    raise AssertionError("deepcopy ran before bounded preflight")

  monkeypatch.setattr(
    event_log_module,
    "deepcopy",
    _unexpected_deepcopy,
  )
  event_log = EventLog(prepare_event=_reject_before_copy)

  with pytest.raises(
    RuntimeError,
    match="bounded preflight rejected event",
  ):
    event_log.append({
      "type": "custom",
      "hostile": ["x"] * 100_000,
    })

  assert ordering == ["prepare"]
  assert event_log.closed is True
  assert event_log.entries == []
