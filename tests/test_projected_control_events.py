from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import Request

from agent_gateway.control_plane import events as events_module
from agent_gateway.control_plane.events import build_events_router
from agent_gateway.event_adapter import adapt_control_event, adapt_event
from agent_gateway.event_log import UserEventBus


class _FakeSession(SimpleNamespace):
  pass


class _FakeSessionStore:
  def __init__(self, sessions: dict[str, _FakeSession] | None = None) -> None:
    self._sessions = dict(sessions or {})

  def get_session(self, session_id: str) -> _FakeSession | None:
    return self._sessions.get(session_id)


class _FakeAuth:
  def __init__(self, tokens: dict[str, _FakeSession], sessions: dict[str, _FakeSession] | None = None) -> None:
    self._tokens = tokens
    self.session_store = _FakeSessionStore(sessions)

  def verify_token(self, token: str) -> _FakeSession:
    return self._tokens[token]


def _session(
  *,
  kind: str = "control",
  user_id: str = "alice",
  session_id: str = "control-1",
  channel: str = "tui",
) -> _FakeSession:
  return _FakeSession(kind=kind, user_id=user_id, session_id=session_id, channel=channel)


def _request(app: Any, token: str) -> Request:
  return Request(
    {
      "type": "http",
      "method": "GET",
      "path": "/events",
      "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))],
      "query_string": b"",
      "app": app,
    }
  )


def _route(auth: _FakeAuth):
  router = build_events_router(auth=auth)  # type: ignore[arg-type]
  return next(route for route in router.routes if getattr(route, "path", None) == "/events")


async def _open_events(
  *,
  bus: UserEventBus,
  auth: _FakeAuth,
  token: str = "control-token",
  control_run_id: str | None = "run-1",
  schema_version: str | None = None,
  after_seq: int = 0,
  tasks: dict[str, Any] | None = None,
):
  app = SimpleNamespace(
    state=SimpleNamespace(
      user_event_bus=bus,
      subprocess_registry=SimpleNamespace(_tasks=dict(tasks or {})),
    )
  )
  response = await _route(auth).endpoint(
    _request(app, token),
    control_run_id=control_run_id,
    run_id=None,
    schema_version=schema_version,
    after_seq=after_seq,
  )
  return response


async def _close_response(response: Any) -> None:
  close = getattr(response.body_iterator, "aclose", None)
  if callable(close):
    await close()
  if response.background is not None:
    await response.background()


def _decode_data(chunk: bytes) -> dict[str, Any]:
  text = chunk.decode("utf-8").strip()
  assert text.startswith("data: ")
  return json.loads(text.removeprefix("data: "))


async def _read_data(response: Any, count: int, *, timeout: float = 0.5) -> list[dict[str, Any]]:
  seen: list[dict[str, Any]] = []
  try:
    while len(seen) < count:
      chunk = await asyncio.wait_for(response.body_iterator.__anext__(), timeout=timeout)
      if chunk.startswith(b":"):
        continue
      seen.append(_decode_data(chunk))
    return seen
  finally:
    await _close_response(response)


def _run(awaitable: Any) -> Any:
  return asyncio.run(awaitable)


def test_control_projection_extends_v1_without_changing_chat_projection() -> None:
  event = {
    "type": "text_delta",
    "text": "hello",
    "run_id": "run-1",
    "control_run_id": "run-1",
    "sub_agent_id": "sub-1",
    "future_only": "strip",
  }

  assert adapt_event(event, 1) == {"type": "text_delta", "text": "hello"}
  assert adapt_control_event(event, 1) == {
    "type": "text_delta",
    "text": "hello",
    "run_id": "run-1",
    "control_run_id": "run-1",
    "sub_agent_id": "sub-1",
  }

  assert adapt_control_event(
    {
      "type": "run_state_changed",
      "run_id": "run-1",
      "control_run_id": "run-1",
      "state": "running",
      "ts": 100,
      "future_only": "strip",
    },
    1,
  ) == {
    "type": "run_state_changed",
    "run_id": "run-1",
    "control_run_id": "run-1",
    "state": "running",
    "ts": 100,
  }
  assert adapt_control_event(
    {"type": "malformed_autonomous_event", "run_id": "run-1", "raw": "{bad", "future_only": "strip"},
    1,
  ) == {"type": "malformed_autonomous_event", "run_id": "run-1", "raw": "{bad"}


def test_chat_wire_projection_for_existing_event_types_stays_byte_identical() -> None:
  event = {
    "type": "artifact_ready",
    "skill_run_id": "run-1",
    "ticker": "MSFT",
    "skill": "model-review",
    "subcommand": "report_model_review",
    "mutation_mode": "preview",
    "artifact_ref": "artifact.json",
    "artifact_id": "artifact-1",
    "artifact_path": "artifact.json",
    "binary_artifact_path": None,
    "proposal_id": None,
    "contract_name": "ModelReview",
    "data_source": "live",
    "status": "noop",
    "gate_code": "PROCEED",
    "sidecar_hash": "sha256:" + "ab" * 32,
    "verdict_echo": {"verdict": "PROCEED"},
    "run_id": "run-1",
    "control_run_id": "run-1",
    "sub_agent_id": "sub-1",
    "ts": 1.0,
    "scope": "ticker",
    "portfolio_id": None,
  }

  expected_chat_shape = {
    "type": "artifact_ready",
    "skill_run_id": "run-1",
    "ticker": "MSFT",
    "skill": "model-review",
    "subcommand": "report_model_review",
    "mutation_mode": "preview",
    "artifact_ref": "artifact.json",
    "artifact_id": "artifact-1",
    "artifact_path": "artifact.json",
    "binary_artifact_path": None,
    "proposal_id": None,
    "contract_name": "ModelReview",
    "data_source": "live",
    "status": "noop",
    "gate_code": "PROCEED",
    "sidecar_hash": "sha256:" + "ab" * 32,
    "verdict_echo": {"verdict": "PROCEED"},
    "ts": 1.0,
    "scope": "ticker",
    "portfolio_id": None,
  }

  assert adapt_event(event, 1) == expected_chat_shape
  assert adapt_control_event(event, 1) == {
    **expected_chat_shape,
    "run_id": "run-1",
    "control_run_id": "run-1",
    "sub_agent_id": "sub-1",
  }


def test_raw_control_stream_stays_raw_while_projected_stream_uses_envelopes() -> None:
  async def case() -> None:
    auth = _FakeAuth({"control-token": _session()})
    bus = UserEventBus()
    event = {
      "type": "tool_call_start",
      "tool_call_id": "tool-1",
      "tool_name": "read_file",
      "tool_input": {"path": "notes.md"},
      "run_id": "run-1",
      "control_run_id": "run-1",
      "sub_agent_id": "sub-1",
      "future_only": "raw-only",
    }
    await bus.publish("alice", "run-1", event)

    raw_response = await _open_events(bus=bus, auth=auth, schema_version=None)
    assert await _read_data(raw_response, 1) == [event]

    projected_response = await _open_events(bus=bus, auth=auth, schema_version="v1")
    assert await _read_data(projected_response, 1) == [
      {
        "run_id": "run-1",
        "seq": 1,
        "event": {
          "type": "tool_call_start",
          "tool_call_id": "tool-1",
          "tool_name": "read_file",
          "tool_input": {"path": "notes.md"},
          "run_id": "run-1",
          "control_run_id": "run-1",
          "sub_agent_id": "sub-1",
        },
      }
    ]

    await bus.shutdown()

  _run(case())


def test_projected_stream_seq_resume_and_truncation_marker() -> None:
  async def case() -> None:
    auth = _FakeAuth({"control-token": _session()})
    bus = UserEventBus(replay_buffer_max=2)
    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "one", "run_id": "run-1"})
    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "two", "run_id": "run-1"})
    await bus.publish("alice", "run-1", {"type": "turn_complete", "turn": 1, "run_id": "run-1"})
    await bus.publish("alice", "run-2", {"type": "text_delta", "text": "other", "run_id": "run-2"})
    await bus.publish("alice", "run-2", {"type": "turn_complete", "turn": 1, "run_id": "run-2"})

    truncated = await _open_events(bus=bus, auth=auth, schema_version="v1", after_seq=0)
    assert await _read_data(truncated, 3) == [
      {
        "run_id": "run-1",
        "seq": None,
        "event": {
          "type": "replay_truncated",
          "run_id": "run-1",
          "control_run_id": "run-1",
          "dropped_before_seq": 2,
        },
      },
      {"run_id": "run-1", "seq": 2, "event": {"type": "text_delta", "text": "two", "run_id": "run-1"}},
      {"run_id": "run-1", "seq": 3, "event": {"type": "turn_complete", "turn": 1, "run_id": "run-1"}},
    ]

    resumed = await _open_events(bus=bus, auth=auth, schema_version="v1", after_seq=2)
    assert await _read_data(resumed, 1) == [
      {"run_id": "run-1", "seq": 3, "event": {"type": "turn_complete", "turn": 1, "run_id": "run-1"}}
    ]

    other_run = await _open_events(
      bus=bus,
      auth=auth,
      control_run_id="run-2",
      schema_version="v1",
    )
    assert await _read_data(other_run, 1) == [
      {"run_id": "run-2", "seq": 1, "event": {"type": "text_delta", "text": "other", "run_id": "run-2"}}
    ]

    await bus.shutdown()

  _run(case())


def test_projected_stream_replays_rehydrated_autonomous_events_after_restart() -> None:
  async def case() -> None:
    auth = _FakeAuth({"control-token": _session()})
    bus = UserEventBus()
    rehydrated = SimpleNamespace(
      task_id="bg_7",
      control_run_id="run-1",
      user_id="alice",
      channel="tui",
      state="completed",
      proc=None,
      event_lines=[
        {"type": "text_delta", "text": "restored"},
        {"type": "turn_complete", "turn": 1},
      ],
    )

    by_task_id = await _open_events(
      bus=bus,
      auth=auth,
      control_run_id="bg_7",
      schema_version="v1",
      tasks={"bg_7": rehydrated},
    )
    assert await _read_data(by_task_id, 2) == [
      {
        "run_id": "run-1",
        "seq": 1,
        "event": {
          "type": "text_delta",
          "text": "restored",
          "run_id": "run-1",
          "control_run_id": "run-1",
        },
      },
      {
        "run_id": "run-1",
        "seq": 2,
        "event": {
          "type": "turn_complete",
          "turn": 1,
          "run_id": "run-1",
          "control_run_id": "run-1",
        },
      },
    ]

    resumed = await _open_events(
      bus=bus,
      auth=auth,
      control_run_id="run-1",
      schema_version="v1",
      after_seq=1,
      tasks={"bg_7": rehydrated},
    )
    assert await _read_data(resumed, 1) == [
      {
        "run_id": "run-1",
        "seq": 2,
        "event": {
          "type": "turn_complete",
          "turn": 1,
          "run_id": "run-1",
          "control_run_id": "run-1",
        },
      }
    ]

    await bus.shutdown()

  _run(case())


def test_projected_bus_events_dropped_sentinel_has_null_seq_and_drop_cursor() -> None:
  async def case() -> None:
    bus = UserEventBus(subscriber_queue_max=1)
    subscriber = bus.subscribe_entries("alice", control_run_id="run-1")
    first_task = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)

    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "first"})
    assert (await asyncio.wait_for(first_task, timeout=0.5)).seq == 1

    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "second"})
    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "third"})
    await bus.publish("alice", "run-1", {"type": "turn_complete"})

    sentinel = await asyncio.wait_for(subscriber.__anext__(), timeout=0.5)
    assert sentinel.seq is None
    assert sentinel.event == {
      "type": "events_dropped",
      "count": 2,
      "oldest_ts": sentinel.event["oldest_ts"],
      "dropped_through_seq": 3,
      "run_id": "run-1",
      "control_run_id": "run-1",
    }
    assert isinstance(sentinel.event["oldest_ts"], float)
    assert (await asyncio.wait_for(subscriber.__anext__(), timeout=0.5)).seq == 4

    await subscriber.aclose()
    await bus.shutdown()

  _run(case())


def test_projected_visibility_filters_wrong_scope_but_preserves_same_user_unscoped_control() -> None:
  async def case() -> None:
    chat_run = _session(kind="chat", user_id="alice", session_id="chat-1", channel="tui")
    sessions = {"chat-1": chat_run, "chat-2": _session(kind="chat", session_id="chat-2", channel="tui")}
    auth = _FakeAuth(
      {
        "control-token": _session(kind="control", user_id="alice", session_id="control-1", channel="tui"),
        "wrong-channel": _session(kind="control", user_id="alice", session_id="control-2", channel="excel"),
        "other-chat": _session(kind="chat", user_id="alice", session_id="chat-2", channel="tui"),
      },
      sessions=sessions,
    )

    matching_bus = UserEventBus()
    matching_response = await _open_events(
      bus=matching_bus,
      auth=auth,
      token="control-token",
      control_run_id=None,
      schema_version="v1",
    )
    matching_task = asyncio.create_task(matching_response.body_iterator.__anext__())
    await asyncio.sleep(0)
    await matching_bus.publish("alice", "chat-1", {"type": "text_delta", "text": "visible", "run_id": "chat-1"})
    await matching_bus.publish("alice", "chat-1", {"type": "turn_complete", "turn": 1, "run_id": "chat-1"})
    assert _decode_data(await asyncio.wait_for(matching_task, timeout=0.5)) == {
      "run_id": "chat-1",
      "seq": 1,
      "event": {"type": "text_delta", "text": "visible", "run_id": "chat-1"},
    }
    await _close_response(matching_response)
    await matching_bus.shutdown()

    wrong_channel_bus = UserEventBus()
    wrong_channel_response = await _open_events(
      bus=wrong_channel_bus,
      auth=auth,
      token="wrong-channel",
      control_run_id=None,
      schema_version="v1",
    )
    wrong_channel_task = asyncio.create_task(wrong_channel_response.body_iterator.__anext__())
    await asyncio.sleep(0)
    await wrong_channel_bus.publish("alice", "chat-1", {"type": "text_delta", "text": "hidden", "run_id": "chat-1"})
    with pytest.raises(asyncio.TimeoutError):
      await asyncio.wait_for(asyncio.shield(wrong_channel_task), timeout=0.05)
    wrong_channel_task.cancel()
    with suppress(asyncio.CancelledError):
      await wrong_channel_task
    await _close_response(wrong_channel_response)
    await wrong_channel_bus.shutdown()

    other_chat_bus = UserEventBus()
    other_chat_response = await _open_events(
      bus=other_chat_bus,
      auth=auth,
      token="other-chat",
      control_run_id=None,
      schema_version="v1",
    )
    other_chat_task = asyncio.create_task(other_chat_response.body_iterator.__anext__())
    await asyncio.sleep(0)
    await other_chat_bus.publish("alice", "chat-1", {"type": "text_delta", "text": "hidden", "run_id": "chat-1"})
    with pytest.raises(asyncio.TimeoutError):
      await asyncio.wait_for(asyncio.shield(other_chat_task), timeout=0.05)
    other_chat_task.cancel()
    with suppress(asyncio.CancelledError):
      await other_chat_task
    await _close_response(other_chat_response)
    await other_chat_bus.shutdown()

  _run(case())


def test_projected_coalescing_merges_adjacent_deltas_and_flushes_before_non_delta() -> None:
  async def case() -> None:
    auth = _FakeAuth({"control-token": _session()})
    bus = UserEventBus()
    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "a", "run_id": "run-1"})
    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "b", "run_id": "run-1"})
    await bus.publish("alice", "run-1", {"type": "thinking_delta", "text": "c", "run_id": "run-1"})
    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "d", "run_id": "run-1"})
    await bus.publish(
      "alice",
      "run-1",
      {"type": "tool_call_start", "tool_call_id": "tool-1", "tool_name": "read", "tool_input": {}, "run_id": "run-1"},
    )

    response = await _open_events(bus=bus, auth=auth, schema_version="v1")
    assert await _read_data(response, 4) == [
      {"run_id": "run-1", "seq": 2, "event": {"type": "text_delta", "text": "ab", "run_id": "run-1"}},
      {"run_id": "run-1", "seq": 3, "event": {"type": "thinking_delta", "text": "c", "run_id": "run-1"}},
      {"run_id": "run-1", "seq": 4, "event": {"type": "text_delta", "text": "d", "run_id": "run-1"}},
      {
        "run_id": "run-1",
        "seq": 5,
        "event": {
          "type": "tool_call_start",
          "tool_call_id": "tool-1",
          "tool_name": "read",
          "tool_input": {},
          "run_id": "run-1",
        },
      },
    ]
    await bus.shutdown()

  _run(case())


def test_projected_coalescing_flushes_on_timeout_and_keepalive(monkeypatch: pytest.MonkeyPatch) -> None:
  async def case() -> None:
    monkeypatch.setattr(events_module, "_PROJECTED_COALESCE_SECONDS", 0.01)
    monkeypatch.setattr(events_module, "_PROJECTED_KEEPALIVE_SECONDS", 0.01)
    auth = _FakeAuth({"control-token": _session()})

    bus = UserEventBus()
    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "a", "run_id": "run-1"})
    await bus.publish("alice", "run-1", {"type": "text_delta", "text": "b", "run_id": "run-1"})
    response = await _open_events(bus=bus, auth=auth, schema_version="v1")
    assert await _read_data(response, 1, timeout=0.2) == [
      {"run_id": "run-1", "seq": 2, "event": {"type": "text_delta", "text": "ab", "run_id": "run-1"}}
    ]
    await bus.shutdown()

    empty_bus = UserEventBus()
    keepalive_response = await _open_events(bus=empty_bus, auth=auth, schema_version="v1")
    try:
      assert await asyncio.wait_for(keepalive_response.body_iterator.__anext__(), timeout=0.2) == b":keepalive\n\n"
    finally:
      await _close_response(keepalive_response)
      await empty_bus.shutdown()

  _run(case())
