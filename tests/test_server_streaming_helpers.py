from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from functools import partial
from types import AsyncGeneratorType
from typing import Any, Coroutine


import pytest

from agent_gateway import server as gateway_server
from agent_gateway import server_chat_helpers
from agent_gateway import server_chat_stream_core
from agent_gateway import server_streaming
from agent_gateway.event_log import EventLog, LogEntry
from agent_gateway.session import GatewaySession, SessionStream, StreamSubscriber



def test_streaming_parent_aliases_are_available() -> None:
  assert gateway_server._server_streaming is server_streaming
  assert callable(gateway_server._pump_stream_subscriber)
  assert callable(gateway_server._register_stream_subscriber)
  assert callable(gateway_server._cleanup_stream_subscriber)
  assert callable(gateway_server._stream_subscriber_sse)
  assert gateway_server._register_stream_subscriber.__module__ == "agent_gateway.server"


def test_register_stream_subscriber_injects_live_parent_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
  captured: dict[str, object] = {}
  sentinel_subscriber = object()
  sentinel_helpers = object()

  def fake_register(*args, **kwargs):
    captured["args"] = args
    captured.update(kwargs)
    return sentinel_subscriber

  async def fake_pump(*_args, **_kwargs):
    return None

  monkeypatch.setattr(gateway_server, "_server_chat_helpers", sentinel_helpers)
  monkeypatch.setattr(gateway_server, "_STREAM_SUBSCRIBER_QUEUE_MAX", 17)
  monkeypatch.setattr(gateway_server, "_pump_stream_subscriber", fake_pump)
  monkeypatch.setattr(server_streaming, "register_stream_subscriber", fake_register)

  result = gateway_server._register_stream_subscriber(
    "active-turn",
    after_seq=42,
    client_label="client-a",
  )

  assert result is sentinel_subscriber
  assert captured["args"] == ("active-turn",)
  assert captured["after_seq"] == 42
  assert captured["client_label"] == "client-a"
  deps = captured["deps"]
  assert isinstance(deps, server_streaming.StreamingDeps)
  assert deps.chat_helpers is sentinel_helpers
  assert deps.queue_max == 17
  assert deps.pump_stream_subscriber is fake_pump


async def _collect_sse_chunks() -> list[object]:
  return [
    chunk
    async for chunk in gateway_server._stream_subscriber_sse(
      active_turn="turn",
      subscriber="subscriber",
    )
  ]


def test_stream_subscriber_sse_injects_live_parent_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
  captured: dict[str, object] = {}
  sentinel_helpers = object()

  async def fake_sse(**kwargs):
    captured.update(kwargs)
    yield {"event": "ready"}

  async def fake_cleanup(*_args, **_kwargs):
    return None

  monkeypatch.setattr(gateway_server, "_server_chat_helpers", sentinel_helpers)
  monkeypatch.setattr(gateway_server, "_STREAM_SUBSCRIBER_KEEPALIVE_SECONDS", 3.5)
  monkeypatch.setattr(gateway_server, "_STREAM_SUBSCRIBER_DONE", object())
  monkeypatch.setattr(gateway_server, "_cleanup_stream_subscriber", fake_cleanup)
  monkeypatch.setattr(server_streaming, "stream_subscriber_sse", fake_sse)

  chunks = asyncio.run(_collect_sse_chunks())

  assert chunks == [{"event": "ready"}]
  assert captured["active_turn"] == "turn"
  assert captured["subscriber"] == "subscriber"
  deps = captured["deps"]
  assert isinstance(deps, server_streaming.StreamingDeps)
  assert deps.chat_helpers is sentinel_helpers
  assert deps.keepalive_seconds == 3.5
  assert deps.done_marker is gateway_server._STREAM_SUBSCRIBER_DONE
  assert deps.cleanup_stream_subscriber is fake_cleanup


def test_chat_helpers_stream_sse_closes_core_generator_on_outer_aclose(monkeypatch: pytest.MonkeyPatch) -> None:
  cleanup_complete = False

  async def fake_core_sse(**_kwargs):
    nonlocal cleanup_complete
    try:
      yield b"data: first\n\n"
      await asyncio.Event().wait()
    finally:
      cleanup_complete = True

  monkeypatch.setattr(server_chat_helpers._chat_stream_core, "stream_subscriber_sse", fake_core_sse)

  async def run() -> None:
    session = GatewaySession(
      session_id="session-1",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="user-1",
    )
    active_turn = SessionStream(event_log=EventLog(), runner_task=None)
    subscriber = StreamSubscriber(
      subscriber_id="subscriber-1",
      connected_at=1.0,
      last_sent_seq=0,
      queue=asyncio.Queue(),
    )
    stream = server_chat_helpers._stream_subscriber_sse(
      session=session,
      active_turn=active_turn,
      subscriber=subscriber,
      transcript_dir=None,
      channel=None,
      write_transcript=False,
      log=logging.getLogger("test.server_chat_helpers"),
    )
    assert isinstance(stream, AsyncGeneratorType)
    assert await stream.__anext__() == b"data: first\n\n"
    await stream.aclose()

  asyncio.run(run())

  assert cleanup_complete


@pytest.mark.parametrize(
  ("visible_at", "keepalive_times"),
  [(None, [10.0, 20.0, 30.0]), (8, [18.0, 28.0])],
)
def test_stream_keepalive_tracks_wire_idle_during_filtered_traffic(
  visible_at: int | None,
  keepalive_times: list[float],
) -> None:
  async def run() -> None:
    now = 0.0
    done_marker = object()
    session = GatewaySession(
      session_id="session-1",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="user-1",
      schema_version=1,
    )
    active_turn = SessionStream(event_log=EventLog(), runner_task=None)
    subscriber = StreamSubscriber(
      subscriber_id="subscriber-1",
      connected_at=0.0,
      last_sent_seq=0,
      queue=asyncio.Queue(),
    )
    active_turn.subscribers[subscriber.subscriber_id] = subscriber
    arrivals: deque[tuple[float, object]] = deque(
      (
        float(seq),
        LogEntry(
          seq=seq,
          timestamp=float(seq),
          event=(
            {"type": "text_delta", "text": "hello"}
            if seq == visible_at
            else {"type": "assistant_message", "content": "internal"}
          ),
        ),
      )
      for seq in range(1, 35)
    )
    arrivals.append((35.0, done_marker))

    async def scheduled_wait_for(
      awaitable: Coroutine[Any, Any, object],
      *,
      timeout: float,
    ) -> object:
      nonlocal now
      # Advance virtual time to whichever happens first: arrival or timeout.
      arrival_time, item = arrivals[0]
      if timeout <= 0 or now + timeout < arrival_time:
        now += timeout
        awaitable.close()
        raise asyncio.TimeoutError
      now = arrival_time
      arrivals.popleft()
      subscriber.queue.put_nowait(item)
      return await awaitable

    chunks = [
      (now, chunk)
      async for chunk in server_chat_stream_core.stream_subscriber_sse(
        session=session,
        active_turn=active_turn,
        subscriber=subscriber,
        transcript_dir=None,
        channel=None,
        write_transcript=False,
        log=logging.getLogger("test.server_chat_stream_core"),
        keepalive_seconds=10.0,
        done_marker=done_marker,
        cleanup_stream_subscriber=partial(
          server_chat_stream_core.cleanup_stream_subscriber,
          gather_fn=asyncio.gather,
        ),
        wait_for_fn=scheduled_wait_for,
        timeout_error=asyncio.TimeoutError,
        event_for_wire_fn=partial(
          server_chat_stream_core.event_for_wire,
          product_id_resolver=lambda: None,
        ),
        write_turn_transcript_entry_fn=lambda *_args, **_kwargs: None,
        monotonic_fn=lambda: now,
      )
    ]

    assert [at for at, chunk in chunks if chunk == b":keepalive\n\n"] == keepalive_times
    data = [(at, json.loads(chunk[len(b"data: "):])) for at, chunk in chunks if chunk.startswith(b"data: ")]
    if visible_at is None:
      assert data == []
    else:
      assert data == [
        (
          float(visible_at),
          {
            "seq": visible_at,
            "session_id": session.session_id,
            "schema_version": 1,
            "event": {"type": "text_delta", "text": "hello"},
          },
        )
      ]
    assert len(chunks) == len(keepalive_times) + (visible_at is not None)
    assert subscriber.last_sent_seq == 34
    assert not arrivals
    assert active_turn.subscribers == {}

  asyncio.run(run())


def _citation_hold_log(
  events: list[dict[str, Any]],
  *,
  first_seq: int = 1,
) -> EventLog:
  event_log = EventLog(session_id="session-1", first_seq=first_seq)
  for event in events:
    assert event_log.append(event) is not None
  if not event_log.closed:
    event_log.close_without_terminal()
  return event_log


async def _collect_citation_hold_stream(
  event_log: EventLog,
  *,
  after_seq: int = 0,
  transcript_entries: list[tuple[int, dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[int | None, list[dict[str, Any]]]]:
  session = GatewaySession(
    session_id=event_log.session_id,
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="user-1",
    schema_version=1,
  )
  active_turn = SessionStream(event_log=event_log, runner_task=None)
  done_marker = object()
  subscriber = server_chat_stream_core.register_stream_subscriber(
    active_turn,
    after_seq=after_seq,
    client_label="citation-hold-test",
    queue_max=100,
    queue_factory=asyncio.Queue,
    task_factory=asyncio.create_task,
    uuid_hex_factory=lambda: "citation-hold",
    time_fn=lambda: 0.0,
    default_pump_stream_subscriber=partial(
      server_chat_stream_core.pump_stream_subscriber,
      done_marker=done_marker,
    ),
  )
  envelopes: list[dict[str, Any]] = []
  before_input: dict[int | None, list[dict[str, Any]]] = {}
  durable_entries = event_log.entries

  async def observed_wait_for(
    awaitable: Coroutine[Any, Any, object],
    *,
    timeout: float,
  ) -> object:
    item = await asyncio.wait_for(awaitable, timeout=timeout)
    # The previous input has finished yielding before the next queue read.
    key = item.seq if isinstance(item, LogEntry) else None
    before_input[key] = [envelope["event"] for envelope in envelopes]
    return item

  def record_transcript(
    _active_turn: SessionStream,
    seq: int,
    event: dict[str, Any],
    **_kwargs: Any,
  ) -> None:
    if transcript_entries is not None:
      transcript_entries.append((seq, event))

  async def consume() -> None:
    async for chunk in server_chat_stream_core.stream_subscriber_sse(
      session=session,
      active_turn=active_turn,
      subscriber=subscriber,
      transcript_dir=None,
      channel=None,
      write_transcript=transcript_entries is not None,
      log=logging.getLogger("test.citation_hold_stream"),
      keepalive_seconds=0.05,
      done_marker=done_marker,
      cleanup_stream_subscriber=partial(
        server_chat_stream_core.cleanup_stream_subscriber,
        gather_fn=asyncio.gather,
      ),
      wait_for_fn=observed_wait_for,
      timeout_error=asyncio.TimeoutError,
      event_for_wire_fn=partial(
        server_chat_stream_core.event_for_wire,
        product_id_resolver=lambda: None,
      ),
      write_turn_transcript_entry_fn=record_transcript,
    ):
      if chunk.startswith(b"data: "):
        envelopes.append(json.loads(chunk[len(b"data: "):]))

  await asyncio.wait_for(consume(), timeout=5.0)
  sequences = [envelope["seq"] for envelope in envelopes]
  assert sequences == sorted(set(sequences))
  assert all(seq > after_seq for seq in sequences)
  assert event_log.entries == durable_entries
  assert active_turn.subscribers == {}
  return envelopes, before_input


@pytest.mark.parametrize(
  "release_event",
  [
    {
      "type": "tool_calls_planned",
      "turn": 1,
      "calls": [{"tool_call_id": "call-1", "tool_name": "search"}],
    },
    {
      "type": "tool_call_start",
      "tool_call_id": "call-1",
      "tool_name": "search",
      "tool_input": {"query": "filing"},
    },
    {
      "type": "tool_approval_request",
      "tool_call_id": "call-1",
      "tool_name": "search",
      "tool_input": {"query": "filing"},
      "reason": "Approval required",
    },
  ],
  ids=["planned", "start", "approval"],
)
def test_citation_hold_tool_signal_releases_every_queued_event_in_log_order(
  release_event: dict[str, Any],
) -> None:
  async def run() -> None:
    events = [
      {"type": "citation_hold_active", "mode": "hold"},
      {"type": "thinking_delta", "text": "Before any answer text"},
      {"type": "text_delta", "text": "I will search the filing."},
      {"type": "thinking_delta", "text": "Queued after the first text"},
      {"type": "stream_retry", "attempt": 1},
      release_event,
      {"type": "tool_call_complete", "tool_call_id": "call-1", "tool_name": "search", "result": "found"},
      {"type": "turn_complete", "turn": 1},
      {"type": "text_delta", "text": "The sourced final answer."},
      {"type": "thinking_delta", "text": "Queued final-answer reasoning"},
      {"type": "turn_complete", "turn": 2},
      {"type": "citation_check_started", "turn": 2, "mode": "hold"},
      {"type": "stream_complete", "terminal_disposition": "completed"},
    ]
    event_log = _citation_hold_log(events)
    envelopes, before_input = await _collect_citation_hold_stream(event_log)

    assert before_input[3] == events[:2]
    assert before_input[6] == events[:2]
    assert before_input[7] == events[:6]
    assert before_input[9] == events[:8]
    assert before_input[10] == events[:8]
    assert before_input[13] == events[:8]
    assert [envelope["event"] for envelope in envelopes] == events
    assert [envelope["seq"] for envelope in envelopes] == list(range(1, len(events) + 1))

  asyncio.run(run())


@pytest.mark.parametrize(
  "terminal",
  [
    {"type": "stream_complete", "terminal_disposition": "completed"},
    {"type": "error", "error": "Finalization failed"},
  ],
  ids=["complete", "error"],
)
def test_citation_hold_pass_keeps_text_and_check_queued_until_terminal(
  terminal: dict[str, Any],
) -> None:
  async def run() -> None:
    events = [
      {"type": "citation_hold_active", "mode": "hold"},
      {"type": "text_delta", "text": "A sourced "},
      {"type": "thinking_delta", "text": "Still preparing"},
      {"type": "text_delta", "text": "answer."},
      {"type": "assistant_message", "content": "A sourced answer."},
      {"type": "turn_complete", "turn": 1},
      {"type": "citation_check_started", "turn": 1, "mode": "hold"},
      {"type": "citation_validation", "turn": 1, "violation_count": 0, "revised": False},
      terminal,
    ]
    event_log = _citation_hold_log(events)
    envelopes, before_input = await _collect_citation_hold_stream(event_log)

    for seq in range(3, len(events) + 1):
      assert before_input[seq] == events[:1]
    visible_entries = [entry for entry in event_log.entries if entry.event["type"] != "assistant_message"]
    assert [envelope["event"] for envelope in envelopes] == [entry.event for entry in visible_entries]
    assert [envelope["seq"] for envelope in envelopes] == [entry.seq for entry in visible_entries]
    assert [entry.event for entry in event_log.entries] == events

  asyncio.run(run())


def test_citation_hold_revision_omits_only_draft_text_and_streams_revision_immediately() -> None:
  async def run() -> None:
    events = [
      {"type": "citation_hold_active", "mode": "hold"},
      {"type": "text_delta", "text": "Rejected draft: "},
      {"type": "thinking_delta", "text": "Retain this nontext event"},
      {"type": "text_delta", "text": "unsupported claim."},
      {"type": "assistant_message", "content": "Rejected draft: unsupported claim."},
      {"type": "turn_complete", "turn": 1},
      {"type": "citation_check_started", "turn": 1, "mode": "hold"},
      {"type": "citation_validation", "turn": 1, "violation_count": 1, "revised": True},
      {"type": "answer_revision_started", "turn": 1, "mode": "hold", "violation_count": 1},
      {"type": "thinking_delta", "text": "Revising"},
      {"type": "text_delta", "text": "Corrected answer."},
      {"type": "turn_complete", "turn": 2},
      {"type": "stream_complete", "terminal_disposition": "completed"},
    ]
    event_log = _citation_hold_log(events)
    transcript_entries: list[tuple[int, dict[str, Any]]] = []
    envelopes, before_input = await _collect_citation_hold_stream(
      event_log,
      transcript_entries=transcript_entries,
    )
    retained = [events[0], events[2], *events[5:]]

    assert before_input[9] == events[:1]
    assert before_input[10] == retained[:6]
    assert before_input[12] == retained[:8]
    assert [envelope["event"] for envelope in envelopes] == retained
    assert [envelope["seq"] for envelope in envelopes] == [1, 3, *range(6, 14)]
    assert [envelope["event"]["text"] for envelope in envelopes if envelope["event"]["type"] == "text_delta"] == [
      "Corrected answer.",
    ]
    assert [entry.event for entry in event_log.entries] == events
    assert transcript_entries == [(entry.seq, entry.event) for entry in event_log.entries]

  asyncio.run(run())


@pytest.mark.parametrize("check_mode", [None, "append", "hold"], ids=["off", "append", "legacy-hold"])
def test_citation_hold_without_hold_marker_leaves_stream_immediate(
  check_mode: str | None,
) -> None:
  async def run() -> None:
    events: list[dict[str, Any]] = [
      {"type": "text_delta", "text": "Original answer."},
      {"type": "thinking_delta", "text": "Still immediate"},
      {"type": "turn_complete", "turn": 1},
    ]
    if check_mode is not None:
      events.extend([
        {"type": "citation_check_started", "turn": 1, "mode": check_mode},
        {"type": "answer_revision_started", "turn": 1, "mode": check_mode, "violation_count": 1},
        {"type": "text_delta", "text": "Revised answer."},
      ])
    events.append({"type": "stream_complete", "terminal_disposition": "completed"})
    envelopes, before_input = await _collect_citation_hold_stream(_citation_hold_log(events))

    for seq in range(1, len(events) + 1):
      assert before_input[seq] == events[:seq - 1]
    assert [envelope["event"] for envelope in envelopes] == events
    assert [envelope["seq"] for envelope in envelopes] == list(range(1, len(events) + 1))

  asyncio.run(run())


def test_citation_hold_subscription_end_flushes_queue_without_a_terminal_event() -> None:
  async def run() -> None:
    events = [
      {"type": "citation_hold_active", "mode": "hold"},
      {"type": "text_delta", "text": "Answer pending stream closure."},
      {"type": "thinking_delta", "text": "Queued nontext"},
      {"type": "turn_complete", "turn": 1},
      {"type": "citation_check_started", "turn": 1, "mode": "hold"},
    ]
    event_log = _citation_hold_log(events)
    envelopes, before_input = await _collect_citation_hold_stream(event_log)

    assert before_input[None] == events[:1]
    assert [envelope["event"] for envelope in envelopes] == events
    assert not event_log.has_terminal

  asyncio.run(run())


def test_citation_hold_next_turn_text_releases_unchecked_completed_turn() -> None:
  async def run() -> None:
    events = [
      {"type": "citation_hold_active", "mode": "hold"},
      {"type": "text_delta", "text": "First turn."},
      {"type": "thinking_delta", "text": "First turn reasoning"},
      {"type": "turn_complete", "turn": 1},
      {"type": "text_delta", "text": "Second turn."},
      {"type": "thinking_delta", "text": "Second turn reasoning"},
      {"type": "turn_complete", "turn": 2},
      {"type": "citation_check_started", "turn": 2, "mode": "hold"},
      {"type": "stream_complete", "terminal_disposition": "completed"},
    ]
    envelopes, before_input = await _collect_citation_hold_stream(_citation_hold_log(events))

    assert before_input[5] == events[:1]
    assert before_input[6] == events[:4]
    assert before_input[9] == events[:4]
    assert [envelope["event"] for envelope in envelopes] == events

  asyncio.run(run())


@pytest.mark.parametrize("first_seq", [1, 41], ids=["initial", "continuation"])
@pytest.mark.parametrize("revised", [False, True], ids=["pass", "revision"])
def test_citation_hold_reconnect_reconstructs_projection_before_after_seq(
  first_seq: int,
  revised: bool,
) -> None:
  async def run() -> None:
    events = [
      {"type": "citation_hold_active", "mode": "hold"},
      {"type": "text_delta", "text": "Original draft."},
      {"type": "thinking_delta", "text": "Retained nontext"},
      {"type": "turn_complete", "turn": 1},
      {"type": "citation_check_started", "turn": 1, "mode": "hold"},
    ]
    if revised:
      events.extend([
        {"type": "answer_revision_started", "turn": 1, "mode": "hold", "violation_count": 1},
        {"type": "text_delta", "text": "Corrected answer."},
        {"type": "turn_complete", "turn": 2},
      ])
    events.append({"type": "stream_complete", "terminal_disposition": "completed"})
    event_log = _citation_hold_log(events, first_seq=first_seq)
    full, _ = await _collect_citation_hold_stream(event_log)
    visible_entries = [
      entry
      for entry in event_log.entries
      if not (revised and entry.event["type"] == "text_delta" and entry.event["text"] == "Original draft.")
    ]
    assert [envelope["event"] for envelope in full] == [entry.event for entry in visible_entries]
    assert [envelope["seq"] for envelope in full] == [entry.seq for entry in visible_entries]

    # Include cursors inside the draft, after its turn_complete/check, and after
    # revision_started, even though a held draft never supplies a wire cursor.
    for after_seq in range(first_seq - 1, event_log.next_seq):
      resumed, before_input = await _collect_citation_hold_stream(event_log, after_seq=after_seq)
      assert resumed == [envelope for envelope in full if envelope["seq"] > after_seq]
      if revised:
        assert all(
          envelope["event"].get("text") != "Original draft."
          for envelope in resumed
          if envelope["event"]["type"] == "text_delta"
        )
        revision_seq = first_seq + 5
        if first_seq <= after_seq < revision_seq:
          assert before_input[revision_seq] == []
      else:
        terminal_seq = event_log.next_seq - 1
        if first_seq <= after_seq < terminal_seq:
          assert before_input[terminal_seq] == []

  asyncio.run(run())


def test_stream_subscriber_sse_emits_terminal_when_pump_ends_abnormally() -> None:
  def build_turn() -> tuple[GatewaySession, SessionStream, StreamSubscriber, EventLog]:
    session = GatewaySession(
      session_id="session-1",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="user-1",
    )
    event_log = EventLog(session_id=session.session_id)
    active_turn = SessionStream(event_log=event_log, runner_task=None)
    subscriber = StreamSubscriber(
      subscriber_id="subscriber-1",
      connected_at=0.0,
      last_sent_seq=0,
      queue=asyncio.Queue(),
    )
    active_turn.subscribers[subscriber.subscriber_id] = subscriber
    return session, active_turn, subscriber, event_log

  async def wire_events(
    session: GatewaySession,
    active_turn: SessionStream,
    subscriber: StreamSubscriber,
  ) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    async def consume() -> None:
      async for chunk in server_chat_helpers._stream_subscriber_sse(
        session=session,
        active_turn=active_turn,
        subscriber=subscriber,
        transcript_dir=None,
        channel=None,
        write_transcript=False,
        log=logging.getLogger("test.server_chat_helpers"),
        keepalive_seconds=0.05,
      ):
        if chunk.startswith(b"data: "):
          events.append(json.loads(chunk[len(b"data: "):])["event"])

    # A stream that loops on `:keepalive` instead of ending trips this timeout.
    await asyncio.wait_for(consume(), timeout=5.0)
    return events

  async def pump_raises_after_first_delta() -> None:
    session, active_turn, subscriber, event_log = build_turn()
    first = event_log.append({"type": "text_delta", "text": "hello"})
    assert first is not None

    async def failing_pump() -> None:
      await subscriber.queue.put(first)
      raise RuntimeError("pump boom")

    subscriber.pump_task = asyncio.create_task(failing_pump())

    events = await wire_events(session, active_turn, subscriber)

    assert [event["type"] for event in events] == ["text_delta", "error"]
    assert events[-1]["error"] == "pump boom"
    assert active_turn.subscribers == {}

  async def pump_cancelled_after_interrupted_terminal() -> None:
    session, active_turn, subscriber, event_log = build_turn()
    first = event_log.append({"type": "text_delta", "text": "hello"})
    assert first is not None
    event_log.append({"type": "interrupted", "reason": "turn_cancelled"})
    event_log.append({
      "type": "stream_complete",
      "terminal_disposition": "interrupted",
      "reason": "turn_cancelled",
    })
    subscriber.queue.put_nowait(first)
    pump_task = asyncio.create_task(asyncio.Event().wait())
    subscriber.pump_task = pump_task
    pump_task.cancel()
    await asyncio.gather(pump_task, return_exceptions=True)
    assert pump_task.cancelled()

    events = await wire_events(session, active_turn, subscriber)

    assert [event["type"] for event in events][-2:] == ["interrupted", "stream_complete"]
    assert events[0]["type"] == "text_delta"
    assert events[-1]["terminal_disposition"] == "interrupted"
    assert active_turn.subscribers == {}

  async def run() -> None:
    await pump_raises_after_first_delta()
    await pump_cancelled_after_interrupted_terminal()

  asyncio.run(run())


def test_lingering_subscriber_on_cleared_turn_never_settles_next_turn_pending_client_tool(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(server_chat_helpers, "_ACTIVE_TURN_GRACE_SECONDS", 0.02)
  monkeypatch.setattr(server_chat_stream_core.server_models, "_ACTIVE_TURN_GRACE_SECONDS", 0.02)

  async def run() -> None:
    session = GatewaySession(
      session_id="session-1",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="user-1",
      result_queue=asyncio.Queue(),
    )
    turn_a_log = EventLog(session_id=session.session_id)
    turn_a_log.append({"type": "text_delta", "text": "a"})
    turn_a_log.append({"type": "stream_complete", "terminal_disposition": "completed"})
    turn_a_log.close()
    turn_a = SessionStream(event_log=turn_a_log, runner_task=None, session=session)
    session.active_turn = turn_a
    sub_a = StreamSubscriber(
      subscriber_id="sub-a",
      connected_at=0.0,
      last_sent_seq=0,
      queue=asyncio.Queue(),
    )
    turn_a.subscribers[sub_a.subscriber_id] = sub_a

    # Turn A clears to last_completed_turn while sub_a is still attached; Turn B starts with a pending client tool.
    server_chat_helpers._clear_active_turn(session, turn_a)
    turn_b = SessionStream(event_log=EventLog(session_id=session.session_id), runner_task=None, session=session)
    session.active_turn = turn_b
    session.pending_tools["call-b"] = {"status": "pending"}

    await server_chat_helpers._cleanup_stream_subscriber(turn_a, sub_a.subscriber_id)
    assert turn_a.client_disconnect_handle is None
    await asyncio.sleep(0.05)
    assert session.pending_tools == {"call-b": {"status": "pending"}}
    assert session.result_queue is not None and session.result_queue.empty()

  asyncio.run(run())
