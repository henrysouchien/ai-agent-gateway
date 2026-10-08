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
  assert callable(gateway_server._disconnect_stream_subscriber_for_backpressure)
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
