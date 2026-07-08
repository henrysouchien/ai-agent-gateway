from __future__ import annotations

import asyncio
import logging

import pytest

from agent_gateway import server as gateway_server
from agent_gateway import server_chat_helpers
from agent_gateway import server_streaming


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
    stream = server_chat_helpers._stream_subscriber_sse(
      session=object(),
      active_turn=object(),
      subscriber=object(),
      transcript_dir=None,
      channel=None,
      write_transcript=False,
      log=logging.getLogger("test.server_chat_helpers"),
    )
    assert await stream.__anext__() == b"data: first\n\n"
    await stream.aclose()

  asyncio.run(run())

  assert cleanup_complete
