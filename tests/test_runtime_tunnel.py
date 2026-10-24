"""The session tunnel stays inside AgentCore's WebSocket limits (64 KB frames, 250 frames/s)."""
from __future__ import annotations

import asyncio
import json
import os

from agent_gateway.runtime_tunnel import (
  MAX_FRAME_CHARS, MAX_FRAMES_PER_SECOND, RESP_MAX_BODY_BYTES,
  TunnelEndpoint, TunnelRequest, TunnelResponse,
)

def test_large_bodies_and_many_chunks_cross_within_frame_limits() -> None:
  async def scenario() -> None:
    frames: list[tuple[float, str]] = []
    to_runtime: asyncio.Queue[str] = asyncio.Queue()
    to_edge: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def edge_send(text: str) -> None:
      frames.append((loop.time(), text))
      await to_runtime.put(text)

    async def runtime_send(text: str) -> None:
      frames.append((loop.time(), text))
      await to_edge.put(text)

    body = os.urandom(300_000)  # a model request with its system prompt and tools

    async def handler(request: TunnelRequest) -> TunnelResponse:
      assert request.body == body

      async def chunks():
        for index in range(300):  # token deltas
          yield f"data: {index}\n\n".encode()
      return TunnelResponse(200, [("content-type", "text/event-stream")], chunks())

    edge = TunnelEndpoint(send=edge_send, handler=None, odd_ids=False)
    runtime = TunnelEndpoint(send=runtime_send, handler=handler, odd_ids=True)

    async def pump(queue: asyncio.Queue[str], endpoint: TunnelEndpoint) -> None:
      while True:
        await endpoint.feed(await queue.get())

    pumps = [asyncio.create_task(pump(to_runtime, runtime)), asyncio.create_task(pump(to_edge, edge))]
    response = await edge.request("POST", "/egress/anthropic/v1/messages", [], body)
    received = b"".join([chunk async for chunk in response.body])
    for task in pumps:
      task.cancel()
    assert received == b"".join(f"data: {index}\n\n".encode() for index in range(300))
    assert max(len(text) for _, text in frames) <= MAX_FRAME_CHARS
    # No one-second window carries more frames than the pace allows.
    times = sorted(at for at, _ in frames)
    for index, start in enumerate(times):
      in_window = sum(1 for at in times[index:] if at < start + 1.0)
      assert in_window <= MAX_FRAMES_PER_SECOND * 2  # two independent senders
  asyncio.run(scenario())


def test_a_cancelled_sender_never_leaves_a_partial_message() -> None:
  # Audit P1 on 554d099338: a caller cancelled between fragments made the peer
  # join the abandoned prefix to the next message.
  async def scenario() -> None:
    received: list[str] = []
    wire: asyncio.Queue[str] = asyncio.Queue()

    async def send(text: str) -> None:
      await asyncio.sleep(0.01)  # a slow socket, so cancellation lands mid-message
      await wire.put(text)

    async def on_control(kind: str, body: dict) -> None:
      received.append(body["payload"])

    sender = TunnelEndpoint(send=send, handler=None, odd_ids=False)
    receiver = TunnelEndpoint(send=send, handler=None, on_control=on_control, odd_ids=True)
    first, second = "a" * 90_000, "b" * 90_000
    task = asyncio.create_task(sender.control("probe", {"payload": first}))
    await asyncio.sleep(0.015)
    task.cancel()
    await sender.control("probe", {"payload": second})
    while not wire.empty():
      await receiver.feed(wire.get_nowait())
    assert received == [first, second]
  asyncio.run(scenario())


def test_a_request_cancelled_before_its_first_frame_never_sends() -> None:
  # Audit P2 on 1685672fc5: shielding alone dispatched a request its caller had
  # already cancelled while it waited behind another send.
  async def scenario() -> None:
    received: list[str] = []
    wire: asyncio.Queue[str] = asyncio.Queue()
    first_started = asyncio.Event()

    async def send(text: str) -> None:
      first_started.set()
      await asyncio.sleep(0.05)
      await wire.put(text)

    async def on_control(kind: str, body: dict) -> None:
      received.append(body["payload"])

    sender = TunnelEndpoint(send=send, handler=None, odd_ids=False)
    receiver = TunnelEndpoint(send=send, handler=None, on_control=on_control, odd_ids=True)
    first = asyncio.create_task(sender.control("probe", {"payload": "first"}))
    await first_started.wait()  # first holds the send lock
    queued = asyncio.create_task(sender.control("probe", {"payload": "abandoned"}))
    await asyncio.sleep(0.01)
    queued.cancel()
    await first
    await sender.control("probe", {"payload": "after"})
    while not wire.empty():
      await receiver.feed(wire.get_nowait())
    assert received == ["first", "after"]
  asyncio.run(scenario())


def test_single_frame_resp_negotiation_and_streaming_fallbacks() -> None:
  async def scenario() -> None:
    to_runtime: asyncio.Queue[str] = asyncio.Queue()
    to_edge: asyncio.Queue[str] = asyncio.Queue()
    runtime_frames: list[dict] = []

    async def edge_send(text: str) -> None:
      await to_runtime.put(text)

    async def runtime_send(text: str) -> None:
      runtime_frames.append(json.loads(text))
      await to_edge.put(text)

    async def handler(request: TunnelRequest) -> TunnelResponse:
      if request.path == "/sse":
        payload = b"data: hello\n\n"

        async def sse_chunks():
          yield payload

        return TunnelResponse(
          200,
          [("content-type", "text/event-stream; charset=utf-8"), ("content-length", str(len(payload)))],
          sse_chunks(),
        )
      if request.path == "/no-length":
        async def unlengthed():
          yield b'{"ok":true}'

        return TunnelResponse(200, [("content-type", "application/json")], unlengthed())
      if request.path == "/large":
        big = b"x" * (RESP_MAX_BODY_BYTES + 1)

        async def big_chunks():
          yield big

        return TunnelResponse(
          200,
          [("content-type", "application/json"), ("content-length", str(len(big)))],
          big_chunks(),
        )
      if request.path == "/huge-headers":
        medium = b"x" * 35_000

        async def medium_chunks():
          yield medium

        return TunnelResponse(
          200,
          [
            ("content-type", "application/json"),
            ("content-length", str(len(medium))),
            ("x-pad", "p" * 20_000),
          ],
          medium_chunks(),
        )
      body = b'{"status":"ok"}'

      async def ok_chunks():
        yield body

      return TunnelResponse(
        200,
        [("content-type", "application/json"), ("content-length", str(len(body)))],
        ok_chunks(),
      )

    edge = TunnelEndpoint(send=edge_send, handler=None, odd_ids=False)
    runtime = TunnelEndpoint(send=runtime_send, handler=handler, odd_ids=True)

    async def pump(queue: asyncio.Queue[str], endpoint: TunnelEndpoint) -> None:
      while True:
        await endpoint.feed(await queue.get())

    pumps = [asyncio.create_task(pump(to_runtime, runtime)), asyncio.create_task(pump(to_edge, edge))]
    try:
      # Inert before negotiation: small JSON response uses head + data + end.
      assert not runtime.single_frame_responses
      resp0 = await edge.request("GET", "/api/health", [], b"")
      assert b"".join([c async for c in resp0.body]) == b'{"status":"ok"}'
      assert [f["t"] for f in runtime_frames] == ["head", "data", "end"]
      runtime_frames.clear()

      # Bootstrap control frame with tunnel_features: ["resp"] enables single-frame responses.
      await edge.control("bootstrap", {"generation": 1, "tunnel_features": ["resp"]})
      await asyncio.sleep(0)
      assert runtime.single_frame_responses

      # 100 back-to-back small JSON responses take 100 "resp" frames (no pacer sleep at 67).
      loop = asyncio.get_running_loop()
      started = loop.time()
      for _ in range(100):
        resp = await edge.request("GET", "/api/health", [], b"")
        assert resp.status == 200
        assert b"".join([c async for c in resp.body]) == b'{"status":"ok"}'
      elapsed = loop.time() - started
      assert [f["t"] for f in runtime_frames] == ["resp"] * 100
      assert elapsed < 0.5
      runtime_frames.clear()

      # SSE, missing content-length, body > RESP_MAX_BODY_BYTES, and frame > MAX_FRAME_CHARS still stream.
      for path, expected_len in (
        ("/sse", len(b"data: hello\n\n")),
        ("/no-length", len(b'{"ok":true}')),
        ("/large", RESP_MAX_BODY_BYTES + 1),
        ("/huge-headers", 35_000),
      ):
        resp = await edge.request("GET", path, [], b"")
        body = b"".join([c async for c in resp.body])
        assert len(body) == expected_len
        types = [f["t"] for f in runtime_frames]
        assert "resp" not in types
        assert types[0] == "head" and types[-1] == "end"
        runtime_frames.clear()
    finally:
      for task in pumps:
        task.cancel()
  asyncio.run(scenario())
