"""The session tunnel stays inside AgentCore's WebSocket limits (64 KB frames, 250 frames/s)."""
from __future__ import annotations

import asyncio
import os

from agent_gateway.runtime_tunnel import (
  MAX_FRAME_CHARS, MAX_FRAMES_PER_SECOND, TunnelEndpoint, TunnelRequest, TunnelResponse,
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
