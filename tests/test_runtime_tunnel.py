"""The session tunnel stays inside AgentCore's WebSocket limits (64 KB frames, 250 frames/s)."""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from agent_gateway.runtime_tunnel import (
  MAX_FRAME_CHARS, MAX_FRAMES_PER_SECOND, RESP_MAX_BODY_BYTES,
  TunnelEndpoint, TunnelFrameError, TunnelRequest, TunnelResponse,
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


def _frame(**fields: object) -> str:
  return json.dumps(fields)


def test_malformed_request_frame_fails_that_request_and_never_reaches_the_handler() -> None:
  async def scenario() -> None:
    delivered: list[TunnelRequest] = []
    sent: list[dict] = []
    complete = asyncio.Event()

    async def send(text: str) -> None:
      frame = json.loads(text)
      sent.append(frame)
      complete.set()

    async def handler(request: TunnelRequest) -> TunnelResponse:
      delivered.append(request)

      async def chunks():
        yield b"ok"
      return TunnelResponse(200, [], chunks())

    endpoint = TunnelEndpoint(send=send, handler=handler, odd_ids=False)
    good = dict(t="req", method="POST", path="/p", headers=[["a", "b"]], body="ZXhhY3Q=")
    try:
      for index, bad in enumerate((
        {**good, "body": "!"}, {**good, "body": "ZXhhY3Q"}, {**good, "body": None}, {**good, "body": 5},
        {**good, "method": 7}, {**good, "path": None}, {**good, "headers": [["a"]]},
        {**good, "headers": [["a", 1]]}, {**good, "headers": {"a": "b"}},
        {key: value for key, value in good.items() if key != "body"},
      )):
        complete.clear()
        await endpoint.feed(json.dumps({**bad, "id": 2 * index + 1}))
        await asyncio.wait_for(complete.wait(), 1)
        assert sent[-1] == {"t": "fail", "id": 2 * index + 1, "error": "TunnelFrameError"}
      assert delivered == []
      complete.clear()
      await endpoint.feed(json.dumps({**good, "id": 99}))
      await asyncio.wait_for(complete.wait(), 1)
      assert delivered == [TunnelRequest("POST", "/p", [("a", "b")], b"exact")]
      assert not endpoint.closed
    finally:
      endpoint.close()
  asyncio.run(scenario())


@pytest.mark.parametrize("text", [
  "{", "[]", "null", _frame(t="frag", s=1, last=True), _frame(t="frag", s="x", last=1),
  _frame(t="req", id=True, method="GET", path="/", headers=[], body=""),
  _frame(t="req", id="1", method="GET", path="/", headers=[], body=""),
  _frame(t="control", kind="bootstrap", body=[]), _frame(t="control", kind="bootstrap"),
  _frame(t="control", kind=None, body={}), _frame(t="control", kind="bootstrap", body={"tunnel_features": "resp"}),
  _frame(t="control", kind="bootstrap", body={"tunnel_features": [1]}),
  _frame(t="resp", id=1, status=True, headers=[], b64=""), _frame(t="resp", id=1, status=200, headers=[], b64="!"),
  _frame(t="resp", id=1, status=200, headers=[], b64=None), _frame(t="head", id=1, status="200", headers=[]),
  _frame(t="head", id=1, status=200, headers=[["a", None]]), _frame(t="data", id=1, b64="!"),
  _frame(t="data", id=1), _frame(t="end", id=None), _frame(t="fail", id=1, error=None),
])
def test_malformed_frame_raises_the_declared_error_and_changes_nothing(text: str) -> None:
  async def scenario() -> None:
    controls: list[tuple[str, dict]] = []

    async def send(_text: str) -> None:
      return None

    async def on_control(kind: str, body: dict) -> None:
      controls.append((kind, body))

    endpoint = TunnelEndpoint(send=send, handler=None, on_control=on_control, odd_ids=False)
    request = asyncio.create_task(endpoint.request("GET", "/", [], b""))
    await asyncio.sleep(0)
    try:
      with pytest.raises(TunnelFrameError):
        await endpoint.feed(text)
      assert controls == []
      assert not endpoint.single_frame_responses
      assert not request.done()
    finally:
      endpoint.close()
      await asyncio.gather(request, return_exceptions=True)
  asyncio.run(scenario())


def test_exact_frames_still_decode_and_unknown_frame_kinds_stay_ignored() -> None:
  async def scenario() -> None:
    controls: list[tuple[str, dict]] = []

    async def send(_text: str) -> None:
      return None

    async def on_control(kind: str, body: dict) -> None:
      controls.append((kind, body))

    endpoint = TunnelEndpoint(send=send, handler=None, on_control=on_control, odd_ids=False)
    try:
      await endpoint.feed(_frame(t="control", kind="bootstrap", body={"tunnel_features": ["resp"], "x": 1}))
      assert controls == [("bootstrap", {"tunnel_features": ["resp"], "x": 1})]
      assert endpoint.single_frame_responses
      await endpoint.feed(_frame(t="control", kind="lease", body={}))
      assert controls[-1] == ("lease", {})
      await endpoint.feed(_frame(t="unheard-of", id=1))
      await endpoint.feed(_frame(t="resp", id=404, status=200, headers=[], b64=""))
    finally:
      endpoint.close()
  asyncio.run(scenario())
