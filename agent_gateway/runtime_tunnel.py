"""One multiplexed HTTP tunnel between the trusted edge and a hosted user runtime.

A hosted runtime (an AgentCore microVM session) is reachable only through the
session's WebSocket, which the edge opens; the runtime cannot dial the shared
tier. Both directions ride that one socket as JSON text frames:

- edge -> runtime: product HTTP requests (chat, control, artifacts), plus the
  session's ``bootstrap`` and ``credentials`` control frames;
- runtime -> edge: egress requests to allowlisted shared services, which the
  edge executes with credentials the runtime never holds.

Frames (``id`` is unique per originating side; the edge uses even ids, the
runtime odd ids, so the two sides never collide):

  {"t": "req",  "id": n, "method": str, "path": str, "headers": [[k, v]], "body": b64}
  {"t": "head", "id": n, "status": int, "headers": [[k, v]]}
  {"t": "data", "id": n, "b64": str}
  {"t": "end",  "id": n}
  {"t": "fail", "id": n, "error": str}
  {"t": "control", "kind": str, "body": {...}}   # bootstrap / credentials / ready
  {"t": "frag", "last": bool, "s": str}          # consecutive pieces of one larger frame

A request body travels whole; a response streams (SSE stays incremental).
"""
from __future__ import annotations

import asyncio
import base64
import collections
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
import itertools
import json
from typing import Any

Headers = list[tuple[str, str]]

# AgentCore session WebSocket limits: 64 KB per frame and 250 frames per
# second per connection; exceeding either closes the socket. Frames are ASCII
# JSON, so characters are bytes.
MAX_FRAME_CHARS = 60_000
MAX_FRAMES_PER_SECOND = 200


@dataclass(frozen=True)
class TunnelRequest:
  method: str
  path: str
  headers: Headers
  body: bytes


@dataclass
class TunnelResponse:
  status: int
  headers: Headers
  body: AsyncIterator[bytes]


@dataclass
class _Pending:
  head: asyncio.Future[tuple[int, Headers]]
  chunks: asyncio.Queue[bytes | BaseException | None] = field(default_factory=asyncio.Queue)


class TunnelClosed(ConnectionError):
  """The socket ended; in-flight requests on it fail."""


Handler = Callable[[TunnelRequest], Awaitable[TunnelResponse]]
ControlHandler = Callable[[str, dict[str, Any]], Awaitable[None]]


def _pairs(raw: Any) -> Headers:
  return [(str(name), str(value)) for name, value in raw or ()]


class TunnelEndpoint:
  """One side of the tunnel. ``feed`` every received text frame; ``request`` to call the peer."""

  def __init__(
    self, *, send: Callable[[str], Awaitable[None]], handler: Handler | None,
    on_control: ControlHandler | None = None, odd_ids: bool,
  ) -> None:
    self._send_text = send
    self._handler = handler
    self._on_control = on_control
    self._ids = itertools.count(1 if odd_ids else 2, 2)
    self._pending: dict[int, _Pending] = {}
    self._tasks: set[asyncio.Task[None]] = set()
    self._send_lock = asyncio.Lock()
    self._closed: BaseException | None = None
    self._fragments: list[str] = []
    self._sent_at: collections.deque[float] = collections.deque()

  async def _paced_send(self, text: str, started: asyncio.Event | None = None) -> None:
    # AgentCore closes a session WebSocket that exceeds its frame rate.
    loop = asyncio.get_running_loop()
    while len(self._sent_at) >= MAX_FRAMES_PER_SECOND:
      wait = self._sent_at[0] + 1.0 - loop.time()
      if wait > 0:
        await asyncio.sleep(wait)
      self._sent_at.popleft()
    self._sent_at.append(loop.time())
    if started is not None:
      started.set()
    await self._send_text(text)

  async def _send(self, frame: dict[str, Any]) -> None:
    if self._closed is not None:
      raise TunnelClosed(str(self._closed))
    text = json.dumps(frame, separators=(",", ":"))
    # A message is abandoned or sent whole, never part-sent: cancellation
    # before its first frame drops it, but a caller cancelled mid-message
    # must not leave the peer joining the abandoned prefix to the next one.
    started = asyncio.Event()
    task = asyncio.ensure_future(self._send_whole(text, started))
    task.add_done_callback(lambda done: done.cancelled() or done.exception())
    try:
      await asyncio.shield(task)
    except asyncio.CancelledError:
      if not started.is_set():
        task.cancel()
      raise

  async def _send_whole(self, text: str, started: asyncio.Event) -> None:
    async with self._send_lock:
      if len(text) <= MAX_FRAME_CHARS:
        await self._paced_send(text, started)
        return
      # AgentCore also closes on any frame over 64 KB: a large frame travels as
      # consecutive fragments, which the lock keeps from interleaving.
      # Quotes and backslashes are escaped again inside a fragment, so a piece
      # holds at most half a frame.
      step = (MAX_FRAME_CHARS - 64) // 2
      for start in range(0, len(text), step):
        last = start + step >= len(text)
        await self._paced_send(json.dumps(
          {"t": "frag", "last": last, "s": text[start:start + step]}, separators=(",", ":"),
        ), started if start == 0 else None)

  async def control(self, kind: str, body: dict[str, Any]) -> None:
    await self._send({"t": "control", "kind": kind, "body": body})

  async def request(
    self, method: str, path: str, headers: Sequence[tuple[str, str]], body: bytes,
  ) -> TunnelResponse:
    request_id = next(self._ids)
    pending = _Pending(head=asyncio.get_running_loop().create_future())
    self._pending[request_id] = pending
    try:
      await self._send({
        "t": "req", "id": request_id, "method": method, "path": path,
        "headers": [list(pair) for pair in headers],
        "body": base64.b64encode(body).decode("ascii"),
      })
      status, response_headers = await pending.head
    except BaseException:
      self._pending.pop(request_id, None)
      raise

    async def chunks() -> AsyncIterator[bytes]:
      try:
        while True:
          item = await pending.chunks.get()
          if item is None:
            return
          if isinstance(item, BaseException):
            raise item
          yield item
      finally:
        self._pending.pop(request_id, None)

    return TunnelResponse(status, response_headers, chunks())

  async def feed(self, text: str) -> None:
    frame = json.loads(text)
    kind = frame.get("t")
    if kind == "frag":
      self._fragments.append(str(frame.get("s") or ""))
      if not frame.get("last"):
        return
      whole, self._fragments = "".join(self._fragments), []
      await self.feed(whole)
      return
    if kind == "req":
      task = asyncio.create_task(self._serve(frame))
      self._tasks.add(task)
      task.add_done_callback(self._tasks.discard)
      return
    if kind == "control":
      if self._on_control is not None:
        await self._on_control(str(frame.get("kind")), dict(frame.get("body") or {}))
      return
    pending = self._pending.get(int(frame.get("id", 0)))
    if pending is None:
      return
    if kind == "head":
      if not pending.head.done():
        pending.head.set_result((int(frame["status"]), _pairs(frame.get("headers"))))
    elif kind == "data":
      pending.chunks.put_nowait(base64.b64decode(frame["b64"]))
    elif kind == "end":
      pending.chunks.put_nowait(None)
    elif kind == "fail":
      error = TunnelClosed(str(frame.get("error") or "peer failed the request"))
      if not pending.head.done():
        pending.head.set_exception(error)
      pending.chunks.put_nowait(error)

  async def _serve(self, frame: dict[str, Any]) -> None:
    request_id = int(frame["id"])
    try:
      if self._handler is None:
        raise RuntimeError("this endpoint serves no requests")
      response = await self._handler(TunnelRequest(
        method=str(frame["method"]), path=str(frame["path"]),
        headers=_pairs(frame.get("headers")), body=base64.b64decode(frame.get("body") or ""),
      ))
      await self._send({
        "t": "head", "id": request_id, "status": response.status,
        "headers": [list(pair) for pair in response.headers],
      })
      async for chunk in response.body:
        if chunk:
          await self._send({"t": "data", "id": request_id, "b64": base64.b64encode(chunk).decode("ascii")})
      await self._send({"t": "end", "id": request_id})
    except TunnelClosed:
      return
    except Exception as exc:  # The peer must learn the request failed, not hang.
      try:
        await self._send({"t": "fail", "id": request_id, "error": type(exc).__name__})
      except TunnelClosed:
        return

  def close(self, reason: BaseException | None = None) -> None:
    """Fail every in-flight request; the socket is gone."""
    self._closed = reason or TunnelClosed("tunnel closed")
    error = self._closed if isinstance(self._closed, TunnelClosed) else TunnelClosed(str(self._closed))
    for pending in self._pending.values():
      if not pending.head.done():
        pending.head.set_exception(error)
      pending.chunks.put_nowait(error)
    for task in self._tasks:
      task.cancel()

  @property
  def closed(self) -> bool:
    return self._closed is not None
