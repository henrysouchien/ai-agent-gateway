from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from starlette.requests import ClientDisconnect

@pytest.fixture(autouse=True)
def _patch_asgi_transport_streaming(monkeypatch: pytest.MonkeyPatch):
  sentinel = object()

  class _StreamingResponseStream(httpx.AsyncByteStream):
    def __init__(
      self,
      body_queue: asyncio.Queue[Any],
      *,
      response_closed: asyncio.Event,
      app_task: asyncio.Task[Any],
    ) -> None:
      self._body_queue = body_queue
      self._response_closed = response_closed
      self._app_task = app_task
      self._closed = False

    async def __aiter__(self):
      try:
        while True:
          chunk = await self._body_queue.get()
          if chunk is sentinel:
            break
          yield chunk
      finally:
        await self.aclose()

    async def aclose(self) -> None:
      if self._closed:
        return
      self._closed = True
      self._response_closed.set()
      try:
        await self._app_task
      except (ClientDisconnect, OSError):
        pass

  async def _handle_async_request(self, request: httpx.Request) -> httpx.Response:
    assert isinstance(request.stream, httpx.AsyncByteStream)

    scope = {
      "type": "http",
      "asgi": {"version": "3.0", "spec_version": "2.3"},
      "http_version": "1.1",
      "method": request.method,
      "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
      "scheme": request.url.scheme,
      "path": request.url.path,
      "raw_path": request.url.raw_path.split(b"?")[0],
      "query_string": request.url.query,
      "server": (request.url.host, request.url.port),
      "client": self.client,
      "root_path": self.root_path,
    }

    request_body = request.stream.__aiter__()
    request_complete = False
    response_started = asyncio.Event()
    response_closed = asyncio.Event()
    body_queue: asyncio.Queue[Any] = asyncio.Queue()
    status_code: int | None = None
    response_headers: list[tuple[bytes, bytes]] | None = None

    async def receive() -> dict[str, Any]:
      nonlocal request_complete

      if request_complete:
        await response_closed.wait()
        return {"type": "http.disconnect"}

      try:
        body = await request_body.__anext__()
      except StopAsyncIteration:
        request_complete = True
        return {"type": "http.request", "body": b"", "more_body": False}
      return {"type": "http.request", "body": body, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
      nonlocal status_code, response_headers

      if message["type"] == "http.response.start":
        status_code = message["status"]
        response_headers = message.get("headers", [])
        response_started.set()
        return

      if message["type"] == "http.response.body":
        if response_closed.is_set():
          raise OSError("response stream closed")
        body = message.get("body", b"")
        more_body = message.get("more_body", False)
        if body and request.method != "HEAD":
          await body_queue.put(body)
        if not more_body:
          await body_queue.put(sentinel)

    async def _run_app() -> None:
      try:
        await self.app(scope, receive, send)
      finally:
        response_started.set()
        await body_queue.put(sentinel)

    app_task = asyncio.create_task(_run_app())
    await response_started.wait()
    if app_task.done():
      await app_task

    assert status_code is not None
    assert response_headers is not None

    return httpx.Response(
      status_code,
      headers=response_headers,
      stream=_StreamingResponseStream(
        body_queue,
        response_closed=response_closed,
        app_task=app_task,
      ),
    )

  monkeypatch.setattr(httpx.ASGITransport, "handle_async_request", _handle_async_request)
