from __future__ import annotations

import pytest
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message

from agent_gateway.control_plane.health import CONTROL_PLANE_VERSION
from agent_gateway.control_plane.middleware import (
  CONTROL_PLANE_VERSION_HEADER,
  add_control_plane_version_header_middleware,
)


def _middleware():
  app = FastAPI()
  add_control_plane_version_header_middleware(
    app, path_prefixes=("/api/control",),
  )
  registered = app.user_middleware[0]
  middleware = registered.cls(app, *registered.args, **registered.kwargs)
  assert isinstance(middleware, BaseHTTPMiddleware)
  return middleware.dispatch_func


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat/tool-approval", "/api/control/health"])
async def test_no_response_disconnect_returns_empty_499(path: str) -> None:
  async def receive() -> Message:
    return {"type": "http.disconnect"}

  request = Request(
    {"type": "http", "method": "POST", "path": path, "headers": []},
    receive=receive,
  )

  async def call_next(request: Request) -> Response:
    raise RuntimeError("No response returned.")

  response = await _middleware()(request, call_next)

  assert response.status_code == 499
  assert response.body == b""
  assert CONTROL_PLANE_VERSION_HEADER not in response.headers


@pytest.mark.asyncio
async def test_connected_no_response_error_propagates() -> None:
  async def receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}

  request = Request(
    {
      "type": "http", "method": "POST",
      "path": "/api/chat/tool-approval", "headers": [],
    },
    receive=receive,
  )
  error = RuntimeError("No response returned.")

  async def call_next(request: Request) -> Response:
    raise error

  with pytest.raises(RuntimeError) as caught:
    await _middleware()(request, call_next)

  assert caught.value is error


@pytest.mark.asyncio
async def test_normal_control_response_keeps_version_header() -> None:
  request = Request({
    "type": "http", "method": "GET", "path": "/api/control/health", "headers": [],
  })
  original = Response(content=b"healthy", status_code=200)

  async def call_next(request: Request) -> Response:
    return original

  response = await _middleware()(request, call_next)

  assert response is original
  assert response.status_code == 200
  assert response.body == b"healthy"
  assert response.headers[CONTROL_PLANE_VERSION_HEADER] == CONTROL_PLANE_VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
  RuntimeError("No response returned"),
  RuntimeError("upstream failed"),
  ValueError("No response returned."),
])
async def test_other_errors_propagate(error: Exception) -> None:
  request = Request({
    "type": "http", "method": "POST", "path": "/api/chat/tool-approval", "headers": [],
  })

  async def call_next(request: Request) -> Response:
    raise error

  with pytest.raises(type(error)) as caught:
    await _middleware()(request, call_next)

  assert caught.value is error
