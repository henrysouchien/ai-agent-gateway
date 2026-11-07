from __future__ import annotations

import pytest
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

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
