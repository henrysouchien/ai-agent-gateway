from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from agent_gateway.session import AuthManager, GatewaySession


def _require_bearer_session(request: Request, auth: AuthManager) -> GatewaySession:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  return auth.verify_token(token)


def _sse_data(event: dict[str, Any]) -> bytes:
  return f"data: {json.dumps(event, sort_keys=True, default=str)}\n\n".encode("utf-8")


async def _shielded_aclose(iterator: Any) -> None:
  close = getattr(iterator, "aclose", None)
  if not callable(close):
    return
  close_task = asyncio.create_task(close())
  await asyncio.shield(close_task)


def build_events_router(*, auth: AuthManager) -> APIRouter:
  router = APIRouter()

  @router.get("/events")
  async def control_events(
    request: Request,
    control_run_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
  ) -> StreamingResponse:
    authenticated = _require_bearer_session(request, auth)
    scoped_run_id = control_run_id or run_id
    if authenticated.kind == "chat":
      if scoped_run_id is not None and scoped_run_id != authenticated.session_id:
        raise HTTPException(status_code=401, detail="Chat session token cannot subscribe to another run")
      scoped_run_id = authenticated.session_id

    user_event_bus = getattr(request.app.state, "user_event_bus", None)
    if user_event_bus is None:
      raise HTTPException(status_code=503, detail="User event bus unavailable")

    subscription: AsyncIterator[dict[str, Any]] = user_event_bus.subscribe(
      authenticated.user_id,
      control_run_id=scoped_run_id,
    )

    async def event_generator() -> AsyncIterator[bytes]:
      async for event in subscription:
        try:
          yield _sse_data(dict(event))
        except Exception as exc:
          yield _sse_data({"type": "stream_error", "error": f"SSE serialization failed: {exc}"})
          return

    headers = {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "X-Accel-Buffering": "no",
      "Connection": "keep-alive",
    }
    return StreamingResponse(
      event_generator(),
      headers=headers,
      background=BackgroundTask(_shielded_aclose, subscription),
    )

  return router


__all__ = ["build_events_router"]
