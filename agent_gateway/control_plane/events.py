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


def _normalize_channel(channel: str | None) -> str | None:
  if not isinstance(channel, str):
    return None
  normalized = channel.strip().lower()
  return normalized or None


def _channel_matches(run_channel: str | None, authenticated_channel: str | None) -> bool:
  normalized_run_channel = _normalize_channel(run_channel)
  if normalized_run_channel is None:
    return True
  return _normalize_channel(authenticated_channel) == normalized_run_channel


def _event_run_id(event: dict[str, Any]) -> str | None:
  for key in ("control_run_id", "run_id", "session_id", "task_id"):
    value = event.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()
  return None


def _run_visible_to_session(
  *,
  auth: AuthManager,
  app_state: Any,
  authenticated: GatewaySession,
  run_id: str,
) -> bool:
  session = auth.session_store.get_session(run_id)
  if session is not None:
    return (
      session.kind == "chat"
      and session.user_id == authenticated.user_id
      and _channel_matches(session.channel, authenticated.channel)
    )

  _ = app_state
  return True


def _event_visible_to_session(
  *,
  auth: AuthManager,
  app_state: Any,
  authenticated: GatewaySession,
  event: dict[str, Any],
) -> bool:
  if authenticated.kind == "chat":
    return _event_run_id(event) == authenticated.session_id
  run_id = _event_run_id(event)
  if run_id is None:
    return True
  return _run_visible_to_session(
    auth=auth,
    app_state=app_state,
    authenticated=authenticated,
    run_id=run_id,
  )


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
    elif scoped_run_id is not None and not _run_visible_to_session(
      auth=auth,
      app_state=request.app.state,
      authenticated=authenticated,
      run_id=scoped_run_id,
    ):
      raise HTTPException(status_code=404, detail="Run not found")

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
          event_dict = dict(event)
          if not _event_visible_to_session(
            auth=auth,
            app_state=request.app.state,
            authenticated=authenticated,
            event=event_dict,
          ):
            continue
          yield _sse_data(event_dict)
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
