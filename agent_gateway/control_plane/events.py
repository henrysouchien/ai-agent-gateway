from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from agent_gateway.event_adapter import adapt_control_event
from agent_gateway.events import DEFAULT_SCHEMA_VERSION
from agent_gateway.session import AuthManager, GatewaySession

_PROJECTED_SCHEMA_VERSION_ALIASES = {"1", "v1"}
_PROJECTED_KEEPALIVE_SECONDS = 15.0
_PROJECTED_COALESCE_SECONDS = 0.5
_DELTA_EVENT_TYPES = {"text_delta", "thinking_delta"}


def _require_bearer_session(request: Request, auth: AuthManager) -> GatewaySession:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  return auth.verify_token(token)


def _sse_data(event: dict[str, Any]) -> bytes:
  return f"data: {json.dumps(event, sort_keys=True, default=str)}\n\n".encode("utf-8")


def _sse_keepalive() -> bytes:
  return b":keepalive\n\n"


def _parse_projected_schema_version(raw: str | None) -> int | None:
  raw = getattr(raw, "default", raw)
  if raw is None:
    return None
  normalized = str(raw).strip().lower()
  if normalized not in _PROJECTED_SCHEMA_VERSION_ALIASES:
    raise HTTPException(status_code=400, detail="schema_version must be v1")
  return DEFAULT_SCHEMA_VERSION


def _normalize_after_seq(raw: Any) -> int:
  value = getattr(raw, "default", raw)
  try:
    return max(int(value), 0)
  except (TypeError, ValueError):
    raise HTTPException(status_code=400, detail="after_seq must be an integer")


def _projected_envelope(entry: Any, event: dict[str, Any]) -> dict[str, Any]:
  run_id = getattr(entry, "control_run_id", None) or _event_run_id(event)
  return {
    "run_id": run_id,
    "seq": getattr(entry, "seq", None),
    "event": event,
  }


def _can_coalesce(left: dict[str, Any], right: dict[str, Any]) -> bool:
  left_event = left.get("event")
  right_event = right.get("event")
  if not isinstance(left_event, dict) or not isinstance(right_event, dict):
    return False
  left_type = left_event.get("type")
  return (
    left_type in _DELTA_EVENT_TYPES
    and right_event.get("type") == left_type
    and left.get("run_id") == right.get("run_id")
  )


def _merge_delta_envelope(left: dict[str, Any], right: dict[str, Any]) -> None:
  left_event = left["event"]
  right_event = right["event"]
  left_event["text"] = f"{left_event.get('text', '')}{right_event.get('text', '')}"
  left["seq"] = right.get("seq")


async def _projected_control_event_chunks(
  *,
  subscription: AsyncIterator[Any],
  auth: AuthManager,
  app_state: Any,
  authenticated: GatewaySession,
  schema_version: int,
) -> AsyncIterator[bytes]:
  pending: dict[str, Any] | None = None
  pending_started = 0.0
  next_keepalive_at = time.monotonic() + _PROJECTED_KEEPALIVE_SECONDS

  async def flush_pending() -> bytes | None:
    nonlocal pending
    if pending is None:
      return None
    envelope = pending
    pending = None
    return _sse_data(envelope)

  while True:
    now = time.monotonic()
    deadlines = [next_keepalive_at]
    if pending is not None:
      deadlines.append(pending_started + _PROJECTED_COALESCE_SECONDS)
    timeout = max(0.0, min(deadlines) - now)
    try:
      entry = await asyncio.wait_for(subscription.__anext__(), timeout=timeout)
    except asyncio.TimeoutError:
      now = time.monotonic()
      if pending is not None and now >= pending_started + _PROJECTED_COALESCE_SECONDS:
        chunk = await flush_pending()
        if chunk is not None:
          next_keepalive_at = time.monotonic() + _PROJECTED_KEEPALIVE_SECONDS
          yield chunk
        continue
      if now >= next_keepalive_at:
        next_keepalive_at = now + _PROJECTED_KEEPALIVE_SECONDS
        yield _sse_keepalive()
      continue
    except StopAsyncIteration:
      chunk = await flush_pending()
      if chunk is not None:
        yield chunk
      return

    raw_event = dict(getattr(entry, "event", {}))
    raw_event_type = raw_event.get("type")
    try:
      if not _event_visible_to_session(
        auth=auth,
        app_state=app_state,
        authenticated=authenticated,
        event=raw_event,
      ):
        continue
      projected_event = adapt_control_event(raw_event, schema_version)
    except ValueError as exc:
      projected_event = adapt_control_event({"type": "stream_error", "error": str(exc)}, DEFAULT_SCHEMA_VERSION)
    if projected_event is None:
      if raw_event_type not in _DELTA_EVENT_TYPES:
        chunk = await flush_pending()
        if chunk is not None:
          next_keepalive_at = time.monotonic() + _PROJECTED_KEEPALIVE_SECONDS
          yield chunk
      continue

    envelope = _projected_envelope(entry, projected_event)
    event_type = projected_event.get("type")
    if event_type in _DELTA_EVENT_TYPES:
      if pending is not None and _can_coalesce(pending, envelope):
        _merge_delta_envelope(pending, envelope)
      else:
        chunk = await flush_pending()
        if chunk is not None:
          next_keepalive_at = time.monotonic() + _PROJECTED_KEEPALIVE_SECONDS
          yield chunk
        pending = envelope
        pending_started = time.monotonic()
      continue

    chunk = await flush_pending()
    if chunk is not None:
      next_keepalive_at = time.monotonic() + _PROJECTED_KEEPALIVE_SECONDS
      yield chunk
    next_keepalive_at = time.monotonic() + _PROJECTED_KEEPALIVE_SECONDS
    yield _sse_data(envelope)


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


def _autonomous_record_for_run(app_state: Any, run_id: str) -> Any | None:
  registry = getattr(app_state, "subprocess_registry", None)
  tasks = getattr(registry, "_tasks", None)
  if not isinstance(tasks, dict):
    return None
  record = tasks.get(run_id)
  if record is not None:
    return record
  return next((task for task in tasks.values() if getattr(task, "control_run_id", None) == run_id), None)


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

  record = _autonomous_record_for_run(app_state, run_id)
  if record is None:
    return True
  return (
    getattr(record, "user_id", None) == authenticated.user_id
    and _channel_matches(getattr(record, "channel", None), authenticated.channel)
  )


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
  try:
    await asyncio.shield(close_task)
  except asyncio.CancelledError:
    close_task.cancel()
    await asyncio.gather(close_task, return_exceptions=True)
    raise


def build_events_router(*, auth: AuthManager) -> APIRouter:
  router = APIRouter()

  @router.get("/events")
  async def control_events(
    request: Request,
    control_run_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    schema_version: str | None = Query(default=None),
    after_seq: int = Query(default=0),
  ) -> StreamingResponse:
    authenticated = _require_bearer_session(request, auth)
    projected_schema_version = _parse_projected_schema_version(schema_version)
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

    if projected_schema_version is None:
      subscription: AsyncIterator[Any] = user_event_bus.subscribe(
        authenticated.user_id,
        control_run_id=scoped_run_id,
      )
    else:
      subscription = user_event_bus.subscribe_entries(
        authenticated.user_id,
        control_run_id=scoped_run_id,
        after_seq=_normalize_after_seq(after_seq),
      )

    async def event_generator() -> AsyncIterator[bytes]:
      if projected_schema_version is not None:
        async for chunk in _projected_control_event_chunks(
          subscription=subscription,
          auth=auth,
          app_state=request.app.state,
          authenticated=authenticated,
          schema_version=projected_schema_version,
        ):
          yield chunk
        return

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
