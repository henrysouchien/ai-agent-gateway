from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from agent_gateway.event_log import EventLog
from agent_gateway.session import GatewaySession

from .runs_helpers import (
  ChatMessage,
  ChatRunResponse,
  _CONTROL_CHAT_TASK_PREFIX,
  _chat_run_from_session,
  _require_run_channel,
  _session_matches_owner,
  _session_owner_user_id,
  _session_has_cancel_event,
  _state_from_session,
)

def _require_control_session(session: GatewaySession) -> None:
  if session.kind != "control":
    raise HTTPException(status_code=401, detail="Control session required")


def _require_chat_session_for_run(authenticated: GatewaySession, target: GatewaySession) -> None:
  if authenticated.kind == "chat" and authenticated.session_id == target.session_id:
    return
  if authenticated.kind == "control" and _session_matches_owner(target, _session_owner_user_id(authenticated)):
    _require_run_channel(target.channel, authenticated.channel)
    return
  raise HTTPException(status_code=401, detail="Chat session token or matching control session required for this run")


def _transcript_dir_from_app_state(app_state: Any) -> Path | None:
  config = getattr(app_state, "gateway_config", None)
  raw = getattr(config, "transcript_dir", None)
  if raw is None:
    return None
  return Path(raw)


def _run_state_event(control_run_id: str, state: str) -> dict[str, Any]:
  return {
    "type": "run_state_changed",
    "run_id": control_run_id,
    "control_run_id": control_run_id,
    "state": state,
    "ts": int(time.time()),
  }


def _event_for_run(event: dict[str, Any], control_run_id: str) -> dict[str, Any]:
  event_copy = dict(event)
  event_copy.setdefault("run_id", control_run_id)
  event_copy.setdefault("control_run_id", control_run_id)
  return event_copy


def _latest_user_message_content(messages: list[ChatMessage]) -> str | None:
  for message in reversed(messages):
    if message.role.strip().lower() != "user":
      continue
    content = message.content.strip()
    if content:
      return content
  return None


def _control_message_id(request_id: str | None) -> str:
  normalized = request_id.strip() if isinstance(request_id, str) else ""
  return normalized or str(uuid.uuid4())


def _has_parent_message_event(session: GatewaySession, message_id: str) -> bool:
  return any(
    event.get("type") == "parent_message_sent"
    and (event.get("message_id") == message_id or event.get("request_id") == message_id)
    for event in session.event_history.snapshot()
  )


async def _maybe_call_on_event(callback: Any, event: dict[str, Any], control_run_id: str) -> None:
  if not callable(callback):
    return
  result = callback(event, control_run_id)
  if inspect.isawaitable(result):
    await result


async def _publish_control_event(app_state: Any, user_id: str, control_run_id: str, event: dict[str, Any]) -> None:
  event_copy = _event_for_run(event, control_run_id)
  user_event_bus = getattr(app_state, "user_event_bus", None)
  if user_event_bus is not None:
    await user_event_bus.publish(
      user_id=user_id,
      control_run_id=control_run_id,
      event=event_copy,
    )
  config = getattr(app_state, "gateway_config", None)
  await _maybe_call_on_event(getattr(config, "on_event", None), event_copy, control_run_id)


async def _cleanup_run_buffer(app_state: Any, user_id: str, control_run_id: str) -> None:
  user_event_bus = getattr(app_state, "user_event_bus", None)
  if user_event_bus is not None:
    await user_event_bus.cleanup_run(user_id, control_run_id)


async def _record_chat_parent_message_event(
  *,
  app_state: Any,
  session: GatewaySession,
  messages: list[ChatMessage],
  request_id: str | None,
) -> None:
  message = _latest_user_message_content(messages)
  if message is None:
    return

  control_run_id = session.session_id
  message_id = _control_message_id(request_id)
  if _has_parent_message_event(session, message_id):
    return

  sent_at = time.time()
  event = {
    "type": "parent_message_sent",
    "run_id": control_run_id,
    "control_run_id": control_run_id,
    "session_id": control_run_id,
    "message_id": message_id,
    "request_id": message_id,
    "message": message,
    "channel": session.channel or "web",
    "sender": {
      "session_id": control_run_id,
      "user_id": _session_owner_user_id(session),
    },
    "sent_at": sent_at,
    "ts": sent_at,
  }
  session.event_history.append(event)
  await _publish_control_event(app_state, _session_owner_user_id(session), control_run_id, event)


async def _cancel_control_chat_background_tasks(
  session: GatewaySession,
  *,
  settle_timeout: float = 0.0,
) -> None:
  task_items: list[tuple[str, asyncio.Future[Any]]] = []
  for task_key, task in list(session.control_chat_tasks.items()):
    if not str(task_key).startswith(_CONTROL_CHAT_TASK_PREFIX):
      continue
    task_items.append((task_key, task))

  if not task_items:
    return

  tasks = [task for _task_key, task in task_items]
  if settle_timeout > 0:
    await asyncio.wait(tasks, timeout=settle_timeout)

  for task_key, task in task_items:
    session.control_chat_tasks.pop(task_key, None)
    if not task.done():
      task.cancel()

  await asyncio.gather(*tasks, return_exceptions=True)


async def cleanup_control_chat_tasks(session: GatewaySession) -> None:
  await _cancel_control_chat_background_tasks(session)


async def _finalize_control_chat_task(
  *,
  task: asyncio.Task[Any],
  session: GatewaySession,
  app_state: Any,
  task_key: str,
) -> None:
  state = "failed"
  try:
    result = await task
    state = str(getattr(result, "state", "") or _state_from_session(session, session.event_history.snapshot()))
  except asyncio.CancelledError:
    state = "cancelled"
  except Exception:
    state = "failed"
  finally:
    session.control_chat_tasks.pop(task_key, None)
    if _session_has_cancel_event(session):
      await _cleanup_run_buffer(app_state, _session_owner_user_id(session), session.session_id)
      return
    if state not in {"approval_pending", "running", "starting"}:
      await _cleanup_run_buffer(app_state, _session_owner_user_id(session), session.session_id)


async def _dispatch_control_chat_turn(
  *,
  request: Request,
  session: GatewaySession,
  messages: list[ChatMessage],
  request_id: str | None,
  context: dict[str, Any],
  model: str | None,
  deadline_sec: int | None,
  record_parent_message: bool = False,
) -> ChatRunResponse:
  from agent_gateway.server import ChatMessage as ServerChatMessage
  from agent_gateway.server import ChatTurnInputs, _dispatch_chat_turn

  app_state = request.app.state
  control_run_id = session.session_id
  pending_seen = asyncio.Event()
  server_messages = [
    message if isinstance(message, ServerChatMessage) else ServerChatMessage(role=message.role, content=message.content)
    for message in messages
  ]
  if record_parent_message:
    if session.stream_active or (session.active_turn is not None and session.active_turn.is_running):
      raise HTTPException(status_code=409, detail="A turn is already running; subscribe via /chat/subscribe")
    await _record_chat_parent_message_event(
      app_state=app_state,
      session=session,
      messages=messages,
      request_id=request_id,
    )

  async def _on_event(event: dict[str, Any]) -> None:
    event_with_run = _event_for_run(event, control_run_id)
    await _publish_control_event(app_state, _session_owner_user_id(session), control_run_id, event_with_run)
    if event_with_run.get("type") == "tool_approval_request":
      await _publish_control_event(
        app_state,
        _session_owner_user_id(session),
        control_run_id,
        _run_state_event(control_run_id, "approval_pending"),
      )
      pending_seen.set()

  event_log = EventLog(session_id=control_run_id)
  dispatch_task = asyncio.create_task(
    _dispatch_chat_turn(
      session,
      ChatTurnInputs(
        messages=server_messages,
        request_id=request_id,
        context=dict(context),
        metadata=None,
        model=model,
      ),
      event_log=event_log,
      on_event=_on_event,
      build_chat_runtime=app_state.gateway_build_chat_runtime,
      credentials_resolver=getattr(getattr(app_state, "gateway_config", None), "credentials_refresh_resolver", None),
      transcript_dir=_transcript_dir_from_app_state(app_state),
      publish_lifecycle_events=True,
    )
  )
  task_key = f"control_chat_turn:{control_run_id}:{id(dispatch_task)}"
  session.control_chat_tasks[task_key] = dispatch_task

  pending_task = asyncio.create_task(pending_seen.wait())
  timeout = float(deadline_sec) if deadline_sec is not None else None
  done, _pending = await asyncio.wait(
    {dispatch_task, pending_task},
    timeout=timeout,
    return_when=asyncio.FIRST_COMPLETED,
  )

  if dispatch_task in done:
    session.control_chat_tasks.pop(task_key, None)
    pending_task.cancel()
    await asyncio.gather(pending_task, return_exceptions=True)
    result = await dispatch_task
    state = str(getattr(result, "state", "") or _state_from_session(session, session.event_history.snapshot()))
    if state not in {"approval_pending", "running", "starting"}:
      await _cleanup_run_buffer(app_state, _session_owner_user_id(session), control_run_id)
    return _chat_run_from_session(session)

  if pending_task in done:
    await asyncio.gather(pending_task, return_exceptions=True)
  else:
    pending_task.cancel()
    await asyncio.gather(pending_task, return_exceptions=True)

  asyncio.create_task(
    _finalize_control_chat_task(
      task=dispatch_task,
      session=session,
      app_state=app_state,
      task_key=task_key,
    )
  )
  return _chat_run_from_session(session)
