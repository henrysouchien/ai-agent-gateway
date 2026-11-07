from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypedDict

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask

from .chat_history import ChatHistoryQuery, read_chat_history
from .event_log import log_has_terminal
from .session import session_owner_user_id


class _BackgroundChildInterruptUnsupportedDetail(TypedDict):
  code: Literal["background_child_interrupt_unsupported"]
  message: str
  task_id: str


def _authorized_chat_session(
  session: Any,
  requested_session_id: str,
  *,
  get_session: Callable[[str], Any | None],
  refusal: str,
) -> Any:
  """Return the session a subscribe/cancel acts on.

  A chat bearer authorizes only the session it was minted for. A control
  bearer may also act on a live session of the same owner; any other
  requested session is refused.
  """
  if requested_session_id == session.session_id:
    return session
  if getattr(session, "kind", "chat") == "control":
    target = get_session(requested_session_id)
    owner = session_owner_user_id(session, required=False)
    if target is not None and owner and owner == session_owner_user_id(target, required=False):
      return target
  raise HTTPException(status_code=403, detail=refusal)


async def chat_history_response(
  request: Request,
  *,
  auth: Any,
  get_bearer_token: Callable[[str | None], str],
) -> JSONResponse:
  token = get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)
  try:
    query = ChatHistoryQuery.model_validate(dict(request.query_params))
  except ValidationError as exc:
    raise RequestValidationError(exc.errors()) from exc
  return JSONResponse(
    content=await read_chat_history(session, query),
    headers={"Cache-Control": "private, no-store"},
  )


async def chat_subscribe_response(
  request: Request,
  *,
  auth: Any,
  get_bearer_token: Callable[[str | None], str],
  register_stream_subscriber: Callable[..., Any],
  cleanup_stream_subscriber: Callable[[Any, str], Awaitable[None]],
  stream_subscriber_sse: Callable[..., Any],
  transcript_dir: Path | None,
  log: Any,
  get_session: Callable[[str], Any | None],
) -> StreamingResponse:
  token = get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)
  if "schema_version" in request.query_params:
    raise HTTPException(
      status_code=400,
      detail="schema_version is set at session init via POST /chat/init; create a new session to change versions",
    )
  requested_session_id = request.query_params.get("session_id") or session.session_id
  session = _authorized_chat_session(
    session,
    requested_session_id,
    get_session=get_session,
    refusal="Cannot subscribe to a different session",
  )

  raw_after_seq = request.query_params.get("after_seq", "0")
  try:
    after_seq = max(int(raw_after_seq), 0)
  except (TypeError, ValueError):
    raise HTTPException(status_code=400, detail="after_seq must be an integer")

  active_turn = session.active_turn
  if active_turn is None:
    # The turn finished and its grace clear ran before this reconnect; the
    # finished log replays from after_seq and ends at its terminal.
    active_turn = session.last_completed_turn
  if active_turn is None:
    raise HTTPException(status_code=404, detail="No active turn for this session")

  subscriber = register_stream_subscriber(
    active_turn,
    after_seq=after_seq,
    client_label=request.query_params.get("client_label") or "subscriber",
  )
  headers = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
  }

  async def cleanup() -> None:
    await cleanup_stream_subscriber(active_turn, subscriber.subscriber_id)

  async def event_generator():
    try:
      async for chunk in stream_subscriber_sse(
        session=session,
        active_turn=active_turn,
        subscriber=subscriber,
        transcript_dir=transcript_dir,
        channel=session.channel,
        write_transcript=True,
        log=log,
      ):
        yield chunk
    finally:
      await cleanup()

  return StreamingResponse(event_generator(), headers=headers, background=BackgroundTask(cleanup))


async def chat_cancel_response(
  request: Request,
  body: Any,
  *,
  auth: Any,
  get_bearer_token: Callable[[str | None], str],
  cancel_active_turn_runner: Callable[[Any], Awaitable[None]],
  clear_active_turn: Callable[[Any, Any], None],
  get_session: Callable[[str], Any | None],
) -> JSONResponse:
  token = get_bearer_token(request.headers.get("Authorization"))
  session = _authorized_chat_session(
    auth.verify_token(token),
    body.session_id,
    get_session=get_session,
    refusal="Cannot cancel a different session",
  )

  active_turn = session.active_turn
  if active_turn is None:
    raise HTTPException(status_code=404, detail="No active turn for this session")

  await cancel_active_turn_runner(active_turn)
  clear_active_turn(session, active_turn)
  return JSONResponse(content={"status": "cancelled", "session_id": session.session_id})


async def chat_interrupt_background_child_response(
  request: Request,
  task_id: str,
  *,
  auth: Any,
  get_bearer_token: Callable[[str | None], str],
) -> JSONResponse:
  token = get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)
  active_turn = session.active_turn
  # Live ownership comes only from this bearer's turn, never a journal lookup
  # or a newly constructed runtime. The registry owns target eligibility.
  if active_turn is None or active_turn.runtime_owner is None:
    raise HTTPException(status_code=409, detail="No live runner for this session")
  interrupt = getattr(active_turn.runtime_owner, "interrupt_background_task", None)
  if not callable(interrupt):
    detail = _BackgroundChildInterruptUnsupportedDetail(
      code="background_child_interrupt_unsupported",
      message="This runner does not support background child interruption",
      task_id=task_id,
    )
    raise HTTPException(status_code=409, detail=detail)
  requested = interrupt(task_id, reason="user_interrupted")
  return JSONResponse(content={
    "task_id": task_id,
    "interrupt_requested": requested,
  })


async def chat_recap_response(
  request: Request,
  body: Any,
  *,
  auth: Any,
  get_bearer_token: Callable[[str | None], str],
  transcript_dir: Path | None,
  compute_session_recap_payload: Callable[..., dict[str, Any]],
  compute_cumulative_session_recap_payload: Callable[..., dict[str, Any]],
  event_for_wire: Callable[[Any, Any], dict[str, Any]],
  write_transcript: Callable[..., None],
) -> JSONResponse:
  token = get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)
  # A chat bearer authorizes only the session it was minted for (auth).
  if body.session_id != session.session_id:
    raise HTTPException(status_code=403, detail="Cannot recap a different session")

  scope = str(body.scope or "active_turn").strip().lower()
  if scope != "active_turn":
    return _session_cumulative_recap_response(
      scope=scope,
      session=session,
      transcript_dir=transcript_dir,
      compute_cumulative_session_recap_payload=compute_cumulative_session_recap_payload,
      event_for_wire=event_for_wire,
      write_transcript=write_transcript,
    )

  active_turn = session.active_turn
  if active_turn is None:
    raise HTTPException(status_code=404, detail="No active turn for this session")

  recap_payload = compute_session_recap_payload(session, active_turn, trigger="explicit")
  if log_has_terminal(active_turn.event_log):



    write_transcript(
      transcript_dir,
      session.session_id,
      recap_payload,
      user_id=session.user_id,
      channel=session.channel,
    )
  else:
    appended = active_turn.event_log.append(recap_payload)
    if appended is not None:
      active_turn.transcript_written_seqs.add(int(appended.seq))
      write_transcript(
        transcript_dir,
        session.session_id,
        event_for_wire(appended, active_turn.event_log),
        user_id=session.user_id,
        channel=session.channel,
      )
    else:
      write_transcript(
        transcript_dir,
        session.session_id,
        recap_payload,
        user_id=session.user_id,
        channel=session.channel,
      )

  return JSONResponse(content=recap_payload)


def _session_cumulative_recap_response(
  *,
  scope: str,
  session: Any,
  transcript_dir: Path | None,
  compute_cumulative_session_recap_payload: Callable[..., dict[str, Any]],
  event_for_wire: Callable[[Any, Any], dict[str, Any]],
  write_transcript: Callable[..., None],
) -> JSONResponse:
  if scope != "session_cumulative":
    raise HTTPException(status_code=400, detail="scope must be active_turn or session_cumulative")
  active_turn = session.active_turn
  recap_payload = compute_cumulative_session_recap_payload(
    session,
    active_turn,
    transcript_dir,
    trigger="explicit",
  )
  if active_turn is not None:
    for entry in active_turn.event_log.entries:
      entry_seq = int(entry.seq)
      if entry_seq in active_turn.transcript_written_seqs:
        continue
      active_turn.transcript_written_seqs.add(entry_seq)
      write_transcript(
        transcript_dir,
        session.session_id,
        event_for_wire(entry, active_turn.event_log),
        user_id=session.user_id,
        channel=session.channel,
      )
  if active_turn is not None and not log_has_terminal(active_turn.event_log):
    appended = active_turn.event_log.append(recap_payload)
    if appended is not None:
      active_turn.transcript_written_seqs.add(int(appended.seq))
    write_transcript(
      transcript_dir,
      session.session_id,
      recap_payload,
      user_id=session.user_id,
      channel=session.channel,
    )
  else:
    write_transcript(
      transcript_dir,
      session.session_id,
      recap_payload,
      user_id=session.user_id,
      channel=session.channel,
    )
  return JSONResponse(content=recap_payload)



async def chat_todo_get_response(
  request: Request,
  *,
  auth: Any,
  get_bearer_token: Callable[[str | None], str],
) -> JSONResponse:
  """Current checklist snapshot for the session, with its revision."""
  token = get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)
  state = getattr(session, "todo_checklist_state", None)
  if state is None:
    from agent_gateway.todo_checklist import ensure_todo_session_state

    state = ensure_todo_session_state(
      session,
      gateway_session_id=getattr(session, "session_id", None) or "",
      scope_to_run=False,
    )
    log = getattr(session, "agent_session_log", None)
    if log is not None:
      await state.load(log)
  return JSONResponse(content=state.checklist.to_payload())


async def chat_todo_patch_response(
  request: Request,
  body: Any,
  *,
  auth: Any,
  get_bearer_token: Callable[[str | None], str],
  append_todo_snapshot: Callable[[Any, dict], Any],
  emit_to_session: Callable[[Any, dict], None],
) -> JSONResponse:
  """Revision-guarded analyst edit: replace the checklist's phases.

  A stale revision is rejected with the current snapshot so the client can
  rebase; the model's in-flight `init`/`done`-all is rejected the same way
  when an analyst edit moves first.
  """
  from agent_gateway.todo_checklist import TodoError

  token = get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)
  state = getattr(session, "todo_checklist_state", None)
  if state is None:
    from agent_gateway.todo_checklist import ensure_todo_session_state

    state = ensure_todo_session_state(
      session,
      gateway_session_id=getattr(session, "session_id", None) or "",
      scope_to_run=False,
    )
    log = getattr(session, "agent_session_log", None)
    if log is not None:
      await state.load(log)
  expected_revision = getattr(body, "expected_revision", None)
  if expected_revision is None or expected_revision != state.checklist.revision:
    raise HTTPException(
      status_code=409,
      detail={
        "code": "stale_revision",
        "message": f"checklist is at revision {state.checklist.revision}, PATCH was made against {expected_revision}",
        "checklist": state.checklist.to_payload(),
      },
    )
  try:
    updated = state.apply_analyst(
      expected_revision=expected_revision,
      phases=getattr(body, "phases", None),
    )
  except TodoError as exc:
    raise HTTPException(
      status_code=422,
      detail={"code": exc.code, "message": exc.message, **exc.fields},
    )
  # The model's in-flight init/done-all formed at an older revision is rejected
  # on its next call; the edit note (session's state.pending_analyst_note) reaches
  # it at its next request boundary.
  event = {
    "type": "todo_snapshot",
    "gateway_session_id": getattr(session, "gateway_session_id", None)
      or getattr(session, "session_id", None),
    "actor": "analyst",
    "op": "patch",
    "revision": updated.revision,
    "phases": [p.to_payload() for p in updated.phases],
    "analyst_revision": updated.analyst_revision,
    "analyst_change": updated.analyst_change,
  }
  if state.scope_to_run and not event["gateway_session_id"]:
    event["gateway_session_id"] = getattr(session, "session_id", None)
  await append_todo_snapshot(session, event)
  emit_to_session(session, event)
  return JSONResponse(content=updated.to_payload())


__all__ = [
  "chat_cancel_response",
  "chat_history_response",
  "chat_interrupt_background_child_response",
  "chat_recap_response",
  "chat_subscribe_response",
  "chat_todo_get_response",
  "chat_todo_patch_response",
]
