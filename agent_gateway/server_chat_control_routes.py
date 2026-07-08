from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask


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
) -> StreamingResponse:
  token = get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)
  if "schema_version" in request.query_params:
    raise HTTPException(
      status_code=400,
      detail="schema_version is set at session init via POST /chat/init; create a new session to change versions",
    )
  requested_session_id = request.query_params.get("session_id") or session.session_id
  if requested_session_id != session.session_id:
    raise HTTPException(status_code=403, detail="Cannot subscribe to a different session")

  raw_after_seq = request.query_params.get("after_seq", "0")
  try:
    after_seq = max(int(raw_after_seq), 0)
  except (TypeError, ValueError):
    raise HTTPException(status_code=400, detail="after_seq must be an integer")

  active_turn = session.active_turn
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
) -> JSONResponse:
  token = get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)
  if body.session_id != session.session_id:
    raise HTTPException(status_code=403, detail="Cannot cancel a different session")

  active_turn = session.active_turn
  if active_turn is None:
    raise HTTPException(status_code=404, detail="No active turn for this session")

  await cancel_active_turn_runner(active_turn)
  clear_active_turn(session, active_turn)
  return JSONResponse(content={"status": "cancelled", "session_id": session.session_id})


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
  if active_turn.event_log.closed:
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
  if active_turn is not None and not active_turn.event_log.closed:
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


__all__ = [
  "chat_cancel_response",
  "chat_recap_response",
  "chat_subscribe_response",
]
