from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

from .approvals import (
  ApprovalActionError,
  _record_durable_vote_and_continue,
  _resolve_store_and_policy,
)
from .session import AuthManager, session_owner_user_id


async def tool_result_response(
  request: Request,
  payload: Any,
  *,
  auth: AuthManager,
  log: Any,
  time_time: Callable[[], float] | None = None,
) -> JSONResponse:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)

  pending = session.pending_tools.get(payload.tool_call_id)
  if not pending:
    return JSONResponse({"error": "Unknown tool_call_id"}, status_code=404)

  # One result per tool call (concurrency): the waiting turn reads exactly one
  # result for this call, so a second submission has no reader.
  if pending.get("status") == "received":
    return JSONResponse({"error": "Result already submitted"}, status_code=409)

  # Concurrency: only a call the turn is waiting on as a client tool takes a
  # result; an entry in any other state (an approval awaiting its vote) has no
  # result reader.
  if pending.get("status") != "pending":
    return JSONResponse({"error": "Invalid pending tool state for result submission"}, status_code=409)

  # The nonce binds the result to the issued call (replay): a result carrying
  # another call's nonce was not produced for this one.
  if pending.get("nonce") != payload.nonce:
    return JSONResponse({"error": "Nonce mismatch"}, status_code=409)

  # The pending call was already released at its deadline, so a late result has
  # no waiting turn to deliver to.
  current_time = time.time if time_time is None else time_time
  if int(current_time()) > int(pending.get("expires_at", 0)):
    session.pending_tools.pop(payload.tool_call_id, None)
    return JSONResponse({"error": "Tool call expired"}, status_code=410)

  pending["status"] = "received"
  if session.result_queue is None:
    session.result_queue = asyncio.Queue()
  await session.result_queue.put({"result": payload.result, "error": payload.error})

  tool_name = pending.get("tool_name", "?")
  if payload.error:
    has_error_code = (
      isinstance(payload.error, dict)
      and isinstance(payload.error.get("code"), str)
      and bool(payload.error.get("code"))
    )
    log.warning(
      "Tool result: %s | error=true | has_code=%s",
      tool_name,
      has_error_code,
    )
  else:
    log.info("Tool result: %s | success", tool_name)
  return JSONResponse({"status": "ok"})


async def tool_approval_response(
  request: Request,
  payload: Any,
  *,
  auth: AuthManager,
  record_vote_and_unblock: Callable[..., Any],
) -> JSONResponse:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)

  pending = session.pending_tools.get(payload.tool_call_id)
  # A live pending entry keeps its own validation (duplicate, nonce, queue);
  # only a call whose live entry is gone resolves through the durable request.
  if pending is None:
    store, policy = _resolve_store_and_policy(
      target_session=session,
      app_state=request.app.state,
    )
    if store is None or policy is None:
      return JSONResponse({"error": "Approval subsystem unavailable"}, status_code=503)
    tenant_storage_id = session_owner_user_id(session)
    located = await store._locate_request_for_decision(
      tenant_storage_id=tenant_storage_id,
      tool_call_id=payload.tool_call_id,
      nonce=payload.nonce,
    )
    if located is None:
      return JSONResponse({"error": "Unknown tool_call_id"}, status_code=404)
    request_record, locator, original_nonce = located
    if request_record.channel and (
      str(request_record.channel).strip().lower()
      != str(session.channel or "").strip().lower()
    ):
      return JSONResponse({"error": "Unknown tool_call_id"}, status_code=404)
    try:
      result = await _record_durable_vote_and_continue(
        authenticated=session,
        tenant_storage_id=tenant_storage_id,
        request_record=request_record,
        locator=locator,
        nonce=original_nonce,
        store=store,
        policy=policy,
        decider_id=session_owner_user_id(session),
        decider_role=getattr(session, "role", None),
        approved=payload.approved,
        allow_tool_type=payload.allow_tool_type,
        reason=None,
        app_state=request.app.state,
      )
    except ApprovalActionError as exc:
      return JSONResponse(exc.payload, status_code=exc.status_code)
    return JSONResponse(result)

  try:
    result = await record_vote_and_unblock(
      target_session=session,
      pending_entry=pending,
      tool_call_id=payload.tool_call_id,
      nonce=payload.nonce,
      decider_id=session_owner_user_id(session),
      decider_role=getattr(session, "role", None),
      approved=payload.approved,
      allow_tool_type=payload.allow_tool_type,
      reason=None,
      app_state=request.app.state,
    )
  except ApprovalActionError as exc:
    return JSONResponse(exc.payload, status_code=exc.status_code)
  return JSONResponse(result)


__all__ = ["tool_approval_response", "tool_result_response"]
