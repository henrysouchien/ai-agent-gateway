from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

from .approvals import ApprovalActionError
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

  if pending.get("status") == "received":
    return JSONResponse({"error": "Result already submitted"}, status_code=409)

  if pending.get("status") != "pending":
    return JSONResponse({"error": "Invalid pending tool state for result submission"}, status_code=409)

  if pending.get("nonce") != payload.nonce:
    return JSONResponse({"error": "Nonce mismatch"}, status_code=409)

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
  if not pending:
    return JSONResponse({"error": "Unknown tool_call_id"}, status_code=404)

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
      denied_by=payload.denied_by,
    )
  except ApprovalActionError as exc:
    return JSONResponse(exc.payload, status_code=exc.status_code)
  return JSONResponse(result)


__all__ = ["tool_approval_response", "tool_result_response"]
