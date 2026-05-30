from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_gateway.approvals import ApprovalActionError, _approval_request_to_dict, _record_vote_and_unblock
from agent_gateway.session import AuthManager, GatewaySession


class ControlApprovalDecisionRequest(BaseModel):
  approved: bool
  allow_tool_type: bool = False
  reason: str | None = None


def _require_bearer_session(request: Request, auth: AuthManager) -> GatewaySession:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  return auth.verify_token(token)


def _json_error(status_code: int, message: str) -> JSONResponse:
  return JSONResponse({"error": message}, status_code=status_code)


def _target_chat_session_for_user(auth: AuthManager, run_id: str, user_id: str) -> GatewaySession | None:
  session = auth.session_store.get_session(run_id)
  if session is None or session.kind != "chat" or session.user_id != user_id:
    return None
  return session


def _find_pending_approval(session: GatewaySession, approval_id: str) -> tuple[str, dict[str, Any]] | None:
  for tool_call_id, pending in session.pending_tools.items():
    if str(pending.get("approval_id") or "") == approval_id:
      return tool_call_id, pending
  return None


async def _approval_records_for_session(store: Any, session: GatewaySession) -> list[dict[str, Any]]:
  approvals: list[dict[str, Any]] = []
  seen: set[str] = set()
  for pending in session.pending_tools.values():
    approval_id = pending.get("approval_id")
    if not approval_id:
      continue
    approval_id_text = str(approval_id)
    if approval_id_text in seen:
      continue
    seen.add(approval_id_text)
    request_record = await store.get(approval_id_text)
    if request_record is not None and request_record.state == "pending_user":
      approvals.append(_approval_request_to_dict(request_record))
  return approvals


def build_approvals_router(*, auth: AuthManager) -> APIRouter:
  router = APIRouter()

  @router.get("/approvals")
  async def list_approvals(request: Request) -> JSONResponse:
    authenticated = _require_bearer_session(request, auth)
    auth.session_store.cleanup_expired()
    store = getattr(request.app.state, "gateway_approval_store", None)
    if store is None:
      return _json_error(503, "Approval subsystem unavailable")

    approvals: list[dict[str, Any]] = []
    for session in auth.session_store.sessions.values():
      if session.kind != "chat" or session.user_id != authenticated.user_id:
        continue
      approvals.extend(await _approval_records_for_session(store, session))
    return JSONResponse({"approvals": approvals})

  @router.post("/runs/{run_id}/approvals/{approval_id}")
  async def resolve_approval(
    request: Request,
    run_id: str,
    approval_id: str,
    payload: ControlApprovalDecisionRequest,
  ) -> JSONResponse:
    authenticated = _require_bearer_session(request, auth)
    target_session = _target_chat_session_for_user(auth, run_id, authenticated.user_id)
    if target_session is None:
      return _json_error(404, "Run approval not found")

    pending = _find_pending_approval(target_session, approval_id)
    if pending is None:
      return _json_error(404, "Run approval not found")
    tool_call_id, pending_entry = pending

    try:
      result = await _record_vote_and_unblock(
        target_session=target_session,
        pending_entry=pending_entry,
        tool_call_id=tool_call_id,
        nonce=str(pending_entry.get("nonce") or ""),
        decider_id=authenticated.user_id,
        decider_role=getattr(authenticated, "role", None),
        approved=payload.approved,
        allow_tool_type=payload.allow_tool_type,
        reason=payload.reason,
        app_state=request.app.state,
      )
    except ApprovalActionError as exc:
      return JSONResponse(exc.payload, status_code=exc.status_code)
    return JSONResponse(result)

  return router


__all__ = [
  "ControlApprovalDecisionRequest",
  "build_approvals_router",
]
