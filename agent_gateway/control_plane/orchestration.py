from __future__ import annotations

import os
from typing import Any, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..excel_dispatch import mint_and_submit


class ExcelDispatchRequest(BaseModel):
  text: Any = None
  workbook: Any = None
  force_compaction: Any = False
  window_seconds: Any = None
  args_predicate: Any = None
  timeout_s: Any = None


def _json_error(status_code: int, code: str, message: str, **details: Any) -> JSONResponse:
  payload: dict[str, Any] = {"code": code, "message": message}
  payload.update(details)
  return JSONResponse(payload, status_code=status_code)


def _dev_enabled() -> bool:
  return os.getenv("EXCEL_ORCHESTRATION_DEV", "").strip() == "1"


def _require_store(request: Request) -> Any | None:
  return getattr(request.app.state, "gateway_approval_store", None)


def _mint_error_status(error: dict[str, Any]) -> int:
  code = str(error.get("code") or "")
  if code == "invalid_input":
    return 400
  if code in {"no_excel_session", "ambiguous_target"}:
    return 409
  if code in {"relay_error", "internal_error"}:
    return 502
  return 500


def _profile_for_auth_context(auth_context: Any) -> str:
  profile = getattr(auth_context, "profile", None)
  if isinstance(profile, str) and profile.strip():
    return profile.strip()
  return "autonomous"


def _relay_status_error(status: str, response: dict[str, Any]) -> JSONResponse | None:
  if status in {"session_mismatch", "user_mismatch"}:
    return _json_error(
      409,
      "relay_owner_mismatch",
      "Bound Excel session cannot poll this relay request",
      status=status,
    )
  if status == "ok" or status == "not_found":
    return None
  return _json_error(
    502,
    "relay_error",
    f"Excel relay result returned status '{status}'",
    status=status,
    response=response,
  )


def _dispatch_state_response(response: dict[str, Any]) -> JSONResponse:
  state = response.get("state")
  if state == "done":
    payload: dict[str, Any] = {"state": "done"}
    if "result" in response:
      payload["result"] = response.get("result")
    return JSONResponse(payload)
  if state in {"failed", "timeout"}:
    error = response.get("error")
    if not isinstance(error, dict):
      error = {"message": f"Excel relay request {state}"}
    return JSONResponse({"state": state, "error": error})
  if state == "pending":
    return JSONResponse({"state": "pending"})
  return _json_error(
    502,
    "relay_error",
    f"Excel relay returned unknown result state '{state}'",
    response=response,
  )


def build_orchestration_router(
  *,
  relay: Any,
  authenticate: Callable[[Request], Any],
) -> APIRouter:
  router = APIRouter()

  @router.post("/api/orchestration/excel-dispatch")
  async def submit_excel_dispatch(request: Request, payload: ExcelDispatchRequest) -> JSONResponse:
    auth_context = authenticate(request)
    if not _dev_enabled():
      return _json_error(403, "orchestration_disabled", "Excel orchestration dispatch is disabled")
    store = _require_store(request)
    if store is None:
      return _json_error(503, "approval_store_unavailable", "Approval subsystem unavailable")

    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id=auth_context.user_id,
      gateway_session_id=auth_context.session_id,
      text=payload.text,
      workbook=payload.workbook,
      force_compaction=payload.force_compaction,
      window_seconds=payload.window_seconds,
      args_predicate=payload.args_predicate,
      delegator_profile=_profile_for_auth_context(auth_context),
      delegator_run_id=None,
    )
    if error is not None:
      return JSONResponse(error, status_code=_mint_error_status(error))
    assert submitted is not None
    return JSONResponse(
      {
        "request_id": submitted["request_id"],
        "delegation_id": submitted["delegation_id"],
      }
    )

  @router.get("/api/orchestration/excel-dispatch/{request_id}")
  async def get_excel_dispatch_result(
    request: Request,
    request_id: str,
    delegation_id: str | None = None,
  ) -> JSONResponse:
    auth_context = authenticate(request)
    if not _dev_enabled():
      return _json_error(403, "orchestration_disabled", "Excel orchestration dispatch is disabled")
    if not isinstance(delegation_id, str) or not delegation_id.strip():
      return _json_error(400, "invalid_input", "'delegation_id' query parameter is required")
    store = _require_store(request)
    if store is None:
      return _json_error(503, "approval_store_unavailable", "Approval subsystem unavailable")

    grant = await store.get_delegation_grant(delegation_id.strip())
    if grant is None or grant.bound_relay_request_id != request_id:
      return _json_error(404, "not_found", "Excel dispatch delegation not found")
    if grant.delegator_user_id != auth_context.user_id:
      return _json_error(403, "forbidden", "Excel dispatch delegation is not visible to this user")

    try:
      status, response = await relay.result(
        request_id,
        gateway_session_id=grant.bound_excel_session_id,
        user_id=auth_context.user_id,
      )
    except PermissionError as exc:
      return _json_error(
        409,
        "relay_owner_mismatch",
        str(exc) or "Bound Excel session cannot poll this relay request",
      )
    except Exception as exc:
      return _json_error(502, "relay_error", str(exc) or exc.__class__.__name__)

    if not isinstance(response, dict):
      return _json_error(
        502,
        "relay_error",
        "Excel relay returned an invalid result response",
        status=status,
      )
    relay_error = _relay_status_error(str(status), response)
    if relay_error is not None:
      return relay_error
    return _dispatch_state_response(response)

  return router


__all__ = ["ExcelDispatchRequest", "build_orchestration_router"]
