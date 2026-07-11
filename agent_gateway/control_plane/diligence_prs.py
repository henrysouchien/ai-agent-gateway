from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_gateway.session import AuthManager

from .batches import _registry_for_user
from .runs import _require_bearer_session, _require_control_session


_ALLOWED_MERGE_FIELDS = frozenset({
  "confirm_merge",
  "expected_handoff_id",
  "expected_proposal_ids",
  "expected_research_file_id",
  "expected_ticker",
  "expected_workspace_id",
  "process_outbox",
})
_INVALID_INPUT_CODES = frozenset({
  "diligence_pr_merge_invalid_input",
  "invalid_tool_input",
  "proposal_series_invalid_input",
})


def build_diligence_prs_router(*, auth: AuthManager) -> APIRouter:
  router = APIRouter(prefix="/diligence-prs")

  @router.post("/{pr_id}/merge", response_model=None)
  async def merge_diligence_pr(
    request: Request,
    pr_id: str,
    payload: dict[str, Any] = Body(...),
  ) -> dict[str, Any] | JSONResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    if str(getattr(authenticated, "role", "") or "") != "owner":
      raise HTTPException(status_code=403, detail="Owner control session required to merge a diligence PR")

    merge_input = _validated_merge_input(pr_id, payload)
    handler = _merge_handler_for_session(authenticated)
    result, error = await handler(merge_input)
    if error is not None:
      return JSONResponse(error, status_code=_merge_error_status_code(error))
    if not isinstance(result, dict):
      raise HTTPException(status_code=500, detail="Diligence PR merge returned an invalid response")
    if str(result.get("status") or "").strip().lower() == "blocked":
      return JSONResponse(result, status_code=409)
    if str(result.get("status") or "").strip().lower() != "success":
      raise HTTPException(status_code=500, detail="Diligence PR merge returned an unknown status")
    return result

  return router


def _validated_merge_input(pr_id: str, payload: dict[str, Any]) -> dict[str, Any]:
  unexpected = sorted(set(payload) - _ALLOWED_MERGE_FIELDS)
  if unexpected:
    raise HTTPException(status_code=422, detail=f"unexpected merge fields: {', '.join(unexpected)}")
  if payload.get("confirm_merge") is not True:
    raise HTTPException(status_code=400, detail="confirm_merge=true is required to merge a diligence PR")

  expected_ticker = _required_text(payload, "expected_ticker").upper()
  expected_workspace_id = _required_text(payload, "expected_workspace_id")
  expected_proposal_ids = _required_proposal_ids(payload)
  process_outbox = payload.get("process_outbox", True)
  if not isinstance(process_outbox, bool):
    raise HTTPException(status_code=422, detail="process_outbox must be a boolean")

  merge_input: dict[str, Any] = {
    "pr_id": str(pr_id or "").strip(),
    "confirm_merge": True,
    "expected_ticker": expected_ticker,
    "expected_workspace_id": expected_workspace_id,
    "expected_proposal_ids": expected_proposal_ids,
    "process_outbox": process_outbox,
  }
  for field in ("expected_research_file_id", "expected_handoff_id"):
    value = _optional_positive_int(payload, field)
    if value is not None:
      merge_input[field] = value
  return merge_input


def _required_text(payload: dict[str, Any], field: str) -> str:
  value = payload.get(field)
  if not isinstance(value, str) or not value.strip():
    raise HTTPException(status_code=422, detail=f"{field} must be a non-empty string")
  return value.strip()


def _required_proposal_ids(payload: dict[str, Any]) -> list[str]:
  raw_ids = payload.get("expected_proposal_ids")
  if not isinstance(raw_ids, list) or not raw_ids:
    raise HTTPException(status_code=422, detail="expected_proposal_ids must be a non-empty array")
  proposal_ids: list[str] = []
  seen: set[str] = set()
  for index, raw_id in enumerate(raw_ids):
    if not isinstance(raw_id, str) or not raw_id.strip():
      raise HTTPException(
        status_code=422,
        detail=f"expected_proposal_ids[{index}] must be a non-empty string",
      )
    proposal_id = raw_id.strip()
    if proposal_id in seen:
      raise HTTPException(status_code=422, detail="expected_proposal_ids must not contain duplicates")
    seen.add(proposal_id)
    proposal_ids.append(proposal_id)
  return proposal_ids


def _optional_positive_int(payload: dict[str, Any], field: str) -> int | None:
  if field not in payload or payload.get(field) is None:
    return None
  value = payload.get(field)
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise HTTPException(status_code=422, detail=f"{field} must be a positive integer")
  return value


def _merge_handler_for_session(session: Any):
  try:
    from agent.shared.tool_handlers.apply_proposals import make_merge_diligence_pr_handler
  except ModuleNotFoundError as exc:
    if exc.name not in {
      "agent",
      "agent.shared",
      "agent.shared.tool_handlers",
      "agent.shared.tool_handlers.apply_proposals",
    }:
      raise
    from api.agent.shared.tool_handlers.apply_proposals import make_merge_diligence_pr_handler

  return make_merge_diligence_pr_handler(
    session=session,
    batch_registry_factory=_registry_for_user,
  )


def _merge_error_status_code(error: dict[str, Any]) -> int:
  code = str(error.get("code") or "").strip()
  if code == "merge_confirmation_required":
    return 400
  if code == "diligence_pr_not_found":
    return 404
  if code in _INVALID_INPUT_CODES:
    return 422
  if code == "merge_diligence_pr_unavailable":
    return 503
  if code == "ValueError":
    return 409
  return 500


__all__ = ["build_diligence_prs_router"]
