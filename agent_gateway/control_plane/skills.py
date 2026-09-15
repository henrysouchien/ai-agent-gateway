from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agent_gateway.control_skill_catalog import (
  ControlSkillCatalog,
  ControlSkillDetail,
  ControlSkillSummary,
  ControlSkillUnavailableError,
)
from agent_gateway.session import AuthManager


class SkillMetadataResponse(BaseModel):
  name: str
  label: str
  description: str
  agent_description: str | None
  version: str
  scope: str
  requires_portfolio_context: bool
  required_context: list[str]
  agent_callable: bool
  resumable: bool
  max_turns: int | None
  max_budget_usd: float | None
  persist_state: bool
  typed_contract: str | None
  catalog: bool
  profiles: list[str]
  modes: list[str]
  outputs: list[str]
  action_class: str
  approval_policy: str
  tier_availability: list[str]
  credential_requirements: list[str]
  schedule_eligible: bool
  can_launch: bool
  can_schedule: bool
  blocked_reason: str | None
  path: str


class SkillsListResponse(BaseModel):
  skills: list[SkillMetadataResponse]


class SkillDetailResponse(SkillMetadataResponse):
  body: str


def _response_from_summary(
  summary: ControlSkillSummary,
) -> SkillMetadataResponse:
  if type(summary) is not ControlSkillSummary:
    raise TypeError("control catalog returned an invalid summary")
  return SkillMetadataResponse(**asdict(summary))


def _require_bearer_session(request: Request, auth: AuthManager) -> None:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  auth.verify_token(token)


def build_skills_router(
  *,
  auth: AuthManager,
  catalog: ControlSkillCatalog,
) -> APIRouter:
  if not isinstance(catalog, ControlSkillCatalog):
    raise TypeError("catalog must implement ControlSkillCatalog")
  router = APIRouter(prefix="/skills")

  @router.get("", response_model=SkillsListResponse)
  async def list_skills(request: Request) -> SkillsListResponse:
    _require_bearer_session(request, auth)
    summaries = catalog.list_skills()
    if type(summaries) is not tuple:
      raise RuntimeError("control skill catalog returned an invalid list")
    return SkillsListResponse(
      skills=[_response_from_summary(summary) for summary in summaries]
    )

  @router.get("/{skill_name}", response_model=SkillDetailResponse)
  async def get_skill(request: Request, skill_name: str) -> SkillDetailResponse:
    _require_bearer_session(request, auth)
    try:
      detail = catalog.resolve_skill(skill_name)
    except ControlSkillUnavailableError as exc:
      raise HTTPException(status_code=404, detail="Skill not found") from exc
    if type(detail) is not ControlSkillDetail:
      raise RuntimeError("control skill catalog returned an invalid detail")
    return SkillDetailResponse(**asdict(detail))

  return router


__all__ = [
  "SkillDetailResponse",
  "SkillMetadataResponse",
  "SkillsListResponse",
  "build_skills_router",
]
