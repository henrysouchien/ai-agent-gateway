from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agent_gateway.session import AuthManager

_SKILL_LOADER_MODULE_NAMES = frozenset({"agent", "agent.skills", "agent.skills.loader"})


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


def _loader_api() -> Any:
  try:
    from agent.skills import loader as skill_loader
  except ModuleNotFoundError as exc:
    if exc.name not in _SKILL_LOADER_MODULE_NAMES:
      raise
    from api.agent.skills import loader as skill_loader  # type: ignore
  return skill_loader


def _response_from_metadata(metadata: Any) -> SkillMetadataResponse:
  return SkillMetadataResponse(**asdict(metadata))


def _require_bearer_session(request: Request, auth: AuthManager) -> None:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  auth.verify_token(token)


def _list_skill_metadata(skills_dir: Path) -> list[Any]:
  entries: list[Any] = []
  if not skills_dir.exists():
    return entries

  skill_loader = _loader_api()
  for path in sorted(skills_dir.glob("*.md")):
    if not path.is_file():
      continue
    if not skill_loader._SKILL_NAME_RE.match(path.stem):
      continue
    metadata = skill_loader.load_skill_metadata(path.stem, skills_dir)
    if metadata is not None:
      entries.append(metadata)
  return sorted(entries, key=lambda entry: entry.name)


def build_skills_router(*, auth: AuthManager, skills_dir: Path) -> APIRouter:
  router = APIRouter(prefix="/skills")
  skills_root = Path(skills_dir)

  @router.get("", response_model=SkillsListResponse)
  async def list_skills(request: Request) -> SkillsListResponse:
    _require_bearer_session(request, auth)
    return SkillsListResponse(
      skills=[_response_from_metadata(metadata) for metadata in _list_skill_metadata(skills_root)]
    )

  @router.get("/{skill_name}", response_model=SkillDetailResponse)
  async def get_skill(request: Request, skill_name: str) -> SkillDetailResponse:
    _require_bearer_session(request, auth)
    skill_loader = _loader_api()
    try:
      metadata = skill_loader.load_skill_metadata(skill_name, skills_root)
    except ValueError as exc:
      raise HTTPException(status_code=404, detail="Skill not found") from exc
    if metadata is None:
      raise HTTPException(status_code=404, detail="Skill not found")

    try:
      _name, body, _version, _scope, _interactive = skill_loader.load_skill(skill_name, skills_root)
    except (FileNotFoundError, ValueError) as exc:
      raise HTTPException(status_code=404, detail="Skill not found") from exc

    return SkillDetailResponse(**asdict(metadata), body=body)

  return router


__all__ = [
  "SkillDetailResponse",
  "SkillMetadataResponse",
  "SkillsListResponse",
  "build_skills_router",
]
