from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from agent_gateway.artifact_paths import (
  ArtifactPathError,
  _validate_artifact_id,
  _validate_skill,
  _validate_ticker,
  user_workspace_root,
)


ArtifactAuthDependency = Callable[[Request], str]


class ArtifactResponse(BaseModel):
  ticker: str
  skill: str
  artifact_id: str
  artifact_path: str
  binary_artifact_path: str | None
  contract_name: str
  data_source: str
  created_at: str
  skill_run_id: str


class ArtifactsListResponse(BaseModel):
  artifacts: list[ArtifactResponse]


def _filter_ticker(value: str | None) -> str | None:
  if value is None:
    return None
  stripped = value.strip()
  if not stripped:
    return None
  try:
    return _validate_ticker(stripped)
  except ArtifactPathError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc


def _filter_skill(value: str | None) -> str | None:
  if value is None:
    return None
  stripped = value.strip()
  if not stripped:
    return None
  try:
    return _validate_skill(stripped)
  except ArtifactPathError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc


def _relative_sidecar_path(workspace_root: Path, path: Path) -> str:
  try:
    return path.relative_to(workspace_root).as_posix()
  except ValueError:
    return path.as_posix()


def _string_or_none(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value).strip()
  return text or None


def _skill_run_id_from_artifact_id(artifact_id: str) -> str:
  parts = artifact_id.split("-", 3)
  if len(parts) == 4 and parts[3]:
    return parts[3]
  return artifact_id


def _created_at_from_mtime(path: Path) -> str:
  return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact_response_from_sidecar(
  *,
  path: Path,
  workspace_root: Path,
  ticker: str,
  skill: str,
  artifact_id: str,
) -> ArtifactResponse | None:
  try:
    with path.open("r", encoding="utf-8") as handle:
      payload = json.load(handle)
  except (OSError, json.JSONDecodeError):
    return None
  if not isinstance(payload, dict):
    return None

  resolved_artifact_id = _string_or_none(payload.get("artifact_id")) or artifact_id
  artifact_path = _string_or_none(payload.get("artifact_path")) or _relative_sidecar_path(workspace_root, path)
  created_at = _string_or_none(payload.get("created_at")) or _created_at_from_mtime(path)
  skill_run_id = _string_or_none(payload.get("skill_run_id")) or _skill_run_id_from_artifact_id(resolved_artifact_id)

  return ArtifactResponse(
    ticker=_string_or_none(payload.get("ticker")) or ticker,
    skill=_string_or_none(payload.get("skill")) or skill,
    artifact_id=resolved_artifact_id,
    artifact_path=artifact_path,
    binary_artifact_path=_string_or_none(payload.get("binary_artifact_path")),
    contract_name=_string_or_none(payload.get("contract_name")) or skill,
    data_source=_string_or_none(payload.get("data_source")) or "live",
    created_at=created_at,
    skill_run_id=skill_run_id,
  )


def _artifact_sidecars_for_user(
  user_id: str,
  *,
  ticker_filter: str | None,
  skill_filter: str | None,
) -> list[tuple[int, Path, str, str, str]]:
  try:
    workspace_root = user_workspace_root(user_id)
  except ArtifactPathError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc

  artifacts_root = workspace_root / "artifacts"
  if not artifacts_root.is_dir():
    return []

  resolved_workspace = workspace_root.resolve()
  entries: list[tuple[int, Path, str, str, str]] = []
  for path in artifacts_root.rglob("*.json"):
    try:
      path.resolve().relative_to(resolved_workspace)
      relative = path.relative_to(artifacts_root)
    except ValueError:
      continue

    parts = relative.parts
    if len(parts) != 3:
      continue
    ticker_component, skill_component, filename = parts
    try:
      normalized_ticker = _validate_ticker(ticker_component)
      normalized_skill = _validate_skill(skill_component)
      artifact_id = _validate_artifact_id(Path(filename).stem)
    except ArtifactPathError:
      continue

    if ticker_filter is not None and normalized_ticker != ticker_filter:
      continue
    if skill_filter is not None and normalized_skill != skill_filter:
      continue

    try:
      mtime_ns = path.stat().st_mtime_ns
    except OSError:
      continue
    entries.append((mtime_ns, path, normalized_ticker, normalized_skill, artifact_id))

  return sorted(entries, key=lambda entry: (entry[0], entry[1].as_posix()), reverse=True)


def build_artifacts_router(*, artifact_auth_dependency: ArtifactAuthDependency) -> APIRouter:
  router = APIRouter(prefix="/artifacts")

  @router.get("", response_model=ArtifactsListResponse)
  async def list_artifacts(
    request: Request,
    ticker: str | None = Query(default=None),
    skill: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1),
  ) -> ArtifactsListResponse:
    user_id = artifact_auth_dependency(request)
    ticker_filter = _filter_ticker(ticker)
    skill_filter = _filter_skill(skill)
    effective_limit = min(limit, 50)

    try:
      workspace_root = user_workspace_root(user_id)
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc

    artifacts: list[ArtifactResponse] = []
    for _mtime_ns, path, path_ticker, path_skill, artifact_id in _artifact_sidecars_for_user(
      user_id,
      ticker_filter=ticker_filter,
      skill_filter=skill_filter,
    ):
      artifact = _artifact_response_from_sidecar(
        path=path,
        workspace_root=workspace_root,
        ticker=path_ticker,
        skill=path_skill,
        artifact_id=artifact_id,
      )
      if artifact is None:
        continue
      artifacts.append(artifact)
      if len(artifacts) >= effective_limit:
        break

    return ArtifactsListResponse(artifacts=artifacts)

  return router


__all__ = [
  "ArtifactResponse",
  "ArtifactsListResponse",
  "build_artifacts_router",
]
