from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from agent_gateway.artifact_paths import ArtifactPathError, _validate_ticker, user_workspace_root
from agent_gateway.control_plane.artifacts import ArtifactAuthDependency


class DashboardArtifactListItem(BaseModel):
  artifact_id: str
  title: str
  summary: str
  ticker: str | None
  scope_label: str | None
  source_skill: str
  readiness_posture: Any
  profile: str
  ts: str


def build_dashboard_artifacts_router(
  *,
  artifact_auth_dependency: ArtifactAuthDependency,
) -> APIRouter:
  router = APIRouter(prefix="/dashboard-artifacts")

  @router.get("", response_model=list[DashboardArtifactListItem])
  async def list_dashboard_artifacts(
    request: Request,
    ticker: str | None = Query(default=None),
    since: float | None = Query(default=None),
    limit: int = Query(default=50, ge=1),
  ) -> list[DashboardArtifactListItem]:
    user_id = artifact_auth_dependency(request)
    workspace_root = _workspace_root_for_user(user_id)
    try:
      artifacts = list_stored_dashboard_artifacts(
        workspace_root,
        ticker=_filter_ticker(ticker),
        since=since,
        limit=min(limit, 200),
      )
    except ValueError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    return [
      DashboardArtifactListItem(
        artifact_id=artifact.artifact_id,
        title=artifact.title,
        summary=artifact.summary,
        ticker=artifact.ticker,
        scope_label=artifact.scope_label,
        source_skill=artifact.source_skill,
        readiness_posture=artifact.readiness_posture,
        profile=artifact.profile,
        ts=artifact.ts,
      )
      for artifact in artifacts
    ]

  @router.get("/{artifact_id}", response_model=None)
  async def get_dashboard_artifact_sidecar(artifact_id: str, request: Request) -> Any:
    user_id = artifact_auth_dependency(request)
    workspace_root = _workspace_root_for_user(user_id)
    try:
      artifact = read_dashboard_artifact_sidecar(workspace_root, artifact_id)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    if artifact is None:
      raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact

  @router.get("/{artifact_id}/payload")
  async def get_dashboard_artifact_payload(artifact_id: str, request: Request) -> Response:
    user_id = artifact_auth_dependency(request)
    workspace_root = _workspace_root_for_user(user_id)
    try:
      payload = read_dashboard_artifact_payload(workspace_root, artifact_id)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    if payload is None:
      raise HTTPException(status_code=404, detail="Artifact not found")
    return Response(
      content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
      media_type="application/json",
      headers={"X-Content-Type-Options": "nosniff"},
    )

  return router


def _dashboard_artifacts_unavailable(exc: ModuleNotFoundError) -> HTTPException:
  if exc.name != "schema":
    raise exc
  return HTTPException(status_code=404, detail="Dashboard artifacts are unavailable")


def list_stored_dashboard_artifacts(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.dashboard_artifact_store import list_dashboard_artifacts
  except ModuleNotFoundError as exc:
    raise _dashboard_artifacts_unavailable(exc) from exc
  return list_dashboard_artifacts(*args, **kwargs)


def read_dashboard_artifact_payload(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.dashboard_artifact_store import read_dashboard_artifact_payload as read_payload
  except ModuleNotFoundError as exc:
    raise _dashboard_artifacts_unavailable(exc) from exc
  return read_payload(*args, **kwargs)


def read_dashboard_artifact_sidecar(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.dashboard_artifact_store import read_dashboard_artifact_sidecar as read_sidecar
  except ModuleNotFoundError as exc:
    raise _dashboard_artifacts_unavailable(exc) from exc
  return read_sidecar(*args, **kwargs)


def _workspace_root_for_user(user_id: str) -> Path:
  try:
    return user_workspace_root(user_id)
  except ArtifactPathError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc


def _filter_ticker(value: str | None) -> str | None:
  value = _filter_text(value)
  if value is None:
    return None
  try:
    return _validate_ticker(value)
  except ArtifactPathError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc


def _filter_text(value: str | None) -> str | None:
  if value is None:
    return None
  stripped = value.strip()
  return stripped or None


__all__ = [
  "DashboardArtifactListItem",
  "build_dashboard_artifacts_router",
]
