from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from agent_gateway.artifact_paths import ArtifactPathError, _validate_ticker, user_workspace_root
from agent_gateway.control_plane.artifacts import ArtifactAuthDependency


class HtmlArtifactListItem(BaseModel):
  artifact_id: str
  title: str
  purpose: Any
  summary: str
  ticker: str | None
  session_id: str | None
  source_skill: str
  ts: str


def build_html_artifacts_router(
  *,
  artifact_auth_dependency: ArtifactAuthDependency,
) -> APIRouter:
  router = APIRouter(prefix="/html-artifacts")

  @router.get("", response_model=list[HtmlArtifactListItem])
  async def list_html_artifacts(
    request: Request,
    ticker: str | None = Query(default=None),
    purpose: str | None = Query(default=None),
    since: float | None = Query(default=None),
    limit: int = Query(default=50, ge=1),
  ) -> list[HtmlArtifactListItem]:
    user_id = artifact_auth_dependency(request)
    workspace_root = _workspace_root_for_user(user_id)
    try:
      artifacts = list_stored_html_artifacts(
        workspace_root,
        ticker=_filter_ticker(ticker),
        purpose=_filter_text(purpose),
        since=since,
        limit=min(limit, 200),
      )
    except ValueError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    return [
      HtmlArtifactListItem(
        artifact_id=artifact.artifact_id,
        title=artifact.title,
        purpose=artifact.purpose,
        summary=artifact.summary,
        ticker=artifact.ticker,
        session_id=artifact.session_id,
        source_skill=artifact.source_skill,
        ts=artifact.ts,
      )
      for artifact in artifacts
    ]

  @router.get("/{artifact_id}", response_model=None)
  async def get_html_artifact_sidecar(artifact_id: str, request: Request) -> Any:
    user_id = artifact_auth_dependency(request)
    workspace_root = _workspace_root_for_user(user_id)
    try:
      artifact = read_html_artifact_sidecar(workspace_root, artifact_id)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    if artifact is None:
      raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact

  @router.get("/{artifact_id}/content")
  async def get_html_artifact_content(artifact_id: str, request: Request) -> Response:
    user_id = artifact_auth_dependency(request)
    workspace_root = _workspace_root_for_user(user_id)
    try:
      content = read_html_artifact_content(workspace_root, artifact_id)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    if content is None:
      raise HTTPException(status_code=404, detail="Artifact not found")
    return Response(
      content=content,
      headers={
        "Content-Security-Policy": "sandbox; default-src 'none'; script-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'",
        "Content-Disposition": "inline",
        "Content-Type": "text/html; charset=utf-8",
        "X-Content-Type-Options": "nosniff",
      },
    )

  return router


def _html_artifacts_unavailable(exc: ModuleNotFoundError) -> HTTPException:
  if exc.name != "schema":
    raise exc
  return HTTPException(status_code=404, detail="HTML artifacts are unavailable")


def list_stored_html_artifacts(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.html_artifact_store import list_html_artifacts
  except ModuleNotFoundError as exc:
    raise _html_artifacts_unavailable(exc) from exc
  return list_html_artifacts(*args, **kwargs)


def read_html_artifact_content(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.html_artifact_store import read_html_artifact_content as read_content
  except ModuleNotFoundError as exc:
    raise _html_artifacts_unavailable(exc) from exc
  return read_content(*args, **kwargs)


def read_html_artifact_sidecar(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.html_artifact_store import read_html_artifact_sidecar as read_sidecar
  except ModuleNotFoundError as exc:
    raise _html_artifacts_unavailable(exc) from exc
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
  "HtmlArtifactListItem",
  "build_html_artifacts_router",
]
