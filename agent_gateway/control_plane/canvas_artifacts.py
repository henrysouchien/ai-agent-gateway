from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Literal
import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_gateway.artifact_paths import ArtifactPathError, _validate_ticker, user_workspace_root
from agent_gateway.control_plane.artifacts import (
  ArtifactAuthDependency,
  _artifact_filters,
  _effective_artifact_fields,
  _payload_matches_artifact_filters,
)


log = logging.getLogger(__name__)

CANVAS_RENDER_FAILURE_REQUEST_MAX_BYTES = 32 * 1024
CANVAS_RENDER_FAILURE_STORED_CAP = 20
CANVAS_RENDER_FAILURE_REPORTS_PER_RENDER = 3
CANVAS_RENDER_FAILURE_MESSAGE_MAX_CHARS = 2_000
CANVAS_RENDER_FAILURE_COMPONENT_STACK_MAX_CHARS = 8_000
_LOWERCASE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CanvasArtifactListItem(BaseModel):
  artifact_id: str
  title: str
  purpose: Any
  summary: str
  ticker: str | None
  session_id: str | None
  source_skill: str
  ts: str


class CanvasRenderFailure(BaseModel):
  model_config = ConfigDict(extra="forbid")

  kind: Literal["canvas_render_error"]
  nonce: str = Field(..., min_length=1)
  artifact_id: str = Field(..., min_length=1)
  message: str = Field(..., max_length=CANVAS_RENDER_FAILURE_MESSAGE_MAX_CHARS)
  component_stack: str = Field(..., max_length=CANVAS_RENDER_FAILURE_COMPONENT_STACK_MAX_CHARS)


def build_canvas_artifacts_router(
  *,
  artifact_auth_dependency: ArtifactAuthDependency,
) -> APIRouter:
  router = APIRouter(prefix="/canvas-artifacts")

  @router.get("", response_model=list[CanvasArtifactListItem])
  async def list_canvas_artifacts(
    request: Request,
    ticker: str | None = Query(default=None),
    purpose: str | None = Query(default=None),
    since: float | None = Query(default=None),
    research_file_id: int | None = Query(default=None),
    control_run_id: str | None = Query(default=None),
    visibility: str = Query(default="default"),
    origin_kind: str = Query(default="all"),
    limit: int = Query(default=50, ge=1),
  ) -> list[CanvasArtifactListItem]:
    user_id = artifact_auth_dependency(request)
    workspace_root = _workspace_root_for_user(user_id)
    response_limit = min(limit, 200)
    filters = _artifact_filters(
      research_file_id=research_file_id,
      control_run_id=control_run_id,
      visibility=visibility,
      origin_kind=origin_kind,
    )
    try:
      artifacts = list_stored_canvas_artifacts(
        workspace_root,
        ticker=_filter_ticker(ticker),
        purpose=_filter_text(purpose),
        since=since,
        limit=max(response_limit, 500),
      )
    except ValueError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    items: list[CanvasArtifactListItem] = []
    for artifact in artifacts:
      if not _canvas_artifact_matches_filters(artifact, user_id=user_id, filters=filters):
        continue
      items.append(
        CanvasArtifactListItem(
          artifact_id=artifact.artifact_id,
          title=artifact.title,
          purpose=artifact.purpose,
          summary=artifact.summary,
          ticker=artifact.ticker,
          session_id=artifact.session_id,
          source_skill=artifact.source_skill,
          ts=artifact.ts,
        )
      )
      if len(items) >= response_limit:
        break
    return items

  @router.get("/{artifact_id}", response_model=None)
  async def get_canvas_artifact_sidecar(
    artifact_id: str,
    request: Request,
    research_file_id: int | None = Query(default=None),
    control_run_id: str | None = Query(default=None),
    visibility: str = Query(default="default"),
    origin_kind: str = Query(default="all"),
  ) -> Any:
    artifact = _authorized_artifact(
      artifact_id=artifact_id,
      request=request,
      artifact_auth_dependency=artifact_auth_dependency,
      research_file_id=research_file_id,
      control_run_id=control_run_id,
      visibility=visibility,
      origin_kind=origin_kind,
    )
    return artifact

  @router.get("/{artifact_id}/source")
  async def get_canvas_artifact_source(
    artifact_id: str,
    request: Request,
    research_file_id: int | None = Query(default=None),
    control_run_id: str | None = Query(default=None),
    visibility: str = Query(default="default"),
    origin_kind: str = Query(default="all"),
  ) -> Response:
    artifact, workspace_root = _authorized_artifact_and_workspace(
      artifact_id=artifact_id,
      request=request,
      artifact_auth_dependency=artifact_auth_dependency,
      research_file_id=research_file_id,
      control_run_id=control_run_id,
      visibility=visibility,
      origin_kind=origin_kind,
    )
    del artifact
    try:
      source = read_canvas_artifact_source(workspace_root, artifact_id)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    if source is None:
      raise HTTPException(status_code=404, detail="Artifact not found")
    return Response(
      content=source,
      headers={
        "Content-Disposition": "inline",
        "Content-Type": "text/plain; charset=utf-8",
        "X-Content-Type-Options": "nosniff",
      },
    )

  @router.get("/{artifact_id}/bundle/{bundle_digest}.js")
  async def get_canvas_artifact_bundle(
    artifact_id: str,
    bundle_digest: str,
    request: Request,
    research_file_id: int | None = Query(default=None),
    control_run_id: str | None = Query(default=None),
    visibility: str = Query(default="default"),
    origin_kind: str = Query(default="all"),
  ) -> Response:
    artifact, workspace_root = _authorized_artifact_and_workspace(
      artifact_id=artifact_id,
      request=request,
      artifact_auth_dependency=artifact_auth_dependency,
      research_file_id=research_file_id,
      control_run_id=control_run_id,
      visibility=visibility,
      origin_kind=origin_kind,
    )
    if not _LOWERCASE_SHA256_RE.fullmatch(bundle_digest) or artifact.bundle_digest != bundle_digest:
      raise HTTPException(status_code=404, detail="Artifact not found")
    try:
      bundle = read_canvas_artifact_bundle(workspace_root, artifact_id)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    if bundle is None or hashlib.sha256(bundle).hexdigest() != bundle_digest:
      raise HTTPException(status_code=404, detail="Artifact not found")
    return Response(
      content=bundle,
      headers={
        "Cache-Control": "public, max-age=31536000, immutable",
        "Content-Disposition": "inline",
        "Content-Type": "application/javascript",
        "X-Content-Type-Options": "nosniff",
      },
    )

  @router.post("/{artifact_id}/render-failures", status_code=204)
  async def post_canvas_render_failure(artifact_id: str, request: Request) -> Response:
    _artifact, workspace_root = _authorized_artifact_and_workspace(
      artifact_id=artifact_id,
      request=request,
      artifact_auth_dependency=artifact_auth_dependency,
    )
    raw = await _read_capped_request_body(request)
    try:
      report = CanvasRenderFailure.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
      # Renderers swallow this rejection client-side, so without a server-side record a
      # schema drift between the connectors type and this model silently discards EVERY
      # render failure and nobody notices. Log the offending fields (never the raw,
      # untrusted body) so the drift is observable here.
      if isinstance(exc, ValidationError):
        fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        reason = f"fields={fields}"
      else:
        reason = f"malformed_json ({type(exc).__name__})"
      log.warning(
        "canvas render failure report rejected: artifact_id=%s bytes=%d %s",
        artifact_id, len(raw), reason,
      )
      raise HTTPException(status_code=400, detail="Invalid render failure report") from exc
    if report.artifact_id != artifact_id:
      log.warning(
        "canvas render failure report rejected: artifact_id=%s body_artifact_id_mismatch",
        artifact_id,
      )
      raise HTTPException(status_code=400, detail="Invalid render failure report")
    try:
      await asyncio.to_thread(
        _append_render_failure,
        workspace_root,
        artifact_id,
        report.model_dump(mode="json"),
      )
    except (OSError, ValueError) as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    return Response(status_code=204)

  return router


async def _read_capped_request_body(request: Request) -> bytes:
  body = bytearray()
  async for chunk in request.stream():
    remaining = CANVAS_RENDER_FAILURE_REQUEST_MAX_BYTES + 1 - len(body)
    if remaining > 0:
      body.extend(chunk[:remaining])
    if len(body) > CANVAS_RENDER_FAILURE_REQUEST_MAX_BYTES:
      raise HTTPException(status_code=413, detail="Render failure report too large")
  return bytes(body)


def _append_render_failure(workspace_root: Path, artifact_id: str, report: dict[str, Any]) -> None:
  from agent_gateway.canvas_artifact_store import (
    _canvas_artifact_path,
    _canvas_artifacts_dir,
    _ensure_under_workspace,
  )

  directory = _canvas_artifacts_dir(workspace_root)
  report_path = _canvas_artifact_path(workspace_root, artifact_id, suffix=".render_failures.json")
  lock_path = _canvas_artifact_path(workspace_root, artifact_id, suffix=".render_failures.lock")
  open_flags = os.O_CREAT | os.O_RDWR
  if hasattr(os, "O_NOFOLLOW"):
    open_flags |= os.O_NOFOLLOW
  lock_fd = os.open(lock_path, open_flags, 0o600)
  try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    _ensure_under_workspace(report_path, workspace_root)
    if report_path.exists():
      current = json.loads(report_path.read_text(encoding="utf-8"))
      if not isinstance(current, list):
        raise ValueError("render failure log must be a JSON array")
    else:
      current = []
    report_nonce = report.get("nonce")
    reports_for_render = sum(
      1 for existing in current
      if isinstance(existing, dict) and existing.get("nonce") == report_nonce
    )
    if reports_for_render >= CANVAS_RENDER_FAILURE_REPORTS_PER_RENDER:
      return
    current.append(report)
    current = current[-CANVAS_RENDER_FAILURE_STORED_CAP:]
    tmp_path = _ensure_under_workspace(
      directory / f".{artifact_id}.render_failures.{os.getpid()}.{uuid.uuid4().hex}.tmp",
      workspace_root,
    )
    try:
      tmp_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
      if hasattr(os, "O_NOFOLLOW"):
        tmp_flags |= os.O_NOFOLLOW
      tmp_fd = os.open(tmp_path, tmp_flags, 0o600)
      with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
        json.dump(current, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
      os.replace(tmp_path, report_path)
      directory_fd = os.open(directory, os.O_RDONLY)
      try:
        os.fsync(directory_fd)
      finally:
        os.close(directory_fd)
    finally:
      try:
        tmp_path.unlink()
      except FileNotFoundError:
        pass
  finally:
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)


def _authorized_artifact(
  *,
  artifact_id: str,
  request: Request,
  artifact_auth_dependency: ArtifactAuthDependency,
  research_file_id: int | None = None,
  control_run_id: str | None = None,
  visibility: str = "default",
  origin_kind: str = "all",
) -> Any:
  artifact, _workspace = _authorized_artifact_and_workspace(
    artifact_id=artifact_id,
    request=request,
    artifact_auth_dependency=artifact_auth_dependency,
    research_file_id=research_file_id,
    control_run_id=control_run_id,
    visibility=visibility,
    origin_kind=origin_kind,
  )
  return artifact


def _authorized_artifact_and_workspace(
  *,
  artifact_id: str,
  request: Request,
  artifact_auth_dependency: ArtifactAuthDependency,
  research_file_id: int | None = None,
  control_run_id: str | None = None,
  visibility: str = "default",
  origin_kind: str = "all",
) -> tuple[Any, Path]:
  user_id = artifact_auth_dependency(request)
  workspace_root = _workspace_root_for_user(user_id)
  filters = _artifact_filters(
    research_file_id=research_file_id,
    control_run_id=control_run_id,
    visibility=visibility,
    origin_kind=origin_kind,
  )
  try:
    artifact = read_canvas_artifact_sidecar(workspace_root, artifact_id)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
  if artifact is None or not _canvas_artifact_matches_filters(artifact, user_id=user_id, filters=filters):
    raise HTTPException(status_code=404, detail="Artifact not found")
  return artifact, workspace_root


def _canvas_artifacts_unavailable(exc: ModuleNotFoundError) -> HTTPException:
  if exc.name != "schema":
    raise exc
  return HTTPException(status_code=404, detail="Canvas artifacts are unavailable")


def list_stored_canvas_artifacts(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.canvas_artifact_store import list_canvas_artifacts
  except ModuleNotFoundError as exc:
    raise _canvas_artifacts_unavailable(exc) from exc
  return list_canvas_artifacts(*args, **kwargs)


def read_canvas_artifact_bundle(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.canvas_artifact_store import read_canvas_artifact_bundle as read_bundle
  except ModuleNotFoundError as exc:
    raise _canvas_artifacts_unavailable(exc) from exc
  return read_bundle(*args, **kwargs)


def read_canvas_artifact_source(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.canvas_artifact_store import read_canvas_artifact_source as read_source
  except ModuleNotFoundError as exc:
    raise _canvas_artifacts_unavailable(exc) from exc
  return read_source(*args, **kwargs)


def read_canvas_artifact_sidecar(*args: Any, **kwargs: Any) -> Any:
  try:
    from agent_gateway.canvas_artifact_store import read_canvas_artifact_sidecar as read_sidecar
  except ModuleNotFoundError as exc:
    raise _canvas_artifacts_unavailable(exc) from exc
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


def _canvas_artifact_matches_filters(artifact: Any, *, user_id: str, filters: dict[str, Any]) -> bool:
  payload = artifact.model_dump(mode="json")
  payload.update(_effective_artifact_fields(payload, user_id=user_id))
  return _payload_matches_artifact_filters(payload, filters=filters)


__all__ = [
  "CANVAS_RENDER_FAILURE_COMPONENT_STACK_MAX_CHARS",
  "CANVAS_RENDER_FAILURE_MESSAGE_MAX_CHARS",
  "CANVAS_RENDER_FAILURE_REPORTS_PER_RENDER",
  "CANVAS_RENDER_FAILURE_REQUEST_MAX_BYTES",
  "CANVAS_RENDER_FAILURE_STORED_CAP",
  "CanvasArtifactListItem",
  "CanvasRenderFailure",
  "build_canvas_artifacts_router",
]
