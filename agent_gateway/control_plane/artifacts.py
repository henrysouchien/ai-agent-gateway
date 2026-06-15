from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_gateway.artifact_paths import (
  ArtifactPath,
  ArtifactPathError,
  _validate_artifact_id,
  _validate_skill,
  _validate_ticker,
  artifact_json_path_for_request,
  latest_artifact_json_path_for_request,
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
  run_id: str | None = None


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


def _list_stored_html_artifacts(*args: Any, **kwargs: Any) -> Any:
  from agent_gateway.html_artifact_store import list_html_artifacts

  return list_html_artifacts(*args, **kwargs)


def _list_stored_dashboard_artifacts(*args: Any, **kwargs: Any) -> Any:
  from agent_gateway.dashboard_artifact_store import list_dashboard_artifacts

  return list_dashboard_artifacts(*args, **kwargs)


def _skill_run_id_from_artifact_id(artifact_id: str) -> str:
  parts = artifact_id.split("-", 3)
  if len(parts) == 4 and parts[3]:
    return parts[3]
  return artifact_id


def _created_at_from_mtime(path: Path) -> str:
  return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_from_iso(value: str | None) -> float | None:
  if value is None:
    return None
  text = value.strip()
  if not text:
    return None
  if text.endswith("Z"):
    text = f"{text[:-1]}+00:00"
  try:
    return datetime.fromisoformat(text).timestamp()
  except ValueError:
    return None


def _artifact_event_keys(event: dict[str, Any]) -> set[str]:
  keys: set[str] = set()
  for field in ("artifact_id", "artifact_path", "binary_artifact_path", "path", "url"):
    value = _string_or_none(event.get(field))
    if value is not None:
      keys.add(value)
  return keys


def _artifact_run_index_from_registry(registry: Any, user_id: str) -> dict[str, dict[str, str]]:
  tasks = getattr(registry, "_tasks", {})
  if not isinstance(tasks, dict):
    return {}

  index: dict[str, dict[str, str]] = {}
  for record in tasks.values():
    if _string_or_none(getattr(record, "user_id", None)) != str(user_id):
      continue
    run_id = _string_or_none(getattr(record, "control_run_id", None))
    if run_id is None:
      continue
    events = getattr(record, "event_lines", None)
    if not isinstance(events, list):
      continue
    for event in events:
      if not isinstance(event, dict):
        continue
      if _string_or_none(event.get("type")) not in {"artifact_ready", "artifact_created", "artifact_updated"}:
        continue
      event_keys = _artifact_event_keys(event)
      if not event_keys:
        continue
      metadata = {"run_id": run_id}
      skill_run_id = _string_or_none(event.get("skill_run_id"))
      if skill_run_id is not None:
        metadata["skill_run_id"] = skill_run_id
      for key in event_keys:
        index[key] = metadata
  return index


def _artifact_run_metadata(
  run_index: dict[str, dict[str, str]],
  *,
  artifact_id: str,
  artifact_path: str,
  binary_artifact_path: str | None,
) -> dict[str, str]:
  for key in (artifact_id, artifact_path, binary_artifact_path):
    if key and key in run_index:
      return run_index[key]
  return {}


def _artifact_run_windows_from_registry(registry: Any, user_id: str) -> dict[str, float]:
  tasks = getattr(registry, "_tasks", {})
  if not isinstance(tasks, dict):
    return {}

  windows: dict[str, float] = {}
  for record in tasks.values():
    if _string_or_none(getattr(record, "user_id", None)) != str(user_id):
      continue
    run_id = _string_or_none(getattr(record, "control_run_id", None))
    if run_id is None:
      continue
    try:
      started_at = float(getattr(record, "started_at", 0) or 0)
    except (TypeError, ValueError):
      started_at = 0.0
    if started_at > 0:
      windows[run_id] = started_at
  return windows


def _artifact_matches_run_window(
  artifact: ArtifactResponse,
  *,
  run_id: str,
  run_windows: dict[str, float],
) -> bool:
  started_at = run_windows.get(run_id)
  if started_at is None:
    return True
  artifact_created_at = _timestamp_from_iso(artifact.created_at)
  if artifact_created_at is None:
    return True
  return artifact_created_at >= started_at - 1.0


def _assert_artifact_path_still_safe(artifact: ArtifactPath) -> Path:
  try:
    resolved = artifact.path.resolve()
    resolved.relative_to(artifact.workspace_root.resolve())
  except ValueError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
  return resolved


def _file_cache_headers(path: Path) -> dict[str, str]:
  stat = path.stat()
  return {
    "Cache-Control": "private, max-age=0",
    "ETag": f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"',
  }


def _artifact_json_response(artifact: ArtifactPath | None) -> JSONResponse:
  if artifact is None:
    raise HTTPException(status_code=404, detail="Artifact not found")
  path = _assert_artifact_path_still_safe(artifact)
  if not path.is_file():
    raise HTTPException(status_code=404, detail="Artifact not found")
  with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  return JSONResponse(content=payload, headers=_file_cache_headers(path))


def _artifact_response_from_sidecar(
  *,
  path: Path,
  workspace_root: Path,
  ticker: str,
  skill: str,
  artifact_id: str,
  run_index: dict[str, dict[str, str]],
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
  binary_artifact_path = _string_or_none(payload.get("binary_artifact_path"))
  created_at = _string_or_none(payload.get("created_at")) or _created_at_from_mtime(path)
  run_metadata = _artifact_run_metadata(
    run_index,
    artifact_id=resolved_artifact_id,
    artifact_path=artifact_path,
    binary_artifact_path=binary_artifact_path,
  )
  skill_run_id = (
    run_metadata.get("skill_run_id")
    or _string_or_none(payload.get("skill_run_id"))
    or _skill_run_id_from_artifact_id(resolved_artifact_id)
  )

  return ArtifactResponse(
    ticker=_string_or_none(payload.get("ticker")) or ticker,
    skill=_string_or_none(payload.get("skill")) or skill,
    artifact_id=resolved_artifact_id,
    artifact_path=artifact_path,
    binary_artifact_path=binary_artifact_path,
    contract_name=_string_or_none(payload.get("contract_name")) or skill,
    data_source=_string_or_none(payload.get("data_source")) or "live",
    created_at=created_at,
    skill_run_id=skill_run_id,
    run_id=run_metadata.get("run_id") or _string_or_none(payload.get("run_id")) or _string_or_none(payload.get("control_run_id")),
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


def _html_artifact_responses_for_user(
  workspace_root: Path,
  *,
  ticker_filter: str | None,
  skill_filter: str | None,
  run_index: dict[str, dict[str, str]],
  limit: int,
) -> list[tuple[int, str, ArtifactResponse]]:
  try:
    html_artifacts = _list_stored_html_artifacts(
      workspace_root,
      ticker=ticker_filter,
      limit=limit,
    )
  except ModuleNotFoundError as exc:
    if exc.name == "schema":
      return []
    raise
  except ValueError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc

  responses: list[tuple[int, str, ArtifactResponse]] = []
  for artifact in html_artifacts:
    source_skill = _string_or_none(artifact.source_skill) or "_html"
    if skill_filter is not None and source_skill != skill_filter:
      continue
    artifact_id = _string_or_none(artifact.artifact_id)
    if artifact_id is None:
      continue
    try:
      sidecar_path = (workspace_root / "artifacts" / "_html" / f"{artifact_id}.json").resolve()
      sidecar_path.relative_to(workspace_root.resolve())
      mtime_ns = sidecar_path.stat().st_mtime_ns
    except (OSError, ValueError):
      continue
    artifact_path = f"artifacts/_html/{artifact_id}.json"
    binary_artifact_path = f"artifacts/_html/{artifact_id}.html"
    run_metadata = _artifact_run_metadata(
      run_index,
      artifact_id=artifact_id,
      artifact_path=artifact_path,
      binary_artifact_path=binary_artifact_path,
    )
    run_id = run_metadata.get("run_id") or _string_or_none(artifact.session_id)
    responses.append(
      (
        mtime_ns,
        artifact_path,
        ArtifactResponse(
          ticker=_string_or_none(artifact.ticker) or "HTML",
          skill=source_skill,
          artifact_id=artifact_id,
          artifact_path=artifact_path,
          binary_artifact_path=binary_artifact_path,
          contract_name="HtmlArtifact",
          data_source="live",
          created_at=_string_or_none(artifact.ts) or _created_at_from_mtime(sidecar_path),
          skill_run_id=run_metadata.get("skill_run_id") or _string_or_none(artifact.session_id) or artifact_id,
          run_id=run_id,
        ),
      )
    )
  return responses


def _dashboard_artifact_responses_for_user(
  workspace_root: Path,
  *,
  ticker_filter: str | None,
  skill_filter: str | None,
  run_index: dict[str, dict[str, str]],
  limit: int,
) -> list[tuple[int, str, ArtifactResponse]]:
  try:
    dashboard_artifacts = _list_stored_dashboard_artifacts(
      workspace_root,
      ticker=ticker_filter,
      limit=limit,
    )
  except ModuleNotFoundError as exc:
    if exc.name == "schema":
      return []
    raise
  except ValueError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc

  responses: list[tuple[int, str, ArtifactResponse]] = []
  for artifact in dashboard_artifacts:
    source_skill = _string_or_none(artifact.source_skill) or "_dashboard"
    if skill_filter is not None and source_skill != skill_filter:
      continue
    artifact_id = _string_or_none(artifact.artifact_id)
    if artifact_id is None:
      continue
    try:
      sidecar_path = (workspace_root / "artifacts" / "_dashboards" / f"{artifact_id}.json").resolve()
      sidecar_path.relative_to(workspace_root.resolve())
      mtime_ns = sidecar_path.stat().st_mtime_ns
    except (OSError, ValueError):
      continue
    artifact_path = f"artifacts/_dashboards/{artifact_id}.json"
    binary_artifact_path = f"artifacts/_dashboards/{artifact_id}.payload.json"
    run_metadata = _artifact_run_metadata(
      run_index,
      artifact_id=artifact_id,
      artifact_path=artifact_path,
      binary_artifact_path=binary_artifact_path,
    )
    responses.append(
      (
        mtime_ns,
        artifact_path,
        ArtifactResponse(
          ticker=_string_or_none(artifact.ticker) or "Dashboard",
          skill=source_skill,
          artifact_id=artifact_id,
          artifact_path=artifact_path,
          binary_artifact_path=binary_artifact_path,
          contract_name="DashboardArtifact",
          data_source="live",
          created_at=_string_or_none(artifact.ts) or _created_at_from_mtime(sidecar_path),
          skill_run_id=run_metadata.get("skill_run_id") or artifact_id,
          run_id=run_metadata.get("run_id"),
        ),
      )
    )
  return responses


def build_artifacts_router(
  *,
  artifact_auth_dependency: ArtifactAuthDependency,
  autonomous_registry: Any | None = None,
) -> APIRouter:
  router = APIRouter(prefix="/artifacts")

  @router.get("", response_model=ArtifactsListResponse)
  async def list_artifacts(
    request: Request,
    ticker: str | None = Query(default=None),
    skill: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1),
  ) -> ArtifactsListResponse:
    user_id = artifact_auth_dependency(request)
    ticker_filter = _filter_ticker(ticker)
    skill_filter = _filter_skill(skill)
    run_id_filter = _string_or_none(run_id)
    effective_limit = min(limit, 50)

    try:
      workspace_root = user_workspace_root(user_id)
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc

    registry = getattr(request.app.state, "subprocess_registry", None) or autonomous_registry
    run_index = _artifact_run_index_from_registry(registry, user_id)
    run_windows = _artifact_run_windows_from_registry(registry, user_id)

    artifact_entries: list[tuple[int, str, ArtifactResponse]] = []
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
        run_index=run_index,
      )
      if artifact is None:
        continue
      if run_id_filter is not None and artifact.run_id != run_id_filter:
        continue
      if run_id_filter is not None and not _artifact_matches_run_window(
        artifact,
        run_id=run_id_filter,
        run_windows=run_windows,
      ):
        continue
      artifact_entries.append((_mtime_ns, path.as_posix(), artifact))

    for entry in _html_artifact_responses_for_user(
        workspace_root=workspace_root,
        ticker_filter=ticker_filter,
        skill_filter=skill_filter,
        run_index=run_index,
        limit=effective_limit,
    ):
      if run_id_filter is not None and entry[2].run_id != run_id_filter:
        continue
      if run_id_filter is not None and not _artifact_matches_run_window(
        entry[2],
        run_id=run_id_filter,
        run_windows=run_windows,
      ):
        continue
      artifact_entries.append(entry)

    for entry in _dashboard_artifact_responses_for_user(
        workspace_root=workspace_root,
        ticker_filter=ticker_filter,
        skill_filter=skill_filter,
        run_index=run_index,
        limit=effective_limit,
    ):
      if run_id_filter is not None and entry[2].run_id != run_id_filter:
        continue
      if run_id_filter is not None and not _artifact_matches_run_window(
        entry[2],
        run_id=run_id_filter,
        run_windows=run_windows,
      ):
        continue
      artifact_entries.append(entry)

    artifact_entries.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return ArtifactsListResponse(
      artifacts=[artifact for _mtime_ns, _path, artifact in artifact_entries[:effective_limit]]
    )

  @router.get("/{ticker}/{skill}/latest")
  async def latest_artifact(request: Request, ticker: str, skill: str) -> JSONResponse:
    user_id = artifact_auth_dependency(request)
    try:
      artifact = latest_artifact_json_path_for_request(user_id, ticker=ticker, skill=skill)
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    return _artifact_json_response(artifact)

  @router.get("/{ticker}/{skill}/{artifact_id}")
  async def artifact_by_id(request: Request, ticker: str, skill: str, artifact_id: str) -> JSONResponse:
    user_id = artifact_auth_dependency(request)
    try:
      artifact = artifact_json_path_for_request(
        user_id,
        ticker=ticker,
        skill=skill,
        artifact_id=artifact_id,
      )
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    return _artifact_json_response(artifact)

  return router


__all__ = [
  "ArtifactResponse",
  "ArtifactsListResponse",
  "build_artifacts_router",
]
