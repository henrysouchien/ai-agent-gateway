from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from agent_gateway.artifact_paths import (
  ArtifactPathError,
  artifact_json_path_for_request,
  artifact_json_paths_for_request,
)
from agent_gateway.control_plane import artifact_index as _artifact_index
from agent_gateway.control_plane.artifact_index import (
  ArtifactAuthDependency,
  ArtifactResponse,
  ArtifactsListResponse,
  _artifact_event_keys,
  _artifact_filters,
  _artifact_json_response,
  _artifact_matches_run_window,
  _artifact_response_from_sidecar,
  _artifact_run_index_from_registry,
  _artifact_run_metadata,
  _artifact_run_windows_from_registry,
  _artifact_sidecars_for_user,
  _assert_artifact_path_still_safe,
  _created_at_from_mtime,
  _dashboard_artifact_responses_for_user,
  _effective_artifact_fields,
  _effective_auxiliary_artifact_fields,
  _file_cache_headers,
  _filter_skill,
  _filter_ticker,
  _html_artifact_responses_for_user,
  _int_or_none,
  _list_stored_dashboard_artifacts,
  _list_stored_html_artifacts,
  _origin_filter,
  _origin_kind,
  _origin_ref,
  _payload_matches_artifact_filters,
  _relative_sidecar_path,
  _research_file_classification,
  _research_file_id_token_present,
  _sidecar_classification,
  _skill_run_id_from_artifact_id,
  _string_or_none,
  _timestamp_from_iso,
  _visibility,
  _visibility_filter,
  artifact_responses_for_user as _artifact_responses_for_user,
)

_artifact_filters_impl = _artifact_filters
_artifact_json_response_impl = _artifact_json_response
_effective_artifact_fields_impl = _effective_artifact_fields
_effective_auxiliary_artifact_fields_impl = _effective_auxiliary_artifact_fields
_payload_matches_artifact_filters_impl = _payload_matches_artifact_filters


def _sync_compat_globals() -> None:
  for name in (
    "_artifact_event_keys",
    "_artifact_matches_run_window",
    "_artifact_response_from_sidecar",
    "_artifact_run_index_from_registry",
    "_artifact_run_metadata",
    "_artifact_run_windows_from_registry",
    "_artifact_sidecars_for_user",
    "_assert_artifact_path_still_safe",
    "_created_at_from_mtime",
    "_dashboard_artifact_responses_for_user",
    "_effective_artifact_fields",
    "_effective_auxiliary_artifact_fields",
    "_file_cache_headers",
    "_filter_skill",
    "_filter_ticker",
    "_html_artifact_responses_for_user",
    "_int_or_none",
    "_list_stored_dashboard_artifacts",
    "_list_stored_html_artifacts",
    "_origin_filter",
    "_origin_kind",
    "_origin_ref",
    "_payload_matches_artifact_filters",
    "_relative_sidecar_path",
    "_research_file_classification",
    "_research_file_id_token_present",
    "_sidecar_classification",
    "_skill_run_id_from_artifact_id",
    "_string_or_none",
    "_timestamp_from_iso",
    "_visibility",
    "_visibility_filter",
  ):
    setattr(_artifact_index, name, globals()[name])


def _artifact_filters(
  *,
  research_file_id: int | None,
  control_run_id: str | None,
  visibility: str | None,
  origin_kind: str | None,
) -> dict[str, Any]:
  _sync_compat_globals()
  return _artifact_filters_impl(
    research_file_id=research_file_id,
    control_run_id=control_run_id,
    visibility=visibility,
    origin_kind=origin_kind,
  )


def _artifact_json_response(
  artifact: Any,
  *,
  user_id: str,
  filters: dict[str, Any],
) -> JSONResponse:
  _sync_compat_globals()
  return _artifact_json_response_impl(artifact, user_id=user_id, filters=filters)


def _effective_artifact_fields(payload: dict[str, Any], *, user_id: str) -> dict[str, Any]:
  _sync_compat_globals()
  return _effective_artifact_fields_impl(payload, user_id=user_id)


def _effective_auxiliary_artifact_fields(payload: dict[str, Any], *, user_id: str) -> dict[str, Any]:
  _sync_compat_globals()
  return _effective_auxiliary_artifact_fields_impl(payload, user_id=user_id)


def _payload_matches_artifact_filters(payload: dict[str, Any], *, filters: dict[str, Any]) -> bool:
  _sync_compat_globals()
  return _payload_matches_artifact_filters_impl(payload, filters=filters)


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
    research_file_id: int | None = Query(default=None),
    control_run_id: str | None = Query(default=None),
    visibility: str = Query(default="default"),
    origin_kind: str = Query(default="all"),
    limit: int = Query(default=50, ge=1),
  ) -> ArtifactsListResponse:
    user_id = artifact_auth_dependency(request)
    _sync_compat_globals()
    ticker_filter = _filter_ticker(ticker)
    skill_filter = _filter_skill(skill)
    run_id_filter = _string_or_none(run_id)
    effective_limit = min(limit, 50)
    filters = _artifact_filters(
      research_file_id=research_file_id,
      control_run_id=control_run_id,
      visibility=visibility,
      origin_kind=origin_kind,
    )
    return ArtifactsListResponse(
      artifacts=_artifact_responses_for_user(
        request=request,
        user_id=user_id,
        ticker_filter=ticker_filter,
        skill_filter=skill_filter,
        run_id_filter=run_id_filter,
        filters=filters,
        effective_limit=effective_limit,
        autonomous_registry=autonomous_registry,
      )
    )

  @router.get("/{ticker}/{skill}/latest")
  async def latest_artifact(
    request: Request,
    ticker: str,
    skill: str,
    research_file_id: int | None = Query(default=None),
    control_run_id: str | None = Query(default=None),
    visibility: str = Query(default="default"),
    origin_kind: str = Query(default="all"),
  ) -> JSONResponse:
    user_id = artifact_auth_dependency(request)
    _sync_compat_globals()
    filters = _artifact_filters(
      research_file_id=research_file_id,
      control_run_id=control_run_id,
      visibility=visibility,
      origin_kind=origin_kind,
    )
    try:
      artifacts = artifact_json_paths_for_request(user_id, ticker=ticker, skill=skill)
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    for artifact in reversed(artifacts):
      try:
        return _artifact_json_response(artifact, user_id=user_id, filters=filters)
      except HTTPException as exc:
        if exc.status_code != 404:
          raise
    raise HTTPException(status_code=404, detail="Artifact not found")

  @router.get("/{ticker}/{skill}/{artifact_id}")
  async def artifact_by_id(
    request: Request,
    ticker: str,
    skill: str,
    artifact_id: str,
    research_file_id: int | None = Query(default=None),
    control_run_id: str | None = Query(default=None),
    visibility: str = Query(default="default"),
    origin_kind: str = Query(default="all"),
  ) -> JSONResponse:
    user_id = artifact_auth_dependency(request)
    _sync_compat_globals()
    filters = _artifact_filters(
      research_file_id=research_file_id,
      control_run_id=control_run_id,
      visibility=visibility,
      origin_kind=origin_kind,
    )
    try:
      artifact = artifact_json_path_for_request(
        user_id,
        ticker=ticker,
        skill=skill,
        artifact_id=artifact_id,
      )
    except ArtifactPathError as exc:
      raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
    return _artifact_json_response(artifact, user_id=user_id, filters=filters)

  return router


__all__ = [
  "ArtifactAuthDependency",
  "ArtifactResponse",
  "ArtifactsListResponse",
  "_artifact_event_keys",
  "_artifact_filters",
  "_artifact_json_response",
  "_artifact_matches_run_window",
  "_artifact_response_from_sidecar",
  "_artifact_run_index_from_registry",
  "_artifact_run_metadata",
  "_artifact_run_windows_from_registry",
  "_artifact_sidecars_for_user",
  "_assert_artifact_path_still_safe",
  "_created_at_from_mtime",
  "_dashboard_artifact_responses_for_user",
  "_effective_artifact_fields",
  "_effective_auxiliary_artifact_fields",
  "_file_cache_headers",
  "_filter_skill",
  "_filter_ticker",
  "_html_artifact_responses_for_user",
  "_int_or_none",
  "_list_stored_dashboard_artifacts",
  "_list_stored_html_artifacts",
  "_origin_filter",
  "_origin_kind",
  "_origin_ref",
  "_payload_matches_artifact_filters",
  "_relative_sidecar_path",
  "_research_file_classification",
  "_research_file_id_token_present",
  "_sidecar_classification",
  "_skill_run_id_from_artifact_id",
  "_string_or_none",
  "_sync_compat_globals",
  "_timestamp_from_iso",
  "_visibility",
  "_visibility_filter",
  "build_artifacts_router",
]
