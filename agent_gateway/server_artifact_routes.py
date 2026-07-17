from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse


def _first_text_field(payload: Mapping[str, Any], *keys: str) -> str | None:
  for key in keys:
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()
  return None


def _artifact_picker_copy(payload: Mapping[str, Any]) -> dict[str, str]:
  verdict = payload.get("verdict")
  verdict_fields = verdict if isinstance(verdict, Mapping) else {}
  values = {
    "title": _first_text_field(payload, "title", "label"),
    "conclusion": (
      _first_text_field(payload, "conclusion", "judgment", "summary", "decision", "rationale")
      or _first_text_field(
        verdict_fields,
        "one_line_summary",
        "conclusion",
        "judgment",
        "summary",
        "decision",
        "rationale",
      )
    ),
    "created_at": _first_text_field(payload, "created_at", "createdAt", "ts"),
    "contract_name": _first_text_field(payload, "contract_name", "contractName"),
  }
  return {key: value for key, value in values.items() if value is not None}


def artifact_latest_response(
  parent: Mapping[str, Any],
  request: Request,
  ticker: str,
  skill: str,
) -> JSONResponse:
  user_id = parent["_artifact_auth_dependency"](request)
  filters = parent["_artifact_request_filters"](request)
  artifact_path_error = parent["ArtifactPathError"]
  try:
    artifacts = parent["artifact_json_paths_for_request"](user_id, ticker=ticker, skill=skill)
  except artifact_path_error as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
  for artifact in reversed(artifacts):
    try:
      return parent["_artifact_json_response"](artifact, user_id=user_id, filters=filters)
    except HTTPException as exc:
      if exc.status_code != 404:
        raise
  raise HTTPException(status_code=404, detail="Artifact not found")


def artifact_by_id_response(
  parent: Mapping[str, Any],
  request: Request,
  ticker: str,
  skill: str,
  artifact_id: str,
) -> JSONResponse:
  user_id = parent["_artifact_auth_dependency"](request)
  filters = parent["_artifact_request_filters"](request)
  artifact_path_error = parent["ArtifactPathError"]
  try:
    artifact = parent["artifact_json_path_for_request"](
      user_id,
      ticker=ticker,
      skill=skill,
      artifact_id=artifact_id,
    )
  except artifact_path_error as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
  return parent["_artifact_json_response"](artifact, user_id=user_id, filters=filters)


def artifact_index_response(parent: Mapping[str, Any], request: Request, ticker: str) -> JSONResponse:
  user_id = parent["_artifact_auth_dependency"](request)
  filters = parent["_artifact_request_filters"](request)
  artifact_path_error = parent["ArtifactPathError"]
  try:
    artifacts_by_skill = parent["ticker_artifact_paths_for_request"](user_id, ticker=ticker)
  except artifact_path_error as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
  decorated: list[dict[str, Any]] = []
  for skill, artifacts in sorted(artifacts_by_skill.items(), key=lambda item: item[0]):
    matching: list[tuple[Any, dict[str, Any]]] = []
    for artifact in artifacts:
      path = parent["_assert_artifact_path_still_safe"](artifact)
      if not path.is_file():
        continue
      payload = parent["_decorate_artifact_payload"](
        parent["_artifact_payload_from_path"](path),
        user_id=user_id,
      )
      if parent["_artifact_payload_matches_filters"](payload, filters=filters):
        matching.append((artifact, payload))
    if not matching:
      continue
    latest_artifact, latest_payload = matching[-1]
    recent_artifact_ids = [
      str(artifact.artifact_id)
      for artifact, _payload in reversed(matching[-parent["_ARTIFACT_INDEX_RECENT_LIMIT"]:])
      if artifact.artifact_id is not None
    ]
    decorated.append({
      "skill": skill,
      "latest_artifact_id": latest_artifact.artifact_id,
      "artifact_count": len(matching),
      "recent_artifact_ids": recent_artifact_ids,
      "research_file_id": latest_payload.get("research_file_id"),
      "control_run_id": latest_payload.get("control_run_id"),
      "has_research_file": latest_payload.get("has_research_file") is True,
      "origin_kind": latest_payload.get("origin_kind"),
      "visibility": latest_payload.get("visibility"),
      "origin_ref": latest_payload.get("origin_ref"),
      "classification_source": latest_payload.get("classification_source"),
      **_artifact_picker_copy(latest_payload),
    })
  return JSONResponse(content=decorated)


def ui_blocks_by_id_response(
  parent: Mapping[str, Any],
  request: Request,
  ui_blocks_id: str,
) -> JSONResponse:
  user_id = parent["_artifact_auth_dependency"](request)
  if re.fullmatch(r"ub_[0-9a-f]{16}", ui_blocks_id) is None:
    raise HTTPException(status_code=400, detail="Invalid ui_blocks_id")
  envelope = parent["read_ui_blocks_payload"](
    parent["user_workspace_root"](user_id),
    ui_blocks_id,
  )
  if envelope is None:
    raise HTTPException(status_code=404, detail="UI blocks payload not found")
  return JSONResponse(
    content=envelope,
    headers={
      "Cache-Control": "private, max-age=0",
      "X-Content-Type-Options": "nosniff",
    },
  )


def letter_by_id_response(
  parent: Mapping[str, Any],
  request: Request,
  ticker: str,
  artifact_id: str,
) -> FileResponse:
  user_id = parent["_artifact_auth_dependency"](request)
  artifact_path_error = parent["ArtifactPathError"]
  try:
    artifact = parent["letter_docx_path_for_request"](
      user_id,
      ticker=ticker,
      artifact_id=artifact_id,
    )
  except artifact_path_error as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc

  path = parent["_assert_artifact_path_still_safe"](artifact)
  if not path.is_file():
    raise HTTPException(status_code=404, detail="Letter artifact not found")
  headers = parent["_file_cache_headers"](path)
  headers["Content-Disposition"] = (
    f'attachment; filename="{parent["_letter_filename"](artifact.ticker, artifact.artifact_id or artifact_id)}"'
  )
  return FileResponse(path, media_type=parent["_ARTIFACT_DOCX_MEDIA_TYPE"], headers=headers)


def artifact_path_guard_response(parent: Mapping[str, Any], request: Request, artifact_path: str) -> JSONResponse:
  parent["_artifact_auth_dependency"](request)
  artifact_path_error = parent["ArtifactPathError"]
  try:
    parent["reject_unsafe_path"](artifact_path)
  except artifact_path_error as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
  raise HTTPException(status_code=404, detail="Artifact not found")


def letter_path_guard_response(parent: Mapping[str, Any], request: Request, letter_path: str) -> JSONResponse:
  parent["_artifact_auth_dependency"](request)
  artifact_path_error = parent["ArtifactPathError"]
  try:
    parent["reject_unsafe_path"](letter_path)
  except artifact_path_error as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
  raise HTTPException(status_code=404, detail="Letter artifact not found")
