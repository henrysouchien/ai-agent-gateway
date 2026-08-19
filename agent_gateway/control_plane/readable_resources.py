from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from agent_gateway.autonomous_runner import AutonomousRegistry
from agent_gateway.session import AuthManager, GatewaySession

from .runs_chat_helpers import _require_control_session
from .runs_helpers import (
  _chat_session_has_run_activity,
  _normalize_channel,
  _record_owner_user_id,
  _require_bearer_session,
  _run_channel_matches,
  _session_matches_owner,
  _session_owner_user_id,
)

_RESOURCE_ID_RE = re.compile(r"^[^/\\\s]+$")
_CONTENT_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_READABLE_RESOURCE_EVENT_TYPE = "readable_resource_ready"
_READABLE_RESOURCE_MAX_CONTENT_BYTES = 2_000_000


class ReadableResourceResponse(BaseModel):
  resource_id: str
  control_run_id: str | None = None
  run_id: str | None = None
  session_id: str | None = None
  task_id: str | None = None
  skill_run_id: str
  contract_name: str
  content_type: str
  content_class: str
  content_snapshot_id: str
  content_sha256: str
  content_bytes: int
  truncated: bool
  title: str | None = None
  source_path: str | None = None
  byte_start: int | None = None
  byte_end: int | None = None
  tool_name: str | None = None
  created_at: str | None = None


class ReadableResourceDetailResponse(ReadableResourceResponse):
  content: str


class ReadableResourceListResponse(BaseModel):
  readable_resources: list[ReadableResourceResponse]
  next_cursor: str | None = None


class _ReadableResourceInvalid(ValueError):
  pass


@dataclass(frozen=True)
class _ReadableResourceSource:
  control_run_id: str
  run_id: str
  session_id: str | None
  task_id: str | None
  events: list[dict[str, Any]]


def _string_or_none(value: Any) -> str | None:
  if not isinstance(value, str):
    return None
  cleaned = value.strip()
  return cleaned or None


def _validate_resource_id(resource_id: str) -> str:
  cleaned = resource_id.strip()
  if not _resource_id_is_safe(cleaned):
    raise HTTPException(status_code=400, detail="Unsafe readable resource id")
  return cleaned


def _resource_id_is_safe(resource_id: str) -> bool:
  return bool(
    resource_id
    and len(resource_id) <= 512
    and resource_id not in {".", ".."}
    and _RESOURCE_ID_RE.fullmatch(resource_id) is not None
  )


def _non_negative_int(value: Any) -> int | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, int) and value >= 0:
    return value
  return None


def _required_string(event: dict[str, Any], key: str) -> str | None:
  value = _string_or_none(event.get(key))
  if value is None or len(value) > 512:
    return None
  return value


def _optional_metadata_string(value: Any) -> str | None:
  normalized = _string_or_none(value)
  if normalized is None or len(normalized) > 512:
    return None
  return normalized


def _resource_owner_id(event: dict[str, Any]) -> str | None:
  for key in ("control_run_id", "run_id", "session_id", "task_id"):
    value = _required_string(event, key)
    if value is not None:
      return value
  return None


def _event_claims_match_source(event: dict[str, Any], source: _ReadableResourceSource) -> bool:
  expected = {
    "control_run_id": source.control_run_id,
    "run_id": source.run_id,
    "session_id": source.session_id,
    "task_id": source.task_id,
  }
  for key, expected_value in expected.items():
    claimed = _string_or_none(event.get(key))
    if claimed is not None and claimed != expected_value:
      return False
  return True


def _content_payload(event: dict[str, Any]) -> tuple[str, int, str]:
  content = event.get("content")
  if not isinstance(content, str) or not content.strip():
    raise _ReadableResourceInvalid("readable resource snapshot content is missing")
  content_bytes = content.encode("utf-8")
  if len(content_bytes) > _READABLE_RESOURCE_MAX_CONTENT_BYTES:
    raise _ReadableResourceInvalid("readable resource snapshot content exceeds 2 MB")

  declared_bytes = _non_negative_int(event.get("content_bytes"))
  if declared_bytes is None:
    raise _ReadableResourceInvalid("readable resource content_bytes is missing")
  if declared_bytes != len(content_bytes):
    raise _ReadableResourceInvalid("readable resource content byte length mismatch")

  content_sha256 = _required_string(event, "content_sha256")
  if content_sha256 is None or _CONTENT_SHA256_RE.fullmatch(content_sha256) is None:
    raise _ReadableResourceInvalid("readable resource content_sha256 is invalid")
  normalized_sha = content_sha256.lower()
  if hashlib.sha256(content_bytes).hexdigest() != normalized_sha:
    raise _ReadableResourceInvalid("readable resource content sha256 mismatch")
  return content, declared_bytes, normalized_sha


def _resource_payload_from_event(
  event: dict[str, Any],
  *,
  source: _ReadableResourceSource,
  include_content: bool,
) -> tuple[dict[str, Any], float]:
  if event.get("type") != _READABLE_RESOURCE_EVENT_TYPE:
    raise _ReadableResourceInvalid("not a readable resource event")
  if _string_or_none(event.get("content_class")) != "human_readable":
    raise _ReadableResourceInvalid("readable resource is not human readable")
  if _string_or_none(event.get("content_type")) not in {"text/markdown", "text/plain"}:
    raise _ReadableResourceInvalid("readable resource content_type is not allowed")

  resource_id = _required_string(event, "resource_id")
  if resource_id is None:
    raise _ReadableResourceInvalid("readable resource id is missing")
  if not _resource_id_is_safe(resource_id):
    raise _ReadableResourceInvalid("readable resource id is unsafe")
  owner_id = _resource_owner_id(event)
  if owner_id is None:
    raise _ReadableResourceInvalid("readable resource owner provenance is missing")
  if not _event_claims_match_source(event, source):
    raise _ReadableResourceInvalid("readable resource owner provenance conflicts with source run")
  skill_run_id = _required_string(event, "skill_run_id")
  contract_name = _required_string(event, "contract_name")
  snapshot_id = _required_string(event, "content_snapshot_id")
  if skill_run_id is None or contract_name is None or snapshot_id is None:
    raise _ReadableResourceInvalid("readable resource provenance is incomplete")
  content, content_bytes, content_sha256 = _content_payload(event)
  truncated = event.get("truncated")
  if not isinstance(truncated, bool):
    raise _ReadableResourceInvalid("readable resource truncated flag is missing")
  byte_start = _non_negative_int(event.get("byte_start"))
  byte_end = _non_negative_int(event.get("byte_end"))
  if byte_start is not None and byte_end is not None and byte_end < byte_start:
    raise _ReadableResourceInvalid("readable resource byte range is invalid")

  payload: dict[str, Any] = {
    "resource_id": resource_id,
    "control_run_id": source.control_run_id,
    "run_id": source.run_id,
    "session_id": source.session_id,
    "task_id": source.task_id,
    "skill_run_id": skill_run_id,
    "contract_name": contract_name,
    "content_type": str(event["content_type"]),
    "content_class": "human_readable",
    "content_snapshot_id": snapshot_id,
    "content_sha256": content_sha256,
    "content_bytes": content_bytes,
    "truncated": truncated,
    "title": _optional_metadata_string(event.get("title")),
    "source_path": _optional_metadata_string(event.get("source_path")),
    "byte_start": byte_start,
    "byte_end": byte_end,
    "tool_name": _optional_metadata_string(event.get("tool_name")),
    "created_at": _optional_metadata_string(event.get("created_at")),
  }
  if include_content:
    payload["content"] = content

  try:
    sort_ts = float(event.get("ts") or 0)
  except (TypeError, ValueError):
    sort_ts = 0.0
  return payload, sort_ts


def _chat_event_sources(
  *,
  auth: AuthManager,
  owner_user_id: str,
  channel: str | None,
) -> list[_ReadableResourceSource]:
  return [
    _ReadableResourceSource(
      control_run_id=session.session_id,
      run_id=session.session_id,
      session_id=session.session_id,
      task_id=None,
      events=session.event_history.snapshot(),
    )
    for session in auth.session_store.visible_sessions_snapshot()
    if (
      session.kind == "chat"
      and _session_matches_owner(session, owner_user_id)
      and _run_channel_matches(session.channel, channel)
      and _chat_session_has_run_activity(session)
    )
  ]


def _autonomous_event_sources(
  *,
  autonomous_registry: AutonomousRegistry | None,
  owner_user_id: str,
  channel: str | None,
) -> list[_ReadableResourceSource]:
  if autonomous_registry is None:
    return []
  return [
    _ReadableResourceSource(
      control_run_id=str(record.control_run_id),
      run_id=str(record.control_run_id),
      session_id=None,
      task_id=str(record.task_id),
      events=list(record.event_lines or []),
    )
    for record in autonomous_registry._tasks.values()
    if _record_owner_user_id(record) == owner_user_id and _run_channel_matches(record.channel, channel)
  ]


def _visible_event_sources(
  *,
  auth: AuthManager,
  autonomous_registry: AutonomousRegistry | None,
  authenticated: GatewaySession,
) -> list[_ReadableResourceSource]:
  owner_user_id = _session_owner_user_id(authenticated)
  channel = _normalize_channel(authenticated.channel)
  return [
    *_chat_event_sources(auth=auth, owner_user_id=owner_user_id, channel=channel),
    *_autonomous_event_sources(
      autonomous_registry=autonomous_registry,
      owner_user_id=owner_user_id,
      channel=channel,
    ),
  ]


def _find_visible_resources(
  *,
  auth: AuthManager,
  autonomous_registry: AutonomousRegistry | None,
  authenticated: GatewaySession,
  include_content: bool,
) -> list[tuple[float, dict[str, Any]]]:
  resources: list[tuple[float, dict[str, Any]]] = []
  seen: set[str] = set()
  for source in _visible_event_sources(
    auth=auth,
    autonomous_registry=autonomous_registry,
    authenticated=authenticated,
  ):
    for event in source.events:
      if not isinstance(event, dict) or event.get("type") != _READABLE_RESOURCE_EVENT_TYPE:
        continue
      try:
        payload, sort_ts = _resource_payload_from_event(event, source=source, include_content=include_content)
      except _ReadableResourceInvalid:
        continue
      resource_id = payload["resource_id"]
      if resource_id in seen:
        continue
      seen.add(resource_id)
      resources.append((sort_ts, payload))
  resources.sort(key=lambda item: item[0], reverse=True)
  return resources


def build_readable_resources_router(
  *,
  auth: AuthManager,
  autonomous_registry: AutonomousRegistry | None = None,
) -> APIRouter:
  router = APIRouter(prefix="/readable-resources")

  @router.get("", response_model=ReadableResourceListResponse)
  async def list_readable_resources(
    request: Request,
    run_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
  ) -> ReadableResourceListResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    resources = _find_visible_resources(
      auth=auth,
      autonomous_registry=autonomous_registry,
      authenticated=authenticated,
      include_content=False,
    )
    if run_id is not None:
      normalized_run_id = run_id.strip()
      resources = [
        (sort_ts, resource)
        for sort_ts, resource in resources
        if normalized_run_id in {
          resource.get("control_run_id"),
          resource.get("run_id"),
          resource.get("session_id"),
          resource.get("task_id"),
        }
      ]
    return ReadableResourceListResponse(
      readable_resources=[
        ReadableResourceResponse(**resource)
        for _sort_ts, resource in resources[:limit]
      ],
      next_cursor=None,
    )

  @router.get("/{resource_id}", response_model=ReadableResourceDetailResponse)
  async def get_readable_resource(
    request: Request,
    resource_id: str,
  ) -> ReadableResourceDetailResponse:
    normalized_resource_id = _validate_resource_id(resource_id)
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    for _sort_ts, resource in _find_visible_resources(
      auth=auth,
      autonomous_registry=autonomous_registry,
      authenticated=authenticated,
      include_content=True,
    ):
      if resource.get("resource_id") == normalized_resource_id:
        return ReadableResourceDetailResponse(**resource)
    raise HTTPException(status_code=404, detail="Readable resource not found")

  return router


__all__ = [
  "ReadableResourceDetailResponse",
  "ReadableResourceListResponse",
  "ReadableResourceResponse",
  "build_readable_resources_router",
]
