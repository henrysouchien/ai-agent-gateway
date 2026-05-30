from __future__ import annotations

import functools

from typing import Annotated, Any, Literal, Union

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from agent_gateway.session import AuthManager, GatewaySession


ScheduleSource = Literal["launchd", "jobs-mcp"]
JobsFrequency = Literal["daily", "weekly", "monthly", "quarterly"]


@functools.cache
def _scheduler_mcp():
  from mcp_servers.scheduler_mcp import server

  return server


@functools.cache
def _jobs_api():
  from investment_tools.jobs import api

  return api


class ScheduleBaseResponse(BaseModel):
  source: ScheduleSource
  name: str
  enabled: bool
  schedule_description: str
  last_run_at: str | None = None
  next_run_at: str | None = None


class LaunchdScheduleResponse(ScheduleBaseResponse):
  source: Literal["launchd"]
  label: str
  command: list[str]
  working_directory: str
  last_exit_status: int | None = None
  recent_log_lines: list[str] = Field(default_factory=list)


class JobsMcpScheduleResponse(ScheduleBaseResponse):
  source: Literal["jobs-mcp"]
  schedule_id: str
  job_type: str
  frequency: JobsFrequency
  time_of_day: str | None = None
  day_of_week: int | None = None
  day_of_month: int | None = None
  params: dict[str, Any] = Field(default_factory=dict)


ScheduleResponse = Union[LaunchdScheduleResponse, JobsMcpScheduleResponse]


class SchedulesListResponse(BaseModel):
  schedules: list[ScheduleResponse]


class ScheduleEnvelopeResponse(BaseModel):
  schedule: ScheduleResponse


class ScheduleLogsResponse(BaseModel):
  name: str
  log_lines: list[str]


class ScheduleDeleteResponse(BaseModel):
  deleted: bool
  name: str
  source: ScheduleSource


class CreateLaunchdScheduleRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  source: Literal["launchd"]
  name: str = Field(..., min_length=1)
  command: list[str] = Field(..., min_length=1)
  schedule: dict[str, Any] | list[dict[str, Any]]
  working_directory: str = Field(..., min_length=1)
  log_file: str | None = None
  comment: str | None = None


class CreateJobsMcpScheduleRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  source: Literal["jobs-mcp"]
  name: str = Field(..., min_length=1)
  job_type: str = Field(..., min_length=1)
  frequency: JobsFrequency
  time_of_day: str | None = None
  day_of_week: StrictInt | None = None
  day_of_month: StrictInt | None = None
  params: dict[str, Any] = Field(default_factory=dict)


CreateScheduleRequest = Annotated[
  Union[CreateLaunchdScheduleRequest, CreateJobsMcpScheduleRequest],
  Body(discriminator="source"),
]


class ScheduleEnabledRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  enabled: StrictBool


def _require_bearer_session(request: Request, auth: AuthManager) -> GatewaySession:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  return auth.verify_token(token)


def _error_message(result: dict[str, Any], fallback: str) -> str:
  raw = result.get("message") or result.get("error") or fallback
  return str(raw)


def _require_backend_success(
  result: dict[str, Any],
  *,
  action: str,
  not_found_status: int = status.HTTP_400_BAD_REQUEST,
) -> dict[str, Any]:
  if not isinstance(result, dict):
    raise HTTPException(status_code=502, detail=f"{action} returned an invalid response")
  if str(result.get("status") or "").lower() in {"ok", "success"}:
    return result

  message = _error_message(result, f"{action} failed")
  status_code = not_found_status if "not found" in message.lower() else status.HTTP_400_BAD_REQUEST
  raise HTTPException(status_code=status_code, detail=message)


def _optional_text(value: Any) -> str | None:
  if value is None:
    return None
  return str(value)


def _optional_int(value: Any) -> int | None:
  if value is None or isinstance(value, bool):
    return None
  try:
    return int(value)
  except (TypeError, ValueError):
    return None


def _as_string_list(value: Any) -> list[str]:
  if not isinstance(value, list):
    return []
  return [str(item) for item in value]


def _launchd_name(raw: dict[str, Any]) -> str:
  if raw.get("name") is not None:
    return str(raw["name"])
  label = str(raw.get("label") or "")
  prefix = str(getattr(_scheduler_mcp(), "_PLIST_PREFIX", "com.henrychien."))
  return label[len(prefix):] if label.startswith(prefix) else label


def _normalize_schedule(raw: dict[str, Any]) -> ScheduleResponse:
  source = raw.get("source")
  if source == "launchd":
    return LaunchdScheduleResponse(
      source="launchd",
      name=_launchd_name(raw),
      label=str(raw.get("label") or ""),
      enabled=bool(raw.get("enabled")),
      schedule_description=str(raw.get("schedule_description") or ""),
      last_run_at=_optional_text(raw.get("last_run_at")),
      next_run_at=_optional_text(raw.get("next_run_at")),
      command=_as_string_list(raw.get("command")),
      working_directory=str(raw.get("working_directory") or ""),
      last_exit_status=_optional_int(raw.get("last_exit_status")),
      recent_log_lines=_as_string_list(raw.get("recent_log_lines")),
    )

  if source == "jobs-mcp":
    frequency = raw.get("frequency")
    if frequency not in {"daily", "weekly", "monthly", "quarterly"}:
      raise HTTPException(status_code=400, detail="jobs-mcp schedule missing supported frequency")
    return JobsMcpScheduleResponse(
      source="jobs-mcp",
      name=str(raw.get("name") or ""),
      enabled=bool(raw.get("enabled")),
      schedule_description=str(raw.get("schedule_description") or ""),
      last_run_at=_optional_text(raw.get("last_run_at")),
      next_run_at=_optional_text(raw.get("next_run_at")),
      schedule_id=str(raw.get("schedule_id") or raw.get("name") or ""),
      job_type=str(raw.get("job_type") or ""),
      frequency=frequency,
      time_of_day=_optional_text(raw.get("time_of_day")),
      day_of_week=_optional_int(raw.get("day_of_week")),
      day_of_month=_optional_int(raw.get("day_of_month")),
      params=dict(raw.get("params") or {}),
    )

  raise HTTPException(status_code=400, detail=f"Unsupported schedule source: {source!r}")


def _show_schedule(name: str, *, source: ScheduleSource | None = None) -> ScheduleResponse:
  result = _scheduler_mcp().schedule_show(name, source=source)
  return _normalize_schedule(
    _require_backend_success(result, action="schedule_show", not_found_status=status.HTTP_404_NOT_FOUND)
  )


def _show_jobs_schedule(name: str) -> JobsMcpScheduleResponse:
  schedule = _show_schedule(name, source="jobs-mcp")
  if not isinstance(schedule, JobsMcpScheduleResponse):
    raise HTTPException(status_code=400, detail=f"Schedule is not a jobs-mcp schedule: {name}")
  return schedule


def _list_schedules(source: ScheduleSource | None) -> list[ScheduleResponse]:
  result = _scheduler_mcp().schedule_list(source=source)
  payload = _require_backend_success(result, action="schedule_list")
  schedules: list[ScheduleResponse] = []
  for raw in payload.get("schedules") or []:
    if not isinstance(raw, dict):
      continue
    if raw.get("source") == "jobs-mcp" and raw.get("frequency") not in {"daily", "weekly", "monthly", "quarterly"}:
      schedules.append(_show_schedule(str(raw.get("name") or raw.get("schedule_id") or ""), source="jobs-mcp"))
      continue
    schedules.append(_normalize_schedule(raw))
  return schedules


def build_schedules_router(*, auth: AuthManager) -> APIRouter:
  router = APIRouter(prefix="/schedules")
  operator_global_description = (
    "Operator-global schedule management. Schedules are not user-scoped resources; "
    "any authenticated control-plane user sees the same launchd and jobs-mcp schedules."
  )

  @router.get(
    "",
    response_model=SchedulesListResponse,
    description=operator_global_description,
  )
  async def list_schedules(
    request: Request,
    source: ScheduleSource | None = Query(default=None),
  ) -> SchedulesListResponse:
    _require_bearer_session(request, auth)
    return SchedulesListResponse(schedules=_list_schedules(source))

  @router.get(
    "/{name}",
    response_model=ScheduleResponse,
    description=operator_global_description,
  )
  async def get_schedule(request: Request, name: str) -> ScheduleResponse:
    _require_bearer_session(request, auth)
    return _show_schedule(name)

  @router.get(
    "/{name}/logs",
    response_model=ScheduleLogsResponse,
    description=f"{operator_global_description} Logs are available for launchd schedules only.",
  )
  async def get_schedule_logs(
    request: Request,
    name: str,
    lines: int = Query(default=50, ge=0),
  ) -> ScheduleLogsResponse:
    _require_bearer_session(request, auth)
    result = _scheduler_mcp().schedule_logs(name, lines=lines)
    payload = _require_backend_success(result, action="schedule_logs", not_found_status=status.HTTP_404_NOT_FOUND)
    return ScheduleLogsResponse(name=name, log_lines=_as_string_list(payload.get("lines")))

  @router.post(
    "",
    response_model=ScheduleEnvelopeResponse,
    status_code=status.HTTP_201_CREATED,
    description=operator_global_description,
  )
  async def create_schedule(request: Request, payload: CreateScheduleRequest) -> ScheduleEnvelopeResponse:
    _require_bearer_session(request, auth)
    if payload.source == "launchd":
      result = _scheduler_mcp().schedule_create(
        name=payload.name,
        command=payload.command,
        schedule=payload.schedule,
        working_directory=payload.working_directory,
        log_file=payload.log_file,
        comment=payload.comment or "",
      )
      _require_backend_success(result, action="schedule_create")
      return ScheduleEnvelopeResponse(schedule=_show_schedule(payload.name, source="launchd"))

    result = _jobs_api().create_schedule(
      payload.name,
      payload.job_type,
      payload.frequency,
      time_of_day=payload.time_of_day,
      day_of_week=payload.day_of_week,
      day_of_month=payload.day_of_month,
      params=payload.params,
    )
    _require_backend_success(result, action="jobs-mcp create_schedule")
    return ScheduleEnvelopeResponse(schedule=_show_jobs_schedule(payload.name))

  @router.put(
    "/{name}/enabled",
    response_model=ScheduleEnvelopeResponse,
    description=operator_global_description,
  )
  async def set_schedule_enabled(
    request: Request,
    name: str,
    payload: ScheduleEnabledRequest,
  ) -> ScheduleEnvelopeResponse:
    _require_bearer_session(request, auth)
    schedule = _show_schedule(name)
    if isinstance(schedule, LaunchdScheduleResponse):
      result = _scheduler_mcp().schedule_enable(name) if payload.enabled else _scheduler_mcp().schedule_disable(name)
      _require_backend_success(result, action="schedule_enable" if payload.enabled else "schedule_disable")
      return ScheduleEnvelopeResponse(schedule=_show_schedule(name, source="launchd"))

    result = _jobs_api().update_schedule(schedule.schedule_id, enabled=payload.enabled)
    _require_backend_success(
      result,
      action="jobs-mcp update_schedule",
      not_found_status=status.HTTP_404_NOT_FOUND,
    )
    return ScheduleEnvelopeResponse(schedule=_show_jobs_schedule(name))

  @router.delete(
    "/{name}",
    response_model=ScheduleDeleteResponse,
    description=operator_global_description,
  )
  async def delete_schedule(
    request: Request,
    name: str,
    confirm: bool = Query(default=False),
  ) -> ScheduleDeleteResponse:
    _require_bearer_session(request, auth)
    if not confirm:
      raise HTTPException(status_code=400, detail="confirm=true is required to delete a schedule")

    schedule = _show_schedule(name)
    if isinstance(schedule, LaunchdScheduleResponse):
      result = _scheduler_mcp().schedule_delete(name, confirm=True)
      _require_backend_success(result, action="schedule_delete", not_found_status=status.HTTP_404_NOT_FOUND)
      return ScheduleDeleteResponse(deleted=True, name=schedule.name, source="launchd")

    result = _jobs_api().delete_schedule(schedule.schedule_id, dry_run=False)
    _require_backend_success(
      result,
      action="jobs-mcp delete_schedule",
      not_found_status=status.HTTP_404_NOT_FOUND,
    )
    return ScheduleDeleteResponse(deleted=True, name=schedule.name, source="jobs-mcp")

  return router


__all__ = [
  "CreateJobsMcpScheduleRequest",
  "CreateLaunchdScheduleRequest",
  "CreateScheduleRequest",
  "JobsMcpScheduleResponse",
  "LaunchdScheduleResponse",
  "ScheduleDeleteResponse",
  "ScheduleEnvelopeResponse",
  "ScheduleLogsResponse",
  "ScheduleResponse",
  "SchedulesListResponse",
  "build_schedules_router",
]
