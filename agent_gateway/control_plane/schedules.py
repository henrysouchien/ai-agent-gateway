from __future__ import annotations

import functools
import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import secrets
import time as monotonic_time

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationInfo, field_validator, model_validator

from agent_gateway.artifact_paths import user_data_dir
from agent_gateway.session import AuthManager, GatewaySession
from agent_gateway.role_validation import require_exact_role

from .runs_helpers import (
  AutonomousDispatchResponse,
  _autonomous_run_from_task,
  _autonomous_task_for_user,
)
from .runs_models import DispatchScope


logger = logging.getLogger(__name__)


ScheduleSource = Literal["launchd", "jobs-mcp"]
JobsFrequency = Literal["daily", "weekly", "monthly", "quarterly"]
AgentScheduleSource = Literal["agent-gateway"]
_AGENT_RUN_SCHEDULE_KIND = "agent_run_schedule"
_AGENT_RUN_SCHEDULE_BACKEND = "agent-gateway"
_AGENT_RUN_SCHEDULE_FILENAME = "agent-run-schedules.json"
_SCHEDULE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TIME_OF_DAY_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_TICKER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,31}$")
_AGENT_SCHEDULE_POLL_INTERVAL_ENV = "AGENT_GATEWAY_AGENT_RUN_SCHEDULE_POLL_SECONDS"
_AGENT_SCHEDULE_CLAIM_STALE_AFTER_SECONDS = 15 * 60
_WEB_SAFE_SCHEDULE_FIELDS = frozenset({
  "schedule_id",
  "id",
  "name",
  "kind",
  "enabled",
  "state",
  "source",
  "profile",
  "skill",
  "task",
  "ticker",
  "run_id",
  "last_run_id",
  "last_status",
  "last_error",
  "schedule_description",
  "description",
  "cadence_summary",
  "timezone",
  "last_exit_status",
  "last_run_at",
  "next_run_at",
  "created_at",
  "updated_at",
  "owned_by_current_user",
  "editable",
  "can_edit",
  "can_delete",
  "can_enable",
  "can_disable",
  "can_run_now",
  "cadence",
  "dispatch",
})


def _utc_now() -> datetime:
  return datetime.now(timezone.utc)


def _iso_from_datetime(value: datetime) -> str:
  return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc_datetime(value: Any) -> datetime | None:
  if not isinstance(value, str) or not value.strip():
    return None
  normalized = value.strip()
  if normalized.endswith("Z"):
    normalized = f"{normalized[:-1]}+00:00"
  try:
    parsed = datetime.fromisoformat(normalized)
  except ValueError:
    return None
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed.astimezone(timezone.utc)


def _user_data_dir(owner_user_id: object) -> Path:
  normalized = str(owner_user_id or "").strip()
  if not normalized:
    raise ValueError("user_id is required for per-user store access")
  candidate = Path(normalized)
  if candidate.is_absolute() or any(
    part in {"", ".", ".."} for part in candidate.parts
  ):
    raise ValueError(
      f"invalid user_id for filesystem-backed storage: {normalized!r}"
    )
  return user_data_dir() / "users" / normalized


def schedule_store_path_for(owner_user_id: object) -> Path:
  return _user_data_dir(owner_user_id) / _AGENT_RUN_SCHEDULE_FILENAME


def schedule_store_for(owner_user_id: object) -> "AgentRunScheduleStore":
  return AgentRunScheduleStore(schedule_store_path_for(owner_user_id))


def agent_run_schedule_users_root() -> Path:
  return _user_data_dir("schedule-store-root-probe").parent


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


class AgentRunScheduleCadence(BaseModel):
  model_config = ConfigDict(extra="forbid")

  type: Literal["daily", "weekly", "monthly"]
  time_of_day: StrictStr
  days_of_week: list[StrictInt] | None = None
  days_of_month: list[StrictInt] | None = None

  @field_validator("time_of_day")
  @classmethod
  def _require_time_of_day(cls, value: str) -> str:
    if _TIME_OF_DAY_RE.fullmatch(value) is None:
      raise ValueError("time_of_day must use 24-hour HH:MM format")
    return value

  @field_validator("days_of_week")
  @classmethod
  def _validate_days_of_week(cls, value: list[int] | None) -> list[int] | None:
    if value is None:
      return value
    if not value:
      raise ValueError("days_of_week must contain at least one day")
    if len(value) > 7 or len(set(value)) != len(value):
      raise ValueError("days_of_week must contain unique ISO weekday values")
    if any(day < 1 or day > 7 for day in value):
      raise ValueError("days_of_week values must be between 1 and 7")
    return value

  @field_validator("days_of_month")
  @classmethod
  def _validate_days_of_month(cls, value: list[int] | None) -> list[int] | None:
    if value is None:
      return value
    if not value:
      raise ValueError("days_of_month must contain at least one day")
    if len(value) > 31 or len(set(value)) != len(value):
      raise ValueError("days_of_month must contain unique month-day values")
    if any(day < 1 or day > 31 for day in value):
      raise ValueError("days_of_month values must be between 1 and 31")
    return value

  @model_validator(mode="after")
  def _require_matching_day_fields(self) -> "AgentRunScheduleCadence":
    if self.type == "daily":
      if self.days_of_week is not None or self.days_of_month is not None:
        raise ValueError("daily cadence must not include days_of_week or days_of_month")
    elif self.type == "weekly":
      if self.days_of_week is None:
        raise ValueError("weekly cadence requires days_of_week")
      if self.days_of_month is not None:
        raise ValueError("weekly cadence must not include days_of_month")
    elif self.type == "monthly":
      if self.days_of_month is None:
        raise ValueError("monthly cadence requires days_of_month")
      if self.days_of_week is not None:
        raise ValueError("monthly cadence must not include days_of_week")
    return self


class AgentRunScheduleDispatch(BaseModel):
  model_config = ConfigDict(extra="forbid")

  kind: Literal["autonomous"]
  profile: StrictStr
  mode: Literal["task", "skill"]
  skill: StrictStr | None = None
  ticker: StrictStr | None = None
  task: StrictStr | None = None
  context: StrictStr | None = None
  dispatch_scope: DispatchScope | None = None

  @field_validator("profile", "skill")
  @classmethod
  def _validate_short_token(cls, value: str | None, info: ValidationInfo) -> str | None:
    if value is None:
      return value
    if not value.strip():
      raise ValueError(f"{info.field_name} must be omitted, null, or a non-empty string")
    if len(value) > 128:
      raise ValueError(f"{info.field_name} must be 128 characters or fewer")
    return value

  @field_validator("ticker")
  @classmethod
  def _validate_ticker(cls, value: str | None) -> str | None:
    if value is None:
      return value
    if _TICKER_RE.fullmatch(value.strip()) is None:
      raise ValueError("ticker must be a non-empty ticker token")
    return value

  @field_validator("task")
  @classmethod
  def _validate_task(cls, value: str | None) -> str | None:
    if value is None:
      return value
    if not value.strip():
      raise ValueError("task must be omitted, null, or a non-empty string")
    if len(value) > 2000:
      raise ValueError("task must be 2000 characters or fewer")
    return value

  @field_validator("context")
  @classmethod
  def _validate_context(cls, value: str | None) -> str | None:
    if value is None:
      return value
    if not value.strip():
      raise ValueError("context must be omitted, null, or a non-empty string")
    if len(value) > 8000:
      raise ValueError("context must be 8000 characters or fewer")
    return value

  @model_validator(mode="after")
  def _require_mode_payload(self) -> "AgentRunScheduleDispatch":
    if self.mode == "task" and not (self.task or "").strip():
      raise ValueError("task-mode schedule dispatch requires task")
    if self.mode == "skill" and not (self.skill or "").strip():
      raise ValueError("skill-mode schedule dispatch requires skill")
    return self


class CreateAgentRunScheduleRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  kind: Literal["agent_run_schedule"]
  name: StrictStr
  enabled: StrictBool | None = None
  timezone: StrictStr
  cadence: AgentRunScheduleCadence
  dispatch: AgentRunScheduleDispatch
  request_id: StrictStr | None = None

  @field_validator("enabled", mode="before")
  @classmethod
  def _reject_null_enabled(cls, value: Any) -> Any:
    if value is None:
      raise ValueError("enabled must be omitted or a boolean")
    return value

  @field_validator("request_id", mode="before")
  @classmethod
  def _reject_null_request_id(cls, value: Any) -> Any:
    if value is None:
      raise ValueError("request_id must be omitted or a safe non-empty identifier")
    return value

  @field_validator("name", "request_id")
  @classmethod
  def _validate_schedule_token(cls, value: str | None, info: ValidationInfo) -> str | None:
    if value is None:
      return value
    if _SCHEDULE_NAME_RE.fullmatch(value) is None:
      raise ValueError(f"{info.field_name} must be a safe non-empty identifier")
    return value

  @field_validator("timezone")
  @classmethod
  def _validate_timezone(cls, value: str) -> str:
    if not value.strip() or len(value) > 64:
      raise ValueError("timezone must be a non-empty IANA timezone")
    try:
      ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
      raise ValueError("timezone must be a valid IANA timezone") from exc
    return value


class UpdateAgentRunScheduleRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  kind: Literal["agent_run_schedule"] | None = None
  name: StrictStr | None = None
  enabled: StrictBool | None = None
  timezone: StrictStr | None = None
  cadence: AgentRunScheduleCadence | None = None
  dispatch: AgentRunScheduleDispatch | None = None
  request_id: StrictStr | None = None

  @field_validator(
    "kind",
    "name",
    "enabled",
    "timezone",
    "cadence",
    "dispatch",
    "request_id",
    mode="before",
  )
  @classmethod
  def _reject_null_update_field(cls, value: Any, info: ValidationInfo) -> Any:
    if value is None:
      raise ValueError(f"{info.field_name} must be omitted or a concrete value")
    return value

  @field_validator("name", "request_id")
  @classmethod
  def _validate_schedule_token(cls, value: str | None, info: ValidationInfo) -> str | None:
    if value is None:
      return value
    if _SCHEDULE_NAME_RE.fullmatch(value) is None:
      raise ValueError(f"{info.field_name} must be a safe non-empty identifier")
    return value

  @field_validator("timezone")
  @classmethod
  def _validate_timezone(cls, value: str | None) -> str | None:
    if value is None:
      return value
    return CreateAgentRunScheduleRequest._validate_timezone(value)

  @model_validator(mode="after")
  def _require_update_field(self) -> "UpdateAgentRunScheduleRequest":
    if not any(
      value is not None
      for value in (self.name, self.enabled, self.timezone, self.cadence, self.dispatch)
    ):
      raise ValueError("schedule update must include at least one mutable field")
    return self


class BrowserSafeScheduleResponse(BaseModel):
  model_config = ConfigDict(extra="allow")

  schedule_id: str | None = None
  id: str | None = None
  name: str | None = None
  kind: str | None = None
  source: str | None = None
  enabled: bool | None = None
  schedule_description: str | None = None
  last_run_at: str | None = None
  next_run_at: str | None = None
  owned_by_current_user: bool | None = None
  editable: bool | None = None
  can_edit: bool | None = None
  can_delete: bool | None = None
  can_enable: bool | None = None
  can_disable: bool | None = None
  can_run_now: bool | None = None

  @model_validator(mode="after")
  def _require_schedule_identity(self) -> "BrowserSafeScheduleResponse":
    for value in (self.schedule_id, self.id, self.name):
      if isinstance(value, str) and value.strip():
        return self
    raise ValueError("schedule_id, id, or name must be a non-empty string")


ScheduleResponse = Union[LaunchdScheduleResponse, JobsMcpScheduleResponse, BrowserSafeScheduleResponse]


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
  source: ScheduleSource | AgentScheduleSource


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
  Union[CreateLaunchdScheduleRequest, CreateJobsMcpScheduleRequest, CreateAgentRunScheduleRequest],
  Body(),
]


class ScheduleEnabledRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  enabled: StrictBool


def _hhmm(value: str) -> tuple[int, int]:
  if _TIME_OF_DAY_RE.fullmatch(value) is None:
    raise ValueError("time_of_day must use 24-hour HH:MM format")
  hour, minute = value.split(":", 1)
  return int(hour), int(minute)


def _local_candidate(day: date, time_of_day: str, zone: ZoneInfo) -> datetime:
  hour, minute = _hhmm(time_of_day)
  return datetime.combine(day, time(hour=hour, minute=minute), tzinfo=zone)


def _next_month(first_of_month: date) -> date:
  year = first_of_month.year + (1 if first_of_month.month == 12 else 0)
  month = 1 if first_of_month.month == 12 else first_of_month.month + 1
  return date(year, month, 1)


def compute_agent_schedule_next_run_at(
  cadence: AgentRunScheduleCadence | dict[str, Any],
  timezone_name: str,
  *,
  after: datetime | None = None,
) -> str:
  parsed_cadence = (
    cadence if isinstance(cadence, AgentRunScheduleCadence) else AgentRunScheduleCadence.model_validate(cadence)
  )
  zone = ZoneInfo(timezone_name)
  reference_utc = (after or _utc_now()).astimezone(timezone.utc)
  reference_local = reference_utc.astimezone(zone)

  if parsed_cadence.type == "daily":
    for offset in range(0, 3):
      candidate = _local_candidate(reference_local.date() + timedelta(days=offset), parsed_cadence.time_of_day, zone)
      if candidate > reference_local:
        return _iso_from_datetime(candidate)

  if parsed_cadence.type == "weekly":
    days = set(parsed_cadence.days_of_week or ())
    for offset in range(0, 15):
      candidate_day = reference_local.date() + timedelta(days=offset)
      if candidate_day.isoweekday() not in days:
        continue
      candidate = _local_candidate(candidate_day, parsed_cadence.time_of_day, zone)
      if candidate > reference_local:
        return _iso_from_datetime(candidate)

  if parsed_cadence.type == "monthly":
    days = sorted(parsed_cadence.days_of_month or ())
    month_cursor = date(reference_local.year, reference_local.month, 1)
    for _ in range(0, 24):
      for day_number in days:
        try:
          candidate_day = date(month_cursor.year, month_cursor.month, day_number)
        except ValueError:
          continue
        candidate = _local_candidate(candidate_day, parsed_cadence.time_of_day, zone)
        if candidate > reference_local:
          return _iso_from_datetime(candidate)
      month_cursor = _next_month(month_cursor)

  raise ValueError("Unable to compute next schedule run")


def _schedule_owner_user_id(session: GatewaySession) -> str:
  owner_user_id = getattr(session, "owner_user_id", None)
  if isinstance(owner_user_id, str) and owner_user_id.strip():
    return owner_user_id.strip()
  try:
    risk_user_id = int(getattr(session, "risk_user_id", 0) or 0)
  except (TypeError, ValueError):
    risk_user_id = 0
  if risk_user_id > 0:
    return str(risk_user_id)
  return str(session.user_id)


def _capability_fields(*, owned: bool, enabled: bool) -> dict[str, bool]:
  return {
    "owned_by_current_user": bool(owned),
    "editable": bool(owned),
    "can_edit": bool(owned),
    "can_delete": bool(owned),
    "can_enable": bool(owned and not enabled),
    "can_disable": bool(owned and enabled),
    "can_run_now": bool(owned),
  }


def _canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _request_body_hash(value: dict[str, Any]) -> str:
  return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class ScheduleStoreUnreadableError(RuntimeError):
  def __init__(self, path: Path, cause: BaseException) -> None:
    self.path = path
    self.cause = cause
    super().__init__(
      f"Schedule store at {path} is unreadable and was left untouched: {cause}"
    )


class AgentRunScheduleStore:
  def __init__(
    self,
    path: Path,
    *,
    clock: Callable[[], float] | None = None,
  ) -> None:
    self.path = path
    self._clock = clock or monotonic_time.monotonic
    self._store_unreadable = False
    self._last_unreadable_summary_at: float | None = None

  def _empty_payload(self) -> dict[str, Any]:
    return {"version": 1, "schedules": [], "idempotency": {}}

  def _mark_unreadable(self, cause: BaseException) -> None:
    now = self._clock()
    if not self._store_unreadable:
      logger.error("Schedule store became unreadable at %s: %s", self.path, cause)
      self._store_unreadable = True
      self._last_unreadable_summary_at = now
      return
    last_summary_at = self._last_unreadable_summary_at
    if last_summary_at is None or now - last_summary_at >= 60.0:
      logger.warning("Schedule store remains unreadable at %s: %s", self.path, cause)
      self._last_unreadable_summary_at = now

  def _mark_readable(self) -> None:
    if self._store_unreadable:
      logger.info("Schedule store recovered at %s", self.path)
      self._store_unreadable = False
      self._last_unreadable_summary_at = None

  def _load(self) -> dict[str, Any]:
    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
      seen: set[str] = set()
      for key, _value in pairs:
        if key in seen:
          raise ValueError(f"duplicate JSON object member: {key!r}")
        seen.add(key)
      return dict(pairs)

    try:
      payload = json.loads(
        self.path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
      )
      if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
      if "version" not in payload:
        raise ValueError("version is required")
      version = payload["version"]
      if type(version) is not int or version != 1:
        raise ValueError("version must be the integer 1")
      if "schedules" not in payload:
        raise ValueError("schedules is required")
      schedules = payload["schedules"]
      if not isinstance(schedules, list):
        raise ValueError("schedules must be a list")
      if "idempotency" not in payload:
        raise ValueError("idempotency is required")
      if not isinstance(payload["idempotency"], dict):
        raise ValueError("idempotency must be an object")
      if any(not isinstance(item, dict) for item in schedules):
        raise ValueError("every schedules element must be an object")
    except FileNotFoundError:
      self._mark_readable()
      return self._empty_payload()
    except (OSError, ValueError) as exc:
      self._mark_unreadable(exc)
      raise ScheduleStoreUnreadableError(self.path, exc) from exc
    self._mark_readable()
    return payload

  def _save(self, payload: dict[str, Any]) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = self.path.with_name(f"{self.path.name}.{secrets.token_hex(8)}.tmp")
    tmp_path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(tmp_path, self.path)

  def _all_records(self) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in self._load().get("schedules") or []:
      if isinstance(item, dict) and item.get("kind") == _AGENT_RUN_SCHEDULE_KIND:
        records.append(dict(item))
    return records

  def list_for_owner(self, owner_user_id: str) -> list[dict[str, Any]]:
    owner = str(owner_user_id)
    return [record for record in self._all_records() if str(record.get("owner_user_id") or "") == owner]

  def get_for_owner(self, owner_user_id: str, identifier: str) -> dict[str, Any] | None:
    normalized = str(identifier or "").strip()
    if not normalized:
      return None
    for record in self.list_for_owner(owner_user_id):
      if normalized in {str(record.get("schedule_id") or ""), str(record.get("name") or "")}:
        return record
    return None

  def create(self, payload: CreateAgentRunScheduleRequest, session: GatewaySession) -> dict[str, Any]:
    owner_user_id = _schedule_owner_user_id(session)
    dispatch_role = require_exact_role(getattr(session, "role", None))
    request_body = payload.model_dump(mode="json", exclude_unset=True)
    request_hash = _request_body_hash(request_body)
    idempotency_key = (
      f"{owner_user_id}:{payload.request_id}"
      if isinstance(payload.request_id, str) and payload.request_id.strip()
      else None
    )
    stored = self._load()
    idempotency = dict(stored.get("idempotency") or {})
    if idempotency_key is not None and idempotency_key in idempotency:
      entry = idempotency.get(idempotency_key)
      if isinstance(entry, dict) and entry.get("body_hash") == request_hash:
        existing = self.get_for_owner(owner_user_id, str(entry.get("schedule_id") or ""))
        if existing is not None:
          return existing
      raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
          "error": "schedule_request_id_conflict",
          "message": "request_id already exists for a different schedule body.",
        },
      )

    schedules = [item for item in stored.get("schedules") or [] if isinstance(item, dict)]
    if any(
      str(item.get("owner_user_id") or "") == owner_user_id and str(item.get("name") or "") == payload.name
      for item in schedules
    ):
      raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Schedule name already exists")

    now = _utc_now()
    enabled = payload.enabled if payload.enabled is not None else True
    cadence_payload = payload.cadence.model_dump(mode="json", exclude_none=True)
    record = {
      "schedule_id": f"sched_{secrets.token_hex(8)}",
      "id": None,
      "name": payload.name,
      "kind": _AGENT_RUN_SCHEDULE_KIND,
      "source": _AGENT_RUN_SCHEDULE_BACKEND,
      "enabled": enabled,
      "timezone": payload.timezone,
      "cadence": cadence_payload,
      "dispatch": payload.dispatch.model_dump(mode="json", exclude_none=True),
      "owner_user_id": owner_user_id,
      "dispatch_role": dispatch_role,
      "dispatch_revision": 1,
      "dispatch_authored_by": owner_user_id,
      "raw_user_id": getattr(session, "raw_user_id", None) or session.user_id,
      "user_email": session.user_email,
      "user_slug": getattr(session, "user_slug", None),
      "risk_user_id": int(getattr(session, "risk_user_id", 0) or 0),
      "user_aliases": list(getattr(session, "user_aliases", ()) or (owner_user_id,)),
      "identity_status": getattr(session, "identity_status", None),
      "channel": _normalize_channel(getattr(session, "channel", None)) or "web",
      "created_by": owner_user_id,
      "updated_by": owner_user_id,
      "created_at": _iso_from_datetime(now),
      "updated_at": _iso_from_datetime(now),
      "next_run_at": compute_agent_schedule_next_run_at(payload.cadence, payload.timezone, after=now),
      "last_run_id": None,
      "last_run_at": None,
      "last_status": None,
    }
    record.update(_capability_fields(owned=True, enabled=enabled))
    schedules.append(record)
    if idempotency_key is not None:
      idempotency[idempotency_key] = {
        "schedule_id": record["schedule_id"],
        "body_hash": request_hash,
      }
    stored["schedules"] = schedules
    stored["idempotency"] = idempotency
    self._save(stored)
    return record

  def update(
    self,
    owner_user_id: str,
    identifier: str,
    payload: UpdateAgentRunScheduleRequest,
    *,
    updated_by: str,
    live_role: str,
  ) -> dict[str, Any]:
    exact_live_role = require_exact_role(live_role)
    stored = self._load()
    schedules = [item for item in stored.get("schedules") or [] if isinstance(item, dict)]
    target_index: int | None = None
    for index, item in enumerate(schedules):
      if str(item.get("owner_user_id") or "") != owner_user_id:
        continue
      if identifier in {str(item.get("schedule_id") or ""), str(item.get("name") or "")}:
        target_index = index
        break
    if target_index is None:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")

    record = dict(schedules[target_index])
    if payload.name is not None and payload.name != record.get("name"):
      if any(
        index != target_index
        and str(item.get("owner_user_id") or "") == owner_user_id
        and str(item.get("name") or "") == payload.name
        for index, item in enumerate(schedules)
      ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Schedule name already exists")
      record["name"] = payload.name
    if payload.timezone is not None:
      record["timezone"] = payload.timezone
    if payload.cadence is not None:
      record["cadence"] = payload.cadence.model_dump(mode="json", exclude_none=True)
    if payload.dispatch is not None:
      record["dispatch"] = payload.dispatch.model_dump(mode="json", exclude_none=True)
      revision = record.get("dispatch_revision")
      record["dispatch_revision"] = (
        revision + 1
        if type(revision) is int and revision >= 1
        else 1
      )
      record["dispatch_role"] = exact_live_role
      record["dispatch_authored_by"] = owner_user_id
    if (
      payload.enabled is True
      and record.get("enabled") is False
      and record.get("dispatch_role") != exact_live_role
    ):
      raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
          "error": "schedule_dispatch_role_mismatch",
          "message": "Re-submit the dispatch under the current role before enabling it.",
        },
      )
    if payload.enabled is not None:
      record["enabled"] = payload.enabled
    record["updated_by"] = updated_by
    record["updated_at"] = _iso_from_datetime(_utc_now())
    record["next_run_at"] = compute_agent_schedule_next_run_at(record["cadence"], str(record["timezone"]))
    record.update(_capability_fields(owned=True, enabled=bool(record.get("enabled"))))
    schedules[target_index] = record
    stored["schedules"] = schedules
    self._save(stored)
    return record

  def set_enabled(
    self,
    owner_user_id: str,
    identifier: str,
    *,
    enabled: bool,
    updated_by: str,
    live_role: str,
  ) -> dict[str, Any]:
    return self.update(
      owner_user_id,
      identifier,
      UpdateAgentRunScheduleRequest(kind=_AGENT_RUN_SCHEDULE_KIND, enabled=enabled),
      updated_by=updated_by,
      live_role=live_role,
    )

  def delete(self, owner_user_id: str, identifier: str) -> dict[str, Any]:
    stored = self._load()
    schedules = [item for item in stored.get("schedules") or [] if isinstance(item, dict)]
    remaining: list[dict[str, Any]] = []
    deleted: dict[str, Any] | None = None
    for item in schedules:
      if (
        deleted is None
        and str(item.get("owner_user_id") or "") == owner_user_id
        and identifier in {str(item.get("schedule_id") or ""), str(item.get("name") or "")}
      ):
        deleted = dict(item)
        continue
      remaining.append(item)
    if deleted is None:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
    stored["schedules"] = remaining
    self._save(stored)
    return deleted

  def due_records(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
    reference = (now or _utc_now()).astimezone(timezone.utc)
    due: list[dict[str, Any]] = []
    for record in self._all_records():
      if not bool(record.get("enabled")):
        continue
      claim_id = str(record.get("running_claim_id") or "")
      claimed_at = _parse_utc_datetime(record.get("running_claimed_at"))
      if (
        claim_id
        and claimed_at is not None
        and reference - claimed_at < timedelta(seconds=_AGENT_SCHEDULE_CLAIM_STALE_AFTER_SECONDS)
      ):
        continue
      next_run = _parse_utc_datetime(record.get("next_run_at"))
      if next_run is not None and next_run <= reference:
        due.append(record)
    return due

  def claim_due_record(
    self,
    schedule_id: str,
    *,
    claim_id: str,
    now: datetime | None = None,
  ) -> dict[str, Any] | None:
    stored = self._load()
    schedules = [item for item in stored.get("schedules") or [] if isinstance(item, dict)]
    reference = (now or _utc_now()).astimezone(timezone.utc)
    for index, item in enumerate(schedules):
      if str(item.get("schedule_id") or "") != schedule_id:
        continue
      record = dict(item)
      if not bool(record.get("enabled")):
        return None
      next_run = _parse_utc_datetime(record.get("next_run_at"))
      if next_run is None or next_run > reference:
        return None
      existing_claim_id = str(record.get("running_claim_id") or "")
      claimed_at = _parse_utc_datetime(record.get("running_claimed_at"))
      if (
        existing_claim_id
        and claimed_at is not None
        and reference - claimed_at < timedelta(seconds=_AGENT_SCHEDULE_CLAIM_STALE_AFTER_SECONDS)
      ):
        return None
      record["running_claim_id"] = claim_id
      record["running_claimed_at"] = _iso_from_datetime(reference)
      record["last_status"] = "launching"
      record["updated_at"] = _iso_from_datetime(reference)
      record.update(_capability_fields(owned=True, enabled=True))
      schedules[index] = record
      stored["schedules"] = schedules
      self._save(stored)
      return record
    return None

  def record_fire_result(
    self,
    schedule_id: str,
    *,
    run_id: str | None,
    status_text: str,
    error: str | None = None,
    fired_at: datetime | None = None,
    claim_id: str | None = None,
    advance_next_run: bool = True,
  ) -> dict[str, Any] | None:
    stored = self._load()
    schedules = [item for item in stored.get("schedules") or [] if isinstance(item, dict)]
    timestamp = fired_at or _utc_now()
    for index, item in enumerate(schedules):
      if str(item.get("schedule_id") or "") != schedule_id:
        continue
      record = dict(item)
      if claim_id is not None and str(record.get("running_claim_id") or "") != claim_id:
        return None
      record.pop("running_claim_id", None)
      record.pop("running_claimed_at", None)
      record["last_run_id"] = run_id
      record["last_run_at"] = _iso_from_datetime(timestamp)
      record["last_status"] = status_text
      if error:
        record["last_error"] = error
      else:
        record.pop("last_error", None)
      if advance_next_run:
        try:
          record["next_run_at"] = compute_agent_schedule_next_run_at(
            record["cadence"],
            str(record["timezone"]),
            after=timestamp + timedelta(seconds=1),
          )
        except (KeyError, ValueError, ZoneInfoNotFoundError):
          record["enabled"] = False
          record["last_status"] = "failed"
          record["last_error"] = error or "Unable to compute next run"
      record["updated_at"] = _iso_from_datetime(timestamp)
      record.update(_capability_fields(owned=True, enabled=bool(record.get("enabled"))))
      schedules[index] = record
      stored["schedules"] = schedules
      self._save(stored)
      return record
    return None


class AgentRunScheduleRunner:
  def __init__(
    self,
    *,
    store_for: Callable[[object], AgentRunScheduleStore] = schedule_store_for,
    users_root: Path | None = None,
    autonomous_registry: Any | None,
    user_event_bus_factory: Callable[[], Any | None] | None = None,
    poll_interval_seconds: float | None = None,
  ) -> None:
    self.store_for = store_for
    self.users_root = users_root or agent_run_schedule_users_root()
    self._stores_by_path: dict[Path, AgentRunScheduleStore] = {}
    self.autonomous_registry = autonomous_registry
    self.user_event_bus_factory = user_event_bus_factory
    self.poll_interval_seconds = (
      poll_interval_seconds
      if poll_interval_seconds is not None
      else self._configured_poll_interval_seconds()
    )
    self._task: asyncio.Task[Any] | None = None
    self._stopped = asyncio.Event()

  def _configured_poll_interval_seconds(self) -> float:
    raw = os.getenv(_AGENT_SCHEDULE_POLL_INTERVAL_ENV, "").strip()
    if not raw:
      return 60.0
    try:
      interval = float(raw)
    except ValueError:
      return 60.0
    return max(0.0, interval)

  def start(self) -> None:
    if self.poll_interval_seconds <= 0:
      return
    if self._task is not None and not self._task.done():
      return
    self._stopped.clear()
    self._task = asyncio.create_task(self._run_loop())

  async def shutdown(self) -> None:
    self._stopped.set()
    if self._task is None:
      return
    self._task.cancel()
    await asyncio.gather(self._task, return_exceptions=True)
    self._task = None

  async def _run_loop(self) -> None:
    while not self._stopped.is_set():
      await self.fire_due()
      try:
        await asyncio.wait_for(self._stopped.wait(), timeout=self.poll_interval_seconds)
      except asyncio.TimeoutError:
        continue

  async def fire_due(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
    if self.autonomous_registry is None:
      return []
    results: list[dict[str, Any]] = []
    for store in self._existing_stores():
      try:
        due_records = store.due_records(now=now)
      except ScheduleStoreUnreadableError:
        continue
      for record in due_records:
        schedule_id = str(record.get("schedule_id") or "")
        if not schedule_id:
          continue
        claim_id = f"claim_{secrets.token_hex(8)}"
        try:
          claimed = store.claim_due_record(
            schedule_id,
            claim_id=claim_id,
            now=now,
          )
        except ScheduleStoreUnreadableError:
          continue
        if claimed is None:
          continue
        result = await self._fire_record(
          store,
          claimed,
          role=claimed.get("dispatch_role"),
          now=now,
          claim_id=claim_id,
        )
        if result is not None:
          results.append(result)
    return results

  def _existing_stores(self) -> list[AgentRunScheduleStore]:
    stores: list[AgentRunScheduleStore] = []
    for path in sorted(
      self.users_root.glob(f"*/{_AGENT_RUN_SCHEDULE_FILENAME}"),
      key=lambda candidate: candidate.as_posix(),
    ):
      if not path.is_file():
        continue
      canonical = path.resolve(strict=False)
      store = self._stores_by_path.get(canonical)
      if store is None:
        try:
          resolved = self.store_for(path.parent.name)
        except (TypeError, ValueError):
          logger.error("Skipping schedule store with invalid owner path: %s", path)
          continue
        if resolved.path.resolve(strict=False) != canonical:
          logger.error("Skipping schedule store outside canonical owner path: %s", path)
          continue
        store = resolved
        self._stores_by_path[canonical] = store
      stores.append(store)
    return stores

  async def fire_record_now(
    self,
    record: dict[str, Any],
    *,
    live_role: str,
    now: datetime | None = None,
  ) -> dict[str, Any] | None:
    exact_live_role = require_exact_role(live_role)
    store = self.store_for(record.get("owner_user_id"))
    return await self._fire_record(
      store,
      record,
      role=exact_live_role,
      now=now,
      claim_id=None,
      advance_next_run=False,
    )

  async def _fire_record(
    self,
    store: AgentRunScheduleStore,
    record: dict[str, Any],
    *,
    role: object,
    now: datetime | None = None,
    claim_id: str | None = None,
    advance_next_run: bool = True,
  ) -> dict[str, Any] | None:
    registry = self.autonomous_registry
    schedule_id = str(record.get("schedule_id") or "")
    if not schedule_id:
      return None
    try:
      dispatch_role = require_exact_role(role)
    except ValueError:
      # A stored schedule that predates the role plane carries no dispatch_role.
      # Owner is the only role that can own a schedule store, so adopt it and
      # run the work loudly rather than failing the fire and advancing
      # next_run — that silently retires real queued work forever.
      logger.warning(
        "Schedule %s has no attested dispatch role (%r); dispatching as owner",
        schedule_id,
        role,
      )
      dispatch_role = "owner"
    dispatch = record.get("dispatch")
    if not isinstance(dispatch, dict):
      result = {"schedule_id": schedule_id, "status": "failed", "error": "Schedule dispatch is missing"}
      try:
        store.record_fire_result(
          schedule_id,
          run_id=None,
          status_text="failed",
          error="Schedule dispatch is missing",
          fired_at=now,
          claim_id=claim_id,
          advance_next_run=advance_next_run,
        )
      except ScheduleStoreUnreadableError:
        result["result_persisted"] = False
      return result

    try:
      dispatch_scope = dispatch.get("dispatch_scope")
      if not isinstance(dispatch_scope, dict):
        dispatch_scope = (
          record.get("dispatch_scope")
          if isinstance(record.get("dispatch_scope"), dict)
          else None
        )
      registry.set_user_event_bus(self.user_event_bus_factory() if self.user_event_bus_factory else None)
      start_payload = await registry.start(
        role=dispatch_role,
        profile=str(dispatch.get("profile") or ""),
        mode=str(dispatch.get("mode") or ""),
        task=(
          dispatch.get("task")
          if isinstance(dispatch.get("task"), str)
          else None
        ),
        skill=(
          dispatch.get("skill")
          if isinstance(dispatch.get("skill"), str)
          else None
        ),
        context=(
          dispatch.get("context")
          if isinstance(dispatch.get("context"), str)
          else None
        ),
        ticker=(
          dispatch.get("ticker")
          if isinstance(dispatch.get("ticker"), str)
          else None
        ),
        channel=str(record.get("channel") or "web"),
        user_id=str(record.get("raw_user_id") or record.get("owner_user_id") or ""),
        user_email=record.get("user_email") if isinstance(record.get("user_email"), str) else None,
        owner_user_id=str(record.get("owner_user_id") or ""),
        user_slug=record.get("user_slug") if isinstance(record.get("user_slug"), str) else None,
        risk_user_id=record.get("risk_user_id") if isinstance(record.get("risk_user_id"), int) else None,
        user_aliases=record.get("user_aliases") if isinstance(record.get("user_aliases"), list) else None,
        identity_status=record.get("identity_status") if isinstance(record.get("identity_status"), str) else None,
        dispatch_scope=dispatch_scope,
        schedule_id=schedule_id,
        schedule_name=str(record.get("name") or ""),
      )
    except Exception as exc:
      result = {"schedule_id": schedule_id, "status": "failed", "error": str(exc)}
      try:
        store.record_fire_result(
          schedule_id,
          run_id=None,
          status_text="failed",
          error=str(exc),
          fired_at=now,
          claim_id=claim_id,
          advance_next_run=advance_next_run,
        )
      except ScheduleStoreUnreadableError:
        result["result_persisted"] = False
      return result

    run_id = str(start_payload.get("run_id") or start_payload.get("task_id") or "")
    result = {"schedule_id": schedule_id, "status": "started", "run_id": run_id}
    try:
      store.record_fire_result(
        schedule_id,
        run_id=run_id,
        status_text="started",
        fired_at=now,
        claim_id=claim_id,
        advance_next_run=advance_next_run,
      )
    except ScheduleStoreUnreadableError:
      result["result_persisted"] = False
    return result


def _require_bearer_session(request: Request, auth: AuthManager) -> GatewaySession:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  session = auth.verify_token(token)
  return session


def _normalize_channel(channel: str | None) -> str | None:
  if not isinstance(channel, str):
    return None
  normalized = channel.strip().lower()
  return normalized or None


def _is_web_session(session: GatewaySession) -> bool:
  return _normalize_channel(getattr(session, "channel", None)) == "web"


def _raw_web_schedule_write_forbidden() -> HTTPException:
  return HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail={
      "error": "web_control_raw_schedule_forbidden",
      "message": "Web Agent Control cannot use raw launchd or jobs-mcp schedule writes.",
    },
  )


def _launchd_schedule_creation_forbidden() -> HTTPException:
  return HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail={
      "error": "launchd_schedule_creation_forbidden",
      "message": (
        "The Agent Control API cannot create launchd schedules. "
        "Install approved launchd jobs through deployment-managed templates."
      ),
    },
  )


def _web_dev_schedule_dispatch_forbidden() -> HTTPException:
  return HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail={
      "error": "web_control_dev_dispatch_forbidden",
      "message": "Web Agent Control schedules require skill mode.",
    },
  )


def _require_agent_schedule_dispatch_allowed(
  session: GatewaySession,
  dispatch: AgentRunScheduleDispatch,
) -> None:
  if not _is_web_session(session):
    return
  if dispatch.mode != "skill":
    raise _web_dev_schedule_dispatch_forbidden()


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
  if raw.get("kind") == "agent_run_schedule":
    return BrowserSafeScheduleResponse(**_project_schedule_for_web(raw))
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


def _show_schedule_or_none(name: str, *, source: ScheduleSource | None = None) -> ScheduleResponse | None:
  try:
    return _show_schedule(name, source=source)
  except HTTPException as exc:
    if exc.status_code == status.HTTP_404_NOT_FOUND:
      return None
    raise


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
    schedules.append(_normalize_schedule(raw))
  return schedules


def _schedule_to_dict(schedule: ScheduleResponse | dict[str, Any]) -> dict[str, Any]:
  if isinstance(schedule, BaseModel):
    return schedule.model_dump(mode="json")
  return dict(schedule)


def _project_schedule_for_web(schedule: ScheduleResponse | dict[str, Any]) -> dict[str, Any]:
  raw = _schedule_to_dict(schedule)
  projected = {
    key: value
    for key, value in raw.items()
    if key in _WEB_SAFE_SCHEDULE_FIELDS and value is not None
  }
  source = raw.get("source")
  if source in {"launchd", "jobs-mcp"}:
    projected.setdefault("kind", "operator_schedule")
    projected["source"] = source
    projected["owned_by_current_user"] = False
    projected["editable"] = False
    projected["can_edit"] = False
    projected["can_delete"] = False
    projected["can_enable"] = False
    projected["can_disable"] = False
    projected["can_run_now"] = False
  return projected


def build_schedules_router(
  *,
  auth: AuthManager,
  agent_schedule_store_for: Callable[[object], AgentRunScheduleStore] | None = None,
  agent_schedule_runner: AgentRunScheduleRunner | None = None,
  dispatch_scope_validator: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> APIRouter:
  router = APIRouter(prefix="/schedules")
  store_for = agent_schedule_store_for or schedule_store_for
  operator_global_description = (
    "Owner-only operator-global launchd and jobs-mcp schedule management."
  )

  def _can_access_operator_schedules(session: GatewaySession) -> bool:
    return require_exact_role(getattr(session, "role", None)) == "owner"

  def _operator_schedule_not_found(name: str) -> HTTPException:
    return HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Schedule not found: {name}",
    )

  async def _validated_agent_schedule_dispatch(
    dispatch: AgentRunScheduleDispatch,
    *,
    session: GatewaySession,
  ) -> AgentRunScheduleDispatch:
    scope = dispatch.dispatch_scope
    if scope is None or dispatch_scope_validator is None:
      return dispatch
    payload = scope.model_dump()
    try:
      validation_result = dispatch_scope_validator(session, dict(payload))
      if inspect.isawaitable(validation_result):
        validation_result = await validation_result
    except HTTPException:
      raise
    except ValueError as exc:
      raise HTTPException(
        status_code=422,
        detail={
          "error": "dispatch_scope_validation_failed",
          "message": str(exc) or "Selected dispatch scope is not valid.",
        },
      ) from exc
    except Exception as exc:
      raise HTTPException(
        status_code=422,
        detail={
          "error": "dispatch_scope_validation_failed",
          "message": "Selected dispatch scope could not be validated.",
        },
      ) from exc
    if validation_result is None:
      validation_result = payload
    if not isinstance(validation_result, dict):
      raise HTTPException(
        status_code=422,
        detail={
          "error": "dispatch_scope_validation_failed",
          "message": "Dispatch scope validator returned an invalid payload.",
        },
      )
    try:
      validated_scope = DispatchScope.model_validate(validation_result)
    except ValueError as exc:
      raise HTTPException(
        status_code=422,
        detail={
          "error": "dispatch_scope_validation_failed",
          "message": "Dispatch scope validator returned a non-redacted scope.",
        },
      ) from exc
    return dispatch.model_copy(update={"dispatch_scope": validated_scope})

  @router.get(
    "",
    response_model=SchedulesListResponse,
    description=operator_global_description,
  )
  async def list_schedules(
    request: Request,
    source: ScheduleSource | None = Query(default=None),
  ) -> SchedulesListResponse:
    session = _require_bearer_session(request, auth)
    schedules = []
    if _can_access_operator_schedules(session):
      schedules = await asyncio.to_thread(_list_schedules, source)
    if source is None:
      owner_user_id = _schedule_owner_user_id(session)
      schedules.extend(
        _normalize_schedule(record)
        for record in store_for(owner_user_id).list_for_owner(owner_user_id)
      )
    if _is_web_session(session):
      return SchedulesListResponse(schedules=[_project_schedule_for_web(schedule) for schedule in schedules])
    return SchedulesListResponse(schedules=schedules)

  @router.get(
    "/{name}",
    response_model=ScheduleResponse,
    description=operator_global_description,
  )
  async def get_schedule(request: Request, name: str) -> ScheduleResponse:
    session = _require_bearer_session(request, auth)
    owner_user_id = _schedule_owner_user_id(session)
    if not _can_access_operator_schedules(session):
      owned_schedule = store_for(owner_user_id).get_for_owner(owner_user_id, name)
      if owned_schedule is not None:
        return _project_schedule_for_web(owned_schedule)
      raise _operator_schedule_not_found(name)
    if _is_web_session(session):
      owned_schedule = store_for(owner_user_id).get_for_owner(owner_user_id, name)
      if owned_schedule is not None:
        return _project_schedule_for_web(owned_schedule)
      schedule = _show_schedule(name)
      return _project_schedule_for_web(schedule)
    raw_schedule = _show_schedule_or_none(name)
    if raw_schedule is not None:
      return raw_schedule
    owned_schedule = store_for(owner_user_id).get_for_owner(owner_user_id, name)
    if owned_schedule is not None:
      return _project_schedule_for_web(owned_schedule)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Schedule not found: {name}")

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
    session = _require_bearer_session(request, auth)
    if not _can_access_operator_schedules(session):
      raise _operator_schedule_not_found(name)
    if _is_web_session(session):
      raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
          "error": "web_control_schedule_logs_forbidden",
          "message": "Web Agent Control cannot read raw scheduler logs.",
        },
      )
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
    session = _require_bearer_session(request, auth)
    if isinstance(payload, CreateAgentRunScheduleRequest):
      dispatch = await _validated_agent_schedule_dispatch(payload.dispatch, session=session)
      payload = payload.model_copy(update={"dispatch": dispatch})
      _require_agent_schedule_dispatch_allowed(session, payload.dispatch)
      created = store_for(_schedule_owner_user_id(session)).create(payload, session)
      return ScheduleEnvelopeResponse(schedule=_project_schedule_for_web(created))

    if not _can_access_operator_schedules(session):
      raise _operator_schedule_not_found(payload.name)
    if _is_web_session(session):
      raise _raw_web_schedule_write_forbidden()
    if payload.source == "launchd":
      raise _launchd_schedule_creation_forbidden()

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

  @router.patch(
    "/{name}",
    response_model=ScheduleEnvelopeResponse,
    description="User-scoped browser-safe agent-run schedule update.",
  )
  async def update_agent_run_schedule(
    request: Request,
    name: str,
    payload: UpdateAgentRunScheduleRequest,
  ) -> ScheduleEnvelopeResponse:
    session = _require_bearer_session(request, auth)
    owner_user_id = _schedule_owner_user_id(session)
    agent_store = store_for(owner_user_id)
    existing = agent_store.get_for_owner(owner_user_id, name)
    if existing is None:
      if _is_web_session(session):
        raise _raw_web_schedule_write_forbidden()
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
    if payload.dispatch is not None:
      dispatch = await _validated_agent_schedule_dispatch(payload.dispatch, session=session)
      payload = payload.model_copy(update={"dispatch": dispatch})
      _require_agent_schedule_dispatch_allowed(session, payload.dispatch)
    updated = agent_store.update(
      owner_user_id,
      name,
      payload,
      updated_by=owner_user_id,
      live_role=require_exact_role(getattr(session, "role", None)),
    )
    return ScheduleEnvelopeResponse(schedule=_project_schedule_for_web(updated))

  @router.post(
    "/{name}/run-now",
    response_model=AutonomousDispatchResponse,
    description="Run a user-owned browser-safe agent-run schedule immediately without advancing cadence.",
  )
  async def run_agent_run_schedule_now(request: Request, name: str) -> AutonomousDispatchResponse:
    session = _require_bearer_session(request, auth)
    owner_user_id = _schedule_owner_user_id(session)
    record = store_for(owner_user_id).get_for_owner(owner_user_id, name)
    if record is None:
      if _is_web_session(session):
        raise _raw_web_schedule_write_forbidden()
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
    if agent_schedule_runner is None:
      raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent schedule runner unavailable")
    result = await agent_schedule_runner.fire_record_now(
      record,
      live_role=require_exact_role(getattr(session, "role", None)),
    )
    if result is None:
      raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Schedule run-now did not start")
    if result.get("status") != "started":
      raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=str(result.get("error") or "Schedule run-now failed"),
      )
    registry = agent_schedule_runner.autonomous_registry
    task = _autonomous_task_for_user(registry, str(result.get("run_id") or ""), owner_user_id)
    run = _autonomous_run_from_task(task)
    return AutonomousDispatchResponse(
      run=run,
      task_id=task.task_id,
      run_id=task.control_run_id,
      log_path=str(task.log_path),
      started_at=int(task.started_at),
      cmd=list(task.cmd),
    )

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
    session = _require_bearer_session(request, auth)
    owner_user_id = _schedule_owner_user_id(session)
    agent_store = store_for(owner_user_id)
    if not _can_access_operator_schedules(session):
      owned_schedule = agent_store.get_for_owner(owner_user_id, name)
      if owned_schedule is None:
        raise _operator_schedule_not_found(name)
      updated = agent_store.set_enabled(
        owner_user_id,
        name,
        enabled=payload.enabled,
        updated_by=owner_user_id,
        live_role=require_exact_role(getattr(session, "role", None)),
      )
      return ScheduleEnvelopeResponse(schedule=_project_schedule_for_web(updated))
    if _is_web_session(session):
      owned_schedule = agent_store.get_for_owner(owner_user_id, name)
      if owned_schedule is not None:
        updated = agent_store.set_enabled(
          owner_user_id,
          name,
          enabled=payload.enabled,
          updated_by=owner_user_id,
          live_role=require_exact_role(getattr(session, "role", None)),
        )
        return ScheduleEnvelopeResponse(schedule=_project_schedule_for_web(updated))
      raise _raw_web_schedule_write_forbidden()
    schedule = _show_schedule_or_none(name)
    if schedule is None:
      owned_schedule = agent_store.get_for_owner(owner_user_id, name)
      if owned_schedule is not None:
        updated = agent_store.set_enabled(
          owner_user_id,
          name,
          enabled=payload.enabled,
          updated_by=owner_user_id,
          live_role=require_exact_role(getattr(session, "role", None)),
        )
        return ScheduleEnvelopeResponse(schedule=_project_schedule_for_web(updated))
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Schedule not found: {name}")
    if isinstance(schedule, LaunchdScheduleResponse):
      result = (
        _scheduler_mcp().schedule_enable(name)
        if payload.enabled
        else _scheduler_mcp().schedule_disable(name)
      )
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
    session = _require_bearer_session(request, auth)
    owner_user_id = _schedule_owner_user_id(session)
    agent_store = store_for(owner_user_id)
    if not _can_access_operator_schedules(session):
      owned_schedule = agent_store.get_for_owner(owner_user_id, name)
      if owned_schedule is None:
        raise _operator_schedule_not_found(name)
      if not _is_web_session(session) and not confirm:
        raise HTTPException(status_code=400, detail="confirm=true is required to delete a schedule")
      deleted = agent_store.delete(owner_user_id, name)
      return ScheduleDeleteResponse(
        deleted=True,
        name=str(deleted.get("name") or name),
        source=_AGENT_RUN_SCHEDULE_BACKEND,
      )
    if _is_web_session(session):
      owned_schedule = agent_store.get_for_owner(owner_user_id, name)
      if owned_schedule is not None:
        deleted = agent_store.delete(owner_user_id, name)
        return ScheduleDeleteResponse(
          deleted=True,
          name=str(deleted.get("name") or name),
          source=_AGENT_RUN_SCHEDULE_BACKEND,
        )
      raise _raw_web_schedule_write_forbidden()
    if not confirm:
      raise HTTPException(status_code=400, detail="confirm=true is required to delete a schedule")

    schedule = _show_schedule_or_none(name)
    if schedule is None:
      owned_schedule = agent_store.get_for_owner(owner_user_id, name)
      if owned_schedule is not None:
        deleted = agent_store.delete(owner_user_id, name)
        return ScheduleDeleteResponse(
          deleted=True,
          name=str(deleted.get("name") or name),
          source=_AGENT_RUN_SCHEDULE_BACKEND,
        )
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Schedule not found: {name}")
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
  "BrowserSafeScheduleResponse",
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
