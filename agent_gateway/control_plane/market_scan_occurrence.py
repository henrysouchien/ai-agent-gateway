"""Scheduled-occurrence materialization for market-scan shadow definitions."""

from __future__ import annotations

import json
import re

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SHADOW_DISPATCH_SCHEMA = "market-scan-shadow-dispatch/v1"
OCCURRENCE_DISPATCH_SCHEMA = "market-scan-scheduled-occurrence/v1"
SCHEDULED_EXTERNAL_REF_KEYS = frozenset({
  "agent_run_schedule_id",
  "schedule_definition_id",
  "source_schedule_id",
  "scheduled_occurrence_utc",
})
_REQUEST_SCHEMA = "investment-deterministic-run-request/v1"
_SCHEDULE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TIME_OF_DAY_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_STATIC_CONTEXT_KEYS = {
  "schema_version",
  "shadow",
  "execution_authorized",
  "definition_id",
  "capability_id",
  "capability_version",
  "facade_tool",
  "timezone",
  "occurrence_request_schema",
  "idempotency_scheme",
  "source_occurrences",
}
_SOURCE_OCCURRENCE_KEYS = {
  "source_schedule_id",
  "iso_day_of_week",
  "time_of_day",
  "inputs",
}
_EXPECTED_DEFINITIONS: dict[str, tuple[str, tuple[dict[str, Any], ...]]] = {
  "quality-weekly": (
    "quality_screen",
    ({
      "source_schedule_id": "sched_36670bc6",
      "iso_day_of_week": 7,
      "time_of_day": "04:30",
      "inputs": {
        "exchanges": "NASDAQ,NYSE",
        "min_market_cap": 1_000_000_000,
        "min_score": 6,
      },
    },),
  ),
  "insider-mon-thu": (
    "insider_buying",
    (
      {
        "source_schedule_id": "sched_3270bfee",
        "iso_day_of_week": 1,
        "time_of_day": "04:30",
        "inputs": {
          "lookback_days": 14,
          "min_insider_count": 2,
          "min_score": 15,
          "top_n": 10,
        },
      },
      {
        "source_schedule_id": "sched_36859d0f",
        "iso_day_of_week": 4,
        "time_of_day": "04:30",
        "inputs": {},
      },
    ),
  ),
  "fingerprint-weekly": (
    "fingerprint_screen",
    ({
      "source_schedule_id": "sched_2ba0fdea",
      "iso_day_of_week": 3,
      "time_of_day": "04:30",
      "inputs": {"top": 20},
    },),
  ),
}


class MarketScanOccurrenceMaterializationError(ValueError):
  """Raised when a recognized market-scan shadow dispatch is unsafe."""


@dataclass(frozen=True, slots=True)
class ScheduledInvestmentRunAuthority:
  """Exact facade arguments recovered from a signed autonomous workload."""

  agent_run_schedule_id: str
  schedule_definition_id: str
  source_schedule_id: str
  scheduled_occurrence_utc: str
  capability_id: str
  capability_version: str
  idempotency_key: str
  request_hash: str
  _canonical_tool_arguments: str = field(repr=False)

  def bind_tool_arguments(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
    """Bind minimal model intent to a defensive copy of trusted arguments.

    The model may identify only the already-authorized capability. The request,
    idempotency key, and schedule references remain exclusively code-owned.
    """

    if (
      type(supplied) is not dict
      or set(supplied) != {"capability_id"}
      or supplied.get("capability_id") != self.capability_id
    ):
      raise MarketScanOccurrenceMaterializationError(
        "scheduled investment facade intent does not match trusted occurrence"
      )
    return json.loads(self._canonical_tool_arguments)


def _fail() -> MarketScanOccurrenceMaterializationError:
  return MarketScanOccurrenceMaterializationError(
    "market-scan scheduled occurrence context is invalid"
  )


def _load_context(value: object) -> dict[str, Any] | None:
  if not isinstance(value, str):
    return None
  try:
    decoded = json.loads(value)
  except (TypeError, ValueError):
    return None
  if not isinstance(decoded, dict):
    return None
  if decoded.get("schema_version") != SHADOW_DISPATCH_SCHEMA:
    return None
  return decoded


def _load_occurrence_context(value: object) -> dict[str, Any] | None:
  if not isinstance(value, str):
    return None
  try:
    decoded = json.loads(value)
  except (TypeError, ValueError):
    return None
  if not isinstance(decoded, dict):
    return None
  if decoded.get("schema_version") != OCCURRENCE_DISPATCH_SCHEMA:
    return None
  return decoded


def is_market_scan_shadow_dispatch(dispatch: Mapping[str, Any]) -> bool:
  """Return true only for the code-owned versioned shadow context."""

  return _load_context(dispatch.get("context")) is not None


def _require_string(value: object) -> str:
  if not isinstance(value, str) or not value.strip():
    raise _fail()
  return value


def _validate_static_context(
  dispatch: Mapping[str, Any],
  context: dict[str, Any],
) -> tuple[str, str, str, ZoneInfo, list[dict[str, Any]]]:
  if set(context) != _STATIC_CONTEXT_KEYS:
    raise _fail()
  if (
    dispatch.get("kind") != "autonomous"
    or dispatch.get("profile") != "research_producer"
    or dispatch.get("mode") != "skill"
    or dispatch.get("skill") != "market-scan"
    or context.get("shadow") is not True
    or context.get("execution_authorized") is not False
    or context.get("facade_tool") != "start_investment_run"
    or context.get("occurrence_request_schema") != _REQUEST_SCHEMA
    or context.get("idempotency_scheme")
    != "market-scan/{agent_run_schedule_id}/{scheduled_occurrence_utc_compact}"
  ):
    raise _fail()
  definition_id = _require_string(context.get("definition_id"))
  capability_id = _require_string(context.get("capability_id"))
  capability_version = _require_string(context.get("capability_version"))
  if capability_version != "1":
    raise _fail()
  timezone_name = _require_string(context.get("timezone"))
  if timezone_name != "America/New_York":
    raise _fail()
  try:
    zone = ZoneInfo(timezone_name)
  except ZoneInfoNotFoundError as exc:
    raise _fail() from exc
  raw_occurrences = context.get("source_occurrences")
  if not isinstance(raw_occurrences, list) or not raw_occurrences:
    raise _fail()
  occurrences: list[dict[str, Any]] = []
  seen_days: set[int] = set()
  seen_source_ids: set[str] = set()
  for raw in raw_occurrences:
    if not isinstance(raw, dict) or set(raw) != _SOURCE_OCCURRENCE_KEYS:
      raise _fail()
    source_schedule_id = _require_string(raw.get("source_schedule_id"))
    if _SCHEDULE_ID_RE.fullmatch(source_schedule_id) is None:
      raise _fail()
    iso_day = raw.get("iso_day_of_week")
    if type(iso_day) is not int or iso_day < 1 or iso_day > 7:
      raise _fail()
    time_of_day = raw.get("time_of_day")
    if not isinstance(time_of_day, str) or _TIME_OF_DAY_RE.fullmatch(time_of_day) is None:
      raise _fail()
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict):
      raise _fail()
    try:
      json.dumps(inputs, allow_nan=False)
    except (TypeError, ValueError) as exc:
      raise _fail() from exc
    if iso_day in seen_days or source_schedule_id in seen_source_ids:
      raise _fail()
    seen_days.add(iso_day)
    seen_source_ids.add(source_schedule_id)
    occurrences.append({
      "source_schedule_id": source_schedule_id,
      "iso_day_of_week": iso_day,
      "time_of_day": time_of_day,
      "inputs": deepcopy(inputs),
    })
  expected = _EXPECTED_DEFINITIONS.get(definition_id)
  if expected is None or capability_id != expected[0]:
    raise _fail()
  if json.dumps(occurrences, sort_keys=True, separators=(",", ":")) != json.dumps(
    expected[1],
    sort_keys=True,
    separators=(",", ":"),
  ):
    raise _fail()
  return definition_id, capability_id, capability_version, zone, occurrences


def materialize_market_scan_occurrence_dispatch(
  dispatch: Mapping[str, Any],
  *,
  agent_run_schedule_id: str,
  scheduled_for: datetime,
) -> dict[str, Any]:
  """Replace a shadow template with one exact canonical facade dispatch.

  Non-market-scan dispatches are returned unchanged. A recognized shadow
  context is validated strictly and fails closed on any drift.
  """

  context = _load_context(dispatch.get("context"))
  if context is None:
    return deepcopy(dict(dispatch))
  if (
    not isinstance(agent_run_schedule_id, str)
    or _SCHEDULE_ID_RE.fullmatch(agent_run_schedule_id) is None
  ):
    raise _fail()
  if not isinstance(scheduled_for, datetime) or scheduled_for.tzinfo is None:
    raise _fail()
  (
    definition_id,
    capability_id,
    capability_version,
    zone,
    occurrences,
  ) = _validate_static_context(dispatch, context)
  scheduled_utc = scheduled_for.astimezone(timezone.utc)
  if scheduled_utc.second != 0 or scheduled_utc.microsecond != 0:
    raise _fail()
  local = scheduled_utc.astimezone(zone)
  local_time = local.strftime("%H:%M")
  matches = [
    item
    for item in occurrences
    if item["iso_day_of_week"] == local.isoweekday()
    and item["time_of_day"] == local_time
  ]
  if len(matches) != 1:
    raise _fail()
  selected = matches[0]
  scheduled_iso = scheduled_utc.isoformat().replace("+00:00", "Z")
  scheduled_compact = scheduled_utc.strftime("%Y%m%dT%H%M%SZ")
  idempotency_key = (
    f"market-scan/{agent_run_schedule_id}/{scheduled_compact}"
  )
  if len(idempotency_key) > 200:
    raise _fail()
  external_refs = {
    "agent_run_schedule_id": agent_run_schedule_id,
    "schedule_definition_id": definition_id,
    "source_schedule_id": selected["source_schedule_id"],
    "scheduled_occurrence_utc": scheduled_iso,
  }
  request = {
    "schema_version": _REQUEST_SCHEMA,
    "capability_id": capability_id,
    "capability_version": capability_version,
    "inputs": selected["inputs"],
    "idempotency_key": idempotency_key,
    "external_refs": external_refs,
  }
  materialized_context = {
    "schema_version": OCCURRENCE_DISPATCH_SCHEMA,
    "facade_tool": "start_investment_run",
    "deterministic_request": request,
    "tool_arguments": {
      "capability_id": capability_id,
      "request": selected["inputs"],
      "idempotency_key": idempotency_key,
      "external_refs": external_refs,
    },
  }
  result = deepcopy(dict(dispatch))
  result["context"] = json.dumps(
    materialized_context,
    sort_keys=True,
    separators=(",", ":"),
  )
  return result


def scheduled_investment_authority_from_context(
  context_value: object,
  *,
  expected_schedule_id: str | None = None,
) -> ScheduledInvestmentRunAuthority | None:
  """Parse exact authority only from a materialized occurrence context.

  The caller is responsible for establishing that ``context_value`` came from
  the signed autonomous workload. A recognized but drifted context fails
  closed; unrelated contexts return ``None``.
  """

  context = _load_occurrence_context(context_value)
  if context is None:
    return None
  if set(context) != {
    "schema_version",
    "facade_tool",
    "deterministic_request",
    "tool_arguments",
  } or context.get("facade_tool") != "start_investment_run":
    raise _fail()
  request = context.get("deterministic_request")
  tool_arguments = context.get("tool_arguments")
  if (
    not isinstance(request, dict)
    or set(request) != {
      "schema_version",
      "capability_id",
      "capability_version",
      "inputs",
      "idempotency_key",
      "external_refs",
    }
    or request.get("schema_version") != _REQUEST_SCHEMA
    or not isinstance(tool_arguments, dict)
    or set(tool_arguments) != {
      "capability_id",
      "request",
      "idempotency_key",
      "external_refs",
    }
  ):
    raise _fail()
  capability_id = _require_string(request.get("capability_id"))
  capability_version = _require_string(request.get("capability_version"))
  if capability_version != "1":
    raise _fail()
  inputs = request.get("inputs")
  external_refs = request.get("external_refs")
  idempotency_key = _require_string(request.get("idempotency_key"))
  if (
    not isinstance(inputs, dict)
    or not isinstance(external_refs, dict)
    or set(external_refs) != SCHEDULED_EXTERNAL_REF_KEYS
  ):
    raise _fail()
  agent_run_schedule_id = _require_string(
    external_refs.get("agent_run_schedule_id")
  )
  definition_id = _require_string(external_refs.get("schedule_definition_id"))
  source_schedule_id = _require_string(external_refs.get("source_schedule_id"))
  scheduled_occurrence_utc = _require_string(
    external_refs.get("scheduled_occurrence_utc")
  )
  if (
    _SCHEDULE_ID_RE.fullmatch(agent_run_schedule_id) is None
    or expected_schedule_id is not None
    and agent_run_schedule_id != expected_schedule_id
  ):
    raise _fail()
  try:
    scheduled = datetime.fromisoformat(
      scheduled_occurrence_utc.replace("Z", "+00:00")
    )
  except ValueError as exc:
    raise _fail() from exc
  if (
    scheduled.tzinfo is None
    or scheduled.utcoffset() != timezone.utc.utcoffset(scheduled)
    or scheduled.second != 0
    or scheduled.microsecond != 0
    or scheduled.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    != scheduled_occurrence_utc
  ):
    raise _fail()
  expected_key = (
    f"market-scan/{agent_run_schedule_id}/"
    f"{scheduled.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
  )
  expected = _EXPECTED_DEFINITIONS.get(definition_id)
  if expected is None or capability_id != expected[0] or idempotency_key != expected_key:
    raise _fail()
  expected_occurrence = next(
    (
      item
      for item in expected[1]
      if item["source_schedule_id"] == source_schedule_id
    ),
    None,
  )
  if expected_occurrence is None:
    raise _fail()
  expected_inputs_json = json.dumps(
    expected_occurrence["inputs"],
    sort_keys=True,
    separators=(",", ":"),
  )
  try:
    inputs_json = json.dumps(
      inputs,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
      allow_nan=False,
    )
  except (TypeError, ValueError) as exc:
    raise _fail() from exc
  if inputs_json != expected_inputs_json:
    raise _fail()
  expected_tool_arguments = {
    "capability_id": capability_id,
    "request": inputs,
    "idempotency_key": idempotency_key,
    "external_refs": external_refs,
  }
  canonical_tool_arguments = json.dumps(
    expected_tool_arguments,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  )
  if json.dumps(
    tool_arguments,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  ) != canonical_tool_arguments:
    raise _fail()
  canonical_request = json.dumps(
    request,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  )
  return ScheduledInvestmentRunAuthority(
    agent_run_schedule_id=agent_run_schedule_id,
    schedule_definition_id=definition_id,
    source_schedule_id=source_schedule_id,
    scheduled_occurrence_utc=scheduled_occurrence_utc,
    capability_id=capability_id,
    capability_version=capability_version,
    idempotency_key=idempotency_key,
    request_hash=sha256(canonical_request.encode("utf-8")).hexdigest(),
    _canonical_tool_arguments=canonical_tool_arguments,
  )


__all__ = [
  "MarketScanOccurrenceMaterializationError",
  "OCCURRENCE_DISPATCH_SCHEMA",
  "SCHEDULED_EXTERNAL_REF_KEYS",
  "SHADOW_DISPATCH_SCHEMA",
  "ScheduledInvestmentRunAuthority",
  "is_market_scan_shadow_dispatch",
  "materialize_market_scan_occurrence_dispatch",
  "scheduled_investment_authority_from_context",
]
