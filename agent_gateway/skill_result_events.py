from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .artifact_paths import canonicalize_ticker


FMS_DOOR_PREFIX = "fms_"


def build_skill_result_captured_event(
  *,
  skill_run_id: str,
  skill: str,
  ticker: str | None,
  entries: Iterable[Any],
  result: Any | None,
  error: dict[str, Any] | None,
  output_memory_file: str | None = None,
  cost_usd: float | None = None,
  duration_s: float | None = None,
) -> dict[str, Any]:
  entry_list = list(entries)
  fms_results = extract_fms_results(entry_list)
  artifact_events = extract_artifact_events(entry_list)
  primary_fms = _primary_fms(fms_results)
  has_error = error is not None
  outcome = "error" if has_error else "success"
  raw_status = primary_fms.get("status") if isinstance(primary_fms, dict) else None
  return {
    "type": "skill_result_captured",
    "skill_run_id": skill_run_id,
    "skill": skill,
    "ticker": _result_ticker(ticker, fms_results, artifact_events),
    "exit_code": 1 if has_error else 0,
    "outcome": outcome,
    "status": str(raw_status) if raw_status is not None else outcome,
    "gate_code": _gate_code(primary_fms),
    "artifact_refs": _artifact_refs(fms_results, artifact_events),
    "proposal_ids": _proposal_ids(fms_results),
    "verdict_echo": _verdict_echo(primary_fms),
    "fms_results": fms_results,
    "artifact_events": artifact_events,
    "output_memory_file": output_memory_file,
    "cost_usd": cost_usd,
    "duration_s": duration_s,
    "error": _result_error(error, fms_results),
    "warnings": _warnings(result, fms_results),
  }


def extract_fms_results(
  entries: Iterable[Any],
  *,
  door_names: set[str] | None = None,
) -> list[dict[str, Any]]:
  results: list[dict[str, Any]] = []
  for entry in entries:
    event = _entry_event(entry)
    if event is None or not _is_fms_door_event(event, door_names=door_names):
      continue
    result = event.get("result")
    if not isinstance(result, dict):
      continue
    annotated = dict(result)
    annotated["tool_name"] = event.get("tool_name")
    results.append(annotated)
  return results


def extract_artifact_events(entries: Iterable[Any]) -> list[dict[str, Any]]:
  events: list[dict[str, Any]] = []
  for entry in entries:
    event = _entry_event(entry)
    if event is not None and event.get("type") == "artifact_ready":
      events.append(dict(event))
  return events


def _is_fms_door_event(
  event: dict[str, Any],
  *,
  door_names: set[str] | None = None,
) -> bool:
  if event.get("type") != "tool_call_complete":
    return False
  tool_name = event.get("tool_name")
  if door_names is not None:
    if tool_name not in door_names:
      return False
  elif not isinstance(tool_name, str) or not tool_name.startswith(FMS_DOOR_PREFIX):
    return False
  result = event.get("result")
  return isinstance(result, dict) and "subcommand" in result and "mutation_mode" in result


def _entry_event(entry: Any) -> dict[str, Any] | None:
  event = getattr(entry, "event", entry)
  return event if isinstance(event, dict) else None


def _primary_fms(fms_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
  primary: dict[str, Any] = {}
  for item in fms_results:
    if isinstance(item, dict):
      primary = item
  return primary


def _artifact_refs(
  fms_results: Iterable[dict[str, Any]],
  artifact_events: Iterable[dict[str, Any]],
) -> list[str]:
  refs = []
  seen: set[str] = set()
  candidates = [
    *(
      item.get("artifact_ref") or item.get("artifact_path")
      for item in fms_results
      if isinstance(item, dict)
    ),
    *(
      event.get("artifact_ref") or event.get("artifact_path")
      for event in artifact_events
      if isinstance(event, dict)
    ),
  ]
  for ref in candidates:
    if not isinstance(ref, str) or not ref or ref in seen:
      continue
    seen.add(ref)
    refs.append(ref)
  return refs


def _proposal_ids(fms_results: Iterable[dict[str, Any]]) -> list[str]:
  proposal_ids: list[str] = []
  seen: set[str] = set()
  for item in fms_results:
    if not isinstance(item, dict):
      continue
    proposal_id = item.get("proposal_id")
    if not isinstance(proposal_id, str) or not proposal_id or proposal_id in seen:
      continue
    seen.add(proposal_id)
    proposal_ids.append(proposal_id)
  return proposal_ids


def _verdict_echo(primary_fms: dict[str, Any]) -> dict[str, Any] | None:
  verdict_echo = primary_fms.get("verdict_echo") if isinstance(primary_fms, dict) else None
  return dict(verdict_echo) if isinstance(verdict_echo, dict) else None


def _gate_code(primary_fms: dict[str, Any]) -> str | None:
  raw_gate = primary_fms.get("gate_code")
  verdict_echo = _verdict_echo(primary_fms)
  if raw_gate is None and isinstance(verdict_echo, dict):
    raw_gate = verdict_echo.get("gate_code") or verdict_echo.get("verdict")
  return str(raw_gate) if raw_gate is not None else None


def _warnings(result: Any | None, fms_results: Iterable[dict[str, Any]]) -> list[str]:
  warnings: list[str] = []
  if isinstance(result, dict):
    raw_warning = result.get("warning")
    if raw_warning is not None and str(raw_warning).strip():
      warnings.append(str(raw_warning))
  for item in fms_results:
    if not isinstance(item, dict):
      continue
    raw = item.get("warnings")
    if isinstance(raw, list):
      warnings.extend(str(value) for value in raw if value is not None and str(value).strip())
    elif raw is not None and str(raw).strip():
      warnings.append(str(raw))
  return warnings


def _result_error(error: dict[str, Any] | None, fms_results: Iterable[dict[str, Any]]) -> str | None:
  if isinstance(error, dict):
    message = error.get("message") or error.get("error") or error.get("code")
    if message:
      return str(message)
  primary_fms = _primary_fms(fms_results)
  raw_error = primary_fms.get("error") if isinstance(primary_fms, dict) else None
  return str(raw_error) if raw_error else None


def _result_ticker(
  context_ticker: str | None,
  fms_results: Iterable[dict[str, Any]],
  artifact_events: Iterable[dict[str, Any]],
) -> str | None:
  for item in fms_results:
    if not isinstance(item, dict):
      continue
    raw = item.get("ticker")
    if raw is not None and str(raw).strip():
      return canonicalize_ticker(raw)
  for event in artifact_events:
    if not isinstance(event, dict):
      continue
    raw = event.get("ticker")
    if raw is not None and str(raw).strip():
      return canonicalize_ticker(raw)
  return canonicalize_ticker(context_ticker) if context_ticker else None


__all__ = [
  "FMS_DOOR_PREFIX",
  "build_skill_result_captured_event",
  "extract_artifact_events",
  "extract_fms_results",
]
