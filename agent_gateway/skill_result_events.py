from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent_workflow_contracts import TaskResult
from pydantic import ValidationError

from .skill_lifecycle import (
  SkillLifecycleArtifactIdentity,
  SkillLifecycleScope,
  TopLevelSkillLifecycleMetadata,
)
from .sub_agent_skill_state import (
  ChildOutcomeClassification,
  classify_child_outcome,
  is_context_door_name,
  is_declared_terminal_door,
  terminal_fms_result_disposition,
)


FMS_DOOR_PREFIX = "fms_"


def build_skill_result_captured_event(
  *,
  skill_run_id: str,
  skill: str,
  ticker: str | None,
  scope: SkillLifecycleScope,
  portfolio_id: str | None,
  entries: Iterable[Any],
  result: Any | None,
  error: dict[str, Any] | None,
  output_memory_file: str | None = None,
  duration_s: float | None = None,
  canonical_result_evidence_authoritative: bool = False,
  declared_terminal_doors: Iterable[str] | None = None,
) -> dict[str, Any] | None:
  # ``declared_terminal_doors`` is the persist expectation:
  # ``persist_expectation_doors(admitted_task)`` — the terminal-result doors,
  # and only when the operation declares ``result_requires_persist``. A
  # coordinator carries its children's doors without owing a persist of its own.
  artifact_identity = SkillLifecycleArtifactIdentity(
    scope=scope,
    ticker=ticker,
    portfolio_id=portfolio_id,
  )
  lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id=skill_run_id,
    skill=skill,
    **artifact_identity.identity_fields(),
  )
  entry_list = list(entries)
  fms_results = extract_fms_results(entry_list)
  # The door answers the child actually received, in order; the outcome is
  # classified on these on every path. Canonical evidence below replaces only
  # what is published (artifacts), never which doors answered.
  door_results = extract_door_results(entry_list)
  artifact_events = extract_artifact_events(entry_list)
  canonical_evidence = _canonical_task_result_evidence(result)
  if (
    canonical_result_evidence_authoritative
    and canonical_evidence is not None
  ):
    fms_results, artifact_events = canonical_evidence
  artifact_refs = _artifact_refs(fms_results, artifact_events)
  primary_fms = _primary_fms(fms_results)
  door_names = {
    str(item.get("tool_name"))
    for item in door_results
    if item.get("tool_name")
  }
  classification = classify_child_outcome(
    result,
    error,
    declared_terminal_doors=door_names,
    door_results=door_results,
  )
  declared_terminals = frozenset(declared_terminal_doors or ())
  if (
    classification.succeeded
    and declared_terminals
    and not artifact_refs
    and not any(
      item.get("tool_name") in declared_terminals
      and terminal_fms_result_disposition(item) == "success"
      for item in door_results
    )
  ):
    # The grant declares a terminal door; ending with no accepted answer from
    # it and no artifact is not a success, whatever the narrative claims.
    classification = ChildOutcomeClassification(
      outcome="blocked",
      succeeded=False,
      error={
        "code": "no_gate_produced",
        "message": "skill ended without an accepted terminal door result or artifact",
        "child_outcome": "blocked",
      },
    )
  if not classification.is_final:
    return None
  primary_door_status = (
    str(primary_fms.get("status")).strip().lower()
    if isinstance(primary_fms, dict) and primary_fms.get("status") is not None
    else None
  )
  has_error = not classification.succeeded
  outcome = "error" if has_error else "success"
  event = {
    "type": "skill_result_captured",
    **lifecycle.identity_fields(),
    "exit_code": 1 if has_error else 0,
    "outcome": outcome,
    # The child's status is the child's own disposition, from the same owner
    # as exit_code (``classify_child_outcome``, which already fails the run
    # when its declared terminal door failed). Reading it off the last FMS
    # envelope instead paired a failure status with exit code 0 whenever a
    # non-terminal door reported one, which the lifecycle contract rejects.
    # Door statuses stay visible inside ``fms_results``.
    "status": classification.outcome,
    "gate_code": _gate_code(primary_fms),
    "artifact_refs": artifact_refs,
    "proposal_ids": _proposal_ids(fms_results),
    "verdict_echo": _verdict_echo(primary_fms),
    "fms_results": fms_results,
    "last_failed_door_answer": extract_last_failed_door_answer(entry_list),
    "artifact_events": artifact_events,
    "output_memory_file": output_memory_file,
    "cost_usd": _canonical_task_result_cost_usd(result),
    "duration_s": duration_s,
    "compaction_count": _compaction_count(entry_list),
    "error": _result_error(
      classification.error,
      fms_results,
      fallback_status=primary_door_status if has_error else None,
    ),
    "warnings": _warnings(result, fms_results),
    "approval_outcome": None,
    "approval_id": None,
    "approval_tool_name": None,
  }
  return lifecycle.normalize_result_event(event)


def _canonical_task_result_cost_usd(result: Any | None) -> float | None:
  """Read the run's spend off the canonical result that reports it.

  No caller ever passed a cost here, so every ``skill_result_captured`` said
  ``cost_usd: null`` — including the ones a parent had to judge a terminated
  child by (R2, 2026-09-17). The child settlement already carries exact
  provider-reported spend in ``observation.usage``; read it there instead of
  waiting for a caller to hand it over.
  """

  try:
    task_result = TaskResult.model_validate(result)
  except ValidationError:
    return None
  return task_result.observation.usage.cost_usd


def _canonical_task_result_evidence(
  result: Any | None,
) -> tuple[
  list[dict[str, Any]],
  list[dict[str, Any]],
] | None:
  """Recover only durable artifacts published by canonical TaskResult."""

  try:
    task_result = TaskResult.model_validate(result)
  except ValidationError:
    return None
  return (
    [],
    [
      {
        "type": "artifact_ready",
        "artifact_name": artifact.name,
        "artifact_ref": artifact.content.content_id,
      }
      for artifact in task_result.values.artifacts
    ]
  )


def extract_last_failed_door_answer(
  entries: Iterable[Any],
) -> dict[str, Any] | None:
  """Copy the last failed tool's answer, independent of its door namespace."""

  last_answer: dict[str, Any] | None = None
  for entry in entries:
    event = _entry_event(entry)
    if event is None or event.get("type") != "tool_call_complete":
      continue
    result = event.get("result")
    if not isinstance(result, dict):
      continue
    error = result.get("error")
    if isinstance(error, dict):
      data = error.get("data")
      next_call = (
        data["next_call"]
        if isinstance(data, dict) and "next_call" in data
        else error.get("next_call")
      )
      code = error.get("code")
    elif result.get("status") == "error":
      code = result.get("error_type")
      next_call = result.get("next_call")
    else:
      continue
    last_answer = {
      "tool_name": event.get("tool_name"),
      "code": code,
      "next_call": next_call,
    }
  return last_answer


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


def extract_door_results(entries: Iterable[Any]) -> list[dict[str, Any]]:
  """Collect terminal (``fms_*``) and context (``get_*_context``) door answers
  in entry order, each annotated with its ``tool_name``.

  The last answer decides the child's outcome and a later accepted door
  supersedes an earlier refusal (``classify_child_outcome``), so order is
  the contract.
  """

  results: list[dict[str, Any]] = []
  for entry in entries:
    event = _entry_event(entry)
    if event is None or event.get("type") != "tool_call_complete":
      continue
    tool_name = event.get("tool_name")
    if not isinstance(tool_name, str) or not (
      is_declared_terminal_door(tool_name) or is_context_door_name(tool_name)
    ):
      continue
    result = event.get("result")
    if not isinstance(result, dict):
      continue
    annotated = dict(result)
    annotated["tool_name"] = tool_name
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


def _compaction_count(entries: Iterable[Any]) -> int:
  return sum(
    1
    for entry in entries
    if (event := _entry_event(entry)) is not None
    and event.get("type") == "compaction"
  )


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
  # The door envelope carries the normalized FMS gate code; the event
  # normalizer (``skill_lifecycle.GATE_CODES``) owns which values are valid.
  raw_gate = primary_fms.get("gate_code")
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


def _result_error(
  error: dict[str, Any] | None,
  fms_results: Iterable[dict[str, Any]],
  *,
  fallback_status: str | None = None,
) -> str | None:
  if isinstance(error, dict):
    message = error.get("message") or error.get("error") or error.get("code")
    if message:
      return str(message)
  primary_fms = _primary_fms(fms_results)
  raw_error = primary_fms.get("error") if isinstance(primary_fms, dict) else None
  if isinstance(raw_error, dict):
    message = raw_error.get("message") or raw_error.get("type")
    if message:
      return str(message)
  if raw_error:
    return str(raw_error)
  if fallback_status:
    return f"FMS returned {fallback_status}"
  return None


__all__ = [
  "FMS_DOOR_PREFIX",
  "build_skill_result_captured_event",
  "extract_artifact_events",
  "extract_fms_results",
  "extract_door_results",
]
