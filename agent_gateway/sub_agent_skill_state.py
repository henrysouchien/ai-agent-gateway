from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agent_workflow_contracts import TaskResult, ToolGrant
from pydantic import ValidationError

_FAILURE_OUTCOME_BY_STATUS = {
  "failed": "error",
  "interrupted": "interrupted",
  "cancelled": "interrupted",
  "skipped": "skipped",
}

_FAILURE_FMS_STATUSES = frozenset({
  "error",
  "failed",
  "failure",
  "invalid",
  "rejected",
})
_FMS_DOOR_PREFIX = "fms_"
_FMS_NON_TERMINAL_PREFIXES = ("fms_compute_",)
_TERMINAL_SUCCESS_STATUSES = frozenset({
  "applied",
  "noop",
  "ok",
  "staged",
  "success",
})
_TERMINAL_RECEIPT_FIELDS = (
  "artifact_ref",
  "artifact_path",
  "dashboard_ref",
  "proposal_id",
)


@dataclass(frozen=True)
class ChildOutcomeClassification:
  outcome: str
  succeeded: bool
  error: dict[str, Any] | None
  semantic: str | None = None
  execution_status: str = "failed"


def declared_terminal_doors_from_grant(grant: ToolGrant | None) -> frozenset[str]:
  """Return the FMS/product doors already admitted on the child's grant."""

  if grant is None:
    return frozenset()
  return frozenset(
    entry.tool_id
    for entry in grant.tools
    if _is_declared_terminal_door(entry.tool_id)
  )


def _is_declared_terminal_door(tool_id: str) -> bool:
  if not tool_id.startswith(_FMS_DOOR_PREFIX):
    return False
  return not tool_id.startswith(_FMS_NON_TERMINAL_PREFIXES)


def _failure_classification(
  *,
  reason: str,
  message: str,
  outcome: str = "error",
  execution_status: str = "failed",
  semantic: str | None = None,
) -> ChildOutcomeClassification:
  return ChildOutcomeClassification(
    outcome=outcome,
    succeeded=False,
    error={
      "code": reason,
      "message": message,
      "child_outcome": outcome,
    },
    semantic=semantic,
    execution_status=execution_status,
  )


def _named_door_classification(
  *,
  declared_terminal_doors: frozenset[str],
  tools_used: Sequence[str],
  door_results: Sequence[Mapping[str, Any]],
  process_classification: ChildOutcomeClassification,
) -> ChildOutcomeClassification:
  invoked = declared_terminal_doors.intersection(tools_used)
  matching_results = [
    result
    for result in door_results
    if result.get("tool_name") in declared_terminal_doors
  ]
  if not invoked and not matching_results:
    doors = ", ".join(sorted(declared_terminal_doors))
    return _failure_classification(
      reason="terminal_door_not_invoked",
      message=(
        "Named agent-operation ended without invoking its declared "
        f"terminal door ({doors})"
      ),
      outcome="blocked",
      semantic="none",
    )
  terminal = matching_results[-1] if matching_results else None
  if terminal is not None:
    raw_status = str(terminal.get("status") or "").strip().lower()
    gate_code = str(terminal.get("gate_code") or "").strip()
    if gate_code == "STOP" or raw_status in _FAILURE_FMS_STATUSES:
      reason = "terminal_door_stop" if gate_code == "STOP" else "terminal_door_error"
      message = (
        f"Named agent-operation terminal door returned {gate_code or raw_status}"
      )
      error = terminal.get("error")
      if isinstance(error, Mapping):
        detail = error.get("message") or error.get("type")
        if detail:
          message = str(detail)
      return _failure_classification(
        reason=reason,
        message=message,
        outcome="blocked",
        semantic="stop" if gate_code == "STOP" else "none",
      )
  return ChildOutcomeClassification(
    outcome=process_classification.outcome,
    succeeded=True,
    error=None,
    semantic="terminal_door",
    execution_status="succeeded",
  )


def classify_declared_door_outcome(
  *,
  declared_terminal_doors: Iterable[str],
  tools_used: Sequence[str],
  door_results: Sequence[Mapping[str, Any]] = (),
  process_outcome: str = "not_assessed",
) -> ChildOutcomeClassification:
  declared = frozenset(declared_terminal_doors)
  if not declared:
    return ChildOutcomeClassification(
      outcome=process_outcome,
      succeeded=True,
      error=None,
      execution_status="succeeded",
    )
  return _named_door_classification(
    declared_terminal_doors=declared,
    tools_used=tools_used,
    door_results=door_results,
    process_classification=ChildOutcomeClassification(
      outcome=process_outcome,
      succeeded=True,
      error=None,
      execution_status="succeeded",
    ),
  )


def terminal_fms_result_disposition(
  result: Any,
) -> Literal["success", "failure"] | None:
  """Classify a structured terminal FMS result independently of its caller.

  FMS error envelopes always carry gate code STOP, so consult recoverability first.
  """

  if not isinstance(result, Mapping):
    return None
  if result.get("is_error") is True:
    return None
  status = str(result.get("status") or "").strip().lower()
  if status in _FAILURE_FMS_STATUSES:
    error = result.get("error")
    recoverable = (
      isinstance(error, Mapping)
      and error.get("recoverable") is True
    )
    is_fms_envelope = (
      "subcommand" in result and "mutation_mode" in result
    )
    return "failure" if is_fms_envelope and not recoverable else None
  if str(result.get("gate_code") or "").strip() == "STOP":
    return "failure"
  if result.get("success") is False:
    return None
  if result.get("error") not in (None, "", {}, []):
    return None
  if status in _TERMINAL_SUCCESS_STATUSES:
    return "success"
  if any(
    isinstance(result.get(field), str)
    and bool(str(result[field]).strip())
    for field in _TERMINAL_RECEIPT_FIELDS
  ):
    return "success"
  if (
    isinstance(result.get("readback"), Mapping)
    or isinstance(result.get("verdict_echo"), Mapping)
  ):
    return "success"
  return None


def declared_terminal_tool_result_disposition(
  *,
  declared_terminal_doors: Iterable[str],
  tool_name: str,
  result: Any,
) -> Literal["success", "failure"] | None:
  """Classify one exact admitted terminal result at the tool boundary."""

  if tool_name not in frozenset(declared_terminal_doors):
    return None
  return terminal_fms_result_disposition(result)


def accepted_declared_terminal_tool_result_value(
  value: Any,
  *,
  declared_terminal_doors: Iterable[str],
) -> dict[str, Any] | None:
  """Validate the exact canonical value projected for parent delivery."""

  if not isinstance(value, Mapping) or set(value) != {
    "tool_name",
    "result",
  }:
    return None
  tool_name = value.get("tool_name")
  result = value.get("result")
  if (
    not isinstance(tool_name, str)
    or not isinstance(result, Mapping)
    or declared_terminal_tool_result_disposition(
      declared_terminal_doors=declared_terminal_doors,
      tool_name=tool_name,
      result=result,
    )
    != "success"
  ):
    return None
  return {
    "tool_name": tool_name,
    "result": dict(result),
  }


def latest_successful_declared_terminal_tool_result(
  entries: Iterable[Any],
  *,
  declared_terminal_doors: Iterable[str],
) -> dict[str, Any] | None:
  """Project the last accepted terminal result from appended tool events."""

  declared = frozenset(declared_terminal_doors)
  accepted: list[dict[str, Any]] = []
  for entry in entries:
    event = getattr(entry, "event", entry)
    if (
      not isinstance(event, Mapping)
      or event.get("type") != "tool_call_complete"
    ):
      continue
    tool_name = event.get("tool_name")
    result = event.get("result")
    dispatch = event.get("dispatch")
    if (
      not isinstance(tool_name, str)
      or tool_name not in declared
      or not isinstance(result, Mapping)
      or not isinstance(dispatch, Mapping)
      or dispatch.get("outcome") != "ok"
      or event.get("is_error") is True
      or event.get("error") is not None
      or event.get("semantic_error") is not None
    ):
      continue
    candidate = accepted_declared_terminal_tool_result_value(
      {"tool_name": tool_name, "result": result},
      declared_terminal_doors=declared,
    )
    if candidate is not None:
      accepted.append(candidate)
  if len(accepted) > 1:
    raise RuntimeError(
      "child lineage contains multiple accepted terminal tool results"
    )
  return accepted[0] if accepted else None


def classify_child_outcome(
  result: Any | None,
  error: dict[str, Any] | None,
  *,
  declared_terminal_doors: Iterable[str] | None = None,
  door_results: Sequence[Mapping[str, Any]] | None = None,
) -> ChildOutcomeClassification:
  if error is not None:
    normalized_error = dict(error)
    child_outcome = normalized_error.get("child_outcome")
    outcome = (
      str(child_outcome)
      if (
        isinstance(child_outcome, str)
        and child_outcome in set(_FAILURE_OUTCOME_BY_STATUS.values())
      )
      else "error"
    )
    if child_outcome is not None and child_outcome != outcome:
      normalized_error["child_outcome"] = outcome
    return ChildOutcomeClassification(
      outcome=outcome,
      succeeded=False,
      error=normalized_error,
      execution_status="failed",
    )

  try:
    task_result = TaskResult.model_validate(result)
  except ValidationError as exc:
    return _failure_classification(
      reason="invalid_child_result",
      message=f"Sub-agent returned an invalid result: {exc.errors()[0]['msg']}",
    )

  execution = task_result.execution
  if execution.status != "succeeded":
    reason = execution.terminal_reason or execution.status
    return _failure_classification(
      reason=reason,
      message=f"Sub-agent ended with {reason}",
      outcome=_FAILURE_OUTCOME_BY_STATUS[execution.status],
      execution_status=execution.status,
    )

  process_classification = ChildOutcomeClassification(
    outcome=(
      task_result.outcome.disposition
      if task_result.outcome is not None
      else "not_assessed"
    ),
    succeeded=True,
    error=None,
    execution_status="succeeded",
  )
  return classify_declared_door_outcome(
    declared_terminal_doors=declared_terminal_doors or (),
    tools_used=task_result.evidence.tools_used,
    door_results=tuple(door_results or ()),
    process_outcome=process_classification.outcome,
  )


def result_response_text(result: Any | None) -> str:
  try:
    task_result = TaskResult.model_validate(result)
  except ValidationError:
    return ""
  projection = task_result.values.projection
  if projection is None or not isinstance(projection.inline_view, dict):
    return ""
  summary = projection.inline_view.get("summary")
  return summary if isinstance(summary, str) else ""


def skill_state_prompt(skill_name: str, previous_state: dict[str, Any]) -> str:
  state_json = json.dumps(previous_state, indent=2, sort_keys=True)
  return (
    "## Persisted Skill State\n"
    f"Previous state for `{skill_name}`:\n"
    "```json\n"
    f"{state_json}\n"
    "```\n\n"
    "Use this state as continuity context when it is relevant. To update the "
    "persisted state, include a final `## STATE_UPDATE_JSON` section containing "
    "a fenced JSON object. Omitted keys keep their previous values."
  )


async def persist_skill_state(
  result: Any | None,
  error: dict[str, Any] | None,
  *,
  agent_name: str | None,
  profile: Any | None,
  persist_state: bool | None = None,
  operation_version: str | None = None,
  skill_state_store: Any | None,
  skill_state_lock: Any,
  effective_model: str,
  extract_state_update_fn: Any,
  result_response_text_fn: Any = result_response_text,
  logger: Any,
) -> None:
  resolved_persist_state = (
    persist_state
    if persist_state is not None
    else bool(profile is not None and profile.persist_state)
  )
  if not (agent_name and resolved_persist_state and skill_state_store is not None):
    return
  resolved_version = (
    operation_version
    if operation_version is not None
    else getattr(profile, "version", None)
  )
  classification = classify_child_outcome(result, error)
  model_state: dict[str, Any] = {}
  if classification.succeeded:
    response_text = result_response_text_fn(result)
    try:
      model_state = extract_state_update_fn(response_text)
    except Exception:
      logger.warning(
        "Failed to extract state update for skill %s",
        agent_name,
        exc_info=True,
      )
  async with skill_state_lock:
    try:
      def _mutate(
        previous_state: dict[str, Any],
      ) -> dict[str, Any]:
        next_state = dict(previous_state)
        if classification.succeeded:
          next_state.update(model_state)
        next_state["last_run"] = datetime.datetime.now(
          datetime.UTC
        ).isoformat()
        next_state["model"] = effective_model
        next_state["run_count"] = (
          int(previous_state.get("run_count", 0) or 0) + 1
        )
        next_state["last_outcome"] = classification.outcome
        outcome_counts = previous_state.get(
          "outcome_counts",
          {},
        )
        if not isinstance(outcome_counts, dict):
          outcome_counts = {}
        else:
          outcome_counts = dict(outcome_counts)
        outcome_counts[classification.outcome] = (
          int(
            outcome_counts.get(
              classification.outcome,
              0,
            )
            or 0
          )
          + 1
        )
        next_state["outcome_counts"] = outcome_counts
        if resolved_version is not None:
          next_state["version"] = resolved_version
        if classification.error is not None:
          next_state["last_error"] = dict(
            classification.error
          )
        else:
          next_state.pop("last_error", None)
        return next_state

      skill_state_store.update(agent_name, _mutate)
    except Exception:
      logger.warning(
        "Failed to persist state for skill %s",
        agent_name,
        exc_info=True,
      )


__all__ = [
  "ChildOutcomeClassification",
  "classify_child_outcome",
  "classify_declared_door_outcome",
  "accepted_declared_terminal_tool_result_value",
  "declared_terminal_doors_from_grant",
  "declared_terminal_tool_result_disposition",
  "latest_successful_declared_terminal_tool_result",
  "persist_skill_state",
  "result_response_text",
  "skill_state_prompt",
  "terminal_fms_result_disposition",
]
