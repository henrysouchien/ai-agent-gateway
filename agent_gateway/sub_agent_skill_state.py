from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agent_workflow_contracts import AdmittedTask, TaskResult, ToolGrant
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
# A context-bound named operation opens its exact handoff through one granted
# read named ``get_<artifact>_context``; the locator it returns is what the
# terminal door consumes. Same derivation as the ``fms_`` door prefix: read
# off the admitted grant, never enumerated per skill.
_CONTEXT_DOOR_PREFIX = "get_"
_CONTEXT_DOOR_SUFFIX = "_context"
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
  execution_status: str = "failed"
  is_final: bool = True


def declared_terminal_doors_from_grant(grant: ToolGrant | None) -> frozenset[str]:
  """Return the FMS/product doors already admitted on the child's grant."""

  if grant is None:
    return frozenset()
  return frozenset(
    entry.tool_id
    for entry in grant.tools
    if _is_declared_terminal_door(entry.tool_id)
  )


def terminal_result_doors(admitted_task: AdmittedTask | None) -> frozenset[str]:
  """Return the declared doors whose accepted result is the child's result.

  An accepted result from one of these doors ends the run and is what the
  parent reads. An operation that persists several units and then reports
  declares ``terminal_on_persist: false``: none of its doors ends the run, its
  narrative is its result, and a refused door still settles it (the refusal
  reads the grant's doors, not these).
  """

  if admitted_task is None or not admitted_task.operation.terminal_on_persist:
    return frozenset()
  return declared_terminal_doors_from_grant(admitted_task.tool_grant)


def _is_declared_terminal_door(tool_id: str) -> bool:
  if not tool_id.startswith(_FMS_DOOR_PREFIX):
    return False
  return not tool_id.startswith(_FMS_NON_TERMINAL_PREFIXES)


def declared_context_doors_from_grant(grant: ToolGrant | None) -> frozenset[str]:
  """Return the context doors the child's admitted grant declares."""

  if grant is None:
    return frozenset()
  return frozenset(
    entry.tool_id
    for entry in grant.tools
    if entry.effect == "read"
    and entry.tool_id.startswith(_CONTEXT_DOOR_PREFIX)
    and entry.tool_id.endswith(_CONTEXT_DOOR_SUFFIX)
  )


def _door_refused(result: Any) -> bool:
  if not isinstance(result, Mapping):
    return False
  return (
    str(result.get("gate_code") or "").strip() == "STOP"
    or str(result.get("status") or "").strip().lower() in _FAILURE_FMS_STATUSES
  )


def latest_declared_door_refusal(
  declared_doors: Iterable[str],
  tool_events: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
  """Return the child's last declared-door answer, verbatim, when it refused.

  The doors are the grant's terminal and context doors. When the last of them
  the child called answered ``STOP`` or an error status, that answer — with
  its own ``next_call`` — is what the child settles on and the parent reads.
  A later accepted call supersedes an earlier refusal, and a transport
  failure carries no door answer.
  """

  doors = frozenset(declared_doors)
  last: Mapping[str, Any] | None = None
  for event in tool_events:
    if (
      event.get("type") == "tool_call_complete"
      and event.get("tool_name") in doors
    ):
      last = event
  if last is None or not _door_refused(last.get("result")):
    return None
  return {
    "tool_name": str(last["tool_name"]),
    "result": dict(last["result"]),
  }


def _failure_classification(
  *,
  reason: str,
  message: str,
  outcome: str = "error",
  execution_status: str = "failed",
) -> ChildOutcomeClassification:
  return ChildOutcomeClassification(
    outcome=outcome,
    succeeded=False,
    error={
      "code": reason,
      "message": message,
      "child_outcome": outcome,
      "next_call": None,
    },
    execution_status=execution_status,
  )


def _door_result_classification(
  *,
  door_results: Sequence[Mapping[str, Any]],
  process_outcome: str,
) -> ChildOutcomeClassification:
  """Record the child's last FMS door answer on its skill lifecycle event.

  The lifecycle event mirrors what the door returned; the door's own error is
  the event's error. The ``run_agent`` settlement does not read this: it
  settles on the door answer itself (``latest_declared_door_refusal``).
  """

  terminal = door_results[-1] if door_results else None
  if terminal is not None and _door_refused(terminal):
    gate_code = str(terminal.get("gate_code") or "").strip()
    raw_status = str(terminal.get("status") or "").strip().lower()
    message = f"{terminal.get('tool_name')} returned {gate_code or raw_status}"
    error = terminal.get("error")
    if isinstance(error, Mapping):
      detail = error.get("message") or error.get("type")
      if detail:
        message = str(detail)
    return ChildOutcomeClassification(
      outcome="blocked",
      succeeded=False,
      error={
        "code": gate_code or raw_status,
        "message": message,
        "child_outcome": "blocked",
      },
    )
  return ChildOutcomeClassification(
    outcome=process_outcome,
    succeeded=True,
    error=None,
    execution_status="succeeded",
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

  task_result = TaskResult.model_validate(result)

  execution = task_result.execution
  if execution.status == "awaiting_approval":
    return ChildOutcomeClassification(
      outcome="awaiting_approval",
      succeeded=False,
      error=None,
      execution_status=execution.status,
      is_final=False,
    )
  if execution.status != "succeeded":
    reason = execution.terminal_reason or execution.status
    return _failure_classification(
      reason=reason,
      message=f"Sub-agent ended with {reason}",
      outcome=_FAILURE_OUTCOME_BY_STATUS[execution.status],
      execution_status=execution.status,
    )

  declared = frozenset(declared_terminal_doors or ())
  return _door_result_classification(
    door_results=tuple(
      door_result
      for door_result in door_results or ()
      if door_result.get("tool_name") in declared
    ),
    process_outcome=(
      task_result.outcome.disposition
      if task_result.outcome is not None
      else "not_assessed"
    ),
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
  if not classification.is_final:
    return
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
  "latest_declared_door_refusal",
  "accepted_declared_terminal_tool_result_value",
  "declared_context_doors_from_grant",
  "declared_terminal_doors_from_grant",
  "declared_terminal_tool_result_disposition",
  "latest_successful_declared_terminal_tool_result",
  "persist_skill_state",
  "result_response_text",
  "skill_state_prompt",
  "terminal_fms_result_disposition",
  "terminal_result_doors",
]
