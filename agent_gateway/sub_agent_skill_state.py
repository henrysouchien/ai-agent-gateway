from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any

from agent_workflow_contracts import TaskResult
from pydantic import ValidationError

_FAILURE_OUTCOME_BY_STATUS = {
  "failed": "error",
  "interrupted": "interrupted",
  "cancelled": "interrupted",
  "skipped": "skipped",
}


@dataclass(frozen=True)
class ChildOutcomeClassification:
  outcome: str
  succeeded: bool
  error: dict[str, Any] | None


def _failure_classification(
  *,
  reason: str,
  message: str,
  outcome: str = "error",
) -> ChildOutcomeClassification:
  return ChildOutcomeClassification(
    outcome=outcome,
    succeeded=False,
    error={
      "code": reason,
      "message": message,
      "child_outcome": outcome,
    },
  )


def classify_child_outcome(
  result: Any | None,
  error: dict[str, Any] | None,
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
    )

  return ChildOutcomeClassification(
    outcome=(
      task_result.outcome.disposition
      if task_result.outcome is not None
      else "not_assessed"
    ),
    succeeded=True,
    error=None,
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
  skill_state_store: Any | None,
  skill_state_lock: Any,
  effective_model: str,
  extract_state_update_fn: Any,
  result_response_text_fn: Any = result_response_text,
  logger: Any,
) -> None:
  if not (agent_name and profile is not None and profile.persist_state and skill_state_store is not None):
    return
  classification = classify_child_outcome(result, error)
  model_state: dict[str, Any] = {}
  if classification.succeeded:
    response_text = result_response_text_fn(result)
    try:
      model_state = extract_state_update_fn(response_text)
    except Exception:
      logger.warning("Failed to extract state update for skill %s", profile.name, exc_info=True)
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
        if profile.version is not None:
          next_state["version"] = profile.version
        if classification.error is not None:
          next_state["last_error"] = dict(
            classification.error
          )
        else:
          next_state.pop("last_error", None)
        return next_state

      skill_state_store.update(profile.name, _mutate)
    except Exception:
      logger.warning("Failed to persist state for skill %s", profile.name, exc_info=True)


__all__ = [
  "ChildOutcomeClassification",
  "classify_child_outcome",
  "persist_skill_state",
  "result_response_text",
  "skill_state_prompt",
]
