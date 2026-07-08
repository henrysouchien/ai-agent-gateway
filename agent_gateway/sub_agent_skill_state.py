from __future__ import annotations

import datetime
import json
from typing import Any


def result_response_text(result: Any | None) -> str:
  if isinstance(result, dict):
    response = result.get("response")
    return response if isinstance(response, str) else ""
  if result is None:
    return ""
  return str(result)


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
  response_text = result_response_text_fn(result)
  try:
    model_state = extract_state_update_fn(response_text)
  except Exception:
    logger.warning("Failed to extract state update for skill %s", profile.name, exc_info=True)
    model_state = {}
  async with skill_state_lock:
    try:
      previous_state = skill_state_store.get(profile.name)
      next_state = dict(previous_state)
      next_state.update(model_state)
      next_state["last_run"] = datetime.datetime.now(datetime.UTC).isoformat()
      next_state["model"] = effective_model
      next_state["run_count"] = int(previous_state.get("run_count", 0) or 0) + 1
      if profile.version is not None:
        next_state["version"] = profile.version
      if error:
        next_state["last_error"] = dict(error)
      else:
        next_state.pop("last_error", None)
      skill_state_store.set(profile.name, next_state)
    except Exception:
      logger.warning("Failed to persist state for skill %s", profile.name, exc_info=True)


__all__ = [
  "persist_skill_state",
  "result_response_text",
  "skill_state_prompt",
]
