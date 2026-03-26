from __future__ import annotations

import datetime
from typing import Any

from .event_log import EventLog
from .skills import SkillLoader
from .tool_dispatcher import ToolDispatcher

_DEFAULT_EXCLUDED_TOOLS = {"run_agent"}
_SKILL_SYSTEM_PROMPT_TEMPLATE = (
  "{skill_prompt}\n\n"
  "Today's date: {date}\n\n"
  "You are a focused sub-agent working on behalf of another agent. Complete the assigned task "
  "thoroughly and return a clear, concise response for the parent agent. If any tool fails or "
  "returns suspicious data, note that explicitly instead of silently proceeding. You cannot "
  "spawn further sub-agents."
)
_DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
  "You are a focused sub-agent working on behalf of another agent. Complete the assigned task "
  "thoroughly and return a clear, concise response for the parent agent. If any tool fails or "
  "returns suspicious data, note that explicitly instead of silently proceeding. You cannot "
  "spawn further sub-agents.\n\n"
  "Today's date: {date}"
)


def make_run_agent_handler(
  runner_ref: list[Any],
  *,
  skill_loader: SkillLoader | None = None,
  mcp_client: Any,
  local_tool_handlers: dict[str, Any] | None = None,
  excluded_tools: set[str] | None = None,
  default_model: str = "claude-sonnet-4-6",
  default_max_turns: int = 15,
  default_timeout: float = 300.0,
  default_max_tokens: int = 32000,
  allowed_models: set[str] | None = None,
):
  """Build the local handler used by the `run_agent` tool.

  The returned handler validates input, optionally resolves a named skill,
  constructs a constrained sub-agent dispatcher, and delegates execution to
  `AgentRunner.spawn_sub_agent()`.
  """
  effective_allowed_models = (
    allowed_models if allowed_models is not None else {"claude-sonnet-4-6", "claude-opus-4-6"}
  )

  async def _handle_run_agent(tool_input: dict[str, Any], **kwargs: Any):
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}

    task = tool_input.get("task", "")
    if not task or not isinstance(task, str):
      return None, {"code": "invalid_input", "message": "task is required"}

    agent_name = tool_input.get("agent")
    if agent_name is not None:
      if not isinstance(agent_name, str):
        return None, {"code": "invalid_input", "message": "agent must be a string"}
      agent_name = agent_name.strip() or None

    raw_model = tool_input.get("model")
    if raw_model is not None and not isinstance(raw_model, str):
      return None, {"code": "invalid_input", "message": "model must be a string"}

    call_index = int(kwargs.get("call_index", 0) or 0)

    if raw_model is not None and effective_allowed_models and raw_model not in effective_allowed_models:
      return None, {"code": "invalid_input", "message": f"Invalid model: {raw_model}"}

    if agent_name and skill_loader is not None:
      try:
        profile = skill_loader.load(agent_name)
      except FileNotFoundError as exc:
        return None, {"code": "not_found", "message": str(exc)}
      except Exception as exc:
        return None, {"code": "invalid_skill", "message": str(exc)}

      effective_model = raw_model or profile.model or default_model
      if effective_allowed_models and effective_model not in effective_allowed_models:
        return None, {
          "code": "invalid_input",
          "message": f"Invalid model '{effective_model}' for skill '{agent_name}'",
        }

      system_prompt = _SKILL_SYSTEM_PROMPT_TEMPLATE.format(
        skill_prompt=profile.system_prompt,
        date=datetime.date.today().isoformat(),
      )
      effective_max_turns = profile.max_turns if profile.max_turns is not None else default_max_turns
      effective_timeout = profile.timeout if profile.timeout is not None else default_timeout
    elif agent_name and skill_loader is None:
      return None, {"code": "not_available", "message": "Named agents not available"}
    else:
      system_prompt = _DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
        date=datetime.date.today().isoformat(),
      )
      effective_model = raw_model or default_model
      if effective_allowed_models and effective_model not in effective_allowed_models:
        return None, {"code": "invalid_input", "message": f"Invalid model: {effective_model}"}
      effective_max_turns = default_max_turns
      effective_timeout = default_timeout

    effective_excluded = _DEFAULT_EXCLUDED_TOOLS | set(excluded_tools or set())
    sub_local = {
      name: handler
      for name, handler in (local_tool_handlers or {}).items()
      if name not in effective_excluded
    }
    sub_log = EventLog()
    sub_dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=sub_local,
      needs_approval=lambda _name, _input, _qualifier: False,
      event_log=sub_log,
      session_id=getattr(runner, "_full_session_id", ""),
    )

    return await runner.spawn_sub_agent(
      task,
      model=effective_model,
      system_prompt=system_prompt,
      dispatcher=sub_dispatcher,
      excluded_tools=effective_excluded,
      max_turns=effective_max_turns,
      timeout=effective_timeout,
      client_timeout=90,
      max_tokens=default_max_tokens,
      call_index=call_index,
      on_sub_event=lambda event, _sid: sub_log.append(event),
    )

  return _handle_run_agent


def make_run_agent_tool_def(skill_loader: SkillLoader | None = None) -> dict[str, Any]:
  """Build the public tool schema for `run_agent`.

  When a `SkillLoader` is provided, the description includes the currently
  available named skills.
  """
  skills = skill_loader.list_skills() if skill_loader else []
  skill_suffix = f" Available agents: {', '.join(skills)}." if skills else ""

  return {
    "name": "run_agent",
    "description": (
      "Spawn a sub-agent to perform a focused task. The sub-agent runs independently "
      "with its own turn budget and returns its final text response. Use this when you "
      "need to perform a substantial task without cluttering the main conversation. "
      "The sub-agent cannot spawn further sub-agents."
      + skill_suffix
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "agent": {
          "type": "string",
          "description": "Named agent profile to use." + (f" One of: {', '.join(skills)}." if skills else ""),
        },
        "task": {
          "type": "string",
          "description": "Instructions for the sub-agent.",
        },
        "model": {
          "type": "string",
          "description": "Optional model override.",
        },
      },
      "required": ["task"],
    },
  }


__all__ = ["make_run_agent_handler", "make_run_agent_tool_def"]
