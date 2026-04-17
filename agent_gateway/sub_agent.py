from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from .event_log import EventLog
from .session import GatewaySession
from .skills import SkillLoader
from .task_registry import CoordinatorConfig, ProviderResolver
from .tool_dispatcher import ToolDispatcher

_DEFAULT_EXCLUDED_TOOLS = frozenset({"run_agent", "get_background_result", "send_message"})
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
  parent_session: GatewaySession | None = None,
  skill_loader: SkillLoader | None = None,
  mcp_client: Any,
  needs_approval: Callable[..., bool] | None = None,
  mcp_session_inject_servers: set[str] | None = None,
  mcp_meta_inject_servers: frozenset[str] | None = None,
  user_id: str | None = None,
  credentials_resolver_active: bool = False,
  local_tool_handlers: dict[str, Any] | None = None,
  excluded_tools: set[str] | None = None,
  default_model: str = "claude-sonnet-4-6",
  default_max_turns: int = 15,
  default_timeout: float | None = None,
  default_max_tokens: int = 32000,
  allowed_models: set[str] | None = None,
  on_before_background: Callable[[str | None], None] | None = None,
  on_background_complete: Callable[[Any], Awaitable[None]] | None = None,
  provider_resolver: ProviderResolver | None = None,
  outputs_dir: Path | None = None,
  coordinator_config: CoordinatorConfig | None = None,
):
  """Build the local handler used by the `run_agent` tool.

  The returned handler validates input, optionally resolves a named skill,
  constructs a constrained sub-agent dispatcher, and delegates execution to
  `AgentRunner.spawn_sub_agent()`.
  """
  effective_allowed_models = (
    allowed_models if allowed_models is not None else {"claude-sonnet-4-6", "claude-opus-4-6"}
  )
  effective_coordinator = coordinator_config if coordinator_config is not None and coordinator_config.enabled else None

  async def _handle_run_agent(tool_input: dict[str, Any], **kwargs: Any):
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}
    tool_ctx = kwargs.get("tool_ctx")
    parent_turn_id = getattr(tool_ctx, "tool_call_id", None)

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
    raw_provider = tool_input.get("provider")
    if raw_provider is not None:
      if not isinstance(raw_provider, str):
        return None, {"code": "invalid_input", "message": "provider must be a string"}
      raw_provider = raw_provider.strip() or None
    background = tool_input.get("background", False)
    if not isinstance(background, bool):
      return None, {"code": "invalid_input", "message": "background must be a boolean"}

    call_index = int(kwargs.get("call_index", 0) or 0)

    profile = None
    if agent_name and skill_loader is not None:
      try:
        profile = skill_loader.load(agent_name)
      except FileNotFoundError as exc:
        return None, {"code": "not_found", "message": str(exc)}
      except Exception as exc:
        return None, {"code": "invalid_skill", "message": str(exc)}
    elif agent_name and skill_loader is None:
      return None, {"code": "not_available", "message": "Named agents not available"}

    profile_provider = getattr(profile, "provider", None) if profile is not None else None
    if profile_provider:
      raw_provider = raw_provider or profile_provider
    if raw_provider is None and effective_coordinator is not None and effective_coordinator.default_worker_provider:
      raw_provider = effective_coordinator.default_worker_provider

    effective_provider_resolver = provider_resolver
    if effective_provider_resolver is None and effective_coordinator is not None:
      effective_provider_resolver = effective_coordinator.provider_resolver

    resolved = None
    if raw_provider:
      if effective_provider_resolver is None:
        return None, {
          "code": "provider_not_supported",
          "message": f"Provider '{raw_provider}' requested but no provider_resolver configured",
        }
      try:
        resolved = effective_provider_resolver(raw_provider)
      except Exception as exc:
        return None, {"code": "invalid_provider", "message": str(exc)}

    if profile is not None:
      system_prompt = _SKILL_SYSTEM_PROMPT_TEMPLATE.format(
        skill_prompt=profile.system_prompt,
        date=datetime.date.today().isoformat(),
      )
      effective_max_turns = profile.max_turns if profile.max_turns is not None else default_max_turns
      effective_timeout = profile.timeout if profile.timeout is not None else default_timeout
    else:
      system_prompt = _DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
        date=datetime.date.today().isoformat(),
      )
      effective_max_turns = default_max_turns
      effective_timeout = default_timeout

    if resolved is not None:
      effective_allowed = resolved.allowed_models
      effective_model = (
        raw_model
        or (profile.model if profile else None)
        or (
          effective_coordinator.default_worker_model
          if effective_coordinator is not None
          else None
        )
        or resolved.default_model
        or default_model
      )
    else:
      effective_allowed = effective_allowed_models
      effective_model = (
        raw_model
        or (profile.model if profile else None)
        or (
          effective_coordinator.default_worker_model
          if effective_coordinator is not None
          else None
        )
        or default_model
      )

    if effective_allowed and effective_model not in effective_allowed:
      if profile is not None and resolved is None:
        return None, {
          "code": "invalid_input",
          "message": f"Invalid model '{effective_model}' for skill '{agent_name}'",
        }
      return None, {"code": "invalid_input", "message": f"Invalid model: {effective_model}"}

    effective_excluded = _DEFAULT_EXCLUDED_TOOLS | set(excluded_tools or set())
    if effective_coordinator is not None and effective_coordinator.worker_excluded_tools:
      effective_excluded = effective_excluded | effective_coordinator.worker_excluded_tools
    sub_local = {
      name: handler
      for name, handler in (local_tool_handlers or {}).items()
      if name not in effective_excluded
    }
    sub_log = EventLog()
    sub_dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=sub_local,
      needs_approval=needs_approval,
      event_log=sub_log,
      session_id=getattr(runner, "_full_session_id", ""),
      should_avoid_permission_prompts=True,
      mcp_session_inject_servers=mcp_session_inject_servers,
      mcp_meta_inject_servers=mcp_meta_inject_servers,
      user_id=user_id or getattr(parent_session, "user_id", None),
      credentials_resolver_active=credentials_resolver_active,
    )

    async def _dispatch_sub_agent(_background_input: dict[str, Any], **background_kwargs: Any):
      background_call_index = int(background_kwargs.get("call_index", call_index) or 0)
      task_entry = background_kwargs.get("task_entry")
      sub_session = None
      if parent_session is not None:
        sub_session = GatewaySession(
          session_id=f"sub{background_call_index}:{parent_session.session_id}",
          api_key_hash=parent_session.api_key_hash,
          created_at=parent_session.created_at,
          expires_at=parent_session.expires_at,
          user_id=parent_session.user_id,
          auth_config=parent_session.auth_config,
        )
      return await runner.spawn_sub_agent(
        task,
        provider=resolved.provider if resolved else None,
        auth_config=resolved.auth_config if resolved else None,
        model=effective_model,
        system_prompt=system_prompt,
        dispatcher=sub_dispatcher,
        sub_session=sub_session,
        excluded_tools=effective_excluded,
        max_turns=effective_max_turns,
        timeout=effective_timeout,
        client_timeout=90,
        max_tokens=default_max_tokens,
        call_index=background_call_index,
        parent_turn_id=parent_turn_id,
        task_entry=task_entry,
        on_sub_event=lambda event, _sid: sub_log.append(event),
      )

    if background:
      if outputs_dir is not None and agent_name:
        today = datetime.date.today().isoformat()
        stale_path = outputs_dir / agent_name / f"{today}.md"
        try:
          stale_path.unlink(missing_ok=True)
        except OSError as exc:
          return None, {
            "code": "file_cleanup_failed",
            "message": f"Failed to clean stale output {stale_path}: {exc}",
          }
      return await runner._register_background_task(
        tool_input=dict(tool_input),
        handler=_dispatch_sub_agent,
        agent_name=agent_name,
        on_before_start=(lambda: on_before_background(agent_name)) if on_before_background else None,
        on_complete=on_background_complete,
      )
    return await _dispatch_sub_agent(tool_input, call_index=call_index)

  return _handle_run_agent


def make_get_background_result_handler(runner_ref: list[Any]):
  """Build the local handler used by the ``get_background_result`` tool.

  The returned handler delegates to ``AgentRunner.get_background_result()``,
  which polls or waits for background sub-agent tasks registered via
  ``run_agent(background=true)``.

  Args:
    runner_ref: Single-element list holding the active ``AgentRunner`` (or
      ``None`` before the runner is initialized).

  Returns:
    An async handler with signature ``(tool_input, **kwargs) -> (result, error)``.
  """

  async def _handle_get_background_result(tool_input: dict[str, Any], **kwargs: Any):
    _ = kwargs
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}
    return await runner.get_background_result(tool_input)

  return _handle_get_background_result


def make_send_message_handler(runner_ref: list[Any]):
  """Build send_message handler using runner_ref late binding."""

  async def _handle_send_message(tool_input: dict[str, Any], **kwargs: Any):
    _ = kwargs
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Runner not initialized"}
    registry = getattr(runner, "_task_registry", None)
    if registry is None:
      return None, {"code": "not_available", "message": "Task registry not configured"}

    target = tool_input.get("to", "")
    if isinstance(target, str):
      target = target.strip()
    if not target or not isinstance(target, str):
      return None, {"code": "invalid_input", "message": "'to' is required"}

    message = tool_input.get("message", "")
    if isinstance(message, str):
      message = message.strip()
    if not message or not isinstance(message, str):
      return None, {"code": "invalid_input", "message": "'message' is required"}

    entry = registry.get(target)
    if entry is None:
      from .task_registry import TaskState

      matches = [entry for entry in registry.list_tasks(state=TaskState.RUNNING) if entry.agent_name == target]
      if len(matches) > 1:
        ids = ", ".join(entry.task_id for entry in matches)
        return None, {
          "code": "ambiguous_target",
          "message": f"Multiple running agents named '{target}': {ids}. Use task_id instead.",
        }
      entry = matches[0] if matches else None

    if entry is None:
      return None, {"code": "not_found", "message": f"No running agent: {target}"}
    if entry.completed:
      return None, {"code": "already_completed", "message": f"Agent {target} already finished"}

    await entry.message_inbox.put(message)
    return {"status": "delivered", "task_id": entry.task_id}, None

  return _handle_send_message


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
        "provider": {
          "type": "string",
          "description": "Model provider to use (e.g. 'anthropic', 'openai'). Defaults to parent's provider.",
        },
        "background": {
          "type": "boolean",
          "description": (
            "If true, run the sub-agent in the background and return immediately with a task_id. "
            "Use get_background_result to retrieve the result later."
          ),
          "default": False,
        },
      },
      "required": ["task"],
    },
  }


def make_get_background_result_tool_def() -> dict[str, Any]:
  """Build the public tool schema for ``get_background_result``.

  The tool lets the model poll or wait for background sub-agent tasks
  launched via ``run_agent(background=true)``.  Pass ``task_id='*'`` to
  inspect all tracked background tasks at once.

  Returns:
    A tool-definition dict suitable for inclusion in ``get_tool_definitions``.
  """
  return {
    "name": "get_background_result",
    "description": (
      "Check status or wait for a background sub-agent task. Poll returns immediately. "
      "Use task_id='*' to inspect all tracked background tasks."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "string",
          "description": "Task ID from run_agent(background=true), or '*' for all tasks.",
        },
        "wait": {
          "type": "boolean",
          "description": "If true, wait up to timeout seconds for completion.",
          "default": False,
        },
        "timeout": {
          "type": "number",
          "description": "Maximum seconds to wait (clamped to 120).",
          "default": 60,
        },
      },
      "required": ["task_id"],
    },
  }


def make_send_message_tool_def() -> dict[str, Any]:
  return {
    "name": "send_message",
    "description": (
      "Send a message to a running background agent by task_id or agent name. "
      "Use task_id when multiple agents share a name."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "to": {
          "type": "string",
          "description": "Task ID (e.g. bg_0) or agent name.",
        },
        "message": {
          "type": "string",
          "description": "Message content to deliver to the agent.",
        },
      },
      "required": ["to", "message"],
    },
  }


__all__ = [
  "make_get_background_result_handler",
  "make_get_background_result_tool_def",
  "make_send_message_handler",
  "make_send_message_tool_def",
  "make_run_agent_handler",
  "make_run_agent_tool_def",
]
