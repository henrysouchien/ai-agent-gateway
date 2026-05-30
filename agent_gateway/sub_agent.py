from __future__ import annotations

import datetime
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, FrozenSet

from ._provider_utils import _get_allowed_models_for_provider_name
from .events import SkillRunStartedEvent, VerdictEmittedEvent, event_to_dict
from .event_log import EventLog
from .runner import _derive_sub_agent_id
from .session import GatewaySession
from .skills import SkillLoader
from .task_registry import CoordinatorConfig, ParentMessage, ProviderResolver
from .tool_dispatcher import ToolDispatcher
from .transcript import (
  build_synthetic_tool_results,
  detect_orphan_tool_uses,
  place_resume_messages,
  reconstruct_messages_for_task,
  reconstruct_parent_messages,
)
from .verdict_extractor import extract_verdict_payload

_DEFAULT_EXCLUDED_TOOLS = frozenset({"run_agent", "get_background_result", "send_message"})
ExcludedToolsResolver = Callable[[], FrozenSet[str]]
NeedsApprovalResolver = Callable[[FrozenSet[str]], Callable[..., bool] | None]
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
_RUN_AGENT_DESCRIPTION = (
  "Spawn a focused sub-agent. Pass `agent` for a named skill workflow; omit for a generic "
  "sub-agent. Sub-agents run independently with their own turn budget, cannot spawn further "
  "sub-agents, and cannot access Excel tools or trading/order-management tools."
)
_RESUME_AGENT_DESCRIPTION = (
  "Resume an interrupted background sub-agent. Only resumable skills can be resumed; "
  "check skill `resumable` flag in get_background_result response. Returns new task_id."
)
_CONTEXT_TICKER_RE = re.compile(r"\b([A-Z0-9]{1,6}(?:\.[A-Z]{1,2})?)\b")
_TICKER_STOPWORDS = {
  "THE",
  "AND",
  "FOR",
  "NOT",
  "ALL",
  "HAS",
  "ARE",
  "WAS",
  "USE",
  "RUN",
  "INC",
  "LLC",
  "LTD",
  "PLC",
  "ETF",
  "USD",
  "YOY",
  "QOQ",
  "EPS",
  "FCF",
  "CEO",
  "CFO",
  "COO",
  "CTO",
  "IPO",
  "SEC",
  "FY",
  "TTM",
  "BPS",
  "NAV",
  "GDP",
  "CPI",
  "PPI",
  "FMP",
  "EDGAR",
  "MCP",
  "API",
  "SQL",
  "CLI",
}


def _render_agent_param_description(entries: list[tuple[str, str]]) -> str:
  base = "Named agent profile to use. Omit to run without a skill."
  if not entries:
    return base
  lines = [base, "", "Available agents:"]
  lines.extend(f"- {name}: {description}" for name, description in entries)
  return "\n".join(lines)


def _extract_ticker_from_task(task: str) -> str:
  for match in _CONTEXT_TICKER_RE.finditer(task):
    candidate = match.group(1).strip().upper()
    if candidate.startswith("20") or candidate.startswith("Q"):
      continue
    if candidate not in _TICKER_STOPWORDS:
      return candidate
  return ""


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
  user_email: str | None = None,
  parent_user_id: str | None = None,
  parent_user_email: str | None = None,
  credentials_resolver_active: bool = False,
  local_tool_handlers: dict[str, Any] | None = None,
  excluded_tools: set[str] | None = None,
  default_model: str = "claude-opus-4-7",
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
    allowed_models if allowed_models is not None else _get_allowed_models_for_provider_name("anthropic")
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
    context_ticker = ""
    skill_run_id: str | None = None
    skill_run_started_emitted = False
    verdict_emitted = False
    if agent_name and skill_loader is not None:
      try:
        profile = skill_loader.load(agent_name)
      except FileNotFoundError as exc:
        return None, {"code": "not_found", "message": str(exc)}
      except Exception as exc:
        return None, {"code": "invalid_skill", "message": str(exc)}
      if not profile.agent_callable:
        return None, {
          "code": "invalid_skill",
          "message": f"Agent '{agent_name}' is not callable. Choose a callable named agent or omit agent.",
        }
      context_ticker = _extract_ticker_from_task(task)
      skill_run_id = secrets.token_hex(16)
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
    effective_parent_user_id = (
      parent_user_id
      or user_id
      or getattr(parent_session, "user_id", None)
    )
    effective_parent_user_email = (
      parent_user_email
      or user_email
      or getattr(parent_session, "user_email", None)
    )

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
      user_id=effective_parent_user_id,
      risk_user_id=getattr(parent_session, "risk_user_id", None),
      channel=getattr(parent_session, "channel", None),
      role=getattr(parent_session, "role", None),
      credentials_resolver_active=credentials_resolver_active,
    )

    def _emit_parent_event(event: dict[str, Any]) -> None:
      emit = getattr(tool_ctx, "emit", None)
      if callable(emit):
        emit(event)

    def _emit_skill_run_started() -> None:
      nonlocal skill_run_started_emitted
      if skill_run_started_emitted or not (skill_run_id and profile is not None and agent_name):
        return
      _emit_parent_event(
        event_to_dict(
          SkillRunStartedEvent(
            skill_run_id=skill_run_id,
            skill=profile.name,
            ticker=context_ticker,
            ts=time.time(),
          )
        )
      )
      skill_run_started_emitted = True

    def _emit_verdict_if_present() -> None:
      nonlocal verdict_emitted
      if verdict_emitted or not (skill_run_id and profile is not None and agent_name):
        return
      verdict = extract_verdict_payload(sub_log.entries)
      if verdict is None:
        return
      _emit_parent_event(
        event_to_dict(
          VerdictEmittedEvent(
            skill_run_id=skill_run_id,
            skill=profile.name,
            ticker=context_ticker,
            verdict_token=verdict.verdict_token,
            confidence=verdict.confidence,
            materiality_cushion=verdict.materiality_cushion,
            one_line_summary=verdict.one_line_summary,
            ts=time.time(),
          )
        )
      )
      verdict_emitted = True

    async def _dispatch_sub_agent(_background_input: dict[str, Any], **background_kwargs: Any):
      background_call_index = int(background_kwargs.get("call_index", call_index) or 0)
      task_entry = background_kwargs.get("task_entry")
      sub_session = None
      if parent_session is not None:
        sub_session = GatewaySession(
          session_id=_derive_sub_agent_id(parent_session, background_call_index),
          api_key_hash=parent_session.api_key_hash,
          created_at=parent_session.created_at,
          expires_at=parent_session.expires_at,
          user_id=effective_parent_user_id or parent_session.user_id,
          user_email=effective_parent_user_email,
          risk_user_id=getattr(parent_session, "risk_user_id", 0),
          role=getattr(parent_session, "role", "owner"),
          auth_config=parent_session.auth_config,
          channel=getattr(parent_session, "channel", None),
          is_public=getattr(parent_session, "is_public", False),
        )
      _emit_skill_run_started()
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
      enriched_tool_input = dict(tool_input)
      if "resumable" in enriched_tool_input:
        raw_resumable = enriched_tool_input["resumable"]
        if not isinstance(raw_resumable, bool):
          return None, {
            "code": "invalid_input",
            "message": f"resumable must be a bool, got {type(raw_resumable).__name__}: {raw_resumable!r}",
          }
      elif profile is not None:
        enriched_tool_input["resumable"] = profile.resumable

      async def _on_background_complete(bg_task: Any) -> None:
        _emit_verdict_if_present()
        if on_background_complete is not None:
          await on_background_complete(bg_task)

      return await runner._register_background_task(
        tool_input=enriched_tool_input,
        handler=_dispatch_sub_agent,
        agent_name=agent_name,
        parent_turn_id=parent_turn_id,
        on_before_start=(lambda: on_before_background(agent_name)) if on_before_background else None,
        on_complete=_on_background_complete if (skill_run_id or on_background_complete) else None,
      )
    result, error = await _dispatch_sub_agent(tool_input, call_index=call_index)
    _emit_verdict_if_present()
    return result, error

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


def make_resume_handler(
  runner_ref: list[Any],
  *,
  parent_session: GatewaySession | None = None,
  skill_loader: SkillLoader | None = None,
  mcp_client: Any,
  needs_approval: Callable[..., bool] | None = None,
  needs_approval_resolver: NeedsApprovalResolver | None = None,
  mcp_session_inject_servers: set[str] | None = None,
  mcp_meta_inject_servers: frozenset[str] | None = None,
  user_id: str | None = None,
  credentials_resolver_active: bool = False,
  local_tool_handlers: dict[str, Any] | None = None,
  excluded_tools_resolver: ExcludedToolsResolver | None = None,
  excluded_tools: set[str] | None = None,
  default_model: str = "claude-opus-4-7",
  default_max_turns: int = 15,
  default_timeout: float | None = None,
  default_max_tokens: int = 32000,
  allowed_models: set[str] | None = None,
  provider_resolver: ProviderResolver | None = None,
  coordinator_config: CoordinatorConfig | None = None,
):
  if excluded_tools is not None:
    import warnings

    warnings.warn(
      "excluded_tools= is deprecated; use excluded_tools_resolver= so resume reapplies current policy.",
      DeprecationWarning,
      stacklevel=2,
    )
    if excluded_tools_resolver is None:
      captured_excluded = frozenset(excluded_tools)
      excluded_tools_resolver = lambda: captured_excluded
  if excluded_tools_resolver is None:
    raise ValueError("make_resume_handler requires excluded_tools_resolver")

  effective_allowed_models = (
    allowed_models if allowed_models is not None else _get_allowed_models_for_provider_name("anthropic")
  )
  effective_coordinator = coordinator_config if coordinator_config is not None and coordinator_config.enabled else None

  async def _handle_resume(tool_input: dict[str, Any], **kwargs: Any):
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}
    if skill_loader is None:
      return None, {"code": "not_available", "message": "Named agents not available"}
    durable_log = getattr(runner, "_agent_session_log", None)
    if durable_log is None:
      return None, {"code": "not_available", "message": "Durable session log not configured"}

    raw_task_id = tool_input.get("task_id")
    if not isinstance(raw_task_id, str) or not raw_task_id.strip():
      return None, {"code": "invalid_input", "message": "task_id is required"}
    task_id = raw_task_id.strip()
    additional_context = tool_input.get("additional_context")
    if additional_context is not None and not isinstance(additional_context, str):
      return None, {"code": "invalid_input", "message": "additional_context must be a string"}

    await runner._rebuild_task_registry_from_log()
    entry = runner._task_registry.get(task_id)
    if entry is None:
      entry = await runner._lookup_task_in_log(task_id)
    if entry is None:
      return None, {"code": "not_found", "message": f"Unknown background task: {task_id}"}
    if getattr(entry, "state", None).value != "interrupted":
      return None, {"code": "not_interrupted", "message": f"Task {task_id} is not interrupted"}

    depth = await runner._resume_chain_depth(task_id)
    max_depth = getattr(runner, "_max_resume_chain_depth", 3)
    if depth >= max_depth:
      return None, {
        "code": "max_resume_chain_depth",
        "message": f"Resume chain depth limit reached ({max_depth}) for {task_id}",
      }

    agent_name = entry.agent_name
    if not agent_name:
      return None, {"code": "not_resumable", "message": f"Task {task_id} was not started with a resumable skill"}
    try:
      profile = skill_loader.load(agent_name)
    except FileNotFoundError as exc:
      return None, {"code": "not_found", "message": str(exc)}
    except Exception as exc:
      return None, {"code": "invalid_skill", "message": str(exc)}
    if not profile.agent_callable:
      return None, {"code": "invalid_skill", "message": f"Agent '{agent_name}' is not callable"}
    if not profile.resumable:
      return None, {"code": "not_resumable", "message": f"Agent '{agent_name}' is not resumable"}

    transcript = await reconstruct_messages_for_task(durable_log, task_id)
    orphan_ids = detect_orphan_tool_uses(transcript)
    synthetic_results = build_synthetic_tool_results(orphan_ids)
    parent_messages = await reconstruct_parent_messages(
      durable_log,
      task_id,
      datetime.datetime.now(datetime.UTC).timestamp(),
    )
    reconstructed_messages = place_resume_messages(
      transcript,
      synthetic_results,
      parent_messages,
      additional_context,
    )

    raw_provider = tool_input.get("provider")
    if raw_provider is not None:
      if not isinstance(raw_provider, str):
        return None, {"code": "invalid_input", "message": "provider must be a string"}
      raw_provider = raw_provider.strip() or None
    profile_provider = getattr(profile, "provider", None)
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

    raw_model = tool_input.get("model")
    if raw_model is not None and not isinstance(raw_model, str):
      return None, {"code": "invalid_input", "message": "model must be a string"}
    if resolved is not None:
      effective_allowed = resolved.allowed_models
      effective_model = (
        raw_model
        or profile.model
        or (effective_coordinator.default_worker_model if effective_coordinator is not None else None)
        or resolved.default_model
        or default_model
      )
    else:
      effective_allowed = effective_allowed_models
      effective_model = (
        raw_model
        or profile.model
        or (effective_coordinator.default_worker_model if effective_coordinator is not None else None)
        or default_model
      )
    if effective_allowed and effective_model not in effective_allowed:
      return None, {"code": "invalid_input", "message": f"Invalid model '{effective_model}' for skill '{agent_name}'"}

    system_prompt = _SKILL_SYSTEM_PROMPT_TEMPLATE.format(
      skill_prompt=profile.system_prompt,
      date=datetime.date.today().isoformat(),
    )
    effective_max_turns = profile.max_turns if profile.max_turns is not None else default_max_turns
    effective_timeout = profile.timeout if profile.timeout is not None else default_timeout
    effective_excluded = set(_DEFAULT_EXCLUDED_TOOLS | excluded_tools_resolver())
    if effective_coordinator is not None and effective_coordinator.worker_excluded_tools:
      effective_excluded = effective_excluded | effective_coordinator.worker_excluded_tools
    child_needs_approval = (
      needs_approval_resolver(frozenset(effective_excluded))
      if needs_approval_resolver is not None
      else needs_approval
    )
    sub_local = {
      name: handler
      for name, handler in (local_tool_handlers or {}).items()
      if name not in effective_excluded
    }
    sub_log = EventLog()
    sub_dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=sub_local,
      needs_approval=child_needs_approval,
      event_log=sub_log,
      session_id=getattr(runner, "_full_session_id", ""),
      should_avoid_permission_prompts=True,
      mcp_session_inject_servers=mcp_session_inject_servers,
      mcp_meta_inject_servers=mcp_meta_inject_servers,
      user_id=user_id or getattr(parent_session, "user_id", None),
      risk_user_id=getattr(parent_session, "risk_user_id", None),
      channel=getattr(parent_session, "channel", None),
      role=getattr(parent_session, "role", None),
      credentials_resolver_active=credentials_resolver_active,
    )
    tool_ctx = kwargs.get("tool_ctx")
    parent_turn_id = getattr(tool_ctx, "tool_call_id", None)
    call_index = int(kwargs.get("call_index", 0) or 0)

    async def _dispatch_resume(_background_input: dict[str, Any], **background_kwargs: Any):
      background_call_index = int(background_kwargs.get("call_index", call_index) or 0)
      task_entry = background_kwargs.get("task_entry")
      sub_session = None
      if parent_session is not None:
        sub_session = GatewaySession(
          session_id=_derive_sub_agent_id(parent_session, background_call_index),
          api_key_hash=parent_session.api_key_hash,
          created_at=parent_session.created_at,
          expires_at=parent_session.expires_at,
          user_id=parent_session.user_id,
          user_email=parent_session.user_email,
          risk_user_id=getattr(parent_session, "risk_user_id", 0),
          role=getattr(parent_session, "role", "owner"),
          auth_config=parent_session.auth_config,
          channel=getattr(parent_session, "channel", None),
          is_public=getattr(parent_session, "is_public", False),
        )
      return await runner.resume_sub_agent(
        original_task_id=task_id,
        reconstructed_messages=reconstructed_messages,
        parent_messages=parent_messages,
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

    result, error = await runner._register_background_task(
      tool_input={
        "task_id": task_id,
        "agent": agent_name,
        "model": effective_model,
        "provider": raw_provider,
        "resumable": True,
      },
      handler=_dispatch_resume,
      agent_name=agent_name,
      parent_turn_id=parent_turn_id,
      original_task_id=task_id,
    )
    if error is not None:
      return None, error
    result = dict(result or {})
    result["original_task_id"] = task_id
    result["resumed_from"] = task_id
    result["resume_chain_depth"] = depth + 1
    return result, None

  return _handle_resume


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

    message_id = str(tool_input.get("message_id") or uuid.uuid4())
    if message_id in entry.delivered_messages:
      return {"status": "delivered", "task_id": entry.task_id}, None

    sent_at = datetime.datetime.now(datetime.UTC).timestamp()
    append_durable_event = getattr(runner, "_append_durable_event", None)
    if append_durable_event is not None:
      metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
      await append_durable_event(
        {
          "type": "parent_message_sent",
          "task_id": entry.task_id,
          "owner_runner_id": metadata.get("owner_runner_id", getattr(runner, "_runner_id", None)),
          "owner_role": metadata.get("owner_role", getattr(runner, "_role", None)),
          "sub_agent_id": metadata.get("sub_agent_id"),
          "parent_turn_id": metadata.get("parent_turn_id"),
          "call_index": metadata.get("call_index"),
          "task_type": metadata.get("task_type", "background"),
          "provider_name": metadata.get("provider_name", entry.provider_name),
          "model": metadata.get("model", entry.model),
          "message_id": message_id,
          "sender": {
            "session_id": getattr(runner, "_full_session_id", None),
            "user_id": getattr(runner, "_usage_user_id", None),
          },
          "sent_at": sent_at,
          "message": message,
        }
      )
    await entry.message_inbox.put(ParentMessage(message_id=message_id, text=message, sent_at=sent_at))
    entry.delivered_messages.add(message_id)
    return {"status": "delivered", "task_id": entry.task_id}, None

  return _handle_send_message


def make_run_agent_tool_def(skill_loader: SkillLoader | None = None) -> dict[str, Any]:
  """Build the public tool schema for `run_agent`.

  When a `SkillLoader` is provided, the agent parameter description includes
  the currently available callable named skills.
  """
  skills = skill_loader.list_callable_skills_with_descriptions() if skill_loader else []

  return {
    "name": "run_agent",
    "description": _RUN_AGENT_DESCRIPTION,
    "input_schema": {
      "type": "object",
      "properties": {
        "agent": {
          "type": "string",
          "description": _render_agent_param_description(skills),
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


def make_resume_tool_def() -> dict[str, Any]:
  return {
    "name": "resume_background_agent",
    "description": _RESUME_AGENT_DESCRIPTION,
    "input_schema": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "string",
          "description": "Interrupted task to resume.",
        },
        "additional_context": {
          "type": "string",
          "description": "Optional message appended after transcript replay to guide continuation.",
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
  "make_resume_handler",
  "make_resume_tool_def",
  "make_run_agent_handler",
  "make_run_agent_tool_def",
]
