from __future__ import annotations

import asyncio
import datetime
import json
import logging
import secrets
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from .autonomous_output import extract_state_update
from ._provider_utils import _get_allowed_models_for_provider_name, sub_agent_default_model
from .events import SkillRunStartedEvent, event_to_dict
from .event_log import EventLog
from .runner import _derive_sub_agent_id
from .session import GatewaySession
from .skill_result_events import build_skill_result_captured_event
from .skills import SkillLoader, SkillStateStore, resolve_blocks
from .sub_agent_helpers import (
  DEFAULT_SUB_AGENT_TIMEOUT_SECONDS as DEFAULT_SUB_AGENT_TIMEOUT_SECONDS,
  ExcludedToolsResolver as ExcludedToolsResolver,
  MutationModeExclusionsApplier as MutationModeExclusionsApplier,
  NeedsApprovalResolver as NeedsApprovalResolver,
  _ARTIFACT_EMIT_TOOLS as _ARTIFACT_EMIT_TOOLS,
  _CONTEXT_TICKER_RE as _CONTEXT_TICKER_RE,
  _DEFAULT_EXCLUDED_TOOLS as _DEFAULT_EXCLUDED_TOOLS,
  _DEFAULT_SYSTEM_PROMPT_TEMPLATE as _DEFAULT_SYSTEM_PROMPT_TEMPLATE,
  _RESEARCH_FILE_ID_RE as _RESEARCH_FILE_ID_RE,
  _RESUME_AGENT_DESCRIPTION as _RESUME_AGENT_DESCRIPTION,
  _RUN_AGENT_DESCRIPTION as _RUN_AGENT_DESCRIPTION,
  _SKILL_SYSTEM_PROMPT_TEMPLATE as _SKILL_SYSTEM_PROMPT_TEMPLATE,
  _TICKER_STOPWORDS as _TICKER_STOPWORDS,
  _artifact_storage_user_id as _artifact_storage_user_id,
  _dashboard_artifact_scope as _dashboard_artifact_scope,
  _dashboard_artifact_ticker as _dashboard_artifact_ticker,
  _entry_child_budget_usd as _entry_child_budget_usd,
  _extract_research_file_id_from_resume_messages as _extract_research_file_id_from_resume_messages,
  _extract_research_file_id_from_task as _extract_research_file_id_from_task,
  _extract_ticker_from_resume_messages as _extract_ticker_from_resume_messages,
  _extract_ticker_from_task as _extract_ticker_from_task,
  _html_artifact_scope as _html_artifact_scope,
  _html_artifact_ticker as _html_artifact_ticker,
  _install_emit_dashboard_artifact_handler as _install_emit_dashboard_artifact_handler,
  _install_emit_html_artifact_handler as _install_emit_html_artifact_handler,
  _message_content_text as _message_content_text,
  _optional_research_file_id as _optional_research_file_id,
  _render_agent_param_description as _render_agent_param_description,
  _skill_extra_excluded_tool_names as _skill_extra_excluded_tool_names,
  _skill_html_excluded_tools as _skill_html_excluded_tools,
  make_get_background_result_tool_def as make_get_background_result_tool_def,
  make_resume_tool_def as make_resume_tool_def,
  make_run_agent_tool_def as make_run_agent_tool_def,
  make_send_message_tool_def as make_send_message_tool_def,
)
from .sub_agent_messages import make_send_message_handler as _make_send_message_handler
from . import sub_agent_tool_definitions as _sub_agent_tool_definitions
from .task_registry import CoordinatorConfig, ProviderResolver
from .tool_dispatcher import ToolDispatcher
from .transcript import (
  build_synthetic_tool_results,
  detect_orphan_tool_uses,
  place_resume_messages,
  reconstruct_messages_for_task,
  reconstruct_parent_messages,
)


log = logging.getLogger("agent_gateway.sub_agent")


def _child_tool_definitions_getter(
  *,
  runner: Any,
  mcp_client: Any,
  excluded_tools: set[str],
  extra_tool_definitions: list[dict[str, Any]] | None = None,
) -> Callable[[], list[dict[str, Any]]] | None:
  return _sub_agent_tool_definitions.child_tool_definitions_getter(
    runner=runner,
    mcp_client=mcp_client,
    excluded_tools=excluded_tools,
    extra_tool_definitions=extra_tool_definitions,
  )


def _artifact_emit_tool_definitions(installed_tool_names: set[str]) -> list[dict[str, Any]]:
  return _sub_agent_tool_definitions.artifact_emit_tool_definitions(
    installed_tool_names,
    artifact_emit_tools=_ARTIFACT_EMIT_TOOLS,
  )


def _result_response_text(result: Any | None) -> str:
  if isinstance(result, dict):
    response = result.get("response")
    return response if isinstance(response, str) else ""
  if result is None:
    return ""
  return str(result)


def _skill_state_prompt(skill_name: str, previous_state: dict[str, Any]) -> str:
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
  default_timeout: float | None = DEFAULT_SUB_AGENT_TIMEOUT_SECONDS,
  default_max_tokens: int = 64000,
  allowed_models: set[str] | None = None,
  on_before_background: Callable[[str | None], None] | None = None,
  on_background_complete: Callable[[Any], Awaitable[None]] | None = None,
  provider_resolver: ProviderResolver | None = None,
  outputs_dir: Path | None = None,
  skill_state_store: SkillStateStore | None = None,
  coordinator_config: CoordinatorConfig | None = None,
  approval_key_qualifier: Callable[[str, dict[str, Any]], str] | None = None,
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
  skill_state_lock = asyncio.Lock()

  async def _handle_run_agent(tool_input: dict[str, Any], **kwargs: Any):
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}
    tool_ctx = kwargs.get("tool_ctx")
    parent_turn_id = getattr(tool_ctx, "tool_call_id", None)

    task = tool_input.get("task", "")
    if not task or not isinstance(task, str):
      return None, {"code": "invalid_input", "message": "task is required"}
    context_research_file_id = _extract_research_file_id_from_task(task)

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
      skill_prompt = resolve_blocks(profile.system_prompt, skill_loader.skills_dir / "_blocks")
      system_prompt = _SKILL_SYSTEM_PROMPT_TEMPLATE.format(
        skill_prompt=skill_prompt,
        date=datetime.date.today().isoformat(),
      )
      if profile.persist_state and skill_state_store is not None:
        try:
          previous_state = skill_state_store.get(profile.name)
        except Exception:
          log.warning("Failed to load persisted state for skill %s", profile.name, exc_info=True)
          previous_state = {}
        system_prompt = f"{system_prompt}\n\n{_skill_state_prompt(profile.name, previous_state)}"
      effective_max_turns = profile.max_turns if profile.max_turns is not None else default_max_turns
      effective_timeout = profile.timeout if profile.timeout is not None else default_timeout
      effective_max_tokens = profile.max_tokens if profile.max_tokens is not None else default_max_tokens
    else:
      system_prompt = _DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
        date=datetime.date.today().isoformat(),
      )
      effective_max_turns = default_max_turns
      effective_timeout = default_timeout
      effective_max_tokens = default_max_tokens

    if resolved is not None:
      effective_allowed = resolved.allowed_models
      effective_model = (
        raw_model
        or (profile.model if profile else None)
        or sub_agent_default_model(effective_allowed)
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
        or sub_agent_default_model(effective_allowed)
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
    if profile is not None:
      from agent.shared.mutation_enforcement import apply_skill_mutation_mode_exclusions

      effective_excluded = apply_skill_mutation_mode_exclusions(
        profile,
        effective_excluded,
        local_tool_names=set(local_tool_handlers or {}) | set(_ARTIFACT_EMIT_TOOLS),
      )
      try:
        effective_excluded |= _skill_extra_excluded_tool_names(profile)
      except ValueError as exc:
        return None, {
          "code": "invalid_skill_config",
          "message": str(exc),
        }
    sub_local = {
      name: handler
      for name, handler in (local_tool_handlers or {}).items()
      if name not in effective_excluded
    }
    child_excluded = set(effective_excluded)
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

    def _emit_parent_event(event: dict[str, Any]) -> None:
      try:
        sub_log.append(event)
      except NameError:
        pass
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
            ticker=_html_artifact_ticker(profile, context_ticker),
            ts=time.time(),
            scope=_html_artifact_scope(profile, context_ticker),
          )
        )
      )
      skill_run_started_emitted = True

    def _emit_skill_result_captured(result: Any | None, error: dict[str, Any] | None) -> None:
      if not (skill_run_id and profile is not None and agent_name):
        return
      _emit_skill_run_started()
      _emit_parent_event(
        build_skill_result_captured_event(
          skill_run_id=skill_run_id,
          skill=profile.name,
          ticker=_html_artifact_ticker(profile, context_ticker),
          entries=sub_log.entries,
          result=result,
          error=error,
        )
      )

    async def _persist_skill_state(result: Any | None, error: dict[str, Any] | None) -> None:
      if not (agent_name and profile is not None and profile.persist_state and skill_state_store is not None):
        return
      response_text = _result_response_text(result)
      try:
        model_state = extract_state_update(response_text)
      except Exception:
        log.warning("Failed to extract state update for skill %s", profile.name, exc_info=True)
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
          log.warning("Failed to persist state for skill %s", profile.name, exc_info=True)

    if skill_run_id and profile is not None and agent_name:
      child_excluded = _skill_html_excluded_tools(effective_excluded, skill_profile=profile)
      if "emit_html_artifact" not in child_excluded:
        _install_emit_html_artifact_handler(
          sub_local=sub_local,
          profile=profile,
          skill_run_id=skill_run_id,
          context_ticker=context_ticker,
          context_research_file_id=context_research_file_id,
          parent_session=parent_session,
          fallback_user_id=effective_parent_user_id,
          emit_parent_event=_emit_parent_event,
          emit_skill_run_started=_emit_skill_run_started,
        )
      if "emit_dashboard_artifact" not in child_excluded:
        _install_emit_dashboard_artifact_handler(
          sub_local=sub_local,
          profile=profile,
          skill_run_id=skill_run_id,
          context_ticker=context_ticker,
          context_research_file_id=context_research_file_id,
          parent_session=parent_session,
          fallback_user_id=effective_parent_user_id,
          emit_parent_event=_emit_parent_event,
          emit_skill_run_started=_emit_skill_run_started,
        )

    sub_log = EventLog()
    sub_dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=sub_local,
      needs_approval=needs_approval,
      event_log=sub_log,
      session_id=getattr(runner, "_full_session_id", ""),
      should_avoid_permission_prompts=True,
      approval_key_qualifier=approval_key_qualifier,
      mcp_session_inject_servers=mcp_session_inject_servers,
      mcp_meta_inject_servers=mcp_meta_inject_servers,
      user_id=effective_parent_user_id,
      risk_user_id=getattr(parent_session, "risk_user_id", None),
      channel=getattr(parent_session, "channel", None),
      role=getattr(parent_session, "role", None),
      credentials_resolver_active=credentials_resolver_active,
      get_tool_definitions=_child_tool_definitions_getter(
        runner=runner,
        mcp_client=mcp_client,
        excluded_tools=child_excluded,
        extra_tool_definitions=(
          _artifact_emit_tool_definitions(set(sub_local))
          if profile is not None
          else None
        ),
      ),
    )

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
        excluded_tools=child_excluded,
        max_turns=effective_max_turns,
        timeout=effective_timeout,
        client_timeout=90,
        max_tokens=effective_max_tokens,
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
        if on_background_complete is not None:
          await on_background_complete(bg_task)
        await _persist_skill_state(bg_task.result, bg_task.error)
        _emit_skill_result_captured(bg_task.result, bg_task.error)

      return await runner._register_background_task(
        tool_input=enriched_tool_input,
        handler=_dispatch_sub_agent,
        agent_name=agent_name,
        parent_turn_id=parent_turn_id,
        on_before_start=(lambda: on_before_background(agent_name)) if on_before_background else None,
        on_complete=_on_background_complete if (skill_run_id or on_background_complete is not None) else None,
      )
    result, error = await _dispatch_sub_agent(tool_input, call_index=call_index)
    await _persist_skill_state(result, error)
    _emit_skill_result_captured(result, error)
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
  default_model: str = "claude-opus-4-7",
  default_max_turns: int = 15,
  default_timeout: float | None = DEFAULT_SUB_AGENT_TIMEOUT_SECONDS,
  default_max_tokens: int = 64000,
  allowed_models: set[str] | None = None,
  provider_resolver: ProviderResolver | None = None,
  coordinator_config: CoordinatorConfig | None = None,
  mutation_mode_exclusions_applier: MutationModeExclusionsApplier | None = None,
):
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
    child_budget_usd = _entry_child_budget_usd(entry)

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
    if str(getattr(profile, "mutation_mode", "") or "").strip() == "model_writer":
      return None, {
        "code": "model_writer_resume_unsupported",
        "message": "model-writer skills cannot be resumed; re-run the skill",
      }

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
        or sub_agent_default_model(effective_allowed)
        or (effective_coordinator.default_worker_model if effective_coordinator is not None else None)
        or resolved.default_model
        or default_model
      )
    else:
      effective_allowed = effective_allowed_models
      effective_model = (
        raw_model
        or profile.model
        or sub_agent_default_model(effective_allowed)
        or (effective_coordinator.default_worker_model if effective_coordinator is not None else None)
        or default_model
      )
    if effective_allowed and effective_model not in effective_allowed:
      return None, {"code": "invalid_input", "message": f"Invalid model '{effective_model}' for skill '{agent_name}'"}

    skill_prompt = resolve_blocks(profile.system_prompt, skill_loader.skills_dir / "_blocks")
    system_prompt = _SKILL_SYSTEM_PROMPT_TEMPLATE.format(
      skill_prompt=skill_prompt,
      date=datetime.date.today().isoformat(),
    )
    effective_max_turns = profile.max_turns if profile.max_turns is not None else default_max_turns
    effective_timeout = profile.timeout if profile.timeout is not None else default_timeout
    effective_max_tokens = profile.max_tokens if profile.max_tokens is not None else default_max_tokens
    effective_excluded = set(_DEFAULT_EXCLUDED_TOOLS | excluded_tools_resolver())
    if effective_coordinator is not None and effective_coordinator.worker_excluded_tools:
      effective_excluded = effective_excluded | effective_coordinator.worker_excluded_tools
    exclusions_applier = mutation_mode_exclusions_applier
    if exclusions_applier is None:
      from agent.shared.mutation_enforcement import apply_skill_mutation_mode_exclusions

      exclusions_applier = apply_skill_mutation_mode_exclusions
    effective_excluded = exclusions_applier(
      profile,
      effective_excluded,
      local_tool_names=set(local_tool_handlers or {}) | set(_ARTIFACT_EMIT_TOOLS),
    )
    try:
      effective_excluded |= _skill_extra_excluded_tool_names(profile)
    except ValueError as exc:
      return None, {
        "code": "invalid_skill_config",
        "message": str(exc),
      }
    child_excluded = _skill_html_excluded_tools(effective_excluded, skill_profile=profile)
    sub_local = {
      name: handler
      for name, handler in (local_tool_handlers or {}).items()
      if name not in effective_excluded
    }
    child_needs_approval = (
      needs_approval_resolver(frozenset(child_excluded))
      if needs_approval_resolver is not None
      else needs_approval
    )
    tool_ctx = kwargs.get("tool_ctx")
    parent_turn_id = getattr(tool_ctx, "tool_call_id", None)
    call_index = int(kwargs.get("call_index", 0) or 0)
    skill_run_id = secrets.token_hex(16)
    context_ticker = _extract_ticker_from_resume_messages(
      reconstructed_messages,
      parent_messages,
      additional_context,
    )
    context_research_file_id = _extract_research_file_id_from_resume_messages(
      reconstructed_messages,
      parent_messages,
      additional_context,
    )
    skill_run_started_emitted = False

    def _emit_parent_event(event: dict[str, Any]) -> None:
      try:
        sub_log.append(event)
      except NameError:
        pass
      emit = getattr(tool_ctx, "emit", None)
      if callable(emit):
        emit(event)

    def _emit_skill_run_started() -> None:
      nonlocal skill_run_started_emitted
      if skill_run_started_emitted:
        return
      _emit_parent_event(
        event_to_dict(
          SkillRunStartedEvent(
            skill_run_id=skill_run_id,
            skill=profile.name,
            ticker=_html_artifact_ticker(profile, context_ticker),
            ts=time.time(),
            scope=_html_artifact_scope(profile, context_ticker),
          )
        )
      )
      skill_run_started_emitted = True

    def _emit_skill_result_captured(result: Any | None, error: dict[str, Any] | None) -> None:
      _emit_skill_run_started()
      _emit_parent_event(
        build_skill_result_captured_event(
          skill_run_id=skill_run_id,
          skill=profile.name,
          ticker=_html_artifact_ticker(profile, context_ticker),
          entries=sub_log.entries,
          result=result,
          error=error,
        )
      )

    if "emit_html_artifact" not in child_excluded:
      _install_emit_html_artifact_handler(
        sub_local=sub_local,
        profile=profile,
        skill_run_id=skill_run_id,
        context_ticker=context_ticker,
        context_research_file_id=context_research_file_id,
        parent_session=parent_session,
        fallback_user_id=user_id or getattr(parent_session, "user_id", None),
        emit_parent_event=_emit_parent_event,
        emit_skill_run_started=_emit_skill_run_started,
      )
    if "emit_dashboard_artifact" not in child_excluded:
      _install_emit_dashboard_artifact_handler(
        sub_local=sub_local,
        profile=profile,
        skill_run_id=skill_run_id,
        context_ticker=context_ticker,
        context_research_file_id=context_research_file_id,
        parent_session=parent_session,
        fallback_user_id=user_id or getattr(parent_session, "user_id", None),
        emit_parent_event=_emit_parent_event,
        emit_skill_run_started=_emit_skill_run_started,
      )
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
      get_tool_definitions=_child_tool_definitions_getter(
        runner=runner,
        mcp_client=mcp_client,
        excluded_tools=child_excluded,
        extra_tool_definitions=_artifact_emit_tool_definitions(set(sub_local)),
      ),
    )

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
      _emit_skill_run_started()
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
        excluded_tools=child_excluded,
        max_turns=effective_max_turns,
        timeout=effective_timeout,
        client_timeout=90,
        max_tokens=effective_max_tokens,
        call_index=background_call_index,
        parent_turn_id=parent_turn_id,
        task_entry=task_entry,
        max_budget_usd=child_budget_usd,
        on_sub_event=lambda event, _sid: sub_log.append(event),
      )

    resume_tool_input = {
      "task_id": task_id,
      "agent": agent_name,
      "model": effective_model,
      "provider": raw_provider,
      "resumable": True,
    }
    if child_budget_usd is not None:
      resume_tool_input["child_budget_usd"] = child_budget_usd
    result, error = await runner._register_background_task(
      tool_input=resume_tool_input,
      handler=_dispatch_resume,
      agent_name=agent_name,
      parent_turn_id=parent_turn_id,
      on_complete=lambda bg_task: _emit_skill_result_captured(bg_task.result, bg_task.error),
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
  return _make_send_message_handler(runner_ref)



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
