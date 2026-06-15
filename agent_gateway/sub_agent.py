from __future__ import annotations

import datetime
import logging
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, FrozenSet

from ._provider_utils import _get_allowed_models_for_provider_name, sub_agent_default_model
from .events import SkillRunStartedEvent, event_to_dict
from .event_log import EventLog
from .runner import _derive_sub_agent_id
from .session import GatewaySession
from .skill_result_events import build_skill_result_captured_event
from .skills import SkillLoader, SkillProfile, resolve_blocks
from .task_registry import CoordinatorConfig, ParentMessage, ProviderResolver
from .tool_dispatcher import ToolDispatcher
from .transcript import (
  build_synthetic_tool_results,
  detect_orphan_tool_uses,
  place_resume_messages,
  reconstruct_messages_for_task,
  reconstruct_parent_messages,
)

log = logging.getLogger("agent_gateway.sub_agent")
_DEFAULT_EXCLUDED_TOOLS = frozenset({"run_agent", "get_background_result", "send_message"})

# Finite by default: with timeout=None a wedged sub-agent parks the parent's
# run_agent tool call on an unbounded await, which holds the chat turn lock
# forever (stuck "Running" card, dead Esc, 409 on every new message — ACUI-1).
# Skill profiles with an explicit `timeout` still override this.
DEFAULT_SUB_AGENT_TIMEOUT_SECONDS = 1800.0
_ARTIFACT_EMIT_TOOLS = frozenset({"emit_html_artifact", "emit_dashboard_artifact"})
ExcludedToolsResolver = Callable[[], FrozenSet[str]]
NeedsApprovalResolver = Callable[[FrozenSet[str]], Callable[..., bool] | None]
MutationModeExclusionsApplier = Callable[..., set[str]]
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


def _entry_child_budget_usd(entry: Any) -> float | None:
  metadata = getattr(entry, "metadata", None)
  if not isinstance(metadata, dict):
    return None
  raw_budget = metadata.get("child_budget_usd")
  if raw_budget is None or isinstance(raw_budget, bool):
    return None
  try:
    budget = float(raw_budget)
  except (TypeError, ValueError):
    return None
  return budget if budget > 0 else None


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


def _artifact_storage_user_id(parent_session: GatewaySession | None, fallback_user_id: str | None) -> str | None:
  risk_user_id = int(getattr(parent_session, "risk_user_id", 0) or 0)
  if risk_user_id > 0:
    return str(risk_user_id)
  return fallback_user_id


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


def _message_content_text(content: Any) -> str:
  if isinstance(content, str):
    return content
  if not isinstance(content, list):
    return ""

  parts: list[str] = []
  for block in content:
    if isinstance(block, str):
      parts.append(block)
      continue
    if not isinstance(block, dict):
      continue
    for key in ("text", "content"):
      value = block.get(key)
      if isinstance(value, str):
        parts.append(value)
        break
  return "\n".join(parts)


def _extract_ticker_from_resume_messages(
  reconstructed_messages: list[dict[str, Any]],
  parent_messages: list[ParentMessage],
  additional_context: str | None,
) -> str:
  for message in reconstructed_messages:
    if message.get("role") != "user":
      continue
    ticker = _extract_ticker_from_task(_message_content_text(message.get("content")))
    if ticker:
      return ticker
    break
  for message in parent_messages:
    ticker = _extract_ticker_from_task(message.text)
    if ticker:
      return ticker
  return _extract_ticker_from_task(additional_context or "")


def _html_artifact_ticker(profile: SkillProfile, context_ticker: str) -> str | None:
  return (context_ticker or None) if profile.scope == "ticker" else None


def _html_artifact_scope(profile: SkillProfile, context_ticker: str) -> str:
  return "ticker" if _html_artifact_ticker(profile, context_ticker) else "portfolio"


def _dashboard_artifact_ticker(profile: SkillProfile, context_ticker: str) -> str | None:
  return (context_ticker or None) if profile.scope == "ticker" else None


def _dashboard_artifact_scope(profile: SkillProfile, context_ticker: str) -> str:
  return "ticker" if _dashboard_artifact_ticker(profile, context_ticker) else "portfolio"


def _skill_html_excluded_tools(
  effective_excluded: set[str] | FrozenSet[str],
  *,
  skill_profile: SkillProfile | None = None,
) -> set[str]:
  excluded = set(effective_excluded)
  if getattr(skill_profile, "mutation_mode", None) is None:
    excluded.difference_update(_ARTIFACT_EMIT_TOOLS)
  return excluded


def _install_emit_html_artifact_handler(
  *,
  sub_local: dict[str, Any],
  profile: SkillProfile,
  skill_run_id: str,
  context_ticker: str,
  parent_session: GatewaySession | None,
  fallback_user_id: str | None,
  emit_parent_event: Callable[[dict[str, Any]], None],
  emit_skill_run_started: Callable[[], None],
) -> None:
  artifact_storage_user_id = _artifact_storage_user_id(parent_session, fallback_user_id)

  async def _handle_emit_html_artifact(
    tool_input: dict[str, Any],
    **handler_kwargs: Any,
  ):
    html_tool_ctx = handler_kwargs.get("tool_ctx")
    tool_call_id = getattr(html_tool_ctx, "tool_call_id", None)
    artifact_ticker = _html_artifact_ticker(profile, context_ticker)
    emit_skill_run_started()
    try:
      from memory import get_workspace_dir
      from schema.html_artifact import HtmlArtifact, StaticExports
      from schema.thesis_shared_slice import SourceRecord

      from .html_artifact_store import write_html_artifact

      raw_sources = tool_input.get("sources") or []
      if not isinstance(raw_sources, list):
        raise ValueError("sources must be a list")

      now = datetime.datetime.now(datetime.timezone.utc)
      artifact_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(8)}"
      artifact = HtmlArtifact(
        artifact_id=artifact_id,
        title=str(tool_input["title"]),
        purpose=tool_input["purpose"],
        content_ref=f"{artifact_id}.html",
        summary=str(tool_input["summary"]),
        ticker=artifact_ticker,
        session_id=None,
        source_skill=profile.name,
        sources=[SourceRecord.model_validate(source) for source in raw_sources],
        exports=StaticExports(
          copy_as_prompt=tool_input.get("copy_as_prompt"),
          copy_as_markdown=tool_input.get("copy_as_markdown"),
          copy_as_json=tool_input.get("copy_as_json"),
        ),
        ts=now.isoformat(),
      )
      workspace_dir = get_workspace_dir(artifact_storage_user_id)
      write_html_artifact(
        workspace_dir=workspace_dir,
        artifact=artifact,
        html_content=tool_input["html"],
      )
      emit_parent_event({
        "type": "artifact_ready",
        "skill_run_id": skill_run_id,
        "ticker": artifact_ticker,
        "skill": "_html",
        "artifact_id": artifact_id,
        "artifact_path": f"artifacts/_html/{artifact_id}.json",
        "binary_artifact_path": f"artifacts/_html/{artifact_id}.html",
        "contract_name": "HtmlArtifact",
        "data_source": "live",
        "ts": time.time(),
        "scope": "ticker" if artifact_ticker else "portfolio",
        "portfolio_id": None,
      })
      return {"artifact_id": artifact_id, "status": "ok"}, None
    except Exception as exc:
      log.warning("emit_html_artifact failed: %s", exc)
      emit_parent_event({
        "type": "artifact_failed",
        "skill_run_id": skill_run_id,
        "ticker": artifact_ticker,
        "skill": "_html",
        "error_code": "tool_write_failed",
        "error_detail": str(exc),
        "source_path": None,
        "tool_call_id": tool_call_id,
        "ts": time.time(),
      })
      return None, {"code": "internal_error", "message": str(exc)}

  sub_local["emit_html_artifact"] = _handle_emit_html_artifact


def _install_emit_dashboard_artifact_handler(
  *,
  sub_local: dict[str, Any],
  profile: SkillProfile,
  skill_run_id: str,
  context_ticker: str,
  parent_session: GatewaySession | None,
  fallback_user_id: str | None,
  emit_parent_event: Callable[[dict[str, Any]], None],
  emit_skill_run_started: Callable[[], None],
) -> None:
  artifact_storage_user_id = _artifact_storage_user_id(parent_session, fallback_user_id)

  async def _handle_emit_dashboard_artifact(
    tool_input: dict[str, Any],
    **handler_kwargs: Any,
  ):
    dashboard_tool_ctx = handler_kwargs.get("tool_ctx")
    tool_call_id = getattr(dashboard_tool_ctx, "tool_call_id", None)
    artifact_ticker = _dashboard_artifact_ticker(profile, context_ticker)
    try:
      from memory import get_workspace_dir
      from schema.dashboard_artifact import DashboardArtifact

      from .dashboard_artifact import build_dashboard_artifact
      from .dashboard_artifact_store import write_dashboard_artifact

      profile_name = str(tool_input.get("profile") or "production")
      built = build_dashboard_artifact(
        tool_input.get("payload"),
        profile_name,
        str(tool_input.get("summary") or ""),
      )
      if built.get("error") == "dashboard_validation_failed":
        return {
          "error": "dashboard_validation_failed",
          "hard_failures": list(built.get("hard_failures") or []),
          "warnings": list(built.get("warnings") or []),
        }, None

      now = datetime.datetime.now(datetime.timezone.utc)
      artifact_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(8)}"
      artifact_path = f"artifacts/_dashboards/{artifact_id}.json"
      payload_path = f"artifacts/_dashboards/{artifact_id}.payload.json"
      artifact = DashboardArtifact(
        artifact_id=artifact_id,
        source_skill=profile.name,
        payload_ref=f"{artifact_id}.payload.json",
        ts=now.isoformat(),
        **dict(built["sidecar_fields"]),
      )
      workspace_dir = get_workspace_dir(artifact_storage_user_id)
      write_dashboard_artifact(
        workspace_dir=workspace_dir,
        artifact=artifact,
        payload_json=built["payload_json"],
      )
      emit_skill_run_started()
      emit_parent_event({
        "type": "artifact_ready",
        "skill_run_id": skill_run_id,
        "ticker": artifact_ticker,
        "skill": "_dashboard",
        "artifact_id": artifact_id,
        "artifact_path": artifact_path,
        "binary_artifact_path": payload_path,
        "contract_name": "DashboardArtifact",
        "data_source": "live",
        "ts": time.time(),
        "scope": _dashboard_artifact_scope(profile, context_ticker),
        "portfolio_id": None,
      })
      return {
        "artifact_id": artifact_id,
        "sidecar_path": artifact_path,
        "payload_ref": payload_path,
        "warnings": list(built.get("warnings") or []),
      }, None
    except Exception as exc:
      log.warning("emit_dashboard_artifact failed: %s", exc)
      emit_parent_event({
        "type": "artifact_failed",
        "skill_run_id": skill_run_id,
        "ticker": artifact_ticker,
        "skill": "_dashboard",
        "error_code": "tool_write_failed",
        "error_detail": str(exc),
        "source_path": None,
        "tool_call_id": tool_call_id,
        "ts": time.time(),
      })
      return None, {"code": "internal_error", "message": str(exc)}

  sub_local["emit_dashboard_artifact"] = _handle_emit_dashboard_artifact


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

    if skill_run_id and profile is not None and agent_name:
      child_excluded = _skill_html_excluded_tools(effective_excluded, skill_profile=profile)
      if "emit_html_artifact" not in child_excluded:
        _install_emit_html_artifact_handler(
          sub_local=sub_local,
          profile=profile,
          skill_run_id=skill_run_id,
          context_ticker=context_ticker,
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
        if on_background_complete is not None:
          await on_background_complete(bg_task)
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
        max_tokens=default_max_tokens,
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
