from __future__ import annotations

import datetime
import logging
import re
import secrets
import time
from typing import Any, Callable, FrozenSet

from .session import GatewaySession
from .skills import SkillLoader, SkillProfile
from .task_registry import ParentMessage

log = logging.getLogger("agent_gateway.sub_agent")

_DEFAULT_EXCLUDED_TOOLS = frozenset({"run_agent", "get_background_result", "send_message"})

# Finite by default: with timeout=None a wedged sub-agent parks the parent's
# run_agent tool call on an unbounded await, which holds the chat turn lock
# forever (stuck "Running" card, dead Esc, 409 on every new message — ACUI-1).
# Skill profiles with an explicit `timeout` still override this.
DEFAULT_SUB_AGENT_TIMEOUT_SECONDS = 1800.0
_ARTIFACT_EMIT_TOOLS = frozenset({"emit_html_artifact", "emit_dashboard_artifact"})
_RESEARCH_FILE_ID_RE = re.compile(r"\b(?:research[_ ]file[_ ]id|RESEARCH_FILE_ID)\b\s*[:=]\s*(\d+)", re.IGNORECASE)
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


def _extract_research_file_id_from_task(task: object) -> int | None:
  if not isinstance(task, str):
    return None
  match = _RESEARCH_FILE_ID_RE.search(task)
  if match is None:
    return None
  try:
    return int(match.group(1))
  except ValueError:
    return None


def _optional_research_file_id(value: object | None, *, default: int | None = None) -> int | None:
  if value is None or (isinstance(value, str) and not value.strip()):
    return default
  if isinstance(value, bool):
    raise ValueError("research_file_id must be an integer")
  try:
    parsed = int(str(value).strip())
  except (TypeError, ValueError) as exc:
    raise ValueError("research_file_id must be an integer") from exc
  if default is not None and parsed != default:
    raise ValueError("research_file_id does not match the active context")
  return parsed


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


def _extract_research_file_id_from_resume_messages(
  reconstructed_messages: list[dict[str, Any]],
  parent_messages: list[ParentMessage],
  additional_context: str | None,
) -> int | None:
  for message in reconstructed_messages:
    if message.get("role") != "user":
      continue
    research_file_id = _extract_research_file_id_from_task(_message_content_text(message.get("content")))
    if research_file_id is not None:
      return research_file_id
    break
  for message in parent_messages:
    research_file_id = _extract_research_file_id_from_task(message.text)
    if research_file_id is not None:
      return research_file_id
  return _extract_research_file_id_from_task(additional_context or "")


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
  skill_excluded = set(getattr(skill_profile, "extra_excluded_tools", set()) or set())
  if getattr(skill_profile, "mutation_mode", None) is None:
    excluded.difference_update(_ARTIFACT_EMIT_TOOLS - skill_excluded)
  return excluded


def _install_emit_html_artifact_handler(
  *,
  sub_local: dict[str, Any],
  profile: SkillProfile,
  skill_run_id: str,
  context_ticker: str,
  context_research_file_id: int | None,
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
      research_file_id = _optional_research_file_id(
        tool_input.get("research_file_id"),
        default=context_research_file_id,
      )
      control_run_id = str(getattr(parent_session, "session_id", "") or "").strip() or None
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
        research_file_id=research_file_id,
        control_run_id=control_run_id,
        origin_kind=None if research_file_id is not None else "product",
        visibility=None if research_file_id is not None else "default",
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
  context_research_file_id: int | None,
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
      research_file_id = _optional_research_file_id(
        tool_input.get("research_file_id"),
        default=context_research_file_id,
      )
      control_run_id = str(getattr(parent_session, "session_id", "") or "").strip() or None
      artifact = DashboardArtifact(
        artifact_id=artifact_id,
        source_skill=profile.name,
        payload_ref=f"{artifact_id}.payload.json",
        ts=now.isoformat(),
        research_file_id=research_file_id,
        control_run_id=control_run_id,
        origin_kind=None if research_file_id is not None else "product",
        visibility=None if research_file_id is not None else "default",
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
