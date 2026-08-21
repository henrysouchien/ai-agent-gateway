from __future__ import annotations

import datetime
import logging
import re
import secrets
import time
from typing import Any, Callable, FrozenSet

from agent_workflow_contracts.research_file_contract import (
  extract_research_file_id as _extract_research_file_id_from_task,
)
from agent_workflow_contracts.ticker_contract import (
  is_entity_ticker,
  normalize_contract_ticker,
)

from .session import GatewaySession
from .artifact_paths import canonicalize_ticker
from .operation_catalog import AgentOperationCatalog
from .skills import SkillLoader, SkillProfile, operation_tool_ids
from .task_registry import ParentMessage

log = logging.getLogger("agent_gateway.sub_agent")

_DEFAULT_EXCLUDED_TOOLS = frozenset({
  "run_agent",
  "get_agent_result_content",
  "get_background_result",
  "send_message",
})

# Finite by default: with timeout=None a wedged sub-agent parks the parent's
# run_agent tool call on an unbounded await, which holds the chat turn lock
# forever (stuck "Running" card, dead Esc, 409 on every new message — ACUI-1).
# Skill profiles with an explicit `timeout` still override this.
DEFAULT_SUB_AGENT_TIMEOUT_SECONDS = 1800.0
_ARTIFACT_EMIT_TOOLS = frozenset({"emit_canvas_artifact", "emit_dashboard_artifact"})
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
  "Spawn a focused child from a registered executable operation. Pass the "
  "complete operation identity advertised below, or omit it to select the "
  "registered generic explore operation. The server admits model, semantic "
  "capabilities, and an exact tool grant; operation prose does not grant tools."
)
_RESUME_AGENT_DESCRIPTION = (
  "Resume an interrupted background sub-agent. Only resumable skills can be resumed; "
  "read the `resumable` and lineage fields from its automatic interrupted notification "
  "(use get_background_result only for a historical or explicitly omitted notification). "
  "The exact original complete capability bind and credential authority are "
  "reused; selection overrides are not accepted. Returns new task_id."
)


_TICKER_DISCOVERY_TOKEN_RE = re.compile(
  r"(?<![A-Za-z0-9.-])([A-Z0-9][A-Z0-9.-]{0,14})(?![A-Za-z0-9.-])"
)
_DISCOVERY_EXCLUSION_RE = re.compile(
  r"^(?:(?:FY)?\d{2,4}E|FY\d{2,4}E?|(?:[1-4]Q|Q[1-4])(?:FY)?\d{0,4}E?|"
  r"(?:[12]H|H[12])\d{0,4}|\d{1,2}[KQ]|(?:10K|10Q|8K|6K|20F)|"
  r"\d{2,4}Q[1-4]E?|FY\d{2,4}Q[1-4]E?|\d+[KMGTP]?B)$"
)
_B3_DISCOVERY_RE = re.compile(r"^[A-Z]{4}(?:3|4|5|6|7|8|11|31|32|33|34|35)$")
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
  "TRACK",
  "REVIEW",
  "REPORT",
  "FOCUS",
  "BUILD",
  "CREATE",
  "UPDATE",
  "CHECK",
}


def _artifact_storage_user_id(parent_session: GatewaySession | None, fallback_user_id: str | None) -> str | None:
  risk_user_id = int(getattr(parent_session, "risk_user_id", 0) or 0)
  if risk_user_id > 0:
    return str(risk_user_id)
  return fallback_user_id


def _render_agent_param_description(entries: list[tuple[Any, str]]) -> str:
  base = (
    "Full immutable AgentOperationRef. Omit to select the registered generic "
    "explore operation. Bare methodology/skill names are not accepted."
  )
  if not entries:
    return base
  lines = [base, "", "Available operations:"]
  lines.extend(
    (
      f"- {operation.namespace}/{operation.name}@{operation.version} "
      f"({operation.digest}): {description}"
    )
    for operation, description in entries
  )
  return "\n".join(lines)


def _extract_ticker_from_task(task: str) -> str | None:
  candidates: list[str] = []
  for match in _TICKER_DISCOVERY_TOKEN_RE.finditer(task):
    candidate = match.group(1)
    if _DISCOVERY_EXCLUSION_RE.fullmatch(candidate):
      continue
    if any(char.isdigit() for char in candidate) and not _B3_DISCOVERY_RE.fullmatch(candidate):
      continue
    try:
      normalized = normalize_contract_ticker(candidate)
    except ValueError:
      continue
    if normalized in _TICKER_STOPWORDS or not is_entity_ticker(normalized):
      continue
    if normalized not in candidates:
      candidates.append(normalized)
  if len(candidates) > 1:
    raise ValueError("objective contains multiple plausible ticker subjects")
  return candidates[0] if candidates else None


def _resolve_context_ticker(
  *,
  typed_ticker: object | None,
  verified_server_ticker: object | None,
  prose_fallback: Callable[[], str | None] | None,
) -> str | None:
  """Resolve one canonical subject without consulting prose unnecessarily."""

  typed = (
    normalize_contract_ticker(typed_ticker)
    if typed_ticker is not None
    else None
  )
  verified = (
    normalize_contract_ticker(
      verified_server_ticker,
      field_name="verified_server_ticker",
    )
    if verified_server_ticker is not None
    else None
  )
  if typed is not None and verified is not None and typed != verified:
    raise ValueError("typed ticker conflicts with the verified server subject")
  if typed is not None:
    return typed
  if verified is not None:
    return verified
  if prose_fallback is None:
    return None
  fallback = prose_fallback()
  return normalize_contract_ticker(fallback) if fallback is not None else None


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


def _artifact_ticker(profile: SkillProfile, context_ticker: str) -> str | None:
  return canonicalize_ticker(context_ticker) if profile.scope == "ticker" and context_ticker else None


def _artifact_scope(profile: SkillProfile, context_ticker: str) -> str:
  return "ticker" if _artifact_ticker(profile, context_ticker) else "portfolio"


def _dashboard_artifact_ticker(profile: SkillProfile, context_ticker: str) -> str | None:
  return canonicalize_ticker(context_ticker) if profile.scope == "ticker" and context_ticker else None


def _dashboard_artifact_scope(profile: SkillProfile, context_ticker: str) -> str:
  return "ticker" if _dashboard_artifact_ticker(profile, context_ticker) else "portfolio"


def _artifact_ticker_for_scope(
  semantic_scope: str,
  context_ticker: str,
) -> str | None:
  return (
    canonicalize_ticker(context_ticker)
    if semantic_scope == "ticker" and context_ticker
    else None
  )


def _artifact_scope_for_scope(
  semantic_scope: str,
  context_ticker: str,
) -> str:
  return (
    "ticker"
    if _artifact_ticker_for_scope(semantic_scope, context_ticker)
    else "portfolio"
  )


def _current_context_ticker(
  value: str | Callable[[], str | None],
) -> str:
  resolved = value() if callable(value) else value
  return resolved or ""


def _skill_extra_excluded_tool_names(skill_profile: SkillProfile | None) -> set[str]:
  raw_tools = getattr(skill_profile, "extra_excluded_tools", None)
  if raw_tools is None:
    return set()
  if isinstance(raw_tools, str) or not isinstance(raw_tools, (list, tuple, set, frozenset)):
    raise ValueError(
      f"Skill '{getattr(skill_profile, 'name', '<unknown>')}' extra_excluded_tools must be a list of tool names"
    )
  tool_names: set[str] = set()
  for raw_tool in raw_tools:
    if not isinstance(raw_tool, str):
      raise ValueError(
        f"Skill '{getattr(skill_profile, 'name', '<unknown>')}' extra_excluded_tools entries must be strings"
      )
    tool_name = raw_tool.strip()
    if tool_name:
      tool_names.add(tool_name)
  return tool_names


def _skill_artifact_excluded_tools(
  effective_excluded: set[str] | FrozenSet[str],
  *,
  skill_profile: SkillProfile | None = None,
) -> set[str]:
  excluded = set(effective_excluded)
  skill_excluded = _skill_extra_excluded_tool_names(skill_profile)
  declared_artifact_routes = (
    operation_tool_ids(skill_profile) & _ARTIFACT_EMIT_TOOLS
    if skill_profile is not None
    else frozenset()
  )
  excluded.difference_update(declared_artifact_routes - skill_excluded)
  return excluded


def _install_emit_canvas_artifact_handler(
  *, sub_local: dict[str, Any], profile: SkillProfile | None = None,
  skill_name: str | None = None, semantic_scope: str | None = None,
  skill_run_id: str,
  context_ticker: str | Callable[[], str | None],
  context_research_file_id: int | None,
  parent_session: GatewaySession | None, fallback_user_id: str | None,
  emit_parent_event: Callable[[dict[str, Any]], None],
) -> bool:
  resolved_skill_name = skill_name or getattr(profile, "name", None)
  resolved_scope = (
    semantic_scope
    if semantic_scope is not None
    else (getattr(profile, "scope", None) or "global")
  )
  if not resolved_skill_name:
    raise ValueError("artifact handler requires skill identity and scope")
  from .canvas_build_environment import preflight_canvas_build_environment

  preflight = preflight_canvas_build_environment()
  if preflight is None:
    return False
  artifact_storage_user_id = _artifact_storage_user_id(parent_session, fallback_user_id)

  async def _handle_emit_canvas_artifact(tool_input: dict[str, Any], **_: Any):
    try:
      from memory import get_workspace_dir
      from schema.thesis_shared_slice import SourceRecord
      from .canvas_artifact_pipeline import emit_canvas_artifact_async

      raw_sources = tool_input.get("sources") or []
      if not isinstance(raw_sources, list):
        raise ValueError("sources must be a list")
      research_file_id = _optional_research_file_id(
        tool_input.get("research_file_id"), default=context_research_file_id,
      )
      def _emit_canvas_event(event: dict[str, Any]) -> None:
        emit_parent_event(event)

      result = await emit_canvas_artifact_async(
        workspace_dir=get_workspace_dir(artifact_storage_user_id), preflight=preflight,
        title=str(tool_input["title"]), purpose=str(tool_input["purpose"]),
        summary=str(tool_input["summary"]), tsx_source=str(tool_input["tsx_source"]),
        copy_as_markdown=str(tool_input["copy_as_markdown"]),
        copy_as_prompt=tool_input.get("copy_as_prompt"),
        copy_as_json=tool_input.get("copy_as_json"),
        sources=[SourceRecord.model_validate(value) for value in raw_sources],
        source_skill=resolved_skill_name, skill_run_id=skill_run_id,
        ticker=_artifact_ticker_for_scope(
          resolved_scope,
          _current_context_ticker(context_ticker),
        ),
        session_id=str(getattr(parent_session, "session_id", "") or "").strip() or None,
        research_file_id=research_file_id,
        control_run_id=str(getattr(parent_session, "session_id", "") or "").strip() or None,
        user_id=artifact_storage_user_id or "", emit_event=_emit_canvas_event,
      )
      return result, None
    except Exception as exc:
      log.warning("emit_canvas_artifact failed: %s", exc)
      return None, {"code": "internal_error", "message": str(exc)}

  sub_local["emit_canvas_artifact"] = _handle_emit_canvas_artifact
  return True


def _install_emit_dashboard_artifact_handler(
  *,
  sub_local: dict[str, Any],
  profile: SkillProfile | None = None,
  skill_name: str | None = None,
  semantic_scope: str | None = None,
  skill_run_id: str,
  context_ticker: str | Callable[[], str | None],
  context_research_file_id: int | None,
  parent_session: GatewaySession | None,
  fallback_user_id: str | None,
  emit_parent_event: Callable[[dict[str, Any]], None],
) -> None:
  resolved_skill_name = skill_name or getattr(profile, "name", None)
  resolved_scope = (
    semantic_scope
    if semantic_scope is not None
    else (getattr(profile, "scope", None) or "global")
  )
  if not resolved_skill_name:
    raise ValueError("artifact handler requires skill identity and scope")
  artifact_storage_user_id = _artifact_storage_user_id(parent_session, fallback_user_id)

  async def _handle_emit_dashboard_artifact(
    tool_input: dict[str, Any],
    **handler_kwargs: Any,
  ):
    dashboard_tool_ctx = handler_kwargs.get("tool_ctx")
    tool_call_id = getattr(dashboard_tool_ctx, "tool_call_id", None)
    admitted_context_ticker = _current_context_ticker(context_ticker)
    artifact_ticker = _artifact_ticker_for_scope(
      resolved_scope,
      admitted_context_ticker,
    )
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
        source_skill=resolved_skill_name,
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
        "scope": _artifact_scope_for_scope(
          resolved_scope,
          admitted_context_ticker,
        ),
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

def make_run_agent_tool_def(
  operation_source: SkillLoader | AgentOperationCatalog | None = None,
) -> dict[str, Any]:
  """Build the public tool schema for `run_agent`.

  When an operation source is provided, the operation description includes the
  currently available immutable operation identities.
  """
  operations = (
    operation_source.list_callable_operations_with_descriptions()
    if operation_source is not None
    else []
  )

  return {
    "name": "run_agent",
    "description": _RUN_AGENT_DESCRIPTION,
    "input_schema": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "operation": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "namespace": {"type": "string"},
            "name": {"type": "string"},
            "version": {"type": "string"},
            "digest": {
              "type": "string",
              "pattern": "^sha256:[0-9a-f]{64}$",
            },
          },
          "required": ["namespace", "name", "version", "digest"],
          "description": _render_agent_param_description(operations),
        },
        "objective": {
          "type": "string",
          "description": "The objective for the admitted child task.",
        },
        "ticker": {
          "type": "string",
          "minLength": 1,
          "maxLength": 64,
          "description": (
            "Optional typed security subject. The server canonicalizes and "
            "admits this value independently of objective prose; it never "
            "uses the value to widen operation or artifact authority."
          ),
        },
        "research_file_id": {
          "type": "integer",
          "minimum": 1,
          "description": (
            "Optional research-file assertion. When a server-verified active "
            "research turn exists, this value must match it and cannot "
            "override it. At the private Investment quant boundary it never "
            "supplies authority and that verified turn is required. Other "
            "operations retain their declared context semantics."
          ),
        },
        "cost_observation_threshold_usd": {
          "type": "number",
          "exclusiveMinimum": 0,
          "description": (
            "Optional cost observation threshold in USD. Crossing it may emit "
            "telemetry or a runaway warning, but never stops or fails the child."
          ),
        },
        "background": {
          "type": "boolean",
          "description": (
            "Run the sub-agent in the background and return immediately with a task_id. "
            "This defaults to true; set false only when the current turn must wait for "
            "the validated typed child result."
          ),
          "default": True,
        },
      },
      "required": ["objective"],
    },
  }


def make_get_background_result_tool_def() -> dict[str, Any]:
  """Build the public tool schema for ``get_background_result``.

  The tool retrieves historical or explicitly omitted results. When automatic
  notifications are disabled, it can also perform one explicit wait for a
  background sub-agent task launched via ``run_agent(background=true)``.
  Pass ``task_id='*'`` to list eligible tracked task IDs without returning
  their payloads. Oversized exact results are returned in bounded JSON-text
  pages; pass the opaque ``cursor`` back with the same exact task ID.

  Returns:
    A tool-definition dict suitable for inclusion in ``get_tool_definitions``.
  """
  return {
    "name": "get_background_result",
    "description": (
      "Retrieve a historical or explicitly omitted background result. When "
      "automatic notifications are disabled, explicitly wait for a known task. "
      "Current-run outcomes are delivered completely through notifications; "
      "never poll them. Use task_id='*' only to list eligible tracked task IDs. "
      "For a paged exact result, pass each returned opaque cursor unchanged."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "string",
          "description": (
            "Exact task ID from run_agent(background=true), or '*' for the "
            "bounded task-ID/status directory allowed by the active delivery mode."
          ),
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
        "cursor": {
          "type": "string",
          "description": (
            "Opaque cursor returned by a prior page for the same exact task_id. "
            "Omit to start or restart delivery from the first page."
          ),
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
          "minLength": 1,
          "maxLength": 12000,
          "description": "Message content to deliver to the agent.",
        },
      },
      "required": ["to", "message"],
    },
  }
