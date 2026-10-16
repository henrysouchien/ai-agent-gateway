from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Callable, FrozenSet

from agent_workflow_contracts.ticker_contract import (
  is_entity_ticker,
  normalize_contract_ticker,
)

from .operation_catalog import AgentOperationCatalog
from .skills import SkillLoader

def _render_agent_param_description(
  entries: Sequence[tuple[Any, str]],
) -> str:
  base = (
    "Full immutable AgentOperationRef. Omit to select the registered generic "
    "explore operation. Bare methodology/skill names are not accepted."
  )
  if not entries:
    return base
  lines = [
    base,
    "",
    "Registered operations are listed below by identity and one-line "
    "description; call list_skills for the full catalog, including each "
    "operation's exact AgentOperationRef (admission validates its digest).",
    "",
    "Available operations:",
  ]
  for operation, description in entries:
    lines.append(
      f"- {operation.namespace}/{operation.name}@{operation.version}: "
      f"{description}"
    )
  return "\n".join(lines)


def _catalog_operation_entries(
  operation_source: SkillLoader | AgentOperationCatalog,
) -> list[tuple[Any, str]]:
  return list(operation_source.list_callable_operations_with_descriptions())


_ARTIFACT_EMIT_TOOLS = frozenset({"emit_canvas_artifact", "emit_dashboard_artifact"})
# The one spawn door's tool name: admission withholds it from a job whose
# admitted `children` is empty.
RUN_AGENT_TOOL_NAME = "run_agent"
ExcludedToolsResolver = Callable[[], FrozenSet[str]]
NeedsApprovalResolver = Callable[[FrozenSet[str]], Callable[..., bool] | None]
MutationModeExclusionsApplier = Callable[..., set[str]]
_DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
  "You are a focused sub-agent working on behalf of another agent. Complete the assigned task "
  "thoroughly and return a clear, concise response for the parent agent. If any tool fails or "
  "returns suspicious data, note that explicitly instead of silently proceeding.\n\n"
  "Today's date: {date}"
)
_RUN_AGENT_DESCRIPTION = (
  "Spawn, interrupt, or resume a focused child. To interrupt only one live "
  "background child, pass interrupt=task_id; the parent and siblings keep "
  "running. After its interrupted notification, pass resume=task_id and "
  "optional additional_context to continue on the original capability bind "
  "from its durable transcript. Do not mix controls with spawn arguments. "
  "To spawn from a registered executable operation, pass the "
  "complete operation identity advertised below, or omit it to select the "
  "registered generic explore operation. The server admits model, semantic "
  "capabilities, and an exact tool grant; operation prose does not grant tools. "
  "Declared required_context refs are server-set from prior tool results in "
  "this turn, not copied into objective. Missing refs return "
  "required_context_missing without spawning a child; no latest lookup."
)
_RESUME_AGENT_DESCRIPTION = (
  "Resume an interrupted background sub-agent from its durable transcript; "
  "read the lineage fields from its automatic interrupted notification "
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


def make_run_agent_tool_def(
  operation_source: SkillLoader | AgentOperationCatalog | None = None,
) -> dict[str, Any]:
  """Build the public tool schema for `run_agent`.

  When an operation source is provided, each listed operation's identity
  (namespace/name@version) and parent-facing dispatch description appear in
  the `operation` parameter description; digests stay out of prose.
  """
  operations = (
    _catalog_operation_entries(operation_source)
    if operation_source is not None
    else []
  )

  return {
    "name": RUN_AGENT_TOOL_NAME,
    "description": _RUN_AGENT_DESCRIPTION,
    "input_schema": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "interrupt": {
          "type": "string",
          "minLength": 1,
          "description": (
            "Request interruption of one live background child owned by this "
            "job, preserving its transcript and original binding for resume. "
            "Wait for its interrupted notification before resuming."
          ),
        },
        "resume": {
          "type": "string",
          "minLength": 1,
          "description": (
            "Interrupted or parked child task_id to continue from its durable "
            "transcript on its original model and tool grant. Returns a new task_id."
          ),
        },
        "additional_context": {
          "type": "string",
          "description": "Optional continuation guidance; used only with resume.",
        },
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
        "role": {
          "type": "string",
          "description": (
            "Optional model role such as fast or deep selected from policy, "
            "independent of the operation."
          ),
        },
        "effort": {
          "type": "string",
          "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
          "description": "Optional exact model reasoning effort for the child.",
        },
        "handles": {
          "type": "array",
          "items": {"type": "string"},
          "description": (
            "Optional explicit read grants for the child. Pass handles already "
            "available to you or produced by your own job or its descendants."
          ),
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
        "handle_refs": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Handle citations accompanying the message; never grants read access to the recipient.",
        },
      },
      "required": ["to", "message"],
    },
  }
