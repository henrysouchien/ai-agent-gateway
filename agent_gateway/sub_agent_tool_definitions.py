from __future__ import annotations

import copy
from typing import Any, Callable

from .sub_agent_helpers import _ARTIFACT_EMIT_TOOLS


def child_tool_definitions_getter(
  *,
  runner: Any,
  mcp_client: Any,
  excluded_tools: set[str],
  extra_tool_definitions: list[dict[str, Any]] | None = None,
) -> Callable[[], list[dict[str, Any]]] | None:
  child_excluded_tools = {str(name) for name in excluded_tools}
  extra_definitions = [copy.deepcopy(definition) for definition in (extra_tool_definitions or [])]
  parent_get_tool_definitions = getattr(runner, "_get_tool_definitions", None)
  mcp_get_tool_definitions = getattr(mcp_client, "get_tool_definitions", None)
  if (
    not callable(parent_get_tool_definitions)
    and not callable(mcp_get_tool_definitions)
    and not extra_definitions
  ):
    return None

  def _child_tool_definitions() -> list[dict[str, Any]]:
    if callable(parent_get_tool_definitions):
      definitions = list(parent_get_tool_definitions())
    elif callable(mcp_get_tool_definitions):
      definitions = list(mcp_get_tool_definitions())
    else:
      definitions = []
    seen = {str(definition.get("name") or "") for definition in definitions if isinstance(definition, dict)}
    for definition in extra_definitions:
      tool_name = str(definition.get("name") or "")
      if not tool_name or tool_name in seen:
        continue
      definitions.append(copy.deepcopy(definition))
      seen.add(tool_name)
    if not child_excluded_tools:
      return definitions
    return [
      definition
      for definition in definitions
      if str(definition.get("name") or "") not in child_excluded_tools
    ]

  return _child_tool_definitions


def artifact_emit_tool_definitions(
  installed_tool_names: set[str],
  *,
  artifact_emit_tools: set[str] | frozenset[str] = _ARTIFACT_EMIT_TOOLS,
) -> list[dict[str, Any]]:
  requested = set(installed_tool_names) & set(artifact_emit_tools)
  if not requested:
    return []
  try:
    from agent.shared.tool_defs.html_artifact import (
      DASHBOARD_ARTIFACT_TOOL_DEF,
      HTML_ARTIFACT_TOOL_DEF,
    )

    definitions = {
      "emit_dashboard_artifact": DASHBOARD_ARTIFACT_TOOL_DEF,
      "emit_html_artifact": HTML_ARTIFACT_TOOL_DEF,
    }
  except ImportError:
    definitions = {
      "emit_dashboard_artifact": {
        "name": "emit_dashboard_artifact",
        "description": "Emit a typed dashboard artifact for the current named skill run.",
        "input_schema": {
          "type": "object",
          "properties": {
            "payload": {"type": "object"},
            "summary": {"type": "string"},
            "profile": {"type": "string"},
            "research_file_id": {"type": ["integer", "string", "null"]},
          },
          "required": ["payload", "summary"],
        },
      },
      "emit_html_artifact": {
        "name": "emit_html_artifact",
        "description": "Emit an HTML artifact for the current named skill run.",
        "input_schema": {
          "type": "object",
          "properties": {
            "title": {"type": "string"},
            "purpose": {"type": "string"},
            "summary": {"type": "string"},
            "html": {"type": "string"},
            "copy_as_prompt": {"type": ["string", "null"]},
            "copy_as_markdown": {"type": ["string", "null"]},
            "copy_as_json": {"type": ["object", "null"]},
            "sources": {"type": "array", "items": {"type": "object"}},
            "research_file_id": {"type": ["integer", "string", "null"]},
          },
          "required": ["title", "purpose", "summary", "html"],
        },
      },
    }
  return [copy.deepcopy(definitions[name]) for name in sorted(requested)]
