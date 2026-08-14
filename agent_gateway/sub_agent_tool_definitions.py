from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

from .sub_agent_helpers import _ARTIFACT_EMIT_TOOLS


def child_tool_definitions_getter(
  *,
  runner: Any,
  mcp_client: Any,
  excluded_tools: set[str],
  extra_tool_definitions: list[dict[str, Any]] | None = None,
  local_tool_handlers: Mapping[str, Any] | None = None,
) -> Callable[[], list[dict[str, Any]]] | None:
  child_excluded_tools = {str(name) for name in excluded_tools}
  extra_definitions = [copy.deepcopy(definition) for definition in (extra_tool_definitions or [])]
  routable_local_tools = (
    {
      str(name)
      for name, handler in local_tool_handlers.items()
      if callable(handler)
    }
    if local_tool_handlers is not None
    else None
  )
  parent_get_tool_definitions = getattr(runner, "_get_tool_definitions", None)
  mcp_get_tool_definitions = getattr(mcp_client, "get_tool_definitions", None)
  mcp_is_mcp_tool = getattr(mcp_client, "is_mcp_tool", None)
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
    extra_by_name: dict[str, dict[str, Any]] = {}
    for definition in extra_definitions:
      tool_name = str(definition.get("name") or "")
      if not tool_name:
        raise ValueError("Child tool definitions must declare a non-empty name")
      if tool_name in extra_by_name:
        raise ValueError(
          f"Duplicate child tool definition: {tool_name}"
        )
      extra_by_name[tool_name] = definition
    merged_definitions: list[dict[str, Any]] = []
    emitted_extras: set[str] = set()
    for definition in definitions:
      tool_name = (
        str(definition.get("name") or "")
        if isinstance(definition, dict)
        else ""
      )
      replacement = extra_by_name.get(tool_name)
      if replacement is not None:
        if tool_name not in emitted_extras:
          merged_definitions.append(copy.deepcopy(replacement))
          emitted_extras.add(tool_name)
        continue
      merged_definitions.append(definition)
    for tool_name, definition in extra_by_name.items():
      if tool_name in emitted_extras:
        continue
      merged_definitions.append(copy.deepcopy(definition))
    if routable_local_tools is not None:
      routable_definitions: list[dict[str, Any]] = []
      for definition in merged_definitions:
        tool_name = str(definition.get("name") or "")
        if tool_name in routable_local_tools:
          routable_definitions.append(definition)
          continue
        try:
          is_mcp_tool = bool(
            callable(mcp_is_mcp_tool) and mcp_is_mcp_tool(tool_name)
          )
        except Exception:
          is_mcp_tool = False
        if is_mcp_tool:
          routable_definitions.append(definition)
      merged_definitions = routable_definitions
    if not child_excluded_tools:
      return merged_definitions
    return [
      definition
      for definition in merged_definitions
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
    )
    from agent.shared.tool_defs.canvas_artifact import CANVAS_ARTIFACT_TOOL_DEF
    from .canvas_build_environment import canvas_build_enabled

    definitions = {
      "emit_dashboard_artifact": DASHBOARD_ARTIFACT_TOOL_DEF,
    }
    if canvas_build_enabled():
      definitions["emit_canvas_artifact"] = CANVAS_ARTIFACT_TOOL_DEF
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
    }
    from .canvas_build_environment import canvas_build_enabled
    if canvas_build_enabled():
      definitions["emit_canvas_artifact"] = {
        "name": "emit_canvas_artifact",
        "description": "Emit a compiled Canvas artifact for the current named skill run.",
        "input_schema": {
          "type": "object",
          "properties": {
            "title": {"type": "string"}, "purpose": {"type": "string"},
            "summary": {"type": "string"}, "tsx_source": {"type": "string"},
            "copy_as_markdown": {"type": "string"},
          },
          "required": ["title", "purpose", "summary", "tsx_source", "copy_as_markdown"],
        },
      }
  return [copy.deepcopy(definitions[name]) for name in sorted(requested) if name in definitions]
