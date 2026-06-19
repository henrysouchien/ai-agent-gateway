from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class CollisionFilterResult:
  tool_definitions: list[dict[str, Any]]
  tool_to_server: dict[str, str]
  prefixed_to_original: dict[str, str]
  mcp_tool_names: set[str]


def apply_collision_filtering(
  *,
  servers: Mapping[str, Any],
  builtin_tool_names: set[str],
  strip_input_fields: set[str],
  logger: logging.Logger,
) -> CollisionFilterResult:
  existing_names = set(builtin_tool_names)
  seen_mcp_names: dict[str, str] = {}
  merged: list[dict[str, Any]] = []
  tool_to_server: dict[str, str] = {}
  prefixed_to_original: dict[str, str] = {}

  for server_name, state in servers.items():
    prefix = state.tool_prefix
    filtered: list[dict[str, Any]] = []
    filtered_names: set[str] = set()

    for tool in state.tool_definitions:
      original_name = tool["name"]
      tool_name = f"{prefix}{original_name}" if prefix else original_name
      if tool_name in existing_names:
        logger.warning(
          "Skipping MCP tool %s from %s: collides with built-in tool. "
          "Fix: add \"tool_prefix\" to the \"%s\" server config to namespace its tools.",
          tool_name,
          server_name,
          server_name,
        )
        continue

      first_server = seen_mcp_names.get(tool_name)
      if first_server:
        logger.warning(
          "Skipping MCP tool %s from %s: collides with MCP tool from %s. "
          "Fix: add \"tool_prefix\" to the \"%s\" server config to namespace its tools.",
          tool_name,
          server_name,
          first_server,
          server_name,
        )
        continue

      seen_mcp_names[tool_name] = server_name
      tool_to_server[tool_name] = server_name
      if prefix:
        prefixed_to_original[tool_name] = original_name
        tool = {**tool, "name": tool_name}
      filtered.append(tool)
      filtered_names.add(tool_name)

    state.tool_definitions = filtered
    state.tool_names = filtered_names
    merged.extend(filtered)

    logger.info(
      "MCP server %s connected | %d tools: %s",
      server_name,
      len(filtered_names),
      sorted(filtered_names),
    )

  _strip_hidden_input_fields(merged, strip_input_fields)
  return CollisionFilterResult(
    tool_definitions=merged,
    tool_to_server=tool_to_server,
    prefixed_to_original=prefixed_to_original,
    mcp_tool_names=set(tool_to_server.keys()),
  )


def _strip_hidden_input_fields(
  tool_definitions: list[dict[str, Any]],
  strip_input_fields: set[str],
) -> None:
  if not strip_input_fields:
    return
  for tool_def in tool_definitions:
    schema = tool_def.get("input_schema", {})
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for field in strip_input_fields:
      props.pop(field, None)
      if field in required:
        required.remove(field)


__all__ = ["CollisionFilterResult", "apply_collision_filtering"]
