from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


def default_tool_definitions(get_tool_definitions: Any, mcp_client: Any) -> List[Dict[str, Any]]:
  if get_tool_definitions is not None:
    return list(get_tool_definitions())
  if mcp_client is not None:
    return mcp_client.get_tool_definitions()
  return []


def effective_excluded_tools(
  excluded_tools: Set[str],
  active_skill_allow: Set[str],
) -> set[str]:
  return set(excluded_tools) - set(active_skill_allow)


def filter_excluded_tool_definitions(tools: List[Dict[str, Any]], excluded_tools: Set[str]) -> List[Dict[str, Any]]:
  if not excluded_tools:
    return list(tools)
  return [tool for tool in tools if tool["name"] not in excluded_tools]


def normalize_skill_allow(tool_names: Any) -> set[str] | None:
  candidates: Iterable[Any]
  if isinstance(tool_names, str):
    candidates = [tool_names]
  elif isinstance(tool_names, (list, tuple, set, frozenset)):
    candidates = tool_names
  else:
    return None
  return {normalized for name in candidates if (normalized := str(name or "").strip())}
