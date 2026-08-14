from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

_MODEL_WRITER_TERMINAL_DOORS = frozenset({"fms_persist_business_model"})


def default_tool_definitions(get_tool_definitions: Any, mcp_client: Any) -> List[Dict[str, Any]]:
  if get_tool_definitions is not None:
    return list(get_tool_definitions())
  if mcp_client is not None:
    return mcp_client.get_tool_definitions()
  return []


def effective_excluded_tools(
  excluded_tools: Set[str],
  active_skill_deny: Set[str],
  active_skill_allow: Set[str] | None = None,
) -> set[str]:
  return (set(excluded_tools) - set(active_skill_allow or ())) | set(active_skill_deny)


def filter_excluded_tool_definitions(tools: List[Dict[str, Any]], excluded_tools: Set[str]) -> List[Dict[str, Any]]:
  if not excluded_tools:
    return list(tools)
  return [tool for tool in tools if tool["name"] not in excluded_tools]


def normalize_skill_report_doors(value: Any) -> dict[str, str] | None:
  if isinstance(value, dict):
    return {
      str(tool_name).strip(): str(skill_name).strip()
      for tool_name, skill_name in value.items()
      if str(tool_name).strip() and str(skill_name).strip()
    }
  if value is not None:
    return {}
  return None


def normalize_skill_deny(tool_names: Any) -> set[str] | None:
  candidates: Iterable[Any]
  if isinstance(tool_names, str):
    candidates = [tool_names]
  elif isinstance(tool_names, (list, tuple, set, frozenset)):
    candidates = tool_names
  else:
    return None
  return {normalized for name in candidates if (normalized := str(name or "").strip())}


def is_report_door_clear_event(
  event: Dict[str, Any],
  *,
  expected_skill: str | None,
  success_statuses: set[str],
) -> bool:
  if event.get("type") != "tool_call_complete" or event.get("error") is not None:
    return False
  if not expected_skill:
    return False
  tool_name = str(event.get("tool_name") or "").strip()
  result = event.get("result")
  if not (
    isinstance(result, dict)
    and "subcommand" in result
  ):
    return False
  mutation_mode = str(result.get("mutation_mode") or "").strip()
  if mutation_mode != "preview" and not (
    mutation_mode == "model_writer" and tool_name in _MODEL_WRITER_TERMINAL_DOORS
  ):
    return False
  return str(result.get("status") or "").strip().lower() in success_statuses
