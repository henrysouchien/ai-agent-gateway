from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Set

from .tool_dispatcher_helpers import NeedsApprovalCallback


def mcp_scope_error(
  tool_name: str,
  server_name: str | None,
  *,
  allowed_mcp_tools_by_server: Mapping[str, Set[str]] | None,
) -> dict[str, Any] | None:
  if allowed_mcp_tools_by_server is None:
    return None
  if not server_name:
    return {
      "code": "mcp_tool_not_allowed",
      "message": f"MCP tool '{tool_name}' is not allowed in this scoped child run.",
    }
  allowed_tools = allowed_mcp_tools_by_server.get(server_name, set())
  if tool_name in allowed_tools:
    return None
  return {
    "code": "mcp_tool_not_allowed",
    "message": (
      f"MCP tool '{server_name}.{tool_name}' is not allowed in this scoped child run. "
      "Use one of the MCP tools declared by the active skill."
    ),
  }


def callable_accepts_kw(callback: Any, keyword: str) -> bool:
  if callback is None:
    return False
  try:
    params = inspect.signature(callback).parameters
  except (TypeError, ValueError):
    return False
  return keyword in params or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def normalize_needs_approval(
  needs_approval: Callable[..., bool] | None,
) -> NeedsApprovalCallback:
  if needs_approval is None:
    return lambda _name, _tool_input, _qualifier: False

  try:
    arg_count = len(inspect.signature(needs_approval).parameters)
  except (TypeError, ValueError):
    return needs_approval  # type: ignore[return-value]

  if arg_count == 1:
    return lambda name, _tool_input, _qualifier: needs_approval(name)
  if arg_count == 2:
    return lambda name, tool_input, _qualifier: needs_approval(name, tool_input)
  return needs_approval  # type: ignore[return-value]


def qualified_key(tool_name: str, qualifier: str) -> str:
  return f"{tool_name}:{qualifier}" if qualifier else tool_name


def should_request_approval(
  tool_name: str,
  tool_input: dict[str, Any],
  qualifier: str,
  *,
  session_cache_denied: frozenset[str],
  approved_tool_types: Set[str],
  needs_approval: NeedsApprovalCallback,
  qualified_key_fn: Callable[[str, str], str] = qualified_key,
) -> bool:
  if tool_name not in session_cache_denied:
    cache_key = qualified_key_fn(tool_name, qualifier)
    if cache_key in approved_tool_types:
      return False
    if not qualifier and tool_name in approved_tool_types:
      return False
  return needs_approval(tool_name, tool_input, qualifier)


def tool_was_cache_hit(
  tool_name: str,
  qualifier: str,
  *,
  session_cache_denied: frozenset[str],
  approved_tool_types: Set[str],
  qualified_key_fn: Callable[[str, str], str] = qualified_key,
) -> bool:
  if tool_name in session_cache_denied:
    return False
  cache_key = qualified_key_fn(tool_name, qualifier)
  if cache_key in approved_tool_types:
    return True
  if not qualifier and tool_name in approved_tool_types:
    return True
  return False
