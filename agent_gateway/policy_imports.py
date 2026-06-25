from __future__ import annotations

import importlib
from typing import Any


_AGENT_POLICY_MODULE_NAMES = frozenset({"agent", "agent.shared", "agent.shared.server_policies"})
_API_POLICY_MODULE_NAMES = frozenset({"api", "api.agent", "api.agent.shared", "api.agent.shared.server_policies"})


def load_server_policy_module() -> Any | None:
  try:
    return importlib.import_module("agent.shared.server_policies")
  except ModuleNotFoundError as exc:
    if exc.name not in _AGENT_POLICY_MODULE_NAMES:
      raise

  try:
    return importlib.import_module("api.agent.shared.server_policies")
  except ModuleNotFoundError as exc:
    if exc.name not in _API_POLICY_MODULE_NAMES:
      raise
    return None


def load_server_policy_helpers(*, require_tool_class: bool = False) -> tuple[Any | None, Any | None, Any | None]:
  policy_module = load_server_policy_module()
  if policy_module is None:
    return None, None, None
  get_tool_class = (
    policy_module.get_tool_class
    if require_tool_class
    else getattr(policy_module, "get_tool_class", None)
  )
  return (
    policy_module.get_forbidden_tools_for_session,
    policy_module.get_server_for_policy_tool,
    get_tool_class,
  )


def resolve_server_policy_tool_class(
  tool_name: str,
  *,
  policy_tool_name: str | None = None,
  runtime_server: str | None = None,
  default: str = "state_write",
) -> str:
  _get_forbidden_tools_for_session, get_server_for_policy_tool, get_tool_class = load_server_policy_helpers(
    require_tool_class=True
  )
  if get_server_for_policy_tool is None or get_tool_class is None:
    return default

  normalized_tool = policy_tool_name or tool_name
  candidate_servers: list[str] = []
  for server_name in (runtime_server, get_server_for_policy_tool(normalized_tool)):
    if server_name and server_name not in candidate_servers:
      candidate_servers.append(server_name)

  for server_name in candidate_servers:
    cls = get_tool_class(server_name, normalized_tool)
    if cls is not None:
      return cls
  return default
