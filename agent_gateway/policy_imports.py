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


def resolve_effective_role(role: str | None) -> str:
  """Normalize authority without allowing owner-by-omission."""
  policy_module = load_server_policy_module()
  if policy_module is not None:
    resolver = getattr(policy_module, "resolve_effective_role", None)
    if resolver is not None:
      return str(resolver(role))
  return "owner" if (role or "").strip().lower() == "owner" else "invite"


def require_inherited_role(parent_session: Any | None) -> str:
  """Inherit an exact valid role; a null legacy parent remains invite."""
  if parent_session is None:
    return "invite"
  raw_role = getattr(parent_session, "role", None)
  normalized = (
    raw_role.strip().lower()
    if isinstance(raw_role, str)
    else ""
  )
  if normalized not in {"owner", "invite"}:
    raise ValueError("parent session must carry an explicit valid role")
  return normalized


def role_denied_tools_for_session(session: Any | None) -> frozenset[str]:
  """Return enumerable role denials; execution still fails closed if unavailable."""
  policy_module = load_server_policy_module()
  if policy_module is None:
    return frozenset()
  forbidden_for_session = getattr(
    policy_module,
    "get_forbidden_tools_for_session",
    None,
  )
  if forbidden_for_session is None:
    return frozenset()
  return frozenset(forbidden_for_session(session))


def role_policy_denies_tool(
  *,
  role: str | None,
  tool_name: str,
  is_local: bool,
) -> bool:
  """Return the canonical role decision, failing closed for invite authority."""
  if resolve_effective_role(role) == "owner":
    return False
  policy_module = load_server_policy_module()
  if policy_module is None:
    return True
  forbidden_for_session = getattr(
    policy_module,
    "get_forbidden_tools_for_session",
    None,
  )
  invite_denies_local = getattr(
    policy_module,
    "invite_denies_local_tool",
    None,
  )
  if forbidden_for_session is None or invite_denies_local is None:
    return True
  if is_local:
    return bool(invite_denies_local(tool_name))
  return tool_name in forbidden_for_session({"role": "invite"})


def authority_policy_denies_tool(
  *,
  session: Any | None,
  role: str | None,
  tool_name: str,
  is_local: bool,
  profile_name: str | None = None,
) -> bool:
  """Evaluate authenticated role and scoped session capabilities together."""
  if session is None:
    # A sessionless runtime has no scoped capability to evaluate.  Its exact
    # inherited role is therefore owned by the package-level role policy;
    # invite authority still fails closed when the host policy is unavailable.
    return role_policy_denies_tool(
      role=role,
      tool_name=tool_name,
      is_local=is_local,
    )
  policy_module = load_server_policy_module()
  if policy_module is None:
    return True
  evaluator = getattr(policy_module, "authority_denies_tool", None)
  if evaluator is None:
    return True
  authority_session = session if session is not None else {"role": role}
  return bool(evaluator(
    authority_session,
    tool_name,
    is_local=is_local,
    profile_name=profile_name,
  ))


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
