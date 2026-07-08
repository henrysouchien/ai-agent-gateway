from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PolicyOwnerInvariantResult:
  tool_definitions: list[dict[str, Any]]
  tool_to_server: dict[str, str]
  prefixed_to_original: dict[str, str]
  mcp_tool_names: set[str]


def apply_policy_owner_invariant(
  *,
  servers: dict[str, Any],
  tool_definitions: list[dict[str, Any]],
  tool_to_server: dict[str, str],
  prefixed_to_original: dict[str, str],
  mcp_tool_names: set[str],
  policy_server_for_tool: Callable[[str], str | None],
  set_startup_diagnostic: Callable[..., None],
  logger: Any,
) -> PolicyOwnerInvariantResult:
  hidden_by_server: dict[str, list[tuple[str, str, str]]] = {}
  for exposed_name, runtime_server in sorted(tool_to_server.items()):
    original_name = prefixed_to_original.get(exposed_name, exposed_name)
    policy_server = policy_server_for_tool(original_name)
    if policy_server and policy_server != runtime_server:
      hidden_by_server.setdefault(runtime_server, []).append(
        (exposed_name, original_name, policy_server)
      )

  if not hidden_by_server:
    return PolicyOwnerInvariantResult(
      tool_definitions=tool_definitions,
      tool_to_server=tool_to_server,
      prefixed_to_original=prefixed_to_original,
      mcp_tool_names=mcp_tool_names,
    )

  hidden_names = {
    exposed_name
    for mismatches in hidden_by_server.values()
    for exposed_name, _original_name, _policy_server in mismatches
  }
  filtered_tool_definitions = [
    tool_def
    for tool_def in tool_definitions
    if tool_def.get("name") not in hidden_names
  ]
  filtered_tool_to_server = dict(tool_to_server)
  filtered_prefixed_to_original = dict(prefixed_to_original)
  filtered_mcp_tool_names = set(mcp_tool_names)
  for exposed_name in hidden_names:
    filtered_tool_to_server.pop(exposed_name, None)
    filtered_prefixed_to_original.pop(exposed_name, None)
    filtered_mcp_tool_names.discard(exposed_name)

  for runtime_server, mismatches in hidden_by_server.items():
    server_hidden_names = {
      exposed_name
      for exposed_name, _original_name, _policy_server in mismatches
    }
    state = servers.get(runtime_server)
    if state is not None:
      state.tool_definitions = [
        tool_def
        for tool_def in state.tool_definitions
        if tool_def.get("name") not in server_hidden_names
      ]
      state.tool_names.difference_update(server_hidden_names)

    mismatch_summary = ", ".join(
      f"{exposed_name}({original_name}->{policy_server})"
      for exposed_name, original_name, policy_server in mismatches
    )
    message = (
      "MCP runtime owner does not match gateway policy owner; "
      f"hiding tools: {mismatch_summary}"
    )
    set_startup_diagnostic(
      runtime_server,
      category="policy_owner_mismatch",
      message=message,
      retryable=False,
      error_type="PolicyOwnerMismatch",
    )
    logger.error("%s on runtime server %s", message, runtime_server)

  return PolicyOwnerInvariantResult(
    tool_definitions=filtered_tool_definitions,
    tool_to_server=filtered_tool_to_server,
    prefixed_to_original=filtered_prefixed_to_original,
    mcp_tool_names=filtered_mcp_tool_names,
  )


__all__ = ["PolicyOwnerInvariantResult", "apply_policy_owner_invariant"]
