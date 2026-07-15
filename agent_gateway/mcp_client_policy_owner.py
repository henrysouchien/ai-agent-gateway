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
  policy_tool_class: Callable[[str, str], str | None] | None = None,
  strict_runtime_tool_set_for_server: Callable[[str], bool] | None = None,
  transport_server_for_policy_server: Callable[[str], str] | None = None,
  set_startup_diagnostic: Callable[..., None],
  logger: Any,
) -> PolicyOwnerInvariantResult:
  hidden_by_server: dict[str, list[tuple[str, str, str | None, str]]] = {}
  for exposed_name, runtime_server in sorted(tool_to_server.items()):
    original_name = prefixed_to_original.get(exposed_name, exposed_name)
    policy_server = policy_server_for_tool(original_name)
    policy_runtime_server = (
      transport_server_for_policy_server(policy_server)
      if policy_server and transport_server_for_policy_server is not None
      else policy_server
    )
    if policy_server and policy_runtime_server != runtime_server:
      hidden_by_server.setdefault(runtime_server, []).append(
        (exposed_name, original_name, policy_server, "owner_mismatch")
      )
      continue
    policy_context_server = policy_server or runtime_server
    strict_runtime = bool(
      strict_runtime_tool_set_for_server is not None
      and strict_runtime_tool_set_for_server(policy_context_server)
    )
    if strict_runtime and (
      policy_tool_class is None
      or policy_tool_class(policy_context_server, original_name) is None
    ):
      hidden_by_server.setdefault(runtime_server, []).append(
        (exposed_name, original_name, None, "unclassified")
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
    for exposed_name, _original_name, _policy_server, _reason in mismatches
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
      for exposed_name, _original_name, _policy_server, _reason in mismatches
    }
    state = servers.get(runtime_server)
    if state is not None:
      state.tool_definitions = [
        tool_def
        for tool_def in state.tool_definitions
        if tool_def.get("name") not in server_hidden_names
      ]
      state.tool_names.difference_update(server_hidden_names)

    has_unclassified = any(reason == "unclassified" for *_names, reason in mismatches)
    mismatch_summary = ", ".join(
      (
        f"{exposed_name}({original_name}->unclassified)"
        if reason == "unclassified"
        else f"{exposed_name}({original_name}->{policy_server})"
      )
      for exposed_name, original_name, policy_server, reason in mismatches
    )
    if has_unclassified:
      message = (
        "MCP runtime exposed tools outside its strict gateway policy set; "
        f"hiding tools: {mismatch_summary}"
      )
      category = "strict_runtime_tool_set_mismatch"
      error_type = "StrictRuntimeToolSetMismatch"
    else:
      message = (
        "MCP runtime owner does not match gateway policy owner; "
        f"hiding tools: {mismatch_summary}"
      )
      category = "policy_owner_mismatch"
      error_type = "PolicyOwnerMismatch"
    set_startup_diagnostic(
      runtime_server,
      category=category,
      message=message,
      retryable=False,
      error_type=error_type,
    )
    logger.error("%s on runtime server %s", message, runtime_server)

  return PolicyOwnerInvariantResult(
    tool_definitions=filtered_tool_definitions,
    tool_to_server=filtered_tool_to_server,
    prefixed_to_original=filtered_prefixed_to_original,
    mcp_tool_names=filtered_mcp_tool_names,
  )


__all__ = ["PolicyOwnerInvariantResult", "apply_policy_owner_invariant"]
