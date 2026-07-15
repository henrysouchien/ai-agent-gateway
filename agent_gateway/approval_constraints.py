from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from .approval_policy import ApprovalConstraint, ApprovalConstraintError


_SUPPORTED_COMMIT_STRATEGIES = frozenset({
  "JOURNALED_CAS",
  "ARTIFACT_ONLY",
  "PROPOSAL_STAGE",
  "THESIS_TRANSACTION",
  "PROMOTION_SAGA",
})


def trusted_catalog_action(
  tool_name: str,
  *,
  import_module_fn: Callable[[str], Any] = import_module,
) -> Any | None:
  """Load one action from the first supported authoritative catalog layout."""

  action_catalog: Any | None = None
  for module_name in ("fms.action_catalog", "api.fms.action_catalog"):
    try:
      module = import_module_fn(module_name)
    except ModuleNotFoundError as exc:
      parts = module_name.split(".")
      candidate_or_parent = {
        ".".join(parts[:index])
        for index in range(1, len(parts) + 1)
      }
      if exc.name not in candidate_or_parent:
        raise
      continue
    action_catalog = module.ACTION_CATALOG
    break
  if action_catalog is None:
    raise ApprovalConstraintError(
      "trusted FMS action catalog is unavailable in every supported layout"
    )
  matches = [
    action
    for action in action_catalog
    if action.local_tool_name == tool_name
  ]
  if len(matches) > 1:
    raise ApprovalConstraintError(
      f"action catalog contains duplicate local tool {tool_name!r}"
    )
  return None if not matches else matches[0]


def constraint_for_catalog_action(action: Any | None) -> ApprovalConstraint:
  """Classify a trusted catalog row by normalized strategy text."""

  if action is None:
    return "standard"
  strategy = getattr(action, "commit_strategy", None)
  normalized = getattr(strategy, "value", strategy)
  if normalized is None:
    return "standard"
  if not isinstance(normalized, str) or normalized not in _SUPPORTED_COMMIT_STRATEGIES:
    raise ApprovalConstraintError("trusted action has an unsupported commit strategy")
  if normalized == "PROMOTION_SAGA":
    return "fresh_human_owner"
  return "standard"


def constraint_for_catalog_tool(tool_name: str) -> ApprovalConstraint:
  return constraint_for_catalog_action(trusted_catalog_action(tool_name))


def requires_owner_control_route(tool_name: str) -> bool:
  return constraint_for_catalog_tool(tool_name) == "fresh_human_owner"


__all__ = [
  "constraint_for_catalog_action",
  "constraint_for_catalog_tool",
  "requires_owner_control_route",
  "trusted_catalog_action",
]
