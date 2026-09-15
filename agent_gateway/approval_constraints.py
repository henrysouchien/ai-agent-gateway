from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from .approval_policy import ApprovalConstraint, ApprovalConstraintError


_SUPPORTED_COMMIT_STRATEGIES = frozenset({
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
  """Load one action from the authoritative FMS action catalog."""

  try:
    module = import_module_fn("fms.action_catalog")
  except ModuleNotFoundError as exc:
    if exc.name not in {"fms", "fms.action_catalog"}:
      raise
    raise ApprovalConstraintError(
      "trusted FMS action catalog is unavailable"
    ) from exc
  action_catalog = module.ACTION_CATALOG
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


__all__ = [
  "constraint_for_catalog_action",
  "constraint_for_catalog_tool",
  "trusted_catalog_action",
]
