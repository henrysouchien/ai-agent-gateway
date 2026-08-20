from __future__ import annotations

import importlib
from typing import Any

from .policy_imports import load_server_policy_module


_POLICY_CATALOG_MODULE_NAMES = (
  "agent.shared.server_policy_catalog",
  "api.agent.shared.server_policy_catalog",
)
_POLICY_CATALOG_IMPORT_ROOTS = frozenset({
  "agent",
  "agent.shared",
  "agent.shared.server_policy_catalog",
  "api",
  "api.agent",
  "api.agent.shared",
  "api.agent.shared.server_policy_catalog",
})
_FALLBACK_TRADE_OPENING_TOOLS = frozenset({
  "execute_trade",
  "execute_basket_trade",
  "execute_futures_roll",
  "execute_option_trade",
})
_NON_OPENING_IRREVERSIBLE_TOOLS = frozenset({"cancel_order"})
_WRITE_TOOL_CLASSES = (
  "artifact_write",
  "state_write",
  "external_write",
  "portfolio_config",
  "irreversible",
)
HOSTED_PRIVATE_EDGAR_TOOLS = frozenset({"extract_filing_file"})
HOSTED_UNAVAILABLE_NORMALIZER_TOOLS = frozenset({
  "normalizer_activate",
  "normalizer_detect",
  "normalizer_list",
  "normalizer_register_institution",
  "normalizer_sample_csv",
  "normalizer_stage",
  "normalizer_test",
  "normalizer_update",
  "normalizer_validate",
  "statement_normalizer_activate",
  "statement_normalizer_list",
  "statement_normalizer_sample_csv",
  "statement_normalizer_stage",
  "statement_normalizer_test",
})


def _load_portfolio_irreversible_tools() -> frozenset[str]:
  for module_name in _POLICY_CATALOG_MODULE_NAMES:
    try:
      module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
      if exc.name not in _POLICY_CATALOG_IMPORT_ROOTS:
        raise
      continue
    raw_tools = getattr(module, "PORTFOLIO_IRREVERSIBLE_TOOLS", ())
    try:
      tools = frozenset(
        str(tool_name).strip()
        for tool_name in raw_tools
        if str(tool_name or "").strip()
      )
    except TypeError:
      continue
    if tools:
      return tools
  return _FALLBACK_TRADE_OPENING_TOOLS | _NON_OPENING_IRREVERSIBLE_TOOLS


TRADE_OPENING_TOOLS = (
  _load_portfolio_irreversible_tools() - _NON_OPENING_IRREVERSIBLE_TOOLS
) or _FALLBACK_TRADE_OPENING_TOOLS


def effective_allow_tool_type(
  tool_class: str | None,
  tool_name: str | None,
  requested: bool,
) -> bool:
  """Apply the non-persistable approval policy at the relay boundary."""

  normalized_class = str(tool_class or "").strip().lower()
  normalized_name = str(tool_name or "").strip()
  if normalized_class == "irreversible" or normalized_name in TRADE_OPENING_TOOLS:
    return False
  return bool(requested)


def normalizer_excluded_tools() -> frozenset[str]:
  """Return write-class tools that normalizer sessions must never see."""

  excluded = set(TRADE_OPENING_TOOLS)
  policy_module: Any | None = load_server_policy_module()
  get_class_tools = getattr(policy_module, "get_class_tools", None)
  if callable(get_class_tools):
    for tool_class in _WRITE_TOOL_CLASSES:
      raw_tools = get_class_tools(tool_class)
      try:
        excluded.update(
          str(tool_name).strip()
          for tool_name in raw_tools
          if str(tool_name or "").strip()
        )
      except TypeError:
        continue
  local_tool_effect = getattr(policy_module, "LOCAL_TOOL_EFFECT", {})
  try:
    excluded.update(
      str(tool_name).strip()
      for tool_name, effect in local_tool_effect.items()
      if str(tool_name or "").strip() and effect in _WRITE_TOOL_CLASSES
    )
  except AttributeError:
    pass
  return frozenset(
    tool_name
    for tool_name in excluded
    if not tool_name.startswith("normalizer_")
  )
