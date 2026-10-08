from __future__ import annotations


from .policy_imports import load_server_policy_module


def effective_allow_tool_type(
  tool_class: str | None,
  tool_name: str | None,
  requested: bool,
) -> bool:
  """Apply the non-persistable approval policy at the relay boundary."""

  normalized_class = str(tool_class or "").strip().lower()
  normalized_name = str(tool_name or "").strip()
  policy = load_server_policy_module()
  get_trade_opening_tools = getattr(policy, "get_trade_opening_tools", None)
  trade_opening_tools = (
    get_trade_opening_tools() if get_trade_opening_tools is not None else ()
  )
  if normalized_class == "irreversible" or normalized_name in trade_opening_tools:
    return False
  return bool(requested)
