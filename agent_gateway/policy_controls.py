from __future__ import annotations


from .policy_imports import load_server_policy_module


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


def normalizer_excluded_tools() -> frozenset[str]:
  """Return the host's normalizer exclusion policy, or no product exclusions."""
  policy = load_server_policy_module()
  get_excluded_tools = getattr(policy, "get_normalizer_excluded_tools", None)
  return frozenset(get_excluded_tools()) if get_excluded_tools is not None else frozenset()
