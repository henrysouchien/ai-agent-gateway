from __future__ import annotations

import copy
from typing import Any, Dict

from .policy_imports import load_server_policy_module
from .secret_boundary import sanitization_failure_tool_input
from .tool_redaction import get_audit_hmac_secret, redact_tool_input


def get_tool_risk_value(tool_name: str) -> str:
  """Use the embedding application's classification without guessing from names."""
  policy = load_server_policy_module()
  resolver = getattr(policy, "get_tool_risk_value", None)
  return resolver(tool_name) if resolver is not None else "side_effecting"




def redact_tool_input_for_event(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
  try:
    redacted = redact_tool_input(
      tool_name,
      copy.deepcopy(tool_input),
      deployment_secret=get_audit_hmac_secret(),
    )
    if not isinstance(redacted, dict):
      raise TypeError("tool input redactor must return a dict")
    return redacted
  except Exception:
    return sanitization_failure_tool_input()
