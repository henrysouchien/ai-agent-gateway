from __future__ import annotations

import copy
from typing import Any, Dict

from .secret_boundary import sanitization_failure_tool_input


_ABSENT_HOST_REDACTION_MODULES = frozenset({
  "agent",
  "agent.shared",
  "agent.shared.tool_redaction",
})


def _tool_input_redactor() -> tuple[Any, Any]:
  try:
    from agent.shared.tool_redaction import get_audit_hmac_secret, redact_tool_input
  except ModuleNotFoundError as exc:
    if exc.name not in _ABSENT_HOST_REDACTION_MODULES:
      raise
    from .tool_redaction import get_audit_hmac_secret, redact_tool_input

  return get_audit_hmac_secret, redact_tool_input


def get_tool_risk_value(tool_name: str) -> str:
  normalized = str(tool_name or "").strip()
  try:
    from api.agent.shared.tool_risk import get_tool_risk
  except Exception:
    if normalized in {"file_read", "memory_read", "memory_recall", "web_search", "web_fetch"}:
      return "read_only"
    if normalized.startswith(
      (
        "analyze_",
        "check_",
        "compare_",
        "describe_",
        "fetch_",
        "file_read",
        "get_",
        "list_",
        "preview_",
        "query_",
        "read_",
        "recall_",
        "screen_",
        "search_",
        "view_",
      )
    ):
      return "read_only"
    if normalized.startswith(("memory_", "set_", "sync_", "upsert_")):
      return "idempotent_write"
    return "side_effecting"
  try:
    return get_tool_risk(normalized).value
  except Exception:
    return "side_effecting"


def redact_tool_input_for_event(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
  try:
    get_audit_hmac_secret, redact_tool_input = _tool_input_redactor()

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
