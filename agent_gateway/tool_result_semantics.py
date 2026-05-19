from __future__ import annotations

from typing import Any


def classify_semantic_tool_error(result: Any) -> dict[str, Any] | None:
  if not isinstance(result, dict):
    return None

  if result.get("success") is False:
    return {
      "code": "tool_success_false",
      "message": _semantic_error_message(result, "Tool result reported success=false"),
      "source": "success",
      "success": False,
    }

  status = result.get("status")
  if isinstance(status, str) and status.strip().lower() == "error":
    payload = {
      "code": "tool_status_error",
      "message": _semantic_error_message(result, "Tool result reported status=error"),
      "source": "status",
      "status": status,
    }
    nested_code = _nested_error_code(result)
    if nested_code:
      payload["sub_code"] = nested_code
    return payload

  return None


def is_semantic_tool_error(result: Any) -> bool:
  return classify_semantic_tool_error(result) is not None


def _semantic_error_message(result: dict[str, Any], fallback: str) -> str:
  nested = result.get("error")
  if isinstance(nested, dict):
    for key in ("message", "detail", "reason"):
      value = nested.get(key)
      if isinstance(value, str) and value.strip():
        return value.strip()
  elif isinstance(nested, str) and nested.strip():
    return nested.strip()

  for key in ("message", "error_message", "detail", "reason"):
    value = result.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()

  return fallback


def _nested_error_code(result: dict[str, Any]) -> str:
  nested = result.get("error")
  if not isinstance(nested, dict):
    return ""
  for key in ("code", "sub_code", "error_code"):
    value = nested.get(key)
    if value is not None and str(value).strip():
      return str(value).strip()
  return ""
