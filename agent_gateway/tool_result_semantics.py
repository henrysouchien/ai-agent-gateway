from __future__ import annotations

import json
from typing import Any

_REPAIR_HINT_EXAMPLE_LIMIT = 600


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
    empty_detail = not status_error_has_detail(result)
    payload = {
      "code": "tool_status_error",
      "message": _semantic_error_message(
        result,
        "Tool result reported status=error without error detail"
        if empty_detail
        else "Tool result reported status=error",
      ),
      "source": "status",
      "status": status,
    }
    nested_code = _nested_error_code(result)
    if nested_code:
      payload["sub_code"] = nested_code
    elif empty_detail:
      payload["sub_code"] = "empty_error_detail"
    return payload

  return None


def is_semantic_tool_error(result: Any) -> bool:
  return classify_semantic_tool_error(result) is not None


def _semantic_error_message(result: dict[str, Any], fallback: str) -> str:
  message = _base_semantic_error_message(result, fallback)
  validation_summary = _validation_errors_summary(result)
  nested = result.get("error")
  if not validation_summary and isinstance(nested, dict):
    validation_summary = _validation_errors_summary(nested)
  if validation_summary and "validation_errors:" not in message:
    message = f"{message}; {validation_summary}"
  repair_hint = _repair_hint_summary(result, existing_message=message)
  if repair_hint:
    message = f"{message}; {repair_hint}"
  return message


def _repair_hint_summary(result: dict[str, Any], *, existing_message: str = "") -> str:
  """Surface structured repair hints (data.fix / data.example) to the model.

  FMS door rejections ship a `fix` instruction and a concrete passing
  `example` inside error.data; without them the model is left guessing
  field names across retries.
  """
  data = None
  nested = result.get("error")
  if isinstance(nested, dict):
    data = nested.get("data")
  if not isinstance(data, dict):
    data = result.get("data")
  if not isinstance(data, dict):
    return ""

  parts: list[str] = []
  fix = data.get("fix")
  if isinstance(fix, str) and fix.strip() and "fix=" not in existing_message:
    parts.append(f"fix={fix.strip()}")
  example = data.get("example")
  if example not in (None, "", {}, []) and "example=" not in existing_message:
    try:
      rendered = json.dumps(example, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
      rendered = str(example)
    if len(rendered) > _REPAIR_HINT_EXAMPLE_LIMIT:
      rendered = rendered[: _REPAIR_HINT_EXAMPLE_LIMIT - 3].rstrip() + "..."
    parts.append(f"example={rendered}")
  return "; ".join(parts)


def _base_semantic_error_message(result: dict[str, Any], fallback: str) -> str:
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


def status_error_has_detail(result: dict[str, Any]) -> bool:
  nested = result.get("error")
  if isinstance(nested, dict):
    if _nested_error_code(result) or _base_semantic_error_message({"error": nested}, "") or _validation_errors_summary(nested):
      return True
  elif isinstance(nested, str):
    if nested.strip():
      return True
  elif nested not in (None, "", {}, []):
    return True

  for key in ("message", "error_message", "detail", "reason", "code", "sub_code", "error_code", "error_type"):
    value = result.get(key)
    if isinstance(value, str) and value.strip():
      return True
    if value not in (None, "", {}, []):
      return True

  return bool(_validation_errors_summary(result))


def _format_validation_error(error: Any) -> str:
  if not isinstance(error, dict):
    return str(error).strip()

  loc = error.get("loc")
  if isinstance(loc, (list, tuple)):
    location = ".".join(str(part) for part in loc if str(part))
  elif loc is not None:
    location = str(loc)
  else:
    location = ""

  message = ""
  for key in ("msg", "message", "reason", "detail"):
    value = error.get(key)
    if isinstance(value, str) and value.strip():
      message = value.strip()
      break
  if not message:
    ctx = error.get("ctx")
    if isinstance(ctx, dict):
      ctx_error = ctx.get("error")
      if ctx_error is not None:
        message = str(ctx_error).strip()
  if not message:
    error_type = error.get("type")
    message = str(error_type).strip() if error_type is not None else "validation error"

  return f"{location}: {message}" if location else message


def _validation_errors_summary(result: dict[str, Any]) -> str:
  raw_errors = result.get("validation_errors")
  if not isinstance(raw_errors, list) or not raw_errors:
    raw_errors = result.get("errors")
  if not isinstance(raw_errors, list) or not raw_errors:
    return ""

  limit = 4
  formatted = [
    text
    for error in raw_errors[:limit]
    if (text := _format_validation_error(error))
  ]
  if not formatted:
    return ""
  remaining = len(raw_errors) - limit
  if remaining > 0:
    formatted.append(f"+{remaining} more")

  summary = "validation_errors: " + "; ".join(formatted)
  required_fields = result.get("required_fields")
  if isinstance(required_fields, list) and required_fields:
    fields = ", ".join(str(field) for field in required_fields if str(field))
    if fields:
      summary += f"; required_fields: {fields}"
  if len(summary) > 800:
    summary = summary[:797].rstrip() + "..."
  return summary


def _nested_error_code(result: dict[str, Any]) -> str:
  nested = result.get("error")
  if isinstance(nested, dict):
    for key in ("code", "sub_code", "error_code", "error_type"):
      value = nested.get(key)
      if value is not None and str(value).strip():
        return str(value).strip()
  for key in ("code", "sub_code", "error_code", "error_type"):
    value = result.get(key)
    if value is not None and str(value).strip():
      return str(value).strip()
  return ""
