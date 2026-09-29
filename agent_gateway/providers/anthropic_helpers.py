from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .base import ModelInfo


_OAUTH_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
_OAUTH_BETA_SLUGS = [
  "claude-code-20250219",
  "oauth-2025-04-20",
  "fine-grained-tool-streaming-2025-05-14",
]
_COMMON_BETA_SLUGS: list[str] = []
_COMPACTION_BETA_SLUG = "compact-2026-01-12"
_STRUCTURED_OUTPUTS_BETA_SLUG = "structured-outputs-2025-12-15"
_TOOL_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_MAX_TOOL_ID_LEN = 64
_MAX_ERROR_DETAIL_LEN = 800
_ERROR_REDACTION = "[redacted]"
_SENSITIVE_ERROR_KEY_RE = re.compile(r"(api[_-]?key|auth|authorization|bearer|token|secret|password)", re.I)
_SENSITIVE_ERROR_VALUE_RES = (
  re.compile(r"sk-ant-api03-[A-Za-z0-9_-]+"),
  re.compile(r"sk-ant-oat01-[A-Za-z0-9_-]+"),
  re.compile(r"Bearer [A-Za-z0-9_.-]+"),
)

def _thinking_param(model_info: ModelInfo, max_tokens: int) -> dict[str, Any] | None:
  if model_info.thinking_mode == "adaptive":
    # `display: "summarized"` streams thinking text as `thinking_delta`: the
    # stored block keeps the summary beside its signature, and the deltas feed
    # the stream stall guard through long thinking phases that otherwise emit
    # nothing (337 s observed on opus-5 effort high without it).
    return {"type": "adaptive", "display": "summarized"}

  if model_info.thinking_mode == "budget":
    budget_tokens = min(10000, max_tokens - 1024)
    if budget_tokens >= 1024:
      return {"type": "enabled", "budget_tokens": budget_tokens}
    return None

  return None


def _to_plain_dict(value: Any) -> Any:
  if value is None:
    return None
  if hasattr(value, "model_dump"):
    return value.model_dump()
  if isinstance(value, dict):
    return {key: _to_plain_dict(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_to_plain_dict(item) for item in value]
  if hasattr(value, "__dict__"):
    return {key: _to_plain_dict(item) for key, item in vars(value).items() if not key.startswith("_")}
  return value


def _truncate_error_detail(value: Any, limit: int = _MAX_ERROR_DETAIL_LEN) -> str:
  text = str(value).replace("\r", " ").replace("\n", " ").strip()
  for pattern in _SENSITIVE_ERROR_VALUE_RES:
    text = pattern.sub(_ERROR_REDACTION, text)
  if len(text) <= limit:
    return text
  return f"{text[: limit - 3].rstrip()}..."


def _redact_error_body(value: Any) -> Any:
  if isinstance(value, dict):
    redacted: dict[Any, Any] = {}
    for key, item in value.items():
      if _SENSITIVE_ERROR_KEY_RE.search(str(key)):
        redacted[key] = _ERROR_REDACTION
      else:
        redacted[key] = _redact_error_body(item)
    return redacted
  if isinstance(value, list):
    return [_redact_error_body(item) for item in value]
  if isinstance(value, str):
    return _truncate_error_detail(value)
  return value


def _response_header(response: Any, *names: str) -> str:
  headers = getattr(response, "headers", None) or {}
  for name in names:
    try:
      value = headers.get(name)
    except Exception:
      value = None
    if value:
      return _truncate_error_detail(value, limit=160)
  return ""


def _exception_status_code(exc: Exception) -> int | None:
  status_code = getattr(exc, "status_code", None)
  if status_code is None:
    response = getattr(exc, "response", None)
    if response is not None:
      status_code = getattr(response, "status_code", None)
  return status_code if isinstance(status_code, int) else None


def _exception_body(exc: Exception) -> Any:
  body = getattr(exc, "body", None)
  if body is not None:
    return body

  response = getattr(exc, "response", None)
  if response is None:
    return None

  try:
    return response.json()
  except Exception:
    pass

  try:
    text = getattr(response, "text", None)
  except Exception:
    text = None
  if text:
    return text

  try:
    content = getattr(response, "content", None)
  except Exception:
    content = None
  if isinstance(content, bytes):
    return content.decode("utf-8", errors="replace")
  return content


def _format_anthropic_rejection_detail(exc: Exception) -> str | None:
  status_code = _exception_status_code(exc)
  if status_code is None or status_code < 400 or status_code >= 500:
    return None
  if status_code == 429:
    parts = [f"status={status_code}", f"provider_error={type(exc).__name__}"]
    for label, attribute in (
      ("request_id", "provider_request_id"),
      ("limiter", "rate_limit_representative_claim"),
      ("5h_status", "rate_limit_5h_status"),
      ("5h_utilization", "rate_limit_5h_utilization"),
      ("5h_reset", "rate_limit_5h_reset"),
      ("7d_status", "rate_limit_7d_status"),
      ("7d_utilization", "rate_limit_7d_utilization"),
      ("7d_reset", "rate_limit_7d_reset"),
      ("retry_after", "retry_after"),
    ):
      value = getattr(exc, attribute, None)
      if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        continue
      detail = _truncate_error_detail(value, limit=160)
      if detail:
        parts.append(f"{label}={detail}")
    return "; ".join(parts)

  parts = [f"status={status_code}", f"provider_error={type(exc).__name__}"]
  body = _redact_error_body(_exception_body(exc))
  if isinstance(body, dict):
    error = body.get("error") if isinstance(body.get("error"), dict) else body
    error_type = error.get("type") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if error_type:
      parts.append(f"type={_truncate_error_detail(error_type, limit=160)}")
    if message:
      parts.append(f"message={_truncate_error_detail(message)}")
    elif body:
      parts.append(f"body={_truncate_error_detail(json.dumps(body, sort_keys=True, default=str))}")
  elif body:
    parts.append(f"body={_truncate_error_detail(body)}")
  else:
    message = str(exc).strip()
    if message:
      parts.append(f"message={_truncate_error_detail(message)}")

  response = getattr(exc, "response", None)
  request_id = _response_header(
    response,
    "request-id",
    "x-request-id",
    "anthropic-request-id",
    "anthropic-ratelimit-request-id",
  )
  if request_id:
    parts.append(f"request_id={request_id}")
  return "; ".join(parts)


def _stream_request_context(
  stream_params: dict[str, Any],
  *,
  auth_mode: str,
  use_compaction: bool,
  betas: list[str],
) -> str:
  thinking = stream_params.get("thinking")
  if isinstance(thinking, dict):
    thinking_value = str(thinking.get("type") or "enabled")
  elif thinking:
    thinking_value = type(thinking).__name__
  else:
    thinking_value = "disabled"

  messages = stream_params.get("messages")
  tools = stream_params.get("tools")
  parts = [
    f"model={_truncate_error_detail(stream_params.get('model', ''), limit=160)}",
    f"auth_mode={auth_mode or 'api'}",
    f"context_management={'enabled' if use_compaction else 'disabled'}",
    f"thinking={_truncate_error_detail(thinking_value, limit=80)}",
  ]
  if isinstance(messages, list):
    parts.append(f"messages={len(messages)}")
  if isinstance(tools, list):
    parts.append(f"tools={len(tools)}")
  if betas:
    parts.append(f"betas={','.join(betas)}")
  return "; ".join(parts)


def _normalize_tool_call_id(tool_id: str) -> str:
  raw = str(tool_id or "").strip()
  if not raw:
    return "tool"
  normalized = _TOOL_ID_RE.sub("_", raw).strip("_") or "tool"
  if len(normalized) <= _MAX_TOOL_ID_LEN:
    return normalized
  digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
  prefix = normalized[: _MAX_TOOL_ID_LEN - len(digest) - 1].rstrip("_") or "tool"
  return f"{prefix}_{digest}"


def _same_model_message(message: dict[str, Any], model_info: ModelInfo) -> bool:
  if str(message.get("model", "")) != model_info.id:
    return False
  # Session logs written before the provider was recorded on assistant
  # messages carry the model id alone; that id never named another provider.
  provider = message.get("provider")
  return provider is None or str(provider) == model_info.provider


def _synthetic_tool_result(tool_id: str, tool_name: str) -> dict[str, Any]:
  error_message = "No result provided"
  if tool_name:
    error_message = f"No result provided for tool {tool_name}"
  return {
    "type": "tool_result",
    "tool_use_id": tool_id,
    "content": json.dumps({"error": {"code": "missing_tool_result", "message": error_message}}),
    "is_error": True,
  }


def _has_tool_result_block(message: dict[str, Any]) -> bool:
  if message.get("role") != "user":
    return False
  content = message.get("content")
  if not isinstance(content, list):
    return False
  return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
