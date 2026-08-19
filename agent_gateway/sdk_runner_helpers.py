from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Mapping

from .policy_imports import load_server_policy_helpers


_PARENT_MODULE = "agent_gateway.sdk_runner"


def _compat(name: str) -> Any:
  parent = sys.modules.get(_PARENT_MODULE)
  if parent is not None and hasattr(parent, name):
    return getattr(parent, name)
  return globals()[name.lstrip("_")]


def as_plain_dict(value: Any) -> Any:
  if value is None:
    return None
  if isinstance(value, dict):
    plain_dict = _compat("_as_plain_dict")
    return {key: plain_dict(item) for key, item in value.items()}
  if isinstance(value, list):
    plain_dict = _compat("_as_plain_dict")
    return [plain_dict(item) for item in value]
  if hasattr(value, "model_dump"):
    try:
      return _compat("_as_plain_dict")(value.model_dump())
    except Exception:
      pass
  if hasattr(value, "__dict__"):
    plain_dict = _compat("_as_plain_dict")
    return {
      key: plain_dict(item)
      for key, item in vars(value).items()
      if not key.startswith("_")
    }
  return value


def get_attr(value: Any, key: str, default: Any = None) -> Any:
  if isinstance(value, Mapping):
    return value.get(key, default)
  return getattr(value, key, default)


def as_dict(value: Any) -> Dict[str, Any]:
  plain = _compat("_as_plain_dict")(value)
  if isinstance(plain, dict):
    return plain
  return {}


def extract_text(value: Any) -> str:
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  if isinstance(value, list):
    chunks: List[str] = []
    for item in value:
      if isinstance(item, str):
        if item:
          chunks.append(item)
        continue
      text = _compat("_get_attr")(item, "text")
      if isinstance(text, str) and text:
        chunks.append(text)
        continue
      item_type = _compat("_get_attr")(item, "type")
      if item_type == "text":
        block_text = _compat("_get_attr")(item, "text")
        if isinstance(block_text, str) and block_text:
          chunks.append(block_text)
    return "\n".join(chunks).strip()
  return str(value)


def parse_result_payload(value: Any) -> Any:
  if isinstance(value, (dict, list)):
    return _compat("_as_plain_dict")(value)
  if isinstance(value, str):
    stripped = value.strip()
    if not stripped:
      return ""
    try:
      return json.loads(stripped)
    except json.JSONDecodeError:
      return value
  if value is None:
    return None
  return _compat("_as_plain_dict")(value)


def summarize_error_payload(value: Any) -> str:
  parsed = _compat("_parse_result_payload")(value)
  if isinstance(parsed, dict):
    inner = parsed.get("error")
    if isinstance(inner, dict):
      message = inner.get("message")
      if isinstance(message, str) and message.strip():
        return message.strip()
    message = parsed.get("message")
    if isinstance(message, str) and message.strip():
      return message.strip()
    if parsed.get("success") is False:
      return "success=false"
    status = parsed.get("status")
    if isinstance(status, str) and status.strip():
      return f"status={status.strip()}"
    return json.dumps(parsed, default=str)
  if isinstance(parsed, list):
    return json.dumps(parsed, default=str)
  return str(parsed or "Tool failed")


def join_system_prompt(system_prompt: str | List[tuple[str, bool]] | None) -> str:
  if system_prompt is None:
    return ""
  if isinstance(system_prompt, str):
    return system_prompt
  return "\n\n".join(text for text, _should_cache in system_prompt if text)


def server_for_tool(tool_name: str) -> str | None:
  if not tool_name.startswith("mcp__"):
    return None
  parts = tool_name.split("__", 2)
  if len(parts) < 3 or not parts[1]:
    return None
  return parts[1]


def policy_owner_mismatch(tool_name: str) -> tuple[str, str, str] | None:
  runtime_server = _compat("_server_for_tool")(tool_name)
  if not runtime_server:
    return None
  policy_tool = _compat("_policy_tool_name")(tool_name)
  _get_forbidden_tools_for_session, get_server_for_policy_tool, _get_tool_class = load_server_policy_helpers()
  if get_server_for_policy_tool is None:
    return None
  policy_server = get_server_for_policy_tool(policy_tool)
  if policy_server and policy_server != runtime_server:
    return runtime_server, policy_tool, policy_server
  return None


def redact_tool_input_for_event(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
  from .runner_tool_audit import redact_tool_input_for_event as redact

  return redact(tool_name, tool_input)


def policy_tool_name(tool_name: str) -> str:
  if tool_name.startswith("mcp__"):
    parts = tool_name.split("__", 2)
    if len(parts) == 3:
      return parts[2]
  return tool_name


PATCH_OP_RAW_INPUT_TOOLS = frozenset({
  "apply_patch_ops",
  "preview_patch_ops",
  "publish_price_target",
})


def should_escrow_raw_tool_input(tool_name: str) -> bool:
  return _compat("_policy_tool_name")(tool_name) in _compat("_PATCH_OP_RAW_INPUT_TOOLS")


__all__ = [
  "PATCH_OP_RAW_INPUT_TOOLS",
  "as_dict",
  "as_plain_dict",
  "extract_text",
  "get_attr",
  "join_system_prompt",
  "parse_result_payload",
  "policy_owner_mismatch",
  "policy_tool_name",
  "redact_tool_input_for_event",
  "server_for_tool",
  "should_escrow_raw_tool_input",
  "summarize_error_payload",
]
