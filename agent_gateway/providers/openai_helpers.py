from __future__ import annotations

import json
import re
from typing import Any

from .base import ModelInfo, ThinkingLevel

_TOOL_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_MAX_TOOL_ID_LEN = 40


def _base_compat(*, supports_reasoning_effort: bool) -> dict[str, Any]:
  return {
    "supportsStore": True,
    "supportsDeveloperRole": True,
    "supportsReasoningEffort": supports_reasoning_effort,
    "reasoningEffortMap": {},
    "supportsUsageInStreaming": True,
    "maxTokensField": "max_completion_tokens",
    "requiresToolResultName": False,
    "requiresAssistantAfterToolResult": False,
    "requiresThinkingAsText": False,
    "thinkingFormat": "openai",
    "supportsStrictMode": True,
  }


_MODEL_INFO_BY_TAG: list[tuple[tuple[str, ...], ModelInfo]] = [
  (
    ("gpt-5.5",),
    ModelInfo(
      id="gpt-5.5",
      provider="openai",
      context_window=1_050_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=5.00,
      output_cost_per_mtok=30.00,
      cache_read_cost_per_mtok=0.50,
      compat=_base_compat(supports_reasoning_effort=True),
    ),
  ),
  (
    ("gpt-4o-mini",),
    ModelInfo(
      id="gpt-4o-mini",
      provider="openai",
      context_window=128_000,
      max_output_tokens=16_384,
      supports_thinking=False,
      supports_vision=True,
      input_cost_per_mtok=0.15,
      output_cost_per_mtok=0.60,
      cache_read_cost_per_mtok=0.075,
      compat=_base_compat(supports_reasoning_effort=False),
    ),
  ),
  (
    ("gpt-4o",),
    ModelInfo(
      id="gpt-4o",
      provider="openai",
      context_window=128_000,
      max_output_tokens=16_384,
      supports_thinking=False,
      supports_vision=True,
      input_cost_per_mtok=2.50,
      output_cost_per_mtok=10.00,
      cache_read_cost_per_mtok=1.25,
      compat=_base_compat(supports_reasoning_effort=False),
    ),
  ),
  (
    ("o1-mini",),
    ModelInfo(
      id="o1-mini",
      provider="openai",
      context_window=128_000,
      max_output_tokens=65_536,
      supports_thinking=True,
      supports_vision=False,
      input_cost_per_mtok=1.10,
      output_cost_per_mtok=4.40,
      cache_read_cost_per_mtok=0.55,
      compat=_base_compat(supports_reasoning_effort=True),
    ),
  ),
  (
    ("o3-mini",),
    ModelInfo(
      id="o3-mini",
      provider="openai",
      context_window=200_000,
      max_output_tokens=100_000,
      supports_thinking=True,
      supports_vision=False,
      input_cost_per_mtok=1.10,
      output_cost_per_mtok=4.40,
      cache_read_cost_per_mtok=0.55,
      compat=_base_compat(supports_reasoning_effort=True),
    ),
  ),
  (
    ("o1",),
    ModelInfo(
      id="o1",
      provider="openai",
      context_window=200_000,
      max_output_tokens=100_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=15.00,
      output_cost_per_mtok=60.00,
      cache_read_cost_per_mtok=7.50,
      compat=_base_compat(supports_reasoning_effort=True),
    ),
  ),
]


def _field(value: Any, name: str, default: Any = None) -> Any:
  if value is None:
    return default
  if isinstance(value, dict):
    return value.get(name, default)
  return getattr(value, name, default)


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


def _model_matches_tag(model_id: str, tag: str) -> bool:
  candidates = [model_id, model_id.rsplit("/", 1)[-1]]
  return any(candidate == tag or candidate.startswith(f"{tag}-") for candidate in candidates)


def _normalize_tool_call_id(tool_id: str) -> str:
  raw = str(tool_id or "").strip()
  if not raw:
    return "call"
  if "|" in raw:
    raw = raw.split("|", 1)[0]
  normalized = _TOOL_ID_RE.sub("_", raw).strip("_") or "call"
  return normalized[:_MAX_TOOL_ID_LEN]


def _same_model_message(message: dict[str, Any], model_info: ModelInfo) -> bool:
  return (
    str(message.get("provider", "")) == model_info.provider
    and str(message.get("model", "")) == model_info.id
  )


def _synthetic_tool_result(tool_id: str, tool_name: str) -> dict[str, Any]:
  return {
    "type": "tool_result",
    "tool_use_id": tool_id,
    "content": json.dumps({"error": {"code": "missing_tool_result", "message": "No result provided"}}),
    "is_error": True,
    "tool_name": tool_name,
  }


def _is_tool_result_message(message: dict[str, Any]) -> bool:
  if message.get("role") != "user":
    return False
  content = message.get("content")
  if not isinstance(content, list) or not content:
    return False
  return all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def _contains_tool_history(messages: list[dict[str, Any]]) -> bool:
  for message in messages:
    role = message.get("role")
    content = message.get("content")
    if role == "assistant" and isinstance(content, list):
      for block in content:
        if isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use"}:
          return True
    if role == "user" and isinstance(content, list):
      for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
          return True
  return False


def _system_prompt_text(system_prompt: str | list[tuple[str, bool]] | None) -> str:
  if system_prompt is None:
    return ""
  if isinstance(system_prompt, list):
    return "\n\n".join(text for text, _should_cache in system_prompt if text)
  return str(system_prompt)


def _stringify_tool_result_content(content: Any) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    texts = [
      str(block.get("text", ""))
      for block in content
      if isinstance(block, dict) and block.get("type") == "text"
    ]
    if texts:
      return "\n".join(text for text in texts if text)
    return "(see attached image)"
  if isinstance(content, (dict, list)):
    return json.dumps(content, default=str)
  if content is None:
    return ""
  return str(content)


def _map_reasoning_effort(level: ThinkingLevel) -> str | None:
  if level == ThinkingLevel.NONE:
    return None
  if level == ThinkingLevel.MINIMAL:
    return "minimal"
  if level == ThinkingLevel.LOW:
    return "low"
  if level == ThinkingLevel.MEDIUM:
    return "medium"
  return "high"


def _image_part(block: dict[str, Any]) -> dict[str, Any] | None:
  data = block.get("data_base64") or block.get("data")
  media_type = block.get("media_type") or block.get("mime_type") or block.get("mimeType")
  if not data or not media_type:
    return None
  return {
    "type": "image_url",
    "image_url": {
      "url": f"data:{media_type};base64,{data}",
    },
  }


def _detect_compat(base_url: str | None) -> dict[str, Any]:
  base = str(base_url or "").lower()
  is_zai = "api.z.ai" in base
  is_grok = "api.x.ai" in base
  is_non_standard = any(
    part in base
    for part in ("cerebras.ai", "api.x.ai", "chutes.ai", "deepseek.com", "opencode.ai")
  ) or is_zai
  compat = _base_compat(supports_reasoning_effort=not is_grok and not is_zai)
  compat["supportsStore"] = not is_non_standard
  compat["supportsDeveloperRole"] = not is_non_standard
  compat["maxTokensField"] = "max_tokens" if "chutes.ai" in base else "max_completion_tokens"
  compat["thinkingFormat"] = "zai" if is_zai else "openai"
  return compat
