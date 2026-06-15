from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import replace
from typing import Any, AsyncIterator, Dict

from ..rate_limit import get_global_token_bucket
from ..rates import RateTable, UnknownModelError, load_rate_table
from .base import CostEstimate, ModelInfo, ModelProvider, StreamEvent, ThinkingLevel, truncate_to_last_compaction


log = logging.getLogger("agent_gateway.providers.anthropic")

_OAUTH_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
_OAUTH_BETA_SLUGS = [
  "claude-code-20250219",
  "oauth-2025-04-20",
  "fine-grained-tool-streaming-2025-05-14",
]
_COMMON_BETA_SLUGS: list[str] = []
_COMPACTION_BETA_SLUG = "compact-2026-01-12"
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

_MODEL_INFO_BY_TAG: list[tuple[tuple[str, ...], ModelInfo]] = [
  (
    ("claude-fable-5",),
    ModelInfo(
      id="claude-fable-5",
      provider="anthropic",
      context_window=1_000_000,
      max_output_tokens=32_000,
      supports_thinking=True,
      thinking_mode="adaptive",
      input_cost_per_mtok=10.00,
      output_cost_per_mtok=50.00,
      cache_read_cost_per_mtok=1.00,
      cache_write_cost_per_mtok=12.50,
    ),
  ),
  (
    ("claude-opus-4-8",),
    ModelInfo(
      id="claude-opus-4-8",
      provider="anthropic",
      context_window=1_000_000,
      max_output_tokens=32_000,
      supports_thinking=True,
      thinking_mode="adaptive",
      input_cost_per_mtok=5.00,
      output_cost_per_mtok=25.00,
      cache_read_cost_per_mtok=0.50,
      cache_write_cost_per_mtok=6.25,
    ),
  ),
  (
    ("claude-opus-4-7",),
    ModelInfo(
      id="claude-opus-4-7",
      provider="anthropic",
      context_window=1_000_000,
      max_output_tokens=32_000,
      supports_thinking=True,
      thinking_mode="adaptive",
      input_cost_per_mtok=5.00,
      output_cost_per_mtok=25.00,
      cache_read_cost_per_mtok=0.50,
      cache_write_cost_per_mtok=6.25,
    ),
  ),
  (
    ("claude-sonnet-4-6",),
    ModelInfo(
      id="claude-sonnet-4-6",
      provider="anthropic",
      max_output_tokens=64_000,
      supports_thinking=True,
      thinking_mode="adaptive",
      input_cost_per_mtok=3.00,
      output_cost_per_mtok=15.00,
      cache_read_cost_per_mtok=0.30,
      cache_write_cost_per_mtok=3.75,
    ),
  ),
  (
    ("claude-opus-4-6",),
    ModelInfo(
      id="claude-opus-4-6",
      provider="anthropic",
      supports_thinking=True,
      thinking_mode="adaptive",
      input_cost_per_mtok=3.00,
      output_cost_per_mtok=15.00,
      cache_read_cost_per_mtok=0.30,
      cache_write_cost_per_mtok=3.75,
    ),
  ),
  (
    ("claude-sonnet-4-5", "claude-opus-4-5", "claude-sonnet-4"),
    ModelInfo(
      id="claude-sonnet-4-5",
      provider="anthropic",
      max_output_tokens=64_000,
      supports_thinking=True,
      thinking_mode="budget",
      input_cost_per_mtok=3.00,
      output_cost_per_mtok=15.00,
      cache_read_cost_per_mtok=0.30,
      cache_write_cost_per_mtok=3.75,
    ),
  ),
  (
    ("claude-haiku-4-5",),
    ModelInfo(
      id="claude-haiku-4-5",
      provider="anthropic",
      supports_thinking=False,
      thinking_mode="none",
      input_cost_per_mtok=1.00,
      output_cost_per_mtok=5.00,
      cache_read_cost_per_mtok=0.10,
      cache_write_cost_per_mtok=1.25,
    ),
  ),
  (
    ("claude-3",),
    ModelInfo(
      id="claude-3",
      provider="anthropic",
      supports_thinking=False,
      thinking_mode="none",
      input_cost_per_mtok=3.00,
      output_cost_per_mtok=15.00,
      cache_read_cost_per_mtok=0.30,
      cache_write_cost_per_mtok=3.75,
    ),
  ),
]


def _model_matches_tag(model_id: str, tag: str) -> bool:
  return model_id == tag or tag in model_id or model_id.startswith(f"{tag}-")


def _model_info_for_model(model_id: str) -> ModelInfo:
  for tags, info in _MODEL_INFO_BY_TAG:
    if any(_model_matches_tag(model_id, tag) for tag in tags):
      return replace(info, id=model_id)
  return ModelInfo(
    id=model_id,
    provider="anthropic",
    supports_thinking=True,
    thinking_mode="adaptive",
    input_cost_per_mtok=3.00,
    output_cost_per_mtok=15.00,
    cache_read_cost_per_mtok=0.30,
    cache_write_cost_per_mtok=3.75,
  )


def _thinking_param(model_info: ModelInfo, max_tokens: int) -> dict[str, Any] | None:
  if model_info.thinking_mode == "adaptive":
    return {"type": "adaptive"}

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
  if status_code is None or status_code < 400 or status_code == 429 or status_code >= 500:
    return None

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
  return (
    str(message.get("provider", "")) == model_info.provider
    and str(message.get("model", "")) == model_info.id
  )


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


class AnthropicProvider(ModelProvider):
  """`ModelProvider` implementation for Anthropic's Messages API.

  Supports API key and OAuth-style auth, prompt caching blocks, tool use, and
  Anthropic thinking/compaction features when the selected model allows them.
  """

  name = "anthropic"
  supports_compaction = True

  def __init__(self, *, rate_table: RateTable | None = None):
    self._rate_table = rate_table or load_rate_table()

  @staticmethod
  def thinking_param(model: str, max_tokens: int) -> dict[str, Any] | None:
    model_id = str(model or "").strip()
    if not model_id.startswith("claude"):
      return None
    return _thinking_param(_model_info_for_model(model_id), max_tokens)

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    if str(config.get("auth_mode", "api")).strip().lower() == "oauth":
      return bool(config.get("auth_token"))
    return bool(config.get("api_key"))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    try:
      from anthropic import AsyncAnthropic
      import httpx
    except ImportError as exc:
      raise RuntimeError("anthropic dependency is required to use AnthropicProvider") from exc

    mode = str(config.get("auth_mode", "api")).strip().lower()
    client_kwargs: Dict[str, Any] = {}
    if timeout is not None:
      client_kwargs["timeout"] = httpx.Timeout(timeout=timeout, connect=5.0)

    if mode == "oauth":
      from anthropic import Omit
      oauth_headers = {
        "X-Api-Key": Omit(),
        "anthropic-beta": ",".join([*_OAUTH_BETA_SLUGS, *_COMMON_BETA_SLUGS]),
        "user-agent": "claude-cli/2026.3.14",
        "x-app": "cli",
      }

      return AsyncAnthropic(
        api_key="",
        auth_token=str(config.get("auth_token", "")),
        default_headers=oauth_headers,
        **client_kwargs,
      )

    if _COMMON_BETA_SLUGS:
      client_kwargs["default_headers"] = {"anthropic-beta": ",".join(_COMMON_BETA_SLUGS)}
    # Suppress ANTHROPIC_AUTH_TOKEN env pickup — SDK reads it when auth_token kwarg
    # is omitted, and auth_token="" causes LocalProtocolError (Bearer with empty token).
    saved_token = os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    try:
      return AsyncAnthropic(
        api_key=str(config.get("api_key", "")),
        **client_kwargs,
      )
    finally:
      if saved_token is not None:
        os.environ["ANTHROPIC_AUTH_TOKEN"] = saved_token

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    if client is None:
      return
    closer = getattr(client, "aclose", None) or getattr(client, "close", None)
    if closer is None:
      return
    try:
      result = closer()
      if asyncio.iscoroutine(result):
        await asyncio.wait_for(result, timeout=timeout)
    except Exception:
      pass

  def get_model_info(self, model: str) -> ModelInfo:
    model_id = str(model or "").strip()
    if not model_id:
      raise ValueError("Model is required")
    if not model_id.startswith("claude"):
      raise ValueError(f"AnthropicProvider does not recognize model: {model_id}")
    model_info = _model_info_for_model(model_id)
    try:
      rates = self._rate_table.lookup(self.name, model_id)
    except UnknownModelError:
      return model_info
    return replace(
      model_info,
      context_window=rates.context_window or 200_000,
      max_output_tokens=rates.max_tokens or 16_384,
      input_cost_per_mtok=rates.input_cost_per_mtok,
      output_cost_per_mtok=rates.output_cost_per_mtok,
      cache_read_cost_per_mtok=rates.cache_read_cost_per_mtok,
      cache_write_cost_per_mtok=rates.cache_write_cost_per_mtok,
    )

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    thinking_level: ThinkingLevel = ThinkingLevel.HIGH,
    **kwargs: Any,
  ) -> dict[str, Any]:
    auth_mode = str(kwargs.get("auth_mode", "api")).strip().lower()
    system_blocks: list[dict[str, Any]] | None = None
    if auth_mode == "oauth":
      system_blocks = [{"type": "text", "text": _OAUTH_IDENTITY}]
    model_info = self.get_model_info(model)

    if system_prompt:
      if isinstance(system_prompt, list):
        if system_blocks is None:
          system_blocks = []
        for block_text, should_cache in system_prompt:
          if not block_text:
            continue
          block: Dict[str, Any] = {"type": "text", "text": block_text}
          if should_cache:
            block["cache_control"] = {"type": "ephemeral"}
          system_blocks.append(block)
        if not system_blocks:
          system_blocks = None
      else:
        block = {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        if system_blocks is None:
          system_blocks = [block]
        else:
          system_blocks.append(block)

    params: Dict[str, Any] = {
      "model": model,
      "max_tokens": max_tokens,
      "messages": messages,
      "tools": tools,
      "_provider_auth_mode": auth_mode,
    }
    if system_blocks:
      params["system"] = system_blocks

    if thinking_level != ThinkingLevel.NONE and max_tokens >= 2048:
      thinking_param = _thinking_param(model_info, max_tokens)
      if thinking_param is not None:
        params["thinking"] = thinking_param

    compaction_trigger = kwargs.get("compaction_trigger")
    if compaction_trigger is not None:
      compact_edit: Dict[str, Any] = {
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": compaction_trigger},
        "pause_after_compaction": False,
      }
      compaction_instructions = kwargs.get("compaction_instructions")
      if compaction_instructions is not None:
        compact_edit["instructions"] = compaction_instructions
      params["context_management"] = {"edits": [compact_edit]}

    return params

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    tool_id_map: dict[str, str] = {}
    transformed: list[dict[str, Any]] = []

    for message in messages:
      role = message.get("role")
      if role == "assistant":
        if str(message.get("stop_reason", "")) in {"error", "aborted"}:
          continue

        is_same_model = _same_model_message(message, model_info)
        content = message.get("content")
        if not isinstance(content, list):
          transformed.append({"role": "assistant", "content": content})
          continue

        next_content: list[dict[str, Any]] = []
        for block in content:
          if not isinstance(block, dict):
            continue

          block_type = block.get("type")
          if block_type == "thinking":
            thinking_text = str(block.get("thinking", ""))
            if is_same_model:
              next_content.append(dict(block))
            elif thinking_text.strip():
              next_content.append({"type": "text", "text": thinking_text})
            continue

          if block_type in {"tool_use", "server_tool_use"}:
            next_block = dict(block)
            tool_id = str(block.get("id", ""))
            if tool_id:
              normalized_id = _normalize_tool_call_id(tool_id)
              if normalized_id != tool_id:
                tool_id_map[tool_id] = normalized_id
                next_block["id"] = normalized_id
            next_content.append(next_block)
            continue

          if block_type == "text":
            next_content.append({"type": "text", "text": str(block.get("text", ""))})
            continue

          next_content.append(dict(block))

        next_message = {"role": "assistant", "content": next_content}
        transformed.append(next_message)
        continue

      if role == "user":
        content = message.get("content")
        if isinstance(content, list):
          next_content: list[dict[str, Any]] = []
          for block in content:
            if not isinstance(block, dict):
              continue
            if block.get("type") == "tool_result":
              next_block = dict(block)
              next_block.pop("tool_name", None)
              tool_use_id = str(block.get("tool_use_id", ""))
              normalized_id = tool_id_map.get(tool_use_id)
              if normalized_id is not None:
                next_block["tool_use_id"] = normalized_id
              next_content.append(next_block)
            else:
              next_content.append(dict(block))
          transformed.append({"role": "user", "content": next_content})
        else:
          transformed.append({"role": "user", "content": content})
        continue

      transformed.append({"role": role, "content": message.get("content")})

    result: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    existing_tool_result_ids: set[str] = set()

    for message in transformed:
      role = message.get("role")
      if role == "assistant":
        if pending_tool_calls:
          missing = [
            _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
            for block in pending_tool_calls
            if str(block.get("id", "")) not in existing_tool_result_ids
          ]
          if missing:
            result.append({"role": "user", "content": missing})
          pending_tool_calls = []
          existing_tool_result_ids = set()

        content = message.get("content")
        if isinstance(content, list):
          pending_tool_calls = [
            dict(block)
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
          ]
        else:
          pending_tool_calls = []

        result.append(message)
        continue

      if role == "user" and pending_tool_calls and _has_tool_result_block(message):
        content = list(message.get("content") or [])
        pending_ids = {str(block.get("id", "")) for block in pending_tool_calls}
        filtered_content: list[Any] = []
        existing_tool_result_ids = {
          str(block.get("tool_use_id", ""))
          for block in content
          if (
            isinstance(block, dict)
            and block.get("type") == "tool_result"
            and str(block.get("tool_use_id", "")) in pending_ids
          )
        }
        for block in content:
          if not isinstance(block, dict) or block.get("type") != "tool_result":
            filtered_content.append(block)
            continue
          if str(block.get("tool_use_id", "")) in pending_ids:
            filtered_content.append(block)
        missing = [
          _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
          for block in pending_tool_calls
          if str(block.get("id", "")) not in existing_tool_result_ids
        ]
        if missing:
          next_message = dict(message)
          next_message["content"] = [*filtered_content, *missing]
          result.append(next_message)
        else:
          next_message = dict(message)
          next_message["content"] = filtered_content
          result.append(next_message)
        pending_tool_calls = []
        existing_tool_result_ids = set()
        continue

      if pending_tool_calls:
        missing = [
          _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
          for block in pending_tool_calls
          if str(block.get("id", "")) not in existing_tool_result_ids
        ]
        if missing:
          result.append({"role": "user", "content": missing})
        pending_tool_calls = []
        existing_tool_result_ids = set()

      if role == "user" and _has_tool_result_block(message):
        content = [
          block
          for block in list(message.get("content") or [])
          if not (isinstance(block, dict) and block.get("type") == "tool_result")
        ]
        if content:
          next_message = dict(message)
          next_message["content"] = content
          result.append(next_message)
        continue

      result.append(message)

    if pending_tool_calls:
      missing = [
        _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
        for block in pending_tool_calls
        if str(block.get("id", "")) not in existing_tool_result_ids
      ]
      if missing:
        result.append({"role": "user", "content": missing})

    return truncate_to_last_compaction(result)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    bucket = get_global_token_bucket()
    if bucket is not None:
      await bucket.acquire(estimated_tokens=int(params.get("max_tokens") or 0))

    stream_params = dict(params)
    auth_mode = str(stream_params.pop("_provider_auth_mode", "api")).strip().lower()
    use_compaction = "context_management" in stream_params
    betas: list[str] = []
    stop_reason = ""
    current_block_type: str | None = None
    current_tool_id: str | None = None
    current_tool_name: str | None = None
    current_tool_json = ""
    current_tool_block: Any = None
    current_text = ""
    current_thinking = ""
    current_signature = ""
    current_compaction: Any = None

    try:
      if use_compaction:
        betas = [*_COMMON_BETA_SLUGS, _COMPACTION_BETA_SLUG]
        if auth_mode == "oauth":
          betas = [*_OAUTH_BETA_SLUGS, *_COMMON_BETA_SLUGS, _COMPACTION_BETA_SLUG]
        stream_cm = client.beta.messages.stream(**stream_params, betas=betas)
      else:
        stream_cm = client.messages.stream(**stream_params)

      async with stream_cm as stream_obj:
        async for event in stream_obj:
          event_type = getattr(event, "type", None)

          if event_type == "ping":
            yield StreamEvent(type="heartbeat")
            continue

          if event_type == "message_start":
            message = getattr(event, "message", None)
            usage = getattr(message, "usage", None) if message is not None else None
            yield StreamEvent(
              type="message_start",
              input_tokens=getattr(usage, "input_tokens", 0) if usage is not None else 0,
              output_tokens=getattr(usage, "output_tokens", 0) if usage is not None else 0,
              cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) if usage is not None else 0,
              cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) if usage is not None else 0,
            )
            continue

          if event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
              current_block_type = "tool_use"
              current_tool_id = getattr(block, "id", None)
              current_tool_name = getattr(block, "name", None)
              current_tool_json = ""
              current_tool_block = block
              caller = _to_plain_dict(getattr(block, "caller", None))
              yield StreamEvent(
                type="tool_use_start",
                tool_id=str(current_tool_id or ""),
                tool_name=str(current_tool_name or ""),
                raw_block=_to_plain_dict(block),
                caller=caller if isinstance(caller, dict) else None,
              )
            elif block_type == "thinking":
              current_block_type = "thinking"
              current_thinking = ""
              current_signature = ""
              yield StreamEvent(type="stream_progress")
            elif block_type == "text":
              current_block_type = "text"
              current_text = ""
            elif block_type == "compaction":
              current_block_type = "compaction"
              current_compaction = None
            continue

          if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
              text = getattr(delta, "text", "")
              if text:
                current_text += text
                yield StreamEvent(type="text_delta", text=text)
            elif delta_type == "input_json_delta":
              partial = getattr(delta, "partial_json", "")
              if partial:
                current_tool_json += partial
                yield StreamEvent(type="tool_use_delta", tool_input_json=partial)
            elif delta_type == "thinking_delta":
              thinking_text = getattr(delta, "thinking", "")
              if thinking_text:
                current_thinking += thinking_text
                yield StreamEvent(type="thinking_delta", thinking_text=thinking_text)
            elif delta_type == "signature_delta":
              signature = getattr(delta, "signature", "")
              if signature:
                current_signature += signature
                yield StreamEvent(type="stream_progress")
            elif delta_type == "compaction_delta":
              current_compaction = getattr(delta, "content", None)
              yield StreamEvent(type="stream_progress")
            continue

          if event_type == "content_block_stop":
            if current_block_type == "text":
              yield StreamEvent(
                type="text_end",
                text=current_text,
                raw_block={"type": "text", "text": current_text},
              )
              current_text = ""
              current_block_type = None
            elif current_block_type == "thinking":
              block = {
                "type": "thinking",
                "thinking": current_thinking,
                "signature": current_signature,
              }
              yield StreamEvent(
                type="thinking_end",
                thinking_text=current_thinking,
                signature=current_signature,
                raw_block=block,
              )
              current_thinking = ""
              current_signature = ""
              current_block_type = None
            elif current_block_type == "tool_use" and current_tool_id is not None:
              try:
                tool_input = json.loads(current_tool_json) if current_tool_json else {}
              except json.JSONDecodeError:
                tool_input = {}
              block = _to_plain_dict(current_tool_block)
              if not isinstance(block, dict):
                block = {"type": "tool_use", "id": current_tool_id, "name": current_tool_name}
              try:
                from agent.shared.tool_redaction import get_audit_hmac_secret, redact_tool_input

                block["input"] = redact_tool_input(
                  str(current_tool_name or ""),
                  tool_input,
                  deployment_secret=get_audit_hmac_secret(),
                )
              except Exception:
                block["input"] = tool_input
              yield StreamEvent(
                type="tool_use_end",
                tool_id=str(current_tool_id),
                tool_name=str(current_tool_name or ""),
                tool_input_json=current_tool_json,
                tool_input=tool_input,
                raw_block=block,
                caller=_to_plain_dict(getattr(current_tool_block, "caller", None)),
              )
              current_tool_id = None
              current_tool_name = None
              current_tool_json = ""
              current_tool_block = None
              current_block_type = None
            elif current_block_type == "compaction":
              yield StreamEvent(
                type="compaction",
                text=current_compaction,
                raw_block={"type": "compaction", "content": current_compaction},
              )
              current_compaction = None
              current_block_type = None
            continue

          if event_type == "message_delta":
            delta = getattr(event, "delta", None)
            stop_reason = str(getattr(delta, "stop_reason", "") or "")
            usage = getattr(event, "usage", None)
            if usage is not None:
              yield StreamEvent(
                type="usage_update",
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
              )
    except Exception as exc:
      if self.is_retryable_error(exc):
        raise
      detail = _format_anthropic_rejection_detail(exc)
      if detail is None:
        raise
      context = _stream_request_context(
        stream_params,
        auth_mode=auth_mode,
        use_compaction=use_compaction,
        betas=betas,
      )
      log.warning("Anthropic request rejected during stream: %s; %s", detail, context)
      raise RuntimeError(f"Anthropic request rejected (stage=stream): {detail}; {context}") from None

    yield StreamEvent(type="message_end", stop_reason=stop_reason)

  def is_retryable_error(self, exc: Exception) -> bool:
    try:
      import httpx
      from anthropic import APIConnectionError, APIStatusError
    except ImportError:
      return False

    if isinstance(exc, APIStatusError):
      status_code = getattr(exc, "status_code", None)
      if status_code is None:
        response = getattr(exc, "response", None)
        if response is not None:
          status_code = getattr(response, "status_code", None)
      return status_code in {200, 429} or (isinstance(status_code, int) and 500 <= status_code < 600)
    if isinstance(exc, APIConnectionError):
      return True
    if isinstance(exc, (httpx.TransportError, httpx.StreamError)):
      return True
    return False

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    return super().estimate_cost(
      model,
      input_tokens,
      output_tokens,
      cache_read_tokens=cache_read_tokens,
      cache_creation_tokens=cache_creation_tokens,
    )
