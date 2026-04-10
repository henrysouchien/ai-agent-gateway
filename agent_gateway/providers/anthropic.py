from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any, AsyncIterator, Dict, List

from ..rates import RateTable, UnknownModelError, load_rate_table
from .base import CostEstimate, ModelInfo, ModelProvider, StreamEvent, ThinkingLevel


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

_MODEL_INFO_BY_TAG: list[tuple[tuple[str, ...], ModelInfo]] = [
  (
    ("claude-sonnet-4-6",),
    ModelInfo(
      id="claude-sonnet-4-6",
      provider="anthropic",
      supports_thinking=True,
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
      supports_thinking=True,
      input_cost_per_mtok=3.00,
      output_cost_per_mtok=15.00,
      cache_read_cost_per_mtok=0.30,
      cache_write_cost_per_mtok=3.75,
    ),
  ),
]


def _thinking_param(model: str, max_tokens: int) -> dict[str, Any] | None:
  if any(tag in model for tag in ("sonnet-4-6", "opus-4-6")):
    return {"type": "adaptive"}

  if any(tag in model for tag in ("sonnet-4-5", "opus-4-5", "sonnet-4")):
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
    return _thinking_param(model, max_tokens)

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
    if client is None or not hasattr(client, "aclose"):
      return
    try:
      await asyncio.wait_for(client.aclose(), timeout=timeout)
    except Exception:
      pass

  def get_model_info(self, model: str) -> ModelInfo:
    model_id = str(model or "").strip()
    if not model_id:
      raise ValueError("Model is required")
    if not model_id.startswith("claude"):
      raise ValueError(f"AnthropicProvider does not recognize model: {model_id}")
    try:
      rates = self._rate_table.lookup(self.name, model_id)
    except UnknownModelError:
      return ModelInfo(
        id=model_id,
        provider=self.name,
        supports_thinking=_thinking_param(model_id, 4096) is not None,
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_read_cost_per_mtok=0.30,
        cache_write_cost_per_mtok=3.75,
      )
    return ModelInfo(
      id=model_id,
      provider=self.name,
      context_window=rates.context_window or 200_000,
      max_output_tokens=rates.max_tokens or 16_384,
      supports_thinking=_thinking_param(model_id, rates.max_tokens or 4096) is not None,
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
      thinking_param = _thinking_param(model, max_tokens)
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

      if role == "user" and pending_tool_calls and _is_tool_result_message(message):
        content = list(message.get("content") or [])
        existing_tool_result_ids = {
          str(block.get("tool_use_id", ""))
          for block in content
          if isinstance(block, dict) and block.get("type") == "tool_result"
        }
        missing = [
          _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
          for block in pending_tool_calls
          if str(block.get("id", "")) not in existing_tool_result_ids
        ]
        if missing:
          next_message = dict(message)
          next_message["content"] = [*content, *missing]
          result.append(next_message)
        else:
          result.append(message)
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

      result.append(message)

    if pending_tool_calls:
      missing = [
        _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
        for block in pending_tool_calls
        if str(block.get("id", "")) not in existing_tool_result_ids
      ]
      if missing:
        result.append({"role": "user", "content": missing})

    return result

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    stream_params = dict(params)
    auth_mode = str(stream_params.pop("_provider_auth_mode", "api")).strip().lower()
    use_compaction = "context_management" in stream_params
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
          elif delta_type == "compaction_delta":
            current_compaction = getattr(delta, "content", None)
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
      return status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600)
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
