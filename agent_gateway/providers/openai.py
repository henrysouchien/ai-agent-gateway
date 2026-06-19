from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import replace
from typing import Any, AsyncIterator, Dict

from .base import CostEstimate, ModelInfo, ModelProvider, StreamEvent, ThinkingLevel, truncate_to_last_compaction
from .openai_helpers import (
  _MAX_TOOL_ID_LEN as _MAX_TOOL_ID_LEN,
  _MODEL_INFO_BY_TAG as _MODEL_INFO_BY_TAG,
  _TOOL_ID_RE as _TOOL_ID_RE,
  _base_compat as _base_compat,
  _contains_tool_history as _contains_tool_history,
  _detect_compat as _detect_compat,
  _field as _field,
  _image_part as _image_part,
  _is_tool_result_message as _is_tool_result_message,
  _map_reasoning_effort as _map_reasoning_effort,
  _model_matches_tag as _model_matches_tag,
  _normalize_tool_call_id as _normalize_tool_call_id,
  _same_model_message as _same_model_message,
  _stringify_tool_result_content as _stringify_tool_result_content,
  _synthetic_tool_result as _synthetic_tool_result,
  _system_prompt_text as _system_prompt_text,
  _to_plain_dict as _to_plain_dict,
)


log = logging.getLogger("agent_gateway.providers.openai")


class OpenAIProvider(ModelProvider):
  """`ModelProvider` implementation for OpenAI-compatible responses APIs.

  The adapter normalizes OpenAI models into the same runner contract used by the
  Anthropic provider and supports `OPENAI_BASE_URL`-style compatibility targets
  through the `base_url` auth config field.
  """

  name = "openai"

  def __init__(self) -> None:
    self._last_base_url: str | None = None
    self._last_compat_override: dict[str, Any] | None = None

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    if str(config.get("auth_mode", "api")).strip().lower() == "oauth":
      return bool(str(config.get("auth_token", "")).strip())
    return bool(config.get("api_key") or os.environ.get("OPENAI_API_KEY"))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    try:
      import httpx
      from openai import AsyncOpenAI
    except ImportError as exc:
      raise RuntimeError("openai dependency is required to use OpenAIProvider") from exc

    mode = str(config.get("auth_mode", "api")).strip().lower()
    base_url = config.get("base_url") or config.get("baseURL") or config.get("api_base_url") or config.get("api_base")
    self._last_base_url = str(base_url) if base_url else None
    compat_override = config.get("compat")
    self._last_compat_override = dict(compat_override) if isinstance(compat_override, dict) else None

    client_kwargs: Dict[str, Any] = {}
    if base_url:
      client_kwargs["base_url"] = str(base_url)
    if timeout is not None:
      client_kwargs["timeout"] = httpx.Timeout(timeout=timeout, connect=5.0)

    if mode == "oauth":
      client_kwargs["api_key"] = str(config.get("auth_token", ""))
      return AsyncOpenAI(**client_kwargs)

    api_key = str(config.get("api_key") or os.environ.get("OPENAI_API_KEY") or "")
    client_kwargs["api_key"] = api_key
    return AsyncOpenAI(**client_kwargs)

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
    for tags, info in _MODEL_INFO_BY_TAG:
      if any(_model_matches_tag(model_id, tag) for tag in tags):
        return replace(info, id=model_id)
    if "gpt-5" in model_id:
      return ModelInfo(
        id=model_id,
        provider=self.name,
        context_window=400_000,
        max_output_tokens=128_000,
        supports_thinking=True,
        supports_vision=True,
        compat=_base_compat(supports_reasoning_effort=True),
      )
    raise ValueError(f"OpenAIProvider does not recognize model: {model_id}")

  def _resolved_compat(
    self,
    model_info: ModelInfo,
    *,
    base_url: str | None = None,
    compat_override: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    resolved = dict(_base_compat(supports_reasoning_effort=model_info.supports_thinking))
    if isinstance(model_info.compat, dict):
      resolved.update(model_info.compat)

    detected = _detect_compat(base_url if base_url is not None else self._last_base_url)
    resolved["supportsStore"] = bool(resolved.get("supportsStore", True) and detected.get("supportsStore", True))
    resolved["supportsDeveloperRole"] = bool(
      resolved.get("supportsDeveloperRole", True) and detected.get("supportsDeveloperRole", True)
    )
    resolved["supportsReasoningEffort"] = bool(
      resolved.get("supportsReasoningEffort", False) and detected.get("supportsReasoningEffort", True)
    )
    resolved["supportsUsageInStreaming"] = bool(
      resolved.get("supportsUsageInStreaming", True) and detected.get("supportsUsageInStreaming", True)
    )
    resolved["supportsStrictMode"] = bool(
      resolved.get("supportsStrictMode", True) and detected.get("supportsStrictMode", True)
    )
    resolved["requiresToolResultName"] = bool(
      resolved.get("requiresToolResultName", False) or detected.get("requiresToolResultName", False)
    )
    resolved["requiresAssistantAfterToolResult"] = bool(
      resolved.get("requiresAssistantAfterToolResult", False)
      or detected.get("requiresAssistantAfterToolResult", False)
    )
    resolved["requiresThinkingAsText"] = bool(
      resolved.get("requiresThinkingAsText", False) or detected.get("requiresThinkingAsText", False)
    )
    resolved["maxTokensField"] = detected.get("maxTokensField", resolved.get("maxTokensField", "max_completion_tokens"))
    resolved["thinkingFormat"] = detected.get("thinkingFormat", resolved.get("thinkingFormat", "openai"))
    resolved["reasoningEffortMap"] = {
      **dict(detected.get("reasoningEffortMap") or {}),
      **dict(resolved.get("reasoningEffortMap") or {}),
    }

    if isinstance(self._last_compat_override, dict):
      resolved.update(self._last_compat_override)
    if isinstance(compat_override, dict):
      resolved.update(compat_override)
    return resolved

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
    model_info = self.get_model_info(model)
    compat = self._resolved_compat(
      model_info,
      base_url=kwargs.get("base_url"),
      compat_override=kwargs.get("compat"),
    )
    normalized_messages = self.normalize_messages(messages, model_info)

    openai_messages: list[dict[str, Any]] = []
    system_text = _system_prompt_text(system_prompt).strip()
    if system_text:
      system_role = "developer" if model_info.supports_thinking and compat.get("supportsDeveloperRole", True) else "system"
      openai_messages.append({"role": system_role, "content": system_text})

    for message in normalized_messages:
      role = message.get("role")
      if role == "assistant":
        content = message.get("content")
        if not isinstance(content, list):
          text = str(content or "")
          if text:
            openai_messages.append({"role": "assistant", "content": text})
          continue

        assistant_msg: dict[str, Any] = {
          "role": "assistant",
          "content": "" if compat.get("requiresAssistantAfterToolResult") else None,
        }
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        thinking_signature = ""
        tool_calls: list[dict[str, Any]] = []
        reasoning_details: list[dict[str, Any]] = []

        for block in content:
          if not isinstance(block, dict):
            continue
          block_type = block.get("type")
          if block_type == "text":
            text = str(block.get("text", ""))
            if text.strip():
              text_parts.append(text)
            continue
          if block_type == "thinking":
            thinking_text = str(block.get("thinking", ""))
            if thinking_text.strip():
              thinking_parts.append(thinking_text)
            if not thinking_signature:
              thinking_signature = str(block.get("signature", ""))
            continue
          if block_type not in {"tool_use", "server_tool_use"}:
            continue

          tool_call_id = _normalize_tool_call_id(str(block.get("id", "")))
          tool_call = {
            "id": tool_call_id,
            "type": "function",
            "function": {
              "name": str(block.get("name", "tool")),
              "arguments": json.dumps(block.get("input", {}), default=str),
            },
          }
          tool_calls.append(tool_call)
          thought_signature = block.get("thought_signature") or block.get("thoughtSignature")
          if thought_signature:
            try:
              parsed_signature = json.loads(str(thought_signature))
            except json.JSONDecodeError:
              parsed_signature = None
            if isinstance(parsed_signature, dict):
              reasoning_details.append(parsed_signature)

        assistant_text = "".join(text_parts)
        if thinking_parts:
          if not model_info.supports_thinking or compat.get("requiresThinkingAsText"):
            thinking_text = "\n\n".join(thinking_parts)
            assistant_text = thinking_text if not assistant_text else f"{thinking_text}\n\n{assistant_text}"
          elif thinking_signature:
            assistant_msg[thinking_signature] = "\n".join(thinking_parts)

        if assistant_text:
          assistant_msg["content"] = assistant_text
        if tool_calls:
          assistant_msg["tool_calls"] = tool_calls
        if reasoning_details:
          assistant_msg["reasoning_details"] = reasoning_details

        content_value = assistant_msg.get("content")
        has_content = isinstance(content_value, str) and len(content_value) > 0
        if not has_content and not assistant_msg.get("tool_calls"):
          continue
        openai_messages.append(assistant_msg)
        continue

      if role == "user":
        content = message.get("content")
        if isinstance(content, str):
          if content:
            openai_messages.append({"role": "user", "content": content})
          continue

        if not isinstance(content, list):
          text = str(content or "")
          if text:
            openai_messages.append({"role": "user", "content": text})
          continue

        user_parts: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []
        for block in content:
          if not isinstance(block, dict):
            continue
          block_type = block.get("type")
          if block_type == "text":
            text = str(block.get("text", ""))
            if text:
              user_parts.append({"type": "text", "text": text})
            continue
          if block_type == "image" and model_info.supports_vision:
            image_part = _image_part(block)
            if image_part is not None:
              user_parts.append(image_part)
            continue
          if block_type == "tool_result":
            pending_tool_results.append(block)

        if user_parts:
          if all(part.get("type") == "text" for part in user_parts):
            openai_messages.append({"role": "user", "content": "".join(part["text"] for part in user_parts)})
          else:
            openai_messages.append({"role": "user", "content": user_parts})

        for block in pending_tool_results:
          tool_message = {
            "role": "tool",
            "tool_call_id": _normalize_tool_call_id(str(block.get("tool_use_id", ""))),
            "content": _stringify_tool_result_content(block.get("content")),
          }
          if compat.get("requiresToolResultName") and block.get("tool_name"):
            tool_message["name"] = str(block.get("tool_name"))
          openai_messages.append(tool_message)
        continue

      if role == "tool":
        openai_messages.append(
          {
            "role": "tool",
            "tool_call_id": _normalize_tool_call_id(str(message.get("tool_call_id", ""))),
            "content": _stringify_tool_result_content(message.get("content")),
          }
        )

    params: dict[str, Any] = {
      "model": model,
      "messages": openai_messages,
      "stream": True,
    }

    if compat.get("supportsUsageInStreaming", True):
      params["stream_options"] = {"include_usage": True}
    if compat.get("supportsStore", True):
      params["store"] = False

    max_tokens_field = str(compat.get("maxTokensField", "max_completion_tokens"))
    params[max_tokens_field] = max_tokens

    if tools:
      params["tools"] = [
        {
          "type": "function",
          "function": {
            "name": str(tool.get("name", "")),
            "description": str(tool.get("description", "")),
            "parameters": tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}},
            **({"strict": False} if compat.get("supportsStrictMode", True) else {}),
          },
        }
        for tool in tools
      ]
    elif _contains_tool_history(normalized_messages):
      params["tools"] = []

    reasoning_effort = _map_reasoning_effort(thinking_level)
    if model_info.supports_thinking and reasoning_effort and compat.get("supportsReasoningEffort", False):
      params["reasoning_effort"] = compat.get("reasoningEffortMap", {}).get(reasoning_effort, reasoning_effort)
    elif compat.get("thinkingFormat") in {"zai", "qwen"} and model_info.supports_thinking:
      params["enable_thinking"] = thinking_level != ThinkingLevel.NONE

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
          transformed.append(dict(message))
          continue

        next_content: list[dict[str, Any]] = []
        for block in content:
          if not isinstance(block, dict):
            continue

          block_type = block.get("type")
          if block_type == "thinking":
            thinking_text = str(block.get("thinking", ""))
            signature = str(block.get("signature", ""))
            if is_same_model and (thinking_text.strip() or signature):
              next_content.append(dict(block))
            elif thinking_text.strip():
              next_content.append({"type": "text", "text": thinking_text})
            continue

          if block_type in {"tool_use", "server_tool_use"}:
            next_block = dict(block)
            tool_id = str(block.get("id", ""))
            normalized_id = _normalize_tool_call_id(tool_id)
            if tool_id and normalized_id != tool_id:
              tool_id_map[tool_id] = normalized_id
              next_block["id"] = normalized_id
            next_content.append(next_block)
            continue

          if block_type == "text":
            next_content.append({"type": "text", "text": str(block.get("text", ""))})
            continue

          next_content.append(dict(block))

        next_message = dict(message)
        next_message["content"] = next_content
        transformed.append(next_message)
        continue

      if role == "user":
        content = message.get("content")
        if isinstance(content, list):
          next_message = dict(message)
          next_content: list[dict[str, Any]] = []
          for block in content:
            if not isinstance(block, dict):
              continue
            if block.get("type") == "tool_result":
              next_block = dict(block)
              tool_use_id = str(block.get("tool_use_id", ""))
              normalized_id = tool_id_map.get(tool_use_id) or _normalize_tool_call_id(tool_use_id)
              if normalized_id:
                next_block["tool_use_id"] = normalized_id
              next_content.append(next_block)
            else:
              next_content.append(dict(block))
          next_message["content"] = next_content
          transformed.append(next_message)
        else:
          transformed.append(dict(message))
        continue

      transformed.append(dict(message))

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
            if isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use"}
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

    return truncate_to_last_compaction(result, compaction_as_text=True)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    stream = await client.chat.completions.create(**params)
    stop_reason = ""
    current_block_type: str | None = None
    current_text = ""
    current_thinking = ""
    current_signature = ""
    current_tool_id = ""
    current_tool_name = ""
    current_tool_json = ""
    current_tool_index: int | None = None
    current_tool_signature = ""
    message_started = False
    reported_output_tokens = 0

    def _finish_current_block() -> list[StreamEvent]:
      nonlocal current_block_type
      nonlocal current_text
      nonlocal current_thinking
      nonlocal current_signature
      nonlocal current_tool_id
      nonlocal current_tool_name
      nonlocal current_tool_json
      nonlocal current_tool_index
      nonlocal current_tool_signature

      events: list[StreamEvent] = []
      if current_block_type == "text":
        events.append(
          StreamEvent(
            type="text_end",
            text=current_text,
            raw_block={"type": "text", "text": current_text},
          )
        )
        current_text = ""
      elif current_block_type == "thinking":
        block = {
          "type": "thinking",
          "thinking": current_thinking,
          "signature": current_signature,
        }
        events.append(
          StreamEvent(
            type="thinking_end",
            thinking_text=current_thinking,
            signature=current_signature,
            raw_block=block,
          )
        )
        current_thinking = ""
        current_signature = ""
      elif current_block_type == "tool_use":
        try:
          tool_input = json.loads(current_tool_json) if current_tool_json else {}
        except json.JSONDecodeError:
          tool_input = {}
        try:
          from agent.shared.tool_redaction import get_audit_hmac_secret, redact_tool_input

          redacted_tool_input = redact_tool_input(
            str(current_tool_name or ""),
            tool_input,
            deployment_secret=get_audit_hmac_secret(),
          )
        except Exception:
          redacted_tool_input = tool_input
        raw_block = {
          "type": "tool_use",
          "id": current_tool_id,
          "name": current_tool_name,
          "input": redacted_tool_input,
        }
        if current_tool_signature:
          raw_block["thought_signature"] = current_tool_signature
        events.append(
          StreamEvent(
            type="tool_use_end",
            tool_id=current_tool_id,
            tool_name=current_tool_name,
            tool_input_json=current_tool_json,
            tool_input=tool_input,
            raw_block=raw_block,
          )
        )
        current_tool_id = ""
        current_tool_name = ""
        current_tool_json = ""
        current_tool_index = None
        current_tool_signature = ""
      current_block_type = None
      return events

    async for chunk in stream:
      usage = _field(chunk, "usage")
      choices = _field(chunk, "choices") or []
      choice = choices[0] if choices else None
      if usage is None and choice is not None:
        usage = _field(choice, "usage")
      if usage is not None:
        prompt_tokens = int(_field(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(_field(usage, "completion_tokens", 0) or 0)
        prompt_details = _field(usage, "prompt_tokens_details") or {}
        completion_details = _field(usage, "completion_tokens_details") or {}
        cache_read_tokens = int(_field(prompt_details, "cached_tokens", 0) or 0)
        reasoning_tokens = int(_field(completion_details, "reasoning_tokens", 0) or 0)
        output_tokens = completion_tokens + reasoning_tokens
        if not message_started:
          message_started = True
          yield StreamEvent(
            type="message_start",
            input_tokens=prompt_tokens,
            cache_read_tokens=cache_read_tokens,
          )
        output_delta = max(0, output_tokens - reported_output_tokens)
        if output_delta:
          reported_output_tokens = output_tokens
          yield StreamEvent(type="usage_update", output_tokens=output_delta)

      if choice is None:
        continue

      finish_reason = _field(choice, "finish_reason")
      if finish_reason:
        if finish_reason == "stop":
          stop_reason = "end_turn"
        elif finish_reason == "length":
          stop_reason = "max_tokens"
        elif finish_reason in {"tool_calls", "function_call"}:
          stop_reason = "tool_use"
        elif finish_reason == "content_filter":
          stop_reason = "error"
        else:
          stop_reason = str(finish_reason)

      delta = _field(choice, "delta")
      if delta is None:
        continue

      content_delta = _field(delta, "content")
      if isinstance(content_delta, str) and content_delta:
        if current_block_type != "text":
          for event in _finish_current_block():
            yield event
          current_block_type = "text"
          current_text = ""
        current_text += content_delta
        yield StreamEvent(type="text_delta", text=content_delta)

      reasoning_field = ""
      reasoning_delta = ""
      for field_name in ("reasoning_content", "reasoning", "reasoning_text"):
        value = _field(delta, field_name)
        if isinstance(value, str) and value:
          reasoning_field = field_name
          reasoning_delta = value
          break
      if reasoning_delta:
        if current_block_type != "thinking" or (current_signature and current_signature != reasoning_field):
          for event in _finish_current_block():
            yield event
          current_block_type = "thinking"
          current_thinking = ""
          current_signature = reasoning_field
        current_thinking += reasoning_delta
        yield StreamEvent(type="thinking_delta", thinking_text=reasoning_delta)

      tool_calls = _field(delta, "tool_calls") or []
      for tool_call in tool_calls:
        tool_index = _field(tool_call, "index")
        tool_id = _field(tool_call, "id")
        function = _field(tool_call, "function") or {}
        tool_name = _field(function, "name")
        args_delta = _field(function, "arguments") or ""

        is_new_tool = (
          current_block_type != "tool_use"
          or (tool_index is not None and current_tool_index != tool_index)
          or (tool_id and current_tool_id and current_tool_id != tool_id)
        )
        if is_new_tool:
          for event in _finish_current_block():
            yield event
          current_block_type = "tool_use"
          current_tool_index = int(tool_index) if tool_index is not None else None
          current_tool_id = _normalize_tool_call_id(str(tool_id or ""))
          current_tool_name = str(tool_name or "")
          current_tool_json = ""
          current_tool_signature = ""
          yield StreamEvent(
            type="tool_use_start",
            tool_id=current_tool_id,
            tool_name=current_tool_name,
            raw_block={"type": "tool_use", "id": current_tool_id, "name": current_tool_name},
          )

        if tool_id:
          current_tool_id = _normalize_tool_call_id(str(tool_id))
        if tool_name:
          current_tool_name = str(tool_name)
        if args_delta:
          current_tool_json += str(args_delta)
          yield StreamEvent(type="tool_use_delta", tool_input_json=str(args_delta))

      reasoning_details = _field(delta, "reasoning_details") or []
      if current_block_type == "tool_use":
        for detail in reasoning_details:
          detail_type = str(_field(detail, "type", ""))
          detail_id = _normalize_tool_call_id(str(_field(detail, "id", "")))
          detail_data = _field(detail, "data")
          if detail_type == "reasoning.encrypted" and detail_id and detail_data and detail_id == current_tool_id:
            current_tool_signature = json.dumps(_to_plain_dict(detail), default=str)
            break

    for event in _finish_current_block():
      yield event
    yield StreamEvent(type="message_end", stop_reason=stop_reason or "end_turn")

  def is_retryable_error(self, exc: Exception) -> bool:
    try:
      import httpx
    except ImportError:
      httpx = None  # type: ignore[assignment]

    try:
      from openai import APIConnectionError, APIStatusError, RateLimitError
    except ImportError:
      APIConnectionError = APIStatusError = RateLimitError = None  # type: ignore[assignment]

    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
      status_code = getattr(response, "status_code", None)

    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
      return True
    if RateLimitError is not None and isinstance(exc, RateLimitError):
      return True
    if APIStatusError is not None and isinstance(exc, APIStatusError):
      return bool(status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600))
    if httpx is not None and isinstance(exc, (httpx.TransportError, httpx.StreamError)):
      return True
    return bool(status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600))

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
