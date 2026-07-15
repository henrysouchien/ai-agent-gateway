from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, AsyncIterator
from weakref import WeakKeyDictionary

import httpx

from .base import ModelInfo, ModelProvider, StreamEvent, ThinkingLevel, truncate_to_last_compaction
from ..thinking import EffortResolution, clamp_effort
from .codex_helpers import (
  DEFAULT_CODEX_BASE_URL as DEFAULT_CODEX_BASE_URL,
  JWT_CLAIM_PATH as JWT_CLAIM_PATH,
  DEFAULT_INSTRUCTIONS as DEFAULT_INSTRUCTIONS,
  _BETA_HEADER as _BETA_HEADER,
  _CODEX_ORIGINATOR as _CODEX_ORIGINATOR,
  _CODEX_USER_AGENT as _CODEX_USER_AGENT,
  _RETRYABLE_STATUSES as _RETRYABLE_STATUSES,
  _RETRYABLE_RE as _RETRYABLE_RE,
  _CODEX_RESPONSE_STATUSES as _CODEX_RESPONSE_STATUSES,
  _TOOL_ID_RE as _TOOL_ID_RE,
  _SURROGATE_RE as _SURROGATE_RE,
  _MODEL_INFO_BY_TAG as _MODEL_INFO_BY_TAG,
  _ResponsesStreamState as _ResponsesStreamState,
  _config_base_url as _config_base_url,
  _model_matches_tag as _model_matches_tag,
  _credential_token as _credential_token,
  _json_dumps_compact as _json_dumps_compact,
  _encode_text_signature_v1 as _encode_text_signature_v1,
  _parse_text_signature as _parse_text_signature,
  _short_hash as _short_hash,
  _sanitize_surrogates as _sanitize_surrogates,
  _system_prompt_text as _system_prompt_text,
  _resolve_codex_url as _resolve_codex_url,
  _normalize_codex_status as _normalize_codex_status,
  _map_reasoning_effort as _map_reasoning_effort,
  _clamp_reasoning_effort as _clamp_reasoning_effort,
  _same_model_message as _same_model_message,
  _synthetic_tool_result as _synthetic_tool_result,
  _normalize_responses_tool_call_id as _normalize_responses_tool_call_id,
  _assistant_text_block as _assistant_text_block,
  _tool_result_output as _tool_result_output,
  _parse_streaming_json as _parse_streaming_json,
  _extract_account_id as _extract_account_id,
  _build_headers as _build_headers,
  _convert_tools as _convert_tools,
  _convert_messages as _convert_messages,
  _parse_sse as _parse_sse,
  _map_stop_reason as _map_stop_reason,
  _map_event as _map_event,
  _parse_error_response as _parse_error_response,
)


class CodexProvider(ModelProvider):
  """`ModelProvider` implementation for the ChatGPT Codex responses backend."""

  name = "codex"

  def __init__(self) -> None:
    self._client_state: WeakKeyDictionary[httpx.AsyncClient, dict[str, Any]] = WeakKeyDictionary()

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(_credential_token(config))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    token = _credential_token(config)
    if not token:
      raise RuntimeError("No Codex token configured")

    account_id = _extract_account_id(token)
    base_url = _config_base_url(config)
    endpoint_url = _resolve_codex_url(base_url)

    client_kwargs: dict[str, Any] = {}
    effective_timeout = timeout if timeout is not None else 120.0
    client_kwargs["timeout"] = httpx.Timeout(timeout=effective_timeout, connect=10.0)
    client = httpx.AsyncClient(**client_kwargs)
    self._client_state[client] = {
      "token": token,
      "account_id": account_id,
      "endpoint_url": endpoint_url,
      "init_headers": dict(config.get("headers") or {}) if isinstance(config.get("headers"), dict) else {},
    }
    return client

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    if client is None or not hasattr(client, "aclose"):
      return
    self._client_state.pop(client, None)
    try:
      await asyncio.wait_for(client.aclose(), timeout=timeout)
    except Exception:
      pass

  def get_model_info(self, model: str) -> ModelInfo:
    model_id = str(model or "").strip()
    if not model_id:
      raise ValueError("Model is required")
    for tags, info in sorted(_MODEL_INFO_BY_TAG, key=lambda row: max(map(len, row[0])), reverse=True):
      if any(_model_matches_tag(model_id, tag) for tag in tags):
        return replace(info, id=model_id)
    if "gpt-5" in model_id:
      return ModelInfo(
        id=model_id,
        provider=self.name,
        context_window=272_000,
        max_output_tokens=128_000,
        supports_thinking=True,
        supports_vision=True,
        compat={
          "supportsReasoningEffort": True,
          "reasoningEffortValues": ("low", "medium", "high"),
          "reasoningEffortDefault": "medium",
          "omitEqualsNone": False,
        },
      )
    return ModelInfo(id=model_id, provider=self.name)

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
    normalized_messages = self.normalize_messages(messages, model_info)

    params: dict[str, Any] = {
      "model": model,
      "store": False,
      "stream": True,
      "input": _convert_messages(normalized_messages, model_info),
      "text": {"verbosity": str(kwargs.get("text_verbosity") or "medium")},
      "include": ["reasoning.encrypted_content"],
      "tool_choice": "auto",
      "parallel_tool_calls": True,
    }

    system_text = _system_prompt_text(system_prompt).strip()
    params["instructions"] = _sanitize_surrogates(system_text or DEFAULT_INSTRUCTIONS)

    session_id = str(kwargs.get("session_id") or kwargs.get("_session_id") or "")
    if session_id:
      params["prompt_cache_key"] = session_id
      params["_session_id"] = session_id

    headers = kwargs.get("headers")
    if isinstance(headers, dict) and headers:
      params["_headers"] = dict(headers)

    base_url = kwargs.get("base_url")
    if base_url:
      params["_endpoint_url"] = _resolve_codex_url(str(base_url))

    if tools:
      params["tools"] = _convert_tools(tools, strict=None)

    effort_resolution = kwargs.get("effort_resolution")
    if not isinstance(effort_resolution, EffortResolution):
      effort_resolution = self.resolve_effort(
        requested=thinking_level,
        model=model,
        model_info=model_info,
        max_tokens=max_tokens,
      )
    reasoning_fragment = effort_resolution.payload_fragments.get("reasoning")
    if isinstance(reasoning_fragment, dict):
      params["reasoning"] = {
        "summary": str(kwargs.get("reasoning_summary") or "auto"),
        **reasoning_fragment,
      }

    return params

  def resolve_effort(
    self,
    *,
    requested: ThinkingLevel,
    model: str,
    model_info: ModelInfo,
    max_tokens: int,
    **request_context: Any,
  ) -> EffortResolution:
    del model, max_tokens, request_context
    if not model_info.supports_thinking:
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})
    compat = dict(model_info.compat or {})
    raw_values = tuple(compat.get("reasoningEffortValues") or ("low", "medium", "high"))
    supported = tuple(ThinkingLevel(str(value)) for value in raw_values)
    normalized = ThinkingLevel.LOW if requested == ThinkingLevel.MINIMAL and ThinkingLevel.MINIMAL not in supported else requested
    effective = clamp_effort(normalized, supported)
    return EffortResolution(
      requested,
      effective,
      effective != ThinkingLevel.NONE,
      {"reasoning": {"effort": effective.value}},
    )

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    tool_call_id_map: dict[str, str] = {}
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
            if block.get("redacted"):
              if is_same_model:
                next_content.append(dict(block))
              continue
            thinking_text = str(block.get("thinking", ""))
            thinking_signature = str(block.get("thinkingSignature") or block.get("signature") or "")
            if is_same_model and thinking_signature:
              next_block = dict(block)
              next_block["thinkingSignature"] = thinking_signature
              next_content.append(next_block)
              continue
            if not thinking_text.strip():
              continue
            if is_same_model:
              next_content.append(dict(block))
            else:
              next_content.append({"type": "text", "text": thinking_text})
            continue
          if block_type == "text":
            if is_same_model:
              next_content.append(dict(block))
            else:
              next_content.append({"type": "text", "text": str(block.get("text", ""))})
            continue
          if block_type in {"tool_use", "server_tool_use"}:
            next_block = dict(block)
            if not is_same_model:
              next_block.pop("thought_signature", None)
              next_block.pop("thoughtSignature", None)
              original_id = str(block.get("id") or "")
              normalized_id = _normalize_responses_tool_call_id(original_id)
              if normalized_id and normalized_id != original_id:
                tool_call_id_map[original_id] = normalized_id
                next_block["id"] = normalized_id
            next_content.append(next_block)
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
              tool_use_id = str(block.get("tool_use_id") or "")
              normalized_id = tool_call_id_map.get(tool_use_id)
              if normalized_id:
                next_block["tool_use_id"] = normalized_id
              next_content.append(next_block)
            else:
              next_content.append(dict(block))
          next_message["content"] = next_content
          transformed.append(next_message)
          continue

      transformed.append(dict(message))

    result: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    existing_tool_result_ids: set[str] = set()

    for message in transformed:
      role = message.get("role")
      if role == "assistant":
        if pending_tool_calls:
          for tool_call in pending_tool_calls:
            tool_id = str(tool_call.get("id") or "")
            if tool_id and tool_id not in existing_tool_result_ids:
              result.append(
                {
                  "role": "user",
                  "content": [_synthetic_tool_result(tool_id, str(tool_call.get("name") or ""))],
                }
              )
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

      if role == "user":
        content = message.get("content")
        tool_result_blocks = [
          block
          for block in content
          if isinstance(block, dict) and block.get("type") == "tool_result"
        ] if isinstance(content, list) else []
        if pending_tool_calls and tool_result_blocks:
          existing_tool_result_ids = {str(block.get("tool_use_id") or "") for block in tool_result_blocks}
          missing = [
            _synthetic_tool_result(str(tool_call.get("id") or ""), str(tool_call.get("name") or ""))
            for tool_call in pending_tool_calls
            if str(tool_call.get("id") or "") not in existing_tool_result_ids
          ]
          if missing:
            next_message = dict(message)
            next_message["content"] = [*list(content or []), *missing]
            result.append(next_message)
          else:
            result.append(message)
          pending_tool_calls = []
          existing_tool_result_ids = set()
          continue
        if pending_tool_calls:
          for tool_call in pending_tool_calls:
            tool_id = str(tool_call.get("id") or "")
            if tool_id and tool_id not in existing_tool_result_ids:
              result.append(
                {
                  "role": "user",
                  "content": [_synthetic_tool_result(tool_id, str(tool_call.get("name") or ""))],
                }
              )
          pending_tool_calls = []
          existing_tool_result_ids = set()
        result.append(message)
        continue

      result.append(message)

    return truncate_to_last_compaction(result, compaction_as_text=True)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    if not isinstance(client, httpx.AsyncClient):
      raise RuntimeError("CodexProvider requires an httpx.AsyncClient")

    state = self._client_state.get(client)
    if state is None:
      raise RuntimeError("Codex client state is missing")

    request_params = dict(params)
    session_id = str(request_params.pop("_session_id", "") or "")
    additional_headers = request_params.pop("_headers", None)
    endpoint_url = str(request_params.pop("_endpoint_url", state["endpoint_url"]))

    headers = _build_headers(
      state.get("init_headers"),
      additional_headers if isinstance(additional_headers, dict) else None,
      str(state["account_id"]),
      str(state["token"]),
      session_id or None,
    )

    async with client.stream("POST", endpoint_url, json=request_params, headers=headers) as response:
      if response.status_code != 200:
        info = await _parse_error_response(response)
        raise httpx.HTTPStatusError(str(info["message"] or "Request failed"), request=response.request, response=response)

      stream_state = _ResponsesStreamState()
      buffer = ""
      async for chunk in response.aiter_text():
        buffer += chunk
        events, buffer = _parse_sse(buffer)
        for event in events:
          for mapped_event in _map_event(event, stream_state):
            yield mapped_event

  def is_retryable_error(self, exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
      return exc.response.status_code in _RETRYABLE_STATUSES
    if isinstance(exc, (httpx.TransportError, httpx.StreamError)):
      return True
    return bool(_RETRYABLE_RE.search(str(exc)))


__all__ = [
  "CodexProvider",
  "_build_headers",
  "_convert_messages",
  "_convert_tools",
  "_extract_account_id",
  "_map_event",
  "_parse_sse",
  "_resolve_codex_url",
]
