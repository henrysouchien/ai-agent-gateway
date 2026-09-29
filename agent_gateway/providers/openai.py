from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, AsyncGenerator, Dict
from urllib.parse import urlparse

from model_authority.registry import AdapterRouteSupport
from model_authority.rates import RateTable
from model_authority.thinking import EffortResolution, ThinkingLevel, clamp_effort
from .base import (
  ModelInfo,
  ModelProvider,
  StreamEvent,
  authority_rate_table,
  truncate_to_last_compaction,
)
from .openai_responses_helpers import (
  _ResponsesStreamState,
  _convert_messages,
  convert_openai_response_tools,
  _is_tool_result_message,
  _normalize_tool_call_id,
  _same_model_message,
  _synthetic_tool_result,
  _system_prompt_text,
  map_event,
  reasoning_effort_fragment,
  responses_compat,
  responses_model_info,
)

log = logging.getLogger(__name__)


_BASE_URL_KEYS = ("base_url", "baseURL", "api_base_url", "api_base")
_OFFICIAL_OPENAI_HOST = "api.openai.com"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_OPENAI_CONTEXT_LENGTH_PATTERNS = (
  re.compile(r"\bcontext[_\s-]*length[_\s-]*(?:exceeded|error)\b", re.IGNORECASE),
  re.compile(r"\bcontext[_\s-]+window\b", re.IGNORECASE),
  re.compile(r"\bmaximum\s+context\s+length\b", re.IGNORECASE),
  re.compile(r"\bprompt\s+(?:is\s+)?too\s+long\b", re.IGNORECASE),
  re.compile(r"\btoo\s+many\s+input\s+tokens\b", re.IGNORECASE),
  re.compile(r"\binput[_\s-]+(?:is[_\s-]+)?too[_\s-]+long\b", re.IGNORECASE),
)
_OPENAI_OUTPUT_TOKEN_PARAMETER_PATTERN = re.compile(
  r"\bmax(?:imum)?[_\s-]*(?:output|completion)[_\s-]*tokens?\b"
  r"|\b(?:output|completion)[_\s-]*tokens?\s+(?:limit|maximum|parameter)\b",
  re.IGNORECASE,
)


class OpenAIConfigurationError(ValueError):
  """Invalid configuration for the first-party Responses-only provider."""


def _normalize_official_base_url(value: Any) -> str:
  raw = str(value or "").strip()
  parsed = urlparse(raw)
  if (
    parsed.scheme.lower() != "https"
    or (parsed.hostname or "").lower() != _OFFICIAL_OPENAI_HOST
    or parsed.username is not None
    or parsed.password is not None
    or parsed.port is not None
    or parsed.query
    or parsed.fragment
  ):
    raise OpenAIConfigurationError(
      "OpenAIProvider is Responses-only and accepts only the first-party "
      "https://api.openai.com API base; use a first-class provider for other vendors."
    )
  path = parsed.path.rstrip("/")
  if path not in {"", "/v1"}:
    raise OpenAIConfigurationError(
      "OpenAIProvider base_url must be https://api.openai.com or https://api.openai.com/v1."
    )
  return "https://api.openai.com/v1"


def _normalized_client_config(config: dict[str, Any]) -> dict[str, Any]:
  normalized = dict(config)
  compat = normalized.get("compat")
  if compat not in (None, "", {}, []):
    raise OpenAIConfigurationError(
      "OpenAIProvider compatibility overrides were removed with the Responses-only cutover."
    )
  normalized.pop("compat", None)
  configured_urls = [(key, normalized.get(key)) for key in _BASE_URL_KEYS if normalized.get(key)]
  canonical_urls = {_normalize_official_base_url(value) for _key, value in configured_urls}
  if len(canonical_urls) > 1:
    raise OpenAIConfigurationError("Conflicting OpenAI base URL aliases were provided.")
  for key in _BASE_URL_KEYS:
    normalized.pop(key, None)
  normalized["base_url"] = (
    canonical_urls.pop()
    if canonical_urls
    else _DEFAULT_OPENAI_BASE_URL
  )
  normalized["organization"] = str(normalized.get("organization") or "")
  normalized["project"] = str(normalized.get("project") or "")
  return normalized


class OpenAIProvider(ModelProvider):
  """First-party OpenAI provider using the Responses API exclusively."""

  name = "openai"

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport:
    # This implementation speaks ONLY the Responses API (`responses.create`
    # is required at client creation and streaming) against the public
    # api.openai.com base.  It does not implement Chat Completions; the
    # `openai.sdk.chat_completions` adapter is a Risk-local implementation in
    # the Risk serving process and must never be declared here.
    return AdapterRouteSupport(
      adapter="openai.responses",
      provider="openai",
      protocol_profiles=frozenset({"responses.reasoning"}),
      routes=frozenset({"openai.public"}),
    )

  def __init__(self, *, rate_table: RateTable | None = None) -> None:
    self._rate_table = authority_rate_table(self.name, rate_table)

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    if str(config.get("auth_mode", "api")).strip().lower() == "oauth":
      return bool(str(config.get("auth_token", "")).strip())
    return bool(str(config.get("api_key", "")).strip())

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    normalized = _normalized_client_config(config)
    mode = str(normalized.get("auth_mode", "api")).strip().lower()
    credential = str(
      normalized.get("auth_token" if mode == "oauth" else "api_key", "")
    ).strip()
    if not credential:
      raise RuntimeError(f"No OpenAI {mode} credential configured")

    import httpx2
    from openai import AsyncOpenAI, Omit

    client_kwargs: Dict[str, Any] = {
      "base_url": normalized["base_url"],
      "organization": normalized["organization"],
      "project": normalized["project"],
      # openai>=3.14 parses `OPENAI_CUSTOM_HEADERS` from the process
      # environment into the client's default headers, which are applied after
      # the credential, organization and project passed here.  Restating the
      # bound principal as explicit headers keeps that ambient input from
      # redirecting a request to another account, organization or project;
      # an unbound organization or project stays absent rather than inheritable.
      "default_headers": {
        "Authorization": f"Bearer {credential}",
        "OpenAI-Organization": normalized["organization"] or Omit(),
        "OpenAI-Project": normalized["project"] or Omit(),
      },
    }
    if timeout is not None:
      client_kwargs["timeout"] = httpx2.Timeout(timeout=timeout, connect=5.0)
    client_kwargs["api_key"] = credential
    return AsyncOpenAI(**client_kwargs)

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    if client is None:
      return
    try:
      await asyncio.wait_for(client.close(), timeout=timeout)
    except Exception:
      pass

  def get_model_info(self, model: str) -> ModelInfo:
    return responses_model_info(self.adapter_route_support(), model, self._rate_table)

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
    compat = responses_compat(model_info)
    if not model_info.supports_thinking or not compat.reasoning_control.values:
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})
    supported = tuple(ThinkingLevel(value) for value in compat.reasoning_control.values)
    normalized = requested
    if requested == ThinkingLevel.MINIMAL and ThinkingLevel.MINIMAL not in supported:
      normalized = ThinkingLevel.LOW
    effective = clamp_effort(normalized, supported)
    return EffortResolution(
      requested=requested,
      effective=effective,
      thinking_enabled_effective=effective != ThinkingLevel.NONE,
      payload_fragments=reasoning_effort_fragment(compat, effective.value),
    )

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    tool_id_map: dict[str, str] = {}
    transformed: list[dict[str, Any]] = []
    for message in messages:
      role = message.get("role")
      if role == "assistant":
        if str(message.get("stop_reason") or "") in {"error", "aborted"}:
          continue
        content = message.get("content")
        if not isinstance(content, list):
          transformed.append(dict(message))
          continue
        same_model = _same_model_message(message, model_info)
        next_content: list[dict[str, Any]] = []
        for block in content:
          if not isinstance(block, dict):
            continue
          block_type = block.get("type")
          if block_type == "thinking":
            thinking = str(block.get("thinking") or "")
            signature = str(block.get("signature") or block.get("thinkingSignature") or "")
            if same_model and signature:
              next_content.append(dict(block))
            elif thinking.strip():
              next_content.append({"type": "text", "text": thinking})
          elif block_type in {"tool_use", "server_tool_use"}:
            next_block = dict(block)
            original = str(block.get("id") or "")
            normalized = _normalize_tool_call_id(original)
            if original and normalized != original:
              tool_id_map[original] = normalized
              next_block["id"] = normalized
            next_content.append(next_block)
          elif block_type == "text":
            next_content.append(dict(block))
          elif block_type == "compaction":
            next_content.append(dict(block))
        next_message = dict(message)
        next_message["content"] = next_content
        transformed.append(next_message)
        continue
      if role == "user" and isinstance(message.get("content"), list):
        next_message = dict(message)
        next_content = []
        for block in message["content"]:
          if not isinstance(block, dict):
            continue
          next_block = dict(block)
          if block.get("type") == "tool_result":
            original = str(block.get("tool_use_id") or "")
            next_block["tool_use_id"] = tool_id_map.get(original) or _normalize_tool_call_id(original)
          next_content.append(next_block)
        next_message["content"] = next_content
        transformed.append(next_message)
      else:
        transformed.append(dict(message))

    result: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []
    for message in transformed:
      if pending_calls and not _is_tool_result_message(message):
        result.append({
          "role": "user",
          "content": [_synthetic_tool_result(str(block.get("id") or ""), str(block.get("name") or "")) for block in pending_calls],
        })
        pending_calls = []
      if message.get("role") == "assistant" and isinstance(message.get("content"), list):
        pending_calls = [
          dict(block) for block in message["content"]
          if isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use"}
        ]
      elif pending_calls and _is_tool_result_message(message):
        result_ids = {str(block.get("tool_use_id") or "") for block in message["content"]}
        missing = [block for block in pending_calls if str(block.get("id") or "") not in result_ids]
        if missing:
          result.append({
            "role": "user",
            "content": [_synthetic_tool_result(str(block.get("id") or ""), str(block.get("name") or "")) for block in missing],
          })
        pending_calls = []
      result.append(message)
    if pending_calls:
      result.append({
        "role": "user",
        "content": [_synthetic_tool_result(str(block.get("id") or ""), str(block.get("name") or "")) for block in pending_calls],
      })
    return truncate_to_last_compaction(result, compaction_as_text=True)

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
    compat = responses_compat(model_info)
    if tools and not compat.function_tools:
      raise ValueError(f"OpenAI model {model!r} does not support Responses function tools")
    normalized_messages = self.normalize_messages(messages, model_info)
    params: dict[str, Any] = {
      "model": model,
      "stream": True,
      "store": False,
      "instructions": _system_prompt_text(system_prompt).strip(),
      "input": _convert_messages(normalized_messages, model_info),
      "include": ["reasoning.encrypted_content"],
      "max_output_tokens": max_tokens,
    }
    if tools:
      params["tools"] = convert_openai_response_tools(tools)
      params["tool_choice"] = "auto"
      params["parallel_tool_calls"] = True
    resolution = kwargs.get("effort_resolution")
    if not isinstance(resolution, EffortResolution):
      resolution = self.resolve_effort(
        requested=thinking_level, model=model, model_info=model_info, max_tokens=max_tokens
      )
    reasoning = resolution.payload_fragments.get("reasoning")
    if isinstance(reasoning, dict):
      params["reasoning"] = dict(reasoning)
      if resolution.effective != ThinkingLevel.NONE and compat.reasoning_summary:
        params["reasoning"]["summary"] = "auto"
    return params

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
    state = _ResponsesStreamState()
    async with await client.responses.create(**params) as stream:
      async for event in stream:
        for mapped in map_event(event, state):
          yield mapped
        if state.terminal_error is not None:
          terminal_error = state.terminal_error
          state.terminal_error = None
          raise terminal_error

  def is_retryable_error(self, exc: Exception) -> bool:
    import httpx2
    from openai import APIConnectionError, APIStatusError, RateLimitError
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
      status_code = getattr(response, "status_code", None)
    if isinstance(exc, (APIConnectionError, RateLimitError)):
      return True
    if isinstance(exc, APIStatusError):
      return bool(status_code == 429 or isinstance(status_code, int) and 500 <= status_code < 600)
    if isinstance(exc, (httpx2.TransportError, httpx2.StreamError)):
      return True
    return bool(status_code == 429 or isinstance(status_code, int) and 500 <= status_code < 600)

  def is_context_length_error(self, exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
      error = body if isinstance(body, dict) else {}

    error_codes = {
      str(value).strip().lower()
      for value in (
        getattr(exc, "code", None),
        getattr(exc, "type", None),
        error.get("code"),
        error.get("type"),
      )
      if value
    }
    if "context_length_exceeded" in error_codes:
      return True

    response = getattr(exc, "response", None)
    try:
      response_text = getattr(response, "text", "") if response is not None else ""
    except Exception:
      response_text = ""
    searchable = " ".join(
      str(value)
      for value in (
        exc,
        response_text,
        error.get("message"),
        error.get("param"),
      )
      if value
    )
    if _OPENAI_OUTPUT_TOKEN_PARAMETER_PATTERN.search(searchable):
      return False
    return any(pattern.search(searchable) for pattern in _OPENAI_CONTEXT_LENGTH_PATTERNS)


__all__ = ["OpenAIConfigurationError", "OpenAIProvider"]
