from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from typing import Any, AsyncIterator, Dict
from urllib.parse import urlparse

from ..model_registry import AdapterRouteSupport
from ..thinking import EffortResolution, clamp_effort
from .base import (
  CostEstimate,
  ModelInfo,
  ModelProvider,
  StreamEvent,
  ThinkingLevel,
  truncate_to_last_compaction,
)
from .openai_responses_helpers import (
  _MODEL_INFO_BY_TAG,
  _ResponsesStreamState,
  _convert_messages,
  _convert_tools,
  _is_tool_result_message,
  _model_matches_tag,
  _normalize_tool_call_id,
  _same_model_message,
  _synthetic_tool_result,
  _system_prompt_text,
  map_event,
)


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

    try:
      import httpx
      from openai import AsyncOpenAI
    except ImportError as exc:
      raise RuntimeError("openai>=2.31.0 is required to use OpenAIProvider") from exc

    client_kwargs: Dict[str, Any] = {
      "base_url": normalized["base_url"],
      "organization": normalized["organization"],
      "project": normalized["project"],
    }
    if timeout is not None:
      client_kwargs["timeout"] = httpx.Timeout(timeout=timeout, connect=5.0)
    client_kwargs["api_key"] = credential
    client = AsyncOpenAI(**client_kwargs)
    responses = getattr(client, "responses", None)
    if responses is None or not callable(getattr(responses, "create", None)):
      raise RuntimeError("openai>=2.31.0 with AsyncOpenAI.responses.create is required")
    return client

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
    for tags, info in sorted(_MODEL_INFO_BY_TAG, key=lambda row: max(map(len, row[0])), reverse=True):
      if any(_model_matches_tag(model_id, tag) for tag in tags):
        resolved = replace(info, id=model_id)
        if not bool((resolved.compat or {}).get("supportsResponsesStreaming")):
          raise ValueError(f"OpenAI model {model_id!r} has no verified Responses streaming capability")
        return resolved
    raise ValueError(f"OpenAIProvider has no verified Responses capability row for model: {model_id}")

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
    compat = dict(model_info.compat or {})
    if not model_info.supports_thinking or not compat.get("supportsReasoningEffort"):
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})
    supported = tuple(ThinkingLevel(str(value)) for value in compat.get("reasoningEffortValues") or ())
    normalized = requested
    if requested == ThinkingLevel.MINIMAL and ThinkingLevel.MINIMAL not in supported:
      normalized = ThinkingLevel.LOW
    effective = clamp_effort(normalized, supported)
    return EffortResolution(
      requested=requested,
      effective=effective,
      thinking_enabled_effective=effective != ThinkingLevel.NONE,
      payload_fragments={"reasoning": {"effort": effective.value}},
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
    compat = dict(model_info.compat or {})
    if tools and not compat.get("supportsResponsesFunctionTools"):
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
      params["tools"] = _convert_tools(tools)
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
      if resolution.effective != ThinkingLevel.NONE and compat.get("supportsResponsesReasoningSummary"):
        params["reasoning"]["summary"] = "auto"
    return params

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    responses = getattr(client, "responses", None)
    create = getattr(responses, "create", None)
    if not callable(create):
      raise RuntimeError("OpenAI client does not expose responses.create; openai>=2.31.0 is required")
    stream = await create(**params)
    state = _ResponsesStreamState()
    async for event in stream:
      for mapped in map_event(event, state):
        yield mapped
      if state.terminal_error is not None:
        terminal_error = state.terminal_error
        state.terminal_error = None
        raise terminal_error

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
      return bool(status_code == 429 or isinstance(status_code, int) and 500 <= status_code < 600)
    if httpx is not None and isinstance(exc, (httpx.TransportError, httpx.StreamError)):
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


__all__ = ["OpenAIConfigurationError", "OpenAIProvider"]
