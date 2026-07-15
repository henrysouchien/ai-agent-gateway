from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator
from weakref import WeakKeyDictionary

import httpx

from ..thinking import EffortResolution, ThinkingLevel
from .base import ModelInfo, ModelProvider, StreamEvent
from .codex import CodexProvider
from .xai_helpers import (
  DEFAULT_INSTRUCTIONS,
  DEFAULT_XAI_BASE_URL,
  RETRYABLE_STATUSES,
  _ResponsesStreamState,
  _convert_messages,
  _convert_tools,
  _parse_sse,
  _system_prompt_text,
  build_headers,
  map_event,
  parse_error_response,
  resolve_responses_url,
)
from .xai_oauth import (
  oauth_record_from_config,
  refresh_xai_oauth_token,
  resolve_xai_auth_mode,
  token_needs_refresh,
)


class XAIProvider(ModelProvider):
  """First-class xAI Grok provider using the Responses API."""

  name = "xai"

  def __init__(self) -> None:
    self._client_state: WeakKeyDictionary[httpx.AsyncClient, dict[str, Any]] = WeakKeyDictionary()

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    mode = resolve_xai_auth_mode(config)
    if mode == "oauth":
      record, _settings = oauth_record_from_config(config)
      return bool(record and (record.get("access_token") or record.get("refresh_token")))
    return bool(str(config.get("api_key") or os.environ.get("XAI_API_KEY") or "").strip())

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    mode = resolve_xai_auth_mode(config)
    oauth_record = None
    oauth_settings = None
    if mode == "oauth":
      oauth_record, oauth_settings = oauth_record_from_config(config)
      token = str((oauth_record or {}).get("access_token") or "").strip()
      has_refresh = bool(str((oauth_record or {}).get("refresh_token") or "").strip())
    else:
      token = str(config.get("api_key") or os.environ.get("XAI_API_KEY") or "").strip()
      has_refresh = False
    if not token and not has_refresh:
      raise RuntimeError(f"No xAI {mode} credential configured")
    client_kwargs: dict[str, Any] = {
      "timeout": httpx.Timeout(timeout=timeout or 120.0, connect=10.0),
    }
    if isinstance(config.get("_transport"), httpx.AsyncBaseTransport):
      client_kwargs["transport"] = config["_transport"]
    client = httpx.AsyncClient(**client_kwargs)
    self._client_state[client] = {
      "token": token,
      "endpoint_url": resolve_responses_url(str(config.get("base_url") or os.environ.get("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL)),
      "headers": dict(config.get("headers") or {}) if isinstance(config.get("headers"), dict) else {},
      "auth_mode": mode,
      "oauth_record": oauth_record,
      "oauth_settings": oauth_settings,
      "refresh_lock": asyncio.Lock(),
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
    common = dict(
      id=model_id,
      provider=self.name,
      context_window=1_000_000,
      max_output_tokens=128_000,
      supports_vision=False,
      supports_tool_use=True,
    )
    if model_id == "grok-4.3" or model_id.startswith("grok-4.3-") or model_id == "grok-latest":
      return ModelInfo(
        **common,
        supports_thinking=True,
        compat={
          "supportsReasoningEffort": True,
          "reasoningEffortValues": ("none", "low", "medium", "high"),
          "reasoningEffortDefault": "medium",
        },
      )
    if model_id == "grok-4.5" or model_id.startswith("grok-4.5-"):
      return ModelInfo(
        **common,
        supports_thinking=True,
        compat={
          "supportsReasoningEffort": True,
          "reasoningEffortValues": ("low", "medium", "high"),
          "reasoningEffortDefault": "medium",
        },
      )
    if model_id in {
      "grok-build-0.1",
      "grok-4.20-beta-latest-reasoning",
      "grok-4.20-beta-latest-non-reasoning",
    }:
      return ModelInfo(
        **common,
        supports_thinking=model_id.endswith("-reasoning"),
        compat={"supportsReasoningEffort": False},
      )
    # Allowlist enforcement happens above the provider. Unknown explicitly
    # allowlisted Grok models use the spec's conservative reasoning dial.
    return ModelInfo(
      **common,
      supports_thinking=True,
      compat={
        "supportsReasoningEffort": True,
        "reasoningEffortValues": ("low", "medium", "high"),
        "reasoningEffortDefault": "medium",
      },
    )

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
    if not compat.get("supportsReasoningEffort"):
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})
    supported = tuple(ThinkingLevel(str(value)) for value in compat.get("reasoningEffortValues", ()))
    normalized = ThinkingLevel.LOW if requested in {ThinkingLevel.MINIMAL, ThinkingLevel.NONE} and ThinkingLevel.NONE not in supported else requested
    if normalized in {ThinkingLevel.XHIGH, ThinkingLevel.MAX}:
      normalized = ThinkingLevel.HIGH
    effective = normalized if normalized in supported else ThinkingLevel.LOW
    return EffortResolution(
      requested=requested,
      effective=effective,
      thinking_enabled_effective=effective != ThinkingLevel.NONE,
      payload_fragments={"reasoning": {"effort": effective.value}},
    )

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    # The Responses transcript contract is provider-neutral. Composition keeps
    # xAI independent from Codex/OpenAI inheritance and auth routing.
    return CodexProvider().normalize_messages(messages, model_info)

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
      "instructions": _system_prompt_text(system_prompt).strip() or DEFAULT_INSTRUCTIONS,
      "include": ["reasoning.encrypted_content"],
      "max_output_tokens": max_tokens,
    }
    if tools:
      params["tools"] = _convert_tools(tools, strict=None)
      params["tool_choice"] = "auto"
      params["parallel_tool_calls"] = True
    effort_resolution = kwargs.get("effort_resolution")
    if not isinstance(effort_resolution, EffortResolution):
      effort_resolution = self.resolve_effort(
        requested=thinking_level,
        model=model,
        model_info=model_info,
        max_tokens=max_tokens,
      )
    reasoning = effort_resolution.payload_fragments.get("reasoning")
    if isinstance(reasoning, dict):
      params["reasoning"] = dict(reasoning)
    headers = kwargs.get("headers")
    if isinstance(headers, dict) and headers:
      params["_headers"] = dict(headers)
    if kwargs.get("base_url"):
      params["_endpoint_url"] = resolve_responses_url(str(kwargs["base_url"]))
    return params

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    if not isinstance(client, httpx.AsyncClient):
      raise RuntimeError("XAIProvider requires an httpx.AsyncClient")
    state = self._client_state.get(client)
    if state is None:
      raise RuntimeError("xAI client state is missing")
    if state.get("auth_mode") == "oauth":
      record = state.get("oauth_record")
      if isinstance(record, dict) and (not state.get("token") or token_needs_refresh(record)):
        await self._refresh_oauth_state(state, client)

    request_params = dict(params)
    additional_headers = request_params.pop("_headers", None)
    endpoint_url = str(request_params.pop("_endpoint_url", state["endpoint_url"]))
    merged_headers = (
      {**state["headers"], **(additional_headers or {})}
      if isinstance(additional_headers, dict)
      else state["headers"]
    )
    for auth_attempt in range(2):
      headers = build_headers(str(state["token"]), merged_headers)
      refresh_after_response = False
      async with client.stream("POST", endpoint_url, json=request_params, headers=headers) as response:
        if response.status_code == 401 and state.get("auth_mode") == "oauth" and auth_attempt == 0:
          await response.aread()
          refresh_after_response = True
        elif response.status_code != 200:
          message = await parse_error_response(response)
          if response.status_code == 403 and state.get("auth_mode") == "oauth":
            message += (
              " xAI may not have enabled subscription OAuth API access for this account; "
              "use XAI_AUTH_MODE=api with XAI_API_KEY as a fallback."
            )
          raise httpx.HTTPStatusError(message, request=response.request, response=response)
        else:
          stream_state = _ResponsesStreamState()
          buffer = ""
          async for chunk in response.aiter_text():
            buffer += chunk
            events, buffer = _parse_sse(buffer)
            for event in events:
              for mapped in map_event(event, stream_state):
                yield mapped
          return
      if refresh_after_response:
        await self._refresh_oauth_state(state, client)

  async def _refresh_oauth_state(self, state: dict[str, Any], client: httpx.AsyncClient) -> None:
    lock = state.get("refresh_lock")
    if lock is None:
      lock = asyncio.Lock()
      state["refresh_lock"] = lock
    async with lock:
      record = state.get("oauth_record")
      settings = state.get("oauth_settings")
      if not isinstance(record, dict) or settings is None:
        raise RuntimeError("xAI OAuth refresh state is missing; run `agent auth login xai`")
      refreshed = await refresh_xai_oauth_token(record, settings=settings, client=client)
      state["oauth_record"] = refreshed
      state["token"] = str(refreshed["access_token"])

  def is_retryable_error(self, exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
      return exc.response.status_code in RETRYABLE_STATUSES
    return isinstance(exc, (httpx.TransportError, httpx.StreamError))


__all__ = ["XAIProvider"]
