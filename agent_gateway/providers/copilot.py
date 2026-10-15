from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, AsyncGenerator, Mapping
from weakref import WeakKeyDictionary

import httpx

from model_authority.registry import AdapterRouteSupport
from model_authority.rates import RateTable
from model_authority.thinking import EffortResolution, ThinkingLevel

from ..auth import ProviderCredentialFailure
from ..package_info import __version__ as _GATEWAY_VERSION
from .base import (
  ModelInfo,
  ModelProvider,
  StreamEvent,
  _classify_provider_credential_failure,
  _error_code_from_exception,
  _is_context_length_exception,
  _response_text,
  _status_code_from_exception,
  authority_rate_table,
)
from .codex import CodexProvider
from .credential_pool import rotate_provider_credential
from .openai_responses_helpers import (
  _is_tool_result_message,
  reasoning_effort_fragment,
  responses_compat,
  responses_model_info,
)
from .xai_helpers import (
  DEFAULT_INSTRUCTIONS,
  _ResponsesStreamState,
  _convert_messages,
  _convert_tools,
  _parse_sse,
  _system_prompt_text,
  build_headers,
  map_event,
  resolve_responses_url,
)

log = logging.getLogger(__name__)


def _rate_table_for(provider: str, override: RateTable | None) -> RateTable:
  try:
    return authority_rate_table(provider, override)
  except ValueError:
    if override is not None and override.providers.get(provider):
      return override
    log.warning("%s has no authority rate table; cost estimates stay zero", provider)
    return RateTable(
      version="0",
      source="absent",
      providers={provider: {}},
      _path=Path(provider),
    )


# Copilot's proxy also speaks Chat Completions and Anthropic Messages.
# This class does not. One wire: OpenAI Responses at ``{base}/responses``.

DEFAULT_COPILOT_BASE_URL = "https://api.githubcopilot.com"
_RETRYABLE_STATUSES = frozenset({408, 409})
_USAGE_LIMIT_CODES = frozenset({
  "insufficient_quota",
  "quota_exceeded",
  "rate_limit",
  "rate_limit_exceeded",
  "too_many_requests",
  "usage_limit",
})
_USAGE_LIMIT_RE = re.compile(
  r"\b("
  r"rate[\s_-]*limit(?:ed|ing)?"
  r"|too many requests"
  r"|quota[\s_-]*(?:exhausted|exceeded|limit)"
  r"|usage[\s_-]*limit"
  r"|monthly[\s_-]*quota"
  r"|premium[\s_-]*interactions?"
  r")\b",
  re.IGNORECASE,
)
_CONTENT_REFUSAL_RE = re.compile(
  r"\b("
  r"content[\s_-]*(?:filter|policy)"
  r"|content_filter"
  r"|content_policy_violation"
  r"|responsible[\s_-]*ai"
  r"|safety[\s_-]*(?:filter|check|system)"
  r"|moderation"
  r")\b",
  re.IGNORECASE,
)


def _header_token(value: Any) -> str | None:
  text = str(value or "").strip()
  if not text or any(char in text for char in "\r\n"):
    return None
  return text


def _editor_version() -> str | None:
  version = _header_token(_GATEWAY_VERSION)
  if version is None:
    return None
  return f"hank/{version}"


def _bound_base_url(config: dict[str, Any]) -> str:
  configured = str(config.get("base_url") or "").strip()
  return configured or DEFAULT_COPILOT_BASE_URL


def _drop_header(headers: dict[str, Any], name: str) -> None:
  for key in list(headers):
    if str(key).lower() == name.lower():
      headers.pop(key, None)


def _initiator_for(messages: list[dict[str, Any]]) -> str:
  if not messages or not isinstance(messages[-1], dict):
    return CopilotResponsesProvider.INITIATOR_USER
  last = messages[-1]
  if str(last.get("role") or "") == "tool" or _is_tool_result_message(last):
    return CopilotResponsesProvider.INITIATOR_AGENT
  return CopilotResponsesProvider.INITIATOR_USER


def _searchable(exc: Exception) -> str:
  code = _error_code_from_exception(exc) or ""
  return " ".join(part for part in (str(exc), _response_text(exc), code) if part)


def _is_content_refusal(exc: Exception) -> bool:
  return _CONTENT_REFUSAL_RE.search(_searchable(exc)) is not None


def _is_usage_limit_only(exc: Exception) -> bool:
  """True when the body is a rate/usage/quota exhaustion and nothing else."""

  status = _status_code_from_exception(exc)
  if status in {400, 401, 403}:
    return False
  if _is_context_length_exception(exc) or _is_content_refusal(exc):
    return False
  code = str(_error_code_from_exception(exc) or "").strip().lower()
  if code in _USAGE_LIMIT_CODES:
    return True
  return _USAGE_LIMIT_RE.search(_searchable(exc)) is not None


def _retry_after_raw(exc: Exception) -> str | None:
  response = getattr(exc, "response", None)
  headers = getattr(response, "headers", None)
  if headers is None:
    headers = getattr(exc, "headers", None)
  getter = getattr(headers, "get", None)
  if callable(getter):
    raw = getter("retry-after")
    text = str(raw).strip() if raw is not None else ""
    if text:
      return text
  attached = getattr(exc, "retry_after", None)
  if isinstance(attached, str) and attached.strip():
    return attached.strip()
  return None


def _retry_after_reset_at(exc: Exception, *, now: float | None = None) -> float | None:
  """Epoch seconds from ``Retry-After`` (delta-seconds or HTTP-date), else None."""

  raw = _retry_after_raw(exc)
  if raw is None:
    return None
  moment = time.time() if now is None else now
  try:
    seconds = float(raw)
  except ValueError:
    seconds = None
  if seconds is not None:
    if seconds < 0:
      return None
    return moment + seconds
  try:
    parsed = parsedate_to_datetime(raw)
  except (TypeError, ValueError, OverflowError, IndexError):
    return None
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed.timestamp()


def _failure_fields(exc: Exception) -> tuple[str | None, str]:
  message = " ".join(part for part in (str(exc), _response_text(exc)) if part).strip()
  return _error_code_from_exception(exc), message


def _rate_limit_failure(exc: Exception, status: int | None) -> ProviderCredentialFailure:
  error_code, message = _failure_fields(exc)
  return ProviderCredentialFailure(
    provider=CopilotResponsesProvider.name,
    kind="rate_limit",
    status_code=status,
    error_code=error_code,
    message=message,
    reset_at=_retry_after_reset_at(exc),
  )


def _map_copilot_event(event: dict[str, Any], state: _ResponsesStreamState) -> list[StreamEvent]:
  try:
    return map_event(event, state)
  except RuntimeError as exc:
    message = str(exc)
    # The reused mapper names xAI. This wire is Copilot.
    if message.startswith("xAI "):
      raise RuntimeError(f"copilot {message[4:]}") from exc
    raise


async def _error_message(response: httpx.Response) -> str:
  raw = await response.aread()
  raw_text = raw.decode("utf-8", errors="replace")
  message = raw_text or response.reason_phrase or "Request failed"
  try:
    parsed = json.loads(raw_text)
  except json.JSONDecodeError:
    parsed = None
  if isinstance(parsed, dict):
    error = parsed.get("error")
    if isinstance(error, dict):
      message = str(error.get("message") or error.get("code") or message)
    elif isinstance(error, str) and error.strip():
      message = error
  return f"copilot API error (status={response.status_code}): {message}"


class CopilotResponsesProvider(ModelProvider):
  """GitHub Copilot proxy using the OpenAI Responses wire only."""

  name = "copilot"

  HEADER_INTEGRATION_ID = "Copilot-Integration-Id"
  HEADER_EDITOR_VERSION = "Editor-Version"
  HEADER_OPENAI_INTENT = "Openai-Intent"
  HEADER_INITIATOR = "X-Initiator"
  OPENAI_INTENT = "conversation-edits"
  INITIATOR_AGENT = "agent"
  INITIATOR_USER = "user"

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport:
    return AdapterRouteSupport(
      adapter="copilot.responses",
      provider="copilot",
      protocol_profiles=frozenset({"responses.reasoning"}),
      routes=frozenset({"copilot.api"}),
    )

  def __init__(self, *, rate_table: RateTable | None = None) -> None:
    self._rate_table = _rate_table_for(self.name, rate_table)
    self._client_state: WeakKeyDictionary[httpx.AsyncClient, dict[str, Any]] = WeakKeyDictionary()

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(str(config.get("auth_token") or "").strip())

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    token = str(config.get("auth_token") or "").strip()
    if not token:
      raise RuntimeError("No copilot auth_token configured")
    # No read deadline of the provider's own: a quiet stream is bounded once, by
    # the runner's stream watchdog; connect stays bounded.
    client_kwargs: dict[str, Any] = {
      "timeout": httpx.Timeout(timeout=timeout, connect=10.0),
    }
    transport = config.get("_transport")
    if isinstance(transport, httpx.AsyncBaseTransport):
      client_kwargs["transport"] = transport
    client = httpx.AsyncClient(**client_kwargs)
    extra_headers = config.get("headers")
    self._client_state[client] = {
      "token": token,
      "endpoint_url": resolve_responses_url(_bound_base_url(config)),
      "headers": dict(extra_headers) if isinstance(extra_headers, dict) else {},
      "integration_id": _header_token(config.get("integration_id")),
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
    del max_tokens, request_context
    compat = responses_compat(model_info)
    if not compat.reasoning_control.values:
      if requested != ThinkingLevel.NONE:
        raise ValueError(
          f"copilot model {model!r} does not support reasoning effort "
          f"{requested.value!r}; unsupported effort is refused, never repaired"
        )
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})
    supported = tuple(ThinkingLevel(value) for value in compat.reasoning_control.values)
    if requested not in supported:
      raise ValueError(
        f"copilot model {model!r} does not support reasoning effort "
        f"{requested.value!r} (supported: "
        f"{', '.join(level.value for level in supported)}); unsupported "
        "effort is refused, never clamped"
      )
    return EffortResolution(
      requested=requested,
      effective=requested,
      thinking_enabled_effective=requested != ThinkingLevel.NONE,
      payload_fragments=reasoning_effort_fragment(compat, requested.value),
    )

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    # The Responses transcript contract is provider-neutral. Composition keeps
    # Copilot independent from Codex/OpenAI inheritance and auth routing.
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
    del thinking_level
    model_info = self.get_model_info(model)
    compat = responses_compat(model_info)
    if tools and not compat.function_tools:
      raise ValueError(f"copilot model {model!r} does not support Responses function tools")
    normalized_messages = self.normalize_messages(messages, model_info)
    params: dict[str, Any] = {
      "model": model,
      "store": False,
      "stream": True,
      "input": _convert_messages(normalized_messages, model_info),
      "instructions": _system_prompt_text(system_prompt).strip() or DEFAULT_INSTRUCTIONS,
      "include": ["reasoning.encrypted_content"],
      "max_output_tokens": max_tokens,
      "_initiator": _initiator_for(normalized_messages),
    }
    if tools:
      params["tools"] = _convert_tools(tools, strict=None)
      params["tool_choice"] = "auto"
      params["parallel_tool_calls"] = True
    effort_resolution = kwargs.get("effort_resolution")
    if not isinstance(effort_resolution, EffortResolution):
      raise ValueError(
        "copilot build_request_params requires an admitted EffortResolution; "
        "it does not resolve effort locally"
      )
    reasoning = effort_resolution.payload_fragments.get("reasoning")
    if isinstance(reasoning, dict):
      params["reasoning"] = dict(reasoning)
      if effort_resolution.effective != ThinkingLevel.NONE and compat.reasoning_summary:
        params["reasoning"]["summary"] = "auto"
    headers = kwargs.get("headers")
    if isinstance(headers, dict) and headers:
      params["_headers"] = dict(headers)
    if kwargs.get("base_url"):
      params["_endpoint_url"] = resolve_responses_url(str(kwargs["base_url"]))
    return params

  def _request_headers(
    self,
    state: dict[str, Any],
    *,
    initiator: str,
    extra: dict[str, Any] | None,
  ) -> dict[str, str]:
    merged: dict[str, Any] = dict(state.get("headers") or {})
    if isinstance(extra, dict):
      merged.update(extra)
    integration_id = state.get("integration_id")
    _drop_header(merged, self.HEADER_INTEGRATION_ID)
    if integration_id:
      merged[self.HEADER_INTEGRATION_ID] = integration_id
    editor_version = _editor_version()
    if editor_version:
      _drop_header(merged, self.HEADER_EDITOR_VERSION)
      merged[self.HEADER_EDITOR_VERSION] = editor_version
    merged[self.HEADER_OPENAI_INTENT] = self.OPENAI_INTENT
    merged[self.HEADER_INITIATOR] = (
      initiator if initiator in {self.INITIATOR_AGENT, self.INITIATOR_USER} else self.INITIATOR_USER
    )
    return build_headers(str(state["token"]), merged)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
    if not isinstance(client, httpx.AsyncClient):
      raise RuntimeError("CopilotResponsesProvider requires an httpx.AsyncClient")
    state = self._client_state.get(client)
    if state is None:
      raise RuntimeError("copilot client state is missing")
    request_params = dict(params)
    additional_headers = request_params.pop("_headers", None)
    endpoint_url = str(request_params.pop("_endpoint_url", state["endpoint_url"]))
    initiator = str(request_params.pop("_initiator", self.INITIATOR_USER) or self.INITIATOR_USER)
    headers = self._request_headers(
      state,
      initiator=initiator,
      extra=additional_headers if isinstance(additional_headers, dict) else None,
    )
    async with client.stream("POST", endpoint_url, json=request_params, headers=headers) as response:
      if response.status_code != 200:
        message = await _error_message(response)
        raise httpx.HTTPStatusError(message, request=response.request, response=response)
      stream_state = _ResponsesStreamState()
      buffer = ""
      async for chunk in response.aiter_text():
        buffer += chunk
        events, buffer = _parse_sse(buffer)
        for event in events:
          for mapped in _map_copilot_event(event, stream_state):
            yield mapped

  def is_retryable_error(self, exc: Exception) -> bool:
    # 429 belongs to the credential path, not a transport retry.
    status = _status_code_from_exception(exc)
    if status in _RETRYABLE_STATUSES or (isinstance(status, int) and 500 <= status <= 599):
      return True
    return isinstance(exc, (httpx.TransportError, httpx.StreamError))

  def is_context_length_error(self, exc: Exception) -> bool:
    return _is_context_length_exception(exc)

  def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
    status = _status_code_from_exception(exc)
    if status == 429:
      return _rate_limit_failure(exc, status)
    if status == 401:
      error_code, message = _failure_fields(exc)
      return ProviderCredentialFailure(
        provider=self.name,
        kind="auth",
        status_code=status,
        error_code=error_code,
        message=message,
      )
    if status == 403:
      error_code, message = _failure_fields(exc)
      return ProviderCredentialFailure(
        provider=self.name,
        kind="billing",
        status_code=status,
        error_code=error_code,
        message=message,
      )
    # A 400, a content refusal, and a context-length error are not a hop.
    if status == 400 or _is_context_length_exception(exc) or _is_content_refusal(exc):
      return None
    if _is_usage_limit_only(exc):
      return _rate_limit_failure(exc, status)
    return _classify_provider_credential_failure(provider=self.name, exc=exc)

  def next_credential(
    self,
    config: Mapping[str, Any],
    failure: ProviderCredentialFailure,
  ) -> dict[str, Any] | None:
    from . import copilot_oauth  # noqa: F401

    return rotate_provider_credential(
      self.name, config, failure_kind=failure.kind,
      reset_at=failure.reset_at, status_code=failure.status_code,
    )


__all__ = ["CopilotResponsesProvider"]
