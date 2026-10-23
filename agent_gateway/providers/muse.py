from __future__ import annotations

import asyncio
import json
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Mapping
from weakref import WeakKeyDictionary

import httpx

from model_authority.registry import AdapterRouteSupport
from model_authority.rates import RateTable
from model_authority.thinking import EffortResolution, ThinkingLevel
from ..auth import CredentialFailureKind, ProviderCredentialFailure
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
from .credential_pool import rotate_provider_credential

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
from .codex import CodexProvider
from .openai_responses_helpers import (
  _ResponsesStreamState,
  _convert_messages,
  convert_openai_response_tools,
  map_event,
  reasoning_effort_fragment,
  responses_compat,
  responses_model_info,
  _system_prompt_text,
)
from .response_diagnostics import response_diagnostics
from .xai_helpers import (
  DEFAULT_INSTRUCTIONS,
  _parse_sse,
  build_headers,
  resolve_responses_url,
)


# Meta Model API Responses route. Quickstart and chain-of-thought continuity
# are POST /v1/responses; Chat Completions is offered but does not carry
# reasoning across turns.
# https://dev.meta.ai/resources/blog/build-with-muse-spark
# https://dev.meta.ai/docs/protocols/responses
# Auth is an API key sent as Authorization: Bearer, not an OAuth token.
# https://dev.meta.ai/docs/authentication
# No required version header is documented (create-response curl sends only
# Authorization and Content-Type).
# https://dev.meta.ai/docs/api-reference/responses/create-response
# [UNCONFIRMED] a usage-quota body distinct from the documented 429
# rate_limit_exceeded / rate_limit_error envelope. A body that is only a
# rate, usage, or quota exhaustion is still classified as rate_limit.
# https://dev.meta.ai/docs/error-handling
DEFAULT_MUSE_BASE_URL = "https://api.meta.ai/v1"

_RETRYABLE_STATUSES = frozenset({408, 409})
_REFUSAL_RE_TEXT = (
  "content_policy_violation",
  "content policy",
  "content-policy",
)
_USAGE_LIMIT_MARKERS = (
  "rate_limit_exceeded",
  "rate_limit_error",
  "rate limit",
  "rate-limit",
  "ratelimit",
  "too many requests",
  "quota exhausted",
  "quota exceeded",
  "usage limit",
  "usage quota",
  "usage_limit",
)


def _bound_base_url(config: dict[str, Any]) -> str:
  configured = str(config.get("base_url") or "").strip()
  return configured or DEFAULT_MUSE_BASE_URL


def _api_key(config: dict[str, Any]) -> str:
  return str(config.get("api_key") or "").strip()


def _searchable(exc: Exception) -> str:
  return " ".join(
    part for part in (str(exc), _response_text(exc), _error_code_from_exception(exc) or "") if part
  ).lower()


def _error_object(exc: Exception) -> dict[str, Any]:
  text = _response_text(exc) or str(exc)
  start = text.find("{")
  if start < 0:
    return {}
  try:
    parsed = json.loads(text[start:])
  except (json.JSONDecodeError, TypeError, ValueError):
    return {}
  if not isinstance(parsed, dict):
    return {}
  error = parsed.get("error")
  return error if isinstance(error, dict) else {}


def _muse_error_code(exc: Exception) -> str | None:
  error = _error_object(exc)
  for key in ("code", "type"):
    value = error.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()
  return _error_code_from_exception(exc)


def _failure_message(exc: Exception) -> str:
  error = _error_object(exc)
  message = error.get("message")
  if isinstance(message, str) and message.strip():
    return message.strip()
  return str(exc).strip()


def _is_content_refusal(exc: Exception) -> bool:
  error = _error_object(exc)
  code = str(error.get("code") or "").strip().lower()
  if code == "content_policy_violation":
    return True
  searchable = _searchable(exc)
  return any(marker in searchable for marker in _REFUSAL_RE_TEXT)


def _not_a_rate_limit(exc: Exception, status: int | None) -> bool:
  return status == 400 or _is_content_refusal(exc) or _is_context_length_exception(exc)


def _only_usage_limit(exc: Exception, status: int | None) -> bool:
  """True when the body is a rate/usage/quota exhaustion and nothing else."""
  if status in {400, 401, 403} or _not_a_rate_limit(exc, status):
    return False
  error = _error_object(exc)
  kind = str(error.get("type") or "").strip().lower()
  code = str(error.get("code") or "").strip().lower()
  if kind in {"invalid_request_error", "authentication_error", "billing_error", "server_error"}:
    return False
  if kind == "rate_limit_error" or code == "rate_limit_exceeded":
    return True
  usage_tokens = ("rate_limit", "ratelimit", "quota", "usage_limit", "usage_quota")
  if any(token in kind for token in usage_tokens) or any(token in code for token in usage_tokens):
    return True
  if kind or code:
    return False
  return any(marker in _searchable(exc) for marker in _USAGE_LIMIT_MARKERS)


def _retry_after_reset(exc: Exception) -> float | None:
  response = getattr(exc, "response", None)
  headers = getattr(response, "headers", None)
  if headers is None:
    return None
  raw = headers.get("Retry-After")
  if raw is None:
    raw = headers.get("retry-after")
  text = str(raw or "").strip()
  if not text:
    return None
  try:
    seconds = float(text)
  except ValueError:
    seconds = None
  if seconds is not None:
    if seconds < 0:
      return None
    return time.time() + seconds
  try:
    parsed = parsedate_to_datetime(text)
  except (TypeError, ValueError):
    return None
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed.timestamp()


def _credential_failure(
  exc: Exception,
  *,
  kind: CredentialFailureKind,
  status: int | None,
  reset_at: float | None = None,
) -> ProviderCredentialFailure:
  return ProviderCredentialFailure(
    provider="muse",
    kind=kind,
    status_code=status,
    error_code=_muse_error_code(exc),
    message=_failure_message(exc),
    reset_at=reset_at,
  )


def _defer_credential_failure(
  exc: Exception,
  status: int | None,
) -> ProviderCredentialFailure | None:
  if status == 429:
    return None
  failure = _classify_provider_credential_failure(provider="muse", exc=exc)
  if failure is not None and failure.kind == "rate_limit":
    return None
  return failure


async def _error_detail(response: httpx.Response) -> str:
  raw = await response.aread()
  text = raw.decode("utf-8", errors="replace").strip()
  message = text or response.reason_phrase or "Request failed"
  try:
    parsed = json.loads(text) if text else None
  except json.JSONDecodeError:
    parsed = None
  if isinstance(parsed, dict):
    error = parsed.get("error")
    if isinstance(error, dict):
      message = str(error.get("message") or error.get("code") or message)
  return f"Muse API error (status={response.status_code}): {message}"


def _map_muse_event(event: dict[str, Any], state: _ResponsesStreamState) -> list[StreamEvent]:
  try:
    mapped = map_event(event, state)
  except RuntimeError as exc:
    raise RuntimeError(_muse_stream_error(str(exc))) from exc
  terminal = state.terminal_error
  if terminal is not None:
    state.terminal_error = RuntimeError(_muse_stream_error(str(terminal)))
  return mapped


def _muse_stream_error(detail: str) -> str:
  for prefix in ("OpenAI Responses error: ", "OpenAI Responses failed: "):
    if detail.startswith(prefix):
      return f"Muse error: {detail[len(prefix):]}"
  if detail.startswith("Muse error: "):
    return detail
  return f"Muse error: {detail}"


class MuseResponsesProvider(ModelProvider):
  """Meta Model API adapter for the OpenAI-compatible Responses route.

  Family is Responses (``muse.responses`` / ``responses.reasoning``), not
  Chat Completions: Muse Spark is a reasoning model and the documented
  quickstart plus cross-turn chain of thought are ``POST /v1/responses``.
  Credential field is ``config["api_key"]`` (the docs call it an API key,
  sent as ``Authorization: Bearer``). Default base URL is
  ``https://api.meta.ai/v1``.
  """

  name = "muse"

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport:
    return AdapterRouteSupport(
      adapter="muse.responses",
      provider="muse",
      protocol_profiles=frozenset({"responses.reasoning"}),
      routes=frozenset({"muse.api"}),
    )

  def __init__(self, *, rate_table: RateTable | None = None) -> None:
    self._rate_table = _rate_table_for(self.name, rate_table)
    self._client_state: WeakKeyDictionary[httpx.AsyncClient, dict[str, Any]] = WeakKeyDictionary()

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(_api_key(config))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    token = _api_key(config)
    if not token:
      raise RuntimeError("No muse api credential configured")
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
          f"muse model {model!r} does not support reasoning effort "
          f"{requested.value!r}; unsupported effort is refused, never repaired"
        )
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})
    supported = tuple(ThinkingLevel(value) for value in compat.reasoning_control.values)
    if requested not in supported:
      raise ValueError(
        f"muse model {model!r} does not support reasoning effort "
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
    # Responses transcript shape is shared with the other Responses adapters.
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
      raise ValueError(f"muse model {model!r} does not support Responses function tools")
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
      params["tools"] = convert_openai_response_tools(tools)
      params["tool_choice"] = "auto"
      params["parallel_tool_calls"] = True
    effort_resolution = kwargs.get("effort_resolution")
    if not isinstance(effort_resolution, EffortResolution):
      raise ValueError(
        "muse build_request_params requires an admitted EffortResolution; "
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

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
    if not isinstance(client, httpx.AsyncClient):
      raise RuntimeError("MuseResponsesProvider requires an httpx.AsyncClient")
    state = self._client_state.get(client)
    if state is None:
      raise RuntimeError("muse client state is missing")
    request_params = dict(params)
    additional_headers = request_params.pop("_headers", None)
    endpoint_url = str(request_params.pop("_endpoint_url", state["endpoint_url"]))
    merged_headers = (
      {**state["headers"], **additional_headers}
      if isinstance(additional_headers, dict)
      else state["headers"]
    )
    headers = build_headers(str(state["token"]), merged_headers)
    async with client.stream("POST", endpoint_url, json=request_params, headers=headers) as response:
      if response.status_code != 200:
        message = await _error_detail(response)
        raise httpx.HTTPStatusError(message, request=response.request, response=response)
      stream_state = _ResponsesStreamState()
      stream_state.provider_request_id, stream_state.rate_limit = response_diagnostics(response.headers)
      buffer = ""
      async for chunk in response.aiter_text():
        buffer += chunk
        events, buffer = _parse_sse(buffer)
        for event in events:
          for mapped in _map_muse_event(event, stream_state):
            yield mapped
          if stream_state.terminal_error is not None:
            terminal_error = stream_state.terminal_error
            stream_state.terminal_error = None
            raise terminal_error

  def is_retryable_error(self, exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
      status = exc.response.status_code
      return status in _RETRYABLE_STATUSES or (isinstance(status, int) and 500 <= status < 600)
    return isinstance(exc, (httpx.TransportError, httpx.StreamError))

  def is_context_length_error(self, exc: Exception) -> bool:
    return _is_context_length_exception(exc)

  def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
    status = _status_code_from_exception(exc)
    blocked = _not_a_rate_limit(exc, status)
    if status == 429 and not blocked:
      return _credential_failure(exc, kind="rate_limit", status=status, reset_at=_retry_after_reset(exc))
    if blocked:
      return _defer_credential_failure(exc, status)
    if status == 401:
      return _credential_failure(exc, kind="auth", status=status)
    if status == 403:
      return _credential_failure(exc, kind="billing", status=status)
    if _only_usage_limit(exc, status):
      return _credential_failure(exc, kind="rate_limit", status=status, reset_at=_retry_after_reset(exc))
    if status != 429:
      return _classify_provider_credential_failure(provider=self.name, exc=exc)
    return None

  def next_credential(
    self,
    config: Mapping[str, Any],
    failure: ProviderCredentialFailure,
  ) -> dict[str, Any] | None:
    from . import muse_oauth  # noqa: F401

    return rotate_provider_credential(
      self.name, config, failure_kind=failure.kind,
      reset_at=failure.reset_at, status_code=failure.status_code,
    )


__all__ = ["MuseResponsesProvider"]
