from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping

from model_authority.registry import AdapterRouteSupport
from model_authority.thinking import ThinkingLevel

from ..auth import CredentialFailureKind, ProviderCredentialFailure
from .base import (
  ModelInfo,
  _classify_provider_credential_failure,
  _error_code_from_exception,
  _is_context_length_exception,
  _response_text,
  _status_code_from_exception,
)
from .ollama import OllamaLocalProvider
from .api_key_store import resolve_api_key_store_paths
from .credential_pool import (
  PoolAccountCandidate,
  load_pool_accounts,
  normalize_user_scope,
  register_provider_pool_hooks,
  rotate_provider_credential,
)

# Direct cloud inference. https://docs.ollama.com/api/authentication
_DEFAULT_BASE_URL = "https://ollama.com/v1"
# Cloud rejects an output-token request above this (omp
# OLLAMA_CLOUD_NUM_PREDICT_CAP). Chat completions sends the cap under the
# entry's output-token parameter; this route does not send native num_predict.
_OUTPUT_TOKEN_CAP = 65_536
_RETRY_AFTER_HEADER_NAMES = ("retry-after", "Retry-After", "RETRY-AFTER")
# Fields Cloud rejects on replayed assistant history. Native /api/chat rejects
# `thinking`; this OpenAI-compat route also emits thinking as `reasoning`.
_REPLAYED_REASONING_KEYS = frozenset({
  "thinking",
  "reasoning",
  "reasoning_content",
})
_REASONING_BLOCK_TYPES = frozenset({
  "thinking",
  "redacted_thinking",
  "reasoning",
})
_USAGE_LIMIT_RE = re.compile(
  r"\b(?:rate(?:\s+limit(?:ed|ing)?|\s+limited)|too many requests"
  r"|usage(?:\s+limit|\s+exhausted)|quota(?:\s+exhausted|\s+exceeded))\b",
  re.IGNORECASE,
)
_CONTENT_REFUSAL_RE = re.compile(
  r"\b(?:content[_\s-]*filter|content[_\s-]*policy|safety[_\s-]*violation|refused to)\b",
  re.IGNORECASE,
)
_BILLING_NOT_USAGE_RE = re.compile(
  r"\b(?:payment|credit(?:s)?|subscription|billing|insufficient[_\s-]+quota|monthly\s+(?:usage\s+)?limit)\b",
  re.IGNORECASE,
)
_MONTHLY_LIMIT_RE = re.compile(
  r"\b(?:monthly\s+(?:usage\s+)?limit|add\s+usage\s+credits)\b",
  re.IGNORECASE,
)
# Monthly usage-limit 429s from ollama.com carry no Retry-After header; park
# the exhausted credential for 24h rather than the 5-minute burst default.
_MONTHLY_LIMIT_COOLDOWN_SECONDS = 86_400.0
_CONTENT_REFUSAL_CODES = frozenset({"content_filter", "refusal"})


def resolve_ollama_cloud_pool_candidates(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[list[PoolAccountCandidate], None]:
  env = os.environ if environ is None else environ
  user_path, org_path = resolve_api_key_store_paths(
    "ollama_cloud", config, user_scope=user_scope, environ=env,
  )
  candidates: list[PoolAccountCandidate] = []
  seen: set[str] = set()
  for path in (user_path, org_path):
    if path is None:
      continue
    for record in load_pool_accounts(
      path, default_prefix="ollama_cloud", required_any_keys=("api_key",),
    ):
      key = str(record["identity"])
      if key not in seen:
        seen.add(key)
        candidates.append(PoolAccountCandidate(
          provider="ollama_cloud", key=key, record=record,
          store_path=path, is_user_store=path == user_path, auth_mode="api",
        ))
  for name in ("OLLAMA_CLOUD_API_KEY", "OLLAMA_API_KEY"):
    token = str(env.get(name) or "").strip()
    if token:
      record = {
        "api_key": token,
        "identity": f"env:{name}:{hashlib.sha256(token.encode()).hexdigest()[:12]}",
      }
      key = record["identity"]
      if key not in seen:
        seen.add(key)
        candidates.append(PoolAccountCandidate(
          provider="ollama_cloud", key=key, record=record, auth_mode="api",
        ))
  return candidates, None


def _materialize_ollama_cloud_candidate(
  selected: PoolAccountCandidate, config: Mapping[str, Any],
) -> dict[str, Any]:
  materialized = dict(config)
  materialized.update(
    auth_mode="api", api_key=selected.token,
    auth_mechanism=selected.auth_mechanism, entitlement_kind=selected.entitlement_kind,
    account_identity=selected.account_identity,
    auth_user_scope=normalize_user_scope(config.get("auth_user_scope")),
    auth_token="", auth_store_path=str(selected.store_path or ""), api_key_store_path="",
    refresh_token=str(selected.record.get("refresh_token") or "").strip(),
    token_expires_at=selected.record.get("expires_at"),
  )
  return materialized


register_provider_pool_hooks(
  "ollama_cloud", resolver=resolve_ollama_cloud_pool_candidates,
  materializer=_materialize_ollama_cloud_candidate,
)


def _api_key(config: dict[str, Any]) -> str:
  return str(config.get("api_key") or "").strip()


def _is_reasoning_block(block: object) -> bool:
  if not isinstance(block, dict):
    return False
  return block.get("type") in _REASONING_BLOCK_TYPES


def _without_replayed_reasoning(message: dict[str, Any]) -> dict[str, Any]:
  """Drop thinking and reasoning from one assistant turn.

  Cloud accepts those fields on a model response and rejects them on the next
  request with 400. The text is omitted, not demoted into ``content``.
  """

  if message.get("role") != "assistant":
    return message
  stripped = {
    key: value
    for key, value in message.items()
    if key not in _REPLAYED_REASONING_KEYS
  }
  content = stripped.get("content")
  if isinstance(content, list):
    stripped["content"] = [
      block for block in content if not _is_reasoning_block(block)
    ]
  return stripped


def _exception_text(exc: Exception) -> str:
  return " ".join(part for part in (str(exc), _response_text(exc)) if part).strip()


def _is_content_refusal(exc: Exception) -> bool:
  code = str(_error_code_from_exception(exc) or "")
  searchable = f"{_exception_text(exc)} {code}"
  return bool(_CONTENT_REFUSAL_RE.search(searchable) or code in _CONTENT_REFUSAL_CODES)


def _is_usage_limit_only(exc: Exception) -> bool:
  """True when the body is only a rate, usage, or quota-exhausted limit."""

  status = _status_code_from_exception(exc)
  if status in {400, 401, 403}:
    return False
  if _is_context_length_exception(exc) or _is_content_refusal(exc):
    return False
  searchable = _exception_text(exc)
  if _USAGE_LIMIT_RE.search(searchable) is None:
    return False
  return _BILLING_NOT_USAGE_RE.search(searchable) is None


def _header_text(headers: object, name: str) -> str:
  getter = getattr(headers, "get", None)
  if not callable(getter):
    return ""
  raw = getter(name)
  if raw is None:
    return ""
  return str(raw).strip()


def _retry_after_raw(exc: Exception) -> str | None:
  attached = getattr(exc, "retry_after", None)
  if attached is not None:
    text = str(attached).strip()
    if text:
      return text
  response = getattr(exc, "response", None)
  headers = getattr(response, "headers", None) if response is not None else None
  if headers is None:
    headers = getattr(exc, "headers", None)
  if headers is None:
    return None
  for name in _RETRY_AFTER_HEADER_NAMES:
    text = _header_text(headers, name)
    if text:
      return text
  return None


def _retry_after_reset_at(exc: Exception) -> float | None:
  """Epoch seconds from ``Retry-After`` (delta-seconds or HTTP-date)."""

  raw = _retry_after_raw(exc)
  if raw is None:
    return None
  try:
    seconds = float(raw)
  except ValueError:
    seconds = None
  if seconds is not None:
    if seconds < 0:
      return None
    return time.time() + seconds
  try:
    parsed = parsedate_to_datetime(raw)
  except (TypeError, ValueError, OverflowError, IndexError):
    return None
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed.timestamp()


def _credential_failure(
  provider: str,
  exc: Exception,
  *,
  kind: CredentialFailureKind,
  status_code: int | None,
  reset_at: float | None = None,
) -> ProviderCredentialFailure:
  return ProviderCredentialFailure(
    provider=provider,
    kind=kind,
    status_code=status_code,
    error_code=_error_code_from_exception(exc),
    message=_exception_text(exc),
    reset_at=reset_at,
  )


class OllamaCloudProvider(OllamaLocalProvider):
  """Chat completions against Ollama Cloud.

  Wire is ``POST {base}/chat/completions``. Default base is
  ``https://ollama.com/v1``. Auth is ``Authorization: Bearer`` from
  ``config["api_key"]`` (https://docs.ollama.com/api/authentication).
  Request facts come from the bound entry's chat-completions compat.
  Replayed assistant history omits thinking and reasoning fields. The
  output-token request is clamped to 65536.
  """

  name = "ollama_cloud"

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport:
    return AdapterRouteSupport(
      adapter="ollama_cloud.chat_completions",
      provider="ollama_cloud",
      protocol_profiles=frozenset({"chat_completions.standard"}),
      routes=frozenset({"ollama.cloud"}),
    )

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(_api_key(config))

  def _base_url(self, config: dict[str, Any]) -> str:
    configured = str(config.get("base_url") or "").strip()
    return configured or _DEFAULT_BASE_URL

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    if not self.has_active_credential(config):
      raise RuntimeError("No ollama_cloud api credential configured")
    return super().create_client(config, timeout=timeout)

  def normalize_messages(
    self,
    messages: list[dict[str, Any]],
    model_info: ModelInfo,
  ) -> list[dict[str, Any]]:
    normalized = super().normalize_messages(messages, model_info)
    return [_without_replayed_reasoning(message) for message in normalized]

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
    capped = max_tokens if max_tokens <= _OUTPUT_TOKEN_CAP else _OUTPUT_TOKEN_CAP
    params = super().build_request_params(
      model=model,
      messages=messages,
      system_prompt=system_prompt,
      tools=tools,
      max_tokens=capped,
      thinking_level=thinking_level,
      **kwargs,
    )
    wire_messages = params.get("messages")
    if isinstance(wire_messages, list):
      params["messages"] = [
        _without_replayed_reasoning(message) if isinstance(message, dict) else message
        for message in wire_messages
      ]
    return params

  def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
    status = _status_code_from_exception(exc)
    # A 400, a content refusal, and a context-length error are not a hop,
    # and are never rate_limit even when the body also says quota.
    if status == 400 or _is_context_length_exception(exc) or _is_content_refusal(exc):
      return None
    if status == 429:
      reset_at = _retry_after_reset_at(exc)
      if reset_at is None and _MONTHLY_LIMIT_RE.search(_exception_text(exc)):
        reset_at = time.time() + _MONTHLY_LIMIT_COOLDOWN_SECONDS
      return _credential_failure(
        self.name,
        exc,
        kind="rate_limit",
        status_code=status,
        reset_at=reset_at,
      )
    if status == 401:
      return _credential_failure(self.name, exc, kind="auth", status_code=status)
    if status == 403:
      return _credential_failure(self.name, exc, kind="billing", status_code=status)
    if _is_usage_limit_only(exc):
      return _credential_failure(
        self.name,
        exc,
        kind="rate_limit",
        status_code=status,
        reset_at=_retry_after_reset_at(exc),
      )
    return _classify_provider_credential_failure(provider=self.name, exc=exc)

  def next_credential(
    self,
    config: Mapping[str, Any],
    failure: ProviderCredentialFailure,
  ) -> dict[str, Any] | None:
    return rotate_provider_credential(
      self.name, config, failure_kind=failure.kind,
      reset_at=failure.reset_at, status_code=failure.status_code,
    )


__all__ = ["OllamaCloudProvider", "resolve_ollama_cloud_pool_candidates"]
