"""Kimi Code (`kimi_code`) chat-completions provider adapter (`https://api.kimi.com/coding/v1`)."""

from __future__ import annotations

from typing import Any, Mapping

import httpx
from model_authority.registry import AdapterRouteSupport
from model_authority.thinking import ThinkingLevel

from ..auth import ProviderCredentialFailure
from .base import (
  EffortResolution,
  ModelInfo,
  _classify_provider_credential_failure,
  _status_code_from_exception,
)
from .credential_pool import rotate_provider_credential
from .kimi_code_oauth import kimi_common_headers
from .ollama import OllamaLocalProvider, _chat_compat, _chat_completions_url
from .ollama_cloud import (
  _credential_failure,
  _is_content_refusal,
  _is_usage_limit_only,
  _retry_after_reset_at,
  _without_replayed_reasoning,
)

DEFAULT_KIMI_CODE_BASE_URL = "https://api.kimi.com/coding/v1"


def _bearer_token(config: Mapping[str, Any]) -> str:
  mode = str(config.get("auth_mode") or "").strip().lower()
  if mode == "api":
    return str(config.get("api_key") or config.get("auth_token") or "").strip()
  if mode == "oauth":
    return str(config.get("auth_token") or config.get("api_key") or "").strip()
  return str(config.get("auth_token") or config.get("api_key") or "").strip()


class KimiCodeProvider(OllamaLocalProvider):
  """Chat completions against Kimi Code (`https://api.kimi.com/coding/v1/chat/completions`).

  Authenticates with `Authorization: Bearer <auth_token or api_key>` plus Kimi CLI
  attribution headers (`User-Agent: KimiCLI/1.0`, `X-Msh-Platform: kimi_cli`,
  `X-Msh-Device-*`). Reasoning effort is encoded in Kimi's native `thinking`
  request field (`{"type": "enabled", "effort": "<level>"}` when enabled, or
  `{"type": "disabled"}` when `none`).
  """

  name = "kimi_code"

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport:
    return AdapterRouteSupport(
      adapter="kimi_code.chat_completions",
      provider="kimi_code",
      protocol_profiles=frozenset({"chat_completions.standard"}),
      routes=frozenset({"kimi.code"}),
    )

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(_bearer_token(config))

  def _base_url(self, config: dict[str, Any]) -> str:
    configured = str(config.get("base_url") or "").strip()
    return configured or DEFAULT_KIMI_CODE_BASE_URL

  def _prepared_headers(self, config: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {
      "Accept": "text/event-stream",
      "Content-Type": "application/json",
      **kimi_common_headers(),
    }
    extra = config.get("headers")
    if isinstance(extra, Mapping):
      for key, value in extra.items():
        if str(key).lower() == "authorization":
          continue
        headers[str(key)] = str(value)
    token = _bearer_token(config)
    if token:
      headers["Authorization"] = f"Bearer {token}"
    return headers

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    token = _bearer_token(config)
    if not token:
      raise RuntimeError("No kimi_code credential configured; run `agent auth login kimi-code`")
    client_kwargs: dict[str, Any] = {
      "timeout": httpx.Timeout(timeout=timeout, connect=10.0),
    }
    transport = config.get("_transport")
    if isinstance(transport, httpx.AsyncBaseTransport):
      client_kwargs["transport"] = transport
    client = httpx.AsyncClient(**client_kwargs)
    self._client_state[client] = {
      "endpoint_url": _chat_completions_url(self._base_url(config)),
      "headers": self._prepared_headers(config),
      "send_authorization": True,
    }
    return client

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
    model_info = self.get_model_info(model)
    compat = _chat_compat(model_info)
    clamped_max_tokens = min(max_tokens, model_info.max_output_tokens)
    params = super().build_request_params(
      model=model,
      messages=messages,
      system_prompt=system_prompt,
      tools=tools,
      max_tokens=clamped_max_tokens,
      thinking_level=thinking_level,
      **kwargs,
    )
    effort_resolution = kwargs.get("effort_resolution")
    if compat.reasoning_control is not None and isinstance(effort_resolution, EffortResolution):
      params.pop("reasoning_effort", None)
      if effort_resolution.thinking_enabled_effective:
        params["thinking"] = {
          "type": "enabled",
          "effort": effort_resolution.effective.value,
        }
      else:
        params["thinking"] = {"type": "disabled"}
    return params

  def classify_credential_failure(
    self,
    exc: Exception,
  ) -> ProviderCredentialFailure | None:
    if _is_content_refusal(exc):
      return None
    status = _status_code_from_exception(exc)
    reset_at = _retry_after_reset_at(exc)
    if status == 401:
      return _credential_failure(self.name, exc, kind="auth", status_code=status)
    if status == 402:
      return _credential_failure(
        self.name, exc, kind="billing", status_code=status, reset_at=reset_at
      )
    if status == 429:
      if _is_usage_limit_only(exc):
        return _credential_failure(
          self.name, exc, kind="rate_limit", status_code=status, reset_at=reset_at
        )
      classified = _classify_provider_credential_failure(provider=self.name, exc=exc)
      if classified is not None and reset_at is not None and classified.reset_at is None:
        return ProviderCredentialFailure(
          provider=classified.provider,
          kind=classified.kind,
          status_code=classified.status_code,
          error_code=classified.error_code,
          message=classified.message,
          reset_at=reset_at,
        )
      return classified
    if _is_usage_limit_only(exc):
      return _credential_failure(
        self.name, exc, kind="rate_limit", status_code=status, reset_at=reset_at
      )
    return _classify_provider_credential_failure(provider=self.name, exc=exc)

  def next_credential(
    self,
    config: Mapping[str, Any],
    failure: ProviderCredentialFailure,
  ) -> dict[str, Any] | None:
    from . import kimi_code_oauth  # noqa: F401

    return rotate_provider_credential(
      self.name,
      config,
      failure_kind=failure.kind,
      reset_at=failure.reset_at,
      status_code=failure.status_code,
    )


__all__ = [
  "DEFAULT_KIMI_CODE_BASE_URL",
  "KimiCodeProvider",
]
