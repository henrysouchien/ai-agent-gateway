from __future__ import annotations

import os
from typing import Any

from .providers import AnthropicProvider, ModelProvider, OpenAIProvider


_PROVIDER_DEFAULT_MODELS = {
  "anthropic": "claude-sonnet-4-6",
  "openai": "gpt-4o",
}

_ANTHROPIC_ALLOWED_MODELS = {"claude-sonnet-4-6", "claude-opus-4-6"}


def _resolve_provider(
  provider: str | ModelProvider,
  model: str | None,
  api_key: str | None,
  auth_token: str | None,
  provider_config: dict[str, Any] | None,
  *,
  auth_config: dict[str, Any] | None = None,
  max_tokens: int = 16_000,
) -> tuple[ModelProvider, str, dict[str, Any]]:
  provider_name: str
  if isinstance(provider, str):
    provider_name = provider.strip().lower()
    if provider_name == "anthropic":
      provider_instance: ModelProvider = AnthropicProvider()
    elif provider_name == "openai":
      provider_instance = OpenAIProvider()
    else:
      raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")
    if model is None:
      model = _PROVIDER_DEFAULT_MODELS.get(provider_name, "gpt-4o")
  elif isinstance(provider, ModelProvider):
    provider_instance = provider
    provider_name = str(getattr(provider, "name", "custom") or "custom")
    if model is None:
      # Fall back to auth_config["model"] if provided
      model = str((auth_config or {}).get("model", "")).strip() or None
      if model is None:
        raise ValueError("model is required when passing a ModelProvider instance (via arg or auth_config)")
  else:
    raise TypeError("provider must be a string ('anthropic', 'openai') or a ModelProvider instance")

  resolved_auth_config = dict(auth_config or {})
  if auth_config is None:
    if isinstance(provider_instance, AnthropicProvider):
      resolved_key = (api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
      resolved_token = (auth_token or "").strip() or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
      if resolved_key:
        resolved_auth_config = {
          "auth_mode": "api",
          "api_key": resolved_key,
          "auth_token": "",
        }
      elif resolved_token:
        resolved_auth_config = {
          "auth_mode": "oauth",
          "api_key": "",
          "auth_token": resolved_token,
        }
      else:
        resolved_auth_config = {
          "auth_mode": "api",
          "api_key": "",
          "auth_token": "",
        }
    else:
      resolved_key = (api_key or "").strip()
      resolved_auth_config = {"api_key": resolved_key} if resolved_key else {}
  elif isinstance(provider_instance, AnthropicProvider):
    auth_mode = str(resolved_auth_config.get("auth_mode", "")).strip().lower()
    if not auth_mode:
      if str(resolved_auth_config.get("api_key", "")).strip():
        auth_mode = "api"
      elif str(resolved_auth_config.get("auth_token", "")).strip():
        auth_mode = "oauth"
      else:
        auth_mode = "api"
    resolved_auth_config["auth_mode"] = auth_mode

  # auth_config wins for model/max_tokens if explicitly provided;
  # otherwise use the resolved values from args/defaults.
  if "model" not in resolved_auth_config or auth_config is None:
    resolved_auth_config["model"] = model
  if "max_tokens" not in resolved_auth_config or auth_config is None:
    resolved_auth_config["max_tokens"] = max_tokens
  if provider_config:
    resolved_auth_config.update(provider_config)

  return provider_instance, provider_name, resolved_auth_config


def _allowed_models_for_provider(
  provider: ModelProvider,
  model: str,
) -> set[str]:
  if isinstance(provider, AnthropicProvider):
    allowed_models = set(_ANTHROPIC_ALLOWED_MODELS)
    allowed_models.add(model)
    return allowed_models
  return set()


__all__ = [
  "_allowed_models_for_provider",
  "_resolve_provider",
]
