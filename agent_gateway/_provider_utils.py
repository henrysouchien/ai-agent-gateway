from __future__ import annotations

import os
from typing import Any

from .providers import AnthropicProvider, CodexProvider, ModelProvider, OpenAIProvider, XAIProvider


_SELECTION_AUTH_CONFIG_FIELDS = frozenset({
  "effort",
  "model",
  "model_key",
  "thinking",
  "thinking_enabled_requested",
})


def _reject_selection_auth_config(
  config: dict[str, Any] | None,
  *,
  field_name: str,
) -> None:
  duplicated = sorted(_SELECTION_AUTH_CONFIG_FIELDS & set(config or {}))
  if duplicated:
    raise ValueError(
      f"{field_name} is not model-selection authority: "
      + ", ".join(duplicated)
    )

def _classify_anthropic_credential(raw: str) -> dict[str, Any]:
  """Return the Anthropic credential field matching the raw token format."""
  if raw.startswith("sk-ant-oat01-"):
    return {"auth_token": raw}
  return {"api_key": raw}


def resolve_auth_config(
  *,
  provider: str = "anthropic",
  api_key: str | None = None,
  auth_token: str | None = None,
  auth_config: dict[str, Any] | None = None,
  max_tokens: int | None = None,
  infer_mode_from_prefix: bool = False,
  read_env: bool = True,
  read_env_auth_mode: bool = False,
  raise_on_missing: bool = False,
) -> dict[str, Any]:
  """Resolve provider auth config from args, config dict, and env vars.

  Precedence: explicit args > auth_config dict > environment variables.
  Env vars are only consulted when ``auth_config`` is None and ``read_env``
  is True.  Extra keys in ``auth_config`` are preserved in the result.

  This function resolves credential and provider request-limit material only.
  Model and effort selection live exclusively in a ``CapabilityBind``.
  """
  provider_name = provider.strip().lower() if isinstance(provider, str) else "anthropic"
  _reject_selection_auth_config(auth_config, field_name="auth_config")

  # Start with a copy of auth_config to preserve extra keys.
  result: dict[str, Any] = dict(auth_config or {})

  # --- Non-Anthropic path ---
  if provider_name != "anthropic":
    resolved_key = (
      (api_key or "").strip()
      or str(result.get("api_key", "")).strip()
    )
    resolved_token = (
      (auth_token or "").strip()
      or str(result.get("auth_token", "")).strip()
    )
    if read_env and auth_config is None and provider_name in {"openai", "xai"}:
      prefix = provider_name.upper()
      resolved_key = resolved_key or os.environ.get(f"{prefix}_API_KEY", "").strip()
      resolved_token = resolved_token or os.environ.get(f"{prefix}_AUTH_TOKEN", "").strip()

    explicit_mode = str(result.get("auth_mode", "")).strip().lower()

    if explicit_mode:
      auth_mode = explicit_mode
    elif resolved_key:
      auth_mode = "api"
    elif resolved_token:
      auth_mode = "oauth"
    else:
      if max_tokens is not None:
        result["max_tokens"] = max_tokens
      return result

    result["auth_mode"] = auth_mode
    if auth_config is None:
      result["api_key"] = resolved_key if auth_mode == "api" else ""
      result["auth_token"] = resolved_token if auth_mode == "oauth" else ""
    else:
      result["api_key"] = resolved_key
      result["auth_token"] = resolved_token

    if max_tokens is not None:
      result["max_tokens"] = max_tokens
    return result

  # --- Anthropic path ---
  # Resolve credentials: args > config dict > env (env only if no config)
  use_env = read_env and auth_config is None

  resolved_key = (api_key or "").strip() or str(result.get("api_key", "")).strip()
  resolved_token = (auth_token or "").strip() or str(result.get("auth_token", "")).strip()

  if use_env:
    resolved_key = resolved_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    resolved_token = resolved_token or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    if not resolved_token:
      from .providers.anthropic_oauth import resolve_anthropic_oauth_token

      resolved_token, _store_path, _record = resolve_anthropic_oauth_token()

  # Resolve auth_mode with precedence chain
  explicit_mode = str(result.get("auth_mode", "")).strip().lower()
  if not explicit_mode and read_env_auth_mode and auth_config is None:
    explicit_mode = os.environ.get("ANTHROPIC_AUTH_MODE", "").strip().lower()

  if explicit_mode:
    auth_mode = explicit_mode
  elif infer_mode_from_prefix and resolved_key and resolved_key.startswith("sk-ant-oat"):
    # Single-field consumers (e.g. finance_cli) store OAuth token as "api_key"
    auth_mode = "oauth"
    resolved_token = resolved_key
    resolved_key = ""
  elif resolved_key:
    auth_mode = "api"  # api_key wins when both present (matches _resolve_provider)
  elif resolved_token:
    auth_mode = "oauth"
  else:
    auth_mode = "api"

  if raise_on_missing:
    if auth_mode == "oauth" and not resolved_token:
      raise RuntimeError(
        "auth_mode is 'oauth' but no auth_token found. "
        "Set ANTHROPIC_AUTH_TOKEN or pass auth_token=."
      )
    if auth_mode == "api" and not resolved_key:
      raise RuntimeError(
        "auth_mode is 'api' but no api_key found. "
        "Set ANTHROPIC_API_KEY or pass api_key=."
      )

  result["auth_mode"] = auth_mode
  if auth_config is None:
    # Building from scratch (args + env) — normalize: blank the inactive credential
    result["api_key"] = resolved_key if auth_mode == "api" else ""
    result["auth_token"] = resolved_token if auth_mode == "oauth" else ""
  else:
    # Config dict provided — preserve both fields as-is (matches current
    # _resolve_provider lines 75-84 and _prepare_auth_config lines 41-55
    # which only set auth_mode without blanking credentials).
    result["api_key"] = resolved_key
    result["auth_token"] = resolved_token

  # Request limits may be carried with credential material; model selection may not.
  if max_tokens is not None:
    result["max_tokens"] = max_tokens
  return result


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
  normalized_model = str(model or "").strip()
  if not normalized_model:
    raise ValueError(
      "adapter resolution requires an upstream model from a CapabilityBind"
    )
  _reject_selection_auth_config(
    provider_config,
    field_name="provider_config",
  )
  _reject_selection_auth_config(auth_config, field_name="auth_config")
  if isinstance(provider, str):
    provider_name = provider.strip().lower()
    if provider_name == "anthropic":
      provider_instance: ModelProvider = AnthropicProvider()
    elif provider_name == "codex":
      provider_instance = CodexProvider()
    elif provider_name == "openai":
      provider_instance = OpenAIProvider()
    elif provider_name == "xai":
      provider_instance = XAIProvider()
    else:
      raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'codex', 'openai', or 'xai'.")
  elif isinstance(provider, ModelProvider):
    provider_instance = provider
    provider_name = str(getattr(provider, "name", "custom") or "custom")
  else:
    raise TypeError("provider must be a string ('anthropic', 'codex', 'openai', 'xai') or a ModelProvider instance")

  if auth_config is None and isinstance(provider_instance, AnthropicProvider):
    resolved_auth_config = resolve_auth_config(
      provider="anthropic",
      api_key=api_key,
      auth_token=auth_token,
    )
  elif auth_config is None:
    resolved_auth_config = resolve_auth_config(
      provider=provider_name,
      api_key=api_key,
      auth_token=auth_token,
    )
  elif isinstance(provider_instance, AnthropicProvider):
    resolved_auth_config = resolve_auth_config(
      provider="anthropic",
      auth_config=auth_config,
    )
  elif isinstance(provider_instance, OpenAIProvider):
    resolved_auth_config = resolve_auth_config(
      provider="openai",
      auth_config=auth_config,
    )
  elif isinstance(provider_instance, CodexProvider):
    resolved_auth_config = resolve_auth_config(
      provider="codex",
      auth_config=auth_config,
    )
  elif isinstance(provider_instance, XAIProvider):
    resolved_auth_config = resolve_auth_config(
      provider="xai",
      auth_config=auth_config,
    )
  else:
    resolved_auth_config = dict(auth_config or {})

  if "max_tokens" not in resolved_auth_config or auth_config is None:
    resolved_auth_config["max_tokens"] = max_tokens
  if provider_config:
    resolved_auth_config.update(provider_config)

  return provider_instance, provider_name, resolved_auth_config
__all__ = [
  "_classify_anthropic_credential",
  "_resolve_provider",
  "resolve_auth_config",
]
