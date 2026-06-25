from __future__ import annotations

import logging
import os
from typing import Any

from .fixture_gate import FIXTURE_MODEL_ID, require_fixture_provider_available
from .providers import AnthropicProvider, CodexProvider, FixtureProvider, ModelProvider, OpenAIProvider

log = logging.getLogger("agent_gateway.provider_utils")
_SUB_AGENT_DEFAULT_MODEL_WARNED: set[str] = set()
_API_CREDENTIALS_MODULE_NAMES = frozenset({"api", "api.credentials"})
_API_TOOL_CATALOG_MODULE_NAMES = frozenset(
  {"api", "api.agent", "api.agent.shared", "api.agent.shared.tool_catalog"}
)
_CREDENTIALS_MODULE_NAMES = frozenset({"credentials"})
_TOOL_CATALOG_MODULE_NAMES = frozenset({"agent", "agent.shared", "agent.shared.tool_catalog"})

_FALLBACK_DEFAULT_MODELS = {
  "agent-sdk": "claude-sonnet-4-6",
  "anthropic": "claude-sonnet-4-6",
  "codex": "gpt-5.4",
  "fixture": FIXTURE_MODEL_ID,
  "openai": "gpt-4o",
}

_FALLBACK_ALLOWED_MODELS = {
  "agent-sdk": {"claude-fable-5", "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8"},
  "anthropic": {"claude-fable-5", "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8"},
  "codex": {
    "gpt-5.5",
    "gpt-5.1",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
  },
  "fixture": {FIXTURE_MODEL_ID},
  "openai": {"gpt-5.5", "gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"},
}


def _get_default_model_for_provider(provider: str | None = None) -> str:
  resolved = str(provider or "anthropic").strip().lower() or "anthropic"
  try:
    from credentials import get_default_model as get_default_model
  except ModuleNotFoundError as exc:
    if exc.name not in _CREDENTIALS_MODULE_NAMES:
      raise
    try:
      from api.credentials import get_default_model as get_default_model
    except ModuleNotFoundError as exc:
      if exc.name not in _API_CREDENTIALS_MODULE_NAMES:
        raise
      return _FALLBACK_DEFAULT_MODELS.get(resolved, _FALLBACK_DEFAULT_MODELS["anthropic"])
  try:
    model = str(get_default_model(resolved)).strip()
  except Exception:
    model = ""
  if model:
    return model
  return _FALLBACK_DEFAULT_MODELS.get(resolved, _FALLBACK_DEFAULT_MODELS["anthropic"])


def _get_allowed_models_for_provider_name(provider: str | None = None) -> set[str]:
  resolved = str(provider or "anthropic").strip().lower() or "anthropic"
  try:
    from agent.shared.tool_catalog import get_allowed_models as get_allowed_models
  except ModuleNotFoundError as exc:
    if exc.name not in _TOOL_CATALOG_MODULE_NAMES:
      raise
    try:
      from api.agent.shared.tool_catalog import get_allowed_models as get_allowed_models
    except ModuleNotFoundError as exc:
      if exc.name not in _API_TOOL_CATALOG_MODULE_NAMES:
        raise
      models = set(_FALLBACK_ALLOWED_MODELS.get(resolved, set()))
      default_model = _get_default_model_for_provider(resolved)
      if default_model:
        models.add(default_model)
      return models
  return set(get_allowed_models(resolved))


def sub_agent_default_model(allowed_models: set[str] | frozenset[str] | None) -> str | None:
  """Return the operator's default sub-agent model when valid for this provider."""
  raw_model = os.environ.get("SUB_AGENT_DEFAULT_MODEL", "").strip()
  if not raw_model:
    return None

  allowed = set(allowed_models or set())
  if raw_model in allowed:
    return raw_model

  if raw_model not in _SUB_AGENT_DEFAULT_MODEL_WARNED:
    _SUB_AGENT_DEFAULT_MODEL_WARNED.add(raw_model)
    log.warning(
      "Ignoring SUB_AGENT_DEFAULT_MODEL=%r because it is not in the resolved provider allowlist",
      raw_model,
    )
  return None


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
  model: str | None = None,
  max_tokens: int | None = None,
  thinking: bool | None = None,
  infer_mode_from_prefix: bool = False,
  read_env: bool = True,
  read_env_auth_mode: bool = False,
  raise_on_missing: bool = False,
) -> dict[str, Any]:
  """Resolve provider auth config from args, config dict, and env vars.

  Precedence: explicit args > auth_config dict > environment variables.
  Env vars are only consulted when ``auth_config`` is None and ``read_env``
  is True.  Extra keys in ``auth_config`` are preserved in the result.

  This function does NOT inject default values for ``model``, ``max_tokens``,
  or ``thinking`` — it only sets them when explicitly passed as kwargs.
  Callers like ``_resolve_provider()`` handle their own model defaulting.
  """
  provider_name = provider.strip().lower() if isinstance(provider, str) else "anthropic"

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

    explicit_mode = str(result.get("auth_mode", "")).strip().lower()

    if explicit_mode:
      auth_mode = explicit_mode
    elif resolved_key:
      auth_mode = "api"
    elif resolved_token:
      auth_mode = "oauth"
    else:
      if model is not None:
        result["model"] = model
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

    if model is not None:
      result["model"] = model
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

  # Only set model/max_tokens/thinking when explicitly passed — no defaults injected.
  if model is not None:
    result["model"] = model
  if max_tokens is not None:
    result["max_tokens"] = max_tokens
  if thinking is not None:
    result["thinking"] = thinking

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
  if isinstance(provider, str):
    provider_name = provider.strip().lower()
    if provider_name == "anthropic":
      provider_instance: ModelProvider = AnthropicProvider()
    elif provider_name == "codex":
      provider_instance = CodexProvider()
    elif provider_name == "openai":
      provider_instance = OpenAIProvider()
    elif provider_name == "fixture":
      require_fixture_provider_available("fixture provider resolver", error_type=ValueError)
      provider_instance = FixtureProvider()
    else:
      raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'codex', 'openai', or dev-only 'fixture'.")
    if model is None:
      model = _get_default_model_for_provider(provider_name)
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
  elif isinstance(provider_instance, FixtureProvider):
    resolved_auth_config = dict(auth_config or {})
  else:
    resolved_auth_config = dict(auth_config)

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
    allowed_models = _get_allowed_models_for_provider_name("anthropic")
    if model:
      allowed_models.add(model)
    return allowed_models
  if isinstance(provider, FixtureProvider):
    return {model or FIXTURE_MODEL_ID}
  return set()


__all__ = [
  "_classify_anthropic_credential",
  "_allowed_models_for_provider",
  "_resolve_provider",
  "resolve_auth_config",
]
