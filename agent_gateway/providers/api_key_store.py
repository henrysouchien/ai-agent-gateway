from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .credential_pool import (
  load_pool_accounts,
  remove_pool_account,
  resolve_provider_store_path,
  resolve_provider_store_paths,
  upsert_pool_account,
)

# OAuth lives in oauth.json for these families. auth_store_path selects that
# file and must not also override the API-key store (credentials.json).
_DUAL_STORE_API_KEY_FAMILIES = {"google", "gemini", "kimi_code"}


def _api_key_store_overrides(family: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
  override_keys = (
    ("api_key_store_path",)
    if family in _DUAL_STORE_API_KEY_FAMILIES
    else ("api_key_store_path", "auth_store_path")
  )
  if family in {"google", "gemini"}:
    override_envs = ("GOOGLE_API_KEY_STORE_PATH", "GEMINI_API_KEY_STORE_PATH")
  elif family in _DUAL_STORE_API_KEY_FAMILIES:
    override_envs = (f"{family.upper()}_API_KEY_STORE_PATH",)
  else:
    override_envs = (f"{family.upper()}_API_KEY_STORE_PATH", f"{family.upper()}_AUTH_STORE_PATH")
  return override_keys, override_envs


def resolve_api_key_store_path(
  family: str,
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> Path:
  override_keys, override_envs = _api_key_store_overrides(family)
  return resolve_provider_store_path(
    family,
    "credentials.json",
    config,
    environ=environ,
    user_scope=user_scope,
    override_config_keys=override_keys,
    override_env_vars=override_envs,
  )


def resolve_api_key_store_paths(
  family: str,
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[Path | None, Path]:
  override_keys, override_envs = _api_key_store_overrides(family)
  return resolve_provider_store_paths(
    family,
    "credentials.json",
    config,
    environ=environ,
    user_scope=user_scope,
    override_config_keys=override_keys,
    override_env_vars=override_envs,
  )


def load_api_key_accounts(path: Path, *, family: str = "api-key") -> list[dict[str, Any]]:
  return load_pool_accounts(path, default_prefix=family, required_any_keys=("api_key",))


def load_api_key_record(path: Path) -> dict[str, Any] | None:
  return accounts[0] if (accounts := load_api_key_accounts(path)) else None


def upsert_api_key_record(
  path: Path,
  record: Mapping[str, Any],
  *,
  family: str = "api-key",
) -> list[dict[str, Any]]:
  return upsert_pool_account(path, record, default_prefix=family, required_any_keys=("api_key",))


def save_api_key_record(path: Path, record: Mapping[str, Any]) -> None:
  upsert_api_key_record(path, record)


def remove_api_key_record(path: Path, identity: str | None = None) -> bool:
  return remove_pool_account(
    path,
    identity,
    default_prefix="api-key",
    required_any_keys=("api_key",),
  )
