from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import httpx

from .api_key_store import load_api_key_accounts, resolve_api_key_store_paths
from .credential_pool import (
  PoolAccountCandidate,
  derive_account_identity,
  load_pool_accounts,
  normalize_user_scope,
  register_provider_pool_hooks,
  remove_pool_account,
  resolve_provider_store_path,
  resolve_provider_store_paths,
  upsert_pool_account,
)

DEFAULT_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
DEFAULT_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_GOOGLE_SCOPE = "https://www.googleapis.com/auth/generative-language"
EXPIRY_MARGIN_SECONDS = 60
_HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


@dataclass(frozen=True)
class GoogleOAuthSettings:
  client_id: str
  client_secret: str | None
  auth_url: str
  token_url: str
  scope: str
  store_path: Path


def resolve_google_oauth_settings(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> GoogleOAuthSettings:
  cfg = config or {}
  env = os.environ if environ is None else environ
  client_id = str(
    cfg.get("client_id")
    or cfg.get("oauth_client_id")
    or env.get("GOOGLE_OAUTH_CLIENT_ID")
    or env.get("GEMINI_OAUTH_CLIENT_ID")
    or ""
  ).strip()
  client_secret_raw = (
    cfg.get("client_secret")
    or cfg.get("oauth_client_secret")
    or env.get("GOOGLE_OAUTH_CLIENT_SECRET")
    or env.get("GEMINI_OAUTH_CLIENT_SECRET")
  )
  client_secret = str(client_secret_raw).strip() if client_secret_raw is not None else None
  auth_url = str(
    cfg.get("auth_url")
    or cfg.get("oauth_auth_url")
    or env.get("GOOGLE_OAUTH_AUTH_URL")
    or DEFAULT_GOOGLE_AUTH_URL
  ).strip()
  token_url = str(
    cfg.get("token_url")
    or cfg.get("oauth_token_url")
    or env.get("GOOGLE_OAUTH_TOKEN_URL")
    or DEFAULT_GOOGLE_TOKEN_URL
  ).strip()
  scope = str(
    cfg.get("scope")
    or cfg.get("oauth_scope")
    or env.get("GOOGLE_OAUTH_SCOPES")
    or DEFAULT_GOOGLE_SCOPE
  ).strip()

  store_path = resolve_provider_store_path(
    "google",
    "oauth.json",
    cfg,
    environ=env,
    user_scope=user_scope,
    override_config_keys=("auth_store_path",),
    override_env_vars=("GOOGLE_AUTH_STORE_PATH", "GEMINI_AUTH_STORE_PATH"),
  )

  return GoogleOAuthSettings(
    client_id=client_id,
    client_secret=client_secret,
    auth_url=auth_url,
    token_url=token_url,
    scope=scope,
    store_path=store_path,
  )


def resolve_google_oauth_store_paths(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[Path | None, Path]:
  return resolve_provider_store_paths(
    "google",
    "oauth.json",
    config,
    environ=environ,
    user_scope=user_scope,
    override_config_keys=("auth_store_path",),
    override_env_vars=("GOOGLE_AUTH_STORE_PATH", "GEMINI_AUTH_STORE_PATH"),
  )


def load_google_accounts(path: Path) -> list[dict[str, Any]]:
  return load_pool_accounts(
    path,
    default_prefix="google",
    required_any_keys=("access_token", "refresh_token"),
    strict=True,
  )


def load_google_token_record(path: Path) -> dict[str, Any] | None:
  return accounts[0] if (accounts := load_google_accounts(path)) else None


def upsert_google_token_record(path: Path, record: Mapping[str, Any]) -> list[dict[str, Any]]:
  return upsert_pool_account(
    path,
    record,
    default_prefix="google",
    required_any_keys=("access_token", "refresh_token"),
  )


def remove_google_token_record(path: Path, identity: str | None = None) -> bool:
  return remove_pool_account(
    path,
    identity,
    default_prefix="google",
    required_any_keys=("access_token", "refresh_token"),
  )


def save_google_token_record(path: Path, record: Mapping[str, Any]) -> None:
  access_token = str(record.get("access_token") or "").strip()
  if not access_token:
    raise ValueError("Google OAuth token record missing required 'access_token'")

  upsert_google_token_record(path, record)


def token_needs_refresh(record: Mapping[str, Any], *, margin_seconds: int = EXPIRY_MARGIN_SECONDS) -> bool:
  expires_at = record.get("expires_at")
  if expires_at is None:
    return False
  try:
    exp_float = float(expires_at)
  except (ValueError, TypeError):
    return True
  return exp_float <= (time.time() + margin_seconds)


def refresh_google_tokens(
  settings: GoogleOAuthSettings,
  record: Mapping[str, Any],
  *,
  client: httpx.Client | None = None,
) -> dict[str, Any]:
  refresh_token = str(record.get("refresh_token") or "").strip()
  if not refresh_token:
    raise RuntimeError("Cannot refresh Google OAuth token without a refresh_token")

  data: dict[str, str] = {
    "grant_type": "refresh_token",
    "refresh_token": refresh_token,
    "client_id": settings.client_id,
  }
  if settings.client_secret:
    data["client_secret"] = settings.client_secret

  headers = {"Accept": "application/json"}
  own_client = client is None
  http = client or httpx.Client(timeout=_HTTP_TIMEOUT)
  try:
    response = http.post(settings.token_url, data=data, headers=headers)
  finally:
    if own_client:
      http.close()

  if response.status_code != 200:
    raise RuntimeError(
      f"Google OAuth token refresh failed ({response.status_code}): {response.text}"
    )

  payload = response.json()
  new_access_token = str(payload.get("access_token") or "").strip()
  if not new_access_token:
    raise RuntimeError("Google token endpoint response missing access_token")

  expires_in = payload.get("expires_in")
  expires_at = (time.time() + float(expires_in)) if expires_in is not None else None

  updated = dict(record)
  updated["access_token"] = new_access_token
  if expires_at is not None:
    updated["expires_at"] = expires_at
  if "refresh_token" in payload and str(payload["refresh_token"]).strip():
    updated["refresh_token"] = str(payload["refresh_token"]).strip()
  if "scope" in payload:
    updated["scope"] = payload["scope"]

  save_google_token_record(settings.store_path, updated)
  return updated


def resolve_google_pool_candidates(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[list[PoolAccountCandidate], Callable[[PoolAccountCandidate], PoolAccountCandidate]]:
  cfg = config or {}
  env = os.environ if environ is None else environ
  settings = resolve_google_oauth_settings(cfg, user_scope=user_scope, environ=env)
  user_oauth_path, org_oauth_path = resolve_google_oauth_store_paths(
    cfg, user_scope=user_scope, environ=env,
  )
  user_api_path, org_api_path = resolve_api_key_store_paths(
    "google", cfg, user_scope=user_scope, environ=env,
  )
  candidates: list[PoolAccountCandidate] = []
  seen: set[str] = set()
  for path, auth_mode, is_user_store in (
    (user_oauth_path, "oauth", True),
    (user_api_path, "api", True),
    (org_oauth_path, "oauth", False),
    (org_api_path, "api", False),
  ):
    if path is None:
      continue
    accounts = (
      load_google_accounts(path)
      if auth_mode == "oauth"
      else load_api_key_accounts(path, family="google")
    )
    for account in accounts:
      key = str(account["identity"])
      if key in seen:
        continue
      seen.add(key)
      candidates.append(PoolAccountCandidate(
        provider="google",
        key=key,
        record=account,
        store_path=path,
        is_user_store=is_user_store,
        auth_mode=auth_mode,
      ))
  for env_name, auth_mode in (
    ("GOOGLE_AUTH_TOKEN", "oauth"),
    ("GEMINI_AUTH_TOKEN", "oauth"),
    ("GEMINI_API_KEY", "api"),
    ("GOOGLE_API_KEY", "api"),
  ):
    token = str(env.get(env_name) or "").strip()
    if not token:
      continue
    record = {"api_key" if auth_mode == "api" else "access_token": token}
    digest_identity = derive_account_identity(record, prefix="google")
    key = f"env:{env_name}:{digest_identity.rsplit('-', 1)[-1]}"
    record["identity"] = key
    if key in seen:
      continue
    seen.add(key)
    candidates.append(PoolAccountCandidate(
      provider="google",
      key=key,
      record=record,
      auth_mode=auth_mode,
    ))

  def refresh(candidate: PoolAccountCandidate) -> PoolAccountCandidate:
    if (
      candidate.auth_mode == "oauth"
      and settings.client_id
      and candidate.record.get("refresh_token")
      and token_needs_refresh(candidate.record)
    ):
      candidate_settings = replace(
        settings, store_path=candidate.store_path or settings.store_path,
      )
      updated = refresh_google_tokens(candidate_settings, candidate.record)
      return replace(candidate, record=updated)
    return candidate

  return candidates, refresh


def _materialize_google_pool_candidate(
  selected: PoolAccountCandidate,
  config: Mapping[str, Any],
) -> dict[str, Any]:
  return {
    "auth_mode": selected.auth_mode,
    "auth_token": selected.token if selected.auth_mode == "oauth" else "",
    "api_key": selected.token if selected.auth_mode == "api" else "",
    "refresh_token": str(selected.record.get("refresh_token") or "").strip(),
    "token_expires_at": selected.record.get("expires_at"),
    "account_identity": selected.account_identity,
    "auth_store_path": str(selected.store_path) if selected.store_path is not None else "",
    "auth_user_scope": normalize_user_scope(config.get("auth_user_scope")),
  }


register_provider_pool_hooks(
  "google",
  resolver=resolve_google_pool_candidates,
  materializer=_materialize_google_pool_candidate,
)
