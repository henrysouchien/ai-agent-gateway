from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import httpx

from .credential_pool import (
  PoolAccountCandidate,
  load_pool_accounts,
  normalize_user_scope,
  register_provider_pool_hooks,
  remove_pool_account,
  resolve_provider_store_path,
  resolve_provider_store_paths,
  upsert_pool_account,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_COPILOT_DEVICE_CODE_URL = "https://github.com/login/device/code"
DEFAULT_COPILOT_TOKEN_URL = "https://github.com/login/oauth/access_token"
DEFAULT_COPILOT_SCOPE = "read:user"
_HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


@dataclass(frozen=True)
class CopilotOAuthSettings:
  client_id: str
  device_code_url: str
  token_url: str
  scope: str
  store_path: Path


@dataclass(frozen=True)
class CopilotDeviceCode:
  device_code: str
  user_code: str
  verification_uri: str
  expires_in: int
  interval: float


def resolve_copilot_oauth_settings(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> CopilotOAuthSettings:
  cfg = config or {}
  env = os.environ if environ is None else environ
  client_id = str(
    cfg.get("client_id")
    or cfg.get("oauth_client_id")
    or env.get("COPILOT_OAUTH_CLIENT_ID")
    or env.get("GITHUB_COPILOT_CLIENT_ID")
    or ""
  ).strip()
  device_code_url = str(
    cfg.get("device_code_url")
    or env.get("COPILOT_OAUTH_DEVICE_CODE_URL")
    or DEFAULT_COPILOT_DEVICE_CODE_URL
  ).strip()
  token_url = str(
    cfg.get("token_url")
    or env.get("COPILOT_OAUTH_TOKEN_URL")
    or DEFAULT_COPILOT_TOKEN_URL
  ).strip()
  scope = str(
    cfg.get("scope")
    or env.get("COPILOT_OAUTH_SCOPES")
    or DEFAULT_COPILOT_SCOPE
  ).strip()

  store_path = resolve_provider_store_path(
    "copilot", "oauth.json", cfg, environ=env, user_scope=user_scope,
    override_config_keys=("auth_store_path",),
    override_env_vars=("COPILOT_AUTH_STORE_PATH",),
  )

  return CopilotOAuthSettings(
    client_id=client_id,
    device_code_url=device_code_url,
    token_url=token_url,
    scope=scope,
    store_path=store_path,
  )


def resolve_copilot_store_paths(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[Path | None, Path]:
  return resolve_provider_store_paths(
    "copilot", "oauth.json", config, environ=environ, user_scope=user_scope,
    override_config_keys=("auth_store_path",),
    override_env_vars=("COPILOT_AUTH_STORE_PATH",),
  )


def load_copilot_accounts(path: Path) -> list[dict[str, Any]]:
  return load_pool_accounts(
    path, default_prefix="copilot", required_any_keys=("access_token",), strict=True,
  )


def load_copilot_token_record(path: Path) -> dict[str, Any] | None:
  return accounts[0] if (accounts := load_copilot_accounts(path)) else None


def upsert_copilot_token_record(path: Path, record: Mapping[str, Any]) -> list[dict[str, Any]]:
  return upsert_pool_account(
    path, record, default_prefix="copilot", required_any_keys=("access_token",),
  )


def remove_copilot_token_record(path: Path, identity: str | None = None) -> bool:
  return remove_pool_account(
    path, identity, default_prefix="copilot", required_any_keys=("access_token",),
  )


def save_copilot_token_record(path: Path, record: Mapping[str, Any]) -> None:
  if not str(record.get("access_token") or "").strip():
    raise ValueError("Copilot OAuth token record missing required 'access_token'")
  upsert_copilot_token_record(path, record)


def resolve_copilot_pool_candidates(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[list[PoolAccountCandidate], None]:
  env = os.environ if environ is None else environ
  user_path, org_path = resolve_copilot_store_paths(config, user_scope=user_scope, environ=env)
  candidates: list[PoolAccountCandidate] = []
  seen: set[str] = set()
  for path in (user_path, org_path):
    if path is None:
      continue
    for record in load_copilot_accounts(path):
      key = str(record["identity"])
      if key not in seen:
        seen.add(key)
        candidates.append(PoolAccountCandidate(
          provider="copilot", key=key, record=record,
          store_path=path, is_user_store=path == user_path, auth_mode="oauth",
        ))
  for name in ("COPILOT_GITHUB_TOKEN", "COPILOT_AUTH_TOKEN"):
    token = str(env.get(name) or "").strip()
    if token:
      record = {"access_token": token}
      record["identity"] = f"env:{name}:{hashlib.sha256(token.encode()).hexdigest()[:12]}"
      key = record["identity"]
      if key not in seen:
        seen.add(key)
        candidates.append(PoolAccountCandidate(
          provider="copilot", key=key, record=record, auth_mode="oauth",
        ))
  return candidates, None


def _materialize_copilot_candidate(
  selected: PoolAccountCandidate, config: Mapping[str, Any],
) -> dict[str, Any]:
  materialized = dict(config)
  materialized.update(
    auth_mode="oauth", auth_token=selected.token,
    account_identity=selected.account_identity,
    auth_user_scope=normalize_user_scope(config.get("auth_user_scope")),
    api_key="", auth_store_path=str(selected.store_path or ""), api_key_store_path="",
    refresh_token=str(selected.record.get("refresh_token") or "").strip(),
    token_expires_at=selected.record.get("expires_at"),
  )
  return materialized


register_provider_pool_hooks(
  "copilot", resolver=resolve_copilot_pool_candidates, materializer=_materialize_copilot_candidate,
)


def request_copilot_device_code(
  settings: CopilotOAuthSettings,
  *,
  client: httpx.Client | None = None,
) -> CopilotDeviceCode:
  if not settings.client_id:
    raise ValueError("COPILOT_OAUTH_CLIENT_ID must be set for Copilot device code login")
  data = {
    "client_id": settings.client_id,
    "scope": settings.scope,
  }
  headers = {"Accept": "application/json"}
  own_client = client is None
  http = client or httpx.Client(timeout=_HTTP_TIMEOUT)
  try:
    response = http.post(settings.device_code_url, data=data, headers=headers)
  finally:
    if own_client:
      http.close()

  if response.status_code != 200:
    raise RuntimeError(
      f"Copilot device code request failed ({response.status_code}): {response.text}"
    )

  payload = response.json()
  device_code = str(payload.get("device_code") or "").strip()
  user_code = str(payload.get("user_code") or "").strip()
  verification_uri = str(payload.get("verification_uri") or "").strip()
  expires_in = int(payload.get("expires_in") or 900)
  interval = float(payload.get("interval") or 5.0)

  if not device_code or not user_code or not verification_uri:
    raise RuntimeError(f"Copilot device endpoint returned incomplete payload: {payload}")

  return CopilotDeviceCode(
    device_code=device_code,
    user_code=user_code,
    verification_uri=verification_uri,
    expires_in=expires_in,
    interval=interval,
  )


def poll_copilot_device_token(
  settings: CopilotOAuthSettings,
  device: CopilotDeviceCode,
  *,
  client: httpx.Client | None = None,
  max_wait_seconds: float | None = None,
) -> dict[str, Any]:
  poll_interval = max(1.0, device.interval)
  deadline = time.time() + (max_wait_seconds if max_wait_seconds is not None else float(device.expires_in))
  data = {
    "client_id": settings.client_id,
    "device_code": device.device_code,
    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
  }
  headers = {"Accept": "application/json"}
  own_client = client is None
  http = client or httpx.Client(timeout=_HTTP_TIMEOUT)
  try:
    while time.time() < deadline:
      response = http.post(settings.token_url, data=data, headers=headers)
      if response.status_code == 200:
        payload = response.json()
        if "access_token" in payload and payload["access_token"]:
          record: dict[str, Any] = {
            "access_token": str(payload["access_token"]).strip(),
            "token_type": str(payload.get("token_type") or "bearer"),
            "scope": str(payload.get("scope") or settings.scope),
          }
          if "refresh_token" in payload and payload["refresh_token"]:
            record["refresh_token"] = str(payload["refresh_token"]).strip()
          if "expires_in" in payload and payload["expires_in"]:
            record["expires_at"] = time.time() + float(payload["expires_in"])
          save_copilot_token_record(settings.store_path, record)
          return record

        error = str(payload.get("error") or "")
        if error == "authorization_pending":
          time.sleep(poll_interval)
          continue
        if error == "slow_down":
          poll_interval += 5.0
          time.sleep(poll_interval)
          continue
        if error == "expired_token":
          raise RuntimeError("Copilot device authorization code expired; re-run login.")
        if error:
          desc = str(payload.get("error_description") or error)
          raise RuntimeError(f"Copilot device authorization failed: {desc}")
      elif response.status_code in {400, 403}:
        try:
          err_payload = response.json()
          error = str(err_payload.get("error") or "")
          if error == "authorization_pending":
            time.sleep(poll_interval)
            continue
          if error == "slow_down":
            poll_interval += 5.0
            time.sleep(poll_interval)
            continue
        except Exception:
          pass
        raise RuntimeError(f"Copilot token request failed ({response.status_code}): {response.text}")
      else:
        raise RuntimeError(f"Copilot token request failed ({response.status_code}): {response.text}")
    raise TimeoutError("Copilot device authorization polling timed out")
  finally:
    if own_client:
      http.close()
