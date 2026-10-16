from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import time
from typing import Any, Mapping

import httpx

from ..auth import derive_auth_mechanism_and_entitlement
from .api_key_store import resolve_api_key_store_paths
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

MUSE_CLIENT_ID = "1031625952748946"
MUSE_DEVICE_AUTH_URL = "https://auth.meta.com/oidc/device/authorization/"
MUSE_DEVICE_TOKEN_URL = "https://auth.meta.com/oidc/device/token/"
MUSE_KEY_URL = "https://api.meta.ai/muse-code/key"
MUSE_API_VERSION = "1.0.0"


def resolve_muse_store_path(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> Path:
  return resolve_provider_store_path(
    "muse", "credentials.json", config, environ=environ, user_scope=user_scope,
    override_config_keys=("auth_store_path",), override_env_vars=("MUSE_AUTH_STORE_PATH",),
  )


def resolve_muse_store_paths(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[Path | None, Path]:
  return resolve_provider_store_paths(
    "muse", "credentials.json", config, environ=environ, user_scope=user_scope,
    override_config_keys=("auth_store_path",), override_env_vars=("MUSE_AUTH_STORE_PATH",),
  )


def load_muse_accounts(path: Path) -> list[dict[str, Any]]:
  return load_pool_accounts(path, default_prefix="muse", required_any_keys=("api_key",))


def load_muse_token_record(path: Path) -> dict[str, Any] | None:
  return accounts[0] if (accounts := load_muse_accounts(path)) else None


def upsert_muse_token_record(path: Path, record: Mapping[str, Any]) -> list[dict[str, Any]]:
  return upsert_pool_account(path, record, default_prefix="muse", required_any_keys=("api_key",))


def remove_muse_token_record(path: Path, identity: str | None = None) -> bool:
  return remove_pool_account(path, identity, default_prefix="muse", required_any_keys=("api_key",))


def save_muse_token_record(path: Path, record: Mapping[str, Any]) -> None:
  upsert_muse_token_record(path, record)


def resolve_muse_pool_candidates(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[list[PoolAccountCandidate], None]:
  env = os.environ if environ is None else environ
  subscription_paths = resolve_muse_store_paths(config, user_scope=user_scope, environ=env)
  api_paths = resolve_api_key_store_paths("muse", config, user_scope=user_scope, environ=env)
  candidates: list[PoolAccountCandidate] = []
  seen_paths: set[Path] = set()
  seen: set[str] = set()
  for index in (0, 1):
    for path in (subscription_paths[index], api_paths[index]):
      if path is None or path in seen_paths:
        continue
      seen_paths.add(path)
      for record in load_muse_accounts(path):
        key = str(record["identity"])
        if key not in seen:
          seen.add(key)
          is_user_store = index == 0
          is_sub = str(record.get("source") or "").strip().lower() == "subscription"
          cand_auth_mode = "oauth" if is_sub else "api"
          cand_mechanism, cand_entitlement = derive_auth_mechanism_and_entitlement(
            {**record, "provider": "muse"}, is_user_store=is_user_store,
          )
          candidates.append(PoolAccountCandidate(
            provider="muse", key=key, record=record,
            store_path=path, is_user_store=is_user_store,
            auth_mode=cand_auth_mode, auth_mechanism=cand_mechanism,
            entitlement_kind=cand_entitlement,
          ))
  for name in ("MUSE_API_KEY", "META_API_KEY"):
    token = str(env.get(name) or "").strip()
    if token:
      record = {"api_key": token}
      record["identity"] = f"env:{name}:{hashlib.sha256(token.encode()).hexdigest()[:12]}"
      key = record["identity"]
      if key not in seen:
        seen.add(key)
        candidates.append(PoolAccountCandidate(
          provider="muse", key=key, record=record, auth_mode="api",
          auth_mechanism="api_key", entitlement_kind="org_commercial",
        ))
  return candidates, None


def _materialize_muse_candidate(
  selected: PoolAccountCandidate, config: Mapping[str, Any],
) -> dict[str, Any]:
  materialized = dict(config)
  materialized.update(
    auth_mode=selected.auth_mode, api_key=selected.token,
    auth_mechanism=selected.auth_mechanism, entitlement_kind=selected.entitlement_kind,
    account_identity=selected.account_identity,
    auth_user_scope=normalize_user_scope(config.get("auth_user_scope")),
    auth_token="", auth_store_path=str(selected.store_path or ""), api_key_store_path="",
    refresh_token=str(selected.record.get("refresh_token") or "").strip(),
    token_expires_at=selected.record.get("expires_at"),
  )
  return materialized


register_provider_pool_hooks(
  "muse", resolver=resolve_muse_pool_candidates, materializer=_materialize_muse_candidate,
)


def request_muse_device_code(
  *,
  client: httpx.Client | None = None,
) -> dict[str, Any]:
  headers = {
    "Accept": "application/json",
    "x-api-version": MUSE_API_VERSION,
  }
  data = {
    "client_id": MUSE_CLIENT_ID,
  }

  def _post(c: httpx.Client) -> httpx.Response:
    return c.post(MUSE_DEVICE_AUTH_URL, headers=headers, data=data, timeout=30.0)

  if client is not None:
    resp = _post(client)
  else:
    with httpx.Client() as c:
      resp = _post(c)

  if resp.status_code != 200:
    raise RuntimeError(f"Muse device code request failed: {resp.status_code} {resp.text}")
  return resp.json()


def poll_muse_device_token(
  device_code: str,
  *,
  interval: float = 5.0,
  expires_in: float = 900.0,
  client: httpx.Client | None = None,
) -> dict[str, Any]:
  headers = {
    "Accept": "application/json",
    "x-api-version": MUSE_API_VERSION,
  }
  data = {
    "client_id": MUSE_CLIENT_ID,
    "device_code": device_code,
    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
  }
  deadline = time.time() + expires_in

  def _poll(c: httpx.Client) -> dict[str, Any]:
    while time.time() < deadline:
      resp = c.post(MUSE_DEVICE_TOKEN_URL, headers=headers, data=data, timeout=30.0)
      if resp.status_code == 200:
        return resp.json()
      payload = {}
      try:
        payload = resp.json()
      except Exception:
        pass
      error = payload.get("error")
      if error in ("authorization_pending", "slow_down"):
        poll_interval = interval + (5.0 if error == "slow_down" else 0.0)
        time.sleep(poll_interval)
        continue
      raise RuntimeError(f"Muse device authorization failed: {resp.status_code} {resp.text}")
    raise TimeoutError("Muse device authorization timed out")

  if client is not None:
    return _poll(client)
  with httpx.Client() as c:
    return _poll(c)


def mint_muse_api_key(
  access_token: str,
  *,
  client: httpx.Client | None = None,
) -> dict[str, Any]:
  headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "x-api-version": MUSE_API_VERSION,
  }
  body = {"onboard": True}

  def _post(c: httpx.Client) -> httpx.Response:
    return c.post(MUSE_KEY_URL, headers=headers, json=body, timeout=30.0)

  if client is not None:
    resp = _post(client)
  else:
    with httpx.Client() as c:
      resp = _post(c)

  if resp.status_code != 200:
    raise RuntimeError(f"Muse key exchange failed: {resp.status_code} {resp.text}")
  payload = resp.json()
  api_key = str(payload.get("api_key") or "").strip()
  if not api_key:
    raise RuntimeError("Muse key exchange response is missing api_key")
  return payload
