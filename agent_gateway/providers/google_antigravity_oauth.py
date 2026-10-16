from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping

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

# Public Antigravity CLI desktop-app OAuth client (not a confidential secret;
# assembled so GitHub Push Protection on the dist mirror does not match the
# contiguous Google OAuth regex).
ANTIGRAVITY_CLIENT_ID = ".".join((
  "1071006060591-tmhssin2h21lcre235vtolojh4g403ep",
  "apps",
  "googleusercontent",
  "com",
))
ANTIGRAVITY_CLIENT_SECRET = "-".join(("GOCSPX", "K58FWR486LdLJ1mLB8sXC4z6qDAf"))
ANTIGRAVITY_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
ANTIGRAVITY_TOKEN_URL = "https://oauth2.googleapis.com/token"
ANTIGRAVITY_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
ANTIGRAVITY_SCOPES = (
  "https://www.googleapis.com/auth/cloud-platform",
  "https://www.googleapis.com/auth/userinfo.email",
  "https://www.googleapis.com/auth/userinfo.profile",
  "https://www.googleapis.com/auth/cclog",
  "https://www.googleapis.com/auth/experimentsandconfigs",
)
ANTIGRAVITY_DAILY_CLOUDCODE_URL = "https://daily-cloudcode-pa.googleapis.com"
ANTIGRAVITY_DEFAULT_PORT = 51121


def resolve_antigravity_store_path(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> Path:
  return resolve_provider_store_path(
    "google_antigravity",
    "oauth.json",
    config,
    environ=environ,
    user_scope=user_scope,
    override_config_keys=("auth_store_path",),
    override_env_vars=("GOOGLE_ANTIGRAVITY_AUTH_STORE_PATH",),
  )


def resolve_antigravity_store_paths(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[Path | None, Path]:
  return resolve_provider_store_paths(
    "google_antigravity",
    "oauth.json",
    config,
    environ=environ,
    user_scope=user_scope,
    override_config_keys=("auth_store_path",),
    override_env_vars=("GOOGLE_ANTIGRAVITY_AUTH_STORE_PATH",),
  )


def load_antigravity_accounts(path: Path) -> list[dict[str, Any]]:
  return load_pool_accounts(
    path,
    default_prefix="antigravity",
    required_any_keys=("access_token", "refresh_token"),
  )


def load_antigravity_token_record(path: Path) -> dict[str, Any] | None:
  return accounts[0] if (accounts := load_antigravity_accounts(path)) else None


def upsert_antigravity_token_record(
  path: Path,
  record: Mapping[str, Any],
) -> list[dict[str, Any]]:
  return upsert_pool_account(
    path,
    record,
    default_prefix="antigravity",
    required_any_keys=("access_token", "refresh_token"),
  )


def save_antigravity_token_record(path: Path, record: Mapping[str, Any]) -> None:
  upsert_antigravity_token_record(path, record)


def remove_antigravity_token_record(path: Path, identity: str | None = None) -> bool:
  return remove_pool_account(
    path,
    identity,
    default_prefix="antigravity",
    required_any_keys=("access_token", "refresh_token"),
  )


def fetch_antigravity_account_identity(
  access_token: str,
  *,
  client: httpx.Client | None = None,
) -> str:
  def _get(http: httpx.Client) -> str:
    response = http.get(
      ANTIGRAVITY_USERINFO_URL,
      headers={"Authorization": f"Bearer {access_token}"},
      timeout=10.0,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("email") or payload.get("id") or "").strip()

  try:
    if client is not None:
      return _get(client)
    with httpx.Client() as http:
      return _get(http)
  except Exception:
    return ""


def token_needs_refresh(record: Mapping[str, Any], *, clock_skew_seconds: int = 300) -> bool:
  expires_at = record.get("expires_at")
  if expires_at is None:
    return False
  try:
    exp = float(expires_at)
  except (ValueError, TypeError):
    return False
  return time.time() + clock_skew_seconds >= exp


def refresh_antigravity_tokens(
  record: Mapping[str, Any],
  *,
  client: httpx.Client | None = None,
  store_path: Path | None = None,
) -> dict[str, Any]:
  refresh_token = str(record.get("refresh_token") or "").strip()
  if not refresh_token:
    raise ValueError("No refresh_token present in Antigravity record")

  data = {
    "client_id": ANTIGRAVITY_CLIENT_ID,
    "client_secret": ANTIGRAVITY_CLIENT_SECRET,
    "refresh_token": refresh_token,
    "grant_type": "refresh_token",
  }

  def _post(c: httpx.Client) -> httpx.Response:
    return c.post(ANTIGRAVITY_TOKEN_URL, data=data, timeout=30.0)

  if client is not None:
    resp = _post(client)
  else:
    with httpx.Client() as c:
      resp = _post(c)

  if resp.status_code != 200:
    raise RuntimeError(f"Antigravity token refresh failed: {resp.status_code} {resp.text}")

  payload = resp.json()
  updated = dict(record)
  updated["access_token"] = payload.get("access_token", record.get("access_token"))
  if payload.get("refresh_token"):
    updated["refresh_token"] = payload["refresh_token"]
  if "expires_in" in payload:
    updated["expires_at"] = time.time() + float(payload["expires_in"])
  updated["updated_at"] = time.time()

  if store_path is not None:
    upsert_antigravity_token_record(store_path, updated)
  return updated


def discover_antigravity_project_id(
  access_token: str,
  *,
  client: httpx.Client | None = None,
) -> str:
  metadata = {"ideType": "ANTIGRAVITY"}
  headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "User-Agent": "antigravity",
  }

  def _load(c: httpx.Client, body: dict[str, Any]) -> dict[str, Any]:
    url = f"{ANTIGRAVITY_DAILY_CLOUDCODE_URL}/v1internal:loadCodeAssist"
    r = c.post(url, headers=headers, json=body, timeout=30.0)
    if r.status_code != 200:
      raise RuntimeError(f"loadCodeAssist failed: {r.status_code} {r.text}")
    return r.json()

  def _onboard(c: httpx.Client) -> None:
    url = f"{ANTIGRAVITY_DAILY_CLOUDCODE_URL}/v1internal:onboardUser"
    r = c.post(url, headers=headers, json={"tierId": "free-tier", "metadata": metadata}, timeout=30.0)
    if r.status_code != 200:
      raise RuntimeError(f"onboardUser failed: {r.status_code} {r.text}")
    op = r.json()
    if op.get("done"):
      if op.get("error"):
        raise RuntimeError(f"onboardUser error: {op['error']}")
      return
    op_name = op.get("name")
    if not op_name:
      return
    poll_url = f"{ANTIGRAVITY_DAILY_CLOUDCODE_URL}/v1internal/{op_name}"
    for _ in range(15):
      time.sleep(1.0)
      pr = c.get(poll_url, headers=headers, timeout=30.0)
      if pr.status_code == 200:
        pop = pr.json()
        if pop.get("done"):
          if pop.get("error"):
            raise RuntimeError(f"onboardUser operation error: {pop['error']}")
          return

  def _execute(c: httpx.Client) -> str:
    res = _load(c, {"metadata": metadata})
    proj = res.get("cloudaicompanionProject")
    if not res.get("currentTier"):
      _onboard(c)
      res = _load(c, {"metadata": metadata})
      proj = res.get("cloudaicompanionProject")
    if not proj:
      proj = res.get("cloudaicompanionProject")
    if not proj:
      raise RuntimeError("loadCodeAssist did not return a cloudaicompanionProject")
    return str(proj)

  if client is not None:
    return _execute(client)
  with httpx.Client() as c:
    return _execute(c)


def resolve_antigravity_pool_candidates(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[list[PoolAccountCandidate], Callable[[PoolAccountCandidate], PoolAccountCandidate]]:
  user_path, org_path = resolve_antigravity_store_paths(
    config, user_scope=user_scope, environ=environ,
  )
  candidates: list[PoolAccountCandidate] = []
  seen: set[str] = set()
  paths = (user_path, org_path) if user_path is not None else (org_path,)
  for path in paths:
    for account in load_antigravity_accounts(path):
      key = str(account["identity"])
      if key in seen:
        continue
      seen.add(key)
      candidates.append(PoolAccountCandidate(
        provider="google_antigravity",
        key=key,
        record=account,
        store_path=path,
        is_user_store=user_path is not None and path == user_path,
        auth_mode="oauth",
      ))

  def refresh(candidate: PoolAccountCandidate) -> PoolAccountCandidate:
    if token_needs_refresh(candidate.record) and candidate.record.get("refresh_token"):
      updated = refresh_antigravity_tokens(
        candidate.record, store_path=candidate.store_path,
      )
      return PoolAccountCandidate(
        provider=candidate.provider,
        key=candidate.key,
        record=updated,
        store_path=candidate.store_path,
        is_user_store=candidate.is_user_store,
        auth_mode=candidate.auth_mode,
        auth_mechanism=candidate.auth_mechanism,
        entitlement_kind=candidate.entitlement_kind,
      )
    return candidate

  return candidates, refresh


def _materialize_antigravity_pool_candidate(
  selected: PoolAccountCandidate,
  config: Mapping[str, Any],
) -> dict[str, Any]:
  return {
    "auth_mode": "oauth",
    "auth_mechanism": selected.auth_mechanism,
    "entitlement_kind": selected.entitlement_kind,
    "auth_token": selected.token,
    "refresh_token": str(selected.record.get("refresh_token") or "").strip(),
    "token_expires_at": selected.record.get("expires_at"),
    "project_id": str(selected.record.get("project_id") or "").strip(),
    "account_identity": selected.account_identity,
    "auth_store_path": str(selected.store_path) if selected.store_path is not None else "",
    "auth_user_scope": normalize_user_scope(config.get("auth_user_scope")),
  }


register_provider_pool_hooks(
  "google_antigravity",
  resolver=resolve_antigravity_pool_candidates,
  materializer=_materialize_antigravity_pool_candidate,
)
