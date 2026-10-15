"""Kimi Code OAuth device-code flow, common headers, and multi-account pool integration."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import socket
import time
from typing import Any, Callable, Mapping
import uuid

import httpx

from .api_key_store import load_api_key_accounts, resolve_api_key_store_paths
from .credential_pool import (
  ACCOUNT_IDENTITY_FIELD,
  USER_SCOPE_FIELD,
  PoolAccountCandidate,
  derive_account_identity,
  load_pool_accounts,
  normalize_user_scope,
  register_provider_pool_hooks,
  remove_pool_account,
  resolve_provider_store_path,
  resolve_provider_store_paths,
  store_base_dir,
  store_write_lock,
  upsert_pool_account,
  upsert_pool_account_unlocked,
)

LOGGER = logging.getLogger(__name__)

KIMI_CODE_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_KIMI_CODE_OAUTH_HOST = "https://auth.kimi.com"
DEFAULT_KIMI_CODE_CLI_VERSION = "1.0"
KIMI_CODE_DEVICE_ID_FILENAME = "device_id"
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_DEVICE_FLOW_TTL_SECONDS = 900.0
EXPIRY_MARGIN_SECONDS = 300
_HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

_CACHED_DEVICE_ID: str | None = None


def _sanitize_header_value(value: str, fallback: str = "unknown") -> str:
  cleaned = "".join(ch for ch in str(value or "") if 0x20 <= ord(ch) <= 0x7E).strip()
  return cleaned or fallback


def _device_model() -> str:
  sys_name = platform.system()
  label = "macOS" if sys_name == "Darwin" else ("Windows" if sys_name == "Windows" else sys_name)
  parts = [label, platform.release(), platform.machine()]
  return " ".join(p.strip() for p in parts if p and p.strip())


def get_kimi_device_id(environ: Mapping[str, str] | None = None) -> str:
  """Return a stable per-install device ID, falling back to an ephemeral ID if unwritable."""
  global _CACHED_DEVICE_ID
  if _CACHED_DEVICE_ID and environ is None:
    return _CACHED_DEVICE_ID
  device_path = store_base_dir(environ) / "kimi_code" / KIMI_CODE_DEVICE_ID_FILENAME
  try:
    if device_path.is_file():
      existing = device_path.read_text(encoding="utf-8").strip()
      if existing:
        if environ is None:
          _CACHED_DEVICE_ID = existing
        return existing
  except OSError:
    pass
  generated = uuid.uuid4().hex
  try:
    device_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    device_path.write_text(f"{generated}\n", encoding="utf-8")
    device_path.chmod(0o600)
  except OSError:
    pass
  if environ is None:
    _CACHED_DEVICE_ID = generated
  return generated


def kimi_common_headers(environ: Mapping[str, str] | None = None) -> dict[str, str]:
  """Client attribution headers required by Kimi Code OAuth and inference endpoints."""
  try:
    hostname = socket.gethostname()
  except OSError:
    hostname = "unknown"
  return {
    "User-Agent": f"KimiCLI/{DEFAULT_KIMI_CODE_CLI_VERSION}",
    "X-Msh-Platform": "kimi_cli",
    "X-Msh-Version": DEFAULT_KIMI_CODE_CLI_VERSION,
    "X-Msh-Device-Name": _sanitize_header_value(hostname, "unknown"),
    "X-Msh-Device-Model": _sanitize_header_value(_device_model(), "unknown"),
    "X-Msh-Os-Version": _sanitize_header_value(platform.version(), "unknown"),
    "X-Msh-Device-Id": _sanitize_header_value(get_kimi_device_id(environ), "unknown"),
  }


def resolve_kimi_code_oauth_host(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
) -> str:
  cfg = config or {}
  env = os.environ if environ is None else environ
  host = str(
    cfg.get("oauth_host")
    or env.get("KIMI_CODE_OAUTH_HOST")
    or env.get("KIMI_OAUTH_HOST")
    or DEFAULT_KIMI_CODE_OAUTH_HOST
  ).strip()
  return host.rstrip("/") or DEFAULT_KIMI_CODE_OAUTH_HOST


def resolve_kimi_code_client_id(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
) -> str:
  cfg = config or {}
  env = os.environ if environ is None else environ
  return str(
    cfg.get("client_id")
    or cfg.get("oauth_client_id")
    or env.get("KIMI_CODE_OAUTH_CLIENT_ID")
    or KIMI_CODE_CLIENT_ID
  ).strip() or KIMI_CODE_CLIENT_ID


def extract_kimi_account_identity(access_token: str) -> str:
  """Extract account identity (`email`, `user_id`, or `sub`) from a JWT access token when present."""
  parts = str(access_token or "").strip().split(".")
  if len(parts) != 3 or not parts[1]:
    return ""
  segment = parts[1]
  padding = "=" * ((4 - len(segment) % 4) % 4)
  try:
    decoded = base64.urlsafe_b64decode((segment + padding).encode("ascii"))
    payload = json.loads(decoded.decode("utf-8"))
  except Exception:
    return ""
  if not isinstance(payload, Mapping):
    return ""
  for field in ("email", "user_email", "user_id", "sub", "account_id"):
    value = str(payload.get(field) or "").strip()
    if value:
      return value
  return ""


def resolve_kimi_code_store_path(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> Path:
  return resolve_provider_store_path(
    "kimi_code",
    "oauth.json",
    config,
    environ=environ,
    user_scope=user_scope,
    override_config_keys=("auth_store_path",),
    override_env_vars=("KIMI_CODE_AUTH_STORE_PATH", "KIMI_AUTH_STORE_PATH"),
  )


def resolve_kimi_code_store_paths(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[Path | None, Path]:
  return resolve_provider_store_paths(
    "kimi_code",
    "oauth.json",
    config,
    environ=environ,
    user_scope=user_scope,
    override_config_keys=("auth_store_path",),
    override_env_vars=("KIMI_CODE_AUTH_STORE_PATH", "KIMI_AUTH_STORE_PATH"),
  )


def load_kimi_code_accounts(path: Path) -> list[dict[str, Any]]:
  return load_pool_accounts(
    path,
    default_prefix="kimi_code",
    required_any_keys=("access_token", "refresh_token"),
  )


def load_kimi_code_token_record(path: Path) -> dict[str, Any] | None:
  accounts = load_kimi_code_accounts(path)
  return accounts[0] if accounts else None


def _kimi_code_entry(record: Mapping[str, Any]) -> dict[str, Any]:
  entry = dict(record)
  if not any(str(entry.get(k) or "").strip() for k in ("identity", "email", "account_id", "user_id")):
    jwt_identity = extract_kimi_account_identity(str(entry.get("access_token") or ""))
    if jwt_identity:
      entry["identity"] = jwt_identity
  entry["identity"] = derive_account_identity(
    entry,
    prefix="kimi_code",
    material_keys=("refresh_token", "access_token"),
  )
  return entry


def upsert_kimi_code_token_record(
  path: Path,
  record: Mapping[str, Any],
) -> list[dict[str, Any]]:
  return upsert_pool_account(
    path,
    _kimi_code_entry(record),
    default_prefix="kimi_code",
    required_any_keys=("access_token", "refresh_token"),
  )


def save_kimi_code_token_record(path: Path, record: Mapping[str, Any]) -> None:
  upsert_kimi_code_token_record(path, record)


def remove_kimi_code_token_record(
  path: Path,
  identity: str | None = None,
) -> bool:
  return remove_pool_account(
    path,
    identity,
    default_prefix="kimi_code",
    required_any_keys=("access_token", "refresh_token"),
  )


def token_needs_refresh(
  record: Mapping[str, Any],
  *,
  clock_skew_seconds: int = EXPIRY_MARGIN_SECONDS,
) -> bool:
  expires_at = record.get("expires_at")
  if expires_at is None:
    return False
  try:
    exp = float(expires_at)
  except (TypeError, ValueError):
    return False
  return time.time() + clock_skew_seconds >= exp


def request_kimi_code_device_code(
  *,
  config: Mapping[str, Any] | None = None,
  environ: Mapping[str, str] | None = None,
  client: httpx.Client | None = None,
) -> dict[str, Any]:
  oauth_host = resolve_kimi_code_oauth_host(config, environ=environ)
  client_id = resolve_kimi_code_client_id(config, environ=environ)
  headers = {
    "Accept": "application/json",
    **kimi_common_headers(environ),
  }
  data = {"client_id": client_id}

  def _post(c: httpx.Client) -> httpx.Response:
    return c.post(
      f"{oauth_host}/api/oauth/device_authorization",
      headers=headers,
      data=data,
      timeout=_HTTP_TIMEOUT,
    )

  resp = _post(client) if client is not None else httpx.Client().post(
    f"{oauth_host}/api/oauth/device_authorization",
    headers=headers,
    data=data,
    timeout=_HTTP_TIMEOUT,
  )
  if resp.status_code != 200:
    raise RuntimeError(f"Kimi Code device authorization failed: {resp.status_code} {resp.text}")
  payload = resp.json()
  if not isinstance(payload, dict):
    raise RuntimeError("Kimi Code device authorization returned non-object JSON")
  user_code = str(payload.get("user_code") or "").strip()
  device_code = str(payload.get("device_code") or "").strip()
  verification_uri = str(payload.get("verification_uri") or "").strip()
  if not user_code or not device_code or not verification_uri:
    raise RuntimeError("Kimi Code device authorization response missing required fields")
  return payload


def _parse_kimi_token_payload(
  payload: Mapping[str, Any],
  *,
  refresh_token_fallback: str = "",
  existing_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  access_token = str(payload.get("access_token") or "").strip()
  if not access_token:
    raise RuntimeError("Kimi Code token response missing access_token")
  refresh_token = str(payload.get("refresh_token") or refresh_token_fallback or "").strip()
  if not refresh_token:
    raise RuntimeError("Kimi Code token response missing refresh_token")
  expires_in_raw = payload.get("expires_in")
  now = time.time()
  try:
    expires_at = now + float(expires_in_raw) if expires_in_raw is not None else None
  except (TypeError, ValueError):
    expires_at = None

  record: dict[str, Any] = dict(existing_record or {})
  record.update({
    "access_token": access_token,
    "refresh_token": refresh_token,
    "token_type": str(payload.get("token_type") or record.get("token_type") or "Bearer"),
    "scope": str(payload.get("scope") or record.get("scope") or ""),
    "source": str(record.get("source") or "subscription"),
    "updated_at": now,
  })
  if expires_at is not None:
    record["expires_at"] = expires_at
  jwt_id = extract_kimi_account_identity(access_token)
  if jwt_id and not str(record.get("identity") or "").strip():
    record["identity"] = jwt_id
    if "@" in jwt_id:
      record.setdefault("email", jwt_id)
  record["identity"] = derive_account_identity(
    record,
    prefix="kimi_code",
    material_keys=("refresh_token", "access_token"),
  )
  return record


def poll_kimi_code_device_token(
  device_code: str,
  *,
  interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
  expires_in: float = DEFAULT_DEVICE_FLOW_TTL_SECONDS,
  config: Mapping[str, Any] | None = None,
  environ: Mapping[str, str] | None = None,
  client: httpx.Client | None = None,
  sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
  oauth_host = resolve_kimi_code_oauth_host(config, environ=environ)
  client_id = resolve_kimi_code_client_id(config, environ=environ)
  headers = {
    "Accept": "application/json",
    **kimi_common_headers(environ),
  }
  data = {
    "client_id": client_id,
    "device_code": device_code,
    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
  }
  deadline = time.time() + float(expires_in)
  wait_seconds = max(1.0, float(interval))

  def _poll_loop(c: httpx.Client) -> dict[str, Any]:
    nonlocal wait_seconds
    while time.time() < deadline:
      resp = c.post(
        f"{oauth_host}/api/oauth/token",
        headers=headers,
        data=data,
        timeout=_HTTP_TIMEOUT,
      )
      payload: dict[str, Any] = {}
      try:
        parsed = resp.json()
        if isinstance(parsed, dict):
          payload = parsed
      except Exception:
        pass
      if resp.status_code == 200 and payload.get("access_token"):
        return _parse_kimi_token_payload(payload)
      error = str(payload.get("error") or "").strip()
      if error == "authorization_pending":
        sleep(wait_seconds)
        continue
      if error == "slow_down":
        wait_seconds += 5.0
        retry_interval = payload.get("interval")
        if isinstance(retry_interval, (int, float)) and float(retry_interval) > wait_seconds:
          wait_seconds = float(retry_interval)
        sleep(wait_seconds)
        continue
      if error == "expired_token":
        raise RuntimeError("Kimi Code device authorization expired")
      if error in {"access_denied", "authorization_denied"}:
        raise RuntimeError("Kimi Code device authorization denied")
      desc = str(payload.get("error_description") or resp.text or "").strip()
      detail = f": {desc}" if desc else ""
      raise RuntimeError(f"Kimi Code device token poll failed ({error or resp.status_code}){detail}")
    raise TimeoutError("Kimi Code device authorization timed out")

  if client is not None:
    return _poll_loop(client)
  with httpx.Client() as c:
    return _poll_loop(c)


def refresh_kimi_code_tokens(
  record: Mapping[str, Any],
  *,
  store_path: Path | None = None,
  config: Mapping[str, Any] | None = None,
  environ: Mapping[str, str] | None = None,
  client: httpx.Client | None = None,
) -> dict[str, Any]:
  """Exchange `record`'s refresh token and persist the result to `store_path`.

  Kimi refresh grants are single-use, and one store may be shared in place by
  several processes (an isolated gateway pointed at the host store via
  `KIMI_CODE_AUTH_STORE_PATH`). With a `store_path`, the read-POST-write runs
  under the store's write lock; a seat another process already refreshed is
  returned as stored, without a second POST of the spent token.
  """
  if not str(record.get("refresh_token") or "").strip():
    raise ValueError("No refresh_token present in Kimi Code credential record")
  oauth_host = resolve_kimi_code_oauth_host(config, environ=environ)
  client_id = resolve_kimi_code_client_id(config, environ=environ)
  headers = {
    "Accept": "application/json",
    **kimi_common_headers(environ),
  }

  def _exchange(base: Mapping[str, Any]) -> dict[str, Any]:
    refresh_token = str(base.get("refresh_token") or "").strip()
    data = {
      "grant_type": "refresh_token",
      "refresh_token": refresh_token,
      "client_id": client_id,
    }

    def _post(c: httpx.Client) -> httpx.Response:
      return c.post(
        f"{oauth_host}/api/oauth/token",
        headers=headers,
        data=data,
        timeout=_HTTP_TIMEOUT,
      )

    if client is not None:
      resp = _post(client)
    else:
      with httpx.Client() as c:
        resp = _post(c)

    if resp.status_code != 200:
      raise RuntimeError(f"Kimi Code token refresh failed: {resp.status_code} {resp.text}")
    payload = resp.json()
    if not isinstance(payload, dict):
      raise RuntimeError("Kimi Code token refresh returned non-object JSON")
    return _parse_kimi_token_payload(
      payload,
      refresh_token_fallback=refresh_token,
      existing_record=base,
    )

  if store_path is None:
    return _exchange(record)

  identity = _kimi_code_entry(record)["identity"]
  with store_write_lock(store_path):
    # Reads under our own exclusive lock fall back to the atomic file snapshot
    # (`store_read_lock` never blocks), so they cannot self-deadlock.
    current = next(
      (a for a in load_kimi_code_accounts(store_path) if a.get("identity") == identity),
      None,
    )
    base: Mapping[str, Any] = record
    if current is not None and str(current.get("refresh_token") or "").strip():
      if (
        current["refresh_token"] != record.get("refresh_token")
        and str(current.get("access_token") or "").strip()
        and not token_needs_refresh(current)
      ):
        return current
      base = current
    updated = _exchange(base)
    upsert_pool_account_unlocked(
      store_path,
      _kimi_code_entry(updated),
      default_prefix="kimi_code",
      required_any_keys=("access_token", "refresh_token"),
    )
    return updated


def resolve_kimi_code_pool_candidates(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[
  list[PoolAccountCandidate],
  Callable[[PoolAccountCandidate], PoolAccountCandidate],
]:
  cfg = config or {}
  env = os.environ if environ is None else environ
  oauth_user_path, oauth_org_path = resolve_kimi_code_store_paths(
    cfg,
    user_scope=user_scope,
    environ=env,
  )
  api_user_path, api_org_path = resolve_api_key_store_paths(
    "kimi_code",
    cfg,
    user_scope=user_scope,
    environ=env,
  )

  candidates: list[PoolAccountCandidate] = []
  seen: set[str] = set()

  def _add_candidate(candidate: PoolAccountCandidate) -> None:
    if candidate.key in seen:
      return
    seen.add(candidate.key)
    candidates.append(candidate)

  for is_user, oauth_path, api_path in (
    (True, oauth_user_path, api_user_path),
    (False, oauth_org_path, api_org_path),
  ):
    if oauth_path is not None:
      for record in load_kimi_code_accounts(oauth_path):
        if not str(record.get("access_token") or "").strip():
          continue
        key = str(record["identity"])
        _add_candidate(
          PoolAccountCandidate(
            provider="kimi_code",
            key=key,
            record=record,
            store_path=oauth_path,
            is_user_store=is_user,
            auth_mode="oauth",
          )
        )
    if api_path is not None:
      for record in load_api_key_accounts(api_path, family="kimi_code"):
        if not str(record.get("api_key") or "").strip():
          continue
        key = str(record["identity"])
        _add_candidate(
          PoolAccountCandidate(
            provider="kimi_code",
            key=key,
            record=record,
            store_path=api_path,
            is_user_store=is_user,
            auth_mode="api",
          )
        )

  for auth_mode, env_name in (
    ("oauth", "KIMI_CODE_AUTH_TOKEN"),
    ("api", "KIMI_CODE_API_KEY"),
    ("api", "KIMI_API_KEY"),
  ):
    token = str(env.get(env_name) or "").strip()
    if not token:
      continue
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    key = f"env:{env_name}:{digest}"
    material_field = "access_token" if auth_mode == "oauth" else "api_key"
    _add_candidate(
      PoolAccountCandidate(
        provider="kimi_code",
        key=key,
        record={"identity": key, material_field: token, "source": f"env:{env_name}"},
        store_path=None,
        is_user_store=False,
        auth_mode=auth_mode,
      )
    )

  def _refresh_candidate(candidate: PoolAccountCandidate) -> PoolAccountCandidate:
    if candidate.auth_mode != "oauth":
      return candidate
    if not token_needs_refresh(candidate.record) or not candidate.record.get("refresh_token"):
      return candidate
    refreshed = refresh_kimi_code_tokens(
      candidate.record,
      store_path=candidate.store_path,
      config=cfg,
      environ=env,
    )
    return PoolAccountCandidate(
      provider=candidate.provider,
      key=candidate.key,
      record=refreshed,
      store_path=candidate.store_path,
      is_user_store=candidate.is_user_store,
      auth_mode=candidate.auth_mode,
    )

  return candidates, _refresh_candidate


def _materialize_kimi_code_candidate(
  selected: PoolAccountCandidate,
  config: Mapping[str, Any],
) -> dict[str, Any]:
  is_api = selected.auth_mode == "api"
  store_path_str = str(selected.store_path) if selected.store_path is not None else ""
  prev_auth_store = str(config.get("auth_store_path") or "").strip()
  prev_api_store = str(config.get("api_key_store_path") or "").strip()
  auth_store_path = (
    (prev_auth_store if prev_auth_store != store_path_str else "")
    if is_api
    else store_path_str
  )
  api_key_store_path = (
    store_path_str
    if is_api
    else (prev_api_store if prev_api_store != store_path_str else "")
  )
  materialized = dict(config)
  materialized.update({
    "provider": "kimi_code",
    "auth_mode": "api" if is_api else "oauth",
    "api_key": selected.token if is_api else "",
    "auth_token": "" if is_api else selected.token,
    ACCOUNT_IDENTITY_FIELD: selected.account_identity,
    USER_SCOPE_FIELD: normalize_user_scope(config.get(USER_SCOPE_FIELD)),
    "auth_store_path": auth_store_path,
    "api_key_store_path": api_key_store_path,
    "refresh_token": str(selected.record.get("refresh_token") or "").strip(),
    "token_expires_at": selected.record.get("expires_at"),
  })
  return materialized


register_provider_pool_hooks(
  "kimi_code",
  resolver=resolve_kimi_code_pool_candidates,
  materializer=_materialize_kimi_code_candidate,
)


__all__ = [
  "DEFAULT_KIMI_CODE_CLI_VERSION",
  "DEFAULT_KIMI_CODE_OAUTH_HOST",
  "DEFAULT_DEVICE_FLOW_TTL_SECONDS",
  "DEFAULT_POLL_INTERVAL_SECONDS",
  "EXPIRY_MARGIN_SECONDS",
  "KIMI_CODE_CLIENT_ID",
  "KIMI_CODE_DEVICE_ID_FILENAME",
  "extract_kimi_account_identity",
  "get_kimi_device_id",
  "kimi_common_headers",
  "load_kimi_code_accounts",
  "load_kimi_code_token_record",
  "poll_kimi_code_device_token",
  "refresh_kimi_code_tokens",
  "remove_kimi_code_token_record",
  "request_kimi_code_device_code",
  "resolve_kimi_code_client_id",
  "resolve_kimi_code_oauth_host",
  "resolve_kimi_code_pool_candidates",
  "resolve_kimi_code_store_path",
  "resolve_kimi_code_store_paths",
  "save_kimi_code_token_record",
  "token_needs_refresh",
  "upsert_kimi_code_token_record",
]
