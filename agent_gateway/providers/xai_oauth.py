from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Awaitable, Callable, Mapping, TypeAlias
from urllib.parse import urlparse

import httpx

from .api_key_store import resolve_api_key_store_paths
from .credential_pool import (
  PoolAccountCandidate,
  _load_pool_accounts_unlocked,
  derive_account_identity,
  load_pool_accounts,
  normalize_user_scope,
  register_provider_pool_hooks,
  resolve_provider_store_path,
  resolve_provider_store_paths,
  write_pool_accounts_unlocked,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_XAI_OAUTH_ISSUER = "https://auth.x.ai"
DEFAULT_XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
DEFAULT_XAI_OAUTH_SCOPE = "openid profile email offline_access grok-cli:access api:access"
DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
DEFAULT_DEVICE_TIMEOUT_SECONDS = 300
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
MIN_POLL_INTERVAL_SECONDS = 1.0
SLOW_DOWN_INCREMENT_SECONDS = 5.0
EXPIRY_MARGIN_SECONDS = 60
_SAVE_ATTEMPTS = 3
_SAVE_RETRY_DELAY_SECONDS = 0.01
# A refresh POST whose answer was lost may already have consumed the single-use
# refresh token, so one retry on a fresh connection is allowed, but only inside
# this many seconds of the first send: past it a hung token endpoint is a failed
# refresh the caller surfaces (re-login), and a retry with under the minimum
# budget left cannot complete a round trip.
_GRACE_HARD_BUDGET_S = 5.0
_MIN_RETRY_BUDGET_S = 1.0
# Bounds a hung issuer endpoint (discovery, device-code, token) so a login or
# refresh cannot wait on it for good; the device-code poll's own expiry is the
# issuer's `expires_in`, not this.
_HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_PRESEND_EXC = (
  httpx.ConnectError,
  httpx.ConnectTimeout,
  httpx.PoolTimeout,
  httpx.UnsupportedProtocol,
  httpx.ProxyError,
)
_STORE_REFRESH_CONSUMED: set[str] = set()
_REAUTH_REQUIRED_MESSAGE = (
  "xAI OAuth refresh did not complete previously; the refresh token may be invalidated — "
  "run `python3 -m agent_gateway.cli auth login xai`."
)
JSONValue: TypeAlias = bool | int | float | str | None | list["JSONValue"] | dict[str, "JSONValue"]
XAIOAuthTokenRecord: TypeAlias = dict[str, JSONValue]



@dataclass(frozen=True)
class XAIOAuthSettings:
  issuer: str
  client_id: str
  scope: str
  discovery_url: str
  store_path: Path


@dataclass(frozen=True)
class XAIDeviceCode:
  device_code: str
  user_code: str
  verification_uri: str
  verification_uri_complete: str | None
  expires_in: int
  interval: float


VerificationCallback = Callable[[XAIDeviceCode], Awaitable[None] | None]


def resolve_xai_oauth_settings(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
  user_scope: Any = None,
) -> XAIOAuthSettings:
  cfg = config or {}
  env = os.environ if environ is None else environ
  issuer = str(cfg.get("oauth_issuer") or env.get("XAI_OAUTH_ISSUER") or DEFAULT_XAI_OAUTH_ISSUER).strip().rstrip("/")
  client_id = str(cfg.get("oauth_client_id") or env.get("XAI_OAUTH_CLIENT_ID") or DEFAULT_XAI_OAUTH_CLIENT_ID).strip()
  scope = str(cfg.get("oauth_scope") or env.get("XAI_OAUTH_SCOPES") or DEFAULT_XAI_OAUTH_SCOPE).strip()
  discovery_url = str(
    cfg.get("oauth_discovery_url")
    or env.get("XAI_OAUTH_DISCOVERY_URL")
    or f"{issuer}/.well-known/openid-configuration"
  ).strip()
  store_path = resolve_provider_store_path(
    "xai", "oauth.json", cfg, environ=env, user_scope=user_scope,
    override_config_keys=("auth_store_path",), override_env_vars=("XAI_AUTH_STORE_PATH",),
  )
  if not issuer or not client_id or not scope:
    raise ValueError("xAI OAuth issuer, client ID, and scopes must not be blank")
  _require_trusted_endpoint(discovery_url, issuer=issuer, label="discovery URL")
  return XAIOAuthSettings(issuer, client_id, scope, discovery_url, store_path)


def resolve_xai_store_paths(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
  user_scope: Any = None,
) -> tuple[Path | None, Path]:
  return resolve_provider_store_paths(
    "xai", "oauth.json", config, environ=environ, user_scope=user_scope,
    override_config_keys=("auth_store_path",), override_env_vars=("XAI_AUTH_STORE_PATH",),
  )


def load_xai_accounts(path: Path) -> list[dict[str, Any]]:
  return load_pool_accounts(
    path, default_prefix="xai", required_any_keys=("refresh_token", "access_token"), strict=True,
  )


def _selected_xai_record(
  stored: Mapping[str, Any] | None, identity: str,
) -> dict[str, Any] | None:
  if stored is None:
    return None
  accounts = stored.get("accounts")
  if isinstance(accounts, list):
    if identity:
      return next(
        (dict(account) for account in accounts
         if isinstance(account, Mapping) and str(account.get("identity") or "") == identity),
        None,
      )
    # No bound identity: the first account is the seat a fresh read serves. Its
    # sibling's fields (a quarantine marker, another refresh grant) never bleed in.
    first = next((account for account in accounts if isinstance(account, Mapping)), None)
    return dict(first) if first is not None else None
  return dict(stored)


def resolve_xai_auth_mode(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
) -> str:
  cfg = config or {}
  env = os.environ if environ is None else environ
  explicit = str(cfg.get("auth_mode") or env.get("XAI_AUTH_MODE") or "").strip().lower()
  if explicit:
    if explicit not in {"api", "oauth"}:
      raise ValueError(f"Unknown XAI_AUTH_MODE={explicit!r}. Expected 'api' or 'oauth'.")
    return explicit
  record = load_xai_token_record(resolve_xai_oauth_settings(cfg, environ=env).store_path)
  return "oauth" if record and str(record.get("refresh_token") or "").strip() else "api"


def _load_xai_raw_store(path: Path) -> dict[str, Any] | None:
  try:
    raw = path.read_text(encoding="utf-8")
  except FileNotFoundError:
    return None
  except OSError as exc:
    raise RuntimeError(f"Unable to read xAI OAuth token store: {path}") from exc
  try:
    parsed = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise RuntimeError(f"Invalid xAI OAuth token store JSON: {path}") from exc
  if not isinstance(parsed, dict):
    raise RuntimeError(f"Invalid xAI OAuth token store payload: {path}")
  return dict(parsed)


def load_xai_token_record(path: Path) -> XAIOAuthTokenRecord | None:
  parsed = _load_xai_raw_store(path)
  if parsed is None:
    return None
  if isinstance(parsed.get("accounts"), list):
    return {
      key: value
      for key, value in parsed.items()
      if key not in {"accounts", "version", "identity"}
    }
  return parsed


def upsert_xai_token_record(path: Path, record: Mapping[str, Any]) -> list[dict[str, Any]]:
  required = ("access_token", "refresh_token", "expires_at", "scope", "issuer", "client_id")
  missing = [key for key in required if record.get(key) in {None, ""}]
  if missing:
    raise ValueError(f"xAI OAuth token record is missing: {', '.join(missing)}")
  accounts = _load_pool_accounts_unlocked(
    path, default_prefix="xai", required_any_keys=("refresh_token", "access_token"),
  )
  entry = {key: value for key, value in record.items() if key not in {"accounts", "version"}}
  entry["identity"] = derive_account_identity(entry, prefix="xai")
  for index, existing in enumerate(accounts):
    if existing["identity"] == entry["identity"]:
      accounts[index] = entry
      break
  else:
    accounts.append(entry)
  write_pool_accounts_unlocked(path, accounts, extra_top_level=entry)
  return accounts


def save_xai_token_record(path: Path, record: Mapping[str, Any]) -> None:
  required = ("access_token", "refresh_token", "expires_at", "scope", "issuer", "client_id")
  missing = [key for key in required if record.get(key) in {None, ""}]
  if missing:
    raise ValueError(f"xAI OAuth token record is missing: {', '.join(missing)}")
  existing_raw = _load_xai_raw_store(path) if path.exists() else None
  if record.get("identity"):
    upsert_xai_token_record(path, record)
    return
  if isinstance((existing_raw or {}).get("accounts"), list):
    accounts = _load_pool_accounts_unlocked(
      path, default_prefix="xai", required_any_keys=("refresh_token", "access_token"),
    )
    entry = {key: value for key, value in record.items() if key not in {"accounts", "version"}}
    target_idx = next(
      (
        i for i, acc in enumerate(accounts)
        if acc.get("refresh_token") == entry.get("refresh_token")
        or acc.get("refresh_pending") is True
      ),
      0 if accounts else None,
    )
    if target_idx is not None and target_idx < len(accounts):
      entry["identity"] = accounts[target_idx]["identity"]
      accounts[target_idx] = entry
    else:
      entry["identity"] = derive_account_identity(entry, prefix="xai")
      accounts.append(entry)
    write_pool_accounts_unlocked(path, accounts, extra_top_level=entry)
    return
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  try:
    path.parent.chmod(0o700)
  except OSError:
    pass
  temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
  fd = os.open(temp_path, flags, 0o600)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(dict(record), handle, indent=2, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, path)
    os.chmod(path, 0o600)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    try:
      temp_path.unlink()
    except FileNotFoundError:
      pass

def _save_token_record_durable(
  path: Path,
  record: Mapping[str, Any],
  attempts: int = _SAVE_ATTEMPTS,
) -> None:
  for attempt in range(attempts):
    try:
      save_xai_token_record(path, record)
      return
    except OSError:
      if attempt + 1 >= attempts:
        raise
      time.sleep(_SAVE_RETRY_DELAY_SECONDS * (attempt + 1))


def _write_reauth_tombstone(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  try:
    path.parent.chmod(0o700)
  except OSError:
    pass
  temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
  fd = os.open(temp_path, flags, 0o600)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump({"reauth_required": True}, handle, indent=2, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, path)
    os.chmod(path, 0o600)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    try:
      temp_path.unlink()
    except FileNotFoundError:
      pass


def token_store_is_private(path: Path) -> bool:
  try:
    return stat.S_IMODE(path.stat().st_mode) == 0o600
  except OSError:
    return False


@asynccontextmanager
async def _store_lock(store_path: Path):
  canonical_store = Path(os.path.realpath(store_path.expanduser()))
  canonical_store.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  try:
    canonical_store.parent.chmod(0o700)
  except OSError:
    pass
  # Same lockfile convention as credential_pool.store_write_lock, so removal,
  # enrollment and refresh serialize on the store: `.oauth.json.lock`, not
  # `oauth.json.lock`.
  lock_path = canonical_store.with_name(f".{canonical_store.name}.lock")
  flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
  fd = os.open(lock_path, flags, 0o600)
  acquired = False
  try:
    while True:
      try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        break
      except BlockingIOError:
        await asyncio.sleep(0.05)
    yield
  finally:
    if acquired:
      fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _record_is_valid(rec: Mapping[str, Any], settings: XAIOAuthSettings) -> bool:
  try:
    access_token = rec.get("access_token")
    refresh_token = rec.get("refresh_token")
    if not isinstance(access_token, str) or not access_token.strip():
      return False
    if not isinstance(refresh_token, str) or not refresh_token.strip():
      return False
    expires_at = float(rec["expires_at"])
    if not math.isfinite(expires_at) or expires_at <= 0:
      return False
    if rec.get("issuer") != settings.issuer or rec.get("client_id") != settings.client_id:
      return False
    token_endpoint = rec.get("token_endpoint")
    if token_endpoint is not None:
      if not isinstance(token_endpoint, str) or not token_endpoint.strip():
        return False
      _require_trusted_endpoint(
        token_endpoint.strip(),
        issuer=settings.issuer,
        label="token endpoint",
      )
    return True
  except Exception:
    return False


def oauth_record_from_config(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any] | None, XAIOAuthSettings]:
  cfg = config or {}
  env = os.environ if environ is None else environ
  settings = resolve_xai_oauth_settings(cfg, environ=env)
  identity = str(cfg.get("account_identity") or "").strip()
  stored = _selected_xai_record(_load_xai_raw_store(settings.store_path), identity) or {}
  access_token = str(cfg.get("auth_token") or env.get("XAI_AUTH_TOKEN") or stored.get("access_token") or "").strip()
  refresh_token = str(cfg.get("refresh_token") or env.get("XAI_REFRESH_TOKEN") or stored.get("refresh_token") or "").strip()
  record = {
    **stored,
    "access_token": access_token,
    "refresh_token": refresh_token,
    "expires_at": cfg.get("token_expires_at") or env.get("XAI_TOKEN_EXPIRES_AT") or stored.get("expires_at"),
    "scope": str(stored.get("scope") or settings.scope),
    "issuer": str(stored.get("issuer") or settings.issuer),
    "client_id": str(stored.get("client_id") or settings.client_id),
  }
  if identity:
    record["identity"] = identity
  try:
    raw_expires = record["expires_at"]
    record["expires_at"] = float(raw_expires) if raw_expires not in {None, ""} else 0.0
  except (TypeError, ValueError):
    record["expires_at"] = 0.0
  if not access_token and not refresh_token:
    return None, settings
  return record, settings


def token_needs_refresh(record: Mapping[str, Any], *, now: float | None = None) -> bool:
  try:
    expires_at = float(record.get("expires_at") or 0)
  except (TypeError, ValueError):
    # A corrupt expiry in the own-written store must trigger a proactive
    # refresh (which rewrites a clean record) instead of silently disabling
    # refresh forever.
    return True
  return bool(expires_at and expires_at <= (time.time() if now is None else now) + EXPIRY_MARGIN_SECONDS)


def xai_record_requires_reauth(record: Any) -> bool:
  return bool(
    isinstance(record, dict)
    and (record.get("refresh_pending") or record.get("reauth_required"))
  )


async def discover_xai_oauth(client: httpx.AsyncClient, settings: XAIOAuthSettings) -> dict[str, str]:
  response = await client.get(settings.discovery_url, headers={"Accept": "application/json"})
  response.raise_for_status()
  payload = _response_object(response, "xAI OAuth discovery")
  device_endpoint = str(payload.get("device_authorization_endpoint") or "").strip()
  token_endpoint = str(payload.get("token_endpoint") or "").strip()
  if not device_endpoint or not token_endpoint:
    raise RuntimeError("xAI OAuth discovery response is missing device-code endpoints")
  _require_trusted_endpoint(device_endpoint, issuer=settings.issuer, label="device authorization endpoint")
  _require_trusted_endpoint(token_endpoint, issuer=settings.issuer, label="token endpoint")
  return {"device_authorization_endpoint": device_endpoint, "token_endpoint": token_endpoint}


async def request_xai_device_code(
  client: httpx.AsyncClient,
  settings: XAIOAuthSettings,
  device_endpoint: str,
) -> XAIDeviceCode:
  response = await client.post(
    device_endpoint,
    data={"client_id": settings.client_id, "scope": settings.scope},
    headers={"Accept": "application/json"},
  )
  response.raise_for_status()
  payload = _response_object(response, "xAI device code request")
  device_code = str(payload.get("device_code") or "").strip()
  user_code = str(payload.get("user_code") or "").strip()
  verification_uri = str(payload.get("verification_uri") or "").strip()
  complete = str(payload.get("verification_uri_complete") or "").strip() or None
  if not device_code or not user_code or not verification_uri:
    raise RuntimeError("xAI device code response is missing device_code, user_code, or verification_uri")
  _require_trusted_endpoint(verification_uri, issuer=settings.issuer, label="verification URI")
  if complete:
    _require_trusted_endpoint(complete, issuer=settings.issuer, label="complete verification URI")
  expires_in = _positive_int(payload.get("expires_in"), DEFAULT_DEVICE_TIMEOUT_SECONDS)
  interval = float(_positive_int(payload.get("interval"), int(DEFAULT_POLL_INTERVAL_SECONDS)))
  return XAIDeviceCode(device_code, user_code, verification_uri, complete, expires_in, interval)


async def poll_xai_device_token(
  client: httpx.AsyncClient,
  settings: XAIOAuthSettings,
  token_endpoint: str,
  device: XAIDeviceCode,
  *,
  sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
  now: Callable[[], float] = time.time,
) -> dict[str, Any]:
  deadline = now() + device.expires_in
  interval = max(MIN_POLL_INTERVAL_SECONDS, device.interval)
  while now() < deadline:
    response = await client.post(
      token_endpoint,
      data={
        "grant_type": DEVICE_CODE_GRANT_TYPE,
        "client_id": settings.client_id,
        "device_code": device.device_code,
      },
      headers={"Accept": "application/json"},
    )
    payload = _response_object(response, "xAI device token exchange", allow_error=True)
    if response.is_success:
      return _parse_token_response(payload, settings, token_endpoint, require_refresh=True, now=now())
    error = str(payload.get("error") or "")
    if error == "authorization_pending":
      await sleep(interval)
      continue
    if error == "slow_down":
      interval += SLOW_DOWN_INCREMENT_SECONDS
      await sleep(interval)
      continue
    if error in {"access_denied", "authorization_denied"}:
      raise RuntimeError("xAI device authorization was denied")
    if error == "expired_token":
      raise RuntimeError("xAI device code expired; run the login again")
    _raise_oauth_response_error(response, payload, "xAI device token exchange")
  raise RuntimeError("xAI device authorization timed out")


async def login_xai_device_code(
  *,
  config: Mapping[str, Any] | None = None,
  on_verification: VerificationCallback | None = None,
  client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, Any], Path]:
  settings = resolve_xai_oauth_settings(config)
  owns_client = client is None
  oauth_client = client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
  try:
    discovery = await discover_xai_oauth(oauth_client, settings)
    device = await request_xai_device_code(oauth_client, settings, discovery["device_authorization_endpoint"])
    if on_verification is not None:
      result = on_verification(device)
      if asyncio.iscoroutine(result):
        await result
    record = await poll_xai_device_token(oauth_client, settings, discovery["token_endpoint"], device)
    record["device_authorization_endpoint"] = discovery["device_authorization_endpoint"]
    async with _store_lock(settings.store_path):
      upsert_xai_token_record(settings.store_path, record)
    return record, settings.store_path
  finally:
    if owns_client:
      await oauth_client.aclose()


async def refresh_xai_oauth_token(
  record: Mapping[str, Any],
  *,
  settings: XAIOAuthSettings,
  client: httpx.AsyncClient | None = None,
  force: bool = False,
) -> dict[str, Any]:
  # Run the whole lock-held read→POST→save as an explicit task. Shielding alone is NOT enough:
  # asyncio.shield raises CancelledError in THIS coroutine immediately while the inner task runs
  # detached, which would let the caller's `finally` (e.g. provider_summarize.py:144
  # close_client) close the shared httpx client mid-POST and abort the save → R0 stays on disk →
  # reuse. So on caller cancellation we keep THIS frame on the stack until the inner task is
  # terminal (client stays open through save), THEN re-raise the cancellation.
  task = asyncio.ensure_future(_locked_refresh(record, settings, client, force))
  try:
    return await asyncio.shield(task)
  except asyncio.CancelledError:
    # Caller cancelled. Keep THIS frame (and the shared client) alive until the inner refresh
    # is terminal, so close_client cannot abort the POST/save. Loop swallows BOTH re-cancels
    # and the inner task's own failure; we recover the real outcome after the loop.
    while not task.done():
      try:
        await asyncio.shield(task)
      except BaseException:               # re-cancel OR inner failure — keep waiting for done()
        pass
    # Terminal. Retrieve the outcome deterministically so it is never "never retrieved", and
    # make a durable-save failure VISIBLE instead of masking it behind CancelledError.
    if not task.cancelled():
      inner_exc = task.exception()
      if inner_exc is not None and not isinstance(inner_exc, asyncio.CancelledError):
        LOGGER.error("xAI OAuth refresh failed during caller cancellation; token store may "
                     "be stale (rotated token unsaved): %r", inner_exc)
    raise                                   # honor the caller's cancellation deterministically
  # Non-cancel path: a failed POST/save propagates as its own exception via the initial await.


async def _locked_refresh(
  record: Mapping[str, Any],
  settings: XAIOAuthSettings,
  client: httpx.AsyncClient | None,
  force: bool,
) -> dict[str, Any]:
  async with _store_lock(settings.store_path):
    canonical_path = os.path.realpath(settings.store_path.expanduser())
    identity = str(record.get("identity") or record.get("account_identity") or "").strip()
    on_disk = _selected_xai_record(_load_xai_raw_store(settings.store_path), identity)
    if on_disk is None:
      if canonical_path in _STORE_REFRESH_CONSUMED or xai_record_requires_reauth(record):
        raise RuntimeError(_REAUTH_REQUIRED_MESSAGE)
      post_refresh = str(record.get("refresh_token") or "").strip()
    elif xai_record_requires_reauth(on_disk):
      raise RuntimeError(_REAUTH_REQUIRED_MESSAGE)
    elif _record_is_valid(on_disk, settings):
      if not force and not token_needs_refresh(on_disk):
        return dict(on_disk)
      post_refresh = str(on_disk["refresh_token"]).strip()
    else:
      raise RuntimeError(
        "xAI OAuth store record is present but not valid for the current "
        "issuer/client; run `python3 -m agent_gateway.cli auth login xai`"
      )

    if not post_refresh:
      raise RuntimeError(
        "xAI OAuth credential is missing refresh token; run "
        "`python3 -m agent_gateway.cli auth login xai`"
      )

    source_record = dict(on_disk if on_disk is not None else record)
    if identity:
      source_record["identity"] = identity
    owns_client = client is None
    oauth_client = client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
    try:
      token_endpoint = str(
        source_record.get("token_endpoint") or record.get("token_endpoint") or ""
      ).strip()
      if not token_endpoint:
        token_endpoint = (await discover_xai_oauth(oauth_client, settings))["token_endpoint"]
      _require_trusted_endpoint(token_endpoint, issuer=settings.issuer, label="token endpoint")
      base_record = dict(source_record)
      pending_record = {**base_record, "refresh_pending": True}
      _save_token_record_durable(settings.store_path, pending_record)
      was_present = canonical_path in _STORE_REFRESH_CONSUMED
      _STORE_REFRESH_CONSUMED.add(canonical_path)
      post_started_monotonic = time.monotonic()
      try:
        response = await oauth_client.post(
          token_endpoint,
          data={
            "grant_type": "refresh_token",
            "client_id": settings.client_id,
            "refresh_token": post_refresh,
          },
          headers={"Accept": "application/json"},
        )
      except Exception as exc:
        if isinstance(exc, _PRESEND_EXC):
          cleared_record = dict(base_record)
          cleared_record.pop("refresh_pending", None)
          try:
            _save_token_record_durable(settings.store_path, cleared_record)
          except OSError:
            LOGGER.exception("Unable to clear xAI OAuth refresh-pending marker after pre-send failure")
          if not was_present:
            _STORE_REFRESH_CONSUMED.remove(canonical_path)
          raise

        remaining = (
          post_started_monotonic + _GRACE_HARD_BUDGET_S
        ) - time.monotonic()
        if remaining < _MIN_RETRY_BUDGET_S:
          raise
        retry_client = httpx.AsyncClient(
          timeout=httpx.Timeout(remaining, connect=remaining)
        )
        try:
          budget = (
            post_started_monotonic + _GRACE_HARD_BUDGET_S
          ) - time.monotonic()
          if budget < _MIN_RETRY_BUDGET_S:
            raise exc
          retry_resp = await asyncio.wait_for(
            retry_client.post(
              token_endpoint,
              data={
                "grant_type": "refresh_token",
                "client_id": settings.client_id,
                "refresh_token": post_refresh,
              },
              headers={"Accept": "application/json"},
            ),
            timeout=budget,
          )
        except asyncio.TimeoutError:
          raise
        except asyncio.CancelledError:
          raise
        except Exception:
          raise
        finally:
          try:
            await retry_client.aclose()
          except Exception:
            pass
        retry_payload = _response_object(
          retry_resp,
          "xAI OAuth refresh",
          allow_error=True,
        )
        if not retry_resp.is_success:
          _raise_oauth_response_error(
            retry_resp,
            retry_payload,
            "xAI OAuth refresh",
          )
        refreshed = _parse_token_response(
          retry_payload,
          settings,
          token_endpoint,
          require_refresh=True,
        )
        for key in ("identity", "device_authorization_endpoint", "token_endpoint"):
          value = source_record.get(key) or record.get(key)
          if value:
            refreshed[key] = value
        _save_token_record_durable(settings.store_path, refreshed)
        return refreshed
      payload = _response_object(response, "xAI OAuth refresh", allow_error=True)
      if not response.is_success:
        _raise_oauth_response_error(response, payload, "xAI OAuth refresh")
      refreshed = _parse_token_response(payload, settings, token_endpoint, require_refresh=True)
      for key in ("identity", "device_authorization_endpoint", "token_endpoint"):
        value = source_record.get(key) or record.get(key)
        if value:
          refreshed[key] = value
      _save_token_record_durable(settings.store_path, refreshed)
      return refreshed
    finally:
      if owns_client:
        await oauth_client.aclose()


def _parse_token_response(
  payload: Mapping[str, Any],
  settings: XAIOAuthSettings,
  token_endpoint: str,
  *,
  require_refresh: bool,
  now: float | None = None,
) -> dict[str, Any]:
  access_token = str(payload.get("access_token") or "").strip()
  refresh_token = str(payload.get("refresh_token") or "").strip()
  if not access_token:
    raise RuntimeError("xAI OAuth token response is missing access_token")
  if require_refresh and not refresh_token:
    raise RuntimeError("xAI OAuth token response is missing refresh_token; offline_access may have been rejected")
  expires_in = _positive_int(payload.get("expires_in"), 3600)
  return {
    "access_token": access_token,
    "refresh_token": refresh_token,
    "expires_at": (time.time() if now is None else now) + expires_in,
    "scope": str(payload.get("scope") or settings.scope),
    "issuer": settings.issuer,
    "client_id": settings.client_id,
    "token_endpoint": token_endpoint,
    **({"id_token": str(payload["id_token"])} if payload.get("id_token") else {}),
  }


def _require_trusted_endpoint(endpoint: str, *, issuer: str, label: str) -> None:
  parsed = urlparse(endpoint)
  issuer_host = urlparse(issuer).hostname
  host = parsed.hostname
  trusted = bool(
    parsed.scheme == "https"
    and host
    and (host == issuer_host or host == "x.ai" or host.endswith(".x.ai"))
  )
  if not trusted:
    raise ValueError(f"xAI OAuth returned untrusted {label}")


def _response_object(response: httpx.Response, context: str, *, allow_error: bool = False) -> dict[str, Any]:
  try:
    payload = response.json()
  except ValueError as exc:
    raise RuntimeError(f"{context} returned invalid JSON") from exc
  if not isinstance(payload, dict):
    raise RuntimeError(f"{context} returned an invalid payload")
  if not allow_error and not response.is_success:
    _raise_oauth_response_error(response, payload, context)
  return dict(payload)


def _raise_oauth_response_error(response: httpx.Response, payload: Mapping[str, Any], context: str) -> None:
  error = str(payload.get("error") or "").strip()
  description = str(payload.get("error_description") or "").strip()
  detail = f": {error}" if error else ""
  if description:
    detail += f" ({description})"
  raise RuntimeError(f"{context} failed ({response.status_code}){detail}")


def _positive_int(value: Any, default: int) -> int:
  if isinstance(value, bool):
    return default
  try:
    parsed = int(value)
  except (TypeError, ValueError, OverflowError):
    return default
  return parsed if 0 < parsed <= 86_400 else default


def resolve_xai_pool_candidates(
  config: Mapping[str, Any] | None = None,
  *,
  user_scope: Any = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[list[PoolAccountCandidate], None]:
  cfg = config or {}
  env = os.environ if environ is None else environ
  mode = str(cfg.get("auth_mode") or env.get("XAI_AUTH_MODE") or "").strip().lower()
  mode = resolve_xai_auth_mode(cfg, environ=env) if mode else ""
  oauth_paths = resolve_xai_store_paths(cfg, user_scope=user_scope, environ=env)
  api_paths = resolve_api_key_store_paths("xai", cfg, user_scope=user_scope, environ=env)
  candidates: list[PoolAccountCandidate] = []
  seen: set[str] = set()
  for index in (0, 1):
    for auth_mode, path in (("oauth", oauth_paths[index]), ("api", api_paths[index])):
      if path is None or (mode and mode != auth_mode):
        continue
      accounts = (
        load_xai_accounts(path) if auth_mode == "oauth"
        else load_pool_accounts(path, default_prefix="xai", required_any_keys=("api_key",))
      )
      for record in accounts:
        key = str(record["identity"])
        if key not in seen:
          seen.add(key)
          candidates.append(PoolAccountCandidate(
            provider="xai", key=key, record=record, store_path=path,
            is_user_store=index == 0, auth_mode=auth_mode,
          ))
  for auth_mode, name in (("oauth", "XAI_AUTH_TOKEN"), ("api", "XAI_API_KEY")):
    token = str(env.get(name) or "").strip()
    if token and (not mode or mode == auth_mode):
      record: dict[str, Any] = {
        "identity": f"env:{name}:{hashlib.sha256(token.encode()).hexdigest()[:12]}",
        "access_token" if auth_mode == "oauth" else "api_key": token,
      }
      if auth_mode == "oauth":
        record.update(
          refresh_token=str(env.get("XAI_REFRESH_TOKEN") or "").strip(),
          expires_at=env.get("XAI_TOKEN_EXPIRES_AT") or 0,
        )
      key = str(record["identity"])
      if key not in seen:
        seen.add(key)
        candidates.append(PoolAccountCandidate(
          provider="xai", key=key, record=record, auth_mode=auth_mode,
        ))
  return candidates, None


def materialize_xai_pool_candidate(
  selected: PoolAccountCandidate, config: Mapping[str, Any],
) -> dict[str, Any]:
  materialized = dict(config)
  materialized.update(
    api_key="", auth_token="",
    refresh_token=str(selected.record.get("refresh_token") or "").strip(),
    token_expires_at=selected.record.get("expires_at"),
    auth_store_path=str(selected.store_path or ""), api_key_store_path="",
    device_authorization_endpoint="", token_endpoint="",
  )
  materialized.update(
    auth_mode=selected.auth_mode, account_identity=selected.account_identity,
    auth_user_scope=normalize_user_scope(config.get("auth_user_scope")),
  )
  if selected.auth_mode == "api":
    materialized["api_key"] = selected.token
  else:
    materialized["auth_token"] = selected.token
    for source, target in (
      ("issuer", "oauth_issuer"), ("client_id", "oauth_client_id"), ("scope", "oauth_scope"),
      ("device_authorization_endpoint", "device_authorization_endpoint"),
      ("token_endpoint", "token_endpoint"),
    ):
      if source in selected.record:
        materialized[target] = selected.record[source]
  return materialized


register_provider_pool_hooks(
  "xai", resolver=resolve_xai_pool_candidates, materializer=materialize_xai_pool_candidate,
)


__all__ = [
  "DEFAULT_XAI_OAUTH_CLIENT_ID",
  "DEFAULT_XAI_OAUTH_ISSUER",
  "DEFAULT_XAI_OAUTH_SCOPE",
  "XAIDeviceCode",
  "XAIOAuthSettings",
  "XAIOAuthTokenRecord",
  "discover_xai_oauth",
  "load_xai_token_record",
  "load_xai_accounts",
  "materialize_xai_pool_candidate",
  "login_xai_device_code",
  "oauth_record_from_config",
  "poll_xai_device_token",
  "refresh_xai_oauth_token",
  "resolve_xai_auth_mode",
  "resolve_xai_oauth_settings",
  "resolve_xai_pool_candidates",
  "resolve_xai_store_paths",
  "save_xai_token_record",
  "token_needs_refresh",
  "token_store_is_private",
  "xai_record_requires_reauth",
]
