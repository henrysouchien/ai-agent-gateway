from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urlparse

import httpx


DEFAULT_XAI_OAUTH_ISSUER = "https://auth.x.ai"
DEFAULT_XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
DEFAULT_XAI_OAUTH_SCOPE = "openid profile email offline_access grok-cli:access api:access"
DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
DEFAULT_DEVICE_TIMEOUT_SECONDS = 300
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
MIN_POLL_INTERVAL_SECONDS = 1.0
SLOW_DOWN_INCREMENT_SECONDS = 5.0
EXPIRY_MARGIN_SECONDS = 60


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
  raw_store = str(cfg.get("auth_store_path") or env.get("XAI_AUTH_STORE_PATH") or "").strip()
  if raw_store:
    store_path = Path(raw_store).expanduser()
  else:
    user_data = str(env.get("USER_DATA_DIR") or "").strip()
    base = Path(user_data).expanduser() if user_data else Path.home() / ".agent_gateway"
    store_path = base / "xai" / "oauth.json"
  if not issuer or not client_id or not scope:
    raise ValueError("xAI OAuth issuer, client ID, and scopes must not be blank")
  _require_trusted_endpoint(discovery_url, issuer=issuer, label="discovery URL")
  return XAIOAuthSettings(issuer, client_id, scope, discovery_url, store_path)


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


def load_xai_token_record(path: Path) -> dict[str, Any] | None:
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


def save_xai_token_record(path: Path, record: Mapping[str, Any]) -> None:
  required = ("access_token", "refresh_token", "expires_at", "scope", "issuer", "client_id")
  missing = [key for key in required if record.get(key) in {None, ""}]
  if missing:
    raise ValueError(f"xAI OAuth token record is missing: {', '.join(missing)}")
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


def oauth_record_from_config(
  config: Mapping[str, Any] | None = None,
  *,
  environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any] | None, XAIOAuthSettings]:
  cfg = config or {}
  env = os.environ if environ is None else environ
  settings = resolve_xai_oauth_settings(cfg, environ=env)
  stored = load_xai_token_record(settings.store_path) or {}
  access_token = str(cfg.get("auth_token") or env.get("XAI_AUTH_TOKEN") or stored.get("access_token") or "").strip()
  refresh_token = str(cfg.get("refresh_token") or env.get("XAI_REFRESH_TOKEN") or stored.get("refresh_token") or "").strip()
  raw_expires = cfg.get("token_expires_at") or env.get("XAI_TOKEN_EXPIRES_AT") or stored.get("expires_at")
  try:
    expires_at = float(raw_expires) if raw_expires not in {None, ""} else 0.0
  except (TypeError, ValueError):
    expires_at = 0.0
  if not access_token and not refresh_token:
    return None, settings
  record = {
    **stored,
    "access_token": access_token,
    "refresh_token": refresh_token,
    "expires_at": expires_at,
    "scope": str(stored.get("scope") or settings.scope),
    "issuer": str(stored.get("issuer") or settings.issuer),
    "client_id": str(stored.get("client_id") or settings.client_id),
  }
  return record, settings


def token_needs_refresh(record: Mapping[str, Any], *, now: float | None = None) -> bool:
  try:
    expires_at = float(record.get("expires_at") or 0)
  except (TypeError, ValueError):
    return False
  return bool(expires_at and expires_at <= (time.time() if now is None else now) + EXPIRY_MARGIN_SECONDS)


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
  oauth_client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
  try:
    discovery = await discover_xai_oauth(oauth_client, settings)
    device = await request_xai_device_code(oauth_client, settings, discovery["device_authorization_endpoint"])
    if on_verification is not None:
      result = on_verification(device)
      if asyncio.iscoroutine(result):
        await result
    record = await poll_xai_device_token(oauth_client, settings, discovery["token_endpoint"], device)
    record["device_authorization_endpoint"] = discovery["device_authorization_endpoint"]
    save_xai_token_record(settings.store_path, record)
    return record, settings.store_path
  finally:
    if owns_client:
      await oauth_client.aclose()


async def refresh_xai_oauth_token(
  record: Mapping[str, Any],
  *,
  settings: XAIOAuthSettings,
  client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
  refresh_token = str(record.get("refresh_token") or "").strip()
  if not refresh_token:
    raise RuntimeError("xAI OAuth credential is missing refresh token; run `agent auth login xai`")
  owns_client = client is None
  oauth_client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
  try:
    token_endpoint = str(record.get("token_endpoint") or "").strip()
    if not token_endpoint:
      token_endpoint = (await discover_xai_oauth(oauth_client, settings))["token_endpoint"]
    _require_trusted_endpoint(token_endpoint, issuer=settings.issuer, label="token endpoint")
    response = await oauth_client.post(
      token_endpoint,
      data={
        "grant_type": "refresh_token",
        "client_id": settings.client_id,
        "refresh_token": refresh_token,
      },
      headers={"Accept": "application/json"},
    )
    payload = _response_object(response, "xAI OAuth refresh", allow_error=True)
    if not response.is_success:
      _raise_oauth_response_error(response, payload, "xAI OAuth refresh")
    refreshed = _parse_token_response(payload, settings, token_endpoint, require_refresh=False)
    if not refreshed.get("refresh_token"):
      refreshed["refresh_token"] = refresh_token
    for key in ("device_authorization_endpoint",):
      if record.get(key):
        refreshed[key] = record[key]
    save_xai_token_record(settings.store_path, refreshed)
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


__all__ = [
  "DEFAULT_XAI_OAUTH_CLIENT_ID",
  "DEFAULT_XAI_OAUTH_ISSUER",
  "DEFAULT_XAI_OAUTH_SCOPE",
  "XAIDeviceCode",
  "XAIOAuthSettings",
  "discover_xai_oauth",
  "load_xai_token_record",
  "login_xai_device_code",
  "oauth_record_from_config",
  "poll_xai_device_token",
  "refresh_xai_oauth_token",
  "resolve_xai_auth_mode",
  "resolve_xai_oauth_settings",
  "save_xai_token_record",
  "token_needs_refresh",
  "token_store_is_private",
]
