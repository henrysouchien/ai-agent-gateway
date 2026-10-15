from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import http.server
import json
import logging
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Protocol, TextIO
import urllib.parse
import webbrowser

import httpx

from .providers.google_oauth import (
  load_google_token_record,
  resolve_google_oauth_settings,
  save_google_token_record,
)
from .providers.copilot_oauth import (
  CopilotOAuthSettings,
  load_copilot_token_record,
  poll_copilot_device_token,
  request_copilot_device_code,
  resolve_copilot_oauth_settings,
)
from .providers.credential_pool import (
  USER_SCOPE_FIELD,
  derive_account_identity,
  load_pool_accounts,
  remove_pool_account,
)
from .providers.google_antigravity_oauth import (
  ANTIGRAVITY_AUTH_URL,
  ANTIGRAVITY_CLIENT_ID,
  ANTIGRAVITY_CLIENT_SECRET,
  ANTIGRAVITY_DEFAULT_PORT,
  ANTIGRAVITY_SCOPES,
  ANTIGRAVITY_TOKEN_URL,
  discover_antigravity_project_id,
  fetch_antigravity_account_identity,
  load_antigravity_token_record,
  refresh_antigravity_tokens,
  resolve_antigravity_store_path,
  save_antigravity_token_record,
  token_needs_refresh as antigravity_token_needs_refresh,
)
from .providers.muse_oauth import (
  load_muse_token_record,
  mint_muse_api_key,
  poll_muse_device_token,
  request_muse_device_code,
  resolve_muse_store_path,
  save_muse_token_record,
)
from .providers.kimi_code_oauth import (
  extract_kimi_account_identity,
  poll_kimi_code_device_token,
  request_kimi_code_device_code,
  resolve_kimi_code_store_path,
  save_kimi_code_token_record,
)
from .providers.api_key_store import (
  load_api_key_record,
  resolve_api_key_store_path,
  save_api_key_record,
)
LOGGER = logging.getLogger(__name__)


def _generate_pkce() -> tuple[str, str]:
  verifier = secrets.token_urlsafe(32)
  digest = hashlib.sha256(verifier.encode("ascii")).digest()
  challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
  return verifier, challenge


def _format_expires_suffix(expires_at: Any) -> str:
  if expires_at is None:
    return ""
  try:
    from datetime import datetime, timezone
    exp_dt = datetime.fromtimestamp(float(expires_at), tz=timezone.utc)
    return f"; expires={exp_dt.isoformat()}"
  except Exception:
    return f"; expires={expires_at}"


def run_loopback_oauth_flow(
  *,
  auth_url_builder: Callable[[str, str, str], str],
  token_exchange: Callable[[str, str, str], dict[str, Any]],
  host: str = "127.0.0.1",
  redirect_host: str | None = None,
  callback_path: str = "/oauth-callback",
  port: int = 0,
  timeout: float = 120.0,
  console_fallback: bool = True,
  stdout: TextIO | None = None,
  no_browser: bool = False,
) -> dict[str, Any]:
  state = secrets.token_urlsafe(16)
  verifier, challenge = _generate_pkce()

  callback_params: dict[str, list[str]] = {}
  callback_event = threading.Event()

  class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
      parsed = urllib.parse.urlsplit(self.path)
      params = urllib.parse.parse_qs(parsed.query)
      callback_params.update(params)
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.end_headers()
      self.wfile.write(
        b"<!DOCTYPE html><html><body style='font-family: sans-serif; text-align: center; padding: 40px;'>"
        b"<h2>Authentication successful</h2><p>You can return to the console.</p></body></html>"
      )
      callback_event.set()

    def log_message(self, format: str, *args: Any) -> None:
      pass

  server: http.server.HTTPServer | None = None
  actual_port = port
  if port != 0:
    try:
      server = http.server.HTTPServer((host, port), _Handler)
      actual_port = port
    except OSError:
      server = None

  if server is None:
    server = http.server.HTTPServer((host, 0), _Handler)
    actual_port = server.server_port

  server_thread = threading.Thread(target=server.serve_forever, daemon=True)
  server_thread.start()

  redirect_target_host = redirect_host or host
  redirect_uri = f"http://{redirect_target_host}:{actual_port}{callback_path}"
  auth_url = auth_url_builder(redirect_uri, state, challenge)

  out = stdout or LOGGER.info
  if stdout:
    stdout.write("\nOAuth Authorization\n")
    stdout.write(f"Open: {auth_url}\n")
    stdout.write("Waiting for authorization callback in browser...\n")
    stdout.flush()

  if not no_browser:
    try:
      webbrowser.open(auth_url)
    except Exception:
      pass

  try:
    if callback_event.wait(timeout):
      if "error" in callback_params:
        err = callback_params["error"][0]
        desc = callback_params.get("error_description", [""])[0]
        raise RuntimeError(f"OAuth authorization failed: {err} ({desc})")
      if (
        "state" not in callback_params
        or not callback_params["state"][0]
        or callback_params["state"][0] != state
      ):
        raise RuntimeError("OAuth state mismatch in callback; authorization rejected")
      if "code" not in callback_params or not callback_params["code"][0]:
        raise RuntimeError("No authorization code in callback")
      code = callback_params["code"][0]
      return token_exchange(code, verifier, redirect_uri)
  finally:
    server.shutdown()
    server.server_close()

  if console_fallback and stdout:
    stdout.write("\nCallback not received automatically.\n")
    pasted = getpass.getpass("Paste the full callback URL or code: ").strip()
    if pasted:
      parsed_pasted = urllib.parse.urlsplit(pasted)
      if (
        "code=" in pasted
        or "state=" in pasted
        or parsed_pasted.scheme
        or parsed_pasted.query
      ):
        q = urllib.parse.parse_qs(parsed_pasted.query or parsed_pasted.path)
        if "state" not in q or not q["state"][0] or q["state"][0] != state:
          raise RuntimeError("OAuth state mismatch in callback; authorization rejected")
        if "code" not in q or not q["code"][0]:
          raise RuntimeError("No authorization code in callback")
        code = q["code"][0]
      else:
        code = pasted
      return token_exchange(code, verifier, redirect_uri)
  raise TimeoutError("OAuth authorization callback timed out or was cancelled")


def run_device_code_flow(
  *,
  settings: CopilotOAuthSettings,
  stdout: TextIO,
  no_browser: bool = False,
  client: httpx.Client | None = None,
) -> dict[str, Any]:
  device = request_copilot_device_code(settings=settings, client=client)
  user_code = device.user_code
  device_code = device.device_code
  verification_uri = device.verification_uri
  expires_in = float(device.expires_in)

  stdout.write("\nGitHub Copilot device authorization\n")
  stdout.write(f"Open: {verification_uri}\n")
  stdout.write(f"Enter code: {user_code}\n")
  stdout.write(f"Expires in approximately {max(1, round(expires_in / 60))} minutes.\n")
  stdout.write("Waiting for authorization...\n")
  stdout.flush()

  if not no_browser:
    try:
      webbrowser.open(verification_uri)
    except Exception:
      pass

  record = poll_copilot_device_token(
    settings,
    device,
    client=client,
  )
  stdout.write(f"Copilot device authorization complete. Token store: {settings.store_path}\n")
  return record


def run_delegated_flow(
  *,
  command: list[str],
  stdout: TextIO,
) -> int:
  cmd_str = " ".join(command)
  stdout.write(f"Launching delegated signin: {cmd_str}...\n")
  stdout.flush()
  try:
    res = subprocess.run(command, check=False)
    if res.returncode == 0:
      stdout.write(f"Delegated signin with `{command[0]}` complete.\n")
      return 0
    stdout.write(f"Delegated signin with `{command[0]}` exited with code {res.returncode}.\n")
    return res.returncode
  except FileNotFoundError as exc:
    raise RuntimeError(f"Command `{command[0]}` not found; install `{command[0]}` first.") from exc


def run_api_key_prompt_flow(
  *,
  provider_name: str,
  family: str,
  setup_url: str,
  stdout: TextIO,
  no_browser: bool = False,
  user_scope: Any = None,
  store_path_override: str | None = None,
) -> int:
  cfg = {"api_key_store_path": store_path_override} if store_path_override else {}
  store_path = resolve_api_key_store_path(family, cfg, user_scope=user_scope)

  stdout.write(f"\n{provider_name} API Key Setup\n")
  stdout.write(f"Create or copy your API key at: {setup_url}\n")
  stdout.flush()

  if not no_browser:
    try:
      webbrowser.open(setup_url)
    except Exception:
      pass

  key = getpass.getpass(f"Paste your {provider_name} API key: ").strip()
  if not key:
    raise RuntimeError(f"No {provider_name} API key provided.")

  record = {
    "api_key": key,
    "source": "api_key",
    "updated_at": time.time(),
  }
  save_api_key_record(store_path, record)
  stdout.write(f"{provider_name} API key saved. Store: {store_path}\n")
  return 0


class LoginFlow(Protocol):
  name: str
  kind: str

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    ...

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    ...

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    ...


class GeminiLoginFlow:
  name = "gemini"
  kind = "oauth_code"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    if getattr(args, "api_key", False):
      store = str(getattr(args, "store", None) or "").strip()
      return run_api_key_prompt_flow(
        provider_name="Gemini (Google AI Studio)",
        family="google",
        setup_url="https://aistudio.google.com/apikey",
        stdout=stdout,
        no_browser=bool(getattr(args, "no_browser", False)),
        user_scope=getattr(args, "user", None),
        store_path_override=store or None,
      )

    store = str(getattr(args, "store", None) or "").strip()
    cfg = {"auth_store_path": store} if store else {}
    store_path = resolve_antigravity_store_path(cfg, user_scope=getattr(args, "user", None))

    def build_auth_url(redirect_uri: str, state: str, challenge: str) -> str:
      del challenge
      params = {
        "client_id": ANTIGRAVITY_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(ANTIGRAVITY_SCOPES),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
      }
      return f"{ANTIGRAVITY_AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code(code: str, verifier: str, redirect_uri: str) -> dict[str, Any]:
      del verifier
      data = {
        "client_id": ANTIGRAVITY_CLIENT_ID,
        "client_secret": ANTIGRAVITY_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
      }
      with httpx.Client() as client:
        resp = client.post(ANTIGRAVITY_TOKEN_URL, data=data, timeout=30.0)
        if resp.status_code != 200:
          raise RuntimeError(f"Antigravity token exchange failed: {resp.status_code} {resp.text}")
        payload = resp.json()
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
          raise RuntimeError("No access_token returned in Antigravity token exchange")

        stdout.write("Discovering Antigravity project ID...\n")
        stdout.flush()
        try:
          project_id = discover_antigravity_project_id(access_token, client=client)
        except Exception as exc:
          LOGGER.warning("Could not auto-discover Antigravity project: %s", exc)
          project_id = ""
        try:
          discovered_identity = fetch_antigravity_account_identity(
            access_token,
            client=client,
          )
        except Exception as exc:
          LOGGER.warning("Could not fetch Antigravity account identity: %s", exc)
          discovered_identity = ""

      expires_in = payload.get("expires_in")
      expires_at = (time.time() + float(expires_in)) if expires_in is not None else None
      record: dict[str, Any] = {
        "access_token": access_token,
        "token_type": str(payload.get("token_type") or "Bearer"),
        "scope": str(payload.get("scope") or " ".join(ANTIGRAVITY_SCOPES)),
        "source": "subscription",
      }
      if project_id:
        record["project_id"] = project_id
      if expires_at is not None:
        record["expires_at"] = expires_at
      if "refresh_token" in payload and str(payload["refresh_token"]).strip():
        record["refresh_token"] = str(payload["refresh_token"]).strip()

      identity = str(
        discovered_identity
        or getattr(args, "email", None)
        or project_id
        or ""
      ).strip()
      if not identity:
        identity = derive_account_identity(
          record,
          prefix="antigravity",
          material_keys=("refresh_token", "access_token"),
        )
      record["identity"] = identity
      if "@" in identity:
        record["email"] = identity

      save_antigravity_token_record(store_path, record)
      return record

    try:
      enrolled = run_loopback_oauth_flow(
        auth_url_builder=build_auth_url,
        token_exchange=exchange_code,
        port=ANTIGRAVITY_DEFAULT_PORT,
        stdout=stdout,
        no_browser=bool(getattr(args, "no_browser", False)),
      )
    except Exception as exc:
      raise RuntimeError(f"Gemini Antigravity OAuth login failed: {exc}") from exc

    accounts = load_pool_accounts(
      store_path,
      default_prefix="antigravity",
      required_any_keys=("access_token", "refresh_token"),
    )
    enrolled_id = str((enrolled or {}).get("identity") or "").strip()
    stdout.write(
      f"Gemini Antigravity OAuth login complete. Token store: {store_path} "
      f"(account={enrolled_id}, accounts={len(accounts)})\n"
    )
    return 0

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    api_key_mode = bool(getattr(args, "api_key", False))
    sub_cfg = {"auth_store_path": store} if (store and not api_key_mode) else {}
    api_cfg = {"api_key_store_path": store} if (store and api_key_mode) else {}
    user_scope = getattr(args, "user", None)

    sub_path = resolve_antigravity_store_path(sub_cfg, user_scope=user_scope)
    sub_accounts = load_pool_accounts(
      sub_path,
      default_prefix="antigravity",
      required_any_keys=("access_token", "refresh_token"),
    )

    api_path = resolve_api_key_store_path("google", api_cfg, user_scope=user_scope)
    api_accounts = load_pool_accounts(
      api_path,
      default_prefix="google",
      required_any_keys=("api_key",),
    )

    env_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()

    reported = False
    if sub_accounts:
      sub_record = sub_accounts[0]
      exp_str = _format_expires_suffix(sub_record.get("expires_at"))
      proj = f"; project={sub_record.get('project_id')}" if sub_record.get("project_id") else ""
      stdout.write(
        f"Gemini (Antigravity subscription): active; store={sub_path}{proj}{exp_str}; "
        f"accounts={len(sub_accounts)}\n"
      )
      for acc in sub_accounts:
        acc_id = acc.get("identity") or ""
        acc_proj = f"; project={acc.get('project_id')}" if acc.get("project_id") else ""
        acc_exp = _format_expires_suffix(acc.get("expires_at"))
        stdout.write(f"  account={acc_id}{acc_proj}{acc_exp}\n")
      reported = True

    if api_accounts:
      stdout.write(
        f"Gemini (API key): active; store={api_path}; accounts={len(api_accounts)}\n"
      )
      for acc in api_accounts:
        stdout.write(f"  account={acc.get('identity') or ''}\n")
      reported = True
    elif env_key:
      stdout.write("Gemini (API key): configured via environment variable\n")
      reported = True

    if not reported:
      stdout.write(f"Gemini: not logged in ({sub_path})\n")
      return 1
    return 0

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    api_key_mode = bool(getattr(args, "api_key", False))
    account = str(getattr(args, "account", None) or "").strip() or None
    user_scope = getattr(args, "user", None)
    if api_key_mode:
      api_cfg = {"api_key_store_path": store} if store else {}
      api_path = resolve_api_key_store_path("google", api_cfg, user_scope=user_scope)
      targets: list[tuple[Path, str, tuple[str, ...]]] = [(api_path, "google", ("api_key",))]
    elif store:
      sub_path = resolve_antigravity_store_path({"auth_store_path": store}, user_scope=user_scope)
      targets = [(sub_path, "antigravity", ("access_token", "refresh_token"))]
    else:
      sub_path = resolve_antigravity_store_path({}, user_scope=user_scope)
      api_path = resolve_api_key_store_path("google", {}, user_scope=user_scope)
      targets = [
        (sub_path, "antigravity", ("access_token", "refresh_token")),
        (api_path, "google", ("api_key",)),
      ]
    for p, prefix, req_keys in targets:
      try:
        removed = remove_pool_account(
          p,
          account,
          default_prefix=prefix,
          required_any_keys=req_keys,
        )
      except OSError as exc:
        raise RuntimeError(f"Unable to remove {p}: {exc}") from exc
      if removed:
        stdout.write(f"Removed Gemini credential store: {p}\n")
    return 0

class MuseLoginFlow:
  name = "muse"
  kind = "device_code"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    if getattr(args, "api_key", False):
      store = str(getattr(args, "store", None) or "").strip()
      return run_api_key_prompt_flow(
        provider_name="Muse (Meta Model API)",
        family="muse",
        setup_url="https://dev.meta.ai",
        stdout=stdout,
        no_browser=bool(getattr(args, "no_browser", False)),
        user_scope=getattr(args, "user", None),
        store_path_override=store or None,
      )

    store = str(getattr(args, "store", None) or "").strip()
    cfg = {"auth_store_path": store} if store else {}
    store_path = resolve_muse_store_path(cfg, user_scope=getattr(args, "user", None))

    device = request_muse_device_code()
    user_code = device["user_code"]
    device_code = device["device_code"]
    verification_uri = (
      device.get("verification_uri_complete")
      or device.get("verification_uri")
      or "https://auth.meta.com/oidc/device/authorization/"
    )
    interval = float(device.get("interval", 5.0))
    expires_in = float(device.get("expires_in", 900.0))

    stdout.write("\nMeta Muse Code device authorization\n")
    stdout.write(f"Open: {verification_uri}\n")
    stdout.write(f"Enter code: {user_code}\n")
    stdout.write(f"Expires in approximately {max(1, round(expires_in / 60))} minutes.\n")
    stdout.write("Waiting for authorization...\n")
    stdout.flush()

    if not getattr(args, "no_browser", False):
      try:
        webbrowser.open(verification_uri)
      except Exception:
        pass

    token_payload = poll_muse_device_token(
      device_code,
      interval=interval,
      expires_in=expires_in,
    )
    access_token = token_payload.get("access_token")
    if not access_token:
      raise RuntimeError(f"Muse device authorization returned no access token: {token_payload}")

    stdout.write("Minting Meta Model API key from subscription token...\n")
    stdout.flush()
    key_payload = mint_muse_api_key(access_token)
    api_key = key_payload["api_key"]

    record = {
      "api_key": api_key,
      "source": "subscription",
      "account_id": key_payload.get("user_id") or key_payload.get("user_email") or "",
      "email": key_payload.get("user_email") or "",
      "created_at": time.time(),
    }
    save_muse_token_record(store_path, record)
    stdout.write(f"Muse device authorization complete. Token store: {store_path}\n")
    return 0

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    if getattr(args, "api_key", False):
      cfg = {"api_key_store_path": store} if store else {}
      store_path = resolve_api_key_store_path("muse", cfg, user_scope=getattr(args, "user", None))
    else:
      cfg = {"auth_store_path": store} if store else {}
      store_path = resolve_muse_store_path(cfg, user_scope=getattr(args, "user", None))
    accounts = load_pool_accounts(
      store_path,
      default_prefix="muse",
      required_any_keys=("api_key",),
    )
    env_key = os.getenv("MUSE_API_KEY", "").strip() or os.getenv("META_API_KEY", "").strip()

    if accounts:
      record = accounts[0]
      src = record.get("source", "subscription")
      email_info = f" ({record['email']})" if record.get("email") else ""
      stdout.write(
        f"Muse ({src}): active; store={store_path}{email_info}; accounts={len(accounts)}\n"
      )
      for acc in accounts:
        acc_id = acc.get("identity") or ""
        acc_src = acc.get("source", "subscription")
        stdout.write(f"  account={acc_id}; source={acc_src}\n")
      return 0
    if env_key:
      stdout.write("Muse (API key): configured via environment variable\n")
      return 0
    stdout.write(f"Muse: not logged in ({store_path})\n")
    return 1

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    if getattr(args, "api_key", False):
      cfg = {"api_key_store_path": store} if store else {}
      store_path = resolve_api_key_store_path("muse", cfg, user_scope=getattr(args, "user", None))
    else:
      cfg = {"auth_store_path": store} if store else {}
      store_path = resolve_muse_store_path(cfg, user_scope=getattr(args, "user", None))
    account = str(getattr(args, "account", None) or "").strip() or None
    try:
      removed = remove_pool_account(
        store_path,
        account,
        default_prefix="muse",
        required_any_keys=("api_key",),
      )
    except OSError as exc:
      raise RuntimeError(f"Unable to remove {store_path}: {exc}") from exc
    if removed:
      stdout.write(f"Removed Muse credential store: {store_path}\n")
    return 0


class KimiCodeLoginFlow:
  name = "kimi-code"
  kind = "device_code"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    if getattr(args, "api_key", False):
      return run_api_key_prompt_flow(
        provider_name="Kimi Code",
        family="kimi_code",
        setup_url="https://www.kimi.com/code",
        stdout=stdout,
        no_browser=bool(getattr(args, "no_browser", False)),
        user_scope=getattr(args, "user", None),
        store_path_override=store or None,
      )

    cfg = {"auth_store_path": store} if store else {}
    store_path = resolve_kimi_code_store_path(cfg, user_scope=getattr(args, "user", None))

    device = request_kimi_code_device_code(config=cfg)
    user_code = device["user_code"]
    device_code = device["device_code"]
    verification_uri = device.get("verification_uri_complete") or device["verification_uri"]
    interval = float(device.get("interval", 5.0))
    expires_in = float(device.get("expires_in", 900.0))

    stdout.write("\nKimi Code device authorization\n")
    stdout.write(f"Open: {verification_uri}\n")
    stdout.write(f"Enter code: {user_code}\n")
    stdout.write(f"Expires in approximately {max(1, round(expires_in / 60))} minutes.\n")
    stdout.write("Waiting for authorization...\n")
    stdout.flush()

    if not getattr(args, "no_browser", False):
      try:
        webbrowser.open(verification_uri)
      except Exception:
        pass

    record = poll_kimi_code_device_token(
      device_code,
      interval=interval,
      expires_in=expires_in,
      config=cfg,
    )
    email = str(getattr(args, "email", None) or "").strip()
    if email and not extract_kimi_account_identity(str(record.get("access_token") or "")):
      record["identity"] = email
      record["email"] = email
    save_kimi_code_token_record(store_path, record)
    accounts = load_pool_accounts(
      store_path,
      default_prefix="kimi_code",
      required_any_keys=("access_token", "refresh_token"),
    )
    stdout.write(
      f"Kimi Code device authorization complete. Token store: {store_path} "
      f"(account={record.get('identity')}, accounts={len(accounts)})\n"
    )
    return 0

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    api_key_mode = bool(getattr(args, "api_key", False))
    user_scope = getattr(args, "user", None)
    api_cfg = {"api_key_store_path": store} if (store and api_key_mode) else {}
    api_path = resolve_api_key_store_path("kimi_code", api_cfg, user_scope=user_scope)
    api_accounts = load_pool_accounts(
      api_path,
      default_prefix="kimi_code",
      required_any_keys=("api_key",),
    )

    sub_path = None
    sub_accounts: list[dict[str, Any]] = []
    if not api_key_mode:
      sub_cfg = {"auth_store_path": store} if store else {}
      sub_path = resolve_kimi_code_store_path(sub_cfg, user_scope=user_scope)
      sub_accounts = load_pool_accounts(
        sub_path,
        default_prefix="kimi_code",
        required_any_keys=("access_token", "refresh_token"),
      )
      env_set = any(
        os.getenv(name, "").strip()
        for name in ("KIMI_CODE_AUTH_TOKEN", "KIMI_CODE_API_KEY", "KIMI_API_KEY")
      )
    else:
      env_set = False

    reported = False
    if sub_accounts:
      stdout.write(
        f"Kimi Code (subscription): active; store={sub_path}; accounts={len(sub_accounts)}\n"
      )
      for acc in sub_accounts:
        stdout.write(
          f"  account={acc.get('identity') or ''}"
          f"{_format_expires_suffix(acc.get('expires_at'))}\n"
        )
      reported = True
    if api_accounts:
      stdout.write(
        f"Kimi Code (API key): active; store={api_path}; accounts={len(api_accounts)}\n"
      )
      for acc in api_accounts:
        stdout.write(f"  account={acc.get('identity') or ''}\n")
      reported = True
    if not reported and env_set:
      stdout.write("Kimi Code: configured via environment variable\n")
      reported = True

    if not reported:
      stdout.write(f"Kimi Code: not logged in ({api_path if api_key_mode else sub_path})\n")
      return 1
    return 0

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    api_key_mode = bool(getattr(args, "api_key", False))
    account = str(getattr(args, "account", None) or "").strip() or None
    user_scope = getattr(args, "user", None)
    if api_key_mode:
      api_cfg = {"api_key_store_path": store} if store else {}
      api_path = resolve_api_key_store_path("kimi_code", api_cfg, user_scope=user_scope)
      targets: list[tuple[Path, str, tuple[str, ...]]] = [(api_path, "kimi_code", ("api_key",))]
    elif store:
      sub_path = resolve_kimi_code_store_path({"auth_store_path": store}, user_scope=user_scope)
      targets = [(sub_path, "kimi_code", ("access_token", "refresh_token"))]
    else:
      sub_path = resolve_kimi_code_store_path({}, user_scope=user_scope)
      api_path = resolve_api_key_store_path("kimi_code", {}, user_scope=user_scope)
      targets = [
        (sub_path, "kimi_code", ("access_token", "refresh_token")),
        (api_path, "kimi_code", ("api_key",)),
      ]
    for p, prefix, req_keys in targets:
      try:
        removed = remove_pool_account(
          p,
          account,
          default_prefix=prefix,
          required_any_keys=req_keys,
        )
      except OSError as exc:
        raise RuntimeError(f"Unable to remove {p}: {exc}") from exc
      if removed:
        stdout.write(f"Removed Kimi Code credential store: {p}\n")
    return 0



class OllamaCloudLoginFlow:
  name = "ollama-cloud"
  kind = "api_key"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    return run_api_key_prompt_flow(
      provider_name="Ollama Cloud",
      family="ollama_cloud",
      setup_url="https://ollama.com/settings/keys",
      stdout=stdout,
      no_browser=bool(getattr(args, "no_browser", False)),
      user_scope=getattr(args, "user", None),
      store_path_override=store or None,
    )

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    cfg = {"api_key_store_path": store} if store else {}
    store_path = resolve_api_key_store_path("ollama_cloud", cfg, user_scope=getattr(args, "user", None))
    accounts = load_pool_accounts(
      store_path,
      default_prefix="ollama_cloud",
      required_any_keys=("api_key",),
    )
    env_key = os.getenv("OLLAMA_CLOUD_API_KEY", "").strip() or os.getenv("OLLAMA_API_KEY", "").strip()

    if accounts:
      stdout.write(f"Ollama Cloud: active; store={store_path}; accounts={len(accounts)}\n")
      for acc in accounts:
        stdout.write(f"  account={acc.get('identity') or ''}\n")
      return 0
    if env_key:
      stdout.write("Ollama Cloud: configured via environment variable\n")
      return 0
    stdout.write(f"Ollama Cloud: not configured ({store_path})\n")
    return 1

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    cfg = {"api_key_store_path": store} if store else {}
    store_path = resolve_api_key_store_path("ollama_cloud", cfg, user_scope=getattr(args, "user", None))
    account = str(getattr(args, "account", None) or "").strip() or None
    try:
      removed = remove_pool_account(
        store_path,
        account,
        default_prefix="ollama_cloud",
        required_any_keys=("api_key",),
      )
    except OSError as exc:
      raise RuntimeError(f"Unable to remove {store_path}: {exc}") from exc
    if removed:
      stdout.write(f"Removed Ollama Cloud credential store: {store_path}\n")
    return 0

class GoogleLoginFlow:
  name = "google"
  kind = "oauth_code"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    if getattr(args, "api_key", False):
      store = str(getattr(args, "store", None) or "").strip()
      return run_api_key_prompt_flow(
        provider_name="Google AI Studio",
        family="google",
        setup_url="https://aistudio.google.com/apikey",
        stdout=stdout,
        no_browser=bool(getattr(args, "no_browser", False)),
        user_scope=getattr(args, "user", None),
        store_path_override=store or None,
      )

    store = str(getattr(args, "store", None) or "").strip()
    cfg = {"auth_store_path": store} if store else {}
    settings = resolve_google_oauth_settings(cfg, user_scope=getattr(args, "user", None))

    if not settings.client_id:
      return run_api_key_prompt_flow(
        provider_name="Google AI Studio",
        family="google",
        setup_url="https://aistudio.google.com/apikey",
        stdout=stdout,
        no_browser=bool(getattr(args, "no_browser", False)),
        user_scope=getattr(args, "user", None),
        store_path_override=store or None,
      )

    def build_auth_url(redirect_uri: str, state: str, challenge: str) -> str:
      params = {
        "client_id": settings.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": settings.scope,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
      }
      return f"{settings.auth_url}?{urllib.parse.urlencode(params)}"

    def exchange_code(code: str, verifier: str, redirect_uri: str) -> dict[str, Any]:
      data = {
        "client_id": settings.client_id,
        "code": code,
        "code_verifier": verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
      }
      if settings.client_secret:
        data["client_secret"] = settings.client_secret

      with httpx.Client() as client:
        resp = client.post(settings.token_url, data=data, timeout=30.0)
        if resp.status_code != 200:
          raise RuntimeError(f"Google token exchange failed: {resp.status_code} {resp.text}")
        payload = resp.json()

      access_token = str(payload.get("access_token") or "").strip()
      if not access_token:
        raise RuntimeError("No access_token returned in Google token exchange")

      expires_in = payload.get("expires_in")
      expires_at = (time.time() + float(expires_in)) if expires_in is not None else None
      record: dict[str, Any] = {
        "access_token": access_token,
        "token_type": str(payload.get("token_type") or "Bearer"),
        "scope": str(payload.get("scope") or settings.scope),
      }
      if expires_at is not None:
        record["expires_at"] = expires_at
      if "refresh_token" in payload and str(payload["refresh_token"]).strip():
        record["refresh_token"] = str(payload["refresh_token"]).strip()
      email_arg = str(getattr(args, "email", None) or "").strip()
      if email_arg:
        record["identity"] = email_arg
        if "@" in email_arg:
          record["email"] = email_arg
      save_google_token_record(settings.store_path, record)
      return record

    try:
      run_loopback_oauth_flow(
        auth_url_builder=build_auth_url,
        token_exchange=exchange_code,
        stdout=stdout,
        no_browser=bool(getattr(args, "no_browser", False)),
      )
    except Exception as exc:
      raise RuntimeError(f"Google OAuth login failed: {exc}") from exc

    accounts = load_pool_accounts(
      settings.store_path,
      default_prefix="google",
      required_any_keys=("access_token", "refresh_token"),
    )
    stdout.write(
      f"Google OAuth login complete. Token store: {settings.store_path} "
      f"(accounts={len(accounts)})\n"
    )
    return 0

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    api_key_mode = bool(getattr(args, "api_key", False))
    oauth_cfg = {"auth_store_path": store} if (store and not api_key_mode) else {}
    api_cfg = {"api_key_store_path": store} if (store and api_key_mode) else {}
    user_scope = getattr(args, "user", None)

    settings = resolve_google_oauth_settings(oauth_cfg, user_scope=user_scope)
    accounts = load_pool_accounts(
      settings.store_path,
      default_prefix="google",
      required_any_keys=("access_token", "refresh_token"),
    )
    api_path = resolve_api_key_store_path("google", api_cfg, user_scope=user_scope)
    api_accounts = load_pool_accounts(
      api_path,
      default_prefix="google",
      required_any_keys=("api_key",),
    )
    env_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()

    reported = False
    if accounts:
      record = accounts[0]
      exp_str = _format_expires_suffix(record.get("expires_at"))
      stdout.write(
        f"Google OAuth: active; store={settings.store_path}{exp_str}; "
        f"accounts={len(accounts)}\n"
      )
      for acc in accounts:
        acc_id = acc.get("identity") or ""
        acc_exp = _format_expires_suffix(acc.get("expires_at"))
        stdout.write(f"  account={acc_id}{acc_exp}\n")
      reported = True

    if api_accounts:
      stdout.write(
        f"Google (API key): active; store={api_path}; accounts={len(api_accounts)}\n"
      )
      for acc in api_accounts:
        stdout.write(f"  account={acc.get('identity') or ''}\n")
      reported = True
    elif env_key:
      stdout.write("Google (API key): configured via environment variable\n")
      reported = True

    if not reported:
      stdout.write(f"Google: not logged in ({settings.store_path})\n")
      return 1
    return 0

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    api_key_mode = bool(getattr(args, "api_key", False))
    account = str(getattr(args, "account", None) or "").strip() or None
    user_scope = getattr(args, "user", None)
    if api_key_mode:
      api_cfg = {"api_key_store_path": store} if store else {}
      api_path = resolve_api_key_store_path("google", api_cfg, user_scope=user_scope)
      targets: list[tuple[Path, str, tuple[str, ...]]] = [(api_path, "google", ("api_key",))]
    elif store:
      settings = resolve_google_oauth_settings({"auth_store_path": store}, user_scope=user_scope)
      targets = [(settings.store_path, "google", ("access_token", "refresh_token"))]
    else:
      settings = resolve_google_oauth_settings({}, user_scope=user_scope)
      api_path = resolve_api_key_store_path("google", {}, user_scope=user_scope)
      targets = [
        (settings.store_path, "google", ("access_token", "refresh_token")),
        (api_path, "google", ("api_key",)),
      ]
    for p, prefix, req_keys in targets:
      try:
        removed = remove_pool_account(
          p,
          account,
          default_prefix=prefix,
          required_any_keys=req_keys,
        )
      except OSError as exc:
        raise RuntimeError(f"Unable to remove {p}: {exc}") from exc
      if removed:
        stdout.write(f"Removed Google token store: {p}\n")
    return 0


class CopilotLoginFlow:
  name = "copilot"
  kind = "device_code"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    cfg = {"auth_store_path": store} if store else {}
    settings = resolve_copilot_oauth_settings(cfg, user_scope=getattr(args, "user", None))

    try:
      run_device_code_flow(
        settings=settings,
        stdout=stdout,
        no_browser=bool(getattr(args, "no_browser", False)),
      )
    except Exception as exc:
      raise RuntimeError(f"Copilot OAuth login failed: {exc}") from exc
    return 0

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    cfg = {"auth_store_path": store} if store else {}
    settings = resolve_copilot_oauth_settings(cfg, user_scope=getattr(args, "user", None))
    accounts = load_pool_accounts(
      settings.store_path,
      default_prefix="copilot",
      required_any_keys=("access_token", "refresh_token"),
    )
    if not accounts:
      stdout.write(f"Copilot OAuth: not logged in ({settings.store_path})\n")
      return 1
    stdout.write(f"Copilot OAuth: active; store={settings.store_path}; accounts={len(accounts)}\n")
    for acc in accounts:
      acc_id = acc.get("identity") or ""
      acc_exp = _format_expires_suffix(acc.get("expires_at"))
      stdout.write(f"  account={acc_id}{acc_exp}\n")
    return 0

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    store = str(getattr(args, "store", None) or "").strip()
    cfg = {"auth_store_path": store} if store else {}
    settings = resolve_copilot_oauth_settings(cfg, user_scope=getattr(args, "user", None))
    account = str(getattr(args, "account", None) or "").strip() or None
    try:
      remove_pool_account(
        settings.store_path,
        account,
        default_prefix="copilot",
        required_any_keys=("access_token", "refresh_token"),
      )
    except OSError as exc:
      raise RuntimeError(f"Unable to remove Copilot OAuth token store: {settings.store_path}") from exc
    stdout.write(f"Removed Copilot OAuth token store: {settings.store_path}\n")
    return 0

class OllamaLoginFlow:
  name = "ollama"
  kind = "delegated"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    try:
      return run_delegated_flow(command=["ollama", "signin"], stdout=stdout)
    except Exception as exc:
      raise RuntimeError(str(exc)) from exc

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    stdout.write(f"Ollama: local daemon at {base_url}; delegated signin via 'ollama signin'.\n")
    return 0

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    stdout.write("Ollama does not store gateway credentials; use 'ollama signout' to sign out of the local daemon.\n")
    return 0


class AnthropicLoginFlow:
  name = "anthropic"
  kind = "oauth_code"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    from .cli import _auth_login_anthropic
    return _auth_login_anthropic(args, stdout=stdout)

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    from .cli import _auth_status_anthropic
    return _auth_status_anthropic(args, stdout=stdout)

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    from .cli import _anthropic_store_path
    from .providers.anthropic_oauth import remove_anthropic_oauth_record
    path = _anthropic_store_path(args)
    account = str(getattr(args, "account", None) or "").strip()
    if account:
      remove_anthropic_oauth_record(path, account)
      stdout.write(f"Removed Anthropic OAuth token store: {path}\n")
      stdout.write("Claude Code's saved login was not modified.\n")
      return 0
    try:
      path.unlink()
    except FileNotFoundError:
      pass
    except OSError as exc:
      raise RuntimeError(f"Unable to remove Anthropic OAuth token store: {path}") from exc
    stdout.write(f"Removed Anthropic OAuth token store: {path}\n")
    stdout.write("Claude Code's saved login was not modified.\n")
    return 0

class CodexLoginFlow:
  name = "codex"
  kind = "delegated"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    from .cli import _auth_login_codex
    return _auth_login_codex(args, stdout=stdout)

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    try:
      status = subprocess.run(
        ["codex", "login", "status"],
        check=False,
        capture_output=True,
        text=True,
      )
    except OSError as exc:
      raise RuntimeError("Unable to launch `codex`; install Codex first") from exc
    detail = str(status.stdout or status.stderr or "").strip()
    stdout.write(f"Codex: {detail or 'not logged in'}\n")
    return 0 if status.returncode == 0 else 1

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    stdout.write("Codex OAuth credentials are managed by CAAM; use `codex logout`.\n")
    return 0


class XAILoginFlow:
  name = "xai"
  kind = "device_code"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    from .providers.xai_oauth import XAIDeviceCode, login_xai_device_code

    store = str(getattr(args, "store", None) or "").strip()
    cfg: dict[str, Any] = {"auth_store_path": store} if store else {}
    if getattr(args, "user", None) is not None:
      cfg[USER_SCOPE_FIELD] = getattr(args, "user", None)

    async def on_verification(device: XAIDeviceCode) -> None:
      url = device.verification_uri_complete or device.verification_uri
      stdout.write("\nxAI OAuth device authorization\n")
      stdout.write(f"Open: {url}\n")
      stdout.write(f"Code: {device.user_code}\n")
      stdout.write(f"Expires in approximately {max(1, round(device.expires_in / 60))} minutes.\n")
      stdout.write("Waiting for authorization...\n")
      stdout.flush()
      if not getattr(args, "no_browser", False):
        webbrowser.open(url)

    import asyncio
    _record, path = asyncio.run(
      login_xai_device_code(config=cfg, on_verification=on_verification)
    )
    stdout.write(f"xAI OAuth login complete. Token store: {path}\n")
    return 0

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    from .providers.xai_oauth import (
      load_xai_token_record,
      resolve_xai_oauth_settings,
      token_needs_refresh,
      token_store_is_private,
      xai_record_requires_reauth,
    )

    store = str(getattr(args, "store", None) or "").strip()
    cfg: dict[str, Any] = {"auth_store_path": store} if store else {}
    if getattr(args, "user", None) is not None:
      cfg[USER_SCOPE_FIELD] = getattr(args, "user", None)
    settings = resolve_xai_oauth_settings(config=cfg)
    record = load_xai_token_record(settings.store_path)

    if not record:
      stdout.write(f"xAI OAuth: not logged in ({settings.store_path})\n")
      return 1
    if xai_record_requires_reauth(record):
      if record.get("reauth_required"):
        stdout.write("xAI OAuth: logged out — re-authentication required\n")
      else:
        stdout.write(
          "xAI OAuth: previous refresh did not complete — re-authentication required\n"
        )
      return 1
    accounts = load_pool_accounts(
      settings.store_path,
      default_prefix="xai",
      required_any_keys=("access_token", "refresh_token"),
    )
    state = "refresh required" if token_needs_refresh(record) else "stored; not server-validated"
    permissions = "0600" if token_store_is_private(settings.store_path) else "insecure permissions"
    stdout.write(
      f"xAI OAuth: {state}; store={settings.store_path}; permissions={permissions}; "
      f"accounts={len(accounts) or 1}\n"
    )
    for acc in accounts:
      acc_id = acc.get("identity") or ""
      acc_exp = _format_expires_suffix(acc.get("expires_at"))
      stdout.write(f"  account={acc_id}{acc_exp}\n")
    return 0

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    from .providers.xai_oauth import (
      _store_lock,
      _write_reauth_tombstone,
      resolve_xai_oauth_settings,
    )

    store = str(getattr(args, "store", None) or "").strip()
    cfg: dict[str, Any] = {"auth_store_path": store} if store else {}
    if getattr(args, "user", None) is not None:
      cfg[USER_SCOPE_FIELD] = getattr(args, "user", None)
    settings = resolve_xai_oauth_settings(cfg)
    path = settings.store_path
    account = str(getattr(args, "account", None) or "").strip()
    if account:
      remove_pool_account(
        path,
        account,
        default_prefix="xai",
        required_any_keys=("access_token", "refresh_token"),
      )
      stdout.write(f"Removed xAI OAuth token store: {path}\n")
      return 0

    import asyncio
    async def logout_xai() -> None:
      async with _store_lock(path):
        _write_reauth_tombstone(path)

    asyncio.run(logout_xai())
    stdout.write(f"Removed xAI OAuth token store: {path}\n")
    return 0

class OpenAILoginFlow:
  name = "openai"
  kind = "api_key"

  def login(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    raise RuntimeError(
      "The OpenAI API does not provide ChatGPT subscription OAuth login. "
      "Use OPENAI_API_KEY for provider=openai, or run "
      "`python3 -m agent_gateway.cli auth login codex` "
      "for ChatGPT subscription authentication. No credentials were changed."
    )

  def status(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    api_key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    access_token_present = bool(os.environ.get("OPENAI_AUTH_TOKEN", "").strip())
    if api_key_present or access_token_present:
      kind = "API key" if api_key_present else "bearer access token"
      stdout.write(f"OpenAI API: {kind} configured in the environment; no gateway OAuth store.\n")
      return 0
    stdout.write(
      "OpenAI API: no credential configured. Set OPENAI_API_KEY; for ChatGPT subscription "
      "auth use provider=codex and `python3 -m agent_gateway.cli auth login codex`.\n"
    )
    return 1

  def logout(self, args: argparse.Namespace, *, stdout: TextIO) -> int:
    stdout.write("OpenAI uses OPENAI_API_KEY from environment; no stored token to remove.\n")
    return 0


_REGISTERED_FLOWS: dict[str, LoginFlow] = {
  "anthropic": AnthropicLoginFlow(),
  "codex": CodexLoginFlow(),
  "copilot": CopilotLoginFlow(),
  "gemini": GeminiLoginFlow(),
  "google": GoogleLoginFlow(),
  "kimi-code": KimiCodeLoginFlow(),
  "muse": MuseLoginFlow(),
  "ollama": OllamaLoginFlow(),
  "ollama-cloud": OllamaCloudLoginFlow(),
  "openai": OpenAILoginFlow(),
  "xai": XAILoginFlow(),
}


def registered_login_flows() -> Mapping[str, LoginFlow]:
  return _REGISTERED_FLOWS


def get_login_flow(name: str) -> LoginFlow:
  normalized = str(name or "").strip().lower()
  flow = _REGISTERED_FLOWS.get(normalized)
  if flow is None:
    raise ValueError(f"Unknown login provider family {name!r}. Available: {', '.join(sorted(_REGISTERED_FLOWS))}")
  return flow
