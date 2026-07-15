from __future__ import annotations

import asyncio
import json
from pathlib import Path
import stat
from urllib.parse import parse_qs

import httpx
import pytest

from agent_gateway.providers.xai import XAIProvider
from agent_gateway.providers.xai_oauth import (
  DEFAULT_XAI_OAUTH_CLIENT_ID,
  DEFAULT_XAI_OAUTH_SCOPE,
  load_xai_token_record,
  login_xai_device_code,
  refresh_xai_oauth_token,
  resolve_xai_auth_mode,
  resolve_xai_oauth_settings,
  save_xai_token_record,
)


def _run(coro):
  return asyncio.run(coro)


def _record(**overrides):
  return {
    "access_token": "access-1",
    "refresh_token": "refresh-1",
    "expires_at": 4_000_000_000,
    "scope": DEFAULT_XAI_OAUTH_SCOPE,
    "issuer": "https://auth.x.ai",
    "client_id": DEFAULT_XAI_OAUTH_CLIENT_ID,
    "token_endpoint": "https://auth.x.ai/oauth2/token",
    **overrides,
  }


def test_store_defaults_under_user_data_dir_and_is_mode_0600(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings(environ={"USER_DATA_DIR": str(tmp_path)})
  assert settings.store_path == tmp_path / "xai" / "oauth.json"
  save_xai_token_record(settings.store_path, _record())
  assert stat.S_IMODE(settings.store_path.stat().st_mode) == 0o600
  assert load_xai_token_record(settings.store_path) == _record()


def test_explicit_auth_mode_wins_and_auto_detects_refreshable_store(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record())
  env = {"XAI_AUTH_STORE_PATH": str(store), "XAI_API_KEY": "api-key"}
  assert resolve_xai_auth_mode(environ=env) == "oauth"
  assert resolve_xai_auth_mode(environ={**env, "XAI_AUTH_MODE": "api"}) == "api"
  assert resolve_xai_auth_mode(environ={**env, "XAI_AUTH_MODE": "oauth"}) == "oauth"


def test_device_code_login_discovers_polls_and_persists(tmp_path: Path) -> None:
  requests: list[httpx.Request] = []

  def handler(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    if request.url.path.endswith("openid-configuration"):
      return httpx.Response(
        200,
        json={
          "device_authorization_endpoint": "https://auth.x.ai/oauth2/device/code",
          "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
      )
    if request.url.path.endswith("/device/code"):
      return httpx.Response(
        200,
        json={
          "device_code": "device-1",
          "user_code": "ABCD-1234",
          "verification_uri": "https://accounts.x.ai/oauth2/device",
          "verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234",
          "expires_in": 900,
          "interval": 1,
        },
      )
    return httpx.Response(
      200,
      json={
        "access_token": "access-1",
        "refresh_token": "refresh-1",
        "expires_in": 3600,
      },
    )

  seen_codes = []
  client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
  try:
    record, path = _run(
      login_xai_device_code(
        config={"auth_store_path": str(tmp_path / "oauth.json")},
        on_verification=lambda code: seen_codes.append(code),
        client=client,
      )
    )
  finally:
    _run(client.aclose())

  assert path == tmp_path / "oauth.json"
  assert record["refresh_token"] == "refresh-1"
  assert seen_codes[0].user_code == "ABCD-1234"
  device_form = parse_qs(requests[1].content.decode())
  assert device_form["client_id"] == [DEFAULT_XAI_OAUTH_CLIENT_ID]
  assert device_form["scope"] == [DEFAULT_XAI_OAUTH_SCOPE]
  token_form = parse_qs(requests[2].content.decode())
  assert token_form["device_code"] == ["device-1"]
  assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_refresh_rotates_tokens_and_updates_store(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  old = _record()
  save_xai_token_record(settings.store_path, old)

  def handler(request: httpx.Request) -> httpx.Response:
    form = parse_qs(request.content.decode())
    assert form["grant_type"] == ["refresh_token"]
    assert form["refresh_token"] == ["refresh-1"]
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
  try:
    refreshed = _run(refresh_xai_oauth_token(old, settings=settings, client=client))
  finally:
    _run(client.aclose())
  assert refreshed["access_token"] == "access-2"
  assert refreshed["refresh_token"] == "refresh-2"
  assert load_xai_token_record(settings.store_path)["refresh_token"] == "refresh-2"


def test_provider_refreshes_once_on_401_then_retries_response(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record())
  response_attempts = 0
  authorization_headers: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    nonlocal response_attempts
    if request.url.path.endswith("/oauth2/token"):
      return httpx.Response(
        200,
        json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
      )
    response_attempts += 1
    authorization_headers.append(str(request.headers.get("authorization")))
    if response_attempts == 1:
      return httpx.Response(401, json={"error": {"message": "expired"}})
    payload = {
      "type": "response.completed",
      "response": {"status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    }
    return httpx.Response(200, text=f"data: {json.dumps(payload)}\n\n")

  provider = XAIProvider()
  config = {
    "auth_mode": "oauth",
    "auth_store_path": str(store),
    "_transport": httpx.MockTransport(handler),
  }
  client = provider.create_client(config)

  async def collect():
    try:
      return [event async for event in provider.stream(client, {"model": "grok-4.5", "stream": True})]
    finally:
      await provider.close_client(client)

  events = _run(collect())
  assert response_attempts == 2
  assert authorization_headers == ["Bearer access-1", "Bearer access-2"]
  assert events[-1].type == "message_end"
  assert load_xai_token_record(store)["refresh_token"] == "refresh-2"


def test_untrusted_discovery_override_is_rejected() -> None:
  with pytest.raises(ValueError, match="untrusted discovery URL"):
    resolve_xai_oauth_settings({"oauth_discovery_url": "https://evil.example/.well-known/openid"})
