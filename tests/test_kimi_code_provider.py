from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import time
from typing import Any, Iterator
import urllib.parse

import httpx
import pytest

from agent_gateway.auth import CredentialFailureKind, ProviderCredentialFailure
from agent_gateway.providers.api_key_store import upsert_api_key_record
from agent_gateway.providers.credential_pool import (
  ProviderCredentialPool,
  get_provider_credential_pool,
  store_write_lock,
)
from agent_gateway.providers.kimi_code import KimiCodeProvider
from agent_gateway.providers.kimi_code_oauth import (
  extract_kimi_account_identity,
  load_kimi_code_accounts,
  refresh_kimi_code_tokens,
  resolve_kimi_code_pool_candidates,
  save_kimi_code_token_record,
  upsert_kimi_code_token_record,
)
from model_authority.thinking import ThinkingLevel

_ENV_TOKEN_NAMES = (
  "KIMI_CODE_AUTH_TOKEN",
  "KIMI_CODE_API_KEY",
  "KIMI_API_KEY",
  "KIMI_CODE_AUTH_STORE_PATH",
  "KIMI_AUTH_STORE_PATH",
  "KIMI_CODE_API_KEY_STORE_PATH",
  "KIMI_CODE_OAUTH_HOST",
  "KIMI_OAUTH_HOST",
)


def _jwt(payload: dict[str, Any]) -> str:
  def segment(value: dict[str, Any]) -> str:
    raw = json.dumps(value).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

  return f"{segment({'alg': 'none', 'typ': 'JWT'})}.{segment(payload)}.signature"


def _seat(email: str, *, access_suffix: str = "a") -> dict[str, Any]:
  return {
    "access_token": _jwt({"email": email, "sub": f"sub-{email}", "n": access_suffix}),
    "refresh_token": f"refresh-{email}",
    "expires_at": time.time() + 3600,
    "source": "subscription",
  }


@pytest.fixture
def kimi_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  for name in _ENV_TOKEN_NAMES:
    monkeypatch.delenv(name, raising=False)
  pool = get_provider_credential_pool("kimi_code")
  pool.clear()
  try:
    yield tmp_path
  finally:
    pool.clear()


def test_extract_kimi_account_identity_prefers_email_then_user_id_then_sub() -> None:
  assert extract_kimi_account_identity(
    _jwt({"email": "seat@example.com", "user_id": "u-1", "sub": "s-1"})
  ) == "seat@example.com"
  assert extract_kimi_account_identity(_jwt({"user_id": "u-1", "sub": "s-1"})) == "u-1"
  assert extract_kimi_account_identity(_jwt({"sub": "s-1"})) == "s-1"


@pytest.mark.parametrize(
  "token",
  [
    "",
    "sk-kimi-opaque-token",
    "only.two",
    "a.!!!not-base64!!!.c",
    f"a.{base64.urlsafe_b64encode(b'not json').decode('ascii')}.c",
    _jwt({"unrelated": "claim"}),
  ],
)
def test_extract_kimi_account_identity_falls_back_to_empty_on_opaque_tokens(token: str) -> None:
  assert extract_kimi_account_identity(token) == ""


def test_two_kimi_accounts_enroll_without_overwriting_and_update_in_place(tmp_path: Path) -> None:
  store = tmp_path / "kimi_code" / "oauth.json"
  first = _seat("first@example.com")
  second = _seat("second@example.com")
  save_kimi_code_token_record(store, first)
  accounts = upsert_kimi_code_token_record(store, second)

  assert [a["identity"] for a in accounts] == ["first@example.com", "second@example.com"]
  on_disk = json.loads(store.read_text(encoding="utf-8"))["accounts"]
  assert [a["identity"] for a in on_disk] == ["first@example.com", "second@example.com"]
  assert on_disk[0]["refresh_token"] == "refresh-first@example.com"
  assert on_disk[1]["refresh_token"] == "refresh-second@example.com"

  reenrolled = _seat("first@example.com", access_suffix="b")
  reenrolled["refresh_token"] = "refresh-first-rotated"
  save_kimi_code_token_record(store, reenrolled)

  seats = load_kimi_code_accounts(store)
  assert [a["identity"] for a in seats] == ["first@example.com", "second@example.com"]
  assert seats[0]["access_token"] == reenrolled["access_token"]
  assert seats[0]["refresh_token"] == "refresh-first-rotated"
  assert seats[1]["access_token"] == second["access_token"]


def test_refresh_kimi_code_tokens_updates_matching_seat_in_place(tmp_path: Path) -> None:
  store = tmp_path / "kimi_code" / "oauth.json"
  first = _seat("first@example.com")
  second = _seat("second@example.com")
  save_kimi_code_token_record(store, first)
  save_kimi_code_token_record(store, second)
  refreshed_access = _jwt({"email": "first@example.com", "n": "refreshed"})
  seen: list[dict[str, str]] = []

  def handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/oauth/token"
    seen.append(dict(urllib.parse.parse_qsl(request.content.decode("utf-8"))))
    return httpx.Response(
      200,
      json={"access_token": refreshed_access, "expires_in": 7200, "token_type": "Bearer"},
    )

  record = load_kimi_code_accounts(store)[0]
  with httpx.Client(transport=httpx.MockTransport(handler)) as client:
    updated = refresh_kimi_code_tokens(record, store_path=store, client=client)

  assert seen == [{
    "grant_type": "refresh_token",
    "refresh_token": "refresh-first@example.com",
    "client_id": "17e5f671-d194-4dfb-9706-5516cb48c098",
  }]
  assert updated["access_token"] == refreshed_access
  assert updated["refresh_token"] == "refresh-first@example.com"
  seats = load_kimi_code_accounts(store)
  assert [a["identity"] for a in seats] == ["first@example.com", "second@example.com"]
  assert seats[0]["access_token"] == refreshed_access
  assert seats[1]["access_token"] == second["access_token"]


def _expired_seat(email: str) -> dict[str, Any]:
  seat = _seat(email)
  seat["expires_at"] = time.time() - 60
  return seat


def test_refresh_kimi_code_tokens_persists_rotated_refresh_token_with_one_post(tmp_path: Path) -> None:
  store = tmp_path / "kimi_code" / "oauth.json"
  save_kimi_code_token_record(store, _expired_seat("first@example.com"))
  refreshed_access = _jwt({"email": "first@example.com", "n": "refreshed"})
  seen: list[dict[str, str]] = []

  def handler(request: httpx.Request) -> httpx.Response:
    seen.append(dict(urllib.parse.parse_qsl(request.content.decode("utf-8"))))
    return httpx.Response(200, json={
      "access_token": refreshed_access,
      "refresh_token": "refresh-rotated",
      "expires_in": 7200,
    })

  record = load_kimi_code_accounts(store)[0]
  with httpx.Client(transport=httpx.MockTransport(handler)) as client:
    updated = refresh_kimi_code_tokens(record, store_path=store, client=client)

  assert [s["refresh_token"] for s in seen] == ["refresh-first@example.com"]
  assert updated["refresh_token"] == "refresh-rotated"
  (stored,) = load_kimi_code_accounts(store)
  assert stored["access_token"] == refreshed_access
  assert stored["refresh_token"] == "refresh-rotated"


def test_refresh_kimi_code_tokens_returns_seat_another_process_already_refreshed(tmp_path: Path) -> None:
  store = tmp_path / "kimi_code" / "oauth.json"
  save_kimi_code_token_record(store, _expired_seat("first@example.com"))
  stale = load_kimi_code_accounts(store)[0]
  # Process B refreshed: new single-use refresh token, fresh access token.
  other = _seat("first@example.com", access_suffix="b")
  other["refresh_token"] = "refresh-rotated-by-b"
  save_kimi_code_token_record(store, other)

  def handler(request: httpx.Request) -> httpx.Response:
    pytest.fail("the spent refresh token must not be POSTed again")

  with httpx.Client(transport=httpx.MockTransport(handler)) as client:
    result = refresh_kimi_code_tokens(stale, store_path=store, client=client)

  assert result["access_token"] == other["access_token"]
  assert result["refresh_token"] == "refresh-rotated-by-b"
  (stored,) = load_kimi_code_accounts(store)
  assert stored["refresh_token"] == "refresh-rotated-by-b"


def test_refresh_kimi_code_tokens_posts_stored_refresh_token_when_stored_seat_also_expired(
  tmp_path: Path,
) -> None:
  store = tmp_path / "kimi_code" / "oauth.json"
  save_kimi_code_token_record(store, _expired_seat("first@example.com"))
  stale = load_kimi_code_accounts(store)[0]
  newer = _expired_seat("first@example.com")
  newer["refresh_token"] = "refresh-rotated-by-b"
  save_kimi_code_token_record(store, newer)
  seen: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    seen.append(dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))["refresh_token"])
    return httpx.Response(200, json={"access_token": "fresh-access", "expires_in": 7200})

  with httpx.Client(transport=httpx.MockTransport(handler)) as client:
    updated = refresh_kimi_code_tokens(stale, store_path=store, client=client)

  assert seen == ["refresh-rotated-by-b"]
  assert updated["refresh_token"] == "refresh-rotated-by-b"


def test_refresh_kimi_code_tokens_waits_for_store_write_lock_before_posting(tmp_path: Path) -> None:
  import threading

  store = tmp_path / "kimi_code" / "oauth.json"
  save_kimi_code_token_record(store, _expired_seat("first@example.com"))
  record = load_kimi_code_accounts(store)[0]
  locked = threading.Event()
  release = threading.Event()
  posted = threading.Event()
  done = threading.Event()
  errors: list[BaseException] = []

  def handler(_request: httpx.Request) -> httpx.Response:
    posted.set()
    return httpx.Response(200, json={"access_token": "fresh-access", "expires_in": 7200})

  def hold() -> None:
    with store_write_lock(store):
      locked.set()
      release.wait(timeout=5)

  def run_refresh() -> None:
    try:
      with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        refresh_kimi_code_tokens(record, store_path=store, client=client)
    except BaseException as exc:  # surfaced via assertion below
      errors.append(exc)
    finally:
      done.set()

  holder = threading.Thread(target=hold)
  holder.start()
  assert locked.wait(timeout=2)
  refresher = threading.Thread(target=run_refresh)
  refresher.start()
  assert not posted.wait(timeout=0.3)  # blocked behind the lock holder
  release.set()
  assert done.wait(timeout=5)
  holder.join(timeout=2)
  refresher.join(timeout=2)
  assert errors == []
  assert posted.is_set()
  assert load_kimi_code_accounts(store)[0]["access_token"] == "fresh-access"



def test_pool_candidates_order_user_oauth_user_api_org_oauth_org_api_then_env(kimi_env: Path) -> None:
  user_oauth = kimi_env / "users" / "42" / "kimi_code" / "oauth.json"
  user_api = kimi_env / "users" / "42" / "kimi_code" / "credentials.json"
  org_oauth = kimi_env / "kimi_code" / "oauth.json"
  org_api = kimi_env / "kimi_code" / "credentials.json"
  save_kimi_code_token_record(user_oauth, _seat("user-oauth@example.com"))
  upsert_api_key_record(
    user_api, {"identity": "user-api", "api_key": "sk-user-api"}, family="kimi_code"
  )
  save_kimi_code_token_record(org_oauth, _seat("org-oauth@example.com"))
  upsert_api_key_record(
    org_api, {"identity": "org-api", "api_key": "sk-org-api"}, family="kimi_code"
  )
  environ = {
    "USER_DATA_DIR": str(kimi_env),
    "KIMI_CODE_AUTH_TOKEN": "env-oauth-token",
    "KIMI_CODE_API_KEY": "env-code-api-key",
    "KIMI_API_KEY": "env-kimi-api-key",
  }

  candidates, _refresh = resolve_kimi_code_pool_candidates(user_scope="42", environ=environ)

  assert [(c.auth_mode, c.is_user_store, c.store_path) for c in candidates[:4]] == [
    ("oauth", True, user_oauth),
    ("api", True, user_api),
    ("oauth", False, org_oauth),
    ("api", False, org_api),
  ]
  assert [c.account_identity for c in candidates[:4]] == [
    "user-oauth@example.com", "user-api", "org-oauth@example.com", "org-api",
  ]
  env_candidates = candidates[4:]
  assert [(c.auth_mode, c.store_path, c.token) for c in env_candidates] == [
    ("oauth", None, "env-oauth-token"),
    ("api", None, "env-code-api-key"),
    ("api", None, "env-kimi-api-key"),
  ]
  assert [c.key.rsplit(":", 1)[0] for c in env_candidates] == [
    "env:KIMI_CODE_AUTH_TOKEN", "env:KIMI_CODE_API_KEY", "env:KIMI_API_KEY",
  ]
  assert all(c.provider == "kimi_code" for c in candidates)


def test_pool_candidates_without_user_scope_start_at_org_store(kimi_env: Path) -> None:
  save_kimi_code_token_record(
    kimi_env / "users" / "42" / "kimi_code" / "oauth.json", _seat("user-oauth@example.com")
  )
  save_kimi_code_token_record(kimi_env / "kimi_code" / "oauth.json", _seat("org-oauth@example.com"))

  candidates, _refresh = resolve_kimi_code_pool_candidates(environ={"USER_DATA_DIR": str(kimi_env)})

  assert [c.account_identity for c in candidates] == ["org-oauth@example.com"]
  assert not candidates[0].is_user_store


def test_cli_pool_store_target_kimi_code_uses_underscore_oauth_and_credentials(kimi_env: Path) -> None:
  from agent_gateway.cli import _resolve_pool_store_target

  oauth_path, family, keys = _resolve_pool_store_target(argparse.Namespace(
    provider="kimi-code", store=None, api_key=False, user=None,
  ))
  assert oauth_path == kimi_env / "kimi_code" / "oauth.json"
  assert family == "kimi_code"
  assert keys == ("access_token", "refresh_token")

  api_path, family, keys = _resolve_pool_store_target(argparse.Namespace(
    provider="kimi-code", store=None, api_key=True, user=None,
  ))
  assert api_path == kimi_env / "kimi_code" / "credentials.json"
  assert family == "kimi_code"
  assert keys == ("api_key",)


def test_pool_candidates_keep_both_stores_when_config_auth_store_is_oauth(kimi_env: Path) -> None:
  oauth = kimi_env / "kimi_code" / "oauth.json"
  api = kimi_env / "kimi_code" / "credentials.json"
  save_kimi_code_token_record(oauth, _seat("oauth@example.com"))
  upsert_api_key_record(api, {"identity": "api-seat", "api_key": "sk-api"}, family="kimi_code")

  candidates, _refresh = resolve_kimi_code_pool_candidates(
    {"auth_store_path": str(oauth)},
    environ={"USER_DATA_DIR": str(kimi_env)},
  )

  assert {(c.auth_mode, c.account_identity, c.store_path) for c in candidates} == {
    ("oauth", "oauth@example.com", oauth),
    ("api", "api-seat", api),
  }

  # The production rotation config names the selected seat's store under its
  # own mode's key (`api/credentials.py` get_kimi_code_config): an API seat
  # sets api_key_store_path, and the OAuth store must still resolve.
  candidates, _refresh = resolve_kimi_code_pool_candidates(
    {"api_key_store_path": str(api)},
    environ={"USER_DATA_DIR": str(kimi_env)},
  )

  assert {(c.auth_mode, c.account_identity, c.store_path) for c in candidates} == {
    ("oauth", "oauth@example.com", oauth),
    ("api", "api-seat", api),
  }

  # Rotating onto an API seat via the pool materializer (`next_credential`)
  # writes the API seat's path to `api_key_store_path` and preserves the OAuth
  # store path, so a subsequent re-resolve still sees both stores.
  api_candidate = next(c for c in candidates if c.auth_mode == "api")
  from agent_gateway.providers.kimi_code_oauth import _materialize_kimi_code_candidate

  rotated_cfg = _materialize_kimi_code_candidate(api_candidate, {"auth_store_path": str(oauth)})
  assert rotated_cfg["api_key_store_path"] == str(api)
  assert rotated_cfg["auth_store_path"] == str(oauth)
  candidates_after_api, _ = resolve_kimi_code_pool_candidates(
    rotated_cfg,
    environ={"USER_DATA_DIR": str(kimi_env)},
  )
  assert {(c.auth_mode, c.account_identity, c.store_path) for c in candidates_after_api} == {
    ("oauth", "oauth@example.com", oauth),
    ("api", "api-seat", api),
  }


def test_adapter_route_support_declares_kimi_code_route() -> None:
  support = KimiCodeProvider.adapter_route_support()
  assert support is not None
  assert support.adapter == "kimi_code.chat_completions"
  assert support.provider == "kimi_code"
  assert support.routes == frozenset({"kimi.code"})
  assert support.protocol_profiles == frozenset({"chat_completions.standard"})


@pytest.mark.parametrize(
  ("config", "expected_token"),
  [
    ({"auth_mode": "oauth", "auth_token": "oauth-token", "api_key": ""}, "oauth-token"),
    ({"auth_mode": "api", "auth_token": "", "api_key": "sk-api-key"}, "sk-api-key"),
  ],
)
def test_create_client_sends_bearer_token_and_kimi_cli_headers(
  kimi_env: Path, config: dict[str, Any], expected_token: str,
) -> None:
  provider = KimiCodeProvider()
  assert provider.has_active_credential(config)
  client = provider.create_client({"provider": "kimi_code", **config}, timeout=30.0)
  try:
    state = provider._client_state[client]
    headers = state["headers"]
    assert state["endpoint_url"] == "https://api.kimi.com/coding/v1/chat/completions"
    assert state["send_authorization"] is True
    assert headers["Authorization"] == f"Bearer {expected_token}"
    assert headers["User-Agent"] == "KimiCLI/1.0"
    assert headers["X-Msh-Platform"] == "kimi_cli"
    assert headers["X-Msh-Version"] == "1.0"
    for name in ("X-Msh-Device-Name", "X-Msh-Device-Model", "X-Msh-Device-Id"):
      assert headers[name]
  finally:
    asyncio.run(client.aclose())


def test_create_client_without_credential_names_the_login_command() -> None:
  provider = KimiCodeProvider()
  assert not provider.has_active_credential({"auth_mode": "oauth", "auth_token": ""})
  with pytest.raises(RuntimeError, match="agent auth login kimi-code"):
    provider.create_client({"provider": "kimi_code", "auth_mode": "oauth", "auth_token": ""})


def _request_params(provider: KimiCodeProvider, level: ThinkingLevel, max_tokens: int) -> dict[str, Any]:
  info = provider.get_model_info("k3")
  resolution = provider.resolve_effort(
    requested=level, model="k3", model_info=info, max_tokens=max_tokens,
  )
  return provider.build_request_params(
    model="k3",
    messages=[{"role": "user", "content": "hello"}],
    system_prompt="be brief",
    tools=[],
    max_tokens=max_tokens,
    thinking_level=level,
    effort_resolution=resolution,
  )


def test_build_request_params_encodes_enabled_thinking_effort_and_clamps_max_tokens() -> None:
  params = _request_params(KimiCodeProvider(), ThinkingLevel.HIGH, 500_000)
  assert params["model"] == "k3"
  assert params["thinking"] == {"type": "enabled", "effort": "high"}
  assert "reasoning_effort" not in params
  assert params["max_tokens"] == 131072


def test_build_request_params_encodes_disabled_thinking_when_effort_is_none() -> None:
  params = _request_params(KimiCodeProvider(), ThinkingLevel.NONE, 4096)
  assert params["thinking"] == {"type": "disabled"}
  assert params["max_tokens"] == 4096


@pytest.mark.parametrize("failure_kind", ["rate_limit", "auth"])
def test_next_credential_rotates_to_seat_two_and_parks_seat_one(
  kimi_env: Path, failure_kind: CredentialFailureKind,
) -> None:
  store = kimi_env / "kimi_code" / "oauth.json"
  first = _seat("first@example.com")
  second = _seat("second@example.com")
  save_kimi_code_token_record(store, first)
  save_kimi_code_token_record(store, second)
  pool: ProviderCredentialPool = get_provider_credential_pool("kimi_code")
  config = {
    "provider": "kimi_code",
    "auth_mode": "oauth",
    "auth_token": first["access_token"],
    "api_key": "",
    "account_identity": "first@example.com",
  }
  reset_at = time.time() + 600
  failure = ProviderCredentialFailure(
    provider="kimi_code", kind=failure_kind, status_code=429, reset_at=reset_at,
  )

  rotated = KimiCodeProvider().next_credential(config, failure)

  assert rotated is not None
  assert rotated["provider"] == "kimi_code"
  assert rotated["auth_mode"] == "oauth"
  assert rotated["auth_token"] == second["access_token"]
  assert rotated["account_identity"] == "second@example.com"
  assert rotated["refresh_token"] == "refresh-second@example.com"
  assert pool.blocked_until("first@example.com") == reset_at
  assert pool.blocked_until("second@example.com") == 0.0

  # Seat 1 stays parked, so a failure on seat 2 has no sibling left to serve.
  exhausted = KimiCodeProvider().next_credential(rotated, failure)
  assert exhausted is None
  assert pool.blocked_until("second@example.com") == reset_at


def test_next_credential_does_not_rotate_oauth_seat_on_billing_failure(kimi_env: Path) -> None:
  store = kimi_env / "kimi_code" / "oauth.json"
  first = _seat("first@example.com")
  save_kimi_code_token_record(store, first)
  save_kimi_code_token_record(store, _seat("second@example.com"))
  config = {
    "provider": "kimi_code",
    "auth_mode": "oauth",
    "auth_token": first["access_token"],
    "account_identity": "first@example.com",
  }
  failure = ProviderCredentialFailure(provider="kimi_code", kind="billing", status_code=402)
  assert KimiCodeProvider().next_credential(config, failure) is None


@pytest.mark.parametrize(
  ("status", "body", "kind"),
  [(401, "unauthorized", "auth"), (402, "payment required", "billing")],
)
def test_classify_credential_failure_maps_auth_and_billing_statuses(
  status: int, body: str, kind: str,
) -> None:
  request = httpx.Request("POST", "https://api.kimi.com/coding/v1/chat/completions")
  exc = httpx.HTTPStatusError(
    body, request=request, response=httpx.Response(status, text=body, request=request),
  )
  failure = KimiCodeProvider().classify_credential_failure(exc)
  assert failure is not None
  assert failure.provider == "kimi_code"
  assert failure.kind == kind
  assert failure.status_code == status
