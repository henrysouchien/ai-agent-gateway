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

from hank_model_transport.auth import CredentialFailureKind, ProviderCredentialFailure
from hank_model_transport.providers.api_key_store import upsert_api_key_record
from hank_model_transport.providers.credential_pool import (
  ProviderCredentialPool,
  get_provider_credential_pool,
  select_pool_candidate,
  store_write_lock,
)
from hank_model_transport.providers.kimi_code import KimiCodeProvider
from hank_model_transport.providers.kimi_code_oauth import (
  extract_kimi_account_identity,
  load_kimi_code_accounts,
  refresh_kimi_code_tokens,
  resolve_kimi_code_pool_candidates,
  save_kimi_code_token_record,
  upsert_kimi_code_token_record,
)
from model_authority.thinking import ThinkingLevel

# The hop's stamp on the Kimi pool config (`kimi_code_oauth`), spelled out so
# these tests run against the pre-fix owner and fail on behavior, not import.
_HOP_STAMP = "metered_api_key_admitted"
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


@pytest.fixture
def owner_scope(monkeypatch: pytest.MonkeyPatch) -> str:
  """Deployment owner roster; returns the owner's run scope, the only one an org OAuth seat serves."""
  monkeypatch.delenv("HANK_USER_RUNTIME_CREDENTIALS_FILE", raising=False)
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([{"slug": "operator", "risk_user_id": 1, "role": "owner"}]),
  )
  return "1"


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
  from hank_model_transport.providers.kimi_code_oauth import _materialize_kimi_code_candidate

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


def _enroll_every_kimi_source(kimi_env: Path) -> tuple[dict[str, Path], dict[str, str]]:
  stores = {
    "user_oauth": kimi_env / "users" / "42" / "kimi_code" / "oauth.json",
    "user_api": kimi_env / "users" / "42" / "kimi_code" / "credentials.json",
    "org_oauth": kimi_env / "kimi_code" / "oauth.json",
    "org_api": kimi_env / "kimi_code" / "credentials.json",
  }
  save_kimi_code_token_record(stores["user_oauth"], _seat("user-oauth@example.com"))
  upsert_api_key_record(
    stores["user_api"], {"identity": "user-api", "api_key": "sk-user-api"}, family="kimi_code"
  )
  save_kimi_code_token_record(stores["org_oauth"], _seat("org-oauth@example.com"))
  upsert_api_key_record(
    stores["org_api"], {"identity": "org-api", "api_key": "sk-org-api"}, family="kimi_code"
  )
  environ = {
    "USER_DATA_DIR": str(kimi_env),
    "KIMI_CODE_AUTH_TOKEN": "env-oauth-token",
    "KIMI_CODE_API_KEY": "env-code-api-key",
    "KIMI_API_KEY": "env-kimi-api-key",
  }
  return stores, environ


def test_hop_stamped_pool_keeps_oauth_seats_only(kimi_env: Path) -> None:
  """A metered key reached through a fallback hop is PAYG nobody chose (C1 06a4b3ba)."""
  stores, environ = _enroll_every_kimi_source(kimi_env)

  candidates, _refresh = resolve_kimi_code_pool_candidates(
    {_HOP_STAMP: False}, user_scope="42", environ=environ,
  )

  assert [(c.auth_mode, c.store_path, c.token) for c in candidates] == [
    ("oauth", stores["user_oauth"], _seat("user-oauth@example.com")["access_token"]),
    ("oauth", stores["org_oauth"], _seat("org-oauth@example.com")["access_token"]),
    ("oauth", None, "env-oauth-token"),
  ]


@pytest.mark.parametrize(
  "stamp",
  [{}, {_HOP_STAMP: True}],
  ids=["unstamped-named-selection", "explicitly-admitted"],
)
def test_named_selection_still_selects_its_metered_key(
  kimi_env: Path, stamp: dict[str, bool],
) -> None:
  stores, environ = _enroll_every_kimi_source(kimi_env)
  pool = get_provider_credential_pool("kimi_code")
  candidates, refresh = resolve_kimi_code_pool_candidates(stamp, user_scope="42", environ=environ)
  for candidate in candidates:
    if candidate.auth_mode == "oauth":
      pool.block(candidate.key, until=time.time() + 600)

  selected = select_pool_candidate(
    candidates, pool=pool, refresh_fn=refresh, user_scope="42", environ=environ,
  )

  assert selected is not None
  assert (selected.auth_mode, selected.store_path, selected.token) == (
    "api", stores["user_api"], "sk-user-api",
  )


@pytest.mark.parametrize("malformed", ["false", 0, 1, "no"])
def test_malformed_metered_stamp_refuses_instead_of_choosing_a_lane(
  kimi_env: Path, malformed: object,
) -> None:
  with pytest.raises(ValueError, match=_HOP_STAMP):
    resolve_kimi_code_pool_candidates(
      {_HOP_STAMP: malformed},
      environ={"USER_DATA_DIR": str(kimi_env)},
    )


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


def test_build_request_params_encodes_enabled_thinking_effort() -> None:
  params = _request_params(KimiCodeProvider(), ThinkingLevel.HIGH, 500_000)
  assert params["model"] == "k3"
  assert params["thinking"] == {"type": "enabled", "effort": "high"}
  assert "reasoning_effort" not in params


def test_build_request_params_encodes_disabled_thinking_when_effort_is_none() -> None:
  params = _request_params(KimiCodeProvider(), ThinkingLevel.NONE, 4096)
  assert params["thinking"] == {"type": "disabled"}
  assert params["max_tokens"] == 4096


@pytest.mark.parametrize("failure_kind", ["rate_limit", "auth"])
def test_next_credential_rotates_to_seat_two_and_parks_seat_one(
  kimi_env: Path, owner_scope: str, failure_kind: CredentialFailureKind,
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
    "auth_user_scope": owner_scope,
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


def _seat_with_metered_sibling(kimi_env: Path) -> dict[str, Any]:
  user_store = kimi_env / "users" / "42" / "kimi_code"
  seat = _seat("seat@example.com")
  save_kimi_code_token_record(user_store / "oauth.json", seat)
  upsert_api_key_record(
    user_store / "credentials.json", {"identity": "user-api", "api_key": "sk-user-api"},
    family="kimi_code",
  )
  return {
    "provider": "kimi_code",
    "auth_mode": "oauth",
    "auth_token": seat["access_token"],
    "api_key": "",
    "account_identity": "seat@example.com",
    "auth_user_scope": "42",
  }


def test_rotation_after_a_hop_never_lands_on_a_metered_key(kimi_env: Path) -> None:
  config = {
    **_seat_with_metered_sibling(kimi_env),
    _HOP_STAMP: False,
  }
  reset_at = time.time() + 600
  failure = ProviderCredentialFailure(
    provider="kimi_code", kind="rate_limit", status_code=429, reset_at=reset_at,
  )

  assert KimiCodeProvider().next_credential(config, failure) is None
  assert get_provider_credential_pool("kimi_code").blocked_until("seat@example.com") == reset_at


def test_rotation_on_a_named_kimi_selection_still_reaches_its_metered_key(kimi_env: Path) -> None:
  config = _seat_with_metered_sibling(kimi_env)
  failure = ProviderCredentialFailure(
    provider="kimi_code", kind="rate_limit", status_code=429, reset_at=time.time() + 600,
  )

  rotated = KimiCodeProvider().next_credential(config, failure)

  assert rotated is not None
  assert (rotated["auth_mode"], rotated["api_key"], rotated["auth_token"]) == (
    "api", "sk-user-api", "",
  )
  assert _HOP_STAMP not in rotated


def test_next_credential_does_not_rotate_oauth_seat_on_billing_failure(
  kimi_env: Path, owner_scope: str,
) -> None:
  store = kimi_env / "kimi_code" / "oauth.json"
  first = _seat("first@example.com")
  save_kimi_code_token_record(store, first)
  save_kimi_code_token_record(store, _seat("second@example.com"))
  config = {
    "provider": "kimi_code",
    "auth_mode": "oauth",
    "auth_token": first["access_token"],
    "account_identity": "first@example.com",
    "auth_user_scope": owner_scope,
  }
  failure = ProviderCredentialFailure(provider="kimi_code", kind="billing", status_code=402)
  assert KimiCodeProvider().next_credential(config, failure) is None


@pytest.mark.parametrize("scope", ["13", ""])
def test_next_credential_never_rotates_onto_org_oauth_seat_for_non_owner(
  kimi_env: Path, owner_scope: str, scope: str,
) -> None:
  store = kimi_env / "kimi_code" / "oauth.json"
  first = _seat("first@example.com")
  save_kimi_code_token_record(store, first)
  save_kimi_code_token_record(store, _seat("second@example.com"))
  config = {
    "provider": "kimi_code",
    "auth_mode": "oauth",
    "auth_token": first["access_token"],
    "account_identity": "first@example.com",
    "auth_user_scope": scope,
  }
  failure = ProviderCredentialFailure(
    provider="kimi_code", kind="rate_limit", status_code=429, reset_at=time.time() + 600,
  )
  assert KimiCodeProvider().next_credential(config, failure) is None


@pytest.mark.parametrize(
  ("status", "body", "kind"),
  [(401, "unauthorized", "auth"), (402, "payment required", "billing"), (403, "forbidden", "billing")],
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


@pytest.fixture
def kimi_monthly_limit_detail() -> str:
  fixture = Path(__file__).parent / "fixtures" / "kimi-code-monthly-usage-limit-sse.json"
  capture = json.loads(fixture.read_text(encoding="utf-8"))
  return capture["envelope"]["event"]["primary_failure"]["detail"]


@pytest.mark.parametrize(
  ("retry_after", "expected_reset"),
  [(None, None), ("Tue, 01 Jan 2030 00:00:00 GMT", 1893456000.0)],
)
def test_classify_code_less_monthly_429_from_retained_sse(
  kimi_monthly_limit_detail: str, retry_after: str | None, expected_reset: float | None,
) -> None:
  request = httpx.Request("POST", "https://api.kimi.com/coding/v1/chat/completions")
  # The SSE retained the formatted exception, not its upstream body/headers.
  # An empty response is the constructed no-code case, not a captured raw body.
  response = httpx.Response(
    429, request=request, headers={} if retry_after is None else {"retry-after": retry_after},
  )
  exc = httpx.HTTPStatusError(kimi_monthly_limit_detail, request=request, response=response)

  failure = KimiCodeProvider().classify_credential_failure(exc)

  assert failure is not None
  assert failure.provider == "kimi_code"
  assert failure.kind == "rate_limit"
  assert failure.status_code == 429
  assert failure.error_code is None
  assert failure.reset_at == expected_reset
  assert kimi_monthly_limit_detail in failure.message


@pytest.mark.parametrize("source", ["code", "error_code", "type", "response.code", "response.type"])
@pytest.mark.parametrize(
  ("code", "expected_kind"),
  [
    ("usage_limit", "rate_limit"),
    ("quota_exceeded", "rate_limit"),
    ("rate_limit_error", "rate_limit"),
    ("invalid_api_key", "billing"),
    ("unknown_provider_code", "billing"),
  ],
)
def test_monthly_text_does_not_override_structured_code_classification(
  kimi_monthly_limit_detail: str, source: str, code: str, expected_kind: str,
) -> None:
  request = httpx.Request("POST", "https://api.kimi.com/coding/v1/chat/completions")
  # These declarations are synthetic precedence cases, not upstream capture data.
  if source.startswith("response."):
    field = source.split(".")[1]
    response = httpx.Response(429, request=request, json={"error": {field: code}})
  else:
    response = httpx.Response(429, request=request)
  exc = httpx.HTTPStatusError(kimi_monthly_limit_detail, request=request, response=response)
  if not source.startswith("response."):
    setattr(exc, source, code)

  failure = KimiCodeProvider().classify_credential_failure(exc)

  assert failure is not None
  # Existing generic billing prose still applies to non-exhaustion codes; only
  # the established exhaustion vocabulary takes the rate-limit override.
  assert failure.kind == expected_kind
  assert failure.error_code == code
  assert failure.reset_at is None


@pytest.mark.parametrize(("status", "expected_kind"), [(400, "billing"), (401, "auth"), (403, "billing")])
def test_monthly_text_does_not_change_other_http_status_classification(
  kimi_monthly_limit_detail: str, status: int, expected_kind: str,
) -> None:
  request = httpx.Request("POST", "https://api.kimi.com/coding/v1/chat/completions")
  exc = httpx.HTTPStatusError(
    kimi_monthly_limit_detail, request=request, response=httpx.Response(status, request=request),
  )

  failure = KimiCodeProvider().classify_credential_failure(exc)

  assert failure is not None
  assert failure.kind == expected_kind
  assert failure.status_code == status
  assert failure.reset_at is None


def test_other_usage_limit_wording_keeps_existing_billing_classification() -> None:
  request = httpx.Request("POST", "https://api.kimi.com/coding/v1/chat/completions")
  exc = httpx.HTTPStatusError(
    "API usage limit exceeded", request=request, response=httpx.Response(429, request=request),
  )

  failure = KimiCodeProvider().classify_credential_failure(exc)

  assert failure is not None
  assert failure.kind == "billing"
  assert failure.reset_at is None


def _kimi_sse(reasoning_key: str) -> bytes:
  chunks = [
    {"id": "cmpl-kimi-1", "model": "k3", "choices": [{"delta": {reasoning_key: "weigh it"}}]},
    {"id": "cmpl-kimi-1", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
    {
      "id": "cmpl-kimi-1",
      "choices": [],
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 7,
        "completion_tokens_details": {"reasoning_tokens": 5},
      },
    },
  ]
  return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"


@pytest.mark.parametrize("reasoning_key", ["reasoning_content", "reasoning", "thinking"])
def test_stream_surfaces_reasoning_ids_headers_and_reasoning_tokens(
  kimi_env: Path, reasoning_key: str,
) -> None:
  def handle(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
      200,
      headers={
        "content-type": "text/event-stream",
        "x-request-id": "req-kimi-1",
        "x-ratelimit-remaining-requests": "99",
        "retry-after": "3",
      },
      content=_kimi_sse(reasoning_key),
    )

  async def collect() -> list[Any]:
    provider = KimiCodeProvider()
    client = provider.create_client({
      "provider": "kimi_code",
      "auth_mode": "api",
      "api_key": "sk-api-key",
      "_transport": httpx.MockTransport(handle),
    })
    try:
      return [event async for event in provider.stream(client, {"model": "k3"})]
    finally:
      await provider.close_client(client)

  events = asyncio.run(collect())
  start = next(event for event in events if event.type == "message_start")
  end = next(event for event in events if event.type == "message_end")
  usage = next(event for event in events if event.type == "usage_update")

  assert [event.thinking_text for event in events if event.type == "thinking_delta"] == ["weigh it"]
  assert next(event for event in events if event.type == "thinking_end").raw_block == {
    "type": "thinking", "thinking": "weigh it",
  }
  assert start.provider_response_id == "cmpl-kimi-1"
  assert start.provider_request_id == "req-kimi-1"
  assert start.rate_limit == {"x-ratelimit-remaining-requests": "99", "retry-after": "3"}
  assert end.provider_response_id == "cmpl-kimi-1"
  assert usage.reasoning_tokens == 5
