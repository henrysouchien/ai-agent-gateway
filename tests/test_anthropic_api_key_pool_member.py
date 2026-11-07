from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Literal

import pytest

from hank_model_transport.auth import ProviderCredentialFailure
from hank_model_transport.providers import anthropic as anthropic_provider
from hank_model_transport.capability_execution import CapabilityExecutionResolver
from hank_model_transport.providers import anthropic_oauth
from hank_model_transport.providers.anthropic_quota import Cooldown, record_cooldown
from model_authority.key_pool import fingerprint, load_key_pool, pool_config_path, pool_state_path
from model_authority.spend_ledger import SpendLedgerUnavailable
from model_authority.spend_settings import SpendBreach, SpendSettingsError
from hank_model_transport.providers.anthropic import AnthropicProvider
from hank_model_transport.providers.anthropic_oauth import (
  ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES,
  ANTHROPIC_DEPLOYMENT_API_KEY_FIELD,
  ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD,
  ANTHROPIC_USER_SCOPE_FIELD,
  AnthropicCredential,
  AnthropicOAuthError,
  AnthropicSourceError,
  anthropic_pool_config,
  anthropic_credential_blocked_until,
  anthropic_generation_auth_config,
  enroll_anthropic_setup_token,
  register_anthropic_static_credential,
  resolve_anthropic_auth_store_path,
  resolve_anthropic_credentials,
  select_anthropic_credential_source,
)
from hank_model_transport.runtime_helpers import merge_refreshed_auth_config
from gateway_test_support.capability_execution_test_support import stub_bound_capability_execution


_API_KEY = "sk-ant-api03-synthetic-pool-key"
_SEAT_A = "sk-ant-oat01-synthetic-setup-a"
_SEAT_B = "sk-ant-oat01-synthetic-setup-b"
_KEY_IDENTITY = "api-key-" + hashlib.sha256(_API_KEY.encode("utf-8")).hexdigest()[:16]
_RESET_AT = 2_000.0


class _Sidecar:
  """Cooldown access through the shared selection sidecar, the one limiter store."""

  @staticmethod
  def block(credential: AnthropicCredential, *, until: float) -> None:
    record_cooldown(
      credential.sidecar_store_path, credential.locator, Cooldown(scope="shared", until=until, reason="rate_limit"),
    )

  @staticmethod
  def blocked_until(credential: AnthropicCredential) -> float:
    return anthropic_credential_blocked_until(credential)


def _selected_source(credential: AnthropicCredential) -> dict[str, object]:
  """The `credential_source` a selection-issued binding carries for this member."""
  return {**credential.source_metadata(), "automatic": True}


@pytest.fixture
def anthropic_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  # Product imports need the checkout root even in package-only pytest runs.
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
  credentials = importlib.import_module("api.credentials")
  pool = _Sidecar()
  for name in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES:
    monkeypatch.delenv(name, raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_MODE", raising=False)
  monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
  monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "user-data"))
  monkeypatch.setenv("HANK_KEY_POOL_DIR", str(tmp_path / "key-pool"))
  monkeypatch.setenv("HANK_KEY_POOL_STATE_DIR", str(tmp_path / "key-pool-state" / "credentials"))
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))
  monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_SEAT_A, _SEAT_B]))
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([
      {
        "key": "test-owner-cli-key",
        "channel": "cli",
        "slug": "henry",
        "email": "henry@example.com",
        "risk_user_id": 1,
        "role": "owner",
      },
      {
        "key": "test-invite-mcp-key",
        "channel": "mcp",
        "slug": "invitee",
        "email": "invitee@example.com",
        "risk_user_id": 13,
        "role": "invite",
      },
    ]),
  )
  monkeypatch.setattr(anthropic_provider.time, "time", lambda: 1_000.0)
  return pool, credentials, tmp_path / "oauth.json"


def _gateway_sources(credentials, *, user_scope: object = 1, call_class: str = "agent"):
  config = credentials.get_anthropic_config(user_scope=user_scope, call_class=call_class)
  return resolve_anthropic_credentials(
    anthropic_pool_config(config), user_scope=user_scope, call_class=call_class,
  )


def test_setup_tokens_precede_payg_key_with_exact_kind_source_and_identity(
  anthropic_pool,
) -> None:
  _pool, credentials, _store = anthropic_pool
  sources = _gateway_sources(credentials)
  assert sources.tokens == (_SEAT_A, _SEAT_B, _API_KEY)
  assert len(sources.credentials) == 3
  seat_a, seat_b, key = sources.credentials
  assert (seat_a.tier, seat_b.tier, key.tier) == ("org", "org", "payg")
  assert key.record is not None
  assert key.record.credential_kind == "api_key"
  assert key.record.source == "env:ANTHROPIC_API_KEY"
  assert key.record.identity == _KEY_IDENTITY
  assert key.key == _KEY_IDENTITY
  assert key.generation_ref
  for seat in (seat_a, seat_b):
    assert seat.record is not None
    assert seat.record.credential_kind == "auth_token"
    assert seat.record.source != key.record.source
    assert seat.key != key.key
  assert sum(
    member.record is not None and member.record.credential_kind == "api_key"
    for member in sources.credentials
  ) == 1


def test_programmatic_calls_only_use_payg_even_with_unparked_oauth_seats(anthropic_pool) -> None:
  pool, credentials, store = anthropic_pool
  enroll_anthropic_setup_token(
    "sk-ant-oat01-user-own-account",
    store_path=resolve_anthropic_auth_store_path(user_scope=1),
  )
  enroll_anthropic_setup_token("sk-ant-oat01-org-account", store_path=store)
  agent_sources = _gateway_sources(credentials, user_scope=1)
  assert all(pool.blocked_until(member) == 0.0 for member in agent_sources.credentials)

  config = credentials.get_anthropic_config(user_scope=1, call_class="programmatic")
  assert config["call_class"] == "programmatic"
  assert config["auth_mode"] == "api"
  assert config["api_key"] == _API_KEY
  assert config["auth_token"] == ""
  pool_config = {ANTHROPIC_DEPLOYMENT_API_KEY_FIELD: config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD]}
  sources = resolve_anthropic_credentials(
    pool_config, user_scope=1, call_class="programmatic",
  )
  assert sources.tokens == (_API_KEY,)
  assert sources.credentials[0].tier == "payg"
  assert config["credential_source"] == _selected_source(sources.credentials[0])
  assert resolve_anthropic_credentials({
    **pool_config, "call_class": "programmatic",
  }, user_scope=1).tokens == (_API_KEY,)


@pytest.mark.parametrize("call_class", ["invalid", ""])
def test_unknown_or_empty_call_class_is_rejected(anthropic_pool, call_class: str) -> None:
  _pool, credentials, _store = anthropic_pool
  with pytest.raises(ValueError, match="call_class"):
    credentials.get_anthropic_config(user_scope=1, call_class=call_class)
  with pytest.raises(ValueError, match="call_class"):
    resolve_anthropic_credentials({}, call_class=call_class)
  with pytest.raises(ValueError, match="call_class"):
    resolve_anthropic_credentials({"call_class": call_class})


@pytest.mark.parametrize("legacy_mode", [None, "api", "oauth"])
def test_config_derives_oauth_mode_from_first_member_not_legacy_mode(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch, legacy_mode: str | None,
) -> None:
  pool, credentials, _store = anthropic_pool
  if legacy_mode is not None:
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", legacy_mode)
  config = credentials.get_anthropic_config(user_scope=1)
  assert config["auth_mode"] == "oauth"
  assert config["auth_token"] == _SEAT_A
  assert config["api_key"] == ""
  seat_a, seat_b, key = _gateway_sources(credentials, user_scope=1).credentials
  assert config["credential_source"] == _selected_source(seat_a)

  pool.block(seat_a, until=_RESET_AT)
  seat_config = credentials.get_anthropic_config(user_scope=1)
  assert seat_config["auth_mode"] == "oauth"
  assert seat_config["auth_token"] == _SEAT_B
  assert seat_config["api_key"] == ""
  assert seat_config["credential_source"] == _selected_source(seat_b)

  pool.block(seat_b, until=_RESET_AT)
  key_config = credentials.get_anthropic_config(user_scope=1)
  assert key_config["auth_mode"] == "api"
  assert key_config["api_key"] == _API_KEY
  assert key_config["auth_token"] == ""
  assert key_config["credential_source"] == _selected_source(key)


@pytest.mark.parametrize("kind", ["rate_limit", "auth"])
def test_seat_failure_parks_until_reset_and_rotates_next_oauth_seat(
  anthropic_pool, kind: Literal["rate_limit", "auth"],
) -> None:
  pool, credentials, _store = anthropic_pool
  config = credentials.get_anthropic_config(user_scope=1)
  seat_a = _gateway_sources(credentials, user_scope=1).credentials[0]
  failure = ProviderCredentialFailure(
    provider="anthropic", kind=kind,
    status_code=429 if kind == "rate_limit" else 403, reset_at=_RESET_AT,
  )
  rotated = AnthropicProvider().next_credential(config, failure)
  assert rotated is not None
  assert rotated["auth_mode"] == "oauth"
  assert rotated["auth_token"] == _SEAT_B
  assert rotated["api_key"] == ""
  assert rotated[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD]
  merged = merge_refreshed_auth_config(config, rotated)
  assert merged[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD]
  later_pool = resolve_anthropic_credentials({
    name: value for name, value in merged.items() if name != "credential_source"
  })
  assert _API_KEY in later_pool.tokens
  sources = _gateway_sources(credentials, user_scope=1)
  assert rotated["credential_source"] == _selected_source(sources.credentials[1])
  assert rotated["credential_source"] != config["credential_source"]
  assert pool.blocked_until(seat_a) == _RESET_AT
  assert pool.blocked_until(sources.credentials[1]) == 0.0
  before = select_anthropic_credential_source(sources, now=_RESET_AT - 1)
  at_reset = select_anthropic_credential_source(sources, now=_RESET_AT)
  assert before is not None and before.token == _SEAT_B
  assert at_reset is not None and at_reset.token == _SEAT_A


@pytest.mark.parametrize("kind", ["rate_limit", "billing", "auth"])
def test_bound_payg_key_failure_parks_key_and_returns_to_oauth_pool(
  anthropic_pool, kind: Literal["rate_limit", "billing", "auth"],
) -> None:
  pool, credentials, _store = anthropic_pool
  base_config = credentials.get_anthropic_config(user_scope=1)
  seat_a, _seat_b, key = _gateway_sources(credentials, user_scope=1).credentials
  config = anthropic_generation_auth_config(
    key.generation_ref, user_scope=1, base_config=base_config,
  )
  rotated = AnthropicProvider().next_credential(config, ProviderCredentialFailure(
    provider="anthropic", kind=kind,
    status_code=429 if kind == "rate_limit" else 403, reset_at=_RESET_AT,
  ))
  assert rotated is not None
  assert rotated["auth_mode"] == "oauth"
  assert rotated["auth_token"] == _SEAT_A
  assert rotated["api_key"] == ""
  assert rotated["credential_source"] == _selected_source(seat_a)
  assert rotated[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == key.generation_ref
  assert pool.blocked_until(key) == _RESET_AT
  assert pool.blocked_until(seat_a) == 0.0


def test_oauth_seat_failure_falls_through_to_api_key_only_after_both_seats_cool_down(anthropic_pool) -> None:
  pool, credentials, _store = anthropic_pool
  sources = _gateway_sources(credentials, user_scope=1)
  seat_a, seat_b, key = sources.credentials
  config = credentials.get_anthropic_config(user_scope=1)
  assert config["auth_token"] == _SEAT_A
  pool.block(seat_b, until=_RESET_AT)
  rotated = AnthropicProvider().next_credential(config, ProviderCredentialFailure(
    provider="anthropic", kind="rate_limit", status_code=429, reset_at=_RESET_AT,
  ))
  assert rotated is not None
  assert rotated["auth_mode"] == "api"
  assert rotated["api_key"] == _API_KEY
  assert rotated["auth_token"] == ""
  assert rotated["credential_source"] == _selected_source(key)
  assert pool.blocked_until(seat_a) == _RESET_AT
  assert pool.blocked_until(seat_b) == _RESET_AT
  assert pool.blocked_until(key) == 0.0
  at_reset = select_anthropic_credential_source(sources, now=_RESET_AT)
  assert at_reset is not None and at_reset.token == _SEAT_A


def test_absent_api_key_preserves_generation_but_empty_retires_it(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  _pool, credentials, store = anthropic_pool
  config = credentials.get_anthropic_config(user_scope=1)
  key = _gateway_sources(credentials, user_scope=1).credentials[-1]
  config = anthropic_generation_auth_config(
    key.generation_ref, user_scope=1, base_config=config,
  )
  original_bytes = store.read_bytes()
  monkeypatch.delenv("ANTHROPIC_API_KEY")
  absent_config = credentials.get_anthropic_config(user_scope=1)
  assert ANTHROPIC_DEPLOYMENT_API_KEY_FIELD not in absent_config
  absent_sources = _gateway_sources(credentials, user_scope=1)
  assert _API_KEY not in absent_sources.tokens
  assert absent_sources.tokens == (_SEAT_A, _SEAT_B)
  assert store.read_bytes() == original_bytes
  # Bound gateway material explicitly retains membership even without ambient
  # visibility; absence does not retire the stored generation.
  assert resolve_anthropic_credentials(config).credentials[0].generation_ref == key.generation_ref
  exact_config = anthropic_generation_auth_config(
    key.generation_ref, user_scope=1, base_config=config,
  )
  assert exact_config["api_key"] == _API_KEY
  assert exact_config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == key.generation_ref
  with pytest.raises(AnthropicSourceError) as absent_exc:
    resolve_anthropic_credentials({"credential_source": key.source_metadata()})
  assert absent_exc.value.reason == "unobservable"
  assert store.read_bytes() == original_bytes
  monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
  exact = resolve_anthropic_credentials(config)
  assert exact.tokens == (_API_KEY,)
  assert exact.credentials[0].generation_ref == key.generation_ref
  assert store.read_bytes() == original_bytes

  monkeypatch.setenv("ANTHROPIC_API_KEY", "")
  empty_sources = _gateway_sources(credentials, user_scope=1)
  assert empty_sources.tokens == (_SEAT_A, _SEAT_B)
  assert store.read_bytes() != original_bytes
  retired_bytes = store.read_bytes()
  with pytest.raises(AnthropicSourceError) as exc:
    resolve_anthropic_credentials(config)
  assert exc.value.reason == "removed"
  assert store.read_bytes() == retired_bytes
  rotated = AnthropicProvider().next_credential(config, ProviderCredentialFailure(
    provider="anthropic", kind="rate_limit", status_code=429, reset_at=_RESET_AT,
  ))
  assert rotated is not None and rotated["auth_token"] == _SEAT_A
  assert rotated[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == key.generation_ref


def test_exact_generation_lookup_is_read_only_for_key_and_seat(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  _pool, credentials, store = anthropic_pool
  sources = _gateway_sources(credentials, user_scope=1)
  deployment_key_ref = sources.credentials[-1].generation_ref
  original_bytes = store.read_bytes()
  monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-unobserved-new-seat")
  for member in sources.credentials:
    exact = resolve_anthropic_credentials({
      "credential_source": member.source_metadata(),
      ANTHROPIC_DEPLOYMENT_API_KEY_FIELD: deployment_key_ref,
      ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD: True,
    })
    assert exact.tokens == (member.token,)
    assert exact.credentials[0].source_metadata() == member.source_metadata()
    assert len(exact.credentials) == 1
    assert store.read_bytes() == original_bytes


def test_rotating_last_seat_to_payg_changes_request_headers_without_network(anthropic_pool) -> None:
  pytest.importorskip("anthropic")
  httpx2 = pytest.importorskip("httpx2")
  pool, credentials, _store = anthropic_pool
  provider = AnthropicProvider()
  config = credentials.get_anthropic_config(user_scope=1)
  seat_b = _gateway_sources(credentials, user_scope=1).credentials[1]
  pool.block(seat_b, until=_RESET_AT)
  rotated = provider.next_credential(config, ProviderCredentialFailure(
    provider="anthropic", kind="rate_limit", status_code=429, reset_at=_RESET_AT,
  ))
  assert rotated is not None
  requests = []

  def handle_request(request):
    requests.append(request)
    return httpx2.Response(200, json={
      "id": "msg_synthetic", "type": "message", "role": "assistant",
      "model": "claude-sonnet-5", "content": [{"type": "text", "text": "hello"}],
      "stop_reason": "end_turn", "stop_sequence": None,
      "usage": {"input_tokens": 1, "output_tokens": 1},
    })

  async def send_requests():
    for bound in (config, merge_refreshed_auth_config(config, rotated)):
      client = provider.create_client(bound)
      try:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle_request)) as http_client:
          async with client.with_options(http_client=http_client) as local_client:
            await local_client.messages.create(
              model="claude-sonnet-5", max_tokens=16,
              messages=[{"role": "user", "content": "hello"}],
            )
      finally:
        await provider.close_client(client)

  asyncio.run(send_requests())
  assert len(requests) == 2
  assert requests[0].headers.get_list("authorization") == [f"Bearer {_SEAT_A}"]
  assert "x-api-key" not in requests[0].headers
  assert requests[1].headers.get_list("x-api-key") == [_API_KEY]
  assert "authorization" not in requests[1].headers


def test_ambient_api_key_is_not_a_pool_input_even_after_gateway_enrollment(anthropic_pool) -> None:
  _pool, credentials, _store = anthropic_pool
  assert resolve_anthropic_credentials({ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD: True}).tokens == (_SEAT_A, _SEAT_B)
  credentials.get_anthropic_config(user_scope=1)
  assert resolve_anthropic_credentials({ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD: True}).tokens == (_SEAT_A, _SEAT_B)
  rotated = AnthropicProvider().next_credential(
    {"auth_mode": "oauth", "auth_token": _SEAT_A, ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD: True},
    ProviderCredentialFailure(
      provider="anthropic", kind="rate_limit", status_code=429, reset_at=_RESET_AT,
    ),
  )
  assert rotated is not None and rotated["auth_token"] == _SEAT_B
  assert rotated["api_key"] == ""
  assert ANTHROPIC_DEPLOYMENT_API_KEY_FIELD not in rotated


def test_child_pool_environment_names_exclude_operator_api_key() -> None:
  assert "ANTHROPIC_API_KEY" not in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES


def test_gateway_pool_ranks_user_accounts_then_setup_then_org_then_payg(anthropic_pool) -> None:
  _pool, credentials, store = anthropic_pool
  user_token = "sk-ant-oat01-user-own-account"
  org_token = "sk-ant-oat01-org-account"
  enroll_anthropic_setup_token(
    user_token, store_path=resolve_anthropic_auth_store_path(user_scope=1),
  )
  enroll_anthropic_setup_token(
    "sk-ant-oat01-other-user-account",
    store_path=resolve_anthropic_auth_store_path(user_scope=2),
  )
  enroll_anthropic_setup_token(org_token, store_path=store)
  sources = _gateway_sources(credentials, user_scope=1)
  assert sources.tokens == (user_token, _SEAT_A, _SEAT_B, org_token, _API_KEY)
  assert tuple(member.tier for member in sources.credentials) == ("private", "org", "org", "org", "payg")
  config = credentials.get_anthropic_config(user_scope=1)
  assert config["auth_mode"] == "oauth"
  assert config["auth_token"] == user_token
  assert config["credential_source"] == _selected_source(sources.credentials[0])
  # Non-owner (user 2 with own store, user 13 invite, user 3 with no store),
  # unscoped (None), and ownerless environments never admit operator seats.
  assert _gateway_sources(credentials, user_scope=2).tokens == (
    "sk-ant-oat01-other-user-account", _API_KEY,
  )
  assert _gateway_sources(credentials, user_scope=13).tokens == (_API_KEY,)
  assert _gateway_sources(credentials, user_scope=3).tokens == (_API_KEY,)
  assert _gateway_sources(credentials, user_scope=None).tokens == (_API_KEY,)
  invite_cfg = credentials.get_anthropic_config(user_scope=13)
  assert invite_cfg["auth_mode"] == "api"
  assert invite_cfg["api_key"] == _API_KEY
  assert invite_cfg["auth_token"] == ""
  with pytest.raises(AnthropicSourceError) as seat_denied:
    anthropic_generation_auth_config(
      sources.credentials[1].generation_ref, user_scope=13, base_config=invite_cfg,
    )
  assert seat_denied.value.reason == "unobservable"
  bound_sources = resolve_anthropic_credentials({
    ANTHROPIC_DEPLOYMENT_API_KEY_FIELD: config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD],
    "auth_token": _SEAT_B,
    ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD: True,
  }, user_scope=1)
  assert bound_sources.tokens == (_SEAT_B, user_token, _SEAT_A, org_token, _API_KEY)


def test_without_configured_owner_operator_seats_are_admitted_to_nobody(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  _pool, credentials, _store = anthropic_pool
  monkeypatch.delenv("GATEWAY_USER_KEYS", raising=False)
  for scope in (1, 13, None):
    cfg = credentials.get_anthropic_config(user_scope=scope)
    assert cfg["auth_mode"] == "api"
    assert cfg["api_key"] == _API_KEY
    assert cfg["auth_token"] == ""
    assert _gateway_sources(credentials, user_scope=scope).tokens == (_API_KEY,)


def test_unstamped_web_user_byok_config_draws_no_operator_seat_on_any_placement(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  seat = "sk-ant-oat01-synthetic-operator-seat"
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([seat]))
  config = {
    "provider": "anthropic", "billing_mode": "user",
    "auth_mode": "api", "api_key": "sk-ant-api03-userbyok",
  }
  for placement in ("local", "hosted_external"):
    sources = resolve_anthropic_credentials(config)
    assert seat not in sources.tokens
    assert seat not in [
      member.token for member in sources.credentials if member.auth_mechanism == "oauth_bearer"
    ]
    selected = select_anthropic_credential_source(sources, placement=placement)
    assert selected is None or selected.token != seat


def test_operator_seats_refused_flag_draws_no_operator_seat_on_any_placement(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  seat = "sk-ant-oat01-synthetic-operator-seat"
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([seat]))
  config = {ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD: False}
  for placement in ("local", "hosted_external"):
    sources = resolve_anthropic_credentials(config)
    assert seat not in sources.tokens
    assert seat not in [
      member.token for member in sources.credentials if member.auth_mechanism == "oauth_bearer"
    ]
    selected = select_anthropic_credential_source(sources, placement=placement)
    assert selected is None or selected.token != seat


def test_operator_seats_admitted_owner_config_draws_and_selects_seat_on_any_placement(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  seat = "sk-ant-oat01-synthetic-operator-seat"
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([seat]))
  config = {ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD: True}
  for placement in ("local", "hosted_external"):
    sources = resolve_anthropic_credentials(config)
    assert sources.tokens == (seat,)
    selected = select_anthropic_credential_source(sources, placement=placement)
    assert selected is not None and selected.token == seat


def test_gateway_hop_rebind_preserves_deployment_key_membership(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  pool, credentials, _store = anthropic_pool
  config = credentials.get_anthropic_config(user_scope=1)
  seat_a, seat_b, key = _gateway_sources(credentials, user_scope=1).credentials
  execution = stub_bound_capability_execution(
    provider=AnthropicProvider(), model="claude-sonnet-5", effort="none",
    auth_config=config, credential_ref=seat_a.generation_ref,
  )
  pool.block(seat_a, until=_RESET_AT)
  rebound = CapabilityExecutionResolver._rebind_unparked_account(execution)
  assert rebound is not None
  assert rebound.auth_config["auth_token"] == _SEAT_B
  assert rebound.auth_config["api_key"] == ""
  assert rebound.auth_config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == key.generation_ref
  pool.block(seat_b, until=_RESET_AT + 1_000.0)
  payg = CapabilityExecutionResolver._rebind_unparked_account(rebound)
  assert payg is not None
  assert payg.auth_config["api_key"] == _API_KEY
  assert payg.auth_config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == key.generation_ref
  monkeypatch.setattr(anthropic_provider.time, "time", lambda: _RESET_AT + 1.0)
  returned = CapabilityExecutionResolver._rebind_unparked_account(payg)
  assert returned is not None
  assert returned.auth_config["auth_token"] == _SEAT_A
  assert returned.auth_config["api_key"] == ""
  assert returned.auth_config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == key.generation_ref


def test_non_gateway_hop_never_admits_ambient_deployment_key(anthropic_pool) -> None:
  pool, credentials, _store = anthropic_pool
  credentials.get_anthropic_config(user_scope=1)
  sources = resolve_anthropic_credentials({ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD: True})
  seat = sources.credentials[0]
  execution = stub_bound_capability_execution(
    provider=AnthropicProvider(), model="claude-sonnet-5", effort="none",
    auth_config={
      "auth_mode": "oauth", "auth_token": seat.token,
      "credential_source": seat.source_metadata(),
      # SEAT-SERVES-RUN-IFF-OWNER-EQUALS-RUN-OWNER: preserve the issued owner's admission.
      ANTHROPIC_OPERATOR_SEATS_ADMITTED_FIELD: True,
      ANTHROPIC_USER_SCOPE_FIELD: "1",
    },
    credential_ref=seat.generation_ref,
  )
  for member in sources.credentials:
    pool.block(member, until=_RESET_AT)
  assert CapabilityExecutionResolver._rebind_unparked_account(execution) is None


_DRAINED_KEY = "sk-ant-api03-synthetic-drained-key"
_USER_KEY = "sk-ant-usr-synthetic-key-without-workspace"
_HEALTHY_KEY = "sk-ant-api03-synthetic-healthy-key"
_SECOND_HEALTHY_KEY = "sk-ant-api03-synthetic-second-healthy-key"
_WORKSPACE_ID = "wrkspc_test_123"


def _write_shared_pool_file(keys: list[dict[str, str]]) -> None:
  path = pool_config_path("anthropic")
  assert path is not None and not path.exists()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps({"schema": "hank-key-pool/1", "provider": "anthropic", "keys": keys}),
    encoding="utf-8",
  )


def _shared_state_status(key: str) -> str:
  state = json.loads(pool_state_path("anthropic").read_text(encoding="utf-8"))
  return state["keys"][fingerprint(key)]["status"]


def test_gateway_config_skips_out_of_credit_and_unusable_keys_from_shared_key_pool(
  anthropic_pool,
) -> None:
  pytest.importorskip("anthropic")
  _pool, credentials, _store = anthropic_pool
  _write_shared_pool_file([
    {"key": _DRAINED_KEY},
    {"key": _USER_KEY},
    {"key": _HEALTHY_KEY, "workspace_id": _WORKSPACE_ID},
  ])
  load_key_pool(
    "anthropic", env_var="ANTHROPIC_API_KEY", repo="ai-excel-addin",
  ).mark_exhausted(fingerprint(_DRAINED_KEY), reason="credit_balance_too_low")
  assert _shared_state_status(_DRAINED_KEY) == "out_of_credit"
  assert _shared_state_status(_USER_KEY) == "unusable_config"

  config = credentials.get_anthropic_config(call_class="programmatic")

  assert config["auth_mode"] == "api"
  assert config["api_key"] == _HEALTHY_KEY
  assert config["workspace_id"] == _WORKSPACE_ID
  provider = AnthropicProvider()
  client = provider.create_client(config)
  try:
    assert client.default_headers.get("anthropic-workspace-id") == _WORKSPACE_ID
  finally:
    asyncio.run(provider.close_client(client))


def test_programmatic_next_credential_marks_shared_key_pool_out_of_credit_and_never_uses_oauth_seats(
  anthropic_pool,
) -> None:
  _pool, credentials, _store = anthropic_pool
  _write_shared_pool_file([
    {"key": _HEALTHY_KEY, "workspace_id": _WORKSPACE_ID},
    {"key": _SECOND_HEALTHY_KEY},
  ])
  provider = AnthropicProvider()
  config = credentials.get_anthropic_config(call_class="programmatic")
  assert config["api_key"] == _HEALTHY_KEY
  assert config["workspace_id"] == _WORKSPACE_ID

  billing = ProviderCredentialFailure(provider="anthropic", kind="billing", status_code=400)
  second = provider.next_credential(config, billing)

  assert second is not None
  assert _shared_state_status(_HEALTHY_KEY) == "out_of_credit"
  assert second["auth_mode"] == "api"
  assert second["api_key"] == _SECOND_HEALTHY_KEY
  assert second["workspace_id"] == ""
  assert second[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] != config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD]

  merged = merge_refreshed_auth_config(config, second)
  exhausted = provider.next_credential(merged, billing)

  assert _shared_state_status(_SECOND_HEALTHY_KEY) == "out_of_credit"
  assert exhausted is None


def test_credential_key_pool_is_exhausted_only_when_every_key_is_out_of_credit(
  anthropic_pool,
) -> None:
  _pool, credentials, _store = anthropic_pool
  _write_shared_pool_file([
    {"key": _HEALTHY_KEY, "workspace_id": _WORKSPACE_ID},
    {"key": _SECOND_HEALTHY_KEY},
  ])
  provider = AnthropicProvider()
  config = credentials.get_anthropic_config(call_class="programmatic")

  limited = provider.next_credential(
    config,
    ProviderCredentialFailure(provider="anthropic", kind="rate_limit", status_code=429, reset_at=_RESET_AT),
  )
  assert limited is not None
  pool = provider.credential_key_pool(merge_refreshed_auth_config(config, limited))
  assert pool is not None
  # One key cooling on a rate limit: not credit exhaustion.
  assert pool.credit_exhaustion().exhausted is False

  billing = ProviderCredentialFailure(provider="anthropic", kind="billing", status_code=400)
  merged = merge_refreshed_auth_config(config, limited)
  provider.next_credential(merged, billing)
  pool = provider.credential_key_pool(merged)
  assert pool is not None
  assert pool.credit_exhaustion().exhausted is False
  load_key_pool("anthropic", env_var="ANTHROPIC_API_KEY", repo="ai-excel-addin").mark_exhausted(
    fingerprint(_HEALTHY_KEY), reason="credit_balance_too_low",
  )
  load_key_pool("anthropic", env_var="ANTHROPIC_API_KEY", repo="ai-excel-addin").mark_exhausted(
    fingerprint(_SECOND_HEALTHY_KEY), reason="credit_balance_too_low",
  )
  pool = provider.credential_key_pool(merged)
  assert pool is not None
  assert pool.credit_exhaustion().exhausted is True


def test_credential_key_pool_is_absent_for_a_seat_or_a_non_pool_key(anthropic_pool) -> None:
  provider = AnthropicProvider()
  seat = {"provider": "anthropic", "auth_mode": "oauth", "auth_token": _SEAT_A}
  assert provider.credential_key_pool(seat) is None
  assert provider.credential_key_pool(
    {"provider": "anthropic", "auth_mode": "api", "api_key": "sk-ant-api03-not-in-pool"},
  ) is None


def _rewrite_shared_pool_file(keys: list[dict[str, str]]) -> None:
  path = pool_config_path("anthropic")
  assert path is not None and path.exists()
  path.write_text(
    json.dumps({"schema": "hank-key-pool/1", "provider": "anthropic", "keys": keys}),
    encoding="utf-8",
  )


def _bound_pool_key(credentials) -> tuple[dict[str, object], str]:
  config = credentials.get_anthropic_config(call_class="programmatic")
  assert config["api_key"] == _HEALTHY_KEY
  return config, str(config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD])


def test_bound_pool_key_removed_from_pool_file_is_superseded_and_retired_on_next_registration(
  anthropic_pool,
) -> None:
  _pool, credentials, store = anthropic_pool
  _write_shared_pool_file([
    {"key": _HEALTHY_KEY, "workspace_id": _WORKSPACE_ID},
    {"key": _SECOND_HEALTHY_KEY},
  ])
  config, bound_ref = _bound_pool_key(credentials)
  assert anthropic_generation_auth_config(
    bound_ref, user_scope=None, base_config=config,
  )["api_key"] == _HEALTHY_KEY

  _rewrite_shared_pool_file([{"key": _SECOND_HEALTHY_KEY}])

  with pytest.raises(AnthropicSourceError) as bound:
    anthropic_generation_auth_config(bound_ref, user_scope=None, base_config=config)
  assert bound.value.reason == "superseded"
  assert _HEALTHY_KEY not in resolve_anthropic_credentials(anthropic_pool_config(config)).tokens

  replacement = credentials.get_anthropic_config(call_class="programmatic")
  assert replacement["api_key"] == _SECOND_HEALTHY_KEY
  assert _HEALTHY_KEY not in store.read_text(encoding="utf-8")
  with pytest.raises(AnthropicSourceError) as retired:
    anthropic_generation_auth_config(bound_ref, user_scope=None, base_config=config)
  assert retired.value.reason == "removed"


def test_bound_pool_key_marked_out_of_credit_is_superseded_then_removed_when_no_key_works(
  anthropic_pool,
) -> None:
  _pool, credentials, _store = anthropic_pool
  _write_shared_pool_file([
    {"key": _HEALTHY_KEY, "workspace_id": _WORKSPACE_ID},
    {"key": _SECOND_HEALTHY_KEY},
  ])
  config, bound_ref = _bound_pool_key(credentials)
  shared = load_key_pool("anthropic", env_var="ANTHROPIC_API_KEY", repo="ai-excel-addin")
  shared.mark_exhausted(fingerprint(_HEALTHY_KEY), reason="credit_balance_too_low")
  assert _shared_state_status(_HEALTHY_KEY) == "out_of_credit"

  with pytest.raises(AnthropicSourceError) as bound:
    anthropic_generation_auth_config(bound_ref, user_scope=None, base_config=config)
  assert bound.value.reason == "superseded"
  assert _HEALTHY_KEY not in resolve_anthropic_credentials(anthropic_pool_config(config)).tokens

  shared.mark_exhausted(fingerprint(_SECOND_HEALTHY_KEY), reason="credit_balance_too_low")

  with pytest.raises(AnthropicSourceError) as none_working:
    anthropic_generation_auth_config(bound_ref, user_scope=None, base_config=config)
  assert none_working.value.reason == "removed"



def test_bound_pool_key_blocked_by_spend_ceiling_is_superseded_then_removed(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  _pool, credentials, _store = anthropic_pool
  _write_shared_pool_file([
    {"key": _HEALTHY_KEY, "workspace_id": _WORKSPACE_ID},
    {"key": _SECOND_HEALTHY_KEY},
  ])
  config, bound_ref = _bound_pool_key(credentials)
  shared = load_key_pool("anthropic", env_var="ANTHROPIC_API_KEY", repo="ai-excel-addin")
  first_fp = shared.keys[0].credential_fingerprint

  def _per_key_breach(**kwargs):
    if kwargs.get("credential_fingerprint") == first_fp:
      return (SpendBreach(ceiling="per_key_daily_payg_usd", scope=first_fp, used=10.0, limit=5.0),)
    return ()

  monkeypatch.setattr(anthropic_oauth, "spend_breaches", _per_key_breach)
  with pytest.raises(AnthropicSourceError) as per_key:
    anthropic_generation_auth_config(bound_ref, user_scope=None, base_config=config)
  assert per_key.value.reason == "superseded"
  assert _HEALTHY_KEY not in resolve_anthropic_credentials(anthropic_pool_config(config)).tokens

  def _pool_breach(**_kwargs):
    return (SpendBreach(ceiling="per_pool_daily_payg_usd", scope="anthropic-key-pool", used=50.0, limit=25.0),)

  monkeypatch.setattr(anthropic_oauth, "spend_breaches", _pool_breach)
  with pytest.raises(AnthropicSourceError) as pool_wide:
    anthropic_generation_auth_config(bound_ref, user_scope=None, base_config=config)
  assert pool_wide.value.reason == "removed"

  for err in (
    SpendSettingsError("spend_settings", "corrupt settings"),
    SpendLedgerUnavailable(Path("/tmp/unreadable-ledger.sqlite"), RuntimeError("disk I/O")),
  ):
    def _raise_err(**_kwargs):
      raise err

    monkeypatch.setattr(anthropic_oauth, "spend_breaches", _raise_err)
    with pytest.raises(AnthropicSourceError) as blocked_err:
      anthropic_generation_auth_config(bound_ref, user_scope=None, base_config=config)
    assert blocked_err.value.reason == "removed"
    resolved_tokens = resolve_anthropic_credentials(anthropic_pool_config(config)).tokens
    assert _HEALTHY_KEY not in resolved_tokens
    assert resolved_tokens == ()

def test_static_credential_registration_refuses_a_non_string_workspace_id(anthropic_pool) -> None:
  with pytest.raises(AnthropicOAuthError, match="workspace_id"):
    register_anthropic_static_credential(
      _HEALTHY_KEY, credential_kind="api_key", source="env:ANTHROPIC_API_KEY", workspace_id=123,  # type: ignore[arg-type]
    )
