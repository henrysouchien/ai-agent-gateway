from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from pathlib import Path
from typing import Literal

import pytest

from agent_gateway.auth import ProviderCredentialFailure
from agent_gateway.providers import anthropic as anthropic_provider
from agent_gateway.capability_execution import CapabilityExecutionResolver
from agent_gateway.providers import anthropic_oauth
from agent_gateway.providers.anthropic import AnthropicProvider
from agent_gateway.providers.anthropic_oauth import (
  ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES,
  ANTHROPIC_DEPLOYMENT_API_KEY_FIELD,
  AnthropicCredentialPool,
  AnthropicSourceError,
  anthropic_generation_auth_config,
  enroll_anthropic_setup_token,
  resolve_anthropic_auth_store_path,
  resolve_anthropic_credentials,
  select_anthropic_credential_source,
)
from agent_gateway.runner_auth import merge_refreshed_auth_config
from gateway_test_support.capability_execution_test_support import stub_bound_capability_execution


_API_KEY = "sk-ant-api03-synthetic-pool-key"
_SEAT_A = "sk-ant-oat01-synthetic-setup-a"
_SEAT_B = "sk-ant-oat01-synthetic-setup-b"
_KEY_IDENTITY = "api-key-" + hashlib.sha256(_API_KEY.encode("utf-8")).hexdigest()[:16]
_RESET_AT = 2_000.0


@pytest.fixture
def anthropic_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  # Product imports need the checkout root even in package-only pytest runs.
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
  credentials = importlib.import_module("api.credentials")
  pool = AnthropicCredentialPool()
  for module in (anthropic_provider, anthropic_oauth, credentials):
    monkeypatch.setattr(module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  for name in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES:
    monkeypatch.delenv(name, raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_MODE", raising=False)
  monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
  monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "user-data"))
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))
  monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_SEAT_A, _SEAT_B]))
  monkeypatch.setattr(anthropic_provider.time, "time", lambda: 1_000.0)
  return pool, credentials, tmp_path / "oauth.json"


def _gateway_sources(credentials, *, user_scope=None):
  config = credentials.get_anthropic_config(user_scope=user_scope)
  pool_config = {
    ANTHROPIC_DEPLOYMENT_API_KEY_FIELD: config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD],
  } if ANTHROPIC_DEPLOYMENT_API_KEY_FIELD in config else {}
  return resolve_anthropic_credentials(pool_config, user_scope=user_scope)


def test_api_key_precedes_two_setup_tokens_with_exact_kind_source_and_identity(
  anthropic_pool,
) -> None:
  _pool, credentials, _store = anthropic_pool
  sources = _gateway_sources(credentials)
  assert sources.tokens == (_API_KEY, _SEAT_A, _SEAT_B)
  assert len(sources.credentials) == 3
  key, seat_a, seat_b = sources.credentials
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


@pytest.mark.parametrize("legacy_mode", [None, "api", "oauth"])
def test_config_derives_api_mode_from_first_member_not_legacy_mode(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch, legacy_mode: str | None,
) -> None:
  pool, credentials, _store = anthropic_pool
  if legacy_mode is not None:
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", legacy_mode)
  config = credentials.get_anthropic_config()
  assert config["auth_mode"] == "api"
  assert config["api_key"] == _API_KEY
  assert config["auth_token"] == ""
  key = _gateway_sources(credentials).credentials[0]
  assert config["credential_source"] == key.source_metadata()

  pool.block(key.key, until=_RESET_AT)
  seat_config = credentials.get_anthropic_config()
  assert seat_config["auth_mode"] == "oauth"
  assert seat_config["auth_token"] == _SEAT_A
  assert seat_config["api_key"] == ""
  assert seat_config["credential_source"] != config["credential_source"]


@pytest.mark.parametrize("kind", ["rate_limit", "billing", "auth"])
def test_key_failure_parks_until_reset_and_rotates_first_oauth_seat(
  anthropic_pool, kind: Literal["rate_limit", "billing", "auth"],
) -> None:
  pool, credentials, _store = anthropic_pool
  config = credentials.get_anthropic_config()
  key = _gateway_sources(credentials).credentials[0]
  failure = ProviderCredentialFailure(
    provider="anthropic", kind=kind,
    status_code=429 if kind == "rate_limit" else 403, reset_at=_RESET_AT,
  )
  rotated = AnthropicProvider().next_credential(config, failure)
  assert rotated is not None
  assert rotated["auth_mode"] == "oauth"
  assert rotated["auth_token"] == _SEAT_A
  assert rotated["api_key"] == ""
  assert rotated[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD]
  merged = merge_refreshed_auth_config(config, rotated)
  assert merged[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD]
  later_pool = resolve_anthropic_credentials({
    name: value for name, value in merged.items() if name != "credential_source"
  })
  assert _API_KEY in later_pool.tokens
  sources = _gateway_sources(credentials)
  assert rotated["credential_source"] == sources.credentials[1].source_metadata()
  assert rotated["credential_source"] != config["credential_source"]
  assert pool.blocked_until(key.key) == _RESET_AT
  assert pool.blocked_until(sources.credentials[1].key) == 0.0
  before = select_anthropic_credential_source(sources, pool=pool, now=_RESET_AT - 1)
  at_reset = select_anthropic_credential_source(sources, pool=pool, now=_RESET_AT)
  assert before is not None and before.token == _SEAT_A
  assert at_reset is not None and at_reset.token == _API_KEY


def test_oauth_seat_failure_can_rotate_back_to_api_key(anthropic_pool) -> None:
  pool, credentials, _store = anthropic_pool
  sources = _gateway_sources(credentials)
  key, seat = sources.credentials[:2]
  pool.block(key.key, until=999.0)
  config = {
    **credentials.get_anthropic_config(),
    "auth_mode": "oauth", "api_key": "", "auth_token": _SEAT_A,
    "credential_source": seat.source_metadata(),
  }
  rotated = AnthropicProvider().next_credential(config, ProviderCredentialFailure(
    provider="anthropic", kind="rate_limit", status_code=429, reset_at=_RESET_AT,
  ))
  assert rotated is not None
  assert rotated["auth_mode"] == "api"
  assert rotated["api_key"] == _API_KEY
  assert rotated["auth_token"] == ""
  assert rotated["credential_source"] == key.source_metadata()
  assert pool.blocked_until(seat.key) == _RESET_AT
  assert pool.blocked_until(key.key) < 1_000.0


def test_absent_api_key_preserves_generation_but_empty_retires_it(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  _pool, credentials, store = anthropic_pool
  config = credentials.get_anthropic_config()
  key = resolve_anthropic_credentials(config).credentials[0]
  original_bytes = store.read_bytes()
  monkeypatch.delenv("ANTHROPIC_API_KEY")
  absent_config = credentials.get_anthropic_config()
  assert ANTHROPIC_DEPLOYMENT_API_KEY_FIELD not in absent_config
  absent_sources = _gateway_sources(credentials)
  assert _API_KEY not in absent_sources.tokens
  assert absent_sources.tokens == (_SEAT_A, _SEAT_B)
  assert store.read_bytes() == original_bytes
  # Bound gateway material explicitly retains membership even without ambient
  # visibility; absence does not retire the stored generation.
  assert resolve_anthropic_credentials(config).credentials[0].generation_ref == key.generation_ref
  exact_config = anthropic_generation_auth_config(
    key.generation_ref, user_scope=None, base_config=config,
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
  empty_sources = _gateway_sources(credentials)
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
  sources = _gateway_sources(credentials)
  deployment_key_ref = sources.credentials[0].generation_ref
  original_bytes = store.read_bytes()
  monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-unobserved-new-seat")
  for member in sources.credentials:
    exact = resolve_anthropic_credentials({
      "credential_source": member.source_metadata(),
      ANTHROPIC_DEPLOYMENT_API_KEY_FIELD: deployment_key_ref,
    })
    assert exact.tokens == (member.token,)
    assert exact.credentials[0].source_metadata() == member.source_metadata()
    assert len(exact.credentials) == 1
    assert store.read_bytes() == original_bytes


def test_rotating_key_to_seat_changes_request_headers_without_network(anthropic_pool) -> None:
  pytest.importorskip("anthropic")
  httpx2 = pytest.importorskip("httpx2")
  _pool, credentials, _store = anthropic_pool
  provider = AnthropicProvider()
  config = credentials.get_anthropic_config()
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
  assert requests[0].headers.get_list("x-api-key") == [_API_KEY]
  assert "authorization" not in requests[0].headers
  assert requests[1].headers.get_list("authorization") == [f"Bearer {_SEAT_A}"]
  assert "x-api-key" not in requests[1].headers


def test_ambient_api_key_is_not_a_pool_input_even_after_gateway_enrollment(anthropic_pool) -> None:
  _pool, credentials, _store = anthropic_pool
  assert resolve_anthropic_credentials().tokens == (_SEAT_A, _SEAT_B)
  credentials.get_anthropic_config()
  assert resolve_anthropic_credentials().tokens == (_SEAT_A, _SEAT_B)
  rotated = AnthropicProvider().next_credential(
    {"auth_mode": "oauth", "auth_token": _SEAT_A},
    ProviderCredentialFailure(
      provider="anthropic", kind="rate_limit", status_code=429, reset_at=_RESET_AT,
    ),
  )
  assert rotated is not None and rotated["auth_token"] == _SEAT_B
  assert rotated["api_key"] == ""
  assert ANTHROPIC_DEPLOYMENT_API_KEY_FIELD not in rotated


def test_child_pool_environment_names_exclude_operator_api_key() -> None:
  assert "ANTHROPIC_API_KEY" not in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES


def test_gateway_pool_ranks_user_accounts_then_key_then_setup_then_org(anthropic_pool) -> None:
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
  assert sources.tokens == (user_token, _API_KEY, _SEAT_A, _SEAT_B, org_token)
  config = credentials.get_anthropic_config(user_scope=1)
  assert config["auth_mode"] == "oauth"
  assert config["auth_token"] == user_token
  assert config["credential_source"] == sources.credentials[0].source_metadata()
  assert _gateway_sources(credentials, user_scope=3).tokens == (
    _API_KEY, _SEAT_A, _SEAT_B, org_token,
  )
  bound_sources = resolve_anthropic_credentials({
    ANTHROPIC_DEPLOYMENT_API_KEY_FIELD: config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD],
    "auth_token": _SEAT_B,
  }, user_scope=1)
  assert bound_sources.tokens == (_SEAT_B, user_token, _API_KEY, _SEAT_A, org_token)


def test_gateway_hop_rebind_preserves_deployment_key_membership(
  anthropic_pool, monkeypatch: pytest.MonkeyPatch,
) -> None:
  pool, credentials, _store = anthropic_pool
  config = credentials.get_anthropic_config()
  key = resolve_anthropic_credentials(config).credentials[0]
  execution = stub_bound_capability_execution(
    provider=AnthropicProvider(), model="claude-sonnet-5", effort="none",
    auth_config=config, credential_ref=key.generation_ref,
  )
  pool.block(key.key, until=_RESET_AT)
  rebound = CapabilityExecutionResolver._rebind_unparked_account(execution)
  assert rebound is not None
  assert rebound.auth_config["auth_token"] == _SEAT_A
  assert rebound.auth_config["api_key"] == ""
  assert rebound.auth_config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == key.generation_ref
  pool.block(_SEAT_A, until=_RESET_AT + 1_000.0)
  pool.block(_SEAT_B, until=_RESET_AT + 1_000.0)
  monkeypatch.setattr(anthropic_provider.time, "time", lambda: _RESET_AT + 1.0)
  returned = CapabilityExecutionResolver._rebind_unparked_account(rebound)
  assert returned is not None
  assert returned.auth_config["api_key"] == _API_KEY
  assert returned.auth_config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] == key.generation_ref


def test_non_gateway_hop_never_admits_ambient_deployment_key(anthropic_pool) -> None:
  pool, credentials, _store = anthropic_pool
  credentials.get_anthropic_config()
  sources = resolve_anthropic_credentials()
  seat = sources.credentials[0]
  execution = stub_bound_capability_execution(
    provider=AnthropicProvider(), model="claude-sonnet-5", effort="none",
    auth_config={
      "auth_mode": "oauth", "auth_token": seat.token,
      "credential_source": seat.source_metadata(),
    },
    credential_ref=seat.generation_ref,
  )
  for member in sources.credentials:
    pool.block(member.key, until=_RESET_AT)
  assert CapabilityExecutionResolver._rebind_unparked_account(execution) is None
