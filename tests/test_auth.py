# ruff: noqa: E402

import asyncio
import inspect
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway._provider_utils import _classify_anthropic_credential
from agent_gateway.auth import (
  AuthConfig,
  AuthExpiredError,
  CredentialsResolver,
  CredentialsTimeoutError,
  CrossUserReuseError,
  MissingUserIdError,
  NoCredentialError,
  ResolverResult,
)
from agent_gateway.fork_request_handoff import ForkRequestHandoff
from agent_gateway.providers import AnthropicProvider
from agent_gateway.session import GatewaySession
from tests.capability_execution_test_support import stub_runner_capability_execution


def test_auth_config_round_trips_and_preserves_all_fields() -> None:
  raw = {
    "provider": "anthropic",
    "billing_mode": "byok",
    "max_tokens": 16000,
    "auth_mode": "oauth",
    "auth_token": "tok-123",
    "base_url": "https://example.test",
    "compat": {"beta": True},
    "extra": {"k": "v"},
  }

  config = AuthConfig.from_dict(raw)

  assert config.provider == "anthropic"
  assert config.billing_mode == "byok"
  assert config.max_tokens == 16000
  assert config._raw["auth_mode"] == "oauth"
  assert config._raw["base_url"] == "https://example.test"
  assert config._raw["compat"] == {"beta": True}
  assert config.to_dict() == raw


def test_credential_material_is_omitted_from_runtime_object_repr() -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-repr-8f21d7"
  auth_config = AuthConfig.from_dict({
    "provider": "anthropic",
    "billing_mode": "byok",
    "api_key": secret,
  })
  resolver_result = ResolverResult(
    user_id="alice",
    channel="excel",
    auth_config=auth_config,
    credential_principal="service",
  )
  session = GatewaySession(
    session_id="session-repr",
    api_key_hash="safe-hash",
    created_at=1,
    expires_at=2,
    user_id="alice",
    auth_config=auth_config.to_dict(),
  )
  execution = stub_runner_capability_execution(
    provider=AnthropicProvider(),
    model="claude-test",
    effort="none",
    auth_config={"api_key": secret},
  )
  handoff = ForkRequestHandoff(
    _messages=(),
    rendered_system_blocks=(),
    _wire_tools=(),
    max_tokens=100,
    _auth_config=auth_config.to_dict(),
    capability_bind=execution.bind,
    tenant_id="tenant-1",
    billing_mode="byok",
    message_marker_position=(0, 0),
    boundary_kind="mid_turn",
  )

  for value in (auth_config, resolver_result, session, handoff):
    assert secret not in repr(value)


@pytest.mark.parametrize(
  ("payload", "missing_key"),
  [
    ({"billing_mode": "byok"}, "provider"),
    ({"provider": "anthropic"}, "billing_mode"),
  ],
)
def test_auth_config_requires_provider_and_billing_mode(payload: dict[str, str], missing_key: str) -> None:
  with pytest.raises(ValueError, match=missing_key):
    AuthConfig.from_dict(payload)


def test_resolver_result_dataclass_and_credentials_resolver_async_contract() -> None:
  auth_config = AuthConfig.from_dict(
    {
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": "resolver-key",
    }
  )

  async def _resolver(api_key: str, payload) -> ResolverResult:
    assert api_key == "gateway-key"
    assert payload == {"request": "init"}
    return ResolverResult(
      user_id="alice",
      channel="excel",
      auth_config=auth_config,
      credential_principal="service",
      allow_service_for_interactive=True,
    )

  resolver: CredentialsResolver = _resolver
  awaitable = resolver("gateway-key", {"request": "init"})

  assert inspect.isawaitable(awaitable)
  result = asyncio.run(awaitable)
  assert is_dataclass(result)
  assert [field.name for field in fields(result)] == [
    "user_id",
    "channel",
    "auth_config",
    "credential_principal",
    "allow_service_for_interactive",
    "risk_user_id",
    "role",
    "user_email",
    "capabilities",
    "model_entitled_capabilities",
    "model_entitled_keys",
  ]
  assert result.user_id == "alice"
  assert result.channel == "excel"
  assert result.credential_principal == "service"
  assert result.allow_service_for_interactive is True
  assert result.risk_user_id is None
  assert result.role == "invite"
  assert result.user_email is None
  assert result.capabilities == frozenset()
  assert result.model_entitled_capabilities == frozenset()
  assert result.model_entitled_keys == frozenset()
  assert result.auth_config is auth_config


def test_resolver_result_requires_explicit_credential_principal() -> None:
  auth_config = AuthConfig.from_dict(
    {
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": "resolver-key",
    }
  )

  with pytest.raises(ValueError, match="credential_principal"):
    ResolverResult(
      user_id="alice",
      channel="web",
      auth_config=auth_config,
      credential_principal="inferred",  # type: ignore[arg-type]
    )


def test_resolver_result_requires_boolean_interactive_service_policy() -> None:
  auth_config = AuthConfig.from_dict(
    {
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": "resolver-key",
    }
  )

  with pytest.raises(ValueError, match="must be a bool"):
    ResolverResult(
      user_id="alice",
      channel="web",
      auth_config=auth_config,
      credential_principal="service",
      allow_service_for_interactive="false",  # type: ignore[arg-type]
    )


def test_classify_anthropic_credential_distinguishes_oauth_and_api_key() -> None:
  assert _classify_anthropic_credential("sk-ant-oat01-token") == {"auth_token": "sk-ant-oat01-token"}
  assert _classify_anthropic_credential("sk-ant-api-token") == {"api_key": "sk-ant-api-token"}


def test_auth_exceptions_have_actionable_default_messages() -> None:
  messages = {
    NoCredentialError(): "credential",
    CredentialsTimeoutError(): "resolver",
    MissingUserIdError(): "user_id",
    CrossUserReuseError(): "session",
    AuthExpiredError(): "retry",
  }

  for exc, keyword in messages.items():
    message = str(exc)
    assert message
    assert keyword in message.lower()


@pytest.mark.parametrize(
  ("message", "kind"),
  [
    ("429 Too Many Requests: rate limit exceeded", "rate_limit"),
    ("401 Unauthorized: invalid api key", "auth"),
    ("403 PermissionDeniedError: billing quota exhausted", "billing"),
    ("insufficient_quota: exceeded your current quota", "billing"),
  ],
)
def test_provider_classifies_credential_refresh_failures(message: str, kind: str) -> None:
  failure = AnthropicProvider().classify_credential_failure(RuntimeError(message))

  assert failure is not None
  assert failure.provider == "anthropic"
  assert failure.kind == kind
  assert failure.retryable_with_new_credentials is True
