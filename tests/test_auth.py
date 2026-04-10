import sys
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
  CredentialsTimeoutError,
  CrossUserReuseError,
  MissingUserIdError,
  NoCredentialError,
  StrictModeDefaultUserError,
)


def test_auth_config_round_trips_and_preserves_all_fields() -> None:
  raw = {
    "provider": "anthropic",
    "billing_mode": "byok",
    "model": "claude-sonnet-4-6",
    "max_tokens": 16000,
    "auth_mode": "oauth",
    "auth_token": "tok-123",
    "thinking": False,
    "base_url": "https://example.test",
    "compat": {"beta": True},
    "extra": {"k": "v"},
  }

  config = AuthConfig.from_dict(raw)

  assert config.provider == "anthropic"
  assert config.billing_mode == "byok"
  assert config.model == "claude-sonnet-4-6"
  assert config.max_tokens == 16000
  assert config._raw["auth_mode"] == "oauth"
  assert config._raw["thinking"] is False
  assert config._raw["base_url"] == "https://example.test"
  assert config._raw["compat"] == {"beta": True}
  assert config.to_dict() == raw


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


def test_classify_anthropic_credential_distinguishes_oauth_and_api_key() -> None:
  assert _classify_anthropic_credential("sk-ant-oat01-token") == {"auth_token": "sk-ant-oat01-token"}
  assert _classify_anthropic_credential("sk-ant-api-token") == {"api_key": "sk-ant-api-token"}


def test_auth_exceptions_have_actionable_default_messages() -> None:
  messages = {
    NoCredentialError(): "credential",
    CredentialsTimeoutError(): "resolver",
    StrictModeDefaultUserError(): "user_id",
    MissingUserIdError(): "user_id",
    CrossUserReuseError(): "session",
    AuthExpiredError(): "retry",
  }

  for exc, keyword in messages.items():
    message = str(exc)
    assert message
    assert keyword in message.lower()
