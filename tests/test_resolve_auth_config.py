import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from api.credentials import get_default_model as get_canonical_default_model
from agent_gateway import resolve_auth_config
from agent_gateway._provider_utils import _resolve_provider


@pytest.fixture(autouse=True)
def _clear_anthropic_env(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_MODE", raising=False)
  monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
  monkeypatch.delenv("ALLOWED_MODELS_ANTHROPIC", raising=False)
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_resolve_auth_config_explicit_api_key_arg() -> None:
  assert resolve_auth_config(api_key="sk-123") == {
    "auth_mode": "api",
    "api_key": "sk-123",
    "auth_token": "",
  }


def test_resolve_auth_config_explicit_auth_token_arg() -> None:
  assert resolve_auth_config(auth_token="tok-123") == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok-123",
  }


def test_resolve_auth_config_both_args_api_key_wins() -> None:
  resolved = resolve_auth_config(api_key="sk-123", auth_token="tok-123")

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "sk-123"


def test_resolve_auth_config_config_api_key_infers_api_mode() -> None:
  assert resolve_auth_config(auth_config={"api_key": "cfg-key"}) == {
    "auth_mode": "api",
    "api_key": "cfg-key",
    "auth_token": "",
  }


def test_resolve_auth_config_config_auth_token_infers_oauth_mode() -> None:
  assert resolve_auth_config(auth_config={"auth_token": "cfg-token"}) == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "cfg-token",
  }


def test_resolve_auth_config_config_with_both_credentials_api_key_wins() -> None:
  resolved = resolve_auth_config(auth_config={"api_key": "cfg-key", "auth_token": "cfg-token"})

  assert resolved["auth_mode"] == "api"


def test_resolve_auth_config_explicit_auth_mode_is_used() -> None:
  resolved = resolve_auth_config(
    auth_config={
      "auth_mode": "oauth",
      "api_key": "cfg-key",
      "auth_token": "cfg-token",
    }
  )

  assert resolved["auth_mode"] == "oauth"
  assert resolved["api_key"] == "cfg-key"
  assert resolved["auth_token"] == "cfg-token"


def test_resolve_auth_config_env_fallback_api_key() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  try:
    assert resolve_auth_config() == {
      "auth_mode": "api",
      "api_key": "env-key",
      "auth_token": "",
    }
  finally:
    monkeypatch.undo()


def test_resolve_auth_config_env_fallback_auth_token() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  try:
    assert resolve_auth_config() == {
      "auth_mode": "oauth",
      "api_key": "",
      "auth_token": "env-token",
    }
  finally:
    monkeypatch.undo()


def test_resolve_auth_config_env_auth_mode_takes_precedence_when_enabled() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "oauth")
  try:
    resolved = resolve_auth_config(read_env_auth_mode=True)
  finally:
    monkeypatch.undo()

  assert resolved["auth_mode"] == "oauth"
  assert resolved["api_key"] == ""
  assert resolved["auth_token"] == "env-token"


def test_resolve_auth_config_env_auth_mode_ignored_by_default() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "oauth")
  try:
    resolved = resolve_auth_config()
  finally:
    monkeypatch.undo()

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "env-key"
  assert resolved["auth_token"] == ""


def test_resolve_auth_config_env_auth_mode_ignored_when_auth_config_provided() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "oauth")
  try:
    resolved = resolve_auth_config(
      auth_config={"api_key": "cfg-key"},
      read_env_auth_mode=True,
    )
  finally:
    monkeypatch.undo()

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "cfg-key"


def test_resolve_auth_config_precedence_arg_over_config_over_env() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  try:
    resolved = resolve_auth_config(
      api_key="arg-key",
      auth_config={"api_key": "cfg-key"},
    )
  finally:
    monkeypatch.undo()

  assert resolved["api_key"] == "arg-key"
  assert resolved["auth_mode"] == "api"


def test_resolve_auth_config_no_env_fallback_when_auth_config_provided() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  try:
    resolved = resolve_auth_config(auth_config={})
  finally:
    monkeypatch.undo()

  assert resolved == {
    "auth_mode": "api",
    "api_key": "",
    "auth_token": "",
  }


def test_resolve_auth_config_prefix_sniffing_reclassifies_api_key_as_oauth() -> None:
  assert resolve_auth_config(
    api_key="sk-ant-oat-123",
    infer_mode_from_prefix=True,
  ) == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "sk-ant-oat-123",
  }


def test_resolve_auth_config_prefix_sniffing_is_off_by_default() -> None:
  assert resolve_auth_config(api_key="sk-ant-oat-123") == {
    "auth_mode": "api",
    "api_key": "sk-ant-oat-123",
    "auth_token": "",
  }


def test_resolve_auth_config_explicit_auth_mode_beats_prefix_sniffing() -> None:
  resolved = resolve_auth_config(
    api_key="sk-ant-oat-123",
    auth_config={"auth_mode": "api"},
    infer_mode_from_prefix=True,
  )

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "sk-ant-oat-123"
  assert resolved["auth_token"] == ""


def test_resolve_auth_config_no_credentials_without_raise_returns_empty_api_mode() -> None:
  assert resolve_auth_config() == {
    "auth_mode": "api",
    "api_key": "",
    "auth_token": "",
  }


def test_resolve_auth_config_no_credentials_with_raise_errors() -> None:
  with pytest.raises(RuntimeError, match="auth_mode is 'api' but no api_key found"):
    resolve_auth_config(raise_on_missing=True)


def test_resolve_auth_config_openai_explicit_api_key_includes_auth_fields() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    api_key="sk-openai",
    model="gpt-4o",
  )

  assert resolved == {
    "auth_mode": "api",
    "api_key": "sk-openai",
    "auth_token": "",
    "model": "gpt-4o",
  }


def test_resolve_auth_config_openai_explicit_auth_token() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    auth_token="oat-xxx",
  )

  assert resolved == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "oat-xxx",
  }


def test_resolve_auth_config_openai_api_key_wins_over_auth_token() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    api_key="sk-openai",
    auth_token="oat-xxx",
  )

  assert resolved == {
    "auth_mode": "api",
    "api_key": "sk-openai",
    "auth_token": "",
  }


def test_resolve_auth_config_openai_auth_config_with_explicit_mode() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    auth_config={
      "auth_mode": "oauth",
      "auth_token": "tok",
    },
  )

  assert resolved == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
  }


def test_resolve_auth_config_openai_no_credentials_returns_minimal() -> None:
  resolved = resolve_auth_config(provider="openai")

  assert "auth_mode" not in resolved
  assert "auth_token" not in resolved
  assert "api_key" not in resolved


def test_resolve_auth_config_openai_whitespace_token_ignored() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    auth_token="   ",
  )

  assert "auth_mode" not in resolved
  assert "auth_token" not in resolved
  assert "api_key" not in resolved


def test_resolve_auth_config_openai_auth_config_infers_mode_from_token() -> None:
  resolved = resolve_auth_config(
    provider="openai",
    auth_config={"auth_token": "tok"},
  )

  assert resolved == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
  }


def test_resolve_auth_config_sets_model_max_tokens_and_thinking_when_passed() -> None:
  resolved = resolve_auth_config(
    api_key="sk-123",
    model="claude-test",
    max_tokens=123,
    thinking=True,
  )

  assert resolved["model"] == "claude-test"
  assert resolved["max_tokens"] == 123
  assert resolved["effort"] == "high"
  assert resolved["thinking_enabled_requested"] is True
  assert "thinking" not in resolved


def test_resolve_auth_config_does_not_inject_model_defaults() -> None:
  resolved = resolve_auth_config(api_key="sk-123")

  assert "model" not in resolved
  assert "max_tokens" not in resolved
  assert "thinking" not in resolved


def test_resolve_auth_config_blank_args_fall_through_to_next_tier() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  try:
    resolved = resolve_auth_config(api_key="   ", auth_token="")
  finally:
    monkeypatch.undo()

  assert resolved["auth_mode"] == "oauth"
  assert resolved["api_key"] == ""
  assert resolved["auth_token"] == "env-token"


def test_resolve_auth_config_preserves_extra_auth_config_keys() -> None:
  resolved = resolve_auth_config(
    auth_config={
      "api_key": "cfg-key",
      "custom_field": 42,
      "nested": {"enabled": True},
    }
  )

  assert resolved["custom_field"] == 42
  assert resolved["nested"] == {"enabled": True}
  assert resolved["api_key"] == "cfg-key"
  assert resolved["auth_mode"] == "api"


def test_resolve_auth_config_read_env_false_skips_env_fallback() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  try:
    resolved = resolve_auth_config(read_env=False)
  finally:
    monkeypatch.undo()

  assert resolved == {
    "auth_mode": "api",
    "api_key": "",
    "auth_token": "",
  }


def test_resolve_auth_config_preserves_non_active_credential_with_auth_config() -> None:
  resolved = resolve_auth_config(auth_config={"api_key": "cfg-key", "auth_token": "cfg-token"})

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "cfg-key"
  assert resolved["auth_token"] == "cfg-token"


def test_resolve_auth_config_blanks_non_active_credential_without_auth_config() -> None:
  resolved = resolve_auth_config(api_key="arg-key", auth_token="arg-token")

  assert resolved["auth_mode"] == "api"
  assert resolved["api_key"] == "arg-key"
  assert resolved["auth_token"] == ""


def test_resolve_provider_does_not_read_anthropic_auth_mode_env() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "oauth")
  try:
    _provider, _provider_name, config = _resolve_provider("anthropic", None, None, None, None)
  finally:
    monkeypatch.undo()

  assert config["auth_mode"] == "api"
  assert config["api_key"] == "env-key"
  assert config["auth_token"] == ""


def test_resolve_provider_model_free_auth_config_uses_canonical_default() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "anthropic",
    None,
    None,
    None,
    None,
    auth_config={"api_key": "cfg-key"},
  )

  assert config["model"] == get_canonical_default_model("anthropic")


def test_resolve_provider_preserves_explicit_model_via_post_processing() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "anthropic",
    "opus",
    None,
    None,
    None,
    auth_config={"api_key": "cfg-key"},
  )

  assert config["model"] == "opus"
  assert config["auth_mode"] == "api"
  assert config["api_key"] == "cfg-key"


def test_resolve_provider_uses_auth_config_credentials_instead_of_env() -> None:
  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  try:
    _provider, _provider_name, config = _resolve_provider(
      "anthropic",
      None,
      None,
      None,
      None,
      auth_config={"api_key": "cfg-key"},
    )
  finally:
    monkeypatch.undo()

  assert config["api_key"] == "cfg-key"
  assert config["auth_mode"] == "api"
