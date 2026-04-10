import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.providers import OpenAIProvider


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_openai_has_active_credential_oauth_mode() -> None:
  provider = OpenAIProvider()

  assert provider.has_active_credential(
    {"auth_mode": "oauth", "auth_token": "tok"}
  ) is True


def test_openai_has_active_credential_oauth_mode_empty_token() -> None:
  provider = OpenAIProvider()

  assert provider.has_active_credential(
    {"auth_mode": "oauth", "auth_token": "   "}
  ) is False


def test_openai_create_client_oauth_uses_auth_token_as_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
  class _FakeAsyncOpenAI:
    def __init__(self, **kwargs):
      self.kwargs = kwargs

  fake_openai = types.ModuleType("openai")
  fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
  monkeypatch.setitem(sys.modules, "openai", fake_openai)

  provider = OpenAIProvider()
  client = provider.create_client(
    {
      "auth_mode": "oauth",
      "auth_token": "oauth-token",
      "base_url": "https://custom.example/v1",
      "compat": {"streaming": True},
    }
  )

  assert isinstance(client, _FakeAsyncOpenAI)
  assert client.kwargs == {
    "api_key": "oauth-token",
    "base_url": "https://custom.example/v1",
  }
  assert provider._last_base_url == "https://custom.example/v1"
  assert provider._last_compat_override == {"streaming": True}


def test_openai_has_active_credential_oauth_ignores_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("OPENAI_API_KEY", "env-key")
  provider = OpenAIProvider()

  assert provider.has_active_credential(
    {"auth_mode": "oauth", "auth_token": "  "}
  ) is False


def test_openai_has_active_credential_api_mode() -> None:
  provider = OpenAIProvider()

  assert provider.has_active_credential({"api_key": "sk-x"}) is True


def test_openai_create_client_api_mode_uses_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
  class _FakeAsyncOpenAI:
    def __init__(self, **kwargs):
      self.kwargs = kwargs

  fake_openai = types.ModuleType("openai")
  fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
  monkeypatch.setitem(sys.modules, "openai", fake_openai)

  provider = OpenAIProvider()
  client = provider.create_client({"auth_mode": "api", "api_key": "sk-test"})

  assert isinstance(client, _FakeAsyncOpenAI)
  assert client.kwargs["api_key"] == "sk-test"


def test_openai_create_client_oauth_ignores_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("OPENAI_API_KEY", "env-key-should-be-ignored")

  class _FakeAsyncOpenAI:
    def __init__(self, **kwargs):
      self.kwargs = kwargs

  fake_openai = types.ModuleType("openai")
  fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
  monkeypatch.setitem(sys.modules, "openai", fake_openai)

  provider = OpenAIProvider()
  client = provider.create_client({"auth_mode": "oauth", "auth_token": "my-bearer"})

  assert client.kwargs["api_key"] == "my-bearer"
