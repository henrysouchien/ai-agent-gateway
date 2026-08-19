# ruff: noqa: E402

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
API_DIR = ROOT / "api"
for path in (ROOT, PKG_DIR, API_DIR):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent.profiles import load_profile
from agent_gateway._provider_utils import _resolve_provider
from agent_gateway.auth import AuthConfig
from agent_gateway.providers import installed_adapter_providers
from tests.deterministic_fixture_support import (
  FIXTURE_MODEL_ID,
  FixtureProvider,
  build_fixture_profile,
)


@pytest.mark.parametrize("app_env", [None, "development", "test", "production"])
def test_product_provider_and_credentials_never_register_fixture(monkeypatch, app_env) -> None:
  if app_env is None:
    monkeypatch.delenv("APP_ENV", raising=False)
  else:
    monkeypatch.setenv("APP_ENV", app_env)

  with pytest.raises(ValueError, match="Unknown provider"):
    _resolve_provider("fixture", FIXTURE_MODEL_ID, None, None, None)

  from credentials import get_provider_config, get_provider_instance

  with pytest.raises(RuntimeError, match="Unknown provider"):
    get_provider_config("fixture")
  with pytest.raises(RuntimeError, match="Unknown provider"):
    get_provider_instance("fixture")
  with pytest.raises(ValueError, match="Unsupported provider 'fixture'"):
    AuthConfig.from_dict({"provider": "fixture", "billing_mode": "byok"})
  assert "fixture.responses" not in installed_adapter_providers()


def test_product_profile_registry_does_not_contain_fixture() -> None:
  with pytest.raises(ModuleNotFoundError, match=r"agent\.profiles\._fixture"):
    load_profile("_fixture")


def test_test_harness_can_inject_fixture_provider_and_profile_explicitly() -> None:
  injected = FixtureProvider()
  provider, provider_name, auth_config = _resolve_provider(
    injected,
    FIXTURE_MODEL_ID,
    None,
    None,
    None,
  )

  assert provider is injected
  assert provider_name == "fixture"
  assert auth_config == {"max_tokens": 16_000}
  assert build_fixture_profile().name == "_fixture"


def test_preserved_fixture_skill_records_remain_available_to_test_harnesses() -> None:
  skills_root = ROOT / "api" / "memory" / "workspace" / "notes" / "skills"
  names = {
    "fixture-sleep",
    "fixture-canvas-artifact",
    "fixture-dashboard-artifact",
    "fixture-approval-canvas-artifact",
    "fixture-terminal-failure",
  }

  assert all((skills_root / f"{name}.md").is_file() for name in names)
