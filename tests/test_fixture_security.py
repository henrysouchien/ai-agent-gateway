import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
API_DIR = ROOT / "api"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))

from agent_gateway._provider_utils import _resolve_provider
from agent_gateway.auth import AuthConfig
from agent_gateway.autonomous_runner import AutonomousRegistry
from agent_gateway.fixture_gate import (
  FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME,
  FIXTURE_CANVAS_ARTIFACT_SKILL_NAME,
  FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME,
  FIXTURE_MODEL_ID,
  FIXTURE_TERMINAL_FAILURE_SKILL_NAME,
)
from agent_gateway.skills import SkillLoader


def _skills_root() -> Path:
  return ROOT / "api" / "memory" / "workspace" / "notes" / "skills"


def test_fixture_surfaces_fail_closed_when_environment_is_ambiguous(monkeypatch, tmp_path) -> None:
  for name in ("APP_ENV", "ENVIRONMENT", "AGENT_GATEWAY_ENV", "NODE_ENV"):
    monkeypatch.delenv(name, raising=False)

  with pytest.raises(ValueError, match="dev-only"):
    _resolve_provider("fixture", FIXTURE_MODEL_ID, None, None, None)

  with pytest.raises(ValueError, match="dev-only"):
    SkillLoader(_skills_root()).load("fixture-sleep")
  with pytest.raises(ValueError, match="dev-only"):
    SkillLoader(_skills_root()).load(FIXTURE_CANVAS_ARTIFACT_SKILL_NAME)
  with pytest.raises(ValueError, match="dev-only"):
    SkillLoader(_skills_root()).load(FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME)
  with pytest.raises(ValueError, match="dev-only"):
    SkillLoader(_skills_root()).load(FIXTURE_TERMINAL_FAILURE_SKILL_NAME)

  registry = AutonomousRegistry(
    python_executable=sys.executable,
    api_dir=ROOT / "api",
    log_dir=tmp_path,
  )
  with pytest.raises(ValueError, match="dev-only"):
    registry._build_cmd(
      profile="_fixture",
      mode="skill",
      task=None,
      skill="fixture-sleep",
      context=None,
      dev_mode=True,
    )


def test_fixture_surfaces_refuse_production_even_with_dev_flags(monkeypatch, tmp_path) -> None:
  monkeypatch.setenv("APP_ENV", "production")
  monkeypatch.setenv("ANALYST_DEV_MODE", "true")
  monkeypatch.setenv("AGENT_PROVIDER", "fixture")
  monkeypatch.setenv("AGENT_GATEWAY_ENV", "development")

  with pytest.raises(ValueError, match="dev-only"):
    _resolve_provider("fixture", FIXTURE_MODEL_ID, None, None, None)

  with pytest.raises(RuntimeError, match="dev-only"):
    from credentials import get_provider_instance

    get_provider_instance("fixture")

  with pytest.raises(ValueError, match="Unsupported provider 'fixture'"):
    AuthConfig.from_dict({"provider": "fixture", "billing_mode": "byok"})

  with pytest.raises(ValueError, match="dev-only"):
    from agent.profiles import load_profile

    load_profile("_fixture")

  with pytest.raises(ValueError, match="dev-only"):
    SkillLoader(_skills_root()).load("fixture-sleep")
  with pytest.raises(ValueError, match="dev-only"):
    SkillLoader(_skills_root()).load(FIXTURE_CANVAS_ARTIFACT_SKILL_NAME)
  with pytest.raises(ValueError, match="dev-only"):
    SkillLoader(_skills_root()).load(FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME)
  with pytest.raises(ValueError, match="dev-only"):
    SkillLoader(_skills_root()).load(FIXTURE_TERMINAL_FAILURE_SKILL_NAME)

  registry = AutonomousRegistry(
    python_executable=sys.executable,
    api_dir=ROOT / "api",
    log_dir=tmp_path,
  )
  with pytest.raises(ValueError, match="dev-only"):
    registry._build_cmd(
      profile="analyst",
      mode="skill",
      task=None,
      skill="fixture-sleep",
      context=None,
      dev_mode=True,
    )


def test_fixture_surfaces_available_in_explicit_test_environment(monkeypatch) -> None:
  monkeypatch.setenv("APP_ENV", "test")
  monkeypatch.delenv("ENVIRONMENT", raising=False)
  monkeypatch.delenv("AGENT_GATEWAY_ENV", raising=False)
  monkeypatch.delenv("NODE_ENV", raising=False)

  provider, provider_name, config = _resolve_provider(
    "fixture",
    FIXTURE_MODEL_ID,
    None,
    None,
    None,
  )

  assert provider_name == "fixture"
  assert provider.name == "fixture"
  assert {
    "model",
    "model_key",
    "effort",
    "thinking",
    "thinking_enabled_requested",
  }.isdisjoint(config)
  assert SkillLoader(_skills_root()).load("fixture-sleep").name == "fixture-sleep"
  canvas_fixture = SkillLoader(_skills_root()).load(FIXTURE_CANVAS_ARTIFACT_SKILL_NAME)
  assert canvas_fixture.name == FIXTURE_CANVAS_ARTIFACT_SKILL_NAME
  assert canvas_fixture.scope == "ticker"
  assert canvas_fixture.metadata is not None
  assert canvas_fixture.metadata["catalog"] is False
  assert canvas_fixture.agent_callable is False
  dashboard_fixture = SkillLoader(_skills_root()).load(FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME)
  assert dashboard_fixture.name == FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME
  assert dashboard_fixture.scope == "ticker"
  assert dashboard_fixture.metadata is not None
  assert dashboard_fixture.metadata["catalog"] is False
  assert dashboard_fixture.agent_callable is False
  approval_fixture = SkillLoader(_skills_root()).load(FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME)
  assert approval_fixture.name == FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME
  assert approval_fixture.scope == "ticker"
  assert approval_fixture.metadata is not None
  assert approval_fixture.metadata["catalog"] is False
  assert approval_fixture.agent_callable is False
  failure_fixture = SkillLoader(_skills_root()).load(FIXTURE_TERMINAL_FAILURE_SKILL_NAME)
  assert failure_fixture.name == FIXTURE_TERMINAL_FAILURE_SKILL_NAME
  assert failure_fixture.scope == "ticker"
  assert failure_fixture.metadata is not None
  assert failure_fixture.metadata["catalog"] is False
  assert failure_fixture.agent_callable is False
  from agent.profiles import load_profile

  fixture_profile = load_profile("_fixture")
  assert fixture_profile.name == "_fixture"
  assert not hasattr(fixture_profile, "provider")
