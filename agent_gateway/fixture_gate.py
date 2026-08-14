from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TypeVar


FIXTURE_PROVIDER_NAME = "fixture"
FIXTURE_PROFILE_NAME = "_fixture"
FIXTURE_SLEEP_SKILL_NAME = "fixture-sleep"
FIXTURE_CANVAS_ARTIFACT_SKILL_NAME = "fixture-canvas-artifact"
FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME = "fixture-dashboard-artifact"
FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME = "fixture-approval-canvas-artifact"
FIXTURE_TERMINAL_FAILURE_SKILL_NAME = "fixture-terminal-failure"
FIXTURE_SKILL_NAMES = frozenset({
  FIXTURE_SLEEP_SKILL_NAME,
  FIXTURE_CANVAS_ARTIFACT_SKILL_NAME,
  FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME,
  FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME,
  FIXTURE_TERMINAL_FAILURE_SKILL_NAME,
})
FIXTURE_APPROVAL_TOOL_NAME = "fixture_approval_gate"
FIXTURE_MODEL_ID = "fixture-scripted"

_ENV_NAMES = ("APP_ENV", "ENVIRONMENT", "AGENT_GATEWAY_ENV", "NODE_ENV")
_DEV_ENV_NAMES = ("APP_ENV", "ENVIRONMENT", "AGENT_GATEWAY_ENV")
_PRODUCTION_VALUES = frozenset({"production", "prod"})
_EXPLICIT_DEV_VALUES = frozenset({"development", "dev", "local", "test", "testing"})

_ExcT = TypeVar("_ExcT", bound=Exception)


def _normalized_env_values(names: Iterable[str] = _ENV_NAMES) -> dict[str, str]:
  return {
    name: value
    for name in names
    if (value := os.getenv(name, "").strip().lower())
  }


def fixture_production_detected() -> bool:
  """Return True when an environment variable positively identifies production."""
  return any(value in _PRODUCTION_VALUES for value in _normalized_env_values().values())


def fixture_provider_available() -> bool:
  """Fail-closed availability check for the deterministic fixture surface.

  The existing Python-side production convention is ``APP_ENV=production``.
  To avoid stale dev flags exposing the fixture, any known production value in
  common environment selectors wins. If no explicit dev/test environment is
  present, the fixture stays unavailable.
  """
  values = _normalized_env_values()
  if any(value in _PRODUCTION_VALUES for value in values.values()):
    return False
  dev_values = _normalized_env_values(_DEV_ENV_NAMES)
  return any(value in _EXPLICIT_DEV_VALUES for value in dev_values.values())


def fixture_unavailable_message(surface: str = "fixture provider") -> str:
  return (
    f"{surface} is dev-only and is unavailable in this environment. "
    "Set APP_ENV=development or APP_ENV=test for local fixture runs; "
    "APP_ENV=production/prod always refuses it."
  )


def require_fixture_provider_available(
  surface: str = "fixture provider",
  *,
  error_type: type[_ExcT] = RuntimeError,
) -> None:
  if not fixture_provider_available():
    raise error_type(fixture_unavailable_message(surface))


def is_fixture_skill_name(name: str | None) -> bool:
  return str(name or "").strip() in FIXTURE_SKILL_NAMES


def is_fixture_profile_name(name: str | None) -> bool:
  return str(name or "").strip() == FIXTURE_PROFILE_NAME


__all__ = [
  "FIXTURE_APPROVAL_TOOL_NAME",
  "FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME",
  "FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME",
  "FIXTURE_CANVAS_ARTIFACT_SKILL_NAME",
  "FIXTURE_MODEL_ID",
  "FIXTURE_PROFILE_NAME",
  "FIXTURE_PROVIDER_NAME",
  "FIXTURE_SLEEP_SKILL_NAME",
  "FIXTURE_SKILL_NAMES",
  "FIXTURE_TERMINAL_FAILURE_SKILL_NAME",
  "fixture_production_detected",
  "fixture_provider_available",
  "fixture_unavailable_message",
  "is_fixture_profile_name",
  "is_fixture_skill_name",
  "require_fixture_provider_available",
]
