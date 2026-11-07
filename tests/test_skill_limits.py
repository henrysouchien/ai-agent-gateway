from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_gateway.skill_limits import (  # noqa: E402
  ActiveSkillAdmission,
  AutonomousSkillAdmissionPolicy,
  SkillExecutionLimits,
  active_skill_admission_from_mapping,
  active_skill_admission_to_mapping,
  autonomous_skill_admission_policy_from_mapping,
  autonomous_skill_admission_policy_to_mapping,
  reconcile_skill_admission,
  skill_execution_limits_from_mapping,
  skill_execution_limits_to_mapping,
)


def _limits() -> SkillExecutionLimits:
  return SkillExecutionLimits()


@pytest.mark.parametrize("field", ["max_tokens", "max_turns", "max_budget_usd"])
def test_limits_carry_no_retired_ceiling(field: str) -> None:
  with pytest.raises(TypeError):
    SkillExecutionLimits(**{field: 1})


def test_closed_codecs_round_trip_fresh_mappings() -> None:
  limits = _limits()
  active = ActiveSkillAdmission("quant-research", limits)
  policy = AutonomousSkillAdmissionPolicy(True, limits)

  limits_payload = skill_execution_limits_to_mapping(limits)
  active_payload = active_skill_admission_to_mapping(active)
  policy_payload = autonomous_skill_admission_policy_to_mapping(policy)

  assert limits_payload == {}
  assert skill_execution_limits_from_mapping(limits_payload) == limits
  assert active_payload == {
    "skill_name": "quant-research",
    "execution_limits": {},
  }
  assert active_skill_admission_from_mapping(active_payload) == active
  assert policy_payload["execution_limits"] == {}
  assert autonomous_skill_admission_policy_from_mapping(policy_payload) == policy
  limits_payload["max_tokens"] = 99
  assert skill_execution_limits_to_mapping(limits) == {}


def test_limits_codec_reads_legacy_record_and_drops_its_ceilings() -> None:
  assert skill_execution_limits_from_mapping({
    "max_turns": 20,
    "max_tokens": 32_000,
    "max_budget_usd": 20.0,
  }) == SkillExecutionLimits()
  assert skill_execution_limits_from_mapping({"max_tokens": True}) == (
    SkillExecutionLimits()
  )
  assert skill_execution_limits_from_mapping({}) == SkillExecutionLimits()


@pytest.mark.parametrize(
  "payload",
  [
    {"max_tokens": None, "extra": None},
    {
      "max_turns": None,
      "max_tokens": None,
      "max_budget_usd": None,
      "extra": None,
    },
  ],
)
def test_limits_codec_rejects_extra_fields(payload: object) -> None:
  with pytest.raises(ValueError):
    skill_execution_limits_from_mapping(payload)


def test_policy_codec_rejects_bool_coercion() -> None:
  with pytest.raises(TypeError):
    autonomous_skill_admission_policy_from_mapping({
      "skill_resume_allowed": 1,
      "execution_limits": skill_execution_limits_to_mapping(_limits()),
    })



def test_active_admission_requires_exact_nested_type_and_canonical_name() -> None:
  with pytest.raises(TypeError):
    ActiveSkillAdmission("quant-research", object())  # type: ignore[arg-type]
  with pytest.raises(ValueError):
    ActiveSkillAdmission(" quant-research", _limits())


def test_reconciliation_keeps_the_active_admission_on_mismatch() -> None:
  active = ActiveSkillAdmission("quant-research", _limits())

  assert reconcile_skill_admission(
    skill_name="quant-research",
    execution_limits=_limits(),
    active_admission=active,
  ) is active
  assert reconcile_skill_admission(
    skill_name=None,
    execution_limits=None,
    active_admission=active,
  ) is active
  assert reconcile_skill_admission(
    skill_name="market-scan",
    execution_limits=_limits(),
    active_admission=active,
  ) is active
  with pytest.raises(TypeError):
    reconcile_skill_admission(
      skill_name="quant-research",
      execution_limits=None,
      active_admission=None,
    )


def test_skill_limits_leaf_import_does_not_load_application_modules() -> None:
  script = """
import json
import sys
from agent_gateway.skill_limits import SkillExecutionLimits
print(json.dumps({
  'module': SkillExecutionLimits.__module__,
  'application_modules': sorted(name for name in sys.modules if name == 'agent' or name.startswith('agent.skills') or name.startswith('api.agent')),
}))
"""
  result = subprocess.run(
    [sys.executable, "-c", script],
    cwd=ROOT,
    env={
      key: value
      for key, value in os.environ.items()
      if key != "PYTHONPATH"
    },
    text=True,
    capture_output=True,
    check=True,
  )
  assert json.loads(result.stdout) == {
    "module": "agent_gateway.skill_limits",
    "application_modules": [],
  }


def test_skill_limits_leaf_is_importable_from_built_wheel(tmp_path: Path) -> None:
  wheel_dir = tmp_path / "wheel"
  installed = tmp_path / "installed"
  wheel_dir.mkdir()
  installed.mkdir()
  subprocess.run(
    [
      sys.executable,
      "-m",
      "pip",
      "wheel",
      "--no-deps",
      "--no-build-isolation",
      "--wheel-dir",
      str(wheel_dir),
      str(ROOT),
    ],
    check=True,
    capture_output=True,
    text=True,
    env={
      **{k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("PIP_")},
      "PIP_NO_INDEX": "1",
      "PIP_DISABLE_PIP_VERSION_CHECK": "1",
      "PIP_CACHE_DIR": str(tmp_path / "pip-cache"),
    },
  )
  wheel = next(wheel_dir.glob("ai_agent_gateway-*.whl"))
  with ZipFile(wheel) as archive:
    assert "agent_gateway/skill_limits.py" in archive.namelist()
    archive.extractall(installed)

  result = subprocess.run(
    [
      sys.executable,
      "-c",
      (
        "from agent_gateway.skill_limits import "
        "SkillExecutionLimits; "
        "print(list(SkillExecutionLimits().__dataclass_fields__))"
      ),
    ],
    cwd=installed,
    env={**os.environ, "PYTHONPATH": str(installed)},
    check=True,
    capture_output=True,
    text=True,
  )
  assert result.stdout.strip() == "[]"
