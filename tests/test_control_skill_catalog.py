from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent_gateway.control_skill_catalog import (
  ControlSkillCatalog,
  ControlSkillDetail,
  ControlSkillSummary,
  ControlSkillUnavailableError,
  is_control_skill_name,
)


ROOT = Path(__file__).resolve().parents[1]


def _summary_arguments() -> dict[str, object]:
  return {
    "name": "strict-skill",
    "label": "Strict Skill",
    "description": "Strict description.",
    "agent_description": "Agent-facing description.",
    "version": "2.3",
    "scope": "ticker",
    "requires_portfolio_context": False,
    "required_context": ["ticker"],
    "agent_callable": True,
    "resumable": True,
    "persist_state": True,
    "typed_contract": None,
    "catalog": True,
    "profiles": ["analyst"],
    "modes": ["skill"],
    "outputs": ["platform:skill-result-envelope@1"],
    "action_class": "read_only",
    "approval_policy": "runtime_policy",
    "tier_availability": ["paid"],
    "credential_requirements": ["market_data"],
    "schedule_eligible": True,
    "can_launch": True,
    "can_schedule": True,
    "blocked_reason": None,
    "path": "skills/strict-skill.md",
  }


def _summary(**overrides: object) -> ControlSkillSummary:
  return ControlSkillSummary(**{
    **_summary_arguments(),
    **overrides,
  })  # type: ignore[arg-type]


def _detail(**overrides: object) -> ControlSkillDetail:
  values = {**_summary_arguments(), "body": "Resolved methodology."}
  values.update(overrides)
  return ControlSkillDetail(**values)


def test_control_summary_and_detail_have_exact_frozen_wire_fields() -> None:
  expected_summary = (
    "name",
    "label",
    "description",
    "agent_description",
    "version",
    "scope",
    "requires_portfolio_context",
    "required_context",
    "agent_callable",
    "resumable",
    "persist_state",
    "typed_contract",
    "catalog",
    "profiles",
    "modes",
    "outputs",
    "action_class",
    "approval_policy",
    "tier_availability",
    "credential_requirements",
    "schedule_eligible",
    "can_launch",
    "can_schedule",
    "blocked_reason",
    "path",
  )
  assert tuple(field.name for field in fields(ControlSkillSummary)) == (
    expected_summary
  )
  assert tuple(field.name for field in fields(ControlSkillDetail)) == (
    *expected_summary,
    "body",
  )
  assert len(expected_summary) == 25
  summary = _summary()
  detail = _detail()
  with pytest.raises(FrozenInstanceError):
    summary.name = "changed"  # type: ignore[misc]
  with pytest.raises(FrozenInstanceError):
    detail.body = "changed"  # type: ignore[misc]


def test_control_wire_snapshots_all_six_sequence_fields() -> None:
  source_values = {
    "required_context": ["ticker"],
    "profiles": ["analyst"],
    "modes": ["skill"],
    "outputs": ["platform:result@1"],
    "tier_availability": ["paid"],
    "credential_requirements": ["market_data"],
  }
  summary = _summary(**source_values)
  for values in source_values.values():
    values.append("mutated")

  assert summary.required_context == ("ticker",)
  assert summary.profiles == ("analyst",)
  assert summary.modes == ("skill",)
  assert summary.outputs == ("platform:result@1",)
  assert summary.tier_availability == ("paid",)
  assert summary.credential_requirements == ("market_data",)
  assert all(
    type(getattr(summary, field_name)) is tuple
    for field_name in source_values
  )


@pytest.mark.parametrize(
  "field_name",
  ["name", "label", "scope", "action_class", "approval_policy", "path"],
)
@pytest.mark.parametrize("value", [None, 1, b"text", "", " padded "])
def test_control_summary_rejects_invalid_canonical_text(
  field_name: str,
  value: object,
) -> None:
  with pytest.raises((TypeError, ValueError), match=field_name):
    _summary(**{field_name: value})


@pytest.mark.parametrize("value", ["Bad_Name", "-bad", "bad-", "bad--name"])
def test_control_name_grammar_is_adapter_owned(value: str) -> None:
  assert _summary(name=value).name == value
  assert is_control_skill_name(value) is False
  assert is_control_skill_name(object()) is False
  assert is_control_skill_name("1-valid-skill") is True


def test_control_wire_preserves_shallow_external_text_shapes() -> None:
  summary = _summary(
    description="",
    version=" padded version ",
    typed_contract="",
    required_context=(),
    profiles=(),
    modes=(),
    outputs=(),
    tier_availability=(),
    credential_requirements=(),
  )
  detail = _detail(body="", description="", version="")

  assert summary.description == ""
  assert summary.version == " padded version "
  assert summary.typed_contract == ""
  assert detail.body == ""


@pytest.mark.parametrize("field_name", ["agent_description", "blocked_reason"])
@pytest.mark.parametrize("value", [1, b"text", "", " padded "])
def test_control_summary_rejects_invalid_optional_text(
  field_name: str,
  value: object,
) -> None:
  with pytest.raises((TypeError, ValueError), match=field_name):
    _summary(**{field_name: value})


@pytest.mark.parametrize(
  "field_name",
  [
    "requires_portfolio_context",
    "agent_callable",
    "resumable",
    "persist_state",
    "catalog",
    "schedule_eligible",
    "can_launch",
    "can_schedule",
  ],
)
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_control_summary_requires_exact_booleans(
  field_name: str,
  value: object,
) -> None:
  with pytest.raises(TypeError, match=field_name):
    _summary(**{field_name: value})


@pytest.mark.parametrize("field_name", ["max_turns", "max_budget_usd"])
def test_control_summary_carries_no_turn_or_cost_number(field_name: str) -> None:
  with pytest.raises(TypeError, match=field_name):
    _summary(**{field_name: 1})


@pytest.mark.parametrize("value", [1, b"contract", object()])
def test_control_summary_requires_typed_contract_string_or_none(
  value: object,
) -> None:
  with pytest.raises(TypeError, match="typed_contract"):
    _summary(typed_contract=value)


@pytest.mark.parametrize(
  "field_name",
  [
    "required_context",
    "profiles",
    "modes",
    "outputs",
    "tier_availability",
    "credential_requirements",
  ],
)
@pytest.mark.parametrize(
  "value",
  ["ticker", b"ticker", {"ticker"}, (item for item in ("ticker",)), 1],
)
def test_control_summary_rejects_non_sequence_fields(
  field_name: str,
  value: object,
) -> None:
  with pytest.raises(TypeError, match=field_name):
    _summary(**{field_name: value})


@pytest.mark.parametrize("value", [(1,), ("",), (" padded ",), (b"ticker",)])
def test_control_summary_rejects_invalid_sequence_members(value: object) -> None:
  with pytest.raises((TypeError, ValueError), match="required_context item"):
    _summary(required_context=value)


def test_control_catalog_protocol_and_typed_error_are_dependency_neutral() -> None:
  summary = _summary()
  detail = _detail()

  class Catalog:
    def list_skills(self) -> tuple[ControlSkillSummary, ...]:
      return (summary,)

    def resolve_skill(self, _selector: object) -> ControlSkillDetail:
      return detail

  assert isinstance(Catalog(), ControlSkillCatalog)
  error = ControlSkillUnavailableError(
    code="unknown",
    selector="hidden-skill",
  )
  assert error.code == "unknown"
  assert error.selector == "hidden-skill"
  assert not hasattr(error, "diagnostic")
  assert "hidden-skill" not in str(error)


@pytest.mark.parametrize(
  ("code", "error_type"),
  [
    (None, TypeError),
    (True, TypeError),
    (1, TypeError),
    ("hidden", ValueError),
    ("", ValueError),
  ],
)
def test_control_unavailable_error_validates_exact_code_membership(
  code: object,
  error_type: type[Exception],
) -> None:
  with pytest.raises(error_type, match="code"):
    ControlSkillUnavailableError(
      code=code,  # type: ignore[arg-type]
      selector="strict-skill",
    )


def test_control_contract_imports_without_application_modules() -> None:
  environment = dict(os.environ)
  environment["PYTHONPATH"] = str(ROOT)
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      (
        "import json, sys; "
        "from agent_gateway.control_skill_catalog import "
        "ControlSkillDetail, ControlSkillSummary; "
        "print(json.dumps({'summary': ControlSkillSummary.__module__, "
        "'detail': ControlSkillDetail.__module__, "
        "'application_modules': sorted(name for name in sys.modules "
        "if name == 'agent' or name.startswith('agent.skills') "
        "or name.startswith('api.agent'))}))"
      ),
    ],
    cwd=ROOT,
    env=environment,
    check=True,
    capture_output=True,
    text=True,
  )

  assert json.loads(result.stdout) == {
    "summary": "agent_gateway.control_skill_catalog",
    "detail": "agent_gateway.control_skill_catalog",
    "application_modules": [],
  }
