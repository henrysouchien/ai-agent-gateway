from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import json
import math
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

_PACKAGE_IMPORTS = {
  "__future__": frozenset({"annotations"}),
  "collections.abc": frozenset({"Sequence"}),
  "dataclasses": frozenset({"dataclass"}),
  "typing": frozenset({"Literal", "Protocol", "runtime_checkable"}),
}


def _assert_package_import_boundary(source: str) -> None:
  tree = ast.parse(source)
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      assert tuple((alias.name, alias.asname) for alias in node.names) == (
        ("re", None),
      )
      continue
    if not isinstance(node, ast.ImportFrom):
      continue
    assert node.level == 0
    allowed = _PACKAGE_IMPORTS.get(node.module or "")
    assert allowed is not None
    for alias in node.names:
      assert alias.asname is None
      assert alias.name in allowed


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
    "max_turns": 7,
    "max_budget_usd": 1.75,
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
    "max_turns",
    "max_budget_usd",
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
  assert len(expected_summary) == 27
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


def test_control_wire_preserves_shallow_external_text_and_limit_shapes() -> None:
  summary = _summary(
    description="",
    version=" padded version ",
    max_turns=-3,
    max_budget_usd=math.inf,
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
  assert summary.max_turns == -3
  assert summary.max_budget_usd == math.inf
  assert type(summary.max_budget_usd) is float
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


@pytest.mark.parametrize("value", [True, False, 1.5, "2"])
def test_control_summary_rejects_non_exact_max_turns(value: object) -> None:
  with pytest.raises(TypeError, match="max_turns"):
    _summary(max_turns=value)


@pytest.mark.parametrize("value", [True, False, "1.5", object()])
def test_control_summary_rejects_invalid_budget_shape(value: object) -> None:
  with pytest.raises(TypeError, match="max_budget_usd"):
    _summary(max_budget_usd=value)


def test_control_summary_normalizes_budget_to_float() -> None:
  assert _summary(max_budget_usd=None).max_budget_usd is None
  assert _summary(max_budget_usd=-2).max_budget_usd == -2.0
  assert type(_summary(max_budget_usd=2).max_budget_usd) is float


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


def test_control_contract_module_is_stdlib_only_and_not_reexported() -> None:
  source_path = ROOT / "agent_gateway" / "control_skill_catalog.py"
  _assert_package_import_boundary(source_path.read_text(encoding="utf-8"))

  package_init = (ROOT / "agent_gateway" / "__init__.py").read_text(
    encoding="utf-8"
  )
  assert "control_skill_catalog" not in package_init


@pytest.mark.parametrize(
  "statement",
  [
    "import os",
    "import re as regex",
    "from pathlib import Path",
    "from dataclasses import dataclass as record",
  ],
)
def test_control_contract_import_guard_rejects_foreign_or_aliased_imports(
  statement: str,
) -> None:
  with pytest.raises(AssertionError):
    _assert_package_import_boundary(statement)


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
