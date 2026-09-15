"""Dependency-neutral skill admission facts shared across Gateway runtimes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


_LIMIT_FIELDS = frozenset({"max_turns", "max_tokens", "max_budget_usd"})
_ACTIVE_FIELDS = frozenset({"skill_name", "execution_limits"})
_AUTONOMOUS_POLICY_FIELDS = frozenset({
  "skill_resume_allowed",
  "execution_limits",
})


def _closed_mapping(
  value: object,
  *,
  field_name: str,
  fields: frozenset[str],
) -> dict[str, object]:
  if not isinstance(value, Mapping):
    raise TypeError(f"{field_name} must be a mapping")
  try:
    items = tuple(value.items())
  except Exception as exc:
    raise TypeError(f"{field_name} must be a readable mapping") from exc
  payload: dict[str, object] = {}
  for key, item in items:
    if type(key) is not str:
      raise TypeError(f"{field_name} keys must be exact strings")
    if key in payload:
      raise ValueError(f"{field_name} contains duplicate keys")
    payload[key] = item
  if frozenset(payload) != fields:
    raise ValueError(f"{field_name} has invalid fields")
  return payload


def _optional_positive_int(value: object, *, field_name: str) -> int | None:
  if value is None:
    return None
  if type(value) is not int:
    raise TypeError(f"{field_name} must be an exact integer or None")
  if value <= 0:
    raise ValueError(f"{field_name} must be positive")
  return value


def _optional_positive_float(value: object, *, field_name: str) -> float | None:
  if value is None:
    return None
  if type(value) is not int and type(value) is not float:
    raise TypeError(f"{field_name} must be an exact number or None")
  try:
    normalized = float(value)
  except OverflowError as exc:
    raise ValueError(f"{field_name} must be finite and positive") from exc
  if not math.isfinite(normalized) or normalized <= 0:
    raise ValueError(f"{field_name} must be finite and positive")
  return normalized


def _canonical_skill_name(value: object, *, field_name: str) -> str:
  if type(value) is not str:
    raise TypeError(f"{field_name} must be an exact string")
  if not value or value != value.strip():
    raise ValueError(f"{field_name} must be canonical non-empty text")
  return value


@dataclass(frozen=True, slots=True)
class SkillExecutionLimits:
  max_turns: int | None
  max_tokens: int | None
  max_budget_usd: float | None

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "max_turns",
      _optional_positive_int(self.max_turns, field_name="max_turns"),
    )
    object.__setattr__(
      self,
      "max_tokens",
      _optional_positive_int(self.max_tokens, field_name="max_tokens"),
    )
    object.__setattr__(
      self,
      "max_budget_usd",
      _optional_positive_float(
        self.max_budget_usd,
        field_name="max_budget_usd",
      ),
    )


@dataclass(frozen=True, slots=True)
class ActiveSkillAdmission:
  skill_name: str
  execution_limits: SkillExecutionLimits

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "skill_name",
      _canonical_skill_name(self.skill_name, field_name="skill_name"),
    )
    if type(self.execution_limits) is not SkillExecutionLimits:
      raise TypeError("execution_limits must be exact SkillExecutionLimits")


@dataclass(frozen=True, slots=True)
class AutonomousSkillAdmissionPolicy:
  skill_resume_allowed: bool
  execution_limits: SkillExecutionLimits

  def __post_init__(self) -> None:
    if type(self.skill_resume_allowed) is not bool:
      raise TypeError("skill_resume_allowed must be an exact bool")
    if type(self.execution_limits) is not SkillExecutionLimits:
      raise TypeError("execution_limits must be exact SkillExecutionLimits")


@runtime_checkable
class AutonomousSkillAdmissionPolicyResolver(Protocol):
  def __call__(self, skill_name: str) -> AutonomousSkillAdmissionPolicy: ...


def reconcile_skill_admission(
  *,
  skill_name: str | None,
  execution_limits: SkillExecutionLimits | None,
  active_admission: ActiveSkillAdmission | None,
) -> ActiveSkillAdmission | None:
  """Reconcile named and inline facts without allowing either to override."""

  if active_admission is not None and type(active_admission) is not ActiveSkillAdmission:
    raise TypeError("active_admission must be exact ActiveSkillAdmission or None")
  if skill_name is None:
    if execution_limits is not None:
      raise ValueError("execution limits require a named skill")
    return active_admission
  if type(execution_limits) is not SkillExecutionLimits:
    raise TypeError(
      "execution_limits must be exact SkillExecutionLimits"
    )
  named = ActiveSkillAdmission(
    skill_name=skill_name,
    execution_limits=execution_limits,
  )
  if active_admission is None:
    return named
  if active_admission != named:
    raise ValueError("named and active skill admission facts do not match")
  return active_admission


def skill_execution_limits_to_mapping(
  value: SkillExecutionLimits,
) -> dict[str, int | float | None]:
  if type(value) is not SkillExecutionLimits:
    raise TypeError("value must be exact SkillExecutionLimits")
  return {
    "max_turns": value.max_turns,
    "max_tokens": value.max_tokens,
    "max_budget_usd": value.max_budget_usd,
  }


def skill_execution_limits_from_mapping(
  value: object,
) -> SkillExecutionLimits:
  payload = _closed_mapping(
    value,
    field_name="admitted skill execution limits",
    fields=_LIMIT_FIELDS,
  )
  max_turns = _optional_positive_int(
    payload["max_turns"],
    field_name="max_turns",
  )
  max_tokens = _optional_positive_int(
    payload["max_tokens"],
    field_name="max_tokens",
  )
  max_budget_usd = _optional_positive_float(
    payload["max_budget_usd"],
    field_name="max_budget_usd",
  )
  return SkillExecutionLimits(
    max_turns=max_turns,
    max_tokens=max_tokens,
    max_budget_usd=max_budget_usd,
  )


def active_skill_admission_to_mapping(
  value: ActiveSkillAdmission,
) -> dict[str, object]:
  if type(value) is not ActiveSkillAdmission:
    raise TypeError("value must be exact ActiveSkillAdmission")
  return {
    "skill_name": value.skill_name,
    "execution_limits": skill_execution_limits_to_mapping(
      value.execution_limits
    ),
  }


def active_skill_admission_from_mapping(value: object) -> ActiveSkillAdmission:
  payload = _closed_mapping(
    value,
    field_name="active skill admission",
    fields=_ACTIVE_FIELDS,
  )
  skill_name = _canonical_skill_name(
    payload["skill_name"],
    field_name="skill_name",
  )
  return ActiveSkillAdmission(
    skill_name=skill_name,
    execution_limits=skill_execution_limits_from_mapping(
      payload["execution_limits"]
    ),
  )


def autonomous_skill_admission_policy_to_mapping(
  value: AutonomousSkillAdmissionPolicy,
) -> dict[str, object]:
  if type(value) is not AutonomousSkillAdmissionPolicy:
    raise TypeError("value must be exact AutonomousSkillAdmissionPolicy")
  return {
    "skill_resume_allowed": value.skill_resume_allowed,
    "execution_limits": skill_execution_limits_to_mapping(
      value.execution_limits
    ),
  }


def autonomous_skill_admission_policy_from_mapping(
  value: object,
) -> AutonomousSkillAdmissionPolicy:
  payload = _closed_mapping(
    value,
    field_name="autonomous skill admission policy",
    fields=_AUTONOMOUS_POLICY_FIELDS,
  )
  resume_allowed = payload["skill_resume_allowed"]
  if type(resume_allowed) is not bool:
    raise TypeError("skill_resume_allowed must be an exact bool")
  return AutonomousSkillAdmissionPolicy(
    skill_resume_allowed=resume_allowed,
    execution_limits=skill_execution_limits_from_mapping(
      payload["execution_limits"]
    ),
  )


__all__ = [
  "ActiveSkillAdmission",
  "AutonomousSkillAdmissionPolicy",
  "AutonomousSkillAdmissionPolicyResolver",
  "SkillExecutionLimits",
  "active_skill_admission_from_mapping",
  "active_skill_admission_to_mapping",
  "autonomous_skill_admission_policy_from_mapping",
  "autonomous_skill_admission_policy_to_mapping",
  "reconcile_skill_admission",
  "skill_execution_limits_from_mapping",
  "skill_execution_limits_to_mapping",
]
