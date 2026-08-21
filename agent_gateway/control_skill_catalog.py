"""Dependency-neutral control-plane skill catalog contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Literal, Protocol, runtime_checkable


ControlSkillUnavailableCode = Literal[
  "invalid_selector",
  "invalid",
  "unknown",
]
_CONTROL_SKILL_UNAVAILABLE_CODES = frozenset({
  "invalid_selector",
  "invalid",
  "unknown",
})


def _require_text(value: object, *, field_name: str) -> None:
  if type(value) is not str:
    raise TypeError(f"{field_name} must be an exact str")
  if not value or value != value.strip():
    raise ValueError(f"{field_name} must be canonical non-empty text")


def _require_optional_text(value: object, *, field_name: str) -> None:
  if value is None:
    return
  _require_text(value, field_name=field_name)


def _snapshot_text_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
  if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
    raise TypeError(f"{field_name} must be a sequence of exact strings")
  snapshot = tuple(value)
  for item in snapshot:
    _require_text(item, field_name=f"{field_name} item")
  return snapshot


@dataclass(frozen=True, slots=True)
class ControlSkillDefinition:
  """Exact immutable data exposed by the control skill catalog."""

  name: str
  label: str
  description: str
  agent_description: str | None
  version: str
  scope: str
  requires_portfolio_context: bool
  required_context: tuple[str, ...]
  agent_callable: bool
  resumable: bool
  max_turns: int | None
  max_budget_usd: int | float | None
  persist_state: bool
  typed_contract: None
  catalog: bool
  profiles: tuple[str, ...]
  modes: tuple[str, ...]
  outputs: tuple[str, ...]
  action_class: str
  approval_policy: str
  tier_availability: tuple[str, ...]
  credential_requirements: tuple[str, ...]
  schedule_eligible: bool
  can_launch: bool
  can_schedule: bool
  blocked_reason: str | None
  path: str
  body: str

  def __post_init__(self) -> None:
    for field_name in (
      "name",
      "label",
      "description",
      "version",
      "scope",
      "action_class",
      "approval_policy",
      "path",
      "body",
    ):
      _require_text(getattr(self, field_name), field_name=field_name)
    for field_name in ("agent_description", "blocked_reason"):
      _require_optional_text(getattr(self, field_name), field_name=field_name)
    for field_name in (
      "requires_portfolio_context",
      "agent_callable",
      "resumable",
      "persist_state",
      "catalog",
      "schedule_eligible",
      "can_launch",
      "can_schedule",
    ):
      if type(getattr(self, field_name)) is not bool:
        raise TypeError(f"{field_name} must be an exact bool")
    if self.catalog is not True:
      raise ValueError("catalog must be exactly True")
    if self.max_turns is not None:
      if type(self.max_turns) is not int:
        raise TypeError("max_turns must be None or an exact int")
      if self.max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if self.max_budget_usd is not None:
      if type(self.max_budget_usd) not in {int, float}:
        raise TypeError(
          "max_budget_usd must be None or an exact int or float"
        )
      if not math.isfinite(self.max_budget_usd) or self.max_budget_usd <= 0:
        raise ValueError("max_budget_usd must be finite and positive")
    if self.typed_contract is not None:
      raise TypeError("typed_contract must be exactly None")
    for field_name in (
      "required_context",
      "profiles",
      "modes",
      "outputs",
      "tier_availability",
      "credential_requirements",
    ):
      object.__setattr__(
        self,
        field_name,
        _snapshot_text_sequence(
          getattr(self, field_name),
          field_name=field_name,
        ),
      )


@runtime_checkable
class ControlSkillCatalog(Protocol):
  def list_skills(self) -> tuple[ControlSkillDefinition, ...]: ...

  def resolve_skill(self, skill_name: object) -> ControlSkillDefinition: ...


class ControlSkillUnavailableError(LookupError):
  """Typed non-enumerating refusal from a control skill catalog."""

  def __init__(
    self,
    *,
    code: ControlSkillUnavailableCode,
    selector: object,
  ) -> None:
    if type(code) is not str:
      raise TypeError("code must be an exact str")
    if code not in _CONTROL_SKILL_UNAVAILABLE_CODES:
      raise ValueError("code must be a recognized control skill refusal code")
    self.code = code
    self.selector = selector
    super().__init__("control skill is unavailable")


__all__ = [
  "ControlSkillCatalog",
  "ControlSkillDefinition",
  "ControlSkillUnavailableCode",
  "ControlSkillUnavailableError",
]
