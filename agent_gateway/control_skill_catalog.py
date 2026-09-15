"""Dependency-neutral control-plane skill catalog contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
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
_CONTROL_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def is_control_skill_name(value: object) -> bool:
  """Return whether *value* is an exact reusable control skill name."""

  return (
    type(value) is str
    and _CONTROL_SKILL_NAME_RE.fullmatch(value) is not None
  )


def _require_string(value: object, *, field_name: str) -> str:
  if type(value) is not str:
    raise TypeError(f"{field_name} must be an exact str")
  return value


def _require_text(value: object, *, field_name: str) -> None:
  text = _require_string(value, field_name=field_name)
  if not text or text != text.strip():
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
class ControlSkillSummary:
  """Exact immutable metadata exposed by the control skill list wire."""

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
  max_budget_usd: float | None
  persist_state: bool
  typed_contract: str | None
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

  def __post_init__(self) -> None:
    for field_name in (
      "name",
      "label",
      "scope",
      "action_class",
      "approval_policy",
      "path",
    ):
      _require_text(getattr(self, field_name), field_name=field_name)
    for field_name in ("description", "version"):
      _require_string(getattr(self, field_name), field_name=field_name)
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
    if self.max_turns is not None and type(self.max_turns) is not int:
      raise TypeError("max_turns must be None or an exact int")
    if self.max_budget_usd is not None:
      if type(self.max_budget_usd) not in {int, float}:
        raise TypeError(
          "max_budget_usd must be None or an exact int or float"
        )
      object.__setattr__(
        self,
        "max_budget_usd",
        float(self.max_budget_usd),
      )
    if self.typed_contract is not None and type(self.typed_contract) is not str:
      raise TypeError("typed_contract must be None or an exact str")
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


@dataclass(frozen=True, slots=True)
class ControlSkillDetail(ControlSkillSummary):
  """One control skill summary plus its selected resolved body."""

  body: str

  def __post_init__(self) -> None:
    ControlSkillSummary.__post_init__(self)
    _require_string(self.body, field_name="body")


@runtime_checkable
class ControlSkillCatalog(Protocol):
  def list_skills(self) -> tuple[ControlSkillSummary, ...]: ...

  def resolve_skill(self, selector: object) -> ControlSkillDetail: ...


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
  "ControlSkillDetail",
  "ControlSkillSummary",
  "ControlSkillUnavailableCode",
  "ControlSkillUnavailableError",
  "is_control_skill_name",
]
