from __future__ import annotations

from contextvars import ContextVar, Token

from .skill_limits import ActiveSkillAdmission


_ACTIVE_SKILL_ADMISSION: ContextVar[ActiveSkillAdmission | None] = ContextVar(
  "agent_gateway_active_skill_admission",
  default=None,
)


def current_skill_admission() -> ActiveSkillAdmission | None:
  admission = _ACTIVE_SKILL_ADMISSION.get()
  if admission is not None and type(admission) is not ActiveSkillAdmission:
    raise RuntimeError("active skill admission context is malformed")
  return admission


def current_skill() -> str | None:
  admission = current_skill_admission()
  return admission.skill_name if admission is not None else None


def set_current_skill(
  admission: ActiveSkillAdmission | None,
) -> Token[ActiveSkillAdmission | None]:
  if admission is not None and type(admission) is not ActiveSkillAdmission:
    raise TypeError("active skill context requires exact ActiveSkillAdmission")
  return _ACTIVE_SKILL_ADMISSION.set(admission)


def set_current_skill_admission(
  admission: ActiveSkillAdmission | None,
) -> Token[ActiveSkillAdmission | None]:
  return set_current_skill(admission)


def reset_current_skill(token: Token[ActiveSkillAdmission | None]) -> None:
  _ACTIVE_SKILL_ADMISSION.reset(token)


def reset_current_skill_admission(
  token: Token[ActiveSkillAdmission | None],
) -> None:
  reset_current_skill(token)


def clear_current_skill() -> None:
  _ACTIVE_SKILL_ADMISSION.set(None)


__all__ = [
  "clear_current_skill",
  "current_skill",
  "current_skill_admission",
  "reset_current_skill",
  "reset_current_skill_admission",
  "set_current_skill",
  "set_current_skill_admission",
]
