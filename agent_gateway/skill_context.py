from __future__ import annotations

from contextvars import ContextVar, Token


_ACTIVE_SKILL: ContextVar[str | None] = ContextVar("agent_gateway_active_skill", default=None)


def current_skill() -> str | None:
  skill = _ACTIVE_SKILL.get()
  return skill or None


def set_current_skill(skill: str | None) -> Token[str | None]:
  normalized = str(skill or "").strip() or None
  return _ACTIVE_SKILL.set(normalized)


def reset_current_skill(token: Token[str | None]) -> None:
  _ACTIVE_SKILL.reset(token)


def clear_current_skill() -> None:
  _ACTIVE_SKILL.set(None)


__all__ = [
  "clear_current_skill",
  "current_skill",
  "reset_current_skill",
  "set_current_skill",
]
