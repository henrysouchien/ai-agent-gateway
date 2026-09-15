from __future__ import annotations

from typing import Any, Literal, overload

from fastapi import HTTPException, status


ExactRole = Literal["owner", "invite"]


_MISSING = object()
_EXACT_ROLES = frozenset({"owner", "invite"})


@overload
def require_exact_role(value: Any) -> ExactRole: ...


@overload
def require_exact_role(value: Any, role: str) -> None: ...


def require_exact_role(value: Any, role: object = _MISSING) -> str | None:
  """Validate an authority role, optionally requiring it on a session."""
  if role is _MISSING:
    if type(value) is not str or value not in _EXACT_ROLES:
      raise ValueError("role must be exactly 'owner' or 'invite'")
    return value
  expected = require_exact_role(role)
  if getattr(value, "role", None) != expected:
    raise HTTPException(
      status_code=status.HTTP_403_FORBIDDEN,
      detail={
        "error": "role_required",
        "message": f"This operation requires the exact {expected!r} role.",
      },
    )
  return None


__all__ = ["require_exact_role"]
