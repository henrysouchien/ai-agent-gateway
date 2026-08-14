from __future__ import annotations

from typing import Any, overload

from fastapi import HTTPException, status


_MISSING = object()
_EXACT_ROLES = frozenset({"owner", "invite"})


@overload
def require_exact_role(value: Any) -> str: ...


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


def coerce_stored_role(value: Any) -> str:
  """Return a usable role for a role read off a PERSISTED record.

  Authorizing a live request is a security decision and stays strict —
  ``require_exact_role``. Reading a role out of a record we already wrote is
  not: raising there makes the runtime unable to read its own history, and the
  callers turn that into a silent skip. Records written before the role plane
  carry no role; owner is the only role that can own one, so adopt it.
  """

  if type(value) is str and value in _EXACT_ROLES:
    return value
  return "owner"


__all__ = ["require_exact_role", "coerce_stored_role"]
