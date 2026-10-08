from __future__ import annotations

from typing import Any, Literal


ExactRole = Literal["owner", "invite"]


def require_exact_role(value: Any) -> ExactRole:
  """Validate an authority role.

  Role partitions authority: an invite session must never be read as the
  owner (trades, operator schedules and batches), so a value that is not
  exactly 'owner' or 'invite' is refused rather than defaulted to either.
  """
  if type(value) is str and value == "owner":
    return "owner"
  if type(value) is str and value == "invite":
    return "invite"
  raise ValueError("role must be exactly 'owner' or 'invite'")


__all__ = ["require_exact_role"]
