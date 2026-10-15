from __future__ import annotations

from dataclasses import dataclass
import re


RUN_IDENTITY_MAX_LENGTH = 128
_RUN_IDENTITY_PATTERN = re.compile(
  rf"^[A-Za-z0-9][A-Za-z0-9._:-]{{0,{RUN_IDENTITY_MAX_LENGTH - 1}}}$"
)


class RunIdentityCarrierError(ValueError):
  """A server-owned run identity cannot be routed without ambiguity."""

  def __init__(self, code: str, message: str | None = None) -> None:
    if message is None:
      message = code
      code = "run_identity_invalid"
    super().__init__(message)
    self.code = code


def validate_run_identity(value: object) -> str:
  """Return one bounded, path-safe identity or fail closed."""

  if type(value) is not str or _RUN_IDENTITY_PATTERN.fullmatch(value) is None:
    raise RunIdentityCarrierError(
      "run_identity_invalid",
      "run identity must be a canonical ASCII token of at most 128 characters",
    )
  return value


@dataclass(frozen=True, slots=True)
class RunIdentityCarrier:
  """Validated server-owned identity shared by runtime transport carriers."""

  run_id: str

  def __post_init__(self) -> None:
    validate_run_identity(self.run_id)

  @classmethod
  def from_optional(cls, value: str | None) -> "RunIdentityCarrier | None":
    if value is None:
      return None
    return cls(value)


__all__ = [
  "RUN_IDENTITY_MAX_LENGTH",
  "RunIdentityCarrier",
  "RunIdentityCarrierError",
  "validate_run_identity",
]
