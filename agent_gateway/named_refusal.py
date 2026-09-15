from __future__ import annotations

from typing import Literal, Mapping


NamedRefusalTransport = Literal[
  "busy",
  "conflict",
  "invalid",
  "not_found",
  "unavailable",
  "internal",
]

_TRANSPORT_STATUS: Mapping[NamedRefusalTransport, int] = {
  "busy": 429,
  "conflict": 409,
  "invalid": 422,
  "not_found": 404,
  "unavailable": 503,
  "internal": 500,
}


class NamedRefusal(RuntimeError):
  """Owner-minted refusal. Consumers observe; they do not classify."""

  def __init__(self, code: str, message: str, *, transport: NamedRefusalTransport) -> None:
    self.code = code
    self.transport = transport
    self.http_status = _TRANSPORT_STATUS[transport]
    super().__init__(message)


__all__ = [
  "NamedRefusal",
  "NamedRefusalTransport",
]
