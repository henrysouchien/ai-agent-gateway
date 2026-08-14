from __future__ import annotations

from collections.abc import Iterable
from typing import Any


TEAM_WORKSPACE_WRITE_CAPABILITY = "team_workspace_write"
SUPPORTED_SESSION_CAPABILITIES = frozenset({TEAM_WORKSPACE_WRITE_CAPABILITY})


def normalize_session_capabilities(value: Any) -> frozenset[str]:
  """Normalize authenticated session capabilities and reject unknown authority."""
  if value is None:
    return frozenset()
  if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
    raise ValueError("capabilities must be an array of strings")

  normalized: set[str] = set()
  for raw_capability in value:
    if not isinstance(raw_capability, str):
      raise ValueError("capabilities must contain only strings")
    capability = raw_capability.strip()
    if not capability:
      raise ValueError("capabilities must not contain empty names")
    if capability not in SUPPORTED_SESSION_CAPABILITIES:
      raise ValueError(f"unsupported session capability: {capability}")
    normalized.add(capability)
  return frozenset(normalized)


__all__ = [
  "SUPPORTED_SESSION_CAPABILITIES",
  "TEAM_WORKSPACE_WRITE_CAPABILITY",
  "normalize_session_capabilities",
]
