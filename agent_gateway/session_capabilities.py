from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any


TEAM_WORKSPACE_WRITE_CAPABILITY = "team_workspace_write"
SHARED_COMMUNITY_SUBJECT_CAPABILITY = "shared_community_subject"
ENROLLED_ALERTS_WRITE_CAPABILITY = "enrolled_alerts_write"
ENROLLED_DRIVE_WRITE_CAPABILITY = "enrolled_drive_write"
ENROLLED_ROAM_WRITE_CAPABILITY = "enrolled_roam_write"
ENROLLED_IDEA_WORKBENCH_WRITE_CAPABILITY = "enrolled_idea_workbench_write"
ENROLLED_MCP_WRITE_CAPABILITY_BY_SERVER: Mapping[str, str] = MappingProxyType({
  "alerts": ENROLLED_ALERTS_WRITE_CAPABILITY,
  "drive-mcp": ENROLLED_DRIVE_WRITE_CAPABILITY,
  "roam-research": ENROLLED_ROAM_WRITE_CAPABILITY,
  "idea-workbench-mcp": ENROLLED_IDEA_WORKBENCH_WRITE_CAPABILITY,
})
SUPPORTED_SESSION_CAPABILITIES = frozenset({
  TEAM_WORKSPACE_WRITE_CAPABILITY,
  SHARED_COMMUNITY_SUBJECT_CAPABILITY,
  ENROLLED_ALERTS_WRITE_CAPABILITY,
  ENROLLED_DRIVE_WRITE_CAPABILITY,
  ENROLLED_ROAM_WRITE_CAPABILITY,
  ENROLLED_IDEA_WORKBENCH_WRITE_CAPABILITY,
})
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


def is_shared_community_session(session: Any) -> bool:
  """Return whether ``session`` carries the shared-community-subject capability."""
  if session is None:
    return False
  raw_capabilities = (
    session.get("capabilities")
    if isinstance(session, Mapping)
    else getattr(session, "capabilities", None)
  )
  return SHARED_COMMUNITY_SUBJECT_CAPABILITY in normalize_session_capabilities(
    raw_capabilities
  )


def user_has_enrolled_mcp_write_account(
  session: Any,
  server_name: str,
) -> bool:
  """Return whether ``session`` carries the enrolled-account write capability for ``server_name``."""
  required_capability = ENROLLED_MCP_WRITE_CAPABILITY_BY_SERVER.get(server_name)
  if required_capability is None or session is None:
    return False
  raw_capabilities = (
    session.get("capabilities")
    if isinstance(session, Mapping)
    else getattr(session, "capabilities", None)
  )
  return required_capability in normalize_session_capabilities(raw_capabilities)


__all__ = [
  "ENROLLED_ALERTS_WRITE_CAPABILITY",
  "ENROLLED_DRIVE_WRITE_CAPABILITY",
  "ENROLLED_IDEA_WORKBENCH_WRITE_CAPABILITY",
  "ENROLLED_MCP_WRITE_CAPABILITY_BY_SERVER",
  "ENROLLED_ROAM_WRITE_CAPABILITY",
  "SHARED_COMMUNITY_SUBJECT_CAPABILITY",
  "SUPPORTED_SESSION_CAPABILITIES",
  "TEAM_WORKSPACE_WRITE_CAPABILITY",
  "is_shared_community_session",
  "normalize_session_capabilities",
  "user_has_enrolled_mcp_write_account",
]
