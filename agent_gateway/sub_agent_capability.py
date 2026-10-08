from __future__ import annotations

import re
from types import MappingProxyType

from model_authority.binding import CapabilityId


DELEGATION_ROLE_CAPABILITIES = MappingProxyType({
  "adversarial-refute": "node.verify",
  "explore": "node.explore",
  "research-digest": "node.explore",
  "research-digests": "node.explore",
  "summarize": "node.explore",
  "summarize-artifact": "node.explore",
  "verify": "node.verify",
  "verify-finding": "node.verify",
})
DELEGATION_CAPABILITY_IDS = frozenset(DELEGATION_ROLE_CAPABILITIES.values())
_MUTATING_SKILL_MODES = frozenset({
  "apply",
  "model-writer",
  "thesis-writer",
})


def _normalized_alias(value: object) -> str:
  text = str(value or "").strip().lower()
  return re.sub(r"[\s_]+", "-", text)


def is_delegation_profile(
  *,
  validated_agent_name: str | None,
  delegation_role: str | None,
) -> bool:
  """Return whether trusted profile metadata selects the curated role lane."""

  return (
    bool(_normalized_alias(delegation_role))
    or _normalized_alias(validated_agent_name)
    in DELEGATION_ROLE_CAPABILITIES
  )


def resolve_delegation_role_capability(role: object) -> CapabilityId:
  """Resolve a declared delegation role to its node capability.

  A role the curated map does not name runs as ``node.implement``, the
  capability every registered skill without a curated role gets.
  """

  return DELEGATION_ROLE_CAPABILITIES.get(_normalized_alias(role), "node.implement")


def resolve_sub_agent_capability(
  *,
  validated_agent_name: str | None,
  mutation_mode: str | None,
  delegation_role: str | None = None,
) -> CapabilityId:
  """Map server-validated named-agent metadata to one node capability.

  Mutating skill metadata takes precedence over a curated delegation role,
  which takes precedence over a curated alias on the validated agent name.
  Other *validated, registered skill names*, roles the curated map does not
  name, and the permitted unnamed generic worker use ``node.implement``.
  """

  if _normalized_alias(mutation_mode) in _MUTATING_SKILL_MODES:
    return "node.mutate"
  for alias in (delegation_role, validated_agent_name):
    capability = DELEGATION_ROLE_CAPABILITIES.get(_normalized_alias(alias))
    if capability is not None:
      return capability
  return "node.implement"


__all__ = [
  "DELEGATION_CAPABILITY_IDS",
  "DELEGATION_ROLE_CAPABILITIES",
  "is_delegation_profile",
  "resolve_delegation_role_capability",
  "resolve_sub_agent_capability",
]
