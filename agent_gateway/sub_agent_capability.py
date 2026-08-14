from __future__ import annotations

import re
from types import MappingProxyType

from .capability_binding import CapabilityId


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


class DelegationRoleResolutionError(ValueError):
  """Raised when caller-controlled role text has no declared capability."""


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
  """Resolve an explicit delegation role or refuse it before child spawn."""

  alias = _normalized_alias(role)
  capability = DELEGATION_ROLE_CAPABILITIES.get(alias)
  if capability is None:
    rendered = alias or "<blank>"
    raise DelegationRoleResolutionError(
      f"Unmapped delegation role: {rendered}"
    )
  return capability


def resolve_sub_agent_capability(
  *,
  validated_agent_name: str | None,
  mutation_mode: str | None,
  delegation_role: str | None = None,
) -> CapabilityId:
  """Map server-validated named-agent metadata to one node capability.

  Mutating skill metadata takes precedence over purpose-like aliases on a
  validated named agent. Curated role aliases use the strict role resolver.
  Other *validated, registered skill names* and the permitted unnamed generic
  worker use ``node.implement``. Untrusted role text must go through
  :func:`resolve_delegation_role_capability`, which is fail-closed.
  """

  declared_capability: CapabilityId | None = None
  if delegation_role is not None:
    declared_capability = resolve_delegation_role_capability(delegation_role)

  if _normalized_alias(mutation_mode) in _MUTATING_SKILL_MODES:
    return "node.mutate"

  if declared_capability is not None:
    return declared_capability

  alias = _normalized_alias(validated_agent_name)
  if alias in DELEGATION_ROLE_CAPABILITIES:
    return resolve_delegation_role_capability(alias)
  return "node.implement"


__all__ = [
  "DELEGATION_CAPABILITY_IDS",
  "DELEGATION_ROLE_CAPABILITIES",
  "DelegationRoleResolutionError",
  "is_delegation_profile",
  "resolve_delegation_role_capability",
  "resolve_sub_agent_capability",
]
