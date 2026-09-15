"""Resolve active inline-skill tool authority at the gateway boundary.

The compiled skill definition owns inline mutation-mode exceptions.  The
``invoke_skill`` handler projects those declarations onto the live runtime and
passes the exact routes here.  This boundary only combines that resolved fact
with the independent signed investment-capability grant policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .capability_resolution import CapabilityResolutionInputError
from .investment_capability_claim import (
  INVESTMENT_CAPABILITY_CLAIM_SERVER,
  INVESTMENT_CAPABILITY_FACADE_TOOLS,
  INVESTMENT_CAPABILITY_SKILL_GRANTS,
)


@dataclass(frozen=True, slots=True)
class ActiveSkillToolAuthority:
  """What one loaded skill may use, and what its mode ceiling withdraws."""

  granted: frozenset[str] = frozenset()
  denied: frozenset[str] = frozenset()


def _validated_live_tool_routes(
  value: Mapping[str, str],
  *,
  field: str,
) -> Mapping[str, str]:
  if not isinstance(value, Mapping):
    raise CapabilityResolutionInputError(f"{field} must be a mapping")
  routes: dict[str, str] = {}
  exposed_ids: set[str] = set()
  for canonical_id, exposed_name in value.items():
    if (
      type(canonical_id) is not str
      or type(exposed_name) is not str
      or not canonical_id
      or not exposed_name
    ):
      raise CapabilityResolutionInputError(
        f"{field} ids must be non-empty exact strings; got "
        f"{canonical_id!r} -> {exposed_name!r}"
      )
    if exposed_name in exposed_ids:
      raise CapabilityResolutionInputError(
        f"{field} cannot expose {exposed_name!r} from multiple origins"
      )
    routes[canonical_id] = exposed_name
    exposed_ids.add(exposed_name)
  return MappingProxyType(routes)


def _investment_capability_routes(
  skill_name: str,
  *,
  declared_live_routes: Mapping[str, str],
) -> frozenset[str]:
  """Project the signed-claim policy through exact declared live routes."""

  grant = INVESTMENT_CAPABILITY_SKILL_GRANTS.get(skill_name)
  if grant is None:
    return frozenset()
  return frozenset(
    tool_name
    for tool_name in grant.allowed_tool_names
    if declared_live_routes.get(
      f"mcp__{INVESTMENT_CAPABILITY_CLAIM_SERVER}__{tool_name}"
    ) == tool_name
  )


def resolve_active_skill_authority(
  skill_name: str,
  *,
  declared_live_tool_routes: Mapping[str, str],
  inline_mode_exception_routes: Mapping[str, str] = MappingProxyType({}),
  mode_denied_tools: Iterable[str] = (),
  extra_denied_tools: Iterable[str] = (),
) -> ActiveSkillToolAuthority:
  """Combine declared inline exceptions with signed investment grants."""

  declared_routes = _validated_live_tool_routes(
    declared_live_tool_routes,
    field="declared_live_tool_routes",
  )
  inline_routes = _validated_live_tool_routes(
    inline_mode_exception_routes,
    field="inline_mode_exception_routes",
  )
  investment_prefix = f"mcp__{INVESTMENT_CAPABILITY_CLAIM_SERVER}__"
  granted = {
    exposed_name
    for canonical_id, exposed_name in inline_routes.items()
    if declared_routes.get(canonical_id) == exposed_name
    and not canonical_id.startswith(investment_prefix)
    and exposed_name not in INVESTMENT_CAPABILITY_FACADE_TOOLS
  }
  granted.update(
    _investment_capability_routes(
      skill_name,
      declared_live_routes=declared_routes,
    )
  )
  resolved_grants = frozenset(granted)
  denied = (frozenset(mode_denied_tools) - resolved_grants) | frozenset(
    extra_denied_tools
  )
  return ActiveSkillToolAuthority(granted=resolved_grants, denied=denied)


__all__ = [
  "ActiveSkillToolAuthority",
  "resolve_active_skill_authority",
]
