"""The active-skill tool gate, resolved rather than derived in the handler.

Before B-6 the ``invoke_skill`` handler computed its own allow set — a private
intersection of the skill's declared tool ids with a gateway policy row — and
its own deny set, then shipped both on the tool result.  That was a second,
unreviewed authority derivation living next to the model-facing handler.

Here it is one resolver call.  ``granted`` is ``admitted_catalog_routes(...)``
over the policy catalog the gateway already reviewed for that skill: the
**positive** grant, cut by the one resolver from an exact declaration.  The
residual ``denied`` set carries only what a positive grant cannot express — the
mutation-mode ceiling over tools the skill never declared, and the delegation
surfaces a skill file closes explicitly.  The grant wins over the ceiling, as
it always did: a tool gateway policy granted this skill is not withdrawn by a
mode ceiling that never knew about it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agent_workflow_contracts import CatalogToolEntry, PlatformToolCatalog

from .capability_resolution import (
  CapabilityResolutionInputError,
  OperationDeclaration,
  admitted_catalog_routes,
)
from .investment_capability_claim import INVESTMENT_CAPABILITY_SKILL_GRANTS
from .sub_agent_scope_receipt import _normalized_effect


# Least privilege first: a policy row that grants any read authority is
# describable as a read route, which every workspace ceiling admits.
_EFFECT_PRIVILEGE_ORDER = ("read", "propose", "write", "external_effect")


@dataclass(frozen=True, slots=True)
class ActiveSkillToolAuthority:
  """What one loaded skill may use, and what its mode ceiling withdraws."""

  granted: frozenset[str] = frozenset()
  denied: frozenset[str] = frozenset()


def _policy_catalog_effect(effect_classes: Iterable[str]) -> str | None:
  """The least-privilege effect a reviewed policy row describes."""

  normalized = {
    effect
    for raw in effect_classes
    if (effect := _normalized_effect(raw)) is not None
  }
  for candidate in _EFFECT_PRIVILEGE_ORDER:
    if candidate in normalized:
      return candidate
  return None


def skill_policy_catalog(skill_name: str) -> PlatformToolCatalog:
  """Snapshot the gateway policy row for one skill as a tool catalog.

  ``INVESTMENT_CAPABILITY_SKILL_GRANTS`` is gateway policy, never model input:
  it is exactly a catalog of what the platform will route for that skill.
  """

  grant = INVESTMENT_CAPABILITY_SKILL_GRANTS.get(skill_name)
  if grant is None:
    return PlatformToolCatalog()
  effect = _policy_catalog_effect(grant.effect_classes)
  if effect is None:
    return PlatformToolCatalog()
  return PlatformToolCatalog(tools=tuple(
    CatalogToolEntry(tool_id=name, canonical_name=name, effect=effect)
    for name in sorted(set(grant.allowed_tool_names))
  ))


def resolve_active_skill_authority(
  skill_name: str,
  *,
  declared_tool_ids: Iterable[str],
  workspace_scope: str = "model_write",
  mode_denied_tools: Iterable[str] = (),
  extra_denied_tools: Iterable[str] = (),
) -> ActiveSkillToolAuthority:
  """Resolve the exact tool authority one loaded skill runs under."""

  declaration = OperationDeclaration(
    operation_name=f"skill:{skill_name}" if skill_name else "skill:<unnamed>",
    grant_id=f"active-skill-grant:{skill_name}",
    workspace_scope=workspace_scope,
    tool_ceiling=frozenset(declared_tool_ids),
  )
  try:
    admitted = admitted_catalog_routes(
      declaration,
      catalog=skill_policy_catalog(skill_name),
    )
  except CapabilityResolutionInputError:
    admitted = ()
  granted = frozenset(entry.tool_id for entry in admitted)
  denied = (frozenset(mode_denied_tools) - granted) | frozenset(
    extra_denied_tools
  )
  return ActiveSkillToolAuthority(granted=granted, denied=denied)


__all__ = [
  "ActiveSkillToolAuthority",
  "resolve_active_skill_authority",
  "skill_policy_catalog",
]
