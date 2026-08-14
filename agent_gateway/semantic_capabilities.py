"""Canonical semantic-capability admission for every agent execution path.

Operations name semantic requirements.  This module is the single server-side
compiler that proves those requirements against an operation-specific route
ceiling and emits exact capability bindings plus an exact ``ToolGrant``.  It
does not discover tools, infer authority from a methodology, or widen a route
ceiling from effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from agent_workflow_contracts import (
  AdmittedInputBinding,
  CapabilityBinding,
  ContractRef,
  LiveToolCapabilityBinding,
  SemanticCapabilityRequirement,
  ToolGrant,
  ToolGrantEntry,
  TypedInputCapabilityBinding,
  sha256_digest,
)


class SemanticCapabilityCompilationError(ValueError):
  """Exact admitted routes cannot satisfy the semantic requirements."""


@dataclass(frozen=True, slots=True)
class SemanticCapabilitySpec:
  name: str
  live_tool_effects: frozenset[str] = frozenset()
  typed_input_contracts: tuple[ContractRef, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticToolRoute:
  """One exact, server-private tool route inside an operation ceiling."""

  tool_id: str
  effect: str
  server_id: str | None = None
  capability: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticCapabilityCompilation:
  capability_bindings: tuple[CapabilityBinding, ...]
  tool_grant: ToolGrant
  tool_ids: frozenset[str]
  mcp_tools_by_server: Mapping[str, frozenset[str]]


class SemanticCapabilityRegistry:
  """Immutable compatibility registry for semantic requirements."""

  def __init__(self, specs: Iterable[SemanticCapabilitySpec]) -> None:
    by_name: dict[str, SemanticCapabilitySpec] = {}
    for spec in specs:
      if not isinstance(spec, SemanticCapabilitySpec):
        raise TypeError("semantic capability specs must be exact values")
      if not spec.name or spec.name != spec.name.strip():
        raise ValueError("semantic capability names must be canonical text")
      if spec.name in by_name:
        raise ValueError(f"duplicate semantic capability: {spec.name}")
      by_name[spec.name] = spec
    self._by_name = by_name

  def require(self, name: str) -> SemanticCapabilitySpec:
    try:
      return self._by_name[name]
    except KeyError as exc:
      raise SemanticCapabilityCompilationError(
        f"semantic capability {name!r} is not registered"
      ) from exc

DEFAULT_SEMANTIC_CAPABILITY_REGISTRY = SemanticCapabilityRegistry((
  SemanticCapabilitySpec(
    name="research-evidence.read/v1",
    live_tool_effects=frozenset({"read"}),
  ),
  SemanticCapabilitySpec(
    name="artifact.propose/v1",
    live_tool_effects=frozenset({"propose"}),
  ),
  SemanticCapabilitySpec(
    name="state.mutate/v1",
    live_tool_effects=frozenset({"write", "external_effect"}),
  ),
  SemanticCapabilitySpec(
    name="research.source/v1",
  ),
))


def _tool_grant(
  *,
  grant_id: str,
  entries: tuple[ToolGrantEntry, ...],
) -> ToolGrant:
  payload = {
    "grant_id": grant_id,
    "tools": [entry.model_dump(mode="json") for entry in entries],
  }
  return ToolGrant(
    grant_id=grant_id,
    tools=entries,
    digest=sha256_digest(payload),
  )


def compile_semantic_capabilities(
  requirements: tuple[SemanticCapabilityRequirement, ...],
  *,
  grant_id: str,
  tool_routes: Iterable[SemanticToolRoute] = (),
  inputs: tuple[AdmittedInputBinding, ...] = (),
  registry: SemanticCapabilityRegistry = DEFAULT_SEMANTIC_CAPABILITY_REGISTRY,
) -> SemanticCapabilityCompilation:
  """Compile exact routes without widening or positional assignment.

  Untagged tools are accepted only when exactly one required capability is
  compatible with that route.  This keeps the convenient single-capability
  operation case while failing closed for ambiguous multi-capability catalogs.
  """

  if not isinstance(registry, SemanticCapabilityRegistry):
    raise TypeError("registry must be a SemanticCapabilityRegistry")
  by_name: dict[str, SemanticCapabilityRequirement] = {}
  specs: dict[str, SemanticCapabilitySpec] = {}
  for requirement in requirements:
    if requirement.name in by_name:
      raise SemanticCapabilityCompilationError(
        f"duplicate semantic capability requirement: {requirement.name}"
      )
    by_name[requirement.name] = requirement
    specs[requirement.name] = registry.require(requirement.name)

  routes = tuple(tool_routes)
  route_by_id: dict[str, SemanticToolRoute] = {}
  assigned: dict[str, list[SemanticToolRoute]] = {
    name: [] for name in by_name
  }
  for route in routes:
    if not isinstance(route, SemanticToolRoute):
      raise TypeError("tool_routes must contain SemanticToolRoute values")
    if route.tool_id in route_by_id:
      raise SemanticCapabilityCompilationError(
        f"duplicate semantic tool route: {route.tool_id}"
      )
    route_by_id[route.tool_id] = route
    compatible = [
      name
      for name, requirement in by_name.items()
      if "live_tool" in requirement.binding_modes
      and route.effect in specs[name].live_tool_effects
      and (route.capability is None or route.capability == name)
    ]
    if route.capability is not None and route.capability not in by_name:
      raise SemanticCapabilityCompilationError(
        f"tool {route.tool_id!r} targets undeclared capability "
        f"{route.capability!r}"
      )
    if len(compatible) > 1:
      raise SemanticCapabilityCompilationError(
        f"tool {route.tool_id!r} ambiguously matches semantic capabilities: "
        + ", ".join(sorted(compatible))
      )
    if len(compatible) == 1:
      assigned[compatible[0]].append(route)
    elif route.capability is not None:
      raise SemanticCapabilityCompilationError(
        f"tool {route.tool_id!r} is incompatible with semantic capability "
        f"{route.capability!r}"
      )

  bindings: list[CapabilityBinding] = []
  entries: dict[str, ToolGrantEntry] = {}
  for name, requirement in by_name.items():
    spec = specs[name]
    operation_contracts = frozenset(requirement.compatible_input_contracts)
    registry_contracts = frozenset(spec.typed_input_contracts)
    accepted_contracts = (
      operation_contracts & registry_contracts
      if registry_contracts
      else operation_contracts
    )
    compatible_inputs = tuple(
      item
      for item in inputs
      if "typed_input" in requirement.binding_modes
      and item.source.actual_contract in accepted_contracts
    )
    if len(compatible_inputs) > 1:
      raise SemanticCapabilityCompilationError(
        f"semantic capability {name!r} has ambiguous typed inputs: "
        + ", ".join(sorted(item.name for item in compatible_inputs))
      )
    selected_routes = tuple(sorted(assigned[name], key=lambda item: item.tool_id))
    if compatible_inputs:
      selected = compatible_inputs[0]
      bindings.append(TypedInputCapabilityBinding(
        capability=name,
        input_name=selected.name,
        input_contract=selected.source.actual_contract,
      ))
      continue
    if selected_routes:
      route_id = f"semantic:{name}"
      tool_ids = tuple(route.tool_id for route in selected_routes)
      bindings.append(LiveToolCapabilityBinding(
        capability=name,
        route_id=route_id,
        tool_ids=tool_ids,
      ))
      for route in selected_routes:
        entries[route.tool_id] = ToolGrantEntry(
          tool_id=route.tool_id,
          route_id=route_id,
          effect=route.effect,
        )
      continue
    if requirement.required:
      raise SemanticCapabilityCompilationError(
        f"required semantic capability {name!r} has no compatible admitted route"
      )

  ordered_entries = tuple(entries[name] for name in sorted(entries))
  mcp_scope: dict[str, set[str]] = {}
  for name in entries:
    server_id = route_by_id[name].server_id
    if server_id is not None:
      mcp_scope.setdefault(server_id, set()).add(name)
  return SemanticCapabilityCompilation(
    capability_bindings=tuple(bindings),
    tool_grant=_tool_grant(grant_id=grant_id, entries=ordered_entries),
    tool_ids=frozenset(entries),
    mcp_tools_by_server={
      server_id: frozenset(tool_ids)
      for server_id, tool_ids in sorted(mcp_scope.items())
    },
  )


__all__ = [
  "DEFAULT_SEMANTIC_CAPABILITY_REGISTRY",
  "SemanticCapabilityCompilation",
  "SemanticCapabilityCompilationError",
  "SemanticCapabilityRegistry",
  "SemanticCapabilitySpec",
  "SemanticToolRoute",
  "compile_semantic_capabilities",
]
