"""One resolver, one artifact (T3-I11).

Authority for an operation was decided in roughly ten places over three
different universes — a parent's live tool definitions, an operation's declared
ceiling, and the MCP client's routing table — each with its own silent-drop
rule.  This module replaces that with two values and three functions:

``snapshot_platform_catalog``
  Freeze what the platform can route *right now* into a
  :class:`~agent_workflow_contracts.PlatformToolCatalog`.  The declarative
  columns (``effect``, ``idempotent``, ``success_signal``, ``source_identity``)
  are seeded from the gateway-local
  :mod:`agent_gateway.tool_dispatch_declarations` table, which is this
  function's construction-only input (D-B1-2) and has no other reader.

``resolve_operation_authority``
  Answer, for one operation declaration against one catalog snapshot, either
  the exact :class:`~agent_workflow_contracts.ResolvedAuthority` or a visible
  :class:`~agent_workflow_contracts.OperationUnavailable`.  It ships in
  **effect-compatible mode** (D-B5-1): the satisfaction verdict is produced by
  calling :func:`agent_gateway.semantic_capabilities.compile_semantic_capabilities`
  — this function *wraps* today's semantics rather than restating them, so the
  verdict is provably identical.

``derive_dispatcher_allowlist``
  Project the frozen authority into the per-server tool scope the dispatcher
  enforces.  Nothing re-discovers the surface at dispatch time.

The module lives in ``agent_gateway`` deliberately: ``runtime_provider`` (the
workflow path) and ``sub_agent`` (the ordinary delegation path) both need it,
and the gateway never statically imports ``agent.*``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from agent_workflow_contracts import (
  AdmittedInputBinding,
  CatalogToolEntry,
  ExecutionIdentity,
  OperationUnavailable,
  PlatformToolCatalog,
  ResolvedAuthority,
  SemanticCapabilityRequirement,
  UnsatisfiedCapability,
)

from .mcp_activation import LiveToolSurface
from .semantic_capability_routing import capability_for_tool
from .semantic_capabilities import (
  DEFAULT_SEMANTIC_CAPABILITY_REGISTRY,
  SemanticCapabilityCompilationError,
  SemanticCapabilityRegistry,
  SemanticToolRoute,
  compile_semantic_capabilities,
)
from .tool_dispatch_declarations import (
  ToolDispatchDecl,
  canonical_dispatch_tool_name,
  tool_dispatch_declarations,
)


class CapabilityResolutionInputError(ValueError):
  """The resolver was handed something that is not an exact declaration."""


# ``(policy_tool_id, server_id, is_local) -> effect | None``
ToolEffectResolver = Any

_WORKSPACE_EFFECT_CEILINGS: Mapping[str, frozenset[str]] = MappingProxyType({
  "read_only": frozenset({"read"}),
  "workspace_write": frozenset({"read", "propose", "write"}),
  "model_write": frozenset({"read", "propose", "write", "external_effect"}),
})


def _json_descriptor(value: Any) -> Any:
  """Normalize a declarative descriptor into exact JSON.

  The declaration table stores ``MappingProxyType`` rows with tuple literals;
  the wire type accepts only JSON values.  No semantics change — only the
  container types.
  """

  if isinstance(value, Mapping):
    return {str(key): _json_descriptor(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_json_descriptor(item) for item in value]
  return value


def _default_effect_resolver(
  tool_id: str,
  server_id: str | None,
  is_local: bool,
) -> str | None:
  """Delegate to the one server-owned effect table.

  Resolved through the module object so a test that pins
  ``sub_agent_scope_receipt._server_owned_effect`` still pins this path.
  """

  from . import sub_agent_scope_receipt

  return sub_agent_scope_receipt._server_owned_effect(tool_id, server_id, is_local)


def _catalog_tool_entry(
  *,
  tool_id: str,
  canonical_name: str,
  effect: str | None,
  server_id: str | None,
  declaration: ToolDispatchDecl | None,
) -> CatalogToolEntry:
  capability = capability_for_tool(
    canonical_name=canonical_name,
    server_id=server_id,
    effect=effect,
  )
  success_signal = (
    _json_descriptor(declaration.success_signal)
    if declaration is not None and declaration.success_signal is not None
    else None
  )
  source_identity = (
    _json_descriptor(declaration.source_identity)
    if declaration is not None and declaration.source_identity is not None
    else None
  )
  return CatalogToolEntry(
    tool_id=tool_id,
    canonical_name=canonical_name,
    effect=effect,
    server_id=server_id,
    capability=capability,
    idempotent=declaration.idempotent if declaration is not None else None,
    success_signal=success_signal,
    source_identity=source_identity,
  )


def snapshot_platform_catalog(
  *,
  tool_ids: Iterable[str] | None = None,
  local_tool_handlers: Mapping[str, Any] | None = None,
  mcp_client: Any = None,
  effect_resolver: ToolEffectResolver | None = None,
  declarations: Mapping[str, ToolDispatchDecl] | None = None,
) -> PlatformToolCatalog:
  """Freeze the describable platform tools into one exact snapshot.

  Two shapes, one function:

  * **Declarative** (``tool_ids is None``): one entry per declaration row,
    keyed by canonical name with no server route.  This is what the dispatch
    boundary reads.
  * **Routed** (``tool_ids`` given): each candidate id is resolved against the
    live routing facts exactly as the operation-admission receipt does — a
    tool is either local **or** MCP-routed, never both and never neither, and
    an MCP route needs a named server.  An id whose effect cannot be resolved
    is still *described* (``effect=None``) but can never become authority.
  """

  table = declarations if declarations is not None else tool_dispatch_declarations()
  if tool_ids is None:
    entries = [
      _catalog_tool_entry(
        tool_id=name,
        canonical_name=name,
        effect=row.effect,
        server_id=None,
        declaration=row,
      )
      for name, row in table.items()
    ]
    return PlatformToolCatalog(
      tools=tuple(sorted(entries, key=lambda item: item.tool_id))
    )

  handlers = local_tool_handlers if local_tool_handlers is not None else {}
  resolver = effect_resolver if effect_resolver is not None else _default_effect_resolver
  is_mcp_tool = getattr(mcp_client, "is_mcp_tool", None)
  get_server = getattr(mcp_client, "get_server_for_tool", None)
  get_original = getattr(mcp_client, "get_original_tool_name", None)
  entries: dict[str, CatalogToolEntry] = {}
  for raw in tool_ids:
    tool_id = str(raw or "").strip()
    if not tool_id or tool_id in entries:
      continue
    is_local = tool_id in handlers
    is_mcp = bool(callable(is_mcp_tool) and is_mcp_tool(tool_id))
    if is_local == is_mcp:
      # Ambiguous or unroutable ids cannot become authority; the platform
      # does not describe what it cannot route.
      continue
    server_id = (
      str(get_server(tool_id) or "").strip() or None
      if is_mcp and callable(get_server)
      else None
    )
    if is_mcp and server_id is None:
      continue
    policy_tool_id = (
      str(get_original(tool_id) or tool_id).strip()
      if is_mcp and callable(get_original)
      else tool_id
    )
    canonical = canonical_dispatch_tool_name(policy_tool_id)
    entries[tool_id] = _catalog_tool_entry(
      tool_id=tool_id,
      canonical_name=canonical,
      effect=resolver(policy_tool_id, server_id, is_local),
      server_id=server_id,
      declaration=table.get(canonical),
    )
  return PlatformToolCatalog(
    tools=tuple(entries[name] for name in sorted(entries))
  )


_CACHED_DECLARATIVE_CATALOG: tuple[object, PlatformToolCatalog, Mapping[str, CatalogToolEntry]] | None = None


def declarative_platform_catalog() -> PlatformToolCatalog:
  """Return the process-wide declarative catalog, built on first use.

  Keyed on the declaration table's object identity so a test that swaps the
  table (the derived-effect pin) is honoured without a second reset call.
  """

  return _declarative_catalog_index()[0]


def _declarative_catalog_index() -> tuple[
  PlatformToolCatalog,
  Mapping[str, CatalogToolEntry],
]:
  global _CACHED_DECLARATIVE_CATALOG
  table = tool_dispatch_declarations()
  cached = _CACHED_DECLARATIVE_CATALOG
  if cached is not None and cached[0] is table:
    return cached[1], cached[2]
  catalog = snapshot_platform_catalog(declarations=table)
  index = MappingProxyType({entry.tool_id: entry for entry in catalog.tools})
  _CACHED_DECLARATIVE_CATALOG = (table, catalog, index)
  return catalog, index


def reset_platform_catalog_cache() -> None:
  """Drop the cached declarative catalog (tests and policy-module reloads)."""

  global _CACHED_DECLARATIVE_CATALOG
  _CACHED_DECLARATIVE_CATALOG = None


def lookup_catalog_entry(tool_name: str) -> CatalogToolEntry | None:
  """Return the catalog entry for ``tool_name``, or ``None`` when undescribed."""

  _catalog, index = _declarative_catalog_index()
  return index.get(canonical_dispatch_tool_name(tool_name))


@dataclass(frozen=True, slots=True)
class OperationDeclaration:
  """What one operation declares it needs, free of live facts.

  ``tool_ceiling`` is the operation's exact declared tool ceiling.  When
  ``mcp_tool_ceiling`` is not ``None`` an MCP-routed catalog entry is admitted
  only when the declaration named it as an MCP route — the workflow path's
  rule, preserved exactly.  ``None`` means the caller imposes no extra rule
  (the ordinary-delegation path).
  """

  operation_name: str
  grant_id: str
  workspace_scope: str
  required_capabilities: tuple[SemanticCapabilityRequirement, ...] = ()
  tool_ceiling: frozenset[str] = frozenset()
  mcp_tool_ceiling: frozenset[str] | None = None
  inputs: tuple[AdmittedInputBinding, ...] = ()


def admitted_catalog_routes(
  declaration: OperationDeclaration,
  *,
  catalog: PlatformToolCatalog,
  exclusions: Iterable[str] = (),
) -> tuple[CatalogToolEntry, ...]:
  """The exact catalog entries one declaration may draw authority from."""

  ceiling = _WORKSPACE_EFFECT_CEILINGS.get(declaration.workspace_scope)
  if ceiling is None:
    raise CapabilityResolutionInputError(
      f"unknown operation workspace scope: {declaration.workspace_scope}"
    )
  excluded = frozenset(exclusions)
  mcp_ceiling = declaration.mcp_tool_ceiling
  admitted: list[CatalogToolEntry] = []
  for entry in catalog.tools:
    if entry.tool_id not in declaration.tool_ceiling:
      continue
    if entry.tool_id in excluded:
      continue
    if entry.effect is None or entry.effect not in ceiling:
      continue
    if (
      entry.server_id is not None
      and mcp_ceiling is not None
      and entry.tool_id not in mcp_ceiling
    ):
      continue
    admitted.append(entry)
  return tuple(sorted(admitted, key=lambda item: item.tool_id))


def _unsatisfied_capabilities(
  declaration: OperationDeclaration,
  *,
  routes: Sequence[SemanticToolRoute],
  registry: SemanticCapabilityRegistry,
) -> tuple[UnsatisfiedCapability, ...]:
  """Name which declared capabilities the admitted routes cannot serve.

  This is a *diagnostic*, not the verdict — the verdict is always the
  compiler's.  It uses the compiler's own compatibility predicate so the two
  can never disagree about what "compatible" means.
  """

  unsatisfied: list[UnsatisfiedCapability] = []
  for requirement in declaration.required_capabilities:
    try:
      spec = registry.require(requirement.name)
    except SemanticCapabilityCompilationError as exc:
      unsatisfied.append(UnsatisfiedCapability(
        capability=requirement.name,
        required=requirement.required,
        reason="unregistered_capability",
        detail=str(exc)[:2_048],
      ))
      continue
    if "live_tool" not in requirement.binding_modes:
      continue
    compatible = [
      route
      for route in routes
      if route.effect in spec.live_tool_effects
      and (route.capability is None or route.capability == requirement.name)
    ]
    if compatible:
      continue
    unsatisfied.append(UnsatisfiedCapability(
      capability=requirement.name,
      required=requirement.required,
      reason="no_compatible_route",
      detail=(
        f"no admitted route carries an effect compatible with "
        f"{requirement.name!r}"
      ),
    ))
  return tuple(unsatisfied)


def resolve_operation_authority(
  declaration: OperationDeclaration,
  *,
  catalog: PlatformToolCatalog,
  registry: SemanticCapabilityRegistry = DEFAULT_SEMANTIC_CAPABILITY_REGISTRY,
  identity: ExecutionIdentity | None = None,
  exclusions: Iterable[str] = (),
) -> ResolvedAuthority | OperationUnavailable:
  """Resolve one operation's exact authority, or say why it is unavailable.

  Effect-compatible mode (D-B5-1): the satisfaction verdict is produced by
  ``compile_semantic_capabilities``.  This function selects the admitted
  routes and shapes the answer; it never decides satisfaction itself.
  """

  if not isinstance(declaration, OperationDeclaration):
    raise CapabilityResolutionInputError(
      "resolve_operation_authority requires an OperationDeclaration"
    )
  if not isinstance(catalog, PlatformToolCatalog):
    raise CapabilityResolutionInputError(
      "resolve_operation_authority requires a PlatformToolCatalog"
    )
  try:
    admitted = admitted_catalog_routes(
      declaration,
      catalog=catalog,
      exclusions=exclusions,
    )
  except CapabilityResolutionInputError as exc:
    return OperationUnavailable(
      operation_name=declaration.operation_name,
      code="invalid_metadata",
      detail=str(exc)[:2_048],
    )
  declared = frozenset(
    requirement.name for requirement in declaration.required_capabilities
  )
  routes = tuple(
    SemanticToolRoute(
      tool_id=entry.tool_id,
      effect=str(entry.effect),
      server_id=entry.server_id,
      capability=entry.capability,
    )
    for entry in admitted
    # A route serving a capability the operation never declared is not this
    # operation's authority.  Before B-8 the effect ceiling admitted such a
    # route and the compiler dropped it unassigned; the drop now happens where
    # the reason is legible, and the granted set is identical either way.
    if entry.capability is None or entry.capability in declared
  )
  try:
    compiled = compile_semantic_capabilities(
      declaration.required_capabilities,
      grant_id=declaration.grant_id,
      tool_routes=routes,
      inputs=declaration.inputs,
      registry=registry,
    )
  except SemanticCapabilityCompilationError as exc:
    return OperationUnavailable(
      operation_name=declaration.operation_name,
      code="missing_route",
      detail=str(exc)[:2_048],
      unsatisfied=_unsatisfied_capabilities(
        declaration,
        routes=routes,
        registry=registry,
      ),
    )
  granted = compiled.tool_ids
  return ResolvedAuthority(
    operation_name=declaration.operation_name,
    grant=compiled.tool_grant,
    bindings=compiled.capability_bindings,
    routes=tuple(entry for entry in admitted if entry.tool_id in granted),
    identity=identity,
  )


def granted_tool_ids(authority: ResolvedAuthority) -> frozenset[str]:
  """The exact tool ids one resolved authority grants."""

  return frozenset(entry.tool_id for entry in authority.grant.tools)


def derive_dispatcher_allowlist(
  authority: ResolvedAuthority | LiveToolSurface,
) -> dict[str, frozenset[str]]:
  """Project one settled decision into the dispatcher's per-server scope.

  Both paths that grant MCP tools end here.  The delegation path passes the
  frozen :class:`ResolvedAuthority`; the interactive path passes the
  :class:`~agent_gateway.mcp_activation.LiveToolSurface` derived from the
  session's activation fold.  One function, so a tool the dispatcher admits is
  a tool something actually granted — never a second, separately maintained
  allowlist (T3-I12).
  """

  if isinstance(authority, LiveToolSurface):
    return {
      server_id: frozenset(tool_ids)
      for server_id, tool_ids in sorted(
        authority.allowed_mcp_tools_by_server.items()
      )
    }
  granted = granted_tool_ids(authority)
  scope: dict[str, set[str]] = {}
  for route in authority.routes:
    if route.tool_id not in granted or route.server_id is None:
      continue
    scope.setdefault(route.server_id, set()).add(route.tool_id)
  return {
    server_id: frozenset(tool_ids)
    for server_id, tool_ids in sorted(scope.items())
  }


__all__ = [
  "CapabilityResolutionInputError",
  "OperationDeclaration",
  "admitted_catalog_routes",
  "canonical_dispatch_tool_name",
  "declarative_platform_catalog",
  "derive_dispatcher_allowlist",
  "granted_tool_ids",
  "lookup_catalog_entry",
  "reset_platform_catalog_cache",
  "resolve_operation_authority",
  "snapshot_platform_catalog",
]
