"""The platform catalog snapshot and the single capability resolver (B-5).

The resolver ships in effect-compatible mode: every satisfaction verdict is
``compile_semantic_capabilities``'s.  These tests pin that the wrap is exact —
the resolver's grant, bindings and dispatcher allowlist are byte-for-byte the
legacy answer over the same routes — plus the catalog snapshot's routing rules
and the visible Left.
"""

from __future__ import annotations

import pytest

from agent_gateway.capability_resolution import (
  CapabilityResolutionInputError,
  OperationDeclaration,
  admitted_catalog_routes,
  declarative_platform_catalog,
  derive_dispatcher_allowlist,
  granted_tool_ids,
  lookup_catalog_entry,
  reset_platform_catalog_cache,
  resolve_operation_authority,
  snapshot_platform_catalog,
)
from agent_gateway.semantic_capabilities import (
  DEFAULT_SEMANTIC_CAPABILITY_REGISTRY,
  SemanticToolRoute,
  compile_semantic_capabilities,
)
from agent_gateway.semantic_capability_routing import capability_for_tool
from agent_gateway.sub_agent_scope_receipt import (
  admit_operation_tools,
  semantic_tool_routes,
)
from agent_gateway.skills import compile_agent_operation, generic_explore_profile
from agent_gateway.tool_dispatch_declarations import (
  build_tool_dispatch_declarations,
  tool_dispatch_declarations,
)
from agent_workflow_contracts import (
  CatalogToolEntry,
  ExecutionIdentity,
  OperationUnavailable,
  PlatformToolCatalog,
  ResolvedAuthority,
  SemanticCapabilityRequirement,
)


_MCP_ROUTES = {
  "filings_search": "research-corpus-mcp",
  "transcripts_search": "research-corpus-mcp",
  "compare_peers": "market-data-mcp",
}


class _Mcp:
  def __init__(self, routes: dict[str, str] | None = None) -> None:
    self.routes = dict(_MCP_ROUTES if routes is None else routes)

  def is_mcp_tool(self, name: str) -> bool:
    return name in self.routes

  def get_server_for_tool(self, name: str) -> str | None:
    return self.routes.get(name)

  def get_original_tool_name(self, name: str) -> str:
    return name


def _effect(tool_id: str, _server: str | None, _local: bool) -> str | None:
  if tool_id in {"web_search", "web_fetch", "filings_search", "transcripts_search"}:
    return "read"
  if tool_id == "compare_peers":
    return "propose"
  return None


def _requirement(
  name: str = "web.read/v1",
  *,
  required: bool = True,
) -> SemanticCapabilityRequirement:
  return SemanticCapabilityRequirement(
    name=name,
    required=required,
    binding_modes=("live_tool",),
  )


def _evidence_requirements(
  *names: str,
) -> tuple[SemanticCapabilityRequirement, ...]:
  """The granular read domains this fixture's routes actually serve (B-8).

  Routes carry a capability now, so a declaration must name the domain it
  wants: the single coarse row every read tool satisfied is gone.
  """

  return tuple(_requirement(name) for name in sorted(names))


def _routed_catalog(
  *,
  tool_ids: tuple[str, ...] = (
    "web_search",
    "web_fetch",
    "filings_search",
    "transcripts_search",
    "compare_peers",
    "memory_read",
  ),
  local: dict[str, object] | None = None,
  mcp: _Mcp | None = None,
) -> PlatformToolCatalog:
  return snapshot_platform_catalog(
    tool_ids=tool_ids,
    local_tool_handlers=(
      {"web_search": object(), "web_fetch": object(), "memory_read": object()}
      if local is None
      else local
    ),
    mcp_client=mcp if mcp is not None else _Mcp(),
    effect_resolver=_effect,
  )


# --- the snapshot ----------------------------------------------------------


def test_declarative_snapshot_is_seeded_from_the_dispatch_declaration_table() -> None:
  table = tool_dispatch_declarations()
  catalog = snapshot_platform_catalog(declarations=table)

  assert {entry.tool_id for entry in catalog.tools} == set(table)
  for entry in catalog.tools:
    row = table[entry.tool_id]
    assert entry.canonical_name == entry.tool_id
    assert entry.effect == row.effect
    assert entry.idempotent == row.idempotent
    assert (entry.success_signal is None) == (row.success_signal is None)
    assert (entry.source_identity is None) == (row.source_identity is None)
    # A seeded snapshot names no server: routes are a live fact, not a
    # declaration.
    assert entry.server_id is None
    # `capability` is SINGULAR (D-B5-2) and B-8 populates it from the one
    # derivation the codegen also reads.
    assert entry.capability == capability_for_tool(
      canonical_name=entry.canonical_name,
      server_id=entry.server_id,
      effect=entry.effect,
    )
  assert catalog.entry("filings_search").capability == "filings.read/v1"
  assert catalog.entry("transcripts_search").capability == "transcripts.read/v1"
  assert catalog.entry("web_fetch").capability == "web.read/v1"


def test_declarative_snapshot_normalizes_descriptors_into_exact_json() -> None:
  catalog = snapshot_platform_catalog()
  entry = catalog.entry("filings_search")

  assert entry is not None
  assert entry.success_signal == {
    "kind": "status_equals",
    "field": "status",
    "values": ["success"],
  }
  assert entry.source_identity == {
    "kind": "search_hits",
    "container": "hits",
    "default_source_kind": "filing",
  }


def test_declarative_catalog_lookup_canonicalizes_namespaced_names() -> None:
  assert (
    lookup_catalog_entry("mcp__research-corpus-mcp__filings_search")
    == lookup_catalog_entry("filings_search")
  )
  assert lookup_catalog_entry("not_a_declared_tool") is None


def test_declarative_catalog_cache_follows_the_declaration_table_identity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  reset_platform_catalog_cache()
  baseline = declarative_platform_catalog()
  assert declarative_platform_catalog() is baseline

  from agent_gateway import tool_dispatch_declarations as declarations

  monkeypatch.setattr(
    declarations,
    "_CACHED_DECLARATIONS",
    declarations.build_tool_dispatch_declarations(
      effect_resolver=lambda _tool_name: "propose"
    ),
  )
  swapped = declarative_platform_catalog()

  assert swapped is not baseline
  assert all(entry.effect == "propose" for entry in swapped.tools)
  reset_platform_catalog_cache()


def test_routed_snapshot_describes_local_and_mcp_routes_exactly() -> None:
  catalog = _routed_catalog()
  by_id = {entry.tool_id: entry for entry in catalog.tools}

  assert by_id["web_search"].server_id is None
  assert by_id["filings_search"].server_id == "research-corpus-mcp"
  assert by_id["compare_peers"].server_id == "market-data-mcp"
  assert by_id["filings_search"].effect == "read"
  assert by_id["compare_peers"].effect == "propose"
  # Seeded declarative columns survive the routed snapshot.
  assert by_id["filings_search"].source_identity is not None


def test_routed_snapshot_drops_ambiguous_and_unroutable_ids() -> None:
  catalog = snapshot_platform_catalog(
    tool_ids=("filings_search", "orphan_tool", "web_search"),
    # `filings_search` is BOTH a local handler and an MCP route here.
    local_tool_handlers={"filings_search": object(), "web_search": object()},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
  )

  assert [entry.tool_id for entry in catalog.tools] == ["web_search"]


def test_routed_snapshot_drops_mcp_routes_without_a_named_server() -> None:
  catalog = snapshot_platform_catalog(
    tool_ids=("filings_search",),
    local_tool_handlers={},
    mcp_client=_Mcp({"filings_search": ""}),
    effect_resolver=_effect,
  )

  assert catalog.tools == ()


def test_routed_snapshot_describes_effectless_tools_but_never_grants_them() -> None:
  catalog = _routed_catalog()
  by_id = {entry.tool_id: entry for entry in catalog.tools}

  assert by_id["memory_read"].effect is None
  declaration = OperationDeclaration(
    operation_name="explore",
    grant_id="grant:effectless",
    workspace_scope="read_only",
    required_capabilities=(_requirement(),),
    tool_ceiling=frozenset({"memory_read", "web_search"}),
  )

  admitted = admitted_catalog_routes(declaration, catalog=catalog)

  assert [entry.tool_id for entry in admitted] == ["web_search"]


def test_platform_catalog_rejects_unsorted_or_duplicate_entries() -> None:
  entry = CatalogToolEntry(tool_id="b", canonical_name="b")
  other = CatalogToolEntry(tool_id="a", canonical_name="a")

  with pytest.raises(ValueError, match="sorted and unique"):
    PlatformToolCatalog(tools=(entry, other))
  with pytest.raises(ValueError, match="sorted and unique"):
    PlatformToolCatalog(tools=(other, other))


# --- the resolver ----------------------------------------------------------


def test_resolver_verdict_is_exactly_compile_semantic_capabilities() -> None:
  """The effect-compatible wrap (D-B5-1): identical, by construction."""

  catalog = _routed_catalog()
  declaration = OperationDeclaration(
    operation_name="explore",
    grant_id="grant:wrap",
    workspace_scope="read_only",
    required_capabilities=_evidence_requirements(
      "web.read/v1",
      "filings.read/v1",
      "transcripts.read/v1",
    ),
    tool_ceiling=frozenset({
      "web_search",
      "web_fetch",
      "filings_search",
      "transcripts_search",
      "compare_peers",
      "memory_read",
    }),
  )

  authority = resolve_operation_authority(declaration, catalog=catalog)
  assert isinstance(authority, ResolvedAuthority)

  admitted = admitted_catalog_routes(declaration, catalog=catalog)
  legacy = compile_semantic_capabilities(
    declaration.required_capabilities,
    grant_id=declaration.grant_id,
    tool_routes=tuple(
      SemanticToolRoute(
        tool_id=entry.tool_id,
        effect=str(entry.effect),
        server_id=entry.server_id,
        capability=entry.capability,
      )
      for entry in admitted
    ),
    registry=DEFAULT_SEMANTIC_CAPABILITY_REGISTRY,
  )

  assert authority.grant == legacy.tool_grant
  assert authority.bindings == legacy.capability_bindings
  assert granted_tool_ids(authority) == legacy.tool_ids
  assert derive_dispatcher_allowlist(authority) == dict(legacy.mcp_tools_by_server)
  # `compare_peers` is a `propose` route: a read-only workspace never grants it.
  assert "compare_peers" not in granted_tool_ids(authority)


def test_admit_operation_tools_is_now_the_resolver_itself() -> None:
  """B-6: the ordinary-delegation admission IS the resolver call.

  WP6a pinned this as *parity* between two implementations. WP7 removed the
  second implementation: ``admit_operation_tools`` is a thin adapter that
  builds the declaration and the catalog snapshot, so the two sides of this
  assertion are now the same value produced twice.
  """

  operation = compile_agent_operation(
    generic_explore_profile(),
    execution_class="node.explore",
  )
  local = {"web_fetch": object(), "web_search": object()}
  mcp = _Mcp()
  ceiling = ("web_fetch", "web_search", "filings_search", "compare_peers")

  legacy = admit_operation_tools(
    operation,
    grant_id="grant:parity",
    operation_tool_ids=ceiling,
    definitions=tuple({"name": name} for name in ceiling),
    local_tool_handlers=local,
    mcp_client=mcp,
    effect_resolver=_effect,
  )
  authority = resolve_operation_authority(
    OperationDeclaration(
      operation_name=operation.operation.name,
      grant_id="grant:parity",
      workspace_scope=operation.workspace_scope,
      required_capabilities=operation.required_capabilities,
      tool_ceiling=frozenset(ceiling),
    ),
    catalog=snapshot_platform_catalog(
      tool_ids=ceiling,
      local_tool_handlers=local,
      mcp_client=mcp,
      effect_resolver=_effect,
    ),
  )

  assert isinstance(authority, ResolvedAuthority)
  assert isinstance(legacy, ResolvedAuthority)
  assert authority.grant == legacy.grant
  assert authority.bindings == legacy.bindings
  assert granted_tool_ids(authority) == granted_tool_ids(legacy)
  assert derive_dispatcher_allowlist(authority) == derive_dispatcher_allowlist(
    legacy
  )


def test_resolver_reuses_the_live_semantic_tool_routes_facts() -> None:
  """The catalog's routing facts are the receipt's, not a second dialect."""

  local = {"web_search": object()}
  mcp = _Mcp()
  ceiling = ("web_search", "filings_search", "memory_read")
  legacy_routes = semantic_tool_routes(
    ceiling,
    local_tool_handlers=local,
    mcp_client=mcp,
    effect_resolver=_effect,
  )
  catalog = snapshot_platform_catalog(
    tool_ids=ceiling,
    local_tool_handlers=local,
    mcp_client=mcp,
    effect_resolver=_effect,
  )
  declaration = OperationDeclaration(
    operation_name="explore",
    grant_id="grant:routes",
    workspace_scope="model_write",
    required_capabilities=(),
    tool_ceiling=frozenset(ceiling),
  )

  admitted = admitted_catalog_routes(declaration, catalog=catalog)

  assert tuple(
    (entry.tool_id, entry.effect, entry.server_id) for entry in admitted
  ) == tuple(
    (route.tool_id, route.effect, route.server_id) for route in legacy_routes
  )


def test_mcp_tool_ceiling_restricts_mcp_routes_to_declared_ones() -> None:
  catalog = _routed_catalog()
  declaration = OperationDeclaration(
    operation_name="explore",
    grant_id="grant:mcp-ceiling",
    workspace_scope="read_only",
    required_capabilities=_evidence_requirements(
      "web.read/v1",
      "filings.read/v1",
    ),
    tool_ceiling=frozenset({"web_search", "filings_search", "transcripts_search"}),
    mcp_tool_ceiling=frozenset({"filings_search"}),
  )

  authority = resolve_operation_authority(declaration, catalog=catalog)

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == {"web_search", "filings_search"}


def test_exclusions_are_a_resolver_input_not_a_post_hoc_subtraction() -> None:
  catalog = _routed_catalog()
  declaration = OperationDeclaration(
    operation_name="explore",
    grant_id="grant:exclusions",
    workspace_scope="read_only",
    required_capabilities=(_requirement(),),  # web.read/v1 only
    tool_ceiling=frozenset({"web_search", "web_fetch", "filings_search"}),
  )

  authority = resolve_operation_authority(
    declaration,
    catalog=catalog,
    exclusions=("web_fetch", "filings_search"),
  )

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == {"web_search"}
  assert derive_dispatcher_allowlist(authority) == {}


def test_unavailable_operation_is_a_visible_left_naming_the_capability() -> None:
  catalog = _routed_catalog()
  declaration = OperationDeclaration(
    operation_name="explore",
    grant_id="grant:starved",
    workspace_scope="read_only",
    required_capabilities=(_requirement("filings.read/v1"),),
    tool_ceiling=frozenset({"memory_read"}),
  )

  unavailable = resolve_operation_authority(declaration, catalog=catalog)

  assert isinstance(unavailable, OperationUnavailable)
  assert unavailable.operation_name == "explore"
  assert unavailable.code == "missing_route"
  assert [item.capability for item in unavailable.unsatisfied] == [
    "filings.read/v1",
  ]
  assert unavailable.unsatisfied[0].reason == "no_compatible_route"
  assert unavailable.unsatisfied[0].required is True


def test_optional_capability_without_a_route_still_resolves() -> None:
  catalog = _routed_catalog()
  declaration = OperationDeclaration(
    operation_name="coverage-synthesis",
    grant_id="grant:optional",
    workspace_scope="read_only",
    required_capabilities=(_requirement("filings.read/v1", required=False),),
    tool_ceiling=frozenset({"memory_read"}),
  )

  authority = resolve_operation_authority(declaration, catalog=catalog)

  assert isinstance(authority, ResolvedAuthority)
  assert authority.grant.tools == ()
  assert authority.routes == ()


def test_unknown_workspace_scope_is_a_visible_left_not_a_raise() -> None:
  declaration = OperationDeclaration(
    operation_name="explore",
    grant_id="grant:scope",
    workspace_scope="omnipotent",
    required_capabilities=(),
    tool_ceiling=frozenset({"web_search"}),
  )

  unavailable = resolve_operation_authority(declaration, catalog=_routed_catalog())

  assert isinstance(unavailable, OperationUnavailable)
  assert unavailable.code == "invalid_metadata"
  assert "workspace scope" in unavailable.detail


def test_resolver_rejects_inexact_inputs() -> None:
  with pytest.raises(CapabilityResolutionInputError):
    resolve_operation_authority(object(), catalog=_routed_catalog())  # type: ignore[arg-type]
  with pytest.raises(CapabilityResolutionInputError):
    resolve_operation_authority(
      OperationDeclaration(
        operation_name="explore",
        grant_id="grant:bad-catalog",
        workspace_scope="read_only",
      ),
      catalog=object(),  # type: ignore[arg-type]
    )


def test_identity_rides_on_the_resolved_authority() -> None:
  identity = ExecutionIdentity(
    tenant_id="test-tenant",
    credential_handle_id="handle-1",
  )
  authority = resolve_operation_authority(
    OperationDeclaration(
      operation_name="explore",
      grant_id="grant:identity",
      workspace_scope="read_only",
      required_capabilities=(_requirement(),),
      tool_ceiling=frozenset({"web_search"}),
    ),
    catalog=_routed_catalog(),
    identity=identity,
  )

  assert isinstance(authority, ResolvedAuthority)
  assert authority.identity == identity


def test_resolved_routes_are_exactly_the_granted_tools() -> None:
  authority = resolve_operation_authority(
    OperationDeclaration(
      operation_name="explore",
      grant_id="grant:routes-cover",
      workspace_scope="read_only",
      required_capabilities=_evidence_requirements(
        "web.read/v1",
        "filings.read/v1",
      ),
      tool_ceiling=frozenset({"web_search", "filings_search", "memory_read"}),
    ),
    catalog=_routed_catalog(),
  )

  assert isinstance(authority, ResolvedAuthority)
  assert {route.tool_id for route in authority.routes} == granted_tool_ids(authority)
  with pytest.raises(ValueError, match="exactly the granted tools"):
    ResolvedAuthority(
      operation_name="explore",
      grant=authority.grant,
      bindings=authority.bindings,
      routes=(),
    )


def test_dispatcher_allowlist_groups_granted_tools_by_server() -> None:
  authority = resolve_operation_authority(
    OperationDeclaration(
      operation_name="explore",
      grant_id="grant:allowlist",
      workspace_scope="read_only",
      required_capabilities=_evidence_requirements(
        "web.read/v1",
        "filings.read/v1",
        "transcripts.read/v1",
      ),
      tool_ceiling=frozenset({
        "web_search",
        "filings_search",
        "transcripts_search",
      }),
    ),
    catalog=_routed_catalog(),
  )

  assert isinstance(authority, ResolvedAuthority)
  assert derive_dispatcher_allowlist(authority) == {
    "research-corpus-mcp": frozenset({"filings_search", "transcripts_search"}),
  }


def test_snapshot_declarations_override_is_the_only_table_input() -> None:
  table = build_tool_dispatch_declarations(effect_resolver=lambda _name: "write")
  catalog = snapshot_platform_catalog(declarations=table)

  assert catalog.tools
  assert all(entry.effect == "write" for entry in catalog.tools)
