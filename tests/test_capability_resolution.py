"""The platform catalog snapshot and the single capability resolver (B-5).

The resolver ships in effect-compatible mode: every satisfaction verdict is
``compile_semantic_capabilities``'s.  These tests pin that the wrap is exact —
the resolver's grant, bindings and dispatcher allowlist are byte-for-byte the
legacy answer over the same routes — plus the catalog snapshot's routing rules
and the visible Left.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_gateway.capability_resolution import (
  CapabilityResolutionInputError,
  OperationDeclaration,
  admitted_tool_routes,
  admitted_catalog_routes,
  derive_dispatcher_allowlist,
  granted_tool_ids,
  inherited_operation_declaration,
  lookup_catalog_entry,
  resolve_operation_authority,
  snapshot_platform_catalog,
)
from agent_gateway.semantic_capabilities import (
  DEFAULT_SEMANTIC_CAPABILITY_REGISTRY,
  SemanticToolRoute,
  compile_semantic_capabilities,
)
from agent_gateway.semantic_capability_routing import capability_for_tool
from agent_gateway.tool_dispatch_declarations import ToolDispatchDecl
from agent_gateway.sub_agent_scope_receipt import (
  admit_operation_tools,
)
from agent_gateway.skills import compile_agent_operation, generic_explore_profile
from agent_gateway.tool_dispatch_declarations import (
  ToolDispatchSemantics,
  build_tool_dispatch_declarations,
)
from agent_workflow_contracts import (
  CatalogToolEntry,
  ExecutionIdentity,
  OperationUnavailable,
  PlatformToolCatalog,
  ResolvedAuthority,
  SemanticCapabilityRequirement,
  ToolGrant,
  ToolGrantEntry,
  sha256_digest,
)
from agent_workflow_contracts.models import CatalogToolEffect
from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  VersionedPolicyRef,
)


_MCP_ROUTES = {
  "filings_search": "research-corpus-mcp",
  "transcripts_search": "research-corpus-mcp",
  "compare_peers": "market-data-mcp",
  "fetch_financials": "market-data-mcp",
  "corpus_inventory": "research-corpus-mcp",
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

  def get_policy_tool_name(self, name: str) -> str | None:
    return name if name in self.routes else None


def _effect(
  tool_id: str,
  _server: str | None,
  _local: bool,
) -> CatalogToolEffect | None:
  if tool_id in {
    "web_search",
    "web_fetch",
    "filings_search",
    "transcripts_search",
  }:
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


_DECLARED_READ_CAPABILITY = {
  "web_search": "web.read/v1",
  "web_fetch": "web.read/v1",
  "filings_search": "filings.read/v1",
  "transcripts_search": "transcripts.read/v1",
  "memory_read": "corpus.read/v1",
  "fetch_financials": "market-data.read/v1",
}


@pytest.fixture(autouse=True)
def declared_read_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
  """The registration owner declares each read route's capability; the
  gateway reads that declaration and keeps no per-tool list of its own."""
  from agent_gateway import semantic_capability_routing

  monkeypatch.setattr(
    semantic_capability_routing,
    "load_server_policy_module",
    lambda: SimpleNamespace(
      get_read_capability=lambda name, _server: _DECLARED_READ_CAPABILITY.get(name),
    ),
  )


def _declared_semantics(
  *,
  outcome_id: str = "current_generic_non_error",
  outcome_parameters: dict[str, object] | None = None,
  source_id: str = "current_no_gateway_extraction",
  source_parameters: dict[str, object] | None = None,
) -> ToolDispatchSemantics:
  return ToolDispatchSemantics(
    idempotent=True,
    outcome_policy=VersionedPolicyRef(
      kind="outcome",
      policy_id=outcome_id,
      version="v1",
      parameters=outcome_parameters or {},
    ),
    source_identity_policy=VersionedPolicyRef(
      kind="source_identity",
      policy_id=source_id,
      version="v1",
      parameters=source_parameters or {},
    ),
  )


_SUCCESS_STATUS = {"field": "status", "values": ("success",)}
_DECLARED_DISPATCH_SEMANTICS = {
  RegisteredToolIdentity(
    route_kind="local_handler",
    logical_name="web_fetch",
  ): _declared_semantics(
    outcome_id="current-declared-status-equals",
    outcome_parameters=_SUCCESS_STATUS,
    source_id="web_fetch",
  ),
  RegisteredToolIdentity(
    route_kind="mcp",
    logical_server_id="research-corpus-mcp",
    logical_name="filings_search",
  ): _declared_semantics(
    outcome_id="current-declared-status-equals",
    outcome_parameters=_SUCCESS_STATUS,
    source_id="search_hits",
    source_parameters={"container": "hits", "default_source_kind": "filing"},
  ),
  RegisteredToolIdentity(
    route_kind="mcp",
    logical_server_id="research-corpus-mcp",
    logical_name="transcripts_search",
  ): _declared_semantics(
    outcome_id="current-declared-status-equals",
    outcome_parameters=_SUCCESS_STATUS,
    source_id="search_hits",
    source_parameters={
      "container": "hits",
      "default_source_kind": "transcript",
    },
  ),
  RegisteredToolIdentity(
    route_kind="mcp",
    logical_server_id="market-data-mcp",
    logical_name="fetch_financials",
  ): _declared_semantics(
    source_id="vendor_sources",
    source_parameters={"provider": "fmp"},
  ),
}


@pytest.fixture(autouse=True)
def declared_dispatch_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
  """The registration owner declares each route's dispatch semantics; the
  gateway reads that declaration and keeps no per-tool list of its own."""
  from agent_gateway import tool_dispatch_declarations

  monkeypatch.setattr(
    tool_dispatch_declarations,
    "load_server_policy_module",
    lambda: SimpleNamespace(
      get_dispatch_semantics_overrides=lambda: _DECLARED_DISPATCH_SEMANTICS,
    ),
  )
  monkeypatch.setattr(tool_dispatch_declarations, "_CACHED_DECLARATIONS", None)


@pytest.fixture
def declared_route_policy(monkeypatch: pytest.MonkeyPatch) -> None:
  from agent_gateway import capability_resolution

  policy = SimpleNamespace(
    get_server_for_policy_tool=_MCP_ROUTES.get,
    get_local_tool_effect=lambda name: (
      _effect(name, None, True) if name not in _MCP_ROUTES else None
    ),
  )
  monkeypatch.setattr(
    capability_resolution,
    "load_server_policy_module",
    lambda: policy,
  )


# --- the snapshot ----------------------------------------------------------


def test_declarative_snapshot_is_seeded_from_the_dispatch_declaration_table() -> None:
  table = build_tool_dispatch_declarations(
    effect_resolver=lambda name: _effect(name, None, False),
  )
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
    assert entry.origin is None
    assert "origin" not in entry.model_dump(mode="json")
    assert entry.server_id is None
    # `capability` is SINGULAR (D-B5-2) and B-8 populates it from the one
    # derivation the codegen also reads.
    assert entry.capability == capability_for_tool(
      canonical_name=entry.canonical_name,
      server_id=entry.server_id,
      effect=entry.effect,
    )
  filings_search = catalog.entry("filings_search")
  transcripts_search = catalog.entry("transcripts_search")
  web_fetch = catalog.entry("web_fetch")
  assert filings_search is not None
  assert transcripts_search is not None
  assert web_fetch is not None
  assert filings_search.capability == "filings.read/v1"
  assert transcripts_search.capability == "transcripts.read/v1"
  assert web_fetch.capability == "web.read/v1"


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


@pytest.mark.usefixtures("declared_route_policy")
def test_dispatch_lookup_requires_exact_live_route_and_policy_owner() -> None:
  entry = lookup_catalog_entry(
    "mcp__research-corpus-mcp__filings_search",
    origin="mcp",
    server="research-corpus-mcp",
    original_tool_name="filings_search",
  )

  assert entry is not None
  assert entry.origin == "mcp"
  assert entry.server_id == "research-corpus-mcp"
  assert entry.canonical_name == "filings_search"
  assert lookup_catalog_entry(
    "mcp__wrong-mcp__filings_search",
    origin="mcp",
    server="research-corpus-mcp",
    original_tool_name="filings_search",
  ) is None

  logical_alias = lookup_catalog_entry(
    "fetch_financials",
    origin="mcp",
    server="market-data-mcp",
    original_tool_name="fetch_financials",
  )
  assert logical_alias is not None
  assert logical_alias.server_id == "market-data-mcp"
  assert lookup_catalog_entry(
    "fmp_fetch",
    origin="mcp",
    server="fmp-mcp",
    original_tool_name="fmp_fetch",
  ) is None
  assert lookup_catalog_entry(
    "mcp__research-corpus-mcp__transcripts_search",
    origin="mcp",
    server="research-corpus-mcp",
    original_tool_name="filings_search",
  ) is None


@pytest.mark.usefixtures("declared_route_policy")
def test_dispatch_lookup_refuses_unroutable_local_and_policy_unavailability(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  assert lookup_catalog_entry(
    "web_fetch",
    origin="local",
    server=None,
    original_tool_name="web_fetch",
  ) is not None
  assert lookup_catalog_entry(
    "web_search",
    origin="local",
    server=None,
    original_tool_name="web_fetch",
  ) is None
  assert lookup_catalog_entry(
    "web_fetch",
    origin="mcp",
    server="research-corpus-mcp",
    original_tool_name="web_fetch",
  ) is None
  assert lookup_catalog_entry(
    "filings_search",
    origin="local",
    server=None,
    original_tool_name="filings_search",
  ) is None

  import agent_gateway.capability_resolution as resolution

  monkeypatch.setattr(resolution, "load_server_policy_module", lambda: None)
  assert lookup_catalog_entry(
    "filings_search",
    origin="mcp",
    server="research-corpus-mcp",
    original_tool_name="filings_search",
  ) is None


@pytest.mark.usefixtures("declared_route_policy")
def test_routed_snapshot_keeps_route_effect_but_withholds_wrong_owner_declaration() -> None:
  catalog = snapshot_platform_catalog(
    tool_ids=("filings_search",),
    local_tool_handlers={},
    mcp_client=_Mcp({"filings_search": "wrong-mcp"}),
    effect_resolver=lambda _name, _server, _local: "read",
  )
  entry = catalog.entry("filings_search")

  assert entry is not None
  assert entry.effect == "read"
  assert entry.server_id == "wrong-mcp"
  assert entry.idempotent is None
  assert entry.success_signal is None
  assert entry.source_identity is None


@pytest.mark.usefixtures("declared_route_policy")
def test_routed_snapshot_uses_injected_declaration_table_after_owner_match() -> None:
  custom = {
    "corpus_inventory": ToolDispatchDecl(
      success_signal={"kind": "status_equals", "field": "status", "values": ("ok",)},
      source_identity=None,
      effect="read",
      idempotent=True,
    ),
  }
  catalog = snapshot_platform_catalog(
    tool_ids=("corpus_inventory",),
    local_tool_handlers={},
    mcp_client=_Mcp({"corpus_inventory": "research-corpus-mcp"}),
    declarations=custom,
    effect_resolver=lambda _name, _server, _local: "read",
  )
  entry = catalog.entry("corpus_inventory")

  assert entry is not None
  assert entry.idempotent is True
  assert entry.success_signal == {
    "kind": "status_equals",
    "field": "status",
    "values": ["ok"],
  }

  empty = snapshot_platform_catalog(
    tool_ids=("filings_search",),
    local_tool_handlers={},
    mcp_client=_Mcp(),
    declarations={},
    effect_resolver=lambda _name, _server, _local: "read",
  ).entry("filings_search")
  assert empty is not None
  assert empty.idempotent is None
  assert empty.success_signal is None
  assert empty.source_identity is None




@pytest.mark.usefixtures("declared_route_policy")
def test_routed_snapshot_describes_local_and_mcp_routes_exactly() -> None:
  catalog = _routed_catalog()
  by_id = {entry.tool_id: entry for entry in catalog.tools}

  assert by_id["web_search"].server_id is None
  assert by_id["web_search"].origin == "local"
  assert by_id["filings_search"].server_id == "research-corpus-mcp"
  assert by_id["filings_search"].origin == "mcp"
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


def test_catalog_route_origin_and_server_are_one_coherent_fact() -> None:
  with pytest.raises(ValueError, match="unrouted catalog entries"):
    CatalogToolEntry(
      tool_id="filings_search",
      canonical_name="filings_search",
      server_id="research-corpus-mcp",
    )
  with pytest.raises(ValueError, match="local catalog routes"):
    CatalogToolEntry(
      tool_id="web_search",
      canonical_name="web_search",
      origin="local",
      server_id="research-corpus-mcp",
    )
  with pytest.raises(ValueError, match="MCP catalog routes require"):
    CatalogToolEntry(
      tool_id="filings_search",
      canonical_name="filings_search",
      origin="mcp",
    )


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
  semantic_routes: list[SemanticToolRoute] = []
  for entry in admitted:
    effect = entry.effect
    assert effect is not None
    semantic_routes.append(SemanticToolRoute(
      tool_id=entry.tool_id,
      effect=effect,
      server_id=entry.server_id,
      capability=entry.capability,
    ))
  legacy = compile_semantic_capabilities(
    declaration.required_capabilities,
    grant_id=declaration.grant_id,
    tool_routes=tuple(semantic_routes),
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
  with pytest.raises(ValueError, match="exact physical origin"):
    ResolvedAuthority(
      operation_name="explore",
      grant=authority.grant,
      bindings=authority.bindings,
      routes=tuple(
        route.model_copy(
          update={"origin": None, "server_id": None},
        )
        for route in authority.routes
      ),
    )


def test_admitted_route_projection_uses_grant_order_and_frozen_origin() -> None:
  tools = (
    ToolGrantEntry(tool_id="web_search", route_id="local", effect="read"),
    ToolGrantEntry(
      tool_id="filings_search",
      route_id="research-corpus-mcp",
      effect="read",
    ),
  )
  grant_id = "grant:route-order"
  grant = ToolGrant(
    grant_id=grant_id,
    tools=tools,
    digest=sha256_digest({
      "grant_id": grant_id,
      "tools": [entry.model_dump(mode="json") for entry in tools],
    }),
  )
  authority = ResolvedAuthority(
    operation_name="explore",
    grant=grant,
    # ResolvedAuthority keeps catalog order; the admitted projection follows
    # the grant that provider definitions and dispatch consume.
    routes=(
      CatalogToolEntry(
        tool_id="filings_search",
        canonical_name="filings_search",
        effect="read",
        origin="mcp",
        server_id="server-a",
      ),
      CatalogToolEntry(
        tool_id="web_search",
        canonical_name="web_search",
        effect="read",
        origin="local",
      ),
    ),
  )

  routes = admitted_tool_routes(authority)

  assert tuple(route.tool_id for route in routes) == (
    "web_search",
    "filings_search",
  )
  assert routes[0].origin == "local"
  assert routes[0].server_id is None
  assert routes[1].origin == "mcp"
  assert routes[1].server_id == "server-a"


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


@pytest.fixture
def inherited_catalog() -> PlatformToolCatalog:
  return PlatformToolCatalog(tools=(
    CatalogToolEntry(
      tool_id="external_write",
      canonical_name="external_write",
      effect="external_effect",
      origin="mcp",
      server_id="external-mcp",
      capability="state.mutate/v1",
    ),
    CatalogToolEntry(
      tool_id="filings_search",
      canonical_name="filings_search",
      effect="read",
      origin="mcp",
      server_id="research-corpus-mcp",
      capability="filings.read/v1",
    ),
    CatalogToolEntry(
      tool_id="memory_read",
      canonical_name="memory_read",
      effect="read",
      origin="local",
    ),
    CatalogToolEntry(
      tool_id="memory_write",
      canonical_name="memory_write",
      effect="write",
      origin="local",
      capability="state.mutate/v1",
    ),
    CatalogToolEntry(
      tool_id="web_fetch",
      canonical_name="web_fetch",
      effect="read",
      origin="local",
      capability="web.read/v1",
    ),
  ))


def test_inherited_declaration_keeps_read_write_external_and_local_routes(
  inherited_catalog: PlatformToolCatalog,
) -> None:
  ceiling = frozenset({
    "external_write", "memory_write", "web_fetch",
  })
  declaration = inherited_operation_declaration(
    inherited_catalog,
    operation_name="learning-fork",
    grant_id="grant:inherited",
    tool_ceiling=ceiling,
  )

  assert declaration.operation_name == "learning-fork"
  assert declaration.grant_id == "grant:inherited"
  assert declaration.tool_ceiling is ceiling
  assert declaration.workspace_scope == "model_write"
  assert declaration.mcp_tool_ceiling is None
  assert declaration.inputs == ()
  # The two state mutation routes produce one requirement, not duplicates.
  assert declaration.required_capabilities == (
    _requirement("state.mutate/v1"),
    _requirement("web.read/v1"),
  )
  authority = resolve_operation_authority(declaration, catalog=inherited_catalog)

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == ceiling
  assert "memory_write" in granted_tool_ids(authority)
  assert "external_write" in granted_tool_ids(authority)
  assert "filings_search" not in granted_tool_ids(authority)
  assert derive_dispatcher_allowlist(authority) == {
    "external-mcp": frozenset({"external_write"}),
  }


def test_ceiling_routes_are_granted_without_declared_capabilities(
  inherited_catalog: PlatformToolCatalog,
) -> None:
  """Admission follows the ceilings, not the operation's capability declarations.

  The operation declares nothing; its ceiling names tagged routes spanning
  capabilities it never declared (web read, workspace write, external write).
  Every ceiling-admitted route is granted, unbound, under a deterministic id.
  """

  ceiling = frozenset({"external_write", "memory_write", "web_fetch"})
  declaration = OperationDeclaration(
    operation_name="learning-fork",
    grant_id="grant:empty-requirements",
    workspace_scope="model_write",
    required_capabilities=(),
    tool_ceiling=ceiling,
  )
  authority = resolve_operation_authority(declaration, catalog=inherited_catalog)

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == ceiling
  assert authority.bindings == ()
  assert {entry.tool_id: entry.route_id for entry in authority.grant.tools} == {
    tool_id: f"tool:{tool_id}" for tool_id in ceiling
  }
  assert {route.tool_id for route in authority.routes} == ceiling


@pytest.mark.parametrize(
  ("ceiling", "workspace_scope"),
  (
    (frozenset(), "read_only"),
    (frozenset({"web_fetch"}), "read_only"),
    (frozenset({"web_fetch", "memory_write"}), "workspace_write"),
    (frozenset({"external_write"}), "model_write"),
    (frozenset({"web_fetch", "absent_tool"}), "read_only"),
  ),
)
def test_inherited_declaration_uses_narrowest_scope_and_catalog_routes(
  inherited_catalog: PlatformToolCatalog,
  ceiling: frozenset[str],
  workspace_scope: str,
) -> None:
  declaration = inherited_operation_declaration(
    inherited_catalog,
    operation_name="learning-fork",
    grant_id="grant:inherited-scope",
    tool_ceiling=ceiling,
  )

  assert declaration.workspace_scope == workspace_scope
  assert declaration.tool_ceiling is ceiling
  authority = resolve_operation_authority(declaration, catalog=inherited_catalog)

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == ceiling & frozenset(
    entry.tool_id for entry in inherited_catalog.tools
  )
  if not ceiling:
    assert declaration.required_capabilities == ()


def test_inherited_declaration_logs_and_omits_effectless_routes(
  inherited_catalog: PlatformToolCatalog,
  caplog: pytest.LogCaptureFixture,
) -> None:
  effectless = CatalogToolEntry(
    tool_id="effectless_tool",
    canonical_name="effectless_tool",
    origin="local",
  )
  catalog = PlatformToolCatalog(tools=tuple(sorted(
    (*inherited_catalog.tools, effectless),
    key=lambda entry: entry.tool_id,
  )))
  ceiling = frozenset({
    "effectless_tool", "external_write", "memory_write", "web_fetch",
  })

  declaration = inherited_operation_declaration(
    catalog,
    operation_name="learning-fork",
    grant_id="grant:effectless-inheritance",
    tool_ceiling=ceiling,
  )
  authority = resolve_operation_authority(declaration, catalog=catalog)

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == ceiling - {"effectless_tool"}
  assert declaration.tool_ceiling == ceiling - {"effectless_tool"}
  assert declaration.workspace_scope == "model_write"
  assert declaration.required_capabilities == (
    _requirement("state.mutate/v1"),
    _requirement("web.read/v1"),
  )
  warnings = [
    record for record in caplog.records
    if record.name == "agent_gateway.capability_resolution"
    and record.levelname == "WARNING"
  ]
  assert len(warnings) == 1
  assert "inherited route has no declared effect; not delegable" in warnings[0].message
  assert "learning-fork" in warnings[0].message
  assert "effectless_tool" in warnings[0].message
  assert "admitted without it" in warnings[0].message


def test_inherited_declaration_rejects_a_non_catalog_input() -> None:
  with pytest.raises(CapabilityResolutionInputError, match="PlatformToolCatalog"):
    inherited_operation_declaration(
      object(),  # type: ignore[arg-type]
      operation_name="learning-fork",
      grant_id="grant:malformed-inheritance",
      tool_ceiling=frozenset(),
    )


@pytest.mark.parametrize(
  "ceiling",
  (
    frozenset({"memory_read"}),
    frozenset({"memory_read", "web_fetch", "filings_search"}),
  ),
)
def test_inherited_declaration_rejects_untagged_routes_before_resolution(
  inherited_catalog: PlatformToolCatalog,
  ceiling: frozenset[str],
) -> None:
  with pytest.raises(CapabilityResolutionInputError) as error:
    inherited_operation_declaration(
      inherited_catalog,
      operation_name="learning-fork",
      grant_id="grant:untagged-inheritance",
      tool_ceiling=ceiling,
    )

  assert str(error.value) == (
    "learning-fork: an inherited ceiling can only name routes the catalog tags; "
    "memory_read have no capability"
  )


def test_inherited_tagged_ceiling_grants_exactly_its_routes(
  inherited_catalog: PlatformToolCatalog,
) -> None:
  ceiling = frozenset({"web_fetch", "filings_search"})
  declaration = inherited_operation_declaration(
    inherited_catalog,
    operation_name="learning-fork",
    grant_id="grant:tagged-inheritance",
    tool_ceiling=ceiling,
  )
  authority = resolve_operation_authority(declaration, catalog=inherited_catalog)

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == ceiling
