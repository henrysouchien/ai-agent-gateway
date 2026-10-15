from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_gateway import tool_dispatch_declarations as declarations
from agent_gateway.tool_dispatch_declarations import (
  DEFAULT_TOOL_DISPATCH_SEMANTICS,
  ToolDispatchSemantics,
  build_tool_dispatch_declarations,
  tool_dispatch_semantics_for,
  tool_dispatch_semantics_overrides,
)
from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  VersionedPolicyRef,
)


def _local(name: str) -> RegisteredToolIdentity:
  return RegisteredToolIdentity(route_kind="local_handler", logical_name=name)


def _mcp(server: str, name: str) -> RegisteredToolIdentity:
  return RegisteredToolIdentity(
    route_kind="mcp",
    logical_server_id=server,
    logical_name=name,
  )


def _semantics(
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


_EXACT = _mcp("research-corpus-mcp", "filings_search")
_VENDOR = _mcp("market-data-mcp", "fetch_financials")
_POSITIONING = _mcp("positioning-mcp", "get_topdown_positioning")
_DECLARED = {
  _EXACT: _semantics(
    outcome_id="current-declared-status-equals",
    outcome_parameters={"field": "status", "values": ("success",)},
    source_id="search_hits",
    source_parameters={"container": "hits", "default_source_kind": "filing"},
  ),
  _VENDOR: _semantics(
    source_id="vendor_sources",
    source_parameters={"provider": "fmp"},
  ),
  _POSITIONING: _semantics(source_id="positioning_filing_sources"),
}


@pytest.fixture(autouse=True)
def declared_dispatch_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
  """Registration owners declare each route's dispatch semantics; the gateway
  reads that declaration through the bound policy and keeps no list of its own."""

  monkeypatch.setattr(
    declarations,
    "load_server_policy_module",
    lambda: SimpleNamespace(get_dispatch_semantics_overrides=lambda: _DECLARED),
  )
  monkeypatch.setattr(declarations, "_CACHED_DECLARATIONS", None)


class _Text(str):
  pass


def _forged_identity(
  route_kind: object,
  logical_name: object,
  logical_server_id: object,
) -> RegisteredToolIdentity:
  identity = object.__new__(RegisteredToolIdentity)
  object.__setattr__(identity, "route_kind", route_kind)
  object.__setattr__(identity, "logical_name", logical_name)
  object.__setattr__(identity, "logical_server_id", logical_server_id)
  return identity


def test_unbound_policy_declares_no_dispatch_semantics(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(declarations, "load_server_policy_module", lambda: None)

  assert dict(tool_dispatch_semantics_overrides()) == {}
  assert tool_dispatch_semantics_for(_EXACT) is DEFAULT_TOOL_DISPATCH_SEMANTICS
  assert dict(build_tool_dispatch_declarations()) == {}


def test_dispatch_semantics_are_total_by_exact_identity_only() -> None:
  wrong_local = _local("filings_search")
  wrong_mcp = _mcp("market-data-mcp", "filings_search")

  assert tool_dispatch_semantics_for(_EXACT) is _DECLARED[_EXACT]
  assert tool_dispatch_semantics_for(wrong_local) is DEFAULT_TOOL_DISPATCH_SEMANTICS
  assert tool_dispatch_semantics_for(wrong_mcp) is DEFAULT_TOOL_DISPATCH_SEMANTICS
  assert DEFAULT_TOOL_DISPATCH_SEMANTICS.idempotent is False
  assert (
    DEFAULT_TOOL_DISPATCH_SEMANTICS.outcome_policy.policy_id
    == "current_generic_non_error"
  )
  assert DEFAULT_TOOL_DISPATCH_SEMANTICS.outcome_policy.version == "v1"
  assert (
    DEFAULT_TOOL_DISPATCH_SEMANTICS.source_identity_policy.policy_id
    == "current_no_gateway_extraction"
  )
  assert DEFAULT_TOOL_DISPATCH_SEMANTICS.source_identity_policy.version == "v1"
  with pytest.raises(TypeError, match="exact RegisteredToolIdentity"):
    tool_dispatch_semantics_for(_EXACT.materialize())  # type: ignore[arg-type]


@pytest.mark.parametrize(
  "identity",
  [
    _forged_identity(
      _Text("mcp"),
      "filings_search",
      "research-corpus-mcp",
    ),
    _forged_identity(
      "mcp",
      _Text("filings_search"),
      "research-corpus-mcp",
    ),
    _forged_identity(
      "mcp",
      "filings_search",
      _Text("research-corpus-mcp"),
    ),
    _forged_identity("mcp", "filings_search", None),
  ],
)
def test_dispatch_semantics_revalidate_forged_exact_class_identities(
  identity: RegisteredToolIdentity,
) -> None:
  with pytest.raises(TypeError):
    tool_dispatch_semantics_for(identity)


def test_dispatch_projection_names_must_be_unique(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    declarations,
    "load_server_policy_module",
    lambda: SimpleNamespace(
      get_dispatch_semantics_overrides=lambda: {
        _mcp("a-mcp", "shared_name"): _semantics(),
        _mcp("b-mcp", "shared_name"): _semantics(),
      },
    ),
  )

  with pytest.raises(RuntimeError, match="must be unique"):
    build_tool_dispatch_declarations()


def test_dispatch_projection_keeps_registered_vendor_sources_out_of_legacy_reader() -> None:
  actual = build_tool_dispatch_declarations(effect_resolver=lambda _name: "read")

  assert actual["fetch_financials"].source_identity is None
  assert actual["fetch_financials"].idempotent is True
  assert actual["filings_search"].success_signal == {
    "kind": "status_equals",
    "field": "status",
    "values": ("success",),
  }
  assert actual["filings_search"].source_identity == {
    "kind": "search_hits",
    "container": "hits",
    "default_source_kind": "filing",
  }


@pytest.mark.parametrize("status", [
  "ok",
  "success",
  "methodology_validation_not_run",
  "methodology_validation_incomplete",
  "methodology_validation_failed",
  "insufficient_residual_signal_quality",
  "no_published_period",
])
def test_positioning_filing_sources_declaration_and_reader(status: str) -> None:
  from agent_gateway.tool_dispatch_source_identity import read_source_identities

  declaration = build_tool_dispatch_declarations()["get_topdown_positioning"]
  assert declaration.source_identity == {"kind": "positioning_filing_sources"}
  from agent_gateway.tool_dispatch_classification import classify_tool_outcome

  assert declaration.success_signal is None
  result = {
    "status": status,
    "resolved_period": {"year": 2026, "quarter": 2, "selection": "latest"},
    "filing_sources": [
      {
        "manager_cik": "0001067983",
        "year": 2026,
        "quarter": 2,
        "source_url": "https://financialmodelingprep.com/13f/1067983/2026/2",
      },
      {
        "manager_cik": "0001364742",
        "year": 2026,
        "quarter": 2,
        "source_url": "https://financialmodelingprep.com/13f/1364742/2026/2",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1364742/filing",
        "accession": "0001364742-26-000010",
      },
    ],
  }
  assert classify_tool_outcome(None, result, None) == "ok"
  assert read_source_identities(declaration.source_identity, result) == (
    {
      "document_id": "fmp:13f:0001067983:2026:Q2",
      "source_kind": "vendor",
      "provider": "fmp",
      "source_url": "https://financialmodelingprep.com/13f/1067983/2026/2",
      "endpoint": "get_topdown_positioning",
      "key_fields": {"manager_cik": "0001067983", "year": 2026, "quarter": 2},
    },
    {
      "document_id": "fmp:13f:0001364742:2026:Q2:0001364742-26-000010",
      "source_kind": "vendor",
      "provider": "fmp",
      "source_url": "https://financialmodelingprep.com/13f/1364742/2026/2",
      "endpoint": "get_topdown_positioning",
      "key_fields": {
        "manager_cik": "0001364742",
        "year": 2026,
        "quarter": 2,
        "accession": "0001364742-26-000010",
      },
    },
  )


def test_positioning_error_results_mint_no_sources() -> None:
  from agent_gateway.tool_dispatch_classification import (
    DispatchEntry,
    settle_catalogless_tool_result,
  )
  from agent_workflow_contracts import CatalogToolEntry

  declaration = build_tool_dispatch_declarations()["get_topdown_positioning"]
  assert declaration.success_signal is None
  assert declaration.source_identity is not None
  entry = DispatchEntry(
    tool_name="get_topdown_positioning",
    canonical_name="get_topdown_positioning",
    route_id="positioning-mcp",
    catalog_entry=CatalogToolEntry(
      tool_id="get_topdown_positioning",
      canonical_name="get_topdown_positioning",
      success_signal=declaration.success_signal,
      source_identity=dict(declaration.source_identity),
    ),
  )

  result = {
    "status": "error",
    "reason": "no_published_period",
    "filing_sources": [{
      "manager_cik": "0001067983",
      "year": 2026,
      "quarter": 2,
      "source_url": "https://financialmodelingprep.com/13f/1067983/2026/2",
    }],
  }
  settlement = settle_catalogless_tool_result(entry=entry, result=result, error=None)
  assert settlement.outcome == "error_semantic"
  assert settlement.sources == ()


@pytest.mark.parametrize("filing_sources", [None, []])
def test_positioning_without_filing_sources_mints_nothing(
  filing_sources: object,
) -> None:
  from agent_gateway.tool_dispatch_source_identity import read_source_identities

  result: dict[str, object] = {"status": "ok"}
  if filing_sources is not None:
    result["filing_sources"] = filing_sources
  assert read_source_identities({"kind": "positioning_filing_sources"}, result) == ()


@pytest.mark.parametrize(("provider", "key_fields", "source_url"), [
  ("fred", {"series_id": "DGS10", "observation_dates": ["2026-10-07"]},
   "https://api.stlouisfed.org/fred/series/observations"),
  ("fmp", {"symbol": "SPY", "timestamp": 1791324000},
   "https://financialmodelingprep.com/stable/quote"),
  ("orats", {"ticker": "SPY", "trade_dates": ["2026-10-07"]},
   "https://api.orats.io/datav2/hist/cores"),
])
def test_producer_vendor_reader_preserves_coordinates(
  provider: str, key_fields: dict[str, object], source_url: str,
) -> None:
  from agent_gateway.tool_dispatch_source_identity import read_source_identities

  row = {
    "source_kind": "vendor", "provider": provider, "endpoint": "observations",
    "key_fields": key_fields, "source_url": source_url,
    "section_label": "Macro observations", "as_of": "2026-10-07",
    "as_of_basis": "observation",
  }
  assert read_source_identities({"kind": "producer_vendor_sources"}, {
    "sources": [row],
  }) == ({"document_id": source_url, **row},)


@pytest.mark.parametrize("result", [{}, {"sources": None}, {"sources": {}}])
def test_producer_vendor_reader_without_source_rows(result: dict[str, object]) -> None:
  from agent_gateway.tool_dispatch_source_identity import read_source_identities

  assert read_source_identities({"kind": "producer_vendor_sources"}, result) == ()
