from __future__ import annotations

from collections import Counter
from types import MappingProxyType

import pytest

from agent_gateway.tool_dispatch_declarations import (
  DEFAULT_TOOL_DISPATCH_SEMANTICS,
  ToolDispatchSemantics,
  build_tool_dispatch_declarations,
  tool_dispatch_semantics_for,
  tool_dispatch_semantics_overrides,
)
from agent_workflow_contracts.tool_registration import RegisteredToolIdentity


def _local(name: str) -> RegisteredToolIdentity:
  return RegisteredToolIdentity(route_kind="local_handler", logical_name=name)


def _mcp(server: str, name: str) -> RegisteredToolIdentity:
  return RegisteredToolIdentity(
    route_kind="mcp",
    logical_server_id=server,
    logical_name=name,
  )


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


def test_exact_dispatch_overrides_route_population() -> None:
  overrides = tool_dispatch_semantics_overrides()

  assert isinstance(overrides, MappingProxyType)
  assert set(
    identity.logical_name
    for identity in overrides
    if identity.route_kind == "local_handler"
  ) == {
    "code_execute",
    "code_execute_status",
    "fms_compute_quantifying_risk",
    "web_fetch",
  }
  assert all(type(value) is ToolDispatchSemantics for value in overrides.values())
  for name in ("code_execute", "code_execute_status"):
    semantics = overrides[_local(name)]
    assert semantics.idempotent is False
    assert semantics.outcome_policy.policy_id == "current_generic_non_error"
    assert semantics.source_identity_policy.policy_id == "sandbox_computations"
  compute = overrides[_local("fms_compute_quantifying_risk")]
  assert compute.idempotent is True
  assert compute.outcome_policy.parameters == {
    "field": "status",
    "values": ("ok",),
  }
  assert compute.source_identity_policy.policy_id == "fms-computation"


def test_dispatch_semantics_are_total_by_exact_identity_only() -> None:
  exact = _mcp("research-corpus-mcp", "filings_search")
  wrong_local = _local("filings_search")
  wrong_mcp = _mcp("market-data-mcp", "filings_search")

  assert tool_dispatch_semantics_for(exact) is tool_dispatch_semantics_overrides()[exact]
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
    tool_dispatch_semantics_for(exact.materialize())  # type: ignore[arg-type]


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




_SUCCESS_SOURCE_DESCRIPTORS = {
  "fms_compute_quantifying_risk": {"kind": "fms-computation"},
  "web_fetch": {"kind": "web_fetch"},
  "filings_search": {
    "kind": "search_hits",
    "container": "hits",
    "default_source_kind": "filing",
  },
  "transcripts_search": {
    "kind": "search_hits",
    "container": "hits",
    "default_source_kind": "transcript",
  },
  "filings_list": {
    "kind": "documents",
    "container": "documents",
    "source_kind": "filing",
  },
  "transcripts_list": {
    "kind": "documents",
    "container": "documents",
    "source_kind": "transcript",
  },
  "filings_read": {"kind": "single_document", "source_kind": "filing"},
  "transcripts_read": {
    "kind": "single_document",
    "source_kind": "transcript",
  },
  "filings_source_excerpt": {
    "kind": "single_document",
    "source_kind": "filing",
  },
  "transcripts_source_excerpt": {
    "kind": "single_document",
    "source_kind": "transcript",
  },
  "get_filings": {"kind": "parser_filings"},
  "get_filing_sections": {"kind": "parser_filing_sections"},
  "search_filing_text": {
    "kind": "parser_items",
    "container": "hits",
    "default_source_kind": "filing",
  },
  "get_filing_evidence": {
    "kind": "parser_items",
    "container": "evidence",
    "default_source_kind": "filing_evidence",
  },
  "cite_concept": {
    "kind": "parser_items",
    "container": "citations",
    "default_source_kind": "concept_citation",
  },
  "get_filing_document": {"kind": "parser_document"},
  "get_metric": {"kind": "metric_citations"},
}


def test_dispatch_projection_keeps_registered_vendor_sources_out_of_legacy_reader() -> None:
  actual = build_tool_dispatch_declarations(effect_resolver=lambda _name: "read")
  exact_vendor_sources = {
    identity: semantics.source_identity_policy.parameters["provider"]
    for identity, semantics in tool_dispatch_semantics_overrides().items()
    if semantics.source_identity_policy.policy_id == "vendor_sources"
  }

  assert all(
    actual[identity.logical_name].source_identity is None
    for identity in exact_vendor_sources
  )
  assert Counter(
    (identity.logical_server_id, provider)
    for identity, provider in exact_vendor_sources.items()
  ) == {
    ("market-data-mcp", "fmp"): 11,
    ("fred-mcp", "fred"): 2,
    ("portfolio-reads-mcp", "portfolio"): 4,
    ("portfolio-reads-mcp", "fmp"): 1,
    ("gsheets-mcp", "gsheets"): 1,
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

  identity = _mcp("positioning-mcp", "get_topdown_positioning")
  semantics = tool_dispatch_semantics_for(identity)
  assert semantics.idempotent is True
  assert semantics.outcome_policy.policy_id == "current_generic_non_error"
  assert semantics.outcome_policy.parameters == {}
  assert semantics.source_identity_policy.policy_id == "positioning_filing_sources"
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
      {"manager_cik": "0001166559", "year": 2026, "quarter": 2},
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


@pytest.mark.parametrize("missing_field", ["manager_cik", "year", "quarter"])
def test_positioning_skips_incomplete_filing_identity(missing_field: str) -> None:
  from agent_gateway.tool_dispatch_source_identity import read_source_identities

  filing = {
    "manager_cik": "0001067983",
    "year": 2026,
    "quarter": 2,
    "source_url": "https://financialmodelingprep.com/13f/1067983/2026/2",
  }
  del filing[missing_field]
  assert read_source_identities({"kind": "positioning_filing_sources"}, {
    "status": "success",
    "filing_sources": [filing],
  }) == ()


@pytest.mark.parametrize("name", [
  "get_macro_dashboard", "get_rates_and_curves", "get_economic_indicators",
  "get_sector_analysis", "get_technical_summary", "get_positioning",
])
def test_macro_declares_producer_vendor_sources(name: str) -> None:
  semantics = tool_dispatch_semantics_for(_mcp("macro-mcp", name))
  assert semantics.idempotent is True
  assert semantics.outcome_policy.policy_id == "current_generic_non_error"
  assert semantics.outcome_policy.parameters == {}
  assert semantics.source_identity_policy.policy_id == "producer_vendor_sources"
  declaration = build_tool_dispatch_declarations()[name]
  assert declaration.success_signal is None
  assert declaration.source_identity == {"kind": "producer_vendor_sources"}


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


@pytest.mark.parametrize(("field", "value"), [
  ("source_kind", "filing"), ("provider", ""), ("provider", 1),
  ("endpoint", " "), ("endpoint", None), ("source_url", ""),
  ("source_url", 1), ("key_fields", {}), ("key_fields", []),
])
def test_producer_vendor_reader_skips_malformed_rows(field: str, value: object) -> None:
  from agent_gateway.tool_dispatch_source_identity import read_source_identities

  valid: dict[str, object] = {
    "source_kind": "vendor", "provider": "fred", "endpoint": "observations",
    "key_fields": {"series_id": "DGS10"},
    "source_url": "https://api.stlouisfed.org/fred/series/observations",
  }
  assert read_source_identities({"kind": "producer_vendor_sources"}, {
    "sources": [None, "invalid", {**valid, field: value}, valid],
  }) == ({"document_id": valid["source_url"], **valid},)


@pytest.mark.parametrize("result", [{}, {"sources": None}, {"sources": {}}])
def test_producer_vendor_reader_without_source_rows(result: dict[str, object]) -> None:
  from agent_gateway.tool_dispatch_source_identity import read_source_identities

  assert read_source_identities({"kind": "producer_vendor_sources"}, result) == ()
