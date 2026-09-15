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
from api.agent.shared.tool_policy_implementations import (
  build_product_tool_policy_implementation_registry,
)


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


def test_exact_dispatch_overrides_have_the_fixed_40_route_population() -> None:
  overrides = tool_dispatch_semantics_overrides()

  assert isinstance(overrides, MappingProxyType)
  assert len(overrides) == 40
  assert sum(identity.route_kind == "local_handler" for identity in overrides) == 4
  assert sum(identity.route_kind == "mcp" for identity in overrides) == 36
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
  mcp_names_by_server = {
    server: {
      identity.logical_name
      for identity in overrides
      if identity.logical_server_id == server
    }
    for server in {
      identity.logical_server_id
      for identity in overrides
      if identity.logical_server_id is not None
    }
  }
  assert mcp_names_by_server == {
    "edgar-parser-mcp": {
      "cite_concept",
      "get_filing_document",
      "get_filing_evidence",
      "get_filing_sections",
      "get_filings",
      "get_metric",
      "search_filing_text",
    },
    "fred-mcp": {
      "fred_get_multiple",
      "fred_get_series",
      "fred_list_series",
      "fred_search",
    },
    "gsheets-mcp": {"gsheets_read_range"},
    "market-data-mcp": {
      "compare_peers",
      "fetch_company_profile",
      "fetch_financials",
      "get_economic_data",
      "get_estimate_revisions",
      "get_insider_trades",
      "get_institutional_ownership",
      "get_market_context",
      "get_price_performance_windows",
      "get_sector_overview",
      "screen_estimate_revisions",
    },
    "portfolio-reads-mcp": {
      "get_positions",
      "get_quote",
      "get_risk_analysis",
      "industry_peer_comparison",
      "run_whatif",
    },
    "research-corpus-mcp": {
      "filings_list",
      "filings_read",
      "filings_search",
      "filings_source_excerpt",
      "transcripts_list",
      "transcripts_read",
      "transcripts_search",
      "transcripts_source_excerpt",
    },
  }
  assert all(type(value) is ToolDispatchSemantics for value in overrides.values())
  assert sum(value.idempotent for value in overrides.values()) == 38
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


def test_every_final_dispatch_policy_ref_exists_and_accepts_its_parameters() -> None:
  registry = build_product_tool_policy_implementation_registry()
  semantics = (
    DEFAULT_TOOL_DISPATCH_SEMANTICS,
    *tool_dispatch_semantics_overrides().values(),
  )

  for value in semantics:
    registry.validate_reference(value.outcome_policy)
    registry.validate_reference(value.source_identity_policy)


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

  assert len(actual) == 40
  assert len(exact_vendor_sources) == 19
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
