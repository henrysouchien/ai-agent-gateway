import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
GATEWAY_DIR = ROOT / "packages" / "agent-gateway"
if str(GATEWAY_DIR) not in sys.path:
  sys.path.insert(0, str(GATEWAY_DIR))

from agent_gateway.tool_dispatch_classification import (  # noqa: E402
  DEFAULT_TOOL_RETRY_POLICY,
  RETRYABLE_OUTCOMES,
  DispatchEntry,
  RetryPolicy,
  build_dispatch_record,
  build_dispatch_record_for_outcome,
  build_dispatch_record_from_sources,
  build_route_id,
  classify_semantic_tool_error,
  classify_tool_outcome,
  extract_source_identities,
  resolve_dispatch_entry,
  retry_backoff_seconds,
  retry_decision,
  retry_eligible,
)
from agent_gateway.capability_resolution import (  # noqa: E402
  canonical_dispatch_tool_name,
  lookup_catalog_entry,
)
from agent_gateway.tool_dispatch_declarations import (  # noqa: E402
  build_tool_dispatch_declarations,
)
from agent_gateway.tool_dispatch_source_identity import (  # noqa: E402
  read_source_identities,
)
from agent_workflow_contracts import CatalogToolEntry  # noqa: E402
from agent_workflow_contracts.models import CatalogToolEffect  # noqa: E402


def _entry(
  tool_name: str = "filings_search",
  *,
  effect: CatalogToolEffect | None = "read",
  idempotent: bool | None = True,
  success_signal: dict[str, Any] | None = None,
  source_identity: dict[str, Any] | None = None,
) -> DispatchEntry:
  canonical = canonical_dispatch_tool_name(tool_name)
  described = lookup_catalog_entry(
    tool_name,
    origin="mcp",
    server="research-corpus-mcp",
    original_tool_name=canonical,
  )
  return DispatchEntry(
    tool_name=tool_name,
    canonical_name=canonical,
    route_id=build_route_id(tool_name=tool_name),
    catalog_entry=CatalogToolEntry(
      tool_id=canonical,
      canonical_name=canonical,
      success_signal=(
        success_signal
        if success_signal is not None
        else (described.success_signal if described is not None else None)
      ),
      source_identity=(
        source_identity
        if source_identity is not None
        else (described.source_identity if described is not None else None)
      ),
      effect=effect,
      idempotent=idempotent,
    ),
  )


# --- outcome mapping, one case per exit path -------------------------------


@pytest.mark.parametrize(
  ("error", "expected"),
  [
    ({"code": "cancelled", "message": "Task was cancelled"}, "cancelled"),
    (
      {"code": "tool_timeout", "sub_code": "timeout", "message": "timed out"},
      "error_timeout",
    ),
    ({"code": "rate_limited", "message": "Rate limit: max 5 calls"}, "error_rate_limited"),
    (
      {"code": "broker_rate_limited", "message": "Google Sheets is unavailable"},
      "error_rate_limited",
    ),
    (
      {"code": "tool_error", "message": "upstream returned HTTP 429"},
      "error_rate_limited",
    ),
    ({"code": "internal_error", "message": "boom"}, "error_transport"),
    (
      {"code": "tool_error", "message": "upstream returned HTTP 503"},
      "error_transport",
    ),
    ({"code": "tool_excluded", "message": "not available"}, "error_semantic"),
    ({"code": "mcp_tool_not_allowed", "message": "denied"}, "error_semantic"),
    ({"code": "role_policy_denied", "message": "denied"}, "error_semantic"),
    ({"code": "tool_not_advertised", "message": "denied"}, "error_semantic"),
    ({"code": "invalid_tool_input_schema", "message": "bad schema"}, "error_semantic"),
    (
      {"code": "planned_write_contract_invalid", "message": "replan"},
      "error_semantic",
    ),
  ],
)
def test_dispatcher_error_codes_map_to_normalized_outcomes(
  error: dict[str, Any],
  expected: str,
) -> None:
  assert classify_tool_outcome(_entry(), None, error) == expected


def test_success_signal_satisfied_classifies_ok() -> None:
  result = {"status": "success", "hits": []}

  assert classify_tool_outcome(_entry(), result, None) == "ok"


def test_declared_success_signal_unmatched_classifies_error_semantic() -> None:
  result = {"status": "partial", "hits": []}

  assert classify_tool_outcome(_entry(), result, None) == "error_semantic"


def test_undeclared_tool_with_non_error_result_classifies_ok() -> None:
  entry = resolve_dispatch_entry("some_unregistered_tool")

  assert entry.catalog_entry is None
  assert classify_tool_outcome(entry, {"anything": 1}, None) == "ok"


def test_result_borne_semantic_error_classifies_error_semantic() -> None:
  result = {"status": "error", "error": {"code": "not_found", "message": "no rows"}}

  assert classify_tool_outcome(_entry(), result, None) == "error_semantic"


def test_success_false_dialect_classifies_error_semantic() -> None:
  assert classify_tool_outcome(_entry(), {"success": False}, None) == "error_semantic"


def test_explicit_semantic_error_argument_is_honored() -> None:
  semantic_error = {"code": "tool_status_error", "sub_code": "rate_limited"}

  assert (
    classify_tool_outcome(_entry(), {"status": "error"}, None, semantic_error)
    == "error_rate_limited"
  )


# --- the exit gate: a fabricated 429 vendor payload ------------------------


VENDOR_429_PAYLOAD = {
  "status": "error",
  "provider": "fmp",
  "error": {
    "code": "rate_limited",
    "message": "HTTP 429 Too Many Requests from api.fmp.test",
  },
}


def test_fabricated_429_vendor_payload_classifies_rate_limited_and_mints_nothing() -> None:
  """The 429-minting hole closes by sequencing, not by another guard."""

  entry = resolve_dispatch_entry("get_quote")
  record = build_dispatch_record(entry=entry, result=VENDOR_429_PAYLOAD, error=None)

  assert record["outcome"] == "error_rate_limited"
  assert record["sources"] == []
  assert extract_source_identities(entry, VENDOR_429_PAYLOAD) == ()
  assert classify_semantic_tool_error(VENDOR_429_PAYLOAD) is not None


def test_fabricated_429_on_a_source_tool_also_mints_nothing() -> None:
  payload = {
    "status": "error",
    "error": {"code": "rate_limited", "message": "HTTP 429 Too Many Requests"},
    "hits": [
      {
        "document_id": "edgar:0000789019-26-000012",
        "ticker": "MSFT",
        "source": "filing",
        "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
      }
    ],
  }
  entry = resolve_dispatch_entry("mcp__research-corpus-mcp__filings_search")

  record = build_dispatch_record(entry=entry, result=payload, error=None)

  assert record["outcome"] == "error_rate_limited"
  assert record["sources"] == []


# --- the dispatch record ---------------------------------------------------


def test_dispatch_record_carries_outcome_attempts_route_and_plural_sources() -> None:
  result = {
    "status": "success",
    "hits": [
      {
        "document_id": "edgar:0000789019-26-000012",
        "ticker": "MSFT",
        "source": "filing",
        "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
      }
    ],
  }
  entry = resolve_dispatch_entry(
    "mcp__research-corpus-mcp__filings_search",
    origin="mcp",
    server="research-corpus-mcp",
    original_tool_name="filings_search",
    provider_id=None,
  )

  record = build_dispatch_record(entry=entry, result=result, error=None, attempts=3)

  assert record == {
    "outcome": "ok",
    "attempts": 3,
    "route_id": "mcp:research-corpus-mcp/mcp__research-corpus-mcp__filings_search",
    "sources": [
      {
        "document_id": "edgar:0000789019-26-000012",
        "source_kind": "filing",
        "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
      }
    ],
  }


def test_dispatch_record_detaches_nested_registered_source_fields() -> None:
  nested = {"factors": {"momentum": "MTUM"}, "peers": ("PAYC", "ADP")}

  record = build_dispatch_record_from_sources(
    entry=None,
    outcome="ok",
    sources=({
      "document_id": "fms:compute_quantifying_risk:ticker=PCTY",
      "source_kind": "computation",
      "key_fields": nested,
    },),
  )

  assert record["sources"][0]["key_fields"] == {
    "factors": {"momentum": "MTUM"},
    "peers": ["PAYC", "ADP"],
  }
  assert record["sources"][0]["key_fields"] is not nested


def test_dispatch_record_sources_stay_empty_unless_the_outcome_is_ok() -> None:
  result = {
    "status": "success",
    "hits": [{"document_id": "edgar:0000789019-26-000012", "source": "filing"}],
  }
  entry = resolve_dispatch_entry(
    "filings_search",
    origin="mcp",
    server="research-corpus-mcp",
    original_tool_name="filings_search",
  )

  ok_record = build_dispatch_record(entry=entry, result=result, error=None)
  failed_record = build_dispatch_record(
    entry=entry,
    result=result,
    error={"code": "internal_error", "message": "boom"},
  )

  assert ok_record["sources"]
  assert failed_record["outcome"] == "error_transport"
  assert failed_record["sources"] == []


def test_dispatch_record_marks_retries_exhausted_when_asked() -> None:
  record = build_dispatch_record(
    entry=resolve_dispatch_entry("filings_search"),
    result=None,
    error={"code": "tool_timeout", "sub_code": "timeout"},
    attempts=3,
    retries_exhausted=True,
  )

  assert record["outcome"] == "error_timeout"
  assert record["attempts"] == 3
  assert record["retries_exhausted"] is True


def test_dispatch_record_accepts_a_route_owned_settled_outcome() -> None:
  record = build_dispatch_record_for_outcome(
    entry=_entry(success_signal={
      "kind": "status_equals",
      "field": "status",
      "values": ["success"],
    }),
    result={"status": "partial"},
    outcome="ok",
  )

  assert record["outcome"] == "ok"


def test_route_id_names_the_route_actually_taken() -> None:
  assert build_route_id(tool_name="file_read") == "local/file_read"
  assert (
    build_route_id(tool_name="get_quote", server="portfolio-mcp", provider_id="fmp")
    == "mcp:portfolio-mcp/provider:fmp/get_quote"
  )


# --- B-2 retry -------------------------------------------------------------


def test_only_transport_timeout_and_rate_limited_are_retryable() -> None:
  assert RETRYABLE_OUTCOMES == {
    "error_transport",
    "error_timeout",
    "error_rate_limited",
  }


@pytest.mark.parametrize(
  "outcome", ["ok", "cancelled", "error_semantic"]
)
def test_non_retryable_outcomes_settle(outcome: str) -> None:
  assert retry_decision(_entry(), outcome, 1) == "settle"


@pytest.mark.parametrize("outcome", sorted(RETRYABLE_OUTCOMES))
def test_reads_retry_by_default(outcome: str) -> None:
  assert retry_decision(_entry(effect="read", idempotent=True), outcome, 1) == "retry"


@pytest.mark.parametrize("effect", ["write", "propose", "external_effect", None])
def test_writes_never_retry(effect: CatalogToolEffect | None) -> None:
  assert retry_decision(_entry(effect=effect), "error_transport", 1) == "settle"


def test_explicitly_non_idempotent_reads_never_retry() -> None:
  assert (
    retry_decision(_entry(effect="read", idempotent=False), "error_transport", 1)
    == "settle"
  )


def test_unknown_idempotence_still_retries_a_read() -> None:
  assert (
    retry_decision(_entry(effect="read", idempotent=None), "error_transport", 1)
    == "retry"
  )


def test_undeclared_tools_never_retry() -> None:
  entry = resolve_dispatch_entry("some_unregistered_tool")

  assert retry_eligible(entry) is False
  assert retry_decision(entry, "error_transport", 1) == "settle"


def test_retries_are_bounded_at_two() -> None:
  entry = _entry()

  assert retry_decision(entry, "error_transport", 1) == "retry"
  assert retry_decision(entry, "error_transport", 2) == "retry"
  assert retry_decision(entry, "error_transport", 3) == "settle"

  for name in ("fred_list_series", "fred_search"):
    unavailable = _entry(name, effect=None, idempotent=True)
    assert unavailable.effect is None
    assert retry_decision(unavailable, "error_transport", 1) == "settle"
  assert DEFAULT_TOOL_RETRY_POLICY.max_retries == 2


def test_approval_gated_calls_never_retry() -> None:
  assert (
    retry_decision(_entry(), "error_transport", 1, needs_approval=True) == "settle"
  )


def test_abort_between_attempts_settles() -> None:
  assert retry_decision(_entry(), "error_transport", 1, aborted=True) == "settle"


def test_wall_clock_exhaustion_settles() -> None:
  assert (
    retry_decision(_entry(), "error_transport", 1, wall_clock_exhausted=True)
    == "settle"
  )


def test_backoff_is_jittered_and_bounded() -> None:
  policy = RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=4.0)

  for attempt in (1, 2, 3, 9):
    delay = retry_backoff_seconds(attempt, policy)
    assert 0.0 <= delay <= 4.0


# --- the declaration table -------------------------------------------------


def test_declaration_table_derives_effect_and_never_restates_it() -> None:
  seen: list[str] = []

  def _resolver(tool_name: str) -> CatalogToolEffect | None:
    seen.append(tool_name)
    return "read"

  table = build_tool_dispatch_declarations(effect_resolver=_resolver)

  assert seen, "every row's effect must be derived, not literal"
  assert set(seen) == set(table)
  assert all(row.effect == "read" for row in table.values())


def test_declaration_table_covers_the_recognized_source_population() -> None:
  table = build_tool_dispatch_declarations(effect_resolver=lambda _name: "read")

  recognized = {
    "code_execute",
    "code_execute_status",
    "web_fetch",
    "filings_search",
    "transcripts_search",
    "filings_list",
    "transcripts_list",
    "filings_read",
    "transcripts_read",
    "filings_source_excerpt",
    "transcripts_source_excerpt",
    "get_filings",
    "get_filing_sections",
    "search_filing_text",
    "get_filing_evidence",
    "cite_concept",
    "get_filing_document",
    "get_metric",
  }

  assert recognized <= set(table)
  assert all(table[name].source_identity is not None for name in recognized)
  for name in ("code_execute", "code_execute_status"):
    assert table[name].source_identity == {"kind": "sandbox_computations"}
    assert table[name].idempotent is False
    assert table[name].success_signal is None
  assert table["gsheets_read_range"].success_signal == {
    "kind": "status_equals",
    "field": "status",
    "values": ("ok",),
  }


def test_sandbox_computation_source_reader_preserves_order_and_current_gates() -> None:
  descriptor = {"kind": "sandbox_computations"}
  load = {
    "function": " load_statements ",
    "output_sha256": f" {'a' * 64} ",
    "tool_version": " sourced-statements-v1 ",
  }
  render = {
    "function": "render_sourced_table",
    "output_sha256": "b" * 64,
    "tool_version": "sourced-tables-v1",
  }

  assert read_source_identities(
    descriptor,
    {
      "return_code": 0,
      "timed_out": False,
      "computations": [load, {"malformed": True}, render],
    },
  ) == (
    {
      "document_id": "sandbox:load_statements:sha=aaaaaaaaaaaa",
      "source_kind": "computation",
    },
    {
      "document_id": "sandbox:render_sourced_table:sha=bbbbbbbbbbbb",
      "source_kind": "computation",
    },
  )

  non_sources = (
    {"status": "running"},
    {"return_code": 1, "timed_out": False, "computations": [load]},
    {"return_code": 0, "timed_out": True, "computations": [load]},
    {
      "return_code": 0,
      "timed_out": False,
      "computations": [{**load, "function": "unknown_helper"}],
    },
    {
      "return_code": 0,
      "timed_out": False,
      "computations": [{**load, "output_sha256": " "}],
    },
    {
      "return_code": 0,
      "timed_out": False,
      "computations": [{**load, "tool_version": " "}],
    },
  )
  assert all(
    read_source_identities(descriptor, result) == ()
    for result in non_sources
  )


def test_fred_declaration_population_marks_only_data_reads_as_vendor_sources() -> None:
  from api.agent.shared.server_policies import MCP_SERVER_POLICIES

  table = build_tool_dispatch_declarations(
    effect_resolver=lambda name: "read" if name.startswith("fred_") else None
  )
  fred_names = {
    "fred_get_multiple",
    "fred_get_series",
    "fred_list_series",
    "fred_search",
  }

  assert {name for name in table if name.startswith("fred_")} == fred_names
  assert MCP_SERVER_POLICIES["fred-mcp"].read_tools == fred_names
  for name in fred_names:
    row = table[name]
    assert row.effect == "read"
    assert row.idempotent is True
    assert row.success_signal is None
    assert row.source_identity is None

  entry = resolve_dispatch_entry(
    "fred_search",
    origin="mcp",
    server="fred-mcp",
    original_tool_name="fred_search",
  )
  assert entry.catalog_entry is not None
  assert entry.catalog_entry.capability == "market-data.read/v1"
  assert classify_tool_outcome(
    entry,
    {"status": "error", "error": {"code": "not_found"}},
    None,
  ) == "error_semantic"
  assert retry_decision(entry, "error_transport", 1) == "retry"
  assert retry_decision(entry, "error_timeout", 1) == "retry"
  assert retry_decision(entry, "error_rate_limited", 1) == "retry"
  assert retry_decision(entry, "error_transport", 3) == "settle"


def test_broker_session_expired_is_outer_retryable_only_for_declared_reads() -> None:
  read_entry = _entry(effect="read", idempotent=True)
  outcome = classify_tool_outcome(
    read_entry,
    None,
    {
      "code": "mcp_tool_error",
      "sub_code": "broker_session_expired",
      "message": "The broker session expired.",
    },
  )

  assert outcome == "error_transport"
  assert retry_decision(read_entry, outcome, 1) == "retry"
  assert retry_decision(_entry(effect="write"), outcome, 1) == "settle"
  assert retry_decision(_entry(effect=None), outcome, 1) == "settle"


def test_declaration_lookup_requires_explicit_exact_route() -> None:
  assert lookup_catalog_entry(
    "mcp__research-corpus-mcp__filings_search",
    origin="mcp",
    server="research-corpus-mcp",
    original_tool_name="filings_search",
  ) is not None
  assert lookup_catalog_entry(
    "filings_search",
    origin=None,
    server=None,
    original_tool_name="filings_search",
  ) is None
