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
from agent_workflow_contracts import CatalogToolEntry  # noqa: E402


def _entry(
  tool_name: str = "filings_search",
  *,
  effect: str | None = "read",
  idempotent: bool | None = True,
  success_signal: dict[str, Any] | None = None,
  source_identity: dict[str, Any] | None = None,
) -> DispatchEntry:
  described = lookup_catalog_entry(tool_name)
  canonical = canonical_dispatch_tool_name(tool_name)
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
    server="research-corpus-mcp",
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


def test_dispatch_record_sources_stay_empty_unless_the_outcome_is_ok() -> None:
  result = {
    "status": "success",
    "hits": [{"document_id": "edgar:0000789019-26-000012", "source": "filing"}],
  }
  entry = resolve_dispatch_entry("filings_search")

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
def test_writes_never_retry(effect: str | None) -> None:
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

  def _resolver(tool_name: str) -> str | None:
    seen.append(tool_name)
    return "read"

  table = build_tool_dispatch_declarations(effect_resolver=_resolver)

  assert seen, "every row's effect must be derived, not literal"
  assert set(seen) == set(table)
  assert all(row.effect == "read" for row in table.values())


def test_declaration_table_covers_the_recognized_source_population() -> None:
  table = build_tool_dispatch_declarations(effect_resolver=lambda _name: "read")

  recognized = {
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
  assert table["gsheets_read_range"].success_signal == {
    "kind": "status_equals",
    "field": "status",
    "values": ("ok",),
  }


def test_declaration_lookup_canonicalizes_namespaced_tool_names() -> None:
  assert (
    lookup_catalog_entry("mcp__research-corpus-mcp__filings_search")
    == lookup_catalog_entry("filings_search")
  )
