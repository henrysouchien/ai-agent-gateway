import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
GATEWAY_DIR = Path(__file__).resolve().parents[1]
if str(GATEWAY_DIR) not in sys.path:
  sys.path.insert(0, str(GATEWAY_DIR))

from agent_gateway.tool_dispatch_classification import (  # noqa: E402
  RETRYABLE_OUTCOMES,
  DispatchEntry,
  build_dispatch_record,
  build_dispatch_record_for_outcome,
  build_dispatch_record_from_sources,
  build_route_id,
  classify_semantic_tool_error,
  classify_tool_outcome,
  classify_tool_error_type,
  extract_source_identities,
  resolve_dispatch_entry,
  retry_decision,
  retry_eligible,
)
from agent_gateway.capability_resolution import (  # noqa: E402
  canonical_dispatch_tool_name,
  lookup_catalog_entry,
)
from agent_gateway.tool_dispatch_declarations import (  # noqa: E402
  ToolDispatchSemantics,
  build_tool_dispatch_declarations,
)
from agent_gateway.tool_dispatch_source_identity import (  # noqa: E402
  UnknownSourceIdentityDescriptor,
  read_source_identities,
)
from agent_gateway import tool_dispatch_declarations  # noqa: E402
from agent_workflow_contracts import CatalogToolEntry  # noqa: E402
from agent_workflow_contracts.models import CatalogToolEffect  # noqa: E402
from agent_workflow_contracts.tool_registration import (  # noqa: E402
  RegisteredToolIdentity,
  VersionedPolicyRef,
)
from gateway_test_support.host_policy import owner_session_host_policy


_FILINGS_SEARCH = RegisteredToolIdentity(
  route_kind="mcp",
  logical_server_id="research-corpus-mcp",
  logical_name="filings_search",
)


@pytest.fixture(autouse=True)
def filing_route_policy(owner_session_host_policy, monkeypatch):
  owner_session_host_policy.get_server_for_policy_tool = lambda name: (
    "research-corpus-mcp" if name == "filings_search" else None
  )
  owner_session_host_policy.get_local_tool_effect = lambda _name: None
  owner_session_host_policy.get_tool_class = lambda _server, _name: "read"
  # The registration owner declares filings_search's dispatch semantics.
  owner_session_host_policy.get_dispatch_semantics_overrides = lambda: {
    _FILINGS_SEARCH: ToolDispatchSemantics(
      idempotent=True,
      outcome_policy=VersionedPolicyRef(
        kind="outcome",
        policy_id="current-declared-status-equals",
        version="v1",
        parameters={"field": "status", "values": ("success",)},
      ),
      source_identity_policy=VersionedPolicyRef(
        kind="source_identity",
        policy_id="search_hits",
        version="v1",
        parameters={"container": "hits", "default_source_kind": "filing"},
      ),
    ),
  }
  monkeypatch.setattr(tool_dispatch_declarations, "_CACHED_DECLARATIONS", None)


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


def test_a_server_recorded_down_is_a_settled_upstream_failure_never_retried() -> None:
  # The message names "Connection closed", which text classification reads as a
  # transient transport failure; the manager's recovery owns the retry, so the
  # dispatcher must settle this answer instead of waiting on it.
  error = {
    "code": "mcp_tool_error",
    "sub_code": "mcp_server_unavailable",
    "message": "MCP server unavailable: edgar-parser-mcp; could not be reconnected: Connection closed",
  }

  outcome = classify_tool_outcome(_entry(), None, error)

  assert outcome == "error_semantic"
  assert outcome not in RETRYABLE_OUTCOMES
  assert classify_tool_error_type(error) == "upstream_failure"

  # The relayed connect error may itself contain words the text classifier reads as a
  # rate limit; the typed sub_code still settles first.
  relayed = {**error, "message": "MCP server unavailable: x; could not be reconnected: upstream returned HTTP 429 overloaded"}
  assert classify_tool_outcome(_entry(), None, relayed) == "error_semantic"


@pytest.mark.parametrize(
  "error",
  [
    {"code": "tool_error", "sub_code": "connection_error", "message": "MCPError: Connection closed"},
    {"code": "mcp_tool_error", "message": "MCP server unavailable: edgar-parser-mcp; Connection closed"},
  ],
  ids=["plain-transport-error", "same-words-without-the-typed-sub-code"],
)
def test_only_the_typed_down_server_code_stops_a_connection_closed_retry(
  error: dict[str, Any],
) -> None:
  outcome = classify_tool_outcome(_entry(), None, error)

  assert outcome == "error_transport"
  assert outcome in RETRYABLE_OUTCOMES


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


def test_tools_with_no_declared_effect_never_retry() -> None:
  for name in ("fred_list_series", "fred_search"):
    unavailable = _entry(name, effect=None, idempotent=True)
    assert unavailable.effect is None
    assert retry_decision(unavailable, "error_transport", 1) == "settle"


def test_approval_gated_calls_never_retry() -> None:
  assert (
    retry_decision(_entry(), "error_transport", 1, needs_approval=True) == "settle"
  )


def test_abort_between_attempts_settles() -> None:
  assert retry_decision(_entry(), "error_transport", 1, aborted=True) == "settle"


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


def test_statement_citation_reader_preserves_row_and_period_cell_order() -> None:
  row_citation = {
    "document_id": "edgar:0000320193-24-000001",
    "source_url": "https://www.sec.gov/Archives/row.htm",
  }
  current_citation = {
    "document_id": "edgar:0000320193-24-000002",
    "source_url": "https://www.sec.gov/Archives/current.htm",
  }
  prior_citation = {
    "document_id": "edgar:0000320193-23-000001",
    "source_url": "https://www.sec.gov/Archives/prior.htm",
  }
  next_row_citation = {
    "document_id": "edgar:0000320193-23-000002",
    "source_url": "https://www.sec.gov/Archives/next.htm",
  }
  payload = {
    "rows": [
      {
        "citation": row_citation,
        "values": {
          "2024": {"value": 120, "citation": current_citation},
          "2023": {"value": 100, "citation": prior_citation},
          "2022": {"value": 90},
        },
      },
      {"citation": next_row_citation},
      {"values": {"2024": {"value": 42}}},
      None,
    ],
  }

  assert read_source_identities({"kind": "statement_citations"}, payload) == tuple(
    {"source_kind": "xbrl_fact", **citation}
    for citation in (
      row_citation, current_citation, prior_citation, next_row_citation
    )
  )


def test_financials_citation_reader_preserves_current_and_prior_identities() -> None:
  current_citation = {
    "document_id": "edgar:0000320193-24-000001",
    "source_url": "https://www.sec.gov/Archives/current.htm",
  }
  prior_citation = {
    "document_id": "edgar:0000320193-23-000001",
    "source_url": "https://www.sec.gov/Archives/prior.htm",
  }
  single_citation = {
    "document_id": "edgar:0000320193-22-000001",
    "source_url": "https://www.sec.gov/Archives/single.htm",
  }
  payload = {
    "facts": [
      {
        "citation": current_citation,
        "citations": {"current": current_citation, "prior": prior_citation},
      },
      {"citation": single_citation},
      {"value": 42},
      None,
    ],
  }

  assert read_source_identities({"kind": "financials_citations"}, payload) == tuple(
    {"source_kind": "xbrl_fact", **citation}
    for citation in (current_citation, prior_citation, single_citation)
  )


@pytest.mark.parametrize(
  ("kind", "payload"),
  [
    ("statement_citations", {"rows": [{"values": {"2024": {"value": 42}}}]}),
    ("financials_citations", {"facts": [{"value": 42}]}),
  ],
)
def test_statement_and_financials_readers_ignore_uncited_items(
  kind: str, payload: dict[str, Any],
) -> None:
  assert read_source_identities({"kind": kind}, payload) == ()


@pytest.mark.parametrize(
  ("kind", "container"),
  [
    ("statement_citations", "rows"),
    ("financials_citations", "facts"),
  ],
)
@pytest.mark.parametrize("items", [None, {}, "not-a-list", (), []])
def test_statement_and_financials_readers_require_list_containers(
  kind: str, container: str, items: Any,
) -> None:
  assert read_source_identities({"kind": kind}, {container: items}) == ()


def test_source_identity_reader_rejects_unknown_kind() -> None:
  with pytest.raises(UnknownSourceIdentityDescriptor, match="unknown source_identity"):
    read_source_identities({"kind": "unknown_citations"}, {"facts": []})




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


# The declared-type table, the payload key and the payload location are independent
# lookups: every declared type is read once (type key, dispatcher error), and every
# key/location pair once with one declared type that the payload's code contradicts.
@pytest.mark.parametrize("declared, expected, key, location", [
  ("bad_input", "bad_input", "type", "error"),
  ("not_found", "integrity_refusal", "type", "error"),
  ("integrity_refusal", "integrity_refusal", "type", "error"),
  ("confirmation_required", "confirmation_required", "type", "error"),
  ("needs_decision", "needs_decision", "type", "error"),
  ("upstream_failure", "upstream_failure", "type", "error"),
  ("timeout", "timeout", "type", "error"),
  ("interrupted", "interrupted", "type", "error"),
  ("operational_failure", "operational_failure", "type", "error"),
  *(("upstream_failure", "upstream_failure", key, location)
    for key in ("type", "error_type")
    for location in ("error", "data", "nested", "result", "semantic")
    if (key, location) != ("type", "error")),
])
def test_owner_error_type_wins(declared, expected, key, location):
  owner = {key: declared, "code": "invalid_input"}
  error, result, semantic = None, None, None
  if location == "error":
    error = owner
  elif location == "data":
    error = {"code": "timeout", "data": owner}
  elif location == "nested":
    result = {"status": "error", "error": owner}
  elif location == "result":
    result = {"status": "error", **owner}
  else:
    semantic = owner
  assert classify_tool_error_type(error, result, semantic) == expected


_ERROR_CODE_FAMILIES = [
  ("tool_timeout", "timeout"), ("provider_timeout_error", "timeout"),
  ("cancelled", "interrupted"), ("interrupted", "interrupted"),
  ("invalid_period", "bad_input"), ("validation_error", "bad_input"),
  ("bad_input", "bad_input"), ("workflow_invalid_input", "bad_input"),
  ("not_found", "integrity_refusal"), ("permission_denied", "integrity_refusal"),
  ("unknown_tool", "integrity_refusal"), ("artifact_not_found", "integrity_refusal"),
  ("spill_unavailable", "bad_input"), ("spill_access_denied", "integrity_refusal"),
  ("role_policy_denied", "confirmation_required"), ("approval_denied", "confirmation_required"),
  ("user_denied", "confirmation_required"), ("needs_decision", "needs_decision"),
  ("transport_error", "upstream_failure"), ("rate_limited", "upstream_failure"),
  ("unclassified", "operational_failure"),
]
# `code` and `sub_code` are read through the same list in the owner: every code is
# pinned on `code`, and `sub_code` carries one code of each family.
_SUB_CODE_REPRESENTATIVES = frozenset({
  "tool_timeout", "cancelled", "invalid_period", "not_found", "role_policy_denied",
  "needs_decision", "transport_error", "rate_limited", "unclassified",
})


@pytest.mark.parametrize("code, expected, key", [
  *((code, expected, "code") for code, expected in _ERROR_CODE_FAMILIES),
  *((code, expected, "sub_code") for code, expected in _ERROR_CODE_FAMILIES
    if code in _SUB_CODE_REPRESENTATIVES),
])
def test_error_code_family(code, expected, key):
  assert classify_tool_error_type({key: code.upper()}) == expected


def test_fastmcp_validation_message_and_type_precedence():
  assert classify_tool_error_type({"code": "mcp_tool_error", "message": "1 validation error"}) == "bad_input"
  assert classify_tool_error_type({"type": "upstream_failure", "error_type": "bad_input"}) == "upstream_failure"
  assert classify_tool_error_type(None) == "operational_failure"
  assert classify_tool_error_type(None, {"status": "error", "error": {"data": {"code": "invalid_period"}}}) == "bad_input"


def test_producer_validation_error_flag_is_a_caller_refusal():
  refusal = {
    "status": "error",
    "error": "unverified_citation: register_sources did not register S1",
    "validation_error": True,
    "error_type": "fms_invalid_judgment",
  }
  semantic = {"code": "tool_status_error", "sub_code": "fms_invalid_judgment"}
  assert classify_tool_error_type(None, refusal, semantic) == "bad_input"
  # Without the producer's flag the same unknown code stays an operational failure.
  assert classify_tool_error_type(
    None, {key: value for key, value in refusal.items() if key != "validation_error"}, semantic,
  ) == "operational_failure"
  # Declared families and transport/timeout settlements still win over the flag.
  assert classify_tool_error_type(None, {**refusal, "error_type": "upstream_failure"}, semantic) == "upstream_failure"
  assert classify_tool_error_type(None, {**refusal, "code": "tool_timeout"}, semantic) == "timeout"
