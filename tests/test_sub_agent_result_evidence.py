from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import pytest

from agent_gateway.sub_agent_result_evidence import (
  SubAgentResultEvidence,
  collect_sub_agent_result_evidence,
  fold_dispatch_failures,
  fold_observed_sources,
  merge_sub_agent_result_evidence,
  merge_usage_payloads,
)


@dataclass(frozen=True)
class _Entry:
  event: dict[str, Any]


def _entry(event: dict[str, Any]) -> _Entry:
  return _Entry(event=event)


def test_collect_live_evidence_sums_turn_usage_without_stream_complete() -> None:
  evidence = collect_sub_agent_result_evidence(
    [
      _entry({
        "type": "turn_complete",
        "usage": {
          "input_tokens": 7,
          "provider_unit_deltas": {"web_search": 1},
          "estimated_cost": 0.125,
        },
      }),
      _entry({"type": "tool_call_start", "tool_name": "docs_search"}),
      _entry({"type": "stream_retry"}),
      _entry({
        "type": "turn_complete",
        "usage": {
          "input_tokens": 3,
          "output_tokens": 5,
          "provider_unit_deltas": {
            "web_search": 2,
            "web_fetch": 1,
          },
          "estimated_cost": 0.25,
        },
      }),
    ],
    durable=False,
  )

  assert evidence.usage == {
    "input_tokens": 10,
    "provider_unit_deltas": {
      "web_search": 3,
      "web_fetch": 1,
    },
    "estimated_cost": 0.375,
    "output_tokens": 5,
  }
  assert evidence.tools_used == ("docs_search",)


def test_stream_complete_is_authoritative_for_current_segment() -> None:
  evidence = collect_sub_agent_result_evidence(
    [
      _entry({
        "type": "turn_complete",
        "usage": {"input_tokens": 3, "estimated_cost": 0.1},
      }),
      _entry({
        "type": "stream_complete",
        "usage": {
          "input_tokens": 10,
          "output_tokens": 4,
          "estimated_cost": 0.3,
        },
      }),
    ],
    durable=False,
  )

  assert evidence.usage == {
    "input_tokens": 10,
    "output_tokens": 4,
    "estimated_cost": 0.3,
  }


def test_durable_evidence_uses_assistant_and_guarded_draft_usage() -> None:
  evidence = collect_sub_agent_result_evidence(
    [
      _entry({
        "type": "turn_complete",
        "usage": {"input_tokens": 999},
      }),
      _entry({
        "type": "assistant_message",
        "usage": {"input_tokens": 4, "estimated_cost": 0.1},
      }),
      _entry({
        "type": "runtime_guard",
        "guard": "final_answer",
        "draft_usage": {"output_tokens": 6, "estimated_cost": 0.2},
      }),
      _entry({
        "type": "interrupted",
        "message": "process restart",
      }),
    ],
    durable=True,
  )

  assert evidence.usage == {
    "input_tokens": 4,
    "estimated_cost": 0.30000000000000004,
    "output_tokens": 6,
  }
  assert evidence.warning_parts == (
    "Prior child run was interrupted: process restart",
  )


def test_merge_evidence_preserves_lineage_order_and_adds_usage() -> None:
  first = SubAgentResultEvidence(
    usage={
      "input_tokens": 2,
      "provider_unit_deltas": {"web_search": 1},
    },
    tools_used=("first_tool",),
    fms_results=({"status": "staged"},),
    artifact_events=({"type": "artifact_ready", "artifact_ref": "a"},),
    warning_parts=("first warning",),
  )
  second = SubAgentResultEvidence(
    usage={
      "input_tokens": 3,
      "estimated_cost": 0.5,
      "provider_unit_deltas": {"web_search": 2},
    },
    tools_used=("second_tool",),
    fms_results=({"status": "applied"},),
    artifact_events=({"type": "artifact_ready", "artifact_ref": "b"},),
    warning_parts=("second warning",),
  )

  merged = merge_sub_agent_result_evidence(first, second)

  assert merged.usage == {
    "input_tokens": 5,
    "provider_unit_deltas": {"web_search": 3},
    "estimated_cost": 0.5,
  }
  assert merged.tools_used == ("first_tool", "second_tool")
  assert merged.fms_results == (
    {"status": "staged"},
    {"status": "applied"},
  )
  assert merged.artifact_events == (
    {"type": "artifact_ready", "artifact_ref": "a"},
    {"type": "artifact_ready", "artifact_ref": "b"},
  )
  assert merged.warning_parts == ("first warning", "second warning")


def test_merge_usage_retains_newest_non_numeric_metadata() -> None:
  assert merge_usage_payloads(
    {"input_tokens": 1, "billing_mode": "estimated"},
    {"input_tokens": 2, "billing_mode": "actual"},
  ) == {
    "input_tokens": 3,
    "billing_mode": "actual",
  }


@pytest.mark.parametrize(
  "usage_key",
  ["counter", "provider_unit_deltas"],
)
def test_collect_live_usage_arithmetic_overflow_fails_closed(
  usage_key: str,
) -> None:
  huge_integer = 10**4000
  first_value: Any = huge_integer
  second_value: Any = 1.0
  if usage_key == "provider_unit_deltas":
    first_value = {"operation": huge_integer}
    second_value = {"operation": 1.0}

  evidence = collect_sub_agent_result_evidence(
    [
      _entry({
        "type": "turn_complete",
        "usage": {usage_key: first_value},
      }),
      _entry({
        "type": "turn_complete",
        "usage": {usage_key: second_value},
      }),
    ],
    durable=False,
  )

  assert evidence.admission_rejected is True
  assert evidence.usage == {}
  assert evidence.fms_results == ()


def test_collect_durable_usage_arithmetic_overflow_fails_closed() -> None:
  evidence = collect_sub_agent_result_evidence(
    [
      _entry({
        "type": "assistant_message",
        "usage": {"counter": 10**4000},
      }),
      _entry({
        "type": "runtime_guard",
        "draft_usage": {"counter": 1.0},
      }),
    ],
    durable=True,
  )

  assert evidence.admission_rejected is True
  assert evidence.usage == {}


def test_merge_preflights_exact_canonical_usage_after_metadata_overwrite() -> None:
  stale_trace = [None] * 60_000
  current_trace = [None] * 60_000
  first = SubAgentResultEvidence(
    usage={"trace": stale_trace, "marker": "stale"},
    tools_used=(),
    fms_results=(),
    artifact_events=(),
    warning_parts=(),
  )
  second = SubAgentResultEvidence(
    usage={"trace": current_trace, "marker": "current"},
    tools_used=(),
    fms_results=(),
    artifact_events=(),
    warning_parts=(),
  )

  merged = merge_sub_agent_result_evidence(first, second)

  assert merged.admission_rejected is False
  assert merged.usage["trace"] is current_trace
  assert merged.usage["marker"] == "current"


def test_merge_rejects_aggregate_evidence_that_only_overflows_together() -> None:
  first = SubAgentResultEvidence(
    usage={},
    tools_used=(),
    fms_results=({"wide": [None] * 60_000},),
    artifact_events=(),
    warning_parts=("first",),
  )
  second = SubAgentResultEvidence(
    usage={},
    tools_used=(),
    fms_results=({"wide": [None] * 60_000},),
    artifact_events=(),
    warning_parts=("second",),
  )

  merged = merge_sub_agent_result_evidence(first, second)

  assert merged.admission_rejected is True
  assert merged.fms_results == ()
  assert merged.warning_parts[:2] == ("first", "second")


def test_merge_rejects_aggregate_evidence_over_byte_ceiling() -> None:
  first = SubAgentResultEvidence(
    usage={},
    tools_used=(),
    fms_results=({"blob": "x" * (35 * 1024 * 1024)},),
    artifact_events=(),
    warning_parts=(),
  )
  second = SubAgentResultEvidence(
    usage={},
    tools_used=(),
    fms_results=({"blob": "y" * (35 * 1024 * 1024)},),
    artifact_events=(),
    warning_parts=(),
  )

  merged = merge_sub_agent_result_evidence(first, second)

  assert merged.admission_rejected is True
  assert merged.fms_results == ()


def test_merge_rejects_integer_sum_that_crosses_digit_ceiling() -> None:
  digit_limit = sys.get_int_max_str_digits()
  if digit_limit == 0:
    pytest.skip("runtime integer string conversion limit is disabled")
  addend = 5 * (10 ** (digit_limit - 1))
  first = SubAgentResultEvidence(
    usage={"counter": addend},
    tools_used=(),
    fms_results=(),
    artifact_events=(),
    warning_parts=(),
  )
  second = SubAgentResultEvidence(
    usage={"counter": addend},
    tools_used=(),
    fms_results=(),
    artifact_events=(),
    warning_parts=(),
  )

  merged = merge_sub_agent_result_evidence(first, second)

  assert merged.admission_rejected is True
  assert merged.usage == {}


def _tool_call_complete_with_blocks(
  tool_name: str,
  blocks: list[dict[str, Any]],
) -> _Entry:
  return _entry({
    "type": "tool_call_complete",
    "tool_name": tool_name,
    "final_tool_result_blocks": [
      {"type": "tool_result", "tool_use_id": "tool_1", "content": "{}"},
      *blocks,
    ],
  })


def test_collect_captures_observed_sources_from_source_envelope() -> None:
  evidence = collect_sub_agent_result_evidence(
    [
      _entry({"type": "tool_call_start", "tool_name": "filings_search"}),
      _tool_call_complete_with_blocks(
        "filings_search",
        [
          {
            "type": "source_envelope",
            "_event_only": True,
            "schema_version": 1,
            "tool_name": "filings_search",
            "sources_for_call": [
              {
                "index": 1,
                "document_id": "edgar:0000789019-26-000012",
                "source_kind": "filing",
                "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
                "produced_by_tool": "filings_search",
              },
              {"index": 2, "document_id": "", "source_kind": "filing"},
            ],
            "excerpt_handles_for_call": [
              {
                "handle_id": "h_0123456789ab",
                "document_id": "edgar:0000789019-26-000012",
                "source_kind": "filing",
              },
              {
                "handle_id": "h_ba9876543210",
                "document_id": "web:deadbeef",
                "handle_class": "web",
              },
            ],
          },
        ],
      ),
    ],
    durable=False,
  )

  assert evidence.tools_used == ("filings_search",)
  assert evidence.observed_sources == (
    {
      "kind": "observed_source",
      "source_kind": "filing",
      "document_id": "edgar:0000789019-26-000012",
      "produced_by_tool": "filings_search",
      "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
      "excerpt_handle_id": "h_0123456789ab",
    },
    {
      "kind": "observed_source",
      "source_kind": "web",
      "document_id": "web:deadbeef",
      "produced_by_tool": "filings_search",
      "excerpt_handle_id": "h_ba9876543210",
    },
  )


def test_collect_captures_suppressed_context_source_observations() -> None:
  evidence = collect_sub_agent_result_evidence(
    [
      _tool_call_complete_with_blocks(
        "filings_search",
        [
          {
            "type": "source_observation",
            "_event_only": True,
            "schema_version": 1,
            "tool_name": "filings_search",
            "observed_sources": [
              {
                "document_id": "edgar:0000789019-26-000012",
                "source_kind": "filing",
                "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
              },
              {"document_id": "", "source_kind": "filing"},
            ],
          },
        ],
      ),
    ],
    durable=False,
  )

  assert evidence.observed_sources == (
    {
      "kind": "observed_source",
      "source_kind": "filing",
      "document_id": "edgar:0000789019-26-000012",
      "produced_by_tool": "filings_search",
      "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
    },
  )
  # Observation-only records never carry a citation identity.
  assert all(
    "excerpt_handle_id" not in record
    for record in evidence.observed_sources
  )


def test_merge_dedups_observed_sources_across_segments() -> None:
  record = {
    "kind": "observed_source",
    "source_kind": "filing",
    "document_id": "edgar:0000789019-26-000012",
    "produced_by_tool": "filings_search",
  }
  other = {
    "kind": "observed_source",
    "source_kind": "web",
    "document_id": "web:deadbeef",
    "produced_by_tool": "web_fetch",
  }
  first = SubAgentResultEvidence(
    usage={},
    tools_used=(),
    fms_results=(),
    artifact_events=(),
    warning_parts=(),
    observed_sources=(record,),
  )
  second = SubAgentResultEvidence(
    usage={},
    tools_used=(),
    fms_results=(),
    artifact_events=(),
    warning_parts=(),
    observed_sources=(dict(record), other),
  )

  merged = merge_sub_agent_result_evidence(first, second)

  assert merged.observed_sources == (record, other)


# --- B-1: the dispatch fold ------------------------------------------------


def _dispatch_event(
  tool_name: str,
  sources: list[dict[str, Any]],
  *,
  outcome: str = "ok",
  blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
  event: dict[str, Any] = {
    "type": "tool_call_complete",
    "tool_name": tool_name,
    "is_error": outcome != "ok",
    "dispatch": {
      "outcome": outcome,
      "attempts": 1,
      "route_id": f"local/{tool_name}",
      "sources": sources,
    },
  }
  if blocks is not None:
    event["final_tool_result_blocks"] = blocks
  return event


def test_fold_reads_observed_sources_from_the_dispatch_record() -> None:
  folded = fold_observed_sources(
    [
      _dispatch_event(
        "filings_search",
        [
          {
            "document_id": "edgar:0000789019-26-000012",
            "source_kind": "filing",
            "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
          },
          {"document_id": "", "source_kind": "filing"},
        ],
      )
    ]
  )

  assert folded == (
    {
      "kind": "observed_source",
      "source_kind": "filing",
      "document_id": "edgar:0000789019-26-000012",
      "produced_by_tool": "filings_search",
      "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
    },
  )


def test_fold_falls_back_to_blocks_for_pre_train_events() -> None:
  legacy_event = {
    "type": "tool_call_complete",
    "tool_name": "filings_search",
    "final_tool_result_blocks": [
      {
        "type": "source_observation",
        "tool_name": "filings_search",
        "observed_sources": [
          {"document_id": "edgar:0000789019-26-000012", "source_kind": "filing"},
        ],
      },
    ],
  }

  assert "dispatch" not in legacy_event
  assert fold_observed_sources([legacy_event]) == (
    {
      "kind": "observed_source",
      "source_kind": "filing",
      "document_id": "edgar:0000789019-26-000012",
      "produced_by_tool": "filings_search",
    },
  )


def test_dispatch_wins_when_both_the_record_and_the_blocks_exist() -> None:
  folded = fold_observed_sources(
    [
      _dispatch_event(
        "filings_search",
        [{"document_id": "edgar:0000789019-26-000012", "source_kind": "filing"}],
        blocks=[
          {
            "type": "source_observation",
            "tool_name": "filings_search",
            "observed_sources": [
              {"document_id": "edgar:9999999999-99-999999", "source_kind": "filing"},
            ],
          },
        ],
      )
    ]
  )

  assert folded == (
    {
      "kind": "observed_source",
      "source_kind": "filing",
      "document_id": "edgar:0000789019-26-000012",
      "produced_by_tool": "filings_search",
    },
  )


def test_fold_falls_back_to_blocks_when_the_dispatch_record_settles_no_sources() -> None:
  """Vendor and computation citations still mint on the api-side ledger path."""

  folded = fold_observed_sources(
    [
      _dispatch_event(
        "get_quote",
        [],
        blocks=[
          {
            "type": "source_envelope",
            "tool_name": "get_quote",
            "sources_for_call": [
              {
                "document_id": "fmp:quote:MSFT",
                "source_kind": "vendor",
                "produced_by_tool": "get_quote",
              },
            ],
          },
        ],
      )
    ]
  )

  assert folded == (
    {
      "kind": "observed_source",
      "source_kind": "vendor",
      "document_id": "fmp:quote:MSFT",
      "produced_by_tool": "get_quote",
    },
  )


@pytest.mark.parametrize(
  "outcome",
  [
    "error_rate_limited",
    "error_timeout",
    "error_transport",
    "error_semantic",
    "cancelled",
  ],
)
def test_a_failed_dispatch_contributes_nothing_even_when_blocks_name_sources(
  outcome: str,
) -> None:
  """The 429 shape: a failed call observed nothing, blocks included (T3-I06)."""

  folded = fold_observed_sources(
    [
      _dispatch_event(
        "get_quote",
        [],
        outcome=outcome,
        blocks=[
          {
            "type": "source_envelope",
            "tool_name": "get_quote",
            "sources_for_call": [
              {
                "document_id": "fmp:quote:AAPL",
                "source_kind": "vendor",
                "produced_by_tool": "get_quote",
              },
            ],
          },
          {
            "type": "source_observation",
            "tool_name": "get_quote",
            "observed_sources": [
              {"document_id": "fmp:quote:AAPL", "source_kind": "vendor"},
            ],
          },
        ],
      )
    ]
  )

  assert folded == ()


def test_a_dispatch_record_with_no_outcome_still_reads_the_legacy_blocks() -> None:
  """A degenerate record is not a settled failure; keep the old reading."""

  folded = fold_observed_sources(
    [
      {
        "type": "tool_call_complete",
        "tool_name": "filings_search",
        "dispatch": {"attempts": 1, "sources": []},
        "final_tool_result_blocks": [
          {
            "type": "source_observation",
            "tool_name": "filings_search",
            "observed_sources": [
              {"document_id": "edgar:0000789019-26-000012", "source_kind": "filing"},
            ],
          },
        ],
      }
    ]
  )

  assert folded == (
    {
      "kind": "observed_source",
      "source_kind": "filing",
      "document_id": "edgar:0000789019-26-000012",
      "produced_by_tool": "filings_search",
    },
  )


def test_collect_uses_the_dispatch_fold_for_observed_sources() -> None:
  evidence = collect_sub_agent_result_evidence(
    [
      _entry({"type": "tool_call_start", "tool_name": "filings_search"}),
      _entry(
        _dispatch_event(
          "filings_search",
          [{"document_id": "edgar:0000789019-26-000012", "source_kind": "filing"}],
        )
      ),
    ],
    durable=False,
  )

  assert evidence.tools_used == ("filings_search",)
  assert evidence.observed_sources == (
    {
      "kind": "observed_source",
      "source_kind": "filing",
      "document_id": "edgar:0000789019-26-000012",
      "produced_by_tool": "filings_search",
    },
  )


def test_fold_dispatch_failures_counts_outcomes_per_tool() -> None:
  counts = fold_dispatch_failures(
    [
      _dispatch_event("filings_search", [], outcome="ok"),
      _dispatch_event("filings_search", [], outcome="error_rate_limited"),
      _dispatch_event("web_fetch", [], outcome="error_transport"),
      {"type": "tool_call_start", "tool_name": "filings_search"},
    ]
  )

  assert counts == {"filings_search": (1, 1), "web_fetch": (0, 1)}


def test_fold_dispatch_failures_falls_back_to_is_error_for_old_logs() -> None:
  counts = fold_dispatch_failures(
    [
      {"type": "tool_call_complete", "tool_name": "filings_search", "is_error": False},
      {"type": "tool_call_complete", "tool_name": "filings_search", "is_error": True},
    ]
  )

  assert counts == {"filings_search": (1, 1)}


def test_tools_used_still_comes_from_tool_call_start_so_failed_calls_count() -> None:
  evidence = collect_sub_agent_result_evidence(
    [
      _entry({"type": "tool_call_start", "tool_name": "filings_search"}),
      _entry(
        _dispatch_event("filings_search", [], outcome="error_rate_limited")
      ),
    ],
    durable=False,
  )

  assert evidence.tools_used == ("filings_search",)
  assert evidence.observed_sources == ()
