from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway import AgentRunner, EventLog
from agent_gateway.event_adapter import adapt_control_event, adapt_event
from agent_gateway.tool_display import DETAIL_MAX_CHARS, resolve_display
from agent_gateway.tool_dispatch_classification import (
  ToolResultSettlement,
  settle_catalogless_tool_result,
)
from agent_gateway.tool_policy_registry import PreparedToolCall
from gateway_test_support.capability_execution_test_support import (
  stub_runner_capability_execution,
)




def test_resolve_display_seed_map_renders_product_label_and_detail() -> None:
  display = resolve_display(
    "mcp__edgar-parser-mcp__get_metric",
    {
      "ticker": "msft",
      "metric_name": "revenue",
      "quarter": 3,
      "year": 2026,
      "role": "income statement",
    },
  )

  assert display == {
    "label": "Pulling MSFT revenue",
    "detail": "Q3 2026 - income statement",
  }


@pytest.mark.parametrize(
  ("tool_name", "label", "detail"),
  [
    ("write_cells", "Writing cells", "PCTY Comps - C4:C6"),
    ("read_cells", "Reading cells", "PCTY Comps - C4:C6"),
    ("list_sheets", "Listing workbook sheets", None),
    ("get_selection", "Reading the current selection", "PCTY Comps"),
    ("get_used_range", "Reading used range", "PCTY Comps"),
  ],
)
@pytest.mark.parametrize("prefix", ["", "mcp__excel__"])
def test_excel_display_omits_workbook_routing_session(
  tool_name: str, label: str, detail: str | None, prefix: str,
) -> None:
  tool_input = {
    "_workbook": "a77b9ec6-b2ab-4d63-9245-5dced32baf78",
    "sheet_name": "PCTY Comps",
    "range": "C4:C6",
  }
  expected = {"label": label}
  if detail is not None:
    expected["detail"] = detail

  assert resolve_display(prefix + tool_name, tool_input) == expected
  assert tool_input["_workbook"] == "a77b9ec6-b2ab-4d63-9245-5dced32baf78"


@pytest.mark.parametrize("workbook_key", ["workbook", "workbook_name"])
def test_excel_display_preserves_human_workbook_names(workbook_key: str) -> None:
  display = resolve_display("write_cells", {
    "_workbook": "a77b9ec6-b2ab-4d63-9245-5dced32baf78",
    workbook_key: "Coverage Model.xlsx",
    "sheet": "PCTY Comps",
    "range": "C4:C6",
  })

  assert display == {
    "label": "Writing cells",
    "detail": "Coverage Model.xlsx - PCTY Comps - C4:C6",
  }


@pytest.mark.parametrize(
  "tool_name",
  ["write_cells", "read_cells", "list_sheets", "get_selection", "get_used_range"],
)
def test_excel_display_omits_detail_with_only_routing_session(tool_name: str) -> None:
  display = resolve_display(tool_name, {
    "_workbook": "a77b9ec6-b2ab-4d63-9245-5dced32baf78",
  })

  assert display is not None
  assert "detail" not in display


def test_excel_display_prefers_workbook_name_over_injected_routing_session() -> None:
  assert resolve_display("list_sheets", {
    "_workbook": "a77b9ec6-b2ab-4d63-9245-5dced32baf78",
    "workbook": "",
    "workbook_name": "Coverage Model.xlsx",
  }) == {"label": "Listing workbook sheets", "detail": "Coverage Model.xlsx"}


@pytest.mark.parametrize(
  ("tool_name", "label"),
  [
    ("fms_report_sniff_test", "Writing sniff-test report"),
    ("fms_report_capital_allocation", "Writing capital-allocation report"),
    ("fms_propose_capital_allocation", "Staging capital-allocation proposal"),
    ("fms_persist_capital_allocation", "Saving capital-allocation"),
    ("fms_report_business_quality_assessment", "Writing business-quality report"),
    ("fms_propose_fundamental_research", "Drafting fundamental research"),
    ("fms_persist_business_model", "Saving business model work"),
  ],
)
@pytest.mark.parametrize("prefix", ["", "mcp__research__"])
def test_fms_display_names_research_steps(
  tool_name: str, label: str, prefix: str,
) -> None:
  assert resolve_display(prefix + tool_name, {"ticker": "pcty"}) == {
    "label": label, "detail": "PCTY",
  }


@pytest.mark.parametrize(
  ("tool_name", "label"),
  [
    ("docs_fetch", "Reading documentation"),
    ("load_tools", "Preparing research tools"),
    ("list_skills", "Finding research workflows"),
    ("transcripts_read", "Reading earnings transcripts"),
    ("transcripts_search", "Searching earnings transcripts"),
    ("transcripts_list", "Finding earnings transcripts"),
    ("filings_read", "Reading filing excerpts"),
    ("filings_search", "Searching filings"),
    ("filings_list", "Finding filings"),
    ("get_quote", "Pulling a market quote"),
    ("memory_store", "Saving memory"),
    ("memory_recall", "Recalling memory"),
    ("invoke_skill", "Running sniff test"),
  ],
)
@pytest.mark.parametrize("prefix", ["", "mcp__research__"])
def test_common_tool_display_uses_reader_facing_actions(
  tool_name: str, label: str, prefix: str,
) -> None:
  display = resolve_display(prefix + tool_name, {"skill": "sniff-test"})

  assert display == {"label": label}


@pytest.mark.parametrize("skill_key", ["skill", "skill_name", "workflow_name"])
def test_skill_display_renders_workflow_words(skill_key: str) -> None:
  assert resolve_display("invoke_skill", {
    skill_key: "business-quality-assessment", "ticker": "PCTY",
  }) == {"label": "Running business quality assessment", "detail": "PCTY"}


@pytest.mark.parametrize(
  ("concept", "label"),
  [
    ("WeightedAverageNumberOfDilutedSharesOutstanding", "weighted average diluted shares"),
    ("PaymentsForRepurchaseOfCommonStock", "payments for repurchase of common stock"),
    ("OperatingIncomeLoss", "operating income (loss)"),
    ("GrossProfit", "gross profit"),
    ("CostOfRevenue", "cost of revenue"),
    ("EarningsPerShareDiluted", "diluted EPS"),
    ("RevenueFromContractWithCustomerExcludingAssessedTax",
     "revenue from contract with customer excluding assessed tax"),
    ("EBITDAMargin", "ebitda margin"),
    ("ShareBasedCompensation", "share based compensation"),
    ("gross_profit", "gross profit"),
    ("gross-profit", "gross profit"),
    ("revenue", "revenue"),
  ],
)
@pytest.mark.parametrize("namespace", ["", "us-gaap:"])
@pytest.mark.parametrize(
  ("tool_name", "suffix"), [("get_metric", ""), ("get_metric_series", " history")],
)
def test_metric_display_formats_concept_labels(
  concept: str, label: str, namespace: str, tool_name: str, suffix: str,
) -> None:
  display = resolve_display("mcp__edgar-parser-mcp__" + tool_name, {
    "ticker": "pcty", "metric_name": namespace + concept,
  })

  assert display is not None
  assert display == {"label": f"Pulling PCTY {label}{suffix}"}
  assert re.search(r"[a-z][A-Z]", display["label"]) is None


@pytest.mark.parametrize("metric_key", ["metric", "concept", "tag"])
def test_metric_display_formats_concept_aliases(metric_key: str) -> None:
  assert resolve_display("get_metric", {
    "ticker": "PCTY", metric_key: "us-gaap:OperatingIncomeLoss",
    "quarter": 4, "year": 2026,
  }) == {"label": "Pulling PCTY operating income (loss)", "detail": "Q4 2026"}


def test_resolve_display_price_target_uses_research_identity_not_ticker() -> None:
  display = resolve_display(
    "mcp__portfolio-reads-mcp__get_price_target",
    {"ticker": "MSCI", "research_file_id": 42, "handoff_id": 314},
  )

  assert display == {
    "label": "Pulling price target",
    "detail": "42 - 314",
  }


def test_resolve_display_model_insights_uses_research_identity_not_ticker() -> None:
  display = resolve_display(
    "mcp__portfolio-reads-mcp__get_model_insights",
    {"ticker": "MSCI", "research_file_id": 42, "model_insights_id": "mi_123"},
  )

  assert display == {
    "label": "Reviewing model insights",
    "detail": "42 - mi_123",
  }


def test_resolve_display_generic_fallback_strips_mcp_prefix_and_uses_salient_arg() -> None:
  display = resolve_display(
    "mcp__research-corpus-mcp__custom_lookup_tool",
    {"query": "Azure gross margin inflection"},
  )

  assert display == {
    "label": "Custom lookup tool",
    "detail": "Azure gross margin inflection",
  }


def test_resolve_display_url_detail_uses_origin_and_path_without_query_string() -> None:
  display = resolve_display(
    "fetch_remote_page",
    {"url": "https://example.com/research/reports/msft-quarterly-review.pdf?token=secret#frag"},
  )

  assert display is not None
  assert display["detail"] == "https://example.com/research/reports/msft-quarterly-review.pdf"
  assert "token" not in display["detail"]
  assert "secret" not in display["detail"]
  assert "?" not in display["detail"]


def test_resolve_display_bounds_detail_length() -> None:
  display = resolve_display("custom_lookup", {"query": "x" * 200})

  assert display is not None
  assert len(display["detail"]) <= DETAIL_MAX_CHARS
  assert display["detail"].endswith("...")


def test_resolve_display_uses_only_redacted_input_values() -> None:
  display = resolve_display(
    "memory_write",
    {"file": "<redacted>", "mode": "append", "content": "raw secret should not be read"},
  )

  assert display == {"label": "Writing <redacted>", "detail": "append"}
  assert "raw secret" not in str(display)


def test_display_is_preserved_when_present_on_chat_and_control_projections() -> None:
  display = {"label": "Pulling MSFT revenue", "detail": "Q3 2026 - income statement"}
  tool_call_start = {
    "type": "tool_call_start",
    "tool_call_id": "toolu_1",
    "tool_name": "get_metric",
    "tool_input": {"ticker": "MSFT"},
    "display": display,
    "run_id": "run-1",
    "control_run_id": "run-1",
    "sub_agent_id": "sub-1",
    "future_only": "strip",
  }
  tool_execute_request = {
    "type": "tool_execute_request",
    "tool_call_id": "toolu_2",
    "nonce": "nonce-1",
    "expires_at": 123,
    "tool_name": "read_cells",
    "tool_input": {"range": "A1:B2"},
    "display": {"label": "Reading cells", "detail": "A1:B2"},
    "run_id": "run-1",
    "control_run_id": "run-1",
    "future_only": "strip",
  }

  assert adapt_event(tool_call_start, 1) == {
    "type": "tool_call_start",
    "tool_call_id": "toolu_1",
    "tool_name": "get_metric",
    "tool_input": {"ticker": "MSFT"},
    "display": display,
  }
  assert adapt_control_event(tool_call_start, 1) == {
    "type": "tool_call_start",
    "tool_call_id": "toolu_1",
    "tool_name": "get_metric",
    "tool_input": {"ticker": "MSFT"},
    "display": display,
    "run_id": "run-1",
    "control_run_id": "run-1",
    "sub_agent_id": "sub-1",
  }
  assert adapt_event(tool_execute_request, 1) == {
    "type": "tool_execute_request",
    "tool_call_id": "toolu_2",
    "nonce": "nonce-1",
    "expires_at": 123,
    "tool_name": "read_cells",
    "tool_input": {"range": "A1:B2"},
    "display": {"label": "Reading cells", "detail": "A1:B2"},
  }
  assert adapt_control_event(tool_execute_request, 1) == {
    "type": "tool_execute_request",
    "tool_call_id": "toolu_2",
    "nonce": "nonce-1",
    "expires_at": 123,
    "tool_name": "read_cells",
    "tool_input": {"range": "A1:B2"},
    "display": {"label": "Reading cells", "detail": "A1:B2"},
    "run_id": "run-1",
    "control_run_id": "run-1",
  }


def test_absent_display_keeps_existing_projection_shape() -> None:
  event = {
    "type": "tool_call_start",
    "tool_call_id": "toolu_1",
    "tool_name": "code_execute",
    "tool_input": {"code": "<redacted>"},
    "execution_location": "local",
    "call_index": 2,
    "server": "local",
    "started_at": 123.4,
    "parent_assistant_message_seq": 7,
    "run_id": "run-1",
    "control_run_id": "run-1",
    "sub_agent_id": "sub-1",
    "future_only": "strip",
  }
  expected_chat = {
    "type": "tool_call_start",
    "tool_call_id": "toolu_1",
    "tool_name": "code_execute",
    "tool_input": {"code": "<redacted>"},
    "execution_location": "local",
    "call_index": 2,
    "server": "local",
    "started_at": 123.4,
    "parent_assistant_message_seq": 7,
  }

  assert adapt_event(event, 1) == expected_chat
  assert adapt_control_event(event, 1) == {
    **expected_chat,
    "run_id": "run-1",
    "control_run_id": "run-1",
    "sub_agent_id": "sub-1",
  }


def test_clean_emission_sites_stamp_display_from_redacted_input() -> None:
  raw_input = {
    "file": "secret.txt",
    "mode": "append",
    "content": "raw secret should not be read",
  }
  redacted_input = {
    "file": "<redacted>",
    "mode": "append",
    "content": "<redacted>",
  }

  class _Provider:
    name = "stub"

    def get_model_info(self, model: str) -> SimpleNamespace:
      return SimpleNamespace(
        model_id=model,
        context_window=200_000,
        max_output_tokens=8192,
      )

    def estimate_cost(
      self,
      model: str,
      uncached: int,
      cache_read: int,
      cache_write: int,
      output: int,
    ) -> SimpleNamespace:
      _ = model, uncached, cache_read, cache_write, output
      return SimpleNamespace(total_usd=0.0, breakdown={})

  class _Dispatcher:
    @staticmethod
    def redact_prepared_tool_input(
      tool_name: str,
      prepared_call: PreparedToolCall,
    ) -> dict[str, object]:
      assert tool_name == "memory_write"
      assert prepared_call.materialize_input() == raw_input
      return dict(redacted_input)

    @staticmethod
    def settle_tool_result(
      _tool_name: str,
      dispatch_entry: Any,
      result: Any,
      error: Any,
      semantic_error: Any = None,
      **_kwargs: Any,
    ) -> ToolResultSettlement:
      return settle_catalogless_tool_result(
        entry=dispatch_entry,
        result=result,
        error=error,
        semantic_error=semantic_error,
      )

    async def dispatch(
      self,
      tool_id: str,
      tool_name: str,
      tool_input: dict[str, Any],
      **_kwargs: Any,
    ):
      _ = tool_id, tool_name, tool_input
      return {"status": "ok"}, None

    async def prepare_tool_call_async(
      self,
      tool_name: str,
      tool_input: dict[str, Any],
    ) -> PreparedToolCall:
      _ = tool_name
      return PreparedToolCall(dict(tool_input))

    async def dispatch_prepared(
      self,
      tool_id: str,
      tool_name: str,
      prepared_call: PreparedToolCall,
      **kwargs: Any,
    ):
      return await self.dispatch(
        tool_id,
        tool_name,
        prepared_call.materialize_input(),
        **kwargs,
      )

    def requires_approval(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
      _ = tool_name, tool_input
      return False

  runner = AgentRunner(
    event_log=EventLog(session_id="display-redaction"),
    dispatcher=_Dispatcher(),  # type: ignore[arg-type]
    session_id="display-redaction",
    capability_execution=stub_runner_capability_execution(
      provider=_Provider(),
      model="stub-model",
      effort="none",
      auth_config={"api_key": "k"},
    ),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  asyncio.run(runner._execute_single_tool(
    "tool-1",
    "memory_write",
    raw_input,
    {"tools": []},
  ))

  start = next(
    entry.event
    for entry in runner._log.entries
    if entry.event.get("type") == "tool_call_start"
  )
  serialized = str(start)
  assert start["tool_input"] == redacted_input
  assert start["display"] == resolve_display("memory_write", redacted_input)
  assert "raw secret" not in serialized
  assert "secret.txt" not in serialized
