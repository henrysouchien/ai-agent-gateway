from __future__ import annotations

from pathlib import Path

from agent_gateway.event_adapter import adapt_control_event, adapt_event
from agent_gateway.tool_display import DETAIL_MAX_CHARS, resolve_display


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
  repo_root = Path(__file__).resolve().parents[3]
  runner = (repo_root / "packages/agent-gateway/agent_gateway/runner.py").read_text(encoding="utf-8")
  sdk_runner = (repo_root / "packages/agent-gateway/agent_gateway/sdk_runner.py").read_text(encoding="utf-8")
  addin_runtime = (repo_root / "api/agent/interactive/runtime.py").read_text(encoding="utf-8")

  assert "from .tool_display import resolve_display" in runner
  assert "display = resolve_display(tool_name, redacted_tool_input)" in runner
  assert 'tool_start_event["display"] = display' in runner

  assert "from .tool_display import resolve_display" in sdk_runner
  assert "display = resolve_display(tool_name, redacted_tool_input)" in sdk_runner
  assert 'tool_start_event["display"] = display' in sdk_runner

  assert "from agent_gateway.tool_display import resolve_display" in addin_runtime
  assert "redacted_tool_input = redact_tool_input(" in addin_runtime
  assert "display = resolve_display(payload.tool_name, redacted_tool_input)" in addin_runtime
  assert 'tool_execute_event["display"] = display' in addin_runtime
