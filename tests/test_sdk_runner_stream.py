from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (  # noqa: E402
  AgentSDKConfig,
  AgentSDKRunner,
  CapabilityResolutionError,
  EventLog,
  ProductModelRegistry,
)
import agent_gateway.sdk_runner as sdk_runner  # noqa: E402
from agent_gateway.sdk_runner_stream import ToolCallInfo, _SDKRunnerStreamMixin  # noqa: E402
from tests.sdk_capability_execution_test_support import stub_sdk_capability_execution  # noqa: E402


def _make_runner(
  *,
  on_tool_timing=None,
  api_key: str = "test-secret",
  capability_execution=None,
) -> AgentSDKRunner:
  return AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-stream",
    sdk_config=AgentSDKConfig(
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=(
      capability_execution
      or stub_sdk_capability_execution(api_key=api_key)
    ),
    system_prompt="test",
    on_tool_timing=on_tool_timing,
  )


def test_sdk_runner_stream_methods_remain_on_runner_class() -> None:
  assert issubclass(AgentSDKRunner, _SDKRunnerStreamMixin)
  assert AgentSDKRunner._handle_stream_event is _SDKRunnerStreamMixin._handle_stream_event
  assert AgentSDKRunner._normalize_tool_result is _SDKRunnerStreamMixin._normalize_tool_result
  assert AgentSDKRunner._complete_tool_call is _SDKRunnerStreamMixin._complete_tool_call
  assert AgentSDKRunner._emit_stream_complete is _SDKRunnerStreamMixin._emit_stream_complete


def test_sdk_runner_exposes_exact_capability_execution() -> None:
  capability_execution = stub_sdk_capability_execution()
  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-exact-execution",
    sdk_config=AgentSDKConfig(
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=capability_execution,
    system_prompt="test",
  )

  assert runner.capability_execution is capability_execution


def test_sdk_stream_validates_and_keeps_provider_reported_model_distinct() -> None:
  execution = stub_sdk_capability_execution()
  entry = execution.registry.require(execution.bind.model_key)
  reported_model = "claude-sonnet-4-6-20260801"
  admitted_entry = replace(
    entry,
    reported_identities=frozenset({entry.upstream_model, reported_model}),
  )
  execution = replace(
    execution,
    registry=ProductModelRegistry(
      schema=execution.registry.schema,
      revision=execution.registry.revision,
      models={admitted_entry.key: admitted_entry},
    ),
  )
  runner = _make_runner(capability_execution=execution)
  bind = runner.capability_execution.bind

  runner._handle_stream_event({
    "type": "message_start",
    "message": {
      "model": reported_model,
      "usage": {"input_tokens": 1},
    },
  })

  assert runner._usage["provider_reported_model"] == reported_model
  assert runner.capability_execution.bind.model_key == bind.model_key
  assert runner.capability_execution.bind.upstream_model == bind.upstream_model

  runner._emit_stream_complete()
  usage = runner._log.entries[-1].event["usage"]
  assert usage["capability_bind"] == bind.receipt()
  assert usage["provider_reported_model"] == reported_model


def test_sdk_stream_rejects_unadmitted_provider_reported_model() -> None:
  runner = _make_runner()

  with pytest.raises(CapabilityResolutionError) as caught:
    runner._handle_stream_event({
      "type": "message_start",
      "message": {
        "model": "unreviewed-provider-snapshot",
        "usage": {"input_tokens": 1},
      },
    })

  assert caught.value.code == "reported_identity_mismatch"


def test_sdk_system_and_assistant_logs_use_runner_secret_boundary(caplog) -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-sdk-system-8f21d7"
  runner = _make_runner(api_key=secret)

  with caplog.at_level("INFO", logger="agent_gateway.sdk_runner"):
    runner._handle_system_message(SimpleNamespace(
      subtype="status",
      data={
        "credential": secret,
        "api_key_set": True,
        "path": "/Users/alice/Documents/report.xlsx",
      },
    ))
    runner._handle_assistant_message(SimpleNamespace(
      error=f"provider rejected {secret}",
    ))

  assert secret not in caplog.text
  assert "<redacted-secret>" in caplog.text
  assert "api_key_set" in caplog.text
  assert "/Users/alice/Documents/report.xlsx" in caplog.text


def test_sdk_runner_stream_complete_requires_closed_terminal_contract() -> None:
  runner = _make_runner()

  with pytest.raises(ValueError, match="terminal_disposition"):
    runner._emit_stream_complete(terminal_disposition="future_value")

  with pytest.raises(ValueError, match="requires a reason"):
    runner._emit_stream_complete(terminal_disposition="interrupted")

  runner._emit_stream_complete(
    terminal_disposition="interrupted",
    reason="operator_pause",
  )

  terminal_event = runner._log.entries[-1].event
  assert terminal_event["terminal_disposition"] == "interrupted"
  assert terminal_event["reason"] == "operator_pause"


def test_sdk_runner_non_success_flush_interrupts_pending_tool() -> None:
  runner = _make_runner()
  runner._pending_tool_calls["tool-1"] = ToolCallInfo(
    tool_call_id="tool-1",
    tool_name="Read",
    tool_input={"file_path": "partial.txt"},
    started_at=10.0,
  )

  runner._flush_pending_tool_calls(outcome="tool_error")

  event = runner._log.entries[-1].event
  assert event["type"] == "tool_call_interrupted"
  assert event["tool_call_id"] == "tool-1"
  assert event["reason"] == "tool_error"
  assert event["tool_risk"] == "side_effecting"
  assert event["role"] == "writer"
  assert "tool-1" not in runner._pending_tool_calls


def test_sdk_runner_stream_uses_canonical_display_for_tool_start(monkeypatch: pytest.MonkeyPatch) -> None:
  runner = _make_runner()
  monkeypatch.setattr(sdk_runner, "_redact_tool_input_for_event", lambda _name, _payload: {"redacted": True})
  monkeypatch.setattr(sdk_runner, "_should_escrow_raw_tool_input", lambda _name: False)
  monkeypatch.setattr(sdk_runner, "gateway_product_id", lambda: "hank-test")

  runner._handle_stream_event(
    {
      "type": "content_block_start",
      "content_block": {"type": "tool_use", "id": "tool-1", "name": "mcp__portfolio-reads-mcp__preview_trade"},
    }
  )
  runner._handle_stream_event(
    {
      "type": "content_block_delta",
      "delta": {"type": "input_json_delta", "partial_json": json.dumps({"ticker": "MSFT"})},
    }
  )
  runner._handle_stream_event({"type": "content_block_stop"})

  events = [entry.event for entry in runner._log.entries]
  assert events == [
    {
      "type": "tool_call_start",
      "tool_call_id": "tool-1",
      "tool_name": "mcp__portfolio-reads-mcp__preview_trade",
      "tool_input": {"redacted": True},
      "display": {"label": "Preview trade"},
      "product_id": "hank-test",
    }
  ]


def test_sdk_runner_stream_parent_monkeypatches_drive_tool_completion(monkeypatch: pytest.MonkeyPatch) -> None:
  timing_calls: list[tuple] = []
  runner = _make_runner(on_tool_timing=lambda *args: timing_calls.append(args))
  monkeypatch.setattr(sdk_runner, "_server_for_tool", lambda _name: "patched-server")
  monkeypatch.setattr(sdk_runner, "time", SimpleNamespace(time=lambda: 12.5))
  runner._pending_tool_calls["tool-1"] = ToolCallInfo(
    tool_call_id="tool-1",
    tool_name="tool",
    tool_input={},
    started_at=10.0,
  )

  runner._complete_tool_call("tool-1", result={"success": False, "message": "rejected"})

  events = [entry.event for entry in runner._log.entries]
  assert events[0]["server"] == "patched-server"
  assert events[0]["duration_ms"] == 2500
  assert events[0]["is_error"] is True
  assert events[0]["semantic_error"] == {
    "code": "tool_success_false",
    "message": "rejected",
    "source": "success",
    "success": False,
  }
  assert timing_calls[0][2] == "patched-server"
  assert timing_calls[0][3] == 2500
  assert timing_calls[0][4] is True


def test_sdk_runner_suppresses_text_after_accepted_ui_blocks() -> None:
  runner = _make_runner()
  runner._pending_tool_calls["tool-ui"] = ToolCallInfo(
    tool_call_id="tool-ui",
    tool_name="emit_ui_blocks",
    tool_input={},
    started_at=0.0,
  )

  runner._complete_tool_call(
    "tool-ui",
    result={"accepted": {"ui_blocks_id": "ub_test", "emission_index": 0}},
  )
  runner._handle_stream_event({
    "type": "content_block_delta",
    "delta": {"type": "text_delta", "text": "The visual is ready."},
  })

  events = [entry.event for entry in runner._log.entries]
  assert any(event.get("type") == "tool_call_complete" for event in events)
  assert not any(event.get("type") == "text_delta" for event in events)


def test_sdk_runner_keeps_text_after_ui_blocks_validation_failure() -> None:
  runner = _make_runner()
  runner._pending_tool_calls["tool-ui"] = ToolCallInfo(
    tool_call_id="tool-ui",
    tool_name="emit_ui_blocks",
    tool_input={},
    started_at=0.0,
  )

  runner._complete_tool_call(
    "tool-ui",
    result={"validation_failed": {"failures": [{"code": "unknown_block"}]}},
  )
  runner._handle_stream_event({
    "type": "content_block_delta",
    "delta": {"type": "text_delta", "text": "I'll repair the visual."},
  })

  events = [entry.event for entry in runner._log.entries]
  assert any(event.get("type") == "text_delta" for event in events)


def test_sdk_producer_settles_a_dispatch_record_through_the_shared_builder() -> None:
  """D-B1-1: the second producer must not degrade the fold to the fallback."""

  runner = _make_runner()
  runner._pending_tool_calls["tool-1"] = ToolCallInfo(
    tool_call_id="tool-1",
    tool_name="filings_search",
    tool_input={},
    started_at=0.0,
  )

  runner._complete_tool_call(
    "tool-1",
    result={
      "status": "success",
      "hits": [
        {
          "document_id": "edgar:0000789019-26-000012",
          "ticker": "MSFT",
          "source": "filing",
          "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
        }
      ],
    },
  )

  event = next(
    entry.event
    for entry in runner._log.entries
    if entry.event.get("type") == "tool_call_complete"
  )
  assert event["dispatch"]["outcome"] == "ok"
  # The SDK owns its own dispatch: this producer never retries.
  assert event["dispatch"]["attempts"] == 1
  assert event["dispatch"]["sources"] == [
    {
      "document_id": "edgar:0000789019-26-000012",
      "source_kind": "filing",
      "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
    }
  ]


def test_sdk_producer_settles_a_failure_outcome_with_no_sources() -> None:
  runner = _make_runner()
  runner._pending_tool_calls["tool-1"] = ToolCallInfo(
    tool_call_id="tool-1",
    tool_name="get_quote",
    tool_input={},
    started_at=0.0,
  )

  runner._complete_tool_call(
    "tool-1",
    result={
      "status": "error",
      "error": {"code": "rate_limited", "message": "HTTP 429 Too Many Requests"},
    },
  )

  event = next(
    entry.event
    for entry in runner._log.entries
    if entry.event.get("type") == "tool_call_complete"
  )
  assert event["dispatch"]["outcome"] == "error_rate_limited"
  assert event["dispatch"]["sources"] == []
  assert event["is_error"] is True
