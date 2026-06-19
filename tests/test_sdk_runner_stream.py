from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentSDKConfig, AgentSDKRunner, EventLog  # noqa: E402
import agent_gateway.sdk_runner as sdk_runner  # noqa: E402
from agent_gateway.sdk_runner_stream import _SDKRunnerStreamMixin  # noqa: E402


def _make_runner(*, on_tool_timing=None) -> AgentSDKRunner:
  return AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-stream",
    sdk_config=AgentSDKConfig(
      api_key="k",
      model="claude-sonnet-4-6",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
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
  assert sdk_runner.ToolCallInfo.__module__ == "agent_gateway.sdk_runner"
  assert sdk_runner._ActiveToolUse.__module__ == "agent_gateway.sdk_runner"


def test_sdk_runner_stream_parent_monkeypatches_drive_tool_start(monkeypatch: pytest.MonkeyPatch) -> None:
  runner = _make_runner()
  monkeypatch.setattr(sdk_runner, "_redact_tool_input_for_event", lambda _name, _payload: {"redacted": True})
  monkeypatch.setattr(sdk_runner, "resolve_display", lambda _name, _payload: {"summary": "patched"})
  monkeypatch.setattr(sdk_runner, "_should_escrow_raw_tool_input", lambda _name: False)
  monkeypatch.setattr(sdk_runner, "gateway_product_id", lambda: "hank-test")

  runner._handle_stream_event(
    {
      "type": "content_block_start",
      "content_block": {"type": "tool_use", "id": "tool-1", "name": "mcp__portfolio-mcp__preview_trade"},
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
      "tool_name": "mcp__portfolio-mcp__preview_trade",
      "tool_input": {"redacted": True},
      "display": {"summary": "patched"},
      "product_id": "hank-test",
    }
  ]


def test_sdk_runner_stream_parent_monkeypatches_drive_tool_completion(monkeypatch: pytest.MonkeyPatch) -> None:
  timing_calls: list[tuple] = []
  runner = _make_runner(on_tool_timing=lambda *args: timing_calls.append(args))
  monkeypatch.setattr(sdk_runner, "_server_for_tool", lambda _name: "patched-server")
  monkeypatch.setattr(sdk_runner, "classify_semantic_tool_error", lambda _result: {"code": "patched"})
  monkeypatch.setattr(sdk_runner, "time", SimpleNamespace(time=lambda: 12.5))
  runner._pending_tool_calls["tool-1"] = sdk_runner.ToolCallInfo(
    tool_call_id="tool-1",
    tool_name="tool",
    tool_input={},
    started_at=10.0,
  )

  runner._complete_tool_call("tool-1", result={"status": "ok"})

  events = [entry.event for entry in runner._log.entries]
  assert events[0]["server"] == "patched-server"
  assert events[0]["duration_ms"] == 2500
  assert events[0]["is_error"] is True
  assert events[0]["semantic_error"] == {"code": "patched"}
  assert timing_calls[0][2] == "patched-server"
  assert timing_calls[0][3] == 2500
  assert timing_calls[0][4] is True
