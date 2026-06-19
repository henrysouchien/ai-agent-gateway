from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentSDKConfig, AgentSDKRunner, EventLog  # noqa: E402
import agent_gateway.sdk_runner as sdk_runner  # noqa: E402
from agent_gateway import sdk_runner_helpers  # noqa: E402


def _make_runner() -> AgentSDKRunner:
  return AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-helpers",
    sdk_config=AgentSDKConfig(
      api_key="k",
      model="claude-sonnet-4-6",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    system_prompt="test",
  )


def test_sdk_runner_helper_aliases_remain_on_parent_module() -> None:
  assert sdk_runner._as_dict is sdk_runner_helpers.as_dict
  assert sdk_runner._as_plain_dict is sdk_runner_helpers.as_plain_dict
  assert sdk_runner._extract_text is sdk_runner_helpers.extract_text
  assert sdk_runner._get_attr is sdk_runner_helpers.get_attr
  assert sdk_runner._join_system_prompt is sdk_runner_helpers.join_system_prompt
  assert sdk_runner._parse_result_payload is sdk_runner_helpers.parse_result_payload
  assert sdk_runner._policy_tool_name is sdk_runner_helpers.policy_tool_name
  assert sdk_runner._redact_tool_input_for_event is sdk_runner_helpers.redact_tool_input_for_event
  assert sdk_runner._server_for_tool is sdk_runner_helpers.server_for_tool
  assert sdk_runner._should_escrow_raw_tool_input is sdk_runner_helpers.should_escrow_raw_tool_input
  assert sdk_runner._summarize_error_payload is sdk_runner_helpers.summarize_error_payload
  assert sdk_runner._PATCH_OP_RAW_INPUT_TOOLS is sdk_runner_helpers.PATCH_OP_RAW_INPUT_TOOLS


def test_sdk_runner_helpers_preserve_core_payload_behavior() -> None:
  assert sdk_runner_helpers.server_for_tool("mcp__portfolio-mcp__preview_trade") == "portfolio-mcp"
  assert sdk_runner_helpers.policy_tool_name("mcp__portfolio-mcp__preview_trade") == "preview_trade"
  assert sdk_runner_helpers.policy_tool_name("file_write") == "file_write"
  assert sdk_runner_helpers.should_escrow_raw_tool_input("mcp__portfolio-mcp__apply_patch_ops") is True
  assert sdk_runner_helpers.should_escrow_raw_tool_input("mcp__portfolio-mcp__preview_trade") is False
  assert sdk_runner_helpers.join_system_prompt([("a", True), ("", False), ("b", False)]) == "a\n\nb"
  assert sdk_runner_helpers.extract_text([{"type": "text", "text": "hello"}, "world"]) == "hello\nworld"
  assert sdk_runner_helpers.summarize_error_payload({"error": {"message": "bad"}}) == "bad"


def test_sdk_runner_parent_helper_monkeypatches_still_drive_methods(monkeypatch: pytest.MonkeyPatch) -> None:
  runner = _make_runner()
  monkeypatch.setattr(sdk_runner, "_parse_result_payload", lambda _value: {"patched": True})
  assert runner._normalize_tool_result({"content": "ignored"}) == ({"patched": True}, None)
  assert sdk_runner._summarize_error_payload("ignored") == '{"patched": true}'

  monkeypatch.setattr(sdk_runner, "_summarize_error_payload", lambda _value: "patched summary")
  additional_context = runner._format_additional_context(
    tool_name="tool",
    result_entry={"is_error": True, "content": {"anything": "goes"}},
    extra_blocks=[],
  )

  assert additional_context is not None
  assert "patched summary" in additional_context


def test_sdk_runner_nested_helper_monkeypatches_resolve_parent_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sdk_runner, "_as_plain_dict", lambda _value: {"plain": True})
  assert sdk_runner._as_dict(object()) == {"plain": True}
  assert sdk_runner._parse_result_payload({"x": 1}) == {"plain": True}

  monkeypatch.setattr(sdk_runner, "_get_attr", lambda _value, key, default=None: "patched" if key == "text" else default)
  assert sdk_runner._extract_text([object()]) == "patched"

  monkeypatch.setattr(sdk_runner, "_policy_tool_name", lambda _tool_name: "preview_patch_ops")
  monkeypatch.setattr(sdk_runner, "_PATCH_OP_RAW_INPUT_TOOLS", frozenset({"preview_patch_ops"}))
  assert sdk_runner._should_escrow_raw_tool_input("anything") is True
