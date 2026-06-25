from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))

from agent_gateway import AgentSDKConfig, AgentSDKRunner, EventLog  # noqa: E402
import agent_gateway.sdk_runner as sdk_runner  # noqa: E402
import agent_gateway.sdk_runner_context as sdk_runner_context  # noqa: E402
from agent_gateway import policy_imports  # noqa: E402
from agent_gateway import sdk_runner_helpers  # noqa: E402
from agent.shared import hooks  # noqa: E402
from logs import cost_tracker  # noqa: E402


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
  assert sdk_runner._policy_owner_mismatch is sdk_runner_helpers.policy_owner_mismatch
  assert sdk_runner._policy_tool_name is sdk_runner_helpers.policy_tool_name
  assert sdk_runner._redact_tool_input_for_event is sdk_runner_helpers.redact_tool_input_for_event
  assert sdk_runner._server_for_tool is sdk_runner_helpers.server_for_tool
  assert sdk_runner._should_escrow_raw_tool_input is sdk_runner_helpers.should_escrow_raw_tool_input
  assert sdk_runner._summarize_error_payload is sdk_runner_helpers.summarize_error_payload
  assert sdk_runner._PATCH_OP_RAW_INPUT_TOOLS is sdk_runner_helpers.PATCH_OP_RAW_INPUT_TOOLS


def test_sdk_runner_context_sidecar_preserves_prompt_and_parent_alias(monkeypatch: pytest.MonkeyPatch) -> None:
  runner = _make_runner()
  messages = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "second"},
    {"role": "user", "content": "third"},
  ]

  assert runner._build_prompt(messages) == sdk_runner_context.build_prompt(messages)

  monkeypatch.setattr(
    sdk_runner,
    "classify_semantic_tool_error",
    lambda result: {"code": "patched"} if result == {"success": False} else None,
  )

  entry = runner._make_result_entry("tool-1", {"success": False}, None)

  assert entry["is_error"] is True


def test_sdk_runner_context_surfaces_use_parent_normalizer(monkeypatch: pytest.MonkeyPatch) -> None:
  runner = AgentSDKRunner(
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
    context_surfaces=lambda: [{"name": "brief"}, "ignored", {"name": "tooling"}],
  )
  normalized_inputs = []

  def normalize(surfaces):
    normalized_inputs.append(surfaces)
    return [{"patched": True}]

  monkeypatch.setattr(runner, "_normalize_context_surfaces", normalize)

  assert runner._context_surface_records() == [{"patched": True}]
  assert normalized_inputs == [[{"name": "brief"}, "ignored", {"name": "tooling"}]]


def test_sdk_runner_context_hook_resolves_parent_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
  runner = _make_runner()
  contexts = []

  async def on_tool_result(ctx):
    contexts.append(ctx)
    return [{"type": "text", "text": f"{ctx.server}:{ctx.duration_ms}"}]

  runner._on_tool_result = on_tool_result
  runner._pending_tool_calls["tool-1"] = sdk_runner.ToolCallInfo(
    tool_call_id="tool-1",
    tool_name="tool",
    tool_input={},
    started_at=10.0,
  )
  monkeypatch.setattr(sdk_runner, "_server_for_tool", lambda _name: "patched-server")
  monkeypatch.setattr(sdk_runner, "time", SimpleNamespace(time=lambda: 12.5))

  additional_context = asyncio.run(
    runner._build_hook_additional_context(
      tool_call_id="tool-1",
      tool_name="tool",
      tool_input={},
      result={"status": "ok"},
      error=None,
    )
  )

  assert contexts[0].server == "patched-server"
  assert contexts[0].duration_ms == 2500
  assert additional_context == "patched-server:2500"


def test_sdk_runner_stream_forwards_ids_and_hook_resolves_namespaced_mcp(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  db_path = tmp_path / "cost.db"
  monkeypatch.setattr(cost_tracker, "_DB_PATH", db_path)
  monkeypatch.setattr(sdk_runner, "time", SimpleNamespace(time=lambda: 12.5))
  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-timing",
    sdk_config=AgentSDKConfig(
      api_key="k",
      model="claude-sonnet-4-6",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
      request_id="req-sdk-timing",
    ),
    system_prompt="test",
    on_tool_timing=hooks.tool_timing_hook,
  )
  runner._pending_tool_calls["tool-sdk-1"] = sdk_runner.ToolCallInfo(
    tool_call_id="tool-sdk-1",
    tool_name="mcp__portfolio-mcp__documents_search",
    tool_input={},
    started_at=10.0,
  )

  runner._complete_tool_call(
    "tool-sdk-1",
    result={"status": "ok"},
  )

  with sqlite3.connect(str(db_path)) as conn:
    row = conn.execute(
      """
      SELECT capability_id, transport, request_id, tool_call_id, server, tool
      FROM tool_timing
      WHERE session_id = 'sess-sdk-timing'
      """
    ).fetchone()

  assert tuple(row) == (
    "mcp:portfolio-mcp:documents_search",
    "mcp",
    "req-sdk-timing",
    "tool-sdk-1",
    "portfolio-mcp",
    "mcp__portfolio-mcp__documents_search",
  )


def test_sdk_runner_helpers_preserve_core_payload_behavior() -> None:
  assert sdk_runner_helpers.server_for_tool("mcp__portfolio-mcp__preview_trade") == "portfolio-mcp"
  assert sdk_runner_helpers.policy_tool_name("mcp__portfolio-mcp__preview_trade") == "preview_trade"
  assert sdk_runner_helpers.policy_tool_name("file_write") == "file_write"
  assert sdk_runner_helpers.policy_owner_mismatch("file_write") is None
  assert sdk_runner_helpers.should_escrow_raw_tool_input("mcp__portfolio-mcp__apply_patch_ops") is True
  assert sdk_runner_helpers.should_escrow_raw_tool_input("mcp__portfolio-mcp__preview_trade") is False
  assert sdk_runner_helpers.join_system_prompt([("a", True), ("", False), ("b", False)]) == "a\n\nb"
  assert sdk_runner_helpers.extract_text([{"type": "text", "text": "hello"}, "world"]) == "hello\nworld"
  assert sdk_runner_helpers.summarize_error_payload({"error": {"message": "bad"}}) == "bad"


def test_sdk_runner_helpers_detect_policy_owner_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
  from agent.shared import server_policies

  monkeypatch.setattr(
    server_policies,
    "get_server_for_policy_tool",
    lambda tool_name: "portfolio-trades-mcp" if tool_name == "execute_trade" else None,
  )

  assert sdk_runner_helpers.policy_owner_mismatch(
    "mcp__portfolio-mcp__execute_trade"
  ) == ("portfolio-mcp", "execute_trade", "portfolio-trades-mcp")
  assert sdk_runner_helpers.policy_owner_mismatch("mcp__portfolio-trades-mcp__execute_trade") is None


def test_sdk_runner_helpers_policy_owner_mismatch_falls_back_when_policy_modules_absent(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fake_import_module(name: str):
    if name == "agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'agent'", name="agent")
    if name == "api.agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'api'", name="api")
    raise AssertionError(f"unexpected import: {name}")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)

  assert sdk_runner_helpers.policy_owner_mismatch("mcp__portfolio-mcp__execute_trade") is None


def test_sdk_runner_helpers_policy_owner_mismatch_raises_when_agent_policy_import_breaks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fake_import_module(_name: str):
    raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)

  with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
    sdk_runner_helpers.policy_owner_mismatch("mcp__portfolio-mcp__execute_trade")


def test_sdk_runner_helpers_policy_owner_mismatch_raises_when_api_policy_import_breaks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fake_import_module(name: str):
    if name == "agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'agent'", name="agent")
    if name == "api.agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")
    raise AssertionError(f"unexpected import: {name}")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)

  with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
    sdk_runner_helpers.policy_owner_mismatch("mcp__portfolio-mcp__execute_trade")


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
