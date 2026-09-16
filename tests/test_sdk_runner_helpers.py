from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))



from agent_gateway import AgentSDKConfig, AgentSDKRunner, EventLog  # noqa: E402
import agent_gateway.sdk_runner as sdk_runner  # noqa: E402
import agent_gateway.sdk_runner_approval as sdk_runner_approval  # noqa: E402
import agent_gateway.sdk_runner_context as sdk_runner_context  # noqa: E402
from agent_gateway import policy_imports  # noqa: E402
from agent_gateway import sdk_runner_helpers  # noqa: E402
from agent_gateway.sdk_runner_stream import ToolCallInfo  # noqa: E402
from agent_gateway.tool_dispatch_classification import ToolResultSettlement  # noqa: E402
from gateway_test_support.sdk_capability_execution_test_support import stub_sdk_capability_execution  # noqa: E402


def _identity_registered_redaction(_tool_name, tool_input):
  return dict(tool_input)


def _unexpected_registered_preparation(tool_name, *_args):
  raise AssertionError(
    f"helper test did not expect registered preparation for {tool_name!r}"
  )


def _make_runner(
  *,
  registered_mcp_descriptor_for_sdk_tool: Callable[[str], Any] | None = None,
  prepare_registered_mcp_tool_call_for_sdk_tool: Callable[..., Any] | None = None,
  redact_registered_mcp_tool_input_for_sdk_tool: Callable[..., Any] | None = None,
  settle_registered_mcp_tool_result_for_sdk_tool: Callable[..., Any] | None = None,
) -> AgentSDKRunner:
  if (
    registered_mcp_descriptor_for_sdk_tool is not None
    and prepare_registered_mcp_tool_call_for_sdk_tool is None
  ):
    prepare_registered_mcp_tool_call_for_sdk_tool = (
      _unexpected_registered_preparation
    )
  if (
    registered_mcp_descriptor_for_sdk_tool is not None
    and redact_registered_mcp_tool_input_for_sdk_tool is None
  ):
    redact_registered_mcp_tool_input_for_sdk_tool = _identity_registered_redaction
  return AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-helpers",
    sdk_config=AgentSDKConfig(
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=stub_sdk_capability_execution(),
    system_prompt="test",
    registered_mcp_descriptor_for_sdk_tool=(
      registered_mcp_descriptor_for_sdk_tool
    ),
    prepare_registered_mcp_tool_call_for_sdk_tool=(
      prepare_registered_mcp_tool_call_for_sdk_tool
    ),
    redact_registered_mcp_tool_input_for_sdk_tool=(
      redact_registered_mcp_tool_input_for_sdk_tool
    ),
    settle_registered_mcp_tool_result_for_sdk_tool=(
      settle_registered_mcp_tool_result_for_sdk_tool
    ),
  )




def test_catalogless_approval_redacts_secrets_and_hashes_original_input() -> None:
  redacted, args_hash = sdk_runner_approval.redact_for_approval_request(
    "data_historical_prices",
    {"symbol": "AAPL", "credential": "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"},
  )

  assert redacted.get("symbol") == "AAPL"
  assert redacted["credential"] == "<redacted-secret>"
  assert args_hash.startswith("hmac-sha256-v1:")


def test_registered_sdk_approval_uses_exact_manager_redaction(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[tuple[str, dict[str, object]]] = []

  def _redact(tool_name: str, tool_input: dict[str, object]):
    calls.append((tool_name, dict(tool_input)))
    return {"oauth_token": "<redacted>"}

  runner = _make_runner(
    registered_mcp_descriptor_for_sdk_tool=lambda _tool_name: object(),
    redact_registered_mcp_tool_input_for_sdk_tool=_redact,
    settle_registered_mcp_tool_result_for_sdk_tool=lambda *_args: object(),
  )
  monkeypatch.setattr(
    sdk_runner_approval,
    "redact_for_approval_request",
    lambda *_args: pytest.fail("registered SDK route used bare-name redaction"),
  )
  tool_name = "mcp__portfolio-config-mcp__complete_brokerage_connection"

  redacted, args_hash = runner._redact_for_approval_request(
    tool_name,
    {"oauth_token": "raw-token"},
  )

  assert calls == [(tool_name, {"oauth_token": "raw-token"})]
  assert redacted == {"oauth_token": "<redacted>"}
  assert args_hash.startswith("hmac-sha256-v1:")


def test_sdk_runner_catalogless_approval_identity_uses_legacy_class_owner(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runner = _make_runner()
  captured: dict[str, object] = {}

  def fake_resolve_tool_class(tool_name: str, **kwargs):
    captured["tool_name"] = tool_name
    captured.update(kwargs)
    return "patched-class"

  identity = sdk_runner_approval.resolve_catalogless_approval_identity(
    "runtime-tool",
    resolve_server_policy_tool_class_fn=fake_resolve_tool_class,
  )

  assert identity == ("runtime-tool", "patched-class")
  assert captured["tool_name"] == "runtime-tool"
  assert captured["policy_tool_name"] == "runtime-tool"
  assert captured["runtime_server"] is None
  assert runner._resolve_sdk_approval_identity("runtime-tool")[0] == "runtime-tool"


def test_sdk_runner_registered_mode_keeps_builtin_approval_catalogless() -> None:
  descriptor_calls: list[str] = []

  def descriptor_for(tool_name: str) -> Any:
    descriptor_calls.append(tool_name)
    raise AssertionError("builtins do not resolve through the MCP manager")

  runner = _make_runner(
    registered_mcp_descriptor_for_sdk_tool=descriptor_for,
    settle_registered_mcp_tool_result_for_sdk_tool=lambda *_args: object(),
  )

  policy_tool, _tool_class = runner._resolve_sdk_approval_identity("file_write")

  assert policy_tool == "file_write"
  assert descriptor_calls == []


def test_sdk_runner_registered_mode_keeps_sdk_local_mcp_catalogless() -> None:
  local_tool_id = "mcp__gateway-tools__load_tools"

  class LocalMcpConfig(dict[str, Any]):
    catalogless_mcp_tool_ids = {local_tool_id}

  descriptor_calls: list[str] = []

  def descriptor_for(tool_name: str) -> Any:
    descriptor_calls.append(tool_name)
    raise AssertionError("SDK-local MCP tools do not resolve through the manager")

  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-local",
    sdk_config=AgentSDKConfig(
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=stub_sdk_capability_execution(),
    system_prompt="test",
    mcp_server_configs=LocalMcpConfig({
      "gateway-tools": {"type": "sdk"},
    }),
    registered_mcp_descriptor_for_sdk_tool=descriptor_for,
    prepare_registered_mcp_tool_call_for_sdk_tool=(
      _unexpected_registered_preparation
    ),
    redact_registered_mcp_tool_input_for_sdk_tool=(
      lambda _tool_name, tool_input: dict(tool_input)
    ),
    settle_registered_mcp_tool_result_for_sdk_tool=lambda *_args: ToolResultSettlement("ok"),
  )

  policy_tool, _tool_class = runner._resolve_sdk_approval_identity(
    local_tool_id
  )

  assert policy_tool == "load_tools"
  assert descriptor_calls == []


def test_sdk_runner_context_sidecar_preserves_prompt_and_semantic_error() -> None:
  runner = _make_runner()
  messages = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "second"},
    {"role": "user", "content": "third"},
  ]

  assert runner._build_prompt(messages) == sdk_runner_context.build_prompt(messages)

  entry = runner._make_result_entry("tool-1", {"success": False}, None)

  assert entry["is_error"] is True


def test_sdk_runner_context_surfaces_use_parent_normalizer(monkeypatch: pytest.MonkeyPatch) -> None:
  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-helpers",
    sdk_config=AgentSDKConfig(
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=stub_sdk_capability_execution(),
    system_prompt="test",
    context_surfaces=lambda: [{"name": "brief"}, "ignored", {"name": "tooling"}],  # pyright: ignore[reportArgumentType]  # negative: mixed context surface filtering
  )
  normalized_inputs = []

  def normalize(surfaces):
    normalized_inputs.append(surfaces)
    return [{"patched": True}]

  monkeypatch.setattr(runner, "_normalize_context_surfaces", normalize)

  assert runner._context_surface_records() == [{"patched": True}]
  assert normalized_inputs == [[{"name": "brief"}, "ignored", {"name": "tooling"}]]


def test_sdk_context_surface_failure_log_is_value_free(caplog) -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-sdk-context-8f21d7"

  def fail_context_surfaces():
    raise RuntimeError(secret)

  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-context-failure",
    sdk_config=AgentSDKConfig(
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=stub_sdk_capability_execution(api_key=secret),
    system_prompt="test",
    context_surfaces=fail_context_surfaces,
  )

  with caplog.at_level("WARNING", logger="agent_gateway.sdk_runner"):
    assert runner._context_surface_records() == []

  assert secret not in caplog.text
  assert "exception_type=RuntimeError" in caplog.text




def test_sdk_runner_helpers_preserve_core_payload_behavior() -> None:
  assert sdk_runner_helpers.server_for_tool("mcp__portfolio-reads-mcp__preview_trade") == "portfolio-reads-mcp"
  assert sdk_runner_helpers.catalogless_tool_name("mcp__portfolio-reads-mcp__preview_trade") == "preview_trade"
  assert sdk_runner_helpers.catalogless_tool_name("file_write") == "file_write"
  assert sdk_runner_helpers.catalogless_policy_owner_mismatch("file_write") is None
  assert sdk_runner_helpers.should_escrow_raw_tool_input("mcp__portfolio-reads-mcp__apply_patch_ops") is True
  assert sdk_runner_helpers.should_escrow_raw_tool_input("mcp__portfolio-reads-mcp__preview_trade") is False
  assert sdk_runner_helpers.join_system_prompt([("a", True), ("", False), ("b", False)]) == "a\n\nb"
  assert sdk_runner_helpers.extract_text([{"type": "text", "text": "hello"}, "world"]) == "hello\nworld"
  assert sdk_runner_helpers.summarize_error_payload({"error": {"message": "bad"}}) == "bad"


def test_sdk_runner_helpers_detect_catalogless_policy_owner_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    policy_imports,
    "_server_policy",
    SimpleNamespace(
      get_forbidden_tools_for_session=lambda _session: frozenset(),
      get_server_for_policy_tool=lambda name: (
        "portfolio-trades-mcp" if name == "execute_trade" else None
      ),
    ),
  )

  assert sdk_runner_helpers.catalogless_policy_owner_mismatch(
    "mcp__portfolio-reads-mcp__execute_trade"
  ) == ("portfolio-reads-mcp", "execute_trade", "portfolio-trades-mcp")
  assert sdk_runner_helpers.catalogless_policy_owner_mismatch("mcp__portfolio-trades-mcp__execute_trade") is None




def test_sdk_runner_helpers_catalogless_owner_is_unset_without_bound_policy() -> None:
  policy_imports.configure_server_policy(None)

  assert sdk_runner_helpers.catalogless_policy_owner_mismatch("mcp__portfolio-reads-mcp__execute_trade") is None




def test_sdk_runner_parent_helper_monkeypatches_still_drive_nested_helpers(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(sdk_runner, "_parse_result_payload", lambda _value: {"patched": True})
  assert sdk_runner._summarize_error_payload("ignored") == '{"patched": true}'


def test_sdk_runner_nested_helper_monkeypatches_resolve_parent_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sdk_runner, "_as_plain_dict", lambda _value: {"plain": True})
  assert sdk_runner._as_dict(object()) == {"plain": True}
  assert sdk_runner._parse_result_payload({"x": 1}) == {"plain": True}

  monkeypatch.setattr(sdk_runner, "_get_attr", lambda _value, key, default=None: "patched" if key == "text" else default)
  assert sdk_runner._extract_text([object()]) == "patched"

  monkeypatch.setattr(sdk_runner, "_catalogless_tool_name", lambda _tool_name: "preview_patch_ops")
  monkeypatch.setattr(sdk_runner, "_PATCH_OP_RAW_INPUT_TOOLS", frozenset({"preview_patch_ops"}))
  assert sdk_runner._should_escrow_raw_tool_input("anything") is True


def test_absent_audit_hmac_secret_defaults_loudly(
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
) -> None:
  from agent_gateway.tool_redaction import get_audit_hmac_secret

  monkeypatch.delenv("GATEWAY_AUDIT_HMAC_SECRET", raising=False)
  with caplog.at_level("WARNING", logger="agent_gateway.tool_redaction"):
    assert get_audit_hmac_secret() == b"dev-secret"
  assert any(
    "GATEWAY_AUDIT_HMAC_SECRET" in record.message for record in caplog.records
  )


def test_configured_audit_hmac_secret_is_silent(
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
) -> None:
  from agent_gateway.tool_redaction import get_audit_hmac_secret

  monkeypatch.setenv("GATEWAY_AUDIT_HMAC_SECRET", "configured-secret")
  with caplog.at_level("WARNING", logger="agent_gateway.tool_redaction"):
    assert get_audit_hmac_secret() == b"configured-secret"
  assert not caplog.records
