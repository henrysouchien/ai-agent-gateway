# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (
  EventLog,
  PolicyApprovalDecision,
  RunContext,
  SessionStore,
  ToolDispatcher,
)
from agent_gateway.approval_store import SQLiteApprovalStore
import agent_gateway.tool_dispatcher_helpers as dispatcher_helpers


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient:
  def is_mcp_tool(self, _tool_name: str) -> bool:
    return False

  def get_server_for_tool(self, _tool_name: str) -> str | None:
    return None

  async def call_tool(self, _tool_name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    raise AssertionError("MCP should not execute")


async def _unexpected_handler(_tool_input: dict[str, Any], **_kwargs: Any):
  raise AssertionError("handler should not execute")


async def _ok_handler(tool_input: dict[str, Any], **_kwargs: Any):
  return {"received": dict(tool_input)}, None


class _ModifiedArgsPolicy:
  policy_bundle_hash = "modified-args-test-policy"
  policy_version = "1"

  async def decide(self, *, payload, request, run_context):
    _ = payload, request, run_context
    return PolicyApprovalDecision(
      outcome="auto_approve",
      reason="test modified args",
      modified_tool_args={},
    )

  async def on_resolve(self, *, request) -> None:
    _ = request

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = grant_id, reason

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
    _ = decider_role, tool_class
    return True


def _tool_defs() -> list[dict[str, Any]]:
  return [
    {
      "name": "structured_write",
      "description": "test",
      "input_schema": {
        "type": "object",
        "properties": {
          "judgment": {"type": "object"},
        },
        "required": ["judgment"],
        "additionalProperties": False,
      },
    }
  ]


def test_local_tool_schema_validation_rejects_missing_required_before_handler() -> None:
  event_log = EventLog()
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _unexpected_handler},
    event_log=event_log,
    get_tool_definitions=_tool_defs,
    role="owner",
  )

  result, error = _run(dispatcher.dispatch("call-1", "structured_write", {}))

  assert result is None
  assert error is not None
  assert error["code"] == "invalid_tool_input_schema"
  assert error["details"]["missing"] == ["judgment"]
  events = [entry.event for entry in event_log.entries]
  assert events == [
    {
      "type": "tool_input_validation_failed",
      "tool_call_id": "call-1",
      "tool_name": "structured_write",
      "code": "invalid_tool_input_schema",
      "message": error["message"],
      "details": error["details"],
    }
  ]


def test_local_tool_schema_validation_methods_delegate_to_extracted_helpers() -> None:
  schema = _tool_defs()[0]["input_schema"]
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _unexpected_handler},
    get_tool_definitions=_tool_defs,
  )

  assert ToolDispatcher._json_type_name(True) == dispatcher_helpers.json_type_name(True)
  assert ToolDispatcher._matches_json_type({}, ["array", "object"]) is True
  assert ToolDispatcher._format_expected_type(["string", "null"]) == "string|null"
  assert dispatcher._active_local_tool_schema("structured_write") == (
    dispatcher_helpers.active_local_tool_schema(_tool_defs, "structured_write")
  )
  assert dispatcher._tool_input_schema_error(
    "structured_write",
    message="bad",
    details={"missing": ["judgment"]},
  ) == dispatcher_helpers.tool_input_schema_error(
    "structured_write",
    message="bad",
    details={"missing": ["judgment"]},
  )
  assert dispatcher._validate_against_local_schema(
    "structured_write",
    {},
    schema,
  ) == dispatcher_helpers.validate_against_local_schema("structured_write", {}, schema)


def test_local_tool_schema_validation_preserves_dispatcher_override_seam() -> None:
  calls: list[tuple[str, str, Any]] = []
  handler_calls: list[dict[str, Any]] = []

  async def _handler(tool_input: dict[str, Any], **_kwargs: Any):
    handler_calls.append(dict(tool_input))
    return {"ok": True}, None

  class _OverrideDispatcher(ToolDispatcher):
    def _validate_local_tool_input(
      self,
      tool_call_id: str,
      tool_name: str,
      tool_input: Any,
    ) -> dict[str, Any] | None:
      calls.append((tool_call_id, tool_name, tool_input))
      return {"code": "custom_schema_gate", "message": "blocked by override"}

  dispatcher = _OverrideDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _handler},
    get_tool_definitions=_tool_defs,
    role="owner",
  )

  result, error = _run(
    dispatcher.dispatch("call-1", "structured_write", {"judgment": {"ticker": "PAYC"}})
  )

  assert result is None
  assert error == {"code": "custom_schema_gate", "message": "blocked by override"}
  assert calls == [("call-1", "structured_write", {"judgment": {"ticker": "PAYC"}})]
  assert handler_calls == []


def test_local_tool_schema_validation_preserves_active_schema_override_seam() -> None:
  class _ActiveSchemaOverrideDispatcher(ToolDispatcher):
    def _active_local_tool_schema(
      self,
      tool_name: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
      return None, {"code": "custom_active_schema", "message": f"{tool_name} blocked"}

  dispatcher = _ActiveSchemaOverrideDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _ok_handler},
    get_tool_definitions=_tool_defs,
    role="owner",
  )

  result, error = _run(
    dispatcher.dispatch("call-1", "structured_write", {"judgment": {"ticker": "PAYC"}})
  )

  assert result is None
  assert error == {"code": "custom_active_schema", "message": "structured_write blocked"}


def test_local_tool_schema_validation_preserves_type_match_override_seam() -> None:
  handler_calls: list[dict[str, Any]] = []

  async def _handler(tool_input: dict[str, Any], **_kwargs: Any):
    handler_calls.append(dict(tool_input))
    return {"ok": True}, None

  class _LenientTypeDispatcher(ToolDispatcher):
    @classmethod
    def _matches_json_type(cls, value: Any, expected_type: Any) -> bool:
      _ = value, expected_type
      return True

  dispatcher = _LenientTypeDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _handler},
    get_tool_definitions=_tool_defs,
    role="owner",
  )

  result, error = _run(
    dispatcher.dispatch("call-1", "structured_write", {"judgment": "accepted by override"})
  )

  assert error is None
  assert result == {"ok": True}
  assert handler_calls == [{"judgment": "accepted by override"}]


def test_local_tool_schema_validation_rejects_unknown_top_level_field() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _unexpected_handler},
    get_tool_definitions=_tool_defs,
    role="owner",
  )

  result, error = _run(
    dispatcher.dispatch("call-1", "structured_write", {"judgment": {}, "ops": []})
  )

  assert result is None
  assert error is not None
  assert error["code"] == "invalid_tool_input_schema"
  assert error["details"]["unexpected"] == ["ops"]


def test_local_tool_schema_validation_rejects_top_level_type_mismatch() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _unexpected_handler},
    get_tool_definitions=_tool_defs,
    role="owner",
  )

  result, error = _run(
    dispatcher.dispatch("call-1", "structured_write", {"judgment": "not an object"})
  )

  assert result is None
  assert error is not None
  assert error["code"] == "invalid_tool_input_schema"
  assert error["details"]["type_errors"] == [
    {"field": "judgment", "expected": "object", "got": "string"}
  ]


def test_local_tool_schema_validation_allows_valid_input() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _ok_handler},
    get_tool_definitions=_tool_defs,
    role="owner",
  )

  result, error = _run(
    dispatcher.dispatch("call-1", "structured_write", {"judgment": {"ticker": "PAYC"}})
  )

  assert error is None
  assert result == {"received": {"judgment": {"ticker": "PAYC"}}}


def test_local_tool_schema_validation_blocks_unadvertised_local_tool() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"hidden_write": _unexpected_handler},
    get_tool_definitions=_tool_defs,
    role="owner",
  )

  result, error = _run(dispatcher.dispatch("call-1", "hidden_write", {}))

  assert result is None
  assert error is not None
  assert error["code"] == "tool_not_advertised"


def test_local_tool_schema_validation_rechecks_approval_modified_args(tmp_path: Path) -> None:
  calls: list[dict[str, Any]] = []

  async def _handler(tool_input: dict[str, Any], **_kwargs: Any):
    calls.append(dict(tool_input))
    return {"ok": True}, None

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
    role="owner",
  )
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _handler},
    needs_approval=lambda _name, _tool_input, _qualifier: True,
    session=session,
    role="owner",
    store=store,
    policy=_ModifiedArgsPolicy(),
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      session_id=session.session_id,
      channel="cli",
      policy_bundle_hash="modified-args-test-policy",
    ),
    get_tool_definitions=_tool_defs,
  )

  result, error = _run(
    dispatcher.dispatch("call-1", "structured_write", {"judgment": {"ticker": "PAYC"}})
  )

  assert result is None
  assert error is not None
  assert error["code"] == "invalid_tool_input_schema"
  assert error["details"]["missing"] == ["judgment"]
  assert calls == []
