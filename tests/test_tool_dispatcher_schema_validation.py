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


def test_local_tool_schema_validation_rejects_unknown_top_level_field() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _unexpected_handler},
    get_tool_definitions=_tool_defs,
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
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"structured_write": _handler},
    needs_approval=lambda _name, _tool_input, _qualifier: True,
    session=session,
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
