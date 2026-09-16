# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_route import DurableLocalApprovalRoute
from agent_gateway import (
  EventLog,
  McpClientManager,
  PolicyApprovalDecision,
  RunContext,
  SessionStore,
  ToolDispatcher,
)
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.tool_policy_registry import PreparedToolCall
from gateway_test_support.host_policy import owner_session_host_policy


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any] | PreparedToolCall,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> NoReturn:
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


def test_request_snapshot_blocks_unadvertised_local_tool() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"hidden_write": _unexpected_handler},
    get_tool_definitions=_tool_defs,
    role="owner",
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-1",
      "hidden_write",
      {},
      advertised_tool_names=frozenset(),
    )
  )

  assert result is None
  assert error is not None
  assert error["code"] == "tool_not_advertised"


def test_snapshot_admitted_local_tool_requires_live_schema_for_input_validation() -> None:
  live_definitions = [
    {
      "name": "hidden_write",
      "input_schema": {"type": "object"},
    }
  ]
  request_snapshot = frozenset({"hidden_write"})
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"hidden_write": _unexpected_handler},
    get_tool_definitions=lambda: list(live_definitions),
    role="owner",
  )
  live_definitions.clear()

  result, error = _run(
    dispatcher.dispatch(
      "call-1",
      "hidden_write",
      {"payload": "accepted without a live schema"},
      advertised_tool_names=request_snapshot,
    )
  )

  assert result is None
  assert error == {
    "code": "tool_schema_unavailable",
    "message": (
      "Cannot validate local tool 'hidden_write': its definition is absent "
      "from the active tool catalog."
    ),
    "details": {
      "tool_name": "hidden_write",
      "reason": "tool_definition_missing",
    },
    "fix": "Retry after the active tool definition is available.",
  }


def test_snapshot_admitted_local_tool_requires_schema_in_live_definition() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"hidden_write": _unexpected_handler},
    get_tool_definitions=lambda: [{"name": "hidden_write"}],
    role="owner",
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-1",
      "hidden_write",
      {},
      advertised_tool_names=frozenset({"hidden_write"}),
    )
  )

  assert result is None
  assert error is not None
  assert error["code"] == "tool_schema_unavailable"
  assert error["details"] == {
    "tool_name": "hidden_write",
    "reason": "input_schema_missing",
  }


def test_catalog_free_local_caller_remains_supported_without_request_snapshot() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"package_read": _ok_handler},
    role="owner",
  )

  result, error = _run(
    dispatcher.dispatch("call-1", "package_read", {"ticker": "PAYC"})
  )

  assert error is None
  assert result == {"received": {"ticker": "PAYC"}}


def test_local_tool_schema_validation_rechecks_approval_modified_args(tmp_path: Path, owner_session_host_policy) -> None:
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
    role="owner",
    approval_route=DurableLocalApprovalRoute(
      store,
      _ModifiedArgsPolicy(),
      session,
    ),
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
