# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import tool_dispatcher as dispatcher_module
from agent_gateway import tool_dispatcher_audit as audit
from agent_gateway import ToolDispatcher
from agent_gateway.event_log import EventLog


class _NullMcpClient:
  def is_mcp_tool(self, _tool_name: str) -> bool:
    return False

  def get_server_for_tool(self, _tool_name: str) -> str | None:
    return None

  async def call_tool(self, _tool_name: str, _tool_input: dict[str, Any]):
    raise AssertionError("MCP should not execute in audit helper tests")


class _Emitter:
  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []
    self.raw_args_object: dict[str, Any] | None = None

  async def emit_execution_outcome(self, **kwargs: Any) -> None:
    self.raw_args_object = kwargs["raw_tool_args"]
    self.calls.append(
      {
        "request": kwargs["request"],
        "raw_tool_args": dict(kwargs["raw_tool_args"]),
        "outcome": kwargs["outcome"],
        "error_summary": kwargs["error_summary"],
      }
    )


class _Store:
  def __init__(self, emitter: _Emitter | None) -> None:
    self.audit_emitter = emitter


def test_emit_approval_decided_appends_typed_event_with_injected_clock() -> None:
  event_log = EventLog()

  audit.emit_approval_decided(
    event_log,
    "tool-1",
    "write_file",
    outcome="approved",
    decision_source="user_approved",
    allow_tool_type_applied=True,
    time_fn=lambda: 123.0,
  )

  assert event_log.entries[-1].event == {
    "type": "tool_approval_decided",
    "tool_call_id": "tool-1",
    "tool_name": "write_file",
    "outcome": "approved",
    "decision_source": "user_approved",
    "allow_tool_type_applied": True,
    "ts": 123.0,
  }


def test_emit_execution_audit_copies_and_clears_raw_args() -> None:
  async def _run() -> None:
    emitter = _Emitter()
    request = SimpleNamespace(approval_id="approval-1")
    raw_tool_args = {"path": "x", "nested": {"keep": True}}

    await audit.emit_execution_audit(
      request,
      raw_tool_args,
      approval_store=_Store(emitter),
      outcome="tool_error",
      error_summary="boom",
    )

    assert emitter.calls == [
      {
        "request": request,
        "raw_tool_args": {"path": "x", "nested": {"keep": True}},
        "outcome": "tool_error",
        "error_summary": "boom",
      }
    ]
    assert raw_tool_args == {"path": "x", "nested": {"keep": True}}
    assert emitter.raw_args_object == {}
    assert emitter.raw_args_object is not raw_tool_args

  asyncio.run(_run())


def test_emit_execution_audit_noops_without_request_store_or_emitter() -> None:
  async def _run() -> None:
    raw_tool_args = {"path": "x"}

    await audit.emit_execution_audit(
      None,
      raw_tool_args,
      approval_store=_Store(_Emitter()),
      outcome="success",
    )
    await audit.emit_execution_audit(
      SimpleNamespace(approval_id="approval-1"),
      raw_tool_args,
      approval_store=None,
      outcome="success",
    )
    await audit.emit_execution_audit(
      SimpleNamespace(approval_id="approval-1"),
      raw_tool_args,
      approval_store=_Store(None),
      outcome="success",
    )

    assert raw_tool_args == {"path": "x"}

  asyncio.run(_run())


def test_tool_dispatcher_audit_wrappers_preserve_parent_seams(monkeypatch) -> None:
  async def _run() -> None:
    store = _Store(_Emitter())
    dispatcher = ToolDispatcher(mcp_client=_NullMcpClient(), store=store)
    request = SimpleNamespace(approval_id="approval-1")
    execution_calls: list[dict[str, Any]] = []

    async def fake_emit_execution_audit(
      request_arg: Any,
      raw_tool_args: dict[str, Any],
      *,
      approval_store: Any | None,
      outcome: str,
      error_summary: str | None = None,
    ) -> None:
      execution_calls.append(
        {
          "request": request_arg,
          "raw_tool_args": raw_tool_args,
          "approval_store": approval_store,
          "outcome": outcome,
          "error_summary": error_summary,
        }
      )

    monkeypatch.setattr(audit, "emit_execution_audit", fake_emit_execution_audit)

    await dispatcher._emit_execution_audit(
      request,
      {"path": "x"},
      outcome="tool_error",
      error_summary="boom",
    )

    assert execution_calls == [
      {
        "request": request,
        "raw_tool_args": {"path": "x"},
        "approval_store": store,
        "outcome": "tool_error",
        "error_summary": "boom",
      }
    ]

    event_log = EventLog()
    dispatcher = ToolDispatcher(mcp_client=_NullMcpClient(), event_log=event_log)
    monkeypatch.setattr(dispatcher_module.time, "time", lambda: 456.0)

    dispatcher._emit_approval_decided(
      "tool-2",
      "write_file",
      outcome="denied",
      decision_source="user_denied",
      allow_tool_type_applied=False,
    )

    assert event_log.entries[-1].event["ts"] == 456.0
    assert event_log.entries[-1].event["decision_source"] == "user_denied"

  asyncio.run(_run())
