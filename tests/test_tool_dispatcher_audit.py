# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_route import DurableLocalApprovalRoute
from agent_gateway.approval_policy import ApprovalDecision as PolicyApprovalDecision
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.session import SessionStore
from agent_gateway import tool_dispatcher as dispatcher_module
from agent_gateway import tool_dispatcher_audit as audit
from agent_gateway import McpClientManager, ToolDispatcher
from agent_gateway.event_log import EventLog
from agent_gateway.secret_boundary import SecretBoundary
from agent_gateway.tool_policy_registry import PreparedToolCall


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
    raise AssertionError("MCP should not execute in audit helper tests")


class _Emitter:
  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []
    self.lifecycle_calls: list[dict[str, object]] = []
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

  async def emit_audit_for_lifecycle_event(self, **kwargs: object) -> None:
    self.lifecycle_calls.append(kwargs)


class _Store:
  def __init__(self, emitter: _Emitter | None) -> None:
    self._test_audit_emitter = emitter

  @property
  def audit_emitter(self) -> _Emitter | None:
    return self._test_audit_emitter

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


def test_emit_execution_audit_legacy_emitter_receives_safe_projection() -> None:
  async def _run() -> None:
    secret = "CUSTOM-ACTIVE-CREDENTIAL-LEGACY-AUDIT-8f21d7"
    emitter = _Emitter()
    boundary = SecretBoundary((secret,))

    await audit.emit_execution_audit(
      SimpleNamespace(approval_id="approval-1"),
      {
        "credential": secret,
        "api_key_set": True,
        "path": "/Users/alice/Documents/report.xlsx",
      },
      approval_store=_Store(emitter),
      outcome="tool_error",
      error_summary=f"failed {secret}",
      boundary_sanitizer=lambda value, sink: boundary.sanitize(
        value,
        sink=sink,
      ),
    )

    serialized = repr(emitter.calls)
    assert secret not in serialized
    assert "<redacted-secret>" in serialized
    assert emitter.calls[0]["raw_tool_args"]["api_key_set"] is True
    assert emitter.calls[0]["raw_tool_args"]["path"] == "/Users/alice/Documents/report.xlsx"

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


def test_tool_dispatcher_audit_wrappers_preserve_parent_seams(
  monkeypatch,
  tmp_path: Path,
) -> None:
  async def _run() -> None:
    class Policy:
      policy_id = "audit-wrapper-test"
      policy_version = "1"

      async def decide(self, **_kwargs: object) -> PolicyApprovalDecision:
        return PolicyApprovalDecision(
          outcome="auto_approve",
          reason="exercise the dispatcher-owned audit seam",
        )

      async def on_resolve(self, **_kwargs: object) -> None:
        return None

    async def failing_handler(
      tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[None, dict[str, str]]:
      assert tool_input == {"path": "x"}
      return None, {"code": "audit_test_failure", "message": "boom"}

    emitter = _Emitter()
    store = SQLiteApprovalStore(
      tmp_path / "audit-wrapper-approvals.sqlite3",
      audit_emitter=emitter,
    )
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
      role="owner",
    )
    dispatcher = ToolDispatcher(
      mcp_client=_NullMcpClient(),
      role="owner",
      local_tool_handlers={"audit_failure": failing_handler},
      needs_approval=lambda *_args: True,
      approved_tool_types=set(),
      approval_route=DurableLocalApprovalRoute(store, Policy(), session),
    )
    secret = "CUSTOM-ACTIVE-CREDENTIAL-DISPATCH-AUDIT-3d91"
    dispatcher.bind_secret_boundary(SecretBoundary((secret,)))
    execution_calls: list[dict[str, object]] = []

    async def fake_emit_execution_audit(
      request_arg: object,
      raw_tool_args: dict[str, object],
      *,
      approval_store: object | None,
      outcome: str,
      error_summary: str | None = None,
      boundary_sanitizer: object | None = None,
    ) -> None:
      execution_calls.append(
        {
          "request": request_arg,
          "raw_tool_args": raw_tool_args,
          "approval_store": approval_store,
          "outcome": outcome,
          "error_summary": error_summary,
          "boundary_sanitizer": boundary_sanitizer,
        }
      )

    monkeypatch.setattr(audit, "emit_execution_audit", fake_emit_execution_audit)

    result, error = await dispatcher.dispatch(
      "call-audit-failure",
      "audit_failure",
      {"path": "x"},
    )

    assert result is None
    assert error == {"code": "audit_test_failure", "message": "boom"}
    assert len(execution_calls) == 1
    execution_call = execution_calls[0]
    request = execution_call["request"]
    assert request is not None
    assert getattr(request, "tool_call_id") == "call-audit-failure"
    assert getattr(request, "tool_name") == "audit_failure"
    assert getattr(request, "state") == "auto_approved"
    assert execution_call["raw_tool_args"] == {"path": "x"}
    assert execution_call["approval_store"] is store
    assert execution_call["outcome"] == "tool_error"
    assert execution_call["error_summary"] == (
      "{'code': 'audit_test_failure', 'message': 'boom'}"
    )
    boundary_sanitizer = execution_call["boundary_sanitizer"]
    assert callable(boundary_sanitizer)
    assert boundary_sanitizer(secret, "approval_audit") == "<redacted-secret>"
    assert emitter.lifecycle_calls

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
