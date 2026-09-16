# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import tool_dispatcher_audit as audit
from agent_gateway.event_log import EventLog
from agent_gateway.secret_boundary import SecretBoundary




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


