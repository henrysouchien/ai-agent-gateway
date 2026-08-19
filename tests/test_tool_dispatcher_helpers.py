# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import EventLog
import agent_gateway.tool_dispatcher as dispatcher_module
import agent_gateway.tool_dispatcher_helpers as dispatcher_helpers


def test_tool_dispatcher_helper_exports_are_parent_aliases() -> None:
  helper_names = (
    "ToolResult",
    "NeedsApprovalCallback",
    "ApprovalKeyQualifier",
    "_approval_queue_timeout_seconds",
    "TransportApprovalRequest",
    "TransportApprovalResult",
    "ApprovalCallback",
    "LocalToolHandler",
    "ApprovalRequest",
    "ApprovalDecision",
    "InterceptContext",
    "InterceptDecision",
    "InterceptResult",
    "ToolInterceptor",
    "HeadlessAskCallback",
    "ToolExecutionContext",
  )

  for name in helper_names:
    assert getattr(dispatcher_module, name) is getattr(dispatcher_helpers, name)

def test_tool_execution_context_emits_and_tracks_abort() -> None:
  event_log = EventLog()
  abort_event = asyncio.Event()
  ctx = dispatcher_module.ToolExecutionContext(
    tool_call_id="tool-1",
    tool_name="demo",
    event_log=event_log,
    abort_event=abort_event,
  )

  ctx.emit({"type": "custom", "value": 1})
  assert [entry.event for entry in event_log.entries] == [{"type": "custom", "value": 1}]
  assert ctx.aborted is False

  abort_event.set()
  assert ctx.aborted is True
  asyncio.run(asyncio.wait_for(ctx.wait_aborted(), timeout=0.1))


def test_intercept_decision_rejects_unknown_action() -> None:
  with pytest.raises(ValueError, match="Invalid interceptor action"):
    dispatcher_module.InterceptDecision("maybe")


def test_run_interceptors_helper_returns_first_pending_ask_and_warnings() -> None:
  event_log = EventLog()

  async def _warn(ctx: dispatcher_helpers.InterceptContext) -> dispatcher_helpers.InterceptDecision:
    assert ctx.session_id == "sess-1"
    return dispatcher_helpers.InterceptDecision("warn", message="watch it", code="warn_policy")

  async def _ask(ctx: dispatcher_helpers.InterceptContext) -> dispatcher_helpers.InterceptDecision:
    assert ctx.tool_name == "demo"
    return dispatcher_helpers.InterceptDecision("ask", message="needs review", code="ask_policy")

  result = asyncio.run(
    dispatcher_helpers.run_interceptors(
      "call-1",
      "demo",
      {"x": 1},
      interceptors=[_warn, _ask],
      event_log=event_log,
      session_id="sess-1",
      log=dispatcher_module.log,
    )
  )

  assert result.proceed is True
  assert result.warnings == ["watch it"]
  assert result.pending_ask == dispatcher_helpers.InterceptDecision(
    "ask",
    message="needs review",
    code="ask_policy",
  )
  assert [entry.event["action"] for entry in event_log.entries] == ["warn", "ask"]
