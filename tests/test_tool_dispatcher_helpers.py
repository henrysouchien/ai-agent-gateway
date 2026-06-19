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
    "RELAY_POLICY_DENIED_SUB_CODE",
    "RELAY_POLICY_DENIED_MESSAGE",
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
    "resolve_denied_provenance",
    "InterceptContext",
    "InterceptDecision",
    "InterceptResult",
    "ToolInterceptor",
    "HeadlessAskCallback",
    "ToolExecutionContext",
  )

  for name in helper_names:
    assert getattr(dispatcher_module, name) is getattr(dispatcher_helpers, name)


def test_resolve_denied_provenance_preserves_relay_policy_message() -> None:
  source, error = dispatcher_module.resolve_denied_provenance("relay_policy")

  assert source == dispatcher_module.RELAY_POLICY_DENIED_SUB_CODE
  assert error["sub_code"] == dispatcher_module.RELAY_POLICY_DENIED_SUB_CODE
  assert error["message"] == dispatcher_module.RELAY_POLICY_DENIED_MESSAGE


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
