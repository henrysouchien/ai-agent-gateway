# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
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

def test_tool_execution_context_emit_owns_its_delivery_failure() -> None:
  """A tool's secondary event never fails the tool: the channel logs and the log
  closes itself under its own ``raise`` policy (HUBS run 2 catch-all ratchet)."""

  def failing_subscriber(_event: object, _session_id: str) -> None:
    raise TimeoutError("secondary event delivery timed out")

  event_log = EventLog(on_event=failing_subscriber, on_event_error="raise")
  ctx = dispatcher_module.ToolExecutionContext(
    tool_call_id="tool-1",
    tool_name="demo",
    event_log=event_log,
  )

  ctx.emit({"type": "artifact_ready", "artifact_id": "a-1"})

  assert event_log.closed is True



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


_POSTCOMPILE_SCHEMA = {
  "type": "object",
  "properties": {
    "context_ref": {"type": "string"},
    "judgment": {"type": "object"},
    "credential_note": {"type": "string"},
  },
  "required": ["context_ref", "judgment"],
  "additionalProperties": False,
}


@pytest.mark.parametrize(
  ("tool_input", "fragment"),
  [
    ({}, "missing required field(s): context_ref, judgment"),
    ({"context_ref": "ctx-1", "credential_note": "sk-secret"}, "missing required field(s): judgment"),
    (["not", "an", "object"], "expected object, got array"),
  ],
)
def test_schema_refusal_names_the_fields_and_no_call_without_echoing_input(
  tool_input,
  fragment,
) -> None:
  error = dispatcher_helpers.validate_against_local_schema(
    "fms_persist_postcompile_valuation",
    tool_input,
    _POSTCOMPILE_SCHEMA,
  )

  assert error is not None
  assert error["code"] == "invalid_tool_input_schema"
  assert "'fms_persist_postcompile_valuation'" in error["message"]
  assert fragment in error["message"]
  # Re-sending this tool with an empty or echoed input repeats the refusal.
  assert error["next_call"] is None
  assert "sk-secret" not in repr(error)
