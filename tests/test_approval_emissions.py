import asyncio
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.event_log import EventLog
from agent_gateway.tool_dispatcher import ApprovalDecision, InterceptDecision, ToolDispatcher


class _NullMcpClient:
  def is_mcp_tool(self, _tool_name: str) -> bool:
    return False

  def get_server_for_tool(self, _tool_name: str) -> str | None:
    return None

  async def call_tool(self, _tool_name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    raise AssertionError("MCP should not execute in approval emission tests")


async def _ok_handler(_tool_input: dict[str, Any], **_kwargs: Any):
  return {"ok": True}, None


def _decision_events(event_log: EventLog) -> list[dict[str, Any]]:
  return [entry.event for entry in event_log.entries if entry.event.get("type") == "tool_approval_decided"]


def _legacy_headless_events(event_log: EventLog) -> list[dict[str, Any]]:
  return [entry.event for entry in event_log.entries if entry.event.get("type") == "headless_auto_deny"]


def _dispatcher(
  event_log: EventLog,
  *,
  needs_approval=None,
  request_approval=None,
  approved_tool_types: set[str] | None = None,
  session_cache_denied_tools: frozenset[str] | None = None,
  interceptors=(),
  should_avoid_permission_prompts: bool = False,
  on_headless_ask=None,
) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"write_file": _ok_handler},
    needs_approval=needs_approval or (lambda _name, _tool_input, _qualifier: False),
    request_approval=request_approval,
    approved_tool_types=approved_tool_types,
    event_log=event_log,
    interceptors=interceptors,
    should_avoid_permission_prompts=should_avoid_permission_prompts,
    on_headless_ask=on_headless_ask,
    session_cache_denied_tools=session_cache_denied_tools,
  )


def test_user_approved_emits_decided_and_real_allow_tool_type_gate() -> None:
  async def _run() -> None:
    event_log = EventLog()

    async def approve(_request):
      return ApprovalDecision(approved=True, allow_tool_type=True)

    dispatcher = _dispatcher(
      event_log,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      request_approval=approve,
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result == {"ok": True}
    assert error is None
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "user_approved"
    assert events[0]["outcome"] == "approved"
    assert events[0]["allow_tool_type_applied"] is True

  asyncio.run(_run())


def test_user_denied_emits_decided_without_installing_allow_tool_type() -> None:
  async def _run() -> None:
    event_log = EventLog()

    async def deny(_request):
      return ApprovalDecision(approved=False, allow_tool_type=True)

    dispatcher = _dispatcher(
      event_log,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      request_approval=deny,
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result is None
    assert error == {"code": "user_denied", "message": "User denied execution"}
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "user_denied"
    assert events[0]["outcome"] == "denied"
    assert events[0]["allow_tool_type_applied"] is False

  asyncio.run(_run())


def test_dynamic_ask_user_approval_emits_without_persistent_cache_install() -> None:
  async def _run() -> None:
    event_log = EventLog()

    async def ask(_ctx):
      return InterceptDecision(action="ask", message="Needs confirmation")

    async def approve(_request):
      return ApprovalDecision(approved=True, allow_tool_type=True)

    dispatcher = _dispatcher(event_log, request_approval=approve, interceptors=[ask])

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result == {"ok": True}
    assert error is None
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "user_approved"
    assert events[0]["allow_tool_type_applied"] is False

  asyncio.run(_run())


def test_approval_timeout_emits_decided_event() -> None:
  async def _run() -> None:
    event_log = EventLog()

    async def timeout(_request):
      return None

    dispatcher = _dispatcher(
      event_log,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      request_approval=timeout,
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result is None
    assert error == {"code": "approval_timeout", "message": "User did not respond within timeout"}
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "approval_timeout"
    assert events[0]["outcome"] == "timeout"

  asyncio.run(_run())


def test_headless_static_auto_deny_emits_legacy_and_decided_events() -> None:
  async def _run() -> None:
    event_log = EventLog()
    dispatcher = _dispatcher(
      event_log,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      should_avoid_permission_prompts=True,
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result is None
    assert error is not None
    assert error["code"] == "headless_auto_deny"
    assert len(_legacy_headless_events(event_log)) == 1
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "headless_auto_deny"
    assert events[0]["outcome"] == "denied"

  asyncio.run(_run())


def test_headless_hook_allow_emits_decided_and_executes_tool() -> None:
  async def _run() -> None:
    event_log = EventLog()

    async def ask(_ctx):
      return InterceptDecision(action="ask", message="Needs confirmation")

    dispatcher = _dispatcher(
      event_log,
      interceptors=[ask],
      should_avoid_permission_prompts=True,
      on_headless_ask=lambda _ctx, _decision: "allow",
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result == {"ok": True}
    assert error is None
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "headless_hook_approved"
    assert events[0]["outcome"] == "approved"

  asyncio.run(_run())


def test_session_cache_auto_approval_emits_standalone_decided_event() -> None:
  async def _run() -> None:
    event_log = EventLog()
    dispatcher = _dispatcher(
      event_log,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      approved_tool_types={"write_file"},
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result == {"ok": True}
    assert error is None
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "session_cache_approved"
    assert events[0]["outcome"] == "approved"
    assert not any(event.get("type") == "tool_approval_request" for event in events)

  asyncio.run(_run())


def test_session_cache_denied_guard_suppresses_cache_approved_emission() -> None:
  async def _run() -> None:
    event_log = EventLog()
    dispatcher = _dispatcher(
      event_log,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      approved_tool_types={"write_file"},
      session_cache_denied_tools=frozenset({"write_file"}),
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result is None
    assert error is not None
    assert error["code"] == "approval_required"
    assert _decision_events(event_log) == []

  asyncio.run(_run())


def test_all_package_decision_sources_have_decided_event_coverage() -> None:
  async def _run() -> None:
    sources = set()

    async def approve(_request):
      return ApprovalDecision(approved=True, allow_tool_type=False)

    async def deny(_request):
      return ApprovalDecision(approved=False)

    async def timeout(_request):
      return None

    async def ask(_ctx):
      return InterceptDecision(action="ask", message="Needs confirmation")

    scenarios = [
      _dispatcher(EventLog(), needs_approval=lambda *_args: True, request_approval=approve),
      _dispatcher(EventLog(), needs_approval=lambda *_args: True, request_approval=deny),
      _dispatcher(EventLog(), needs_approval=lambda *_args: True, request_approval=timeout),
      _dispatcher(
        EventLog(),
        needs_approval=lambda *_args: True,
        should_avoid_permission_prompts=True,
      ),
      _dispatcher(
        EventLog(),
        interceptors=[ask],
        should_avoid_permission_prompts=True,
        on_headless_ask=lambda _ctx, _decision: "allow",
      ),
      _dispatcher(
        EventLog(),
        needs_approval=lambda *_args: True,
        approved_tool_types={"write_file"},
      ),
    ]

    for idx, dispatcher in enumerate(scenarios):
      await dispatcher.dispatch(f"tool-{idx}", "write_file", {"path": "x"})
      event_log = dispatcher._event_log
      assert event_log is not None
      sources.update(event["decision_source"] for event in _decision_events(event_log))

    assert sources == {
      "user_approved",
      "user_denied",
      "approval_timeout",
      "headless_auto_deny",
      "headless_hook_approved",
      "session_cache_approved",
    }

  asyncio.run(_run())
