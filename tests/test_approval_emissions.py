# ruff: noqa: E402

from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gateway_test_support.host_policy import owner_session_host_policy


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_route import DurableLocalApprovalRoute
from agent_gateway import SessionStore
from agent_gateway.approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest,
  ApprovalRequestPayload,
  RunContext,
  ToolClass,
  build_approval_request,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.tool_dispatcher_helpers import ToolResult
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.approvals import _record_vote_and_unblock
from agent_gateway.event_log import EventLog
from agent_gateway.single_user_policy import SingleUserApprovalPolicy
from agent_gateway.tool_dispatcher import ApprovalDecision, InterceptDecision, ToolDispatcher


class _NullMcpClient(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  async def call_tool(
    self,
    name: str,
    tool_input: object,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    _ = (
      name,
      tool_input,
      meta,
      abort_event,
      gateway_session,
      allow_uncertain_replay,
      trusted_dispatch_scope,
    )
    raise AssertionError("MCP should not execute in approval emission tests")


async def _ok_handler(_tool_input: dict[str, Any], **_kwargs: Any):
  return {"ok": True}, None


class _ClassifiedToolDispatcher(ToolDispatcher):
  _test_tool_class: ToolClass
  def __init__(self, *args: Any, tool_class: ToolClass, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    self._test_tool_class = tool_class

  def _resolve_tool_class(self, tool_name: str) -> ToolClass:
    _ = tool_name
    return self._test_tool_class


class _ManualApprovalBasePolicy:
  policy_bundle_hash = "manual-test-policy"
  policy_version = "1"

  async def decide(
    self,
    *,
    payload: ApprovalRequestPayload,
    request: ApprovalRequest,
    run_context: RunContext,
  ):
    _ = payload, request, run_context
    return PolicyApprovalDecision(
      outcome="request_user_approval",
      reason="Tool requires approval",
      expiry_seconds=600,
      allow_persistent_grant=False,
    )

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    _ = request

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = grant_id, reason

  def role_authorized_for_class(
    self,
    *,
    decider_role: str | None,
    tool_class: ToolClass,
  ) -> bool:
    _ = decider_role, tool_class
    return True


def _decision_events(event_log: EventLog) -> list[dict[str, Any]]:
  return [entry.event for entry in event_log.entries if entry.event.get("type") == "tool_approval_decided"]


def _dispatcher(
  event_log: EventLog,
  *,
  needs_approval=None,
  request_approval=None,
  interceptors=(),
) -> ToolDispatcher:
  return ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"write_file": _ok_handler},
    role="owner",
    needs_approval=needs_approval or (lambda _name, _tool_input, _qualifier: False),
    request_approval=request_approval,
    event_log=event_log,
    interceptors=interceptors,
  )


def test_user_approved_emits_decided_without_applying_allow_tool_type() -> None:
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
    assert events[0]["allow_tool_type_applied"] is False

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




@pytest.mark.usefixtures("owner_session_host_policy")
def test_lifecycle_user_denial_emits_ordinary_provenance_and_error(tmp_path: Path) -> None:
  async def _run() -> None:
    class _Policy:
      policy_bundle_hash = "test-policy"

      async def decide(
        self,
        *,
        payload: ApprovalRequestPayload,
        request: ApprovalRequest,
        run_context: RunContext,
      ):
        _ = payload, request, run_context
        return PolicyApprovalDecision(
          outcome="request_user_approval",
          reason="Tool requires approval",
          expiry_seconds=600,
          allow_persistent_grant=True,
        )

      async def on_resolve(self, *, request: ApprovalRequest) -> None:
        _ = request

      async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
        _ = grant_id, reason

      def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
        _ = decider_role, tool_class
        return True

    event_log = EventLog()
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    policy = _Policy()
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
      role="owner",
    )
    dispatcher = ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=_NullMcpClient(),
      local_tool_handlers={"write_file": _ok_handler},
      role=session.role,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      event_log=event_log,
      approval_route=DurableLocalApprovalRoute(
        store,
        policy,
        session,
      ),
      run_context=RunContext(
        user_id="alice",
        request_id="request-1",
        session_id=session.session_id,
        channel="web",
      ),
    )

    dispatch_task = asyncio.create_task(dispatcher.dispatch("tool-1", "write_file", {"path": "x"}))
    for _ in range(100):
      pending = session.pending_tools.get("tool-1")
      if pending is not None:
        break
      await asyncio.sleep(0.001)
    else:
      raise AssertionError("approval request was not queued")

    approval_id = str(pending["approval_id"])
    await _record_vote_and_unblock(
      target_session=session,
      pending_entry=pending,
      tool_call_id="tool-1",
      nonce=pending["nonce"],
      decider_id="alice",
      decider_role="owner",
      approved=False,
      allow_tool_type=False,
      reason=None,
      app_state=SimpleNamespace(gateway_approval_store=store, gateway_approval_policy=policy),
    )
    result, error = await dispatch_task

    assert result is None
    assert error == {"code": "user_denied", "message": "User denied execution"}
    stored = await store.get(approval_id)
    assert stored is not None
    assert stored.state == "denied"
    assert stored.decision_reason is None
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "user_denied"
    assert events[0]["outcome"] == "denied"
    assert events[0]["allow_tool_type_applied"] is False

  asyncio.run(_run())


def test_interactive_irreversible_approval_coerces_allow_tool_type_false(tmp_path: Path) -> None:
  async def _run() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    policy = _ManualApprovalBasePolicy()
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    request = build_approval_request(
      tool_call_id="tool-irreversible",
      tool_name="memory_write",
      tool_class="irreversible",
      tool_args_redacted={"file": "notes/test.md"},
      args_hash="irreversible-args-hash",
      run_context=RunContext(
        user_id="alice",
        request_id="request-irreversible",
        session_id=session.session_id,
        channel="web",
      ),
      reason="Tool requires approval",
      state="pending_user",
      approval_constraint="standard",
    )
    request.persistent_grant_scope = "irreversible:memory_write"
    await store.create(request)
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    pending = {
      "approval_id": request.approval_id,
      "nonce": "nonce-irreversible",
      "status": "approval_pending",
      "tool_name": request.tool_name,
    }
    session.approval_store = store
    session.approval_policy = policy
    session.pending_tools[request.tool_call_id] = pending
    session.approval_queues[request.tool_call_id] = queue

    await _record_vote_and_unblock(
      target_session=session,
      pending_entry=pending,
      tool_call_id=request.tool_call_id,
      nonce="nonce-irreversible",
      decider_id="alice",
      decider_role="owner",
      approved=True,
      allow_tool_type=True,
      reason="approved once",
      app_state=SimpleNamespace(
        gateway_approval_store=store,
        gateway_approval_policy=policy,
      ),
    )

    assert pending["allow_tool_type"] is False
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": False,
      "approval_id": request.approval_id,
    }
    assert await store.find_persistent_grant(
      user_id="alice",
      tool_name=request.tool_name,
      scope_hint=request.persistent_grant_scope,
    ) is None

  asyncio.run(_run())


def _background_child_dispatcher(
  tmp_path: Path,
  event_log: EventLog,
  *,
  tool_name: str,
  tool_class: ToolClass,
  executed: list[str],
) -> tuple[ToolDispatcher, SQLiteApprovalStore, Any]:
  async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
    executed.append(tool_name)
    return {"ok": True}, None

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
    role="owner",
  )
  policy = SingleUserApprovalPolicy(store=store)
  dispatcher = _ClassifiedToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={tool_name: _handler},
    role=session.role,
    needs_approval=lambda _name, _tool_input, _qualifier: True,
    event_log=event_log,
    approval_route=DurableLocalApprovalRoute(store, policy, session),
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      session_id=session.session_id,
      channel="web",
    ),
    tool_class=tool_class,
  )
  return dispatcher, store, session


@pytest.mark.usefixtures("owner_session_host_policy")
def test_background_child_static_door_is_auto_approved(tmp_path: Path) -> None:
  async def _run() -> None:
    event_log = EventLog()
    executed: list[str] = []
    dispatcher, store, _session = _background_child_dispatcher(
      tmp_path,
      event_log,
      tool_name="fms_persist_scenario_set",
      tool_class="state_write",
      executed=executed,
    )

    result, error = await dispatcher.dispatch("tool-1", "fms_persist_scenario_set", {"x": 1})

    assert error is None
    assert result == {"ok": True}
    assert executed == ["fms_persist_scenario_set"]
    stored = await store.get_by_tool_call_id("tool-1")
    assert stored is not None
    assert stored.state == "auto_approved"
    assert stored.decision_reason == "Only a live order requires a human decision"
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["outcome"] == "approved"

  asyncio.run(_run())


def test_callback_decision_sources_have_decided_event_coverage() -> None:
  async def _run() -> None:
    sources = set()

    async def approve(_request):
      return ApprovalDecision(approved=True, allow_tool_type=False)

    async def deny(_request):
      return ApprovalDecision(approved=False)

    scenarios = [
      _dispatcher(EventLog(), needs_approval=lambda *_args: True, request_approval=approve),
      _dispatcher(EventLog(), needs_approval=lambda *_args: True, request_approval=deny),
    ]

    for idx, dispatcher in enumerate(scenarios):
      await dispatcher.dispatch(f"tool-{idx}", "write_file", {"path": "x"})
      event_log = dispatcher._event_log
      assert event_log is not None
      sources.update(event["decision_source"] for event in _decision_events(event_log))

    assert sources == {"user_approved", "user_denied"}

  asyncio.run(_run())
