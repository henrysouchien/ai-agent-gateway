from agent_gateway.registry_scope import RegistryScope
from gateway_test_support.run_context_test_support import run_context_for_test
# ruff: noqa: E402

from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

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
from agent_gateway.server_chat_helpers import _legacy_request_approval, _make_request_approval
from agent_gateway.session_recap import compute_recap
from agent_gateway.server_models import ToolApprovalRequest
from agent_gateway.server_tool_routes import tool_approval_response
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
    run_context=run_context_for_test(registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store')),
  )


@pytest.mark.parametrize("approved", [True, False])
@pytest.mark.parametrize("callback_factory", [_make_request_approval, _legacy_request_approval])
@pytest.mark.parametrize("configured_store", [False, True])
def test_callback_approval_http_vote_preserves_event_identity_and_reason(
  approved: bool,
  callback_factory,
  configured_store: bool,
  tmp_path: Path,
) -> None:
  async def _run() -> None:
    event_log = EventLog()
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
      role="owner",
    )
    dispatcher = _dispatcher(
      event_log,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      request_approval=callback_factory(session, event_log),
    )
    app_state = SimpleNamespace()
    if configured_store:
      store = SQLiteApprovalStore(tmp_path / "unused-legacy-approvals.sqlite3")
      app_state.gateway_approval_store = store
      app_state.gateway_approval_policy = SingleUserApprovalPolicy(store=store)
    approval_ids = []
    started_at = time.time()
    for index in range(2):
      tool_call_id = f"callback-tool-{index}"
      task = asyncio.create_task(
        dispatcher.dispatch(tool_call_id, "write_file", {"path": f"x-{index}"})
      )

      async def wait_for_request() -> None:
        while tool_call_id not in session.approval_queues:
          if task.done():
            await task
            raise AssertionError("Dispatch completed without requesting approval")
          await asyncio.sleep(0)

      try:
        await asyncio.wait_for(wait_for_request(), timeout=5)
        request_event = [
          entry.event for entry in event_log.entries
          if entry.event.get("type") == "tool_approval_request"
          and entry.event["tool_call_id"] == tool_call_id
        ][0]
        approval_id = request_event["approval_id"]
        assert isinstance(approval_id, str) and approval_id
        approval_ids.append(approval_id)
        pending = session.pending_tools[tool_call_id]
        assert started_at <= request_event["ts"] <= time.time()
        assert pending["requested_at"] == int(request_event["ts"])
        pending_recap = compute_recap(
          event_log, session_id=session.session_id, started_at=started_at, trigger="explicit",
        )
        assert pending_recap.approvals[-1].approval_id == approval_id
        assert pending_recap.approvals[-1].state == "pending"
        assert pending_recap.approvals[-1].requested_at == request_event["ts"]
        reason = f"Reviewed callback call {index}"
        response = await tool_approval_response(
          cast(Any, SimpleNamespace(
            headers={"Authorization": "Bearer test"},
            app=SimpleNamespace(state=app_state),
          )),
          ToolApprovalRequest(
            tool_call_id=tool_call_id, nonce=pending["nonce"],
            approved=approved, reason=reason,
          ),
          auth=cast(Any, SimpleNamespace(verify_token=lambda _token: session)),
          record_vote_and_unblock=_record_vote_and_unblock,
        )
        assert response.status_code == 200
        result, error = await asyncio.wait_for(task, timeout=5)
      finally:
        if not task.done():
          task.cancel()
          await asyncio.gather(task, return_exceptions=True)

      if approved:
        assert result == {"ok": True}
        assert error is None
      else:
        assert result is None
        assert error == {"code": "user_denied", "message": f"Tool call denied by user: {reason}"}
      decision = _decision_events(event_log)[-1]
      assert decision["approval_id"] == approval_id
      assert decision["decision_reason"] == reason
      assert decision["outcome"] == ("approved" if approved else "denied")
      recap = compute_recap(
        event_log, session_id=session.session_id, started_at=started_at, trigger="explicit",
      )
      assert len(recap.approvals) == index + 1
      assert recap.approvals[-1].approval_id == approval_id
      assert recap.approvals[-1].state == "decided"
      assert recap.approvals[-1].outcome == decision["outcome"]
      assert recap.approvals[-1].requested_at == request_event["ts"]
      assert tool_call_id not in session.pending_tools
    assert len(set(approval_ids)) == 2

  asyncio.run(_run())


def test_user_approved_emits_decided_without_applying_allow_tool_type() -> None:
  async def _run() -> None:
    event_log = EventLog()
    requests = []

    async def approve(request):
      requests.append(request)
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
    assert events[0]["approval_id"] == requests[0].approval_id
    assert events[0]["decision_source"] == "user_approved"
    assert events[0]["outcome"] == "approved"
    assert events[0]["allow_tool_type_applied"] is False

  asyncio.run(_run())


def test_callback_approval_without_allow_tool_type_reports_not_applied() -> None:
  async def _run() -> None:
    event_log = EventLog()

    async def approve_once(_request):
      return ApprovalDecision(approved=True, allow_tool_type=False)

    dispatcher = _dispatcher(
      event_log,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      request_approval=approve_once,
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result == {"ok": True}
    assert error is None
    events = _decision_events(event_log)
    assert len(events) == 1
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


class _GrantingPolicy:
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
      persistent_grant_scope_hint="write_file",
    )

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    _ = request

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = grant_id, reason

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
    _ = decider_role, tool_class
    return True


def _persistent_grant_rows(store: SQLiteApprovalStore) -> list[tuple[str, str]]:
  connection = sqlite3.connect(store.path)
  try:
    return [
      (str(row[0]), str(row[1]))
      for row in connection.execute(
        "SELECT tool_name, granted_via_approval_id FROM persistent_grants"
      )
    ]
  finally:
    connection.close()


@pytest.mark.usefixtures("owner_session_host_policy")
@pytest.mark.parametrize("vote_allow_tool_type", [True, False])
def test_lifecycle_user_approval_reports_the_applied_grant(
  tmp_path: Path,
  vote_allow_tool_type: bool,
) -> None:
  async def _run() -> None:
    event_log = EventLog()
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    policy = _GrantingPolicy()
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
      approval_route=DurableLocalApprovalRoute(store, policy, session),
      run_context=RunContext(
        user_id="alice",
        request_id="request-1",
        session_id=session.session_id,
        channel="web",
        registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store'),
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
      approved=True,
      allow_tool_type=vote_allow_tool_type,
      reason=None,
      app_state=SimpleNamespace(gateway_approval_store=store, gateway_approval_policy=policy),
    )
    result, error = await dispatch_task

    assert result == {"ok": True}
    assert error is None
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "user_approved"
    assert events[0]["outcome"] == "approved"
    assert events[0]["allow_tool_type_applied"] is vote_allow_tool_type
    assert _persistent_grant_rows(store) == (
      [("write_file", approval_id)] if vote_allow_tool_type else []
    )

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
        registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store'),
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
        registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store'),
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
      registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store'),
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
    # Propagation: the stored reason is what the owner decided for this call.
    decided = await SingleUserApprovalPolicy().decide(
      payload=ApprovalRequestPayload(
        stored.approval_id, stored.tool_name, stored.tool_class, {},
      ),
      request=stored,
      run_context=RunContext(
        user_id="alice",
        request_id="request-1",
        session_id=stored.session_id,
        channel="web",
        registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store'),
      ),
    )
    assert decided.outcome == "auto_approve"
    assert stored.decision_reason == decided.reason
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
