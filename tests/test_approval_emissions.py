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

from agent_gateway import SessionStore
from agent_gateway.approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest,
  ApprovalRequestPayload,
  RunContext,
  build_approval_request,
)
from agent_gateway.approval_policy import DelegationGrant, utc_now
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.approvals import _record_vote_and_unblock
from agent_gateway.event_log import EventLog
from agent_gateway.single_user_policy import DelegationApprovalPolicy
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


class _ClassifiedToolDispatcher(ToolDispatcher):
  def __init__(self, *args: Any, tool_class: str, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    self._test_tool_class = tool_class

  def _resolve_tool_class(self, tool_name: str) -> str:
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
      route_target_type="pending_tools",
      expiry_seconds=600,
      allow_persistent_grant=False,
    )

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    _ = request

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = grant_id, reason

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
    _ = decider_role, tool_class
    return True


def _delegation_grant(*, ceiling: frozenset[str] = frozenset({"state_write"})) -> DelegationGrant:
  now = utc_now()
  return DelegationGrant(
    delegation_id="delegation-1",
    delegator_user_id="alice",
    delegator_run_id=None,
    delegator_session_id="orchestrator-session-1",
    delegator_profile="relay_qa_operator",
    delegator_channel="excel",
    bound_excel_session_id="excel-session-1",
    bound_relay_request_id="request-1",
    bound_workbook="Budget.xlsx",
    tool_class_ceiling=ceiling,
    args_predicate=None,
    window_seconds=600,
    exclude_external_write_bypass=True,
    created_at=now,
    expires_at=None,
  )


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
    role="owner",
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


def test_lifecycle_approval_request_times_out_without_user_response(monkeypatch, tmp_path: Path) -> None:
  async def _run() -> None:
    from agent_gateway import approval_settings

    monkeypatch.setattr(approval_settings, "approval_wait_seconds", lambda: 0.01)

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
          route_target_type="pending_tools",
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
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
      role="owner",
    )
    dispatcher = ToolDispatcher(
      mcp_client=_NullMcpClient(),
      local_tool_handlers={"write_file": _ok_handler},
      role=session.role,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      event_log=event_log,
      session=session,
      store=store,
      policy=_Policy(),
      run_context=RunContext(
        user_id="alice",
        request_id="request-1",
        session_id=session.session_id,
        channel="web",
      ),
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result is None
    assert error == {"code": "approval_timeout", "message": "User did not respond within timeout"}
    approval_events = [
      entry.event for entry in event_log.entries if entry.event.get("type") == "tool_approval_request"
    ]
    assert len(approval_events) == 1
    stored = await store.get(approval_events[0]["approval_id"])
    assert stored is not None
    assert stored.state == "expired"
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "approval_timeout"
    assert events[0]["outcome"] == "timeout"

  asyncio.run(_run())


def test_lifecycle_relay_policy_denial_emits_provenance_and_error(tmp_path: Path) -> None:
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
          route_target_type="pending_tools",
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
      mcp_client=_NullMcpClient(),
      local_tool_handlers={"write_file": _ok_handler},
      role=session.role,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      event_log=event_log,
      session=session,
      store=store,
      policy=policy,
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
      denied_by="relay_policy",
    )
    result, error = await dispatch_task

    assert result is None
    assert error is not None
    assert error["code"] == "user_denied"
    assert error["sub_code"] == "relay_policy_denied"
    assert "relay chat policy" in error["message"]
    assert "taskpane composer" in error["message"]
    stored = await store.get(approval_id)
    assert stored is not None
    assert stored.state == "denied"
    assert stored.decision_reason == "Auto-denied by relay chat policy"
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "relay_policy_denied"
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
      "denied_by": None,
    }
    assert await store.find_persistent_grant(
      user_id="alice",
      tool_name=request.tool_name,
      scope_hint=request.persistent_grant_scope,
    ) is None

  asyncio.run(_run())


def test_delegated_lifecycle_auto_approval_emits_delegated_source(tmp_path: Path) -> None:
  async def _run() -> None:
    event_log = EventLog()
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
      role="owner",
    )
    policy = DelegationApprovalPolicy(base=_ManualApprovalBasePolicy())
    dispatcher = _ClassifiedToolDispatcher(
      mcp_client=_NullMcpClient(),
      local_tool_handlers={"write_file": _ok_handler},
      role=session.role,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      event_log=event_log,
      session=session,
      store=store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id="request-1",
        session_id="excel-session-1",
        channel="excel",
        delegation=_delegation_grant(),
        policy_bundle_hash=policy.policy_bundle_hash,
      ),
      tool_class="state_write",
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result == {"ok": True}
    assert error is None
    stored = await store.get_by_tool_call_id("tool-1")
    assert stored is not None
    assert stored.state == "auto_approved"
    assert stored.delegation_id == "delegation-1"
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "delegated_auto_approved"
    assert events[0]["outcome"] == "approved"

  asyncio.run(_run())


def test_delegated_lifecycle_external_write_escalates_without_auto_approval(monkeypatch, tmp_path: Path) -> None:
  async def _run() -> None:
    from agent_gateway import approval_settings

    monkeypatch.setattr(approval_settings, "approval_wait_seconds", lambda: 0.01)

    event_log = EventLog()
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
      role="owner",
    )
    policy = DelegationApprovalPolicy(base=_ManualApprovalBasePolicy())
    dispatcher = _ClassifiedToolDispatcher(
      mcp_client=_NullMcpClient(),
      local_tool_handlers={"write_file": _ok_handler},
      role=session.role,
      needs_approval=lambda _name, _tool_input, _qualifier: True,
      event_log=event_log,
      session=session,
      store=store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id="request-1",
        session_id="excel-session-1",
        channel="excel",
        delegation=_delegation_grant(ceiling=frozenset({"read", "state_write"})),
        policy_bundle_hash=policy.policy_bundle_hash,
      ),
      tool_class="external_write",
    )

    result, error = await dispatcher.dispatch("tool-1", "write_file", {"path": "x"})

    assert result is None
    assert error == {"code": "approval_timeout", "message": "User did not respond within timeout"}
    request_events = [entry.event for entry in event_log.entries if entry.event.get("type") == "tool_approval_request"]
    assert len(request_events) == 1
    stored = await store.get_by_tool_call_id("tool-1")
    assert stored is not None
    assert stored.state == "expired"
    events = _decision_events(event_log)
    assert len(events) == 1
    assert events[0]["decision_source"] == "approval_timeout"
    assert all(event["decision_source"] != "delegated_auto_approved" for event in events)

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
    assert not any(
      entry.event.get("type") == "tool_approval_request"
      for entry in event_log.entries
    )

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


def test_callback_headless_and_cache_decision_sources_have_decided_event_coverage() -> None:
  async def _run() -> None:
    sources = set()

    async def approve(_request):
      return ApprovalDecision(approved=True, allow_tool_type=False)

    async def deny(_request):
      return ApprovalDecision(approved=False)

    async def deny_by_relay_policy(_request):
      return ApprovalDecision(approved=False, denied_by="relay_policy")

    async def timeout(_request):
      return None

    async def ask(_ctx):
      return InterceptDecision(action="ask", message="Needs confirmation")

    scenarios = [
      _dispatcher(EventLog(), needs_approval=lambda *_args: True, request_approval=approve),
      _dispatcher(EventLog(), needs_approval=lambda *_args: True, request_approval=deny),
      _dispatcher(EventLog(), needs_approval=lambda *_args: True, request_approval=deny_by_relay_policy),
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
      "relay_policy_denied",
      "approval_timeout",
      "headless_auto_deny",
      "headless_hook_approved",
      "session_cache_approved",
    }

  asyncio.run(_run())
