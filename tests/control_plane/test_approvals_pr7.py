from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi.testclient import TestClient

from agent_gateway.approval_audit import ApprovalAuditEmitter
from agent_gateway.approval_notifications import ApprovalNotificationDestination
from agent_gateway.approval_policy import (
  ApprovalRequest,
  ApprovalRequestPayload,
  DelegationGrant,
  RunContext,
  utc_now,
)
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.audit_writer import JSONLAuditWriter
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


API_KEY = "approvals-pr7-key"


class _FakeAutonomousProcess:
  returncode: int | None = None

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    return self.returncode

  def terminate(self) -> None:
    if self.returncode is None:
      self.returncode = -15

  def kill(self) -> None:
    if self.returncode is None:
      self.returncode = -9


def _run(coro):
  return asyncio.run(coro)


class _NoopPolicy:
  policy_bundle_hash = "test-policy-bundle"

  def __init__(self, *, owner_only_classes: set[str] | None = None) -> None:
    self.owner_only_classes = set(owner_only_classes or set())
    self.resolved: list[str] = []
    self.authorization_checks: list[tuple[str | None, str]] = []

  async def decide(
    self,
    *,
    payload: ApprovalRequestPayload,
    request: ApprovalRequest,
    run_context: RunContext,
  ):
    raise AssertionError("decide is not used by PR7 endpoint tests")

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    self.resolved.append(request.approval_id)

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = grant_id, reason

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
    self.authorization_checks.append((decider_role, tool_class))
    if tool_class in self.owner_only_classes:
      return decider_role == "owner"
    return True


def _make_app(tmp_path, policy: _NoopPolicy | None = None):
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="approvals-pr7-test-secret-0123456789",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )
  writer = JSONLAuditWriter(tmp_path / "audit")
  emitter = ApprovalAuditEmitter(
    writer=writer,
    deployment_secret=b"approval-test-secret",
    key_id="test-key",
  )
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3", audit_emitter=emitter)
  resolved_policy = policy or _NoopPolicy()
  app.state.gateway_approval_audit_writer = writer
  app.state.gateway_approval_audit_emitter = emitter
  app.state.gateway_approval_store = store
  app.state.gateway_approval_policy = resolved_policy
  return app, store, resolved_policy, writer


def _control_session(client: TestClient, user_id: str, *, channel: str = "tui") -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _chat_session(client: TestClient, user_id: str, *, channel: str = "tui") -> dict[str, Any]:
  response = client.post(
    "/api/chat/init",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def _session(app, session_payload: dict[str, Any]):
  session = app.state.auth.session_store.get_session(session_payload["session_id"])
  assert session is not None
  return session


def _with_role(app, session_payload: dict[str, Any], role: str) -> dict[str, Any]:
  session = _session(app, session_payload)
  session.role = role
  updated = dict(session_payload)
  updated["session_token"] = app.state.auth.issue_token(session)
  return updated


def _approval_record(
  *,
  chat_payload: dict[str, Any],
  tool_call_id: str,
  approval_id: str,
  user_id: str,
  delegation_id: str | None = None,
  channel: str = "tui",
  tool_class: str = "external_write",
  state: str = "pending_user",
  required_decider_count: int = 1,
  eligible_decider_count: int = 1,
  persistent_grant_scope: str | None = None,
) -> ApprovalRequest:
  now = utc_now()
  return ApprovalRequest(
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    parent_approval_id=None,
    approval_chain_id=approval_id,
    delegation_id=delegation_id,
    request_id=f"req-{approval_id}",
    session_id=chat_payload["session_id"],
    run_id=f"skill-run-{approval_id}",
    user_id=user_id,
    profile="chat",
    channel=channel,
    tool_name="execute_trade",
    tool_class=tool_class,  # type: ignore[arg-type]
    tool_args_redacted={"ticker": "AAPL"},
    args_hash=f"hash-{approval_id}",
    reason="needs approval",
    blast_radius_summary="test approval",
    state=state,  # type: ignore[arg-type]
    requested_at=now,
    decided_at=now if state in {"approved", "denied", "expired"} else None,
    decider_id="prior" if state in {"approved", "denied"} else None,
    decision=state if state in {"approved", "denied", "expired"} else None,  # type: ignore[arg-type]
    required_decider_count=required_decider_count,
    eligible_decider_count=eligible_decider_count,
    persistent_grant_scope=persistent_grant_scope,
    policy_id="test-policy",
    policy_version="1",
    policy_bundle_hash="test-policy-bundle",
  )


def _install_pending_approval(
  *,
  app,
  store: SQLiteApprovalStore,
  chat_payload: dict[str, Any],
  user_id: str,
  delegation_id: str | None = None,
  channel: str = "tui",
  tool_class: str = "external_write",
  state: str = "pending_user",
  required_decider_count: int = 1,
  eligible_decider_count: int = 1,
  persistent_grant_scope: str | None = None,
) -> tuple[ApprovalRequest, asyncio.Queue]:
  tool_call_id = f"tool-{uuid.uuid4().hex}"
  approval_id = f"appr-{uuid.uuid4().hex}"
  request_record = _approval_record(
    chat_payload=chat_payload,
    tool_call_id=tool_call_id,
    approval_id=approval_id,
    user_id=user_id,
    delegation_id=delegation_id,
    channel=channel,
    tool_class=tool_class,
    state=state,
    required_decider_count=required_decider_count,
    eligible_decider_count=eligible_decider_count,
    persistent_grant_scope=persistent_grant_scope,
  )
  _run(store.create(request_record))

  session = _session(app, chat_payload)
  queue: asyncio.Queue = asyncio.Queue(maxsize=1)
  session.pending_tools[tool_call_id] = {
    "approval_id": approval_id,
    "nonce": f"nonce-{approval_id}",
    "requested_at": int(time.time()),
    "status": "approval_pending",
    "tool_name": request_record.tool_name,
    "tool_input": dict(request_record.tool_args_redacted),
    "resolved_qualifier": "external_write:execute_trade:AAPL",
    "reason": request_record.reason,
    "allow_persistent_approval": persistent_grant_scope is not None,
  }
  session.approval_queues[tool_call_id] = queue
  return request_record, queue


def _create_delegation_grant(
  store: SQLiteApprovalStore,
  *,
  delegation_id: str,
  delegator_user_id: str,
  excel_session_id: str,
  delegator_channel: str = "web",
) -> DelegationGrant:
  grant = DelegationGrant(
    delegation_id=delegation_id,
    delegator_user_id=delegator_user_id,
    delegator_run_id="delegator-run",
    delegator_session_id="delegator-session",
    delegator_profile="orchestrator",
    delegator_channel=delegator_channel,
    bound_excel_session_id=excel_session_id,
    bound_relay_request_id=f"relay-{delegation_id}",
    bound_workbook="Book1.xlsx",
    tool_class_ceiling=frozenset({"external_write"}),
    args_predicate=None,
    window_seconds=300,
    created_at=utc_now(),
  )
  return _run(store.create_delegation_grant(grant))


def _set_session_channel(app, session_id: str, channel: str) -> None:
  # Test sessions don't bind context.channel to session.channel (it stays None),
  # which would make _channel_matches always true and bypass the cross-channel
  # delegation path. Set distinct channels so the delegation gating is genuinely
  # exercised (mirrors the real excel-channel taskpane vs orchestrator-channel split).
  sess = app.state.auth.session_store.get_session(session_id)
  assert sess is not None, f"session not found: {session_id}"
  sess.channel = channel


def _install_autonomous_pending_approval(
  *,
  app,
  store: SQLiteApprovalStore,
  run_id: str,
  user_id: str,
  channel: str = "tui",
  tool_class: str = "state_write",
) -> ApprovalRequest:
  record = app.state.subprocess_registry._find_by_control_run_id(run_id)
  assert record is not None
  tool_call_id = f"tool-{uuid.uuid4().hex}"
  approval_id = f"appr-{uuid.uuid4().hex}"
  request_record = ApprovalRequest(
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    parent_approval_id=None,
    approval_chain_id=approval_id,
    request_id=f"req-{approval_id}",
    session_id=f"agent-control:{run_id}:123",
    run_id=run_id,
    user_id=user_id,
    profile="analyst",
    channel=channel,
    tool_name="memory_write",
    tool_class=tool_class,  # type: ignore[arg-type]
    tool_args_redacted={"file": "notes/approval.md", "content": "<hash:test>"},
    args_hash=f"hash-{approval_id}",
    reason="Tool requires user approval",
    blast_radius_summary="state_write:memory_write",
    state="pending_user",
    requested_at=utc_now(),
    persistent_grant_scope="state_write:memory_write",
    policy_id="test-policy",
    policy_version="1",
    policy_bundle_hash="test-policy-bundle",
  )
  _run(store.create(request_record))
  event = {
    "type": "tool_approval_request",
    "approval_id": approval_id,
    "tool_call_id": tool_call_id,
    "nonce": f"nonce-{approval_id}",
    "tool_name": "memory_write",
    "tool_input": dict(request_record.tool_args_redacted),
    "resolved_qualifier": "",
    "reason": request_record.reason,
    "allow_persistent_approval": True,
    "ts": time.time(),
  }
  record.event_lines.append(event)
  return request_record


def test_control_approval_list_resolve_unblocks_queue_persists_grant_and_audits(tmp_path) -> None:
  app, store, policy, writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
      persistent_grant_scope="external_write:execute_trade:AAPL",
    )

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    approvals = listed.json()["approvals"]
    assert [item["approval_id"] for item in approvals] == [approval.approval_id]
    assert approvals[0]["state"] == "pending_user"

    response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": True, "reason": "looks safe"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["approval"]["approval_id"] == approval.approval_id
    assert body["approval"]["state"] == "approved"
    assert body["approval"]["decider_id"] == "alice"

    assert policy.resolved == [approval.approval_id]
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": True,
      "approval_id": approval.approval_id,
      "denied_by": None,
    }

    grant = _run(
      store.find_persistent_grant(
        user_id="alice",
        tool_name="execute_trade",
        scope_hint="external_write:execute_trade:AAPL",
      )
    )
    assert grant is not None
    entries, _cursor = _run(writer.query(approval_id=approval.approval_id, event_type="vote_recorded"))
    assert len(entries) == 1
    assert entries[0].decider_id == "alice"
    assert entries[0].decider_role == "owner"
    assert entries[0].decision_reason == "looks safe"


def test_control_approval_endpoints_reject_chat_session_tokens(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
    )

    listed = client.get("/api/control/approvals", headers=_headers(chat))
    assert listed.status_code == 401
    assert listed.json()["detail"] == "Control session required"

    response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(chat),
      json={"approved": True},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Control session required"
    retry_response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}/notifications/retry",
      headers=_headers(chat),
    )
    assert retry_response.status_code == 401
    assert retry_response.json()["detail"] == "Control session required"
    assert queue.empty()


def test_control_approval_notification_retry_requeues_failed_outbox_with_redacted_response(tmp_path) -> None:
  async def failing_sender(_row: dict[str, Any]) -> None:
    raise RuntimeError("telegram down")

  app, store, _policy, _writer = _make_app(tmp_path)
  store._notification_destination_resolver = lambda _request: [  # type: ignore[attr-defined]
    ApprovalNotificationDestination(channel="telegram", destination="chat-private")
  ]
  store._notification_sender = failing_sender  # type: ignore[attr-defined]
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
    )
    _run(store.enqueue_pending_approval_notification(approval))
    assert _run(store.deliver_pending_approval_notifications()) == 1
    failed_rows = _run(store.list_approval_notification_outbox(approval.approval_id))
    assert failed_rows[0]["state"] == "failed_retryable"
    assert failed_rows[0]["destination"] == "chat-private"
    store._notification_sender = None  # type: ignore[attr-defined]

    response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}/notifications/retry",
      headers=_headers(control),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
      "status": "queued",
      "approval_id": approval.approval_id,
      "requeued": 1,
      "delivery_scheduled": False,
      "notification": {"state": "pending", "channels": ["telegram"]},
    }
    assert "chat-private" not in response.text
    assert "message" not in body
    requeued_rows = _run(store.list_approval_notification_outbox(approval.approval_id))
    assert requeued_rows[0]["state"] == "pending"
    assert requeued_rows[0]["last_error"] is None

    repeat = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}/notifications/retry",
      headers=_headers(control),
    )

    assert repeat.status_code == 200, repeat.text
    assert repeat.json() == {
      "status": "queued",
      "approval_id": approval.approval_id,
      "requeued": 0,
      "delivery_scheduled": False,
      "notification": {"state": "pending", "channels": ["telegram"]},
    }
    assert queue.empty()


def test_control_approval_list_resolve_routes_approval_pending_autonomous_decision_to_child_inbox(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeAutonomousProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", "approval-test-hmac")
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    approvals = listed.json()["approvals"]
    assert [item["approval_id"] for item in approvals] == [approval.approval_id]
    assert approvals[0]["session_id"] == run_id
    assert approvals[0]["run_id"] == run_id

    response = client.post(
      f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": False, "reason": "reviewed diff"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["state"] == "approved"
    assert policy.resolved == [approval.approval_id]

    assert record.approval_decisions_path is not None
    decisions = [
      json.loads(line)
      for line in record.approval_decisions_path.read_text(encoding="utf-8").splitlines()
      if line.strip()
    ]
    assert decisions == [
      {
        "approval_id": approval.approval_id,
        "tool_call_id": approval.tool_call_id,
        "nonce": f"nonce-{approval.approval_id}",
        "approved": True,
        "allow_tool_type": False,
        "reason": "reviewed diff",
        "decider": {"user_id": "alice"},
        "channel": "tui",
        "decided_at": decisions[0]["decided_at"],
      }
    ]


def test_cancel_autonomous_run_denies_pending_approval_and_clears_queue(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeAutonomousProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", "approval-test-hmac")
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval cancel bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [item["approval_id"] for item in listed.json()["approvals"]] == [approval.approval_id]

    cancelled = client.delete(f"/api/control/runs/{run_id}", headers=_headers(control))
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["state"] == "cancelled"

    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "denied"
    assert stored.decider_id == "alice"
    assert stored.decision == "denied"
    assert stored.decision_reason == "run_cancelled"
    assert policy.resolved == [approval.approval_id]

    after_cancel = client.get("/api/control/approvals", headers=_headers(control))
    assert after_cancel.status_code == 200, after_cancel.text
    assert after_cancel.json()["approvals"] == []

    assert record is not None
    assert record.approval_decisions_path is not None
    decisions = [
      json.loads(line)
      for line in record.approval_decisions_path.read_text(encoding="utf-8").splitlines()
      if line.strip()
    ]
    assert decisions == [
      {
        "approval_id": approval.approval_id,
        "tool_call_id": approval.tool_call_id,
        "nonce": f"nonce-{approval.approval_id}",
        "approved": False,
        "allow_tool_type": False,
        "reason": "run_cancelled",
        "decider": {"user_id": "alice"},
        "channel": "tui",
        "decided_at": decisions[0]["decided_at"],
      }
    ]


def test_terminal_autonomous_run_hides_stale_pending_approval(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeAutonomousProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", "approval-test-hmac")
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval timeout bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [item["approval_id"] for item in listed.json()["approvals"]] == [approval.approval_id]

    record.state = "failed"
    record.exit_code = 1
    record.completed_at = time.time()
    assert record.proc is not None
    record.proc.returncode = 1

    after_failure = client.get("/api/control/approvals", headers=_headers(control))
    assert after_failure.status_code == 200, after_failure.text
    assert after_failure.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": False, "reason": "too late"},
    )

    assert response.status_code == 409, response.text
    assert response.json()["error"] == "Autonomous run is not running"
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert stored.votes_received_count == 0
    assert policy.resolved == []


def test_exited_autonomous_process_hides_pending_approval_even_before_state_update(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeAutonomousProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", "approval-test-hmac")
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval process exit bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [item["approval_id"] for item in listed.json()["approvals"]] == [approval.approval_id]

    assert record.proc is not None
    record.proc.returncode = 1

    after_process_exit = client.get("/api/control/approvals", headers=_headers(control))
    assert after_process_exit.status_code == 200, after_process_exit.text
    assert after_process_exit.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": False, "reason": "process already exited"},
    )

    assert response.status_code == 409, response.text
    assert response.json()["error"] == "Autonomous run is not running"
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert stored.votes_received_count == 0
    assert policy.resolved == []


def test_autonomous_approval_delivery_failure_does_not_resolve_store(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeAutonomousProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", "approval-test-hmac")
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.approval_decisions_path = None

    response = client.post(
      f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": False, "reason": "reviewed diff"},
    )

    assert response.status_code == 409, response.text
    assert "approval inbox unavailable" in response.json()["error"]
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert stored.votes_received_count == 0
    assert policy.resolved == []


def test_role_class_precheck_blocks_approval_but_allows_denial_for_invite(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path, policy=_NoopPolicy(owner_only_classes={"external_write"}))
  with TestClient(app) as client:
    control = _with_role(app, _control_session(client, "alice"), "invite")
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(app=app, store=store, chat_payload=chat, user_id="alice")

    blocked = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )
    assert blocked.status_code == 403, blocked.text
    after_block = _run(store.get(approval.approval_id))
    assert after_block is not None
    assert after_block.votes_received_count == 0
    assert policy.authorization_checks == [("invite", "external_write")]
    assert queue.empty()

    denied = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": False, "reason": "cancel"},
    )
    assert denied.status_code == 200, denied.text
    assert denied.json()["approval"]["state"] == "denied"
    assert queue.get_nowait() == {
      "approved": False,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
      "denied_by": None,
    }
    assert policy.authorization_checks == [("invite", "external_write")]


def test_control_approval_wrong_run_and_cross_user_are_404(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    bob_chat = _chat_session(client, "bob")
    bob_approval, _queue = _install_pending_approval(app=app, store=store, chat_payload=bob_chat, user_id="bob")

    wrong_run = client.post(
      f"/api/control/runs/not-a-run/approvals/{bob_approval.approval_id}",
      headers=_headers(alice_control),
      json={"approved": True},
    )
    cross_user = client.post(
      f"/api/control/runs/{bob_chat['session_id']}/approvals/{bob_approval.approval_id}",
      headers=_headers(alice_control),
      json={"approved": True},
    )

    assert wrong_run.status_code == 404
    assert cross_user.status_code == 404


def test_control_approval_chat_endpoints_are_channel_scoped(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    web_control = _control_session(client, "alice", channel="web")
    excel_chat = _chat_session(client, "alice", channel="excel")
    _session(app, excel_chat).channel = "excel"
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=excel_chat,
      user_id="alice",
      channel="excel",
    )

    listed = client.get("/api/control/approvals", headers=_headers(web_control))
    assert listed.status_code == 200, listed.text
    assert listed.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{excel_chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(web_control),
      json={"approved": False, "reason": "wrong channel"},
    )

    assert response.status_code == 404, response.text
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert queue.empty()


def test_delegated_chat_approval_visible_and_resolvable_cross_channel(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    web_control = _control_session(client, "alice", channel="web")
    excel_chat = _chat_session(client, "alice", channel="excel")
    _set_session_channel(app, web_control["session_id"], "web")
    _set_session_channel(app, excel_chat["session_id"], "excel")
    delegation_id = f"deleg-{uuid.uuid4().hex}"
    grant = _create_delegation_grant(
      store,
      delegation_id=delegation_id,
      delegator_user_id="alice",
      excel_session_id=excel_chat["session_id"],
    )
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=excel_chat,
      user_id="alice",
      delegation_id=grant.delegation_id,
      channel="excel",
    )

    listed = client.get("/api/control/approvals", headers=_headers(web_control))
    assert listed.status_code == 200, listed.text
    approvals = listed.json()["approvals"]
    assert [item["approval_id"] for item in approvals] == [approval.approval_id]
    assert approvals[0]["delegation_id"] == delegation_id
    assert approvals[0]["channel"] == "excel"

    response = client.post(
      f"/api/control/runs/{excel_chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(web_control),
      json={"approved": True, "allow_tool_type": False, "reason": "delegated review"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["approval"]["state"] == "approved"
    assert response.json()["approval"]["decider_id"] == "alice"
    assert policy.resolved == [approval.approval_id]
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
      "denied_by": None,
    }


def test_chat_tool_approval_endpoint_relay_policy_denial_records_provenance(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(app=app, store=store, chat_payload=chat, user_id="alice")
    session = _session(app, chat)
    pending = next(iter(session.pending_tools.values()))

    response = client.post(
      "/api/chat/tool-approval",
      headers=_headers(chat),
      json={
        "tool_call_id": approval.tool_call_id,
        "nonce": pending["nonce"],
        "approved": False,
        "allow_tool_type": True,
        "denied_by": "relay_policy",
      },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["approval"]["state"] == "denied"
    assert payload["approval"]["decision_reason"] == "Auto-denied by relay chat policy"
    assert policy.resolved == [approval.approval_id]
    assert queue.get_nowait() == {
      "approved": False,
      "allow_tool_type": True,
      "approval_id": approval.approval_id,
      "denied_by": "relay_policy",
    }
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "denied"
    assert stored.decision_reason == "Auto-denied by relay chat policy"


def test_chat_tool_approval_endpoint_approved_ignores_denied_by(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(app=app, store=store, chat_payload=chat, user_id="alice")
    session = _session(app, chat)
    pending = next(iter(session.pending_tools.values()))

    response = client.post(
      "/api/chat/tool-approval",
      headers=_headers(chat),
      json={
        "tool_call_id": approval.tool_call_id,
        "nonce": pending["nonce"],
        "approved": True,
        "allow_tool_type": False,
        "denied_by": "relay_policy",
      },
    )
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["state"] == "approved"
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
      "denied_by": None,
    }


def test_delegated_chat_approval_hidden_from_non_delegator_cross_channel(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    bob_control = _control_session(client, "bob", channel="web")
    excel_chat = _chat_session(client, "alice", channel="excel")
    delegation_id = f"deleg-{uuid.uuid4().hex}"
    grant = _create_delegation_grant(
      store,
      delegation_id=delegation_id,
      delegator_user_id="alice",
      excel_session_id=excel_chat["session_id"],
    )
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=excel_chat,
      user_id="alice",
      delegation_id=grant.delegation_id,
      channel="excel",
    )

    listed = client.get("/api/control/approvals", headers=_headers(bob_control))
    assert listed.status_code == 200, listed.text
    assert listed.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{excel_chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(bob_control),
      json={"approved": True, "reason": "not mine"},
    )

    assert response.status_code == 404, response.text
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert queue.empty()


def test_delegated_chat_approval_hidden_when_grant_delegator_differs(tmp_path) -> None:
  # Same-user floor passes (alice owns the excel session, authed as alice), but the
  # grant's delegator is a DIFFERENT user, so the delegation visibility check must
  # still reject. Pins _approval_visible_via_delegation independently of the same-user
  # floor (the non-delegator test above is short-circuited by that floor first).
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice", channel="web")
    excel_chat = _chat_session(client, "alice", channel="excel")
    _set_session_channel(app, alice_control["session_id"], "web")
    _set_session_channel(app, excel_chat["session_id"], "excel")
    delegation_id = f"deleg-{uuid.uuid4().hex}"
    grant = _create_delegation_grant(
      store,
      delegation_id=delegation_id,
      delegator_user_id="carol",
      excel_session_id=excel_chat["session_id"],
    )
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=excel_chat,
      user_id="alice",
      delegation_id=grant.delegation_id,
      channel="excel",
    )

    listed = client.get("/api/control/approvals", headers=_headers(alice_control))
    assert listed.status_code == 200, listed.text
    assert listed.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{excel_chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(alice_control),
      json={"approved": True, "reason": "wrong delegator"},
    )
    assert response.status_code == 404, response.text
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert queue.empty()


def test_n_of_m_partial_vote_does_not_unblock_queue(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
      required_decider_count=2,
      eligible_decider_count=2,
    )

    response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
      "status": "vote_recorded",
      "votes_received_count": 1,
      "required_decider_count": 2,
      "eligible_decider_count": 2,
    }
    assert queue.empty()
    assert policy.resolved == []
    pending = next(iter(_session(app, chat).pending_tools.values()))
    assert pending["status"] == "approval_pending"


def test_terminal_and_expired_approval_states_are_rejected(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    terminal, _terminal_queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
      state="approved",
    )
    expired, _expired_queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
      state="expired",
    )

    terminal_response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{terminal.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )
    expired_response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{expired.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )

    assert terminal_response.status_code == 409
    assert expired_response.status_code == 410

    terminal_retry = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{terminal.approval_id}/notifications/retry",
      headers=_headers(control),
    )
    expired_retry = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{expired.approval_id}/notifications/retry",
      headers=_headers(control),
    )

    assert terminal_retry.status_code == 409
    assert terminal_retry.json()["error"] == "Approval request already resolved"
    assert expired_retry.status_code == 410
    assert expired_retry.json()["error"] == "Approval request expired"


def test_chat_tool_approval_endpoint_uses_shared_helper(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(app=app, store=store, chat_payload=chat, user_id="alice")
    session = _session(app, chat)
    pending = next(iter(session.pending_tools.values()))

    response = client.post(
      "/api/chat/tool-approval",
      headers=_headers(chat),
      json={
        "tool_call_id": approval.tool_call_id,
        "nonce": pending["nonce"],
        "approved": True,
        "allow_tool_type": False,
      },
    )
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["state"] == "approved"
    assert policy.resolved == [approval.approval_id]
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
      "denied_by": None,
    }
