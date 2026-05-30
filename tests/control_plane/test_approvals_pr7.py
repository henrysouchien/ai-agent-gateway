from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi.testclient import TestClient

from agent_gateway.approval_audit import ApprovalAuditEmitter
from agent_gateway.approval_policy import ApprovalRequest, ApprovalRequestPayload, RunContext, utc_now
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.audit_writer import JSONLAuditWriter
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


API_KEY = "approvals-pr7-key"


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


def _control_session(client: TestClient, user_id: str) -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": "tui"}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _chat_session(client: TestClient, user_id: str) -> dict[str, Any]:
  response = client.post(
    "/api/chat/init",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": "tui"}},
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
    request_id=f"req-{approval_id}",
    session_id=chat_payload["session_id"],
    run_id=f"skill-run-{approval_id}",
    user_id=user_id,
    profile="chat",
    channel="tui",
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
    }
