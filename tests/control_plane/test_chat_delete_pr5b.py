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
from agent_gateway.event_log import EventLog
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


API_KEY = "chat-delete-pr5b-key"


def _run(coro):
  return asyncio.run(coro)


class _NoopPolicy:
  policy_bundle_hash = "test-policy-bundle"

  def __init__(self) -> None:
    self.resolved: list[str] = []
    self.authorization_checks: list[tuple[str | None, str]] = []

  async def decide(
    self,
    *,
    payload: ApprovalRequestPayload,
    request: ApprovalRequest,
    run_context: RunContext,
  ):
    raise AssertionError("decide is not used by chat DELETE tests")

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    self.resolved.append(request.approval_id)

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = grant_id, reason

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
    self.authorization_checks.append((decider_role, tool_class))
    return decider_role == "owner"


class _ApprovalHoldingRunner:
  def __init__(
    self,
    *,
    event_log: EventLog,
    session,
    store: SQLiteApprovalStore,
    approval_id_holder: list[str],
  ) -> None:
    self._event_log = event_log
    self._session = session
    self._store = store
    self._approval_id_holder = approval_id_holder

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, model_override, max_turns
    tool_call_id = f"tool-{uuid.uuid4().hex}"
    approval_id = f"appr-{uuid.uuid4().hex}"
    self._approval_id_holder.append(approval_id)
    now = utc_now()
    await self._store.create(
      ApprovalRequest(
        approval_id=approval_id,
        tool_call_id=tool_call_id,
        parent_approval_id=None,
        approval_chain_id=approval_id,
        request_id=f"req-{approval_id}",
        session_id=self._session.session_id,
        run_id=f"skill-run-{approval_id}",
        user_id=self._session.user_id,
        profile="chat",
        channel="tui",
        tool_name="execute_trade",
        tool_class="external_write",
        tool_args_redacted={"ticker": "AAPL"},
        args_hash=f"hash-{approval_id}",
        reason="needs approval",
        blast_radius_summary="test approval",
        state="pending_user",
        requested_at=now,
        decided_at=None,
        decider_id=None,
        decision=None,
        required_decider_count=1,
        eligible_decider_count=1,
        persistent_grant_scope=None,
        policy_id="test-policy",
        policy_version="1",
        policy_bundle_hash="test-policy-bundle",
      )
    )
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    nonce = f"nonce-{approval_id}"
    self._session.pending_tools[tool_call_id] = {
      "approval_id": approval_id,
      "nonce": nonce,
      "requested_at": int(time.time()),
      "status": "approval_pending",
      "tool_name": "execute_trade",
      "tool_input": {"ticker": "AAPL"},
      "resolved_qualifier": "external_write:execute_trade:AAPL",
      "reason": "needs approval",
      "allow_persistent_approval": False,
    }
    self._session.approval_queues[tool_call_id] = queue
    self._event_log.append(
      {
        "type": "tool_approval_request",
        "tool_call_id": tool_call_id,
        "approval_id": approval_id,
        "nonce": nonce,
        "tool_name": "execute_trade",
        "tool_input": {"ticker": "AAPL"},
        "resolved_qualifier": "external_write:execute_trade:AAPL",
        "reason": "needs approval",
        "allow_persistent_approval": False,
      }
    )
    decision = await queue.get()
    self._event_log.append(
      {
        "type": "tool_approval_decided",
        "tool_call_id": tool_call_id,
        "approval_id": approval_id,
        "approved": bool(decision.get("approved")),
        "decision_source": "user_denied",
      }
    )
    self._event_log.append({"type": "stream_complete", "usage": {}})


def _make_app(tmp_path):
  seen_events: list[dict[str, Any]] = []
  approval_ids: list[str] = []
  app_holder: dict[str, Any] = {}

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = request, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid: _ApprovalHoldingRunner(
        event_log=event_log,
        session=session,
        store=app_holder["app"].state.gateway_approval_store,
        approval_id_holder=approval_ids,
      ),
    )

  def _on_event(event: dict[str, Any], _session_id: str) -> None:
    seen_events.append(dict(event))

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="chat-delete-pr5b-test-secret-0123456789",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
      on_event=_on_event,
    )
  )
  writer = JSONLAuditWriter(tmp_path / "audit")
  emitter = ApprovalAuditEmitter(
    writer=writer,
    deployment_secret=b"approval-test-secret",
    key_id="test-key",
  )
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3", audit_emitter=emitter)
  policy = _NoopPolicy()
  app.state.gateway_approval_audit_writer = writer
  app.state.gateway_approval_audit_emitter = emitter
  app.state.gateway_approval_store = store
  app.state.gateway_approval_policy = policy
  app_holder["app"] = app
  return app, store, policy, seen_events, approval_ids


def _control_session(client: TestClient, user_id: str) -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": "tui"}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def _with_role(app, session_payload: dict[str, Any], role: str) -> dict[str, Any]:
  session = app.state.auth.session_store.get_session(session_payload["session_id"])
  assert session is not None
  session.role = role
  updated = dict(session_payload)
  updated["session_token"] = app.state.auth.issue_token(session)
  return updated


def test_chat_delete_denies_pending_approval_unblocks_loop_and_expires_session(tmp_path) -> None:
  app, store, policy, seen_events, approval_ids = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _with_role(app, _control_session(client, "alice"), "invite")
    dispatch = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={"kind": "chat", "message": "place trade", "channel": "tui"},
    )
    assert dispatch.status_code == 200, dispatch.text
    run_payload = dispatch.json()
    assert run_payload["run"]["state"] == "approval_pending"
    assert run_payload["run"]["pending_approval"] is not None
    assert approval_ids

    deleted = client.delete(f"/api/control/runs/{run_payload['chat_session_id']}", headers=_headers(control))
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["state"] == "cancelled"

    request_record = _run(store.get(approval_ids[0]))
    assert request_record is not None
    assert request_record.state == "denied"
    assert request_record.decider_id == "alice"
    assert request_record.decision == "denied"
    assert policy.resolved == [approval_ids[0]]
    assert policy.authorization_checks == []
    assert app.state.auth.session_store.get_session(run_payload["chat_session_id"]) is None

    for _ in range(20):
      if any(event.get("type") == "tool_approval_decided" for event in seen_events):
        break
      time.sleep(0.05)
    decided_events = [event for event in seen_events if event.get("type") == "tool_approval_decided"]
    assert decided_events
    assert decided_events[-1]["decision_source"] == "user_denied"
    assert decided_events[-1]["approved"] is False
