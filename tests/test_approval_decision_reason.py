from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.approval_audit import ApprovalAuditEmitter
from agent_gateway.approval_policy import RunContext
from agent_gateway.approval_route import DurableLocalApprovalRoute
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.approvals import _record_vote_and_unblock
from agent_gateway.audit_writer import JSONLAuditWriter
from agent_gateway.event_adapter import adapt_event
from agent_gateway.event_log import EventLog
from agent_gateway.events import ToolApprovalDecidedEvent, event_from_dict
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.secret_boundary import SecretBoundary
from agent_gateway.server_models import ToolApprovalRequest
from agent_gateway.server_tool_routes import tool_approval_response
from agent_gateway.session import SessionStore
from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway.tool_policy_registry import PreparedToolCall
from gateway_test_support.approval_policy_test_support import FixedOutcomeApprovalPolicy
from gateway_test_support.capability_execution_test_support import stub_capability_execution_resolver
from gateway_test_support.host_policy import owner_session_host_policy


@pytest.mark.parametrize("reason, expected", [(None, None), ("  ", None), ("  review complete\n", "review complete")])
def test_approval_reason_normalization(reason: str | None, expected: str | None) -> None:
  payload = ToolApprovalRequest(tool_call_id="call-1", nonce="nonce", approved=False, reason=reason)
  assert payload.reason == expected


@pytest.mark.parametrize("approved", [True, False])
def test_interactive_reason_reaches_audit_session_projection_and_tool_result(
  tmp_path: Path, owner_session_host_policy: Any, approved: bool,
) -> None:
  async def scenario() -> None:
    writer = JSONLAuditWriter(tmp_path / "audit")
    store = SQLiteApprovalStore(
      tmp_path / "approvals.sqlite3",
      audit_emitter=ApprovalAuditEmitter(writer=writer, deployment_secret=b"test-secret", key_id="test"),
    )
    policy = FixedOutcomeApprovalPolicy("request_user_approval")
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice", role="owner")
    session.agent_session_log = AgentSessionLog(tmp_path / "session.jsonl")
    route = DurableLocalApprovalRoute(store, policy, session)
    session.approval_route = route
    event_log = EventLog()
    executed: list[dict[str, Any]] = []

    async def write_tool(tool_input: dict[str, Any], **_kwargs: Any) -> Any:
      executed.append(tool_input)
      return {"ok": True}, None

    dispatcher = ToolDispatcher(
      capability_bind=stub_capability_execution_resolver().resolve("session.driver").bind,
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={"write_tool": write_tool},
      needs_approval=lambda *_args: True,
      event_log=event_log,
      session=session,
      role="owner",
      approval_route=route,
      run_context=RunContext(user_id="alice", request_id="request-1", session_id=session.session_id),
    )
    dispatcher.bind_secret_boundary(SecretBoundary(()))
    call = asyncio.create_task(dispatcher.dispatch_prepared(
      "call-1", "write_tool", PreparedToolCall({"ticker": "MSFT"}),
      advertised_tool_names=frozenset({"write_tool"}),
    ))

    async def wait_for_approval() -> dict[str, Any]:
      while "call-1" not in session.approval_queues:
        if call.done():
          raise AssertionError(f"Call ended before approval: {call.result()}")
        await asyncio.sleep(0)
      return session.pending_tools["call-1"]

    try:
      pending = await asyncio.wait_for(wait_for_approval(), 10)
      reason = "Reviewed the proposed change" if approved else "Keep the current valuation assumptions"
      payload = ToolApprovalRequest(
        tool_call_id="call-1", nonce=pending["nonce"], approved=approved, reason=f"  {reason}\n",
      )
      response = await tool_approval_response(
        cast(Any, SimpleNamespace(headers={"Authorization": "Bearer test"}, app=SimpleNamespace(state=SimpleNamespace()))),
        payload,
        auth=cast(Any, SimpleNamespace(verify_token=lambda _token: session)),
        record_vote_and_unblock=partial(
          _record_vote_and_unblock, authoritative_store=store, authoritative_policy=policy,
        ),
      )
      assert response.status_code == 200
      result, error = await asyncio.wait_for(call, 10)
      assert bool(executed) is approved
      if approved:
        assert result == {"ok": True} and error is None
      else:
        assert result is None
        assert error is not None
        assert error["code"] == "user_denied"
        assert error["message"] == f"Tool call denied by user: {reason}"

      approval_id = pending["approval_id"]
      stored = await store.get(approval_id)
      assert stored is not None and stored.decision_reason == reason
      rows, _ = await writer.query(owner_user_id="alice", approval_id=approval_id)
      decided_rows = [row for row in rows if row.decision_reason is not None]
      assert decided_rows and all(row.decision_reason == reason for row in decided_rows)
      event = next(entry.event for entry in reversed(event_log.entries) if entry.event["type"] == "tool_approval_decided")
      assert event["approval_id"] == approval_id
      assert event["decision_reason"] == reason
      projected = adapt_event(event, 1)
      assert projected is not None
      typed = event_from_dict(projected)
      assert isinstance(typed, ToolApprovalDecidedEvent)
      assert typed.approval_id == approval_id and typed.decision_reason == reason

      # A parked call resolves in a later turn; its decision must reach the
      # durable session log as well as that turn's live event stream.
      await dispatcher._record_durable_approval_decided(
        "call-1", "write_tool", outcome="approved" if approved else "denied",
        decision_source="user_approved" if approved else "user_denied", request=stored,
      )
      entries, _ = await session.agent_session_log.query(event_types={"tool_approval_decided"})
      assert entries[-1].event["approval_id"] == approval_id
      assert entries[-1].event["decision_reason"] == reason
    finally:
      if not call.done():
        call.cancel()
      await asyncio.gather(call, return_exceptions=True)

  asyncio.run(scenario())


@pytest.mark.parametrize("approved", [True, False])
@pytest.mark.parametrize("async_transport", [True, False])
def test_cli_transport_forwards_reason(approved: bool, async_transport: bool) -> None:
  from agent_gateway_cli.transport import (
    AsyncGatewayTransport, GatewayEndpoint, GatewaySession, GatewayTransport, MemorySessionHolder,
  )

  bodies: list[dict[str, Any]] = []

  def receive(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/chat/tool-approval"
    bodies.append(json.loads(request.content))
    return httpx.Response(200, json={"status": "ok"})

  endpoint = GatewayEndpoint(base_url="http://gateway.test", api_key="key", channel="cli", user_id="alice")
  session = GatewaySession(token="token", session_id="session", expires_at=4102444800)
  holder = MemorySessionHolder(session)
  kwargs: dict[str, Any] = dict(session=session, tool_call_id="call-1", nonce="nonce", approved=approved, reason="  checked risk  ")
  if async_transport:
    async def scenario() -> None:
      gateway = AsyncGatewayTransport(endpoint=endpoint, http_transport=httpx.MockTransport(receive))
      try:
        await gateway.submit_approval(holder, **kwargs)
      finally:
        await gateway._client.aclose()
    asyncio.run(scenario())
  else:
    gateway = GatewayTransport(endpoint=endpoint, http_transport=httpx.MockTransport(receive))
    try:
      gateway.submit_approval(holder, **kwargs)
    finally:
      gateway._client.close()
  assert bodies == [{"tool_call_id": "call-1", "nonce": "nonce", "approved": approved, "allow_tool_type": False, "reason": "checked risk"}]


@pytest.mark.parametrize("approved", [True, False])
def test_agents_mcp_forwards_reason_to_approval_post(monkeypatch: pytest.MonkeyPatch, approved: bool) -> None:
  from mcp_servers.agents_mcp import gateway_client, server

  async def scenario() -> None:
    config = SimpleNamespace()
    entry = SimpleNamespace(
      lock=asyncio.Lock(), tool_call_id="call-1", nonce="nonce",
      turn_state=SimpleNamespace(session=SimpleNamespace(session_token="token")),
    )
    captured: list[dict[str, Any]] = []

    async def post(_config: Any, path: str, **kwargs: Any) -> httpx.Response:
      assert path == "/chat/tool-approval"
      captured.append(kwargs["json_body"])
      return httpx.Response(200, json={"status": "ok"})

    async def resume(_entry: Any, _pending_id: str) -> dict[str, Any]:
      return {"status": "complete"}

    monkeypatch.setitem(gateway_client._PENDING, "pending-reason", entry)
    monkeypatch.setattr(gateway_client, "_control_post", post)
    monkeypatch.setattr(gateway_client, "_resume_pending_stream", resume)
    monkeypatch.setattr(server, "_state", lambda _ctx: SimpleNamespace(config=config))
    result = await server.agent_chat_approve(
      "pending-reason", approved, reason="  checked risk  ", ctx=cast(Any, SimpleNamespace()),
    )
    assert result == {"status": "complete"}
    assert captured == [{"tool_call_id": "call-1", "nonce": "nonce", "approved": approved, "allow_tool_type": False, "reason": "checked risk"}]

  asyncio.run(scenario())


@pytest.mark.parametrize("choice, approved", [("y", True), ("n", False)])
def test_cli_chat_prompts_for_optional_reason(choice: str, approved: bool) -> None:
  from io import StringIO
  from agent_gateway_cli.chat import DevChatCLI
  from agent_gateway_cli.transport import GatewayEndpoint, GatewaySession, GatewayTransport, MemorySessionHolder

  responses = iter([f"{choice}:  reviewed the assumptions  "])
  output = StringIO()
  cli = DevChatCLI(stdout=output, stderr=StringIO(), input_fn=lambda: next(responses), time_fn=lambda: 0.0)
  bodies: list[dict[str, Any]] = []

  def receive(request: httpx.Request) -> httpx.Response:
    bodies.append(json.loads(request.content))
    return httpx.Response(200, json={"status": "ok"})

  gateway = GatewayTransport(
    endpoint=GatewayEndpoint(base_url="http://gateway.test", api_key="key", channel="cli", user_id="alice"),
    http_transport=httpx.MockTransport(receive),
  )
  session = GatewaySession(token="token", session_id="session", expires_at=4102444800)
  try:
    cli._render_approval_request(
      gateway=gateway, holder=MemorySessionHolder(session), session=session,
      payload={"tool_name": "write_tool", "tool_call_id": "call-1", "nonce": "nonce"},
    )
  finally:
    gateway._client.close()
  assert "y: <reason>" in output.getvalue()
  assert bodies[0]["approved"] is approved
  assert bodies[0]["reason"] == "reviewed the assumptions"
