# ruff: noqa: E402

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import sys
import time
from dataclasses import fields as dataclass_fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Mapping

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import EventLog, ToolDispatcher
from agent_gateway import policy_imports
from agent_gateway import tool_dispatcher as dispatcher_module
from agent_gateway import tool_dispatcher_approval_lifecycle as lifecycle_helpers
from agent_gateway.approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest as PolicyApprovalRequest,
  ApprovalState,
  PersistentGrant,
  RunContext,
  build_approval_request,
  utc_now,
)
from agent_gateway.approval_route import (
  DurableLocalApprovalRoute,
  NoApprovalRoute,
)
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.session import GatewaySession
from agent_gateway.prepared_business_model_store import PreparedBusinessModelLifecycle
from agent_gateway.batch_approval_projection import (
  BatchApprovalProjectionRegistry,
  BatchApprovalScope,
)
from agent_gateway.single_user_policy import SingleUserApprovalPolicy
from agent_gateway.secret_boundary import SecretBoundary
from agent_gateway.skill_limits import (
  ActiveSkillAdmission,
  SkillExecutionLimits,
)
from agent_gateway.tool_dispatcher_helpers import (
  LocalToolHandler,
  PlannedWritePlanningRejected,
  ToolResult,
  TrustedToolPlan,
)
from agent_gateway.mcp_client import McpClientManager, RegisteredMcpDirectToolCall
from agent_gateway.tool_definition import LiveToolRouteBinding, OriginatedToolDefinition
from agent_gateway.tool_registration import RegisteredMcpToolDescriptor
from agent_gateway.tool_policy_registry import PlanDecision, PreparedToolCall
from agent_workflow_contracts.tool_registration import RegisteredToolIdentity


class _NullMcp(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  def is_mcp_tool(self, name: str) -> bool:
    _ = name
    return False

  def get_server_for_tool(self, name: str) -> str | None:
    _ = name
    return None

  async def call_tool(
    self,
    name: str,
    tool_input: object,
    meta: dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: Mapping[str, object] | None = None,
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
    return {"ok": True}, None


class _PrefixedMcp(_NullMcp):
  def is_mcp_tool(self, name: str) -> bool:
    return name == "trades_execute_trade"

  def get_server_for_tool(self, name: str) -> str | None:
    return "portfolio-trades-mcp" if name == "trades_execute_trade" else None

  def get_original_tool_name(self, name: str) -> str:
    return "execute_trade" if name == "trades_execute_trade" else name


class _NoMcpLookup:
  async def call_tool(self, _name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    return {"ok": True}, None


class _PlanningHandlerDouble:
  """Callable local handler with the complete optional planning-hook surface."""

  PLANNING_IDENTITY: str | None
  plan_change: Callable[..., Awaitable[object]] | None
  execute_prepared_change: Callable[..., Awaitable[ToolResult]] | None

  def __init__(
    self,
    handler: LocalToolHandler,
    *,
    planning_identity: str | None = None,
    plan_change: Callable[..., Awaitable[object]] | None = None,
    execute_prepared_change: Callable[..., Awaitable[ToolResult]] | None = None,
  ) -> None:
    self._handler = handler
    self.PLANNING_IDENTITY = planning_identity
    self.plan_change = plan_change
    self.execute_prepared_change = execute_prepared_change

  async def __call__(self, *args: object, **kwargs: object) -> ToolResult:
    return await self._handler(*args, **kwargs)


class _SessionLog:
  def __init__(self) -> None:
    self.events: list[dict[str, Any]] = []

  async def append(self, event: dict[str, Any]) -> None:
    self.events.append(dict(event))


def _request(
  *,
  state: ApprovalState = "pending_user",
  state_version: int = 0,
) -> PolicyApprovalRequest:
  return PolicyApprovalRequest(
    approval_id="approval-1",
    tool_call_id="call-1",
    parent_approval_id=None,
    approval_chain_id="approval-1",
    request_id="request-1",
    session_id="session-1",
    run_id="run-1",
    user_id="1",
    profile="chat",
    channel="tui",
    tool_name="place_order",
    tool_class="state_write",
    tool_args_redacted={"ticker": "MSFT"},
    args_hash="args-1",
    reason="needs approval",
    blast_radius_summary="state_write:place_order",
    state=state,
    requested_at=datetime(2026, 1, 1, tzinfo=UTC),
    state_version=state_version,
  )


def _decision() -> PolicyApprovalDecision:
  return PolicyApprovalDecision(
    outcome="request_user_approval",
    reason="needs approval",
    allow_persistent_grant=True,
  )


class _ApprovalWaitStoreFake:
  def __init__(
    self,
    *,
    request: PolicyApprovalRequest,
    transition_error: Exception | None = None,
    transition_request: PolicyApprovalRequest | None = None,
    transition_event: asyncio.Event | None = None,
  ) -> None:
    self.request = request
    self.transition_error = transition_error
    self.transition_request = transition_request
    self.transition_event = transition_event
    self.transitions: list[dict[str, object]] = []

  async def get(self, approval_id: str) -> PolicyApprovalRequest:
    assert approval_id == self.request.approval_id
    return self.request

  async def transition_state(
    self,
    approval_id: str,
    state: ApprovalState,
    *,
    expected_state_version: int | None = None,
    expires_at: datetime | None = None,
    decider_id: str | None = None,
    decider_role: str | None = None,
    decision: str | None = None,
    decision_reason: str | None = None,
  ) -> PolicyApprovalRequest:
    assert approval_id == self.request.approval_id
    assert state == "expired"
    assert expected_state_version == self.request.state_version
    self.transitions.append({
      "approval_id": approval_id,
      "state": state,
      "expected_state_version": expected_state_version,
      "decision_reason": decision_reason,
    })
    _ = (expires_at, decider_id, decider_role, decision)
    if self.transition_request is not None:
      self.request = self.transition_request
    if self.transition_event is not None:
      self.transition_event.set()
    if self.transition_error is not None:
      raise self.transition_error
    self.request = replace(
      self.request,
      state=state,
      state_version=self.request.state_version + 1,
      decision_reason=decision_reason,
    )
    return self.request


def test_resolve_run_context_uses_canonical_session_owner_for_approval_identity() -> None:
  session = _gateway_session(
    user_id="henry",
    owner_user_id="1",
    request_id="request-1",
    session_id="session-1",
    channel="cli",
  )

  resolved = lifecycle_helpers.resolve_run_context(
    run_context=RunContext(
      user_id="henry",
      request_id="request-1",
      session_id="session-1",
      profile="chat",
      channel="cli",
    ),
    session=session,
    user_id="henry",
    channel="cli",
    role="owner",
    session_id="session-1",
    approval_policy=SimpleNamespace(policy_bundle_hash="policy-1"),
  )

  assert resolved.user_id == "1"


















async def _wait_for_queue(session: GatewaySession, tool_call_id: str) -> asyncio.Queue:
  for _ in range(100):
    queue = session.approval_queues.get(tool_call_id)
    if queue is not None:
      return queue
    await asyncio.sleep(0)
  raise AssertionError("approval queue was not registered")


def test_projected_batch_approval_waits_through_its_advertised_expiry() -> None:
  calls: list[float | int | None] = []

  def globally_capped(expiry_seconds: float | int | None) -> float:
    calls.append(expiry_seconds)
    return 270.0

  assert lifecycle_helpers._approval_wait_timeout_seconds(
    600,
    batch_admission=object(),
    approval_queue_timeout_seconds_fn=globally_capped,
  ) == 600.0
  assert calls == []


def test_non_batch_approval_keeps_global_wait_ceiling() -> None:
  assert lifecycle_helpers._approval_wait_timeout_seconds(
    600,
    batch_admission=None,
    approval_queue_timeout_seconds_fn=lambda _expiry: 270.0,
  ) == 270.0


def test_durable_interactive_approval_event_carries_stable_id_and_cleans_up() -> None:
  async def scenario() -> None:
    session_log = _SessionLog()
    session = _gateway_session(pending_tools={}, approval_queues={}, agent_session_log=session_log)
    event_log = EventLog()
    request = _request()
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        approval_store=None,
        append_event_fn=event_log.append,
        request=request,
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="qual-1",
        allow_persistent=True,
        timeout_seconds=5,
        log=logging.getLogger("test"),
      )
    )

    queue = await _wait_for_queue(session, request.tool_call_id)
    assert session.pending_tools[request.tool_call_id] == {
      "approval_id": "approval-1",
      "nonce": "nonce-1",
      "requested_at": session.pending_tools[request.tool_call_id]["requested_at"],
      "status": "approval_pending",
      "tool_name": "place_order",
      "resolved_qualifier": "qual-1",
    }
    approval_event = event_log.entries[0].event
    assert approval_event["type"] == "tool_approval_request"
    assert str(approval_event["approval_id"]).strip()
    assert approval_event["approval_id"] == request.approval_id
    assert session_log.events[0]["approval_id"] == approval_event["approval_id"]
    assert session_log.events[0]["allow_persistent_approval"] is True

    await queue.put({"approved": True, "allow_tool_type": False})

    assert await task == {"approved": True, "allow_tool_type": False}
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_pending_tool_exposes_only_trusted_planned_change_projection() -> None:
  async def scenario() -> None:
    session = _gateway_session(pending_tools={}, approval_queues={}, agent_session_log=None)
    event_log = EventLog()
    planned_change = {
      "schema_version": "planned-change-review.v1",
      "change_set_id": "change-set-1",
      "change_hash": "a" * 64,
      "intent": {"subcommand": "persist_business_model"},
      "target": {"ticker": "MSFT", "research_file_id": 1},
    }
    request = replace(
      _request(),
      tool_name="fms_persist_business_model",
      tool_args_redacted={
        "judgment": {"ticker": "MSFT", "large": "x" * 10_000},
        "planned_change": planned_change,
      },
      blast_radius_summary="state_write:fms_persist_business_model",
    )
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        approval_store=None,
        append_event_fn=event_log.append,
        request=request,
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="qual-1",
        allow_persistent=False,
        timeout_seconds=5,
        log=logging.getLogger("test"),
      )
    )

    queue = await _wait_for_queue(session, request.tool_call_id)
    pending = session.pending_tools[request.tool_call_id]
    assert pending["planned_change"] == planned_change
    assert "tool_input" not in pending
    assert "judgment" not in pending
    planned_change["target"]["ticker"] = "DRIFT"
    assert pending["planned_change"]["target"]["ticker"] == "MSFT"

    await queue.put({"approved": False, "allow_tool_type": False})
    assert await task == {"approved": False, "allow_tool_type": False}

  asyncio.run(scenario())


def test_projected_pending_tool_binds_stage_identity_to_projection_and_event() -> None:
  async def scenario() -> None:
    session_log = _SessionLog()
    session = _gateway_session(
      pending_tools={},
      approval_queues={},
      agent_session_log=session_log,
      batch_stage_run_seq=3,
    )
    event_log = EventLog()
    request = _request()

    class Admission:
      def publish_pending(self) -> None:
        session.approval_queues[request.tool_call_id].put_nowait({"approved": False})

    result = await lifecycle_helpers.await_user_approval_via_pending_tools(
      session=session,
      approval_store=None,
      append_event_fn=event_log.append,
      request=request,
      decision=_decision(),
      nonce="nonce-1",
      resolved_qualifier="qual-1",
      allow_persistent=False,
      timeout_seconds=5,
      log=logging.getLogger("test"),
      batch_admission=Admission(),
    )

    assert result == {"approved": False}
    assert event_log.entries[0].event["stage_run_seq"] == 3
    assert session_log.events[0]["stage_run_seq"] == 3
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


@pytest.mark.parametrize("stage_run_seq", [None, 0, -1, True, "3"])
def test_projected_pending_tool_rejects_invalid_stage_identity(
  stage_run_seq: object,
) -> None:
  session = _gateway_session(
    pending_tools={},
    approval_queues={},
    batch_stage_run_seq=stage_run_seq,
  )

  with pytest.raises(
    ValueError,
    match="stage_run_seq must be a positive integer",
  ):
    asyncio.run(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        approval_store=None,
        append_event_fn=None,
        request=_request(),
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="",
        allow_persistent=False,
        timeout_seconds=5,
        log=logging.getLogger("test"),
        batch_admission=object(),
      )
    )
  assert session.pending_tools == {}
  assert session.approval_queues == {}


def test_pending_tool_helper_expires_store_request_on_timeout() -> None:

  async def scenario() -> None:
    store = _ApprovalWaitStoreFake(request=_request(state_version=7))
    session = _gateway_session(pending_tools={}, approval_queues={}, agent_session_log=None)

    result = await lifecycle_helpers.await_user_approval_via_pending_tools(
      session=session,
      approval_store=store,
      append_event_fn=None,
      request=_request(),
      decision=_decision(),
      nonce="nonce-1",
      resolved_qualifier="",
      allow_persistent=False,
      timeout_seconds=0,
      log=logging.getLogger("test"),
    )

    assert result is None
    assert store.transitions == [
      {
        "approval_id": "approval-1",
        "state": "expired",
        "expected_state_version": 7,
        "decision_reason": "Timed out waiting for user approval",
      }
    ]
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_pending_tool_timeout_observes_concurrent_durable_vote() -> None:
  async def scenario() -> None:
    vote_won = asyncio.Event()
    session = _gateway_session(
      pending_tools={},
      approval_queues={},
      agent_session_log=None,
    )


    store = _ApprovalWaitStoreFake(
      request=_request(state_version=7),
      transition_error=RuntimeError("approval request state_version changed"),
      transition_request=_request(state="approved", state_version=8),
      transition_event=vote_won,
    )
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        approval_store=store,
        append_event_fn=None,
        request=_request(),
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="",
        allow_persistent=False,
        timeout_seconds=0,
        log=logging.getLogger("test"),
      )
    )

    await asyncio.wait_for(vote_won.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not task.done()
    assert "call-1" in session.pending_tools
    approval_queue = session.approval_queues["call-1"]
    approval_queue.put_nowait({
      "approved": True,
      "allow_tool_type": False,
      "approval_id": "approval-1",
    })

    assert await asyncio.wait_for(task, timeout=1) == {
      "approved": True,
      "allow_tool_type": False,
      "approval_id": "approval-1",
    }
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_pending_tool_timeout_rejects_mismatched_vote_delivery() -> None:
  async def scenario() -> None:
    session = _gateway_session(
      pending_tools={},
      approval_queues={},
      agent_session_log=None,
    )


    store = _ApprovalWaitStoreFake(
      request=_request(state_version=7),
      transition_error=RuntimeError("approval request state_version changed"),
      transition_request=_request(state="denied", state_version=8),
    )
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        approval_store=store,
        append_event_fn=None,
        request=_request(),
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="",
        allow_persistent=False,
        timeout_seconds=0,
        log=logging.getLogger("test"),
      )
    )

    queue = await _wait_for_queue(session, "call-1")
    await asyncio.sleep(0.11)
    queue.put_nowait({
      "approved": False,
      "allow_tool_type": False,
      "approval_id": "different-approval",
    })

    with pytest.raises(
      RuntimeError,
      match="different approval",
    ):
      await asyncio.wait_for(task, timeout=1)
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_pending_tool_timeout_bounds_missing_winner_delivery(
  monkeypatch,
) -> None:
  async def scenario() -> None:
    session = _gateway_session(
      pending_tools={},
      approval_queues={},
      agent_session_log=None,
    )


    store = _ApprovalWaitStoreFake(
      request=_request(state="approved", state_version=8),
    )
    monkeypatch.setattr(
      lifecycle_helpers,
      "_APPROVAL_WINNER_DELIVERY_TIMEOUT_SECONDS",
      0.01,
    )
    with pytest.raises(
      RuntimeError,
      match="reconciliation deadline",
    ):
      await lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        approval_store=store,
        append_event_fn=None,
        request=_request(),
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="",
        allow_persistent=False,
        timeout_seconds=0,
        log=logging.getLogger("test"),
      )
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())










def _gateway_session(**overrides: Any) -> GatewaySession:
  """A real GatewaySession: the route the ledger row binds to is type-exact."""

  now = time.time()
  defaults: dict[str, Any] = {
    "session_id": "sess-1",
    "api_key_hash": "hash",
    "created_at": now,
    "expires_at": now + 600,
    "user_id": "alice",
    "channel": "web",
    "role": "owner",
  }
  field_names = {f.name for f in dataclass_fields(GatewaySession)}
  init_kwargs = {**defaults, **{k: v for k, v in overrides.items() if k in field_names}}
  session = GatewaySession(**init_kwargs)
  for key, value in overrides.items():
    if key not in field_names:
      setattr(session, key, value)
  return session




def _exact_staged_workbook_execution(
  *,
  target_hash: str,
  operations: list[dict[str, Any]],
) -> dict[str, Any]:
  workbook_bytes = b"exact staged workbook"
  sidecar_bytes = b"sidecar"
  return {
    "execution_kind": "canonical_normal_workbook_bundle_v1",
    "workbook_content_base64": base64.b64encode(workbook_bytes).decode("ascii"),
    "workbook_content_sha256": hashlib.sha256(workbook_bytes).hexdigest(),
    "sidecar_content_base64": base64.b64encode(sidecar_bytes).decode("ascii"),
    "sidecar_content_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
    "workbook_source_target_hash": target_hash,
    "compute_engine_version": "engine-v1",
    "mutation": {
      "operations": operations,
      "force_overwrite": True,
      "refresh_schema_cache": False,
    },
    "expected_readback": {"gross_margin": {"2027": 0.42}},
  }




def _planned_session() -> GatewaySession:
  return _gateway_session(
    session_id="sess-1",
    user_id="alice",
    channel="web",
    role="owner",
    pending_tools={},
    approval_queues={},
  )




_DEFAULT_BUSINESS_MODEL_RESTORE = object()


























































@pytest.mark.parametrize(
  ("session_cache_approved", "automatic_approval_reason"),
  [
    (True, None),
    (False, "headless hook allowed"),
    (False, None),
  ],
)
def test_fresh_owner_lifecycle_blocks_every_automatic_source(
  tmp_path: Path,
  session_cache_approved: bool,
  automatic_approval_reason: str | None,
) -> None:
  class AutoPolicy:
    policy_id = "auto-policy"
    policy_version = "1"

    def __init__(self) -> None:
      self.decide_calls = 0
      self.resolved: list[str] = []

    async def decide(self, **_kwargs: Any) -> PolicyApprovalDecision:
      self.decide_calls += 1
      return PolicyApprovalDecision(
        outcome="auto_approve",
        reason="custom automatic policy",
        allow_persistent_grant=True,
        persistent_grant_scope_hint="state_write:apply_proposal_series",
        grant_reference="grant-should-not-survive",
      )

    async def on_resolve(self, *, request: Any) -> None:
      self.resolved.append(request.approval_id)

  async def fail_if_prompted(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("headless fresh-owner lifecycle must not prompt")

  store = SQLiteApprovalStore(tmp_path / "fresh-owner.sqlite3")
  policy = AutoPolicy()
  session = _gateway_session()
  result = asyncio.run(
    lifecycle_helpers.run_approval_lifecycle(
      route=DurableLocalApprovalRoute(store, policy, session),
      session=session,
      tool_call_id="promotion-call",
      tool_name="apply_proposal_series",
      tool_input={"proposal_ids": ["proposal-1"]},
      qualifier="",
      reason="promotion",
      allow_persistent=True,
      approval_constraint="fresh_human_owner",
      required_owner_user_id="owner-1",
      session_cache_approved=session_cache_approved,
      automatic_approval_reason=automatic_approval_reason,
      deny_user_prompt=True,
      resolve_run_context_fn=lambda: RunContext(
        user_id="owner-1",
        request_id="request-1",
        decider_role="owner",
      ),
      current_skill_admission_fn=lambda: None,
      redact_for_approval_request_fn=lambda *_args: ({}, "args-hash"),
      resolve_tool_class_fn=lambda _tool_name: "state_write",
      effective_trade_approval_decision_fn=lambda _name, _args, decision: decision,
      await_user_approval_via_pending_tools_fn=fail_if_prompted,
      approval_queue_timeout_seconds_fn=lambda _expiry: 1.0,
    )
  )

  request = result["request"]
  assert result["approved"] is False
  assert result["allow_tool_type"] is False
  assert request.state == "auto_denied"
  assert request.authorization_mode == "HUMAN"
  assert request.grant_reference is None
  assert request.persistent_grant_scope is None
  assert request.approval_constraint == "fresh_human_owner"
  assert request.required_owner_user_id == "owner-1"
  assert policy.decide_calls == 1
  assert policy.resolved == [request.approval_id]


def test_native_approval_refuses_inline_limit_mismatch_before_policy(
  tmp_path: Path,
) -> None:
  calls: list[str] = []

  class Policy:
    async def decide(self, **_kwargs: Any) -> PolicyApprovalDecision:
      calls.append("policy")
      raise AssertionError("mismatched admission must precede policy")

  limits = SkillExecutionLimits(20, 32_000, 20.0)
  with pytest.raises(ValueError, match="do not match"):
    asyncio.run(
      lifecycle_helpers.run_approval_lifecycle(
        route=DurableLocalApprovalRoute(
          SQLiteApprovalStore(tmp_path / "admission-mismatch.sqlite3"),
          Policy(),
          _gateway_session(),
        ),
        session=_gateway_session(),
        tool_call_id="call-mismatch",
        tool_name="file_write",
        tool_input={"path": "x"},
        qualifier="",
        reason="write",
        allow_persistent=False,
        resolve_run_context_fn=lambda: RunContext(
          user_id="owner-1",
          request_id="request-1",
          skill="quant-research",
          admitted_skill_execution_limits=limits,
        ),
        current_skill_admission_fn=lambda: ActiveSkillAdmission(
          "quant-research",
          SkillExecutionLimits(19, 32_000, 20.0),
        ),
        redact_for_approval_request_fn=lambda *_args: (_ for _ in ()).throw(
          AssertionError("mismatch must precede redaction")
        ),
        resolve_tool_class_fn=lambda _tool_name: "state_write",
        effective_trade_approval_decision_fn=(
          lambda _name, _args, decision: decision
        ),
        await_user_approval_via_pending_tools_fn=lambda *_args: None,
        approval_queue_timeout_seconds_fn=lambda _expiry: 1.0,
      )
    )

  assert calls == []


