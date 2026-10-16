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
from typing import Any, Awaitable, Callable, Literal, Mapping

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog, ToolDispatcher
from agent_gateway import policy_imports
from agent_gateway import tool_dispatcher as dispatcher_module
from agent_gateway import tool_dispatcher_approval_lifecycle as lifecycle_helpers
from agent_gateway.admission import (
  AdmittedContext, ContinuationOwners, ContinueRequest, admit, admit_root_turn,
  capture_job_context, capture_root_turn, context_for_child, current_parent_context,
  current_root_turn, record_root_turn_admission, root_context,
)
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.approvals import _approval_request_to_dict
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.capability_execution import CapabilityExecutionResolver
from agent_gateway.execution_identity import DispatchIdentity
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent
from agent_gateway.skills import SkillLoader
from agent_gateway.sub_agent import make_run_agent_handler
from gateway_test_support.capability_execution_test_support import stub_capability_execution_resolver
from gateway_test_support.host_policy import owner_session_host_policy
from gateway_test_support.approval_policy_test_support import FixedOutcomeApprovalPolicy
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
  NoApprovalRoute,
  DurableLocalApprovalRoute,
  InProcessDelegatedApprovalRoute,
  ParentDelegatedApprovalRoute,
  child_approval_route,
)
from agent_gateway.autonomous_approval_channel import (
  AutonomousApprovalChannelChild,
)
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


def test_durable_interactive_approval_event_carries_stable_id_and_cleans_up() -> None:
  async def scenario() -> None:
    session_log = _SessionLog()
    session = _gateway_session(pending_tools={}, approval_queues={}, agent_session_log=session_log)
    event_log = EventLog()
    request = _request()
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        append_event_fn=event_log.append,
        request=request,
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="qual-1",
        allow_persistent=True,
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
        append_event_fn=event_log.append,
        request=request,
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="qual-1",
        allow_persistent=False,
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
      append_event_fn=event_log.append,
      request=request,
      decision=_decision(),
      nonce="nonce-1",
      resolved_qualifier="qual-1",
      allow_persistent=False,
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
        append_event_fn=None,
        request=_request(),
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="",
        allow_persistent=False,
        log=logging.getLogger("test"),
        batch_admission=object(),
      )
    )
  assert session.pending_tools == {}
  assert session.approval_queues == {}


class _AdvancingClockLoop(asyncio.SelectorEventLoop):
  """An event loop whose clock the test moves forward without sleeping.

  Every asyncio timer is scheduled against ``loop.time()``, so advancing the
  offset fires any deadline a wait would have set as of that elapsed time.
  """

  def __init__(self) -> None:
    super().__init__()
    self.offset = 0.0

  def time(self) -> float:
    return super().time() + self.offset


def _run_on_advancing_clock(
  scenario: Callable[[_AdvancingClockLoop], Awaitable[None]],
) -> None:
  loop = _AdvancingClockLoop()
  try:
    loop.run_until_complete(scenario(loop))
  finally:
    loop.close()


def test_late_vote_reaches_the_live_wait_after_hours_elapse() -> None:
  async def scenario(loop: _AdvancingClockLoop) -> None:
    session = _gateway_session(pending_tools={}, approval_queues={}, agent_session_log=None)
    request = _request()
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        append_event_fn=None,
        request=request,
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="",
        allow_persistent=False,
        log=logging.getLogger("test"),
      )
    )
    queue = await _wait_for_queue(session, request.tool_call_id)

    loop.offset += 6 * 60 * 60
    for _ in range(3):
      await asyncio.sleep(0)

    assert not task.done()
    assert session.approval_queues[request.tool_call_id] is queue
    queue.put_nowait({"approved": True, "allow_tool_type": False, "approval_id": request.approval_id})
    assert await task == {"approved": True, "allow_tool_type": False, "approval_id": request.approval_id}
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  _run_on_advancing_clock(scenario)


def test_cancelled_wait_releases_the_live_entry_and_leaves_no_decision() -> None:
  async def scenario() -> None:
    session = _gateway_session(pending_tools={}, approval_queues={}, agent_session_log=None)
    request = _request()
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        append_event_fn=None,
        request=request,
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="",
        allow_persistent=False,
        log=logging.getLogger("test"),
      )
    )
    await _wait_for_queue(session, request.tool_call_id)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_wait_cancelled_during_publication_releases_the_live_entry() -> None:
  async def scenario() -> None:
    published = asyncio.Event()

    class _BlockingSessionLog:
      async def append(self, _event: dict[str, Any]) -> None:
        published.set()
        await asyncio.Event().wait()

    session = _gateway_session(
      pending_tools={},
      approval_queues={},
      agent_session_log=_BlockingSessionLog(),
    )
    request = _request()
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        append_event_fn=None,
        request=request,
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="",
        allow_persistent=False,
        log=logging.getLogger("test"),
      )
    )
    await published.wait()
    assert request.tool_call_id in session.pending_tools

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_delegated_money_wait_is_not_bounded_by_elapsed_time() -> None:
  async def scenario(loop: _AdvancingClockLoop) -> None:
    session = _gateway_session(pending_tools={}, approval_queues={}, agent_session_log=None)
    tool_call_id = "call-delegated"
    projected: list[dict[str, Any]] = []

    async def wait(
      request: PolicyApprovalRequest,
      decision: PolicyApprovalDecision,
      **kwargs: Any,
    ) -> dict[str, Any]:
      projected.append(dict(kwargs["durable_request"]))
      assert decision.expiry_seconds is None
      return await lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        append_event_fn=None,
        request=request,
        decision=decision,
        log=logging.getLogger("test"),
        **kwargs,
      )

    task = asyncio.create_task(
      lifecycle_helpers.run_approval_lifecycle(
        route=ParentDelegatedApprovalRoute(
          object.__new__(AutonomousApprovalChannelChild),
          session,
        ),
        session=session,
        producer_session=session,
        executable_capture=None,
        tool_call_id=tool_call_id,
        tool_name="place_order",
        tool_input={"ticker": "MSFT"},
        qualifier="",
        reason="live order",
        allow_persistent=False,
        resolve_run_context_fn=lambda: RunContext(
          user_id="owner-1",
          request_id="control-run-1",
          profile="analyst",
          channel="cli",
          decider_role="owner",
        ),
        current_skill_admission_fn=lambda: None,
        redact_for_approval_request_fn=lambda *_args: ({}, "args-hash"),
        resolve_tool_class_fn=lambda _tool_name: "irreversible",
        effective_trade_approval_decision_fn=(
          lambda _name, _args, decision: decision
        ),
        await_user_approval_via_pending_tools_fn=wait,
      )
    )
    queue = await _wait_for_queue(session, tool_call_id)
    assert projected[0]["expires_at"] is None

    loop.offset += 2 * 60 * 60
    for _ in range(3):
      await asyncio.sleep(0)

    assert not task.done()
    approval_id = projected[0]["approval_id"]
    queue.put_nowait({"approved": True, "allow_tool_type": False, "approval_id": approval_id})
    result = await task
    assert result["approved"] is True
    assert result["request"].state == "approved"
    assert result["request"].expires_at is None
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  _run_on_advancing_clock(scenario)









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
  ("tool_class", "expected_state", "expected_mode"),
  [
    ("state_write", "auto_approved", "OWNER_CONTROL_PLANE"),
    ("irreversible", "denied", "HUMAN"),
  ],
)
def test_delegated_run_settles_everything_but_the_money_boundary(
  tool_class: str,
  expected_state: str,
  expected_mode: str,
) -> None:
  """An unattended run settles its own writes; a live order still asks.

  The delegated route is an autonomous child's only route and no operator is
  guaranteed to sit at it, so every class outside the money boundary is settled
  under the run owner's authority. Placing an order is the one class that still
  waits for the operator's decision.
  """

  session = _gateway_session()
  prompts: list[str] = []

  async def record_prompt(
    request: PolicyApprovalRequest,
    *_args: Any,
    **_kwargs: Any,
  ) -> dict[str, Any]:
    prompts.append("prompted")
    return {"approval_id": request.approval_id, "approved": False}

  result = asyncio.run(
    lifecycle_helpers.run_approval_lifecycle(
      route=ParentDelegatedApprovalRoute(
        object.__new__(AutonomousApprovalChannelChild),
        session,
      ),
      session=session,
      producer_session=session,
      executable_capture=None,
      tool_call_id="call-idea-triage",
      tool_name="idea_triage_decide",
      tool_input={"decisions": [{"id": 1, "verdict": "dismiss"}]},
      qualifier="",
      reason="triage verdict",
      allow_persistent=False,
      resolve_run_context_fn=lambda: RunContext(
        user_id="owner-1",
        request_id="control-run-1",
        profile="analyst",
        channel="cli",
        decider_role="owner",
      ),
      current_skill_admission_fn=lambda: None,
      redact_for_approval_request_fn=lambda *_args: ({}, "args-hash"),
      resolve_tool_class_fn=lambda _tool_name: tool_class,
      effective_trade_approval_decision_fn=(
        lambda _name, _args, decision: decision
      ),
      await_user_approval_via_pending_tools_fn=record_prompt,
    )
  )

  request = result["request"]
  assert request.state == expected_state
  assert request.authorization_mode == expected_mode
  if expected_state == "auto_approved":
    assert result["approved"] is True
    assert request.decider_id == "owner-1"
    assert request.decider_role == "owner"
    assert prompts == []
  else:
    assert result["approved"] is False
    assert prompts == ["prompted"]





def _native_capture_dispatcher(
  tmp_path: Path,
  *,
  outcome: Literal["request_user_approval", "auto_approve", "auto_deny"] = "request_user_approval",
  delegated: bool = False,
  admitted: bool = True,
  credential_generation_ref: str | None = "anthropic-generation-original",
) -> tuple[ToolDispatcher, SQLiteApprovalStore, GatewaySession]:
  store = SQLiteApprovalStore(tmp_path / "native-capture.sqlite3")
  parent = _gateway_session()
  producer = _gateway_session(session_id="child-session") if delegated else parent
  # Mutable session configuration deliberately disagrees with the frozen bind.
  producer.auth_config = {
    "credential_source": {"generation_ref": "anthropic-generation-session-rotated"},
  }
  ledger = DurableLocalApprovalRoute(store, FixedOutcomeApprovalPolicy(outcome), parent)
  route = InProcessDelegatedApprovalRoute(ledger) if delegated else ledger
  producer.approval_route = route
  if admitted:
    context = root_context(
      producer,
      withheld_tools=frozenset(),
      research_file_id=None,
    )
    capture_job_context(
      producer,
      replace(context, job_id="job-child" if delegated else "job-native"),
    )

  async def handler(_input: dict[str, Any], **_kwargs: Any) -> ToolResult:
    return {"ok": True}, None

  dispatcher = ToolDispatcher(
    capability_bind=stub_capability_execution_resolver().resolve("session.driver").bind.model_copy(
      update={"credential_ref": credential_generation_ref or "service:openai"},
    ),
    mcp_client=_NullMcp(),
    local_tool_handlers={"native_write": handler},
    needs_approval=lambda *_args: True,
    session=producer,
    role="owner",
    approval_route=route,
    run_context=RunContext(user_id="alice", request_id="capture-request"),
  )
  dispatcher.bind_secret_boundary(SecretBoundary(("raw-original-operand", "private-root-operand")))
  return dispatcher, store, parent


async def _wait_for_native_queue(
  session: GatewaySession,
  task: asyncio.Task[Any],
  tool_call_id: str,
) -> asyncio.Queue:
  async def wait_for_publication() -> asyncio.Queue:
    while tool_call_id not in session.approval_queues:
      if task.done():
        raise AssertionError(f"Native call exited before parking: {task.result()}")
      await asyncio.sleep(0.01)
    return session.approval_queues[tool_call_id]

  return await asyncio.wait_for(wait_for_publication(), timeout=10)


@pytest.mark.parametrize("advertised", [None, frozenset({"z_tool", "native_write", "a_tool"})])
@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("credential_generation_ref", [None, "anthropic-generation-original"])
def test_native_park_captures_exact_call_and_dispatch_operands(
  tmp_path: Path,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
  advertised: frozenset[str] | None,
  delegated: bool,
  credential_generation_ref: str | None,
) -> None:
  async def scenario() -> None:
    dispatcher, store, parent = _native_capture_dispatcher(
      tmp_path,
      delegated=delegated,
      credential_generation_ref=credential_generation_ref,
    )
    original = PreparedToolCall(
      {"ticker": "MSFT", "api_key": "raw-original-operand"},
      exact_backend="original-backend",
    )
    recorded_calls: list[PreparedToolCall] = []
    park_binding = store.park_executable_binding

    async def observe_binding(**kwargs: Any) -> str:
      recorded_calls.append(kwargs["prepared_call"])
      return await park_binding(**kwargs)

    monkeypatch.setattr(store, "park_executable_binding", observe_binding)
    with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
      await dispatcher.dispatch_prepared(
        "call-capture",
        "native_write",
        original,
        call_index=7,
        advertised_tool_names=advertised,
        skill_run_id="original-skill",
        step_id="original-step",
        workspace_dir="/original/workspace",
        batch_id="original-batch",
        capture_readable_resource_snapshot=True,
        allow_uncertain_mcp_replay=False,
        abort_event=asyncio.Event(),
        on_executed_prepared_call=lambda _call: None,
      )
    park = raised.value
    job_id = "job-child" if delegated else "job-native"
    assert (park.job_id, park.original_tool_call_id, park.root_turn_id) == (
      job_id, "call-capture", None,
    )
    assert len(recorded_calls) == 1 and recorded_calls[0] is original
    assert parent.pending_tools == {} and parent.approval_queues == {}
    binding, request = await store._read_executable_binding_for_executor(
      job_id=job_id,
      tool_call_id="call-capture",
      nonce=park.nonce,
      tenant_storage_id="alice",
    )
    assert request.state == "pending_user"
    assert binding.approval_id == park.approval_id
    assert binding.prepared_input == original.prepared_input
    assert binding.exact_backend == original.exact_backend
    assert binding.credential_generation_ref == credential_generation_ref
    assert binding.dispatch_context == PreparedToolCall({
      "call_index": 7,
      "advertised_tool_names": sorted(advertised) if advertised is not None else None,
      "skill_run_id": "original-skill",
      "step_id": "original-step",
      "workspace_dir": "/original/workspace",
      "batch_id": "original-batch",
      "capture_readable_resource_snapshot": True,
      "allow_uncertain_mcp_replay": False,
    }).prepared_input
    projection = _approval_request_to_dict(request)
    assert "raw-original-operand" not in json.dumps(projection)
    assert "prepared_input" not in projection
    assert "dispatch_context" not in projection
    assert "original-backend" not in json.dumps(projection)

  asyncio.run(scenario())


@pytest.mark.parametrize("with_writer", [False, True])
@pytest.mark.parametrize("exact_backend", [None, "private-backend"])
@pytest.mark.parametrize("generation_ref", [None, "anthropic-generation-original"])
def test_detached_park_sends_private_capture_before_public_approval(
  tmp_path: Path,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
  with_writer: bool,
  exact_backend: str | None,
  generation_ref: str | None,
) -> None:
  async def scenario() -> None:
    dispatcher, store, session = _native_capture_dispatcher(
      tmp_path, credential_generation_ref=generation_ref,
    )
    capture_job_context(
      session, replace(current_parent_context(session), job_id="job-detached"),
    )
    captures: list[dict[str, Any]] = []
    order: list[str] = []
    event_log = EventLog()
    session_log = AgentSessionLog(path=tmp_path / "child-events.jsonl")
    session_events: list[dict[str, Any]] = []
    append_session_event = session_log.append

    async def observe_session_event(event: dict[str, Any]):
      session_events.append(dict(event))
      return await append_session_event(event)

    monkeypatch.setattr(session_log, "append", observe_session_event)
    session.agent_session_log = session_log
    dispatcher._event_log = event_log

    async def write_private_capture(payload: dict[str, Any]) -> None:
      assert session.pending_tools == {}
      assert event_log.entries == []
      assert session_events == []
      captures.append(payload)
      order.append("private")
      await asyncio.sleep(0)

    route = ParentDelegatedApprovalRoute(
      object.__new__(AutonomousApprovalChannelChild),
      session,
      private_capture_writer=write_private_capture if with_writer else None,
    )
    session.approval_route = route
    dispatcher._approval_route = route
    monkeypatch.setattr(dispatcher, "_resolve_tool_class", lambda _name: "irreversible")
    append_event = dispatcher._append_event

    def observe_public(event: dict[str, Any]) -> Any:
      if event["type"] == "tool_approval_request":
        assert len(captures) == int(with_writer)
        order.append("public")
      return append_event(event)

    monkeypatch.setattr(dispatcher, "_append_event", observe_public)
    original = PreparedToolCall(
      {"ticker": "MSFT", "api_key": "raw-original-operand"},
      exact_backend=exact_backend,
    )
    task = asyncio.create_task(dispatcher.dispatch_prepared(
      "call-detached",
      "native_write",
      original,
      call_index=7,
      advertised_tool_names=frozenset({"z_tool", "native_write", "a_tool"}),
      skill_run_id="private-skill",
      step_id="private-step",
      workspace_dir="/private/workspace",
      batch_id="private-batch",
      capture_readable_resource_snapshot=True,
      allow_uncertain_mcp_replay=False,
    ))
    queue: asyncio.Queue | None = None
    if with_writer:
      with pytest.raises(lifecycle_helpers.ApprovalPark) as parked:
        await task
      pending = {
        "approval_id": parked.value.approval_id,
        "nonce": parked.value.nonce,
      }
      assert session.pending_tools == {}
      assert session.approval_queues == {}
    else:
      queue = await _wait_for_native_queue(session, task, "call-detached")
      pending = session.pending_tools["call-detached"]
    assert order == (["private", "public"] if with_writer else ["public"])
    if with_writer:
      assert captures == [{
        "approval_id": pending["approval_id"],
        "tool_call_id": "call-detached",
        "nonce": pending["nonce"],
        "tool_name": "native_write",
        "prepared_input": original.materialize_input(),
        "exact_backend": exact_backend,
        "dispatch_context": {
          "call_index": 7,
          "advertised_tool_names": ["a_tool", "native_write", "z_tool"],
          "skill_run_id": "private-skill",
          "step_id": "private-step",
          "workspace_dir": "/private/workspace",
          "batch_id": "private-batch",
          "capture_readable_resource_snapshot": True,
          "allow_uncertain_mcp_replay": False,
        },
        "job_id": "job-detached",
        "tenant_storage_id": "alice",
        "credential_generation_ref": generation_ref,
        "session_log_path": str(session_log.path),
        "pack_cursor": None,
      }]
    else:
      assert captures == []
    public = json.dumps({
      "events": [entry.event for entry in event_log.entries],
      "session_log": session_events,
      "pending_tools": session.pending_tools,
    })
    for private_value in (
      "raw-original-operand", "private-backend", "prepared_input",
      "dispatch_context", "private-skill", "private-step", "/private/workspace",
      "private-batch", "credential_generation_ref", "anthropic-generation-original",
    ):
      assert private_value not in public
    with store._connection() as conn:
      assert conn.execute("SELECT COUNT(*) FROM approval_executable_bindings").fetchone()[0] == 0
    if not with_writer:
      assert queue is not None
      await queue.put({
        "approval_id": pending["approval_id"],
        "approved": False,
        "allow_tool_type": False,
      })
      result, error = await task
      assert result is None
      assert error is not None and error["code"] == "user_denied"

  asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["auto_approve", "auto_deny"])
def test_native_automatic_decisions_record_no_executable_binding(
  tmp_path: Path,
  outcome: Literal["auto_approve", "auto_deny"],
  owner_session_host_policy: Any,
) -> None:
  dispatcher, store, parent = _native_capture_dispatcher(tmp_path, outcome=outcome)
  result, error = asyncio.run(dispatcher.dispatch_prepared(
    "call-automatic",
    "native_write",
    PreparedToolCall({"ticker": "MSFT"}, exact_backend="original-backend"),
  ))
  assert (error is None) == (outcome == "auto_approve")
  assert (result is not None) == (outcome == "auto_approve")
  assert parent.pending_tools == {}
  with store._connection() as conn:
    assert conn.execute("SELECT COUNT(*) FROM approval_executable_bindings").fetchone()[0] == 0


@pytest.mark.parametrize("missing_context", ["job", "tenant"])
def test_native_park_without_admitted_job_observes_and_continues(
  tmp_path: Path,
  owner_session_host_policy: Any,
  caplog: pytest.LogCaptureFixture,
  missing_context: str,
) -> None:
  async def scenario() -> None:
    dispatcher, store, parent = _native_capture_dispatcher(
      tmp_path, admitted=missing_context == "tenant",
    )
    if missing_context == "tenant":
      capture_job_context(parent, replace(current_parent_context(parent), tenant=None))
    else:
      assert parent.admitted_job_context is None
      assert dispatcher._executing_context() is None
    task = asyncio.create_task(dispatcher.dispatch_prepared(
      "call-no-job",
      "native_write",
      PreparedToolCall({"ticker": "MSFT"}),
    ))
    queue = await _wait_for_native_queue(parent, task, "call-no-job")
    assert parent.pending_tools["call-no-job"]["status"] == "approval_pending"
    with store._connection() as conn:
      assert conn.execute("SELECT COUNT(*) FROM approval_executable_bindings").fetchone()[0] == 0
    await queue.put({"approved": False, "allow_tool_type": False})
    result, error = await task
    assert result is None
    assert error is not None and error["code"] == "user_denied"
    warnings = [record for record in caplog.records if "approval_capture_without_job_context" in record.message]
    assert len(warnings) == 1
    assert "native_write" in warnings[0].message
    assert "call-no-job" in warnings[0].message
    assert f"missing_job_id={missing_context == 'job'}" in warnings[0].message
    assert "missing_tenant_storage_id=True" in warnings[0].message

  asyncio.run(scenario())


class _NativeChildCaptureProvider(ModelProvider):
  name = "openai"

  def __init__(self) -> None:
    self.turn = 0

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    return None

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return kwargs

  async def stream(self, client: Any, params: dict[str, Any]):
    self.turn += 1
    if self.turn == 1:
      assert "web_search" in {definition["name"] for definition in params["tools"]}
      tool_input = {"query": "exact child operand"}
      yield StreamEvent(type="tool_use_start", tool_id="call-real-child", tool_name="web_search")
      yield StreamEvent(
        type="tool_use_end",
        tool_id="call-real-child",
        tool_name="web_search",
        tool_input=tool_input,
        raw_block={
          "type": "tool_use",
          "id": "call-real-child",
          "name": "web_search",
          "input": tool_input,
        },
      )
      yield StreamEvent(type="message_end", stop_reason="tool_use")
      return
    text = "Child finished after the approval decision."
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def test_real_in_process_child_capture_uses_child_job_not_parent_session(
  tmp_path: Path,
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  owner_session_host_policy.get_read_capability = lambda name, _server: (
    "web.read/v1" if name == "web_search" else None
  )

  async def scenario() -> None:
    parent = _gateway_session(
      session_id="parent-capture-session",
      tenant_id="tenant-capture",
      auth_config={
        "provider": "openai",
        "api_key": "opaque",
        "credential_source": {"generation_ref": "parent-bound-generation"},
      },
    )
    store = SQLiteApprovalStore(tmp_path / "real-child.sqlite3")
    policy = FixedOutcomeApprovalPolicy("request_user_approval")
    parent.approval_route = DurableLocalApprovalRoute(store, policy, parent)
    parent.approval_store = store
    parent.approval_policy = policy
    parent_context = root_context(
      parent,
      withheld_tools=frozenset(),
      research_file_id=None,
    )

    async def native_tool(_input: dict[str, Any], **_kwargs: Any) -> ToolResult:
      raise AssertionError("The denied child's tool must not execute")

    definition = {
      "name": "web_search",
      "description": "Native test tool requiring approval.",
      "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
    handlers: dict[str, LocalToolHandler] = {"web_search": native_tool}
    mcp = _NullMcp()
    dispatcher = ToolDispatcher(
      capability_bind=stub_capability_execution_resolver(
        default_provider="openai", default_model="gpt-5.6-sol",
      ).resolve("session.driver").bind,
      mcp_client=mcp,
      local_tool_handlers=handlers,
      session=parent,
      approval_route=parent.approval_route,
      session_id=parent.session_id,
      event_log=EventLog(),
      get_tool_definitions=lambda: [definition],
    )
    provider = _NativeChildCaptureProvider()
    base = stub_capability_execution_resolver(
      default_provider="openai",
      default_model="gpt-5.6-sol",
    )
    resolver = CapabilityExecutionResolver(
      registry=base.registry,
      selection_policy=base.selection_policy,
      auth_context=base.auth_context,
      credential_materializer=base.credential_materializer,
      adapter_resolver=lambda _adapter: provider,
      trusted_channel=base.trusted_channel,
      authenticated_run_overrides=base.authenticated_run_overrides,
      executable_capability_ids=base.executable_capability_ids,
    )
    observed: dict[str, Any] = {}

    class ObservingParentRunner(AgentRunner):
      async def spawn_sub_agent(self, task: str, **kwargs: Any):
        observed.update(kwargs)
        return await super().spawn_sub_agent(task, **kwargs)

    runner = ObservingParentRunner(
      event_log=EventLog(),
      dispatcher=dispatcher,
      session_id=parent.session_id,
      capability_execution=resolver.resolve("session.driver"),
      get_tool_definitions=dispatcher.get_tool_definitions,
      agent_session_log=AgentSessionLog(path=tmp_path / "sessions" / "capture.jsonl"),
      workspace_dir=str(tmp_path),
      user_id=parent.user_id,
      billing_mode="byok",
      rate_table_version="unknown",
    )
    runner._runner_id = "capture-parent"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "native-capture-child.md").write_text(
      "---\nname: native-capture-child\nexecution_class: node.explore\n"
      "version: '1.0'\nagent_callable: true\nagent_description: Exercise native approval.\n"
      "mutation_mode: read_only\nallowed_tools: [web_search]\nmcp_tools: {}\n"
      "semantic_metadata:\n  required_context: []\n"
      "  tool_refs:\n    - kind: local\n      tool_id: web_search\n"
      "  capability_requirements:\n    - name: web.read/v1\n"
      "      required: true\n      binding_modes: [live_tool]\n"
      "---\nMake the admitted tool call.\n",
      encoding="utf-8",
    )
    loader = SkillLoader(skills_dir)
    operation = next(
      item.snapshot.operation.model_dump(mode="json")
      for item in loader.list_callable_operations()
      if item.snapshot.operation.name == "native-capture-child"
    )
    spawn = make_run_agent_handler(
      [runner],
      parent_session=parent,
      skill_loader=loader,
      mcp_client=mcp,
      local_tool_handlers=handlers,
      capability_execution_resolver=resolver,
      needs_approval=lambda *_args: True,
      declared_local_tool_definitions=(
        lambda requested: [definition] if "web_search" in requested else []
      ),
    )
    # The child parks its call; its spawn settles without a decision.
    await asyncio.wait_for(spawn({
      "operation": operation,
      "objective": "Make the native child tool call.",
      "background": False,
    }), timeout=10)
    assert provider.turn == 1
    assert parent.pending_tools == {} and parent.approval_queues == {}
    child_session = observed["sub_session"]
    child_context = current_parent_context(child_session)
    child_job_id = observed["admitted_task"].attempt.physical_task_id
    assert child_session is not parent
    assert observed["dispatcher"]._session is parent
    assert child_context.job_id == child_job_id
    assert child_job_id != parent_context.job_id
    assert current_parent_context(parent).job_id == parent_context.job_id
    assert child_context.tenant is not None
    with store._connection() as conn:
      approval_id, nonce = conn.execute(
        "SELECT approval_id, nonce FROM approval_executable_bindings WHERE tool_call_id = ?",
        ("call-real-child",),
      ).fetchone()
    binding, request = await store._read_executable_binding_for_executor(
      job_id=child_job_id,
      tool_call_id="call-real-child",
      nonce=nonce,
      tenant_storage_id=child_context.tenant.storage_id,
    )
    assert binding.job_id == child_job_id
    assert binding.job_id != parent_context.job_id
    assert binding.approval_id == approval_id
    assert binding.prepared_input == {"query": "exact child operand"}
    assert request.state == "pending_user"

    # A continuation retains its durable child's bind after parent rotation.
    durable_child = observed["admitted_task"]
    frozen_bind = durable_child.model_bind.model_copy(
      update={"credential_ref": "anthropic-generation-child-frozen"},
    )
    durable_child = durable_child.model_copy(update={"model_bind": frozen_bind})
    assert parent.auth_config is not None
    parent.auth_config["credential_source"] = {
      "generation_ref": "anthropic-generation-parent-rotated",
    }
    continued = admit(
      ContinueRequest(child_job_id),
      current_parent_context(parent),
      owners=ContinuationOwners(
        task=durable_child,
        research_binding=child_context.research_binding,
      ),
    )
    assert isinstance(continued, AdmittedContext)
    assert continued.wire.capability_bind == frozen_bind
    resumed_dispatcher = ToolDispatcher(
      capability_bind=continued.wire.capability_bind,
      mcp_client=mcp,
      local_tool_handlers=handlers,
      needs_approval=lambda *_args: True,
      role="owner",
      approval_route=continued.approval_route,
      identity=DispatchIdentity(
        context=context_for_child(continued, job_id=child_job_id),
        session=parent,
      ),
    )
    with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
      await resumed_dispatcher.dispatch_prepared(
        "call-resumed-child", "web_search", PreparedToolCall({"query": "resumed child"}),
      )
    resumed_park = raised.value
    assert (resumed_park.job_id, resumed_park.root_turn_id) == (child_job_id, None)
    resumed_binding, resumed_request = await store._read_executable_binding_for_executor(
      job_id=child_job_id,
      tool_call_id="call-resumed-child",
      nonce=resumed_park.nonce,
      tenant_storage_id=child_context.tenant.storage_id,
    )
    assert resumed_binding.credential_generation_ref == frozen_bind.credential_ref
    assert resumed_binding.job_id == child_job_id
    assert resumed_request.state == "pending_user"
    assert resumed_dispatcher._executing_root_turn() is None

  asyncio.run(scenario())


def test_native_root_park_links_captured_turn_on_ledger_store(
  tmp_path: Path,
  owner_session_host_policy: Any,
) -> None:
  async def scenario() -> None:
    dispatcher, store, session = _native_capture_dispatcher(tmp_path)
    execution = stub_capability_execution_resolver().resolve("session.driver")
    execution = replace(execution, bind=dispatcher._capability_bind)
    root_turn = admit_root_turn(
      current_parent_context(session),
      execution=execution,
      tool_grant={"tool_ids": ["native_write"]},
      approval_route=session.approval_route,
    )
    root_turn = await record_root_turn_admission(store, root_turn)
    capture_root_turn(session, root_turn)
    assert current_root_turn(session) is root_turn
    assert dispatcher._executing_root_turn() is root_turn
    with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
      await dispatcher.dispatch_prepared(
        "call-root-park", "native_write", PreparedToolCall({"api_key": "private-root-operand"}),
      )
    park = raised.value
    assert (park.job_id, park.original_tool_call_id, park.root_turn_id) == (
      "job-native", "call-root-park", root_turn.turn_id,
    )
    assert session.pending_tools == {} and session.approval_queues == {}
    reopened = SQLiteApprovalStore(store.path)
    assert await reopened._locate_root_turn_for_approval(
      approval_id=park.approval_id, tenant_storage_id="alice",
    ) == ("job-native", "call-root-park", park.nonce, root_turn.turn_id)
    record = await reopened._read_root_turn_for_executor(
      job_id="job-native", turn_id=root_turn.turn_id, tenant_storage_id="alice",
    )
    assert record is not None
    assert len(record.executable_refs) == 1
    request = await store.get(park.approval_id)
    assert request is not None and request.state == "pending_user"
    assert "private-root-operand" not in json.dumps(_approval_request_to_dict(request))

  asyncio.run(scenario())


def _binding_rows(store: SQLiteApprovalStore) -> list[tuple[str, str, str]]:
  with store._connection() as conn:
    return [
      (row["approval_id"], row["tool_call_id"], row["nonce"])
      for row in conn.execute(
        "SELECT approval_id, tool_call_id, nonce FROM approval_executable_bindings",
      )
    ]


@pytest.mark.parametrize("pending_call_id", ["call-resumed", "call-retry"])
def test_native_repark_of_resumed_approval_binds_the_pending_call(
  tmp_path: Path,
  owner_session_host_policy: Any,
  pending_call_id: str,
) -> None:
  """A producer that died while parked leaves a pending, bound approval.

  Re-dispatching the resumed request parks again on that approval: one binding,
  re-presented to the call now pending, and the Continue executes that call.
  """

  async def scenario() -> None:
    dispatcher, store, _session = _native_capture_dispatcher(tmp_path)
    original = PreparedToolCall({"ticker": "MSFT"}, exact_backend="original-backend")
    with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
      await dispatcher.dispatch_prepared(
        "call-resumed", "native_write", original, call_index=3,
      )
    first = raised.value
    resumed = await store.get(first.approval_id)
    assert resumed is not None and resumed.state == "pending_user"
    first_binding, _ = await store._read_executable_binding_for_executor(
      job_id="job-native", tool_call_id="call-resumed", nonce=first.nonce,
      tenant_storage_id="alice",
    )
    resumed_context = {**first_binding.dispatch_context, "call_index": 4}
    event_log = EventLog()
    dispatcher._event_log = event_log
    with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
      await dispatcher._run_approval_lifecycle(
        executable_capture=lifecycle_helpers.ExecutableCapture(
          prepared_call=original,
          dispatch_context=resumed_context,
          job_id="job-native",
          tenant_storage_id="alice",
          credential_generation_ref="anthropic-generation-original",
          root_turn=None,
        ),
        tool_call_id=pending_call_id,
        tool_name="native_write",
        tool_input=original.materialize_input(),
        qualifier="",
        reason="",
        allow_persistent=False,
        prepared_business_model_change={},
        resume_approval_request=resumed,
      )
    park = raised.value
    assert (park.job_id, park.approval_id, park.original_tool_call_id) == (
      "job-native", first.approval_id, pending_call_id,
    )
    if pending_call_id == "call-resumed":
      assert park.nonce == first.nonce
    [asked] = [
      entry.event for entry in event_log.entries
      if entry.event["type"] == "tool_approval_request"
    ]
    assert (asked["approval_id"], asked["tool_call_id"], asked["nonce"]) == (
      first.approval_id, pending_call_id, park.nonce,
    )
    assert _binding_rows(store) == [(first.approval_id, pending_call_id, park.nonce)]
    pending = await store.get(first.approval_id)
    assert pending is not None
    assert (pending.state, pending.tool_call_id) == ("pending_user", pending_call_id)
    binding, _ = await store._read_executable_binding_for_executor(
      job_id="job-native", tool_call_id=pending_call_id, nonce=park.nonce,
      tenant_storage_id="alice",
    )
    assert binding.dispatch_context == resumed_context
    await store.transition_state(
      first.approval_id, "approved", expected_state_version=pending.state_version,
      decider_id="alice", decider_role="owner",
    )
    result, error = await dispatcher.execute_approved_binding(
      job_id="job-native", tool_call_id=pending_call_id, nonce=park.nonce,
      abort_event=None, on_executed_prepared_call=None,
    )
    assert error is None and result == {"ok": True}
    if pending_call_id != "call-resumed":
      _stale, stale_error = await dispatcher.execute_approved_binding(
        job_id="job-native", tool_call_id="call-resumed", nonce=first.nonce,
        abort_event=None, on_executed_prepared_call=None,
      )
      assert stale_error is not None
      assert stale_error["code"] == "approval_binding_missing_binding"

  asyncio.run(scenario())


def test_native_call_parks_again_after_its_denied_approval(
  tmp_path: Path,
  owner_session_host_policy: Any,
) -> None:
  """The same call id parking twice in one job binds its new approval."""

  async def scenario() -> None:
    dispatcher, store, _session = _native_capture_dispatcher(tmp_path)
    original = PreparedToolCall({"ticker": "MSFT"}, exact_backend="original-backend")
    with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
      await dispatcher.dispatch_prepared("call-twice", "native_write", original)
    first = raised.value
    denied = await store.get(first.approval_id)
    assert denied is not None
    await store.transition_state(
      first.approval_id, "denied", expected_state_version=denied.state_version,
    )
    with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
      await dispatcher.dispatch_prepared("call-twice", "native_write", original)
    second = raised.value
    assert second.approval_id != first.approval_id
    assert (second.job_id, second.original_tool_call_id, second.nonce) == (
      "job-native", "call-twice", first.nonce,
    )
    assert _binding_rows(store) == [(second.approval_id, "call-twice", second.nonce)]
    first_request = await store.get(first.approval_id)
    pending = await store.get(second.approval_id)
    assert first_request is not None and first_request.state == "denied"
    assert pending is not None and pending.state == "pending_user"
    await store.transition_state(
      second.approval_id, "approved", expected_state_version=pending.state_version,
      decider_id="alice", decider_role="owner",
    )
    result, error = await dispatcher.execute_approved_binding(
      job_id="job-native", tool_call_id="call-twice", nonce=second.nonce,
      abort_event=None, on_executed_prepared_call=None,
    )
    assert error is None and result == {"ok": True}

  asyncio.run(scenario())


@pytest.mark.parametrize("child_identity", [False, True])
@pytest.mark.parametrize("scheduled_learning_fork", [False, True])
def test_fork_parks_against_its_spawning_turn_after_a_later_turn_begins(
  tmp_path: Path,
  owner_session_host_policy: Any,
  scheduled_learning_fork: bool,
  child_identity: bool,
) -> None:
  async def scenario() -> None:
    dispatcher, store, session = _native_capture_dispatcher(tmp_path)
    execution = replace(
      stub_capability_execution_resolver().resolve("session.driver"),
      bind=dispatcher._capability_bind,
    )

    async def begin_turn(job_id: str):
      capture_job_context(session, replace(
        root_context(session, withheld_tools=frozenset(), research_file_id=None),
        job_id=job_id,
      ))
      turn = await record_root_turn_admission(store, admit_root_turn(
        current_parent_context(session),
        execution=execution,
        tool_grant={"tool_ids": ["native_write"]},
        approval_route=session.approval_route,
      ))
      capture_root_turn(session, turn)
      return turn

    turn_a = await begin_turn("job-turn-a")
    from agent_gateway.fork_ledger import ForkLedger
    from agent_gateway.fork_request_handoff import build_post_turn_handoff
    from agent_gateway.fork_task_registry import ForkTaskRegistry
    from agent_gateway.task_registry import TaskEntry
    from agent_workflow_contracts import (
      AgentOperationRef, AttemptRef, OrdinaryDelegationTaskRef, OutcomeRequirement,
      ResultRequirement, sha256_digest, terminal_task_result,
    )
    from gateway_test_support.admitted_authority_test_support import sealed_admitted_task
    from model_authority.binding import CredentialHandle

    source = SimpleNamespace(
      _dispatcher=dispatcher,
      _capability_execution=execution,
      _gateway_session=SimpleNamespace(
        tenant_id="alice",
        session_credential_handle=CredentialHandle(
          handle_id=execution.bind.credential_ref,
          provider=execution.bind.provider, principal=execution.bind.credential_principal,
          tenant_id="alice", actor_id="alice",
        ),
      ),
      _billing_mode="byok",
      _last_request_system_blocks=(("system", True),),
      _last_request_wire_tools=[],
      _last_request_message_marker_position=(0, 0),
      _last_request_max_tokens=4096,
    )
    handoff = build_post_turn_handoff(
      source, [{"role": "user", "content": "turn A"}],
      {"role": "assistant", "content": "done"},
    )

    worker_ready = asyncio.Event()
    # The independently admitted child publishes on its own route: a distinct
    # session and ledger, so publication proves the clone installed it.
    child_session = _gateway_session(session_id="learning-child-session")
    child_store = SQLiteApprovalStore(tmp_path / "learning-child.sqlite3")
    child_route = child_approval_route(DurableLocalApprovalRoute(
      child_store, FixedOutcomeApprovalPolicy("request_user_approval"), child_session,
    ))
    child_context = replace(
      current_parent_context(session), job_id="job-child", approval_route=child_route,
    )
    child_scope: dict[str, Any] = (
      {
        "identity": DispatchIdentity(context=child_context, session=child_session),
        "session": child_session,
        "approval_route": child_context.approval_route,
      }
      if child_identity
      else {"identity": None, "session": None, "approval_route": None}
    )
    publication_session = child_session if child_identity else session
    publication_store = child_store if child_identity else store

    parks: list[lifecycle_helpers.ApprovalPark] = []

    async def run_fork(fork_id: str, submitted_handoff: Any) -> TaskEntry:
      await worker_ready.wait()
      fork = dispatcher.fork_child(
        capability_bind=execution.bind,
        event_log=EventLog(), session_id="fork-of-turn-a",
        spawning_turn=submitted_handoff.spawning_turn,
        **child_scope,
      )
      with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
        await fork.dispatch_prepared(
          "call-fork-park", "native_write",
          PreparedToolCall({"api_key": "private-root-operand"}),
        )
      parks.append(raised.value)
      admitted = sealed_admitted_task(
        logical_task=OrdinaryDelegationTaskRef(
          delegation_id=fork_id,
          operation=AgentOperationRef(
            namespace="agent-operation", name="learning-fork", version="1.0",
            digest=sha256_digest({"operation": "learning-fork"}),
          ),
        ),
        attempt=AttemptRef(
          physical_task_id=fork_id, attempt_id=f"{fork_id}:attempt:1", attempt_number=1,
        ),
        result_requirement=ResultRequirement(
          mode="narrative", terminal_narrative="required",
          outcome=OutcomeRequirement(required=False, source="none"),
        ),
      )
      entry = TaskEntry(task_id=fork_id, task_type="agent", admitted_task=admitted)
      entry.task_result = terminal_task_result(
        admitted, status="awaiting_approval",
        reason=f"approval:{raised.value.approval_id} call:call-fork-park",
      )
      return entry

    registry = ForkTaskRegistry(
      ForkLedger(tmp_path / "fork-ledger.sqlite", process_instance_id="test"),
      spawn_fork=run_fork,
    )
    task: asyncio.Task[Any]
    if scheduled_learning_fork:
      launch = registry.submit(
        fork_id="learning-A", session_id=session.session_id, owner="alice",
        handoff=handoff, on_settled=None,
      )
      assert launch.launched
      task = registry._tasks_by_session[session.session_id]
    else:
      fork = dispatcher.fork_child(
        event_log=EventLog(), session_id="fork-of-turn-a",
        capability_bind=execution.bind,
        spawning_turn=handoff.spawning_turn,
        **child_scope,
      )
      task = asyncio.create_task(fork.dispatch_prepared(
        "call-fork-park", "native_write",
        PreparedToolCall({"api_key": "private-root-operand"}),
      ))
    turn_b = await begin_turn("job-turn-b")
    assert turn_b.turn_id != turn_a.turn_id
    worker_ready.set()
    if scheduled_learning_fork:
      await asyncio.wait_for(task, timeout=10)
      (park,) = parks
    else:
      with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
        await asyncio.wait_for(task, timeout=10)
      park = raised.value
    assert publication_session.pending_tools == {}
    assert publication_session.approval_queues == {}
    root_link = await SQLiteApprovalStore(
      publication_store.path,
    )._locate_root_turn_for_approval(
      approval_id=park.approval_id, tenant_storage_id="alice",
    )
    if child_identity:
      assert park.root_turn_id is None
      assert await store.get(park.approval_id) is None
      binding, _ = await child_store._read_executable_binding_for_executor(
        job_id="job-child", tool_call_id="call-fork-park",
        nonce=park.nonce, tenant_storage_id="alice",
      )
      assert binding.job_id == "job-child"
      assert root_link is None
    else:
      assert park.root_turn_id == turn_a.turn_id
      assert root_link == (
        "job-turn-a", "call-fork-park", park.nonce, turn_a.turn_id,
      )
    request = await publication_store.get(park.approval_id)
    assert request is not None and request.state == "pending_user"

  asyncio.run(scenario())


def test_native_parks_before_and_after_rotation_capture_their_original_generations(
  tmp_path: Path,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from agent_gateway.providers.anthropic import AnthropicProvider
  from agent_gateway.providers.anthropic_oauth import resolve_anthropic_credentials
  from gateway_test_support.capability_execution_test_support import stub_bound_capability_execution

  token_a = "sk-ant-oat01-capture-member-A"
  token_b = "sk-ant-oat01-capture-member-B"
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", token_a)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([token_a, token_b]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))
  sources = resolve_anthropic_credentials()
  member_a = sources.by_token(token_a)
  member_b = sources.by_token(token_b)
  assert member_a is not None and member_b is not None
  generation_a = member_a.source_metadata()["generation_ref"]
  generation_b = member_b.source_metadata()["generation_ref"]

  class RateLimitedError(RuntimeError):
    # A subscription-quota 429: only a spent window rotates the account.
    status_code = 429
    rate_limit_unified_status = "rejected"

  async def scenario() -> None:
    dispatcher, store, session = _native_capture_dispatcher(tmp_path)
    parked: dict[str, lifecycle_helpers.ApprovalPark] = {}

    class RotatingProvider(AnthropicProvider):
      def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
        return config["auth_token"]

      async def close_client(self, client: Any, timeout: float = 2.0) -> None:
        pass

      async def stream(self, client: Any, params: dict[str, Any]):
        call_id = "call-before-rotation" if client == token_a else "call-after-rotation"
        with pytest.raises(lifecycle_helpers.ApprovalPark) as raised:
          await dispatcher.dispatch_prepared(
            call_id, "native_write", PreparedToolCall({"ticker": "MSFT"}),
          )
        parked[call_id] = raised.value
        if client == token_a:
          raise RateLimitedError("Anthropic rate limit exceeded")
        yield StreamEvent(type="text_delta", text="rotated")
        yield StreamEvent(type="message_end", stop_reason="end_turn")

    execution = stub_bound_capability_execution(
      provider=RotatingProvider(), model="claude-sonnet-5", effort="none",
      credential_principal=dispatcher._capability_bind.credential_principal,
      credential_ref=generation_a,
      auth_config={
        "auth_mode": "oauth", "auth_token": token_a, "max_tokens": 1024,
        "credential_source": member_a.source_metadata(),
      },
    )
    dispatcher.replace_capability_bind(execution.bind)
    root_turn = await record_root_turn_admission(store, admit_root_turn(
      current_parent_context(session), execution=execution,
      tool_grant={"tool_ids": ["native_write"]},
      approval_route=session.approval_route,
    ))
    capture_root_turn(session, root_turn)
    event_log = EventLog()
    runner = AgentRunner(
      capability_execution=execution, dispatcher=dispatcher, event_log=event_log,
      session_id=session.session_id, user_id="alice", billing_mode="byok",
      rate_table_version="unknown",
    )
    await runner.run(messages=[{"role": "user", "content": "rotate then park"}])
    assert any(entry.event.get("type") == "credential_refreshed" for entry in event_log.entries)
    assert runner.capability_execution.bind.credential_ref == generation_b
    assert dispatcher._capability_bind is runner.capability_execution.bind
    assert set(parked) == {"call-before-rotation", "call-after-rotation"}
    assert session.pending_tools == {} and session.approval_queues == {}
    for call_id, generation in (
      ("call-before-rotation", generation_a), ("call-after-rotation", generation_b),
    ):
      park = parked[call_id]
      assert park.root_turn_id == root_turn.turn_id
      binding, request = await store._read_executable_binding_for_executor(
        job_id="job-native", tool_call_id=call_id, nonce=park.nonce,
        tenant_storage_id="alice",
      )
      assert binding.credential_generation_ref == generation
      assert request.state == "pending_user"
      assert await store._locate_root_turn_for_approval(
        approval_id=request.approval_id, tenant_storage_id="alice",
      ) == ("job-native", call_id, park.nonce, root_turn.turn_id)

  asyncio.run(scenario())
