"""Cancelling a detached autonomous run through ``DELETE /control/runs/{id}``.

A parked run (its ask recorded a durable continuation and its process exited)
has no process to deliver a denial to, so cancelling it ends it: the run is
cancelled first, the request is denied ``run_cancelled`` with no delivery row,
and no successor is launched from its captured call. A live run's cancel still
delivers the denial to the waiting process; an operator's deny of a parked run
still relaunches.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway.approval_policy import ApprovalVote
from agent_gateway.approvals import _record_durable_vote_and_continue
from agent_gateway.approval_store import RUN_CANCELLED_REASON, SQLiteApprovalStore
from agent_gateway.autonomous_runner import AutonomousRegistry
from agent_gateway.control_plane.autonomous_approval_delivery import (
  autonomous_approval_delivery_context,
  autonomous_decision_unavailable,
  deliver_autonomous_approval_outbox,
)
from agent_gateway.control_plane.runs_helpers import (
  _deny_autonomous_pending_approvals_for_cancel,
)
from agent_gateway.control_run_lifecycle import canonical_control_run_state
from agent_gateway.session import GatewaySession
from agent_gateway.single_user_policy import SingleUserApprovalPolicy

from test_autonomous_approval_channel import (  # noqa: F401
  owner_session_host_policy,
  parked_detached_approval,
)


def _owner() -> GatewaySession:
  """The run owner's authenticated control session (`hank control …`)."""

  return GatewaySession(
    session_id="control-owner", api_key_hash="control", created_at=0,
    expires_at=2**31, user_id="42", owner_user_id="42", role="owner",
    kind="control", channel="cli",
  )


class _AuditEmitter:
  async def emit_audit_for_lifecycle_event(self, **_kwargs: Any) -> None:
    return None


def _prepare(parked: Any, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
  """Audited store, the run's published ask on its event record, a launch recorder."""

  store = SQLiteApprovalStore(parked.store.path, audit_emitter=_AuditEmitter())
  parked.registry._approval_store = store
  source = parked.source
  source.event_lines = [{
    "type": "tool_approval_request",
    "approval_id": parked.parked["approval_id"],
    "tool_call_id": "call-original",
    "nonce": parked.parked["nonce"],
    "tool_name": "native_write",
  }]
  # The same ask on the run's durable events file, which a restart reads.
  Path(parked.root, f"{source.task_id}.events.jsonl").write_text(
    json.dumps(source.event_lines[0]) + "\n", encoding="utf-8",
  )
  launches: list[dict[str, Any]] = []

  async def launch_process(**kwargs: Any) -> dict[str, str]:
    launches.append(kwargs)
    successor = replace(
      source, task_id="bg_successor", session_id="bg_successor",
      control_run_id="run-successor", state="running",
      resumed_from=kwargs["resumed_from"], reaper_task=None,
      approval_continuation=None,
      launch_approval_continuation=kwargs["approval_continuation"],
      resumed_approval_id=kwargs["approval_continuation"]["approval_id"],
    )
    parked.registry._tasks[successor.task_id] = successor
    return {"run_id": successor.control_run_id, "task_id": successor.task_id}

  monkeypatch.setattr(parked.registry, "start", launch_process)
  return SimpleNamespace(
    store=store, source=source, launches=launches,
    approval_id=parked.parked["approval_id"], nonce=parked.parked["nonce"],
  )


def _app_state(registry: AutonomousRegistry, store: SQLiteApprovalStore) -> SimpleNamespace:
  async def root_continue(_request: Any) -> SimpleNamespace:
    # A detached run's approval has no root turn in this gateway's ledger.
    return SimpleNamespace(status="not_root", job_id=None, turn_id=None, tool_call_id=None)

  async def child_continue(request: Any) -> Any:
    # `api/agent/interactive/child_continue.py:86-92`: the subprocess registry
    # answers first; a detached run never reaches interactive restoration.
    outcome = await registry.continue_approval(
      approval_id=request.approval_id, tenant_storage_id=request.tenant_storage_id,
    )
    assert outcome is not None, "a detached approval fell through to interactive restore"
    return outcome

  return SimpleNamespace(
    gateway_approval_store=store,
    gateway_approval_policy=SingleUserApprovalPolicy(store=store),
    root_continue_hook=root_continue, child_continue_hook=child_continue,
  )


def _delete_run_endpoint(registry: AutonomousRegistry, store: SQLiteApprovalStore) -> tuple[Any, Any]:
  """``DELETE /control/runs/{id}``: the route's own entry, mounted from ``build_runs_router``."""
  from fastapi import Request
  from fastapi.routing import APIRoute

  from agent_gateway.control_plane import runs

  auth: Any = SimpleNamespace(
    verify_token=lambda _token: _owner(),
    session_store=SimpleNamespace(get_session=lambda _id: None),
  )
  router = runs.build_runs_router(auth=auth, autonomous_registry=registry)
  endpoint = next(
    route.endpoint for route in router.routes
    if isinstance(route, APIRoute) and route.path == "/runs/{control_run_id}"
    and route.methods and "DELETE" in route.methods
  )
  request = Request({
    "type": "http", "headers": [(b"authorization", b"Bearer test")],
    "app": SimpleNamespace(state=_app_state(registry, store)),
  })
  return endpoint, request


async def _delete_run(
  parked: Any,
  store: SQLiteApprovalStore,
  *,
  registry: AutonomousRegistry | None = None,
) -> Exception | None:
  """Cancel through ``control_plane/runs.py::delete_run``; the collected approval error it
  raises (after the cancel) is returned, else ``None``."""
  from fastapi import HTTPException

  run_registry: AutonomousRegistry = registry if registry is not None else parked.registry
  record = run_registry._tasks[parked.source.task_id]
  endpoint, request = _delete_run_endpoint(run_registry, store)
  try:
    response = await endpoint(request, record.task_id)
  except HTTPException as exc:
    return exc
  # The route answers with the record's projection at return: a parked run is already
  # `cancelled`; a live one carries `cancellation_requested` until its process exits.
  assert response.task_id == record.task_id
  assert canonical_control_run_state(response.state) == "cancelled" or record.cancellation_requested is True
  return None


async def _durable_decision(
  registry: AutonomousRegistry,
  store: SQLiteApprovalStore,
  approval_id: str,
  *,
  approved: bool,
) -> dict[str, Any]:
  """The durable decision route ``hank control approve`` takes for a captured call.

  ``control_plane/approvals.py::_resolve_parked_approval_decision`` →
  ``approvals.py::_record_durable_vote_and_continue``.
  """

  located = await store._locate_request_for_decision(
    approval_id=approval_id, tenant_storage_id="42",
  )
  assert located is not None
  request_record, locator, nonce = located
  assert locator.job_id == "bg_7"
  return await _record_durable_vote_and_continue(
    authenticated=_owner(), tenant_storage_id="42", request_record=request_record,
    locator=locator, nonce=nonce, store=store, policy=SingleUserApprovalPolicy(store=store),
    decider_id="42", decider_role="owner", approved=approved,
    allow_tool_type=False, reason=None, app_state=_app_state(registry, store),
  )


async def _crash_after_cancel_before_deny(
  parked: Any,
  prepared: SimpleNamespace,
) -> None:
  """The gateway dies after ``registry.cancel`` and before the deny is written."""

  async def die(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("gateway died after cancel, before the deny")

  store = prepared.store
  store.terminalize_pending_for_cancellation = die
  try:
    with pytest.raises(RuntimeError, match="gateway died after cancel"):
      await _deny_autonomous_pending_approvals_for_cancel(
        registry=parked.registry, record=prepared.source,
        authenticated=_owner(),
        app_state=_app_state(parked.registry, store),
      )
  finally:
    del store.terminalize_pending_for_cancellation
  # The exact crash state: the run is cancelled, the request still pending.
  assert prepared.source.state == "killed"
  assert canonical_control_run_state(prepared.source.state) == "cancelled"
  pending = await prepared.store.get(prepared.approval_id)
  assert pending is not None and pending.state == "pending_user"
  assert await prepared.store.get_autonomous_approval_delivery(
    prepared.approval_id, tool_call_id="call-original", nonce=prepared.nonce,
  ) is None
  assert prepared.launches == []


async def _assert_call_not_executed(prepared: SimpleNamespace) -> None:
  binding, _ = await prepared.store._read_executable_binding_for_executor(
    job_id="bg_7", tool_call_id="call-original", nonce=prepared.nonce,
    tenant_storage_id="42",
  )
  assert await prepared.store._read_completed_executable_result_for_executor(
    binding=binding,
  ) is None


def test_cancel_of_parked_run_ends_cancelled_and_launches_no_successor(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """(a) Fails before: the cancel's denial relaunched the parked run (`5adac74007`)."""

  parked = parked_detached_approval
  prepared = _prepare(parked, monkeypatch)

  async def scenario() -> None:
    assert prepared.source.state == "awaiting_approval"
    assert prepared.source.approval_continuation is not None
    assert await _delete_run(parked, prepared.store) is None
    decided = await prepared.store.get(prepared.approval_id)
    assert decided is not None
    assert (decided.state, decided.decision_reason) == ("denied", RUN_CANCELLED_REASON)
    assert prepared.launches == []
    assert prepared.source.resumed_as == []
    assert prepared.source.state == "killed"
    assert canonical_control_run_state(prepared.source.state) == "cancelled"
    await _assert_call_not_executed(prepared)

  asyncio.run(scenario())


def test_operator_deny_of_parked_run_still_relaunches_with_the_denial(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """(b) Contrast: a human deny reaches the successor, which returns it to the model."""

  parked = parked_detached_approval
  prepared = _prepare(parked, monkeypatch)

  async def scenario() -> None:
    request = await prepared.store.record_vote_with_autonomous_delivery(
      prepared.approval_id,
      ApprovalVote(
        vote_id="owner-deny", approval_id=prepared.approval_id,
        decider_id="42", decider_role="owner", decision="denied",
        decision_reason="Not this account", decided_at=datetime.now(UTC),
      ),
      delivery=autonomous_approval_delivery_context(
        prepared.source, tool_call_id="call-original", nonce=prepared.nonce,
      ),
    )
    delivery = await prepared.store.get_autonomous_approval_delivery(
      prepared.approval_id, tool_call_id="call-original", nonce=prepared.nonce,
    )
    assert delivery is not None
    await deliver_autonomous_approval_outbox(
      registry=parked.registry, store=prepared.store, record=prepared.source,
      request_record=request, delivery=delivery, approval_id=prepared.approval_id,
      tool_call_id="call-original", nonce=prepared.nonce,
      approved=False, user_id="42", channel="cli",
    )
    assert len(prepared.launches) == 1
    signed = prepared.launches[0]["approval_continuation"]
    assert (signed["decision"], signed["decision_reason"]) == ("denied", "Not this account")
    assert signed["claim_token"] is None
    assert prepared.source.state == "continued"
    await _assert_call_not_executed(prepared)

  asyncio.run(scenario())


def test_approve_after_cancel_neither_relaunches_nor_executes(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """(c) The cancelled request is terminal; every later continuation entry refuses."""

  parked = parked_detached_approval
  prepared = _prepare(parked, monkeypatch)

  async def scenario() -> None:
    assert await _delete_run(parked, prepared.store) is None
    late = await prepared.store.record_vote(
      prepared.approval_id,
      ApprovalVote(
        vote_id="late-approve", approval_id=prepared.approval_id,
        decider_id="42", decider_role="owner", decision="approved",
        decision_reason=None, decided_at=datetime.now(UTC),
      ),
    )
    assert (late.state, late.decision_reason) == ("denied", RUN_CANCELLED_REASON)
    # `hank control approve` on the cancelled run: the durable decision route answers
    # `conflict` with the request's state unchanged (R20 leg A), not a refusal.
    late_vote = await _durable_decision(
      parked.registry, prepared.store, prepared.approval_id, approved=True,
    )
    assert late_vote["outcome"] == "conflict"
    after = await prepared.store.get(prepared.approval_id)
    assert after is not None and (after.state, after.decision_reason) == ("denied", RUN_CANCELLED_REASON)
    # The durable decision route's child hook reaches the registry directly.
    continued = await parked.registry.continue_approval(
      approval_id=prepared.approval_id, tenant_storage_id="42",
    )
    assert continued is not None and continued.status == "not_restorable"
    assert prepared.launches == []
    assert prepared.source.resumed_as == []
    assert canonical_control_run_state(prepared.source.state) == "cancelled"
    await _assert_call_not_executed(prepared)

  asyncio.run(scenario())


def test_cancel_of_live_run_still_delivers_the_denial_to_its_process(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """(d) A run waiting live (no durable continuation) is unblocked with the denial."""

  parked = parked_detached_approval
  prepared = _prepare(parked, monkeypatch)
  source = prepared.source
  source.state = "running"
  source.approval_continuation = None
  sent: list[dict[str, Any]] = []

  async def send_approval_decision(control_run_id: str, **kwargs: Any) -> dict[str, Any]:
    sent.append({"control_run_id": control_run_id, **kwargs})
    with kwargs["publication_transaction"]():
      pass
    return {"delivered": True}

  monkeypatch.setattr(parked.registry, "send_approval_decision", send_approval_decision)

  async def scenario() -> None:
    async def reaped() -> None:
      return None

    source.reaper_task = asyncio.create_task(reaped())
    await source.reaper_task
    assert await _delete_run(parked, prepared.store) is None
    assert [(item["control_run_id"], item["approval_id"], item["approved"]) for item in sent] == [
      ("run-7", prepared.approval_id, False),
    ]
    decided = await prepared.store.get(prepared.approval_id)
    assert decided is not None
    assert (decided.state, decided.decision_reason) == ("denied", RUN_CANCELLED_REASON)
    delivery = await prepared.store.get_autonomous_approval_delivery(
      prepared.approval_id, tool_call_id="call-original", nonce=prepared.nonce,
    )
    assert delivery is not None and delivery["state"] == "published"
    assert source.cancellation_requested is True
    assert prepared.launches == []

  asyncio.run(scenario())


def test_crash_between_cancel_and_deny_refuses_a_late_approve(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """(e1) Killed run, pending request: an approve on the durable route launches nothing."""

  parked = parked_detached_approval
  prepared = _prepare(parked, monkeypatch)

  async def scenario() -> None:
    await _crash_after_cancel_before_deny(parked, prepared)
    approved = await _durable_decision(
      parked.registry, prepared.store, prepared.approval_id, approved=True,
    )
    # The registry refuses the continuation of a run that is no longer parked.
    assert approved["continuation"]["status"] == "not_restorable"
    assert prepared.launches == []
    assert prepared.source.state == "killed"
    await _assert_call_not_executed(prepared)

  asyncio.run(scenario())


def test_crash_between_cancel_and_deny_survives_restart_and_redelete(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """(e2) A restart keeps the run killed; re-sending DELETE denies the request."""

  parked = parked_detached_approval
  prepared = _prepare(parked, monkeypatch)

  async def scenario() -> None:
    await _crash_after_cancel_before_deny(parked, prepared)
    restarted = AutonomousRegistry(
      api_dir=Path(__file__).resolve().parents[3] / "api",
      log_dir=parked.root, approval_store=prepared.store,
    )

    async def no_launch(**_kwargs: Any) -> dict[str, str]:
      raise AssertionError("a restarted gateway launched a successor")

    monkeypatch.setattr(restarted, "start", no_launch)
    rehydrated = restarted._tasks[prepared.source.task_id]
    assert rehydrated.state == "killed"
    assert canonical_control_run_state(rehydrated.state) == "cancelled"
    assert rehydrated.approval_continuation is not None
    assert await _delete_run(parked, prepared.store, registry=restarted) is None
    decided = await prepared.store.get(prepared.approval_id)
    assert decided is not None
    assert (decided.state, decided.decision_reason) == ("denied", RUN_CANCELLED_REASON)
    assert rehydrated.state == "killed"
    # Replays after the re-sent DELETE: a matching deny reaches the continuation
    # and is refused there; a conflicting approve stops before it.
    deny = await _durable_decision(
      restarted, prepared.store, prepared.approval_id, approved=False,
    )
    assert deny["outcome"] == "already_decided"
    assert deny["continuation"]["status"] == "not_restorable"
    approve = await _durable_decision(
      restarted, prepared.store, prepared.approval_id, approved=True,
    )
    assert approve["outcome"] == "conflict"
    assert "continuation" not in approve
    assert prepared.launches == []
    assert rehydrated.state == "killed"
    await _assert_call_not_executed(prepared)

  asyncio.run(scenario())
