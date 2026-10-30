"""Injected approval-clock walks, with deterministic replay and seeded discovery.

The package's standalone pytest configuration does not load tests/fuzz's lane
plugin. Both bounded lanes run by default; ``-k generated`` selects discovery.
The denominator comes from the product AST; the clock witnesses actual callers.
"""
from __future__ import annotations

import ast
import asyncio
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import inspect
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from agent_gateway import approval_store as owner
from agent_gateway.approval_policy import ApprovalRequest, ApprovalVote
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.tool_policy_registry import PreparedToolCall
from api import runtime_approval_service as remote


_BASE = 2_051_222_400_123_456_789
_COUNTS: Counter[str] = Counter()
_OPERATIONS = (
  "_init_schema",
  "_insert_autonomous_approval_delivery_in_transaction",
  "ensure_autonomous_approval_delivery_audited",
  "autonomous_approval_delivery_append_transaction",
  "reconcile_autonomous_approval_delivery_duplicate",
  "acknowledge_autonomous_approval_delivery",
  "autonomous_approval_delivery_recovery_window",
  "record_autonomous_approval_delivery_failure",
  "quarantine_autonomous_approval_delivery",
  "_commit",
)


@dataclass
class _Clock:
  value: int = _BASE

  def __call__(self) -> int:
    caller = sys._getframe(1)
    if caller.f_code.co_name == "_trusted_utc_now":
      assert caller.f_back is not None
      _COUNTS[caller.f_back.f_code.co_name] += 1
    return self.value


class _Audit:
  async def emit_audit_for_lifecycle_event(self, **kwargs: Any) -> None:
    # Only the audit transport is inert; the store's durable ready transaction
    # and every approval/delivery transition execute in the real SQLite owner.
    pass


def _request(suffix: str = "clock") -> ApprovalRequest:
  return ApprovalRequest(
    approval_id=f"approval-{suffix}", tool_call_id=f"tool-{suffix}",
    parent_approval_id=None, approval_chain_id=f"approval-{suffix}",
    request_id=f"run-{suffix}", session_id=f"session-{suffix}", run_id=f"run-{suffix}",
    user_id="owner", profile="analyst", channel="tui",
    tool_name="memory_write", tool_class="state_write",
    tool_args_redacted={}, args_hash="clock-hash", reason="clock walk",
    blast_radius_summary="state_write:memory_write", state="pending_user",
    requested_at=datetime(2034, 12, 31, tzinfo=UTC),
    expires_at=datetime(2035, 1, 1, tzinfo=UTC),
    approval_constraint="standard",
  )


def _delivery_context(suffix: str) -> dict[str, str]:
  return {
    "task_id": f"bg_{suffix}", "control_run_id": f"run-{suffix}",
    "session_id": f"session-{suffix}", "channel_id": "a" * 64,
    "tool_call_id": f"tool-{suffix}", "nonce": f"nonce-{suffix}",
  }


def _decide(store: SQLiteApprovalStore, suffix: str) -> None:
  asyncio.run(store.record_vote_with_autonomous_delivery(
    f"approval-{suffix}",
    ApprovalVote(
      vote_id=f"vote-{suffix}", approval_id=f"approval-{suffix}", decider_id="owner",
      decider_role="owner", decision="approved", decision_reason="reviewed",
      decided_at=datetime(2035, 1, 1, tzinfo=UTC),
    ),
    delivery=_delivery_context(suffix),
  ))


def _snapshot(store: SQLiteApprovalStore) -> tuple[str, ...]:
  with store._connection() as conn:
    return tuple(conn.iterdump())


def _high_water(store: SQLiteApprovalStore) -> int:
  with store._connection() as conn:
    return int(conn.execute(
      "SELECT observed_at_ns FROM approval_clock_high_water WHERE clock_id = 1",
    ).fetchone()[0])


def _prepare(path: Path, clock: _Clock, operation: str, suffix: str) -> tuple[SQLiteApprovalStore, Callable[[], Any]]:
  store = SQLiteApprovalStore(path, clock_ns=clock)
  if operation == "_init_schema":
    return store, lambda: SQLiteApprovalStore(path, clock_ns=clock)
  if operation == "autonomous_approval_delivery_recovery_window":
    return store, lambda: asyncio.run(store.autonomous_approval_delivery_recovery_window())
  asyncio.run(store.create(_request(suffix)))
  if operation == "_insert_autonomous_approval_delivery_in_transaction":
    return store, lambda: _decide(store, suffix)
  _decide(store, suffix)
  store._audit_emitter = _Audit()
  identity = {"approval_id": f"approval-{suffix}", "tool_call_id": f"tool-{suffix}", "nonce": f"nonce-{suffix}"}
  if operation == "ensure_autonomous_approval_delivery_audited":
    return store, lambda: asyncio.run(store.ensure_autonomous_approval_delivery_audited(**identity))
  asyncio.run(store.ensure_autonomous_approval_delivery_audited(**identity))
  if operation == "_commit":
    service = remote.RuntimeApprovalService(
      services_root=path.parent, audit_hmac_secret=b"clock-walk",
      audit_hmac_key_id="clock-walk",
    )
    reserved = service._reserve(store, 1, {
      "decision_id": identity["approval_id"], "tool_call_id": identity["tool_call_id"],
      "nonce": identity["nonce"], "approved": True,
    })
    reservation = reserved.reservation
    assert reservation is not None
    return store, lambda: service._commit(store, 1, reservation)
  if operation == "autonomous_approval_delivery_append_transaction":
    def publish() -> None:
      with store.autonomous_approval_delivery_append_transaction(**identity, approved=True):
        pass
    return store, publish
  if operation == "reconcile_autonomous_approval_delivery_duplicate":
    return store, lambda: store.reconcile_autonomous_approval_delivery_duplicate(**identity, approved=True)
  if operation == "acknowledge_autonomous_approval_delivery":
    store.reconcile_autonomous_approval_delivery_duplicate(**identity, approved=True)
    delivery = asyncio.run(store.get_autonomous_approval_delivery(**identity))
    assert delivery is not None
    return store, lambda: asyncio.run(store.acknowledge_autonomous_approval_delivery(
      **identity, **{key: delivery[key] for key in (
        "task_id", "control_run_id", "session_id", "channel_id", "approved", "decided_at_ns",
      )},
    ))
  if operation == "record_autonomous_approval_delivery_failure":
    return store, lambda: asyncio.run(store.record_autonomous_approval_delivery_failure(**identity, error="transport failed"))
  assert operation == "quarantine_autonomous_approval_delivery"
  return store, lambda: asyncio.run(store.quarantine_autonomous_approval_delivery(**identity, error="unreachable channel"))


def _walk(root: Path, steps: list[tuple[str, int]]) -> None:
  high_water = _BASE
  with TemporaryDirectory(dir=root) as directory:
    clock = _Clock(high_water)
    for index, (operation, delta) in enumerate(steps):
      store, invoke = _prepare(Path(directory) / "approvals.sqlite3", clock, operation, str(index))
      before = _snapshot(store)
      reached_before = _COUNTS[operation]
      clock.value = high_water + delta
      if delta < 0:
        with pytest.raises(RuntimeError):
          invoke()
        assert _COUNTS[operation] > reached_before, operation
        assert _snapshot(store) == before, operation
        assert _high_water(store) == high_water, operation
        # The same operation remains usable at the durable boundary; refusal
        # must not have consumed a vote, sequence, receipt or state version.
        clock.value = high_water
        invoke()
      else:
        invoke()
        high_water = clock.value
      assert _COUNTS[operation] > reached_before, operation
      assert _high_water(store) == high_water, operation
      # Reopening observes the same durable clock, not an in-memory model.
      reopened = SQLiteApprovalStore(store.path, clock_ns=clock)
      assert _high_water(reopened) == high_water


def test_trusted_time_operation_denominator() -> None:
  callers = []
  for module in (owner, remote):
    tree = ast.parse(inspect.getsource(module))
    for function in ast.walk(tree):
      if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for node in ast.walk(function):
          if isinstance(node, ast.Call):
            name = (
              node.func.id if isinstance(node.func, ast.Name)
              else node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            if name == "_trusted_utc_now":
              callers.append(function.name)
  assert Counter(callers) == Counter(_OPERATIONS)


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_clock_walk_replay(tmp_path: Path, operation: str) -> None:
  _walk(tmp_path, [(operation, delta) for delta in (0, 1, -1, 1_000_001, -1_000, 0)])


def test_remote_publication_equal_clock_regression(tmp_path: Path) -> None:
  _walk(tmp_path, [("_commit", 0)])


@seed(20261030)
@settings(
  max_examples=int(os.environ.get("FUZZ_DISCOVERY_EXAMPLES", "40")),
  deadline=None, database=None,
  suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.lists(st.tuples(st.sampled_from(_OPERATIONS), st.integers(-1_000_000, 1_000_000)), min_size=1, max_size=12))
def test_generated_clock_walk(tmp_path: Path, steps: list[tuple[str, int]]) -> None:
  _walk(tmp_path, steps)


def test_default_clock_is_resolved_after_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(owner.time, "time_ns", lambda: _BASE)
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  monkeypatch.setattr(owner.time, "time_ns", lambda: _BASE + 999)
  observed = asyncio.run(store.autonomous_approval_delivery_recovery_window())
  assert observed["observed_at_ns"] == _BASE + 999
  assert _high_water(store) == _BASE + 999


@pytest.mark.parametrize("delta", [0, 1_000_001, -1_000_001])
def test_executable_binding_expiry_uses_clock(tmp_path: Path, delta: int) -> None:
  clock = _Clock()
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3", clock_ns=clock)

  async def expire() -> None:
    original = _request()
    await store.create(original)
    old = await store.record_policy_generation(
      policy_id="clock-policy", policy_version="1", policy_bundle_hash="a" * 64,
    )
    await store.record_root_turn(
      job_id="job-clock", turn_id="turn-clock", tenant_storage_id="owner",
      capability_bind={"model_key": "session.driver", "provider": "anthropic"},
      credential_generation_ref=None, effective_grant={"tool_ids": ["memory_write"]},
      research_binding=None, selection_snapshot={"role": "analyst"},
      approval_route_kind="DurableLocalApprovalRoute",
      policy_generation_ref=old.generation_ref,
    )
    binding: dict[str, Any] = {
      "approval_id": original.approval_id, "job_id": "job-clock",
      "tool_call_id": original.tool_call_id, "nonce": "nonce-clock",
      "tenant_storage_id": "owner", "tool_name": original.tool_name,
      "prepared_call": PreparedToolCall(prepared_input={}),
      "dispatch_context": {}, "credential_generation_ref": None,
    }
    await store.record_executable_binding(**binding)
    await store.link_root_turn_executable_binding(
      job_id="job-clock", turn_id="turn-clock", tool_call_id=original.tool_call_id,
      nonce="nonce-clock", tenant_storage_id="owner",
    )
    new = await store.record_policy_generation(
      policy_id="clock-policy", policy_version="1", policy_bundle_hash="b" * 64,
    )
    successor = replace(original, approval_id="approval-successor", approval_chain_id="approval-successor")
    await store.create(successor)
    clock.value += delta
    assert await store.park_executable_binding(**{
      **binding, "approval_id": successor.approval_id, "nonce": "nonce-successor",
      "policy_generation_ref": new.generation_ref,
    }) == "nonce-successor"
    expired = await store.get(original.approval_id)
    assert expired is not None and expired.state == "expired"
    assert expired.state_version == original.state_version + 1
    seconds, nanos = divmod(clock.value, 1_000_000_000)
    assert expired.decided_at == datetime.fromtimestamp(seconds, UTC).replace(microsecond=nanos // 1_000)
    # This existing expiry path observes time but has never advanced the
    # trusted-time ledger. Clock injection adds no new refusal at this seam.
    assert _high_water(store) == _BASE

  asyncio.run(expire())


@pytest.fixture(scope="module", autouse=True)
def _report_coverage(request: pytest.FixtureRequest):
  _COUNTS.clear()
  yield
  reporter = request.config.pluginmanager.get_plugin("terminalreporter")
  assert reporter is not None
  reporter.write_line("trusted-clock operation observations: " + repr(dict(sorted(_COUNTS.items()))))
