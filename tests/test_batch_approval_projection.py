from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_gateway.batch_approval_projection import (
  BatchApprovalProjectionRegistry,
  BatchApprovalScope,
  approval_record_matches_projection,
  bind_batch_approval_scope,
  current_batch_approval_scope,
)


def _session(user_id: str = "1", channel: str = "tui") -> SimpleNamespace:
  return SimpleNamespace(
    session_id="stage-session",
    user_id=user_id,
    channel=channel,
    pending_tools={},
    approval_queues={},
  )


def _install_pending(session: SimpleNamespace, approval_id: str = "approval-1") -> None:
  session.pending_tools["tool-1"] = {
    "approval_id": approval_id,
    "nonce": "nonce-1",
    "status": "approval_pending",
    "stage_run_seq": 3,
  }


def test_projection_is_exact_owner_channel_run_and_identity_scoped() -> None:
  registry = BatchApprovalProjectionRegistry()
  session = _session()
  _install_pending(session)
  registry.register_session(
    batch_id=59,
    owner_user_id="1",
    channel="TUI",
    session=session,
  )
  projection = registry.find_projection(
    run_id="batch_59",
    approval_id="approval-1",
    owner_user_id="1",
    channel="tui",
  )
  assert projection is not None
  assert projection.stage_run_seq == 3
  assert registry.find_projection(
    run_id="batch_59",
    approval_id="approval-1",
    owner_user_id="henry",
    channel="tui",
  ) is None
  assert registry.find_projection(
    run_id="batch_59",
    approval_id="approval-1",
    owner_user_id="1",
    channel="web",
  ) is None

  durable = SimpleNamespace(
    approval_id="approval-1",
    tool_call_id="tool-1",
    user_id="1",
    request_id="batch_59",
    run_id="batch_59",
    session_id="stage-session",
    channel="tui",
  )
  assert approval_record_matches_projection(durable, projection)
  durable.user_id = "henry"
  assert not approval_record_matches_projection(durable, projection)


def test_projection_rejects_cross_user_collision_and_closes_exact_batch() -> None:
  registry = BatchApprovalProjectionRegistry()
  alice = _session("1")
  bob = _session("2")
  _install_pending(alice, "approval-alice")
  _install_pending(bob, "approval-bob")
  registry.register_session(batch_id=1, owner_user_id="1", channel="tui", session=alice)
  registry.register_session(batch_id=1, owner_user_id="2", channel="tui", session=bob)

  registry.close_batch(owner_user_id="1", batch_id=1)
  assert registry.projections_for_owner(owner_user_id="1", channel="tui") == []
  assert [
    item.approval_id
    for item in registry.projections_for_owner(owner_user_id="2", channel="tui")
  ] == ["approval-bob"]


def test_projection_registration_is_idempotent_but_collision_fails_closed() -> None:
  registry = BatchApprovalProjectionRegistry()
  first = _session("1")
  _install_pending(first)
  first_carrier = registry.register_session(
    batch_id=1,
    owner_user_id="1",
    channel="tui",
    session=first,
  )
  assert registry.register_session(
    batch_id=1,
    owner_user_id="1",
    channel="tui",
    session=first,
  ) == first_carrier

  conflicting = _session("1")
  conflicting.session_id = "other-stage"
  _install_pending(conflicting)
  with pytest.raises(ValueError, match="conflicting batch approval identity"):
    registry.register_session(
      batch_id=1,
      owner_user_id="1",
      channel="tui",
      session=conflicting,
    )
  assert ("1", 1, id(conflicting)) not in registry._carriers
  conflicting.pending_tools.clear()
  _install_pending(conflicting, "approval-recovered")
  registry.register_session(
    batch_id=1,
    owner_user_id="1",
    channel="tui",
    session=conflicting,
  )
  assert [
    projection.approval_id
    for projection in registry.projections_for_batch(
      owner_user_id="1",
      batch_id=1,
    )
  ] == ["approval-1", "approval-recovered"]


def test_scope_is_context_local_and_registers_session() -> None:
  registry = BatchApprovalProjectionRegistry()
  store = object()
  policy = object()
  scope = BatchApprovalScope(
    batch_id=7,
    owner_user_id="1",
    channel="tui",
    store=store,
    policy=policy,
    registry=registry,
  )
  session = _session()
  assert current_batch_approval_scope() is None
  with bind_batch_approval_scope(scope):
    assert current_batch_approval_scope() is scope
    scope.register_session(session)
  assert current_batch_approval_scope() is None
  _install_pending(session)
  projection = registry.find_projection(
    run_id="batch_7",
    approval_id="approval-1",
    owner_user_id="1",
    channel="tui",
  )
  assert projection is not None
  assert projection.store is store
  assert projection.policy is policy


def test_batch_freeze_returns_complete_snapshot_and_rejects_late_registration() -> None:
  registry = BatchApprovalProjectionRegistry()
  first = _session()
  _install_pending(first, "approval-first")
  registry.register_session(
    batch_id=7,
    owner_user_id="1",
    channel="tui",
    session=first,
  )

  frozen = asyncio.run(registry.freeze_batch(owner_user_id="1", batch_id=7))
  assert [projection.approval_id for projection in frozen] == ["approval-first"]

  late = _session()
  late.session_id = "late-stage-session"
  _install_pending(late, "approval-late")
  with pytest.raises(RuntimeError, match="admission is fenced"):
    registry.register_session(
      batch_id=7,
      owner_user_id="1",
      channel="tui",
      session=late,
    )
  assert [
    projection.approval_id
    for projection in registry.projections_for_batch(
      owner_user_id="1",
      batch_id=7,
    )
  ] == ["approval-first"]

  registry.release_batch_fence(owner_user_id="1", batch_id=7)
  registry.register_session(
    batch_id=7,
    owner_user_id="1",
    channel="tui",
    session=late,
  )
  assert [
    projection.approval_id
    for projection in registry.projections_for_batch(
      owner_user_id="1",
      batch_id=7,
    )
  ] == ["approval-first", "approval-late"]


def test_shutdown_fence_precedes_snapshot_and_permanently_closes_admission() -> None:
  registry = BatchApprovalProjectionRegistry()
  session = _session()
  registry.register_session(
    batch_id=9,
    owner_user_id="1",
    channel="tui",
    session=session,
  )
  _install_pending(session, "approval-before-shutdown")

  snapshots = asyncio.run(registry.begin_shutdown())
  assert [
    projection.approval_id
    for projection in snapshots[("1", 9)]
  ] == ["approval-before-shutdown"]

  late = _session()
  late.session_id = "post-shutdown-stage"
  with pytest.raises(RuntimeError, match="shutting down"):
    registry.register_session(
      batch_id=10,
      owner_user_id="1",
      channel="tui",
      session=late,
    )


def test_batch_freeze_rollback_restores_exact_preflight_projection_state() -> None:
  registry = BatchApprovalProjectionRegistry()
  session = _session()
  registry.register_session(
    batch_id=11,
    owner_user_id="1",
    channel="tui",
    session=session,
  )
  assert registry._projections == {}
  _install_pending(session, "approval-materialized-by-freeze")

  frozen = asyncio.run(registry.freeze_batch(owner_user_id="1", batch_id=11))
  assert [projection.approval_id for projection in frozen] == [
    "approval-materialized-by-freeze"
  ]
  registry.release_batch_fence(
    owner_user_id="1",
    batch_id=11,
    rollback=True,
  )

  assert registry._projections == {}


def test_batch_freeze_drains_admitted_approval_before_snapshot() -> None:
  async def run_case() -> None:
    registry = BatchApprovalProjectionRegistry()
    session = _session()
    store = SimpleNamespace(
      abort_unpublished_approval=lambda *args, **kwargs: None,
      fence_persistent_grants_for_cancellation=lambda *args, **kwargs: None,
      revoke_persistent_grants_for_approval=lambda *args, **kwargs: None,
    )
    scope = BatchApprovalScope(
      batch_id=12,
      owner_user_id="1",
      channel="tui",
      store=store,
      policy=object(),
      registry=registry,
    )
    scope.register_session(session)
    session.batch_approval_scope = scope
    admission = scope.acquire_admission(session)

    freeze_task = asyncio.create_task(
      registry.freeze_batch(owner_user_id="1", batch_id=12)
    )
    await asyncio.sleep(0)
    assert not freeze_task.done()
    _install_pending(session, "approval-admitted-before-freeze")
    admission.bind_request(
      request=SimpleNamespace(
        approval_id="approval-admitted-before-freeze",
        tool_call_id="tool-1",
        user_id="1",
        request_id="batch_12",
        run_id="batch_12",
        session_id=session.session_id,
        channel="tui",
      ),
      store=store,
    )
    admission.publish_pending()

    frozen = await freeze_task
    assert [projection.approval_id for projection in frozen] == [
      "approval-admitted-before-freeze"
    ]

  asyncio.run(run_case())


def test_batch_drain_retains_projection_published_then_removed_from_carrier() -> None:
  async def run_case() -> None:
    registry = BatchApprovalProjectionRegistry()
    session = _session()
    store = SimpleNamespace(
      abort_unpublished_approval=lambda *args, **kwargs: None,
      fence_persistent_grants_for_cancellation=lambda *args, **kwargs: None,
      revoke_persistent_grants_for_approval=lambda *args, **kwargs: None,
    )
    scope = BatchApprovalScope(
      batch_id=15,
      owner_user_id="1",
      channel="tui",
      store=store,
      policy=object(),
      registry=registry,
    )
    scope.register_session(session)
    admission = scope.acquire_admission(session)
    admission.bind_request(
      request=SimpleNamespace(
        approval_id="approval-transient-after-fence",
        tool_call_id="tool-1",
        user_id="1",
        request_id="batch_15",
        run_id="batch_15",
        session_id=session.session_id,
        channel="tui",
      ),
      store=store,
    )

    initial = registry.close_batch_admission(owner_user_id="1", batch_id=15)
    assert initial == ()
    _install_pending(session, "approval-transient-after-fence")
    admission.publish_pending()
    assert [
      projection.approval_id
      for projection in registry._projection_snapshot_for_batch(("1", 15))
    ] == ["approval-transient-after-fence"]

    session.pending_tools.pop("tool-1")
    session.approval_queues.pop("tool-1", None)
    drained = await registry.snapshot_after_batch_drain(
      owner_user_id="1",
      batch_id=15,
    )

    assert [projection.approval_id for projection in drained] == [
      "approval-transient-after-fence"
    ]
    assert [
      projection.approval_id
      for projection in registry.merge_projection_sets(initial, drained)
    ] == ["approval-transient-after-fence"]

  asyncio.run(run_case())


def test_cancelled_batch_freeze_reopens_admission_without_poisoning_registry() -> None:
  async def run_case() -> None:
    registry = BatchApprovalProjectionRegistry()
    session = _session()
    scope = BatchApprovalScope(
      batch_id=13,
      owner_user_id="1",
      channel="tui",
      store=object(),
      policy=object(),
      registry=registry,
    )
    scope.register_session(session)
    first_admission = scope.acquire_admission(session)

    freeze_task = asyncio.create_task(
      registry.freeze_batch(owner_user_id="1", batch_id=13)
    )
    await asyncio.sleep(0)
    assert not freeze_task.done()
    freeze_task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await freeze_task

    second_admission = scope.acquire_admission(session)
    second_admission.release()
    first_admission.release()
    assert registry._admission_gates[("1", 13)].active == 0

  asyncio.run(run_case())


def test_normal_batch_close_defers_carrier_removal_until_admission_drains() -> None:
  registry = BatchApprovalProjectionRegistry()
  session = _session()
  scope = BatchApprovalScope(
    batch_id=14,
    owner_user_id="1",
    channel="tui",
    store=object(),
    policy=object(),
    registry=registry,
  )
  scope.register_session(session)
  admission = scope.acquire_admission(session)

  registry.close_batch(owner_user_id="1", batch_id=14)

  assert ("1", 14, id(session)) in registry._carriers
  assert registry._admission_gates[("1", 14)].close_when_drained is True
  with pytest.raises(RuntimeError, match="admission is fenced"):
    scope.acquire_admission(session)

  admission.release()

  assert ("1", 14, id(session)) not in registry._carriers
  assert ("1", 14) not in registry._admission_gates
  assert registry.batch_keys() == set()
