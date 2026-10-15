from __future__ import annotations

import asyncio
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway.autonomous_runner import AutonomousTask
from agent_gateway.control_plane import autonomous_approval_drainer as drainer_module
from agent_gateway.control_plane.autonomous_approval_drainer import (
  AutonomousApprovalDeliveryCoordinator,
  PermanentAutonomousApprovalDeliveryError,
)


class FakeDeliveryStore:
  def __init__(
    self,
    row_count: int,
    *,
    retry_delay_ns: int = 1_000_000,
    quarantine_after: int = 5,
    failure_recorder_errors: int = 0,
    quarantine_errors: int = 0,
    recovery_errors: int = 0,
  ) -> None:
    now_ns = time.time_ns()
    self.rows = [
      {
        "delivery_sequence": sequence,
        "approval_id": f"approval-{sequence}",
        "tool_call_id": f"tool-{sequence}",
        "nonce": f"nonce-{sequence}",
        "state": "pending",
        "attempt_count": 0,
        "next_attempt_ns": now_ns,
        "last_attempt_ns": None,
      }
      for sequence in range(1, row_count + 1)
    ]
    self.selector_calls: list[dict[str, int | None]] = []
    self.high_water_calls = 0
    self.failure_calls: list[int] = []
    self.retry_delay_ns = retry_delay_ns
    self.quarantine_after = quarantine_after
    self.failure_recorder_errors = failure_recorder_errors
    self.quarantine_errors = quarantine_errors
    self.recovery_errors = recovery_errors
    self.recovery_error_event = asyncio.Event()

  async def autonomous_approval_delivery_recovery_window(
    self,
  ) -> dict[str, int]:
    self.high_water_calls += 1
    if self.recovery_errors > 0:
      self.recovery_errors -= 1
      if self.recovery_errors == 0:
        self.recovery_error_event.set()
      raise OSError("injected recovery-window read failure")
    return {
      "high_water": max(
        (
          int(row["delivery_sequence"])
          for row in self.rows
        ),
        default=0,
      ),
      "observed_at_ns": time.time_ns(),
    }

  async def list_pending_autonomous_approval_deliveries(
    self,
    *,
    limit: int,
    after_sequence: int,
    through_sequence: int | None,
  ) -> list[dict[str, Any]]:
    self.selector_calls.append({
      "limit": limit,
      "after_sequence": after_sequence,
      "through_sequence": through_sequence,
    })
    selected = [
      dict(row)
      for row in self.rows
      if (
        row["state"] == "pending"
        and int(row["delivery_sequence"]) > after_sequence
        and (
          through_sequence is None
          or int(row["delivery_sequence"]) <= through_sequence
        )
      )
    ]
    selected.sort(key=lambda row: int(row["delivery_sequence"]))
    return selected[:limit]

  async def get_autonomous_approval_delivery(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
  ) -> dict[str, Any] | None:
    return next(
      (
        dict(row)
        for row in self.rows
        if (
          row["approval_id"] == approval_id
          and row["tool_call_id"] == tool_call_id
          and row["nonce"] == nonce
        )
      ),
      None,
    )

  async def record_autonomous_approval_delivery_failure(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    error: str,
  ) -> dict[str, Any]:
    del error
    if self.failure_recorder_errors > 0:
      self.failure_recorder_errors -= 1
      raise OSError("injected failure recorder error")
    row = next(
      row
      for row in self.rows
      if (
        row["approval_id"] == approval_id
        and row["tool_call_id"] == tool_call_id
        and row["nonce"] == nonce
      )
    )
    row["attempt_count"] = int(row["attempt_count"]) + 1
    now_ns = time.time_ns()
    row["last_attempt_ns"] = now_ns
    row["next_attempt_ns"] = now_ns + self.retry_delay_ns
    if int(row["attempt_count"]) >= self.quarantine_after:
      row["state"] = "quarantined"
    self.failure_calls.append(int(row["delivery_sequence"]))
    return dict(row)

  async def quarantine_autonomous_approval_delivery(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    error: str,
  ) -> dict[str, Any]:
    if self.quarantine_errors > 0:
      self.quarantine_errors -= 1
      raise OSError("injected quarantine recorder error")
    outcome = await self.record_autonomous_approval_delivery_failure(
      approval_id,
      tool_call_id=tool_call_id,
      nonce=nonce,
      error=error,
    )
    row = next(
      row
      for row in self.rows
      if row["approval_id"] == approval_id
    )
    row["state"] = "quarantined"
    outcome["state"] = "quarantined"
    return outcome


class FakeRegistry:
  def __init__(self) -> None:
    self.failed: list[tuple[Any, str]] = []
    self.failure_event = asyncio.Event()

  def fail_autonomous_approval_delivery(
    self,
    task_id: Any,
    *,
    error: str,
  ) -> None:
    self.failed.append((task_id, error))
    self.failure_event.set()


def _coordinator(
  store: Any,
  *,
  registry: Any | None = None,
  batch_limit: int = 2,
  shutdown_timeout_seconds: float = 1.0,
) -> AutonomousApprovalDeliveryCoordinator:
  return AutonomousApprovalDeliveryCoordinator(
    store=store,
    registry=registry or object(),  # type: ignore[arg-type]
    batch_limit=batch_limit,
    shutdown_timeout_seconds=shutdown_timeout_seconds,
    fallback_retry_base_seconds=0.001,
  )


def test_snapshot_drain_pages_through_exact_bounded_high_water() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(5)
    coordinator = _coordinator(store)
    attempted: list[int] = []

    async def deliver(delivery: dict[str, Any]) -> None:
      sequence = int(delivery["delivery_sequence"])
      attempted.append(sequence)
      store.rows[sequence - 1]["state"] = "published"

    coordinator._deliver_one = deliver  # type: ignore[method-assign]

    assert await coordinator.drain_once() == 5
    assert attempted == [1, 2, 3, 4, 5]
    assert [
      call["after_sequence"]
      for call in store.selector_calls
    ] == [0, 2, 4]
    assert all(
      call["limit"] == 2
      and call["through_sequence"] == 5
      for call in store.selector_calls
    )

  asyncio.run(scenario())


def test_poison_rows_are_attempted_once_per_snapshot_without_spin() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(5)
    coordinator = _coordinator(store)
    attempted: list[int] = []

    async def fail(delivery: dict[str, Any]) -> None:
      sequence = int(delivery["delivery_sequence"])
      attempted.append(sequence)
      if sequence == 1:
        await store.record_autonomous_approval_delivery_failure(
          str(delivery["approval_id"]),
          tool_call_id=str(delivery["tool_call_id"]),
          nonce=str(delivery["nonce"]),
          error="already recorded by delivery path",
        )
      raise RuntimeError("injected delivery failure")

    coordinator._deliver_one = fail  # type: ignore[method-assign]

    assert await coordinator.drain_once() == 5
    assert attempted == [1, 2, 3, 4, 5]
    assert store.failure_calls == [1, 2, 3, 4, 5]
    assert [
      row["attempt_count"] for row in store.rows
    ] == [1, 1, 1, 1, 1]

  asyncio.run(scenario())


def test_wake_during_drain_is_not_lost() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(1)
    coordinator = _coordinator(store, batch_limit=1)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_delivered = asyncio.Event()
    attempted: list[int] = []

    async def deliver(delivery: dict[str, Any]) -> None:
      sequence = int(delivery["delivery_sequence"])
      attempted.append(sequence)
      if sequence == 1:
        first_entered.set()
        await release_first.wait()
      store.rows[sequence - 1]["state"] = "published"
      if sequence == 2:
        second_delivered.set()

    coordinator._deliver_one = deliver  # type: ignore[method-assign]
    coordinator.start()
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    store.rows.append({
      "delivery_sequence": 2,
      "approval_id": "approval-2",
      "tool_call_id": "tool-2",
      "nonce": "nonce-2",
      "state": "pending",
      "attempt_count": 0,
      "next_attempt_ns": time.time_ns(),
      "last_attempt_ns": None,
    })
    assert coordinator.wake() is True
    release_first.set()
    await asyncio.wait_for(second_delivered.wait(), timeout=1)
    await coordinator.shutdown()

    assert attempted == [1, 2]
    assert store.high_water_calls >= 2

  asyncio.run(scenario())


def test_shutdown_cancels_a_hung_drain_within_bound() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(1)
    coordinator = _coordinator(
      store,
      batch_limit=1,
      shutdown_timeout_seconds=0.01,
    )
    entered = asyncio.Event()
    never = asyncio.Event()

    async def hang(_delivery: dict[str, Any]) -> None:
      entered.set()
      await never.wait()

    coordinator._deliver_one = hang  # type: ignore[method-assign]
    coordinator.start()
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.wait_for(coordinator.shutdown(), timeout=0.5)

    assert coordinator._task is None
    assert coordinator.wake() is False
    assert not [
      task
      for task in asyncio.all_tasks()
      if (
        task is not asyncio.current_task()
        and task.get_name()
        == "autonomous-approval-delivery-drainer"
        and not task.done()
      )
    ]

  asyncio.run(scenario())


def test_transient_failure_gets_one_shot_retry_without_polling() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(
      1,
      retry_delay_ns=1_000_000,
    )
    coordinator = _coordinator(store, batch_limit=1)
    delivered = asyncio.Event()
    attempts = 0

    async def fail_once(delivery: dict[str, Any]) -> None:
      nonlocal attempts
      attempts += 1
      if attempts == 1:
        raise OSError("injected transient append failure")
      store.rows[0]["state"] = "published"
      delivered.set()

    coordinator._deliver_one = fail_once  # type: ignore[method-assign]
    coordinator.start()
    await asyncio.wait_for(delivered.wait(), timeout=1)
    await coordinator.shutdown()

    assert attempts == 2
    assert store.failure_calls == [1]
    assert store.rows[0]["state"] == "published"

  asyncio.run(scenario())


def test_retry_exhaustion_quarantines_and_fails_owner() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(
      1,
      retry_delay_ns=1_000_000,
      quarantine_after=2,
    )
    store.rows[0]["task_id"] = "bg-1"
    registry = FakeRegistry()
    coordinator = _coordinator(
      store,
      registry=registry,
      batch_limit=1,
    )

    async def always_fail(_delivery: dict[str, Any]) -> None:
      raise OSError("injected persistent append failure")

    coordinator._deliver_one = always_fail  # type: ignore[method-assign]
    coordinator.start()
    await asyncio.wait_for(
      registry.failure_event.wait(),
      timeout=1,
    )
    await coordinator.shutdown()

    assert store.rows[0]["state"] == "quarantined"
    assert store.rows[0]["attempt_count"] == 2
    assert len(registry.failed) == 1
    assert registry.failed[0][0] == "bg-1"

  asyncio.run(scenario())


def test_failure_recorder_error_gets_bounded_fallback_retry() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(
      1,
      failure_recorder_errors=1,
    )
    coordinator = _coordinator(store, batch_limit=1)
    delivered = asyncio.Event()
    attempts = 0

    async def fail_then_deliver(
      _delivery: dict[str, Any],
    ) -> None:
      nonlocal attempts
      attempts += 1
      if attempts == 1:
        raise OSError("injected delivery failure")
      store.rows[0]["state"] = "published"
      delivered.set()

    coordinator._deliver_one = fail_then_deliver  # type: ignore[method-assign]
    coordinator.start()
    await asyncio.wait_for(delivered.wait(), timeout=1)
    await coordinator.shutdown()

    assert attempts == 2
    assert store.rows[0]["state"] == "published"

  asyncio.run(scenario())


def test_quarantine_recorder_error_gets_bounded_fallback_retry() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(
      1,
      quarantine_errors=1,
    )
    store.rows[0]["task_id"] = "bg-1"
    registry = FakeRegistry()
    coordinator = _coordinator(
      store,
      registry=registry,
      batch_limit=1,
    )

    async def permanently_fail(
      _delivery: dict[str, Any],
    ) -> None:
      raise PermanentAutonomousApprovalDeliveryError(
        "owner is gone"
      )

    coordinator._deliver_one = permanently_fail  # type: ignore[method-assign]
    coordinator.start()
    await asyncio.wait_for(
      registry.failure_event.wait(),
      timeout=1,
    )
    await coordinator.shutdown()

    assert store.rows[0]["state"] == "quarantined"
    assert len(registry.failed) == 1

  asyncio.run(scenario())


def test_startup_recovery_read_failure_gets_one_shot_retry() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(1, recovery_errors=1)
    coordinator = _coordinator(store, batch_limit=1)
    delivered = asyncio.Event()

    async def deliver(_delivery: dict[str, Any]) -> None:
      store.rows[0]["state"] = "published"
      delivered.set()

    coordinator._deliver_one = deliver  # type: ignore[method-assign]
    coordinator.start()
    await asyncio.wait_for(delivered.wait(), timeout=1)
    await coordinator.shutdown()

    assert store.high_water_calls == 2
    assert coordinator.fatal_error is None

  asyncio.run(scenario())


def test_sustained_recovery_read_failure_reports_health_and_keeps_retrying() -> None:
  async def scenario() -> None:
    store = FakeDeliveryStore(1, recovery_errors=5)
    coordinator = _coordinator(store, batch_limit=1)
    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()
    delivered = asyncio.Event()

    async def deliver(_delivery: dict[str, Any]) -> None:
      delivery_started.set()
      await release_delivery.wait()
      store.rows[0]["state"] = "published"
      delivered.set()

    coordinator._deliver_one = deliver  # type: ignore[method-assign]
    coordinator.start()
    await asyncio.wait_for(delivery_started.wait(), timeout=1)

    # Five failed drains are reported, never latched: the sixth drain ran.
    assert store.high_water_calls == 6
    assert coordinator.fatal_error is not None
    assert "failing after 5 attempts" in coordinator.fatal_error
    assert coordinator.wake() is True

    release_delivery.set()
    await asyncio.wait_for(delivered.wait(), timeout=1)
    for _attempt in range(10):
      if coordinator.fatal_error is None:
        break
      await asyncio.sleep(0)
    await coordinator.shutdown()

    assert coordinator.fatal_error is None

  asyncio.run(scenario())


@pytest.mark.parametrize(
  ("rows", "error"),
  [
    (
      [
        {"delivery_sequence": 2},
        {"delivery_sequence": 1},
      ],
      "broke ordering",
    ),
    (
      [
        {"delivery_sequence": sequence}
        for sequence in range(1, 4)
      ],
      "exceeded its bound",
    ),
  ],
)
def test_selector_contract_fails_closed(
  rows: list[dict[str, int]],
  error: str,
) -> None:
  class InvalidStore:
    async def autonomous_approval_delivery_recovery_window(
      self,
    ) -> dict[str, int]:
      return {
        "high_water": 3,
        "observed_at_ns": time.time_ns(),
      }

    async def list_pending_autonomous_approval_deliveries(
      self,
      **_kwargs: Any,
    ) -> list[dict[str, int]]:
      return rows

  async def scenario() -> None:
    coordinator = _coordinator(InvalidStore(), batch_limit=2)

    async def no_op(_delivery: dict[str, Any]) -> None:
      return None

    coordinator._deliver_one = no_op  # type: ignore[method-assign]
    for row in rows:
      row["next_attempt_ns"] = 1
    with pytest.raises(RuntimeError, match=error):
      await coordinator.drain_once()

  asyncio.run(scenario())


def test_drainer_has_no_polling_sleep() -> None:
  source_path = (
    Path(__file__).parents[1]
    / "agent_gateway"
    / "control_plane"
    / "autonomous_approval_drainer.py"
  )
  source = source_path.read_text(encoding="utf-8")
  assert "asyncio.sleep" not in source
  assert "time.sleep" not in source


def _running_task(**overrides: Any) -> AutonomousTask:
  # Only the fields _deliver_one reads before handing the row to delivery.
  record = object.__new__(AutonomousTask)
  fields = {
    "task_id": "bg_1",
    "control_run_id": "bg_1",
    "session_id": "bg_1",
    "channel_id": "chan_1",
    "owner_user_id": "user-1",
    "channel": "cli",
    "state": "running",
    "proc": None,
    **overrides,
  }
  for name, value in fields.items():
    setattr(record, name, value)
  return record


class _TaskRegistry:
  def __init__(self, record: AutonomousTask) -> None:
    self._tasks = {record.task_id: record}


class _RequestStore:
  def __init__(self, request: Any) -> None:
    self.request = request

  async def get(self, approval_id: str) -> Any:
    assert approval_id == self.request.approval_id
    return self.request


def _delivery_row(**overrides: Any) -> dict[str, Any]:
  return {
    "approval_id": "approval-1",
    "tool_call_id": "tool-1",
    "nonce": "nonce-1",
    "task_id": "bg_1",
    "control_run_id": "bg_1",
    "session_id": "bg_1",
    "channel_id": "chan_1",
    "allow_tool_type": False,
    "approved": True,
    "state": "pending",
    **overrides,
  }


def _durable_request() -> SimpleNamespace:
  return SimpleNamespace(
    approval_id="approval-1",
    tool_call_id="tool-1",
    user_id="user-1",
    request_id="bg_1",
    run_id="bg_1",
    session_id="bg_1",
    channel="cli",
    decider_id="user-1",
    state="approved",
  )


def test_drainer_hands_a_matching_row_to_delivery(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  delivered: list[dict[str, Any]] = []

  async def deliver(**kwargs: Any) -> None:
    delivered.append(kwargs)

  monkeypatch.setattr(drainer_module, "deliver_autonomous_approval_outbox", deliver)
  coordinator = _coordinator(
    _RequestStore(_durable_request()),
    registry=_TaskRegistry(_running_task()),
  )

  asyncio.run(coordinator._deliver_one(_delivery_row()))  # type: ignore[arg-type]

  assert [call["approval_id"] for call in delivered] == ["approval-1"]


def test_drainer_redelivers_uncaptured_row_from_unrestricted_launch(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  delivered: list[dict[str, Any]] = []

  async def deliver(**kwargs: Any) -> None:
    delivered.append(kwargs)

  monkeypatch.setattr(drainer_module, "deliver_autonomous_approval_outbox", deliver)
  record = _running_task(channel=None)
  request = _durable_request()
  coordinator = _coordinator(
    _RequestStore(request),
    registry=_TaskRegistry(record),
  )
  asyncio.run(coordinator._deliver_one(_delivery_row()))  # type: ignore[arg-type]
  assert record.channel is None
  assert len(delivered) == 1
  assert delivered[0]["request_record"] is request
  assert delivered[0]["channel"] == "cli"


@pytest.mark.parametrize(
  "changed",
  [
    {"control_run_id": "bg_other"},
    {"session_id": "bg_other"},
    {"channel_id": "chan_other"},
  ],
)
def test_drainer_quarantines_a_row_whose_run_authority_changed(
  monkeypatch: pytest.MonkeyPatch,
  changed: dict[str, str],
) -> None:
  delivered: list[dict[str, Any]] = []

  async def deliver(**kwargs: Any) -> None:
    delivered.append(kwargs)

  monkeypatch.setattr(drainer_module, "deliver_autonomous_approval_outbox", deliver)
  coordinator = _coordinator(
    _RequestStore(_durable_request()),
    registry=_TaskRegistry(_running_task()),
  )

  with pytest.raises(
    PermanentAutonomousApprovalDeliveryError,
    match="outbox identity mismatch",
  ):
    asyncio.run(coordinator._deliver_one(_delivery_row(**changed)))  # type: ignore[arg-type]
  assert delivered == []


def test_drainer_stops_delivery_when_user_removed_from_roster(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  delivered: list[dict[str, Any]] = []

  async def deliver(**kwargs: Any) -> None:
    delivered.append(kwargs)

  monkeypatch.setattr(drainer_module, "deliver_autonomous_approval_outbox", deliver)

  alice_task = _running_task(task_id="bg_alice", control_run_id="bg_alice", session_id="bg_alice", channel_id="chan_alice", owner_user_id="alice")
  bob_task = _running_task(task_id="bg_bob", control_run_id="bg_bob", session_id="bg_bob", channel_id="chan_bob", owner_user_id="bob")

  class MultiTaskRegistry:
    def __init__(self, tasks: list[AutonomousTask], roster: dict[str, dict[str, Any]]) -> None:
      self._tasks = {t.task_id: t for t in tasks}
      self._roster = roster

    def live_user_role(self, user_id: str, email: str | None = None) -> str | None:
      del email
      entry = self._roster.get(user_id)
      if not entry:
        return None
      return str(entry.get("role") or "owner")

    def _mcp_user_key_lookup(self, user_id: str, email: str | None = None) -> dict[str, Any] | None:
      del email
      return self._roster.get(user_id)

  # alice was removed from the roster; bob is still in the roster
  roster = {"bob": {"role": "owner"}}
  registry = MultiTaskRegistry([alice_task, bob_task], roster)

  req_alice = SimpleNamespace(
    approval_id="app-alice",
    tool_call_id="tool-alice",
    user_id="alice",
    request_id="bg_alice",
    run_id="bg_alice",
    session_id="bg_alice",
    channel="cli",
    decider_id="alice",
    state="approved",
  )
  req_bob = SimpleNamespace(
    approval_id="app-bob",
    tool_call_id="tool-bob",
    user_id="bob",
    request_id="bg_bob",
    run_id="bg_bob",
    session_id="bg_bob",
    channel="cli",
    decider_id="bob",
    state="approved",
  )

  class MultiRequestStore:
    async def get(self, approval_id: str) -> Any:
      if approval_id == "app-alice":
        return req_alice
      if approval_id == "app-bob":
        return req_bob
      return None

  coordinator = _coordinator(MultiRequestStore(), registry=registry)

  row_alice = _delivery_row(approval_id="app-alice", tool_call_id="tool-alice", nonce="n-alice", task_id="bg_alice", control_run_id="bg_alice", session_id="bg_alice", channel_id="chan_alice")
  row_bob = _delivery_row(approval_id="app-bob", tool_call_id="tool-bob", nonce="n-bob", task_id="bg_bob", control_run_id="bg_bob", session_id="bg_bob", channel_id="chan_bob")

  # 1. Attempting delivery for alice fails permanently because she is removed from roster
  with pytest.raises(PermanentAutonomousApprovalDeliveryError, match="no longer admitted on the roster"):
    asyncio.run(coordinator._deliver_one(row_alice))  # type: ignore[arg-type]

  # 2. Attempting delivery for bob succeeds because bob is still on the roster
  asyncio.run(coordinator._deliver_one(row_bob))  # type: ignore[arg-type]
  # Delivered calls contain bob only; alice was stopped
  assert [call["user_id"] for call in delivered] == ["bob"]
