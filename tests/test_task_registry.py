# ruff: noqa: E402

import asyncio
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.runner_background_lifecycle import RunnerBackgroundLifecycleMixin
from agent_gateway.runner_background_tasks import (
  background_task_payload,
)
from agent_gateway.task_registry import (
  ParentMessage,
  ResumeSuccessorConflictError,
  TaskDurableEventConflictError,
  TaskEntry,
  TaskRegistry,
  TaskState,
  _TERMINAL_STATES,
  make_progress_tracker,
)


def _bind_receipt(*, provider: str, model: str) -> dict[str, str]:
  return {
    "capability_id": "node.implement",
    "provider": provider,
    "model": model,
    "effort": "high",
    "policy_id": "test-policy",
    "policy_version": "1",
    "credential_principal": "user",
    "run_mode": "interactive",
  }


def _run(coro):
  return asyncio.run(coro)


def _child_report(summary: str = "done") -> dict[str, object]:
  return {
    "kind": "report",
    "version": "1",
    "report": {
      "summary": summary,
      "findings": [],
      "artifacts": [],
      "caveats": [],
    },
    "usage": {},
    "tools_used": [],
    "fms_results": None,
    "artifact_events": None,
    "warning": None,
  }


def _canonical_skipped_result() -> dict[str, object]:
  digest = f"sha256:{'1' * 64}"
  return {
    "schema_version": "2.0",
    "task_result_id": "result-bg-7",
    "logical_task": {
      "kind": "ordinary_delegation",
      "delegation_id": "delegation-bg-7",
      "operation": {
        "namespace": "agent-workflow",
        "name": "research",
        "version": "v1",
        "digest": digest,
      },
    },
    "attempt": {
      "attempt_number": 1,
      "attempt_id": "attempt-bg-7",
      "physical_task_id": "bg_7",
    },
    "execution": {
      "status": "skipped",
      "terminal_reason": "not required",
    },
    "outcome": None,
    "evidence": {"observed_sources": [], "tools_used": []},
    "values": {
      "terminal_narrative": None,
      "projection": None,
      "artifacts": [],
    },
    "observation": {
      "transcript": {"kind": "child_transcript", "owner_id": "bg_7"},
      "activity": {"kind": "child_activity", "owner_id": "bg_7"},
      "usage": {},
    },
    "provenance": {
      "admitted_task_digest": digest,
      "model_bind_digest": digest,
      "capability_binding_digest": digest,
      "tool_grant_digest": digest,
    },
  }


class _RecordingListener:
  def __init__(self) -> None:
    self.transitions: list[tuple[str, TaskState, TaskState]] = []

  def on_transition(self, entry: TaskEntry, old_state: TaskState, new_state: TaskState) -> None:
    self.transitions.append((entry.task_id, old_state, new_state))


class _BackgroundLifecycleHarness(RunnerBackgroundLifecycleMixin):
  def __init__(self) -> None:
    self._task_registry = TaskRegistry()
    self.listener = _RecordingListener()
    self._task_registry.add_listener(self.listener)
    self._runner_id = "runner-test"
    self._role = "writer"
    self._sid = "session-test"
    self.durable_events: list[dict[str, object]] = []
    self.append_invocations = 0

  def _ensure_sub_agent_semaphore(self) -> None:
    return None

  async def _append_durable_event(self, event: dict[str, object]) -> None:
    self.append_invocations += 1
    self.durable_events.append(event)


def test_task_registry_register_transition_get_and_list_tasks() -> None:
  registry = TaskRegistry()
  first = registry.register("background_agent", agent_name="writer", owner="alice")
  second = registry.register("background_agent", agent_name="reviewer")
  first.started_at = 20.0
  second.started_at = 10.0

  registry.transition(first.task_id, TaskState.RUNNING)
  registry.transition(second.task_id, TaskState.COMPLETED, result={"summary": "done"})

  assert first.metadata == {"owner": "alice"}
  assert registry.get(first.task_id) is first
  assert registry.list_tasks() == [second, first]
  assert registry.list_tasks(state=TaskState.COMPLETED) == [second]
  assert second.result == {"summary": "done"}
  assert second.completed_at is not None


def test_transition_terminal_guard_is_no_op() -> None:
  registry = TaskRegistry()
  listener = _RecordingListener()
  registry.add_listener(listener)
  entry = registry.register("background_agent")

  registry.transition(entry.task_id, TaskState.COMPLETED, result={"ok": True})
  completed_at = entry.completed_at
  result = registry.transition(entry.task_id, TaskState.FAILED, error={"code": "late"})

  assert result is entry
  assert entry.state == TaskState.COMPLETED
  assert entry.result == {"ok": True}
  assert entry.error is None
  assert entry.completed_at == completed_at
  assert listener.transitions == [(entry.task_id, TaskState.PENDING, TaskState.COMPLETED)]


def test_kill_records_intent_and_cancels_without_pretransitioning() -> None:
  async def _case() -> None:
    registry = TaskRegistry()
    entry = registry.register("background_agent")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _worker() -> None:
      started.set()
      try:
        await asyncio.sleep(60)
      except asyncio.CancelledError:
        cancelled.set()
        raise

    task = asyncio.create_task(_worker())
    await started.wait()
    entry.asyncio_task = task
    registry.transition(entry.task_id, TaskState.RUNNING)

    assert registry.kill(entry.task_id) is True
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    await asyncio.gather(task, return_exceptions=True)
    assert entry.termination_intent == "killed"
    assert entry.state == TaskState.RUNNING
    assert entry.completed is False
    assert entry.completed_at is None

  _run(_case())


def test_kill_records_parent_cancellation_and_does_not_repeat_same_request() -> None:
  class _Cancelable:
    def __init__(self) -> None:
      self.cancel_count = 0

    def cancel(self) -> None:
      self.cancel_count += 1

  registry = TaskRegistry()
  entry = registry.register("background_agent")
  task = _Cancelable()
  entry.asyncio_task = task  # type: ignore[assignment]
  registry.transition(entry.task_id, TaskState.RUNNING)

  assert registry.kill(entry.task_id, termination_intent="cancelled") is True
  assert registry.kill(entry.task_id, termination_intent="cancelled") is False
  assert registry.kill(entry.task_id, termination_intent="killed") is False
  assert entry.termination_intent == "cancelled"
  assert task.cancel_count == 1
  assert entry.state == TaskState.RUNNING


@pytest.mark.parametrize("termination_intent", ["cancelled", "killed"])
def test_background_finalizer_persists_canonical_termination_result(
  termination_intent: str,
) -> None:
  async def _case() -> None:
    harness = _BackgroundLifecycleHarness()
    entry = harness._task_registry.register("background_agent")
    entry.metadata["sub_agent_id"] = "sub0:session-test"
    started = asyncio.Event()
    completed: list[str] = []

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[None, None]:
      started.set()
      await asyncio.sleep(60)
      return None, None

    async def _on_complete(completed_entry: TaskEntry) -> None:
      completed.append(completed_entry.task_id)

    task = asyncio.create_task(
      harness._run_background_agent(
        entry,
        _handler,
        {},
        0,
        on_complete=_on_complete,
      )
    )
    entry.asyncio_task = task
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)
    await started.wait()

    assert harness._task_registry.kill(
      entry.task_id,
      termination_intent=termination_intent,  # type: ignore[arg-type]
    )
    await task

    assert entry.state == TaskState.KILLED
    assert entry.error is None
    assert entry.result is not None
    assert entry.result["reason"] == termination_intent
    assert entry.result["usage"] == {
      "input_tokens": 0,
      "output_tokens": 0,
      "turn_count": 0,
    }
    assert entry.result["tools_used"] == []
    assert completed == [entry.task_id]
    assert len(harness.durable_events) == 1
    durable = harness.durable_events[0]
    assert durable["type"] == "task_completed"
    assert durable["final_state"] == "killed"
    assert durable["result"] == entry.result
    assert durable["error"] is None
    assert entry.completion_persistence_state == "committed"
    assert harness.append_invocations == 1
    assert [
      transition
      for transition in harness.listener.transitions
      if transition[2] in _TERMINAL_STATES
    ] == [(entry.task_id, TaskState.RUNNING, TaskState.KILLED)]

  _run(_case())


def test_cancelled_accepted_report_is_durably_killed() -> None:
  async def _case() -> None:
    harness = _BackgroundLifecycleHarness()
    entry = harness._task_registry.register("background_agent")
    entry.metadata["sub_agent_id"] = "sub0:session-test"
    accepted_result = {
      "kind": "report",
      "version": "1",
      "report": {
        "summary": "Accepted before cancellation",
        "findings": [],
        "artifacts": [],
        "caveats": [],
      },
      "usage": {"input_tokens": 3},
      "tools_used": ["lookup"],
      "fms_results": None,
      "artifact_events": None,
      "warning": "observed_terminal_signals=cancelled",
    }
    staged = asyncio.Event()

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[dict[str, object], None]:
      entry.result = accepted_result
      staged.set()
      await asyncio.Event().wait()
      return accepted_result, None

    task = asyncio.create_task(
      harness._run_background_agent(entry, _handler, {}, 0)
    )
    entry.asyncio_task = task
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)
    await staged.wait()

    assert harness._task_registry.kill(
      entry.task_id,
      termination_intent="cancelled",
    )
    await task

    assert entry.state == TaskState.KILLED
    assert entry.termination_intent == "cancelled"
    assert entry.result is not None
    assert entry.result["status"] == "interrupted"
    assert entry.result["reason"] == "cancelled"
    assert entry.error is None
    assert entry.completion_persistence_state == "committed"
    assert harness.append_invocations == 1
    assert len(harness.durable_events) == 1
    durable = harness.durable_events[0]
    assert durable["type"] == "task_completed"
    assert durable["task_id"] == entry.task_id
    assert durable["final_state"] == "killed"
    assert durable["result"] == entry.result
    assert durable["error"] is None

  _run(_case())


def test_background_finalizer_preserves_result_accepted_during_kill_race() -> None:
  async def _case() -> None:
    append_started = asyncio.Event()
    release_append = asyncio.Event()

    class _BlockedAppendHarness(_BackgroundLifecycleHarness):
      async def _append_durable_event(self, event: dict[str, object]) -> None:
        append_started.set()
        await release_append.wait()
        await super()._append_durable_event(event)

    harness = _BlockedAppendHarness()
    entry = harness._task_registry.register("background_agent")
    accepted_result = {
      "kind": "report",
      "version": "1",
      "report": {
        "summary": "Accepted before cancellation",
        "findings": [],
        "artifacts": [],
        "caveats": [],
      },
      "usage": {},
      "tools_used": [],
      "fms_results": None,
      "artifact_events": None,
      "warning": None,
    }

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[dict[str, object], None]:
      return accepted_result, None

    task = asyncio.create_task(
      harness._run_background_agent(entry, _handler, {}, 0)
    )
    entry.asyncio_task = task
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)
    await append_started.wait()

    assert harness._task_registry.kill(entry.task_id)
    release_append.set()
    await task

    assert entry.state == TaskState.COMPLETED
    assert entry.result == accepted_result
    assert entry.error is None
    assert len(harness.durable_events) == 1
    assert harness.durable_events[0]["final_state"] == "completed"
    assert harness.durable_events[0]["result"] == accepted_result
    assert entry.completion_persistence_state == "committed"
    assert harness.append_invocations == 1
    assert [
      transition
      for transition in harness.listener.transitions
      if transition[2] in _TERMINAL_STATES
    ] == [(entry.task_id, TaskState.RUNNING, TaskState.COMPLETED)]

  _run(_case())


def test_background_finalizer_marks_real_failed_child_result_failed_and_visible() -> None:
  async def _case() -> None:
    harness = _BackgroundLifecycleHarness()
    entry = harness._task_registry.register("background_agent")
    failed_error = {"code": "provider_error", "message": "provider failed"}

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[None, dict[str, object]]:
      return None, failed_error

    completed: list[TaskEntry] = []
    task = asyncio.create_task(
      harness._run_background_agent(
        entry,
        _handler,
        {},
        0,
        on_complete=completed.append,
      )
    )
    entry.asyncio_task = task
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)
    await task

    assert entry.state == TaskState.FAILED
    assert entry.error == failed_error
    assert entry.result is None
    assert completed == [entry]
    assert len(harness.durable_events) == 1
    durable = harness.durable_events[0]
    assert durable["final_state"] == "failed"
    assert durable["result"] is None
    assert durable["error"] == failed_error

  _run(_case())


def test_background_finalizer_append_failure_does_not_publish_success() -> None:
  async def _case() -> None:
    class _RaisingAppendHarness(_BackgroundLifecycleHarness):
      async def _append_durable_event(self, event: dict[str, object]) -> None:
        _ = event
        self.append_invocations += 1
        raise RuntimeError("durable append failed")

    harness = _RaisingAppendHarness()
    entry = harness._task_registry.register("background_agent")

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[dict[str, object], None]:
      return {"response": "accepted"}, None

    completed: list[TaskEntry] = []
    task = asyncio.create_task(
      harness._run_background_agent(
        entry,
        _handler,
        {},
        0,
        on_complete=completed.append,
      )
    )
    entry.asyncio_task = task
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)

    with pytest.raises(RuntimeError, match="durable append failed"):
      await task

    assert entry.state == TaskState.INTERRUPTED
    assert entry.result is None
    assert entry.error is not None
    assert (
      entry.error["code"]
      == "background_completion_persistence_uncertain"
    )
    assert entry.pending_final_state == TaskState.FAILED
    assert entry.pending_result is not None
    assert entry.pending_result["reason"] == "runtime_error"
    assert entry.completion_persistence_state == "uncertain"
    assert "durable append failed" in str(entry.completion_persistence_error)
    assert harness.append_invocations == 2
    assert harness.durable_events == []
    assert completed == []
    assert [
      transition
      for transition in harness.listener.transitions
      if transition[2] in _TERMINAL_STATES
    ] == [(entry.task_id, TaskState.RUNNING, TaskState.INTERRUPTED)]

  _run(_case())


@pytest.mark.parametrize(
  "lookup_failure_type",
  [OSError, asyncio.CancelledError],
)
def test_primary_completion_lookup_uncertainty_does_not_write_fallback(
  lookup_failure_type: type[BaseException],
) -> None:
  async def _case() -> None:
    class _PostFsyncUnknownHarness(_BackgroundLifecycleHarness):
      async def _append_durable_event(
        self,
        event: dict[str, object],
      ) -> None:
        self.append_invocations += 1
        self.durable_events.append(event)
        raise RuntimeError("append failed after fsync")

      async def _confirmed_durable_background_completion(
        self,
        _bg_task: TaskEntry,
      ) -> TaskEntry | None:
        raise lookup_failure_type("durable lookup outcome unknown")

    harness = _PostFsyncUnknownHarness()
    harness._writer_lease_poisoned = False
    entry = harness._task_registry.register("background_agent")
    accepted_result = {
      "kind": "report",
      "version": "1",
      "report": {
        "summary": "Durably accepted before lookup failed",
        "findings": [],
        "artifacts": [],
        "caveats": [],
      },
      "usage": {},
      "tools_used": [],
      "fms_results": None,
      "artifact_events": None,
      "warning": None,
    }
    completed: list[TaskEntry] = []

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[dict[str, object], None]:
      return accepted_result, None

    task = asyncio.create_task(
      harness._run_background_agent(
        entry,
        _handler,
        {},
        0,
        on_complete=completed.append,
      )
    )
    entry.asyncio_task = task
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)

    with pytest.raises(lookup_failure_type):
      await task
    await asyncio.sleep(0)
    pending_reconciliations = tuple(
      getattr(harness, "_pending_background_initializations", ())
    )
    if pending_reconciliations:
      await asyncio.gather(
        *pending_reconciliations,
        return_exceptions=True,
      )

    assert harness.append_invocations == 1
    assert len(harness.durable_events) == 1
    assert harness.durable_events[0]["final_state"] == "completed"
    assert harness.durable_events[0]["result"] == accepted_result
    assert entry.state == TaskState.INTERRUPTED
    assert entry.result is None
    assert entry.pending_final_state == TaskState.COMPLETED
    assert entry.completion_persistence_state == "uncertain"
    assert entry.completion_callback_invoked is False
    assert completed == []
    assert harness._writer_lease_poisoned is True

  _run(_case())


def test_background_finalizer_hanging_append_and_shield_cancel_are_bounded() -> None:
  async def _case() -> None:
    append_started = asyncio.Event()
    release_append = asyncio.Event()

    class _HangingAppendHarness(_BackgroundLifecycleHarness):
      async def _append_durable_event(self, event: dict[str, object]) -> None:
        self.append_invocations += 1
        append_started.set()
        while not release_append.is_set():
          try:
            await release_append.wait()
          except asyncio.CancelledError:
            continue
        self.durable_events.append(event)

    harness = _HangingAppendHarness()
    harness._background_completion_persist_timeout_seconds = 0.01
    entry = harness._task_registry.register("background_agent")
    accepted_result = {
      "kind": "report",
      "version": "1",
      "report": {
        "summary": "Accepted before persistence stalled",
        "findings": [],
        "artifacts": [],
        "caveats": [],
      },
      "usage": {},
      "tools_used": [],
      "fms_results": None,
      "artifact_events": None,
      "warning": None,
    }

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[dict[str, object], None]:
      return accepted_result, None

    task = asyncio.create_task(
      harness._run_background_agent(entry, _handler, {}, 0)
    )
    entry.asyncio_task = task
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)
    await append_started.wait()

    started_at = asyncio.get_running_loop().time()
    assert harness._task_registry.kill(entry.task_id)
    with pytest.raises(asyncio.TimeoutError):
      await task
    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < 0.2
    assert entry.state == TaskState.INTERRUPTED
    assert entry.result is None
    assert entry.pending_final_state == TaskState.COMPLETED
    assert entry.completion_persistence_state == "uncertain"
    assert harness.append_invocations == 1
    assert harness.durable_events == []

    release_append.set()
    for _ in range(100):
      if entry.state == TaskState.COMPLETED:
        break
      await asyncio.sleep(0)

    assert entry.state == TaskState.COMPLETED
    assert entry.result == accepted_result
    assert entry.error is None
    assert entry.completion_persistence_state == "committed"
    assert harness.append_invocations == 1
    assert len(harness.durable_events) == 1
    assert harness.durable_events[0]["final_state"] == "completed"
    assert [
      transition
      for transition in harness.listener.transitions
      if transition[2] in _TERMINAL_STATES
    ] == [
      (entry.task_id, TaskState.RUNNING, TaskState.INTERRUPTED),
      (entry.task_id, TaskState.INTERRUPTED, TaskState.COMPLETED),
    ]

  _run(_case())


def test_cancelled_detached_completion_reconciliation_poisons_writer_lease() -> None:
  async def _case() -> None:
    lookup_started = asyncio.Event()

    class _BlockedReconciliationHarness(_BackgroundLifecycleHarness):
      async def _confirmed_durable_background_completion(
        self,
        _bg_task: TaskEntry,
      ) -> TaskEntry | None:
        lookup_started.set()
        await asyncio.Future()

    harness = _BlockedReconciliationHarness()
    harness._writer_lease_poisoned = False
    entry = harness._task_registry.register("background_agent")
    harness._task_registry.transition(entry.task_id, TaskState.INTERRUPTED)
    entry.completion_finalizer_detached = True

    harness._schedule_detached_completion_reconciliation(entry)
    reconciliation_task = next(iter(harness._pending_background_initializations))
    await lookup_started.wait()
    reconciliation_task.cancel()
    await asyncio.gather(reconciliation_task, return_exceptions=True)
    await asyncio.sleep(0)

    assert reconciliation_task.cancelled()
    assert harness._writer_lease_poisoned is True
    assert entry.state == TaskState.INTERRUPTED
    assert entry.completion_finalizer_detached is True

  _run(_case())


def test_failed_detached_completion_lookup_poisons_writer_lease() -> None:
  async def _case() -> None:
    class _FailedReconciliationHarness(_BackgroundLifecycleHarness):
      async def _confirmed_durable_background_completion(
        self,
        _bg_task: TaskEntry,
      ) -> TaskEntry | None:
        raise RuntimeError("durable lookup failed")

    harness = _FailedReconciliationHarness()
    harness._writer_lease_poisoned = False
    entry = harness._task_registry.register("background_agent")
    harness._task_registry.transition(entry.task_id, TaskState.INTERRUPTED)
    entry.completion_finalizer_detached = True

    harness._schedule_detached_completion_reconciliation(entry)
    reconciliation_task = next(iter(harness._pending_background_initializations))
    await reconciliation_task
    await asyncio.sleep(0)

    assert harness._writer_lease_poisoned is True
    assert entry.state == TaskState.INTERRUPTED
    assert entry.completion_finalizer_detached is True

  _run(_case())


@pytest.mark.parametrize(
  ("was_cancelled", "expected_intent"),
  [(True, "cancelled"), (False, "killed")],
)
def test_shutdown_routes_intent_and_durably_reconciles(
  monkeypatch: pytest.MonkeyPatch,
  was_cancelled: bool,
  expected_intent: str,
) -> None:
  import agent_gateway.runner as gateway_runner

  class _Cancelable:
    def __init__(self) -> None:
      self.cancel_count = 0

    def cancel(self) -> None:
      self.cancel_count += 1

  async def _case() -> None:
    harness = _BackgroundLifecycleHarness()
    entry = harness._task_registry.register("background_agent")
    task = _Cancelable()
    entry.asyncio_task = task  # type: ignore[assignment]
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)

    async def _drain(
      running_entries: object,
      _pending: object,
      **kwargs: object,
    ) -> None:
      selected = list(running_entries)  # type: ignore[arg-type]
      kill_task = kwargs["kill_task"]
      kill_task(selected[0].task_id)  # type: ignore[operator]

    monkeypatch.setattr(
      gateway_runner,
      (
        "_drain_cancelled_background_tasks"
        if was_cancelled
        else "_drain_still_pending_background_tasks"
      ),
      _drain,
    )

    await harness._shutdown_background_tasks(was_cancelled)

    assert entry.termination_intent == expected_intent
    assert entry.state == TaskState.KILLED
    assert task.cancel_count == 1
    assert entry.result["reason"] == expected_intent
    assert entry.completion_persistence_state == "committed"
    assert entry.completion_persistence_error is None
    assert harness.append_invocations == 1
    assert len(harness.durable_events) == 1
    assert harness.durable_events[0]["final_state"] == "killed"
    assert harness.durable_events[0]["result"] == entry.result
    assert [
      transition
      for transition in harness.listener.transitions
      if transition[2] in _TERMINAL_STATES
    ] == [(entry.task_id, TaskState.RUNNING, TaskState.KILLED)]

  _run(_case())


@pytest.mark.parametrize(
  ("was_cancelled", "expected_intent"),
  [(True, "cancelled"), (False, "killed")],
)
def test_shutdown_drain_error_still_durably_reconciles_once(
  monkeypatch: pytest.MonkeyPatch,
  was_cancelled: bool,
  expected_intent: str,
) -> None:
  import agent_gateway.runner as gateway_runner

  class _Cancelable:
    def __init__(self) -> None:
      self.cancel_count = 0

    def cancel(self) -> None:
      self.cancel_count += 1

  async def _case() -> None:
    harness = _BackgroundLifecycleHarness()
    entry = harness._task_registry.register("background_agent")
    task = _Cancelable()
    entry.asyncio_task = task  # type: ignore[assignment]
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)

    async def _timeout(*_args: object, **_kwargs: object) -> None:
      raise asyncio.TimeoutError

    monkeypatch.setattr(
      gateway_runner,
      (
        "_drain_cancelled_background_tasks"
        if was_cancelled
        else "_drain_still_pending_background_tasks"
      ),
      _timeout,
    )

    with pytest.raises(asyncio.TimeoutError):
      await harness._shutdown_background_tasks(was_cancelled)
    await harness._finalize_background_agent(
      entry,
      final_state=TaskState.FAILED,
      result=None,
      error={"code": "late_finalizer"},
    )

    assert entry.termination_intent == expected_intent
    assert entry.state == TaskState.KILLED
    assert entry.error is None
    assert entry.result["reason"] == expected_intent
    assert task.cancel_count == 1
    assert entry.completion_persistence_state == "committed"
    assert harness.append_invocations == 1
    assert len(harness.durable_events) == 1
    assert harness.durable_events[0]["final_state"] == "killed"
    assert [
      transition
      for transition in harness.listener.transitions
      if transition[2] in _TERMINAL_STATES
    ] == [(entry.task_id, TaskState.RUNNING, TaskState.KILLED)]

  _run(_case())


def test_shutdown_reconciles_done_but_live_task_without_duplicate_append() -> None:
  async def _case() -> None:
    harness = _BackgroundLifecycleHarness()
    entry = harness._task_registry.register("background_agent")
    result = {"response": "durably complete"}

    async def _done() -> None:
      return None

    asyncio_task = asyncio.create_task(_done())
    await asyncio_task
    entry.asyncio_task = asyncio_task
    entry.pending_final_state = TaskState.COMPLETED
    entry.pending_result = result
    entry.result = result
    entry.completion_persistence_state = "committed"
    harness.durable_events.append({
      "type": "task_completed",
      "task_id": entry.task_id,
      "final_state": "completed",
      "result": result,
    })
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)

    await harness._shutdown_background_tasks(was_cancelled=False)

    assert entry.state == TaskState.COMPLETED
    assert entry.result == result
    assert entry.completion_persistence_state == "committed"
    assert entry.completion_persistence_error is None
    assert harness.append_invocations == 0
    assert len(harness.durable_events) == 1
    assert [
      transition
      for transition in harness.listener.transitions
      if transition[2] in _TERMINAL_STATES
    ] == [(entry.task_id, TaskState.RUNNING, TaskState.COMPLETED)]

  _run(_case())


def test_shutdown_reconciles_running_entry_without_asyncio_handle() -> None:
  async def _case() -> None:
    harness = _BackgroundLifecycleHarness()
    entry = harness._task_registry.register("background_agent")
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)

    await harness._shutdown_background_tasks(was_cancelled=True)

    assert entry.state == TaskState.KILLED
    assert entry.termination_intent == "cancelled"
    assert entry.result["reason"] == "cancelled"
    assert entry.completion_persistence_state == "committed"
    assert harness.append_invocations == 1
    assert len(harness.durable_events) == 1
    assert harness.durable_events[0]["final_state"] == "killed"
    assert [
      transition
      for transition in harness.listener.transitions
      if transition[2] in _TERMINAL_STATES
    ] == [(entry.task_id, TaskState.RUNNING, TaskState.KILLED)]

  _run(_case())


def test_late_cancellation_resistant_handler_cannot_replace_shutdown_result() -> None:
  async def _case() -> None:
    harness = _BackgroundLifecycleHarness()
    harness._background_grace_wait_timeout_seconds = 0.0
    harness._background_kill_drain_timeout_seconds = 0.01
    entry = harness._task_registry.register("background_agent")
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release_handler = asyncio.Event()
    late_result = {"response": "late accepted result"}

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[dict[str, object], None]:
      started.set()
      try:
        await release_handler.wait()
      except asyncio.CancelledError:
        cancelled.set()
        await release_handler.wait()
      return late_result, None

    task = asyncio.create_task(
      harness._run_background_agent(entry, _handler, {}, 0)
    )
    entry.asyncio_task = task
    harness._task_registry.transition(entry.task_id, TaskState.RUNNING)
    await started.wait()

    await harness._shutdown_background_tasks(was_cancelled=False)
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)

    assert entry.state == TaskState.KILLED
    shutdown_result = dict(entry.result or {})
    shutdown_pending_state = entry.pending_final_state
    assert shutdown_result["reason"] == "killed"
    assert shutdown_pending_state == TaskState.KILLED
    assert harness.append_invocations == 1

    release_handler.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert entry.state == TaskState.KILLED
    assert entry.result == shutdown_result
    assert entry.pending_final_state == shutdown_pending_state
    assert entry.error is None
    assert harness.append_invocations == 1
    assert [
      transition
      for transition in harness.listener.transitions
      if transition[2] in _TERMINAL_STATES
    ] == [(entry.task_id, TaskState.RUNNING, TaskState.KILLED)]

  _run(_case())


def test_kill_returns_false_for_terminal_task() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent")
  registry.transition(entry.task_id, TaskState.COMPLETED)

  assert registry.kill(entry.task_id) is False


def test_finalize_interrupted_is_explicit_and_idempotent() -> None:
  registry = TaskRegistry()
  listener = _RecordingListener()
  registry.add_listener(listener)
  entry = registry.register("background_agent")
  entry.state = TaskState.INTERRUPTED
  result = {
    "kind": "unstructured",
    "reason": "resume_abandoned",
  }

  finalized = registry.finalize_interrupted(
    entry.task_id,
    TaskState.FAILED,
    result=result,
  )
  repeated = registry.finalize_interrupted(
    entry.task_id,
    TaskState.FAILED,
    result={"unexpected": True},
  )

  assert finalized is entry
  assert repeated is entry
  assert entry.state == TaskState.FAILED
  assert entry.result == result
  assert listener.transitions == [
    (entry.task_id, TaskState.INTERRUPTED, TaskState.FAILED)
  ]


def test_adopt_interrupted_accepts_only_reconstructed_interrupted_tasks() -> None:
  registry = TaskRegistry()
  entry = TaskEntry(
    task_id="bg_17",
    task_type="background_agent",
    state=TaskState.INTERRUPTED,
    reconstructed_from_log=True,
  )

  adopted = registry.adopt_interrupted(entry)
  repeated = registry.adopt_interrupted(entry)

  assert adopted is entry
  assert repeated is entry
  assert registry.get(entry.task_id) is entry
  assert registry.register("background_agent").task_id == "bg_18"

  with pytest.raises(
    ValueError,
    match="only reconstructed interrupted tasks",
  ):
    registry.adopt_interrupted(
      TaskEntry(
        task_id="bg_live",
        task_type="background_agent",
        state=TaskState.INTERRUPTED,
      )
    )


def test_registry_counts_running_and_admission_slots_separately() -> None:
  registry = TaskRegistry()
  pending = registry.register("background_agent")
  running = registry.register("background_agent")
  completed = registry.register("background_agent")
  failed = registry.register("background_agent")

  registry.transition(running.task_id, TaskState.RUNNING)
  registry.transition(completed.task_id, TaskState.COMPLETED)
  registry.transition(failed.task_id, TaskState.FAILED, error={"code": "boom"})

  assert pending.state == TaskState.PENDING
  assert registry.inflight_count == 1
  assert registry.admission_count == 2


def _reject_over_capacity(
  admission_count: int,
  ceiling: int,
) -> dict[str, str] | None:
  if admission_count < ceiling:
    return None
  return {"code": "max_background_tasks"}


def _reject_retrieval_backpressure(
  pending_omitted_results: int,
  retention_limit: int,
) -> dict[str, object]:
  return {
    "code": "background_notification_retrieval_required",
    "pending_omitted_results": pending_omitted_results,
    "retention_limit": retention_limit,
  }


def _admit(registry: TaskRegistry, **kwargs: object):
  return registry.admit(
    "background_agent",
    reject_over_capacity=_reject_over_capacity,
    reject_retrieval_backpressure=_reject_retrieval_backpressure,
    **kwargs,  # type: ignore[arg-type]
  )


def test_running_transition_over_ceiling_logs_invariant_violation_and_proceeds(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """T3-I04 / D-A8-5: the RUNNING gate is a detector, never a refusal.

  It used to ``raise RuntimeError("Task inflight limit reached")`` after the
  durable ``task_registered`` append, stranding the entry PENDING with no
  terminal (CUR-E2E-03). Capacity is owned by ``admit`` now, so reaching this
  branch is a code defect: record it and proceed.
  """

  registry = TaskRegistry(max_inflight=1)
  first = registry.register("background_agent")
  second = registry.register("background_agent")

  registry.transition(first.task_id, TaskState.RUNNING)
  with caplog.at_level("ERROR", logger="agent_gateway.task_registry"):
    entry = registry.transition(second.task_id, TaskState.RUNNING)

  assert entry.state == TaskState.RUNNING
  assert registry.inflight_count == 2
  violations = [
    record
    for record in caplog.records
    if "task registry invariant violation" in record.getMessage()
  ]
  assert len(violations) == 1
  message = violations[0].getMessage()
  assert second.task_id in message
  assert "max_inflight=1" in message

  registry.transition(first.task_id, TaskState.COMPLETED)
  assert registry.inflight_count == 1


def test_admit_refuses_capacity_instead_of_the_running_transition() -> None:
  """The ceiling now lives at the single admission point (D-A8-1)."""

  registry = TaskRegistry(max_inflight=1)
  first, rejection = _admit(registry)
  assert rejection is None
  assert first is not None

  refused, rejection = _admit(registry)
  assert refused is None
  assert rejection == {"code": "max_background_tasks"}
  assert registry.admission_count == 1

  registry.transition(first.task_id, TaskState.RUNNING)
  registry.transition(first.task_id, TaskState.COMPLETED)

  admitted, rejection = _admit(registry)
  assert rejection is None
  assert admitted is not None
  assert registry.admission_count == 1


def test_admit_discounts_an_already_reserved_pending_successor() -> None:
  registry = TaskRegistry(max_inflight=1)
  reserved = registry.claim_resume_successor(
    "background_agent",
    task_id="bg_root_r1",
    original_task_id="bg_root",
  )[0]
  assert registry.admission_count == 1

  entry, rejection = _admit(
    registry,
    task_id="bg_root_r1",
    original_task_id="bg_root",
  )
  assert rejection is None
  assert entry is reserved
  assert registry.admission_count == 1


def test_admit_refuses_when_notification_retrieval_retention_is_full() -> None:
  registry = TaskRegistry(max_inflight=1, max_retained=1)
  omitted = registry.register("background_agent")
  registry.transition(omitted.task_id, TaskState.COMPLETED)
  omitted.notification_delivery_state = "payload_omitted"
  second = registry.register("background_agent")
  registry.transition(second.task_id, TaskState.COMPLETED)
  second.notification_delivery_state = "payload_omitted"

  entry, rejection = _admit(registry)
  assert entry is None
  assert rejection is not None
  assert rejection["code"] == "background_notification_retrieval_required"
  assert rejection["retention_limit"] == (
    registry.notification_retrieval_retention_limit
  )


def test_admit_returns_a_conflict_rejection_for_a_foreign_resume_lineage() -> None:
  registry = TaskRegistry(max_inflight=4)
  registry.claim_resume_successor(
    "background_agent",
    task_id="bg_root_r1",
    original_task_id="bg_root",
  )

  entry, rejection = _admit(
    registry,
    task_id="bg_root_r1",
    original_task_id="bg_other",
  )
  assert entry is None
  assert rejection is not None
  assert rejection["code"] == "resume_successor_conflict"


def test_admit_is_atomic_under_concurrent_callers() -> None:
  """Kills seam-map Hole 1: the verdict and the reservation are one step.

  Every caller yields to the event loop right before admitting, which is
  exactly what the old ``await self._lookup_task_in_log(...)`` between the
  capacity check and ``register`` did.
  """

  registry = TaskRegistry(max_inflight=3)
  admitted: list[str] = []
  refused: list[dict[str, str]] = []

  async def _case() -> None:
    async def _one() -> None:
      await asyncio.sleep(0)
      entry, rejection = _admit(registry)
      if rejection is not None:
        refused.append(rejection)  # type: ignore[arg-type]
        return
      assert entry is not None
      admitted.append(entry.task_id)

    await asyncio.gather(*[_one() for _ in range(10)])

  asyncio.run(_case())

  assert len(admitted) == 3
  assert len(refused) == 7
  assert {rejection["code"] for rejection in refused} == {"max_background_tasks"}
  assert registry.admission_count == 3


def test_max_retained_auto_evicts_oldest_completed_entries() -> None:
  registry = TaskRegistry(max_retained=2)
  first = registry.register("background_agent")
  first.started_at = 1.0
  registry.transition(first.task_id, TaskState.COMPLETED)

  second = registry.register("background_agent")
  second.started_at = 2.0
  registry.transition(second.task_id, TaskState.COMPLETED)

  third = registry.register("background_agent")
  third.started_at = 3.0

  assert registry.get(first.task_id) is None
  assert registry.list_tasks() == [second, third]


def test_evict_completed_removes_old_terminal_entries() -> None:
  registry = TaskRegistry()
  old_entry = registry.register("background_agent")
  recent_entry = registry.register("background_agent")
  running_entry = registry.register("background_agent")

  registry.transition(old_entry.task_id, TaskState.COMPLETED)
  registry.transition(recent_entry.task_id, TaskState.FAILED, error={"code": "failed"})
  registry.transition(running_entry.task_id, TaskState.RUNNING)

  old_entry.completed_at = time.time() - 600
  recent_entry.completed_at = time.time() - 60

  assert registry.evict_completed(max_age_seconds=300) == 1
  assert registry.get(old_entry.task_id) is None
  assert registry.get(recent_entry.task_id) is recent_entry
  assert registry.get(running_entry.task_id) is running_entry


def test_listener_fires_synchronously_on_transition() -> None:
  registry = TaskRegistry()
  listener = _RecordingListener()
  registry.add_listener(listener)
  entry = registry.register("background_agent")

  registry.transition(entry.task_id, TaskState.RUNNING)

  assert listener.transitions == [(entry.task_id, TaskState.PENDING, TaskState.RUNNING)]


def test_task_entry_completed_property() -> None:
  assert TaskEntry(task_id="t0", task_type="background_agent", state=TaskState.PENDING).completed is False
  assert TaskEntry(task_id="t1", task_type="background_agent", state=TaskState.RUNNING).completed is False
  assert TaskEntry(task_id="t2", task_type="background_agent", state=TaskState.COMPLETED).completed is True
  assert TaskEntry(task_id="t3", task_type="background_agent", state=TaskState.FAILED).completed is True
  assert TaskEntry(task_id="t4", task_type="background_agent", state=TaskState.KILLED).completed is True
  assert TaskEntry(task_id="t5", task_type="background_agent", state=TaskState.INTERRUPTED).completed is True


def test_interrupted_is_terminal_for_send_kill_and_completion() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent")
  entry.state = TaskState.INTERRUPTED
  entry.completed_at = time.time()

  assert TaskState.INTERRUPTED in _TERMINAL_STATES
  assert entry.completed is True
  assert registry.kill(entry.task_id) is False


def test_task_entry_message_inbox_exists_at_creation() -> None:
  entry = TaskRegistry().register("background_agent")

  assert isinstance(entry.message_inbox, asyncio.Queue)
  assert entry.message_inbox.empty()


def test_load_from_events_reconstructs_tasks_restores_seq_and_bypasses_listeners() -> None:
  registry = TaskRegistry()
  listener = _RecordingListener()
  registry.add_listener(listener)
  completed_result = _child_report()

  registry.load_from_events(
    [
      {
        "type": "task_completed",
        "task_id": "bg_7",
        "final_state": "completed",
        "completed_at": 125.0,
        "result": completed_result,
        "owner_runner_id": "runner_old",
        "owner_role": "writer",
        "sub_agent_id": "sub7:sess-parent",
        "parent_turn_id": "turn-1",
        "call_index": 7,
        "task_type": "background",
        "capability_bind": _bind_receipt(
          provider="anthropic",
          model="claude-sonnet-4-6",
        ),
      },
      {
        "type": "task_registered",
        "event_schema_version": 2,
        "task_id": "bg_7",
        "agent_name": "writer",
        "started_at": 100.0,
        "metadata": {"custom": "value"},
        "owner_runner_id": "runner_old",
        "owner_role": "writer",
        "sub_agent_id": "sub7:sess-parent",
        "parent_turn_id": "turn-1",
        "call_index": 7,
        "task_type": "background",
        "capability_bind": _bind_receipt(
          provider="anthropic",
          model="claude-sonnet-4-6",
        ),
      },
      {
        "type": "task_registered",
        "event_schema_version": 2,
        "task_id": "bg_9",
        "agent_name": "reviewer",
        "started_at": 200.0,
        "owner_runner_id": "runner_old",
        "owner_role": "writer",
        "sub_agent_id": "sub9:sess-parent",
        "parent_turn_id": "turn-2",
        "call_index": 9,
        "task_type": "plan_run",
        "capability_bind": _bind_receipt(provider="openai", model="gpt-5.2"),
        "metadata": {
          "owner_runner_id": "runner_old",
          "owner_role": "writer",
          "sub_agent_id": "sub9:sess-parent",
          "parent_turn_id": "turn-2",
          "call_index": 9,
          "task_type": "background",
          "capability_bind": _bind_receipt(
            provider="openai",
            model="gpt-5.2",
          ),
        },
        "_durable_seq": 3,
      },
      {
        "type": "parent_message_sent",
        "task_id": "bg_9",
        "owner_runner_id": "runner_old",
        "owner_role": "writer",
        "sub_agent_id": "sub9:sess-parent",
        "parent_turn_id": "turn-2",
        "call_index": 9,
        "task_type": "background",
        "capability_bind": _bind_receipt(
          provider="openai",
          model="gpt-5.2",
        ),
        "message_id": "msg-1",
        "message": "status",
        "sent_at": 210.0,
        "_durable_seq": 4,
      },
    ]
  )

  completed = registry.get("bg_7")
  interrupted = registry.get("bg_9")
  assert completed is not None
  assert interrupted is not None
  assert completed.state == TaskState.COMPLETED
  assert completed.result == completed_result
  assert completed.completed_at == 125.0
  assert completed.metadata["custom"] == "value"
  assert interrupted.state == TaskState.INTERRUPTED
  assert interrupted.completed_at == 200.0
  assert interrupted.task_type == "plan_run"
  assert interrupted.reconstructed_from_log is True
  assert interrupted.capability_bind_receipt == _bind_receipt(
    provider="openai",
    model="gpt-5.2",
  )
  assert interrupted.metadata["parent_messages"][0]["message_id"] == "msg-1"
  assert interrupted.accepted_parent_messages["msg-1"] == ParentMessage(
    message_id="msg-1",
    text="status",
    sent_at=210.0,
    task_id="bg_9",
    sent_seq=4,
  )
  assert interrupted.delivered_messages == {"msg-1"}
  assert registry.register("background_agent").task_id == "bg_10"
  assert listener.transitions == []


def test_load_from_events_accepts_exact_duplicate_task_events() -> None:
  registry = TaskRegistry()
  registration = {
    "type": "task_registered",
    "event_schema_version": 2,
    "task_id": "bg_7",
    "task_type": "background",
    "agent_name": "writer",
    "started_at": 100.0,
    "capability_bind": _bind_receipt(
      provider="anthropic",
      model="claude-sonnet-4-6",
    ),
  }
  completion = {
    "type": "task_completed",
    "task_id": "bg_7",
    "final_state": "completed",
    "completed_at": 125.0,
    "result": _child_report(),
    "error": None,
  }

  registry.load_from_events([
    {**registration, "_durable_seq": 1},
    {**registration, "_durable_seq": 2},
    {**completion, "_durable_seq": 3},
    {**completion, "_durable_seq": 4},
  ])

  entry = registry.get("bg_7")
  assert entry is not None
  assert entry.state is TaskState.COMPLETED


@pytest.mark.parametrize(
  "event_schema_version",
  [1, None, "2", 2.0, True, 3],
  ids=["v1", "missing", "string", "float", "bool", "future"],
)
def test_load_from_events_loudly_skips_unsupported_schema_version(
  event_schema_version: object,
  caplog: pytest.LogCaptureFixture,
) -> None:
  """Pre-cutover registrations drain with a loud skip, never rebuild.

  A v1 record rebuilt as a bind-less INTERRUPTED entry would fail only at
  resume with a generic invalid_task_metadata (review 2026-08-14, B7).
  """

  registry = TaskRegistry()
  event = {
    "type": "task_registered",
    "task_id": "bg_7",
    "task_type": "background",
    "agent_name": "writer",
    "started_at": 100.0,
    "capability_bind": _bind_receipt(
      provider="anthropic",
      model="claude-sonnet-4-6",
    ),
  }
  if event_schema_version is not None:
    event["event_schema_version"] = event_schema_version

  with caplog.at_level("WARNING", logger="agent_gateway.task_registry"):
    registry.load_from_events([event])

  assert registry.get("bg_7") is None
  assert registry.list_tasks() == []
  skip_records = [
    record
    for record in caplog.records
    if "Skipping durable task bg_7" in record.getMessage()
  ]
  assert len(skip_records) == 1
  assert "event_schema_version" in skip_records[0].getMessage()
  # The skipped identity still advances the sequence so new registrations
  # never collide with the drained durable record.
  assert registry.register("background_agent").task_id == "bg_8"


def test_load_from_events_loudly_skips_v2_registration_without_bind(
  caplog: pytest.LogCaptureFixture,
) -> None:
  registry = TaskRegistry()
  event = {
    "type": "task_registered",
    "event_schema_version": 2,
    "task_id": "bg_7",
    "task_type": "background",
    "agent_name": "writer",
    "started_at": 100.0,
  }

  with caplog.at_level("WARNING", logger="agent_gateway.task_registry"):
    registry.load_from_events([event])

  assert registry.get("bg_7") is None
  skip_records = [
    record
    for record in caplog.records
    if "Skipping durable task bg_7" in record.getMessage()
  ]
  assert len(skip_records) == 1
  assert "capability_bind" in skip_records[0].getMessage()


def test_load_from_events_skip_warns_once_per_task_and_reason(
  caplog: pytest.LogCaptureFixture,
) -> None:
  registry = TaskRegistry()
  event = {
    "type": "task_registered",
    "event_schema_version": 1,
    "task_id": "bg_7",
    "task_type": "background",
    "agent_name": "writer",
    "started_at": 100.0,
  }

  with caplog.at_level("WARNING", logger="agent_gateway.task_registry"):
    registry.load_from_events([dict(event)])
    registry.load_from_events([dict(event)])

  skip_records = [
    record
    for record in caplog.records
    if "Skipping durable task bg_7" in record.getMessage()
  ]
  assert len(skip_records) == 1


@pytest.mark.parametrize(
  ("event_type", "changed_field", "changed_value"),
  [
    ("task_registered", "agent_name", "reviewer"),
    ("task_completed", "final_state", "failed"),
  ],
)
def test_load_from_events_rejects_conflicting_duplicate_task_events(
  event_type: str,
  changed_field: str,
  changed_value: object,
) -> None:
  registry = TaskRegistry()
  registration = {
    "type": "task_registered",
    "event_schema_version": 2,
    "task_id": "bg_7",
    "task_type": "background",
    "agent_name": "writer",
    "started_at": 100.0,
    "capability_bind": _bind_receipt(
      provider="anthropic",
      model="claude-sonnet-4-6",
    ),
  }
  completion = {
    "type": "task_completed",
    "task_id": "bg_7",
    "final_state": "completed",
    "completed_at": 125.0,
    "result": _child_report(),
    "error": None,
  }
  source = registration if event_type == "task_registered" else completion
  events = [registration, completion, {**source, changed_field: changed_value}]

  with pytest.raises(
    TaskDurableEventConflictError,
    match=f"conflicting durable {event_type}",
  ):
    registry.load_from_events(events)


@pytest.mark.parametrize(
  ("final_state", "error"),
  [
    ("failed", None),
    ("completed", {"code": "provider_error", "message": "failed"}),
    ("not-a-state", None),
    (None, None),
  ],
)
def test_load_from_events_quarantines_task_result_settlement_mismatch(
  final_state: object,
  error: dict[str, str] | None,
) -> None:
  registry = TaskRegistry()
  registry.load_from_events([
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "task_id": "bg_7",
      "task_type": "background",
      "agent_name": "writer",
      "started_at": 100.0,
      "capability_bind": _bind_receipt(
        provider="anthropic",
        model="claude-sonnet-4-6",
      ),
    },
    {
      "type": "task_completed",
      "task_id": "bg_7",
      "final_state": final_state,
      "completed_at": 125.0,
      "result": _canonical_skipped_result(),
      "error": error,
    },
  ])

  entry = registry.get("bg_7")
  assert entry is not None
  assert entry.state is TaskState.FAILED
  assert entry.task_result is None
  assert entry.error is not None
  assert entry.error["code"] == "task_result_settlement_mismatch"


def test_load_from_events_rejects_raw_completed_child_payload() -> None:
  registry = TaskRegistry()
  raw_result = {
    "response": "partial",
    "error_detail": "provider failed",
  }

  registry.load_from_events(
    [
      {
        "type": "task_registered",
        "event_schema_version": 2,
        "task_id": "bg_3",
        "agent_name": "reviewer",
        "started_at": 100.0,
        "capability_bind": _bind_receipt(
          provider="anthropic",
          model="claude-sonnet-4-6",
        ),
      },
      {
        "type": "task_completed",
        "task_id": "bg_3",
        "final_state": "completed",
        "completed_at": 125.0,
        "result": raw_result,
        "error": None,
      },
    ]
  )

  reconstructed = registry.get("bg_3")
  assert reconstructed is not None
  assert reconstructed.state == TaskState.FAILED
  assert reconstructed.result == raw_result
  assert reconstructed.error is not None
  assert reconstructed.error["code"] == "invalid_task_result"
  assert reconstructed.completed_at == 125.0
  assert reconstructed.completion_persistence_state == "committed"


@pytest.mark.parametrize(
  "stored_error",
  ["provider failed", 42, ["provider failed"], True],
  ids=["string", "integer", "list", "boolean"],
)
@pytest.mark.parametrize("final_state", ["completed", "failed", "killed"])
def test_load_from_events_rejects_malformed_child_error(
  stored_error: object,
  final_state: str,
) -> None:
  registry = TaskRegistry()
  completed_result = _child_report()

  registry.load_from_events(
    [
      {
        "type": "task_registered",
        "event_schema_version": 2,
        "task_id": "bg_5",
        "agent_name": "reviewer",
        "started_at": 100.0,
        "capability_bind": _bind_receipt(
          provider="anthropic",
          model="claude-sonnet-4-6",
        ),
      },
      {
        "type": "task_completed",
        "task_id": "bg_5",
        "final_state": final_state,
        "completed_at": 125.0,
        "result": completed_result,
        "error": stored_error,
      },
    ]
  )

  reconstructed = registry.get("bg_5")
  assert reconstructed is not None
  assert reconstructed.state == TaskState.FAILED
  assert reconstructed.result == completed_result
  assert reconstructed.error is not None
  assert reconstructed.error["code"] == "invalid_child_error"
  assert type(stored_error).__name__ in reconstructed.error["message"]
  assert reconstructed.completed_at == 125.0
  assert reconstructed.completion_persistence_state == "committed"
  payload = background_task_payload(
    reconstructed,
    elapsed_seconds=25,
    now=126.0,
  )
  assert payload["status"] == "error"
  assert payload["error"]["code"] == "invalid_child_error"


def test_load_from_events_quarantines_legacy_cancelled_report_completion() -> None:
  registry = TaskRegistry()
  accepted_result = {
    "kind": "report",
    "version": "1",
    "report": {
      "summary": "Accepted before cancellation",
      "findings": [],
      "artifacts": [],
      "caveats": [],
    },
    "usage": {"input_tokens": 3},
    "tools_used": ["lookup"],
    "fms_results": None,
    "artifact_events": None,
    "warning": "observed_terminal_signals=cancelled",
  }

  registry.load_from_events(
    [
      {
        "type": "task_completed",
        "task_id": "bg_4",
        "final_state": "completed",
        "completed_at": 125.0,
        "result": accepted_result,
        "error": None,
      },
      {
        "type": "task_registered",
        "event_schema_version": 2,
        "task_id": "bg_4",
        "agent_name": "reviewer",
        "started_at": 100.0,
        "capability_bind": _bind_receipt(
          provider="anthropic",
          model="claude-sonnet-4-6",
        ),
      },
    ]
  )

  reconstructed = registry.get("bg_4")
  assert reconstructed is not None
  assert reconstructed.state == TaskState.FAILED
  assert reconstructed.termination_intent is None
  assert reconstructed.result == accepted_result
  assert reconstructed.error is not None
  assert reconstructed.error["code"] == "invalid_task_result"
  assert reconstructed.completed_at == 125.0
  assert reconstructed.completion_persistence_state == "committed"


def test_load_from_events_ignores_resume_suffix_for_seq_base() -> None:
  registry = TaskRegistry()

  registry.load_from_events(
    [
      {
        "type": "task_registered",
        "event_schema_version": 2,
        "task_id": "bg_3_r1",
        "agent_name": "reviewer",
        "started_at": 100.0,
        "original_task_id": "bg_3",
        "capability_bind": _bind_receipt(
          provider="anthropic",
          model="claude-sonnet-4-6",
        ),
      },
    ]
  )

  assert registry.register("background_agent").task_id == "bg_0"


def test_register_accepts_original_task_id_and_explicit_resume_id() -> None:
  registry = TaskRegistry()

  entry = registry.register(
    "background_agent",
    agent_name="reviewer",
    task_id="bg_3_r1",
    original_task_id="bg_3",
  )

  assert entry.task_id == "bg_3_r1"
  assert entry.original_task_id == "bg_3"


def test_claim_resume_successor_is_idempotent_for_the_same_lineage() -> None:
  registry = TaskRegistry()

  first, first_created = registry.claim_resume_successor(
    "background_agent",
    agent_name="reviewer",
    task_id="bg_3_r1",
    original_task_id="bg_3",
  )
  repeated, repeated_created = registry.claim_resume_successor(
    "background_agent",
    agent_name="ignored-on-replay",
    task_id="bg_3_r1",
    original_task_id="bg_3",
  )

  assert first_created is True
  assert repeated_created is False
  assert repeated is first
  assert first.agent_name == "reviewer"
  assert registry.list_tasks() == [first]


def test_claim_resume_successor_rejects_cross_lineage_collision() -> None:
  registry = TaskRegistry()
  registry.register(
    "background_agent",
    task_id="bg_3_r1",
    original_task_id="bg_other",
  )

  with pytest.raises(
    ResumeSuccessorConflictError,
    match="requested original_task_id='bg_3'",
  ) as exc_info:
    registry.claim_resume_successor(
      "background_agent",
      task_id="bg_3_r1",
      original_task_id="bg_3",
    )

  assert exc_info.value.task_id == "bg_3_r1"
  assert exc_info.value.existing_original_task_id == "bg_other"
  assert len(registry.list_tasks()) == 1


def test_make_progress_tracker_updates_tool_usage_fields() -> None:
  entry = TaskEntry(task_id="bg_0", task_type="background_agent")
  track = make_progress_tracker(entry)

  before = time.time()
  track({"type": "tool_call_start", "tool_name": "search_query"}, "sub0:sess")

  assert entry.progress.tool_use_count == 1
  assert entry.progress.last_tool_name == "search_query"
  assert entry.progress.recent_tools == ["search_query"]
  assert entry.progress.last_activity_at >= before


def test_make_progress_tracker_updates_turn_counts_and_tokens() -> None:
  entry = TaskEntry(task_id="bg_0", task_type="background_agent")
  track = make_progress_tracker(entry)

  track({"type": "turn_complete", "usage": {"input_tokens": 11, "output_tokens": 7}}, "sub0:sess")
  track({"type": "turn_complete", "usage": {"input_tokens": 3}}, "sub0:sess")

  assert entry.progress.turn_count == 2
  assert entry.progress.input_tokens == 14
  assert entry.progress.output_tokens == 7
  assert entry.progress.last_activity_at > 0


def test_make_progress_tracker_caps_recent_tools_at_ten() -> None:
  entry = TaskEntry(task_id="bg_0", task_type="background_agent")
  track = make_progress_tracker(entry)

  for index in range(12):
    track({"type": "tool_call_start", "tool_name": f"tool_{index}"}, "sub0:sess")

  assert entry.progress.tool_use_count == 12
  assert entry.progress.last_tool_name == "tool_11"
  assert len(entry.progress.recent_tools) == 10
  assert entry.progress.recent_tools == [f"tool_{index}" for index in range(2, 12)]
