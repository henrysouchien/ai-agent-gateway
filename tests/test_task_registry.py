# ruff: noqa: E402

from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import gc
import sys
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_workflow_contracts import (
  AdmittedTask,
  AgentOperationRef,
  AttemptRef,
  ContentReadGrant,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  UsageObservation,
  canonical_json_bytes,
  sha256_digest,
  terminal_task_result,
)
from gateway_test_support.admitted_authority_test_support import (
  provenance_of,
  sealed_admitted_task,
)
from gateway_test_support.capability_execution_test_support import stub_runner_capability_execution
from gateway_test_support.resumer_custody_test_support import (
  canonical_read_custody_task,
  custody_registration,
)

from agent_gateway import AgentRunner, ToolDispatcher
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.event_log import EventLog
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.runner_sub_agents import _bind_child_registry, _close_sub_runner
from agent_gateway.runner_background_lifecycle import RunnerBackgroundLifecycleMixin
from agent_gateway.runner_background_tasks import (
  background_task_payload,
  build_agent_completion_envelope,
  ordinary_parent_result_policy,
  task_completed_event_payload,
  task_correlation_payload,
)
from agent_gateway.runner_session_events import build_agent_completion_event
from agent_gateway.sub_agent import seal_admitted_task_payload
from agent_gateway.sub_agent_result_contract import (
  build_task_result,
  canonical_projection,
  report_contract_ref,
)
from agent_gateway.runner_state import BackgroundTask
from agent_gateway.task_registry import (
  ParentMessage,
  RegistryOwner,
  ResumeSuccessorConflictError,
  TaskDurableEventConflictError,
  TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES,
  TaskEntry,
  TaskResultSettlementError,
  TaskRegistry,
  TaskState,
  _TERMINAL_STATES,
  job_output_handles,
  job_read_subtree,
  make_progress_tracker,
  validate_task_result_settlement,
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


def _canonical_admitted_task(task_id: str = "bg_7") -> AdmittedTask:
  task = sealed_admitted_task(
    logical_task=OrdinaryDelegationTaskRef(
      delegation_id=f"delegation:{task_id}",
      operation=AgentOperationRef(
        namespace="agent-operation",
        name="research",
        version="1",
        digest=sha256_digest({"operation": "research"}),
      ),
    ),
    attempt=AttemptRef(
      attempt_number=1,
      attempt_id=f"attempt:{task_id}",
      physical_task_id=task_id,
    ),
    result_requirement=ResultRequirement(outcome=OutcomeRequirement()),
  )
  assert task.model_bind is not None
  payload = task.model_dump(mode="json")
  payload["schema_version"] = "1.2"
  payload["operation"]["execution_class"] = task.model_bind.capability_id
  payload["operation"]["resumable"] = True
  payload["execution_snapshot"]["resume_mechanics"]["resumable"] = True
  payload["execution_snapshot"]["resume_mechanics"]["max_chain_depth"] = 3
  payload["execution_snapshot"]["provider_tool_definitions"] = [
    {"definition": {
      "name": "web_search",
      "description": "Search public sources.",
      "input_schema": {"type": "object"},
    }},
  ]
  payload["tool_routes"] = [
    {"tool_id": "web_search", "origin": "local", "server_id": None},
  ]
  payload["admitted_task_digest"] = sha256_digest({
    key: value
    for key, value in payload.items()
    if key != "admitted_task_digest"
  })
  return AdmittedTask.model_validate(payload)


async def _append_custody_completion(
  durable_log: AgentSessionLog,
  admitted: AdmittedTask,
) -> str:
  task_id = admitted.attempt.physical_task_id
  projection = canonical_projection(
    contract=report_contract_ref("explore-findings-v1"),
    value={"task_id": task_id},
  )
  result = build_task_result(
    logical_task=admitted.logical_task,
    attempt=admitted.attempt,
    requirement=admitted.result_requirement,
    provenance=provenance_of(admitted),
    execution=ExecutionSettlement(status="succeeded"),
    outcome=None,
    terminal_narrative=None,
    projection=None,
    runtime_projection=projection,
  )

  def _narrative(_result):
    raise AssertionError("projection completion does not read a narrative")

  envelope = build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=_narrative,
    read_grant_factory=lambda source: ContentReadGrant(
      grant_id=f"grant:{task_id}",
      content_id=source.content_id,
      scope="direct_parent",
      principal_id="runner-alias",
    ),
  )
  entry = TaskEntry(
    task_id=task_id,
    task_type="background_agent",
    state=TaskState.COMPLETED,
    started_at=1.0,
    completed_at=2.0,
    result=result.model_dump(mode="json"),
    admitted_task=admitted,
    task_result=result,
    completion_envelope=envelope,
  )
  await durable_log.append(task_completed_event_payload(
    entry,
    TaskState.COMPLETED,
    correlation_payload=task_correlation_payload(entry, runner_id="runner-alias", role="owner"),
    completed_at=2.0,
  ))
  await durable_log.append(build_agent_completion_event(
    task_id=task_id,
    envelope=envelope,
    ts=2.0,
  ))
  return projection.content.content_id


def _run(coro):
  return asyncio.run(coro)


def _linked_registry_runner(
  parent: TaskRegistry,
  task_id: str,
  *,
  task_entry: TaskEntry | None = None,
  close_error: type[BaseException] | None = None,
) -> SimpleNamespace:
  async def force_close(timeout: float = 2.0) -> None:
    if close_error is not None:
      raise close_error("close failed")

  child = SimpleNamespace(
    _task_registry=TaskRegistry(),
    force_close=force_close,
  )
  child_runner_ref: list[object] = []
  _bind_child_registry(
    child,
    parent_registry=parent,
    attempt=AttemptRef(
      attempt_number=1,
      attempt_id=f"attempt:{task_id}",
      physical_task_id=task_id,
    ),
    sub_session=None,
    task_entry=task_entry,
    spill_sink=None,
    bind_job_runner=child_runner_ref.append,
  )
  return child


def _notification_registry_runner(
  parent: TaskRegistry,
  task_id: str,
  *,
  task_entry: TaskEntry | None = None,
) -> AgentRunner:
  event_log = EventLog()
  _dispatcher_execution_243 = stub_runner_capability_execution(
        provider=SimpleNamespace(name="stub"),
        model="stub-model",
        effort="none",
        auth_config={"api_key": "test-only"},
      )
  child = AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      capability_bind=_dispatcher_execution_243.bind,
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={},
      event_log=event_log,
      session_id=task_id,
    ),
    session_id=task_id,
    capability_execution=_dispatcher_execution_243,
    user_id="registry-test",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  child_runner_ref: list[object] = []
  _bind_child_registry(
    child,
    parent_registry=parent,
    attempt=AttemptRef(
      attempt_number=1,
      attempt_id=f"attempt:{task_id}",
      physical_task_id=task_id,
    ),
    sub_session=None,
    task_entry=task_entry,
    spill_sink=None,
    bind_job_runner=child_runner_ref.append,
  )
  return child


def _close_registry_runner(
  child: SimpleNamespace | AgentRunner,
  *,
  task_entry: TaskEntry | None = None,
  cancelled_error: asyncio.CancelledError | None = None,
) -> None:
  _run(_close_sub_runner(
    child,
    EventLog(),
    stalled=False,
    cancelled_error=cancelled_error,
    cancellation_signal="cancelled" if cancelled_error is not None else None,
    runtime_exception_detail=None,
    task_entry=task_entry,
    cleanup_warnings=[],
  ))


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
  _background_completion_persist_timeout_seconds: float
  _background_grace_wait_timeout_seconds: float
  _background_kill_drain_timeout_seconds: float
  _pending_background_initializations: set[asyncio.Task[None]]

  def __init__(self) -> None:
    self._task_registry = TaskRegistry()
    self.listener = _RecordingListener()
    self._task_registry.add_listener(self.listener)
    self._runner_id = "runner-test"
    self._role = "writer"
    self._sid = "session-test"
    self.durable_events: list[dict[str, object]] = []
    self.append_invocations = 0
    self._pending_background_initializations = set()

  def _ensure_sub_agent_semaphore(self) -> None:
    return None

  async def _append_durable_event(self, event: dict[str, object]) -> None:
    self.append_invocations += 1
    self.durable_events.append(event)


async def _kill_registry_owner(
  parent: TaskRegistry,
  owner_entry: TaskEntry,
  child: AgentRunner,
) -> None:
  harness = _BackgroundLifecycleHarness()
  harness._task_registry = parent
  started = asyncio.Event()

  async def _handler(
    _tool_input: dict[str, object],
    **_kwargs: object,
  ) -> tuple[dict[str, object], None]:
    started.set()
    try:
      await asyncio.Event().wait()
    except asyncio.CancelledError as exc:
      await _close_sub_runner(
        child,
        EventLog(),
        stalled=False,
        cancelled_error=exc,
        cancellation_signal="killed",
        runtime_exception_detail=None,
        task_entry=owner_entry,
        cleanup_warnings=[],
      )
      raise
    raise AssertionError("owner handler must end by cancellation")

  task = asyncio.create_task(harness._run_background_agent(owner_entry, _handler, {}, 0))
  owner_entry.asyncio_task = task
  await started.wait()
  assert parent.kill(owner_entry.task_id)
  await task

  assert owner_entry.state == TaskState.KILLED
  assert owner_entry.completion_persistence_state == "committed"
  assert harness.durable_events[0]["task_id"] == owner_entry.task_id
  assert harness.durable_events[0]["final_state"] == "killed"


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

    async def _on_complete(
      completed_entry: BackgroundTask | TaskEntry,
    ) -> None:
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
    failed_error: dict[str, object] = {
      "code": "provider_error",
      "message": "provider failed",
    }

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[None, dict[str, object]]:
      return None, failed_error

    completed: list[BackgroundTask | TaskEntry] = []
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

    completed: list[BackgroundTask | TaskEntry] = []
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
        bg_task: TaskEntry,
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
    completed: list[BackgroundTask | TaskEntry] = []

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
      harness._pending_background_initializations
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
        bg_task: TaskEntry,
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
        bg_task: TaskEntry,
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
    assert entry.result is not None
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
    assert entry.result is not None
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
    assert entry.result is not None
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
    late_result: dict[str, object] = {
      "response": "late accepted result",
    }

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
    reject_retrieval_backpressure=_reject_retrieval_backpressure,
    **kwargs,  # type: ignore[arg-type]
  )


def test_running_transition_past_the_in_flight_count_logs_no_violation(
  caplog: pytest.LogCaptureFixture,
) -> None:
  """The in-flight count is a measurement; the RUNNING transition never gates on it."""

  registry = TaskRegistry(max_inflight=1)
  first = registry.register("background_agent")
  second = registry.register("background_agent")

  registry.transition(first.task_id, TaskState.RUNNING)
  with caplog.at_level("ERROR", logger="agent_gateway.task_registry"):
    entry = registry.transition(second.task_id, TaskState.RUNNING)

  assert entry.state == TaskState.RUNNING
  assert registry.inflight_count == 2
  assert not [
    record
    for record in caplog.records
    if "task registry invariant violation" in record.getMessage()
  ]


def test_admit_never_refuses_for_the_in_flight_count() -> None:
  """A child past the old ceiling is admitted; it waits on the semaphore to run."""

  registry = TaskRegistry(max_inflight=1)
  first, rejection = _admit(registry)
  assert rejection is None
  assert first is not None

  second, rejection = _admit(registry)
  assert rejection is None
  assert second is not None
  assert second.task_id != first.task_id
  assert registry.admission_count == 2


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


def test_admit_discounts_a_reserved_successor_that_already_started() -> None:
  """A reservation holds its slot across the PENDING -> RUNNING transition.

  A concurrent resume registration whose pre-decision durable lookup outlives
  that transition reaches ``admit`` with the successor already RUNNING. It is
  replaying the claim, not asking for a second slot, so the ceiling must hand
  the entry back instead of refusing capacity.
  """

  registry = TaskRegistry(max_inflight=1)
  reserved = registry.claim_resume_successor(
    "background_agent",
    task_id="bg_root_r1",
    original_task_id="bg_root",
  )[0]
  registry.transition(reserved.task_id, TaskState.RUNNING)

  entry, rejection = _admit(
    registry,
    task_id="bg_root_r1",
    original_task_id="bg_root",
  )
  assert rejection is None
  assert entry is not None
  assert entry is reserved
  assert entry.state == TaskState.RUNNING
  assert registry.admission_count == 1


def test_admit_returns_a_running_reservation_for_its_own_task_id() -> None:
  registry = TaskRegistry(max_inflight=1)
  reserved = registry.register("background_agent", task_id="bg_override")
  registry.transition(reserved.task_id, TaskState.RUNNING)

  entry, rejection = _admit(registry, task_id="bg_override")

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
  verdict and ``register`` did. The one admission verdict left is retrieval
  backpressure (retention limit ``max_retained + max_inflight`` = 3 here).
  """

  registry = TaskRegistry(max_inflight=1, max_retained=2)
  admitted: list[str] = []
  refused: list[dict[str, str]] = []

  async def _case() -> None:
    async def _one() -> None:
      await asyncio.sleep(0)
      entry, rejection = _admit(registry)
      if rejection is not None:
        refused.append(rejection)
        return
      assert entry is not None
      admitted.append(entry.task_id)

    await asyncio.gather(*[_one() for _ in range(10)])

  asyncio.run(_case())

  assert len(admitted) == 3
  assert len(refused) == 7
  assert {rejection["code"] for rejection in refused} == {
    "background_notification_retrieval_required"
  }
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


@pytest.mark.parametrize(
  ("close_error", "cancelled_error"),
  [
    (None, None),
    (RuntimeError, None),
    (asyncio.CancelledError, None),
    (None, asyncio.CancelledError),
  ],
)
def test_completed_foreground_registry_is_released(
  close_error: type[BaseException] | None,
  cancelled_error: type[asyncio.CancelledError] | None,
) -> None:
  root = TaskRegistry(owner=RegistryOwner(job_id="root", tenant=lambda: None))
  child = _linked_registry_runner(root, "foreground", close_error=close_error)
  child_registry = weakref.ref(child._task_registry)

  _close_registry_runner(
    child,
    cancelled_error=cancelled_error() if cancelled_error is not None else None,
  )
  del child
  gc.collect()

  assert child_registry() is None
  assert [job.job_id for job in root.roster()] == ["root"]


def test_completed_foreground_registry_keeps_live_background_descendant_on_roster() -> None:
  root = TaskRegistry(owner=RegistryOwner(job_id="root", tenant=lambda: None))
  child = _linked_registry_runner(root, "foreground")
  child_registry = weakref.ref(child._task_registry)
  entry = child._task_registry.register("background_agent", agent_name="writer")
  child._task_registry.transition(entry.task_id, TaskState.RUNNING)

  _close_registry_runner(child)
  del child
  gc.collect()

  descendant = root.find_on_roster("writer")
  assert descendant is not None
  assert descendant.entry is entry
  assert descendant.parent_job_id == "foreground"
  assert child_registry() is descendant.registry
  descendant.registry.transition(entry.task_id, TaskState.COMPLETED)
  del descendant
  gc.collect()

  assert root.find_on_roster(entry.task_id) is None
  assert child_registry() is None
  assert [job.job_id for job in root.roster()] == ["root"]


def test_retention_eviction_releases_killed_owner_with_unretrieved_result() -> None:
  async def _case() -> None:
    root = TaskRegistry(
      max_retained=1,
      owner=RegistryOwner(job_id="root", tenant=lambda: None),
    )
    owner_entry = root.register("background_agent", task_id="background-owner")
    root.transition(owner_entry.task_id, TaskState.RUNNING)
    owner = _notification_registry_runner(
      root,
      owner_entry.task_id,
      task_entry=owner_entry,
    )
    inner = _BackgroundLifecycleHarness()
    inner._task_registry = owner._task_registry
    omitted = inner._task_registry.register("background_agent", task_id="omitted-leaf")
    inner._task_registry.transition(omitted.task_id, TaskState.RUNNING)
    report = _child_report("x" * (TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES + 1))

    async def _handler(
      _tool_input: dict[str, object],
      **_kwargs: object,
    ) -> tuple[dict[str, object], None]:
      return report, None

    await inner._run_background_agent(omitted, _handler, {}, 0)
    # Payload omission represents the omitted-delivery state family; both
    # omission states have the same registry retention semantics.
    assert omitted.notification_delivery_state == "payload_omitted"
    generation = omitted.notification_generation
    completion = inner.durable_events[0]

    await _kill_registry_owner(root, owner_entry, owner)
    assert root.get(owner_entry.task_id) is owner_entry
    assert root.find_on_roster(omitted.task_id) is not None
    root.register("background_agent", task_id="replacement")

    assert root.get(owner_entry.task_id) is None
    assert owner_entry.task_id not in root._child_registries
    assert root.find_on_roster(omitted.task_id) is None
    assert [job.job_id for job in root.roster()] == ["root", "replacement"]
    assert omitted.notification_delivery_state == "payload_omitted"
    assert omitted.notification_generation == generation
    assert completion["result"] == omitted.result == report
    assert completion["final_state"] == "completed"

  _run(_case())


@pytest.mark.parametrize("eviction", ["age", "retention"])
def test_background_owner_eviction_preserves_live_nested_descendant(
  eviction: str,
) -> None:
  root = TaskRegistry(
    max_retained=1,
    owner=RegistryOwner(job_id="root", tenant=lambda: None),
  )
  owner_entry = root.register("background_agent", task_id="background-owner")
  root.transition(owner_entry.task_id, TaskState.RUNNING)
  owner = _linked_registry_runner(root, owner_entry.task_id, task_entry=owner_entry)
  owner_registry = weakref.ref(owner._task_registry)
  foreground = _notification_registry_runner(owner._task_registry, "foreground")
  foreground_registry = weakref.ref(foreground._task_registry)
  leaf = foreground._task_registry.register(
    "background_agent",
    task_id="nested-leaf",
    agent_name="leaf",
  )
  foreground._task_registry.transition(leaf.task_id, TaskState.RUNNING)
  omitted = foreground._task_registry.register(
    "background_agent",
    task_id="omitted-leaf",
  )
  foreground._task_registry.transition(
    omitted.task_id,
    TaskState.COMPLETED,
    result=_child_report("x" * (TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES + 1)),
  )
  assert omitted.notification_delivery_state == "payload_omitted"
  _close_registry_runner(foreground)
  _close_registry_runner(owner, task_entry=owner_entry)
  root.transition(owner_entry.task_id, TaskState.COMPLETED)
  del owner, foreground

  if eviction == "age":
    assert root.evict_completed(max_age_seconds=0) == 1
  else:
    root.register("background_agent")

  assert root.get(owner_entry.task_id) is None
  assert root.find_on_roster(omitted.task_id) is not None
  descendant = root.find_on_roster("leaf")
  assert descendant is not None
  assert descendant.entry is leaf
  assert descendant.parent_job_id == "foreground"
  descendant.registry.transition(leaf.task_id, TaskState.FAILED)
  del descendant
  gc.collect()

  assert root.find_on_roster(leaf.task_id) is None
  assert root.find_on_roster(omitted.task_id) is None
  assert omitted.notification_delivery_state == "payload_omitted"
  assert owner_registry() is None
  assert foreground_registry() is None


def test_evicted_owner_waits_for_real_detached_completion_before_unlinking() -> None:
  async def _case() -> None:
    append_started = asyncio.Event()
    release_append = asyncio.Event()
    reconciled = asyncio.Event()

    class _DelayedAppendHarness(_BackgroundLifecycleHarness):
      async def _append_durable_event(self, event: dict[str, object]) -> None:
        if event["task_id"] == "detached-leaf":
          append_started.set()
          await release_append.wait()
        await super()._append_durable_event(event)

    root = TaskRegistry(
      max_retained=1,
      owner=RegistryOwner(job_id="root", tenant=lambda: None),
    )
    owner_entry = root.register("background_agent", task_id="background-owner")
    root.transition(owner_entry.task_id, TaskState.RUNNING)
    owner = _notification_registry_runner(
      root,
      owner_entry.task_id,
      task_entry=owner_entry,
    )
    inner = _DelayedAppendHarness()
    inner._task_registry = owner._task_registry
    omitted = inner._task_registry.register("background_agent", task_id="omitted-leaf")
    inner._task_registry.transition(omitted.task_id, TaskState.RUNNING)
    report = _child_report("x" * (TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES + 1))
    await inner._finalize_background_agent(
      omitted,
      final_state=TaskState.COMPLETED,
      result=report,
      error=None,
    )
    assert omitted.notification_delivery_state == "payload_omitted"
    generation = omitted.notification_generation

    detached = inner._task_registry.register("background_agent", task_id="detached-leaf")
    inner._task_registry.transition(detached.task_id, TaskState.RUNNING)
    detached.completion_callback = lambda _entry: reconciled.set()
    finalizer = asyncio.create_task(inner._finalize_background_agent(
      detached,
      final_state=TaskState.COMPLETED,
      result=_child_report("settled after owner eviction"),
      error=None,
    ))
    await append_started.wait()
    finalizer.cancel()
    with pytest.raises(asyncio.CancelledError):
      await finalizer
    assert detached.state == TaskState.INTERRUPTED
    assert detached.completion_finalizer_detached

    await _kill_registry_owner(root, owner_entry, owner)
    root.register("background_agent", task_id="replacement")
    assert root.get(owner_entry.task_id) is None
    descendant = root.find_on_roster(detached.task_id)
    assert descendant is not None
    assert descendant.entry is detached
    assert descendant.parent_job_id == owner_entry.task_id
    assert owner_entry.task_id in root._child_registries

    release_append.set()
    await asyncio.wait_for(reconciled.wait(), timeout=2.0)

    assert detached.state == TaskState.COMPLETED
    assert detached.completion_persistence_state == "committed"
    assert not detached.completion_finalizer_detached
    assert root.find_on_roster(detached.task_id) is None
    assert root.find_on_roster(omitted.task_id) is None
    assert owner_entry.task_id not in root._child_registries
    assert [job.job_id for job in root.roster()] == ["root", "replacement"]
    assert omitted.notification_delivery_state == "payload_omitted"
    assert omitted.notification_generation == generation
    assert inner.durable_events[0]["result"] == omitted.result == report
    assert [event["final_state"] for event in inner.durable_events] == [
      "completed",
      "completed",
    ]

  _run(_case())


def test_foreground_registry_keeps_detached_completion_until_reconciled() -> None:
  root = TaskRegistry(owner=RegistryOwner(job_id="root", tenant=lambda: None))
  child = _linked_registry_runner(root, "foreground")
  entry = child._task_registry.register("background_agent")
  child._task_registry.transition(entry.task_id, TaskState.RUNNING)
  _close_registry_runner(child)
  entry.completion_finalizer_detached = True
  child._task_registry.transition(entry.task_id, TaskState.INTERRUPTED)

  assert root.find_on_roster(entry.task_id) is not None
  entry.completion_finalizer_detached = False
  child._task_registry.finalize_interrupted(entry.task_id, TaskState.FAILED)

  assert root.find_on_roster(entry.task_id) is None


def test_foreground_registry_prunes_terminal_detached_completion_after_reconciliation() -> None:
  root = TaskRegistry(owner=RegistryOwner(job_id="root", tenant=lambda: None))
  child = _linked_registry_runner(root, "foreground")
  child_registry = weakref.ref(child._task_registry)
  entry = child._task_registry.register("background_agent")
  child._task_registry.transition(entry.task_id, TaskState.RUNNING)
  entry.completion_finalizer_detached = True
  child._task_registry.transition(entry.task_id, TaskState.COMPLETED)
  _close_registry_runner(child)
  del child

  descendant = root.find_on_roster(entry.task_id)
  assert descendant is not None
  entry.completion_finalizer_detached = False
  descendant.registry.transition(entry.task_id, TaskState.COMPLETED)
  del descendant
  gc.collect()

  assert root.find_on_roster(entry.task_id) is None
  assert child_registry() is None


def test_foreground_registry_keeps_omitted_notification_until_retrieved() -> None:
  class _OmittedNotificationListener:
    def on_transition(
      self,
      entry: TaskEntry,
      old_state: TaskState,
      new_state: TaskState,
    ) -> None:
      if new_state == TaskState.COMPLETED:
        entry.notification_delivery_state = "payload_omitted"

  root = TaskRegistry(owner=RegistryOwner(job_id="root", tenant=lambda: None))
  child = _linked_registry_runner(root, "foreground")
  child._task_registry.add_listener(_OmittedNotificationListener())
  entry = child._task_registry.register("background_agent")
  child._task_registry.transition(entry.task_id, TaskState.RUNNING)
  _close_registry_runner(child)
  child._task_registry.transition(entry.task_id, TaskState.COMPLETED)

  assert root.find_on_roster(entry.task_id) is not None
  assert child._task_registry.mark_notification_payload_retrieved(
    entry.task_id,
    notification_generation=entry.notification_generation,
  )
  assert root.find_on_roster(entry.task_id) is None


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


@pytest.mark.parametrize("receipt", ["absent", "conflicting"])
def test_load_from_events_recovers_canonical_authority_independent_of_receipt(
  receipt: str,
) -> None:
  admitted = _canonical_admitted_task()
  assert admitted.model_bind is not None
  correlation = {
    "owner_runner_id": "runner-parent",
    "owner_role": "writer",
    "sub_agent_id": "sub7:session-parent",
    "parent_turn_id": "turn-7",
    "call_index": 7,
    "task_type": "background_agent",
  }
  metadata = {
    **correlation,
    "admitted_task": admitted.model_dump(mode="json"),
    "capability_bind": admitted.model_bind.to_json(),
    "custom": "retained",
  }
  registration = {
    "type": "task_registered",
    "event_schema_version": 2,
    "task_id": "bg_7",
    "agent_name": "research",
    "started_at": 100.0,
    "metadata": metadata,
    **correlation,
    "_durable_seq": 1,
  }
  raw_receipt = None
  if receipt == "conflicting":
    raw_receipt = _bind_receipt(provider="openai", model="gpt-5.2")
    registration["capability_bind"] = raw_receipt
  messages = [
    {
      "type": "parent_message_sent",
      "task_id": "bg_7",
      **correlation,
      "capability_bind": admitted.model_bind.to_json(),
      "message_id": message_id,
      "message": text,
      "sent_at": timestamp,
      "_durable_seq": seq,
    }
    for message_id, text, timestamp, seq in (
      ("msg-first", "Read the filing", 110.0, 2),
      ("msg-next", "Use the amendment", 120.0, 3),
    )
  ]
  registry = TaskRegistry()
  listener = _RecordingListener()
  registry.add_listener(listener)
  registry.load_from_events([registration, *messages])

  entry = registry.get("bg_7")
  assert entry is not None
  assert entry.state is TaskState.INTERRUPTED
  assert entry.error is None
  assert entry.reconstructed_from_log is True
  assert entry.registration_persistence_state == "committed"
  assert entry.completion_persistence_state == "not_started"
  assert entry.admitted_task == admitted
  assert entry.metadata["admitted_task"] == admitted.model_dump(mode="json")
  assert entry.metadata["custom"] == "retained"
  assert entry.capability_bind_receipt == (
    raw_receipt if isinstance(raw_receipt, dict) else None
  )
  assert entry.metadata["capability_bind"] == (
    admitted.model_bind.to_json() if receipt == "absent" else raw_receipt
  )
  assert list(entry.accepted_parent_messages.values()) == [
    ParentMessage("msg-first", "Read the filing", 110.0, "bg_7", 2),
    ParentMessage("msg-next", "Use the amendment", 120.0, "bg_7", 3),
  ]
  assert entry.delivered_messages == {"msg-first", "msg-next"}
  assert listener.transitions == []
  assert registry.register("background_agent").task_id == "bg_8"


def test_load_from_events_retains_noncanonical_registration_without_receipt() -> None:
  metadata = {"custom": {"observation": "raw"}, "owner_role": "writer"}
  registry = TaskRegistry()
  registry.load_from_events([{
    "type": "task_registered",
    "event_schema_version": 2,
    "task_id": "bg_7",
    "task_type": "plan_run",
    "started_at": 100.0,
    "original_task_id": "bg_3",
    "metadata": metadata,
  }])

  entry = registry.get("bg_7")
  assert entry is not None
  assert entry.state is TaskState.INTERRUPTED
  assert entry.admitted_task is None
  assert entry.capability_bind_receipt is None
  assert entry.metadata["custom"] == metadata["custom"]
  assert entry.metadata["owner_role"] == "writer"
  assert entry.original_task_id == "bg_3"
  assert entry.started_at == 100.0
  assert registry.register("background_agent").task_id == "bg_8"


@pytest.mark.parametrize(
  "tamper",
  ["model_bind_digest", "admitted_task_digest", "logical_identity"],
)
def test_load_from_events_detects_canonical_corruption_without_receipt(
  tamper: str,
) -> None:
  payload = _canonical_admitted_task().model_dump(mode="json")
  if tamper == "logical_identity":
    payload["logical_task"]["operation"]["name"] = "different-operation"
  else:
    payload[tamper] = f"sha256:{'f' * 64}"
  registry = TaskRegistry()
  registry.load_from_events([{
    "type": "task_registered",
    "event_schema_version": 2,
    "task_id": "bg_7",
    "task_type": "background_agent",
    "started_at": 100.0,
    "metadata": {"admitted_task": payload},
  }])

  entry = registry.get("bg_7")
  assert entry is not None
  assert entry.state is TaskState.FAILED
  assert entry.admitted_task is None
  assert entry.error is not None
  assert entry.error["code"] == "invalid_admitted_task"
  assert entry.metadata["admitted_task"] == payload


@pytest.mark.parametrize("tamper", [None, "physical_identity", "authority_digest"])
def test_load_from_events_preserves_canonical_result_identity_without_receipt(
  tamper: str | None,
) -> None:
  admitted = _canonical_admitted_task()
  task_result = terminal_task_result(
    admitted,
    status="interrupted",
    reason="Gateway restarted.",
  )
  result = task_result.model_dump(mode="json")
  if tamper == "physical_identity":
    result["attempt"]["physical_task_id"] = "bg_8"
  elif tamper == "authority_digest":
    result["provenance"]["admitted_task_digest"] = f"sha256:{'f' * 64}"
  registry = TaskRegistry()
  registry.load_from_events([
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "task_id": "bg_7",
      "task_type": "background_agent",
      "started_at": 100.0,
      "metadata": {"admitted_task": admitted.model_dump(mode="json")},
    },
    {
      "type": "task_completed",
      "task_id": "bg_7",
      "final_state": "interrupted",
      "completed_at": 125.0,
      "result": result,
      "error": None,
    },
  ])

  entry = registry.get("bg_7")
  assert entry is not None
  assert entry.admitted_task == admitted
  assert entry.result == result
  assert entry.completed_at == 125.0
  assert entry.completion_persistence_state == "committed"
  if tamper is None:
    assert entry.state is TaskState.INTERRUPTED
    assert entry.task_result == task_result
    assert entry.error is None
  else:
    assert entry.state is TaskState.FAILED
    assert entry.task_result is None
    assert entry.error is not None
    assert entry.error["code"] == (
      "invalid_task_result"
      if tamper == "physical_identity"
      else "task_result_admission_mismatch"
    )


def test_load_from_events_preserves_canonical_awaiting_approval_result() -> None:
  admitted = _canonical_admitted_task()
  usage = UsageObservation(
    input_tokens=120,
    output_tokens=35,
    cached_input_tokens=20,
    tool_calls=2,
    cost_usd=0.015,
  )
  task_result = terminal_task_result(
    admitted,
    status="awaiting_approval",
    reason="Waiting for a human approval decision.",
    usage=usage,
  )
  result = task_result.model_dump(mode="json")
  task_id = admitted.attempt.physical_task_id
  registry = TaskRegistry()
  registry.load_from_events([
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "task_id": task_id,
      "task_type": "background_agent",
      "started_at": 100.0,
      "metadata": {"admitted_task": admitted.model_dump(mode="json")},
    },
    {
      "type": "task_completed",
      "task_id": task_id,
      "final_state": "awaiting_approval",
      "completed_at": 125.0,
      "result": result,
      "error": None,
    },
  ])

  entry = registry.get(task_id)
  assert entry is not None
  assert entry.state is TaskState.AWAITING_APPROVAL
  assert entry.completed is False
  assert entry.error is None
  assert entry.task_id == admitted.attempt.physical_task_id
  assert entry.admitted_task == admitted
  assert entry.metadata["admitted_task"] == admitted.model_dump(mode="json")
  assert entry.result == result
  assert entry.task_result == task_result
  assert entry.task_result is not None
  assert entry.task_result.logical_task == admitted.logical_task
  assert entry.task_result.attempt == admitted.attempt
  assert entry.task_result.provenance == provenance_of(admitted)
  assert entry.task_result.observation.usage == usage
  assert entry.task_result.observation.transcript.owner_id == task_id
  assert entry.task_result.observation.activity.owner_id == task_id
  assert entry.completed_at == 125.0
  assert entry.completion_persistence_state == "committed"

  with pytest.raises(
    TaskResultSettlementError,
  ):
    validate_task_result_settlement(
      task_result,
      final_state=TaskState.INTERRUPTED,
      error=None,
    )
  with pytest.raises(
    TaskResultSettlementError,
  ):
    validate_task_result_settlement(
      task_result,
      final_state=TaskState.AWAITING_APPROVAL,
      error={"code": "approval_timeout", "message": "Approval timed out."},
    )


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


def test_job_read_custody_follows_downward_and_forward_edges_only(tmp_path: Path) -> None:
  async def _case() -> None:
    durable_log = AgentSessionLog(tmp_path / "custody.jsonl")
    a0 = canonical_read_custody_task("a0")
    old_child = canonical_read_custody_task("a0-child", parent_job_id="a0")
    old_sibling = canonical_read_custody_task("old-sibling")
    actor_parent = canonical_read_custody_task("actor-parent", parent_job_id="actor-root")
    actor = canonical_read_custody_task("nested-actor", parent_job_id="actor-parent")
    a1 = canonical_read_custody_task("a1", predecessor=a0, resumer_job_id="nested-actor")
    current_child = canonical_read_custody_task("a1-child", parent_job_id="a1")
    later_actor = canonical_read_custody_task("later-actor", parent_job_id="later-root")
    a2 = canonical_read_custody_task("a2", predecessor=a1, resumer_job_id="later-actor")
    later_child = canonical_read_custody_task("a2-child", parent_job_id="a2")
    unrelated = canonical_read_custody_task("unrelated", parent_job_id="unrelated-root")
    unrelated_successor = canonical_read_custody_task("unrelated-successor",
    predecessor=unrelated,
    resumer_job_id="a1",)
    tasks = (
      a0, old_child, old_sibling, actor_parent, actor, a1, current_child,
      later_actor, a2, later_child, unrelated, unrelated_successor,
    )
    outputs = {}
    for task in tasks:
      await durable_log.append(custody_registration(task))
      outputs[task.attempt.physical_task_id] = await _append_custody_completion(durable_log, task)

    resumed = {"a0", "a0-child", "a1", "a1-child", "a2", "a2-child"}
    expected = {
      "original-owner": {"original-owner", "old-sibling"} | resumed,
      "nested-actor": {"nested-actor"} | resumed,
      "later-actor": {"later-actor", "a1", "a1-child", "a2", "a2-child"},
      # Physical ancestors retain their children, not their children's custody.
      "actor-parent": {"actor-parent", "nested-actor"},
      "actor-root": {"actor-root", "actor-parent", "nested-actor"},
      "later-root": {"later-root", "later-actor"},
      # A reached successor does not pass its own unrelated custody onward.
      "a1": {"a1", "a1-child", "a2", "a2-child", "unrelated", "unrelated-successor"},
      "session-alias": {"session-alias"},
      "runner-alias": {"runner-alias"},
      "unrelated-root": {"unrelated-root", "unrelated", "unrelated-successor"},
    }
    for reader, jobs in expected.items():
      assert await job_read_subtree(durable_log, job_id=reader) == jobs
      assert await job_output_handles(durable_log, job_id=reader) == frozenset(
        outputs[job] for job in jobs if job in outputs
      )
    # Cold replay needs neither a retained registry nor a new custody event.
    replay = AgentSessionLog(tmp_path / "custody.jsonl")
    assert await job_read_subtree(replay, job_id="nested-actor") == expected["nested-actor"]
    assert await job_read_subtree(replay, job_id="later-actor") == expected["later-actor"]

  _run(_case())


@pytest.mark.parametrize("tamper", ["digest", "physical_identity", "unsealed_metadata"])
def test_job_read_custody_requires_the_canonical_successor(
  tmp_path: Path,
  tamper: str,
) -> None:
  async def _case() -> None:
    durable_log = AgentSessionLog(tmp_path / "custody.jsonl")
    original = canonical_read_custody_task("original")
    successor = canonical_read_custody_task("successor", predecessor=original, resumer_job_id="actor")
    await durable_log.append(custody_registration(original))
    registration = custody_registration(successor)
    if tamper == "digest":
      registration["metadata"]["admitted_task"]["attempt"]["resumer_job_id"] = "other-actor"
    elif tamper == "physical_identity":
      registration["task_id"] = "physical-alias"
    else:
      registration["metadata"]["admitted_task"] = {
        "attempt": successor.attempt.model_dump(mode="json"),
        "execution_snapshot": {
          "admitted_context": {"lineage": {"parent_job_id": "original-owner"}},
        },
      }
    await durable_log.append(registration)
    for reader in ("actor", "other-actor", "session-alias", "runner-alias"):
      assert await job_read_subtree(durable_log, job_id=reader) == {reader}
    assert await job_read_subtree(durable_log, job_id="original") == {"original"}

  _run(_case())


def test_job_read_custody_preserves_old_attempt_bytes_without_actor(tmp_path: Path) -> None:
  async def _case() -> None:
    durable_log = AgentSessionLog(tmp_path / "historical.jsonl")
    original = canonical_read_custody_task("original")
    successor = canonical_read_custody_task("successor", predecessor=original)
    # Construct the pre-custody attempt wire, not a dump of the new optional field.
    payload = successor.model_dump(mode="json", exclude={"admitted_task_digest"})
    payload["attempt"] = {
      "attempt_number": 2,
      "attempt_id": "attempt:successor",
      "physical_task_id": "successor",
      "restart_of_attempt_id": None,
      "resume_of_task_id": "original",
    }
    seal_admitted_task_payload(payload)
    old_digest = payload["admitted_task_digest"]
    old_wire = canonical_json_bytes(payload)
    replayed = AdmittedTask.model_validate(payload)
    assert replayed.admitted_task_digest == old_digest
    assert canonical_json_bytes(replayed.model_dump(mode="json")) == old_wire
    await durable_log.append(custody_registration(original))
    await durable_log.append(custody_registration(replayed, parent_session_id="later-session"))
    assert await job_read_subtree(durable_log, job_id="original") == {"original", "successor"}
    assert await job_read_subtree(durable_log, job_id="successor") == {"successor"}
    assert await job_read_subtree(durable_log, job_id="later-session") == {"later-session"}

  _run(_case())


def test_job_read_custody_stays_in_its_tenant_journal_after_completion_eviction(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    alice_log = AgentSessionLog(tmp_path / "alice" / "session.jsonl")
    bob_log = AgentSessionLog(tmp_path / "bob" / "session.jsonl")
    original = canonical_read_custody_task("alice-original")
    successor = canonical_read_custody_task("alice-successor", predecessor=original, resumer_job_id="actor")
    stranger = canonical_read_custody_task("bob-task", parent_job_id="bob-owner", tenant_storage_id="bob")
    for task in (original, successor):
      await alice_log.append(custody_registration(task))
    await bob_log.append(custody_registration(stranger))
    original_content = await _append_custody_completion(alice_log, original)
    successor_content = await _append_custody_completion(alice_log, successor)
    stranger_content = await _append_custody_completion(bob_log, stranger)

    events, _ = await alice_log.query(event_types={"task_registered", "task_completed", "agent_completion"})
    retained = TaskRegistry()
    retained.load_from_events([item.event for item in events])
    retained_successor = retained.get("alice-successor")
    assert retained_successor is not None
    assert retained_successor.state is TaskState.COMPLETED
    assert retained_successor.admitted_task == successor
    registry = TaskRegistry(max_retained=0)
    registry.load_from_events([item.event for item in events])
    assert registry.list_tasks() == []
    replay = AgentSessionLog(tmp_path / "alice" / "session.jsonl")
    assert await job_output_handles(replay, job_id="actor") == frozenset({
      original_content, successor_content,
    })
    assert await job_read_subtree(bob_log, job_id="actor") == {"actor"}
    assert await job_output_handles(bob_log, job_id="actor") == frozenset()
    assert await job_output_handles(bob_log, job_id="bob-owner") == frozenset({stranger_content})
    assert await job_read_subtree(replay, job_id="bob-owner") == {"bob-owner"}

  _run(_case())
