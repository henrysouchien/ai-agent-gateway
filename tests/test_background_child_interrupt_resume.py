from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from agent_gateway import AgentRunner, AgentSessionLog, TaskRegistry, TaskState
from agent_gateway.capability_execution import CapabilityExecutionResolver
from agent_gateway.providers import ModelInfo, StreamEvent
from agent_gateway.session import GatewaySession
from agent_gateway.skills import SkillLoader
from agent_gateway.sub_agent import (
  make_resume_handler, make_run_agent_handler, make_run_agent_tool_def,
)
from agent_gateway.transcript import reconstruct_messages_for_task
from gateway_test_support.capability_execution_test_support import (
  stub_capability_execution_resolver,
)
from gateway_test_support.host_policy import owner_session_host_policy
from test_sub_agent_resume import _NullMcpClient, _ResumeTestProvider, _runner, _write_skill


class _LiveChildProvider(_ResumeTestProvider):
  """Persist a real assistant/tool turn before blocking each child's next turn."""

  name = "anthropic"

  def __init__(self) -> None:
    self.ready = {name: asyncio.Event() for name in ("first", "sibling")}
    self.release = {name: asyncio.Event() for name in self.ready}
    self.cancelled = {name: asyncio.Event() for name in self.ready}
    self.calls = dict.fromkeys(self.ready, 0)

  def normalize_messages(
    self, messages: list[dict[str, Any]], model_info: ModelInfo,
  ) -> list[dict[str, Any]]:
    return messages

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return {"messages": copy.deepcopy(kwargs["messages"])}

  async def stream(self, client: Any, params: dict[str, Any]):
    rendered = json.dumps(params["messages"])
    name = next(name for name in self.ready if f"[child:{name}]" in rendered)
    self.calls[name] += 1
    yield StreamEvent(type="message_start", input_tokens=10)
    if self.calls[name] == 1:
      text = f"Retained original assistant evidence for {name}."
      yield StreamEvent(type="text_delta", text=text)
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
      yield StreamEvent(
        type="tool_use_end", tool_id=f"read-{name}",
        tool_name="web_search", tool_input={"query": name},
      )
      yield StreamEvent(type="message_end", stop_reason="tool_use")
      return
    if self.calls[name] == 2:
      self.ready[name].set()
      try:
        await self.release[name].wait()
      except asyncio.CancelledError:
        self.cancelled[name].set()
        raise
    text = f"Completed {name} with the original admitted evidence."
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
    yield StreamEvent(type="usage_update", output_tokens=5)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _Children:
  def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    self.provider = _LiveChildProvider()
    base = stub_capability_execution_resolver()
    self.resolver = CapabilityExecutionResolver(
      registry=base.registry,
      selection_policy=base.selection_policy,
      auth_context=base.auth_context,
      credential_materializer=base.credential_materializer,
      adapter_resolver=lambda _adapter: self.provider,
    )
    self.runner = _runner(
      tmp_path, capability_execution=self.resolver.resolve("session.driver"),
    )
    self.runner._get_tool_definitions = lambda: [{
      "name": "web_search", "description": "Read evidence.",
      "input_schema": {"type": "object"},
    }]
    skills_dir = tmp_path / "skills"
    _write_skill(
      skills_dir,
      "interruptible-child",
      """
allowed_tools:
  - web_search
mcp_tools: {}
semantic_metadata:
  required_context: []
  tool_refs:
    - kind: local
      tool_id: web_search
  capability_requirements:
    - name: web.read/v1
      required: true
      binding_modes: [live_tool]
""",
      body="Read evidence, retain the conversation, and finish the requested report.",
    )
    loader = SkillLoader(skills_dir)
    self.operation = next(
      item.snapshot.operation.model_dump(mode="json")
      for item in loader.list_callable_operations()
      if item.snapshot.operation.name == "interruptible-child"
    )
    session = GatewaySession(
      session_id="sess-parent", api_key_hash="hash", created_at=1, expires_at=2,
      user_id="alice", role="owner",
    )

    async def web_search(tool_input: dict[str, Any], **_kwargs: Any):
      return {"evidence": f"Original evidence for {tool_input['query']}"}, None

    local = {"web_search": web_search}
    self.resume_handler = make_resume_handler(
      [self.runner], operation_resume_bindings=None, parent_session=session,
      mcp_client=_NullMcpClient(), local_tool_handlers=local,
      excluded_tools_resolver=frozenset,
      capability_execution_resolver=self.resolver,
    )
    # Use the same configured resume owner as the existing resume tool.
    getattr(self.runner._dispatcher, "_local")["resume"] = self.resume_handler
    self.run_agent = make_run_agent_handler(
      [self.runner], parent_session=session, skill_loader=loader,
      mcp_client=_NullMcpClient(), local_tool_handlers=local,
      capability_execution_resolver=self.resolver,
    )
    self.resumed_runs: list[dict[str, Any]] = []
    original_run = AgentRunner.run

    async def record_run(child: AgentRunner, *args: Any, **kwargs: Any):
      if "resume_initial_messages" in kwargs:
        self.resumed_runs.append({
          "messages": copy.deepcopy(kwargs["resume_initial_messages"]),
          "bind": child.capability_execution.bind,
        })
      return await original_run(child, *args, **kwargs)

    monkeypatch.setattr(AgentRunner, "run", record_run)

  async def start(self, name: str):
    call_index = 1 if name == "first" else 2
    result, error = await self.run_agent(
      {
        "operation": self.operation,
        "objective": f"[child:{name}] Review the original admitted evidence.",
        "background": True,
      },
      call_index=call_index,
    )
    assert error is None
    assert result is not None
    entry = self.runner._task_registry.get(result["task_id"])
    assert entry is not None
    assert entry.admitted_task is not None
    assert entry.asyncio_task is not None
    await asyncio.wait_for(self.provider.ready[name].wait(), timeout=5)
    assert entry.state == TaskState.RUNNING
    return entry

  async def close(self) -> None:
    for release in self.provider.release.values():
      release.set()
    tasks = [
      entry.asyncio_task for entry in self.runner._task_registry.list_tasks()
      if entry.asyncio_task is not None
    ]
    if tasks:
      await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)


@pytest.fixture
def children(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner_session_host_policy):
  owner_session_host_policy.get_local_tool_effect = lambda name: (
    "read" if name == "web_search" else None
  )
  owner_session_host_policy.get_read_capability = lambda name, _server: (
    "web.read/v1" if name == "web_search" else None
  )
  # Construct asyncio primitives inside the test's running event loop.
  return lambda: _Children(tmp_path, monkeypatch)


async def _assert_durable_settlement(runner, entry, state, status, reason=None):
  log = runner._agent_session_log
  assert log is not None
  reopened = AgentSessionLog(path=log.path)
  rows, _ = await reopened.query(event_types={"task_completed"}, order="asc")
  terminal = [row.event for row in rows if row.event.get("task_id") == entry.task_id]
  assert len(terminal) == 1
  assert terminal[0]["final_state"] == state.value
  assert terminal[0]["result"]["execution"]["status"] == status
  if reason is not None:
    assert terminal[0]["result"]["execution"]["terminal_reason"] == reason
  assert entry.completion_persistence_state == "committed"
  recovered = await runner._lookup_task_in_log(entry.task_id)
  assert recovered is not None
  assert recovered.state == state
  assert recovered.admitted_task == entry.admitted_task
  assert recovered.task_result == entry.task_result
  return reopened


@pytest.mark.parametrize(
  ("control", "reason", "additional_context"),
  [
    ("runner", "user_interrupted", None),
    ("run_agent", "parent_interrupted", "Continue with the original evidence only."),
  ],
)
def test_live_child_interrupt_and_model_resume_preserve_sibling_and_authority(
  children, control: str, reason: str, additional_context: str | None,
) -> None:
  async def case() -> None:
    live = children()
    runner = live.runner
    parent = asyncio.current_task()
    assert parent is not None
    parent_cancels = parent.cancelling()
    try:
      first = await live.start("first")
      sibling = await live.start("sibling")
      original = first.admitted_task
      assert original is not None
      original_payload = original.model_dump(mode="json")
      assert first.asyncio_task is not None
      assert sibling.asyncio_task is not None
      sibling_cancels = sibling.asyncio_task.cancelling()

      if control == "runner":
        assert runner.interrupt_background_task(first.task_id) is True
      else:
        result, error = await live.run_agent({"interrupt": first.task_id})
        assert error is None
        assert result is not None
      # Repeating the request must not queue another cancellation into cleanup.
      assert runner._task_registry.interrupt(first.task_id) is False
      await asyncio.wait_for(asyncio.shield(first.asyncio_task), timeout=5)
      assert live.provider.cancelled["first"].is_set()
      assert first.state == TaskState.INTERRUPTED
      assert first.interruption_reason == reason
      assert first.termination_intent is None
      assert first.task_result is not None
      assert first.task_result.execution.status == "interrupted"
      assert first.task_result.execution.terminal_reason == reason
      assert first.admitted_task.model_dump(mode="json") == original_payload
      assert runner.interrupt_background_task(first.task_id) is False
      assert parent.cancelling() == parent_cancels
      assert not parent.cancelled()
      assert sibling.state == TaskState.RUNNING
      assert sibling.asyncio_task.cancelling() == sibling_cancels
      assert not live.provider.cancelled["sibling"].is_set()

      reopened = await _assert_durable_settlement(
        runner, first, TaskState.INTERRUPTED, "interrupted", reason,
      )
      transcript = await reconstruct_messages_for_task(reopened, first.task_id)
      assert transcript[0]["role"] == "user"
      assert "[child:first]" in json.dumps(transcript[0])
      assert any(
        message["role"] == "assistant"
        and "Retained original assistant evidence for first." in json.dumps(message)
        for message in transcript
      )
      registration, _ = await reopened.query(event_types={"task_registered"})
      original_event = next(
        row.event for row in registration if row.event.get("task_id") == first.task_id
      )
      assert original_event["capability_bind"] == original.model_bind.to_json()

      live.provider.release["sibling"].set()
      await asyncio.wait_for(asyncio.shield(sibling.asyncio_task), timeout=5)
      assert sibling.state == TaskState.COMPLETED
      await _assert_durable_settlement(runner, sibling, TaskState.COMPLETED, "succeeded")

      request = {"resume": first.task_id}
      if additional_context is not None:
        request["additional_context"] = additional_context
      result, error = await live.run_agent(request)
      assert error is None
      assert result is not None
      successor = runner._task_registry.get(result["task_id"])
      assert successor is not None
      assert successor.original_task_id == first.task_id
      assert successor.admitted_task is not None
      assert successor.admitted_task.logical_task == original.logical_task
      assert successor.admitted_task.model_bind == original.model_bind
      assert successor.admitted_task.attempt.resume_of_task_id == first.task_id
      assert successor.asyncio_task is not None
      await asyncio.wait_for(asyncio.shield(successor.asyncio_task), timeout=5)
      assert successor.state == TaskState.COMPLETED
      assert len(live.resumed_runs) == 1
      resumed = live.resumed_runs[0]
      assert resumed["bind"] == original.model_bind
      # Both the original objective and assistant turn survive, not a fresh task.
      assert resumed["messages"][:2] == transcript[:2]
      if additional_context is not None:
        assert additional_context in json.dumps(resumed["messages"])
      await _assert_durable_settlement(
        runner, successor, TaskState.COMPLETED, "succeeded",
      )
      assert first.admitted_task.model_dump(mode="json") == original_payload
      assert parent.cancelling() == parent_cancels
      assert sibling.state == TaskState.COMPLETED
    finally:
      await live.close()

  asyncio.run(case())


def test_run_agent_schema_has_no_top_level_combinators() -> None:
  schema = make_run_agent_tool_def()["input_schema"]

  assert not {"oneOf", "anyOf", "allOf"} & set(schema)
  assert {"objective", "interrupt", "resume", "additional_context"} <= set(
    schema["properties"]
  )


def test_run_agent_resume_without_installed_resume_handler_is_structured(
  children,
) -> None:
  async def case() -> None:
    live = children()
    runner = live.runner
    try:
      first = await live.start("first")
      assert runner.interrupt_background_task(first.task_id) is True
      assert first.asyncio_task is not None
      await asyncio.wait_for(asyncio.shield(first.asyncio_task), timeout=5)
      assert first.state == TaskState.INTERRUPTED

      del getattr(runner._dispatcher, "_local")["resume"]
      result, error = await live.run_agent({"resume": first.task_id})

      assert result is None
      assert error is not None
      assert error["code"] == "invalid_input"
      assert error["message"] == (
        "Background child resume is not configured on this runner."
      )
      assert first.state == TaskState.INTERRUPTED
      assert not live.resumed_runs
    finally:
      await live.close()

  asyncio.run(case())



@pytest.mark.parametrize("control", ["registry_kill", "runner_cancel"])
def test_live_child_kill_remains_killed(children, control: str) -> None:
  async def case() -> None:
    live = children()
    try:
      first = await live.start("first")
      sibling = await live.start("sibling")
      if control == "registry_kill":
        assert live.runner._task_registry.kill(first.task_id) is True
        reason = "killed"
      else:
        assert live.runner.cancel_background_task(first.task_id) is True
        reason = "cancelled"
      assert first.asyncio_task is not None
      await asyncio.wait_for(asyncio.shield(first.asyncio_task), timeout=5)
      assert first.state == TaskState.KILLED
      assert first.interruption_reason is None
      assert first.task_result is not None
      assert first.task_result.execution.status == "cancelled"
      assert first.task_result.execution.terminal_reason == reason
      await _assert_durable_settlement(
        live.runner, first, TaskState.KILLED, "cancelled", reason,
      )
      assert sibling.asyncio_task is not None
      assert sibling.asyncio_task.cancelling() == 0
      live.provider.release["sibling"].set()
      await asyncio.wait_for(asyncio.shield(sibling.asyncio_task), timeout=5)
      assert sibling.state == TaskState.COMPLETED
    finally:
      await live.close()

  asyncio.run(case())


def test_interrupt_rejects_unknown_foreign_and_nonlive_children(children) -> None:
  async def case() -> None:
    live = children()
    try:
      first = await live.start("first")
      sibling = await live.start("sibling")
      registry = live.runner._task_registry
      foreign = TaskRegistry()
      assert live.runner.interrupt_background_task("unknown-child") is False
      assert foreign.interrupt(first.task_id) is False
      assert first.asyncio_task is not None
      assert sibling.asyncio_task is not None
      for state in (TaskState.PENDING, TaskState.PARKED, TaskState.INTERRUPTED,
                    TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED):
        entry = registry.register("background", task_id=f"nonlive-{state.value}")
        entry.state = state
        assert live.runner.interrupt_background_task(entry.task_id) is False
        assert entry.interruption_reason is None
      missing_worker = registry.register("background", task_id="running-without-worker")
      missing_worker.state = TaskState.RUNNING
      assert registry.interrupt(missing_worker.task_id) is False
      finished_worker = registry.register("background", task_id="running-finished-worker")
      finished_worker.state = TaskState.RUNNING
      finished_worker.asyncio_task = asyncio.create_task(asyncio.sleep(0))
      await finished_worker.asyncio_task
      assert registry.interrupt(finished_worker.task_id) is False

      # Staged completion and preexisting control intent are not live targets.
      for field, value in (
        ("pending_final_state", TaskState.COMPLETED),
        ("termination_intent", "killed"),
        ("interruption_reason", "parent_interrupted"),
      ):
        setattr(first, field, value)
        try:
          assert registry.interrupt(first.task_id) is False
          assert first.asyncio_task.cancelling() == 0
        finally:
          setattr(first, field, None)
      assert first.state == sibling.state == TaskState.RUNNING
      assert first.asyncio_task.cancelling() == sibling.asyncio_task.cancelling() == 0
      assert not any(event.is_set() for event in live.provider.cancelled.values())
      for release in live.provider.release.values():
        release.set()
      await asyncio.wait_for(asyncio.gather(first.asyncio_task, sibling.asyncio_task), timeout=5)
      assert first.state == sibling.state == TaskState.COMPLETED
      assert registry.interrupt(first.task_id) is False
    finally:
      await live.close()

  asyncio.run(case())


@pytest.mark.parametrize("consume_cancellation", [False, True])
def test_interrupt_immediately_after_registration_survives_handler_cleanup(
  tmp_path: Path, consume_cancellation: bool,
) -> None:
  async def case() -> None:
    runner = _runner(tmp_path)
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    never_complete = asyncio.Event()

    async def handler(
      _tool_input: dict[str, Any], **_kwargs: Any,
    ) -> tuple[Any | None, dict[str, Any] | None]:
      entered.set()
      try:
        await never_complete.wait()
      except asyncio.CancelledError:
        cancelled.set()
        if consume_cancellation:
          return {"handler_claimed_success": True}, None
        raise
      return None, None
    result, error = await runner._register_background_task(
      tool_input={},
      handler=handler,
      task_type="background_agent",
      capability_bind_receipt=runner.capability_execution.bind.to_json(),
    )
    assert error is None
    assert result is not None
    entry = runner._task_registry.get(result["task_id"])
    assert entry is not None
    assert entry.asyncio_task is not None

    # No readiness wait: control is accepted on the registration-return edge.
    # Hold the real finalizer so a duplicate can race with durable settlement.
    await entry.finalization_lock.acquire()
    try:
      assert runner.interrupt_background_task(entry.task_id) is True
      assert entry.interruption_reason == "user_interrupted"
      assert runner.interrupt_background_task(entry.task_id) is False
      await asyncio.wait_for(cancelled.wait(), timeout=5)
      assert entered.is_set()
      assert entry.pending_final_state == TaskState.INTERRUPTED
      assert not entry.asyncio_task.done()
      cancellation_count = entry.asyncio_task.cancelling()
      assert runner._task_registry.interrupt(entry.task_id) is False
      assert entry.asyncio_task.cancelling() == cancellation_count
    finally:
      entry.finalization_lock.release()
    await asyncio.wait_for(asyncio.shield(entry.asyncio_task), timeout=5)
    assert not entry.asyncio_task.cancelled()
    assert entry.state == TaskState.INTERRUPTED
    assert entry.interruption_reason == "user_interrupted"
    assert entry.termination_intent is None
    assert entry.completion_persistence_state == "committed"
    assert entry.result is not None
    assert entry.result["status"] == "interrupted"
    assert entry.result["reason"] == "user_interrupted"
    assert "handler_claimed_success" not in entry.result
    assert runner._agent_session_log is not None
    reopened = AgentSessionLog(path=runner._agent_session_log.path)
    rows, _ = await reopened.query(event_types={"task_completed"}, order="asc")
    terminal = [row.event for row in rows if row.event.get("task_id") == entry.task_id]
    assert len(terminal) == 1
    assert terminal[0]["final_state"] == TaskState.INTERRUPTED.value
    assert terminal[0]["result"] == entry.result
    assert runner.interrupt_background_task(entry.task_id) is False

  asyncio.run(case())
