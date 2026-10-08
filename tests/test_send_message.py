# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (
  AgentRunner,
  AgentSessionLog,
  CostEstimate,
  EventLog,
  McpClientManager,
  ModelInfo,
  ParentMessage,
  TaskRegistry,
  TaskState,
  ToolDispatcher,
)
from agent_gateway.sub_agent_messages import make_send_message_handler
from agent_gateway.runner import StreamTurnResult
from gateway_test_support.capability_execution_test_support import (
  stub_runner_capability_execution,
)
from agent_gateway.admission import ParentContext
from agent_gateway.approval_route import NoApprovalRoute
from agent_gateway.task_registry import RegistryOwner
from agent_workflow_contracts.models import AdmittedResearchBinding, AdmittedTenant


_TENANT = AdmittedTenant(tenant_id=None, storage_id="alice", role="owner")
_ROOT_JOB = "sess-root"


def _job_context(
  job_id: str,
  *,
  tenant: AdmittedTenant | None = _TENANT,
  handle_grants: frozenset[str] = frozenset(),
) -> ParentContext:
  return ParentContext(
    tenant=tenant,
    research_binding=AdmittedResearchBinding(),
    children=None,
    tool_ids=None,
    withheld=frozenset(),
    handle_grants=handle_grants,
    remaining_usd=None,
    job_id=job_id,
    depth=0 if job_id == _ROOT_JOB else 1,
    approval_route=NoApprovalRoute(),
  )


def _root_caller() -> ParentContext:
  return _job_context(_ROOT_JOB)


def _root_registry(
  *,
  inbox: asyncio.Queue[ParentMessage] | None = None,
  tenant: AdmittedTenant | None = _TENANT,
) -> TaskRegistry:
  return TaskRegistry(owner=RegistryOwner(
    job_id=_ROOT_JOB,
    tenant=lambda: tenant,
    inbox=inbox,
  ))


def _bind_receipt() -> dict[str, str]:
  return {
    "capability_id": "node.implement",
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "effort": "high",
    "policy_id": "test-policy",
    "policy_version": "1",
    "credential_principal": "user",
    "run_mode": "interactive",
  }


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)


class _RegistryRunner:
  def __init__(self, registry: TaskRegistry | None) -> None:
    self._task_registry = registry


class _DurableRegistryRunner(_RegistryRunner):
  _runner_id = "runner_new"
  _role = "writer"
  _full_session_id = "sess-parent"
  _usage_user_id = "alice"

  def __init__(self, registry: TaskRegistry | None) -> None:
    super().__init__(registry)
    self.events: list[dict[str, Any]] = []
    self.order: list[str] = []

  async def _append_durable_event(self, event: dict[str, Any]) -> Any:
    self.order.append("append")
    self.events.append(dict(event))
    return SimpleNamespace(seq=len(self.events))


class _RecordingQueue(asyncio.Queue[ParentMessage]):
  def __init__(self, order: list[str]) -> None:
    super().__init__()
    self.order = order

  async def put(self, item: ParentMessage) -> None:
    self.order.append("put")
    await super().put(item)


class _FailingQueue(asyncio.Queue[ParentMessage]):
  async def put(self, item: ParentMessage) -> None:
    _ = item
    raise RuntimeError("queue closed")


class _StubProvider:
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if False:
      yield

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    _ = model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
    return CostEstimate()


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-send-message",
  )


def test_make_send_message_handler_delivers_message_by_task_id() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)], caller=_root_caller)

  result, error = _run(handler({"to": entry.task_id, "message": "Check the appendix"}))

  assert error is None
  assert result is not None
  assert result["status"] == "queued"
  assert result["task_id"] == entry.task_id
  assert result["message_id"]
  parent_message = entry.message_inbox.get_nowait()
  assert parent_message.text == "Check the appendix"
  assert parent_message.message_id in entry.delivered_messages


def test_make_send_message_handler_emits_parent_message_sent_with_correlation() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  entry.capability_bind_receipt = _bind_receipt()
  entry.metadata.update(
    {
      "owner_runner_id": "runner_old",
      "owner_role": "writer",
      "sub_agent_id": "sub0:sess-parent",
      "parent_turn_id": "turn-1",
      "call_index": 0,
      "task_type": "background",
      "capability_bind": _bind_receipt(),
    }
  )
  registry.transition(entry.task_id, TaskState.RUNNING)
  runner = _DurableRegistryRunner(registry)
  handler = make_send_message_handler([runner], caller=_root_caller)

  result, error = _run(handler({"to": entry.task_id, "message": "Check the appendix"}))

  assert error is None
  assert result is not None
  assert len(runner.events) == 1
  event = runner.events[0]
  assert result["status"] == "accepted"
  assert result["task_id"] == entry.task_id
  assert result["message_id"] == event["message_id"]
  assert event["type"] == "parent_message_sent"
  assert event["task_id"] == entry.task_id
  assert event["owner_runner_id"] == "runner_old"
  assert event["owner_role"] == "writer"
  assert event["sub_agent_id"] == "sub0:sess-parent"
  assert event["parent_turn_id"] == "turn-1"
  assert event["call_index"] == 0
  assert event["task_type"] == "background"
  assert event["capability_bind"] == _bind_receipt()
  assert event["message"] == "Check the appendix"
  assert event["message_id"]
  parent_message = entry.message_inbox.get_nowait()
  assert parent_message.message_id == event["message_id"]
  assert parent_message.sent_at == event["sent_at"]


def test_make_send_message_handler_emits_before_deliver() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  runner = _DurableRegistryRunner(registry)
  entry.message_inbox = _RecordingQueue(runner.order)
  handler = make_send_message_handler([runner], caller=_root_caller)

  result, error = _run(handler({"to": entry.task_id, "message": "Check ordering"}))

  assert error is None
  assert result is not None
  assert result["status"] == "accepted"
  assert result["task_id"] == entry.task_id
  assert result["message_id"] == runner.events[0]["message_id"]
  assert runner.order == ["append", "put"]


def test_make_send_message_handler_is_idempotent_for_delivered_message_id() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  entry.accepted_parent_messages["msg-existing"] = ParentMessage(
    message_id="msg-existing",
    text="Replay",
    sent_at=1.0,
  )
  entry.delivered_messages.add("msg-existing")
  registry.transition(entry.task_id, TaskState.RUNNING)
  runner = _DurableRegistryRunner(registry)
  handler = make_send_message_handler([runner], caller=_root_caller)

  result, error = _run(handler({"to": entry.task_id, "message": "Replay", "message_id": "msg-existing"}))

  assert error is None
  assert result == {
    "status": "queued",
    "task_id": entry.task_id,
    "message_id": "msg-existing",
  }
  assert runner.events == []
  assert entry.message_inbox.empty()


def test_send_message_uses_trusted_tool_call_identity_for_terminal_replay() -> None:
  async def _case() -> None:
    registry = _root_registry()
    entry = registry.register("background_agent", agent_name="writer")
    registry.transition(entry.task_id, TaskState.RUNNING)
    runner = _DurableRegistryRunner(registry)
    handler = make_send_message_handler([runner], caller=_root_caller)
    tool_ctx = SimpleNamespace(tool_call_id="toolu-send-1")

    first, first_error = await handler(
      {
        "to": entry.task_id,
        "message": "Check margins",
        "message_id": "model-spoofed-id",
      },
      tool_ctx=tool_ctx,
    )
    registry.transition(entry.task_id, TaskState.COMPLETED)
    replay, replay_error = await handler(
      {"to": entry.task_id, "message": "Check margins"},
      tool_ctx=tool_ctx,
    )
    conflict, conflict_error = await handler(
      {"to": entry.task_id, "message": "Check revenue"},
      tool_ctx=tool_ctx,
    )
    missing_identity, missing_identity_error = await handler(
      {
        "to": entry.task_id,
        "message": "Check margins",
        "message_id": "model-spoofed-id",
      },
      tool_ctx=SimpleNamespace(),
    )

    assert first_error is None
    assert replay_error is None
    assert first == replay == {
      "status": "accepted",
      "task_id": entry.task_id,
      "message_id": "toolu-send-1",
    }
    assert conflict is None
    assert conflict_error is not None
    assert conflict_error["code"] == "message_id_conflict"
    assert missing_identity is None
    assert missing_identity_error is not None
    assert missing_identity_error["code"] == "message_control_identity_invalid"
    assert len(runner.events) == 1

  _run(_case())


def test_send_message_and_completion_lock_order_is_truthful() -> None:
  async def _send_first() -> None:
    registry = _root_registry()
    entry = registry.register("background_agent", agent_name="writer")
    registry.transition(entry.task_id, TaskState.RUNNING)
    runner = _DurableRegistryRunner(registry)
    append_started = asyncio.Event()
    release_append = asyncio.Event()
    original_append = runner._append_durable_event

    async def _blocking_append(event: dict[str, Any]) -> Any:
      append_started.set()
      await release_append.wait()
      return await original_append(event)

    runner._append_durable_event = _blocking_append
    handler = make_send_message_handler([runner], caller=_root_caller)
    send = asyncio.create_task(handler({
      "to": entry.task_id,
      "message": "Accepted before completion",
      "message_id": "send-first",
    }))
    await append_started.wait()

    async def _complete() -> None:
      async with entry.finalization_lock:
        registry.transition(entry.task_id, TaskState.COMPLETED)

    completion = asyncio.create_task(_complete())
    await asyncio.sleep(0)
    assert not completion.done()
    release_append.set()
    result, error = await send
    await completion
    replay, replay_error = await handler({
      "to": entry.task_id,
      "message": "Accepted before completion",
      "message_id": "send-first",
    })
    assert error is None
    assert replay_error is None
    assert result == replay
    assert result is not None and result["status"] == "accepted"

  async def _completion_first() -> None:
    registry = _root_registry()
    entry = registry.register("background_agent", agent_name="writer")
    registry.transition(entry.task_id, TaskState.RUNNING)
    runner = _DurableRegistryRunner(registry)
    handler = make_send_message_handler([runner], caller=_root_caller)
    completion_started = asyncio.Event()
    release_completion = asyncio.Event()

    async def _complete() -> None:
      async with entry.finalization_lock:
        completion_started.set()
        await release_completion.wait()
        registry.transition(entry.task_id, TaskState.COMPLETED)

    completion = asyncio.create_task(_complete())
    await completion_started.wait()
    send = asyncio.create_task(handler({
      "to": entry.task_id,
      "message": "Too late",
      "message_id": "complete-first",
    }))
    await asyncio.sleep(0)
    assert not send.done()
    release_completion.set()
    await completion
    result, error = await send
    assert result is None
    assert error is not None and error["code"] == "already_completed"
    assert runner.events == []

  async def _case() -> None:
    await _send_first()
    await _completion_first()

  _run(_case())


def test_task_registry_rehydrates_acceptance_for_terminal_replay() -> None:
  correlation = {
    "owner_runner_id": "runner-parent",
    "owner_role": "writer",
    "sub_agent_id": "sub3:sess-parent",
    "parent_turn_id": "turn-3",
    "call_index": 3,
    "capability_bind": _bind_receipt(),
  }
  registry = _root_registry()
  registry.load_from_events([
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "task_id": "bg_3",
      "task_type": "background_agent",
      "started_at": 1.0,
      **correlation,
      "metadata": {**correlation, "task_type": "background"},
      "_durable_seq": 1,
    },
    {
      "type": "parent_message_sent",
      "task_id": "bg_3",
      **correlation,
      "task_type": "background",
      "message_id": "toolu-replay",
      "message": "Use amended filing",
      "sent_at": 2.0,
      "_durable_seq": 2,
    },
    {
      "type": "task_completed",
      "task_id": "bg_3",
      **correlation,
      "task_type": "background",
      "final_state": "completed",
      "completed_at": 3.0,
      "result": {"response": "done"},
      "error": None,
      "_durable_seq": 3,
    },
  ])
  entry = registry.get("bg_3")
  assert entry is not None
  assert entry.delivered_messages == {"toolu-replay"}
  assert entry.accepted_parent_messages["toolu-replay"] == ParentMessage(
    message_id="toolu-replay",
    text="Use amended filing",
    sent_at=2.0,
    task_id="bg_3",
    sent_seq=2,
  )
  handler = make_send_message_handler([_RegistryRunner(registry)], caller=_root_caller)
  result, error = _run(handler({
    "to": "bg_3",
    "message": "Use amended filing",
    "message_id": "toolu-replay",
  }))
  assert error is None
  assert result is not None and result["status"] == "accepted"
  conflict, conflict_error = _run(handler({
    "to": "bg_3",
    "message": "Ignore amended filing",
    "message_id": "toolu-replay",
  }))
  assert conflict is None
  assert conflict_error is not None
  assert conflict_error["code"] == "message_id_conflict"


def test_task_registry_rejects_duplicate_durable_message_identity() -> None:
  correlation = {
    "owner_runner_id": "runner-parent",
    "owner_role": "writer",
    "sub_agent_id": "sub3:sess-parent",
    "parent_turn_id": "turn-3",
    "call_index": 3,
    "capability_bind": _bind_receipt(),
  }
  registry = _root_registry()
  events = [
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "task_id": "bg_3",
      "task_type": "background_agent",
      "started_at": 1.0,
      **correlation,
      "metadata": {**correlation, "task_type": "background"},
      "_durable_seq": 1,
    },
    {
      "type": "parent_message_sent",
      "task_id": "bg_3",
      **correlation,
      "task_type": "background",
      "message_id": "duplicate",
      "message": "one",
      "sent_at": 2.0,
      "_durable_seq": 2,
    },
    {
      "type": "parent_message_sent",
      "task_id": "bg_3",
      **correlation,
      "task_type": "background",
      "message_id": "duplicate",
      "message": "one",
      "sent_at": 2.1,
      "_durable_seq": 3,
    },
  ]
  with pytest.raises(ValueError, match="duplicate durable parent-message"):
    registry.load_from_events(events)


@pytest.mark.parametrize(
  "tamper",
  [
    "sub_agent_id",
    "owner_runner_id",
    "task_type",
  ],
)
def test_task_registry_rejects_parent_message_authority_mismatch(
  tamper: str,
) -> None:
  correlation = {
    "owner_runner_id": "runner-parent",
    "owner_role": "writer",
    "sub_agent_id": "sub3:sess-parent",
    "parent_turn_id": "turn-3",
    "call_index": 3,
    "capability_bind": _bind_receipt(),
  }
  sent = {
    "type": "parent_message_sent",
    "task_id": "bg_3",
    **correlation,
    "task_type": "background",
    "message_id": "toolu-forged",
    "message": "Use amendment",
    "sent_at": 2.0,
    "_durable_seq": 2,
  }
  if tamper == "sub_agent_id":
    sent["sub_agent_id"] = "sub-forged:sess-parent"
  elif tamper == "owner_runner_id":
    sent["owner_runner_id"] = "runner-forged"
  else:
    sent["task_type"] = "workflow_node"
  events = [
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "task_id": "bg_3",
      "task_type": "background_agent",
      "started_at": 1.0,
      **correlation,
      "metadata": {**correlation, "task_type": "background"},
      "_durable_seq": 1,
    },
    sent,
  ]

  with pytest.raises(ValueError, match="invalid durable parent-message"):
    _root_registry().load_from_events(events)


@pytest.mark.parametrize(
  "tamper",
  ["sent_before_registration", "duplicate_registration", "duplicate_completion"],
)
def test_task_registry_rejects_ambiguous_parent_message_lifecycle(
  tamper: str,
) -> None:
  correlation = {
    "owner_runner_id": "runner-parent",
    "owner_role": "writer",
    "sub_agent_id": "sub3:sess-parent",
    "parent_turn_id": "turn-3",
    "call_index": 3,
    "capability_bind": _bind_receipt(),
  }
  registration = {
    "type": "task_registered",
    "event_schema_version": 2,
    "task_id": "bg_3",
    "task_type": "background_agent",
    "started_at": 1.0,
    **correlation,
    "metadata": {**correlation, "task_type": "background"},
    "_durable_seq": 1,
  }
  sent = {
    "type": "parent_message_sent",
    "task_id": "bg_3",
    **correlation,
    "task_type": "background",
    "message_id": "toolu-window",
    "message": "Use amendment",
    "sent_at": 2.0,
    "_durable_seq": 2,
  }
  completion = {
    "type": "task_completed",
    "task_id": "bg_3",
    **correlation,
    "task_type": "background",
    "final_state": "completed",
    "completed_at": 3.0,
    "result": {"response": "done"},
    "error": None,
    "_durable_seq": 3,
  }
  if tamper == "sent_before_registration":
    sent["_durable_seq"] = 1
    registration["_durable_seq"] = 2
    events = [sent, registration]
  elif tamper == "duplicate_registration":
    duplicate = {**registration, "_durable_seq": 2}
    sent["_durable_seq"] = 3
    events = [registration, duplicate, sent]
  else:
    duplicate = {**completion, "_durable_seq": 4}
    events = [registration, sent, completion, duplicate]

  with pytest.raises(ValueError, match="ambiguous|invalid"):
    _root_registry().load_from_events(events)


def test_task_registry_rejects_parent_message_after_completion() -> None:
  correlation = {
    "owner_runner_id": "runner-parent",
    "owner_role": "writer",
    "sub_agent_id": "sub3:sess-parent",
    "parent_turn_id": "turn-3",
    "call_index": 3,
    "capability_bind": _bind_receipt(),
  }
  events = [
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "task_id": "bg_3",
      "task_type": "background_agent",
      "started_at": 1.0,
      **correlation,
      "metadata": {**correlation, "task_type": "background"},
      "_durable_seq": 1,
    },
    {
      "type": "task_completed",
      "task_id": "bg_3",
      **correlation,
      "task_type": "background",
      "final_state": "completed",
      "completed_at": 2.0,
      "result": {"response": "done"},
      "error": None,
      "_durable_seq": 2,
    },
    {
      "type": "parent_message_sent",
      "task_id": "bg_3",
      **correlation,
      "task_type": "background",
      "message_id": "toolu-too-late",
      "message": "Post completion",
      "sent_at": 3.0,
      "_durable_seq": 3,
    },
  ]

  with pytest.raises(ValueError, match="invalid durable parent-message"):
    _root_registry().load_from_events(events)


def test_make_send_message_handler_updates_delivered_messages_after_successful_put() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)], caller=_root_caller)

  result, error = _run(handler({"to": entry.task_id, "message": "Track me", "message_id": "msg-new"}))

  assert error is None
  assert result == {
    "status": "queued",
    "task_id": entry.task_id,
    "message_id": "msg-new",
  }
  assert entry.delivered_messages == {"msg-new"}


def test_make_send_message_handler_replays_exact_id_and_rejects_content_conflict() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)], caller=_root_caller)
  request = {
    "to": entry.task_id,
    "message": "Use the amended filing",
    "message_id": "msg-retry",
  }

  first, first_error = _run(handler(request))
  replay, replay_error = _run(handler(request))
  conflict, conflict_error = _run(handler({
    **request,
    "message": "Use a different filing",
  }))

  assert first_error is None
  assert replay_error is None
  assert first == replay == {
    "status": "queued",
    "task_id": entry.task_id,
    "message_id": "msg-retry",
  }
  assert entry.message_inbox.qsize() == 1
  assert conflict is None
  assert conflict_error == {
    "code": "message_id_conflict",
    "message": "message_id was already accepted with different content",
  }


def test_make_send_message_handler_preserves_resume_identity_and_content() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  restored = ParentMessage(
    message_id="msg-resumed",
    text="Use the amended filing",
    sent_at=1.0,
    task_id="bg-original",
    sent_seq=42,
  )
  entry.accepted_parent_messages[restored.message_id] = restored
  entry.delivered_messages.add(restored.message_id)
  handler = make_send_message_handler([_RegistryRunner(registry)], caller=_root_caller)

  replay, replay_error = _run(handler({
    "to": entry.task_id,
    "message": restored.text,
    "message_id": restored.message_id,
  }))
  conflict, conflict_error = _run(handler({
    "to": entry.task_id,
    "message": "Use another filing",
    "message_id": restored.message_id,
  }))

  assert replay_error is None
  assert replay == {
    "status": "accepted",
    "task_id": entry.task_id,
    "message_id": restored.message_id,
  }
  assert conflict is None
  assert conflict_error == {
    "code": "message_id_conflict",
    "message": "message_id was already accepted with different content",
  }
  assert entry.message_inbox.empty()


@pytest.mark.parametrize(
  "message_id",
  [123, "", " padded ", "x" * 513, "🧪" * 129],
)
def test_make_send_message_handler_rejects_noncanonical_message_id(
  message_id: object,
) -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  runner = _DurableRegistryRunner(registry)
  handler = make_send_message_handler([runner], caller=_root_caller)

  result, error = _run(handler({
    "to": entry.task_id,
    "message": "Use the amended filing",
    "message_id": message_id,
  }))

  assert result is None
  assert error == {
    "code": "invalid_input",
    "message": "message_id must be bounded canonical non-empty text",
  }
  assert runner.events == []
  assert entry.message_inbox.empty()


def test_make_send_message_handler_does_not_update_delivered_messages_when_put_raises() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  entry.message_inbox = _FailingQueue()
  registry.transition(entry.task_id, TaskState.RUNNING)
  runner = _DurableRegistryRunner(registry)
  handler = make_send_message_handler([runner], caller=_root_caller)

  with pytest.raises(RuntimeError, match="queue closed"):
    _run(handler({"to": entry.task_id, "message": "Track me", "message_id": "msg-failed"}))

  assert len(runner.events) == 1
  assert runner.events[0]["message_id"] == "msg-failed"
  assert "msg-failed" not in entry.delivered_messages


def test_make_send_message_handler_delivers_message_by_agent_name() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)], caller=_root_caller)

  result, error = _run(handler({"to": "writer", "message": "Focus on the risks"}))

  assert error is None
  assert result is not None
  assert result["status"] == "queued"
  assert result["task_id"] == entry.task_id
  assert result["message_id"]
  parent_message = entry.message_inbox.get_nowait()
  assert parent_message.text == "Focus on the risks"


def test_make_send_message_handler_returns_ambiguous_target_for_duplicate_names() -> None:
  registry = _root_registry()
  first = registry.register("background_agent", agent_name="writer")
  second = registry.register("background_agent", agent_name="writer")
  first.started_at = 1.0
  second.started_at = 2.0
  registry.transition(first.task_id, TaskState.RUNNING)
  registry.transition(second.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)], caller=_root_caller)

  result, error = _run(handler({"to": "writer", "message": "Status?"}))

  assert result is None
  assert error == {
    "code": "ambiguous_target",
    "message": "Multiple running agents named 'writer': bg_0, bg_1. Use task_id instead.",
  }


def test_make_send_message_handler_returns_not_on_roster_for_unknown_target() -> None:
  handler = make_send_message_handler([_RegistryRunner(_root_registry())], caller=_root_caller)

  result, error = _run(handler({"to": "missing", "message": "Ping"}))

  assert result is None
  assert error is not None and error["code"] == "not_on_roster"


def test_make_send_message_handler_returns_already_completed_for_terminal_task() -> None:
  registry = _root_registry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.COMPLETED, result={"response": "done"})
  handler = make_send_message_handler([_RegistryRunner(registry)], caller=_root_caller)

  result, error = _run(handler({"to": entry.task_id, "message": "One more thing"}))

  assert result is None
  assert error == {"code": "already_completed", "message": f"Agent {entry.task_id} already finished"}


def test_make_send_message_handler_returns_error_when_runner_not_initialized() -> None:
  handler = make_send_message_handler([None], caller=_root_caller)

  result, error = _run(handler({"to": "bg_0", "message": "Ping"}))

  assert result is None
  assert error == {"code": "internal_error", "message": "Runner not initialized"}


def test_make_send_message_handler_returns_error_when_registry_not_configured() -> None:
  handler = make_send_message_handler([object()], caller=_root_caller)

  result, error = _run(handler({"to": "bg_0", "message": "Ping"}))

  assert result is None
  assert error == {"code": "not_available", "message": "Task registry not configured"}


@pytest.mark.parametrize(
  ("tool_input", "expected_message"),
  [
    ({}, "'to' is required"),
    ({"to": "", "message": "Ping"}, "'to' is required"),
    ({"to": 123, "message": "Ping"}, "'to' is required"),
    ({"to": "bg_0"}, "'message' is required"),
    ({"to": "bg_0", "message": ""}, "'message' is required"),
    ({"to": "bg_0", "message": 123}, "'message' is required"),
  ],
)
def test_make_send_message_handler_validates_required_fields(
  tool_input: dict[str, Any],
  expected_message: str,
) -> None:
  handler = make_send_message_handler([_RegistryRunner(_root_registry())], caller=_root_caller)

  result, error = _run(handler(tool_input))

  assert result is None
  assert error == {"code": "invalid_input", "message": expected_message}


def test_parent_message_dataclass_round_trips_through_queue() -> None:
  inbox: asyncio.Queue[ParentMessage] = asyncio.Queue()
  message = ParentMessage(message_id="msg-1", text="Check appendix", sent_at=123.0)

  _run(inbox.put(message))

  assert inbox.get_nowait() == message


def test_run_drains_message_inbox_and_injects_parent_messages_between_turns() -> None:
  async def _case() -> None:
    inbox: asyncio.Queue[ParentMessage] = asyncio.Queue()
    runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-send-message",
      capability_execution=stub_runner_capability_execution(
        provider=_StubProvider(),
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      get_tool_definitions=lambda: [],
      message_inbox=inbox,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    seen_messages: list[list[dict[str, Any]]] = []

    async def _fake_stream_turn(**kwargs: Any):
      seen_messages.append(list(kwargs["current_messages"]))
      if len(seen_messages) == 1:
        await inbox.put(ParentMessage(message_id="msg-1", text="Focus on regressions", sent_at=1.0))
        await inbox.put(ParentMessage(message_id="msg-2", text="Skip polish", sent_at=2.0))
        return object(), StreamTurnResult(
          full_text="working",
          stop_reason="pause_turn",
          content_blocks=[{"type": "text", "text": "working"}],
        )
      return object(), StreamTurnResult(
        full_text="done",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "done"}],
      )

    runner._stream_turn = _fake_stream_turn

    await runner.run(
      messages=[{"role": "user", "content": "Start"}],
      system_prompt="You are helpful.",
    )

    assert len(seen_messages) == 2
    assert seen_messages[1][-1] == {
      "role": "user",
      "content": (
        "Operator update for this task:\n"
        "- id=msg-1: Focus on regressions\n"
        "- id=msg-2: Skip polish"
      ),
    }
    assert seen_messages[1][1]["role"] == "assistant"
    assert inbox.empty()

  _run(_case())


def test_send_message_durable_event_exists_when_queue_put_fails(tmp_path: Path) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "send-message-crash-window.jsonl")
    registry = _root_registry()
    entry = registry.register("background_agent", agent_name="writer")
    entry.message_inbox = _FailingQueue()
    registry.transition(entry.task_id, TaskState.RUNNING)
    runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-send-message",
      capability_execution=stub_runner_capability_execution(
        provider=_StubProvider(),
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      agent_session_log=log,
      task_registry=registry,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    runner._runner_id = "runner-test"
    handler = make_send_message_handler([runner], caller=_root_caller)

    with pytest.raises(RuntimeError, match="queue closed"):
      await handler({"to": entry.task_id, "message": "Persist before delivery", "message_id": "msg-crash"})

    assert entry.message_inbox.empty()
    assert "msg-crash" not in entry.delivered_messages
    entries, _ = await log.query(event_types={"parent_message_sent"}, order="asc")
    assert len(entries) == 1
    assert entries[0].event["message_id"] == "msg-crash"
    assert entries[0].event["message"] == "Persist before delivery"

  _run(_case())


def test_send_message_persists_event_and_child_sees_parent_message_envelope(tmp_path: Path) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "send-message-to-child.jsonl")
    registry = _root_registry()
    entry = registry.register("background_agent", agent_name="writer")
    entry.metadata["sub_agent_id"] = "sub0:sess-send-message"
    registry.transition(entry.task_id, TaskState.RUNNING)
    parent_runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-send-message",
      capability_execution=stub_runner_capability_execution(
        provider=_StubProvider(),
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      agent_session_log=log,
      task_registry=registry,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    parent_runner._runner_id = "runner-parent"
    handler = make_send_message_handler([parent_runner], caller=_root_caller)

    result, error = await handler({"to": entry.task_id, "message": "Focus on regressions", "message_id": "msg-live"})

    assert error is None
    assert result == {
      "status": "accepted",
      "task_id": entry.task_id,
      "message_id": "msg-live",
    }
    entries, _ = await log.query(event_types={"parent_message_sent"}, order="asc")
    assert len(entries) == 1
    assert entries[0].event["message_id"] == "msg-live"

    child_runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-child",
      capability_execution=stub_runner_capability_execution(
        provider=_StubProvider(),
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      agent_session_log=log,
      get_tool_definitions=lambda: [],
      message_inbox=entry.message_inbox,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    child_runner._role = "sub_agent"
    child_runner._sub_agent_id = "sub0:sess-send-message"
    seen_messages: list[list[dict[str, Any]]] = []

    async def _fake_stream_turn(**kwargs: Any):
      seen_messages.append(list(kwargs["current_messages"]))
      if len(seen_messages) == 1:
        return object(), StreamTurnResult(
          full_text="working",
          stop_reason="pause_turn",
          content_blocks=[{"type": "text", "text": "working"}],
        )
      return object(), StreamTurnResult(
        full_text="done",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "done"}],
      )

    child_runner._stream_turn = _fake_stream_turn
    await child_runner.run(messages=[{"role": "user", "content": "Start"}])

    assert seen_messages[0][-1] == {
      "role": "user",
      "content": (
        "Operator update for this task:\n"
        "- id=msg-live: Focus on regressions"
      ),
    }
    consumed_entries, _ = await log.query(
      event_types={"parent_message_consumed"},
      order="asc",
    )
    assert len(consumed_entries) == 1
    consumed = consumed_entries[0]
    assert consumed.event["type"] == "parent_message_consumed"
    assert consumed.event["task_id"] == entry.task_id
    assert consumed.event["message_id"] == "msg-live"
    assert consumed.event["parent_message_seq"] == entries[0].seq
    assert consumed.event["consumer_turn"] == 1
    assert consumed.event["runner_id"]
    assert consumed.event["role"] == "sub_agent"
    assert consumed.event["assistant_message_seq"] < consumed.seq
    assert consumed.event["assistant_message_seq"] > entries[0].seq
    assert "message" not in consumed.event

    await child_runner._materialize_parent_message_consumption_audits()
    replayed_entries, _ = await log.query(
      event_types={"parent_message_consumed"},
      order="asc",
    )
    assert len(replayed_entries) == 1

  _run(_case())


_CITED = "sha256:" + "a" * 64


def _running_child(registry: TaskRegistry, name: str) -> Any:
  entry = registry.register("background_agent", agent_name=name)
  registry.transition(entry.task_id, TaskState.RUNNING)
  return entry


def _child_handler(registry: TaskRegistry, job_id: str, **context: Any):
  return make_send_message_handler(
    [_RegistryRunner(registry)],
    caller=lambda: _job_context(job_id, **context),
  )


def test_sibling_message_is_delivered_as_the_senders_evidence() -> None:
  registry = _root_registry()
  sender = _running_child(registry, "reader")
  recipient = _running_child(registry, "writer")

  result, error = _run(_child_handler(registry, sender.task_id)({
    "to": recipient.task_id,
    "message": "Revenue mix is 93/7 in FY2026",
  }))

  assert error is None
  assert result is not None and result["task_id"] == recipient.task_id
  delivered = recipient.message_inbox.get_nowait()
  assert delivered.sender_job_id == sender.task_id
  assert delivered.evidence is True
  assert sender.message_inbox.empty()


def test_parent_send_to_its_child_is_a_steer_not_evidence() -> None:
  registry = _root_registry()
  child = _running_child(registry, "writer")

  _, error = _run(make_send_message_handler(
    [_RegistryRunner(registry)],
    caller=_root_caller,
  )({"to": child.task_id, "message": "Stop after section 2"}))

  assert error is None
  assert child.message_inbox.get_nowait().evidence is False


def test_child_message_reaches_the_root_jobs_inbox() -> None:
  inbox: asyncio.Queue[ParentMessage] = asyncio.Queue(maxsize=64)
  registry = _root_registry(inbox=inbox)
  child = _running_child(registry, "writer")

  result, error = _run(_child_handler(registry, child.task_id)({
    "to": _ROOT_JOB,
    "message": "Found a restatement in Q3",
  }))

  assert error is None
  assert result is not None and result["task_id"] == _ROOT_JOB
  delivered = inbox.get_nowait()
  assert (delivered.sender_job_id, delivered.evidence) == (child.task_id, True)


def test_root_mailbox_full_is_reported_without_dropping() -> None:
  inbox: asyncio.Queue[ParentMessage] = asyncio.Queue(maxsize=1)
  registry = _root_registry(inbox=inbox)
  child = _running_child(registry, "writer")
  handler = _child_handler(registry, child.task_id)

  _run(handler({"to": _ROOT_JOB, "message": "first"}))
  result, error = _run(handler({"to": _ROOT_JOB, "message": "second"}))

  assert result is None and error is not None
  assert error["code"] == "mailbox_full"
  assert inbox.get_nowait().text == "first"


def test_linked_child_registry_puts_grandchildren_on_the_tree_roster() -> None:
  root = _root_registry()
  lead = _running_child(root, "lead")
  worker_sibling = _running_child(root, "other")
  lead_registry = TaskRegistry(
    id_prefix="lead",
    owner=RegistryOwner(
      job_id=lead.task_id,
      tenant=lambda: _TENANT,
      inbox=lead.message_inbox,
      parent=root,
    ),
  )
  worker = _running_child(lead_registry, "worker")

  # The worker reaches its parent's sibling through the linked registries.
  _, error = _run(_child_handler(lead_registry, worker.task_id)({
    "to": worker_sibling.task_id,
    "message": "question",
  }))
  assert error is None
  assert worker_sibling.message_inbox.get_nowait().sender_job_id == worker.task_id
  # The root addresses the grandchild without sharing a task dict.
  _, error = _run(make_send_message_handler(
    [_RegistryRunner(root)],
    caller=_root_caller,
  )({"to": worker.task_id, "message": "go on"}))
  assert error is None
  assert worker.task_id not in {entry.task_id for entry in root.list_tasks()}
  assert worker.message_inbox.get_nowait().evidence is True


def test_job_in_another_tree_is_not_on_roster() -> None:
  mine = _root_registry()
  sender = _running_child(mine, "reader")
  other = TaskRegistry(
    id_prefix="other",
    owner=RegistryOwner(job_id="sess-other", tenant=lambda: _TENANT),
  )
  stranger = _running_child(other, "writer")

  result, error = _run(_child_handler(mine, sender.task_id)({
    "to": stranger.task_id,
    "message": "hello",
  }))

  assert result is None and error is not None
  assert error["code"] == "not_on_roster"
  assert stranger.message_inbox.empty()


def test_sender_of_another_tenant_is_refused_by_admission() -> None:
  registry = _root_registry()
  recipient = _running_child(registry, "writer")
  intruder = AdmittedTenant(tenant_id=None, storage_id="mallory", role="owner")

  result, error = _run(_child_handler(registry, "bg_x", tenant=intruder)({
    "to": recipient.task_id,
    "message": "hello",
  }))

  assert result is None and error is not None
  assert error["code"] == "tenant_mismatch"
  assert recipient.message_inbox.empty()


def test_sender_cannot_cite_a_handle_it_does_not_hold() -> None:
  registry = _root_registry()
  sender = _running_child(registry, "reader")
  recipient = _running_child(registry, "writer")

  result, error = _run(_child_handler(registry, sender.task_id)({
    "to": recipient.task_id,
    "message": "see this",
    "handle_refs": [_CITED],
  }))

  assert result is None and error is not None
  assert error["code"] == "handle_not_granted"
  assert recipient.message_inbox.empty()


def test_a_cited_handle_is_delivered_but_grants_the_recipient_nothing() -> None:
  from agent_gateway.admission import HandleOwners, HandleRequest, Refusal, admit

  registry = _root_registry()
  sender = _running_child(registry, "reader")
  recipient = _running_child(registry, "writer")

  _, error = _run(_child_handler(
    registry,
    sender.task_id,
    handle_grants=frozenset({_CITED}),
  )({
    "to": recipient.task_id,
    "message": "the DCF is here",
    "handle_refs": [_CITED],
  }))
  assert error is None
  delivered = recipient.message_inbox.get_nowait()
  assert delivered.handle_refs == (_CITED,)

  read = admit(
    HandleRequest(handle=_CITED, reader_job_id=recipient.task_id),
    _job_context(recipient.task_id),
    owners=HandleOwners(outputs=frozenset()),
  )
  assert isinstance(read, Refusal)
  assert read.reason == "handle_not_granted"


def test_evidence_messages_render_with_sender_provenance() -> None:
  from agent_gateway.task_registry import format_parent_messages_for_model

  rendered = format_parent_messages_for_model([
    ParentMessage(message_id="m1", text="Stop early", sent_at=1.0),
    ParentMessage(
      message_id="m2",
      text="Margin is 31%",
      sent_at=2.0,
      sender_job_id="bg_1",
      handle_refs=(_CITED,),
      evidence=True,
    ),
  ])

  assert "Operator update for this task:\n- id=m1: Stop early" in rendered
  assert f"- id=m2 from=bg_1 handles={_CITED}: Margin is 31%" in rendered
  assert "not instructions" in rendered
