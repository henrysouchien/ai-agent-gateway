import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (
  AgentRunner,
  AgentSessionLog,
  CostEstimate,
  EventLog,
  ModelInfo,
  ParentMessage,
  TaskRegistry,
  TaskState,
  ToolDispatcher,
  make_send_message_handler,
  make_send_message_tool_def,
)
from agent_gateway.runner import StreamTurnResult
import agent_gateway.sub_agent as sub_agent_module
from agent_gateway.sub_agent import _DEFAULT_EXCLUDED_TOOLS


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


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

  async def _append_durable_event(self, event: dict[str, Any]) -> None:
    self.order.append("append")
    self.events.append(dict(event))


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
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)])

  result, error = _run(handler({"to": entry.task_id, "message": "Check the appendix"}))

  assert error is None
  assert result == {"status": "delivered", "task_id": entry.task_id}
  parent_message = entry.message_inbox.get_nowait()
  assert parent_message.text == "Check the appendix"
  assert parent_message.message_id in entry.delivered_messages


def test_make_send_message_handler_preserves_sub_agent_export() -> None:
  assert make_send_message_handler is sub_agent_module.make_send_message_handler
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  handler = sub_agent_module.make_send_message_handler([_RegistryRunner(registry)])

  result, error = _run(handler({"to": entry.task_id, "message": "Use direct import"}))

  assert error is None
  assert result == {"status": "delivered", "task_id": entry.task_id}
  assert entry.message_inbox.get_nowait().text == "Use direct import"


def test_make_send_message_handler_emits_parent_message_sent_with_correlation() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  entry.provider_name = "anthropic"
  entry.model = "claude-sonnet-4-6"
  entry.metadata.update(
    {
      "owner_runner_id": "runner_old",
      "owner_role": "writer",
      "sub_agent_id": "sub0:sess-parent",
      "parent_turn_id": "turn-1",
      "call_index": 0,
      "task_type": "background",
      "provider_name": "anthropic",
      "model": "claude-sonnet-4-6",
    }
  )
  registry.transition(entry.task_id, TaskState.RUNNING)
  runner = _DurableRegistryRunner(registry)
  handler = make_send_message_handler([runner])

  result, error = _run(handler({"to": entry.task_id, "message": "Check the appendix"}))

  assert error is None
  assert result == {"status": "delivered", "task_id": entry.task_id}
  assert len(runner.events) == 1
  event = runner.events[0]
  assert event["type"] == "parent_message_sent"
  assert event["task_id"] == entry.task_id
  assert event["owner_runner_id"] == "runner_old"
  assert event["owner_role"] == "writer"
  assert event["sub_agent_id"] == "sub0:sess-parent"
  assert event["parent_turn_id"] == "turn-1"
  assert event["call_index"] == 0
  assert event["task_type"] == "background"
  assert event["provider_name"] == "anthropic"
  assert event["model"] == "claude-sonnet-4-6"
  assert event["message"] == "Check the appendix"
  assert event["message_id"]
  parent_message = entry.message_inbox.get_nowait()
  assert parent_message.message_id == event["message_id"]
  assert parent_message.sent_at == event["sent_at"]


def test_make_send_message_handler_emits_before_deliver() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  runner = _DurableRegistryRunner(registry)
  entry.message_inbox = _RecordingQueue(runner.order)
  handler = make_send_message_handler([runner])

  result, error = _run(handler({"to": entry.task_id, "message": "Check ordering"}))

  assert error is None
  assert result == {"status": "delivered", "task_id": entry.task_id}
  assert runner.order == ["append", "put"]


def test_make_send_message_handler_is_idempotent_for_delivered_message_id() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  entry.delivered_messages.add("msg-existing")
  registry.transition(entry.task_id, TaskState.RUNNING)
  runner = _DurableRegistryRunner(registry)
  handler = make_send_message_handler([runner])

  result, error = _run(handler({"to": entry.task_id, "message": "Replay", "message_id": "msg-existing"}))

  assert error is None
  assert result == {"status": "delivered", "task_id": entry.task_id}
  assert runner.events == []
  assert entry.message_inbox.empty()


def test_make_send_message_handler_updates_delivered_messages_after_successful_put() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)])

  result, error = _run(handler({"to": entry.task_id, "message": "Track me", "message_id": "msg-new"}))

  assert error is None
  assert result == {"status": "delivered", "task_id": entry.task_id}
  assert entry.delivered_messages == {"msg-new"}


def test_make_send_message_handler_does_not_update_delivered_messages_when_put_raises() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  entry.message_inbox = _FailingQueue()
  registry.transition(entry.task_id, TaskState.RUNNING)
  runner = _DurableRegistryRunner(registry)
  handler = make_send_message_handler([runner])

  with pytest.raises(RuntimeError, match="queue closed"):
    _run(handler({"to": entry.task_id, "message": "Track me", "message_id": "msg-failed"}))

  assert len(runner.events) == 1
  assert runner.events[0]["message_id"] == "msg-failed"
  assert "msg-failed" not in entry.delivered_messages


def test_make_send_message_handler_delivers_message_by_agent_name() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)])

  result, error = _run(handler({"to": "writer", "message": "Focus on the risks"}))

  assert error is None
  assert result == {"status": "delivered", "task_id": entry.task_id}
  parent_message = entry.message_inbox.get_nowait()
  assert parent_message.text == "Focus on the risks"


def test_make_send_message_handler_returns_ambiguous_target_for_duplicate_names() -> None:
  registry = TaskRegistry()
  first = registry.register("background_agent", agent_name="writer")
  second = registry.register("background_agent", agent_name="writer")
  first.started_at = 1.0
  second.started_at = 2.0
  registry.transition(first.task_id, TaskState.RUNNING)
  registry.transition(second.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)])

  result, error = _run(handler({"to": "writer", "message": "Status?"}))

  assert result is None
  assert error == {
    "code": "ambiguous_target",
    "message": "Multiple running agents named 'writer': bg_0, bg_1. Use task_id instead.",
  }


def test_make_send_message_handler_returns_not_found_for_unknown_target() -> None:
  handler = make_send_message_handler([_RegistryRunner(TaskRegistry())])

  result, error = _run(handler({"to": "missing", "message": "Ping"}))

  assert result is None
  assert error == {"code": "not_found", "message": "No running agent: missing"}


def test_make_send_message_handler_returns_already_completed_for_terminal_task() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.COMPLETED, result={"response": "done"})
  handler = make_send_message_handler([_RegistryRunner(registry)])

  result, error = _run(handler({"to": entry.task_id, "message": "One more thing"}))

  assert result is None
  assert error == {"code": "already_completed", "message": f"Agent {entry.task_id} already finished"}


def test_make_send_message_handler_returns_error_when_runner_not_initialized() -> None:
  handler = make_send_message_handler([None])

  result, error = _run(handler({"to": "bg_0", "message": "Ping"}))

  assert result is None
  assert error == {"code": "internal_error", "message": "Runner not initialized"}


def test_make_send_message_handler_returns_error_when_registry_not_configured() -> None:
  handler = make_send_message_handler([object()])

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
  handler = make_send_message_handler([_RegistryRunner(TaskRegistry())])

  result, error = _run(handler(tool_input))

  assert result is None
  assert error == {"code": "invalid_input", "message": expected_message}


def test_make_send_message_tool_def_has_expected_schema() -> None:
  tool_def = make_send_message_tool_def()

  assert tool_def["name"] == "send_message"
  assert tool_def["input_schema"]["required"] == ["to", "message"]
  assert set(tool_def["input_schema"]["properties"]) == {"to", "message"}
  assert tool_def["input_schema"]["properties"]["to"]["description"] == "Task ID (e.g. bg_0) or agent name."
  assert tool_def["input_schema"]["properties"]["message"]["description"] == "Message content to deliver to the agent."


def test_send_message_is_in_default_excluded_tools() -> None:
  assert _DEFAULT_EXCLUDED_TOOLS == frozenset({"run_agent", "get_background_result", "send_message"})


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
      provider=_StubProvider(),
      auth_config={"api_key": "k", "model": "stub-model"},
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

    runner._stream_turn = _fake_stream_turn  # type: ignore[method-assign]

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
    registry = TaskRegistry()
    entry = registry.register("background_agent", agent_name="writer")
    entry.message_inbox = _FailingQueue()
    registry.transition(entry.task_id, TaskState.RUNNING)
    runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-send-message",
      provider=_StubProvider(),
      auth_config={"api_key": "k", "model": "stub-model"},
      agent_session_log=log,
      task_registry=registry,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    runner._runner_id = "runner-test"
    handler = make_send_message_handler([runner])

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
    registry = TaskRegistry()
    entry = registry.register("background_agent", agent_name="writer")
    registry.transition(entry.task_id, TaskState.RUNNING)
    parent_runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-send-message",
      provider=_StubProvider(),
      auth_config={"api_key": "k", "model": "stub-model"},
      agent_session_log=log,
      task_registry=registry,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    parent_runner._runner_id = "runner-parent"
    handler = make_send_message_handler([parent_runner])

    result, error = await handler({"to": entry.task_id, "message": "Focus on regressions", "message_id": "msg-live"})

    assert error is None
    assert result == {"status": "delivered", "task_id": entry.task_id}
    entries, _ = await log.query(event_types={"parent_message_sent"}, order="asc")
    assert len(entries) == 1
    assert entries[0].event["message_id"] == "msg-live"

    child_runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-child",
      provider=_StubProvider(),
      auth_config={"api_key": "k", "model": "stub-model"},
      get_tool_definitions=lambda: [],
      message_inbox=entry.message_inbox,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
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

    child_runner._stream_turn = _fake_stream_turn  # type: ignore[method-assign]
    await child_runner.run(messages=[{"role": "user", "content": "Start"}])

    assert seen_messages[0][-1] == {
      "role": "user",
      "content": (
        "Operator update for this task:\n"
        "- id=msg-live: Focus on regressions"
      ),
    }

  _run(_case())
