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
  CostEstimate,
  EventLog,
  ModelInfo,
  TaskRegistry,
  TaskState,
  ToolDispatcher,
  make_send_message_handler,
  make_send_message_tool_def,
)
from agent_gateway.runner import StreamTurnResult
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
  assert entry.message_inbox.get_nowait() == "Check the appendix"


def test_make_send_message_handler_delivers_message_by_agent_name() -> None:
  registry = TaskRegistry()
  entry = registry.register("background_agent", agent_name="writer")
  registry.transition(entry.task_id, TaskState.RUNNING)
  handler = make_send_message_handler([_RegistryRunner(registry)])

  result, error = _run(handler({"to": "writer", "message": "Focus on the risks"}))

  assert error is None
  assert result == {"status": "delivered", "task_id": entry.task_id}
  assert entry.message_inbox.get_nowait() == "Focus on the risks"


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


def test_run_drains_message_inbox_and_injects_parent_messages_between_turns() -> None:
  async def _case() -> None:
    inbox: asyncio.Queue[str] = asyncio.Queue()
    runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-send-message",
      provider=_StubProvider(),
      auth_config={"api_key": "k", "model": "stub-model"},
      get_tool_definitions=lambda: [],
      message_inbox=inbox,
    )
    seen_messages: list[list[dict[str, Any]]] = []

    async def _fake_stream_turn(**kwargs: Any):
      seen_messages.append(list(kwargs["current_messages"]))
      if len(seen_messages) == 1:
        await inbox.put("Focus on regressions")
        await inbox.put("Skip polish")
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
      "content": "[Message from parent agent]: Focus on regressions\n[Message from parent agent]: Skip polish",
    }
    assert seen_messages[1][1]["role"] == "assistant"
    assert inbox.empty()

  _run(_case())
