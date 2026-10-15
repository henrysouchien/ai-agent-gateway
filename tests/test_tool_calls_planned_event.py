"""`tool_calls_planned`: one event per tool-using model turn, before execution."""
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND

import asyncio
from typing import Any

from agent_gateway import (
  AgentRunner,
  EventLog,
  McpClientManager,
  ModelInfo,
  ModelProvider,
  ToolDispatcher,
)
from agent_gateway.event_adapter import adapt_event
from agent_gateway.providers import StreamEvent
from agent_gateway.tool_dispatcher_helpers import ToolResult
from agent_gateway.tool_display import resolve_display
from gateway_test_support.capability_execution_test_support import (
  stub_bound_capability_execution,
)


class _NullMcpClient(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  def is_mcp_tool(self, name: str) -> bool:
    _ = name
    return False

  async def call_tool(
    self,
    name: str,
    tool_input: object,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    _ = (
      tool_input,
      meta,
      abort_event,
      gateway_session,
      allow_uncertain_replay,
      trusted_dispatch_scope,
    )
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []

  def get_server_for_tool(self, name: str) -> str | None:
    _ = name
    return None


class _ScriptedProvider(ModelProvider):
  name = "stub"

  def __init__(self, turns: list[list[StreamEvent]]) -> None:
    self._turns = list(turns)

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
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
    return {
      "model": model,
      "messages": messages,
      "system_prompt": system_prompt,
      "tools": tools,
      "max_tokens": max_tokens,
      **kwargs,
    }

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    events = self._turns.pop(0) if self._turns else _text_turn("done")
    for event in events:
      yield event


def _tool_use_turn(tool_uses: list[tuple[str, str, dict[str, Any]]]) -> list[StreamEvent]:
  events: list[StreamEvent] = []
  for tool_id, tool_name, tool_input in tool_uses:
    events.extend([
      StreamEvent(type="tool_use_start", tool_id=tool_id, tool_name=tool_name),
      StreamEvent(
        type="tool_use_end",
        tool_id=tool_id,
        tool_name=tool_name,
        tool_input=tool_input,
        raw_block={"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input},
      ),
    ])
  events.append(StreamEvent(type="message_end", stop_reason="tool_use"))
  return events


def _text_turn(text: str) -> list[StreamEvent]:
  return [
    StreamEvent(type="text_delta", text=text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]


async def _ok_handler(_tool_input: dict[str, Any], *, call_index: int = 0, tool_ctx: Any = None):
  _ = call_index, tool_ctx
  return {"status": "success"}, None


def _tool_def(name: str) -> dict[str, Any]:
  return {"name": name, "description": "", "input_schema": {"type": "object", "properties": {}}}


def _run_turns(turns: list[list[StreamEvent]]) -> list[dict[str, Any]]:
  event_log = EventLog()
  provider = _ScriptedProvider(turns)
  _dispatcher_execution_145 = stub_bound_capability_execution(
        provider=provider,
        model="stub-model",
        effort="none",
        auth_config={"api_key": "k"},
      )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      capability_bind=_dispatcher_execution_145.bind,
      mcp_client=_NullMcpClient(),
      local_tool_handlers={"lookup": _ok_handler, "get_quote": _ok_handler},
      event_log=event_log,
      session_id="sess-planned",
      role="owner",
    ),
    session_id="sess-planned",
    capability_execution=_dispatcher_execution_145,
    get_tool_definitions=lambda: [_tool_def("lookup"), _tool_def("get_quote")],
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}], system_prompt="x"))
  return [entry.event for entry in event_log.entries]


def test_tool_using_turn_plans_every_call_before_the_first_starts() -> None:
  events = _run_turns([
    _tool_use_turn([
      ("toolu_1", "get_quote", {"ticker": "NVDA"}),
      ("toolu_2", "lookup", {"query": "NVDA news"}),
    ]),
    _text_turn("NVDA is up."),
  ])
  types = [event["type"] for event in events]

  planned = [event for event in events if event["type"] == "tool_calls_planned"]
  assert len(planned) == 1
  planned_index = types.index("tool_calls_planned")
  assert types.index("turn_complete") < planned_index
  assert planned_index < types.index("tool_call_start")

  event = planned[0]
  assert event["turn"] == 1
  starts = {start["tool_call_id"]: start for start in events if start["type"] == "tool_call_start"}
  assert [call["tool_call_id"] for call in event["calls"]] == ["toolu_1", "toolu_2"]
  for call in event["calls"]:
    start = starts[call["tool_call_id"]]
    assert call["tool_name"] == start["tool_name"]
    assert call["tool_input"] == start["tool_input"]
    assert call["display"] == resolve_display(call["tool_name"], call["tool_input"])
    assert call["display"] == start.get("display")

  assert adapt_event(event, 1) == {
    "type": "tool_calls_planned",
    "turn": 1,
    "calls": event["calls"],
  }


def test_text_only_turn_plans_nothing() -> None:
  events = _run_turns([_text_turn("Nothing to look up.")])

  assert "turn_complete" in [event["type"] for event in events]
  assert not any(event["type"] == "tool_calls_planned" for event in events)
