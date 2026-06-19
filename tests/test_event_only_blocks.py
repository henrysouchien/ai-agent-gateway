import asyncio
import json
from typing import Any

from agent_gateway import AgentRunner, AgentSDKConfig, AgentSDKRunner, EventLog, ModelInfo, ModelProvider, ToolDispatcher
from agent_gateway.providers import StreamEvent
from agent_gateway.transcript import _tool_result_blocks_from_event


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []

  def get_server_for_tool(self, _name: str) -> str | None:
    return None


class _RecordingProvider(ModelProvider):
  name = "stub"

  def __init__(self, turns: list[list[StreamEvent]]) -> None:
    self._turns = list(turns)
    self._stream_index = 0
    self.params_history: list[dict[str, Any]] = []

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
    params = {
      "model": model,
      "messages": messages,
      "system_prompt": system_prompt,
      "tools": tools,
      "max_tokens": max_tokens,
      **kwargs,
    }
    self.params_history.append(params)
    return params

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if self._stream_index >= len(self._turns):
      events = [StreamEvent(type="message_end", stop_reason="end_turn")]
    else:
      events = self._turns[self._stream_index]
      self._stream_index += 1
    for event in events:
      yield event


def _run(coro):
  return asyncio.run(coro)


def _tool_def(name: str) -> dict[str, Any]:
  return {"name": name, "description": "", "input_schema": {"type": "object", "properties": {}}}


def _tool_use_turn(tool_uses: list[tuple[str, str, dict[str, Any]]]) -> list[StreamEvent]:
  events: list[StreamEvent] = []
  for tool_id, tool_name, tool_input in tool_uses:
    events.extend(
      [
        StreamEvent(type="tool_use_start", tool_id=tool_id, tool_name=tool_name),
        StreamEvent(
          type="tool_use_end",
          tool_id=tool_id,
          tool_name=tool_name,
          tool_input=tool_input,
          raw_block={"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input},
        ),
      ]
    )
  events.append(StreamEvent(type="message_end", stop_reason="tool_use"))
  return events


def _end_turn() -> list[StreamEvent]:
  return [StreamEvent(type="message_end", stop_reason="end_turn")]


def _dispatcher(event_log: EventLog, handlers: dict[str, Any]) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers=handlers,
    event_log=event_log,
    session_id="sess-event-only",
  )


async def _ok_handler(_tool_input: dict[str, Any], *, call_index: int = 0, tool_ctx: Any = None):
  _ = call_index, tool_ctx
  return {"status": "success"}, None


async def _mixed_extra_blocks(_ctx: Any) -> list[dict[str, Any]]:
  return [
    {"type": "text", "text": "visible citation map"},
    {"type": "source_envelope", "_event_only": True, "data": "hidden from model"},
  ]


def _model_bound_tool_result_content(provider: _RecordingProvider) -> list[dict[str, Any]]:
  for params in reversed(provider.params_history):
    messages = params["messages"]
    if not messages:
      continue
    content = messages[-1].get("content")
    if isinstance(content, list) and any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
      return content
  raise AssertionError("No model-bound tool result message captured")


def test_event_only_blocks_stay_in_sse_but_not_normal_tool_next_turn() -> None:
  event_log = EventLog()
  provider = _RecordingProvider([
    _tool_use_turn([("tool_1", "lookup", {})]),
    _end_turn(),
  ])
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_dispatcher(event_log, {"lookup": _ok_handler}),
    session_id="sess-event-only",
    provider=provider,
    auth_config={"api_key": "k", "model": "stub-model"},
    get_tool_definitions=lambda: [_tool_def("lookup")],
    on_tool_result=_mixed_extra_blocks,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "go"}], system_prompt="x", max_turns=2))

  complete_events = [entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"]
  assert complete_events
  final_blocks = complete_events[-1]["final_tool_result_blocks"]
  assert [block["type"] for block in final_blocks] == ["tool_result", "text", "source_envelope"]

  model_content = _model_bound_tool_result_content(provider)
  assert [block["type"] for block in model_content] == ["tool_result", "text"]
  assert all(block.get("type") != "source_envelope" for block in model_content)


def test_event_only_blocks_are_filtered_from_batched_run_agent_next_turn() -> None:
  event_log = EventLog()
  provider = _RecordingProvider([
    _tool_use_turn([
      ("tool_1", "run_agent", {"task": "a"}),
      ("tool_2", "run_agent", {"task": "b"}),
    ]),
    _end_turn(),
  ])
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_dispatcher(event_log, {"run_agent": _ok_handler}),
    session_id="sess-event-only",
    provider=provider,
    auth_config={"api_key": "k", "model": "stub-model"},
    get_tool_definitions=lambda: [_tool_def("run_agent")],
    on_tool_result=_mixed_extra_blocks,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "go"}], system_prompt="x", max_turns=2))

  complete_events = [entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"]
  assert len(complete_events) == 2
  assert all(
    any(block.get("type") == "source_envelope" for block in event["final_tool_result_blocks"])
    for event in complete_events
  )

  model_content = _model_bound_tool_result_content(provider)
  # Tool_results must come contiguous-first to satisfy Anthropic's
  # "tool_use immediately followed by tool_result" constraint; extras deferred to end.
  assert [block["type"] for block in model_content] == ["tool_result", "tool_result", "text", "text"]
  assert all(block.get("type") != "source_envelope" for block in model_content)


def test_event_only_blocks_are_filtered_from_mixed_run_agent_and_normal_tools() -> None:
  event_log = EventLog()
  tool_calls: list[tuple[str, int]] = []
  provider = _RecordingProvider([
    _tool_use_turn([
      ("tool_1", "run_agent", {"task": "a"}),
      ("tool_2", "run_agent", {"task": "b"}),
      ("tool_3", "lookup", {}),
    ]),
    _end_turn(),
  ])

  async def run_agent_handler(_tool_input: dict[str, Any], *, call_index: int = 0, tool_ctx: Any = None):
    _ = tool_ctx
    tool_calls.append(("run_agent", call_index))
    return {"status": "success"}, None

  async def lookup_handler(_tool_input: dict[str, Any], *, call_index: int = 0, tool_ctx: Any = None):
    _ = tool_ctx
    tool_calls.append(("lookup", call_index))
    return {"status": "success"}, None

  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_dispatcher(event_log, {"run_agent": run_agent_handler, "lookup": lookup_handler}),
    session_id="sess-event-only",
    provider=provider,
    auth_config={"api_key": "k", "model": "stub-model"},
    get_tool_definitions=lambda: [_tool_def("run_agent"), _tool_def("lookup")],
    on_tool_result=_mixed_extra_blocks,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "go"}], system_prompt="x", max_turns=2))

  complete_events = [entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"]
  assert len(complete_events) == 3
  assert sorted(tool_calls) == [("lookup", 0), ("run_agent", 0), ("run_agent", 1)]
  assert all(
    any(block.get("type") == "source_envelope" for block in event["final_tool_result_blocks"])
    for event in complete_events
  )

  model_content = _model_bound_tool_result_content(provider)
  assert [block["type"] for block in model_content] == [
    "tool_result",
    "tool_result",
    "tool_result",
    "text",
    "text",
    "text",
  ]
  assert [block["tool_use_id"] for block in model_content[:3]] == ["tool_1", "tool_2", "tool_3"]
  assert all(block.get("type") != "source_envelope" for block in model_content)


def test_run_agent_call_index_continues_across_separated_batches() -> None:
  event_log = EventLog()
  tool_calls: list[tuple[str, str, int]] = []
  provider = _RecordingProvider([
    _tool_use_turn([
      ("tool_1", "run_agent", {"task": "a"}),
      ("tool_2", "lookup", {}),
      ("tool_3", "run_agent", {"task": "b"}),
    ]),
    _end_turn(),
  ])

  async def run_agent_handler(tool_input: dict[str, Any], *, call_index: int = 0, tool_ctx: Any = None):
    _ = tool_ctx
    tool_calls.append(("run_agent", str(tool_input["task"]), call_index))
    return {"status": "success"}, None

  async def lookup_handler(_tool_input: dict[str, Any], *, call_index: int = 0, tool_ctx: Any = None):
    _ = tool_ctx
    tool_calls.append(("lookup", "", call_index))
    return {"status": "success"}, None

  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_dispatcher(event_log, {"run_agent": run_agent_handler, "lookup": lookup_handler}),
    session_id="sess-event-only",
    provider=provider,
    auth_config={"api_key": "k", "model": "stub-model"},
    get_tool_definitions=lambda: [_tool_def("run_agent"), _tool_def("lookup")],
    on_tool_result=_mixed_extra_blocks,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "go"}], system_prompt="x", max_turns=2))

  assert sorted(tool_calls) == [
    ("lookup", "", 0),
    ("run_agent", "a", 0),
    ("run_agent", "b", 1),
  ]

  model_content = _model_bound_tool_result_content(provider)
  assert [block["type"] for block in model_content] == [
    "tool_result",
    "tool_result",
    "tool_result",
    "text",
    "text",
    "text",
  ]
  assert [block["tool_use_id"] for block in model_content[:3]] == ["tool_1", "tool_2", "tool_3"]
  assert all(block.get("type") != "source_envelope" for block in model_content)


def test_event_only_blocks_are_filtered_from_transcript_replay() -> None:
  blocks = _tool_result_blocks_from_event(
    {
      "final_tool_result_blocks": [
        {"type": "tool_result", "tool_use_id": "tool_1", "content": "{}"},
        {"type": "text", "text": "visible citation map"},
        {"type": "source_envelope", "_event_only": True, "data": "hidden from replay"},
      ]
    }
  )

  assert [block["type"] for block in blocks] == ["tool_result", "text"]


def test_event_only_blocks_are_filtered_from_sdk_additional_context() -> None:
  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk",
    sdk_config=AgentSDKConfig(
      api_key="k",
      model="claude-sonnet-4-6",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    system_prompt="test",
  )

  context = runner._format_additional_context(
    tool_name="lookup",
    result_entry={"type": "tool_result", "tool_use_id": "tool_1", "content": json.dumps({"status": "success"})},
    extra_blocks=[
      {"type": "text", "text": "visible citation map"},
      {"type": "source_envelope", "_event_only": True, "data": "hidden from sdk"},
    ],
  )

  assert context == "visible citation map"
