import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, AgentSessionLog, EventLog, GatewaySession, ModelInfo, ModelProvider, ToolDispatcher
from agent_gateway.providers import StreamEvent


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _ScriptedProvider(ModelProvider):
  name = "stub"

  def __init__(self, turns: list[list[StreamEvent]]) -> None:
    self._turns = [list(turn) for turn in turns]
    self._turn_index = 0

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
    if self._turn_index >= len(self._turns):
      raise AssertionError("unexpected extra turn")
    current_turn = self._turns[self._turn_index]
    self._turn_index += 1
    for event in current_turn:
      yield event


def _make_dispatcher(
  *,
  event_log: EventLog | None = None,
  local_tool_handlers: dict[str, Any] | None = None,
) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers=local_tool_handlers or {},
    event_log=event_log or EventLog(),
    session_id="sess-parent",
  )


def _tool_turn(
  *,
  tool_id: str = "tool-1",
  tool_name: str = "lookup",
  tool_input: dict[str, Any] | None = None,
) -> list[StreamEvent]:
  payload = tool_input or {"query": "AAPL"}
  return [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(
      type="tool_use_end",
      tool_id=tool_id,
      tool_name=tool_name,
      tool_input=payload,
      raw_block={
        "type": "tool_use",
        "id": tool_id,
        "name": tool_name,
        "input": payload,
      },
    ),
    StreamEvent(type="usage_update", output_tokens=3),
    StreamEvent(type="message_end", stop_reason="tool_use"),
  ]


def _text_turn(text: str) -> list[StreamEvent]:
  return [
    StreamEvent(type="message_start", input_tokens=12),
    StreamEvent(type="text_delta", text=text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    StreamEvent(type="usage_update", output_tokens=5),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]


async def _lookup_tool(tool_input: dict[str, Any], **kwargs: Any):
  _ = kwargs
  return {"echo": tool_input}, None


def test_runner_emits_durable_envelope_in_order(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "runner.jsonl")
  provider = _ScriptedProvider([
    _tool_turn(),
    _text_turn("done"),
  ])
  event_log = EventLog()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log, local_tool_handlers={"lookup": _lookup_tool}),
    session_id="sess-parent",
    provider=provider,
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    agent_session_log=log,
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run lookup"}]))

  entries, _ = _run(log.query(order="asc"))
  event_types = [entry.event["type"] for entry in entries]
  assert event_types == [
    "attach",
    "user_message",
    "assistant_message",
    "tool_call_start",
    "tool_call_complete",
    "assistant_message",
    "detach",
  ]
  assert entries[1].event["content"] == "Run lookup"
  assert entries[2].event["stop_reason"] == "tool_use"
  assert entries[2].event["usage"]["input_tokens"] == 10
  assert entries[3].event["parent_assistant_message_seq"] == entries[2].seq
  assert entries[6].event["reason"] == "completed"


def test_runner_stale_recovery_synthesizes_orphan_tool_and_prior_writer_interrupt(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "recovery.jsonl")
  _run(
    log.append(
      {
        "type": "attach",
        "gateway_session_id": "analyst:2026-04-11",
        "runner_id": "runner_old",
        "started_at": 100.0,
        "client_kind": "cron",
        "role": "writer",
      }
    )
  )
  _run(
    log.append(
      {
        "type": "tool_call_start",
        "tool_call_id": "tool-orphan",
        "tool_name": "file_read",
        "tool_input": {"path": "/tmp/report.md"},
        "started_at": 101.0,
        "runner_id": "runner_old",
        "role": "writer",
      }
    )
  )

  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(local_tool_handlers={}),
    session_id="sess-parent",
    provider=_ScriptedProvider([_text_turn("recovered")]),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    agent_session_log=log,
  )

  _run(runner.run(messages=[{"role": "user", "content": "Continue"}], max_turns=1))

  entries, _ = _run(log.query(order="asc"))
  event_types = [entry.event["type"] for entry in entries]
  assert event_types[:6] == [
    "attach",
    "tool_call_start",
    "tool_call_interrupted",
    "interrupted",
    "attach",
    "user_message",
  ]

  orphan = entries[2].event
  recovery = entries[3].event
  new_attach = entries[4].event
  assert orphan["tool_call_id"] == "tool-orphan"
  assert orphan["tool_risk"] == "read_only"
  assert recovery["reason"] == "recovered_on_attach"
  assert recovery["runner_id"] == "runner_old"
  assert recovery["recovered_by_runner_id"] == new_attach["runner_id"]
  assert new_attach["runner_id"] != "runner_old"


def test_second_writer_lease_acquisition_raises(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "lease.jsonl")
  runner_one = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-one",
    provider=_ScriptedProvider([_text_turn("one")]),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    agent_session_log=log,
  )
  runner_two = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-two",
    provider=_ScriptedProvider([_text_turn("two")]),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    agent_session_log=log,
  )
  runner_one._runner_id = "runner_one"
  runner_two._runner_id = "runner_two"

  async def _exercise() -> None:
    await runner_one._acquire_writer_lease_and_recover()
    with pytest.raises(RuntimeError, match="Writer lease already held"):
      await runner_two._acquire_writer_lease_and_recover()
    runner_one._release_write_lease()

  _run(_exercise())


def test_sub_agent_events_are_written_to_parent_log(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "sub-agent.jsonl")
  parent_runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(local_tool_handlers={"lookup": _lookup_tool}),
    session_id="sess-parent",
    provider=_ScriptedProvider([
      _tool_turn(tool_id="tool-sub", tool_input={"query": "MSFT"}),
      _text_turn("sub done"),
    ]),
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
    agent_session_log=log,
  )
  sub_session = GatewaySession(
    session_id="sub0:sess-parent",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="alice",
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
  )

  result, error = _run(
    parent_runner.spawn_sub_agent(
      "Collect background context",
      dispatcher=_make_dispatcher(local_tool_handlers={"lookup": _lookup_tool}),
      sub_session=sub_session,
      max_turns=5,
      timeout=5.0,
    )
  )

  assert error is None
  assert result is not None

  entries, _ = _run(log.query(role="sub_agent", order="asc"))
  event_types = [entry.event["type"] for entry in entries]
  assert event_types == [
    "attach",
    "user_message",
    "assistant_message",
    "tool_call_start",
    "tool_call_complete",
    "assistant_message",
    "detach",
  ]
  assert {entry.event["sub_agent_id"] for entry in entries} == {"sub0:sess-parent"}

