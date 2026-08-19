"""ACUI-27: a turn that exhausts max_tokens with no usable tool call must not
silently end the run — the runner nudges and continues (bounded), and request
max_tokens is clamped to the model's max_output_tokens."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (  # noqa: E402
  AgentRunner,
  AgentSessionLog,
  CostEstimate,
  EventLog,
  ModelInfo,
  ToolDispatcher,
)
from agent_gateway.runner import _MAX_TOKENS_CONTINUATIONS, _MAX_TOKENS_NUDGE, StreamTurnResult  # noqa: E402
from agent_gateway.final_narrative_artifact import read_final_narrative  # noqa: E402
from agent_gateway.sub_agent_narrative_result import final_child_visible_text  # noqa: E402
from agent_gateway.task_registry import ParentMessage  # noqa: E402
from tests.capability_execution_test_support import (  # noqa: E402
  stub_runner_capability_execution,
)


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _StubProvider:
  name = "stub"

  def __init__(self, max_output_tokens: int = 16_384) -> None:
    self._max_output_tokens = max_output_tokens

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name, max_output_tokens=self._max_output_tokens)

  def estimate_cost(self, model, uncached, output, *, cache_read_tokens=0, cache_creation_tokens=0):
    _ = model, uncached, output, cache_read_tokens, cache_creation_tokens
    return CostEstimate()


def _make_runner(
  provider: _StubProvider,
  *,
  auth_config: dict[str, Any] | None = None,
  session_id: str = "sess-max-tokens",
  agent_session_log: AgentSessionLog | None = None,
  message_inbox: asyncio.Queue[ParentMessage] | None = None,
) -> AgentRunner:
  event_log = EventLog()
  return AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      mcp_client=_NullMcpClient(),
      local_tool_handlers={},
      event_log=event_log,
      session_id=session_id,
    ),
    session_id=session_id,
    capability_execution=stub_runner_capability_execution(
      provider=provider,
      model="stub-model",
      effort="none",
      auth_config=auth_config,
    ),
    get_tool_definitions=lambda: [],
    agent_session_log=agent_session_log,
    message_inbox=message_inbox,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_max_tokens_turn_with_no_tool_use_continues_with_nudge() -> None:
  async def _case() -> None:
    runner = _make_runner(_StubProvider())
    seen_messages: list[list[dict[str, Any]]] = []

    async def _fake_stream_turn(**kwargs: Any):
      seen_messages.append(list(kwargs["current_messages"]))
      if len(seen_messages) == 1:
        return object(), StreamTurnResult(
          full_text="",
          stop_reason="max_tokens",
          content_blocks=[
            {"type": "thinking", "thinking": "long reasoning", "signature": "sig"},
            {"type": "tool_use", "id": "tool_partial", "name": "fms_report", "input": {}},
          ],
        )
      return object(), StreamTurnResult(
        full_text="done",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "done"}],
      )

    runner._stream_turn = _fake_stream_turn  # type: ignore[method-assign]
    await runner.run(messages=[{"role": "user", "content": "Start"}], system_prompt="x")

    assert len(seen_messages) == 2, "run must continue past the truncated turn"
    follow_up = seen_messages[1]
    assert follow_up[-1] == {"role": "user", "content": _MAX_TOKENS_NUDGE}
    assert "tool-first response" in _MAX_TOKENS_NUDGE
    assert "smallest valid JSON payload" in _MAX_TOKENS_NUDGE
    assert "Do not spend another turn on hidden analysis" in _MAX_TOKENS_NUDGE
    # the truncated partial tool_use must NOT be replayed to the model
    replayed_assistant = follow_up[-2]
    assert replayed_assistant["role"] == "assistant"
    assert all(block.get("type") != "tool_use" for block in replayed_assistant["content"])

  _run(_case())


def test_max_tokens_continuation_is_bounded(caplog) -> None:
  async def _case() -> None:
    runner = _make_runner(_StubProvider())
    calls = {"n": 0}

    async def _fake_stream_turn(**kwargs: Any):
      calls["n"] += 1
      return object(), StreamTurnResult(
        full_text="",
        stop_reason="max_tokens",
        content_blocks=[{"type": "text", "text": "truncated"}],
      )

    runner._stream_turn = _fake_stream_turn  # type: ignore[method-assign]
    caplog.set_level(logging.WARNING, logger="agent_gateway.runner")
    await runner.run(messages=[{"role": "user", "content": "Start"}], system_prompt="x")

    # initial turn + bounded continuations, then the run ends instead of looping
    assert calls["n"] == 1 + _MAX_TOKENS_CONTINUATIONS
    assert f"continuing with truncation nudge (1/{_MAX_TOKENS_CONTINUATIONS})" in caplog.text
    assert f"continuing with truncation nudge ({_MAX_TOKENS_CONTINUATIONS}/{_MAX_TOKENS_CONTINUATIONS})" in caplog.text
    assert f"after {_MAX_TOKENS_CONTINUATIONS} continuation attempts" in caplog.text

  _run(_case())


def test_child_max_tokens_segments_are_unbounded_and_do_not_consume_max_turns(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    durable_log = AgentSessionLog(tmp_path / "child.jsonl")
    runner = _make_runner(
      _StubProvider(),
      session_id="sub_parent:1",
      agent_session_log=durable_log,
    )
    calls = 0

    async def _fake_stream_turn(**_kwargs: Any):
      nonlocal calls
      calls += 1
      if calls <= 5:
        return object(), StreamTurnResult(
          full_text="repeated provider segment",
          stop_reason="max_tokens",
          content_blocks=[{
            "type": "text",
            "text": "repeated provider segment",
          }],
        )
      return object(), StreamTurnResult(
        full_text="terminal segment",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "terminal segment"}],
      )

    runner._stream_turn = _fake_stream_turn  # type: ignore[method-assign]
    await runner.run(
      messages=[{"role": "user", "content": "Start"}],
      system_prompt="x",
      max_turns=1,
    )

    assert calls == 6
    entries, _ = await durable_log.query(
      event_types={"assistant_message"},
      sub_agent_id="sub_parent:1",
      runner_id=runner._runner_id,
      order="asc",
    )
    assert len(entries) == 6
    assert {
      entry.event["logical_response_id"] for entry in entries
    } == {entries[0].event["logical_response_id"]}
    assert [
      entry.event["logical_response_segment_ordinal"]
      for entry in entries
    ] == list(range(6))
    assert [
      entry.event.get("continued_from_assistant_message_seq")
      for entry in entries
    ] == [None, *[entry.seq for entry in entries[:-1]]]
    assert not any(
      entry.event.get("type") == "max_turns_reached"
      for entry in runner._log.entries
    )

  _run(_case())


def test_parent_message_breaks_child_max_tokens_logical_response(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    durable_log = AgentSessionLog(tmp_path / "steered-child.jsonl")
    inbox: asyncio.Queue[ParentMessage] = asyncio.Queue()
    runner = _make_runner(
      _StubProvider(),
      session_id="sub_parent:1",
      agent_session_log=durable_log,
      message_inbox=inbox,
    )
    seen_messages: list[list[dict[str, Any]]] = []

    async def _fake_stream_turn(**kwargs: Any):
      seen_messages.append(list(kwargs["current_messages"]))
      if len(seen_messages) == 1:
        sent = await durable_log.append({
          "type": "parent_message_sent",
          "runner_id": "runner-parent",
          "role": "writer",
          "task_id": "bg-steered",
          "message_id": "msg-steer",
          "message": "Discard the prior draft. Return only X.",
          "sub_agent_id": "sub_parent:1",
        })
        await inbox.put(ParentMessage(
          message_id="msg-steer",
          text="Discard the prior draft. Return only X.",
          sent_at=1.0,
          task_id="bg-steered",
          sent_seq=sent.seq,
        ))
        return object(), StreamTurnResult(
          full_text="discarded prefix ",
          stop_reason="max_tokens",
          content_blocks=[{
            "type": "text",
            "text": "discarded prefix ",
          }],
        )
      return object(), StreamTurnResult(
        full_text="X",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "X"}],
      )

    runner._stream_turn = _fake_stream_turn  # type: ignore[method-assign]
    await runner.run(
      messages=[{"role": "user", "content": "Start"}],
      system_prompt="x",
      max_turns=2,
    )

    assert len(seen_messages) == 2
    assert seen_messages[1][-1] == {
      "role": "user",
      "content": (
        "Operator update for this task:\n"
        "- id=msg-steer: Discard the prior draft. Return only X."
      ),
    }
    assistant_entries, _ = await durable_log.query(
      event_types={"assistant_message"},
      sub_agent_id="sub_parent:1",
      runner_id=runner._runner_id,
      order="asc",
    )
    assert len(assistant_entries) == 2
    assert assistant_entries[0].event["logical_response_id"] != (
      assistant_entries[1].event["logical_response_id"]
    )
    assert [
      entry.event["logical_response_segment_ordinal"]
      for entry in assistant_entries
    ] == [0, 0]
    visible = await final_child_visible_text(
      durable_log,
      sub_session_id="sub_parent:1",
      workspace_dir=str(tmp_path),
      runner_id=runner._runner_id,
    )
    assert visible.text == "X"
    assert visible.final_narrative is not None
    assert read_final_narrative(
      workspace_dir=tmp_path,
      reference=visible.final_narrative,
    ) == "X"
    consumed_entries, _ = await durable_log.query(
      event_types={"parent_message_consumed"},
      order="asc",
    )
    assert len(consumed_entries) == 1
    assert consumed_entries[0].event["message_id"] == "msg-steer"
    assert consumed_entries[0].event["assistant_message_seq"] == (
      assistant_entries[1].seq
    )
    assert consumed_entries[0].event["consumer_turn"] == 2

  _run(_case())


def test_max_tokens_turn_with_tool_uses_is_not_intercepted() -> None:
  async def _case() -> None:
    runner = _make_runner(_StubProvider())
    calls = {"n": 0}

    async def _fake_stream_turn(**kwargs: Any):
      calls["n"] += 1
      if calls["n"] == 1:
        result = StreamTurnResult(
          full_text="",
          stop_reason="max_tokens",
          content_blocks=[
            {"type": "tool_use", "id": "t1", "name": "nonexistent_tool", "input": {}},
          ],
        )
        result.tool_uses = [("t1", "nonexistent_tool", {})]
        return object(), result
      return object(), StreamTurnResult(
        full_text="done",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "done"}],
      )

    runner._stream_turn = _fake_stream_turn  # type: ignore[method-assign]
    await runner.run(messages=[{"role": "user", "content": "Start"}], system_prompt="x")

    # complete tool calls still execute (loop continues through tool dispatch)
    assert calls["n"] == 2

  _run(_case())


def test_request_max_tokens_clamped_to_model_max_output() -> None:
  async def _case() -> None:
    runner = _make_runner(
      _StubProvider(max_output_tokens=16_384),
      auth_config={"api_key": "k", "max_tokens": 64_000},
    )
    seen: dict[str, Any] = {}

    async def _fake_stream_turn(**kwargs: Any):
      seen["max_tokens"] = kwargs["max_tokens"]
      return object(), StreamTurnResult(
        full_text="done",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "done"}],
      )

    runner._stream_turn = _fake_stream_turn  # type: ignore[method-assign]
    await runner.run(messages=[{"role": "user", "content": "Start"}], system_prompt="x")

    assert seen["max_tokens"] == 16_384

  _run(_case())
