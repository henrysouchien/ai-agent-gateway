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
  CostEstimate,
  EventLog,
  ModelInfo,
  ToolDispatcher,
)
from agent_gateway.runner import _MAX_TOKENS_CONTINUATIONS, _MAX_TOKENS_NUDGE, StreamTurnResult  # noqa: E402


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
  final_answer_guard: Any | None = None,
) -> AgentRunner:
  event_log = EventLog()
  return AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      mcp_client=_NullMcpClient(),
      local_tool_handlers={},
      event_log=event_log,
      session_id="sess-max-tokens",
    ),
    session_id="sess-max-tokens",
    provider=provider,
    auth_config=auth_config or {"api_key": "k", "model": "stub-model"},
    get_tool_definitions=lambda: [],
    final_answer_guard=final_answer_guard,
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


def test_max_tokens_turn_bypasses_final_answer_guard_until_continuation() -> None:
  async def _case() -> None:
    guard_turns: list[int] = []

    def guard(messages, answer_text, tools_used, tool_definitions, turn_count):
      _ = messages, answer_text, tools_used, tool_definitions
      guard_turns.append(turn_count)
      return "verify before final" if turn_count == 1 else None

    runner = _make_runner(_StubProvider(), final_answer_guard=guard)
    seen_messages: list[list[dict[str, Any]]] = []

    async def _fake_stream_turn(**kwargs: Any):
      seen_messages.append(list(kwargs["current_messages"]))
      if len(seen_messages) == 1:
        return object(), StreamTurnResult(
          full_text="truncated 43.5 / 47.0 - 1",
          stop_reason="max_tokens",
          content_blocks=[{"type": "text", "text": "truncated 43.5 / 47.0 - 1"}],
        )
      return object(), StreamTurnResult(
        full_text="done",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "done"}],
      )

    runner._stream_turn = _fake_stream_turn  # type: ignore[method-assign]
    await runner.run(messages=[{"role": "user", "content": "Start"}], system_prompt="x")

    assert len(seen_messages) == 2
    assert seen_messages[1][-1] == {"role": "user", "content": _MAX_TOKENS_NUDGE}
    assert guard_turns == [2]

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
      auth_config={"api_key": "k", "model": "stub-model", "max_tokens": 64_000},
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
