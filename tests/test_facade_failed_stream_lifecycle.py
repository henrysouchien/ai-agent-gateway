from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agent_gateway import (
  AgentRunner,
  AgentSessionLog,
  EventLog,
  ToolDispatcher,
)
from agent_gateway.mcp_client import McpClientManager
from hank_model_transport.providers import CostEstimate, ModelInfo, ModelProvider, StreamEvent
from gateway_test_support.capability_execution_test_support import (
  stub_bound_capability_execution,
)


class _FailingStreamProvider(ModelProvider):
  name = "stub"

  def __init__(self, *, context_length: bool) -> None:
    self.context_length = context_length

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    pass

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(max_output_tokens=16_384, id=model, provider=self.name)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return {
      "model": kwargs["model"],
      "messages": kwargs["messages"],
      "tools": kwargs["tools"],
      "max_tokens": kwargs["max_tokens"],
    }

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    yield StreamEvent(type="message_start", input_tokens=1)
    raise RuntimeError("provider stream failed")

  def is_context_length_error(self, exc: Exception) -> bool:
    return self.context_length

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    return CostEstimate()


@pytest.mark.parametrize("context_length", [False, True], ids=["error-return", "typed-failure"])
def test_failed_stream_turn_detaches_as_error(tmp_path: Path, context_length: bool) -> None:
  async def case() -> None:
    event_log = EventLog()
    durable_log = AgentSessionLog(tmp_path / "failed-stream.jsonl")
    execution = stub_bound_capability_execution(
      provider=_FailingStreamProvider(context_length=context_length),
      model="stub-model",
      effort="none",
      auth_config={"api_key": "test-key"},
    )
    dispatcher = ToolDispatcher(
      capability_bind=execution.bind,
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={},
      event_log=event_log,
      session_id="sess_failed_stream",
    )
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id="sess_failed_stream",
      capability_execution=execution,
      agent_session_log=durable_log,
      allow_stub_response=False,
      get_tool_definitions=lambda: [],
      compaction_trigger=None,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )

    await runner.run(messages=[{"role": "user", "content": "answer"}])

    live = [entry.event for entry in event_log.entries]
    assert [event["type"] for event in live if event["type"] in {
      "error", "stream_complete",
    }] == ["error"]
    assert "provider stream failed" in next(
      event["error"] for event in live if event["type"] == "error"
    )
    entries, _ = await durable_log.query(order="asc")
    durable = [entry.event for entry in entries]
    assert [event["reason"] for event in durable if event["type"] == "detach"] == [
      "error",
    ]
    assert [event["type"] for event in durable if event["type"] in {
      "error", "stream_complete",
    }] == ["error"]

  asyncio.run(case())
