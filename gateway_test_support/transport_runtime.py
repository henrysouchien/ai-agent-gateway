"""Explicit host runtime for transport calls exercised by gateway tests."""

from __future__ import annotations

from typing import Any, AsyncIterator

from hank_model_transport.providers import ModelInfo, StreamEvent
from hank_model_transport.send_prompt import SendPromptRuntime
from model_authority.thinking import EffortResolution


class _GatewayTestSendPromptRuntime:
  def stream_with_stall_timeout(
    self,
    stream: AsyncIterator[StreamEvent],
    *,
    config: dict[str, Any],
    model_info: ModelInfo,
    max_tokens: int,
    effort_resolution: EffortResolution,
  ) -> AsyncIterator[StreamEvent]:
    from agent_gateway import runner_streaming

    return runner_streaming._stream_with_stall_timeout(
      stream,
      config=config,
      model_info=model_info,
      max_tokens=max_tokens,
      effort_resolution=effort_resolution,
    )

  def apply_provider_diagnostics(self, usage_totals: dict[str, Any], event: Any) -> None:
    from agent_gateway import runner_usage

    runner_usage.apply_provider_diagnostics(usage_totals, event)

  def apply_thinking_event(
    self, usage_totals: dict[str, Any], event: Any, streamed_block_chars: int,
  ) -> int:
    from agent_gateway import runner_usage

    return runner_usage.apply_thinking_event(usage_totals, event, streamed_block_chars)


TEST_SEND_PROMPT_RUNTIME: SendPromptRuntime = _GatewayTestSendPromptRuntime()
