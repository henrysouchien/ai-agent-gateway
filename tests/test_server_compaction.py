from __future__ import annotations
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway import (
  AgentRunner,
  AgentSessionLog,
  EventLog,
  ModelInfo,
  ModelProvider,
  ToolDispatcher,
)
from model_authority.thinking import EffortResolution
from agent_gateway.providers import (
  AnthropicProvider,
  CodexProvider,
  OpenAIProvider,
  StreamEvent,
  XAIProvider,
)
from model_authority.thinking import ThinkingLevel
from model_authority.binding import CapabilityEffort
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.providers.base import truncate_to_last_compaction
from agent_gateway.context_builder import SessionContextBuilder
from agent_gateway.server_compaction import (
  CONTINUATION_NUDGE,
  KEEP_ROUNDS,
  MIN_COMPACT_GAIN_TOKENS,
  MIN_SUMMARY_CHARS,
  PORTABLE_COMPACTION_ENV,
  RENDER_CHAR_BUDGET,
  SUMMARY_MAX_TOKENS,
  TAIL_TOKEN_BUDGET,
  apply_compaction_anchor,
  build_summary_prompt,
  maybe_compact_current_messages,
  render_messages_for_summary,
  should_portable_compact,
  split_messages_for_compact,
  summarize_messages,
)
import agent_gateway.runner_run_loop as runner_run_loop
import agent_gateway.server_compaction as server_compaction
from gateway_test_support.capability_execution_test_support import (
  stub_bound_capability_execution,
)


def _run(coro):
  return asyncio.run(coro)


LONG_SUMMARY = (
  "Original request: continue the approved implementation. "
  "Current state: portable compaction has summarized earlier tool work. "
  "Next step: continue from this state with exact file paths and tool ids preserved. "
) * 2
ANCHOR_SUMMARY = LONG_SUMMARY.strip()
LONG_INPUT = "x" * 3_000


class _NullMcpClient(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)


def _make_dispatcher(
  *,
  capability_bind = TEST_CAPABILITY_BIND,
  event_log: EventLog | None = None,
  local_tool_handlers: dict[str, Any] | None = None,
) -> ToolDispatcher:
  return ToolDispatcher(
    capability_bind=capability_bind,
    mcp_client=_NullMcpClient(),
    local_tool_handlers=local_tool_handlers or {},
    event_log=event_log or EventLog(),
    session_id="sess-portable",
  )


async def _lookup_tool(tool_input: dict[str, Any], **kwargs: Any):
  _ = kwargs
  return {"ok": True, "echo": tool_input}, None


class _SummaryClient:
  pass


class _SummaryProvider(ModelProvider):
  name = "codex"

  def __init__(
    self,
    *,
    summary_text: str = LONG_SUMMARY,
    summary_raises: Exception | None = None,
    summary_tool_use: bool = False,
  ) -> None:
    self.summary_text = summary_text
    self.summary_raises = summary_raises
    self.summary_tool_use = summary_tool_use
    self.requests: list[dict[str, Any]] = []

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return _SummaryClient()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name, context_window=200_000, max_output_tokens=4096, supports_thinking=True)

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    _ = model_info
    return truncate_to_last_compaction([dict(message) for message in messages], compaction_as_text=True)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    thinking_level: ThinkingLevel = ThinkingLevel.HIGH,
    **kwargs: Any,
  ) -> dict[str, Any]:
    request = {
      "model": model,
      "messages": [dict(message) for message in messages],
      "system_prompt": system_prompt,
      "tools": [dict(tool) for tool in tools],
      "max_tokens": max_tokens,
      "thinking_level": thinking_level,
      "kwargs": dict(kwargs),
    }
    self.requests.append(request)
    return request

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client
    if self.summary_raises is not None:
      raise self.summary_raises
    yield StreamEvent(type="message_start", input_tokens=31, cache_read_tokens=2, cache_creation_tokens=3)
    if self.summary_tool_use:
      yield StreamEvent(
        type="tool_use_end",
        tool_id="tool-summary",
        tool_name="lookup",
        tool_input={},
        raw_block={"type": "tool_use", "id": "tool-summary", "name": "lookup", "input": {}},
      )
    yield StreamEvent(type="text_delta", text=self.summary_text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": self.summary_text})
    yield StreamEvent(type="usage_update", output_tokens=7, reasoning_tokens=1)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _ClampingSummaryProvider(_SummaryProvider):
  def __init__(self) -> None:
    super().__init__()
    self.effort_resolutions: list[dict[str, Any]] = []
    self.client_creations = 0

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    self.client_creations += 1
    return super().create_client(config, timeout=timeout)

  def resolve_effort(
    self,
    *,
    requested: ThinkingLevel,
    model: str,
    model_info: ModelInfo,
    max_tokens: int,
    **request_context: Any,
  ) -> EffortResolution:
    self.effort_resolutions.append({
      "requested": requested,
      "model": model,
      "model_info": model_info,
      "max_tokens": max_tokens,
      "request_context": request_context,
    })
    return EffortResolution(
      requested=requested,
      effective=(
        requested
        if max_tokens == 512
        else ThinkingLevel.LOW
      ),
      thinking_enabled_effective=True,
      payload_fragments={},
    )


class _PortableRunnerProvider(_SummaryProvider):
  name = "codex"

  def __init__(
    self,
    *,
    main_turns: list[list[StreamEvent] | Exception],
    summary_text: str = LONG_SUMMARY,
  ) -> None:
    super().__init__(summary_text=summary_text)
    self.main_turns = list(main_turns)
    self.main_requests: list[list[dict[str, Any]]] = []
    self.summary_requests = 0
    self.closed_clients = 0

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name, context_window=20, max_output_tokens=4096, supports_thinking=True)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    thinking_level: ThinkingLevel = ThinkingLevel.HIGH,
    **kwargs: Any,
  ) -> dict[str, Any]:
    request = super().build_request_params(
      model=model,
      messages=messages,
      system_prompt=system_prompt,
      tools=tools,
      max_tokens=max_tokens,
      thinking_level=thinking_level,
      **kwargs,
    )
    is_summary = bool(
      messages
      and messages[-1].get("role") == "user"
      and "The <summary> content is the only text" in str(messages[-1].get("content") or "")
    )
    request["kind"] = "summary" if is_summary else "main"
    if is_summary:
      self.summary_requests += 1
    else:
      self.main_requests.append([dict(message) for message in messages])
    return request

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout
    self.closed_clients += 1

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client
    if params.get("kind") == "summary":
      async for event in super().stream(client, params):
        yield event
      return
    if not self.main_turns:
      raise AssertionError("unexpected extra main turn")
    turn = self.main_turns.pop(0)
    if isinstance(turn, Exception):
      raise turn
    for event in turn:
      yield event


class _ContextLengthProvider(_PortableRunnerProvider):
  def is_context_length_error(self, exc: Exception) -> bool:
    return "context length" in str(exc).lower()


class _AnthropicRunnerProvider(AnthropicProvider):
  def __init__(
    self,
    *,
    main_turns: list[list[StreamEvent] | Exception],
  ) -> None:
    super().__init__()
    self.main_turns = list(main_turns)
    self.requests: list[dict[str, Any]] = []
    self.summary_requests = 0

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return _SummaryClient()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return replace(
      super().get_model_info(model),
      context_window=20,
      max_output_tokens=4096,
    )

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    thinking_level: ThinkingLevel = ThinkingLevel.HIGH,
    **kwargs: Any,
  ) -> dict[str, Any]:
    request = super().build_request_params(
      model=model,
      messages=messages,
      system_prompt=system_prompt,
      tools=tools,
      max_tokens=max_tokens,
      thinking_level=thinking_level,
      **kwargs,
    )
    is_summary = bool(
      messages
      and messages[-1].get("role") == "user"
      and "The <summary> content is the only text" in str(messages[-1].get("content") or "")
    )
    request["kind"] = "summary" if is_summary else "main"
    self.requests.append(request)
    if is_summary:
      self.summary_requests += 1
    return request

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client
    if params.get("kind") == "summary":
      yield StreamEvent(type="message_start", input_tokens=31)
      yield StreamEvent(type="text_delta", text=LONG_SUMMARY)
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": LONG_SUMMARY})
      yield StreamEvent(type="usage_update", output_tokens=7)
      yield StreamEvent(type="message_end", stop_reason="end_turn")
      return
    if not self.main_turns:
      raise AssertionError("unexpected extra main turn")
    turn = self.main_turns.pop(0)
    if isinstance(turn, Exception):
      raise turn
    for event in turn:
      yield event


def _tool_turn() -> list[StreamEvent]:
  return [
    StreamEvent(type="message_start", input_tokens=11),
    StreamEvent(
      type="tool_use_end",
      tool_id="tool-1",
      tool_name="lookup",
      tool_input={"query": "AAPL"},
      raw_block={"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"query": "AAPL"}},
    ),
    StreamEvent(type="usage_update", output_tokens=5),
    StreamEvent(type="message_end", stop_reason="tool_use"),
  ]


def _text_turn(text: str = "done") -> list[StreamEvent]:
  return [
    StreamEvent(type="message_start", input_tokens=13),
    StreamEvent(type="text_delta", text=text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    StreamEvent(type="usage_update", output_tokens=3),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]


def _run_runner(
  provider: ModelProvider,
  *,
  messages: list[dict[str, Any]],
  event_log: EventLog | None = None,
  agent_session_log: AgentSessionLog | None = None,
  compaction_trigger: int = 20,
  effort: CapabilityEffort = "high",
  model: str = "model",
) -> EventLog:
  log = event_log or EventLog()
  capability_execution = stub_bound_capability_execution(
    provider=provider,
    model=model,
    effort=effort,
    auth_config={"api_key": "test", "max_tokens": 512},
  )
  runner = AgentRunner(
    event_log=log,
    dispatcher=_make_dispatcher(capability_bind=capability_execution.bind, event_log=log, local_tool_handlers={"lookup": _lookup_tool}),
    session_id="sess-portable",
    capability_execution=capability_execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    compaction_trigger=compaction_trigger,
    agent_session_log=agent_session_log,
    emit_session_recap=False,
  )
  with pytest.MonkeyPatch.context() as monkeypatch:
    monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 500)
    monkeypatch.setattr(runner_run_loop, "MIN_COMPACT_GAIN_TOKENS", 500)
    _run(runner.run(messages))
  return log


def test_server_compaction_constants() -> None:
  assert KEEP_ROUNDS == 0
  assert TAIL_TOKEN_BUDGET == 24_000
  assert MIN_COMPACT_GAIN_TOKENS == 20_000
  assert SUMMARY_MAX_TOKENS == 4_096
  assert RENDER_CHAR_BUDGET == 600_000
  assert MIN_SUMMARY_CHARS == 200
  assert PORTABLE_COMPACTION_ENV == "AGENT_GATEWAY_PORTABLE_COMPACTION"


def test_server_compaction_split_tool_pairs() -> None:
  messages = [
    {"role": "user", "content": "old"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-old", "name": "lookup", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-old", "content": "old result"}]},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-tail", "name": "lookup", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-tail", "content": "tail result"}]},
  ]

  _to_summarize, tail = split_messages_for_compact(messages, keep_rounds=1)

  assert [message["role"] for message in tail] == ["assistant", "user"]
  assert tail[0]["content"][0]["id"] == "tool-tail"
  assert tail[1]["content"][0]["tool_use_id"] == "tool-tail"


def test_server_compaction_apply_truncate_roundtrip() -> None:
  compacted = apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="model")
  messages = [{"role": "user", "content": "old history"}, *compacted, {"role": "user", "content": "next"}]

  truncated = truncate_to_last_compaction(messages, compaction_as_text=True)

  assert len(truncated) == 3
  assert truncated[0]["role"] == "assistant"
  assert truncated[0]["content"][0]["type"] == "text"
  assert "old history" not in str(truncated)
  assert "Summary of the earlier conversation" in truncated[0]["content"][0]["text"]


def test_server_compaction_codex_normalize_text() -> None:
  provider = CodexProvider()
  model_info = provider.get_model_info("gpt-5.6-sol")
  normalized: list[dict[str, Any]] = provider.normalize_messages(
    [{"role": "user", "content": "old"}, *apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="gpt-5.6-sol")],
    model_info,
  )

  assert normalized[0]["content"][0]["type"] == "text"
  assert "Summary of the earlier conversation" in normalized[0]["content"][0]["text"]
  assert all(
    not (isinstance(block, dict) and block.get("type") == "compaction")
    for message in normalized
    for block in (
      lambda content: content if isinstance(content, list) else []
    )(message.get("content"))
  )


def test_server_compaction_skip_when_model_supports_native() -> None:
  should, reason = should_portable_compact(
    est_total_tokens=200_000,
    est_summarize_tokens=100_000,
    trigger=160_000,
    native_compaction=True,
  )

  assert should is False
  assert reason == "native_model"


def test_server_compaction_trigger_respects_effective() -> None:
  should, reason = should_portable_compact(
    est_total_tokens=200_000,
    est_summarize_tokens=100_000,
    trigger=800_000,
    native_compaction=False,
  )

  assert should is False
  assert reason == "below_trigger"


def test_server_compaction_propagates_summary_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 10)
  provider = _SummaryProvider(summary_raises=RuntimeError("provider timeout"))
  messages = [{"role": "user", "content": "x" * 200}]

  execution = stub_bound_capability_execution(
    provider=provider,
    model="model",
    effort="high",
    auth_config={"api_key": "test", "max_tokens": 512},
  )
  with pytest.raises(RuntimeError, match="provider timeout"):
    _run(
      maybe_compact_current_messages(
        messages,
        None,
        capability_execution=execution,
        trigger=1,
        est_total_tokens=100,
      )
    )


def test_server_compaction_gain_guard() -> None:
  should, reason = should_portable_compact(
    est_total_tokens=200_000,
    est_summarize_tokens=MIN_COMPACT_GAIN_TOKENS - 1,
    trigger=160_000,
    native_compaction=False,
  )
  assert (should, reason) == (False, "insufficient_gain")

  should, reason = should_portable_compact(
    est_total_tokens=180_000,
    est_summarize_tokens=100_000,
    trigger=160_000,
    native_compaction=False,
    failed_est_at=170_000,
  )
  assert (should, reason) == (False, "failed_cooldown")

  should, reason = should_portable_compact(
    est_total_tokens=191_000,
    est_summarize_tokens=100_000,
    trigger=160_000,
    native_compaction=False,
    failed_est_at=170_000,
  )
  assert (should, reason) == (True, "triggered")


def test_server_compaction_orphan_passthrough_characterization() -> None:
  messages = [
    *apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="gpt-5.6-sol"),
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "missing-tool", "content": "orphan"}]},
  ]

  codex_normalized = CodexProvider().normalize_messages(messages, CodexProvider().get_model_info("gpt-5.6-sol"))
  openai_normalized = OpenAIProvider().normalize_messages(messages, OpenAIProvider().get_model_info("gpt-5.6"))

  assert codex_normalized[-1]["content"][0]["tool_use_id"] == "missing-tool"
  assert openai_normalized[-1]["content"][0]["tool_use_id"] == "missing-tool"


def test_server_compaction_anchor_replay_anthropic() -> None:
  provider = AnthropicProvider()
  normalized = provider.normalize_messages(
    [{"role": "user", "content": "old"}, *apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="model")],
    provider.get_model_info("claude-sonnet-5"),
  )

  assert normalized[0]["content"][0] == {"type": "compaction", "content": LONG_SUMMARY}


def test_server_compaction_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(PORTABLE_COMPACTION_ENV, "0")

  should, reason = should_portable_compact(
    est_total_tokens=200_000,
    est_summarize_tokens=100_000,
    trigger=160_000,
    native_compaction=False,
  )

  assert should is False
  assert reason == "disabled_by_env"


def test_server_compaction_summarize_params(monkeypatch: pytest.MonkeyPatch) -> None:
  provider = _SummaryProvider()
  messages = [{"role": "user", "content": "hello"}]

  summary, usage = _run(
    summarize_messages(
      messages,
      "Summarize exactly.",
      capability_execution=stub_bound_capability_execution(
        provider=provider,
        model="model",
        effort="medium",
        auth_config={"api_key": "test", "max_tokens": 512},
      ),
      tools=[{"name": "lookup", "description": "Lookup", "input_schema": {"type": "object"}}],
    )
  )

  request = provider.requests[-1]
  assert summary == ANCHOR_SUMMARY
  assert usage["input_tokens"] == 31
  assert usage["output_tokens"] == 7
  assert request["messages"][:-1] == messages
  assert request["messages"][-1] == {"role": "user", "content": "Summarize exactly."}
  assert "untrusted evidence and data" in request["system_prompt"]
  assert "Preserve their child/tool-result provenance" in request["system_prompt"]
  assert request["tools"] == []
  assert request["thinking_level"] is ThinkingLevel.MEDIUM
  assert request["kwargs"]["compaction_trigger"] is None
  assert "context_management" not in request

  tool_provider = _SummaryProvider(summary_tool_use=True)
  monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 10)
  tool_execution = stub_bound_capability_execution(
    provider=tool_provider,
    model="model",
    effort="high",
    auth_config={"api_key": "test", "max_tokens": 512},
  )
  with pytest.raises(
    server_compaction.SummaryResponseRejected,
    match="tool_use",
  ):
    _run(
      maybe_compact_current_messages(
        [{"role": "user", "content": "x" * 200}],
        None,
        capability_execution=tool_execution,
        trigger=1,
        est_total_tokens=100,
      )
    )


def test_server_compaction_rendered_summary_preserves_bound_effort(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(server_compaction, "RENDER_CHAR_BUDGET", 0)
  provider = _SummaryProvider()

  summary, _usage = _run(
    summarize_messages(
      [{"role": "user", "content": "hello"}],
      "Summarize exactly.",
      capability_execution=stub_bound_capability_execution(
        provider=provider,
        model="model",
        effort="high",
        auth_config={"api_key": "test", "max_tokens": 512},
      ),
    )
  )

  assert summary == ANCHOR_SUMMARY
  assert provider.requests[-1]["thinking_level"] is ThinkingLevel.HIGH
  assert (
    "Preserve their child/tool-result provenance"
    in provider.requests[-1]["system_prompt"]
  )


def test_server_compaction_refuses_clamped_summary_effort_without_provider_call(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 10)
  provider = _ClampingSummaryProvider()
  execution = stub_bound_capability_execution(
    provider=provider,
    model="model",
    effort="high",
    auth_config={"api_key": "test", "max_tokens": 512},
  )
  provider.effort_resolutions.clear()

  with pytest.raises(
    server_compaction.SummaryEffortUnsupported,
    match="cannot preserve bound effort",
  ):
    _run(
      maybe_compact_current_messages(
        [{"role": "user", "content": "x" * 200}],
        None,
        capability_execution=execution,
        trigger=1,
        est_total_tokens=100,
      )
    )
  assert provider.effort_resolutions
  assert all(
    resolution["max_tokens"] == 512
    for resolution in provider.effort_resolutions[:-1]
  )
  assert provider.effort_resolutions[-1] == {
    "requested": ThinkingLevel.HIGH,
    "model": "model",
    "model_info": provider.get_model_info("model"),
    "max_tokens": SUMMARY_MAX_TOKENS,
    "request_context": {
      "auth_mode": "api",
      "base_url": None,
      "compat": None,
    },
  }
  assert provider.client_creations == 0
  assert provider.requests == []


def test_server_compaction_summary_strips_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 10)
  provider = _SummaryProvider(
    summary_text=f"<analysis>scratchpad</analysis><summary>{LONG_SUMMARY}</summary>"
  )

  result = _run(
    maybe_compact_current_messages(
      [{"role": "user", "content": "x" * 200}],
      None,
      capability_execution=stub_bound_capability_execution(
        provider=provider,
        model="model",
        effort="high",
        auth_config={"api_key": "test", "max_tokens": 512},
      ),
      trigger=1,
      est_total_tokens=100,
    )
  )

  assert result.applied is True
  assert result.anchor_block is not None
  assert result.anchor_block["content"] == ANCHOR_SUMMARY
  assert "analysis" not in result.anchor_block["content"]
  assert result.summary_chars >= MIN_SUMMARY_CHARS


def test_server_compaction_continuation_request_shape() -> None:
  messages = apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="gpt-5.6-sol")
  normalized = CodexProvider().normalize_messages(messages, CodexProvider().get_model_info("gpt-5.6-sol"))

  assert messages[0]["role"] == "assistant"
  assert messages[0]["content"][0]["type"] == "compaction"
  assert messages[1] == {"role": "user", "content": CONTINUATION_NUDGE}
  assert normalized[0]["content"][0]["type"] == "text"
  assert normalized[-1]["role"] == "user"


def test_runner_portable_compact_mid_loop() -> None:
  provider = _PortableRunnerProvider(main_turns=[_tool_turn(), _text_turn("complete")])
  event_log = _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    effort="medium",
  )

  events = [entry.event for entry in event_log.entries]
  assert any(event.get("type") == "compaction" and event.get("chars") == len(ANCHOR_SUMMARY) for event in events)
  assert any(event.get("type") == "tool_call_complete" for event in events)
  assert provider.summary_requests == 1
  summary_request = next(request for request in provider.requests if request["kind"] == "summary")
  assert summary_request["thinking_level"] is ThinkingLevel.MEDIUM
  first_main = provider.main_requests[0]
  assert first_main[0]["role"] == "assistant"
  assert first_main[0]["content"][0]["type"] == "text"
  assert first_main[1] == {"role": "user", "content": CONTINUATION_NUDGE}
  second_main_text = str(provider.main_requests[1])
  assert CONTINUATION_NUDGE not in second_main_text


def test_runner_haiku_uses_portable_compaction_without_native_payload() -> None:
  provider = _AnthropicRunnerProvider(main_turns=[_text_turn("complete")])

  _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    effort="none",
    model="claude-haiku-4-5",
  )

  assert provider.summary_requests == 1
  assert all("context_management" not in request for request in provider.requests)
  main_request = next(request for request in provider.requests if request["kind"] == "main")
  assert main_request["messages"][0]["content"][0]["type"] == "text"
  assert "Summary of the earlier conversation" in main_request["messages"][0]["content"][0]["text"]


def test_runner_supported_anthropic_model_uses_native_compaction() -> None:
  provider = _AnthropicRunnerProvider(main_turns=[_text_turn("complete")])

  _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    effort="none",
    model="claude-sonnet-5",
  )

  assert provider.summary_requests == 0
  main_request = next(request for request in provider.requests if request["kind"] == "main")
  assert main_request["context_management"]["edits"][0]["type"] == "compact_20260112"


def test_server_compaction_summary_precedes_retained_responses(tmp_path) -> None:
  provider = _PortableRunnerProvider(main_turns=[_tool_turn(), _text_turn("complete")])
  durable_log = AgentSessionLog(path=tmp_path / "session.jsonl")
  _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    agent_session_log=durable_log,
  )

  assistant_entries, _ = _run(durable_log.query(event_types={"assistant_message"}, order="asc"))
  summaries, _ = _run(durable_log.query(event_types={"summary"}, order="asc"))
  assert len(summaries) == 1
  assert summaries[0].event["summary_kind"] == "portable_compaction"
  assert summaries[0].event["text"] == ANCHOR_SUMMARY
  assert summaries[0].event["covers"]["to_seq"] < assistant_entries[0].seq
  assert not any(
    block.get("type") == "compaction"
    for entry in assistant_entries for block in entry.event["content_blocks"]
  )
  rebuilt = _run(SessionContextBuilder(durable_log).build())
  assert rebuilt[0]["content"] == [{"type": "compaction", "content": ANCHOR_SUMMARY}]
  assert any(block.get("type") == "tool_result" for message in rebuilt
             if isinstance(message["content"], list) for block in message["content"])


def test_server_compaction_anchor_persist_on_stream_failure(tmp_path) -> None:
  provider = _PortableRunnerProvider(main_turns=[RuntimeError("stream exploded")])
  durable_log = AgentSessionLog(path=tmp_path / "session.jsonl")
  _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    agent_session_log=durable_log,
  )

  summaries, _ = _run(durable_log.query(event_types={"summary"}, order="asc"))
  anchor_event = summaries[-1].event
  assert anchor_event["summary_kind"] == "portable_compaction"
  assert anchor_event["text"] == ANCHOR_SUMMARY
  assert anchor_event["usage"]["input_tokens"] == 31
  assert anchor_event["usage"]["output_tokens"] == 7

  resume_provider = _PortableRunnerProvider(main_turns=[_tool_turn(), _text_turn("resumed")])
  rebuilt = _run(SessionContextBuilder(durable_log).build())
  anchor_message = rebuilt[0]
  assert anchor_message["content"] == [{"type": "compaction", "content": ANCHOR_SUMMARY}]
  _run_runner(resume_provider, messages=[anchor_message])
  assert resume_provider.main_requests[0][-1] == {"role": "user", "content": CONTINUATION_NUDGE}
  assert CONTINUATION_NUDGE not in str(resume_provider.main_requests[1])


def test_server_compaction_multi_compact_supersede() -> None:
  messages = [
    {"role": "user", "content": "old"},
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "first"},
        {"type": "text", "text": "after first"},
      ],
    },
    {"role": "user", "content": "middle"},
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "second"},
        {"type": "text", "text": "after second"},
      ],
    },
    {"role": "user", "content": "next"},
  ]

  truncated = truncate_to_last_compaction(messages, compaction_as_text=True)

  assert len(truncated) == 2
  assert "second" in truncated[0]["content"][0]["text"]
  assert "first" not in str(truncated)


def test_server_compaction_durable_anchor(tmp_path) -> None:
  provider = _PortableRunnerProvider(main_turns=[_text_turn("complete")])
  durable_log = AgentSessionLog(path=tmp_path / "session.jsonl")
  _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    agent_session_log=durable_log,
  )

  rebuilt = _run(SessionContextBuilder(durable_log).build())
  assert rebuilt[0]["content"] == [{"type": "compaction", "content": ANCHOR_SUMMARY}]
  assert rebuilt[-1]["content"] == [{"type": "text", "text": "complete"}]
  normalized = CodexProvider().normalize_messages(
    rebuilt, CodexProvider().get_model_info("gpt-5.6-sol"),
  )
  assert "complete" in str(normalized)
  assert ANCHOR_SUMMARY in str(normalized)


def test_server_compaction_reactive_hedge() -> None:
  provider = _ContextLengthProvider(
    main_turns=[RuntimeError("context length exceeded"), _text_turn("retried")]
  )
  event_log = _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    compaction_trigger=10_000,
  )

  events = [entry.event for entry in event_log.entries]
  assert provider.summary_requests == 1
  assert any(event.get("type") == "compaction" for event in events)
  assert any(event.get("type") == "stream_complete" for event in events)

  second_failure = _ContextLengthProvider(
    main_turns=[RuntimeError("context length exceeded"), RuntimeError("context length exceeded again")]
  )
  second_log = _run_runner(
    second_failure,
    messages=[{"role": "user", "content": LONG_INPUT}],
    compaction_trigger=10_000,
  )
  assert second_failure.summary_requests == 1
  assert any(entry.event.get("type") == "error" for entry in second_log.entries)

  unclassified = _PortableRunnerProvider(main_turns=[RuntimeError("ordinary failure")])
  unclassified_log = _run_runner(
    unclassified,
    messages=[{"role": "user", "content": LONG_INPUT}],
    compaction_trigger=10_000,
  )
  assert unclassified.summary_requests == 0
  assert any(entry.event.get("type") == "error" for entry in unclassified_log.entries)


def test_server_compaction_provider_context_length_classifiers() -> None:
  assert CodexProvider().is_context_length_error(RuntimeError("context_length_exceeded")) is True
  assert OpenAIProvider().is_context_length_error(RuntimeError("maximum context length exceeded")) is True
  assert XAIProvider().is_context_length_error(RuntimeError("prompt too long")) is True
  assert CodexProvider().is_context_length_error(RuntimeError("ordinary failure")) is False


def test_server_compaction_matches_anthropic_block_shape() -> None:
  block = apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="model")[0]["content"][0]

  assert set(block) == {"type", "content"}
  assert block["type"] == "compaction"
  assert isinstance(block["content"], str)


def test_server_compaction_summarize_usage_reported() -> None:
  provider = _PortableRunnerProvider(main_turns=[_text_turn("complete")])
  event_log = _run_runner(provider, messages=[{"role": "user", "content": LONG_INPUT}])

  turn_complete = next(entry.event for entry in event_log.entries if entry.event.get("type") == "turn_complete")
  usage = turn_complete["usage"]
  assert usage["input_tokens"] == 31 + 13
  assert usage["cache_read_input_tokens"] == 2
  assert usage["cache_creation_input_tokens"] == 3
  assert usage["output_tokens"] == 7 + 3
  assert usage["reasoning_tokens_observed"] == 1


def test_portable_compaction_merges_anthropic_capacity_wait_once(
  monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
  import agent_gateway.providers.anthropic as adapter

  summary_usage = {
    "input_tokens": 31, "cache_read_input_tokens": 2,
    "cache_creation_input_tokens": 3, "output_tokens": 7,
  }
  first_answer_usage = {
    "input_tokens": 100, "cache_read_input_tokens": 200,
    "cache_creation_input_tokens": 30, "output_tokens": 10,
  }
  final_answer_usage = {
    "input_tokens": 20, "cache_read_input_tokens": 40,
    "cache_creation_input_tokens": 5, "output_tokens": 7,
  }

  def sdk_events(text, initial_usage, final_usage):
    return [
      SimpleNamespace(type="message_start", message=SimpleNamespace(
        usage=SimpleNamespace(**{**initial_usage, "output_tokens": 0}),
      )),
      SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
      SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)),
      SimpleNamespace(type="content_block_stop"),
      # Repeated absolute snapshots must not repeat billing or capacity wait.
      *[
        SimpleNamespace(
          type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"),
          usage=final_usage,
        )
        for _ in range(2)
      ],
    ]

  class SDKStream:
    def __init__(self, events):
      self.events = events

    async def __aenter__(self):
      return self

    async def __aexit__(self, *_args):
      return False

    async def __aiter__(self):
      for event in self.events:
        yield event

  scripts = iter([
    SDKStream(sdk_events(LONG_SUMMARY, summary_usage, SimpleNamespace(**summary_usage))),
    SDKStream(sdk_events("complete", first_answer_usage, SimpleNamespace(
      **first_answer_usage,
      iterations=[
        SimpleNamespace(**first_answer_usage),
        SimpleNamespace(**final_answer_usage),
      ],
    ))),
  ])
  requests = []

  def stream(**kwargs):
    requests.append(kwargs)
    return next(scripts)

  messages_api = SimpleNamespace(stream=stream)
  client = SimpleNamespace(messages=messages_api, beta=SimpleNamespace(messages=messages_api))
  waits = iter([1.234, 2.345])
  reservations = []

  async def acquire(*, estimated_tokens):
    reservations.append(estimated_tokens)
    return next(waits)

  # Script external SDK/capacity responses and lower the context window to
  # trigger compaction; keep request building, parsing and usage merging real.
  monkeypatch.setattr(adapter, "get_global_token_bucket", lambda: SimpleNamespace(acquire=acquire))
  provider = AnthropicProvider()
  model_info = provider.get_model_info("claude-haiku-4-5")
  monkeypatch.setattr(provider, "get_model_info", lambda _model: replace(model_info, context_window=20))
  monkeypatch.setattr(provider, "create_client", lambda *_args, **_kwargs: client)
  durable_log = AgentSessionLog(path=tmp_path / "session.jsonl")
  event_log = _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    agent_session_log=durable_log,
    effort="none",
    model="claude-haiku-4-5",
  )

  assert len(requests) == 2
  assert reservations == [SUMMARY_MAX_TOKENS, model_info.max_output_tokens]
  assert all("context_management" not in request for request in requests)
  assert LONG_INPUT in str(requests[0]["messages"])
  assert LONG_INPUT not in str(requests[1]["messages"])
  assert ANCHOR_SUMMARY in str(requests[1]["messages"])
  summaries, _ = _run(durable_log.query(event_types={"summary"}, order="asc"))
  assert len(summaries) == 1
  assert summaries[0].event["summary_kind"] == "portable_compaction"
  assert summaries[0].event["text"] == ANCHOR_SUMMARY
  compact_usage = summaries[0].event["usage"]
  assert compact_usage["provider_wait_ms"] == 1234
  for key, value in summary_usage.items():
    assert compact_usage[key] == value

  events = [entry.event for entry in event_log.entries]
  assert not [event for event in events if event.get("type") == "error"]
  assert next(event for event in events if event.get("type") == "stream_complete")["terminal_disposition"] == "completed"
  turns = [event for event in events if event.get("type") == "turn_complete"]
  assert len(turns) == 1
  usage = turns[0]["usage"]
  assert usage["provider_wait_ms"] == 1234 + 2345
  assert usage["input_tokens"] == 31 + 100 + 20
  assert usage["cache_read_input_tokens"] == 2 + 200 + 40
  assert usage["cache_creation_input_tokens"] == 3 + 30 + 5
  assert usage["output_tokens"] == 7 + 10 + 7
  # Context is the final answering invocation, not its earlier iteration or
  # the portable summary request whose billing shares this turn's payload.
  assert turns[0]["request_input_tokens"] == 20 + 40 + 5
  assert "request_input_tokens" not in usage
  expected_cost = provider.estimate_cost(
    "claude-haiku-4-5", 151, 24, cache_read_tokens=242, cache_creation_tokens=38,
  )
  assert usage["estimated_cost"] == round(expected_cost.total, 4)


def test_server_compaction_render_fallback_prompt_omits_bulk_payloads() -> None:
  rendered = render_messages_for_summary(
    [{"role": "user", "content": [{"type": "tool_result", "content": {"rows": list(range(20)), "answer": "ok"}}]}],
    char_budget=10_000,
  )
  prompt = build_summary_prompt(rendered, "Summarize.")

  assert "<omitted list items=20>" in rendered
  assert "Conversation transcript to compact" in prompt


@pytest.mark.parametrize("keep_rounds,budget", [(0, 0), (1, 1), (20, 100_000)])
def test_pending_batch_is_required_chronological_tail(keep_rounds: int, budget: int) -> None:
  batch = [
    {"role": "assistant", "provider": "codex", "content": [
      {"type": "thinking", "thinking": "retain metadata", "signature": "opaque"},
      {"type": "tool_use", "id": "a", "name": "lookup", "input": {"x": 1}},
      {"type": "tool_use", "id": "b", "name": "lookup", "input": {"x": 2}},
    ]},
    {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "a", "content": "UNDELIVERED_SENTINEL"},
      {"type": "tool_result", "tool_use_id": "b", "content": "error detail", "is_error": True},
      {"type": "text", "text": "companion context"},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "paused"}], "stop_reason": "pause_turn"},
    {"role": "user", "content": "continue"},
  ]
  messages = [{"role": "user", "content": "older"}, *batch]
  pending = {"a"}
  prefix, tail = split_messages_for_compact(
    messages, keep_rounds=keep_rounds, tail_token_budget=budget,
    preserve_tool_result_ids=pending,
  )
  assert prefix + tail == messages
  assert tail[-len(batch):] == batch
  assert pending == {"a"}
  assert sum("UNDELIVERED_SENTINEL" in str(message) for message in tail) == 1
  assert split_messages_for_compact(messages, keep_rounds=0) == (messages, [])


def test_pending_tail_supersedes_old_anchor_and_counts_tail_once(monkeypatch) -> None:
  monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 1)
  tail = [
    {"role": "assistant", "content": [
      {"type": "compaction", "content": "obsolete anchor"},
      {"type": "tool_use", "id": "a", "name": "lookup", "input": {}},
    ]},
    {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "a", "content": "UNDELIVERED_SENTINEL"},
    ]},
  ]
  messages = [{"role": "user", "content": LONG_INPUT}, *tail]
  overhead = 73
  provider = _SummaryProvider()
  result = _run(maybe_compact_current_messages(
    messages, None,
    capability_execution=stub_bound_capability_execution(provider=provider, model="model", effort="high"),
    trigger=1,
    est_total_tokens=overhead + server_compaction._estimate_messages(messages),
    preserve_tool_result_ids={"a"},
  ))
  assert result.applied
  assert result.est_after == overhead + server_compaction._estimate_messages(result.messages)
  assert "UNDELIVERED_SENTINEL" not in str(provider.requests)
  assert "obsolete anchor" in str(provider.requests)
  normalized = CodexProvider().normalize_messages(
    result.messages, CodexProvider().get_model_info("gpt-5.6-sol"),
  )
  assert "obsolete anchor" not in str(normalized)
  assert "UNDELIVERED_SENTINEL" in str(normalized)


class _NormalizedPendingProvider(_ContextLengthProvider):
  def normalize_messages(self, messages, model_info):
    return CodexProvider().normalize_messages(messages, model_info)


@pytest.mark.parametrize("continuation", ["end_turn", "refusal", "pause_turn", "max_tokens", "tool_use"])
def test_pending_result_survives_real_compaction_and_provider_normalization(
  monkeypatch, tmp_path, continuation,
) -> None:
  # Force the trigger only; splitting, summarizing, request normalization,
  # tool execution, durable replay and terminal acknowledgement stay real.
  monkeypatch.setattr(
    server_compaction, "should_portable_compact",
    lambda **kwargs: (kwargs["turn_count"] >= 2, "test_trigger"),
  )
  first = _tool_turn()
  first[1] = replace(first[1], tool_input={"query": "UNDELIVERED_SENTINEL"}, raw_block={
    "type": "tool_use", "id": "tool-1", "name": "lookup",
    "input": {"query": "UNDELIVERED_SENTINEL"},
  })
  middle = _text_turn("consuming result")
  middle[-1] = replace(middle[-1], stop_reason=continuation)
  if continuation == "tool_use":
    middle = _tool_turn()
    middle[1] = replace(middle[1], tool_id="tool-2", raw_block={
      "type": "tool_use", "id": "tool-2", "name": "lookup", "input": {"query": "AAPL"},
    })
  turns: list[list[StreamEvent] | Exception] = [first, middle]
  if continuation in {"pause_turn", "max_tokens", "tool_use"}:
    turns.append(_text_turn("finished"))
  provider = _NormalizedPendingProvider(main_turns=turns)
  durable_log = AgentSessionLog(path=tmp_path / "session.jsonl")
  event_log = _run_runner(
    provider, messages=[{"role": "user", "content": LONG_INPUT}],
    agent_session_log=durable_log, model="gpt-5.6-sol",
  )
  events = [entry.event for entry in event_log.entries]
  assert not [event for event in events if event.get("type") == "error"]
  assert next(event for event in events if event.get("type") == "stream_complete")["terminal_disposition"] == "completed"
  completed, _ = _run(durable_log.query(event_types={"tool_call_complete"}, order="asc"))
  exact_result = next(
    block for block in completed[0].event["final_tool_result_blocks"]
    if block.get("type") == "tool_result"
  )
  relevant_requests = provider.main_requests[1:3] if continuation in {"pause_turn", "max_tokens"} else provider.main_requests[1:2]
  for request in relevant_requests:
    blocks = [block for message in request if isinstance(message.get("content"), list) for block in message["content"]]
    assert sum(block.get("type") == "tool_use" and block.get("id") == exact_result["tool_use_id"] for block in blocks) == 1
    assert [block for block in blocks if block.get("type") == "tool_result" and block.get("tool_use_id") == exact_result["tool_use_id"]] == [exact_result]
  assert provider.summary_requests == len(turns) - 1
  summary_requests = [request for request in provider.requests if request["kind"] == "summary"]
  assert "UNDELIVERED_SENTINEL" not in str(summary_requests[0])
  if continuation in {"pause_turn", "max_tokens"}:
    assert "UNDELIVERED_SENTINEL" not in str(summary_requests)
    rebuilt = _run(SessionContextBuilder(durable_log).build())
    normalized = provider.normalize_messages(rebuilt, provider.get_model_info("gpt-5.6-sol"))
    assert "UNDELIVERED_SENTINEL" in str(normalized)


@pytest.mark.parametrize("failure", [False, True])
def test_reactive_compaction_preserves_pending_result_on_retry_and_error(
  monkeypatch, tmp_path, failure,
) -> None:
  monkeypatch.setattr(
    server_compaction, "should_portable_compact",
    lambda **kwargs: (kwargs["force"], "forced_context_length"),
  )
  provider = _NormalizedPendingProvider(main_turns=[
    _tool_turn(),
    RuntimeError("context length exceeded"),
    RuntimeError("context length exceeded again") if failure else _text_turn("done"),
  ])
  durable_log = AgentSessionLog(path=tmp_path / "session.jsonl")
  event_log = _run_runner(
    provider, messages=[{"role": "user", "content": LONG_INPUT}],
    agent_session_log=durable_log, model="gpt-5.6-sol",
  )
  assert provider.summary_requests == 1
  def results(messages):
    return [block for message in messages if isinstance(message.get("content"), list)
            for block in message["content"] if block.get("type") == "tool_result"]
  assert results(provider.main_requests[1]) == results(provider.main_requests[2])
  assert results(provider.main_requests[2])
  rebuilt = _run(SessionContextBuilder(durable_log).build())
  assert results(provider.normalize_messages(rebuilt, provider.get_model_info("gpt-5.6-sol"))) == results(provider.main_requests[2])
  events = [entry.event for entry in event_log.entries]
  if failure:
    assert any(event.get("type") == "error" and "context length" in event.get("error", "") for event in events)
    assert not any(event.get("type") == "stream_complete" for event in events)
  else:
    assert next(event for event in events if event.get("type") == "stream_complete")["terminal_disposition"] == "completed"
    assert not any(event.get("type") == "error" for event in events)


def test_child_portable_summary_cannot_replace_parent_context(tmp_path) -> None:
  async def case():
    durable_log = AgentSessionLog(tmp_path / "shared-log.jsonl")
    parent = await durable_log.append({
      "type": "user_message", "role": "writer", "content": "parent task",
    })
    await durable_log.append({
      "type": "summary", "role": "sub_agent", "sub_agent_id": "child",
      "summary_kind": "portable_compaction", "text": "child-only summary",
      "covers": {"from_seq": 1, "to_seq": parent.seq},
    })
    assert await SessionContextBuilder(durable_log).build() == [
      {"role": "user", "content": "parent task"},
    ]
  _run(case())


@pytest.mark.parametrize("summary_text", ["short", LONG_SUMMARY])
def test_unsummarizable_pending_work_is_never_acknowledged_or_discarded(
  monkeypatch, summary_text,
) -> None:
  messages = [
    {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "lookup", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "UNDELIVERED_SENTINEL"}]},
  ]
  provider = _SummaryProvider(summary_text=summary_text)
  pending = {"a"}
  result = _run(maybe_compact_current_messages(
    messages, None,
    capability_execution=stub_bound_capability_execution(provider=provider, model="model", effort="high"),
    trigger=1, est_total_tokens=100_000, preserve_tool_result_ids=pending,
  ))
  assert not result.applied
  assert result.reason == "insufficient_gain"
  assert result.messages == messages
  assert pending == {"a"}
  assert provider.requests == []
  monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 1)
  with_prefix = [{"role": "user", "content": LONG_INPUT}, *messages]
  result = _run(maybe_compact_current_messages(
    with_prefix, None,
    capability_execution=stub_bound_capability_execution(provider=provider, model="model", effort="high"),
    trigger=1, est_total_tokens=100_000, preserve_tool_result_ids=pending,
  ))
  if summary_text == "short":
    assert not result.applied
    assert result.reason == "summary_too_short"
    assert result.messages == with_prefix
  else:
    assert result.applied
    assert result.messages[1:] == messages
  assert pending == {"a"}
