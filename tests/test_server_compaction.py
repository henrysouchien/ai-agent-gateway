# ruff: noqa: E402

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from agent_gateway import (
  AgentRunner,
  AgentSessionLog,
  EffortResolution,
  EventLog,
  ModelInfo,
  ModelProvider,
  ToolDispatcher,
)
from agent_gateway.providers import AnthropicProvider, CodexProvider, OpenAIProvider, StreamEvent, ThinkingLevel, XAIProvider
from agent_gateway.providers.base import truncate_to_last_compaction
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
from tests.capability_execution_test_support import (
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


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


def _make_dispatcher(
  *,
  event_log: EventLog | None = None,
  local_tool_handlers: dict[str, Any] | None = None,
) -> ToolDispatcher:
  return ToolDispatcher(
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
  max_turns: int | None = None,
  compaction_trigger: int = 20,
  effort: str = "high",
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
    dispatcher=_make_dispatcher(event_log=log, local_tool_handlers={"lookup": _lookup_tool}),
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
    _run(runner.run(messages, max_turns=max_turns))
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
  model_info = provider.get_model_info("gpt-5.5")
  normalized = provider.normalize_messages(
    [{"role": "user", "content": "old"}, *apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="gpt-5.5")],
    model_info,
  )

  assert normalized[0]["content"][0]["type"] == "text"
  assert "Summary of the earlier conversation" in normalized[0]["content"][0]["text"]
  assert all(
    not (isinstance(block, dict) and block.get("type") == "compaction")
    for message in normalized
    for block in (message.get("content") if isinstance(message.get("content"), list) else [])
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
    *apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="gpt-5.5"),
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "missing-tool", "content": "orphan"}]},
  ]

  codex_normalized = CodexProvider().normalize_messages(messages, CodexProvider().get_model_info("gpt-5.5"))
  openai_normalized = OpenAIProvider().normalize_messages(messages, OpenAIProvider().get_model_info("gpt-5.6"))

  assert codex_normalized[-1]["content"][0]["tool_use_id"] == "missing-tool"
  assert openai_normalized[-1]["content"][0]["tool_use_id"] == "missing-tool"


def test_server_compaction_anchor_replay_anthropic() -> None:
  provider = AnthropicProvider()
  normalized = provider.normalize_messages(
    [{"role": "user", "content": "old"}, *apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="model")],
    provider.get_model_info("claude-sonnet-4-6"),
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
  messages = apply_compaction_anchor([], LONG_SUMMARY, provider="codex", model="gpt-5.5")
  normalized = CodexProvider().normalize_messages(messages, CodexProvider().get_model_info("gpt-5.5"))

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
    max_turns=3,
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
    max_turns=1,
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
    max_turns=1,
    effort="none",
    model="claude-sonnet-4-6",
  )

  assert provider.summary_requests == 0
  main_request = next(request for request in provider.requests if request["kind"] == "main")
  assert main_request["context_management"]["edits"][0]["type"] == "compact_20260112"


def test_server_compaction_one_message_merge(tmp_path) -> None:
  provider = _PortableRunnerProvider(main_turns=[_tool_turn(), _text_turn("complete")])
  durable_log = AgentSessionLog(path=tmp_path / "session.jsonl")
  _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    agent_session_log=durable_log,
    max_turns=3,
  )

  assistant_entries, _ = _run(durable_log.query(event_types={"assistant_message"}, order="asc"))
  compacted = [
    entry.event for entry in assistant_entries
    if entry.event.get("content_blocks", [{}])[0].get("type") == "compaction"
  ]

  assert len(compacted) == 1
  content_blocks = compacted[0]["content_blocks"]
  assert content_blocks[0] == {"type": "compaction", "content": ANCHOR_SUMMARY}
  assert content_blocks[1]["type"] == "tool_use"
  assert CONTINUATION_NUDGE not in str(compacted)


def test_server_compaction_anchor_persist_on_stream_failure(tmp_path) -> None:
  provider = _PortableRunnerProvider(main_turns=[RuntimeError("stream exploded")])
  durable_log = AgentSessionLog(path=tmp_path / "session.jsonl")
  _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    agent_session_log=durable_log,
    max_turns=1,
  )

  assistant_entries, _ = _run(durable_log.query(event_types={"assistant_message"}, order="asc"))
  anchor_event = assistant_entries[-1].event
  assert anchor_event["stop_reason"] == "compaction"
  assert anchor_event["content_blocks"] == [{"type": "compaction", "content": ANCHOR_SUMMARY}]
  assert anchor_event["usage"]["input_tokens"] == 31
  assert anchor_event["usage"]["output_tokens"] == 7

  resume_provider = _PortableRunnerProvider(main_turns=[_tool_turn(), _text_turn("resumed")])
  anchor_message = {
    "role": "assistant",
    "content": anchor_event["content_blocks"],
    "provider": "codex",
    "model": "model",
    "stop_reason": "compaction",
  }
  _run_runner(resume_provider, messages=[anchor_message], max_turns=2)
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
    max_turns=1,
  )

  assistant_entries, _ = _run(durable_log.query(event_types={"assistant_message"}, order="asc"))
  content_blocks = assistant_entries[-1].event["content_blocks"]
  rebuilt = [{
    "role": "assistant",
    "content": content_blocks,
    "provider": "codex",
    "model": "model",
    "stop_reason": assistant_entries[-1].event["stop_reason"],
  }]

  assert content_blocks[0] == {"type": "compaction", "content": ANCHOR_SUMMARY}
  assert truncate_to_last_compaction([{"role": "user", "content": "old"}, *rebuilt], compaction_as_text=True)[0]["role"] == "assistant"


def test_server_compaction_reactive_hedge() -> None:
  provider = _ContextLengthProvider(
    main_turns=[RuntimeError("context length exceeded"), _text_turn("retried")]
  )
  event_log = _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    max_turns=1,
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
    max_turns=1,
    compaction_trigger=10_000,
  )
  assert second_failure.summary_requests == 1
  assert any(entry.event.get("type") == "error" for entry in second_log.entries)

  unclassified = _PortableRunnerProvider(main_turns=[RuntimeError("ordinary failure")])
  unclassified_log = _run_runner(
    unclassified,
    messages=[{"role": "user", "content": LONG_INPUT}],
    max_turns=1,
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
  event_log = _run_runner(provider, messages=[{"role": "user", "content": LONG_INPUT}], max_turns=1)

  turn_complete = next(entry.event for entry in event_log.entries if entry.event.get("type") == "turn_complete")
  usage = turn_complete["usage"]
  assert usage["input_tokens"] == 31 + 13
  assert usage["cache_read_input_tokens"] == 2
  assert usage["cache_creation_input_tokens"] == 3
  assert usage["output_tokens"] == 7 + 3
  assert usage["reasoning_tokens_observed"] == 1


def test_server_compaction_render_fallback_prompt_omits_bulk_payloads() -> None:
  rendered = render_messages_for_summary(
    [{"role": "user", "content": [{"type": "tool_result", "content": {"rows": list(range(20)), "answer": "ok"}}]}],
    char_budget=10_000,
  )
  prompt = build_summary_prompt(rendered, "Summarize.")

  assert "<omitted list items=20>" in rendered
  assert "Conversation transcript to compact" in prompt
