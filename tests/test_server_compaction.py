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
from model_authority.binding import (
  AuthContext,
  CapabilityEffort,
  CapabilityResolutionError,
  CredentialHandle,
  ModelSelectionIntent,
  resolve_capability_model,
)
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
from model_authority.selection import CapabilityDefault, CapabilitySelectionPolicy, TierModelsDefault
from agent_gateway.capability_execution import (
  BoundCapabilityExecution,
  CapabilityExecutionResolver,
  MaterializedCredential,
)
from agent_gateway.providers import installed_adapter_providers
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
  stub_bound_capability_execution as _stub_bound_capability_execution,
)

def stub_bound_capability_execution(**kwargs: Any) -> BoundCapabilityExecution:
  """Keep these behavioral doubles eligible for exact compact inheritance."""
  execution = _stub_bound_capability_execution(**kwargs)
  resolver = execution.resolver
  assert resolver is not None
  entry = execution.registry.require(execution.bind.model_key)
  registry = replace(
    execution.registry,
    models={
      **execution.registry.models,
      entry.key: replace(entry, capabilities={**entry.capabilities, "session.compact": "internal"}),
    },
  )
  policy = replace(
    resolver.selection_policy,
    capabilities={
      **resolver.selection_policy.capabilities,
      "session.compact": CapabilitySelectionPolicy(
        capability_id="session.compact",
        default=CapabilityDefault(kind="inherit_parent"),
        by_channel={},
        allowed_model_keys=frozenset({entry.key}),
      ),
    },
  )
  resolver = replace(
    resolver,
    registry=registry,
    selection_policy=policy,
    auth_context=replace(
      resolver.auth_context,
      entitled_capabilities=resolver.auth_context.entitled_capabilities | {"session.compact"},
    ),
  )
  return resolver.materialize_bind(execution.bind)



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
  capability_execution: BoundCapabilityExecution | None = None,
) -> EventLog:
  log = event_log or EventLog()
  capability_execution = capability_execution or stub_bound_capability_execution(
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


@pytest.mark.parametrize(
  ("model_key", "effort"),
  [
    (model_key, effort)
    for model_key in sorted({
      key
      for slot, policy in INITIAL_MODEL_SELECTION_POLICY.capabilities.items()
      if slot in {"session.driver", "plan.author"} or slot.startswith("node.")
      for key in policy.allowed_model_keys
    })
    for effort in sorted(INITIAL_MODEL_REGISTRY.require(model_key).supported_efforts)
  ],
)
@pytest.mark.parametrize(
  ("principal", "run_mode"),
  [("user", "interactive"), ("service", "autonomous")],
)
@pytest.mark.parametrize("rendered", [False, True])
def test_portable_summary_inherits_packaged_parent_execution_whole(
  monkeypatch: pytest.MonkeyPatch,
  model_key: str,
  effort: CapabilityEffort,
  principal,
  run_mode,
  rendered: bool,
) -> None:
  entry = INITIAL_MODEL_REGISTRY.require(model_key)
  handle = CredentialHandle(
    handle_id=f"compact-test:{principal}:{entry.provider}",
    provider=entry.provider,
    principal=principal,
    tenant_id="tenant",
    actor_id="alice" if principal == "user" else None,
  )
  provider = installed_adapter_providers()[entry.adapter]()
  resolver = CapabilityExecutionResolver(
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth_context=AuthContext(
      run_mode=run_mode,
      actor_id="alice",
      tenant_id="tenant",
      user_provider_handles={entry.provider: handle} if principal == "user" else {},
      service_provider_handles={entry.provider: handle} if principal == "service" else {},
      entitled_capabilities=CAPABILITY_IDS,
      entitled_model_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
    ),
    credential_materializer=lambda selected: MaterializedCredential(
      handle=selected,
      auth_config={
        "api_key": "test-secret",
        "auth_token": "test-token",
        "project_id": "test-project",
        "auth_mode": "api",
        "compat": entry.compat,
      },
    ),
    adapter_resolver=lambda adapter: provider,
  )
  if entry.capabilities.get("session.driver") == "user_selectable":
    parent_bind = resolver.select_bind(
      "session.driver",
      explicit_intent=ModelSelectionIntent(
        model_key=model_key, effort=effort, source="explicit_user",
      ),
    )
  else:
    slot = next(
      name for name in (
        "session.driver", "plan.author", "node.explore", "node.implement",
        "node.mutate", "node.verify", "node.choose", "node.fork",
      )
      if model_key in resolver.selection_policy.capabilities[name].allowed_model_keys
    )
    slot_default = resolver.selection_policy.capabilities[slot].default
    if slot_default.kind != "tier" or slot_default.tier is None:
      raise RuntimeError(f"{slot} default is not a tier")
    role = slot_default.tier
    parent_bind = resolve_capability_model(
      slot,
      registry=resolver.registry,
      selection_policy=resolver.selection_policy,
      auth=resolver.auth_context,
      requested_effort=effort,
      requested_role=role,
      role_map={role: TierModelsDefault((model_key,), effort)},
      role_map_source="internal_policy",
    )
  # A later registry/policy load must not rewrite the parent's frozen provenance.
  parent_bind = parent_bind.model_copy(update={
    "registry_revision": "parent-registry.1",
    "policy_revision": "parent-policy.1",
  })
  parent = resolver.materialize_bind(parent_bind)
  calls: list[dict[str, Any]] = []

  async def _summarize(**kwargs: Any):
    calls.append(kwargs)
    return SimpleNamespace(text=LONG_SUMMARY, usage={"input_tokens": 17}, saw_tool_use=False)

  monkeypatch.setattr(server_compaction, "provider_summarize", _summarize)
  monkeypatch.setattr(server_compaction, "RENDER_CHAR_BUDGET", 0 if rendered else 100_000)
  original_bind = parent.bind.to_json()
  summary, usage = _run(summarize_messages(
    [{"role": "user", "content": "Preserve the approved implementation."}],
    "Summarize exactly.",
    capability_execution=parent,
  ))

  assert summary == ANCHOR_SUMMARY
  assert usage == {"input_tokens": 17}
  assert len(calls) == 1
  compact = calls[0]["capability_execution"]
  assert compact.bind.to_json() == {
    **original_bind,
    "capability_id": "session.compact",
    "selection_source": "parent_binding",
  }
  assert parent.bind.to_json() == original_bind
  assert compact.registry is parent.registry
  assert compact.adapter is parent.adapter
  assert compact.auth_config == parent.auth_config
  assert compact.resolver is resolver
  assert calls[0]["tools"] == []
  assert calls[0]["max_tokens"] == SUMMARY_MAX_TOKENS


@pytest.mark.parametrize("missing_resolver", [False, True])
def test_runner_compact_resolution_failure_preserves_original_history(
  missing_resolver: bool,
) -> None:
  provider = _PortableRunnerProvider(main_turns=[_text_turn("complete")])
  execution = stub_bound_capability_execution(
    provider=provider, model="model", effort="high",
    auth_config={"api_key": "test", "max_tokens": 512},
  )
  resolver = execution.resolver
  assert resolver is not None
  execution = replace(
    execution,
    resolver=None if missing_resolver else replace(
      resolver,
      auth_context=replace(
        resolver.auth_context,
        entitled_capabilities=resolver.auth_context.entitled_capabilities - {"session.compact"},
      ),
    ),
  )
  messages = [{"role": "user", "content": LONG_INPUT}]
  original_messages = [dict(message) for message in messages]
  log = EventLog()
  if missing_resolver:
    expected_error = ValueError
    expected_message = "requires a CapabilityExecutionResolver"
  else:
    expected_error = CapabilityResolutionError
    expected_message = "session.compact"
  with pytest.raises(expected_error, match=expected_message):
    _run_runner(
      provider, messages=messages, event_log=log, capability_execution=execution,
    )
  assert messages == original_messages
  assert provider.summary_requests == 0
  assert provider.main_requests == []
  assert not any(entry.event.get("type") == "compaction" for entry in log.entries)


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


def test_runner_turn_completes_when_the_compaction_summary_is_empty() -> None:
  # Canary #11: an empty provider summary ended the turn; it is a skipped compaction.
  provider = _PortableRunnerProvider(
    main_turns=[_tool_turn(), _text_turn("complete")], summary_text="",
  )
  event_log = _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    effort="medium",
  )

  events = [entry.event for entry in event_log.entries]
  assert provider.summary_requests >= 1
  assert not any(event.get("type") == "compaction" for event in events)
  assert any(event.get("type") == "tool_call_complete" for event in events)
  assert any(event.get("type") == "turn_complete" for event in events)


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


@pytest.mark.parametrize("session_id", ["sess-0123456789abcdef-full-session-id", None])
@pytest.mark.parametrize("rendered", [False, True])
def test_portable_compaction_hands_the_full_session_id_to_the_summary_call(
  monkeypatch, session_id: str | None, rendered: bool,
) -> None:
  monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 1)
  monkeypatch.setattr(server_compaction, "RENDER_CHAR_BUDGET", 0 if rendered else 100_000)
  calls: list[dict[str, Any]] = []

  async def _summarize(**kwargs: Any):
    calls.append(kwargs)
    return SimpleNamespace(text=LONG_SUMMARY, usage={"input_tokens": 17}, saw_tool_use=False)

  monkeypatch.setattr(server_compaction, "provider_summarize", _summarize)
  messages = [{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": "ok"}]
  result = _run(maybe_compact_current_messages(
    messages, None,
    capability_execution=stub_bound_capability_execution(
      provider=_SummaryProvider(), model="model", effort="high",
    ),
    trigger=1,
    est_total_tokens=100_000,
    session_id=session_id,
  ))

  assert result.applied
  assert [call["session_id"] for call in calls] == [session_id]


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


@pytest.mark.parametrize("summary_text", ["", "short", LONG_SUMMARY])
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
  if summary_text in ("", "short"):
    assert not result.applied
    assert result.reason == "summary_too_short"
    assert result.messages == with_prefix
  else:
    assert result.applied
    assert result.messages[1:] == messages
  assert pending == {"a"}


@pytest.mark.parametrize("empty_summary", ["", "   \n\t  "])
def test_apply_compaction_anchor_and_truncate_omit_empty_summary_block(empty_summary: str) -> None:
  from agent_gateway.server_compaction import apply_compaction_anchor

  tail = [{"role": "user", "content": "retained question"}]
  assert apply_compaction_anchor(tail, empty_summary, provider="ollama_cloud", model="m") == tail
  assert apply_compaction_anchor([], empty_summary, provider="ollama_cloud", model="m") == [
    {"role": "user", "content": CONTINUATION_NUDGE},
  ]

  with_empty_anchor = [
    {"role": "user", "content": "old"},
    {"role": "assistant", "content": [{"type": "compaction", "content": empty_summary}]},
    {"role": "user", "content": "next"},
  ]
  assert truncate_to_last_compaction(with_empty_anchor, compaction_as_text=False) == [
    {"role": "assistant", "content": [{"type": "compaction", "content": empty_summary}]},
    {"role": "user", "content": "next"},
  ]
  assert truncate_to_last_compaction(with_empty_anchor, compaction_as_text=True) == [
    {"role": "user", "content": "next"},
  ]


def test_summarize_messages_places_root_deliverable_first_and_strips_skill_summary_narrowing() -> None:
  narrowed_summary = (
    "<analysis>scratch</analysis><summary>"
    "# Task\n"
    "Build a valuation model for **PAYC — Paycom Software, Inc.** (NYSE: PAYC). "
    "Work is the `build-model` chain (v3.0). "
    "The deliverable of the **current stage** is a **critical-factors judgment emitted to "
    "`fms_propose_critical_factors`**, with a short `## Skill Summary` receipt as the *only* chat text.\n\n"
    "## Chain state\n"
    "- FAS accepted (`fas_cc6a7ccfa21545b7a01c44adbb75f854`), next stage scenario_set."
    "</summary>"
  )
  provider = _SummaryProvider(summary_text=narrowed_summary)
  messages = [
    {
      "role": "user",
      "content": "Build me a model for PAYC.\n\n## Current Session Context\n```json\n{\"channel\": \"cli\"}\n```",
    },
    {
      "role": "assistant",
      "content": [
        {"type": "tool_use", "id": "call_1", "name": "invoke_skill", "input": {"skill_name": "critical-factors"}},
      ],
    },
    {
      "role": "user",
      "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "skill loaded"},
      ],
    },
  ]
  summary, _usage = _run(
    summarize_messages(
      messages,
      None,
      capability_execution=stub_bound_capability_execution(provider=provider, model="model", effort="high"),
    )
  )
  assert summary.splitlines()[0] == "Root deliverable: Build me a model for PAYC."
  assert "only* chat text" not in summary
  assert "only chat text" not in summary

  # Subsequent compaction over the compacted anchor preserves the exact root deliverable first line:
  second_messages = [
    {"role": "assistant", "content": [{"type": "compaction", "content": summary}], "stop_reason": "compaction"},
    {"role": "user", "content": CONTINUATION_NUDGE},
  ]
  summary2, _ = _run(
    summarize_messages(
      second_messages,
      None,
      capability_execution=stub_bound_capability_execution(provider=provider, model="model", effort="high"),
    )
  )
  assert summary2.splitlines()[0] == "Root deliverable: Build me a model for PAYC."


def test_strip_analysis_salvages_an_analysis_the_output_budget_cut_off() -> None:
  strip = server_compaction._strip_analysis_and_extract_summary
  analysis = "Reviewed the transcript and the open obligations. " * 40

  assert len(analysis) > MIN_SUMMARY_CHARS
  assert strip(f"<analysis>{analysis}</analysis><summary>") == analysis.strip()
  assert strip(f"<analysis>{analysis}") == analysis.strip()
  assert strip(f"<analysis>{analysis}</analysis><summary>tail</summary>") == f"{analysis.strip()}\n\ntail"


def test_strip_analysis_drops_scratchpad_when_the_summary_is_long_enough() -> None:
  strip = server_compaction._strip_analysis_and_extract_summary
  filler = "scratch " * 100

  assert strip(f"<analysis>{filler}</analysis><summary>{LONG_SUMMARY}</summary>") == LONG_SUMMARY.strip()
  assert strip(f"<summary>{LONG_SUMMARY}</summary><analysis>{filler} cut off") == LONG_SUMMARY.strip()
  assert strip("<analysis>short</analysis><summary>tiny</summary>") == "tiny"


def test_strip_analysis_keeps_a_literal_analysis_tag_inside_the_summary() -> None:
  strip = server_compaction._strip_analysis_and_extract_summary
  body = (
    "Context: the template has an <analysis> tab the model must fill. " * 6
    + "Next steps: finish DCF."
  )
  assert len(body) > MIN_SUMMARY_CHARS

  assert strip(f"<summary>{body}</summary>") == body
  assert strip(f"<analysis>scratch</analysis><summary>{body}</summary>") == body
  assert strip(f"<analysis>cut off before the summary <summary>{body}</summary>") == body


class _ThinkingOnlyProvider(_SummaryProvider):
  """A reasoning model that spends its output budget thinking and is cut off."""

  def __init__(self, *, thinking: str, text: str = "") -> None:
    super().__init__()
    self.thinking = thinking
    self.text = text

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(type="message_start", input_tokens=31, cache_read_tokens=2, cache_creation_tokens=3)
    yield StreamEvent(type="thinking_delta", thinking_text=self.thinking)
    yield StreamEvent(type="thinking_end", raw_block={"type": "thinking", "thinking": self.thinking})
    if self.text:
      yield StreamEvent(type="text_delta", text=self.text)
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": self.text})
    yield StreamEvent(type="usage_update", output_tokens=SUMMARY_MAX_TOKENS)
    yield StreamEvent(type="message_end", stop_reason="max_tokens")


def test_summary_falls_back_to_thinking_when_the_model_never_writes_enough_text() -> None:
  thinking = LONG_SUMMARY * 2
  provider = _ThinkingOnlyProvider(thinking=thinking, text="<summary>")
  messages = [{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": "ok"}]

  summary, usage = _run(summarize_messages(
    messages, None,
    capability_execution=stub_bound_capability_execution(provider=provider, model="model", effort="high"),
  ))

  assert summary == thinking.strip()
  assert usage["output_tokens"] == SUMMARY_MAX_TOKENS


def test_summary_prefers_written_text_over_thinking() -> None:
  provider = _ThinkingOnlyProvider(thinking=LONG_SUMMARY * 2, text=f"<summary>{LONG_SUMMARY}</summary>")
  messages = [{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": "ok"}]

  summary, _usage = _run(summarize_messages(
    messages, None,
    capability_execution=stub_bound_capability_execution(provider=provider, model="model", effort="high"),
  ))

  assert summary == LONG_SUMMARY.strip()


def test_compaction_with_too_little_thinking_and_text_is_summary_too_short(monkeypatch) -> None:
  monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 1)
  messages = [{"role": "user", "content": LONG_INPUT}, {"role": "assistant", "content": "ok"}]
  provider = _ThinkingOnlyProvider(thinking="short reasoning", text="<summary>")
  result = _run(maybe_compact_current_messages(
    messages, None,
    capability_execution=stub_bound_capability_execution(provider=provider, model="model", effort="high"),
    trigger=1,
    est_total_tokens=100_000,
  ))
  assert not result.applied
  assert result.reason == "summary_too_short"
  assert result.messages == messages

  provider = _ThinkingOnlyProvider(thinking=LONG_SUMMARY, text="<summary>")
  result = _run(maybe_compact_current_messages(
    messages, None,
    capability_execution=stub_bound_capability_execution(provider=provider, model="model", effort="high"),
    trigger=1,
    est_total_tokens=100_000,
  ))
  assert result.applied
  assert result.anchor_block == {"type": "compaction", "content": LONG_SUMMARY.strip()}


def _skill_turn_messages(first_request: str, skill: str) -> list[dict[str, Any]]:
  return [
    {"role": "user", "content": f"{first_request}\n\n## Current Session Context\n```json\n{{}}\n```"},
    {"role": "assistant", "content": [
      {"type": "tool_use", "id": f"call_{skill}", "name": "invoke_skill", "input": {"skill_name": skill}},
    ]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"call_{skill}", "content": "skill loaded"}]},
  ]


def test_root_deliverable_is_the_latest_user_turns_request() -> None:
  anchor = {
    "role": "assistant",
    "content": [{"type": "compaction", "content": "Root deliverable: Build me a model for PAYC.\n\n# Task\nBuild PAYC"}],
    "stop_reason": "compaction",
  }
  messages = [
    anchor,
    *_skill_turn_messages("Build me a model for PAYC.", "build-model"),
    {"role": "assistant", "content": "model built"},
    {"role": "user", "content": CONTINUATION_NUDGE},
    {"role": "user", "content": "Now draft a memo comparing ADP and PAYX margins.\n\n## Current Session Context\n```json\n{}\n```"},
    {"role": "assistant", "content": "working"},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_x", "content": "rows"}]},
    {"role": "user", "content": "You stopped with 2 checklist items still open:\n- a\n- b"},
  ]

  assert server_compaction._extract_root_deliverable(messages) == "Now draft a memo comparing ADP and PAYX margins."
  # With no user prompt left, the anchor's own header stands for the compacted request:
  assert server_compaction._extract_root_deliverable(
    [anchor, {"role": "user", "content": CONTINUATION_NUDGE}]
  ) == "Build me a model for PAYC."


def test_runner_injected_user_messages_are_not_the_analysts_request() -> None:
  from agent_gateway.runner_run_loop_defaults import MAX_TOKENS_NUDGE

  def tool_use(call_id: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": "invoke_skill", "input": {"skill_name": "build-model"}}]}

  cut_off = {"role": "assistant", "content": "partial", "stop_reason": "max_tokens"}
  messages = [
    {"role": "user", "content": "Build a DCF for AAPL"},
    cut_off,
    {"role": "user", "content": MAX_TOKENS_NUDGE},
    tool_use("call_1"),
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}]},
  ]
  assert server_compaction._extract_root_deliverable(messages) == "Build a DCF for AAPL"

  # When a turn ends on max_tokens after exhausting continuations, the next real analyst request
  # starts a new turn rather than being skipped as an open-turn continuation.
  after_exhausted_max_tokens = [
    {"role": "user", "content": "Build a DCF for AAPL"},
    tool_use("call_1"),
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}]},
    cut_off,
    {"role": "user", "content": MAX_TOKENS_NUDGE},
    cut_off,
    {"role": "user", "content": MAX_TOKENS_NUDGE},
    cut_off,
    {"role": "user", "content": MAX_TOKENS_NUDGE},
    cut_off,
    {"role": "user", "content": "Write a short memo on MSFT margins"},
    {"role": "assistant", "content": "draft memo", "stop_reason": "end_turn"},
  ]
  assert server_compaction._latest_user_prompt_index(after_exhausted_max_tokens) == 10
  assert server_compaction._extract_root_deliverable(after_exhausted_max_tokens) == "Write a short memo on MSFT margins"
  assert server_compaction._message_has_inline_skill_or_stage_narrowing(after_exhausted_max_tokens, "summary") is False
  # The skill invoked before the injected message still narrows the summary.
  with_skill_first = [
    {"role": "user", "content": "Build a DCF for AAPL"},
    tool_use("call_1"),
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}]},
    {"role": "assistant", "content": "partial", "stop_reason": "max_tokens"},
    {"role": "user", "content": "Continue from where you left off."},
  ]
  assert server_compaction._message_has_inline_skill_or_stage_narrowing(with_skill_first, "summary") is True
  for text in (
    "<background-task-result>x</background-task-result>",
    "<operator-note>x</operator-note>",
    "<task-notification>x</task-notification>",
    "[System: background result]",
  ):
    assert server_compaction.is_runner_injected_user_text(text) is True
  assert server_compaction.is_runner_injected_user_text("Build a DCF for AAPL") is False


def test_earlier_turns_skill_and_header_do_not_leak_into_a_later_turns_summary() -> None:
  messages = [
    *_skill_turn_messages("Build me a model for PAYC.", "build-model"),
    {"role": "assistant", "content": "model built"},
    {"role": "user", "content": "Different question: draft a two-page memo comparing ADP and PAYX margins."},
    {"role": "assistant", "content": "researching"},
  ]
  provider = _SummaryProvider(summary_text=f"<summary>{LONG_SUMMARY}</summary>")
  execution = stub_bound_capability_execution(provider=provider, model="model", effort="high")

  summary, _ = _run(summarize_messages(messages, None, capability_execution=execution))
  assert summary == LONG_SUMMARY.strip()

  # A summary that does carry a skill's narrowing is pinned to the current request, not the first:
  narrowed = f"<summary># Task\nMemo\n\nThe current stage ends with a `## Skill Summary`.\n{LONG_SUMMARY}</summary>"
  narrowed_provider = _SummaryProvider(summary_text=narrowed)
  summary, _ = _run(summarize_messages(
    messages, None,
    capability_execution=stub_bound_capability_execution(provider=narrowed_provider, model="model", effort="high"),
  ))
  assert summary.splitlines()[0] == (
    "Root deliverable: Different question: draft a two-page memo comparing ADP and PAYX margins."
  )

  # An earlier anchor's header is not carried across a new request:
  after_anchor = [
    {"role": "assistant", "content": [{"type": "compaction", "content": "Root deliverable: Build me a model for PAYC.\n\n# Task\nx"}]},
    {"role": "user", "content": "Now draft the ADP memo."},
    {"role": "assistant", "content": "ok"},
  ]
  summary, _ = _run(summarize_messages(after_anchor, None, capability_execution=execution))
  assert summary == LONG_SUMMARY.strip()


def _tool_turn_reporting_real_input(request_input_tokens: int | None) -> list[StreamEvent]:
  turn = _tool_turn()
  turn[-1] = StreamEvent(type="message_end", stop_reason="tool_use", request_input_tokens=request_input_tokens)
  return turn


@pytest.mark.parametrize(
  ("real_input_tokens", "expected_summary_requests"),
  [(200, 0), (None, 1)],
)
def test_real_provider_input_anchors_the_compaction_estimate_when_chars_overshoot(
  monkeypatch: pytest.MonkeyPatch,
  real_input_tokens: int | None,
  expected_summary_requests: int,
) -> None:
  import agent_gateway.runner as gateway_runner
  from agent_gateway.runner_limits import TokenEstimateSnapshot

  def estimate(*, system_text: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> TokenEstimateSnapshot:
    total = 1000 + 400 * len(messages)
    return TokenEstimateSnapshot(
      system_text="", messages_text="", tools_text="",
      system_chars=0, tools_chars=0,
      est_system_tokens=0, est_messages_tokens=total, est_tools_tokens=0,
      est_total_tokens=total, message_count=len(messages), tool_count=len(tools),
    )

  monkeypatch.setattr(gateway_runner, "_token_estimate_snapshot", estimate)
  # Turn 1 estimates 1400 (< trigger 1500) and the provider reports 200 real input
  # tokens. Turn 2 estimates 2200: the old max(est, real) compacted at once; the
  # anchored estimate is 200 + (2200 - 1400) = 1000, below the trigger.
  provider = _PortableRunnerProvider(
    main_turns=[_tool_turn_reporting_real_input(real_input_tokens), _text_turn("complete")],
  )
  _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    compaction_trigger=1500,
    effort="medium",
  )

  assert provider.summary_requests == expected_summary_requests


def test_summary_too_short_holds_the_next_turns_back_until_the_estimate_grows(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import agent_gateway.runner as gateway_runner
  from agent_gateway.runner_limits import TokenEstimateSnapshot

  def estimate(*, system_text: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> TokenEstimateSnapshot:
    total = 2000 + 50 * len(messages)
    return TokenEstimateSnapshot(
      system_text="", messages_text="", tools_text="",
      system_chars=0, tools_chars=0,
      est_system_tokens=0, est_messages_tokens=total, est_tools_tokens=0,
      est_total_tokens=total, message_count=len(messages), tool_count=len(tools),
    )

  monkeypatch.setattr(gateway_runner, "_token_estimate_snapshot", estimate)
  results: list[Any] = []
  real_compact = runner_run_loop.maybe_compact_current_messages

  async def spy(*args: Any, **kwargs: Any) -> Any:
    result = await real_compact(*args, **kwargs)
    results.append(result)
    return result

  monkeypatch.setattr(runner_run_loop, "maybe_compact_current_messages", spy)
  provider = _PortableRunnerProvider(
    main_turns=[_tool_turn(), _tool_turn(), _tool_turn(), _text_turn("complete")],
    summary_text="short",
  )

  _run_runner(
    provider,
    messages=[{"role": "user", "content": LONG_INPUT}],
    compaction_trigger=1500,
    effort="medium",
  )

  # Turn 1 fails with summary_too_short; each later turn grows the estimate by
  # 200 tokens, under MIN_COMPACT_GAIN_TOKENS (500 here), so none summarizes again.
  assert provider.summary_requests == 1
  assert [result.reason for result in results] == [
    "summary_too_short", "failed_cooldown", "failed_cooldown", "failed_cooldown",
  ]


def test_should_portable_compact_gates_on_last_compact_failed_turn() -> None:
  should_compact, reason = server_compaction.should_portable_compact(
    est_total_tokens=200_000,
    est_summarize_tokens=50_000,
    trigger=100_000,
    native_compaction=False,
    turn_count=4,
    last_compact_failed_turn=4,
  )
  assert (should_compact, reason) == (False, "failed_cooldown")
