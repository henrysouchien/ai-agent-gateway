from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from typing import Any

import pytest

from agent_workflow_contracts import (
  AgentOperationRef,
  AttemptRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  TaskResult,
  TaskResultProvenance,
  sha256_digest,
)

from agent_gateway.capability_binding import (
  CapabilityBind,
  CredentialHandle,
)
from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.event_log import EventLog
from agent_gateway.fork_request_handoff import (
  build_mid_turn_handoff,
  build_post_turn_handoff,
  credential_identity_or_none,
)
from agent_gateway.fork_scope_receipt import (
  ForkToolDecision,
  fork_scope_receipt_dict,
  parse_fork_scope_receipt,
)
from agent_gateway.providers.anthropic import AnthropicProvider
from agent_gateway.runner_fork_agents import (
  FORK_PLACEHOLDER,
  ForkPolicyDispatcher,
  _bind_learning_fork_execution,
  build_fork_messages,
  spawn_fork_agent,
)
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.model_registry import ModelRegistryEntry, ProductModelRegistry
from agent_gateway.runner_sub_agents import (
  _authoritative_child_tool_getter,
)


def _bind() -> CapabilityBind:
  return CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key="anthropic.test-sonnet",
    provider="anthropic",
    upstream_model="claude-sonnet-4-6",
    adapter="anthropic.messages",
    protocol_profile="messages.adaptive",
    route="anthropic.public",
    effort="high",
    credential_principal="user",
    credential_ref="credential-1",
    run_mode="interactive",
    registry_revision="registry-1",
    policy_revision="policy-1",
    selection_source="capability_default",
  )


def _registry_for_bind(bind: CapabilityBind) -> ProductModelRegistry:
  return ProductModelRegistry(
    schema="product-model-registry/v1",
    revision=bind.registry_revision,
    models={
      bind.model_key: ModelRegistryEntry(
        key=bind.model_key,
        label="Fork infrastructure test model",
        provider=bind.provider,
        upstream_model=bind.upstream_model,
        adapter=bind.adapter,
        protocol_profile=bind.protocol_profile,
        route=bind.route,
        lifecycle="active",
        capabilities={
          "session.driver": "user_selectable",
          "node.fork": "internal",
        },
        supported_efforts=frozenset({bind.effort}),
        default_effort=bind.effort,
        features=frozenset({"tools", "streaming"}),
        reported_identities=frozenset({bind.upstream_model}),
      )
    },
  )


def _runner() -> SimpleNamespace:
  execution = SimpleNamespace(
    bind=_bind(),
    auth_config={
      "provider": "anthropic",
      "max_tokens": 4096,
    },
    validate=lambda: None,
  )
  return SimpleNamespace(
    _capability_execution=execution,
    _gateway_session=SimpleNamespace(
      session_credential_handle=CredentialHandle(
        handle_id="credential-1",
        provider="anthropic",
        principal="user",
        tenant_id="tenant-1",
        actor_id="user-1",
      ),
      tenant_id="tenant-1",
    ),
    _billing_mode="byok",
    _last_request_system_blocks=(("static", True), ("dynamic", True)),
    _last_request_wire_tools=[
      {
        "name": "read_data",
        "description": "Read",
        "input_schema": {"type": "object"},
      },
      {
        "name": "run_agent",
        "description": "Delegate",
        "input_schema": {"type": "object"},
      },
    ],
    _last_request_message_marker_position=(0, 0),
    _last_request_max_tokens=4096,
  )


@pytest.mark.parametrize(
  ("provider", "principal", "expected"),
  [
    ("openai", "user", None),
    ("anthropic", "service", None),
    ("anthropic", "user", ("credential-1", "tenant-1")),
  ],
)
def test_session_credential_identity_requires_provider_and_principal(
  provider: str,
  principal: str,
  expected: tuple[str, str] | None,
) -> None:
  runner = _runner()
  runner._gateway_session.session_credential_handle = CredentialHandle(
    handle_id="credential-1",
    provider=provider,
    principal=principal,
    tenant_id="tenant-1",
    actor_id="user-1" if principal == "user" else None,
  )

  assert credential_identity_or_none(runner) == expected


def test_credential_identity_sources_are_all_or_nothing() -> None:
  child = _runner()
  child._gateway_session = None
  child._tenant_id = "child-tenant"
  assert credential_identity_or_none(child) == (
    "credential-1",
    "child-tenant",
  )

  autonomous = _runner()
  autonomous._gateway_session = None
  autonomous._capability_execution.auth_config.update({
    "credential_handle_id": "credential-1",
    "tenant_id": "auth-tenant",
  })
  assert credential_identity_or_none(autonomous) == (
    "credential-1",
    "auth-tenant",
  )

  mixed = _runner()
  mixed._gateway_session = None
  mixed._credential_handle_id = "child-handle"
  mixed._capability_execution.auth_config["tenant_id"] = "auth-tenant"
  assert credential_identity_or_none(mixed) is None

  guarded = _runner()
  guarded._credential_handle_id = "child-handle"
  guarded._tenant_id = "child-tenant"
  guarded._gateway_session.session_credential_handle = CredentialHandle(
    handle_id="wrong-provider",
    provider="openai",
    principal="user",
    tenant_id="tenant-1",
    actor_id="user-1",
  )
  assert credential_identity_or_none(guarded) is None


def _assistant(*tool_ids: str) -> dict[str, Any]:
  return {
    "role": "assistant",
    "content": [
      {
        "type": "tool_use",
        "id": tool_id,
        "name": "run_agent",
        "input": {"task": tool_id, "fork": True},
      }
      for tool_id in tool_ids
    ],
  }


def test_handoff_boundaries_are_complete_isolated_and_marked() -> None:
  runner = _runner()
  base = [{"role": "user", "content": [{"type": "text", "text": "seed"}]}]
  final = {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
  post = build_post_turn_handoff(runner, base, final)
  mid_messages = [*base, _assistant("fork-1")]
  mid = build_mid_turn_handoff(runner, mid_messages)

  base[0]["content"][0]["text"] = "mutated"
  runner._last_request_wire_tools[0]["name"] = "mutated"
  mid_messages[-1]["content"][0]["id"] = "mutated"

  assert post.messages[-1] == final
  assert post.messages[0]["content"][0]["text"] == "seed"
  assert mid.messages[-1]["content"][0]["id"] == "fork-1"
  assert post.wire_tools[0]["name"] == "read_data"
  assert post.message_marker_position == (0, 0)


def test_fork_tail_shares_placeholder_prefix_and_requests_normal_message() -> None:
  runner = _runner()
  handoff = build_mid_turn_handoff(
    runner,
    [
      {"role": "user", "content": "investigate"},
      _assistant("fork-1", "other-2"),
    ],
  )
  first = build_fork_messages(
    handoff,
    "first directive",
  )
  second = build_fork_messages(
    handoff,
    "second directive",
  )

  first_tail = first[-1]["content"]
  second_tail = second[-1]["content"]
  first_shared_prefix = [
    *first[:-1],
    {"role": "user", "content": first_tail[:-1]},
  ]
  second_shared_prefix = [
    *second[:-1],
    {"role": "user", "content": second_tail[:-1]},
  ]
  assert first_shared_prefix == second_shared_prefix
  assert [block["tool_use_id"] for block in first_tail[:-1]] == [
    "fork-1",
    "other-2",
  ]
  assert all(
    block["content"] == FORK_PLACEHOLDER
    for block in first_tail[:-1]
  )
  assert first_tail[-1] != second_tail[-1]
  instruction = first_tail[-1]["text"]
  assert "normal assistant message" in instruction
  assert "JSON" not in instruction


class _Dispatcher:
  def __init__(self, definitions: list[dict[str, Any]]) -> None:
    self.definitions = definitions
    self.calls: list[str] = []

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return copy.deepcopy(self.definitions)

  async def dispatch(
    self,
    _tool_call_id: str,
    tool_name: str,
    _tool_input: dict[str, Any],
    **_kwargs: Any,
  ):
    self.calls.append(tool_name)
    return {"ok": True}, None


def test_tool_getter_guard_has_strict_fork_and_nonfork_arms() -> None:
  definitions = [{"name": "read_data"}]
  getter = _authoritative_child_tool_getter(
    _Dispatcher(definitions),
    operation="fork",
  )
  assert getter() == definitions
  with pytest.raises(TypeError, match="authoritative tool catalog"):
    _authoritative_child_tool_getter(
      object(),  # type: ignore[arg-type]
      operation="fork",
    )


def test_fork_dispatch_policy_denies_run_agent_structurally() -> None:
  definitions = [
    {"name": "read_data"},
    {"name": "run_agent"},
  ]
  raw_receipt = fork_scope_receipt_dict(
    tool_decisions=(
      ForkToolDecision("read_data", "allow", "parent surface"),
      ForkToolDecision("run_agent", "deny", "orchestration surface"),
    ),
    capability_bind=_bind().model_copy(
      update={"capability_id": "node.fork"}
    ),
    tenant_id="tenant-1",
    billing_mode="byok",
    resolved_budget_usd=5.0,
    max_turns=20,
    suffix_ceiling=20_000,
  )
  dispatcher = _Dispatcher(definitions)
  policy = ForkPolicyDispatcher(
    dispatcher,
    wire_tools=definitions,
    receipt=parse_fork_scope_receipt(raw_receipt),
  )

  result, error = asyncio.run(
    policy.dispatch("call-1", "run_agent", {"task": "nested"})
  )

  assert result is None
  assert error == {
    "code": "fork_policy_denied",
    "message": (
      "fork policy: run_agent is not available in a forked child"
    ),
    "data": {
      "tool": "run_agent",
      "fork_kind": "side_quest",
      "reason": "orchestration surface",
    },
  }
  assert dispatcher.calls == []


def _message_markers(params: dict[str, Any]) -> list[tuple[int, int]]:
  return [
    (message_index, block_index)
    for message_index, message in enumerate(params["messages"])
    for block_index, block in enumerate(message.get("content") or [])
    if isinstance(block, dict) and "cache_control" in block
  ]


def test_fork_marker_stays_pinned_across_multi_turn_requests() -> None:
  provider = AnthropicProvider()
  messages = [
    {
      "role": "user",
      "content": [{"type": "text", "text": "parent boundary"}],
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "fork work"}],
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": "continue"}],
    },
  ]
  first = provider.build_request_params(
    model="claude-sonnet-4-6",
    messages=messages,
    system_prompt=[("static", True), ("dynamic", True)],
    tools=[],
    max_tokens=4096,
    fork_mode=True,
    fork_marker_position=(0, 0),
  )
  messages.extend([
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "more"}],
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": "finish"}],
    },
  ])
  second = provider.build_request_params(
    model="claude-sonnet-4-6",
    messages=messages,
    system_prompt=[("static", True), ("dynamic", True)],
    tools=[],
    max_tokens=4096,
    fork_mode=True,
    fork_marker_position=(0, 0),
  )

  assert _message_markers(first) == [(0, 0)]
  assert _message_markers(second) == [(0, 0)]


def test_fork_marker_normalizes_bare_string_like_parent_request() -> None:
  provider = AnthropicProvider()
  parent_messages = [
    {"role": "user", "content": "What is my portfolio risk?"},
  ]
  parent = provider.build_request_params(
    model="claude-sonnet-4-6",
    messages=parent_messages,
    system_prompt=[("static", True), ("dynamic", True)],
    tools=[],
    max_tokens=4096,
  )
  fork = provider.build_request_params(
    model="claude-sonnet-4-6",
    messages=[
      *parent_messages,
      {
        "role": "assistant",
        "content": [{"type": "text", "text": "Here is the analysis."}],
      },
      {"role": "user", "content": "Continue the fork."},
    ],
    system_prompt=[("static", True), ("dynamic", True)],
    tools=[],
    max_tokens=4096,
    fork_mode=True,
    fork_marker_position=(0, 0),
  )

  assert _message_markers(parent) == [(0, 0)]
  assert _message_markers(fork) == [(0, 0)]
  assert fork["messages"][0]["content"][0] == (
    parent["messages"][0]["content"][0]
  )
  assert parent_messages[0]["content"] == "What is my portfolio risk?"


class _Provider(ModelProvider):
  name = "anthropic"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return config.get("api_key") == "secret"

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider="anthropic",
      supports_thinking=True,
    )


class _SpawnRunner:
  children: list["_SpawnRunner"] = []

  def __init__(self, **kwargs: Any) -> None:
    self.kwargs = kwargs
    self._log = kwargs["event_log"]
    self._runner_id = f"runner-{kwargs['session_id']}"
    self.__class__.children.append(self)

  async def _append_durable_event(self, _event: dict[str, Any]) -> None:
    return None

  def _append(self, event: dict[str, Any]) -> None:
    self._log.append(event)

  async def run(self, **_kwargs: Any) -> None:
    self._log.append({"type": "text_delta", "text": "fork complete"})
    self._log.append({"type": "stream_complete"})

  async def force_close(self, timeout: float = 2.0) -> None:
    return None


class _ForkSessionLog:
  async def query(self, **_kwargs: Any) -> tuple[list[Any], None]:
    return [SimpleNamespace(seq=7, event={
      "type": "assistant_message",
      "stop_reason": "end_turn",
      "logical_response_id": "fork-response",
      "logical_response_segment_ordinal": 0,
      "content_blocks": [
        {"type": "text", "text": "Complete fork analysis."},
      ],
    })], None


def test_spawn_fork_reuses_resume_seed_and_fresh_tagged_log(
  tmp_path,
) -> None:
  _SpawnRunner.children.clear()
  parent = object.__new__(_SpawnRunner)
  parent._log = EventLog()
  parent._full_session_id = "parent-session"
  parent._cost_accumulator = None
  parent._per_turn_timeout = None
  parent._stream_stall_timeout = 60.0
  parent._mcp_client = None
  parent._loaded_mcp_servers = set()
  parent._on_tool_result = None
  parent._on_usage = None
  parent._on_late_usage_event = None
  parent._on_tool_timing = None
  parent._usage_user_id = "user-1"
  parent._request_id = "request-1"
  parent._rate_table_version = "v1"
  parent._channel = "web"
  parent._usage_ledger_dlq_path = None
  parent._on_metric = None
  parent._compaction_trigger = None
  parent._tool_call_timeout = None
  parent._on_max_turns = None
  parent._aggregator = None
  parent._max_concurrent_sub_agents = 4
  parent._agent_session_log = _ForkSessionLog()
  parent._max_resume_chain_depth = 0
  parent._spill_dir_provider = None
  parent._workspace_dir = str(tmp_path)
  parent._context_surfaces_provider = None
  parent._context_surfaces_static = []
  execution_bind = CapabilityBind(
    schema_version="1.0",
    capability_id="node.fork",
    model_key="anthropic.test-sonnet",
    provider="anthropic",
    upstream_model="claude-sonnet-4-6",
    adapter="anthropic.messages",
    protocol_profile="messages.adaptive",
    route="anthropic.public",
    effort="high",
    credential_principal="user",
    credential_ref="credential-1",
    run_mode="interactive",
    registry_revision="registry-1",
    policy_revision="policy-1",
    selection_source="parent_binding",
  )
  execution = BoundCapabilityExecution(
    bind=execution_bind,
    registry=_registry_for_bind(execution_bind),
    adapter=_Provider(),
    auth_config={
      "api_key": "secret",
      "provider": "anthropic",
    },
  )
  handoff = build_mid_turn_handoff(
    _runner(),
    [
      {
        "role": "user",
        "content": [{"type": "text", "text": "parent"}],
      },
      _assistant("fork-1"),
    ],
  )
  parent._capability_execution = SimpleNamespace(
    bind=handoff.capability_bind,
  )
  decisions = (
    ForkToolDecision("read_data", "allow", "parent surface"),
    ForkToolDecision("run_agent", "deny", "orchestration surface"),
  )
  receipt = fork_scope_receipt_dict(
    tool_decisions=decisions,
    capability_bind=execution.bind,
    tenant_id=handoff.tenant_id,
    billing_mode=handoff.billing_mode,
    resolved_budget_usd=5.0,
    max_turns=20,
    suffix_ceiling=20_000,
  )

  operation = AgentOperationRef(
    namespace="agent-operation",
    name="test-fork",
    version="1.0",
    digest=sha256_digest({"operation": "test-fork"}),
  )
  logical_task = OrdinaryDelegationTaskRef(
    delegation_id="test-fork-1",
    operation=operation,
  )
  attempt = AttemptRef(
    attempt_number=1,
    attempt_id="attempt:test-fork:1",
    physical_task_id="sub0:parent-session",
  )
  digest = sha256_digest({"task": "test-fork"})
  result_provenance = TaskResultProvenance(
    admitted_task_digest=digest,
    model_bind_digest=digest,
    capability_binding_digest=digest,
    tool_grant_digest=digest,
  )
  result, error = asyncio.run(
    spawn_fork_agent(
      parent,
      "finish the side quest",
      handoff=handoff,
      capability_execution=execution,
      logical_task=logical_task,
      attempt=attempt,
      result_requirement=ResultRequirement(
        mode="narrative",
        terminal_narrative="required",
        outcome=OutcomeRequirement(required=False, source="none"),
      ),
      result_provenance=result_provenance,
      dispatcher=_Dispatcher(handoff.wire_tools),
      scope_receipt=receipt,
      max_turns=20,
      max_budget_usd=5.0,
      suffix_ceiling=20_000,
    )
  )

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"
  assert result.values.terminal_narrative is not None
  assert result.values.projection is None
  child = _SpawnRunner.children[-1]
  assert child._fork_mode is True
  assert child._fork_marker_position == (0, 0)
  assert child.kwargs["emit_session_recap"] is False
  assert child.kwargs["on_session_summary"] is None
  assert all(
    entry.event["sub_agent_id"].startswith("sub0:")
    and entry.event["fork"] is True
    for entry in child._log.entries
  )


def test_dynamic_fork_bind_inherits_exact_parent_selection() -> None:
  parent_bind = _bind()
  parent = BoundCapabilityExecution(
    bind=parent_bind,
    registry=_registry_for_bind(parent_bind),
    adapter=_Provider(),
    auth_config={
      "api_key": "secret",
      "provider": "anthropic",
    },
  )

  fork = _bind_learning_fork_execution(parent)

  assert fork.bind.capability_id == "node.fork"
  assert fork.bind.provider == parent.bind.provider
  assert fork.bind.upstream_model == parent.bind.upstream_model
  assert fork.bind.effort == parent.bind.effort
  assert dict(fork.auth_config) == dict(parent.auth_config)
