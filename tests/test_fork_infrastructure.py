from __future__ import annotations
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND

import asyncio
import copy
import json
import logging
import time
from dataclasses import replace
from decimal import Decimal
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from agent_workflow_contracts import (
  AgentOperationRef,
  AgentOperationSnapshot,
  AttemptRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  SemanticCapabilityRequirement,
  TaskResult,
  sha256_digest,
)

from agent_gateway import AgentRunner, AgentSessionLog, runner_sub_agents
from agent_gateway.mcp_activation import McpActivationFold
from model_authority.bind import CapabilityBind
from model_authority.binding import AuthContext, CredentialHandle, CredentialPrincipal
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.selection import (
  CapabilityDefault,
  CapabilitySelectionPolicy,
  ProductModelSelectionPolicy,
)
from agent_gateway.capability_execution import (
  BoundCapabilityExecution,
  CapabilityExecutionResolver,
  MaterializedCredential,
)
from agent_gateway.event_log import EventLog
from tests.host_timing_test_support import harness_wait_seconds
from agent_gateway.context_builder import SessionContextBuilder
from agent_gateway.execution_identity import resolved_execution_identity
from agent_gateway.fork_ledger import ForkLedger
from agent_gateway.learning_fork_trigger import _runtime_for_runner
from agent_gateway.operation_catalog import ResolvedOperationRuntime
from agent_gateway.operation_snapshot import (
  WorkspaceScope,
  build_agent_operation_snapshot,
)
from agent_gateway.containment_settings import ContainmentSettings
from agent_gateway.runner_limits import ContextPressureReminderDecision
from agent_gateway.fork_request_handoff import (
  build_mid_turn_handoff,
  build_post_turn_handoff,
  credential_identity_or_none,
  fork_credential_identity_available,
)
from agent_gateway.providers.anthropic import AnthropicProvider
from agent_gateway.runner_fork_agents import (
  FORK_PLACEHOLDER,
  LearningForkConfig,
  LearningForkWorkItem,
  build_fork_messages,
  spawn_fork_agent,
  spawn_learning_fork,
)
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.providers.base import StreamEvent
from model_authority.registry import ModelRegistryEntry, ProductModelRegistry
from model_authority.schema import SCHEMA
from gateway_test_support.model_defaults import compat_for_profile
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway.task_registry import RegistryOwner, TaskEntry, TaskRegistry
from agent_gateway.tool_dispatcher_approval_lifecycle import ApprovalPark
from agent_gateway.session import GatewaySession
from agent_gateway.tool_policy_registry import PreparedToolCall
from agent_gateway.transcript import (
  detect_orphan_tool_uses,
  reconstruct_messages_for_task,
)
from gateway_test_support.admitted_authority_test_support import (
  SOURCE_TOOL_ID,
  provenance_of,
  sealed_admitted_task,
)
from gateway_test_support.host_policy import owner_session_host_policy  # noqa: F401


def _bind() -> CapabilityBind:
  return CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key="anthropic.test-sonnet",
    provider="anthropic",
    upstream_model="claude-sonnet-5",
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
    schema=SCHEMA,
    revision=bind.registry_revision,
    models={
      bind.model_key: ModelRegistryEntry(
        key=bind.model_key,
        label="Fork infrastructure test model",
        provider=bind.provider,
        upstream_model=bind.upstream_model,
        adapter=bind.adapter,
        protocol_profile=bind.protocol_profile,
        compat=compat_for_profile(bind.protocol_profile),
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


def _inheriting_resolver(
  bind: CapabilityBind,
  provider: ModelProvider,
) -> CapabilityExecutionResolver:
  """The parent's resolver: its model and credential, every slot inheriting."""

  bound = _registry_for_bind(bind).models[bind.model_key]
  qualified = bound.capabilities
  # The policy owner requires every capability slot with a non-empty allowed
  # set; slots the bound model is not qualified for get a separate entry.
  other = replace(
    bound,
    key=f"{bound.key}.other-capabilities",
    route=f"{bound.route}.other-capabilities",
    capabilities={slot: "internal" for slot in CAPABILITY_IDS if slot not in qualified},
  )
  registry = ProductModelRegistry(
    schema=SCHEMA,
    revision=bind.registry_revision,
    models={bound.key: bound, other.key: other},
  )
  handle = CredentialHandle(
    handle_id=bind.credential_ref,
    provider=bind.provider,
    principal=bind.credential_principal,
    tenant_id="tenant-1",
    actor_id="owner-1" if bind.credential_principal == "user" else None,
  )
  return CapabilityExecutionResolver(
    registry=registry,
    selection_policy=ProductModelSelectionPolicy(
      schema=SCHEMA,
      revision=bind.policy_revision,
      capabilities={
        slot: CapabilitySelectionPolicy(
          capability_id=slot,
          default=CapabilityDefault(kind="inherit_parent"),
          by_channel={},
          allowed_model_keys=frozenset(
            {bound.key if slot in qualified else other.key}
          ),
        )
        for slot in CAPABILITY_IDS
      },
    ),
    auth_context=AuthContext(
      run_mode=bind.run_mode,
      actor_id="owner-1",
      tenant_id="tenant-1",
      user_provider_handles=(
        {bind.provider: handle} if bind.credential_principal == "user" else {}
      ),
      service_provider_handles=(
        {bind.provider: handle} if bind.credential_principal == "service" else {}
      ),
      entitled_capabilities=frozenset(qualified),
      entitled_model_keys=frozenset({bind.model_key}),
      allow_service_for_interactive=True,
    ),
    credential_materializer=lambda selected: MaterializedCredential(
      handle=selected,
      auth_config={"api_key": "secret", "provider": bind.provider},
    ),
    adapter_resolver=lambda _adapter: provider,
  )


class _RootOperationCatalog:
  """A session's root catalog as the learning factory reads it: callable names."""

  def __init__(self, *names: str) -> None:
    self._listing = tuple(
      (
        AgentOperationRef(
          namespace="agent-operation",
          name=name,
          version="1.0",
          digest=sha256_digest({"operation": name}),
        ),
        f"{name} operation",
      )
      for name in names
    )

  def resolve_operation(
    self,
    selector: AgentOperationRef | Mapping[str, Any] | None,
  ) -> ResolvedOperationRuntime:
    raise AssertionError(f"learning admission never resolves {selector!r}")

  def list_callable_operations_with_descriptions(
    self,
  ) -> tuple[tuple[AgentOperationRef, str], ...]:
    return self._listing


def _learning_operation(
  directive: str,
  required_capabilities: tuple[SemanticCapabilityRequirement, ...],
  workspace_scope: WorkspaceScope,
) -> AgentOperationSnapshot:
  return build_agent_operation_snapshot(
    name="learning-fork",
    version="1.0",
    methodology_instructions=directive,
    resolved_instructions=directive,
    description="Review the inherited session for reusable learning.",
    execution_class="node.fork",
    required_capabilities=required_capabilities,
    workspace_scope=workspace_scope,
    resumable=True,
    terminal_on_persist=False,
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
    _dispatcher=ToolDispatcher(McpClientManager(), capability_bind=_bind()),
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
  principal: CredentialPrincipal,
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


def test_credential_identity_has_exactly_one_tier() -> None:
  """B-6/D-B6-1: the three-tier fallback chain is gone.

  Before WP7 a runner with no bound session handle still produced a credential
  identity — from ``runner._tenant_id`` (tier 2), or from the
  ``capability_execution.auth_config`` row (tier 3). Both tiers are deleted:
  the session's own credential binding is the ONLY source, and its absence is
  reported as absence.
  """

  child = _runner()
  child._gateway_session = None
  child._tenant_id = "child-tenant"
  assert credential_identity_or_none(child) is None

  autonomous = _runner()
  autonomous._gateway_session = None
  autonomous._capability_execution.auth_config.update({
    "credential_handle_id": "credential-1",
    "tenant_id": "auth-tenant",
  })
  assert credential_identity_or_none(autonomous) is None

  # The one live tier, and its guard: a handle that does not match the
  # capability bind is a different identity, never a weaker one.
  assert credential_identity_or_none(_runner()) == ("credential-1", "tenant-1")

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


def test_fork_capture_predicate_reads_the_resolved_identity() -> None:
  """The two run-loop capture sites read one resolved identity, not a chain."""

  unbound = _runner()
  unbound._gateway_session.session_credential_handle = None
  assert fork_credential_identity_available(unbound) is False

  bound = _runner()
  assert fork_credential_identity_available(bound) is True
  identity = resolved_execution_identity(bound)
  assert identity is not None
  assert (identity.credential_handle_id, identity.tenant_id) == (
    "credential-1",
    "tenant-1",
  )


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


def test_node_parent_bind_freezes_whole_into_both_handoffs() -> None:
  # The handoff freezes whichever exact bind its parent executes; a node
  # parent is not a session driver and is captured whole all the same.
  runner = _runner()
  bind = CapabilityBind.model_validate({
    **_bind().to_json(),
    "capability_id": "node.explore",
  })
  runner._capability_execution.bind = bind
  base = [{"role": "user", "content": [{"type": "text", "text": "seed"}]}]
  post = build_post_turn_handoff(
    runner,
    base,
    {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
  )
  mid = build_mid_turn_handoff(runner, [*base, _assistant("fork-1")])

  assert post.capability_bind == bind
  assert mid.capability_bind == bind
  assert post.capability_bind.capability_id == "node.explore"


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


class _Dispatcher(ToolDispatcher):
  def __init__(self, definitions: list[dict[str, Any]], *, capability_bind = TEST_CAPABILITY_BIND) -> None:
    super().__init__(McpClientManager(), capability_bind=capability_bind)
    self.definitions = definitions
    self.calls: list[str] = []
    self.replay_controls: list[bool] = []

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return copy.deepcopy(self.definitions)

  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    **_kwargs: Any,
  ):
    _ = tool_call_id, tool_input
    self.calls.append(tool_name)
    self.replay_controls.append(_kwargs.get("allow_uncertain_mcp_replay", True))
    return {"ok": True}, None



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
    model="claude-sonnet-5",
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
    model="claude-sonnet-5",
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
    model="claude-sonnet-5",
    messages=parent_messages,
    system_prompt=[("static", True), ("dynamic", True)],
    tools=[],
    max_tokens=4096,
  )
  fork = provider.build_request_params(
    model="claude-sonnet-5",
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
  _fork_mode: bool
  _fork_marker_position: tuple[int, int]
  _fork_suffix_tokens: int
  children: list["_SpawnRunner"] = []

  def __init__(self, **kwargs: Any) -> None:
    self.kwargs = kwargs
    self._log = kwargs["event_log"]
    self._runner_id = f"runner-{kwargs['session_id']}"
    self._approval_park: ApprovalPark | None = None
    self._admitted_self_task = kwargs.get("admitted_self_task")
    self._task_registry = (
      kwargs["task_registry"]
      if kwargs.get("task_registry") is not None
      else TaskRegistry()
    )
    self.__class__.children.append(self)

  async def _append_durable_event(self, _event: dict[str, Any]) -> None:
    return None

  def _append(self, event: dict[str, Any]) -> None:
    self._log.append(event)

  async def run(self, **_kwargs: Any) -> None:
    self._log.append({"type": "text_delta", "text": "fork complete"})
    # A suffix past the ceiling is measured; the fork still finishes.
    self._fork_suffix_tokens = 25_000
    self._log.append({"type": "turn_complete"})
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


def _fork_parent(
  tmp_path,
  cls: type = _SpawnRunner,
  containment: ContainmentSettings = ContainmentSettings(),
) -> Any:
  # ``spawn_fork_agent`` builds the child as ``type(parent)``, so the parent's
  # class chooses the child runner behavior.
  parent = object.__new__(cls)
  parent._log = EventLog()
  parent._full_session_id = "parent-session"
  parent._cost_accumulator = None
  parent._stream_stall_timeout = 60.0
  parent._mcp_client = None
  parent._mcp_activation_fold = McpActivationFold()
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
  parent._aggregator = None
  parent._containment = containment
  parent._agent_session_log = _ForkSessionLog()
  parent._max_resume_chain_depth = 0
  parent._spill_dir_provider = None
  parent._workspace_dir = str(tmp_path)
  parent._context_surfaces_provider = None
  parent._context_surfaces_static = []
  parent._task_registry = TaskRegistry()
  return parent


def test_spawn_fork_reuses_resume_seed_and_fresh_tagged_log(
  tmp_path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  caplog.set_level("INFO", logger="agent_gateway.runner")
  _SpawnRunner.children.clear()
  parent = _fork_parent(tmp_path)
  execution_bind = CapabilityBind(
    schema_version="1.0",
    capability_id="node.fork",
    model_key="anthropic.test-sonnet",
    provider="anthropic",
    upstream_model="claude-sonnet-5",
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
  result_requirement = ResultRequirement(
    mode="narrative", terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )
  admitted = sealed_admitted_task(
    logical_task=logical_task, attempt=attempt, result_requirement=result_requirement,
  )
  entry = TaskEntry(task_id=attempt.physical_task_id, task_type="agent", admitted_task=admitted)
  bound_runners: list[Any] = []
  result_provenance = provenance_of(admitted)
  result, error = asyncio.run(
    spawn_fork_agent(
      parent,
      "finish the side quest",
      handoff=handoff,
      capability_execution=execution,
      logical_task=logical_task,
      attempt=attempt,
      result_requirement=result_requirement,
      result_provenance=result_provenance,
      dispatcher=_Dispatcher(handoff.wire_tools, capability_bind=execution.bind),
      sub_session=None,
      identity=None, approval_route=None,
      bind_job_runner=bound_runners.append,
      task_entry=entry,
    )
  )

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"
  assert result.observation.usage.turns == 1
  assert result.values.terminal_narrative is not None
  assert result.values.projection is None
  child = _SpawnRunner.children[-1]
  # The fork's tool surface is the parent's exact wire surface, minus nothing.
  assert child.kwargs["get_tool_definitions"]() == handoff.wire_tools
  assert "run_agent" in {
    tool["name"] for tool in child.kwargs["get_tool_definitions"]()
  }
  measurements = [
    record.data  # pyright: ignore[reportAttributeAccessIssue]  # structured log payload
    for record in caplog.records
    if getattr(record, "data", {}).get("event") == "fork_measurements"
  ]
  assert measurements == [{
    "event": "fork_measurements",
    "sub_agent_id": child.kwargs["session_id"],
    "turns": 1,
    "suffix_tokens": 25_000,
  }]
  assert all(
    entry.event["sub_agent_id"].startswith("sub0:")
    and entry.event["fork"] is True
    for entry in child._log.entries
  )


def test_learning_receipt_names_the_turns_the_fork_ran() -> None:
  receipts: list[str] = []
  item = LearningForkWorkItem(
    parent=None,  # pyright: ignore[reportArgumentType]
    handoff=None,  # pyright: ignore[reportArgumentType]
    config=None,  # pyright: ignore[reportArgumentType]
    ledger=SimpleNamespace(  # pyright: ignore[reportArgumentType]
      write_receipt=lambda **kw: receipts.append(kw["receipt_text"]),
    ),
    session_id="s",
    owner="o",
    user_id="u",
    fork_events=[
      {"type": "turn_complete"},
      {
        "type": "tool_call_complete",
        "tool_name": "memory_write",
        "is_error": False,
        "error": None,
      },
      {"type": "turn_complete"},
    ],
  )

  item.on_admission_settled("fork-1")

  assert receipts == ["Self-learning fork completed: 1 memory note written in 2 turns"]


class _FailedRetrievalForkRunner(_SpawnRunner):
  """A fork whose only granted source-capability retrieval failed."""

  async def run(self, **_kwargs: Any) -> None:
    self._log.append({
      "type": "tool_call_complete",
      "tool_call_id": "call-web-search",
      "tool_name": SOURCE_TOOL_ID,
      "is_error": True,
    })
    self._log.append({"type": "text_delta", "text": "fork complete"})
    self._log.append({"type": "stream_complete"})


def test_fork_settlement_derives_the_outcome_from_admitted_authority(
  tmp_path,
) -> None:
  # The third settlement site (B-3): ``spawn_fork_agent`` passes the entry's
  # admitted task into settlement, so the fork's result carries the mechanical
  # qualifier derived from the grant and bindings frozen at admission. Drop
  # that pass and the fork settles with no outcome at all.
  _SpawnRunner.children.clear()
  parent = _fork_parent(tmp_path, cls=_FailedRetrievalForkRunner)
  execution_bind = CapabilityBind(
    schema_version="1.0",
    capability_id="node.fork",
    model_key="anthropic.test-sonnet",
    provider="anthropic",
    upstream_model="claude-sonnet-5",
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
  parent._capability_execution = SimpleNamespace(bind=handoff.capability_bind)
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
  result_requirement = ResultRequirement(
    mode="narrative",
    terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )
  admitted = sealed_admitted_task(
    logical_task=logical_task,
    attempt=attempt,
    result_requirement=result_requirement,
  )
  task_entry = TaskEntry(
    task_id=attempt.physical_task_id,
    task_type="agent",
    admitted_task=admitted,
  )
  bound_runners: list[Any] = []

  result, error = asyncio.run(
    spawn_fork_agent(
      parent,
      "finish the side quest",
      handoff=handoff,
      capability_execution=execution,
      logical_task=logical_task,
      attempt=attempt,
      result_requirement=result_requirement,
      result_provenance=provenance_of(admitted),
      dispatcher=_Dispatcher(handoff.wire_tools, capability_bind=execution.bind),
      sub_session=None,
      identity=None, approval_route=None,
      bind_job_runner=bound_runners.append,
      task_entry=task_entry,
    )
  )

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"
  assert result.outcome is not None
  assert result.outcome.assessment_source == "mechanically_derived"
  # The grant intersected with the source-capability binding is exactly
  # ``web_search``; its only retrieval failed.
  assert result.outcome.disposition == "insufficient_evidence"
  assert result.outcome.unmet_requirements == (SOURCE_TOOL_ID,)


class _WedgedForkRunner(_SpawnRunner):
  """A fork whose inner await never resolves and that emits nothing."""

  cancelled = False

  async def run(self, **_kwargs: Any) -> None:
    try:
      await asyncio.Event().wait()
    except asyncio.CancelledError:
      self.cancelled = True
      raise


class _HeartbeatForkRunner(_SpawnRunner):
  """A live fork whose only events, past the gap, are stream heartbeats."""

  cancelled = False
  guard_silence_seconds = 0.0  # set by _spawn_minimal_fork before the fork runs

  async def run(self, **kwargs: Any) -> None:
    try:
      # Outlast the activity gap by 2x on whatever clock the host delivers.
      deadline = time.monotonic() + 2 * self.guard_silence_seconds
      while time.monotonic() < deadline:
        await asyncio.sleep(0.02)
        self._log.append({
          "type": "heartbeat",
          "elapsed_s": 0,
          "last_progress_s": 0,
          "events": 1,
        })
    except asyncio.CancelledError:
      self.cancelled = True
      raise
    await super().run(**kwargs)


def _spawn_minimal_fork(tmp_path, cls: type[_SpawnRunner]) -> TaskResult:
  _SpawnRunner.children.clear()
  # The activity gap is 25x the 20 ms heartbeat cadence and scales with host load: a
  # 50 ms gap was below what a loaded 12-worker gate delivers to the heartbeat.
  silence_seconds = harness_wait_seconds(0.5)
  _HeartbeatForkRunner.guard_silence_seconds = silence_seconds
  parent = _fork_parent(
    tmp_path, cls=cls,
    containment=ContainmentSettings(child_silence_timeout_seconds=silence_seconds),
  )
  execution_bind = CapabilityBind(
    schema_version="1.0",
    capability_id="node.fork",
    model_key="anthropic.test-sonnet",
    provider="anthropic",
    upstream_model="claude-sonnet-5",
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
    auth_config={"api_key": "secret", "provider": "anthropic"},
  )
  handoff = build_mid_turn_handoff(
    _runner(),
    [
      {"role": "user", "content": [{"type": "text", "text": "parent"}]},
      _assistant("fork-1"),
    ],
  )
  parent._capability_execution = SimpleNamespace(bind=handoff.capability_bind)
  operation = AgentOperationRef(
    namespace="agent-operation",
    name="test-fork",
    version="1.0",
    digest=sha256_digest({"operation": "test-fork"}),
  )
  logical_task = OrdinaryDelegationTaskRef(
    delegation_id="test-fork-1", operation=operation,
  )
  attempt = AttemptRef(
    attempt_number=1, attempt_id="attempt:test-fork:1",
    physical_task_id="sub0:parent-session",
  )
  result_requirement = ResultRequirement(
    mode="narrative", terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )
  admitted = sealed_admitted_task(
    logical_task=logical_task, attempt=attempt, result_requirement=result_requirement,
  )
  entry = TaskEntry(task_id=attempt.physical_task_id, task_type="agent", admitted_task=admitted)
  bound_runners: list[Any] = []
  result, error = asyncio.run(asyncio.wait_for(
    spawn_fork_agent(
      parent,
      "finish the side quest",
      handoff=handoff,
      capability_execution=execution,
      logical_task=logical_task,
      attempt=attempt,
      result_requirement=result_requirement,
      result_provenance=provenance_of(admitted),
      dispatcher=_Dispatcher(handoff.wire_tools, capability_bind=execution.bind),
      sub_session=None,
      identity=None, approval_route=None,
      bind_job_runner=bound_runners.append,
      task_entry=entry,
    ),
    # Far past the silence timeout: a fork the guard does not reach
    # fails here instead of hanging the suite.
    timeout=40 * silence_seconds,
  ))
  assert error is None
  assert isinstance(result, TaskResult)
  return result


@pytest.fixture
def _short_guard_poll(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(runner_sub_agents, "STREAM_GUARD_POLL_INTERVAL", 0.01)


@pytest.mark.usefixtures("_short_guard_poll")
def test_wedged_fork_is_settled_stalled_by_the_activity_guard(
  tmp_path,
) -> None:
  # ACUI-1 for forks: an await outside stream and tool, no event, no clock.
  # The parent's activity guard, not elapsed time, is what ends it.
  result = _spawn_minimal_fork(tmp_path, _WedgedForkRunner)

  child = _SpawnRunner.children[-1]
  assert isinstance(child, _WedgedForkRunner)
  assert child.cancelled is True
  assert result.execution.status == "interrupted"
  assert result.execution.terminal_reason is not None
  errors = [
    entry.event
    for entry in child._log.entries
    if entry.event.get("type") == "error"
  ]
  assert [event.get("error_sub_code") for event in errors] == ["stalled"]


@pytest.mark.usefixtures("_short_guard_poll")
def test_live_fork_outlasts_the_activity_gap(tmp_path) -> None:
  result = _spawn_minimal_fork(tmp_path, _HeartbeatForkRunner)

  child = _SpawnRunner.children[-1]
  assert isinstance(child, _HeartbeatForkRunner)
  assert child.cancelled is False
  assert result.execution.status == "succeeded"


def test_run_loop_measures_fork_suffix_and_never_stops_the_fork() -> None:
  # GC13/GC14: the run loop owns the measurement. A suffix far past the old
  # 20k ceiling is recorded, and the request opens no forced-final turn.
  from agent_gateway.runner_run_loop import RunnerRunLoopMixin

  async def _record_context(**kwargs: Any) -> tuple[bool, bool]:
    _ = kwargs
    return False, False

  fork_runner = SimpleNamespace(
    _containment=ContainmentSettings(),
    _fork_mode=True,
    _fork_marker_position=(0, 0),
    _fork_suffix_tokens=0,
    _inject_system_prompt_reminder=lambda prompt, _reminder: prompt,
    _record_provider_request_context=_record_context,
  )
  prefix = {"role": "user", "content": [
    {"type": "text", "text": "parent history"},
    {"type": "text", "text": "fork directive"},
  ]}
  suffix = {"role": "assistant", "content": [{"type": "text", "text": "x" * 90}]}
  measured: list[list[dict[str, Any]]] = []

  def _estimate(*, system_text: str, messages: list[dict[str, Any]], tools: list[Any]) -> Any:
    _ = system_text, tools
    measured.append(copy.deepcopy(messages))
    tokens = 1_000_000 * len(messages)
    return SimpleNamespace(est_messages_tokens=tokens, est_total_tokens=tokens)

  result = asyncio.run(RunnerRunLoopMixin._prepare_provider_request(
    fork_runner,  # pyright: ignore[reportArgumentType]  # duck-typed runner
    active_delivery_obligation=None,
    base_kwargs={},
    build_context_pressure_reminder=lambda **_kwargs: "",
    build_runtime_guard_event=lambda **_kwargs: {},
    context_limit=10_000_000,
    context_pressure_reminder_decision=lambda **_kwargs: ContextPressureReminderDecision(
      reminder_pct=None,
      next_threshold_pct=50,
    ),
    current_messages=[prefix, suffix],
    delivered_notifications=[],
    delivered_notifications_recorded=False,
    delivery_request_nudge_message=None,
    delivery_request_nudge_recorded=False,
    wrap_up_reminder_text=None,
    logger=logging.getLogger("test"),
    max_tokens=1_000,
    last_ordinary_turn=None,
    pending_notifications_at_render=0,
    system_prompt="system",
    system_prompt_estimate_text=lambda _prompt: "",
    token_estimate_snapshot=_estimate,
    turn_count=3,
    turn_reminder=SimpleNamespace(text=""),  # pyright: ignore[reportArgumentType]  # duck-typed reminder
    user_turn_message=lambda text: {"role": "user", "content": text},
  ))

  # The first estimate is the suffix: the boundary message's blocks after the
  # marker, then every later message.
  assert measured[0] == [
    {"role": "user", "content": [{"type": "text", "text": "fork directive"}]},
    suffix,
  ]
  assert fork_runner._fork_suffix_tokens == 2_000_000
  last_ordinary_turn, wrap_up_reminder_text = result[0], result[1]
  assert last_ordinary_turn is None
  assert wrap_up_reminder_text is None


@pytest.mark.parametrize(
  "cleanup_failure",
  ["clean", "close_error", "close_cancel", "commit_error", "nonawait_commit"],
)
def test_approval_park_preserves_fork_admission_usage_and_owner(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path,
  cleanup_failure: str,
  owner_session_host_policy: Any,
) -> None:
  from agent_gateway.admission import (
    admit_root_turn,
    capture_job_context,
    capture_root_turn,
    root_context_from_identity,
  )
  from agent_gateway.approval_route import NoApprovalRoute
  from agent_workflow_contracts.models import AdmittedResearchBinding, AdmittedTenant

  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"

  async def case() -> None:
    parent_journal = AgentSessionLog(tmp_path / "parent-session.jsonl")
    child_journals: list[AgentSessionLog] = []
    fork_id = "learn-0123abcd"
    park = ApprovalPark(
      job_id=fork_id,
      original_tool_call_id="call-fork-approval",
      approval_id="approval-fork", nonce="private-fork-nonce",
    )
    runners: list[AgentRunner] = []
    completed_inputs: list[dict[str, Any]] = []

    class ObservedNativeForkRunner(AgentRunner):
      def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        runners.append(self)

    class Provider(_Provider):
      def __init__(self) -> None:
        self.stream_calls = 0
        self.closed_clients: list[object] = []

      def get_model_info(self, model: str) -> ModelInfo:
        return ModelInfo(
          id=model, provider=self.name, supports_thinking=True,
          input_cost_per_mtok=1_000_000,
          output_cost_per_mtok=1_000_000,
          cache_read_cost_per_mtok=1_000_000,
        )

      def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
        return object()

      async def close_client(self, client: Any, timeout: float = 2.0) -> None:
        self.closed_clients.append(client)
        if self.stream_calls > 1:
          if cleanup_failure == "close_error":
            raise RuntimeError("fork provider resource close failed")
          if cleanup_failure == "close_cancel":
            raise asyncio.CancelledError

      def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
        messages = copy.deepcopy(kwargs["messages"])
        for message in messages:
          if isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]
          for block in message["content"]:
            block.pop("cache_control", None)
        marker = kwargs["fork_marker_position"] if kwargs["fork_mode"] else (0, 0)
        messages[marker[0]]["content"][marker[1]]["cache_control"] = {"type": "ephemeral"}
        system_prompt = kwargs["system_prompt"]
        system = (
          [{"type": "text", "text": system_prompt}]
          if isinstance(system_prompt, str)
          else [{"type": "text", "text": text} for text, _cached in system_prompt]
        )
        return {"messages": messages, "tools": kwargs["tools"], "system": system}

      async def stream(self, client: Any, params: dict[str, Any]):
        self.stream_calls += 1
        if self.stream_calls == 1:
          yield StreamEvent(type="message_start")
          yield StreamEvent(type="text_delta", text="Ready to fork.")
          yield StreamEvent(
            type="text_end",
            raw_block={"type": "text", "text": "Ready to fork."},
          )
          yield StreamEvent(type="message_end", stop_reason="end_turn")
          return
        yield StreamEvent(type="message_start", input_tokens=13, cache_read_tokens=8)
        for call_id, query in (
          ("call-fork-sibling", "completed"),
          (park.original_tool_call_id, "pending"),
          ("call-fork-unscheduled", "later"),
        ):
          payload = {"query": query}
          yield StreamEvent(
            type="tool_use_end",
            tool_id=call_id, tool_name=SOURCE_TOOL_ID, tool_input=payload,
            raw_block={
              "type": "tool_use", "id": call_id,
              "name": SOURCE_TOOL_ID, "input": payload,
            },
          )
        yield StreamEvent(type="usage_update", output_tokens=5)
        yield StreamEvent(type="message_end", stop_reason="tool_use", request_input_tokens=21)

    class ParkingDispatcher(ToolDispatcher):
      async def dispatch_prepared(
        self,
        tool_call_id: str,
        tool_name: str,
        prepared_call: PreparedToolCall,
        **kwargs: Any,
      ):
        if tool_call_id == park.original_tool_call_id:
          assert prepared_call.materialize_input() == {"query": "pending"}
          raise park
        return await super().dispatch_prepared(
          tool_call_id, tool_name, prepared_call, **kwargs,
        )

    async def handler(tool_input: dict[str, Any], **_kwargs: Any):
      completed_inputs.append(dict(tool_input))
      return {"value": "completed evidence"}, None

    async def stock_memory_write(tool_input: dict[str, Any], **_kwargs: Any):
      raise AssertionError(f"the parent's stock memory_write ran: {tool_input}")

    provider = Provider()
    parent_bind = _bind()
    parent_execution = _inheriting_resolver(parent_bind, provider).materialize_bind(
      parent_bind,
    )
    definitions = [{
      "name": SOURCE_TOOL_ID, "description": "Read admitted evidence",
      "input_schema": {"type": "object"},
    }]
    dispatcher = ParkingDispatcher(
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={
        SOURCE_TOOL_ID: handler,
        "memory_write": stock_memory_write,
      },
      get_tool_definitions=lambda: definitions,
      capability_bind=parent_execution.bind,
      role="owner",
      needs_approval=lambda *_args: False,
      # An authenticated parent runs in strict credential mode; its fork
      # must not run under a weaker MCP identity check.
      credentials_resolver_active=True,
    )
    tenant = AdmittedTenant(tenant_id="tenant-1", storage_id="fork-owner", role="owner")
    session = GatewaySession(
      session_id="parent-session", api_key_hash="hash",
      created_at=1, expires_at=2, user_id="fork-owner",
      tenant_id=tenant.tenant_id,
      session_credential_handle=CredentialHandle(
        handle_id=parent_bind.credential_ref,
        provider=parent_bind.provider, principal="user",
        tenant_id="tenant-1", actor_id="fork-owner",
      ),
      learning_fork_operation_catalog=_RootOperationCatalog("explore"),
    )
    context = root_context_from_identity(
      tenant=tenant, job_id="parent-session",
      tool_ids=frozenset({SOURCE_TOOL_ID}), withheld_tools=frozenset(),
      research_binding=AdmittedResearchBinding(), approval_route=NoApprovalRoute(),
    )
    capture_job_context(session, context)
    capture_root_turn(session, admit_root_turn(
      context,
      execution=parent_execution,
      tool_grant={"tool_ids": [SOURCE_TOOL_ID], "withheld_tools": []},
      approval_route=NoApprovalRoute(),
    ))
    parent = ObservedNativeForkRunner(
      event_log=EventLog(), dispatcher=dispatcher,
      session_id="parent-session", capability_execution=parent_execution,
      user_id="fork-owner", billing_mode="byok", rate_table_version="v1",
      agent_session_log=parent_journal, workspace_dir=tmp_path,
      get_tool_definitions=lambda: definitions,
      max_tokens_override=4096,
      gateway_session=session,
    )
    parent._task_registry.bind_owner(RegistryOwner(
      job_id="parent-session", tenant=lambda: tenant,
    ))
    seed = [{"role": "user", "content": [{"type": "text", "text": "Prepare the admitted fork."}]}]
    await parent.run(seed, system_prompt="Read evidence for the admitted task.")
    handoff = build_post_turn_handoff(
      parent, seed,
      {"role": "assistant", "content": [{"type": "text", "text": "Ready to fork."}]},
    )

    def read_fork_journal(
      journal_fork_id: str,
    ) -> AgentSessionLog | None:
      path = tmp_path / f"{journal_fork_id}.jsonl"
      if not path.is_file():
        return None
      return AgentSessionLog(
        path, _create_active_on_open=False, _repair_manifest_on_open=False,
      )

    # A previous gateway process left this fork and a crashed one started.
    dead_process = ForkLedger(
      tmp_path / "fork-ledger.sqlite3", process_instance_id="gateway-dead",
    )
    for started_fork_id in (fork_id, "learn-crashed"):
      dead_process.reserve_admission(
        fork_id=started_fork_id, owner="fork-owner", max_reserved_usd="2",
      )
      assert dead_process.mark_admission_started(fork_id=started_fork_id)

    def child_journal(child_session: GatewaySession) -> AgentSessionLog:
      journal = AgentSessionLog(tmp_path / f"{child_session.session_id}.jsonl")
      original_append = journal.append_sync

      def append_with_fault(event: dict[str, Any]):
        if (
          cleanup_failure in {"commit_error", "nonawait_commit"}
          and event.get("type") == "stream_complete"
          and event.get("terminal_disposition") == "awaiting_approval"
        ):
          raise OSError("fork park journal unavailable")
        if cleanup_failure == "commit_error" and event.get("type") == "run_error":
          raise OSError("fork failure journal unavailable")
        return original_append(event)

      monkeypatch.setattr(journal, "append_sync", append_with_fault)
      child_journals.append(journal)
      return journal

    work_item = LearningForkWorkItem(
      parent=parent,
      handoff=handoff,
      config=LearningForkConfig(
        directive="finish the side quest",
        scope_memory_write=lambda handler, *, fork_id, user_id: handler,
        ledger_path=lambda _owner: tmp_path / "fork-ledger.sqlite3",
        operation_snapshot_builder=_learning_operation,
        child_session_log_builder=child_journal,
        fork_journal_reader=read_fork_journal,
      ),
      ledger=ForkLedger(
        tmp_path / "fork-ledger.sqlite3", process_instance_id="gateway-test",
      ),
      session_id="parent-session",
      owner="fork-owner",
      user_id="fork-owner",
    )
    run = spawn_learning_fork(fork_id, work_item)
    # Admission, materialization and registration happen at submission.
    entry = parent._task_registry.get(fork_id)
    assert entry is not None and entry.admitted_task is not None
    admitted = entry.admitted_task
    assert admitted.execution_snapshot is not None
    admitted_context = admitted.execution_snapshot.admitted_context
    assert admitted_context is not None
    assert [item.tool_id for item in admitted.tool_grant.tools] == [SOURCE_TOOL_ID]
    assert admitted_context.tool_grant == (SOURCE_TOOL_ID,)
    assert admitted_context.children == ("explore",)
    assert admitted_context.lineage.parent_job_id == "parent-session"
    assert admitted_context.lineage.depth == 1
    assert admitted.operation.execution_class == "node.fork"
    assert admitted.model_bind is not None
    assert admitted.model_bind.capability_id == "node.fork"
    assert admitted.model_bind.model_key == parent_bind.model_key
    assert admitted.model_bind.effort == parent_bind.effort
    assert admitted.model_bind.credential_ref == parent_bind.credential_ref
    settled: TaskEntry | None = None
    try:
      settled = await run
    except asyncio.CancelledError:
      assert cleanup_failure == "close_cancel"
    child = runners[1]
    assert settled is None or settled is child._admitted_self_task
    result = entry.task_result
    assert isinstance(result, TaskResult)
    # The own writer settles its job before closing, so a cleanup failure
    # after that settlement leaves the committed park standing.
    assert result.execution.status == (
      "failed" if cleanup_failure in {"commit_error", "nonawait_commit"}
      else "awaiting_approval"
    )
    assert child._full_session_id == fork_id
    fork_dispatcher = child._dispatcher
    assert isinstance(fork_dispatcher, ParkingDispatcher)
    assert fork_dispatcher is not dispatcher
    assert fork_dispatcher._credentials_resolver_active is True
    assert child._role == "writer"
    assert child._agent_session_log is child_journals[0]
    assert child._active_client is None
    assert len(provider.closed_clients) == 2
    assert provider.stream_calls == 2
    assert completed_inputs == [{"query": "completed"}]
    assert child._task_registry.owner is not None
    assert child._task_registry.owner.job_id == fork_id
    assert child._task_registry._owner_completed is (cleanup_failure != "clean")
    assert parent._task_registry._child_registries[fork_id] is child._task_registry
    durable_entries, _ = await child_journals[0].query(
      order="asc", runner_id=child._runner_id,
    )
    durable_events = [item.event for item in durable_entries]
    awaiting = [
      event for event in durable_events
      if event.get("type") == "stream_complete"
      and event.get("terminal_disposition") == "awaiting_approval"
    ]
    if cleanup_failure in {"commit_error", "nonawait_commit"}:
      assert awaiting == []
      assert child._approval_park is (park if cleanup_failure == "commit_error" else None)
    else:
      assert len(awaiting) == 1
      assert child._terminal_closure_event is not None
      assert child._terminal_closure_event["terminal_disposition"] == "awaiting_approval"
      assert child._approval_park is park
      assert awaiting[0]["usage"]["input_tokens"] == 13
      assert awaiting[0]["usage"]["output_tokens"] == 5
    for events in ([item.event for item in child._log.entries], durable_events):
      assert any(
        event.get("type") == "tool_call_complete"
        and event.get("tool_call_id") == "call-fork-sibling"
        for event in events
      )
      if cleanup_failure == "clean":
        assert not any(
          event.get("type") in {"tool_call_complete", "tool_call_interrupted"}
          and event.get("tool_call_id") in {park.original_tool_call_id, "call-fork-unscheduled"}
          for event in events
        )
        assert not {"error", "run_error", "interrupted", "skill_result"} & {
          event.get("type") for event in events
        }
    parent_events, _ = await parent_journal.query(order="asc")
    assert not any(
      item.event.get("tool_call_id") == "call-fork-sibling"
      for item in parent_events
    )
    if cleanup_failure == "clean":
      # The fork's own journal holds the conversation it started from: the
      # inherited prefix before its registration, the directive after it, so
      # a cold reader rebuilds prefix + directive + the parked turn.
      fork_messages = build_fork_messages(handoff, "finish the side quest")
      assert [message["role"] for message in fork_messages] == [
        "user", "assistant", "user",
      ]
      opening = [
        (event["type"], event.get("content", event.get("content_blocks")))
        for event in durable_events
        if event["type"] in {"user_message", "assistant_message", "task_registered"}
      ]
      assert opening[:4] == [
        ("user_message", fork_messages[0]["content"]),
        ("assistant_message", fork_messages[1]["content"]),
        ("task_registered", None),
        ("user_message", fork_messages[2]["content"]),
      ]
      assert opening[4][0] == "assistant_message"
      assistant_turns: list[list[dict[str, Any]]] = [
        event["content_blocks"] for event in durable_events
        if event["type"] == "assistant_message"
      ]
      parked_blocks = assistant_turns[1]
      assert [block["id"] for block in parked_blocks] == [
        "call-fork-sibling", park.original_tool_call_id, "call-fork-unscheduled",
      ]
      transcript = await reconstruct_messages_for_task(child_journals[0], fork_id)
      replayed = await SessionContextBuilder(agent_session_log=child_journals[0]).build()
      for rebuilt in (transcript, replayed):
        assert [
          {"role": message["role"], "content": message["content"]}
          for message in rebuilt[:4]
        ] == [
          *fork_messages,
          {"role": "assistant", "content": parked_blocks},
        ]
      # The resume reconstruction leaves the parked call for the approved-call
      # executor; the context replay shows it waiting on the user, not failed.
      assert park.original_tool_call_id in detect_orphan_tool_uses(transcript)
      parked_results = [
        block for message in replayed if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
        and block.get("tool_use_id") == park.original_tool_call_id
      ]
      assert len(parked_results) == 1 and not parked_results[0].get("is_error")
      assert json.loads(parked_results[0]["content"]) == {
        "status": "awaiting_approval", "approval_id": park.approval_id,
      }
      # Only the asked call is projected; an unscheduled sibling stays an orphan.
      assert "call-fork-unscheduled" in detect_orphan_tool_uses(replayed)
      assert result.logical_task == admitted.logical_task
      assert result.attempt == admitted.attempt
      assert result.provenance == provenance_of(admitted)
      assert result.observation.transcript.owner_id == fork_id
      assert result.observation.activity.owner_id == fork_id
      assert result.observation.usage.input_tokens == 13
      assert result.observation.usage.output_tokens == 5
      assert result.observation.usage.cached_input_tokens == 8
      assert result.observation.usage.cost_usd == 26.0
      assert result.evidence.tools_used == (SOURCE_TOOL_ID, SOURCE_TOOL_ID)
      assert result.values.terminal_narrative is None
      assert result.values.projection is None
      assert result.values.artifacts == ()
      assert result.outcome is None
      assert park.nonce not in result.model_dump_json()
      assert not any(park.nonce in str(event) for event in durable_events)
    if cleanup_failure in {"clean", "close_error"}:
      if cleanup_failure == "close_error":
        # A later writer died mid-append: its torn tail was never committed,
        # so the journal's committed prefix still shows the park.
        with open(child_journals[0].path, "ab") as torn_tail:
          torn_tail.write(b'{"seq": 9999, "timestamp": 1.0, "event": {"type": "task_')
      # The parent's runtime reconciles first: the parked fork's own journal
      # holds its committed park, so its started row moves to this process
      # and settles once through the owner-checked transition; a fork with
      # no journal is abandoned at its reservation.
      session.learning_fork_config = work_item.config
      runtime = _runtime_for_runner(parent, create=True)
      assert runtime is not None
      ledger, _registry = runtime
      adopted = ledger.get_admission(fork_id)
      assert adopted is not None
      assert adopted.state == "started"
      assert adopted.process_instance_id == ledger.process_instance_id
      crashed = ledger.get_admission("learn-crashed")
      assert crashed is not None
      assert crashed.state == "abandoned"
      assert crashed.settled_usd == crashed.max_reserved_usd
      assert ledger.settle_admission(
        fork_id=fork_id, actual_cost_usd=result.observation.usage.cost_usd,
      )
      assert not ledger.settle_admission(
        fork_id=fork_id, actual_cost_usd=result.observation.usage.cost_usd,
      )
      settled_row = ledger.get_admission(fork_id)
      assert settled_row is not None
      assert settled_row.state == "settled"
      assert settled_row.settled_usd == Decimal(str(result.observation.usage.cost_usd))

  asyncio.run(case())


def test_learning_fork_gains_no_tool_the_parents_effective_policy_excludes(
  owner_session_host_policy: Any,
) -> None:
  """A tool on the wire but not advertised never reaches fork authority.

  The parent's last request carries three non-source tools: one the turn
  withheld (a skill-allow tool after the skill cleared), an anchor tool the
  session stopped advertising (the prefix-stable layout re-sends it), and a
  definition kept deferred so a `tool_reference` stays valid.  All are routable
  local tools whose effect would otherwise grant; the fork's authority is
  derived from the advertised surface and must hold none of them.
  """

  from agent_gateway.admission import (
    admit_root_turn,
    capture_job_context,
    capture_root_turn,
    root_context_from_identity,
  )
  from agent_gateway.approval_route import NoApprovalRoute
  from agent_workflow_contracts.models import AdmittedResearchBinding, AdmittedTenant
  from pathlib import Path

  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  gated, retained, left = "skill_gated_tool", "retained_only_tool", "anchor_left_tool"

  async def noop(tool_input: dict[str, Any], **_kwargs: Any):
    return {}, None

  async def case() -> None:
    parent_bind = _bind()
    provider = _Provider()
    parent_execution = _inheriting_resolver(parent_bind, provider).materialize_bind(parent_bind)
    wire_tools = [
      {
        "name": name, "description": name, "input_schema": {"type": "object"},
        **({"defer_loading": True} if name == retained else {}),
      }
      for name in (SOURCE_TOOL_ID, left, gated, retained)
    ]
    dispatcher = ToolDispatcher(
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={
        SOURCE_TOOL_ID: noop, gated: noop, retained: noop, left: noop, "memory_write": noop,
      },
      get_tool_definitions=lambda: wire_tools,
      capability_bind=parent_execution.bind,
      role="owner",
    )
    tenant = AdmittedTenant(tenant_id="tenant-1", storage_id="fork-owner", role="owner")
    session = GatewaySession(
      session_id="parent-session", api_key_hash="hash", created_at=1, expires_at=2,
      user_id="fork-owner", tenant_id=tenant.tenant_id,
      session_credential_handle=CredentialHandle(
        handle_id=parent_bind.credential_ref, provider=parent_bind.provider,
        principal="user", tenant_id="tenant-1", actor_id="fork-owner",
      ),
      learning_fork_operation_catalog=_RootOperationCatalog("explore"),
    )
    context = root_context_from_identity(
      tenant=tenant, job_id="parent-session",
      tool_ids=None, withheld_tools=frozenset({gated}),
      research_binding=AdmittedResearchBinding(), approval_route=NoApprovalRoute(),
    )
    capture_job_context(session, context)
    capture_root_turn(session, admit_root_turn(
      context, execution=parent_execution,
      tool_grant={"tool_ids": None, "withheld_tools": [gated]},
      approval_route=NoApprovalRoute(),
    ))
    parent = AgentRunner(
      event_log=EventLog(), dispatcher=dispatcher, session_id="parent-session",
      capability_execution=parent_execution, user_id="fork-owner",
      billing_mode="byok", rate_table_version="v1",
      get_tool_definitions=lambda: wire_tools, max_tokens_override=4096,
      gateway_session=session, excluded_tools={gated},
    )
    parent._task_registry.bind_owner(RegistryOwner(
      job_id="parent-session", tenant=lambda: tenant,
    ))
    # What the parent's last request put on the wire (`_make_params` records it).
    parent._last_request_system_blocks = (("system", True),)
    parent._last_request_wire_tools = copy.deepcopy(wire_tools)
    parent._last_request_advertised_tool_names = frozenset({SOURCE_TOOL_ID, gated})
    parent._last_request_message_marker_position = (0, 0)
    parent._last_request_max_tokens = 4096
    seed = [{"role": "user", "content": [{"type": "text", "text": "Prepare the fork."}]}]
    handoff = build_post_turn_handoff(
      parent, seed, {"role": "assistant", "content": [{"type": "text", "text": "Ready."}]},
    )
    work_item = LearningForkWorkItem(
      parent=parent, handoff=handoff,
      config=LearningForkConfig(
        directive="learn",
        scope_memory_write=lambda handler, *, fork_id, user_id: handler,
        ledger_path=lambda _owner: Path("/nonexistent/fork-ledger.sqlite3"),
        operation_snapshot_builder=_learning_operation,
        child_session_log_builder=lambda child_session: None,  # type: ignore[arg-type,return-value]
        fork_journal_reader=lambda _fork_id: None,
      ),
      ledger=None,  # type: ignore[arg-type]
      session_id="parent-session", owner="fork-owner", user_id="fork-owner",
    )
    fork_id = "learn-0123abcd"
    try:
      run = spawn_learning_fork(fork_id, work_item)
    except Exception as error:  # the child journal builder is a stub
      run = None
      admitted_error = error
    entry = parent._task_registry.get(fork_id)
    assert entry is not None and entry.admitted_task is not None, (
      f"admission did not complete: {locals().get('admitted_error')!r}"
    )
    admitted = entry.admitted_task
    granted = {item.tool_id for item in admitted.tool_grant.tools}
    assert SOURCE_TOOL_ID in granted
    assert not granted & {gated, retained, left}
    assert admitted.execution_snapshot is not None
    for definition in admitted.execution_snapshot.provider_tool_definitions or ():
      assert "defer_loading" not in definition.definition
    if run is not None:
      unstarted = asyncio.ensure_future(run)
      unstarted.cancel()
      with pytest.raises(asyncio.CancelledError):
        await unstarted

  asyncio.run(case())
