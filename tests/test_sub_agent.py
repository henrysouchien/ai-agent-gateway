from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import hashlib
import inspect
from io import StringIO
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from pydantic import JsonValue, ValidationError

from agent_gateway import AgentRunner, ToolDispatcher
from agent_gateway.admission import capture_job_context, current_parent_context
from agent_gateway.job_caller import JobScope, rebind_job_handlers
from agent_gateway.tool_result_spill import (
  SpillCapabilities,
  SpillSink,
  job_spill_sink,
  make_tool_result_read_handler,
  write_spill_set,
)
from agent_gateway.code_execution import (
  CodeExecutionConfig,
  build_code_execution,
  cleanup_code_execution,
)
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent
from agent_workflow_contracts import (
  ActivityHandle,
  AdmittedTask,
  AgentCompletionEnvelope,
  AgentOperationRef,
  AgentOperationSnapshot,
  AnalyticalOutcome,
  AttemptRef,
  CanonicalProjection,
  ContentHandle,
  ContentReadGrant,
  ContractRef,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  ResolvedAuthority,
  SemanticCapabilityRequirement,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultValues,
  TranscriptHandle,
  UsageObservation,
  canonical_json_bytes,
  sha256_digest,
)
from agent_workflow_contracts.ticker_contract import TICKER_INPUT_CONTRACT
from agent_gateway.capability_execution import (
  BoundCapabilityExecution,
  CapabilityBind,
  CapabilityExecutionResolver,
)
from agent_gateway.agent_result_content import (
  make_get_agent_result_content_handler,
)
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.approval_route import (
  DurableLocalApprovalRoute,
  InProcessDelegatedApprovalRoute,
  bind_session_approval_route,
  route_ledger,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.event_log import EventLog
from agent_gateway.runner_background_tasks import (
  DEFAULT_PARENT_RESULT_MAX_INLINE_BYTES,
  ParentResultMaterializationError,
  build_agent_completion_envelope,
  ordinary_parent_result_policy,
)
from agent_gateway.final_narrative_artifact import publish_final_narrative
from agent_gateway.operation_catalog import (
  Effort,
  MutationMode,
  OperationRuntimePolicy,
  ResolvedOperationRuntime,
)
from agent_gateway.operation_snapshot import WorkspaceScope, build_agent_operation_snapshot
from agent_gateway.tool_dispatcher_helpers import ToolExecutionContext
from agent_gateway.session import GatewaySession
from agent_gateway.skills import SkillLoader, operation_tool_ids
from agent_gateway.sub_agent import (
  _child_run_context,
  _ticker_admitted_input,
  _ticker_from_admitted_inputs,
  _runtime_policy_dispatch_projection,
  make_get_background_result_handler,
  make_get_background_result_tool_def,
  make_run_agent_handler,
  make_run_agent_tool_def,
  make_resume_tool_def,
  make_resume_handler,
  seal_admitted_task_payload,
)
from agent_gateway.sub_agent_helpers import (
  _catalog_operation_entries,
  _resolve_context_ticker,
)
from agent_gateway.capability_resolution import granted_tool_ids
from agent_gateway.sub_agent_scope_receipt import (
  ADMITTED_TASK_METADATA_KEY,
  admit_operation_tools,
)
from agent_gateway.sub_agent_result_contract import (
  TERMINAL_TOOL_RESULT_RETURN_CONTRACT,
  canonical_projection,
  report_contract_ref,
  terminal_narrative_content_handle,
)
from agent_gateway.sub_agent_narrative_result import task_result_from_execution
from agent_gateway.sub_agent_skill_events import (
  DurableSkillEventPersistenceError,
  SkillRunEventEmitter,
)
from agent_gateway.tool_definition import (
  LiveToolRouteBinding,
  LiveToolRouteKind,
  OriginatedToolDefinition,
)
from gateway_test_support.capability_execution_test_support import (
  stub_capability_execution_resolver,
)
from gateway_test_support.host_policy import owner_session_host_policy


async def _tool(_tool_input: dict[str, Any], **_kwargs: Any):
  return {"ok": True}, None


def _mcp_route_binding(
  *,
  server_id: str,
  logical_name: str,
  exposed_name: str,
  transport_server_id: str | None = None,
  provider_original_name: str | None = None,
  route_kind: LiveToolRouteKind = "physical",
) -> LiveToolRouteBinding:
  return LiveToolRouteBinding(
    originated_definition=OriginatedToolDefinition(
      definition={
        "name": exposed_name,
        "description": "Exact live test route.",
        "input_schema": {"type": "object"},
      },
      origin="mcp",
      server_id=server_id,
    ),
    route_kind=route_kind,
    logical_name=logical_name,
    transport_server_id=transport_server_id or server_id,
    provider_original_name=provider_original_name or logical_name,
    provider_id=None,
  )


class _McpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": name}


class _InvestmentMcpClient(_McpClient):
  def is_mcp_tool(self, name: str) -> bool:
    return name == "start_quant_research"

  def get_server_for_tool(self, name: str) -> str | None:
    return (
      "idea-workbench-mcp"
      if name == "start_quant_research"
      else None
    )

  def get_policy_tool_name(self, name: str) -> str | None:
    return name if name == "start_quant_research" else None

  def get_server_tool_definitions(
    self,
    server_names: set[str],
  ) -> list[dict[str, Any]]:
    if "idea-workbench-mcp" not in server_names:
      return []
    return [{
      "name": "start_quant_research",
      "description": "Start exact quant research.",
      "input_schema": {"type": "object"},
    }]

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return self.get_server_tool_definitions({"idea-workbench-mcp"})


class _CapabilityResolver(CapabilityExecutionResolver):
  __slots__ = ("calls",)
  calls: list[dict[str, Any]]

  def __init__(self) -> None:
    resolver = stub_capability_execution_resolver(
      default_provider="openai",
      default_model="gpt-5.6-sol",
      default_effort="high",
    )
    super().__init__(
      registry=resolver.registry,
      selection_policy=resolver.selection_policy,
      auth_context=resolver.auth_context,
      credential_materializer=resolver.credential_materializer,
      adapter_resolver=resolver.adapter_resolver,
      trusted_channel=resolver.trusted_channel,
      authenticated_run_overrides=resolver.authenticated_run_overrides,
      executable_capability_ids=resolver.executable_capability_ids,
    )
    object.__setattr__(self, "calls", [])

  def select_bind(self, capability_id: str, **kwargs: Any) -> Any:
    self.calls.append({"capability_id": capability_id, **kwargs})
    return super().select_bind(capability_id, **kwargs)


def _content(
  value: object,
  contract: ContractRef,
  *,
  media_type: str = "application/json",
) -> ContentHandle:
  raw = (
    value.encode("utf-8")
    if isinstance(value, str)
    else canonical_json_bytes(value)
  )
  digest = hashlib.sha256(raw).hexdigest()
  return ContentHandle(
    content_id=f"sha256:{digest}",
    content_sha256=digest,
    content_bytes=len(raw),
    content_chars=len(raw.decode("utf-8")),
    contract=contract,
    media_type=media_type,
    encoding="utf-8",
    retention="durable",
  )


def _spawn_result(kwargs: dict[str, Any]) -> TaskResult:
  requirement = kwargs["result_requirement"]
  provenance = kwargs["result_provenance"]
  narrative_contract = ContractRef(
    namespace="agent-gateway",
    name="terminal-narrative",
    version="1.0",
    digest=sha256_digest({"contract": "terminal-narrative"}),
  )
  terminal = (
    _content(
      "Canonical child narrative.",
      narrative_contract,
      media_type="text/plain; charset=utf-8",
    )
    if requirement.terminal_narrative == "required"
    else None
  )
  projection = None
  if requirement.projection is not None:
    inline: JsonValue = {"summary": "Canonical child summary."}
    projection = CanonicalProjection(
      contract=requirement.projection.contract,
      content=_content(inline, requirement.projection.contract),
      inline_view=inline,
    )
  outcome = (
    AnalyticalOutcome(
      disposition="complete",
      assessment_source="domain_tool",
    )
    if requirement.outcome.required
    else None
  )
  attempt = kwargs["attempt"]
  return TaskResult(
    task_result_id=f"result:{attempt.attempt_id}",
    logical_task=kwargs["logical_task"],
    attempt=attempt,
    execution=ExecutionSettlement(status="succeeded"),
    outcome=outcome,
    values=TaskResultValues(
      terminal_narrative=terminal,
      projection=projection,
    ),
    observation=TaskObservation(
      transcript=TranscriptHandle(
        kind="child_transcript",
        owner_id=attempt.physical_task_id,
      ),
      activity=ActivityHandle(
        kind="child_activity",
        owner_id=attempt.physical_task_id,
      ),
      usage=UsageObservation(),
    ),
    provenance=TaskResultProvenance.model_validate(provenance),
  )


class _Runner:
  def __init__(self) -> None:
    self._full_session_id = "session-test"
    self._agent_session_log: object | None = object()
    self._spill_dir_provider: SpillSink | None = None
    self._log = EventLog()
    self.spawn_calls: list[dict[str, Any]] = []
    self.background_calls: list[dict[str, Any]] = []
    self.durable_events: list[dict[str, Any]] = []
    self.background_result_calls: list[dict[str, Any]] = []

  def _get_tool_definitions(self) -> list[dict[str, Any]]:
    return [{
      "name": "web_search",
      "description": "Read-only evidence search.",
      "input_schema": {"type": "object"},
    }]

  async def _append_durable_event(self, event: dict[str, Any]) -> object:
    self.durable_events.append(dict(event))
    return object()

  async def _confirm_durable_skill_event(
    self,
    event: dict[str, Any],
  ) -> dict[str, Any] | None:
    return dict(event) if event in self.durable_events else None

  async def spawn_sub_agent(self, task: str, **kwargs: Any):
    call = {"task": task, **kwargs}
    self.spawn_calls.append(call)
    return _spawn_result(call), None

  async def _register_background_task(self, **kwargs: Any):
    self.background_calls.append(dict(kwargs))
    return {"task_id": kwargs["task_id_override"], "status": "running"}, None

  async def get_background_result(
    self,
    tool_input: dict[str, Any],
  ):
    self.background_result_calls.append(dict(tool_input))
    return {"task_id": tool_input["task_id"], "status": "completed"}, None


def _handler(
  runner: _Runner | None,
  *,
  loader: SkillLoader | None,
  operation_catalog: Any | None = None,
  resolver: _CapabilityResolver | None = None,
  background_handlers: dict[str, Any] | None = None,
  operation_mcp_activator: Any | None = None,
  mcp_meta_inject_servers: frozenset[str] | None = None,
  mcp_session_inject_servers: frozenset[str] | None = None,
  fms_rebinder: Any | None = None,
  approval_store: Any | None = None,
  approval_policy: Any | None = None,
  trusted_research_file_id: int | None = None,
  mcp_client: Any | None = None,
  declared_local_tool_definitions: Any | None = None,
  parent_sessions: list[GatewaySession] | None = None,
):
  handlers = {"web_search": _tool}
  handlers.update(background_handlers or {})
  parent_session = GatewaySession(
    session_id="session-test",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="actor-test",
    owner_user_id="actor-test",
    user_email="actor@example.com",
    role="owner",
    tenant_id="tenant-test",
    channel="cli",
    auth_config={"provider": "openai", "api_key": "opaque"},
  )
  bind_session_approval_route(parent_session, approval_store, approval_policy)
  if parent_sessions is not None:
    parent_sessions.append(parent_session)
  return make_run_agent_handler(
    [runner],
    parent_session=parent_session,
    skill_loader=loader,
    operation_catalog=operation_catalog,
    mcp_client=mcp_client or _McpClient(),
    local_tool_handlers=handlers,
    capability_execution_resolver=resolver or _CapabilityResolver(),
    operation_mcp_activator=operation_mcp_activator,
    mcp_session_inject_servers=mcp_session_inject_servers,
    mcp_meta_inject_servers=mcp_meta_inject_servers,
    fms_rebinder=fms_rebinder,
    trusted_research_file_id=trusted_research_file_id,
    declared_local_tool_definitions=declared_local_tool_definitions,
  )


def test_rebound_run_agent_uses_child_authority_and_runner(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  owner_session_host_policy: Any,
) -> None:
  monkeypatch.delenv("HANK_MAX_SPAWN_DEPTH", raising=False)
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"

  _write_operation(tmp_path, execution_class="node.implement")
  profile_path = tmp_path / "filing-review.md"
  profile_path.write_text(
    profile_path.read_text(encoding="utf-8").replace(
      "agent_callable: true", "agent_callable: true\nchildren: [filing-review]",
    ),
    encoding="utf-8",
  )
  loader = SkillLoader(tmp_path)
  operation = _operation(loader)
  async def exercise() -> None:
    calls: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    root_sessions: list[GatewaySession] = []
    handler: Any = None

    class SlotRunner(_Runner):
      def __init__(self, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.slot = asyncio.Semaphore(1)

      async def spawn_sub_agent(self, task: str, **kwargs: Any):
        async with self.slot:
          call = {"task": task, **kwargs}
          calls.append(call)
          child_session = kwargs["sub_session"]
          child_runner = SlotRunner(self.depth + 1)
          handlers = {"run_agent": handler}
          rebind_job_handlers(handlers, JobScope(
            caller=lambda: current_parent_context(child_session),
            spill_sink=lambda: None,
            runner=lambda: child_runner,
            session=child_session,
          ))
          result, error = await handlers["run_agent"]({
            "operation": operation,
            "objective": "Collect nested evidence.",
            "background": False,
          })
          if self.depth == 0:
            assert error is None and result is not None
            assert self.slot.locked()
          else:
            assert result is None and error is not None
            refusals.append(error)
          return _spawn_result(call), None

    root_runner = SlotRunner(0)
    handler = _handler(
      root_runner, loader=loader, parent_sessions=root_sessions,
    )
    result, error = await asyncio.wait_for(handler({
      "objective": "Collect evidence.",
      "operation": operation,
      "background": False,
    }), timeout=5)
    assert error is None and result is not None
    assert len(calls) == 2
    child, grandchild = [
      call["admitted_task"].execution_snapshot.admitted_context
      for call in calls
    ]
    assert child.lineage.depth == 1
    assert grandchild.lineage.depth == 2
    assert grandchild.lineage.parent_job_id == (
      calls[0]["admitted_task"].attempt.physical_task_id
    )
    assert child.tool_grant == ("web_search",)
    assert grandchild.tool_grant == child.tool_grant
    assert len(refusals) == 1
    assert refusals[0]["code"] == "depth_exceeded"

  asyncio.run(exercise())


def test_rebound_run_agent_admits_typed_ticker_with_child_session_tenant(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  owner_session_host_policy: Any,
) -> None:
  monkeypatch.delenv("HANK_MAX_SPAWN_DEPTH", raising=False)
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"

  _write_operation(tmp_path, required_context=("ticker",), execution_class="node.implement")
  profile_path = tmp_path / "filing-review.md"
  profile_path.write_text(
    profile_path.read_text(encoding="utf-8").replace(
      "agent_callable: true", "agent_callable: true\nchildren: [filing-review]",
    ),
    encoding="utf-8",
  )
  loader = SkillLoader(tmp_path)
  operation = _operation(loader)

  async def exercise() -> None:
    calls: list[dict[str, Any]] = []
    root_sessions: list[GatewaySession] = []
    handler: Any = None

    class SlotRunner(_Runner):
      def __init__(self, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.slot = asyncio.Semaphore(1)

      async def spawn_sub_agent(self, task: str, **kwargs: Any):
        async with self.slot:
          call = {"task": task, **kwargs}
          calls.append(call)
          if self.depth == 0:
            child_session = kwargs["sub_session"]
            child_runner = SlotRunner(self.depth + 1)
            handlers = {"run_agent": handler}
            rebind_job_handlers(handlers, JobScope(
              caller=lambda: current_parent_context(child_session),
              spill_sink=lambda: None,
              runner=lambda: child_runner,
              session=child_session,
            ))
            result, error = await handlers["run_agent"]({
              "operation": operation,
              "objective": "Collect nested filing evidence.",
              "ticker": "AAPL",
              "background": False,
            })
            assert error is None and result is not None
            assert self.slot.locked()
          return _spawn_result(call), None

    root_runner = SlotRunner(0)
    handler = _handler(
      root_runner, loader=loader, parent_sessions=root_sessions,
    )
    result, error = await asyncio.wait_for(handler({
      "objective": "Lead the filing review.",
      "operation": operation,
      "ticker": "AAPL",
      "background": False,
    }), timeout=5)
    assert error is None and result is not None
    assert len(calls) == 2
    lead, grandchild = [call["admitted_task"] for call in calls]
    context = grandchild.execution_snapshot.admitted_context
    assert context.lineage.depth == 2
    assert context.lineage.parent_job_id == lead.attempt.physical_task_id
    ticker_input = next(
      binding for binding in grandchild.inputs if binding.name == "ticker"
    )
    assert ticker_input.context.content == "AAPL"
    assert ticker_input.source.actual_contract == TICKER_INPUT_CONTRACT
    assert ticker_input.source.owner.tenant_id == root_sessions[0].tenant_id
    assert ticker_input.source.owner.tenant_id == "tenant-test"

  asyncio.run(exercise())


def _write_operation(
  path: Path,
  *,
  execution_class: str,
  resumable: bool = True,
  required_context: tuple[str, ...] = (),
  operation_name: str = "filing-review",
  investment_start_route: bool = False,
) -> SkillLoader:
  path.mkdir(parents=True, exist_ok=True)
  required_context_yaml = (
    "\n".join(f"    - {name}" for name in required_context)
    if required_context
    else "    []"
  )
  scope_yaml = "scope: ticker\n" if "ticker" in required_context else ""
  allowed_tools_yaml = (
    "  - web_search\n  - start_quant_research"
    if investment_start_route
    else "  - web_search"
  )
  mcp_tools_yaml = (
    """mcp_tools:
  idea-workbench-mcp:
    - start_quant_research"""
    if investment_start_route
    else "mcp_tools: {}"
  )
  investment_tool_ref_yaml = (
    """
    - kind: mcp
      server_id: idea-workbench-mcp
      tool_id: start_quant_research"""
    if investment_start_route
    else ""
  )
  mutation_mode = "thesis_writer" if investment_start_route else "read_only"
  investment_capability_yaml = (
    """
    - name: state.mutate/v1
      required: true
      binding_modes: [live_tool]"""
    if investment_start_route
    else ""
  )
  (path / f"{operation_name}.md").write_text(
    f"""---
name: {operation_name}
execution_class: {execution_class}
version: '1.0'
agent_callable: true
agent_description: Review filing evidence.
mutation_mode: {mutation_mode}
resumable: {str(resumable).lower()}
{scope_yaml}allowed_tools:
{allowed_tools_yaml}
{mcp_tools_yaml}
semantic_metadata:
  required_context:
{required_context_yaml}
  tool_refs:
    - kind: local
      tool_id: web_search
{investment_tool_ref_yaml}
  capability_requirements:
    - name: web.read/v1
      required: true
      binding_modes: [live_tool]
{investment_capability_yaml}
---
Review the admitted filing evidence and return a source-aware conclusion.
""",
    encoding="utf-8",
  )
  return SkillLoader(path)


def _operation(loader: SkillLoader) -> dict[str, Any]:
  return next(
    item.snapshot.operation.model_dump(mode="json")
    for item in loader.list_callable_operations()
    if item.snapshot.operation.name != "explore"
  )


def _catalog_runtime(
  *,
  name: str = "catalog-review",
  exact_tool_ids: frozenset[str] = frozenset({"web_search"}),
  mcp_tools_by_server: dict[str, frozenset[str]] | None = None,
  required_capabilities: tuple[SemanticCapabilityRequirement, ...] = (),
  session_inject_servers: frozenset[str] = frozenset(),
  workspace_scope: WorkspaceScope = "read_only",
  mutation_mode: MutationMode = "read_only",
  effort: Effort | None = None,
) -> ResolvedOperationRuntime:
  snapshot = build_agent_operation_snapshot(
    name=name,
    version="1.0",
    methodology_instructions="Review canonical evidence.",
    resolved_instructions="Review canonical evidence.",
    description="Review canonical evidence.",
    execution_class="node.explore",
    effort=effort,
    required_capabilities=required_capabilities,
    workspace_scope=workspace_scope,
    resumable=True,
  )
  mcp_tools = mcp_tools_by_server or {}
  return ResolvedOperationRuntime(
    snapshot=snapshot,
    policy=OperationRuntimePolicy(
      semantic_scope="global",
      state_class="producer",
      persist_state=False,
      resumable=True,
      resume_mcp_session_reset_ok=True,
      state_dir=None,
      mutation_mode=mutation_mode,
      run_mode="full",
      exact_tool_ids=exact_tool_ids,
      mcp_tools_by_server=mcp_tools,
      runtime_server_refs=frozenset(mcp_tools),
      session_inject_servers=session_inject_servers,
      max_tokens=2_222,
      effort=effort,
    ),
  )


class _Catalog:
  def __init__(self, resolved: ResolvedOperationRuntime) -> None:
    self.resolved = resolved
    self.selectors: list[object] = []

  def resolve_operation(self, selector: object) -> ResolvedOperationRuntime:
    self.selectors.append(selector)
    if selector is not None:
      requested = AgentOperationRef.model_validate(selector)
      if requested != self.resolved.snapshot.operation:
        raise ValueError("unknown operation")
    return self.resolved

  def list_callable_operations_with_descriptions(
    self,
  ) -> list[tuple[AgentOperationRef, str]]:
    return [(self.resolved.snapshot.operation, self.resolved.snapshot.description)]


class _PrefixedMcpClient(_McpClient):
  def __init__(self) -> None:
    self.active = False

  def activate(self) -> None:
    self.active = True

  def resolve_tool_name(
    self,
    server_name: str,
    original_name: str,
  ) -> str | None:
    if (
      self.active
      and server_name == "research-corpus-mcp"
      and original_name == "thesis_read"
    ):
      return "private_thesis_read"
    return None

  def get_server_for_tool(self, name: str) -> str | None:
    return (
      "research-corpus-mcp"
      if self.active and name == "private_thesis_read"
      else None
    )

  def get_original_tool_name(self, name: str) -> str:
    return (
      "thesis_read"
      if name == "private_thesis_read"
      else name
    )

  def get_policy_tool_name(self, name: str) -> str | None:
    if not self.is_mcp_tool(name):
      return None
    return self.get_original_tool_name(name)

  def is_mcp_tool(self, name: str) -> bool:
    return self.get_server_for_tool(name) is not None

  def get_server_tool_definitions(
    self,
    server_names: set[str],
  ) -> list[dict[str, Any]]:
    if not self.active or "research-corpus-mcp" not in server_names:
      return []
    return [{
      "name": "private_thesis_read",
      "description": "Read private research evidence.",
      "input_schema": {"type": "object"},
    }]

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return self.get_server_tool_definitions({"research-corpus-mcp"})

  def get_server_tool_route_bindings(
    self,
    server_names: set[str],
  ) -> tuple[LiveToolRouteBinding, ...]:
    if not self.active or "research-corpus-mcp" not in server_names:
      return ()
    return (_mcp_route_binding(
      server_id="research-corpus-mcp",
      logical_name="thesis_read",
      exposed_name="private_thesis_read",
    ),)


@pytest.fixture(autouse=True)
def delegation_tool_policy(owner_session_host_policy):
  """Describe this module's fake local handlers and live MCP surfaces."""
  local_effects = {
    "web_search": "read",
    "propose_record": "preview",
    "file_read": "read",
    "write_record": "state_write",
    "code_execute": "read",
    "code_execute_status": "read",
  }
  corpus = _PrefixedMcpClient()
  corpus.activate()
  server_effects = {}
  server_by_tool = {}
  for client, effect in (
    (corpus, "read"),
    (_InvestmentMcpClient(), "state_write"),
  ):
    for definition in client.get_tool_definitions():
      name = definition["name"]
      policy_name = client.get_policy_tool_name(name)
      server = client.get_server_for_tool(name)
      server_by_tool[policy_name] = server
      server_effects[server, policy_name] = effect
  owner_session_host_policy.get_local_tool_effect = local_effects.get
  owner_session_host_policy.get_server_for_policy_tool = server_by_tool.get
  owner_session_host_policy.get_tool_class = (
    lambda server, name: server_effects.get((server, name))
  )


def test_run_agent_tool_schema_is_operation_first(tmp_path: Path) -> None:
  schema = make_run_agent_tool_def(_write_operation(tmp_path, execution_class="node.implement"))["input_schema"]

  assert schema["required"] == ["objective"]
  assert schema["additionalProperties"] is False
  assert set(schema["properties"]) >= {
    "operation", "objective", "ticker", "research_file_id", "background",
  }
  ticker = schema["properties"]["ticker"]
  assert "pattern" not in ticker
  assert "independently of objective prose" in ticker["description"]
  assert schema["properties"]["research_file_id"]["minimum"] == 1
  threshold = schema["properties"]["cost_observation_threshold_usd"]
  assert threshold["exclusiveMinimum"] == 0
  assert "never stops or fails" in threshold["description"]
  assert "max_budget_usd" not in schema["properties"]
  assert "agent" not in schema["properties"]
  assert "task" not in schema["properties"]


def _dispatch_grant_for_runtime(runtime: ResolvedOperationRuntime) -> frozenset[str]:
  ceiling = runtime.policy.exact_tool_ids
  authority = admit_operation_tools(
    runtime.snapshot,
    grant_id="grant:catalog-render",
    operation_tool_ids=ceiling,
    definitions=tuple({"name": name} for name in sorted(ceiling)),
    local_tool_handlers={name: object() for name in ceiling},
    mcp_client=_McpClient(),
  )
  assert isinstance(authority, ResolvedAuthority)
  return granted_tool_ids(authority)


def test_catalog_dispatch_grant_filters_child_surface_not_ceiling() -> None:
  class _CeilingRunner(_Runner):
    def _get_tool_definitions(self) -> list[dict[str, Any]]:
      return [{
        "name": name,
        "description": f"Definition for {name}.",
        "input_schema": {"type": "object"},
      } for name in ("file_read", "web_search", "write_record")]

  runtime = _catalog_runtime(
    exact_tool_ids=frozenset({"file_read", "web_search", "write_record"}),
    required_capabilities=(
      SemanticCapabilityRequirement(
        name="web.read/v1",
        required=True,
        binding_modes=("live_tool",),
      ),
      SemanticCapabilityRequirement(
        name="corpus.read/v1",
        required=True,
        binding_modes=("live_tool",),
      ),
    ),
  )
  runner = _CeilingRunner()
  result, error = asyncio.run(_handler(
    runner,
    loader=None,
    operation_catalog=_Catalog(runtime),
    background_handlers={"file_read": _tool, "write_record": _tool},
  )({
    "operation": runtime.snapshot.operation.model_dump(mode="json"),
    "objective": "Review canonical evidence.",
    "background": False,
  }))

  assert error is None
  assert result is not None
  spawn = runner.spawn_calls[0]
  expected = _dispatch_grant_for_runtime(runtime)
  admitted = frozenset(
    entry.tool_id for entry in spawn["admitted_task"].tool_grant.tools
  )
  assert admitted == expected
  assert set(spawn["dispatcher"]._local) == expected
  assert {
    definition["name"]
    for definition in spawn["dispatcher"].get_tool_definitions()
  } == expected
  assert "write_record" in runtime.policy.exact_tool_ids
  assert "write_record" not in admitted


def test_declared_local_propose_route_is_admitted_when_parent_has_not_loaded_it() -> None:
  """A loadable local propose tool missing from the parent payload is still granted.

  Parent-prompt deferral withholds the schema until load_tools. The operation
  declared the tool and the parent can route it, so admission must not treat
  the missing advertisement as a missing route.
  """
  propose = "propose_record"

  class _DeferredParent(_Runner):
    def _get_tool_definitions(self) -> list[dict[str, Any]]:
      return [{
        "name": "web_search",
        "description": "Read-only evidence search.",
        "input_schema": {"type": "object"},
      }]

  runtime = _catalog_runtime(
    name="position-initiation-predecision",
    exact_tool_ids=frozenset({"web_search", propose}),
    workspace_scope="workspace_write",
    mutation_mode="preview",
    required_capabilities=(
      SemanticCapabilityRequirement(
        name="artifact.propose/v1",
        required=True,
        binding_modes=("live_tool",),
      ),
    ),
  )
  runner = _DeferredParent()
  result, error = asyncio.run(_handler(
    runner,
    loader=None,
    operation_catalog=_Catalog(runtime),
    background_handlers={propose: _tool},
    declared_local_tool_definitions=lambda names: (
      [{
        "name": propose,
        "description": "Propose a position-initiation predecision.",
        "input_schema": {"type": "object"},
      }]
      if propose in names
      else []
    ),
  )({
    "operation": runtime.snapshot.operation.model_dump(mode="json"),
    "objective": "Work the pre-decision checks.",
    "background": False,
  }))

  assert error is None, error
  assert result is not None
  spawn = runner.spawn_calls[0]
  admitted = frozenset(
    entry.tool_id for entry in spawn["admitted_task"].tool_grant.tools
  )
  assert propose in admitted
  assert {
    definition["name"]
    for definition in spawn["dispatcher"].get_tool_definitions()
  } >= {propose}



def test_run_agent_handler_rejects_catalog_and_loader_together(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  catalog = _Catalog(_catalog_runtime())

  with pytest.raises(ValueError, match="either skill_loader or operation_catalog"):
    _handler(None, loader=loader, operation_catalog=catalog)


def test_falsey_catalog_has_same_schema_and_handler_operation_view() -> None:
  class _FalseyCatalog(_Catalog):
    def __bool__(self) -> bool:
      return False

  runtime = _catalog_runtime()
  catalog = _FalseyCatalog(runtime)
  entries = _catalog_operation_entries(catalog)
  schema = make_run_agent_tool_def(catalog)["input_schema"]
  runner = _Runner()

  result, error = asyncio.run(_handler(
    runner,
    loader=None,
    operation_catalog=catalog,
  )({
    "operation": runtime.snapshot.operation.model_dump(mode="json"),
    "objective": "Review canonical evidence.",
    "background": False,
  }))

  assert "catalog-review@1.0" in schema["properties"]["operation"][
    "description"
  ]
  assert error is None
  assert result is not None
  assert entries == [(
    runtime.snapshot.operation,
    runtime.snapshot.description,
  )]
  assert catalog.selectors == [
    runtime.snapshot.operation.model_dump(mode="json"),
  ]


def test_injected_catalog_admission_consumes_snapshot_and_policy_directly() -> None:
  resolved = _catalog_runtime(effort="low")
  catalog = _Catalog(resolved)
  runner = _Runner()
  operation_ref = resolved.snapshot.operation.model_dump(mode="json")

  result, error = asyncio.run(_handler(
    runner,
    loader=None,
    operation_catalog=catalog,
  )({
    "operation": operation_ref,
    "objective": "Review canonical evidence.",
    "background": False,
  }))

  assert error is None
  assert result is not None
  assert catalog.selectors == [operation_ref]
  assert len(runner.spawn_calls) == 1
  spawn = runner.spawn_calls[0]
  assert spawn["capability_execution"].bind.effort == "low"
  assert "max_turns" not in spawn
  assert spawn["max_tokens"] == 2_222
  assert "max_budget_usd" not in spawn
  assert TaskResult.model_validate(result).logical_task.operation == (
    resolved.snapshot.operation
  )


@pytest.mark.parametrize(
  ("result_modes", "with_projection"),
  [
    (("domain_result",), False),
    (("narrative", "domain_result"), False),
    (("narrative",), True),
  ],
)
def test_delegation_does_not_refuse_declared_result_metadata(
  result_modes: tuple[str, ...],
  with_projection: bool,
) -> None:
  runtime = _catalog_runtime()
  payload = runtime.snapshot.model_dump(mode="json")
  payload["result_modes"] = result_modes
  payload["projection_contracts"] = (
    (runtime.snapshot.prompt.model_dump(mode="json"),)
    if with_projection else ()
  )
  payload["operation"].pop("digest")
  payload["operation"]["digest"] = sha256_digest(payload)
  operation = AgentOperationSnapshot.model_validate(payload)
  catalog = _Catalog(ResolvedOperationRuntime(
    snapshot=operation,
    policy=runtime.policy,
  ))
  runner = _Runner()

  result, error = asyncio.run(_handler(
    runner,
    loader=None,
    operation_catalog=catalog,
  )({
    "operation": operation.operation.model_dump(mode="json"),
    "objective": "Review canonical evidence.",
    "background": False,
  }))

  assert error is None
  canonical = TaskResult.model_validate(result)
  assert canonical.execution.status == "succeeded"
  assert canonical.logical_task.operation == operation.operation
  assert len(runner.spawn_calls) == 1


def test_child_provider_definitions_reuse_the_admission_snapshot() -> None:
  class _ChangingRunner(_Runner):
    def __init__(self) -> None:
      super().__init__()
      self.definition_reads = 0

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
      self.definition_reads += 1
      return [{
        "name": "web_search",
        "description": f"schema read {self.definition_reads}",
        "input_schema": {
          "type": "object",
          "properties": {
            f"query_{self.definition_reads}": {"type": "string"},
          },
        },
      }]

  resolved = _catalog_runtime(
    required_capabilities=(SemanticCapabilityRequirement(
      name="web.read/v1",
      required=True,
      binding_modes=("live_tool",),
    ),),
  )
  runner = _ChangingRunner()
  result, error = asyncio.run(_handler(
    runner,
    loader=None,
    operation_catalog=_Catalog(resolved),
  )({
    "operation": resolved.snapshot.operation.model_dump(mode="json"),
    "objective": "Review canonical evidence.",
    "background": False,
  }))

  assert error is None
  assert result is not None
  assert runner.definition_reads == 1
  dispatcher = runner.spawn_calls[0]["dispatcher"]
  first = dispatcher.get_tool_definitions()
  first[0]["description"] = "caller mutation"
  second = dispatcher.get_tool_definitions()
  assert runner.definition_reads == 1
  assert second == [{
    "name": "web_search",
    "description": "schema read 1",
    "input_schema": {
      "type": "object",
      "properties": {"query_1": {"type": "string"}},
    },
  }]


def test_default_catalog_explore_activates_prefixed_mcp_before_projection() -> None:
  runtime = _catalog_runtime(
    name="explore",
    exact_tool_ids=frozenset({
      "mcp__research-corpus-mcp__thesis_read"
    }),
    mcp_tools_by_server={
      "research-corpus-mcp": frozenset({"thesis_read"}),
    },
    required_capabilities=(SemanticCapabilityRequirement(
      name="corpus.read/v1",
      required=False,
      binding_modes=("live_tool",),
    ),),
    session_inject_servers=frozenset({"research-corpus-mcp"}),
  )
  catalog = _Catalog(runtime)
  mcp_client = _PrefixedMcpClient()
  activated: list[ResolvedOperationRuntime] = []
  runner = _Runner()

  def _activate(resolved: ResolvedOperationRuntime) -> None:
    activated.append(resolved)
    mcp_client.activate()

  result, error = asyncio.run(_handler(
    runner,
    loader=None,
    operation_catalog=catalog,
    operation_mcp_activator=_activate,
    mcp_client=mcp_client,
  )({
    "objective": "Explore the admitted research evidence.",
    "background": False,
  }))

  assert error is None
  assert result is not None
  assert catalog.selectors == [None]
  assert activated == [runtime]
  spawn = runner.spawn_calls[0]
  assert spawn["dispatcher"]._allowed_mcp_tools_by_server == {
    "research-corpus-mcp": {"private_thesis_read"},
  }
  assert {
    definition["name"]
    for definition in spawn["dispatcher"].get_tool_definitions()
  } == {"private_thesis_read"}


def test_catalog_session_injection_cannot_widen_caller_allowlist() -> None:
  runtime = _catalog_runtime(
    exact_tool_ids=frozenset({
      "mcp__research-corpus-mcp__thesis_read"
    }),
    mcp_tools_by_server={
      "research-corpus-mcp": frozenset({"thesis_read"}),
    },
    required_capabilities=(SemanticCapabilityRequirement(
      name="corpus.read/v1",
      required=False,
      binding_modes=("live_tool",),
    ),),
    session_inject_servers=frozenset({"research-corpus-mcp"}),
  )
  mcp_client = _PrefixedMcpClient()
  runner = _Runner()

  result, error = asyncio.run(_handler(
    runner,
    loader=None,
    operation_catalog=_Catalog(runtime),
    operation_mcp_activator=lambda _policy: mcp_client.activate(),
    mcp_client=mcp_client,
    mcp_session_inject_servers=frozenset(),
  )({
    "operation": runtime.snapshot.operation.model_dump(mode="json"),
    "objective": "Review the admitted research evidence.",
    "background": False,
  }))

  assert error is None
  assert result is not None
  dispatcher = runner.spawn_calls[0]["dispatcher"]
  assert dispatcher._mcp_session_inject_servers == set()


def test_default_catalog_explore_has_explicit_ref_lifecycle_parity() -> None:
  runtime = _catalog_runtime(
    name="explore",
    required_capabilities=(SemanticCapabilityRequirement(
      name="web.read/v1",
      required=False,
      binding_modes=("live_tool",),
    ),),
  )
  runs: list[
    tuple[_Runner, dict[str, Any] | AgentCompletionEnvelope]
  ] = []
  for selector in (
    None,
    runtime.snapshot.operation.model_dump(mode="json"),
  ):
    runner = _Runner()
    invocation: dict[str, Any] = {
      "objective": "Explore the admitted web evidence.",
      "background": False,
    }
    if selector is not None:
      invocation["operation"] = selector

    result, error = asyncio.run(_handler(
      runner,
      loader=None,
      operation_catalog=_Catalog(runtime),
    )(invocation))

    assert error is None
    assert result is not None
    runs.append((runner, result))

  for runner, result in runs:
    assert TaskResult.model_validate(result).execution.status == "succeeded"
    assert [event["type"] for event in runner.durable_events] == [
      "skill_run_started",
      "skill_result_captured",
    ]
    started = runner.durable_events[0]
    assert runner.spawn_calls[0]["skill_run_id"] == started["skill_run_id"]
    assert (
      runner.spawn_calls[0]["dispatcher"]._run_context.run_id
      == started["skill_run_id"]
    )


def test_policy_projection_preserves_distinct_prefixed_same_name_routes() -> None:
  class _TwoServerMcpClient:
    routes = {
      ("server-a", "shared_tool"): "a_shared_tool",
      ("server-b", "shared_tool"): "b_shared_tool",
    }

    def get_server_tool_route_bindings(
      self,
      server_names: set[str],
    ) -> tuple[LiveToolRouteBinding, ...]:
      return tuple(
        _mcp_route_binding(
          server_id=server,
          logical_name=original,
          exposed_name=exposed,
        )
        for (server, original), exposed in self.routes.items()
        if server in server_names
      )

  runtime = _catalog_runtime(
    exact_tool_ids=frozenset({
      "mcp__server-a__shared_tool",
      "mcp__server-b__shared_tool",
    }),
    mcp_tools_by_server={
      "server-a": frozenset({"shared_tool"}),
      "server-b": frozenset({"shared_tool"}),
    },
  )

  exact, canonical_to_exposed = (
    _runtime_policy_dispatch_projection(
      runtime.policy,
      mcp_client=_TwoServerMcpClient(),
    )
  )

  assert exact == frozenset({"a_shared_tool", "b_shared_tool"})
  assert canonical_to_exposed == {
    "mcp__server-a__shared_tool": "a_shared_tool",
    "mcp__server-b__shared_tool": "b_shared_tool",
  }


def test_policy_projection_uses_logical_live_route_identity() -> None:
  class _LogicalRouteMcpClient:
    @staticmethod
    def get_server_tool_route_bindings(
      server_names: set[str],
    ) -> tuple[LiveToolRouteBinding, ...]:
      if "market-data-mcp" not in server_names:
        return ()
      return (_mcp_route_binding(
        server_id="market-data-mcp",
        logical_name="fetch_company_profile",
        exposed_name="fetch_company_profile",
        transport_server_id="fmp-mcp",
        provider_original_name="fmp_profile",
        route_kind="logical",
      ),)

  runtime = _catalog_runtime(
    exact_tool_ids=frozenset({
      "mcp__market-data-mcp__fetch_company_profile",
    }),
    mcp_tools_by_server={
      "market-data-mcp": frozenset({"fetch_company_profile"}),
    },
  )

  exact, canonical_to_exposed = (
    _runtime_policy_dispatch_projection(
      runtime.policy,
      mcp_client=_LogicalRouteMcpClient(),
    )
  )

  assert exact == frozenset({"fetch_company_profile"})
  assert canonical_to_exposed == {
    "mcp__market-data-mcp__fetch_company_profile": (
      "fetch_company_profile"
    ),
  }


def test_catalog_projection_callback_failure_is_operation_unavailable() -> None:
  class _FailingMcpClient(_PrefixedMcpClient):
    def get_server_tool_route_bindings(
      self,
      server_names: set[str],
    ) -> tuple[LiveToolRouteBinding, ...]:
      _ = server_names
      raise RuntimeError("route registry failed")

  runtime = _catalog_runtime(
    exact_tool_ids=frozenset({
      "mcp__research-corpus-mcp__thesis_read"
    }),
    mcp_tools_by_server={
      "research-corpus-mcp": frozenset({"thesis_read"}),
    },
  )

  result, error = asyncio.run(_handler(
    _Runner(),
    loader=None,
    operation_catalog=_Catalog(runtime),
    mcp_client=_FailingMcpClient(),
  )({
    "operation": runtime.snapshot.operation.model_dump(mode="json"),
    "objective": "Review the canonical evidence.",
    "background": False,
  }))

  assert result is None
  assert error is not None
  assert error["code"] == "operation_unavailable"
  assert "live route resolution failed" in error["message"]


def test_catalog_mcp_alias_cannot_impersonate_live_local_handler() -> None:
  class _LocalAliasMcpClient(_PrefixedMcpClient):
    def get_server_tool_route_bindings(
      self,
      server_names: set[str],
    ) -> tuple[LiveToolRouteBinding, ...]:
      if not self.active or "research-corpus-mcp" not in server_names:
        return ()
      return (_mcp_route_binding(
        server_id="research-corpus-mcp",
        logical_name="thesis_read",
        exposed_name="web_search",
      ),)

    def get_server_for_tool(self, name: str) -> str | None:
      return (
        "research-corpus-mcp"
        if self.active and name == "web_search"
        else None
      )

    def get_original_tool_name(self, name: str) -> str:
      return "thesis_read" if name == "web_search" else name

  runtime = _catalog_runtime(
    exact_tool_ids=frozenset({"mcp__research-corpus-mcp__thesis_read"}),
    mcp_tools_by_server={"research-corpus-mcp": frozenset({"thesis_read"})},
  )
  mcp_client = _LocalAliasMcpClient()
  runner = _Runner()

  result, error = asyncio.run(_handler(
    runner,
    loader=None,
    operation_catalog=_Catalog(runtime),
    operation_mcp_activator=lambda _resolved: mcp_client.activate(),
    mcp_client=mcp_client,
  )({
    "operation": runtime.snapshot.operation.model_dump(mode="json"),
    "objective": "Review the canonical evidence.",
    "background": False,
  }))

  assert result is None
  assert error is not None
  assert error["code"] == "operation_unavailable"
  assert "live local handlers collide" in error["message"]
  assert runner.spawn_calls == []
  assert runner.background_calls == []


def test_policy_projection_does_not_mask_base_exceptions() -> None:
  class _InterruptedMcpClient(_PrefixedMcpClient):
    def get_server_tool_route_bindings(
      self,
      server_names: set[str],
    ) -> tuple[LiveToolRouteBinding, ...]:
      _ = server_names
      raise KeyboardInterrupt

  runtime = _catalog_runtime(
    exact_tool_ids=frozenset({
      "mcp__research-corpus-mcp__thesis_read"
    }),
    mcp_tools_by_server={
      "research-corpus-mcp": frozenset({"thesis_read"}),
    },
  )

  with pytest.raises(KeyboardInterrupt):
    _runtime_policy_dispatch_projection(
      runtime.policy,
      mcp_client=_InterruptedMcpClient(),
    )


def test_resume_handler_has_no_live_catalog_dependency() -> None:
  assert "operation_catalog" not in inspect.signature(
    make_resume_handler
  ).parameters


def test_ticker_precedence_equal_typed_and_verified_skips_fallback() -> None:
  calls = 0

  def fallback() -> str:
    nonlocal calls
    calls += 1
    return "MSFT"

  assert _resolve_context_ticker(
    typed_ticker=" googl ",
    verified_server_ticker="GOOGL",
    prose_fallback=fallback,
  ) == "GOOGL"
  assert calls == 0


def test_ticker_precedence_conflicting_typed_facts_fail_without_fallback() -> None:
  calls = 0

  def fallback() -> str:
    nonlocal calls
    calls += 1
    return "MSFT"

  with pytest.raises(ValueError, match="conflicts"):
    _resolve_context_ticker(
      typed_ticker="GOOGL",
      verified_server_ticker="MSFT",
      prose_fallback=fallback,
    )
  assert calls == 0


def test_ticker_precedence_verified_server_subject_skips_fallback() -> None:
  calls = 0

  def fallback() -> str:
    nonlocal calls
    calls += 1
    return "MSFT"

  assert _resolve_context_ticker(
    typed_ticker=None,
    verified_server_ticker=" qcom ",
    prose_fallback=fallback,
  ) == "QCOM"
  assert calls == 0


def test_ticker_precedence_only_absent_typed_facts_invoke_fallback_once() -> None:
  calls = 0

  def fallback() -> str:
    nonlocal calls
    calls += 1
    return "PCTY"

  assert _resolve_context_ticker(
    typed_ticker=None,
    verified_server_ticker=None,
    prose_fallback=fallback,
  ) == "PCTY"
  assert calls == 1


def test_ticker_admitted_binding_rejects_alternate_mechanics() -> None:
  binding = _ticker_admitted_input(
    "GOOGL",
    invocation_id="task-1",
    parent_session=SimpleNamespace(
      tenant_id="tenant-1",
      session_id="session-1",
    ),
  )
  assert _ticker_from_admitted_inputs(
    (binding,),
    required=True,
    owner_invocation_id="task-1",
  ) == "GOOGL"

  with_context_policy = binding.model_copy(update={
    "source": binding.source.model_copy(update={
      "request": binding.source.request.model_copy(update={
        "context_policy": object(),
      }),
    }),
  })
  with_read_grant = binding.model_copy(update={
    "source": binding.source.model_copy(update={"read_grant": object()}),
  })
  wrong_owner = binding.model_copy(update={
    "source": binding.source.model_copy(update={
      "owner": binding.source.owner.model_copy(update={
        "tenant_id": "",
      }),
    }),
  })

  for alternate in (with_context_policy, with_read_grant, wrong_owner):
    with pytest.raises(ValueError):
      _ticker_from_admitted_inputs(
        (alternate,),
        required=True,
        owner_invocation_id="task-1",
      )


def test_typed_ticker_admission_requires_exact_tenant_and_session_owner() -> None:
  with pytest.raises(ValueError, match="tenant and session ownership"):
    _ticker_admitted_input(
      "GOOGL",
      invocation_id="task-1",
      parent_session=SimpleNamespace(user_id="alice"),
    )


def test_direct_handler_rejects_retired_budget_input_without_dispatch(
  tmp_path: Path,
) -> None:
  runner = _Runner()

  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path),
  )({
    "objective": "Research the question.",
    "max_budget_usd": 10,
  }))

  assert result is None
  assert error is not None and error["code"] == "invalid_input"
  assert "max_budget_usd" in error["message"]
  assert runner.spawn_calls == []
  assert runner.background_calls == []


def test_direct_handler_typed_ticker_bypasses_prose_and_is_admitted(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  loader = _write_operation(tmp_path, required_context=("ticker",), execution_class="node.implement")
  runner = _Runner()
  extractor_calls = 0

  def _unexpected_extractor(_objective: str) -> str | None:
    nonlocal extractor_calls
    extractor_calls += 1
    return "TRACK"

  monkeypatch.setattr(
    "agent_gateway.sub_agent._extract_ticker_from_task",
    _unexpected_extractor,
  )

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "TRACK 1: review the filing evidence.",
    "ticker": "GOOGL",
    "background": True,
  }))

  assert error is None
  assert isinstance(result, dict)
  assert result["status"] == "running"
  assert extractor_calls == 0
  admitted = runner.background_calls[0]["admitted_task"]
  assert len(admitted.inputs) == 1
  binding = admitted.inputs[0]
  assert binding.name == "ticker"
  assert binding.source.source_kind == "invocation_argument"
  assert binding.source.request.selector.argument_name == "ticker"
  assert binding.source.actual_contract == TICKER_INPUT_CONTRACT
  assert binding.source.owner.tenant_id == "tenant-test"
  assert binding.source.owner.session_id == "session-test"
  assert binding.source.owner.invocation_id == admitted.attempt.physical_task_id
  assert binding.context.content == "GOOGL"
  dispatch_result, dispatch_error = asyncio.run(
    runner.background_calls[0]["handler"](
      runner.background_calls[0]["tool_input"],
    )
  )
  assert dispatch_error is None
  assert dispatch_result is not None
  assert runner.spawn_calls != []
  started = next(
    event for event in runner.durable_events
    if event["type"] == "skill_run_started"
  )
  assert started["ticker"] == "GOOGL"
  assert extractor_calls == 0


@pytest.mark.parametrize("operation_name", ["filing-review", "quant-research"])
def test_non_investment_research_context_keeps_existing_asserted_id_dispatch(
  tmp_path: Path,
  operation_name: str,
) -> None:
  loader = _write_operation(
    tmp_path,
    required_context=("research_file_id",),
    operation_name=operation_name,
    execution_class="node.implement",
  )
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review the admitted research file.",
    "research_file_id": 42,
    "background": False,
  }))

  assert error is None
  assert result is not None
  assert len(runner.spawn_calls) == 1
  assert runner.spawn_calls[0]["dispatcher"]._run_context.research_file_id == 42


# The child must receive exact server-returned refs, not copies in objective prose.
@pytest.mark.parametrize(
  ("operation_name", "required_context", "objective", "prior_results", "expected_refs"),
  [
    (
      "postcompile-valuation",
      ("ticker", "research_file_id", "model_version_ref", "source_bundle_id", "valuation_policy_ref"),
      "Persist the postcompile valuation for the current PCTY model through "
      "your context-bound terminal door, passing the typed context_ref your "
      "context tool returns.",
      [{
        "model_version_ref": {"model_id": "model-1", "version": 7},
        "source_bundle_id": "bundle-exact",
        "valuation_policy_ref": {"policy_id": "vp-1", "content_sha256": "a" * 64},
      }],
      {
        "model_version_ref": {"model_id": "model-1", "version": 7},
        "source_bundle_id": "bundle-exact",
        "valuation_policy_ref": {"policy_id": "vp-1", "content_sha256": "a" * 64},
      },
    ),
    (
      "earnings-scenarios",
      ("ticker", "research_file_id", "forecast_assumption_set_ref", "critical_factors_ref"),
      "Build and persist the PCTY ScenarioSet through fms_persist_scenario_set.",
      [
        {"readback": {"forecast_assumption_set_ref": {"set_id": "fas-1", "content_sha256": "b" * 64}}},
        {"readback": {
          "skill": "critical-factors",
          "artifact_id": "cf-exact",
          "content_sha256": "c" * 64,
          "ticker": "PCTY",
          "schema_version": "critical-factors-ref.v1",
        }},
      ],
      {
        "forecast_assumption_set_ref": {"set_id": "fas-1", "content_sha256": "b" * 64},
        "critical_factors_ref": {
          "skill": "critical-factors", "artifact_id": "cf-exact", "content_sha256": "c" * 64,
        },
      },
    ),
    (
      "build-model",
      ("ticker",),
      "Build the PCTY model from the current research file state and persist "
      "it through your terminal door.",
      [],
      {},
    ),
  ],
)
def test_named_child_first_message_carries_the_run_binding(
  tmp_path: Path,
  operation_name: str,
  required_context: tuple[str, ...],
  objective: str,
  prior_results: list[dict[str, Any]],
  expected_refs: dict[str, Any],
) -> None:
  loader = _write_operation(
    tmp_path,
    required_context=required_context,
    operation_name=operation_name,
    execution_class="node.implement",
  )
  runner = _Runner()
  for index, prior_result in enumerate(prior_results):
    runner._log.append({
      "type": "tool_call_complete",
      "tool_call_id": f"prior-{index}",
      "result": prior_result,
    })

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    trusted_research_file_id=1,
  )({
    "operation": _operation(loader),
    "objective": objective,
    "research_file_id": 1,
    "ticker": "PCTY",
    "background": False,
  }))

  assert error is None
  assert result is not None
  spawn = runner.spawn_calls[0]
  assert spawn["task"].startswith(f"Ticker: PCTY\nRESEARCH_FILE_ID=1\n{objective}")
  for key, value in expected_refs.items():
    line = next(line for line in spawn["task"].splitlines() if line.startswith(f"{key}="))
    assert json.loads(line.split("=", 1)[1]) == value
    binding = next(binding for binding in spawn["admitted_task"].inputs if binding.name == key)
    assert binding.context.content == value
  assert spawn["dispatcher"]._run_context.research_file_id == 1
  assert spawn["admitted_task"].objective == objective


@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("research_file_id", [None, 1])
@pytest.mark.parametrize("prior_result", [
  None,
  {"model_version_ref": {"model_id": "model-1", "version": 7}},
  {"model_version_ref": None, "source_bundle_id": None, "valuation_policy_ref": None},
])
def test_named_child_missing_required_refs_does_not_spawn(
  tmp_path: Path,
  prior_result: dict[str, Any] | None,
  research_file_id: int | None,
  background: bool,
) -> None:
  keys = ("model_version_ref", "source_bundle_id", "valuation_policy_ref")
  loader = _write_operation(tmp_path, operation_name="postcompile-valuation", required_context=keys, execution_class="node.implement")
  runner = _Runner()
  if prior_result is not None:
    runner._log.append({"type": "tool_call_complete", "result": prior_result})
  result, error = asyncio.run(_handler(
    runner, loader=loader, trusted_research_file_id=research_file_id,
  )({
    "operation": _operation(loader),
    "objective": "Price PCTY; model_version_ref from prose must not count.",
    "background": background,
  }))
  assert result is None
  assert error is not None
  assert error["code"] == "required_context_missing"
  assert error["missing_keys"] == [
    key for key in keys if (prior_result or {}).get(key) is None
  ]
  # The parent's own current-model read returns the refs; with no research
  # file bound there is no read to name.
  assert error["next_call"] == (
    None if research_file_id is None
    else {"tool_name": "get_current_model", "arguments": {"research_file_id": 1}}
  )
  assert runner.spawn_calls == []
  assert runner.background_calls == []


@pytest.mark.parametrize("subject_key", ["industry", "artifact"])
def test_named_child_subject_context_is_not_a_turn_ref(
  tmp_path: Path,
  subject_key: str,
) -> None:
  loader = _write_operation(
    tmp_path, operation_name="industry-onboarding", required_context=(subject_key,),
    execution_class="node.implement",
  )
  runner = _Runner()
  _, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Onboard the payroll software industry",
    "background": False,
  }))
  assert error is None
  assert len(runner.spawn_calls) == 1


def test_named_child_refs_are_scoped_to_current_turn_successful_results(
  tmp_path: Path,
) -> None:
  loader = _write_operation(
    tmp_path, operation_name="postcompile-valuation", required_context=("model_version_ref",),
    execution_class="node.implement",
  )
  runner = _Runner()
  handler = _handler(runner, loader=loader)
  call = {
    "operation": _operation(loader), "objective": "Price PCTY", "background": False,
  }
  runner._log.append({
    "type": "tool_call_complete",
    "result": {"model_version_ref": {"model_id": "model-1", "version": 1}},
  })
  runner._log.append({
    "type": "tool_call_complete",
    "result": {"model_version_ref": {"model_id": "model-1", "version": 2}},
  })
  runner._log.append({
    "type": "tool_call_complete", "is_error": True,
    "result": {"model_version_ref": {"model_id": "wrong-model", "version": 3}},
  })
  _, error = asyncio.run(handler(call))
  assert error is None
  binding = next(
    binding for binding in runner.spawn_calls[0]["admitted_task"].inputs
    if binding.name == "model_version_ref"
  )
  assert binding.context.content == {"model_id": "model-1", "version": 2}
  runner._log = EventLog()
  _, error = asyncio.run(handler(call))
  assert error is not None
  assert error["code"] == "required_context_missing"
  assert len(runner.spawn_calls) == 1


_SCENARIO_KEYS = (
  "ticker", "research_file_id", "forecast_assumption_set_ref", "critical_factors_ref",
)
_FAS_REF = {
  "schema_version": "forecast-assumption-set-ref.v1",
  "set_id": "fas-exact",
  "content_sha256": "b" * 64,
}
_CF_REF = {"skill": "critical-factors", "artifact_id": "cf-exact", "content_sha256": "c" * 64}


def _scenario_child_refs(runner: _Runner, tmp_path: Path) -> tuple[Any, Any]:
  loader = _write_operation(
    tmp_path, operation_name="earnings-scenarios", required_context=_SCENARIO_KEYS,
    execution_class="node.implement",
  )
  result, error = asyncio.run(_handler(runner, loader=loader, trusted_research_file_id=1)({
    "operation": _operation(loader),
    "objective": (
      "Build the PCTY ScenarioSet from forecast_assumption_set_ref="
      + json.dumps(_FAS_REF) + " and critical_factors_ref=" + json.dumps(_CF_REF)
    ),
    "research_file_id": 1,
    "ticker": "PCTY",
    "background": False,
  }))
  return result, error


def _bound_refs(runner: _Runner) -> dict[str, Any]:
  return {
    binding.name: binding.context.content
    for binding in runner.spawn_calls[0]["admitted_task"].inputs
    if binding.name.endswith("_ref")
  }


def _settled_task_result(
  name: str,
  values: TaskResultValues | None = None,
) -> TaskResult:
  return TaskResult(
    task_result_id=f"result:{name}",
    logical_task=OrdinaryDelegationTaskRef(
      delegation_id=f"delegation-{name}",
      operation=AgentOperationRef(
        namespace="agent-operation", name="child", version="1.0",
        digest=sha256_digest({"operation": "child"}),
      ),
    ),
    attempt=AttemptRef(attempt_number=1, attempt_id=f"{name}:1", physical_task_id=name),
    execution=ExecutionSettlement(status="succeeded"),
    outcome=AnalyticalOutcome(disposition="complete", assessment_source="domain_tool"),
    values=values if values is not None else TaskResultValues(
      projection=canonical_projection(
        contract=report_contract_ref(TERMINAL_TOOL_RESULT_RETURN_CONTRACT),
        value={"tool_name": name, "result": {"status": "done"}},
      ),
    ),
    observation=TaskObservation(
      transcript=TranscriptHandle(kind="child_transcript", owner_id=name),
      activity=ActivityHandle(kind="child_activity", owner_id=name),
      usage=UsageObservation(),
    ),
    provenance=TaskResultProvenance(
      admitted_task_digest=sha256_digest({"admitted": name}),
      model_bind_digest=sha256_digest({"model": name}),
      capability_binding_digest=sha256_digest({"capability": name}),
      tool_grant_digest=sha256_digest({"tools": name}),
    ),
  )


def _child_completion(
  tool_name: str,
  readback: dict[str, Any],
  *,
  door_bytes: int = 0,
) -> dict[str, Any]:
  """A ``run_agent`` result as the parent's turn records it: the compiled envelope.

  ``door_bytes`` pads the door's own answer (critical-factors returns ~365 KB)
  so the child's projection overflows the parent's inline bound.
  """
  value: JsonValue = {
    "tool_name": tool_name,
    "result": {"status": "staged", "readback": readback, "preview": "x" * door_bytes},
  }
  task_result = _settled_task_result(
    tool_name,
    TaskResultValues(projection=canonical_projection(
      contract=report_contract_ref(TERMINAL_TOOL_RESULT_RETURN_CONTRACT),
      value=value,
    )),
  )
  return build_agent_completion_envelope(
    task_result,
    policy=ordinary_parent_result_policy(task_result),
    terminal_narrative_reader=lambda _result: "unused",
    read_grant_factory=lambda source: ContentReadGrant(
      grant_id=f"grant:{source.content_sha256}",
      content_id=source.content_id,
      scope="direct_parent",
      principal_id="parent-runner",
    ),
  ).model_dump(mode="json")


def test_named_child_refs_come_from_prior_run_agent_child_results(tmp_path: Path) -> None:
  runner = _Runner()
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "fas-child", "tool_name": "run_agent",
    "result": _child_completion(
      "fms_persist_forecast_assumption_set",
      {"forecast_assumption_set_ref": _FAS_REF, "historical_basis_ref": {"artifact_id": "hf"}},
    ),
  })
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "cf-child", "tool_name": "run_agent",
    "result": _child_completion(
      "fms_persist_skill_artifact",
      {**_CF_REF, "ticker": "PCTY", "schema_version": "critical-factors-ref.v1"},
    ),
  })

  _, error = _scenario_child_refs(runner, tmp_path)

  assert error is None
  assert _bound_refs(runner) == {
    "forecast_assumption_set_ref": _FAS_REF,
    "critical_factors_ref": _CF_REF,
  }


def test_named_child_refs_come_from_an_overflowed_run_agent_child(tmp_path: Path) -> None:
  # PCTY, 2026-10-04 (sess_47cc5130…, seq 153 → 179): the critical-factors
  # child's door answer overflowed to a result handle, and earnings-scenarios
  # was refused required_context_missing critical_factors_ref.
  runner = _Runner()
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "fas-child", "tool_name": "run_agent",
    "result": _child_completion(
      "fms_persist_forecast_assumption_set", {"forecast_assumption_set_ref": _FAS_REF},
    ),
  })
  overflowed = _child_completion(
    "fms_propose_critical_factors",
    {**_CF_REF, "contract_name": "CriticalFactors"},
    door_bytes=DEFAULT_PARENT_RESULT_MAX_INLINE_BYTES,
  )
  assert overflowed["parent_materialization"]["kind"] == "result_handle"
  assert "value" not in overflowed["parent_materialization"]
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "cf-child", "tool_name": "run_agent",
    "result": overflowed,
  })

  _, error = _scenario_child_refs(runner, tmp_path)

  assert error is None
  assert _bound_refs(runner) == {
    "forecast_assumption_set_ref": _FAS_REF,
    "critical_factors_ref": _CF_REF,
  }


def _pipeline_result(stage_refs: dict[str, Any]) -> dict[str, Any]:
  """A ``run_name_pipeline`` result: artifact paths, and the bag at ``stage_refs``."""
  return {
    "ticker": "PCTY", "pipeline_id": "0daccdc75ee1", "status": "failed",
    "artifact_refs": ["artifacts/PCTY/critical-factors/cf-exact.json"],
    "stages": [{"skill": "critical-factors", "status": "completed", "artifact_refs": [
      "artifacts/PCTY/critical-factors/cf-exact.json",
    ]}],
    "stage_refs": stage_refs,
  }


@pytest.mark.parametrize("later_current_model", [False, True])
def test_named_child_refs_come_from_a_prior_pipelines_stage_refs(
  tmp_path: Path,
  later_current_model: bool,
) -> None:
  # PAYC, 2026-10-04: critical-factors completed inside run_name_pipeline, then
  # a run_agent forecast-assumptions child persisted the FAS. The pipeline's own
  # stages read the CF ref from its bag; the named child reads the same bag
  # from the pipeline's result. A ref a stage's door minted is not displaced by
  # a later current-model read.
  fas_compiled = {**_FAS_REF, "set_id": "fas-compiled"}
  runner = _Runner()
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "pipeline", "tool_name": "run_name_pipeline",
    "result": _pipeline_result({"critical_factors_ref": _CF_REF, "forecast_assumption_set_ref": _FAS_REF}),
  })
  if later_current_model:
    runner._log.append({
      "type": "tool_call_complete", "tool_call_id": "current-model", "tool_name": "get_current_model",
      "result": {"forecast_assumption_set_ref": fas_compiled},
    })

  _, error = _scenario_child_refs(runner, tmp_path)

  assert error is None
  assert _bound_refs(runner) == {
    "forecast_assumption_set_ref": _FAS_REF,
    "critical_factors_ref": _CF_REF,
  }


def test_a_pipeline_whose_stage_read_back_nothing_supplies_no_ref(tmp_path: Path) -> None:
  runner = _Runner()
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "pipeline", "tool_name": "run_name_pipeline",
    "result": _pipeline_result({}),
  })
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "fas-child", "tool_name": "run_agent",
    "result": _child_completion("fms_persist_forecast_assumption_set", {"forecast_assumption_set_ref": _FAS_REF}),
  })

  result, error = _scenario_child_refs(runner, tmp_path)

  assert result is None
  assert error is not None
  assert error["code"] == "required_context_missing"
  assert error["missing_keys"] == ["critical_factors_ref"]
  assert error["next_call"] == {
    "tool_name": "get_skill_artifact",
    "arguments": {"ticker": "PCTY", "skill": "critical-factors"},
  }
  assert runner.spawn_calls == []


@pytest.mark.parametrize("returns_header", [False, True])
def test_named_child_refs_come_from_a_successful_calls_input(
  tmp_path: Path,
  returns_header: bool,
) -> None:
  passed_cf_ref = {"artifact_id": "cf-exact", "content_sha256": "c" * 64}
  echoed_cf_ref = {**_CF_REF, "schema_version": "critical-factors-artifact-ref.v1"}
  runner = _Runner()
  runner._log.append({
    "type": "tool_call_start", "tool_call_id": "scenario-read",
    "tool_name": "get_scenario_set_context",
    "tool_input": {
      "research_file_id": 1,
      "forecast_assumption_set_ref": _FAS_REF,
      "critical_factors_ref": passed_cf_ref,
    },
  })
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "scenario-read",
    "tool_name": "get_scenario_set_context", "is_error": False,
    "result": (
      {"header": {"forecast_assumption_set_ref": _FAS_REF, "critical_factors_ref": echoed_cf_ref}}
      if returns_header else {"projection": {"periods": []}}
    ),
  })
  runner._log.append({
    "type": "tool_call_start", "tool_call_id": "refused-read",
    "tool_name": "get_scenario_set_context",
    "tool_input": {"critical_factors_ref": {"artifact_id": "cf-wrong", "content_sha256": "d" * 64}},
  })
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "refused-read",
    "tool_name": "get_scenario_set_context", "is_error": True,
    "result": {"status": "error", "error": "unknown critical_factors_ref"},
  })

  _, error = _scenario_child_refs(runner, tmp_path)

  assert error is None
  assert _bound_refs(runner) == {
    "forecast_assumption_set_ref": _FAS_REF,
    "critical_factors_ref": echoed_cf_ref if returns_header else passed_cf_ref,
  }


@pytest.mark.parametrize("later_door", [False, True])
def test_a_current_model_read_does_not_displace_a_stage_set_a_door_persisted(
  tmp_path: Path,
  later_door: bool,
) -> None:
  keys = (
    "model_version_ref", "forecast_assumption_set_ref", "scenario_set_ref", "valuation_policy_ref",
  )
  loader = _write_operation(tmp_path, operation_name="valuation-policy-precompile", required_context=keys, execution_class="node.implement")
  fas = {name: {"set_id": f"fas-{name}", "content_sha256": "b" * 64} for name in "ABC"}
  scenario = {name: {"set_id": f"ss-{name}", "content_sha256": "c" * 64} for name in "AB"}
  policy = {name: {"policy_id": f"vp-{name}", "content_sha256": "d" * 64} for name in "AB"}
  runner = _Runner()
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "fas-door",
    "result": {"status": "persisted", "readback": {
      "forecast_assumption_set_ref": fas["B"], "valuation_policy_ref": policy["B"],
    }},
  })
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "scenario-child", "tool_name": "run_agent",
    "result": _child_completion("fms_persist_scenario_set", {"scenario_set_ref": scenario["B"]}),
  })
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "current-model",
    "result": {
      "model_version_ref": {"model_id": "model-1", "version": 7},
      "forecast_assumption_set_ref": fas["A"],
      "scenario_set_ref": scenario["A"],
      "valuation_policy_ref": policy["A"],
    },
  })
  if later_door:
    runner._log.append({
      "type": "tool_call_complete", "tool_call_id": "fas-door-2",
      "result": {"status": "persisted", "readback": {"forecast_assumption_set_ref": fas["C"]}},
    })

  _, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader), "objective": "Precompile the PCTY policy", "background": False,
  }))

  assert error is None
  assert _bound_refs(runner) == {
    "model_version_ref": {"model_id": "model-1", "version": 7},
    "forecast_assumption_set_ref": fas["C"] if later_door else fas["B"],
    "scenario_set_ref": scenario["B"],
    "valuation_policy_ref": policy["A"],
  }


def test_named_child_refs_named_only_in_prose_or_refused_calls_do_not_spawn(
  tmp_path: Path,
) -> None:
  runner = _Runner()
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "note", "tool_name": "memory_read",
    "result": {"content": "forecast_assumption_set_ref=" + json.dumps(_FAS_REF)},
  })
  runner._log.append({
    "type": "tool_call_start", "tool_call_id": "refused-read",
    "tool_name": "get_scenario_set_context",
    "tool_input": {"forecast_assumption_set_ref": _FAS_REF, "critical_factors_ref": _CF_REF},
  })
  runner._log.append({
    "type": "tool_call_complete", "tool_call_id": "refused-read",
    "tool_name": "get_scenario_set_context", "is_error": True,
    "result": {"status": "error", "error": "stale forecast_assumption_set_ref"},
  })

  result, error = _scenario_child_refs(runner, tmp_path)

  assert result is None
  assert error is not None
  assert error["code"] == "required_context_missing"
  assert error["missing_keys"] == ["critical_factors_ref", "forecast_assumption_set_ref"]
  assert error["next_call"] == {
    "tool_name": "get_skill_artifact",
    "arguments": {"ticker": "PCTY", "skill": "critical-factors"},
  }
  assert runner.spawn_calls == []
  assert runner.background_calls == []


@pytest.mark.parametrize(
  "required_context",
  [("research_file_id",), ()],
)
def test_investment_start_route_requires_verified_turn_despite_metadata_drift(
  tmp_path: Path,
  required_context: tuple[str, ...],
) -> None:
  loader = _write_operation(
    tmp_path,
    required_context=required_context,
    operation_name="alternate-quant-method",
    investment_start_route=True,
    execution_class="node.mutate",
  )
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Run a bounded quantitative study.",
    "research_file_id": 42,
    "background": False,
  }))

  assert result is None
  assert error is not None
  assert error["code"] == "required_context_missing"
  assert runner.spawn_calls == []
  assert runner.background_calls == []

  trusted_runner = _Runner()
  trusted_result, trusted_error = asyncio.run(_handler(
    trusted_runner,
    loader=loader,
    trusted_research_file_id=42,
    mcp_client=_InvestmentMcpClient(),
  )({
    "operation": _operation(loader),
    "objective": "Run a bounded quantitative study.",
    "background": True,
  }))

  assert trusted_error is None
  assert isinstance(trusted_result, dict)
  assert trusted_result["status"] == "running"
  registration = trusted_runner.background_calls[0]
  admitted_task = registration["admitted_task"]
  assert len(admitted_task.inputs) == 1
  assert admitted_task.inputs[0].name == "research_file_id"
  assert admitted_task.inputs[0].context.content == 42
  dispatch_result, dispatch_error = asyncio.run(
    registration["handler"](registration["tool_input"])
  )
  assert dispatch_error is None
  assert dispatch_result is not None
  assert (
    trusted_runner.spawn_calls[0]["dispatcher"]._run_context.research_file_id
    == 42
  )


def test_verified_turn_remains_authoritative_for_non_investment_child(
  tmp_path: Path,
) -> None:
  loader = _write_operation(
    tmp_path,
    required_context=("research_file_id",),
    execution_class="node.implement",
  )
  runner = _Runner()
  handler = _handler(
    runner,
    loader=loader,
    trusted_research_file_id=42,
  )

  result, error = asyncio.run(handler({
    "operation": _operation(loader),
    "objective": "Review the active research file.",
    "background": False,
  }))

  assert error is None
  assert result is not None
  assert runner.spawn_calls[0]["dispatcher"]._run_context.research_file_id == 42

  conflicting_runner = _Runner()
  conflicting_handler = _handler(
    conflicting_runner,
    loader=loader,
    trusted_research_file_id=42,
  )
  mismatch_result, mismatch_error = asyncio.run(conflicting_handler({
    "operation": _operation(loader),
    "objective": "Review the active research file.",
    "research_file_id": 43,
    "background": False,
  }))
  assert mismatch_result is None
  assert mismatch_error is not None
  assert mismatch_error["code"] == "context_research_file_id_mismatch"
  assert mismatch_error["next_call"] == {
    "tool_name": "run_agent",
    "arguments": {
      "operation": _operation(loader),
      "objective": "Review the active research file.",
      "research_file_id": 42,
      "background": False,
    },
  }
  assert conflicting_runner.spawn_calls == []


@pytest.mark.parametrize(
  ("objective", "message"),
  [
    ("Review the filing evidence.", "requires one unambiguous"),
    ("Compare AAPL and MSFT.", "multiple plausible ticker"),
    ("TRACK 1: review the filing evidence.", "requires one unambiguous"),
    ("REVIEW REPORT.", "requires one unambiguous"),
    ("INVESTIGATIONONLY", "requires one unambiguous"),
  ],
)
def test_direct_handler_required_ticker_fails_before_effects(
  tmp_path: Path,
  objective: str,
  message: str,
) -> None:
  loader = _write_operation(tmp_path, required_context=("ticker",), execution_class="node.implement")
  runner = _Runner()
  resolver = _CapabilityResolver()

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    resolver=resolver,
  )({
    "operation": _operation(loader),
    "objective": objective,
    "background": True,
  }))

  assert result is None
  assert error is not None and error["code"] in {
    "invalid_input",
    "required_context_missing",
  }
  assert message in error["message"]
  assert error["missing_keys"] == ["ticker"]
  assert error["next_call"] == {
    "tool_name": "run_agent",
    "arguments": {
      "operation": _operation(loader),
      "objective": objective,
      "background": True,
    },
  }
  assert resolver.calls == []
  assert runner.spawn_calls == []
  assert runner.background_calls == []


def test_malformed_ticker_on_ticker_required_operation_names_ticker_to_add(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, required_context=("ticker",), execution_class="node.implement")
  runner = _Runner()

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    resolver=_CapabilityResolver(),
  )({
    "operation": _operation(loader),
    "objective": "Normalize the issuer's reported results.",
    "ticker": "not a ticker!!",
    "background": True,
  }))

  assert result is None
  assert error is not None and error["code"] == "invalid_input"
  assert error["missing_keys"] == ["ticker"]
  assert "ticker" not in error["next_call"]["arguments"]
  assert runner.spawn_calls == []


@pytest.mark.parametrize("ticker", ["", "not a ticker!!"])
def test_malformed_ticker_on_ticker_free_operation_repairs_by_dropping_it(
  tmp_path: Path,
  ticker: str,
) -> None:
  loader = _write_operation(tmp_path, required_context=(), execution_class="node.implement")
  runner = _Runner()
  resolver = _CapabilityResolver()

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    resolver=resolver,
  )({
    "operation": _operation(loader),
    "objective": "Summarize the macro outlook.",
    "ticker": ticker,
    "background": True,
  }))

  assert result is None
  assert error is not None and error["code"] == "invalid_input"
  assert "missing_keys" not in error
  assert error["next_call"] == {
    "tool_name": "run_agent",
    "arguments": {
      "operation": _operation(loader),
      "objective": "Summarize the macro outlook.",
      "background": True,
    },
  }
  assert runner.spawn_calls == []


@pytest.mark.parametrize("ticker", ["PCTY", "QCOM"])
def test_direct_handler_unambiguous_legacy_prose_is_promoted_once(
  tmp_path: Path,
  ticker: str,
) -> None:
  loader = _write_operation(tmp_path, required_context=("ticker",), execution_class="node.implement")
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": f"Analyze {ticker} filing evidence.",
    "background": True,
  }))

  assert error is None
  assert isinstance(result, dict)
  assert result["status"] == "running"
  admitted = runner.background_calls[0]["admitted_task"]
  assert admitted.inputs[0].context.content == ticker


def test_direct_handler_binds_typed_research_file_over_objective_prose(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()
  rebound: list[int] = []

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    fms_rebinder=lambda _handlers, research_file_id: rebound.append(
      research_file_id
    ),
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT with research_file_id=42.",
    "research_file_id": 43,
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  # The typed assertion is the only binding channel; objective prose is inert.
  assert rebound == [43]


def test_generic_delegation_uses_exact_canonical_contracts(tmp_path: Path) -> None:
  runner = _Runner()
  resolver = _CapabilityResolver()

  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path),
    resolver=resolver,
  )({
    "objective": "Find the load-bearing evidence.",
    "background": False,
    "cost_observation_threshold_usd": 2.75,
  }))

  assert error is None
  canonical = TaskResult.model_validate(result)
  call = runner.spawn_calls[0]
  assert canonical.logical_task == call["logical_task"]
  assert canonical.attempt == call["attempt"]
  assert call["result_requirement"].mode == "narrative"
  assert call["result_provenance"] == canonical.provenance
  assert call["skill_name"] == "explore"
  assert call["cost_observation_threshold_usd"] == 2.75
  assert resolver.calls[0]["capability_id"] == "node.explore"
  assert set(call["dispatcher"]._local) == {"web_search"}


def test_foreground_delegation_returns_readable_completion_handle(
  tmp_path: Path,
) -> None:
  terminal_text = "Exact child result.\n" * 1_100

  class _DurableForegroundRunner(_Runner):
    def __init__(self) -> None:
      super().__init__()
      self._gateway_session_id = "session-test"
      self._runner_id = "parent-runner"
      self._role = "writer"
      self._workspace_dir = tmp_path / "workspace"
      self._workspace_dir.mkdir()
      self.session_log = AgentSessionLog(
        tmp_path / "agent-session.jsonl"
      )
      self._agent_session_log = self.session_log

    async def _append_durable_event(
      self,
      event: dict[str, Any],
    ) -> object:
      return await self.session_log.append(dict(event))

    async def spawn_sub_agent(self, task: str, **kwargs: Any):
      call = {"task": task, **kwargs}
      self.spawn_calls.append(call)
      result = _spawn_result(call)
      reference = publish_final_narrative(
        workspace_dir=self._workspace_dir,
        sub_agent_id=kwargs["attempt"].physical_task_id,
        terminal_event_seq=1,
        text=terminal_text,
      )
      result = result.model_copy(update={
        "values": TaskResultValues(
          terminal_narrative=terminal_narrative_content_handle(reference),
        ),
      })
      return result, None

  runner = _DurableForegroundRunner()
  parent_sessions: list[GatewaySession] = []
  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path / "skills"),
    parent_sessions=parent_sessions,
  )({
    "objective": "Return the complete evidence report.",
    "background": False,
  }))

  assert error is None
  envelope = AgentCompletionEnvelope.model_validate(result)
  materialization = envelope.parent_materialization
  assert materialization is not None
  assert materialization.kind == "result_handle"
  source = materialization.source
  grant = materialization.read_grant
  page, read_error = asyncio.run(
    make_get_agent_result_content_handler(
      [runner],
      caller=lambda: current_parent_context(parent_sessions[0]),
    )({
      "content_id": source.content_id,
      "read_grant_id": grant.grant_id,
    })
  )
  assert read_error is None
  assert page is not None
  assert page["content"] == terminal_text
  assert page["end"] is True

  # A later sibling receives only the explicitly passed durable output.
  sibling, sibling_error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path / "skills"),
  )({
    "objective": "Read the earlier child's complete evidence report.",
    "background": False,
    "handles": [source.content_id, source.content_id],
  }))
  assert sibling_error is None
  assert sibling is not None
  sibling_session = runner.spawn_calls[-1]["sub_session"]
  sibling_context = current_parent_context(sibling_session)
  assert sibling_context.handle_grants == frozenset({source.content_id})
  sibling_page, sibling_read_error = asyncio.run(
    make_get_agent_result_content_handler(
      [runner], caller=lambda: current_parent_context(sibling_session),
    )({
      "content_id": source.content_id,
      "read_grant_id": grant.grant_id,
    })
  )
  assert sibling_read_error is None
  assert sibling_page is not None
  assert sibling_page["content"] == terminal_text

  denied, denied_error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path / "skills"),
  )({
    "objective": "Read an unowned result.",
    "handles": ["unowned-content"],
  }))
  assert denied is None
  assert denied_error is not None
  assert denied_error["code"] == "handle_not_granted"
  assert len(runner.spawn_calls) == 2
  events, _cursor = asyncio.run(runner.session_log.query(
    event_types={
      "task_registered",
      "task_completed",
      "agent_completion",
    },
    order="asc",
  ))
  assert [entry.event["type"] for entry in events] == [
    "task_registered", "task_completed", "agent_completion",
    "task_registered", "task_completed", "agent_completion",
  ]


def test_spawn_passes_spill_output_without_re_rooting_nested_caller(
  tmp_path: Path,
) -> None:
  from gateway_test_support.resumer_custody_test_support import (
    canonical_read_custody_task, custody_registration,
  )
  from gateway_test_support.spill_custody_test_support import publish_custody_spill
  from agent_gateway.sub_agent import seal_admitted_task_payload
  from agent_workflow_contracts import AdmittedTask
  (tmp_path / "spills").mkdir()
  root_sink = SpillSink(
    lambda: str(tmp_path / "spills"),
    capabilities=SpillCapabilities(spill_read=True),
  )
  child_sink = job_spill_sink(root_sink, "producer")
  text = "Evidence from a durable sibling spill.\n" * 100
  publication = write_spill_set(
    sink=child_sink, tool_name="web_search", tool_use_id="evidence",
    content=text, model_max_chars=100,
  )
  runner = _Runner()
  runner._agent_session_log = AgentSessionLog(tmp_path / "session.jsonl")
  runner._spill_dir_provider = root_sink
  result, error = asyncio.run(_handler(
    runner, loader=SkillLoader(tmp_path / "skills"),
  )({
    "objective": "Read the spilled evidence.", "background": False,
    "handles": [publication.spill_ref],
  }))
  assert error is None
  assert result is not None
  child_session = runner.spawn_calls[-1]["sub_session"]
  child_context = current_parent_context(child_session)
  assert child_context.handle_grants == frozenset({publication.spill_ref})
  child_context = replace(
    child_context,
    children=frozenset({runner.spawn_calls[-1]["skill_name"]}),
  )
  assert child_context.job_id is not None
  reader_sink = job_spill_sink(root_sink, child_context.job_id)
  page, read_error = asyncio.run(make_tool_result_read_handler(
    lambda: reader_sink, caller=lambda: child_context,
    journal=lambda: runner._agent_session_log,
  )({"spill_ref": publication.spill_ref}))
  assert read_error is None
  assert page is not None
  assert page["content"] == text

  unpassed = write_spill_set(
    sink=child_sink, tool_name="web_search", tool_use_id="unpassed",
    content="Unpassed sibling evidence.", model_max_chars=100,
  )
  sessions: list[GatewaySession] = []
  nested_handler = _handler(
    runner, loader=SkillLoader(tmp_path / "skills"), parent_sessions=sessions,
  )
  capture_job_context(sessions[0], child_context)
  runner._spill_dir_provider = reader_sink
  denied, denied_error = asyncio.run(nested_handler({
    "objective": "Read unpassed sibling evidence.",
    "handles": [unpassed.spill_ref],
  }))
  assert denied is None
  assert denied_error is not None
  assert denied_error["code"] == "handle_not_granted"
  assert current_parent_context(sessions[0]) is child_context

  async def _cross_root_case() -> None:
    old_root = tmp_path / "old-root"
    new_root = tmp_path / "new-root"
    old_root.mkdir()
    new_root.mkdir()
    old_sink = SpillSink(
      lambda: str(old_root),
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    )
    current_sink = SpillSink(
      lambda: str(new_root),
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    )
    journal = AgentSessionLog(tmp_path / "cross-root.jsonl")
    assert child_context.tenant is not None and child_context.job_id is not None
    source = canonical_read_custody_task(
      "custody-source", parent_job_id="old-owner",
      tenant_storage_id=child_context.tenant.storage_id,
    )
    payload = source.model_dump(mode="json", exclude={"admitted_task_digest"})
    payload["execution_snapshot"]["admitted_context"]["tenant"] = child_context.tenant.model_dump(mode="json")
    seal_admitted_task_payload(payload)
    source = AdmittedTask.model_validate(payload)
    successor = canonical_read_custody_task(
      "custody-successor", predecessor=source,
      resumer_job_id=child_context.job_id,
    )
    for admitted in (source, successor):
      await journal.append(custody_registration(admitted))
    source_ref, _durable, _live, _provider = await publish_custody_spill(
      job_spill_sink(old_sink, "custody-source"),
      replace(child_context, job_id="custody-source"), journal,
      tool_use_id="source-call", payload={"payload": "source evidence" * 6000},
    )
    successor_ref, _durable, _live, _provider = await publish_custody_spill(
      job_spill_sink(current_sink, "custody-successor"),
      replace(child_context, job_id="custody-successor"), journal,
      tool_use_id="successor-call", payload={"payload": "successor evidence" * 6000},
    )
    unrelated_ref, _durable, _live, _provider = await publish_custody_spill(
      job_spill_sink(old_sink, "old-sibling"),
      replace(child_context, job_id="old-sibling"), journal,
      tool_use_id="sibling-call", payload={"payload": "sibling evidence" * 6000},
    )
    runner._agent_session_log = journal
    runner._spill_dir_provider = job_spill_sink(current_sink, child_context.job_id)
    before_calls = len(runner.spawn_calls)
    delegated, delegated_error = await nested_handler({
      "objective": "Read the continued source and successor.",
      "background": False, "handles": [source_ref, successor_ref],
    })
    assert delegated_error is None and delegated is not None
    delegated_context = current_parent_context(runner.spawn_calls[-1]["sub_session"])
    assert {source_ref, successor_ref} <= delegated_context.handle_grants
    for spill_ref in (source_ref, successor_ref):
      page, page_error = await make_tool_result_read_handler(
        lambda: None, caller=lambda: delegated_context, journal=lambda: journal,
      )({"spill_ref": spill_ref})
      assert page_error is None and page is not None
      assert "evidence" in page["content"]
    denied, denied_error = await nested_handler({
      "objective": "Read unrelated old-root evidence.", "handles": [unrelated_ref],
    })
    assert denied is None and denied_error is not None
    assert denied_error["code"] == "handle_not_granted"
    assert len(runner.spawn_calls) == before_calls + 1
    assert current_parent_context(sessions[0]) is child_context

  asyncio.run(_cross_root_case())


def test_foreground_delegation_publishes_soft_budget_abort_without_narrative(
  tmp_path: Path,
) -> None:
  class _BudgetExhaustedRunner(_Runner):
    def __init__(self) -> None:
      super().__init__()
      self._gateway_session_id = "session-test"
      self._runner_id = "parent-runner"
      self._role = "writer"
      self._workspace_dir = tmp_path
      self.session_log = AgentSessionLog(tmp_path / "session.jsonl")
      self._agent_session_log = self.session_log

    async def _append_durable_event(self, event: dict[str, Any]) -> object:
      return await self.session_log.append(event)

    async def spawn_sub_agent(self, task: str, **kwargs: Any):
      # Buyer 3.3: the child stopped after tool use, before any end_turn.
      result = task_result_from_execution(
        [
          {
            "type": "assistant_message",
            "stop_reason": "tool_use",
            "content_blocks": [{"type": "text", "text": "Pulling peer filings."}],
          },
          {"type": "tool_call_start", "tool_name": "fetch_financials"},
          {"type": "interrupted", "reason": "soft_budget_abort"},
        ],
        logical_task=kwargs["logical_task"],
        attempt=kwargs["attempt"],
        requirement=kwargs["result_requirement"],
        provenance=kwargs["result_provenance"],
        final_narrative=None,
        stalled=False,
      )
      return result, None

  runner = _BudgetExhaustedRunner()
  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path / "skills"),
  )({
    "objective": "Sweep MSCI peer disclosures.",
    "background": False,
  }))

  assert error is None
  envelope = AgentCompletionEnvelope.model_validate(result)
  assert envelope.settlement_projection.execution_status == "failed"
  assert envelope.settlement_projection.terminal_reason == "soft_budget_abort"
  assert envelope.parent_materialization is None
  assert envelope.child_evidence is not None
  assert envelope.child_evidence.evidence_tools == ("fetch_financials",)
  events, _ = asyncio.run(runner.session_log.query(
    event_types={"task_completed", "agent_completion"},
    order="asc",
  ))
  assert [entry.event["type"] for entry in events] == [
    "task_completed", "agent_completion",
  ]
  persisted = AgentCompletionEnvelope.model_validate(events[-1].event["envelope"])
  assert persisted == envelope
  assert "Pulling peer filings." not in persisted.model_dump_json()


class _UnpublishableForegroundRunner(_Runner):
  def __init__(self, tmp_path: Path) -> None:
    super().__init__()
    self._gateway_session_id = "session-test"
    self._runner_id = "parent-runner"
    self._role = "writer"
    self._workspace_dir = tmp_path

  async def _append_durable_event(self, event: dict[str, Any]) -> object:
    if event.get("type") in {
      "task_registered",
      "task_completed",
      "agent_completion",
    }:
      raise ParentResultMaterializationError("forced publication failure")
    self.durable_events.append(dict(event))
    return object()


@pytest.mark.parametrize(
  ("interrupt_events", "external_signals", "expected_status", "expected_reason"),
  [
    (
      [{"type": "interrupted", "reason": "soft_budget_abort"}],
      (),
      "failed",
      "soft_budget_abort",
    ),
    ([], ("cancelled",), "cancelled", "cancelled"),
  ],
)
def test_foreground_interrupt_publication_failure_names_terminal_reason(
  tmp_path: Path,
  interrupt_events: list[dict[str, Any]],
  external_signals: tuple[str, ...],
  expected_status: str,
  expected_reason: str,
) -> None:
  class _InterruptedRunner(_UnpublishableForegroundRunner):
    async def spawn_sub_agent(self, task: str, **kwargs: Any):
      result = task_result_from_execution(
        list(interrupt_events),
        logical_task=kwargs["logical_task"],
        attempt=kwargs["attempt"],
        requirement=kwargs["result_requirement"],
        provenance=kwargs["result_provenance"],
        final_narrative=None,
        stalled=False,
        external_terminal_signals=external_signals,
      )
      return result, None

  result, error = asyncio.run(_handler(
    _InterruptedRunner(tmp_path),
    loader=SkillLoader(tmp_path / "skills"),
  )({
    "objective": "Sweep MSCI peer disclosures.",
    "background": False,
  }))

  assert result is None
  assert error is not None
  assert error == {
    "code": "agent_completion_materialization_failed",
    "message": (
      f"Foreground agent {expected_status} ({expected_reason}) but its exact "
      "parent-readable result could not be published: "
      "ParentResultMaterializationError"
    ),
  }
  assert "completed" not in error["message"]


def test_foreground_completed_publication_failure_keeps_completed_wording(
  tmp_path: Path,
) -> None:
  projection_value: dict[str, Any] = {"summary": "Canonical child summary."}
  projection_contract = ContractRef(
    namespace="agent-gateway",
    name="terminal-tool-result",
    version="1.0",
    digest=sha256_digest({"contract": "terminal-tool-result"}),
  )

  class _CompletedRunner(_UnpublishableForegroundRunner):
    async def spawn_sub_agent(self, task: str, **kwargs: Any):
      call = {"task": task, **kwargs}
      self.spawn_calls.append(call)
      result = _spawn_result(call).model_copy(update={
        "values": TaskResultValues(
          terminal_narrative=None,
          projection=CanonicalProjection(
            contract=projection_contract,
            content=_content(projection_value, projection_contract),
            inline_view=projection_value,
          ),
        ),
      })
      return result, None

  result, error = asyncio.run(_handler(
    _CompletedRunner(tmp_path),
    loader=SkillLoader(tmp_path / "skills"),
  )({
    "objective": "Return the complete evidence report.",
    "background": False,
  }))

  assert result is None
  assert error == {
    "code": "agent_completion_materialization_failed",
    "message": (
      "Foreground agent completed but its exact parent-readable result "
      "could not be published: ParentResultMaterializationError"
    ),
  }


def test_foreground_delegation_materializes_projection_only_result(
  tmp_path: Path,
) -> None:
  projection_value = {
    "tool_name": "fms_propose_demo",
    "result": {"status": "staged", "proposal_id": "proposal-1"},
  }
  projection_contract = ContractRef(
    namespace="agent-gateway",
    name="terminal-tool-result",
    version="1.0",
    digest=sha256_digest({"contract": "terminal-tool-result"}),
  )

  class _DurableForegroundRunner(_Runner):
    def __init__(self) -> None:
      super().__init__()
      self._gateway_session_id = "session-test"
      self._runner_id = "parent-runner"
      self._role = "writer"
      self._workspace_dir = tmp_path / "workspace"
      self._workspace_dir.mkdir()
      self.session_log = AgentSessionLog(
        tmp_path / "agent-session.jsonl"
      )
      self._agent_session_log = self.session_log

    async def _append_durable_event(
      self,
      event: dict[str, Any],
    ) -> object:
      return await self.session_log.append(dict(event))

    async def spawn_sub_agent(self, task: str, **kwargs: Any):
      call = {"task": task, **kwargs}
      self.spawn_calls.append(call)
      result = _spawn_result(call).model_copy(update={
        "values": TaskResultValues(
          terminal_narrative=None,
          projection=CanonicalProjection(
            contract=projection_contract,
            content=_content(projection_value, projection_contract),
            inline_view=projection_value,
          ),
        ),
      })
      return result, None

  result, error = asyncio.run(_handler(
    _DurableForegroundRunner(),
    loader=SkillLoader(tmp_path / "skills"),
  )({
    "objective": "Return the exact structured result.",
    "background": False,
  }))

  assert error is None
  envelope = AgentCompletionEnvelope.model_validate(result)
  materialization = envelope.parent_materialization
  assert materialization is not None
  assert materialization.kind == "projection_inline"
  assert materialization.value == projection_value


def test_registered_operation_requires_full_ref_and_emits_lifecycle(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  call = runner.spawn_calls[0]
  assert call["logical_task"].operation.name == "filing-review"
  assert call["result_requirement"].mode == "narrative"
  assert [event["type"] for event in runner.durable_events] == [
    "skill_run_started",
    "skill_result_captured",
  ]


def test_registered_operation_settles_lifecycle_when_runner_result_is_invalid(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # A runner that returns a non-TaskResult is a runner defect: it still
  # raises to the dispatcher, and the started run still settles as failed.
  async def _invalid_spawn(task: str, **kwargs: Any):
    runner.spawn_calls.append({"task": task, **kwargs})
    return {"response": "ok"}, None

  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()
  monkeypatch.setattr(runner, "spawn_sub_agent", _invalid_spawn)

  with pytest.raises(ValidationError):
    asyncio.run(_handler(runner, loader=loader)({
      "operation": _operation(loader),
      "objective": "Review MSFT filing evidence.",
      "background": False,
    }))

  assert [event["type"] for event in runner.durable_events] == [
    "skill_run_started",
    "skill_result_captured",
  ]
  captured = runner.durable_events[-1]
  assert captured["exit_code"] == 1
  assert captured["status"] == "error"
  assert captured["error"] == (
    "Sub-agent runner returned an invalid result: Field required"
  )


def test_registered_operation_captures_recoverable_fms_error_without_internal_error(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()
  spawn_sub_agent = runner.spawn_sub_agent

  async def _spawn_with_fms_error(task: str, **kwargs: Any):
    result, error = await spawn_sub_agent(task, **kwargs)
    kwargs["dispatcher"]._event_log.append({
      "type": "tool_call_complete",
      "tool_name": "fms_persist_test_artifact",
      "result": {
        "status": "error",
        "subcommand": "persist_test_artifact",
        "mutation_mode": "model_writer",
        "error": {
          "type": "INVALID_JUDGMENT",
          "message": "active research file is required",
          "recoverable": True,
        },
      },
    })
    return result, error

  runner.spawn_sub_agent = _spawn_with_fms_error
  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  captured = runner.durable_events[-1]
  assert captured["type"] == "skill_result_captured"
  assert captured["exit_code"] == 1
  assert captured["outcome"] == "error"
  # The child's status is its own disposition; the door's status stays on the
  # door's row.
  assert captured["status"] == "blocked"
  assert captured["fms_results"][0]["status"] == "error"
  assert captured["error"] == "active research file is required"


def test_registered_operation_activates_declared_mcp_before_admission(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()
  activated: list[str] = []

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    operation_mcp_activator=lambda profile: activated.append(
      profile.name
    ),
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "ticker": "MSFT",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  assert activated == ["filing-review"]
  assert len(runner.spawn_calls) == 1


def test_registered_foreground_operation_inherits_parent_approval_lifecycle(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()
  approval_store = object()
  approval_policy = object()

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    approval_store=approval_store,
    approval_policy=approval_policy,
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  dispatcher = runner.spawn_calls[0]["dispatcher"]
  # The child delegates to the parent's admitted route — the same durable
  # ledger object the parent session carries — rather than re-deriving one
  # from handles. The headless flag is decided independently, so a background
  # child pairs should_avoid_permission_prompts with this same live ledger.
  parent_route = dispatcher._session.approval_route
  assert isinstance(parent_route, DurableLocalApprovalRoute)
  assert isinstance(dispatcher._approval_route, InProcessDelegatedApprovalRoute)
  assert route_ledger(dispatcher._approval_route) is parent_route
  assert parent_route.session is dispatcher._session
  assert dispatcher._approval_store is approval_store
  assert dispatcher._approval_policy is approval_policy
  assert dispatcher._session is not None
  assert dispatcher._should_avoid_permission_prompts is False
  started = next(
    event for event in runner.durable_events
    if event.get("type") == "skill_run_started"
  )
  assert dispatcher._run_context.run_id == started["skill_run_id"]
  assert runner.spawn_calls[0]["skill_run_id"] == started["skill_run_id"]
  assert dispatcher._run_context.skill == "filing-review"
  assert dispatcher._run_context.profile == "filing-review"


def test_registered_operation_never_rebinds_fms_from_objective_prose(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()
  rebound: list[tuple[int, set[str]]] = []

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    fms_rebinder=lambda handlers, research_file_id: rebound.append((
      research_file_id,
      set(handlers),
    )),
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT with research_file_id=42.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  # No typed assertion and no verified turn: the child runs unbound.
  assert rebound == []


def test_registered_operation_binds_explicit_research_file_to_child_run(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()
  rebound: list[int] = []

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    fms_rebinder=lambda _handlers, research_file_id: rebound.append(
      research_file_id
    ),
  )({
    "operation": _operation(loader),
    "objective": "Review the exact PCTY research file.",
    "ticker": "PCTY",
    "research_file_id": 42,
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  assert rebound == [42]
  dispatcher = runner.spawn_calls[0]["dispatcher"]
  assert dispatcher._run_context.research_file_id == 42


def test_registered_operation_mcp_activation_error_prevents_dispatch(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()
  activation_error = {
    "code": "mcp_tool_unavailable",
    "message": "Declared MCP tool is unavailable.",
  }

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    operation_mcp_activator=lambda _profile: activation_error,
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert result is None
  assert error == activation_error
  assert runner.spawn_calls == []


def test_generic_delegation_does_not_activate_named_operation_mcp(
  tmp_path: Path,
) -> None:
  runner = _Runner()
  activated: list[str] = []

  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path),
    operation_mcp_activator=lambda profile: activated.append(
      profile.name
    ),
  )({
    "objective": "Find the load-bearing evidence.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  assert activated == []


def test_registered_operation_fails_closed_without_durable_log(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()
  runner._agent_session_log = None

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert result is None
  assert error is not None
  assert error["code"] == "durable_session_log_required"
  assert runner.spawn_calls == []


def test_bare_operation_name_is_not_an_authority_reference(tmp_path: Path) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": "filing-review",
    "objective": "Review the filing.",
  }))

  assert result is None
  assert error is not None and error["code"] == "invalid_operation"
  assert "full AgentOperationRef" in error["message"]


def test_background_registration_persists_exact_admitted_task(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path, execution_class="node.implement")
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review the filing in the background.",
    "background": True,
    "cost_observation_threshold_usd": 3.25,
  }))

  assert error is None
  assert isinstance(result, dict)
  assert result["status"] == "running"
  registration = runner.background_calls[0]
  admitted = registration["admitted_task"]
  assert isinstance(admitted, AdmittedTask)
  assert admitted.execution_snapshot is not None
  assert result["granted_tools"] == sorted(
    entry.tool_id for entry in admitted.tool_grant.tools
  )
  assert admitted.execution_disposition.kind == "execute"
  assert admitted.operation.operation.name == "filing-review"
  assert admitted.execution_snapshot.cost_observation_threshold_usd == 3.25
  assert admitted.execution_snapshot.max_turns is None
  assert admitted.execution_snapshot.max_budget_usd is None
  assert registration["task_id_override"] == admitted.attempt.physical_task_id
  assert registration["tool_input"]["operation"] == _operation(loader)
  assert registration["tool_input"]["result_requirement"] == (
    admitted.result_requirement.model_dump(mode="json")
  )
  assert registration["tool_input"]["cost_observation_threshold_usd"] == 3.25
  assert "max_budget_usd" not in registration["tool_input"]
  assert ADMITTED_TASK_METADATA_KEY not in registration["tool_input"]
  assert "agent" not in registration["tool_input"]
  assert "task" not in registration["tool_input"]
  assert "child_tool_scope_receipt" not in registration["tool_input"]


def test_unnamed_delegation_passes_no_turn_or_cost_ceiling(tmp_path: Path) -> None:
  runner = _Runner()

  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path),
  )({
    "objective": "Explore the admitted evidence.",
    "background": False,
  }))

  assert error is None
  assert result is not None
  assert "max_turns" not in runner.spawn_calls[0]
  assert "max_budget_usd" not in runner.spawn_calls[0]


@pytest.mark.parametrize("objective", [None, "", 0])
def test_objective_validation_is_canonical(
  tmp_path: Path,
  objective: object,
) -> None:
  result, error = asyncio.run(_handler(
    _Runner(),
    loader=SkillLoader(tmp_path),
  )({"objective": objective}))

  assert result is None
  assert error == {
    "code": "invalid_input",
    "message": "objective is required",
    "next_call": None,
  }


def test_missing_runner_fails_before_admission(tmp_path: Path) -> None:
  result, error = asyncio.run(_handler(
    None,
    loader=SkillLoader(tmp_path),
  )({"objective": "Explore."}))

  assert result is None
  assert error == {
    "code": "internal_error",
    "message": "Sub-agent runner not initialized",
  }


def test_background_result_handler_proxies_exact_task_id() -> None:
  runner = _Runner()

  result, error = asyncio.run(
    make_get_background_result_handler([runner])({"task_id": "bg_7"})
  )

  assert error is None
  assert result == {"task_id": "bg_7", "status": "completed"}
  assert runner.background_result_calls == [{"task_id": "bg_7"}]


def test_background_and_resume_tool_schemas_are_explicit() -> None:
  background = make_get_background_result_tool_def()["input_schema"]
  resume = make_resume_tool_def()["input_schema"]

  assert background["required"] == ["task_id"]
  assert resume["required"] == ["task_id"]
  assert "additional_context" in resume["properties"]


def test_skill_lifecycle_emitter_projects_canonical_task_result_once(
  tmp_path: Path,
) -> None:
  runner = _Runner()
  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path),
  )({"objective": "Collect evidence.", "background": False}))
  assert error is None
  durable: list[dict[str, Any]] = []
  projected: list[dict[str, Any]] = []
  event_log = EventLog()

  async def append(event: dict[str, Any]) -> object:
    durable.append(dict(event))
    return object()

  async def confirm(event: dict[str, Any]) -> dict[str, Any] | None:
    return dict(event) if event in durable else None

  emitter = SkillRunEventEmitter(
    skill_run_id="skill-run-1",
    profile=SimpleNamespace(name="filing-review"),
    semantic_scope="ticker",
    context_ticker="msft",
    portfolio_id=None,
    event_log_getter=lambda: event_log,
    tool_ctx=SimpleNamespace(emit=projected.append),
    durable_appender=append,
    durable_confirmer=confirm,
    time_fn=lambda: 1.0,
  )

  assert asyncio.run(emitter.emit_started()) is True
  assert asyncio.run(emitter.emit_started()) is True
  assert asyncio.run(emitter.emit_result_captured(result, None)) is True
  assert [event["type"] for event in durable] == [
    "skill_run_started",
    "skill_result_captured",
  ]
  assert [event["type"] for event in projected] == [
    "skill_run_started",
    "skill_result_captured",
  ]
  assert projected[-1]["status"] == "not_assessed"
  assert len(event_log.entries) == 2


def test_skill_lifecycle_result_projects_after_child_stream_closes() -> None:
  durable: list[dict[str, Any]] = []
  parent_log = EventLog()
  child_log = EventLog()

  async def append(event: dict[str, Any]) -> object:
    durable.append(dict(event))
    return object()

  async def confirm(event: dict[str, Any]) -> dict[str, Any] | None:
    return dict(event) if event in durable else None

  emitter = SkillRunEventEmitter(
    skill_run_id="skill-run-terminal-child",
    profile=SimpleNamespace(name="valuation-policy-precompile"),
    semantic_scope="ticker",
    context_ticker="PCTY",
    portfolio_id=None,
    event_log_getter=lambda: child_log,
    tool_ctx=ToolExecutionContext(
      tool_call_id="tool-1",
      tool_name="run_agent",
      event_log=parent_log,
    ),
    durable_appender=append,
    durable_confirmer=confirm,
    time_fn=lambda: 1.0,
  )

  assert asyncio.run(emitter.emit_started()) is True
  child_log.append({"type": "stream_complete"})
  assert asyncio.run(
    emitter.emit_result_captured(
      _settled_task_result("terminal-child").model_dump(mode="json"),
      None,
    )
  ) is True

  assert [entry.event["type"] for entry in parent_log.entries] == [
    "skill_run_started",
    "skill_result_captured",
  ]
  assert [entry.event["type"] for entry in child_log.entries] == [
    "stream_complete",
  ]


def test_skill_lifecycle_result_settles_after_parent_stream_closes() -> None:
  durable: list[dict[str, Any]] = []
  parent_log = EventLog()

  async def append(event: dict[str, Any]) -> object:
    durable.append(dict(event))
    return object()

  async def confirm(event: dict[str, Any]) -> dict[str, Any] | None:
    return dict(event) if event in durable else None

  emitter = SkillRunEventEmitter(
    skill_run_id="skill-run-closed-parent",
    profile=SimpleNamespace(name="valuation-policy-precompile"),
    semantic_scope="ticker",
    context_ticker="PCTY",
    portfolio_id=None,
    event_log_getter=lambda: parent_log,
    tool_ctx=ToolExecutionContext(
      tool_call_id="tool-1",
      tool_name="run_agent",
      event_log=parent_log,
    ),
    durable_appender=append,
    durable_confirmer=confirm,
    time_fn=lambda: 1.0,
  )

  assert asyncio.run(emitter.emit_started()) is True
  # The parent run reached its own terminal event before the background
  # child's result was settled: the live stream no longer takes events.
  parent_log.append({"type": "error", "error": "run_failed"})

  assert asyncio.run(
    emitter.emit_result_captured(
      _settled_task_result("closed-parent").model_dump(mode="json"),
      None,
    )
  ) is True

  assert [event["type"] for event in durable] == [
    "skill_run_started",
    "skill_result_captured",
  ]
  assert [entry.event["type"] for entry in parent_log.entries] == [
    "skill_run_started",
    "error",
  ]


def test_skill_lifecycle_emitter_fails_closed_without_confirmation() -> None:
  async def append(_event: dict[str, Any]) -> None:
    return None

  async def never_confirm(
    _event: dict[str, Any],
  ) -> dict[str, Any] | None:
    return None

  emitter = SkillRunEventEmitter(
    skill_run_id="skill-run-unconfirmed",
    profile=SimpleNamespace(name="filing-review"),
    semantic_scope=None,
    context_ticker=None,
    portfolio_id=None,
    event_log_getter=EventLog,
    tool_ctx=None,
    durable_appender=append,
    durable_confirmer=never_confirm,
  )

  with pytest.raises(DurableSkillEventPersistenceError):
    asyncio.run(emitter.emit_started())


def test_child_run_context_inherits_the_calling_run_profile() -> None:
  # The authority owner reads RunContext.profile as the profile partition; a
  # child runs inside its parent's, so the skill name rides `skill` only.
  parent_session = GatewaySession(
    session_id="parent-session",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="user-1",
    role="invite",
    channel="discord",
  )
  run_context = _child_run_context(
    parent_session=parent_session,
    tool_ctx=ToolExecutionContext(
      tool_call_id="tc-run-agent",
      tool_name="run_agent",
      event_log=None,
      profile="community",
    ),
    skill_run_id="skill-run-1",
    skill_name="risk-review",
    admitted_skill_execution_limits=None,
    research_file_id=None,
    user_id=None,
    session_id="parent-session",
    approval_policy=None,
  )

  assert run_context.profile == "community"
  assert run_context.skill == "risk-review"


class _KernelChildProvider(ModelProvider):
  name = "openai"

  def __init__(self, termination: str, on_cells_complete: Callable[[], None]) -> None:
    self.termination = termination
    self.turn = 0
    self.paused = asyncio.Event()
    self.results: list[dict[str, Any]] = []
    self.available_tools: list[frozenset[str]] = []
    self.status_task_id: str | None = None
    self.on_cells_complete = on_cells_complete

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    pass

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return kwargs

  async def stream(self, client: Any, params: dict[str, Any]):
    self.available_tools.append(frozenset(
      definition["name"] for definition in params["tools"]
    ))
    for message in params["messages"]:
      content = message.get("content")
      if not isinstance(content, list):
        continue
      for block in content:
        if (
          isinstance(block, dict)
          and block.get("type") == "tool_result"
          and str(block.get("tool_use_id", "")).startswith("kernel-cell-")
        ):
          result = json.loads(block["content"])
          if result not in self.results:
            self.results.append(result)
    self.turn += 1
    if self.turn <= 2:
      code = (
        "import os\nprint('parent_token' in globals())\n"
        "child_token = 'child-local'\nprint(os.getpid())"
        if self.turn == 1
        else "print(child_token)\nprint(os.getpid())"
      )
      call_id = f"kernel-cell-{self.turn}"
      tool_input = {"code": code, "host": "kernel"}
      tool_name = "code_execute"
      if self.status_task_id is not None:
        tool_name = "code_execute_status"
        tool_input = {"task_id": self.status_task_id}
      yield StreamEvent(
        type="tool_use_start", tool_id=call_id, tool_name=tool_name,
      )
      yield StreamEvent(
        type="tool_use_end",
        tool_id=call_id,
        tool_name=tool_name,
        tool_input=tool_input,
        raw_block={
          "type": "tool_use", "id": call_id,
          "name": tool_name, "input": tool_input,
        },
      )
      yield StreamEvent(type="message_end", stop_reason="tool_use")
      return
    self.on_cells_complete()
    if self.termination == "error":
      raise RuntimeError("child provider failed after persistent cells")
    if self.termination == "cancel":
      self.paused.set()
      await asyncio.Event().wait()
    text = "Both child-local cells completed."
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def _kernel_child_resolver(provider: ModelProvider) -> CapabilityExecutionResolver:
  base = stub_capability_execution_resolver(
    default_provider="openai",
    default_model="gpt-5.6-sol",
  )
  return CapabilityExecutionResolver(
    registry=base.registry,
    selection_policy=base.selection_policy,
    auth_context=base.auth_context,
    credential_materializer=base.credential_materializer,
    adapter_resolver=lambda _adapter: provider,
    trusted_channel=base.trusted_channel,
    authenticated_run_overrides=base.authenticated_run_overrides,
    executable_capability_ids=base.executable_capability_ids,
  )


def _write_kernel_child_operation(path: Path) -> SkillLoader:
  path.mkdir(parents=True)
  (path / "kernel-child.md").write_text(
    """---
name: kernel-child
execution_class: node.implement
version: '1.0'
agent_callable: true
agent_description: Compute in the child's private interpreter.
mutation_mode: read_only
allowed_tools:
  - code_execute
mcp_tools: {}
semantic_metadata:
  required_context: []
  tool_refs:
    - kind: local
      tool_id: code_execute
  capability_requirements:
    - name: computation.execute/v1
      required: true
      binding_modes: [live_tool]
---
Compute two cells and return their results.
""",
    encoding="utf-8",
  )
  return SkillLoader(path)


@pytest.mark.parametrize(
  ("hop", "termination", "authority_variant"),
  [
    (hop, termination, "same")
    for hop in ("spawn", "resume")
    for termination in (
      "success", "error", "cancel", "binding_error",
      "unavailable", "unavailable_status", "custom",
    )
  ] + [
    ("resume", "success", "different"),
    ("resume", "success", "absent"),
    ("resume", "success", "legacy"),
    ("resume", "success", "historical"),
  ],
)
def test_child_kernel_is_isolated_and_disposed_through_real_run_lifetime(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  hop: str,
  termination: str,
  authority_variant: str,
) -> None:
  materialized_binds: list[CapabilityBind] = []
  materialize_bind = CapabilityExecutionResolver.materialize_bind

  def record_materialization(
    resolver: CapabilityExecutionResolver,
    bind: CapabilityBind,
  ) -> BoundCapabilityExecution:
    materialized_binds.append(bind)
    return materialize_bind(resolver, bind)

  monkeypatch.setattr(
    CapabilityExecutionResolver, "materialize_bind", record_materialization,
  )

  async def run() -> None:
    parent_session = GatewaySession(
      session_id="parent-kernel-session",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="actor-test",
      owner_user_id="actor-test",
      user_email="actor@example.com",
      role="owner",
      tenant_id="tenant-test",
      channel="cli",
    )
    bind_session_approval_route(parent_session, None, None)
    config = CodeExecutionConfig(
      register_docker=False,
      register_subprocess=False,
      register_kernel=True,
      work_dir_root=str(tmp_path),
    )
    parent_bundle = build_code_execution(parent_session, config=config)
    mcp_client = McpClientManager(config_path=None)
    parent_dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=parent_bundle.handlers,
      event_log=EventLog(),
      session_id=parent_session.session_id,
      role="owner",
      approval_key_qualifier=parent_bundle.approval_qualifier,
      get_tool_definitions=lambda: parent_bundle.tool_definitions,
    )
    parent_bundle.bind_dispatcher(
      parent_dispatcher,
      tool_definitions=parent_dispatcher.get_tool_definitions,
    )
    child_sessions: list[GatewaySession] = []
    child_readers: list[Any] = []
    child_processes: list[Any] = []
    child_workdirs: list[Path] = []

    def record_child_process() -> None:
      if termination in {"unavailable", "unavailable_status", "custom"}:
        return
      child_runtime = child_sessions[-1].code_execution_kernels["kernel"].runner
      assert child_runtime is not None and child_runtime.process is not None
      child_processes.append(child_runtime.process)

    provider = _KernelChildProvider(termination, record_child_process)
    resolver = _kernel_child_resolver(provider)
    session_log = AgentSessionLog(path=tmp_path / "sessions" / "kernel.jsonl")
    runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=parent_dispatcher,
      session_id=parent_session.session_id,
      capability_execution=resolver.resolve("session.driver"),
      get_tool_definitions=parent_dispatcher.get_tool_definitions,
      agent_session_log=session_log,
      workspace_dir=str(tmp_path),
      user_id=parent_session.user_id,
      billing_mode="byok",
      rate_table_version="unknown",
    )
    runner._runner_id = "kernel-parent"
    loader = _write_kernel_child_operation(tmp_path / "skills")
    if termination == "unavailable_status":
      skill_path = tmp_path / "skills" / "kernel-child.md"
      skill_path.write_text(skill_path.read_text().replace(
        "  - code_execute\n", "  - code_execute\n  - code_execute_status\n",
      ).replace(
        "      tool_id: code_execute\n",
        "      tool_id: code_execute\n"
        "    - kind: local\n      tool_id: code_execute_status\n",
      ))
    operation = next(
      item.snapshot.operation.model_dump(mode="json")
      for item in loader.list_callable_operations()
      if item.snapshot.operation.name == "kernel-child"
    )

    def declared_definitions(requested: frozenset[str]) -> list[dict[str, Any]]:
      return [
        definition for definition in parent_bundle.tool_definitions
        if definition["name"] in requested
      ]

    def build_child(session: GatewaySession):
      child_sessions.append(session)
      if termination in {"unavailable", "unavailable_status"}:
        return None
      bundle = build_code_execution(session, config=config)
      child_readers.append(bundle.frame_reader)
      child_workdirs.append(Path(bundle.ensure_work_dir()))
      return bundle

    def bind_child(bundle: Any, dispatcher: ToolDispatcher) -> None:
      # The fresh bundle also has a status tool. The exact operation grants
      # only execution; building the child's bundle must not widen that grant.
      assert {item["name"] for item in dispatcher.get_tool_definitions()} == {
        "code_execute",
      }
      assert "code_execute_status" not in dispatcher._local
      bundle.bind_dispatcher(
        dispatcher,
        tool_definitions=dispatcher.get_tool_definitions,
      )
      if termination == "binding_error":
        raise RuntimeError("child dispatcher binding failed")


    child_kwargs = {
      "parent_session": parent_session,
      "mcp_client": mcp_client,
      "local_tool_handlers": parent_bundle.handlers,
      "capability_execution_resolver": resolver,
      "code_execution_factory": build_child,
      "code_execution_binder": bind_child,
    }
    if termination == "custom":
      custom_namespace: dict[str, Any] = {}

      async def caller_execute(tool_input: dict[str, Any], **_kwargs: Any):
        stdout = StringIO()
        with redirect_stdout(stdout):
          exec(tool_input["code"], custom_namespace)
        return {"stdout": stdout.getvalue(), "return_code": 0}, None

      child_kwargs["local_tool_handlers"] = {"code_execute": caller_execute}
      child_kwargs["code_execution_factory"] = None
      child_kwargs["code_execution_binder"] = None
    try:
      initial, error = await parent_dispatcher.dispatch(
        "parent-seed",
        "code_execute",
        {"code": "parent_token = 'parent-local'\nimport os\nprint(os.getpid())", "host": "kernel"},
      )
      assert error is None and initial is not None
      parent_pid = int(initial["stdout"].strip())
      parent_runtime = parent_session.code_execution_kernels["kernel"].runner
      assert parent_runtime is not None and parent_runtime.process is not None
      parent_process = parent_runtime.process
      if termination == "unavailable_status":
        background, background_error = await parent_dispatcher.dispatch(
          "parent-background",
          "code_execute",
          {"code": "print(parent_token)", "host": "kernel", "background": True},
        )
        assert background_error is None and background is not None
        provider.status_task_id = background["task_id"]
        await asyncio.sleep(0.05)
      if hop == "spawn":
        if termination == "unavailable":
          # This is the parent's legitimate placement qualifier; it must not
          # make an inherited interpreter available when child construction fails.
          child_kwargs["approval_key_qualifier"] = parent_bundle.approval_qualifier
        handler = make_run_agent_handler(
          [runner],
          skill_loader=loader,
          declared_local_tool_definitions=declared_definitions,
          **child_kwargs,
        )
        child_task = asyncio.create_task(handler({
          "operation": operation,
          "objective": "Run both private cells.",
          "background": False,
        }))
      else:
        # Persist a canonical interrupted attempt, then drive the actual
        # resume handler, background registration and AgentRunner child loop.
        seed = _Runner()
        seed._get_tool_definitions = lambda: parent_bundle.tool_definitions
        seed_handler = make_run_agent_handler(
          [seed],
          parent_session=parent_session,
          skill_loader=loader,
          mcp_client=mcp_client,
          local_tool_handlers=parent_bundle.handlers,
          declared_local_tool_definitions=declared_definitions,
          capability_execution_resolver=resolver,
        )
        _, seed_error = await seed_handler({
          "operation": operation,
          "objective": "Run both private cells.",
          "background": False,
        })
        assert seed_error is None
        admitted = seed.spawn_calls[0]["admitted_task"]
        if authority_variant in {"legacy", "historical"}:
          snapshot = admitted.execution_snapshot
          assert snapshot is not None
          payload = admitted.model_dump(
            mode="json", exclude={"admitted_task_digest"},
          )
          if authority_variant == "legacy":
            payload["execution_snapshot"].pop("admitted_context")
          else:
            wire = snapshot.admitted_context
            assert wire is not None
            payload["execution_snapshot"]["admitted_context"] = wire.model_copy(
              update={
                "budget_share": -4.5,
                "lineage": wire.lineage.model_copy(update={"depth": 9}),
              },
            ).model_dump(mode="json")
          seal_admitted_task_payload(payload)
          admitted = AdmittedTask.model_validate(payload)
        original_id = admitted.attempt.physical_task_id
        canonical_bind = admitted.model_bind
        assert canonical_bind is not None
        correlation = {
          "owner_runner_id": runner._runner_id,
          "owner_role": runner._role,
          "sub_agent_id": f"sub:{original_id}",
          "parent_turn_id": f"turn:{original_id}",
          "call_index": 0,
          "capability_bind": canonical_bind.to_json(),
        }
        if authority_variant == "different":
          # The duplicate registration receipt is not model authority; only
          # the digested canonical admitted task controls materialization.
          correlation["capability_bind"] = {
            **canonical_bind.to_json(),
            "upstream_model": "incorrect-duplicate-model",
          }
        elif authority_variant == "absent":
          correlation.pop("capability_bind")
        await session_log.append({
          "type": "task_registered",
          "task_id": original_id,
          "task_type": "background",
          "agent_name": "kernel-child",
          "started_at": 1.0,
          **correlation,
          "metadata": {
            **correlation,
            "task_type": "background",
            ADMITTED_TASK_METADATA_KEY: admitted.model_dump(mode="json"),
          },
        })
        await session_log.append({
          "type": "attach",
          "sub_agent_id": f"sub:{original_id}",
          "runner_id": f"runner:{original_id}",
          "role": "sub_agent",
        })
        await session_log.append({
          "type": "user_message",
          "task_id": original_id,
          "sub_agent_id": f"sub:{original_id}",
          "runner_id": f"runner:{original_id}",
          "role": "sub_agent",
          "content": "Run both private cells.",
        })
        handler = make_resume_handler(
          [runner], excluded_tools_resolver=frozenset, **child_kwargs,
        )
        materialized_binds.clear()
        result, resume_error = await handler({"task_id": original_id})
        assert resume_error is None and result is not None
        assert materialized_binds == [canonical_bind]
        entry = runner._task_registry.get(result["task_id"])
        assert entry is not None and entry.asyncio_task is not None
        successor = entry.admitted_task
        assert successor is not None and successor.execution_snapshot is not None
        assert admitted.execution_snapshot is not None
        assert successor.model_bind == admitted.model_bind
        assert successor.tool_routes == admitted.tool_routes
        assert successor.workspace_grant == admitted.workspace_grant
        assert successor.capability_bindings == admitted.capability_bindings
        assert (
          successor.execution_snapshot.provider_tool_definitions
          == admitted.execution_snapshot.provider_tool_definitions
        )
        successor_wire = successor.execution_snapshot.admitted_context
        assert successor_wire is not None
        assert successor_wire.capability_bind == admitted.model_bind
        original_wire = admitted.execution_snapshot.admitted_context
        if original_wire is not None:
          assert successor_wire.budget_share == original_wire.budget_share
          assert successor_wire.lineage == original_wire.lineage
        child_task = entry.asyncio_task
      if termination == "cancel":
        await asyncio.wait_for(provider.paused.wait(), timeout=10)
        child_task.cancel()
      if termination == "binding_error" and hop == "spawn":
        with pytest.raises(RuntimeError, match="child dispatcher binding failed"):
          await child_task
      else:
        outcome = (await asyncio.gather(child_task, return_exceptions=True))[0]
        if termination in {"success", "error"}:
          assert not isinstance(outcome, BaseException)
          if hop == "spawn":
            result, run_error = outcome
            assert run_error is None and result is not None
      for child_session in child_sessions:
        assert child_session is not parent_session
        assert child_session.code_execution_kernels == {}
        assert child_session.code_execution_frame_reader is None
        assert child_session.code_execution_work_dir is None
      assert all(not workdir.exists() for workdir in child_workdirs)
      assert all(reader._closed and not reader._cells for reader in child_readers)
      if termination in {"unavailable", "unavailable_status"}:
        assert provider.available_tools and all(
          not tools for tools in provider.available_tools
        )
        assert provider.results and all(
          result["error"]["code"] == "unknown_tool"
          for result in provider.results
        )
      elif termination != "binding_error":
        assert len(provider.results) == 2
        first, second = provider.results
        assert first["return_code"] == second["return_code"] == 0
        first_lines = first["stdout"].splitlines()
        second_lines = second["stdout"].splitlines()
        assert first_lines[0] == "False"
        assert second_lines[0] == "child-local"
        assert first_lines[1] == second_lines[1]
        assert int(first_lines[1]) != parent_pid
        if termination != "custom":
          assert child_processes
          assert all(
            process.returncode is not None for process in child_processes
          )
      assert parent_process.returncode is None
      alive, alive_error = await parent_dispatcher.dispatch(
        "parent-after-child",
        "code_execute",
        {"code": "print(parent_token)\nprint('child_token' in globals())\nprint(os.getpid())", "host": "kernel"},
      )
      assert alive_error is None and alive is not None
      assert alive["stdout"] == f"parent-local\nFalse\n{parent_pid}\n"
      if provider.status_task_id is not None:
        # The child did not collect or cancel a task belonging to its parent.
        status, status_error = await parent_dispatcher.dispatch(
          "parent-collect", "code_execute_status",
          {"task_id": provider.status_task_id},
        )
        assert status_error is None and status is not None
        assert status["stdout"] == "parent-local\n"
    finally:
      await runner.force_close()
      await cleanup_code_execution(parent_session)

  asyncio.run(run())
