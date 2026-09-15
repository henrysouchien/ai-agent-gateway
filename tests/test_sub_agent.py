from __future__ import annotations

import asyncio
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import JsonValue

from agent_workflow_contracts import (
  ActivityHandle,
  AdmittedTask,
  AgentCompletionEnvelope,
  AgentOperationRef,
  AnalyticalOutcome,
  CanonicalProjection,
  ContentHandle,
  ContractRef,
  ExecutionSettlement,
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
from agent_gateway.capability_execution import CapabilityExecutionResolver
from agent_gateway.agent_result_content import (
  make_get_agent_result_content_handler,
)
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.approval_route import (
  DurableLocalApprovalRoute,
  bind_session_approval_route,
)
from agent_gateway.event_log import EventLog
from agent_gateway.runner_background_tasks import (
  ParentResultMaterializationError,
)
from agent_gateway.final_narrative_artifact import publish_final_narrative
from agent_gateway.operation_catalog import (
  OperationRuntimePolicy,
  ResolvedOperationRuntime,
)
from agent_gateway.operation_snapshot import build_agent_operation_snapshot
from agent_gateway.tool_dispatcher_helpers import ToolExecutionContext
from agent_gateway.session import GatewaySession
from agent_gateway.skills import SkillLoader, operation_tool_ids
from agent_gateway.sub_agent import (
  _ticker_admitted_input,
  _ticker_from_admitted_inputs,
  _runtime_policy_dispatch_projection,
  make_get_background_result_handler,
  make_get_background_result_tool_def,
  make_run_agent_handler,
  make_run_agent_tool_def,
  make_resume_tool_def,
  make_resume_handler,
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
from tests.capability_execution_test_support import (
  stub_capability_execution_resolver,
)


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

  def resolve(self, capability_id: str, **kwargs: Any) -> Any:
    self.calls.append({"capability_id": capability_id, **kwargs})
    return super().resolve(capability_id, **kwargs)


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
  approved_tool_types: set[str] | None = None,
  trusted_research_file_id: int | None = None,
  mcp_client: Any | None = None,
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
  parent_session.approved_tool_types = set(approved_tool_types or ())
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
  )


def _write_operation(
  path: Path,
  *,
  resumable: bool = True,
  required_context: tuple[str, ...] = (),
  operation_name: str = "filing-review",
  investment_start_route: bool = False,
  max_budget_usd: float | None = None,
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
  budget_yaml = (
    f"max_budget_usd: {max_budget_usd}\n"
    if max_budget_usd is not None
    else ""
  )
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
version: '1.0'
agent_callable: true
agent_description: Review filing evidence.
mutation_mode: {mutation_mode}
resumable: {str(resumable).lower()}
{budget_yaml}{scope_yaml}allowed_tools:
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
  extra_excluded_tools: frozenset[str] = frozenset(),
  max_budget_usd: float | None = 4.5,
) -> ResolvedOperationRuntime:
  snapshot = build_agent_operation_snapshot(
    name=name,
    version="1.0",
    methodology_instructions="Review canonical evidence.",
    resolved_instructions="Review canonical evidence.",
    description="Review canonical evidence.",
    execution_class="node.explore",
    required_capabilities=required_capabilities,
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
      mutation_mode="read_only",
      run_mode="full",
      exact_tool_ids=exact_tool_ids,
      mcp_tools_by_server=mcp_tools,
      runtime_server_refs=frozenset(mcp_tools),
      session_inject_servers=session_inject_servers,
      timeout_overrides={},
      extra_excluded_tools=extra_excluded_tools,
      max_turns=7,
      timeout_seconds=45,
      max_tokens=2_222,
      max_budget_usd=max_budget_usd,
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


def test_run_agent_tool_schema_is_operation_first(tmp_path: Path) -> None:
  schema = make_run_agent_tool_def(_write_operation(tmp_path))["input_schema"]

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


def test_run_agent_handler_rejects_catalog_and_loader_together(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
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
  resolved = _catalog_runtime()
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
  assert spawn["max_turns"] == 7
  assert spawn["timeout"] == 45
  assert spawn["max_tokens"] == 2_222
  assert spawn["max_budget_usd"] == 4.5
  assert TaskResult.model_validate(result).logical_task.operation == (
    resolved.snapshot.operation
  )


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
    max_budget_usd=8.25,
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
  assert spawn["max_budget_usd"] == 8.25
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

  exact, excluded, canonical_to_exposed = (
    _runtime_policy_dispatch_projection(
      runtime.policy,
      mcp_client=_TwoServerMcpClient(),
    )
  )

  assert exact == frozenset({"a_shared_tool", "b_shared_tool"})
  assert excluded == frozenset()
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

  exact, excluded, canonical_to_exposed = (
    _runtime_policy_dispatch_projection(
      runtime.policy,
      mcp_client=_LogicalRouteMcpClient(),
    )
  )

  assert exact == frozenset({"fetch_company_profile"})
  assert excluded == frozenset()
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


@pytest.mark.parametrize(
  ("exact_tool_ids", "mcp_tools_by_server", "extra_excluded_tools"),
  (
    (
      frozenset({"mcp__research-corpus-mcp__thesis_read"}),
      {"research-corpus-mcp": frozenset({"thesis_read"})},
      frozenset(),
    ),
    (
      frozenset({"web_search"}),
      {},
      frozenset({"mcp__research-corpus-mcp__thesis_read"}),
    ),
  ),
)
def test_catalog_mcp_alias_cannot_impersonate_live_local_handler(
  exact_tool_ids: frozenset[str],
  mcp_tools_by_server: dict[str, frozenset[str]],
  extra_excluded_tools: frozenset[str],
) -> None:
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
    exact_tool_ids=exact_tool_ids,
    mcp_tools_by_server=mcp_tools_by_server,
    extra_excluded_tools=extra_excluded_tools,
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


@pytest.mark.parametrize(
  ("exact_tool_ids", "mcp_tools_by_server", "extra_excluded_tools"),
  (
    (
      frozenset({"mcp__research-corpus-mcp__thesis_read"}),
      {"research-corpus-mcp": frozenset({"thesis_read"})},
      frozenset({"private_thesis_read"}),
    ),
    (
      frozenset({"private_thesis_read"}),
      {},
      frozenset({"mcp__research-corpus-mcp__thesis_read"}),
    ),
  ),
)
def test_policy_projection_refuses_exact_excluded_alias_collisions(
  exact_tool_ids: frozenset[str],
  mcp_tools_by_server: dict[str, frozenset[str]],
  extra_excluded_tools: frozenset[str],
) -> None:
  runtime = _catalog_runtime(
    exact_tool_ids=exact_tool_ids,
    mcp_tools_by_server=mcp_tools_by_server,
    extra_excluded_tools=extra_excluded_tools,
  )
  mcp_client = _PrefixedMcpClient()
  mcp_client.activate()

  with pytest.raises(
    ValueError,
    match="exact dispatcher tools collide with projected exclusions",
  ):
    _runtime_policy_dispatch_projection(
      runtime.policy,
      mcp_client=mcp_client,
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
  loader = _write_operation(tmp_path, required_context=("ticker",))
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
  loader = _write_operation(tmp_path, required_context=("ticker",))
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
  assert resolver.calls == []
  assert runner.spawn_calls == []
  assert runner.background_calls == []


@pytest.mark.parametrize("ticker", ["PCTY", "QCOM"])
def test_direct_handler_unambiguous_legacy_prose_is_promoted_once(
  tmp_path: Path,
  ticker: str,
) -> None:
  loader = _write_operation(tmp_path, required_context=("ticker",))
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
  loader = _write_operation(tmp_path)
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
  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path / "skills"),
  )({
    "objective": "Return the complete evidence report.",
    "background": False,
  }))

  assert error is None
  envelope = AgentCompletionEnvelope.model_validate(result)
  assert envelope.parent_materialization.kind == "result_handle"
  source = envelope.parent_materialization.source
  grant = envelope.parent_materialization.read_grant
  page, read_error = asyncio.run(
    make_get_agent_result_content_handler([runner])({
      "content_id": source.content_id,
      "read_grant_id": grant.grant_id,
    })
  )
  assert read_error is None
  assert page is not None
  assert page["content"] == terminal_text
  assert page["end"] is True
  events, _cursor = asyncio.run(runner.session_log.query(
    event_types={
      "task_registered",
      "task_completed",
      "agent_completion",
    },
    order="asc",
  ))
  assert [entry.event["type"] for entry in events] == [
    "task_registered",
    "task_completed",
    "agent_completion",
  ]


def test_foreground_delegation_publishes_budget_exhaustion_without_narrative(
  tmp_path: Path,
) -> None:
  class _BudgetExhaustedRunner(_Runner):
    def __init__(self) -> None:
      super().__init__()
      self._gateway_session_id = "session-test"
      self._runner_id = "parent-runner"
      self._role = "writer"
      self._workspace_dir = tmp_path
      self._agent_session_log = AgentSessionLog(tmp_path / "session.jsonl")

    async def _append_durable_event(self, event: dict[str, Any]) -> object:
      return await self._agent_session_log.append(event)

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
          {"type": "budget_exceeded"},
        ],
        logical_task=kwargs["logical_task"],
        attempt=kwargs["attempt"],
        requirement=kwargs["result_requirement"],
        provenance=kwargs["result_provenance"],
        final_narrative=None,
        timed_out=False,
        timeout=None,
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
  assert envelope.settlement_projection.terminal_reason == "budget_exhausted"
  assert envelope.parent_materialization is None
  assert envelope.child_evidence.evidence_tools == ("fetch_financials",)
  events, _ = asyncio.run(runner._agent_session_log.query(
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
    ([{"type": "budget_exceeded"}], (), "failed", "budget_exhausted"),
    ([{"type": "max_turns_reached"}], (), "failed", "turns_exhausted"),
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
        timed_out=False,
        timeout=None,
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
  assert envelope.parent_materialization.kind == "projection_inline"
  assert envelope.parent_materialization.value == projection_value


def test_registered_operation_requires_full_ref_and_emits_lifecycle(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
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


def test_registered_operation_captures_recoverable_fms_error_without_internal_error(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
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
  assert captured["status"] == "error"
  assert captured["error"] == "active research file is required"


def test_registered_operation_activates_declared_mcp_before_admission(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
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
  loader = _write_operation(tmp_path)
  runner = _Runner()
  approval_store = object()
  approval_policy = object()

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    approval_store=approval_store,
    approval_policy=approval_policy,
    approved_tool_types={"fms_persist_test_artifact"},
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  dispatcher = runner.spawn_calls[0]["dispatcher"]
  # The child inherits the parent's admitted route as a value — the same object
  # the parent session carries — rather than re-deriving one from handles. The
  # headless flag is decided independently, so a background child pairs
  # should_avoid_permission_prompts with this same live durable route.
  parent_route = dispatcher._session.approval_route
  assert isinstance(parent_route, DurableLocalApprovalRoute)
  assert dispatcher._approval_route is parent_route
  assert parent_route.session is dispatcher._session
  assert dispatcher._approval_store is approval_store
  assert dispatcher._approval_policy is approval_policy
  assert dispatcher._session is not None
  assert dispatcher._approved_tool_types == {"fms_persist_test_artifact"}
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
  loader = _write_operation(tmp_path)
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
  loader = _write_operation(tmp_path)
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
  loader = _write_operation(tmp_path)
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
  loader = _write_operation(tmp_path)
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
  loader = _write_operation(tmp_path)
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
  loader = _write_operation(tmp_path, max_budget_usd=6.0)
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
  assert admitted.execution_snapshot.max_budget_usd == pytest.approx(6.0)
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


def test_unnamed_delegation_does_not_infer_a_skill_budget(tmp_path: Path) -> None:
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
  assert runner.spawn_calls[0]["max_budget_usd"] is None
  assert "agent.shared.mutation_enforcement" not in inspect.getsource(
    make_run_agent_handler
  )


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
  assert error == {"code": "invalid_input", "message": "objective is required"}


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
      {"status": "completed", "result": "done"},
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
