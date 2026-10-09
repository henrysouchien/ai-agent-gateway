from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request, Response
from fastapi.routing import APIRoute

import agent_gateway.autonomous as autonomous
import model_authority.selection as selection
from agent_gateway import EventLog, create_agent
from agent_gateway.admission import (
  AdmittedContext,
  AdmissionOwners,
  OperationRegistration,
  ParentContext,
  SpawnRequest,
  admit,
  current_parent_context,
  root_context,
)
from agent_gateway.capability_execution import CapabilityExecutionResolver
from agent_gateway.capability_resolution import OperationDeclaration, resolve_operation_authority
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.autonomous_runner import AutonomousRegistry
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.single_user_policy import SingleUserApprovalPolicy
from agent_gateway.control_plane import batches, runs, schedules
from agent_gateway.control_plane.runs_models import AutonomousDispatchRequest
from agent_gateway.server import ChatRequest, ChatTurnInputs, GatewayServerConfig, create_gateway_app
from agent_gateway.server_chat_helpers import prepare_session_driver_turn
from agent_gateway.session import GatewaySession
from agent_workflow_contracts import CatalogToolEntry, PlatformToolCatalog, ResolvedAuthority
from gateway_test_support.capability_execution_test_support import (
  stub_capability_execution_resolver,
)
from gateway_test_support.host_policy import owner_session_host_policy
from model_authority.selection import ProductModelSelectionPolicy, SelectionSnapshot


@dataclass
class _Observation:
  captures: list[tuple[ProductModelSelectionPolicy, SelectionSnapshot]] = field(default_factory=list)
  selections: list[tuple[str, SelectionSnapshot | None]] = field(default_factory=list)


def _observe_snapshots(monkeypatch: pytest.MonkeyPatch) -> _Observation:
  observation = _Observation()
  capture = selection.capture_selection_snapshot
  select_bind = CapabilityExecutionResolver.select_bind

  def recording_capture(policy: ProductModelSelectionPolicy, **kwargs: Any) -> SelectionSnapshot:
    snapshot = capture(policy, **kwargs)
    observation.captures.append((policy, snapshot))
    return snapshot

  def recording_select(self: CapabilityExecutionResolver, capability_id: str, **kwargs: Any) -> Any:
    observation.selections.append((capability_id, kwargs.get("selection_snapshot")))
    return select_bind(self, capability_id, **kwargs)

  monkeypatch.setattr(selection, "capture_selection_snapshot", recording_capture)
  monkeypatch.setattr(runs, "capture_selection_snapshot", recording_capture)
  monkeypatch.setattr(schedules, "capture_selection_snapshot", recording_capture)
  monkeypatch.setattr(CapabilityExecutionResolver, "select_bind", recording_select)
  return observation


def _session(session_id: str = "snapshot-root") -> GatewaySession:
  return GatewaySession(
    session_id=session_id,
    api_key_hash="test",
    created_at=1,
    expires_at=2,
    user_id="test-actor",
    owner_user_id="test-actor",
    role="owner",
    tenant_id="test-tenant",
    channel="cli",
  )


def _admit_child(
  parent: ParentContext,
  resolver: CapabilityExecutionResolver,
  session: GatewaySession,
  request: SpawnRequest | None = None,
) -> AdmittedContext[ResolvedAuthority]:
  catalog = PlatformToolCatalog(tools=(CatalogToolEntry(
    tool_id="file_read", canonical_name="file_read", origin="local", effect="read",
  ),))
  declaration = OperationDeclaration(
    operation_name="explore", grant_id="grant:root-snapshot", workspace_scope="read_only",
    required_capabilities=(),
  )
  admitted = admit(
    request or SpawnRequest(operation="explore", task="Read evidence"),
    parent,
    owners=AdmissionOwners(
      operation=OperationRegistration(
        name="explore", execution_class="node.explore", route_ceiling=frozenset({"file_read"}),
        operation_routes=frozenset(),
      ),
      capability_execution_resolver=resolver,
      session=session,
      route_registered=lambda tool_id: tool_id == "file_read",
      resolve_tool_authority=lambda tool_ids: resolve_operation_authority(
        replace(declaration, tool_ceiling=tool_ids), catalog=catalog,
      ),
      placement="in_process",
    ),
  )
  assert isinstance(admitted, AdmittedContext), admitted
  return admitted


def _assert_root_capture(
  parent: ParentContext,
  resolver: CapabilityExecutionResolver,
  observation: _Observation,
) -> SelectionSnapshot:
  assert isinstance(parent, ParentContext)
  snapshot = parent.selection_snapshot
  assert isinstance(snapshot, SelectionSnapshot)
  assert snapshot.policy_revision == resolver.selection_policy.revision
  assert len(observation.captures) == 1
  policy, captured = observation.captures[0]
  assert policy is resolver.selection_policy
  assert snapshot is captured
  return snapshot


def _assert_child_keeps_capture(
  admitted: AdmittedContext[ResolvedAuthority],
  snapshot: SelectionSnapshot,
  observation: _Observation,
) -> None:
  selections = [value for capability, value in observation.selections if capability == "node.explore"]
  assert len(selections) == 1
  assert selections[0] is snapshot
  assert admitted.wire.selection_snapshot == snapshot.to_json()
  assert len(observation.captures) == 1
  # A later unrelated mint must not inherit this root's capture implicitly.
  later = root_context(_session("later-root"), withheld_tools=frozenset(), research_file_id=None)
  assert later.selection_snapshot is None


def test_easy_runtime_root_captures_once_and_forwards_to_child(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  owner_session_host_policy: Any,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  (tmp_path / "gateway").mkdir(mode=0o700)
  app = create_agent("System", api_key="test-key", session_log_base_dir=tmp_path / "sessions")
  config = app.state.gateway_config
  session = app.state.auth.session_store.create_session(
    api_key_hash="test",
    user_id="test-actor",
    tenant_id=config.tenant_id,
    allow_service_for_interactive=config.allow_service_credentials_for_interactive,
    model_entitled_capabilities=frozenset(config.model_selection_policy.capabilities),
    model_entitled_keys=frozenset(
      key for policy in config.model_selection_policy.capabilities.values()
      for key in policy.allowed_model_keys
    ),
  )

  async def build_and_spawn() -> None:
    request = ChatRequest.model_validate({"messages": [{"role": "user", "content": "Read evidence"}]})
    prepared = prepare_session_driver_turn(
      session,
      ChatTurnInputs(
        messages=list(request.messages), request_id=request.request_id,
        context=dict(request.context), metadata=dict(request.metadata),
        model_key=request.model_key, effort=request.effort,
        catalog_revision=request.catalog_revision, ui_blocks_contract=request.ui_blocks_contract,
        commercial_work_start=request.commercial_work_start,
      ),
      build_chat_runtime=app.state.gateway_build_chat_runtime,
    )
    runtime = await app.state.gateway_build_chat_runtime(
      session=session, request=prepared.request, channel=None, auth_manager=app.state.auth,
    )
    resolver = prepared.request.capability_execution_resolver
    assert isinstance(resolver, CapabilityExecutionResolver)
    observation = _observe_snapshots(monkeypatch)
    runner = runtime.build_runner(EventLog(), session.session_id)
    assert runner is not None
    parent = current_parent_context(session)
    snapshot = _assert_root_capture(parent, resolver, observation)
    child = _admit_child(parent, resolver, session)
    _assert_child_keeps_capture(child, snapshot, observation)

  asyncio.run(build_and_spawn())


def test_batch_stage_factory_captures_once_and_forwards_to_child(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  owner_session_host_policy: Any,
) -> None:
  resolver = stub_capability_execution_resolver(run_mode="batch")
  execution = resolver.resolve("session.driver")
  session = _session("batch-root")
  observation = _observe_snapshots(monkeypatch)
  parents: list[ParentContext] = []
  children: list[AdmittedContext[ResolvedAuthority]] = []
  tasks: list[asyncio.Task[Any]] = []
  operation_catalog = object()

  def stage_admitter(parent: ParentContext, received_resolver: CapabilityExecutionResolver, catalog: Any):
    parents.append(parent)
    assert received_resolver is resolver
    assert catalog is operation_catalog

    def admit_stage(request: SpawnRequest, _tool_ceiling: frozenset[str]):
      return _admit_child(parent, received_resolver, session, request)

    return admit_stage

  class Controller:
    def acquire_batch_run(self, _payload: dict[str, Any], **_kwargs: Any):
      return 17, session.user_id, None

    async def run_acquired_batch(self, _batch_id: int, _payload: dict[str, Any], **kwargs: Any):
      factory = kwargs["admit_stage_factory"]
      stage = factory(SimpleNamespace(research_file_id=42))
      children.append(stage(SpawnRequest(operation="explore", task="Read evidence"), frozenset({"file_read"})))

  async def open_context(**_kwargs: Any) -> None:
    raise AssertionError("this stage admission test does not execute a worker")

  controller = Controller()
  backend = batches.BatchBackend(
    registry_for_user=lambda _user_id: None,
    controller=controller,
    open_captured_run_context=open_context,
    stage_admitter=stage_admitter,
    terminal_event_payload=lambda _batch_id, _digest: {},
    active_batch_error=RuntimeError,
    dispatch_replay=batches.BatchDispatchReplayError,
    dispatch_rejected=batches.BatchDispatchRejectedError,
    dispatch_rejection_record=SimpleNamespace,
  )
  app_state = SimpleNamespace(
    batch_backend=backend,
    gateway_config=SimpleNamespace(tenant_id=session.tenant_id),
    gateway_claim_signing_authority=GatewayClaimSigningAuthority("snapshot-test-secret-at-least-32-bytes"),
    autonomous_storage_root=tmp_path,
    gateway_skill_application=SimpleNamespace(operation_catalog=operation_catalog),
    gateway_approval_store=None,
    gateway_approval_policy=None,
  )
  task_registry: Any = SimpleNamespace(start=lambda **kwargs: tasks.append(kwargs["task"]))

  async def start_and_admit() -> None:
    batch_id = batches._acquire_and_start_batch(
      {}, app_state=app_state, registry=object(), task_registry=task_registry,
      dispatch_key="snapshot-test", dispatch_request_spec={},
      user_id=session.user_id, user_email=None, role="owner", channel="cli",
      authenticated_session=session, capability_execution_resolver=resolver,
      session_driver_execution=execution,
    )
    assert batch_id == 17
    assert len(tasks) == 1
    await tasks[0]
    assert len(parents) == len(children) == 1
    assert parents[0] is current_parent_context(session)
    assert parents[0].research_binding.research_file_id == 42
    snapshot = _assert_root_capture(parents[0], resolver, observation)
    _assert_child_keeps_capture(children[0], snapshot, observation)

  asyncio.run(start_and_admit())


@pytest.mark.parametrize("with_resolver", [True, False], ids=["resolver", "no-resolver"])
def test_run_autonomous_root_snapshot_matches_optional_resolver(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  owner_session_host_policy: Any,
  with_resolver: bool,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  resolver = stub_capability_execution_resolver(run_mode="autonomous")
  execution = resolver.resolve("session.driver")
  execution = replace(execution, auth_config={**execution.auth_config, "max_tokens": 4096})
  session = _session("autonomous-root")
  observation = _observe_snapshots(monkeypatch)

  async def run_session(_runner: Any, _event_log: Any, **_kwargs: Any) -> autonomous.RunOutput:
    parent = current_parent_context(session)
    if with_resolver:
      snapshot = _assert_root_capture(parent, resolver, observation)
      child = _admit_child(parent, resolver, session)
      _assert_child_keeps_capture(child, snapshot, observation)
    else:
      assert isinstance(parent, ParentContext)
      assert parent.selection_snapshot is None
      assert observation.captures == []
    return autonomous.RunOutput("complete", [], {}, None, False)

  monkeypatch.setattr(autonomous, "run_session", run_session)
  output = asyncio.run(autonomous.run_autonomous(
    "System", "Read evidence", capability_execution=execution, session=session,
    capability_execution_resolver=resolver if with_resolver else None,
    user_id=session.user_id, billing_mode="byok", rate_table_version="test",
  ))
  assert output.response == "complete"
  assert len(observation.captures) == (1 if with_resolver else 0)


def test_product_unadmitted_runtime_root_captures_once_and_forwards_to_child(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  owner_session_host_policy: Any,
) -> None:
  # Product imports stay inside the test: the gateway wheel collects without api.
  root = Path(__file__).resolve().parents[3]
  if not (root / "api" / "agent" / "autonomous" / "runner.py").is_file():
    pytest.skip("product runtime is not included in the standalone gateway package")
  monkeypatch.syspath_prepend(str(root))
  monkeypatch.syspath_prepend(str(root / "api"))
  import tests as product_tests
  monkeypatch.setattr(product_tests, "__path__", [str(root / "tests"), *getattr(product_tests, "__path__")])
  import memory
  from api.agent.autonomous import runner as runner_mod
  from agent.profiles import load_profile_composition
  from tests.autonomous_exact_test_support import build_exact_autonomous_test_runtime

  monkeypatch.delenv("ANALYST_DEV_MODE", raising=False)
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "user-data"))
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  monkeypatch.setattr(memory, "get_skills_root", lambda: skills_dir)
  monkeypatch.setattr(runner_mod, "build_session_code_execution", lambda **_kwargs: None)
  monkeypatch.setattr(runner_mod, "get_or_create_interceptors", lambda *_args, **_kwargs: [])

  class McpManager:
    async def startup(self, *, allowed_servers: set[str]) -> None:
      assert not allowed_servers

    def get_server_names(self) -> set[str]:
      return set()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
      return []

    def get_server_tool_definition_records(self, _servers: set[str]) -> tuple[Any, ...]:
      return ()

    def get_server_tool_definitions(self, _servers: set[str]) -> list[dict[str, Any]]:
      return []

    def get_server_catalog(self) -> dict[str, Any]:
      return {}

    def get_startup_diagnostics(self) -> dict[str, Any]:
      return {}

  async def read_memory(_input: Any, **_kwargs: Any):
    return {}, None

  composition = load_profile_composition("analyst")
  composition = replace(composition, config=replace(
    composition.config, local_tool_names={"memory_read"}, always_on_mcp_servers=set(),
    core_mcp_tools={}, deferred_mcp_servers=set(), tool_packs={},
  ))
  runtime = build_exact_autonomous_test_runtime()
  try:
    resolver = runtime.capability_execution_resolver
    observation = _observe_snapshots(monkeypatch)
    config = runner_mod.RuntimeConfig(
      excluded_tools=frozenset(), session_driver_execution=runtime.session_driver_execution,
      execution_policy=composition.config.execution_policy,
    )
    assert config.admitted is None
    context = asyncio.run(runner_mod.setup_runtime_context(
      composition, config, mcp_render=runtime.mcp_render,
      session_id=runtime.gateway_session.session_id,
      user_id=runtime.gateway_session.user_id, user_email=runtime.gateway_session.user_email,
      claim_signer=runtime.claim_signer, tool_packs={}, workspace=tmp_path,
      workspace_context="", mcp_client_manager=McpManager(),
      build_local_tool_handlers_fn=lambda *_args, **_kwargs: {"memory_read": read_memory},
      capability_execution_resolver=resolver, session_driver_execution=runtime.session_driver_execution,
      gateway_session=runtime.gateway_session, event_owner=runtime.event_owner,
      skill_application=runtime.skill_application, context_research_file_id=42,
    ))
    assert context.gateway_session is runtime.gateway_session
    parent = current_parent_context(context.gateway_session)
    assert parent.research_binding.research_file_id == 42
    snapshot = _assert_root_capture(parent, resolver, observation)
    child = _admit_child(parent, resolver, context.gateway_session)
    _assert_child_keeps_capture(child, snapshot, observation)
  finally:
    runtime.close()


class _RootObserved(BaseException):
  """Stop at the injected launch dependency, without starting a subprocess."""


@pytest.mark.parametrize("site", ["control-dispatch", "schedule-occurrence", "schedule-run-now"])
@pytest.mark.parametrize("with_policy", [True, False], ids=["policy", "no-policy"])
def test_control_root_captures_declared_policy_and_forwards_to_child(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  owner_session_host_policy: Any,
  site: str,
  with_policy: bool,
) -> None:
  resolver = stub_capability_execution_resolver(run_mode="autonomous")
  session = _session("control-root")
  session.kind = "control"
  registry = AutonomousRegistry(
    api_dir=tmp_path / "api", log_dir=tmp_path / "logs",
    tenant_id=session.tenant_id,
    approval_store=SQLiteApprovalStore(tmp_path / "control-approvals.sqlite3"),
    approval_policy=SingleUserApprovalPolicy(),
    mcp_user_key_lookup=lambda *_args: {"role": "owner"},
    autonomous_selection_policy=resolver.selection_policy if with_policy else None,
  )
  assert registry.selection_policy is (resolver.selection_policy if with_policy else None)
  observation = _observe_snapshots(monkeypatch)
  parents: list[ParentContext] = []
  children: list[AdmittedContext[ResolvedAuthority]] = []

  async def observe_start(**kwargs: Any) -> Any:
    parent = kwargs["parent"]
    parents.append(parent)
    children.append(_admit_child(parent, resolver, session))
    raise _RootObserved()

  monkeypatch.setattr(registry, "start", observe_start)
  store = schedules.AgentRunScheduleStore(tmp_path / "schedules.json")
  record = store.create(
    schedules.CreateAgentRunScheduleRequest.model_validate({
      "kind": "agent_run_schedule", "name": "root-snapshot",
      "enabled": True, "timezone": "UTC",
      "cadence": {"type": "weekly", "days_of_week": [1], "time_of_day": "08:30"},
      "dispatch": {"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "Read evidence"},
      "request_id": "snapshot-schedule",
    }),
    session,
  )
  schedule_runner = schedules.AgentRunScheduleRunner(
    store_for=lambda _owner: store, users_root=tmp_path / "users",
    autonomous_registry=registry, poll_interval_seconds=0,
  )
  # Auth is a supplied dependency; the real endpoint still checks control kind,
  # role and owner, and mints its own ParentContext.
  auth: Any = SimpleNamespace(verify_token=lambda _token: session)
  request = Request({
    "type": "http", "headers": [(b"authorization", b"Bearer test")],
    "app": SimpleNamespace(state=SimpleNamespace(user_event_bus=None)),
  })

  async def enter_root() -> None:
    if site == "schedule-occurrence":
      result = await schedule_runner._fire_record(store, record, live_role=None, claim_id="occurrence-1")
      raise AssertionError(f"schedule occurrence did not reach the launch dependency: {result!r}")
    elif site == "schedule-run-now":
      router = schedules.build_schedules_router(
        auth=auth, agent_schedule_store_for=lambda _owner: store,
        agent_schedule_runner=schedule_runner,
      )
      endpoint: Any = next(
        route.endpoint for route in router.routes
        if isinstance(route, APIRoute) and route.path == "/schedules/{name}/run-now"
      )
      await endpoint(request, record["schedule_id"])
    else:
      router = runs.build_runs_router(auth=auth, autonomous_registry=registry)
      endpoint = next(
        route.endpoint for route in router.routes
        if isinstance(route, APIRoute) and route.path == "/runs" and route.methods and "POST" in route.methods
      )
      await endpoint(
        request,
        AutonomousDispatchRequest.model_validate({
          "kind": "autonomous", "profile": "analyst", "mode": "task", "task": "Read evidence",
          "research_file_id": 42,
        }),
        Response(),
      )

  with pytest.raises(_RootObserved):
    asyncio.run(enter_root())
  assert len(parents) == len(children) == 1
  parent = parents[0]
  assert isinstance(parent, ParentContext)
  assert parent.depth == 0
  if site == "schedule-occurrence":
    assert parent.job_id == "occurrence-1"
  else:
    assert parent is current_parent_context(session)
  if site == "control-dispatch":
    assert parent.research_binding.research_file_id == 42
  if with_policy:
    snapshot = _assert_root_capture(parent, resolver, observation)
    _assert_child_keeps_capture(children[0], snapshot, observation)
  else:
    assert parent.selection_snapshot is None
    assert children[0].wire.selection_snapshot is None
    assert observation.captures == []
    assert observation.selections == [("node.explore", None)]


def test_server_composition_retains_autonomous_policy_object(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))
  (tmp_path / "data" / "gateway").mkdir(mode=0o700, parents=True)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(tmp_path / "logs"))
  resolver = stub_capability_execution_resolver()
  autonomous_policy = replace(resolver.selection_policy, revision="autonomous-owner-policy")

  async def unused_runtime(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("composition does not build a chat runtime")

  config = GatewayServerConfig(
    jwt_secret="snapshot-composition-secret-at-least-32-bytes",
    valid_api_keys={"test-key"},
    tenant_id="test-tenant",
    model_registry=resolver.registry,
    model_selection_policy=resolver.selection_policy,
    capability_adapter_resolver=resolver.adapter_resolver,
    build_chat_runtime=unused_runtime,
    autonomous_api_dir=tmp_path / "api",
    autonomous_selection_policy=autonomous_policy,
  )
  app = create_gateway_app(config)
  assert app.state.gateway_config.autonomous_selection_policy is autonomous_policy
  registry = app.state.subprocess_registry
  assert isinstance(registry, AutonomousRegistry)
  assert registry.selection_policy is autonomous_policy
  assert registry.selection_policy is not config.model_selection_policy
  assert app.state.agent_run_schedule_runner.autonomous_registry is registry
  with pytest.raises(AttributeError):
    setattr(registry, "selection_policy", resolver.selection_policy)
