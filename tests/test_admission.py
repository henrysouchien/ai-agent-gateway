from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_workflow_contracts import (
  AdmittedTask,
  CatalogToolEntry,
  PlatformToolCatalog,
  SemanticCapabilityRequirement,
)
from agent_workflow_contracts.models import (
  AdmittedContextWire,
  AdmittedResearchBinding,
)
from agent_gateway.admission import (
  AdmittedContext,
  AdmissionOwners,
  ContinuationOwners,
  ContinueRequest,
  HandleOwners,
  HandleRequest,
  MessageOwners,
  MessageRequest,
  OperationRegistration,
  ParentContext,
  Refusal,
  SpawnRequest,
  admit,
  context_for_child,
  current_parent_context,
  root_context,
  root_context_from_identity,
)
from agent_gateway.child_environment import RUN_BASH_ENVIRONMENT
from agent_gateway.approval_route import (
  NO_APPROVAL_ROUTE,
  ControlPlaneApprovalRoute,
  DurableLocalApprovalRoute,
  InProcessDelegatedApprovalRoute,
  route_ledger,
  route_session,
)
from agent_gateway.capability_resolution import OperationDeclaration, resolve_operation_authority
from agent_gateway.session import GatewaySession, SessionStore
from agent_gateway.sub_agent import make_run_agent_handler, seal_admitted_task_payload
from agent_gateway.sub_agent_scope_receipt import dispatcher_scopes_from_admitted_routes
from gateway_test_support.capability_execution_test_support import (
  stub_capability_execution_resolver,
)
from gateway_test_support.host_policy import owner_session_host_policy


def _session() -> GatewaySession:
  return GatewaySession(
    session_id="session-admission",
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


def _resolver():
  return stub_capability_execution_resolver(
    default_provider="openai",
    default_model="gpt-5.6-sol",
    default_effort="high",
  )


_BIND = _resolver().select_bind("node.explore")


class _PlainResolver:
  """A plain owner object: records reads, holds no state admit could write."""

  def __init__(self) -> None:
    self.reads: list[str] = []

  def select_bind(self, capability_id: str, *, effort: str | None = None) -> Any:
    self.reads.append(capability_id)
    return _BIND


def _owners(
  resolver: Any,
  *,
  ceiling: frozenset[str] = frozenset({"web_search", "file_read"}),
  operation_routes: frozenset[str] = frozenset(),
  registered: frozenset[str] | None = None,
  env_contract: str | None = None,
  env_names: frozenset[str] = frozenset(),
) -> AdmissionOwners:
  known = ceiling if registered is None else registered
  declaration = OperationDeclaration(
    operation_name="explore",
    grant_id="grant:admission",
    workspace_scope="read_only",
    required_capabilities=(
      SemanticCapabilityRequirement(
        name="web.read/v1", required=False, binding_modes=("live_tool",),
      ),
    ),
  )
  catalog = PlatformToolCatalog(tools=tuple(
    CatalogToolEntry(
      tool_id=tool_id,
      canonical_name=tool_id,
      origin="local",
      effect="write" if tool_id == "write_record" else "read",
      capability="web.read/v1",
    )
    for tool_id in sorted(known)
  ))
  return AdmissionOwners(
    operation=OperationRegistration(
      name="explore",
      execution_class="node.explore",
      route_ceiling=ceiling,
      operation_routes=operation_routes,
    ),
    capability_execution_resolver=resolver,
    session=_session(),
    route_registered=lambda tool_id: tool_id in known,
    placement="in_process",
    resolve_tool_authority=lambda tool_ids: resolve_operation_authority(
      replace(declaration, tool_ceiling=tool_ids),
      catalog=catalog,
    ),
    env_contract=env_contract,
    env_names=env_names,
  )


def _root(**changes: Any) -> ParentContext:
  root = root_context(
    _session(),
    withheld_tools=frozenset(),
    research_file_id=None,
  )
  return replace(root, **changes)


def _request(**changes: Any) -> SpawnRequest:
  return replace(SpawnRequest(operation="explore", task="Collect"), **changes)


# --- (a) one context for both doors -------------------------------------


class _BackgroundRunner:
  def __init__(self) -> None:
    self._full_session_id = "session-admission"
    self._agent_session_log = object()
    self.background_calls: list[dict[str, Any]] = []

  def _get_tool_definitions(self) -> list[dict[str, Any]]:
    return [
      {"name": name, "description": name, "input_schema": {"type": "object"}}
      for name in ("web_search", "file_read", "memory_read")
    ]

  async def _register_background_task(self, **kwargs: Any):
    self.background_calls.append(dict(kwargs))
    return {"task_id": kwargs["task_id_override"], "status": "running"}, None


class _McpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


async def _tool(_tool_input: dict[str, Any], **_kwargs: Any):
  return {"ok": True}, None


def _door_context(
  runner: _BackgroundRunner,
  door_kwargs: dict[str, Any],
  caplog: pytest.LogCaptureFixture,
) -> tuple[AdmittedContextWire, dict[str, Any]]:
  session = _session()
  handler = make_run_agent_handler(
    [runner],
    parent_session=session,
    mcp_client=_McpClient(),
    local_tool_handlers={
      "web_search": _tool,
      "file_read": _tool,
      "memory_read": _tool,
    },
    excluded_tools={"memory_read"},
    user_id="actor-test",
    trusted_research_file_id=42,
    capability_execution_resolver=_resolver(),
    **door_kwargs,
  )
  caplog.clear()
  with caplog.at_level(logging.INFO, logger="agent_gateway.admission"):
    result, error = asyncio.run(handler({
      "objective": "Collect",
      "background": True,
    }))
  assert error is None, error
  assert result is not None
  logged = [
    record.getMessage().split(": ", 1)[1]
    for record in caplog.records
    if record.getMessage().startswith("admitted ")
  ]
  assert len(logged) == 1
  snapshot = runner.background_calls[0]["admitted_task"].execution_snapshot
  assert snapshot.admitted_context is not None
  return snapshot.admitted_context, json.loads(logged[0])


def test_interactive_and_autonomous_doors_admit_equal_contexts(
  caplog: pytest.LogCaptureFixture,
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  # The kwargs each installer passes beyond the shared admit inputs:
  # api/agent/interactive/runtime_handlers.py and
  # api/agent/autonomous/runtime_child_handlers.py.
  interactive, interactive_logged = _door_context(
    _BackgroundRunner(),
    {
      "needs_approval": lambda *_args, **_kwargs: False,
      "mcp_session_inject_servers": set(),
      "mcp_meta_inject_servers": frozenset(),
      "credentials_resolver_active": True,
    },
    caplog,
  )
  autonomous, autonomous_logged = _door_context(
    _BackgroundRunner(),
    {
      "mcp_session_inject_servers": frozenset(),
      "mcp_meta_inject_servers": frozenset(),
      "credentials_resolver_active": False,
      "fms_rebinder": None,
    },
    caplog,
  )

  assert interactive.model_dump() == autonomous.model_dump()
  assert interactive_logged == autonomous_logged == interactive.model_dump(
    mode="json"
  )
  assert interactive.research_binding.research_file_id == 42
  assert interactive.tool_grant == ("file_read", "web_search")
  assert interactive.children == ()
  assert interactive.tenant is not None
  assert interactive.tenant.storage_id == "actor-test"



def test_spawn_reads_approval_authority_after_runtime_setup(
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  runner = _BackgroundRunner()
  session = _session()
  session.approval_route = NO_APPROVAL_ROUTE
  handler = make_run_agent_handler(
    [runner],
    parent_session=session,
    mcp_client=_McpClient(),
    local_tool_handlers={"web_search": _tool, "file_read": _tool},
    capability_execution_resolver=_resolver(),
  )
  # Autonomous runtime configures its route after installing child handlers.
  store, policy = object(), object()
  session.approval_route = DurableLocalApprovalRoute(store, policy, session)
  result, error = asyncio.run(handler({
    "objective": "Collect",
    "background": True,
  }))
  assert error is None
  assert result is not None
  child = runner.background_calls[0]
  assert (
    child["admitted_task"].execution_snapshot.admitted_context.approval_route
    == "parent_delegated"
  )

# --- (b) narrowing ------------------------------------------------------


def test_operation_outside_parent_children_exceeds_parent_grant() -> None:
  result = admit(
    _request(),
    _root(children=("filing-review",), depth=1),
    owners=_owners(_PlainResolver()),
  )

  assert isinstance(result, Refusal)
  assert result.reason == "grant_exceeds_parent"


def test_child_grant_is_route_ceiling_within_parent_grant() -> None:
  result = admit(
    _request(),
    _root(tool_ids=frozenset({"web_search"}), children=("explore",), depth=1),
    owners=_owners(_PlainResolver()),
  )

  assert isinstance(result, AdmittedContext)
  assert result.wire.tool_grant == ("web_search",)
  assert result.wire.lineage.depth == 2


@pytest.mark.parametrize(
  "withheld_name",
  ["get_current_model", "mcp__portfolio-reads-mcp__get_current_model"],
)
@pytest.mark.parametrize("bounded_parent", [False, True])
def test_withheld_routes_narrow_private_and_inherited_authority(
  withheld_name: str,
  bounded_parent: bool,
) -> None:
  model_route = "mcp__portfolio-reads-mcp__get_current_model"
  owners = _owners(
    _PlainResolver(),
    ceiling=frozenset({model_route, "file_read", "private_read"}),
    operation_routes=frozenset({model_route, "private_read"}),
  )
  owners = replace(
    owners,
    operation=replace(
      owners.operation, route_tool_names={model_route: "get_current_model"},
    ),
  )
  result = admit(
    _request(),
    _root(
      withheld=frozenset({withheld_name, "file_read"}),
      tool_ids=frozenset({model_route, "file_read"}) if bounded_parent else None,
    ),
    owners=owners,
  )

  assert isinstance(result, AdmittedContext)
  # An unwithheld operation-private route need not be on the parent's surface.
  assert result.wire.tool_grant == ("private_read",)
  assert tuple(tool.tool_id for tool in result.tool_authority.grant.tools) == (
    "private_read",
  )


@pytest.mark.parametrize(
  ("private_route", "policy_name", "withheld_name"),
  [
    ("private_read", "private_read", "private_read"),
    (
      "mcp__portfolio-reads-mcp__get_current_model",
      "get_current_model",
      "get_current_model",
    ),
    (
      "mcp__portfolio-reads-mcp__get_current_model",
      "get_current_model",
      "mcp__portfolio-reads-mcp__get_current_model",
    ),
  ],
)
def test_withheld_routes_survive_two_hops_and_wire_round_trip(
  private_route: str,
  policy_name: str,
  withheld_name: str,
) -> None:
  owners = _owners(
    _PlainResolver(),
    ceiling=frozenset({"file_read", private_route}),
    operation_routes=frozenset({private_route}),
  )
  owners = replace(
    owners,
    operation=replace(
      owners.operation,
      route_tool_names={private_route: policy_name},
      children=("explore",),
    ),
  )
  first = admit(
    _request(),
    _root(tool_ids=frozenset({"file_read"}), withheld=frozenset({withheld_name})),
    owners=owners,
  )
  assert isinstance(first, AdmittedContext)
  assert first.wire.tool_grant == ("file_read",)
  assert first.wire.withheld_tools == (withheld_name,)
  restored = replace(
    first,
    wire=AdmittedContextWire.model_validate_json(first.wire.model_dump_json()),
  )
  child = context_for_child(restored, job_id="child")
  assert child.tool_ids == frozenset({"file_read"})
  assert child.withheld == frozenset({withheld_name})

  descendant_owners = _owners(
    _PlainResolver(),
    ceiling=frozenset({
      "file_read", private_route, "operation_read", "outside_read",
    }),
    operation_routes=frozenset({private_route, "operation_read"}),
  )
  descendant_owners = replace(
    descendant_owners,
    operation=replace(
      descendant_owners.operation,
      route_tool_names={private_route: policy_name},
      children=("explore",),
    ),
  )
  second = admit(_request(), child, owners=descendant_owners)
  assert isinstance(second, AdmittedContext)
  # Private routes may extend visible scope, but not bypass inherited exclusions.
  assert second.wire.tool_grant == ("file_read", "operation_read")
  assert tuple(tool.tool_id for tool in second.tool_authority.grant.tools) == (
    "file_read", "operation_read",
  )
  assert second.wire.withheld_tools == (withheld_name,)
  assert context_for_child(second, job_id="grandchild").withheld == child.withheld


def test_spawn_tool_leaves_a_grant_whose_children_are_empty() -> None:
  result = admit(
    _request(),
    _root(),
    owners=_owners(
      _PlainResolver(),
      ceiling=frozenset({"run_agent", "send_message", "web_search"}),
    ),
  )

  assert isinstance(result, AdmittedContext)
  assert result.wire.children == ()
  assert result.wire.tool_grant == ("send_message", "web_search")


def test_deep_spawn_after_budget_overrun_retains_admitted_authority() -> None:
  parent = _root(depth=7, remaining_usd=-1.25)
  result = admit(
    _request(),
    parent,
    owners=_owners(_PlainResolver()),
  )

  assert isinstance(result, AdmittedContext)
  assert result.wire.lineage.depth == 8
  assert result.wire.budget_share == -1.25
  assert result.wire.tool_grant == ("file_read", "web_search")
  assert result.approval_route is parent.approval_route


def test_child_session_cannot_regain_the_roots_spawn_authority() -> None:
  admitted = admit(_request(), _root(), owners=_owners(_PlainResolver()))
  assert isinstance(admitted, AdmittedContext)
  child = context_for_child(admitted, job_id="child-1")
  session = SimpleNamespace(
    session_id="child-session",
    owner_user_id="actor-test",
    user_id="actor-test",
    role="owner",
    tenant_id="tenant-test",
    admitted_job_context=child,
  )
  parent = root_context(
    session, withheld_tools=frozenset(), research_file_id=None,
  )
  result = admit(_request(), parent, owners=_owners(_PlainResolver()))

  assert isinstance(result, Refusal)
  assert result.reason == "grant_exceeds_parent"
  assert parent.job_id == "child-1"
  assert parent.depth == 1


def test_remaining_budget_is_the_share() -> None:
  result = admit(
    _request(),
    _root(remaining_usd=2.5),
    owners=_owners(_PlainResolver()),
  )

  assert isinstance(result, AdmittedContext)
  assert result.wire.budget_share == 2.5


def test_handles_outside_parent_grants_refuse_handle_not_granted() -> None:
  result = admit(
    _request(handles=("h:1",)),
    _root(),
    owners=_owners(_PlainResolver()),
  )

  assert isinstance(result, Refusal)
  assert result.reason == "handle_not_granted"


def test_binding_other_than_parents_refuses_binding_not_granted() -> None:
  refused = admit(
    _request(binding=AdmittedResearchBinding(research_file_id=43)),
    _root(research_binding=AdmittedResearchBinding(research_file_id=42)),
    owners=_owners(_PlainResolver()),
  )
  inherited = admit(
    _request(binding=AdmittedResearchBinding(ticker="AAPL")),
    _root(research_binding=AdmittedResearchBinding(research_file_id=42)),
    owners=_owners(_PlainResolver()),
  )

  assert isinstance(refused, Refusal)
  assert refused.reason == "binding_not_granted"
  assert isinstance(inherited, AdmittedContext)
  assert inherited.wire.research_binding == AdmittedResearchBinding(
    research_file_id=42,
    ticker="AAPL",
  )


# --- (c) no side effect -------------------------------------------------


def test_root_tenant_storage_id_is_the_minted_owner_not_the_channel_user_id() -> None:
  # Live run 2026-10-07 (docs/qa/host-admit-at-every-hop-2026-10-07.md): the interactive
  # door admitted storage_id "henry" and the autonomous door "1" for one user whose desk is
  # users/1. The storage id is the session's minted owner (risk id first), never user_id.
  session = SessionStore(ttl=3600).create_session(
    "hash",
    user_id="henry",
    risk_user_id=1,
    role="owner",
  )
  root = root_context(session, withheld_tools=frozenset(), research_file_id=None)
  assert root.tenant is not None
  assert root.tenant.storage_id == "1"


def test_admitted_grant_is_the_semantic_owners_executable_grant() -> None:
  result = admit(
    _request(),
    _root(),
    owners=_owners(
      _resolver(),
      ceiling=frozenset({"file_read", "web_search", "write_record"}),
    ),
  )

  assert isinstance(result, AdmittedContext)
  assert result.wire.tool_grant == ("file_read", "web_search")
  assert tuple(entry.tool_id for entry in result.tool_authority.grant.tools) == (
    "file_read", "web_search",
  )


def test_child_records_into_the_roots_captured_ledger_by_delegation() -> None:
  session = _session()
  route = DurableLocalApprovalRoute(object(), object(), session)
  session.approval_route = route
  parent = root_context(
    session, withheld_tools=frozenset(), research_file_id=None,
  )
  session.approval_route = NO_APPROVAL_ROUTE
  owners = _owners(_resolver())
  owners = replace(
    owners, operation=replace(owners.operation, children=("explore",)),
  )
  result = admit(_request(), parent, owners=owners)

  assert isinstance(result, AdmittedContext)
  assert isinstance(result.approval_route, InProcessDelegatedApprovalRoute)
  assert route_ledger(result.approval_route) is route
  assert route_session(result.approval_route) is session
  assert result.wire.approval_route == "parent_delegated"
  grandchild = admit(
    _request(), context_for_child(result, job_id="child"), owners=owners,
  )
  assert isinstance(grandchild, AdmittedContext)
  assert grandchild.approval_route is result.approval_route


def test_sessionless_root_children_carry_the_control_plane_route() -> None:
  root = _root()
  control = ControlPlaneApprovalRoute(object(), object())
  parent = root_context_from_identity(
    tenant=root.tenant, job_id="schedule:one", tool_ids=None,
    withheld_tools=frozenset(), research_binding=root.research_binding,
    approval_route=control,
  )
  result = admit(
    _request(), parent,
    owners=replace(_owners(_resolver()), placement="local_stub"),
  )
  assert isinstance(result, AdmittedContext)
  assert result.approval_route is control
  assert result.wire.approval_route == "parent_delegated"
  assert route_ledger(control) is None
  with pytest.raises(TypeError):
    route_session(control)


def test_admit_writes_nothing_to_its_inputs() -> None:
  resolver = _PlainResolver()
  owners = _owners(resolver)
  parent = _root()
  request = _request()
  before = (
    copy.deepcopy(request),
    copy.deepcopy(parent),
    copy.deepcopy(owners.operation),
  )

  first = admit(request, parent, owners=owners)
  second = admit(request, parent, owners=owners)

  assert (request, parent, owners.operation) == before
  assert isinstance(first, AdmittedContext)
  assert first == second
  assert resolver.reads == ["node.explore", "node.explore"]


# --- (d) unknown names are logged, never refused ------------------------


def test_unregistered_route_is_logged_and_admitted_without(
  caplog: pytest.LogCaptureFixture,
) -> None:
  with caplog.at_level(logging.WARNING, logger="agent_gateway.admission"):
    result = admit(
      _request(),
      _root(),
      owners=_owners(
        _PlainResolver(),
        ceiling=frozenset({"web_search", "ghost_tool"}),
        registered=frozenset({"web_search"}),
      ),
    )

  assert isinstance(result, AdmittedContext)
  assert result.wire.tool_grant == ("web_search",)
  assert any(
    "route_unregistered" in record.getMessage()
    and "ghost_tool" in record.getMessage()
    for record in caplog.records
  )


def test_env_name_outside_launch_contract_is_logged_and_omitted(
  caplog: pytest.LogCaptureFixture,
) -> None:
  with caplog.at_level(logging.WARNING, logger="agent_gateway.admission"):
    result = admit(
      _request(),
      _root(),
      owners=_owners(
        _PlainResolver(),
        env_contract=RUN_BASH_ENVIRONMENT,
        env_names=frozenset({"PATH", "GATEWAY_SECRET_TOKEN"}),
      ),
    )

  assert isinstance(result, AdmittedContext)
  assert result.wire.env_projection == ("PATH",)
  assert any(
    "env_name_unknown" in record.getMessage()
    and "GATEWAY_SECRET_TOKEN" in record.getMessage()
    for record in caplog.records
  )


def _continuation_task(
  caplog: pytest.LogCaptureFixture,
  *,
  legacy: bool = False,
  admitted_context: AdmittedContextWire | None = None,
) -> AdmittedTask:
  runner = _BackgroundRunner()
  wire, _ = _door_context(runner, {}, caplog)
  wire = admitted_context if admitted_context is not None else wire
  payload = runner.background_calls[0]["admitted_task"].model_dump(mode="json")
  if wire.children:
    payload["operation"]["children"] = list(wire.children)
  if legacy:
    payload["execution_snapshot"].pop("admitted_context")
  else:
    payload["execution_snapshot"]["admitted_context"] = wire.model_copy(update={
      "budget_share": -4.5,
      "lineage": wire.lineage.model_copy(update={"depth": 9}),
    }).model_dump(mode="json")
  payload.pop("admitted_task_digest")
  seal_admitted_task_payload(payload)
  return AdmittedTask.model_validate(payload)


@pytest.mark.parametrize(
  ("cause", "placement", "environment"),
  [
    ("continuation", "in_process", ()),
    ("placement_lost", "local_stub", ("PATH",)),
  ],
)
def test_continuation_relocates_without_rebinding_or_regranting(
  cause: Any,
  placement: str,
  environment: tuple[str, ...],
  caplog: pytest.LogCaptureFixture,
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  task = _continuation_task(caplog)
  assert task.execution_snapshot is not None
  original = task.execution_snapshot.admitted_context
  assert original is not None
  result = admit(
    ContinueRequest(task.attempt.physical_task_id, cause=cause),
    _root(tool_ids=frozenset({"write_record"}), remaining_usd=100.0),
    owners=ContinuationOwners(
      task=task,
      research_binding=original.research_binding,
      placement="local_stub",
      env_contract=RUN_BASH_ENVIRONMENT,
      env_names=frozenset({"PATH", "GATEWAY_SECRET_TOKEN"}),
    ),
  )

  assert isinstance(result, AdmittedContext)
  assert result.wire.placement == placement
  assert result.wire.env_projection == environment
  assert result.wire.budget_share == -4.5
  assert result.wire.lineage.depth == 9
  assert result.wire.capability_bind == task.model_bind
  authority = result.tool_authority
  assert authority.tool_routes is not None
  tool_ids, local_ids, mcp_scope = dispatcher_scopes_from_admitted_routes(
    authority.tool_grant, authority.tool_routes,
  )
  assert tool_ids == local_ids == frozenset({"file_read", "web_search"})
  assert mcp_scope == {}


@pytest.mark.parametrize("cause", ["continuation", "revive", "placement_lost"])
@pytest.mark.parametrize(
  ("private_route", "policy_name", "withheld_name"),
  [
    ("private_read", "private_read", "private_read"),
    (
      "mcp__portfolio-reads-mcp__get_current_model",
      "get_current_model",
      "get_current_model",
    ),
    (
      "mcp__portfolio-reads-mcp__get_current_model",
      "get_current_model",
      "mcp__portfolio-reads-mcp__get_current_model",
    ),
  ],
)
def test_continuation_preserves_exclusions_for_future_descendants(
  cause: Any,
  private_route: str,
  policy_name: str,
  withheld_name: str,
  caplog: pytest.LogCaptureFixture,
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  owners = _owners(
    _PlainResolver(),
    ceiling=frozenset({"file_read", "web_search", private_route}),
    operation_routes=frozenset({private_route}),
  )
  owners = replace(
    owners,
    operation=replace(
      owners.operation,
      route_tool_names={private_route: policy_name},
      children=("explore",),
    ),
  )
  admitted = admit(
    _request(),
    _root(
      tool_ids=frozenset({"file_read", "web_search"}),
      withheld=frozenset({withheld_name}),
      research_binding=AdmittedResearchBinding(research_file_id=42),
    ),
    owners=owners,
  )
  assert isinstance(admitted, AdmittedContext)
  task = _continuation_task(caplog, admitted_context=admitted.wire)
  before = task.model_dump(mode="json")
  result = admit(
    ContinueRequest(task.attempt.physical_task_id, cause=cause),
    _root(tool_ids=frozenset(), withheld=frozenset({"web_search"})),
    owners=ContinuationOwners(
      task=task,
      research_binding=admitted.wire.research_binding,
      placement="local_stub",
      env_contract=RUN_BASH_ENVIRONMENT,
      env_names=frozenset({"PATH"}),
    ),
  )

  assert isinstance(result, AdmittedContext)
  # Continuation retains its immutable executable grant, even with new exclusions.
  assert result.tool_authority is task
  assert result.wire.tool_grant == ("file_read", "web_search")
  assert task.model_dump(mode="json") == before
  assert result.wire.withheld_tools == tuple(sorted({withheld_name, "web_search"}))
  restored = replace(
    result,
    wire=AdmittedContextWire.model_validate_json(result.wire.model_dump_json()),
  )
  child = context_for_child(restored, job_id="continued-child")
  assert child.withheld == frozenset({withheld_name, "web_search"})
  descendant = admit(_request(), child, owners=owners)
  assert isinstance(descendant, AdmittedContext)
  assert descendant.wire.tool_grant == ("file_read",)
  assert tuple(tool.tool_id for tool in descendant.tool_authority.grant.tools) == (
    "file_read",
  )


def test_historical_wire_keeps_only_known_parent_exclusions(
  caplog: pytest.LogCaptureFixture,
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  payload = _continuation_task(caplog).model_dump(mode="json")
  historical_context = payload["execution_snapshot"]["admitted_context"]
  historical_context.pop("withheld_tools")
  historical_context["children"] = ["explore"]
  payload["operation"]["children"] = ["explore"]
  payload.pop("admitted_task_digest")
  seal_admitted_task_payload(payload)
  task = AdmittedTask.model_validate(payload)
  # Reading the absent field cannot rewrite a historical payload or its digest.
  assert task.model_dump(mode="json") == payload
  result = admit(
    ContinueRequest(task.attempt.physical_task_id, cause="revive"),
    _root(withheld=frozenset({"private_read"})),
    owners=ContinuationOwners(
      task=task,
      research_binding=AdmittedResearchBinding(research_file_id=42),
    ),
  )
  assert isinstance(result, AdmittedContext)
  assert result.tool_authority is task
  assert result.wire.withheld_tools == ("private_read",)
  child = context_for_child(result, job_id="historical-child")
  assert child.withheld == frozenset({"private_read"})
  descendant = admit(
    _request(),
    child,
    owners=_owners(
      _PlainResolver(),
      ceiling=frozenset({"file_read", "web_search", "private_read", "memory_read"}),
      operation_routes=frozenset({"private_read", "memory_read"}),
    ),
  )
  assert isinstance(descendant, AdmittedContext)
  # The discarded historical memory_read exclusion is unknown, not inferred
  # from its absence in the old visible grant.
  assert descendant.wire.tool_grant == ("file_read", "memory_read", "web_search")


def test_legacy_continuation_keeps_durable_authority_without_context_projection(
  caplog: pytest.LogCaptureFixture,
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  task = _continuation_task(caplog, legacy=True)
  result = admit(
    ContinueRequest(task.attempt.physical_task_id, cause="revive"),
    _root(
      tool_ids=frozenset(),
      withheld=frozenset({"private_read"}),
      remaining_usd=-10.0,
    ),
    owners=ContinuationOwners(
      task=task,
      research_binding=AdmittedResearchBinding(research_file_id=42),
    ),
  )

  assert isinstance(result, AdmittedContext)
  assert result.wire.budget_share is None
  assert result.wire.research_binding.research_file_id == 42
  assert result.wire.capability_bind == task.model_bind
  assert result.wire.withheld_tools == ("private_read",)
  assert context_for_child(result, job_id="legacy-child").withheld == frozenset({
    "private_read",
  })
  authority = result.tool_authority
  assert authority.tool_routes is not None
  tool_ids, local_ids, _ = dispatcher_scopes_from_admitted_routes(
    authority.tool_grant, authority.tool_routes,
  )
  assert tool_ids == local_ids == frozenset({"file_read", "web_search"})


def test_continuation_cannot_adopt_another_tenants_job(
  caplog: pytest.LogCaptureFixture,
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  task = _continuation_task(caplog)
  parent = _root()
  assert parent.tenant is not None
  result = admit(
    ContinueRequest(task.attempt.physical_task_id),
    replace(parent, tenant=parent.tenant.model_copy(update={"tenant_id": "foreign"})),
    owners=ContinuationOwners(
      task=task,
      research_binding=AdmittedResearchBinding(research_file_id=42),
    ),
  )

  assert isinstance(result, Refusal)
  assert result.reason == "tenant_mismatch"


def test_root_turn_refreshes_but_reads_keep_the_captured_authority() -> None:
  session = _session()
  first = root_context(
    session, withheld_tools=frozenset({"write_record"}), research_file_id=41,
  )
  session.role = "invite"
  assert current_parent_context(session) is first
  second = root_context(
    session, withheld_tools=frozenset({"run_agent"}), research_file_id=42,
  )
  assert current_parent_context(session) is second
  assert second.tenant is not None and second.tenant.role == "invite"
  assert second.research_binding.research_file_id == 42
  assert second.withheld == frozenset({"run_agent"})


def test_identity_root_cannot_grant_outside_live_roster_or_withheld_tools() -> None:
  root = _root()
  parent = root_context_from_identity(
    tenant=root.tenant,
    job_id="schedule:one",
    tool_ids=frozenset({"file_read", "write_record"}),
    withheld_tools=frozenset({"write_record"}),
    research_binding=root.research_binding,
    approval_route=root.approval_route,
  )
  result = admit(
    _request(), parent,
    owners=_owners(
      _resolver(), ceiling=frozenset({"file_read", "write_record", "web_search"}),
    ),
  )
  assert isinstance(result, AdmittedContext)
  assert result.wire.tool_grant == ("file_read",)
  assert result.wire.lineage.parent_job_id == "schedule:one"


def test_frozen_bind_is_authorized_without_reselection(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  resolver = _resolver()
  frozen = resolver.select_bind("node.explore")

  def reject_selection(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a frozen workflow node must not select another bind")

  monkeypatch.setattr(type(resolver), "select_bind", reject_selection)
  result = admit(
    _request(), _root(),
    owners=replace(_owners(resolver), frozen_bind=frozen),
  )
  assert isinstance(result, AdmittedContext)
  assert result.wire.capability_bind is frozen


def test_declared_child_can_spawn_without_gaining_undeclared_operations() -> None:
  owners = _owners(_resolver(), ceiling=frozenset({"run_agent", "file_read"}))
  owners = replace(
    owners, operation=replace(owners.operation, children=("explore",)),
  )
  admitted = admit(_request(), _root(), owners=owners)
  assert isinstance(admitted, AdmittedContext)
  parent = context_for_child(admitted, job_id="lead")
  worker = admit(_request(), parent, owners=owners)
  refused = admit(_request(operation="mutate"), parent, owners=owners)
  assert isinstance(worker, AdmittedContext)
  assert "run_agent" in worker.wire.tool_grant
  assert worker.wire.lineage.parent_job_id == "lead"
  assert isinstance(refused, Refusal) and refused.reason == "grant_exceeds_parent"


def test_late_output_can_be_read_and_explicitly_passed_without_ambient_access() -> None:
  parent = _root(job_id="root")
  outputs = frozenset({"child:result", "child:private"})
  read = admit(
    HandleRequest("child:result", "root"), parent,
    owners=HandleOwners(outputs=outputs),
  )
  assert isinstance(read, ParentContext) and read.job_id == parent.job_id
  child = admit(
    _request(handles=("child:result",)), parent,
    owners=replace(_owners(_resolver()), outputs=outputs),
  )
  assert isinstance(child, AdmittedContext)
  reader = context_for_child(child, job_id="sibling")
  allowed = admit(
    HandleRequest("child:result", "sibling"), reader, owners=HandleOwners(),
  )
  refused = admit(
    HandleRequest("child:private", "sibling"), reader, owners=HandleOwners(),
  )
  assert isinstance(allowed, ParentContext)
  assert isinstance(refused, Refusal) and refused.reason == "handle_not_granted"
  assert parent.handle_grants == frozenset()


def test_message_reference_does_not_authorize_recipient_to_read() -> None:
  sender = _root(job_id="sender", handle_grants=frozenset({"h:one"}))
  recipient = _root(job_id="recipient")
  sent = admit(
    MessageRequest("sender", "recipient", "See this result", ("h:one",)),
    sender, owners=MessageOwners(recipient_tenant=recipient.tenant),
  )
  read = admit(
    HandleRequest("h:one", "recipient"), recipient, owners=HandleOwners(),
  )
  assert sent is sender
  assert isinstance(read, Refusal) and read.reason == "handle_not_granted"


@pytest.mark.parametrize("foreign_tenant", [False, True])
def test_message_requires_same_authenticated_tenant(foreign_tenant: bool) -> None:
  sender = _root(job_id="sender")
  assert sender.tenant is not None
  recipient_tenant = (
    sender.tenant.model_copy(update={"storage_id": "another-owner"})
    if foreign_tenant else None
  )
  result = admit(
    MessageRequest("sender", "recipient", "hello"), sender,
    owners=MessageOwners(recipient_tenant=recipient_tenant),
  )
  assert isinstance(result, Refusal)
  assert result.reason == (
    "tenant_mismatch" if foreign_tenant else "session_unauthenticated"
  )


def test_message_cannot_cite_an_ungranted_handle() -> None:
  sender = _root(job_id="sender")
  result = admit(
    MessageRequest("sender", "recipient", "hello", ("h:foreign",)), sender,
    owners=MessageOwners(
      recipient_tenant=sender.tenant, outputs=frozenset({"h:own"}),
    ),
  )
  assert isinstance(result, Refusal) and result.reason == "handle_not_granted"
