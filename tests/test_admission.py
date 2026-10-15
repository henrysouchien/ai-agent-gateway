from __future__ import annotations

import asyncio
import copy
import json
import logging
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_workflow_contracts import (
  AdmittedTask,
  CatalogToolEntry,
  PlatformToolCatalog,
  SemanticCapabilityRequirement,
  sha256_digest,
)
from agent_workflow_contracts.models import (
  AdmittedContextWire,
  AdmittedResearchBinding,
)
from agent_gateway.admission import (
  AdmittedChildIdentity,
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
  admit_root_turn,
  capture_root_turn,
  current_root_turn,
  link_root_park,
  record_root_turn_admission,
  capture_job_context,
  context_for_child,
  current_parent_context,
  root_context,
  root_context_from_identity,
)
from agent_gateway.child_environment import RUN_BASH_ENVIRONMENT
from agent_gateway.containment_settings import ContainmentSettings
from agent_gateway.approval_route import (
  NO_APPROVAL_ROUTE,
  ControlPlaneApprovalRoute,
  DurableLocalApprovalRoute,
  InProcessDelegatedApprovalRoute,
  route_ledger,
  route_session,
)
from agent_gateway.approval_policy import RunContext, build_approval_request, sha256_args
from agent_gateway.approval_store import RootTurnRecordError, SQLiteApprovalStore
from agent_gateway.single_user_policy import SingleUserApprovalPolicy
from agent_gateway.tool_policy_registry import PreparedToolCall
from agent_gateway.capability_resolution import OperationDeclaration, resolve_operation_authority
from agent_gateway.session import GatewaySession, SessionStore
from agent_gateway.sub_agent import make_run_agent_handler, seal_admitted_task_payload
from agent_gateway.sub_agent_scope_receipt import dispatcher_scopes_from_admitted_routes
from gateway_test_support.capability_execution_test_support import (
  stub_capability_execution_resolver,
)
from gateway_test_support.host_policy import owner_session_host_policy
from model_authority.selection import SelectionSnapshot, capture_selection_snapshot


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
    self.selections: list[tuple[str | None, str | None]] = []

  def select_bind(
    self, capability_id: str, *, role: str | None = None, effort: str | None = None,
    selection_snapshot: SelectionSnapshot | None = None,
  ) -> Any:
    self.reads.append(capability_id)
    self.selections.append((role, effort))
    return _BIND


def _owners(
  resolver: Any,
  *,
  ceiling: frozenset[str] = frozenset({"web_search", "file_read"}),
  operation_routes: frozenset[str] = frozenset(),
  registered: frozenset[str] | None = None,
  env_contract: str | None = None,
  env_names: frozenset[str] = frozenset(),
  children: tuple[str, ...] = (),
  containment: ContainmentSettings = ContainmentSettings(),
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
      children=children,
    ),
    capability_execution_resolver=resolver,
    session=_session(),
    route_registered=lambda tool_id: tool_id in known,
    placement="in_process",
    containment=containment,
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


def test_historical_wire_omits_absent_selection_snapshot() -> None:
  admitted = admit(_request(), _root(), owners=_owners(_resolver()))
  assert isinstance(admitted, AdmittedContext)
  payload = admitted.wire.model_dump(mode="json", exclude_none=False)
  assert "selection_snapshot" not in payload
  restored = AdmittedContextWire.model_validate(payload)
  assert context_for_child(
    replace(admitted, wire=restored), job_id="historical-child",
  ).selection_snapshot is None


def test_snapshot_survives_two_fresh_hops_and_wire_round_trip(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  resolver = _resolver()
  snapshot = capture_selection_snapshot(resolver.selection_policy)
  selected: list[SelectionSnapshot | None] = []
  real_select = type(resolver).select_bind

  def select(self: Any, *args: Any, **kwargs: Any) -> Any:
    selected.append(kwargs.get("selection_snapshot"))
    return real_select(self, *args, **kwargs)

  monkeypatch.setattr(type(resolver), "select_bind", select)
  first = admit(
    _request(), _root(selection_snapshot=snapshot),
    owners=_owners(resolver, children=("explore",)),
  )
  assert isinstance(first, AdmittedContext)
  first_wire = AdmittedContextWire.model_validate_json(first.wire.model_dump_json())
  assert first_wire.selection_snapshot == snapshot.to_json()
  child = context_for_child(replace(first, wire=first_wire), job_id="child")
  assert child.selection_snapshot == snapshot
  second = admit(_request(), child, owners=_owners(resolver))
  assert isinstance(second, AdmittedContext)
  assert second.wire.selection_snapshot == snapshot.to_json()
  assert selected == [snapshot, snapshot]
  assert first.wire.capability_bind == second.wire.capability_bind


def test_frozen_bind_does_not_select_under_snapshot(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  resolver = _resolver()
  snapshot = capture_selection_snapshot(resolver.selection_policy)

  def unexpected_selection(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("frozen admission must not reselect")

  monkeypatch.setattr(type(resolver), "select_bind", unexpected_selection)
  admitted = admit(
    _request(), _root(selection_snapshot=snapshot),
    owners=replace(_owners(resolver), frozen_bind=_BIND),
  )
  assert isinstance(admitted, AdmittedContext)
  assert admitted.wire.capability_bind is _BIND
  assert admitted.wire.selection_snapshot == snapshot.to_json()


def test_spawn_keeps_the_turns_snapshot_and_a_later_mint_does_not_inherit_it() -> None:
  # Live roles-W3 acceptance at 9e1b7bdec630: the spawn handler re-mints the root
  # and the admitted wire dropped the turn's captured snapshot.
  resolver = _resolver()
  session = _session()
  turn = capture_selection_snapshot(resolver.selection_policy)
  root_context(
    session, withheld_tools=frozenset({"memory_read"}), research_file_id=42,
    selection_snapshot=turn,
  )
  runner = _BackgroundRunner()
  handler = make_run_agent_handler(
    [runner],
    parent_session=session,
    mcp_client=_McpClient(),
    local_tool_handlers={"web_search": _tool, "file_read": _tool, "memory_read": _tool},
    excluded_tools={"memory_read"},
    user_id="actor-test",
    trusted_research_file_id=42,
    capability_execution_resolver=resolver,
  )
  _result, error = asyncio.run(handler({"objective": "Collect", "background": True}))
  assert error is None, error
  admitted = runner.background_calls[0]["admitted_task"].execution_snapshot.admitted_context
  assert admitted is not None
  assert admitted.selection_snapshot == turn.to_json()
  # A root minted later without its own capture (schedule run-now, batch stage)
  # resolves under current policy, not this turn's snapshot.
  later = root_context(session, withheld_tools=frozenset(), research_file_id=None)
  assert later.selection_snapshot is None


# --- (a) one context for both doors -------------------------------------


class _BackgroundRunner:
  _containment = ContainmentSettings()

  def __init__(self) -> None:
    self.capability_execution = _resolver().resolve("session.driver")
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


@pytest.mark.parametrize("remaining_usd", [None, 0.0, -1.25])
def test_spawn_after_zero_or_overrun_spend_retains_admitted_authority(
  remaining_usd: float | None,
) -> None:
  parent = _root(depth=1, remaining_usd=remaining_usd)
  result = admit(_request(), parent, owners=_owners(_PlainResolver()))

  assert isinstance(result, AdmittedContext)
  assert result.wire.lineage.depth == 2
  assert result.wire.budget_share == remaining_usd
  assert result.wire.tool_grant == ("file_read", "web_search")
  assert result.approval_route is parent.approval_route


@pytest.mark.parametrize("maximum", [0, 1, 3])
def test_configured_spawn_depth_omits_only_the_spawn_tool_at_the_boundary(
  maximum: int,
) -> None:
  owners = _owners(
    _PlainResolver(),
    ceiling=frozenset({"file_read", "run_agent"}),
    children=("explore",),
    containment=ContainmentSettings(max_spawn_depth=maximum),
  )
  parent = _root()
  for depth in range(1, maximum + 1):
    result = admit(_request(), parent, owners=owners)
    assert isinstance(result, AdmittedContext)
    assert result.wire.lineage.depth == depth
    assert ("run_agent" in result.wire.tool_grant) == (depth < maximum)
    assert "file_read" in result.wire.tool_grant
    assert tuple(entry.tool_id for entry in result.tool_authority.grant.tools) == (
      result.wire.tool_grant
    )
    parent = context_for_child(result, job_id=f"job-{depth}")

  refused = admit(_request(), parent, owners=owners)
  assert isinstance(refused, Refusal)
  assert refused.reason == "depth_exceeded"
  assert refused.classification == "resource"
  assert refused.next_call == (
    "Return an ordinary final response with your partial work to your "
    "parent; the parent can spawn the remaining work."
  )


@pytest.mark.parametrize("maximum", [-1, -7])
@pytest.mark.parametrize("parent_depth", [0, 2, 7])
def test_negative_spawn_depth_limit_disables_refusal_and_tool_omission(
  maximum: int,
  parent_depth: int,
) -> None:
  result = admit(
    _request(),
    _root(depth=parent_depth, remaining_usd=-1.25),
    owners=_owners(
      _PlainResolver(),
      ceiling=frozenset({"file_read", "run_agent"}),
      children=("explore",),
      containment=ContainmentSettings(max_spawn_depth=maximum),
    ),
  )
  assert isinstance(result, AdmittedContext)
  assert result.wire.lineage.depth == parent_depth + 1
  assert result.wire.budget_share == -1.25
  assert result.wire.tool_grant == ("file_read", "run_agent")


@pytest.mark.parametrize("parent_depth", [2, 7])
@pytest.mark.parametrize("children", [(), ("explore",)])
def test_direct_or_stale_spawn_at_or_past_default_depth_is_a_resource_refusal(
  parent_depth: int,
  children: tuple[str, ...],
) -> None:
  resolver = _PlainResolver()
  result = admit(
    _request(),
    _root(depth=parent_depth, children=children, remaining_usd=0.0),
    owners=_owners(resolver),
  )
  assert isinstance(result, Refusal)
  assert result.reason == "depth_exceeded"
  assert result.owner == "admission spawn containment"
  assert result.classification == "resource"
  assert result.next_call is not None
  assert "ordinary final response" in result.next_call
  assert "parent can spawn" in result.next_call
  assert resolver.reads == []


def test_spawn_boundary_preserves_other_grants_and_does_not_mutate_inputs() -> None:
  resolver = _PlainResolver()
  ceiling = frozenset({
    "file_read", "run_agent", "send_message", "private_read",
    "outside_read", "web_search", "missing_read",
  })
  owners = _owners(
    resolver,
    ceiling=ceiling,
    operation_routes=frozenset({"private_read"}),
    registered=ceiling - {"missing_read"},
    children=("explore",),
  )
  parent = _root(
    depth=1,
    tool_ids=frozenset({"file_read", "run_agent", "send_message", "web_search"}),
    withheld=frozenset({"web_search"}),
    handle_grants=frozenset({"h:1"}),
  )
  before = copy.deepcopy(parent)
  request = _request(handles=("h:1",))
  bounded = admit(request, parent, owners=owners)
  unbounded = admit(
    request, parent,
    owners=replace(owners, containment=ContainmentSettings(max_spawn_depth=-1)),
  )

  assert isinstance(bounded, AdmittedContext)
  assert isinstance(unbounded, AdmittedContext)
  assert bounded.wire.tool_grant == ("file_read", "private_read", "send_message")
  assert tuple(entry.tool_id for entry in bounded.tool_authority.grant.tools) == (
    bounded.wire.tool_grant
  )
  assert unbounded.wire.tool_grant == (
    "file_read", "private_read", "run_agent", "send_message",
  )
  assert bounded.wire.model_copy(update={
    "tool_grant": unbounded.wire.tool_grant,
  }) == unbounded.wire
  assert bounded.wire.children == ("explore",)
  assert bounded.wire.handle_grants == ("h:1",)
  assert parent == before
  assert owners.operation.route_ceiling is ceiling


def test_spawn_adapter_preserves_depth_refusal_and_ordinary_final_recovery(
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  runner = _BackgroundRunner()
  session = _session()
  capture_job_context(session, _root(
    depth=2, job_id="worker", children=("explore",),
    tool_ids=frozenset({"file_read", "web_search"}),
  ))
  handler = make_run_agent_handler(
    [runner],
    parent_session=session,
    mcp_client=_McpClient(),
    local_tool_handlers={"web_search": _tool, "file_read": _tool},
    capability_execution_resolver=_resolver(),
  )
  result, error = asyncio.run(handler({"objective": "Collect", "background": True}))
  assert result is None
  assert error == {
    "code": "depth_exceeded",
    "message": "Spawn depth 2 has reached the configured maximum 2.",
    "owner": "admission spawn containment",
    "classification": "resource",
    "next_call": (
      "Return an ordinary final response with your partial work to your "
      "parent; the parent can spawn the remaining work."
    ),
  }
  assert runner.background_calls == []


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


@pytest.mark.parametrize(
  ("spawn", "operation", "selected"),
  [
    ((None, None), (None, None), (None, None)),
    ((None, None), ("research", "medium"), ("research", "medium")),
    (("drafting", "high"), ("research", "medium"), ("drafting", "high")),
    (("drafting", None), ("research", "medium"), ("drafting", "medium")),
  ],
)
def test_spawn_role_and_effort_win_over_the_operations_then_the_slot(
  spawn: tuple[str | None, str | None],
  operation: tuple[str | None, str | None],
  selected: tuple[str | None, str | None],
) -> None:
  resolver = _PlainResolver()
  owners = _owners(resolver)
  owners = replace(
    owners, operation=replace(owners.operation, role=operation[0], effort=operation[1]),
  )
  admit(_request(role=spawn[0], effort=spawn[1]), _root(), owners=owners)
  assert resolver.reads == ["node.explore"]
  assert resolver.selections == [selected]


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


def test_descendant_output_handles_are_admissible_on_spawn() -> None:
  handle = "sha256:" + "a" * 64
  owners = replace(_owners(_PlainResolver()), outputs=frozenset({handle}))
  admitted = admit(_request(handles=(handle,)), _root(), owners=owners)
  assert isinstance(admitted, AdmittedContext)
  assert admitted.wire.handle_grants == (handle,)
  dropped = admit(
    _request(handles=(handle,)),
    _root(),
    owners=replace(owners, outputs=frozenset()),
  )
  assert isinstance(dropped, Refusal)
  assert dropped.reason == "handle_not_granted"



def test_binding_other_than_parents_refuses_binding_not_granted() -> None:
  refused = admit(
    _request(binding=AdmittedResearchBinding(research_file_id=43)),
    _root(research_binding=AdmittedResearchBinding(research_file_id=42)),
    owners=_owners(_PlainResolver()),
  )
  assert isinstance(refused, Refusal)
  assert refused.reason == "binding_not_granted"


def test_ticker_only_child_under_research_bound_parent_refuses_binding_ambiguous() -> None:
  # Interim (b) under XIB-binding ruling: ticker-only child under a research-bound parent
  # is refused binding_ambiguous rather than inheriting the parent's research file.
  refused = admit(
    _request(binding=AdmittedResearchBinding(ticker="AAPL")),
    _root(research_binding=AdmittedResearchBinding(research_file_id=42)),
    owners=_owners(_PlainResolver()),
  )
  assert isinstance(refused, Refusal)
  assert refused.reason == "binding_ambiguous"
  assert "ticker-only child delegation requested" in refused.detail
  assert refused.owner == "admission research_binding"

  # Under an unbound parent, a ticker-only request is admitted with research_file_id=None.
  unbound_parent_child = admit(
    _request(binding=AdmittedResearchBinding(ticker="AAPL")),
    _root(research_binding=AdmittedResearchBinding(research_file_id=None)),
    owners=_owners(_PlainResolver()),
  )
  assert isinstance(unbound_parent_child, AdmittedContext)
  assert unbound_parent_child.wire.research_binding == AdmittedResearchBinding(
    research_file_id=None,
    ticker="AAPL",
  )

  # Under a bound parent, providing the matching research_file_id is admitted.
  matching_child = admit(
    _request(binding=AdmittedResearchBinding(research_file_id=42, ticker="AAPL")),
    _root(research_binding=AdmittedResearchBinding(research_file_id=42)),
    owners=_owners(_PlainResolver()),
  )
  assert isinstance(matching_child, AdmittedContext)
  assert matching_child.wire.research_binding == AdmittedResearchBinding(
    research_file_id=42,
    ticker="AAPL",
  )

  # When the child requests no binding at all, it inherits the parent binding.
  no_binding_child = admit(
    _request(binding=None),
    _root(research_binding=AdmittedResearchBinding(research_file_id=42)),
    owners=_owners(_PlainResolver()),
  )
  assert isinstance(no_binding_child, AdmittedContext)
  assert no_binding_child.wire.research_binding == AdmittedResearchBinding(
    research_file_id=42,
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


def _child_identity(**changes: Any) -> AdmittedChildIdentity:
  return replace(
    AdmittedChildIdentity(
      user_id="actor-test",
      raw_user_id="actor-raw",
      user_email="actor@example.com",
      user_slug="actor",
      risk_user_id=7,
      user_aliases=("actor-test", "actor-raw", "actor"),
      identity_status="canonical",
      role="owner",
    ),
    **changes,
  )


def test_spawn_returns_the_launch_identity_bound_to_the_parent_tenant() -> None:
  identity = _child_identity()
  admitted = admit(
    _request(),
    _root(),
    owners=replace(_owners(_resolver()), identity=identity),
  )

  assert isinstance(admitted, AdmittedContext)
  assert admitted.identity is identity
  assert "identity" not in admitted.wire.model_dump(mode="json")


@pytest.mark.parametrize(
  ("changes", "parent_changes"),
  [
    ({"user_id": "someone-else"}, {}),
    ({"role": "invite"}, {}),
    ({}, {"tenant": None}),
  ],
)
def test_spawn_refuses_an_identity_outside_the_parent_tenant(
  changes: dict[str, Any],
  parent_changes: dict[str, Any],
) -> None:
  refused = admit(
    _request(),
    _root(**parent_changes),
    owners=replace(_owners(_resolver()), identity=_child_identity(**changes)),
  )

  assert isinstance(refused, Refusal)
  assert refused.reason == "tenant_mismatch"
  assert refused.owner == "admission tenant"


@pytest.mark.parametrize("role", ["Owner", "", "admin"])
def test_spawn_refuses_an_inexact_identity_role(role: str) -> None:
  with pytest.raises(ValueError, match="role must be exactly"):
    admit(
      _request(),
      _root(),
      owners=replace(_owners(_resolver()), identity=_child_identity(role=role)),
    )


def test_spawn_without_a_launch_identity_admits_none() -> None:
  admitted = admit(_request(), _root(), owners=_owners(_resolver()))

  assert isinstance(admitted, AdmittedContext)
  assert admitted.identity is None



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


def test_admit_defaults_selection_provenance_role_when_unspecified() -> None:
  resolver = _PlainResolver()
  owners = _owners(resolver)
  parent = _root()
  request = _request(role=None)
  admitted = admit(request, parent, owners=owners)
  assert isinstance(admitted, AdmittedContext)
  assert admitted.wire.selection_provenance is not None
  assert admitted.wire.selection_provenance.role == "default"

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
  depth: int | None = None,
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
      "lineage": (
        wire.lineage.model_copy(update={"depth": depth})
        if depth is not None
        else wire.lineage
      ),
    }).model_dump(mode="json")
  payload.pop("admitted_task_digest")
  seal_admitted_task_payload(payload)
  return AdmittedTask.model_validate(payload)


@pytest.mark.parametrize(
  ("cause", "placement", "environment"),
  [
    ("continuation", "in_process", ()),
    ("revive", "in_process", ()),
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
  task = _continuation_task(caplog, depth=9)
  assert task.execution_snapshot is not None
  original = task.execution_snapshot.admitted_context
  assert original is not None
  result = admit(
    ContinueRequest(task.attempt.physical_task_id, cause=cause),
    _root(depth=7, tool_ids=frozenset({"write_record"}), remaining_usd=0.0),
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
  assert result.wire.lineage == original.lineage
  assert result.wire.tool_grant == original.tool_grant
  assert result.tool_authority is task
  assert result.wire.capability_bind == task.model_bind
  authority = result.tool_authority
  assert authority.tool_routes is not None
  tool_ids, local_ids, mcp_scope = dispatcher_scopes_from_admitted_routes(
    authority.tool_grant, authority.tool_routes,
  )
  assert tool_ids == local_ids == frozenset({"file_read", "web_search"})
  assert mcp_scope == {}


@pytest.mark.parametrize("cause", ["continuation", "revive", "placement_lost"])
def test_continuation_keeps_frozen_spawn_grant_after_depth_limit_changes(
  cause: Any,
  caplog: pytest.LogCaptureFixture,
  owner_session_host_policy: Any,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  admitted = admit(
    _request(),
    _root(research_binding=AdmittedResearchBinding(research_file_id=42)),
    owners=_owners(
      _PlainResolver(),
      ceiling=frozenset({"file_read", "run_agent", "web_search"}),
      children=("explore",),
    ),
  )
  assert isinstance(admitted, AdmittedContext)
  authority = admitted.tool_authority
  payload = _continuation_task(caplog, depth=9).model_dump(mode="json")
  # Freeze the semantic owner's grant, routes and definitions as one task.
  payload["operation"]["children"] = list(admitted.wire.children)
  payload["tool_grant"] = authority.grant.model_dump(mode="json")
  payload["tool_grant_digest"] = authority.grant.digest
  payload["capability_bindings"] = [
    binding.model_dump(mode="json") for binding in authority.bindings
  ]
  payload["capability_binding_digest"] = sha256_digest(payload["capability_bindings"])
  payload["tool_routes"] = [
    {"tool_id": route.tool_id, "origin": route.origin, "server_id": route.server_id}
    for route in authority.physical_routes
  ]
  snapshot = payload["execution_snapshot"]
  snapshot["provider_tool_definitions"] = [
    {"definition": {
      "name": entry.tool_id,
      "description": entry.tool_id,
      "input_schema": {"type": "object"},
    }}
    for entry in authority.grant.tools
  ]
  snapshot["admitted_context"] = admitted.wire.model_copy(update={
    "lineage": admitted.wire.lineage.model_copy(update={"depth": 9}),
    "budget_share": -4.5,
  }).model_dump(mode="json")
  payload.pop("admitted_task_digest")
  seal_admitted_task_payload(payload)
  task = AdmittedTask.model_validate(payload)
  before = task.model_dump(mode="json")
  result = admit(
    ContinueRequest(task.attempt.physical_task_id, cause=cause),
    _root(depth=7, tool_ids=frozenset(), remaining_usd=0.0),
    owners=ContinuationOwners(
      task=task,
      research_binding=admitted.wire.research_binding,
    ),
  )
  assert isinstance(result, AdmittedContext)
  assert result.tool_authority is task
  assert result.wire.tool_grant == ("file_read", "run_agent", "web_search")
  assert result.wire.lineage.depth == 9
  assert result.wire.lineage.parent_job_id == admitted.wire.lineage.parent_job_id
  assert result.wire.budget_share == -4.5
  assert task.model_dump(mode="json") == before


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


def test_default_root_lead_worker_bound_keeps_declared_child_authority() -> None:
  owners = _owners(
    _resolver(),
    ceiling=frozenset({"run_agent", "file_read"}),
    children=("explore",),
  )
  root = _root()
  assert root.depth == 0
  lead = admit(_request(), root, owners=owners)
  assert isinstance(lead, AdmittedContext)
  assert lead.wire.lineage.depth == 1
  assert lead.wire.tool_grant == ("file_read", "run_agent")
  parent = context_for_child(lead, job_id="lead")
  worker = admit(_request(), parent, owners=owners)
  refused = admit(_request(operation="mutate"), parent, owners=owners)
  assert isinstance(worker, AdmittedContext)
  assert worker.wire.lineage.depth == 2
  assert worker.wire.lineage.parent_job_id == "lead"
  assert worker.wire.tool_grant == ("file_read",)
  assert tuple(entry.tool_id for entry in worker.tool_authority.grant.tools) == (
    "file_read",
  )
  worker_context = context_for_child(worker, job_id="worker")
  assert worker_context.tool_ids == frozenset({"file_read"})
  assert worker.wire.children == ("explore",)
  assert isinstance(refused, Refusal) and refused.reason == "grant_exceeds_parent"
  stale_spawn = admit(_request(), worker_context, owners=owners)
  assert isinstance(stale_spawn, Refusal)
  assert stale_spawn.reason == "depth_exceeded"


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


def _root_execution(credential_ref: str | None = None, *, run_mode: str = "interactive"):
  execution = stub_capability_execution_resolver(
    default_provider="openai", default_model="gpt-5.6-sol",
    default_effort="high", run_mode=run_mode,
  ).resolve("session.driver")
  if credential_ref is not None:
    execution = replace(
      execution, bind=execution.bind.model_copy(update={"credential_ref": credential_ref}),
    )
  return execution


@pytest.mark.parametrize("credential_ref", ["service:openai", "anthropic-generation-test"])
def test_root_turn_admission_preserves_exact_owner_facts_and_mints_once(
  credential_ref: str,
) -> None:
  session = _session()
  snapshot = capture_selection_snapshot(_resolver().selection_policy)
  context = root_context(
    session, withheld_tools=frozenset({"write_record"}),
    research_file_id=42, selection_snapshot=snapshot,
  )
  execution = _root_execution(credential_ref)
  arguments: dict[str, Any] = {
    "execution": execution,
    "tool_grant": {"tool_ids": context.tool_ids, "withheld_tools": sorted(context.withheld)},
    "approval_route": NO_APPROVAL_ROUTE,
  }
  first = admit_root_turn(context, **arguments)
  second = admit_root_turn(context, **arguments)
  assert first.job_id == context.job_id
  assert first.tenant is context.tenant
  assert first.capability_bind is execution.bind
  assert first.research_binding is context.research_binding
  assert first.selection_snapshot is snapshot
  assert first.tool_grant == arguments["tool_grant"]
  assert first.approval_route_kind == "no_approval_route"
  assert first.credential_generation_ref == (
    credential_ref if credential_ref.startswith("anthropic-generation-") else None
  )
  assert first.turn_id.startswith("turn:")
  assert first.turn_id != second.turn_id
  assert first.policy_generation_ref is None
  assert current_root_turn(session) is None
  capture_root_turn(session, first)
  assert current_root_turn(session) is first
  capture_root_turn(session, second)
  assert current_root_turn(session) is second


async def _root_approval(
  store: SQLiteApprovalStore,
  *,
  session: GatewaySession,
  with_binding: bool,
) -> tuple[str, str, str]:
  assert session.owner_user_id is not None
  prepared = PreparedToolCall(prepared_input={"path": "retained.txt"}, exact_backend="local")
  call_id, nonce = "root-call", "root-nonce"
  request = build_approval_request(
    tool_call_id=call_id, tool_name="file_read", tool_class="read",
    tool_args_redacted=prepared.materialize_input(),
    args_hash=sha256_args(prepared.materialize_input()),
    run_context=RunContext(
      user_id=session.owner_user_id, session_id=session.session_id,
      request_id="root-request", run_id="root-run",
    ),
    state="pending_user",
  )
  await store.create(request)
  if with_binding:
    await store.record_executable_binding(
      approval_id=request.approval_id, job_id=session.session_id,
      tool_call_id=call_id, nonce=nonce, tenant_storage_id=session.owner_user_id,
      tool_name="file_read", prepared_call=prepared, dispatch_context={},
      credential_generation_ref=None,
    )
  return request.approval_id, call_id, nonce


@pytest.mark.parametrize(
  "with_binding",
  [
    pytest.param(True, id="autonomous-parks-always-carry-executable-binding"),
    pytest.param(False, id="interactive-sdk-park-without-executable-binding"),
  ],
)
def test_interactive_sdk_and_autonomous_bound_root_parks_locate_after_cold_reopen(
  tmp_path: Path,
  with_binding: bool,
) -> None:
  async def case() -> None:
    path = tmp_path / "root.sqlite3"
    store = SQLiteApprovalStore(path)
    session = _session()
    assert session.owner_user_id is not None
    policy = SingleUserApprovalPolicy(store=store)
    route = DurableLocalApprovalRoute(store, policy, session)
    context = root_context(
      session, withheld_tools=frozenset(), research_file_id=42,
      selection_snapshot=capture_selection_snapshot(_resolver().selection_policy),
    )
    initial = admit_root_turn(
      context, execution=_root_execution(
        "anthropic-generation-retained",
        run_mode="autonomous" if with_binding else "interactive",
      ),
      tool_grant={"tool_ids": context.tool_ids, "withheld_tools": sorted(context.withheld)},
      approval_route=route,
    )
    admission = await record_root_turn_admission(store, initial)
    reopened = SQLiteApprovalStore(path)
    generation = await reopened.current_policy_generation()
    assert generation is not None
    assert (
      generation.policy_id, generation.policy_version, generation.policy_bundle_hash,
    ) == (policy.policy_id, policy.policy_version, policy.policy_bundle_hash)
    assert admission.policy_generation_ref == generation.generation_ref
    record = await reopened._read_root_turn_for_executor(
      job_id=session.session_id, turn_id=admission.turn_id,
      tenant_storage_id=session.owner_user_id,
    )
    assert record is not None
    assert json.loads(record.capability_bind_json) == initial.capability_bind.to_json()
    assert record.credential_generation_ref == "anthropic-generation-retained"
    assert json.loads(record.effective_grant_json) == dict(admission.tool_grant)
    assert json.loads(record.research_binding_json) == context.research_binding.model_dump(mode="json")
    assert context.selection_snapshot is not None
    assert json.loads(record.selection_snapshot_json) == context.selection_snapshot.to_json()
    assert record.approval_route_kind == "durable_local"
    approval_id, call_id, nonce = await _root_approval(
      reopened, session=session, with_binding=with_binding,
    )
    await link_root_park(reopened, admission, tool_call_id=call_id, nonce=nonce)
    cold = SQLiteApprovalStore(path)
    assert await cold._locate_root_turn_for_approval(
      approval_id=approval_id, tenant_storage_id=session.owner_user_id,
    ) == (session.session_id, call_id, nonce, admission.turn_id)
    with pytest.raises(RootTurnRecordError) as tenant_error:
      await cold._locate_root_turn_for_approval(
        approval_id=approval_id, tenant_storage_id="foreign-tenant",
      )
    assert tenant_error.value.code == "tenant_mismatch"
    if not with_binding:
      with cold._connection() as conn:
        assert conn.execute("SELECT count(*) FROM approval_executable_bindings").fetchone()[0] == 0

  asyncio.run(case())


@pytest.mark.parametrize("durable", [True, False])
def test_interactive_root_turn_records_before_tools_in_dispatchers_store(
  tmp_path: Path,
  durable: bool,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  from agent.interactive import runtime_chat_builder
  from agent.shared.tool_registration import build_product_tool_registration_composition
  from agent.skills.composition import compile_skill_application
  from agent_gateway.dispatcher_factory import (
    GatewayDispatcherDeps, InvocationPrincipal, build_tool_dispatcher,
  )
  from agent_gateway.event_log import EventLog

  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "dispatcher.sqlite3")
    old_store = SQLiteApprovalStore(tmp_path / "session.sqlite3")
    session = _session()
    assert session.owner_user_id is not None
    old_route = DurableLocalApprovalRoute(old_store, SingleUserApprovalPolicy(store=old_store), session)
    session.approval_route = old_route
    deps = GatewayDispatcherDeps(
      mcp_client=SimpleNamespace(is_mcp_tool=lambda _name: False),
      approval_store=store if durable else None,
      approval_policy=SingleUserApprovalPolicy(store=store) if durable else None,
      mcp_meta_inject_servers=frozenset(),
    )
    session._gateway_dispatcher_deps = deps
    execution = _root_execution()
    skill_application = compile_skill_application(
      skills_root=Path(__file__).resolve().parents[3] / "api/memory/workspace/notes/skills",
      source_prefix=PurePosixPath("skills"),
      tool_registration_catalog=build_product_tool_registration_composition().catalog,
    )
    calls: list[str] = []

    async def tool_handler(
      _input: dict[str, Any], **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      calls.append("file_read")
      return {"content": "retained"}, None

    def root_count(owner: SQLiteApprovalStore) -> int:
      with owner._connection() as conn:
        return conn.execute("SELECT count(*) FROM approval_root_turns").fetchone()[0]

    def load_citation_validation_settings() -> Any:
      return SimpleNamespace(citation_session_registry_enabled=False)

    def prepare_runtime_tooling(**_kwargs: Any) -> Any:
      return SimpleNamespace(
        active_tool_definitions=lambda: [{"name": "file_read"}, {"name": "unbound"}],
        unregistered_needs_approval=lambda *_args: False,
        input_preparation_context_factory=None,
        ce_bundle=None, child_code_execution_factory=None, child_code_execution_binder=None,
      )

    async def prepare_runtime_coordination(**_kwargs: Any) -> Any:
      return SimpleNamespace(unbound_local_tools=frozenset({"unbound"}))

    def build_chat_runtime_from_coordination(**_kwargs: Any) -> Any:
      # The builder has not dispatched any tools; the private snapshot already exists.
      assert calls == []
      assert root_count(store) == (1 if durable else 0)
      assert root_count(old_store) == 0
      return build_tool_dispatcher(
        deps, principal=InvocationPrincipal.from_session(session), profile="interactive",
        capability_bind=execution.bind,
        event_log=EventLog(), session_id=session.session_id, request_approval=None,
        needs_approval=lambda *_args: False,
        local_tool_handlers={"file_read": tool_handler},
        _excel_tool_dispatcher_cls=lambda **kwargs: kwargs["base"],
      )

    monkeypatch.setattr(
      runtime_chat_builder, "_load_citation_validation_settings",
      load_citation_validation_settings,
    )
    monkeypatch.setattr(
      runtime_chat_builder, "get_or_create_interceptors", lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
      runtime_chat_builder, "_prepare_runtime_tooling", prepare_runtime_tooling,
    )
    monkeypatch.setattr(
      runtime_chat_builder, "_prepare_runtime_coordination", prepare_runtime_coordination,
    )
    monkeypatch.setattr(
      runtime_chat_builder, "_build_chat_runtime_from_coordination",
      build_chat_runtime_from_coordination,
    )

    prompting = SimpleNamespace(
      skill_application=skill_application, research_turn=None, trusted_research_file_id=None,
      prompt_context={}, profile_composition=SimpleNamespace(config=SimpleNamespace(name="analyst")),
      execution_policy=None, profile_denied_mcp_servers=set(), skill_catalog_entries=[],
      workspace_dir=None, local_tool_handlers={"file_read": tool_handler},
      code_execution_extra_env={}, system_prompt="root", approval_fn=lambda *_args: False,
      excluded_tools=set(), profile_deferred_mcp_tools=None, profile_allowed_mcp_tools_by_server=None,
      profile_describe_mcp_scope_block=None, tool_surface=None, live_tool_surface=None,
      fms_rebinder=None, stage_skill_name=None, filesystem_access_mode="trusted_local_unconfined",
    )
    dispatcher = await runtime_chat_builder._assemble_chat_runtime(
      session=session,
      request=SimpleNamespace(
        capability_execution=execution, capability_execution_resolver=_resolver(),
        selection_snapshot=None, selection_provenance=None,
      ),
      channel="cli", auth_manager=None,
      channel_registry=SimpleNamespace(get_channel_for_tool=lambda _name: None),
      mcp_client_manager=deps.mcp_client, mcp_relay=None, sdk_config=None, risk_preamble="",
      clear_source_pack_session_fn=lambda _session: None,
      session_storage_user_id_fn=lambda _session, **_kwargs: session.owner_user_id,
      memory_scope_for_user_fn=lambda _user: nullcontext(),
      prepare_prompting_state_fn=lambda **_kwargs: prompting,
      skill_application=skill_application,
      containment=ContainmentSettings(),
    )
    admission = current_root_turn(session)
    assert admission is not None
    assert admission.capability_bind is execution.bind
    assert admission.tool_grant == {"tool_ids": None, "withheld_tools": ["unbound"]}
    assert session.approval_route is old_route  # Capturing does not rebind session routing.
    if durable:
      assert dispatcher._approval_store is store
      approval_id, call_id, nonce = await _root_approval(
        dispatcher._approval_store, session=session, with_binding=True,
      )
      await link_root_park(store, admission, tool_call_id=call_id, nonce=nonce)
      assert await SQLiteApprovalStore(store.path)._locate_root_turn_for_approval(
        approval_id=approval_id, tenant_storage_id=session.owner_user_id,
      ) == (session.session_id, call_id, nonce, admission.turn_id)
    else:
      assert dispatcher._approval_store is None
      assert admission.policy_generation_ref is None
    result, error = await dispatcher.dispatch("first-tool", "file_read", {"path": "retained.txt"})
    assert error is None, error
    assert result == {"content": "retained"}
    assert calls == ["file_read"]
    assert root_count(store) == (1 if durable else 0)

  asyncio.run(case())
