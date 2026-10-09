"""Host-side admission: the one producer of a job's admitted context.

``admit(request, parent, owners=...)`` is the D2 seam of
``docs/design/execution-boundary-consolidation-design-2026-09-24.md`` §3. It
reads the owner registrations handed to it in ``owners`` and returns an
``AdmittedContext`` or the existing owner's refusal; it writes nothing. The
context holds the exact tool and approval authorities its consumers execute;
``AdmittedContextWire`` is only their journal projection. Its side effects
are log lines: one ``admitted`` line and one line per unknown name.

``budget_share`` records observed remaining spend, including overruns and an
unknown value; ``runner_soft_budget`` owns spend containment. Fresh spawns
default to depth two (root 0 → lead 1 → worker 2); ``HANK_MAX_SPAWN_DEPTH``
sets that resource bound and a negative value disables it. At the deepest
tier the executable grant omits ``run_agent``; continuations retain their
original grant and lineage without applying a fresh-spawn bound.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, Protocol, TypeVar, cast, overload
from types import MappingProxyType
from uuid import uuid4

from agent_workflow_contracts import AdmittedTask, OperationUnavailable, ResolvedAuthority
from agent_workflow_contracts.models import (
  AdmittedContextWire,
  AdmittedLineage,
  AdmittedResearchBinding,
  AdmittedTenant,
)
from model_authority.bind import CapabilityBind
from pydantic import JsonValue
from model_authority.selection import SelectionSnapshot

from .capability_execution import BoundCapabilityExecution, CapabilityExecutionResolver
from .approval_route import (
  ApprovalRoute,
  DurableLocalApprovalRoute,
  NoApprovalRoute,
  child_approval_route,
  route_policy,
  session_approval_route,
)
from .child_environment import CHILD_ENVIRONMENT_ALLOWLISTS
from .policy_imports import require_inherited_role, role_denied_tools_for_session
from .session import session_owner_user_id
from .sub_agent_helpers import RUN_AGENT_TOOL_NAME
from .approval_store import SQLiteApprovalStore
from .providers.anthropic_oauth import is_anthropic_generation_ref

log = logging.getLogger("agent_gateway.admission")

RefusalReason = Literal[
  "session_unauthenticated",
  "tenant_mismatch",
  "grant_exceeds_parent",
  "invite_effect_not_allowed",
  "handle_not_granted",
  "binding_not_granted",
  "depth_exceeded",
]
Placement = Literal["in_process", "local_stub", "hosted_stub"]


@dataclass(frozen=True, slots=True)
class Refusal:
  reason: RefusalReason
  owner: str
  detail: str
  classification: Literal["resource"] | None = None
  next_call: str | None = None


@dataclass(frozen=True, slots=True)
class SpawnRequest:
  """One request to start a job from a registered operation (design §3.1)."""

  operation: str
  task: str
  brief_ref: str | None = None
  handles: tuple[str, ...] = ()
  binding: AdmittedResearchBinding | None = None
  effort: str | None = None
  role: str | None = None
  profile: str | None = None


@dataclass(frozen=True, slots=True)
class ContinueRequest:
  """Continue the exact job resolved by the existing task registry."""

  job_id: str
  cause: Literal["continuation", "revive", "placement_lost"] = "continuation"
  message: str | None = None


@dataclass(frozen=True, slots=True)
class MessageRequest:
  """Send within the existing tree roster; references do not grant access."""

  from_job_id: str
  to_job_id: str
  body: str
  handle_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HandleRequest:
  """Read a handle as the job identified by the trusted caller context."""

  handle: str
  reader_job_id: str


@dataclass(frozen=True, slots=True)
class ParentContext:
  """What ``admit`` narrows from.

  The root is minted from the turn's session by :func:`root_context`; at the
  root ``children`` and ``tool_ids`` are ``None``: the session may start any
  operation its catalog resolves, and its tool surface is everything it does
  not withhold.
  """

  tenant: AdmittedTenant | None
  research_binding: AdmittedResearchBinding
  children: tuple[str, ...] | None
  tool_ids: frozenset[str] | None
  withheld: frozenset[str]
  handle_grants: frozenset[str]
  remaining_usd: float | None
  job_id: str | None
  depth: int
  approval_route: ApprovalRoute[object, object]
  selection_snapshot: SelectionSnapshot | None = None


@dataclass(frozen=True, slots=True)
class RootTurnAdmission:
  """One turn's exact root authority, captured before its first tool call."""

  job_id: str | None
  turn_id: str
  tenant: AdmittedTenant | None
  capability_bind: CapabilityBind
  credential_generation_ref: str | None
  tool_grant: Mapping[str, object]
  research_binding: AdmittedResearchBinding
  selection_snapshot: SelectionSnapshot | None
  approval_route_kind: str
  approval_route: ApprovalRoute[Any, Any] = field(repr=False, compare=False)
  policy_generation_ref: str | None = None


def admit_root_turn(
  context: ParentContext,
  *,
  execution: BoundCapabilityExecution,
  tool_grant: Mapping[str, object],
  approval_route: ApprovalRoute[Any, Any],
) -> RootTurnAdmission:
  """Capture owner facts without selecting authority or materializing credentials."""

  bind = execution.bind
  return RootTurnAdmission(
    job_id=context.job_id,
    turn_id="turn:" + uuid4().hex,
    tenant=context.tenant,
    capability_bind=bind,
    credential_generation_ref=(
      bind.credential_ref if is_anthropic_generation_ref(bind.credential_ref) else None
    ),
    tool_grant=MappingProxyType(dict(tool_grant)),
    research_binding=context.research_binding,
    selection_snapshot=context.selection_snapshot,
    approval_route_kind=_approval_route_name(approval_route),
    approval_route=approval_route,
  )


def capture_root_turn(session: Any, admission: RootTurnAdmission) -> None:
  """Install the exact admission the root park closure reads."""

  session.admitted_root_turn = admission


def current_root_turn(session: Any) -> RootTurnAdmission | None:
  """Read the captured turn; never infer it from a session or skill identity."""

  return getattr(session, "admitted_root_turn", None)


async def record_root_turn_admission(
  store: SQLiteApprovalStore,
  admission: RootTurnAdmission,
) -> RootTurnAdmission:
  """Persist the route owner's policy generation, then the private root snapshot."""

  policy = route_policy(admission.approval_route)
  if policy is None or admission.job_id is None or admission.tenant is None:
    return admission
  generation = await store.record_policy_generation(
    policy_id=cast(Any, policy).policy_id,
    policy_version=cast(Any, policy).policy_version,
    policy_bundle_hash=cast(Any, policy).policy_bundle_hash,
  )
  recorded = replace(admission, policy_generation_ref=generation.generation_ref)
  await store.record_root_turn(
    job_id=admission.job_id,
    turn_id=recorded.turn_id,
    tenant_storage_id=admission.tenant.storage_id,
    capability_bind=recorded.capability_bind.to_json(),
    credential_generation_ref=recorded.credential_generation_ref,
    effective_grant=recorded.tool_grant,
    research_binding=recorded.research_binding.model_dump(mode="json"),
    selection_snapshot=(
      recorded.selection_snapshot.to_json()
      if recorded.selection_snapshot is not None else {}
    ),
    approval_route_kind=recorded.approval_route_kind,
    policy_generation_ref=generation.generation_ref,
  )
  return recorded


class RootParkStore(Protocol):
  """The ledger operation used to attach a root turn's parked executable."""

  async def link_root_turn_executable_binding(
    self,
    *,
    job_id: str,
    turn_id: str,
    tool_call_id: str,
    nonce: str,
    tenant_storage_id: str,
  ) -> None: ...


async def link_root_park(
  store: RootParkStore,
  admission: RootTurnAdmission,
  *,
  tool_call_id: str,
  nonce: str,
) -> None:
  """Attach an existing executable binding without copying its private operands."""

  if admission.job_id is None or admission.tenant is None:
    return
  await store.link_root_turn_executable_binding(
    job_id=admission.job_id,
    turn_id=admission.turn_id,
    tool_call_id=tool_call_id,
    nonce=nonce,
    tenant_storage_id=admission.tenant.storage_id,
  )


def root_context(
  session: Any | None,
  *,
  withheld_tools: frozenset[str],
  research_file_id: int | None,
  remaining_usd: float | None = None,
  selection_snapshot: SelectionSnapshot | None = None,
) -> ParentContext:
  """Capture a root turn after runtime setup; never re-root an admitted child."""

  admitted: ParentContext | None = getattr(session, "admitted_job_context", None)
  if admitted is not None and admitted.children is not None:
    return admitted

  tenant = (
    AdmittedTenant(
      tenant_id=getattr(session, "tenant_id", None),
      storage_id=session_owner_user_id(session),
      role=require_inherited_role(session),
    )
    if session is not None
    else None
  )
  context = root_context_from_identity(
    tenant=tenant,
    job_id=getattr(session, "session_id", None),
    tool_ids=None,
    withheld_tools=withheld_tools,
    research_binding=AdmittedResearchBinding(research_file_id=research_file_id),
    approval_route=session_approval_route(session),
    remaining_usd=remaining_usd,
    selection_snapshot=selection_snapshot,
  )
  if session is not None:
    capture_job_context(session, context)
  return context


def root_context_from_identity(
  *,
  tenant: AdmittedTenant | None,
  job_id: str | None,
  tool_ids: frozenset[str] | None,
  withheld_tools: frozenset[str],
  research_binding: AdmittedResearchBinding,
  approval_route: ApprovalRoute[object, object],
  remaining_usd: float | None = None,
  selection_snapshot: SelectionSnapshot | None = None,
) -> ParentContext:
  """Mint from an authenticated edge's identity, live roster and control owner.

  Sessionless schedule/control edges supply these owner facts directly, not a
  synthetic GatewaySession. ``None`` tool IDs retain the session catalog's root
  scope; the grant loop applies withheld tools to a supplied roster.
  """

  return ParentContext(
    tenant=tenant,
    research_binding=research_binding,
    children=None,
    tool_ids=tool_ids,
    withheld=frozenset(withheld_tools),
    handle_grants=frozenset(),
    remaining_usd=remaining_usd,
    job_id=job_id,
    depth=0,
    approval_route=approval_route,
    selection_snapshot=selection_snapshot,
  )


def capture_job_context(session: Any, context: ParentContext) -> None:
  """Install the scope a session's tools narrow from: its turn root, or the
  exact child context of the job it executes."""

  session.admitted_job_context = context


def current_parent_context(session: Any) -> ParentContext:
  """Read the turn-captured root or exact child context without minting scope."""

  context: ParentContext | None = session.admitted_job_context
  assert context is not None, "the authenticated turn edge must capture its context"
  return context


@dataclass(frozen=True, slots=True)
class OperationRegistration:
  """The operation catalog's facts for one resolved operation.

  ``route_ceiling`` is the operation's exact declared tool ceiling (canonical
  route ids; the caller projects them onto live dispatcher names).
  ``operation_routes`` are the routes the operation brings itself (its private
  MCP routes and artifact doors): they may extend the parent's visible surface,
  but remain narrowed by its withheld set. ``route_tool_names`` maps a canonical
  route id to the tool name the role policy reads; an absent id is its own name.
  ``role`` and ``effort`` are the operation's declared model role and exact
  effort (``AgentOperationSnapshot``); a spawn's own value wins, and an absent
  one leaves the authority's slot default in force.
  """

  name: str
  execution_class: str
  route_ceiling: frozenset[str]
  operation_routes: frozenset[str]
  route_tool_names: Mapping[str, str] = field(default_factory=dict)
  children: tuple[str, ...] = ()
  role: str | None = None
  effort: str | None = None


@dataclass(frozen=True, slots=True)
class AdmissionOwners:
  """The owner objects ``admit`` reads; nothing here is written."""

  operation: OperationRegistration
  capability_execution_resolver: CapabilityExecutionResolver
  # The parent session the role policy reads; ``None`` outside a gateway turn.
  session: Any | None
  # The route registry: whether a declared route id has a registered route.
  route_registered: Callable[[str], bool]
  resolve_tool_authority: Callable[
    [frozenset[str]],
    ResolvedAuthority | OperationUnavailable,
  ]
  # The executor backend's facts: where the job runs and the
  # ``child_environment`` purpose of its launch contract (``None``: the
  # backend launches no process, so nothing is projected).
  placement: Placement
  env_contract: str | None = None
  env_names: frozenset[str] = frozenset()
  outputs: frozenset[str] = frozenset()
  frozen_bind: CapabilityBind | None = None


@dataclass(frozen=True, slots=True)
class ContinuationOwners:
  """Durable authority, not a fresh operation catalog or model resolver.

  The registry resolves ``task`` and its existing resume path validates durable
  routes and definitions. ``research_binding`` comes from that task's admitted
  inputs. Missing historical context projections do not discard those facts.
  """

  task: AdmittedTask
  research_binding: AdmittedResearchBinding
  placement: Placement = "in_process"
  env_contract: str | None = None
  env_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class MessageOwners:
  """Facts resolved by the existing roster and durable output owner."""

  recipient_tenant: AdmittedTenant | None
  outputs: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class HandleOwners:
  """Current outputs of the reader job and its descendants, not message refs."""

  outputs: frozenset[str] = frozenset()


_AuthorityT = TypeVar("_AuthorityT", ResolvedAuthority, AdmittedTask)


@dataclass(frozen=True, slots=True)
class AdmittedContext(Generic[_AuthorityT]):
  """Wire facts plus the exact fresh or durable authority execution consumes."""

  wire: AdmittedContextWire
  tool_authority: _AuthorityT
  approval_route: ApprovalRoute[object, object]


def context_for_child(
  admitted: AdmittedContext[_AuthorityT],
  *,
  job_id: str,
) -> ParentContext:
  """Bind a registry job ID to the authority its descendants may narrow."""

  wire = admitted.wire
  return ParentContext(
    tenant=wire.tenant,
    research_binding=wire.research_binding,
    children=wire.children,
    tool_ids=frozenset(wire.tool_grant),
    withheld=frozenset(wire.withheld_tools),
    handle_grants=frozenset(wire.handle_grants),
    remaining_usd=wire.budget_share,
    job_id=job_id,
    depth=wire.lineage.depth,
    approval_route=admitted.approval_route,
    selection_snapshot=(
      SelectionSnapshot.from_json(wire.selection_snapshot)
      if wire.selection_snapshot is not None
      else None
    ),
  )


@overload
def admit(
  request: SpawnRequest,
  parent: ParentContext,
  *,
  owners: AdmissionOwners,
) -> AdmittedContext[ResolvedAuthority] | Refusal | OperationUnavailable: ...


@overload
def admit(
  request: ContinueRequest,
  parent: ParentContext,
  *,
  owners: ContinuationOwners,
) -> AdmittedContext[AdmittedTask] | Refusal: ...


@overload
def admit(
  request: MessageRequest,
  parent: ParentContext,
  *,
  owners: MessageOwners,
) -> ParentContext | Refusal: ...


@overload
def admit(
  request: HandleRequest,
  parent: ParentContext,
  *,
  owners: HandleOwners,
) -> ParentContext | Refusal: ...


def admit(
  request: SpawnRequest | ContinueRequest | MessageRequest | HandleRequest,
  parent: ParentContext,
  *,
  owners: AdmissionOwners | ContinuationOwners | MessageOwners | HandleOwners,
) -> (
  AdmittedContext[ResolvedAuthority]
  | AdmittedContext[AdmittedTask]
  | ParentContext
  | Refusal
  | OperationUnavailable
):
  """Narrow registered spawn, durable continuation, message or read authority."""

  if isinstance(request, MessageRequest):
    assert isinstance(owners, MessageOwners)
    return _admit_message(request, parent, owners=owners)
  if isinstance(request, HandleRequest):
    assert isinstance(owners, HandleOwners)
    return _admit_handle(request, parent, owners=owners)
  if isinstance(request, ContinueRequest):
    assert isinstance(owners, ContinuationOwners)
    return _admit_continuation(request, parent, owners=owners)
  assert isinstance(owners, AdmissionOwners)
  return _admit_spawn(request, parent, owners=owners)


def _admit_spawn(
  request: SpawnRequest,
  parent: ParentContext,
  *,
  owners: AdmissionOwners,
) -> AdmittedContext[ResolvedAuthority] | Refusal | OperationUnavailable:
  """Derive one child's admitted context, or refuse it (design §3.2-3.4).

  A resolver that cannot bind raises its own ``CapabilityResolutionError``;
  admission adds no refusal around it.
  """

  max_spawn_depth = int(os.getenv("HANK_MAX_SPAWN_DEPTH", "2"))
  if max_spawn_depth >= 0 and parent.depth >= max_spawn_depth:
    return Refusal(
      reason="depth_exceeded",
      owner="admission spawn containment",
      detail=(
        f"Spawn depth {parent.depth} has reached the configured maximum "
        f"{max_spawn_depth}."
      ),
      classification="resource",
      next_call=(
        "Return an ordinary final response with your partial work to your "
        "parent; the parent can spawn the remaining work."
      ),
    )
  child_depth = parent.depth + 1

  registration = owners.operation
  if parent.children is not None and request.operation not in parent.children:
    return Refusal(
      reason="grant_exceeds_parent",
      owner="admission tool_grant",
      detail=f"operation {request.operation!r} is not among the parent's children",
    )
  ungranted_handles = sorted(
    set(request.handles) - parent.handle_grants - owners.outputs
  )
  if ungranted_handles:
    return Refusal(
      reason="handle_not_granted",
      owner="admission handle_grants",
      detail="handles not granted to the parent: " + ", ".join(ungranted_handles),
    )
  parent_binding = parent.research_binding
  requested_binding = request.binding
  if requested_binding is None:
    research_binding = parent_binding
  else:
    if (
      parent_binding.research_file_id is not None
      and requested_binding.research_file_id is not None
      and requested_binding.research_file_id != parent_binding.research_file_id
    ):
      return Refusal(
        reason="binding_not_granted",
        owner="admission research_binding",
        detail=(
          f"research_file_id {requested_binding.research_file_id} is not the "
          f"parent's bound research file {parent_binding.research_file_id}"
        ),
      )
    research_binding = AdmittedResearchBinding(
      research_file_id=(
        requested_binding.research_file_id
        if requested_binding.research_file_id is not None
        else parent_binding.research_file_id
      ),
      ticker=(
        requested_binding.ticker
        if requested_binding.ticker is not None
        else parent_binding.ticker
      ),
    )
  if request.profile is not None:
    log.warning(
      "admission ignored profile for %s: profile=%r",
      request.operation,
      request.profile,
    )

  capability_bind = (
    owners.frozen_bind
    if owners.frozen_bind is not None
    else owners.capability_execution_resolver.select_bind(
      registration.execution_class,
      role=request.role if request.role is not None else registration.role,
      effort=request.effort if request.effort is not None else registration.effort,
      selection_snapshot=parent.selection_snapshot,
    )
  )

  children = registration.children
  role_denied = role_denied_tools_for_session(
    owners.session,
    role=parent.tenant.role if parent.tenant is not None else "invite",
  )
  granted: list[str] = []
  for tool_id in sorted(registration.route_ceiling):
    if not owners.route_registered(tool_id):
      log.warning(
        "route_unregistered: %s declares %s, which has no live route; "
        "admitted without it",
        registration.name,
        tool_id,
      )
      continue
    policy_name = registration.route_tool_names.get(tool_id, tool_id)
    parent_holds = (
      tool_id in registration.operation_routes
      or parent.tool_ids is None
      or tool_id in parent.tool_ids
    )
    if (
      not parent_holds
      or tool_id in parent.withheld
      or policy_name in parent.withheld
      or policy_name in role_denied
    ):
      continue
    if tool_id == RUN_AGENT_TOOL_NAME and (
      not children
      or (max_spawn_depth >= 0 and child_depth >= max_spawn_depth)
    ):
      continue
    granted.append(tool_id)

  tool_authority = owners.resolve_tool_authority(frozenset(granted))
  if isinstance(tool_authority, OperationUnavailable):
    return tool_authority

  approval_route = child_approval_route(parent.approval_route)
  context = AdmittedContextWire(
    tenant=parent.tenant,
    research_binding=research_binding,
    capability_bind=capability_bind,
    selection_snapshot=(
      cast(dict[str, JsonValue], parent.selection_snapshot.to_json())
      if parent.selection_snapshot is not None
      else None
    ),
    children=children,
    tool_grant=tuple(entry.tool_id for entry in tool_authority.grant.tools),
    withheld_tools=tuple(sorted(parent.withheld)),
    budget_share=parent.remaining_usd,
    env_projection=_environment_projection(owners.env_contract, owners.env_names),
    approval_route=_approval_route_name(approval_route),
    handle_grants=tuple(sorted(request.handles)),
    placement=owners.placement,
    lineage=AdmittedLineage(parent_job_id=parent.job_id, depth=child_depth),
  )
  log.info(
    "admitted %s: %s",
    registration.name,
    context.model_dump_json(),
  )
  return AdmittedContext(context, tool_authority, approval_route)


def _admit_continuation(
  request: ContinueRequest,
  parent: ParentContext,
  *,
  owners: ContinuationOwners,
) -> AdmittedContext[AdmittedTask] | Refusal:
  task = owners.task
  snapshot = task.execution_snapshot
  # The existing resume owner resolves executable tasks and validates their
  # snapshot, bind, routes and definitions before this pure authority step.
  assert snapshot is not None
  assert task.model_bind is not None
  original = snapshot.admitted_context
  if original is not None and original.tenant != parent.tenant:
    return Refusal(
      reason="tenant_mismatch",
      owner="admission tenant",
      detail=f"job {request.job_id!r} belongs to a different tenant",
    )
  approval_route = child_approval_route(parent.approval_route)
  if original is None:
    # Older executable tasks have a complete durable grant without this newer
    # journal projection. Preserve that authority; never reselect or recompile.
    # They declared no child operations or historical spend observation.
    context = AdmittedContextWire(
      tenant=parent.tenant,
      research_binding=owners.research_binding,
      capability_bind=task.model_bind,
      children=(),
      tool_grant=tuple(entry.tool_id for entry in task.tool_grant.tools),
      withheld_tools=tuple(sorted(parent.withheld)),
      budget_share=None,
      env_projection=_environment_projection(owners.env_contract, owners.env_names),
      approval_route=_approval_route_name(approval_route),
      handle_grants=tuple(sorted({
        grant.content_id for grant in task.content_read_grants
      })),
      placement=owners.placement,
      lineage=AdmittedLineage(parent_job_id=parent.job_id, depth=parent.depth + 1),
    )
  else:
    context = original.model_copy(update={
      "capability_bind": task.model_bind,
      "tool_grant": tuple(entry.tool_id for entry in task.tool_grant.tools),
      # Narrow future descendants without replacing this task's durable grant.
      "withheld_tools": tuple(sorted(parent.withheld.union(original.withheld_tools))),
      "research_binding": owners.research_binding,
      "approval_route": _approval_route_name(approval_route),
      **(
        {
          "placement": owners.placement,
          "env_projection": _environment_projection(
            owners.env_contract, owners.env_names,
          ),
        }
        if request.cause == "placement_lost"
        else {}
      ),
    })
  log.info("admitted %s: %s", request.job_id, context.model_dump_json())
  return AdmittedContext(context, task, approval_route)


def _context_with_outputs(
  context: ParentContext,
  outputs: frozenset[str],
) -> ParentContext:
  if outputs <= context.handle_grants:
    return context
  return replace(context, handle_grants=context.handle_grants | outputs)


def _admit_message(
  request: MessageRequest,
  sender: ParentContext,
  *,
  owners: MessageOwners,
) -> ParentContext | Refusal:
  if sender.tenant is None or owners.recipient_tenant is None:
    return Refusal(
      "session_unauthenticated", "admission tenant",
      "messaging requires authenticated sender and recipient identities",
    )
  if sender.tenant != owners.recipient_tenant:
    return Refusal(
      "tenant_mismatch", "admission tenant",
      f"recipient {request.to_job_id!r} belongs to a different tenant",
    )
  admitted = _context_with_outputs(sender, owners.outputs)
  ungranted = sorted(set(request.handle_refs) - admitted.handle_grants)
  if ungranted:
    return Refusal(
      "handle_not_granted", "admission handle_grants",
      "handles not granted to the sender: " + ", ".join(ungranted),
    )
  log.info("admitted message %s -> %s", sender.job_id, request.to_job_id)
  return admitted


def _admit_handle(
  request: HandleRequest,
  reader: ParentContext,
  *,
  owners: HandleOwners,
) -> ParentContext | Refusal:
  if reader.tenant is None:
    return Refusal(
      "session_unauthenticated", "admission tenant",
      "handle reads require an authenticated reader identity",
    )
  admitted = _context_with_outputs(reader, owners.outputs)
  if request.handle not in admitted.handle_grants:
    return Refusal(
      "handle_not_granted", "admission handle_grants",
      f"handle {request.handle!r} is not granted to the reader",
    )
  log.info("admitted handle %s for %s", request.handle, reader.job_id)
  return admitted


def _approval_route_name(
  route: ApprovalRoute[object, object],
) -> Literal["no_approval_route", "parent_delegated", "durable_local"]:
  """Label the admitted route at the root or child edge."""

  if isinstance(route, NoApprovalRoute):
    return "no_approval_route"
  if isinstance(route, DurableLocalApprovalRoute):
    return "durable_local"
  return "parent_delegated"


def _environment_projection(
  env_contract: str | None,
  env_names: frozenset[str],
) -> tuple[str, ...]:
  if env_contract is None:
    return ()
  contract = CHILD_ENVIRONMENT_ALLOWLISTS[env_contract]
  projection: list[str] = []
  for name in sorted(env_names):
    if name in contract:
      projection.append(name)
    else:
      log.warning(
        "env_name_unknown: %s is not in the %s launch contract; omitted",
        name,
        env_contract,
      )
  return tuple(projection)


__all__ = [
  "AdmittedContext",
  "AdmissionOwners",
  "ContinuationOwners",
  "ContinueRequest",
  "HandleOwners",
  "HandleRequest",
  "MessageOwners",
  "MessageRequest",
  "OperationRegistration",
  "ParentContext",
  "Refusal",
  "RootTurnAdmission",
  "SpawnRequest",
  "admit",
  "admit_root_turn",
  "capture_job_context",
  "capture_root_turn",
  "context_for_child",
  "current_parent_context",
  "current_root_turn",
  "link_root_park",
  "record_root_turn_admission",
  "root_context",
  "root_context_from_identity",
]
