"""The one admitted approval route for a run.

Admission resolves exactly one route value from the run's control authority and
every consumer reads that value instead of testing whichever handle the
executing process happens to hold.

The variants that reach a ledger in this process carry the `GatewaySession`
their decision queue binds to as a required, type-exact field, so the pairing
"durable-local lifecycle without the session that records its decision" has no
spelling at all. A child never receives `DurableLocalApprovalRoute`: an
in-process child carries `InProcessDelegatedApprovalRoute` over its parent's
ledger, a detached child carries `ParentDelegatedApprovalRoute` over its
inherited channel. `ControlPlaneApprovalRoute` is a sessionless root's
authority (a schedule fire): the gateway registry's ledger, decided by the
control plane's authenticated deciders; no dispatcher executes on it.
`NoApprovalRoute` records nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union

from .autonomous_approval_channel import AutonomousApprovalChannelChild
from .approval_store import SQLiteApprovalStore

if TYPE_CHECKING:
  from .session import GatewaySession

_StoreT = TypeVar("_StoreT")
_PolicyT = TypeVar("_PolicyT")


def _gateway_session_cls() -> type:
  # Resolved on use: a session carries its admitted route as a field, so the
  # module-level import edge only runs session -> approval_route.
  from .session import GatewaySession

  return GatewaySession


@dataclass(frozen=True, slots=True)
class DurableLocalApprovalRoute(Generic[_StoreT, _PolicyT]):
  """The process running the lifecycle owns the ledger, the policy, AND the
  session the row and its decision queue bind to."""

  store: _StoreT
  policy: _PolicyT
  session: "GatewaySession"

  def __post_init__(self) -> None:
    if self.store is None or self.policy is None:
      raise ValueError(
        "durable-local approval route requires a store and a policy"
      )
    if type(self.session) is not _gateway_session_cls():
      raise TypeError(
        "durable-local approval route requires the exact GatewaySession "
        "whose ledger records its decision"
      )


@dataclass(frozen=True, slots=True)
class ParentDelegatedApprovalRoute:
  """The run authors approval requests; its parent records and decides them."""

  channel: AutonomousApprovalChannelChild
  session: "GatewaySession"

  def __post_init__(self) -> None:
    if type(self.channel) is not AutonomousApprovalChannelChild:
      raise TypeError(
        "parent-delegated approval route requires its exact inherited channel"
      )
    if type(self.session) is not _gateway_session_cls():
      raise TypeError(
        "parent-delegated approval route requires the exact GatewaySession "
        "whose decision queue receives its decision"
      )


@dataclass(frozen=True, slots=True)
class InProcessDelegatedApprovalRoute(Generic[_StoreT, _PolicyT]):
  """An in-process child authors requests into its parent's ledger; the
  parent's session records and decides them."""

  parent_ledger: DurableLocalApprovalRoute[_StoreT, _PolicyT]

  def __post_init__(self) -> None:
    if type(self.parent_ledger) is not DurableLocalApprovalRoute:
      raise TypeError(
        "in-process delegated approval route requires its parent's exact "
        "durable-local ledger"
      )


@dataclass(frozen=True, slots=True)
class ControlPlaneApprovalRoute(Generic[_StoreT, _PolicyT]):
  """A sessionless root's approval authority: the gateway registry's ledger
  and policy, decided through the control plane."""

  store: _StoreT
  policy: _PolicyT

  def __post_init__(self) -> None:
    if self.store is None or self.policy is None:
      raise ValueError(
        "control-plane approval route requires a store and a policy"
      )


@dataclass(frozen=True, slots=True)
class NoApprovalRoute:
  """The run reaches no decider; a tool needing approval is refused at the door.

  Only a live order needs approval (`approval_policy.MONEY_BOUNDARY_TOOL_CLASSES`),
  and a live order nobody can decide must not spend the owner's money.
  """


ApprovalRoute = Union[
  DurableLocalApprovalRoute[_StoreT, _PolicyT],
  ParentDelegatedApprovalRoute,
  InProcessDelegatedApprovalRoute[_StoreT, _PolicyT],
  ControlPlaneApprovalRoute[_StoreT, _PolicyT],
  NoApprovalRoute,
]

NO_APPROVAL_ROUTE = NoApprovalRoute()


def route_ledger(
  route: ApprovalRoute[_StoreT, _PolicyT],
) -> DurableLocalApprovalRoute[_StoreT, _PolicyT] | None:
  """The in-process durable ledger a route records into, if any."""

  if isinstance(route, DurableLocalApprovalRoute):
    return route
  if isinstance(route, InProcessDelegatedApprovalRoute):
    return route.parent_ledger
  return None


def child_approval_route(
  route: ApprovalRoute[_StoreT, _PolicyT],
) -> ApprovalRoute[_StoreT, _PolicyT]:
  """The route an in-process child of ``route``'s owner admits with."""

  if isinstance(route, DurableLocalApprovalRoute):
    return InProcessDelegatedApprovalRoute(route)
  return route


def route_store(
  route: ApprovalRoute[_StoreT, _PolicyT],
) -> _StoreT | None:
  """Project the durable ledger a route records into, if it reaches one."""

  ledger = route_ledger(route)
  return ledger.store if ledger is not None else None


def route_policy(
  route: ApprovalRoute[_StoreT, _PolicyT],
) -> _PolicyT | None:
  """Project the opaque policy a route's ledger owner holds, if any."""

  ledger = route_ledger(route)
  return ledger.policy if ledger is not None else None


def route_session(route: ApprovalRoute[_StoreT, _PolicyT]) -> "GatewaySession":
  """The session whose ledger row and decision queue a dispatcher records to."""

  if isinstance(route, (DurableLocalApprovalRoute, ParentDelegatedApprovalRoute)):
    return route.session
  if isinstance(route, InProcessDelegatedApprovalRoute):
    return route.parent_ledger.session
  raise TypeError(f"{type(route).__name__} carries no executing session")


def session_approval_route(session: Any) -> ApprovalRoute[object, object]:
  """Read the route admission stamped on a session."""

  if session is None:
    return NO_APPROVAL_ROUTE
  route = getattr(session, "approval_route", None)
  if route is None:
    return NO_APPROVAL_ROUTE
  return route


def bind_session_approval_route(
  session: Any,
  store: _StoreT | None,
  policy: _PolicyT | None,
) -> ApprovalRoute[_StoreT, _PolicyT]:
  """Stamp the admitted route of a session whose own process owns the ledger.

  A durable-local route exists exactly when this process holds both the ledger
  and the policy; otherwise the session reaches no decider.
  """

  route: ApprovalRoute[_StoreT, _PolicyT] = (
    NO_APPROVAL_ROUTE
    if store is None or policy is None
    else DurableLocalApprovalRoute(store, policy, session)
  )
  session.approval_route = route
  session.approval_store = route_store(route)
  session.approval_policy = route_policy(route)
  return route


__all__ = [
  "NO_APPROVAL_ROUTE",
  "ApprovalRoute",
  "ControlPlaneApprovalRoute",
  "DurableLocalApprovalRoute",
  "InProcessDelegatedApprovalRoute",
  "SQLiteApprovalStore",
  "NoApprovalRoute",
  "ParentDelegatedApprovalRoute",
  "bind_session_approval_route",
  "child_approval_route",
  "route_ledger",
  "route_policy",
  "route_session",
  "route_store",
  "session_approval_route",
]
