"""The one admitted approval route for a run.

Admission resolves exactly one route value from the run's control authority and
every consumer reads that value instead of testing whichever handle the
executing process happens to hold.

The two variants that can reach a ledger or a decision queue carry the
`GatewaySession` they bind to as a required, type-exact field, so the pairing
"durable-local lifecycle without the session that records its decision" has no
spelling at all. `NoApprovalRoute` is the only sessionless variant, and it
records nothing.
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
class NoApprovalRoute:
  """The run reaches no decider; a tool needing approval is refused at the door."""


ApprovalRoute = Union[
  DurableLocalApprovalRoute[_StoreT, _PolicyT],
  ParentDelegatedApprovalRoute,
  NoApprovalRoute,
]

NO_APPROVAL_ROUTE = NoApprovalRoute()


def route_store(
  route: ApprovalRoute[_StoreT, _PolicyT],
) -> _StoreT | None:
  """Project the durable ledger a route owns, if it owns one."""

  return route.store if isinstance(route, DurableLocalApprovalRoute) else None


def route_policy(
  route: ApprovalRoute[_StoreT, _PolicyT],
) -> _PolicyT | None:
  """Project the opaque policy a route owns, if it owns one."""

  return route.policy if isinstance(route, DurableLocalApprovalRoute) else None


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
  "DurableLocalApprovalRoute",
  "SQLiteApprovalStore",
  "NoApprovalRoute",
  "ParentDelegatedApprovalRoute",
  "bind_session_approval_route",
  "route_policy",
  "route_store",
  "session_approval_route",
]
