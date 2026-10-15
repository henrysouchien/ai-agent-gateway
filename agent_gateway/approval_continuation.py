"""Host-private approval continuation contracts installed by gateway configuration.

These requests carry route-verified tenant identity, not model arguments or a new
execution grant. The installed host adapter owns restoration and delivery under
the original grant. The route calls the root hook first; a `not_root` approval
goes to the child hook, whose host adapter restores the owning session runtime
and selects the parked task's owner: a workflow node goes to the workflow hook,
which resolves its session-owned service from the parked task (never an
application-global workflow service); any other child resumes through the
ordinary child continuation. No configured hook means no installed consumer;
these types provide neither execution nor a fallback implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Literal


ContinueStatus = Literal[
  "delivered",
  "already_delivered",
  "outcome_unknown",
  "not_root",
  "credential_unavailable",
  "policy_generation_changed",
  "tenant_mismatch",
  "not_restorable",
]


@dataclass(frozen=True)
class RootContinueRequest:
  """Root approval ID and tenant storage identity verified by the calling route.

  ``reissued_from`` names the approval this one re-issued under the current
  policy generation: the host publishes the new ask on the job's durable log
  instead of delivering anything.
  """

  approval_id: str
  tenant_storage_id: str
  reissued_from: str | None = None


@dataclass(frozen=True)
class RootContinueOutcome:
  """Root delivery status and source-owned execution references, when known."""

  status: ContinueStatus
  job_id: str | None = None
  turn_id: str | None = None
  tool_call_id: str | None = None


RootContinueHook = Callable[[RootContinueRequest], Awaitable[RootContinueOutcome]]


@dataclass(frozen=True)
class WorkflowContinueRequest:
  """Workflow approval, parked physical task, and route-verified tenant identity."""

  approval_id: str
  parked_task_id: str
  tenant_storage_id: str


@dataclass(frozen=True)
class WorkflowContinueOutcome:
  """Workflow delivery status and references resolved by its owning adapter."""

  status: ContinueStatus
  job_id: str | None = None
  tool_call_id: str | None = None


WorkflowContinueHook = Callable[
  [WorkflowContinueRequest], Awaitable[WorkflowContinueOutcome]
]


@dataclass(frozen=True)
class ChildContinueRequest:
  """Non-root approval ID and tenant storage identity verified by the route."""

  approval_id: str
  tenant_storage_id: str


@dataclass(frozen=True)
class ChildContinueOutcome:
  """Non-root delivery status and references resolved by the child adapter."""

  status: ContinueStatus
  job_id: str | None = None
  tool_call_id: str | None = None


ChildContinueHook = Callable[[ChildContinueRequest], Awaitable[ChildContinueOutcome]]
