from __future__ import annotations

import hashlib
import inspect
from typing import Any

from .approval_policy import (
  MONEY_BOUNDARY_TOOL_CLASSES,
  ApprovalDecision,
  ApprovalRequest,
  ApprovalRequestPayload,
  RunContext,
)
from .policy_imports import resolve_effective_role


class SingleUserApprovalPolicy:
  """The one owner of which classes need a human, for every channel.

  Approval is required only where the class is the money boundary (a live
  order). Every other class is approved here, with no prompt and no client
  round-trip; the durable request still records the decision.
  """

  policy_id = "single-user"
  policy_version = "4"

  def __init__(self, *, store: Any | None = None) -> None:
    self._store = store
    try:
      source = inspect.getsource(type(self))
    except Exception:
      source = type(self).__name__
    self.policy_bundle_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

  async def decide(
    self,
    *,
    payload: ApprovalRequestPayload,
    request: ApprovalRequest,
    run_context: RunContext,
  ) -> ApprovalDecision:
    _ = payload, run_context
    # An approval whose constraint is unknown (a row from before the column)
    # never executes (money, integrity): it is denied here and the agent
    # replans for a fresh approval.
    if request.approval_constraint == "legacy_unknown":
      return ApprovalDecision(
        outcome="auto_deny",
        reason="Approval constraint is unknown; replan and obtain a fresh approval",
        allow_persistent_grant=False,
        policy_id=self.policy_id,
        policy_version=self.policy_version,
      )
    # The money boundary: placing or cancelling a live order is the one class
    # whose decision belongs to a human, on every channel. A `fresh_human_owner`
    # request is decided by its frozen owner, not by any owner-role vote: it
    # asks here, and `approval_is_executable` lets only a human vote by
    # `required_owner_user_id` run it, on every path that waits for the vote.
    if (
      request.approval_constraint == "fresh_human_owner"
      or request.tool_class in MONEY_BOUNDARY_TOOL_CLASSES
    ):
      return ApprovalDecision(
        outcome="request_user_approval",
        reason=(
          "Exact promotion requires a fresh decision by its frozen owner"
          if request.approval_constraint == "fresh_human_owner"
          else f"{request.tool_class} tool requires explicit user approval"
        ),
        expiry_seconds=600,
        allow_persistent_grant=False,
        policy_id=self.policy_id,
        policy_version=self.policy_version,
      )
    return ApprovalDecision(
      outcome="auto_approve",
      reason="Only a live order requires a human decision",
      allow_persistent_grant=False,
      policy_id=self.policy_id,
      policy_version=self.policy_version,
    )

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    return None

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = reason
    if self._store is not None:
      await self._store.revoke_persistent_grant(grant_id)

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
    """Only the `owner` role may approve a live order.

    An order spends money in the owner's brokerage account; an `invite`
    collaborator shares the research desk, not that account, so its vote on an
    `irreversible` call is refused. Every other class needs no human at all.
    """

    if tool_class in MONEY_BOUNDARY_TOOL_CLASSES:
      return resolve_effective_role(decider_role) == "owner"
    return True
