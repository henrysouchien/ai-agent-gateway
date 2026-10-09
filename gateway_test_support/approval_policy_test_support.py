"""Approval policy doubles shared by gateway approval tests."""

from __future__ import annotations

from typing import Any, Literal

from agent_gateway.approval_policy import ApprovalDecision
from agent_gateway.single_user_policy import SingleUserApprovalPolicy


class FixedOutcomeApprovalPolicy(SingleUserApprovalPolicy):
  """Decide every approval with one configured policy outcome."""

  def __init__(
    self,
    outcome: Literal["request_user_approval", "auto_approve", "auto_deny"],
  ) -> None:
    super().__init__()
    self.outcome: Literal["request_user_approval", "auto_approve", "auto_deny"] = outcome

  async def decide(self, **_kwargs: Any) -> ApprovalDecision:
    return ApprovalDecision(outcome=self.outcome, reason="capture test")
