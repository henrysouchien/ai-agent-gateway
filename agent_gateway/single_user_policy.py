from __future__ import annotations

import hashlib
import inspect
from typing import Any

from .approval_policy import (
  ApprovalDecision,
  ApprovalPolicy,
  ApprovalRequest,
  ApprovalRequestPayload,
  RunContext,
)


class SingleUserApprovalPolicy:
  """Default policy preserving the existing single-user approval behavior."""

  policy_id = "single-user"
  policy_version = "1"

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
    # Portfolio configuration and irreversible tools require a fresh user decision.
    if request.tool_class in {"portfolio_config", "irreversible"}:
      return self._request_user(
        f"{request.tool_class} tool requires explicit user approval",
        allow_persistent_grant=False,
      )

    scope_hint = self._scope_hint(request, payload)
    if self._store is not None:
      grant = await self._store.find_persistent_grant(
        user_id=request.user_id,
        tool_name=request.tool_name,
        scope_hint=scope_hint,
      )
      if grant is not None:
        emitter = getattr(self._store, "audit_emitter", None)
        emit = getattr(emitter, "emit_grant_event", None) if emitter is not None else None
        if emit is not None:
          await emit(event_type="persistent_grant_used", grant=grant, request=request)
        return ApprovalDecision(
          outcome="auto_approve",
          reason="Persistent approval grant matched",
          persistent_grant_scope_hint=scope_hint,
          policy_id=self.policy_id,
          policy_version=self.policy_version,
        )

    if self._chain_trust_allows(request=request, run_context=run_context):
      return ApprovalDecision(
        outcome="auto_approve",
        reason="Parent approval chain trust matched",
        persistent_grant_scope_hint=scope_hint,
        policy_id=self.policy_id,
        policy_version=self.policy_version,
      )

    return self._request_user(
      "Tool requires user approval",
      allow_persistent_grant=request.tool_class in {"state_write", "external_write", "artifact_write"},
      scope_hint=scope_hint,
    )

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    return None

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = reason
    if self._store is not None:
      await self._store.revoke_persistent_grant(grant_id)

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
    role = str(decider_role or "owner")
    if tool_class == "irreversible":
      return role == "owner"
    return role in {"owner", "invite", "approver", "compliance", "pm"}

  def _request_user(
    self,
    reason: str,
    *,
    allow_persistent_grant: bool,
    scope_hint: str | None = None,
  ) -> ApprovalDecision:
    return ApprovalDecision(
      outcome="request_user_approval",
      reason=reason,
      route_target_type="pending_tools",
      required_decider_count=1,
      eligible_decider_count=1,
      expiry_seconds=600,
      allow_persistent_grant=allow_persistent_grant,
      persistent_grant_scope_hint=scope_hint,
      policy_id=self.policy_id,
      policy_version=self.policy_version,
    )

  @staticmethod
  def _scope_hint(request: ApprovalRequest, payload: ApprovalRequestPayload) -> str:
    qualifier = ""
    args = payload.tool_args
    for key in ("ticker", "symbol", "portfolio_id", "account_id"):
      value = args.get(key)
      if value:
        qualifier = str(value)
        break
    return f"{request.tool_class}:{request.tool_name}:{qualifier}" if qualifier else f"{request.tool_class}:{request.tool_name}"

  @staticmethod
  def _chain_trust_allows(*, request: ApprovalRequest, run_context: RunContext) -> bool:
    if not request.parent_approval_id:
      return False
    if request.tool_class in {"external_write", "portfolio_config", "irreversible"}:
      return False
    if request.user_id != run_context.user_id:
      return False
    if request.profile != run_context.profile:
      return False
    if request.channel != run_context.channel:
      return False
    if request.args_predicate is None:
      return False
    if not request.chain_trust_window_seconds:
      return False
    return True


def make_default_policy(*, store: Any | None = None) -> ApprovalPolicy:
  return SingleUserApprovalPolicy(store=store)
