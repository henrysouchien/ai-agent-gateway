from __future__ import annotations

import hashlib
import inspect
from datetime import timedelta
from typing import Any

from .approval_policy import (
  ApprovalDecision,
  ApprovalPolicy,
  ApprovalRequest,
  ApprovalRequestPayload,
  RunContext,
  ToolClass,
  utc_now,
)
from .policy_imports import resolve_effective_role


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
    if request.approval_constraint == "legacy_unknown":
      return ApprovalDecision(
        outcome="auto_deny",
        reason="Approval constraint is unknown; replan and obtain a fresh approval",
        allow_persistent_grant=False,
        policy_id=self.policy_id,
        policy_version=self.policy_version,
      )
    if request.approval_constraint == "fresh_human_owner":
      return self._request_user(
        "Exact promotion requires a fresh decision by its frozen owner",
        allow_persistent_grant=False,
      )
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
        approval_constraint=request.approval_constraint,
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
          grant_reference=grant.grant_id,
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
    role = resolve_effective_role(decider_role)
    if tool_class == "irreversible":
      return role == "owner"
    return role in {"owner", "invite"}

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

class DelegationApprovalPolicy:
  """Approval policy wrapper for server-authoritative delegated Excel turns."""

  policy_id = "delegation"

  def __init__(self, *, base: ApprovalPolicy) -> None:
    self._base = base
    try:
      source = inspect.getsource(type(self))
    except Exception:
      source = type(self).__name__
    base_hash = getattr(base, "policy_bundle_hash", "unknown")
    self._policy_bundle_hash = hashlib.sha256((source + base_hash).encode("utf-8")).hexdigest()

  @property
  def policy_version(self) -> str:
    return getattr(self._base, "policy_version", "1")

  @property
  def policy_bundle_hash(self) -> str:
    return self._policy_bundle_hash

  async def decide(
    self,
    *,
    payload: ApprovalRequestPayload,
    request: ApprovalRequest,
    run_context: RunContext,
  ) -> ApprovalDecision:
    if request.approval_constraint != "standard":
      return await self._base.decide(
        payload=payload,
        request=request,
        run_context=run_context,
      )
    grant = run_context.delegation
    if grant is None:
      return await self._base.decide(payload=payload, request=request, run_context=run_context)

    if request.tool_class in {"portfolio_config", "irreversible"}:
      return self._request_user(f"{request.tool_class} tool requires explicit user approval")

    if request.tool_class == "external_write":
      return self._request_user("external_write tool requires explicit user approval under delegation")

    if (
      request.tool_class in grant.tool_class_ceiling
      and utc_now() <= grant.created_at + timedelta(seconds=grant.window_seconds)
      and self._predicate_matches(payload=payload, predicate=grant.args_predicate)
    ):
      return ApprovalDecision(
        outcome="auto_approve",
        reason="delegation grant matched",
        policy_id=self.policy_id,
        policy_version=self.policy_version,
      )

    return self._request_user("Tool requires user approval under delegation")

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    await self._base.on_resolve(request=request)

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    await self._base.revoke_persistent_grant(grant_id=grant_id, reason=reason)

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: ToolClass) -> bool:
    return self._base.role_authorized_for_class(decider_role=decider_role, tool_class=tool_class)

  def _request_user(self, reason: str) -> ApprovalDecision:
    return ApprovalDecision(
      outcome="request_user_approval",
      reason=reason,
      expiry_seconds=600,
      allow_persistent_grant=False,
      policy_id=self.policy_id,
      policy_version=self.policy_version,
    )

  @staticmethod
  def _predicate_matches(*, payload: ApprovalRequestPayload, predicate: dict[str, Any] | None) -> bool:
    if predicate is None:
      return True
    return all(payload.tool_args.get(key) == value for key, value in predicate.items())


def make_default_policy(*, store: Any | None = None) -> ApprovalPolicy:
  return DelegationApprovalPolicy(base=SingleUserApprovalPolicy(store=store))
