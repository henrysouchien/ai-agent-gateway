from __future__ import annotations

import hashlib
import inspect
from datetime import timedelta
from typing import Any, Protocol

from .approval_policy import (
  DEFAULT_APPROVAL_PREFERENCE,
  MONEY_BOUNDARY_TOOL_CLASSES,
  STANDING_APPROVED_TOOL_CLASSES,
  ApprovalDecision,
  ApprovalPolicy,
  ApprovalPreference,
  ApprovalRequest,
  ApprovalRequestPayload,
  RunContext,
  ToolClass,
  preference_settles_class,
  utc_now,
)
from .policy_imports import resolve_effective_role


class StandingApprovalPreferenceReader(Protocol):
  def get(self, *, user_id: str) -> Any: ...


class SingleUserApprovalPolicy:
  """The one owner of which classes need a human, for every channel.

  Approval is required where the class is the money boundary (a live order).
  Every other class follows the user's standing approval preference, whose
  product default is ``auto_approve_all_but_trades`` — so no agent-driven
  surface draws an approval card for a read, a write or a config change.
  """

  policy_id = "single-user"
  policy_version = "2"

  def __init__(
    self,
    *,
    store: Any | None = None,
    preference_store: StandingApprovalPreferenceReader | None = None,
  ) -> None:
    self._store = store
    self._preference_store = preference_store
    try:
      source = inspect.getsource(type(self))
    except Exception:
      source = type(self).__name__
    self.policy_bundle_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

  def standing_preference(self, *, user_id: str) -> ApprovalPreference:
    """Read this user's standing answer; unset means the product default."""

    if self._preference_store is None:
      return DEFAULT_APPROVAL_PREFERENCE
    return self._preference_store.get(user_id=user_id).preference

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
    # The money boundary: placing or cancelling a live order is the one class
    # whose decision belongs to a human, on every channel.
    if request.tool_class in MONEY_BOUNDARY_TOOL_CLASSES:
      return self._request_user(
        f"{request.tool_class} tool requires explicit user approval",
        allow_persistent_grant=False,
      )

    # Every other class follows the user's standing answer, decided here and
    # nowhere else. Its default settles the call with no prompt and no client
    # round-trip; the durable request still records the decision.
    preference = self.standing_preference(user_id=request.user_id)
    if preference_settles_class(
      preference=preference,
      tool_class=request.tool_class,
    ):
      return ApprovalDecision(
        outcome="auto_approve",
        reason=f"Standing approval preference: {preference}",
        allow_persistent_grant=False,
        policy_id=self.policy_id,
        policy_version=self.policy_version,
      )

    if request.approval_reuse_mode == "disabled":
      return self._request_user(
        "Tool requires user approval",
        allow_persistent_grant=False,
      )
    scope_hint = (
      request.approval_reuse_key
      if request.approval_reuse_mode == "exact"
      else self._scope_hint(request, payload)
    )
    if not scope_hint:
      raise ValueError("approval reuse scope is unavailable")
    if self._store is not None:
      grant = await self._store.find_persistent_grant(
        user_id=request.user_id,
        tool_name=request.tool_name,
        scope_hint=scope_hint,
        approval_constraint=request.approval_constraint,
        approval_reuse_mode=request.approval_reuse_mode,
        approval_reuse_key=request.approval_reuse_key,
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
  """One minted grant, on top of the owner's rule, for delegated Excel turns.

  The grant can only add an auto-approval for the exact call an operator
  delegated; which classes need a human is the base policy's to decide, so
  there is no second class list here and no arm that escalates past it.
  """

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
    if request.approval_reuse_mode != "legacy":
      return await self._base.decide(
        payload=payload,
        request=request,
        run_context=run_context,
      )
    if request.approval_constraint != "standard":
      return await self._base.decide(
        payload=payload,
        request=request,
        run_context=run_context,
      )
    grant = run_context.delegation
    if grant is None:
      return await self._base.decide(payload=payload, request=request, run_context=run_context)

    if (
      request.tool_class in STANDING_APPROVED_TOOL_CLASSES
      and request.tool_class in grant.tool_class_ceiling
      and utc_now() <= grant.created_at + timedelta(seconds=grant.window_seconds)
      and self._predicate_matches(payload=payload, predicate=grant.args_predicate)
    ):
      return ApprovalDecision(
        outcome="auto_approve",
        reason="delegation grant matched",
        policy_id=self.policy_id,
        policy_version=self.policy_version,
      )

    return await self._base.decide(
      payload=payload,
      request=request,
      run_context=run_context,
    )

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    await self._base.on_resolve(request=request)

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    await self._base.revoke_persistent_grant(grant_id=grant_id, reason=reason)

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: ToolClass) -> bool:
    return self._base.role_authorized_for_class(decider_role=decider_role, tool_class=tool_class)

  @staticmethod
  def _predicate_matches(*, payload: ApprovalRequestPayload, predicate: dict[str, Any] | None) -> bool:
    if predicate is None:
      return True
    return all(payload.tool_args.get(key) == value for key, value in predicate.items())
