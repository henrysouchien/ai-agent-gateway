from __future__ import annotations

import asyncio
from copy import deepcopy
import logging
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol, TYPE_CHECKING

from .approval_policy import (
  MONEY_BOUNDARY_TOOL_CLASSES,
  ApprovalConstraint,
  ApprovalReuseMode,
  ApprovalState,
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest as PolicyApprovalRequest,
  RunContext,
  apply_decision_to_request,
  approval_is_executable,
  approval_reuse_scope_authorized,
  approval_request_projection,
  build_approval_request,
  call_policy_safely,
  constrain_approval_decision,
  utc_now,
)
from .approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args
from .approval_route import (
  ApprovalRoute,
  ParentDelegatedApprovalRoute,
  route_ledger,
)
from .batch_approval_projection import (
  BatchApprovalAdmissionStore,
  abort_unpublished_batch_approval_admission,
  acquire_batch_approval_admission,
  require_batch_stage_run_seq,
)
from .policy_imports import resolve_server_policy_tool_class
from .secret_boundary import (
  SANITIZATION_FAILED,
  SecretBoundary,
  sanitize_approval_decision_projection,
  sanitize_boundary_value,
  sanitization_failure_tool_input,
)
from .skill_limits import reconcile_skill_admission
from . import tool_redaction as _redaction
from .tool_policy_registry import PreparedToolCall
from . import admission

if TYPE_CHECKING:
  from .prepared_business_model_store import PreparedBusinessModelChange


class ApprovalPark(Exception):
  """A physical attempt yielded with its original call still unresolved."""

  __slots__ = (
    "job_id", "original_tool_call_id", "approval_id", "nonce", "root_turn_id",
  )

  def __init__(
    self,
    *,
    job_id: str,
    original_tool_call_id: str,
    approval_id: str,
    nonce: str,
    root_turn_id: str | None = None,
  ) -> None:
    super().__init__("execution awaiting approval")
    self.job_id = job_id
    self.original_tool_call_id = original_tool_call_id
    self.approval_id = approval_id
    self.nonce = nonce
    self.root_turn_id = root_turn_id

  def __str__(self) -> str:
    return "execution awaiting approval"

  def __repr__(self) -> str:
    return "ApprovalPark()"


@dataclass(frozen=True, slots=True, repr=False)
class ExecutableCapture:
  """Private original call operands, separate from public approval data."""

  prepared_call: PreparedToolCall
  dispatch_context: Mapping[str, object]
  job_id: str | None
  tenant_storage_id: str | None
  credential_generation_ref: str | None
  root_turn: admission.RootTurnAdmission | None


class ApprovalLifecycleStore(admission.RootParkStore, Protocol):
  """Durable methods called by one local approval lifecycle."""

  async def get(
    self,
    approval_id: str,
  ) -> PolicyApprovalRequest | None: ...

  async def transition_state(
    self,
    approval_id: str,
    state: ApprovalState,
    *,
    expected_state_version: int | None = None,
    expires_at: datetime | None = None,
    decider_id: str | None = None,
    decider_role: str | None = None,
    decision: str | None = None,
    decision_reason: str | None = None,
  ) -> PolicyApprovalRequest: ...

  async def create(
    self,
    request: PolicyApprovalRequest,
  ) -> PolicyApprovalRequest: ...

  async def park_executable_binding(
    self,
    *,
    approval_id: str,
    job_id: str,
    tool_call_id: str,
    nonce: str,
    tenant_storage_id: str,
    tool_name: str,
    prepared_call: PreparedToolCall,
    dispatch_context: Mapping[str, object],
    credential_generation_ref: str | None,
  ) -> str: ...

  async def create_raw_patch_authorization(
    self,
    request: PolicyApprovalRequest,
    *,
    prepared_payload: bytes,
  ) -> PolicyApprovalRequest: ...

  async def create_or_get_with_prepared_business_model_change(
    self,
    request: PolicyApprovalRequest,
    record: PreparedBusinessModelChange,
  ) -> tuple[
    PolicyApprovalRequest,
    PreparedBusinessModelChange,
    bool,
  ]: ...

  async def update_request(
    self,
    request: PolicyApprovalRequest,
  ) -> PolicyApprovalRequest: ...

  async def get_prepared_business_model_change(
    self,
    *,
    caller_kind: str,
    user_scope: str,
    idempotency_locator: str,
  ) -> PreparedBusinessModelChange | None: ...

  async def enqueue_pending_approval_notification(
    self,
    request: PolicyApprovalRequest,
  ) -> dict[str, Any] | None: ...

  def schedule_approval_notification_delivery(
    self,
    *,
    limit: int = 50,
  ) -> bool: ...


class ApprovalLifecyclePolicy(Protocol):
  """Policy callback invoked after this lifecycle resolves a request."""

  async def on_resolve(
    self,
    *,
    request: PolicyApprovalRequest,
  ) -> None: ...


class _ApprovalLifecycleAuthority(
  ApprovalLifecycleStore,
  BatchApprovalAdmissionStore,
  Protocol,
):
  """The route aggregate needed to reach both lifecycle-local gates."""



async def _run_delegated_approval_lifecycle(
  *,
  route: ParentDelegatedApprovalRoute,
  executable_capture: ExecutableCapture | None,
  request: PolicyApprovalRequest,
  tool_input: dict[str, Any],
  qualifier: str,
  reason: str | None,
  await_user_approval_via_pending_tools_fn: Any,
  utc_now_fn: Any,
  os_urandom_fn: Any,
  run_context: RunContext,
) -> dict[str, Any]:
  """Author the request here; the parent records the row and decides it.

  This run owns no ledger and no policy, so nothing durable is written and no
  policy is evaluated: the request is projected onto the event frame the parent
  drains, and the wait lasts until the parent delivers the decision.

  A delegated run is unattended by construction — no human sits at its decision
  channel — so every class outside the money boundary is settled here under the
  run owner's own authority, the same rule the interactive policy applies
  (`MONEY_BOUNDARY_TOOL_CLASSES`, `agent_gateway.approval_policy`). Only a live
  order still asks.
  """

  if request.tool_class not in MONEY_BOUNDARY_TOOL_CLASSES:
    decided_at = utc_now_fn()
    request = replace(
      request,
      state="auto_approved",
      decision="auto_approved",
      authorization_mode="OWNER_CONTROL_PLANE",
      expires_at=None,
      decided_at=decided_at,
      decider_id=run_context.user_id,
      decider_role=run_context.decider_role,
      reason=reason or "",
    )
    return {
      "approved": True,
      "allow_tool_type": False,
      "request": request,
      "tool_input": tool_input,
    }

  request = replace(
    request,
    state="pending_user",
    authorization_mode="HUMAN",
  )
  decision = PolicyApprovalDecision(
    outcome="request_user_approval",
    reason=reason or "",
    allow_persistent_grant=False,
  )
  nonce = os_urandom_fn(8).hex()
  if executable_capture is not None and route.private_capture_writer is not None:
    prepared = executable_capture.prepared_call
    await route.private_capture_writer({
      "approval_id": request.approval_id,
      "tool_call_id": request.tool_call_id,
      "nonce": nonce,
      "tool_name": request.tool_name,
      "prepared_input": prepared.materialize_input(),
      "exact_backend": prepared.exact_backend,
      "dispatch_context": dict(executable_capture.dispatch_context),
      "job_id": executable_capture.job_id,
      "tenant_storage_id": executable_capture.tenant_storage_id,
      "credential_generation_ref": executable_capture.credential_generation_ref,
    })
  approval = await await_user_approval_via_pending_tools_fn(
    request,
    decision,
    nonce=nonce,
    resolved_qualifier=qualifier,
    allow_persistent=False,
    durable_request=approval_request_projection(request),
  )
  if approval.get("approval_id") != request.approval_id:
    raise RuntimeError(
      "delegated approval decision named a different approval"
    )
  approved = approval.get("approved")
  if type(approved) is not bool:
    raise RuntimeError(
      "delegated approval decision carried an invalid outcome"
    )
  request = replace(
    request,
    state="approved" if approved else "denied",
    decision="approved" if approved else "denied",
    decided_at=utc_now_fn(),
    decider_id=run_context.user_id,
    decider_role=run_context.decider_role,
  )
  return {
    "approved": approved,
    "allow_tool_type": False,
    "request": request,
    "tool_input": tool_input,
  }


async def run_approval_lifecycle(
  *,
  route: ApprovalRoute,
  session: Any | None,
  producer_session: Any | None,
  executable_capture: ExecutableCapture | None,
  root_turn_admission: admission.RootTurnAdmission | None = None,
  tool_call_id: str,
  tool_name: str,
  tool_input: dict[str, Any],
  qualifier: str,
  reason: str,
  allow_persistent: bool,
  approval_constraint: ApprovalConstraint = "standard",
  required_owner_user_id: str | None = None,
  approval_reuse_mode: ApprovalReuseMode = "legacy",
  approval_reuse_key: str | None = None,
  approval_identity: Mapping[str, Any] | None = None,
  prepared_authorization_payload: bytes | None = None,
  prepared_business_model_change: Mapping[str, Any] | None = None,
  resume_approval_request: PolicyApprovalRequest | None = None,
  approval_args_redacted: dict[str, Any] | None = None,
  approval_args_hash: str | None = None,
  resolve_run_context_fn: Any,
  current_skill_admission_fn: Any,
  redact_for_approval_request_fn: Any,
  resolve_tool_class_fn: Any,
  effective_trade_approval_decision_fn: Any,
  await_user_approval_via_pending_tools_fn: Any,
  build_approval_request_fn: Any = build_approval_request,
  call_policy_safely_fn: Any = call_policy_safely,
  apply_decision_to_request_fn: Any = apply_decision_to_request,
  utc_now_fn: Any = utc_now,
  os_urandom_fn: Any = os.urandom,
  secret_boundary: SecretBoundary | None = None,
) -> dict[str, Any]:
  arguments = dict(locals())
  # Batch scope belongs to the executing producer, not its parent's ledger.
  batch_admission = acquire_batch_approval_admission(arguments["producer_session"])
  arguments["batch_admission"] = batch_admission
  try:
    return await _run_approval_lifecycle_impl(**arguments)
  except BaseException:
    if batch_admission is not None and not batch_admission.published:
      await abort_unpublished_batch_approval_admission(batch_admission)
    raise
  finally:
    if batch_admission is not None:
      batch_admission.release()


async def _run_approval_lifecycle_impl(
  *,
  route: ApprovalRoute[
    _ApprovalLifecycleAuthority,
    ApprovalLifecyclePolicy,
  ],
  session: Any | None,
  producer_session: Any | None,
  executable_capture: ExecutableCapture | None,
  root_turn_admission: admission.RootTurnAdmission | None = None,
  tool_call_id: str,
  tool_name: str,
  tool_input: dict[str, Any],
  qualifier: str,
  reason: str,
  allow_persistent: bool,
  approval_constraint: ApprovalConstraint = "standard",
  required_owner_user_id: str | None = None,
  approval_reuse_mode: ApprovalReuseMode = "legacy",
  approval_reuse_key: str | None = None,
  approval_identity: Mapping[str, Any] | None = None,
  prepared_authorization_payload: bytes | None = None,
  prepared_business_model_change: Mapping[str, Any] | None = None,
  resume_approval_request: PolicyApprovalRequest | None = None,
  approval_args_redacted: dict[str, Any] | None = None,
  approval_args_hash: str | None = None,
  resolve_run_context_fn: Any,
  current_skill_admission_fn: Any,
  redact_for_approval_request_fn: Any,
  resolve_tool_class_fn: Any,
  effective_trade_approval_decision_fn: Any,
  await_user_approval_via_pending_tools_fn: Any,
  build_approval_request_fn: Any = build_approval_request,
  call_policy_safely_fn: Any = call_policy_safely,
  apply_decision_to_request_fn: Any = apply_decision_to_request,
  utc_now_fn: Any = utc_now,
  os_urandom_fn: Any = os.urandom,
  batch_admission: Any | None = None,
  secret_boundary: SecretBoundary | None = None,
) -> dict[str, Any]:
  ledger = route_ledger(route)
  if isinstance(route, ParentDelegatedApprovalRoute):
    store = None
    policy = None
  elif ledger is not None:
    store = ledger.store
    policy = ledger.policy
  else:
    raise RuntimeError("approval lifecycle has no admitted route")
  run_context = resolve_run_context_fn()
  active_admission = current_skill_admission_fn()
  reconciled_admission = reconcile_skill_admission(
    skill_name=run_context.skill,
    execution_limits=run_context.admitted_skill_execution_limits,
    active_admission=active_admission,
  )
  if reconciled_admission is not None:
    run_context = replace(
      run_context,
      skill=reconciled_admission.skill_name,
      admitted_skill_execution_limits=(
        reconciled_admission.execution_limits
      ),
    )
  if (approval_args_redacted is None) is not (approval_args_hash is None):
    raise ValueError("approval argument override requires redacted args and hash")
  if approval_args_hash is None:
    redacted, args_hash = redact_for_approval_request_fn(tool_name, tool_input)
  else:
    redacted = dict(approval_args_redacted or {})
    args_hash = approval_args_hash
  redacted = sanitize_boundary_value(
    redacted,
    sink="approval_request",
    boundary=secret_boundary,
  )
  if not isinstance(redacted, dict):
    redacted = sanitization_failure_tool_input()
  safe_reason = sanitize_boundary_value(
    reason or None,
    sink="approval_request",
    boundary=secret_boundary,
  )
  if safe_reason is not None and not isinstance(safe_reason, str):
    safe_reason = SANITIZATION_FAILED
  build_request_kwargs = {
    "tool_call_id": tool_call_id,
    "tool_name": tool_name,
    "tool_class": resolve_tool_class_fn(tool_name),
    "tool_args_redacted": redacted,
    "args_hash": args_hash,
    "run_context": run_context,
    "reason": safe_reason,
    "approval_constraint": approval_constraint,
    "required_owner_user_id": required_owner_user_id,
    "approval_reuse_mode": approval_reuse_mode,
    "approval_reuse_key": approval_reuse_key,
  }
  if approval_identity is not None:
    build_request_kwargs["approval_identity"] = approval_identity
  request = (
    resume_approval_request
    if resume_approval_request is not None
    else build_approval_request_fn(**build_request_kwargs)
  )
  if resume_approval_request is not None and (
    request.approval_constraint != approval_constraint
    or request.required_owner_user_id != required_owner_user_id
    or request.approval_reuse_mode != approval_reuse_mode
    or request.approval_reuse_key != approval_reuse_key
  ):
    raise RuntimeError(
      "prepared approval constraint changed; replan and reauthorize"
    )
  if request.approval_reuse_mode == "disabled":
    allow_persistent = False
  if batch_admission is not None:
    batch_admission.bind_request(request=request, store=store)
  if hasattr(request, "__dataclass_fields__"):
    policy_id = (
      getattr(policy, "policy_id", None)
      or getattr(request, "policy_id", None)
      or "single-user"
    )
    policy_version = (
      getattr(policy, "policy_version", None)
      or getattr(request, "policy_version", None)
      or "1"
    )
    request = replace(
      request,
      policy_id=str(policy_id),
      policy_version=str(policy_version),
    )
  if isinstance(route, ParentDelegatedApprovalRoute):
    return await _run_delegated_approval_lifecycle(
      route=route,
      executable_capture=executable_capture,
      request=request,
      tool_input=tool_input,
      qualifier=qualifier,
      reason=safe_reason,
      await_user_approval_via_pending_tools_fn=(
        await_user_approval_via_pending_tools_fn
      ),
      utc_now_fn=utc_now_fn,
      os_urandom_fn=os_urandom_fn,
      run_context=run_context,
    )
  assert ledger is not None
  store = ledger.store
  policy = ledger.policy
  if resume_approval_request is not None:
    if prepared_business_model_change is None:
      raise RuntimeError("prepared approval resume requires its BusinessModel identity")
  elif prepared_business_model_change is not None:
    from .prepared_business_model_store import (
      PreparedBusinessModelChange,
      PreparedBusinessModelLifecycle,
    )

    prepared_bytes = prepared_business_model_change["prepared_payload"]
    if not isinstance(prepared_bytes, bytes):
      raise RuntimeError("prepared BusinessModel authorization payload must be bytes")
    existing = await store.get_prepared_business_model_change(
      caller_kind=str(prepared_business_model_change["caller_kind"]),
      user_scope=str(prepared_business_model_change["user_scope"]),
      idempotency_locator=str(prepared_business_model_change["idempotency_locator"]),
    )
    if existing is None:
      request, existing, _ = await store.create_or_get_with_prepared_business_model_change(
        request,
        PreparedBusinessModelChange(
          schema_version="prepared-business-model-change.v1",
          caller_kind=str(prepared_business_model_change["caller_kind"]),
          user_scope=str(prepared_business_model_change["user_scope"]),
          idempotency_locator=str(prepared_business_model_change["idempotency_locator"]),
          intent_digest=str(prepared_business_model_change["intent_digest"]),
          prepared_payload=prepared_bytes,
          change_set_id=str(prepared_business_model_change["change_set_id"]),
          change_hash=str(prepared_business_model_change["change_hash"]),
          base_vector_hash=str(prepared_business_model_change["base_vector_hash"]),
          prepared_payload_digest=str(prepared_business_model_change["prepared_payload_digest"]),
          lifecycle=PreparedBusinessModelLifecycle.PENDING,
          created_at=request.requested_at.isoformat(),
          expires_at=(request.expires_at.isoformat() if request.expires_at is not None else None),
          approval_id=request.approval_id,
          approval_chain_id=request.approval_chain_id,
        ),
      )
    else:
      supplied_intent = str(prepared_business_model_change["intent_digest"])
      terminal_non_executable = existing.lifecycle in {
        PreparedBusinessModelLifecycle.DENIED,
        PreparedBusinessModelLifecycle.EXPIRED,
        PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT,
      }
      immutable_matches = (
        existing.prepared_payload == prepared_bytes
        and existing.change_set_id == str(prepared_business_model_change["change_set_id"])
        and existing.change_hash == str(prepared_business_model_change["change_hash"])
        and existing.base_vector_hash == str(prepared_business_model_change["base_vector_hash"])
        and existing.prepared_payload_digest
        == str(prepared_business_model_change["prepared_payload_digest"])
      )
      if existing.intent_digest != supplied_intent or (
        not terminal_non_executable and not immutable_matches
      ):
        raise RuntimeError("FMS skill-run locator conflicts with its prepared BusinessModel intent")
      await store.create(request)
  elif prepared_authorization_payload is None:
    await store.create(request)
  else:
    create_bound = getattr(store, "create_raw_patch_authorization", None)
    if not callable(create_bound):
      raise RuntimeError("approval store cannot persist prepared authorization bytes")
    await store.create_raw_patch_authorization(
      request,
      prepared_payload=prepared_authorization_payload,
    )

  raw_args = dict(tool_input)
  try:
    decision: PolicyApprovalDecision = await call_policy_safely_fn(policy, request, raw_args, run_context)
  finally:
    raw_args.clear()
    del raw_args

  decision = effective_trade_approval_decision_fn(tool_name, request.tool_args_redacted, decision)
  decision = constrain_approval_decision(request, decision)
  decision, raw_modified_tool_args = sanitize_approval_decision_projection(
    decision,
    sink="approval_decision",
    boundary=secret_boundary,
  )
  policy_modified_tool_args = raw_modified_tool_args is not None
  request = apply_decision_to_request_fn(request, decision)
  await store.update_request(request)
  final_tool_input = (
    raw_modified_tool_args
    if raw_modified_tool_args is not None
    else tool_input
  )
  allow_persistent = allow_persistent and not policy_modified_tool_args

  if decision.outcome == "auto_approve":
    request = await store.transition_state(
      request.approval_id,
      "auto_approved",
      expected_state_version=request.state_version,
      decider_id=run_context.user_id,
      decider_role=run_context.decider_role,
      decision_reason=decision.reason,
    )
    await policy.on_resolve(request=request)
    return {
      "approved": True,
      "allow_tool_type": False,
      "request": request,
      "tool_input": final_tool_input,
      "policy_modified_tool_args": policy_modified_tool_args,
    }

  if decision.outcome == "auto_deny":
    request = await store.transition_state(
      request.approval_id,
      "auto_denied",
      expected_state_version=request.state_version,
      decider_id=run_context.user_id,
      decider_role=run_context.decider_role,
      decision_reason=decision.reason,
    )
    await policy.on_resolve(request=request)
    return {
      "approved": False,
      "allow_tool_type": False,
      "request": request,
      "policy_modified_tool_args": policy_modified_tool_args,
    }

  nonce = os_urandom_fn(8).hex()
  root_turn = (
    executable_capture.root_turn
    if executable_capture is not None
    else root_turn_admission
  )
  park: ApprovalPark | None = None
  if executable_capture is not None:
    if executable_capture.job_id is None or executable_capture.tenant_storage_id is None:
      logging.getLogger("agent_gateway.approval_capture").warning(
        "approval_capture_without_job_context | tool_name=%s | tool_call_id=%s"
        " | missing_job_id=%s | missing_tenant_storage_id=%s",
        tool_name,
        tool_call_id,
        executable_capture.job_id is None,
        executable_capture.tenant_storage_id is None,
      )
    else:
      # A re-park (a resumed request, or a call id parking again) keeps the
      # durable binding's identity; the store returns the nonce to publish.
      nonce = await store.park_executable_binding(
        approval_id=request.approval_id,
        job_id=executable_capture.job_id,
        tool_call_id=tool_call_id,
        nonce=nonce,
        tenant_storage_id=executable_capture.tenant_storage_id,
        tool_name=tool_name,
        prepared_call=executable_capture.prepared_call,
        dispatch_context=executable_capture.dispatch_context,
        credential_generation_ref=executable_capture.credential_generation_ref,
      )
      # The bound call's decision reaches it through its Continue hook.
      park = ApprovalPark(
        job_id=executable_capture.job_id,
        original_tool_call_id=tool_call_id,
        approval_id=request.approval_id,
        nonce=nonce,
        root_turn_id=root_turn.turn_id if root_turn is not None else None,
      )

  if root_turn is not None:
    await admission.link_root_park(
      store,
      root_turn,
      tool_call_id=tool_call_id,
      nonce=nonce,
    )

  request = await store.transition_state(
    request.approval_id,
    "pending_user",
    expires_at=(
      utc_now_fn() + timedelta(seconds=decision.expiry_seconds)
      if decision.expiry_seconds is not None
      else None
    ),
    expected_state_version=request.state_version,
  )
  enqueue_notification = getattr(store, "enqueue_pending_approval_notification", None)
  if enqueue_notification is not None:
    try:
      await enqueue_notification(request)
      schedule_notification_delivery = getattr(store, "schedule_approval_notification_delivery", None)
      if schedule_notification_delivery is not None:
        schedule_notification_delivery()
    except Exception:
      logging.getLogger("agent_gateway.approval_notifications").warning(
        "Failed to enqueue approval notification intent for %s | failure=true",
        request.approval_id,
      )
  approval = await await_user_approval_via_pending_tools_fn(
    request,
    decision,
    nonce=nonce,
    resolved_qualifier=qualifier,
    allow_persistent=(allow_persistent and request.approval_constraint == "standard"),
    batch_admission=batch_admission,
    park=park,
  )
  latest = await store.get(request.approval_id)
  if latest is not None:
    request = latest
  executable = approval_is_executable(request)
  return {
    "approved": executable,
    "allow_tool_type": (
      executable
      and request.approval_constraint == "standard"
      and approval_reuse_scope_authorized(request)
      and bool(approval.get("allow_tool_type"))
    ),
    "request": request,
    "tool_input": final_tool_input,
    "policy_modified_tool_args": policy_modified_tool_args,
  }


async def await_user_approval_via_pending_tools(
  *,
  session: Any,
  append_event_fn: Callable[[dict[str, Any]], Any] | None,
  request: PolicyApprovalRequest,
  decision: PolicyApprovalDecision,
  nonce: str,
  resolved_qualifier: str,
  allow_persistent: bool,
  log: logging.Logger,
  batch_admission: Any | None = None,
  durable_request: dict[str, Any] | None = None,
  park: ApprovalPark | None = None,
) -> dict[str, Any]:
  """Publish the ask, then park the call or wait on the session's live queue.

  A call with a durable executable binding raises its ``park`` once the ask is
  published: the turn ends awaiting approval and the decision reaches the
  original call through its Continue hook whenever it arrives. A call without
  one has nothing to continue from and waits here, unbounded by any timer, until
  the decision arrives or the owning turn cancels this task. The durable request
  stays pending until decided, and the live entry is removed whichever way this
  ends.
  """

  approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
  pending_entry = {
    "approval_id": request.approval_id,
    "nonce": nonce,
    "requested_at": int(time.time()),
    "status": "approval_pending",
    "tool_name": request.tool_name,
    "resolved_qualifier": resolved_qualifier,
  }
  planned_change = request.tool_args_redacted.get("planned_change")
  if isinstance(planned_change, dict):
    # Control-plane stage harnesses need only the trusted plan projection. Do
    # not duplicate the agent-authored judgment (which can be very large) into
    # the live pending-tool registry.
    pending_entry["planned_change"] = deepcopy(planned_change)
  if batch_admission is not None:
    pending_entry["stage_run_seq"] = require_batch_stage_run_seq(session)
  session.pending_tools[request.tool_call_id] = pending_entry
  session.approval_queues[request.tool_call_id] = approval_queue
  try:
    if batch_admission is not None:
      batch_admission.publish_pending()

    approval_event = build_tool_approval_request_event(
      request,
      nonce=nonce,
      resolved_qualifier=resolved_qualifier,
      reason=decision.reason,
      allow_persistent_approval=allow_persistent and decision.allow_persistent_grant,
    )
    if batch_admission is not None:
      approval_event["stage_run_seq"] = pending_entry["stage_run_seq"]
    if durable_request is not None:
      # The delegated route's up-path: the parent creates the durable row from
      # this projection before the frame becomes operator-visible.
      approval_event["durable_request"] = durable_request
    if append_event_fn is not None:
      append_event_fn(approval_event)
    session_log = getattr(session, "agent_session_log", None)
    if session_log is not None:
      try:
        await session_log.append(approval_event)
      except Exception:
        log.warning(
          "Failed to persist approval request event for %s | failure=true",
          request.tool_call_id,
        )

    if park is not None:
      raise park
    return await approval_queue.get()
  finally:
    session.pending_tools.pop(request.tool_call_id, None)
    session.approval_queues.pop(request.tool_call_id, None)


def build_tool_approval_request_event(
  request: PolicyApprovalRequest,
  *,
  nonce: str,
  resolved_qualifier: str,
  reason: str | None,
  allow_persistent_approval: bool,
) -> dict[str, Any]:
  """The ask a parked or waiting call publishes to its live and durable logs."""
  return {
    "type": "tool_approval_request",
    "tool_call_id": request.tool_call_id,
    "approval_id": request.approval_id,
    "nonce": nonce,
    "tool_name": request.tool_name,
    "tool_input": request.tool_args_redacted,
    "resolved_qualifier": resolved_qualifier,
    "reason": reason,
    "allow_persistent_approval": allow_persistent_approval,
    "ts": time.time(),
  }


def redact_for_approval_request(
  tool_name: str,
  tool_input: dict[str, Any],
  *,
  event_log: Any | None,
  enrich_trade_approval_args_fn: Any = enrich_trade_approval_args,
) -> tuple[dict[str, Any], str]:
  secret = _redaction.get_audit_hmac_secret()
  key_id = _redaction.get_audit_hmac_key_id()
  redacted = _redaction.redact_tool_input(
    tool_name,
    tool_input,
    deployment_secret=secret,
    key_id=key_id,
    redaction_scope="fresh_raw",
  )
  redacted = enrich_trade_approval_args_fn(tool_name, redacted, event_log=event_log)
  args_hash = hash_approval_arguments(tool_input)
  return redacted, args_hash


def hash_approval_arguments(tool_input: dict[str, Any]) -> str:
  secret = _redaction.get_audit_hmac_secret()
  key_id = _redaction.get_audit_hmac_key_id()
  return _redaction.hmac_value(
    tool_input,
    deployment_secret=secret,
    key_id=key_id,
  )


def approval_transport_input(
  tool_name: str,
  tool_input: dict[str, Any],
  *,
  event_log: Any | None,
  enrich_trade_approval_args_fn: Any = enrich_trade_approval_args,
) -> dict[str, Any]:
  return enrich_trade_approval_args_fn(tool_name, dict(tool_input), event_log=event_log)


def resolve_run_context(
  *,
  run_context: RunContext | None,
  session: Any | None,
  user_id: str | None,
  channel: str | None,
  role: str | None,
  session_id: str,
  approval_policy: object | None,
) -> RunContext:
  if run_context is not None:
    owner_user_id = str(getattr(session, "owner_user_id", None) or "").strip()
    if owner_user_id and run_context.user_id != owner_user_id:
      return replace(run_context, user_id=owner_user_id)
    return run_context
  resolved_user_id = str(
    getattr(session, "owner_user_id", None)
    or user_id
    or getattr(session, "user_id", "")
    or "unknown"
  )
  resolved_channel = str(channel or getattr(session, "channel", None) or "web")
  return RunContext(
    user_id=resolved_user_id,
    request_id=str(getattr(session, "request_id", "") or session_id or "request-unknown"),
    session_id=session_id or getattr(session, "session_id", None),
    profile="chat",
    channel=resolved_channel,
    decider_role=role,
    policy_bundle_hash=str(getattr(approval_policy, "policy_bundle_hash", "unknown")),
  )


def resolve_tool_class(
  tool_name: str,
  *,
  mcp: Any,
  resolve_server_policy_tool_class_fn: Any = resolve_server_policy_tool_class,
) -> str:
  is_mcp_tool = getattr(mcp, "is_mcp_tool", None)
  get_server_for_tool = getattr(mcp, "get_server_for_tool", None)
  server = (
    get_server_for_tool(tool_name)
    if callable(is_mcp_tool) and callable(get_server_for_tool) and is_mcp_tool(tool_name)
    else None
  )
  policy_tool_name = tool_name
  if server:
    original_tool_name = getattr(mcp, "get_original_tool_name", None)
    if callable(original_tool_name):
      policy_tool_name = original_tool_name(tool_name)
  return resolve_server_policy_tool_class_fn(
    tool_name,
    policy_tool_name=policy_tool_name,
    runtime_server=server,
  )


def effective_trade_approval_decision(
  tool_name: str,
  tool_args_redacted: dict[str, Any],
  decision: PolicyApprovalDecision,
  *,
  effective_trade_approval_expiry_seconds_fn: Any = effective_trade_approval_expiry_seconds,
  utc_now_fn: Any = utc_now,
) -> PolicyApprovalDecision:
  expiry_seconds = effective_trade_approval_expiry_seconds_fn(
    tool_name,
    tool_args_redacted,
    requested_expiry_seconds=decision.expiry_seconds,
    now=utc_now_fn(),
  )
  if expiry_seconds == decision.expiry_seconds:
    return decision
  return replace(decision, expiry_seconds=expiry_seconds)
