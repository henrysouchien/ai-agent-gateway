from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import timedelta
from dataclasses import replace
from typing import Any, Mapping

from . import approval_settings
from .approval_policy import (
  ApprovalConstraint,
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest as PolicyApprovalRequest,
  RunContext,
  apply_decision_to_request,
  approval_is_executable,
  build_approval_request,
  call_policy_safely,
  constrain_approval_decision,
  sha256_args,
  utc_now,
)
from .approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args
from .batch_approval_projection import (
  abort_unpublished_batch_approval_admission,
  acquire_batch_approval_admission,
  require_batch_stage_run_seq,
)
from .policy_imports import resolve_server_policy_tool_class


def _approval_wait_timeout_seconds(
  expiry_seconds: float | int | None,
  *,
  batch_admission: Any | None,
  approval_queue_timeout_seconds_fn: Any,
) -> float:
  """Keep projected batch waits aligned with their advertised durable expiry.

  The ordinary queue timeout remains capped by the global approval-wait setting,
  which deliberately protects short-lived trade previews. A projected in-process
  batch approval already has a policy-bounded durable expiry and no independent
  process bridge, so ending its queue wait earlier strands the live stage while
  the control API still advertises a pending request. Trade approvals remain safe
  because ``effective_trade_approval_decision`` first caps their own expiry to the
  preview lifetime.
  """

  if batch_admission is None:
    return float(approval_queue_timeout_seconds_fn(expiry_seconds))
  return float(expiry_seconds or 600)


async def run_approval_lifecycle(
  *,
  store: Any | None,
  policy: Any | None,
  session: Any | None,
  tool_call_id: str,
  tool_name: str,
  tool_input: dict[str, Any],
  qualifier: str,
  reason: str,
  allow_persistent: bool,
  approval_constraint: ApprovalConstraint = "standard",
  required_owner_user_id: str | None = None,
  approval_identity: Mapping[str, Any] | None = None,
  prepared_authorization_payload: bytes | None = None,
  prepared_business_model_change: Mapping[str, Any] | None = None,
  resume_approval_request: PolicyApprovalRequest | None = None,
  approval_args_redacted: dict[str, Any] | None = None,
  approval_args_hash: str | None = None,
  session_cache_approved: bool = False,
  automatic_approval_reason: str | None = None,
  automatic_denial_reason: str | None = None,
  deny_user_prompt: bool = False,
  resolve_run_context_fn: Any,
  current_skill_fn: Any,
  redact_for_approval_request_fn: Any,
  resolve_tool_class_fn: Any,
  effective_trade_approval_decision_fn: Any,
  await_user_approval_via_pending_tools_fn: Any,
  approval_queue_timeout_seconds_fn: Any,
  build_approval_request_fn: Any = build_approval_request,
  call_policy_safely_fn: Any = call_policy_safely,
  apply_decision_to_request_fn: Any = apply_decision_to_request,
  utc_now_fn: Any = utc_now,
  os_urandom_fn: Any = os.urandom,
) -> dict[str, Any]:
  arguments = dict(locals())
  batch_admission = acquire_batch_approval_admission(session)
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
  store: Any | None,
  policy: Any | None,
  session: Any | None,
  tool_call_id: str,
  tool_name: str,
  tool_input: dict[str, Any],
  qualifier: str,
  reason: str,
  allow_persistent: bool,
  approval_constraint: ApprovalConstraint = "standard",
  required_owner_user_id: str | None = None,
  approval_identity: Mapping[str, Any] | None = None,
  prepared_authorization_payload: bytes | None = None,
  prepared_business_model_change: Mapping[str, Any] | None = None,
  resume_approval_request: PolicyApprovalRequest | None = None,
  approval_args_redacted: dict[str, Any] | None = None,
  approval_args_hash: str | None = None,
  session_cache_approved: bool = False,
  automatic_approval_reason: str | None = None,
  automatic_denial_reason: str | None = None,
  deny_user_prompt: bool = False,
  resolve_run_context_fn: Any,
  current_skill_fn: Any,
  redact_for_approval_request_fn: Any,
  resolve_tool_class_fn: Any,
  effective_trade_approval_decision_fn: Any,
  await_user_approval_via_pending_tools_fn: Any,
  approval_queue_timeout_seconds_fn: Any,
  build_approval_request_fn: Any = build_approval_request,
  call_policy_safely_fn: Any = call_policy_safely,
  apply_decision_to_request_fn: Any = apply_decision_to_request,
  utc_now_fn: Any = utc_now,
  os_urandom_fn: Any = os.urandom,
  batch_admission: Any | None = None,
) -> dict[str, Any]:
  if store is None or policy is None:
    raise RuntimeError("approval lifecycle is not configured")
  run_context = resolve_run_context_fn()
  active_skill = current_skill_fn()
  if active_skill and run_context.skill is None:
    run_context = replace(run_context, skill=active_skill)
  if (approval_args_redacted is None) is not (approval_args_hash is None):
    raise ValueError("approval argument override requires redacted args and hash")
  if approval_args_hash is None:
    redacted, args_hash = redact_for_approval_request_fn(tool_name, tool_input)
  else:
    redacted = dict(approval_args_redacted or {})
    args_hash = approval_args_hash
  build_request_kwargs = {
    "tool_call_id": tool_call_id,
    "tool_name": tool_name,
    "tool_class": resolve_tool_class_fn(tool_name),
    "tool_args_redacted": redacted,
    "args_hash": args_hash,
    "run_context": run_context,
    "reason": reason or None,
    "approval_constraint": approval_constraint,
    "required_owner_user_id": required_owner_user_id,
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
  ):
    raise RuntimeError(
      "prepared approval constraint changed; replan and reauthorize"
    )
  if request.approval_constraint == "legacy_unknown":
    session_cache_approved = False
    automatic_approval_reason = None
    allow_persistent = False
  elif request.approval_constraint == "fresh_human_owner":
    session_cache_approved = False
    automatic_approval_reason = None
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
    if session_cache_approved:
      request = replace(
        request,
        authorization_mode="CACHE_HIT",
        cache_reference=f"session-cache:{tool_name}:{qualifier}",
      )
    elif automatic_approval_reason is not None:
      request = replace(request, authorization_mode="AUTO_ALLOW")
  if resume_approval_request is not None:
    if prepared_business_model_change is None:
      raise RuntimeError("prepared approval resume requires its BusinessModel identity")
  elif prepared_business_model_change is not None:
    from .prepared_business_model_store import (
      PreparedBusinessModelChange,
      PreparedBusinessModelLifecycle,
    )

    if request.expires_at is None:
      request = replace(request, expires_at=request.requested_at + timedelta(minutes=15))
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
    await create_bound(
      request,
      prepared_payload=prepared_authorization_payload,
    )

  if session_cache_approved or automatic_approval_reason is not None:
    decision_reason = (
      automatic_approval_reason
      or "Session approval cache matched exact planned identity"
    )
    decision_source = (
      "headless_hook_approved"
      if automatic_approval_reason is not None
      else "session_cache_approved"
    )
    request = await store.transition_state(
      request.approval_id,
      "auto_approved",
      expected_state_version=request.state_version,
      decider_id=run_context.user_id,
      decider_role=run_context.decider_role,
      decision_reason=decision_reason,
    )
    await policy.on_resolve(request=request)
    return {
      "approved": True,
      "allow_tool_type": False,
      "request": request,
      "tool_input": tool_input,
      "decision_source": decision_source,
    }

  if automatic_denial_reason is not None:
    request = await store.transition_state(
      request.approval_id,
      "auto_denied",
      expected_state_version=request.state_version,
      decider_id=run_context.user_id,
      decider_role=run_context.decider_role,
      decision_reason=automatic_denial_reason,
    )
    await policy.on_resolve(request=request)
    return {
      "approved": False,
      "allow_tool_type": False,
      "request": request,
      "denied_by": "headless_policy",
      "decision_source": "headless_auto_deny",
    }

  raw_args = dict(tool_input)
  try:
    decision: PolicyApprovalDecision = await call_policy_safely_fn(policy, request, raw_args, run_context)
  finally:
    raw_args.clear()
    del raw_args

  decision = effective_trade_approval_decision_fn(tool_name, request.tool_args_redacted, decision)
  decision = constrain_approval_decision(request, decision)
  request = apply_decision_to_request_fn(request, decision)
  await store.update_request(request)
  final_tool_input = decision.modified_tool_args if decision.modified_tool_args is not None else tool_input
  policy_modified_tool_args = decision.modified_tool_args is not None

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

  if decision.outcome == "route_external":
    expires_at = utc_now_fn() + timedelta(seconds=decision.expiry_seconds or 600)
    request = await store.transition_state(
      request.approval_id,
      "routed_external",
      route_target=decision.route_target,
      route_target_type=decision.route_target_type,
      expires_at=expires_at,
      expected_state_version=request.state_version,
    )
    return {
      "approved": False,
      "allow_tool_type": False,
      "request": request,
      "timeout": True,
      "policy_modified_tool_args": policy_modified_tool_args,
    }

  if deny_user_prompt:
    request = await store.transition_state(
      request.approval_id,
      "auto_denied",
      expected_state_version=request.state_version,
      decider_id=run_context.user_id,
      decider_role=run_context.decider_role,
      decision_reason="Headless context cannot collect required user approval",
    )
    await policy.on_resolve(request=request)
    return {
      "approved": False,
      "allow_tool_type": False,
      "request": request,
      "denied_by": "headless_policy",
      "decision_source": "headless_auto_deny",
      "policy_modified_tool_args": policy_modified_tool_args,
    }

  expires_at = utc_now_fn() + timedelta(seconds=decision.expiry_seconds or 600)
  request = await store.transition_state(
    request.approval_id,
    "pending_user",
    route_target_type="pending_tools",
    expires_at=expires_at,
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
        "Failed to enqueue approval notification intent for %s",
        request.approval_id,
        exc_info=True,
      )
  nonce = os_urandom_fn(8).hex()
  approval = await await_user_approval_via_pending_tools_fn(
    request,
    decision,
    nonce=nonce,
    resolved_qualifier=qualifier,
    allow_persistent=(allow_persistent and request.approval_constraint == "standard"),
    timeout_seconds=_approval_wait_timeout_seconds(
      decision.expiry_seconds,
      batch_admission=batch_admission,
      approval_queue_timeout_seconds_fn=approval_queue_timeout_seconds_fn,
    ),
    batch_admission=batch_admission,
  )
  if approval is None:
    return {"approved": False, "allow_tool_type": False, "request": request, "timeout": True}
  latest = await store.get(request.approval_id)
  if latest is not None:
    request = latest
  executable = approval_is_executable(request)
  return {
    "approved": executable,
    "allow_tool_type": (
      executable
      and request.approval_constraint == "standard"
      and bool(approval.get("allow_tool_type"))
    ),
    "denied_by": approval.get("denied_by"),
    "request": request,
    "tool_input": final_tool_input,
    "policy_modified_tool_args": policy_modified_tool_args,
  }


async def await_user_approval_via_pending_tools(
  *,
  session: Any | None,
  approval_store: Any | None,
  event_log: Any | None,
  request: PolicyApprovalRequest,
  decision: PolicyApprovalDecision,
  nonce: str,
  resolved_qualifier: str,
  allow_persistent: bool,
  timeout_seconds: float,
  log: logging.Logger,
  batch_admission: Any | None = None,
) -> dict[str, Any] | None:
  if session is None:
    return None
  approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
  pending_entry = {
    "approval_id": request.approval_id,
    "nonce": nonce,
    "requested_at": int(time.time()),
    "status": "approval_pending",
    "tool_name": request.tool_name,
    "resolved_qualifier": resolved_qualifier,
  }
  if batch_admission is not None:
    pending_entry["stage_run_seq"] = require_batch_stage_run_seq(session)
  session.pending_tools[request.tool_call_id] = pending_entry
  session.approval_queues[request.tool_call_id] = approval_queue
  if batch_admission is not None:
    try:
      batch_admission.publish_pending()
    except BaseException:
      session.pending_tools.pop(request.tool_call_id, None)
      session.approval_queues.pop(request.tool_call_id, None)
      raise

  approval_event = {
    "type": "tool_approval_request",
    "tool_call_id": request.tool_call_id,
    "approval_id": request.approval_id,
    "nonce": nonce,
    "tool_name": request.tool_name,
    "tool_input": request.tool_args_redacted,
    "resolved_qualifier": resolved_qualifier,
    "reason": decision.reason,
    "allow_persistent_approval": allow_persistent and decision.allow_persistent_grant,
    "ts": time.time(),
  }
  if batch_admission is not None:
    approval_event["stage_run_seq"] = pending_entry["stage_run_seq"]
  if event_log is not None:
    event_log.append(approval_event)
  session_log = getattr(session, "agent_session_log", None)
  if session_log is not None:
    try:
      await session_log.append(approval_event)
    except Exception:
      log.warning("Failed to persist approval request event for %s", request.tool_call_id, exc_info=True)

  try:
    return await asyncio.wait_for(approval_queue.get(), timeout=max(0.1, timeout_seconds))
  except asyncio.TimeoutError:
    if approval_store is not None:
      try:
        latest = await approval_store.get(request.approval_id)
        if latest is not None and latest.state == "pending_user":
          await approval_store.transition_state(
            latest.approval_id,
            "expired",
            expected_state_version=latest.state_version,
            decision_reason="Timed out waiting for user approval",
          )
      except Exception:
        log.warning("Failed to expire timed-out approval request %s", request.approval_id, exc_info=True)
    return None
  finally:
    session.pending_tools.pop(request.tool_call_id, None)
    session.approval_queues.pop(request.tool_call_id, None)


def redact_for_approval_request(
  tool_name: str,
  tool_input: dict[str, Any],
  *,
  event_log: Any | None,
  enrich_trade_approval_args_fn: Any = enrich_trade_approval_args,
  sha256_args_fn: Any = sha256_args,
) -> tuple[dict[str, Any], str]:
  try:
    from agent.shared.tool_redaction import get_audit_hmac_secret, get_audit_hmac_key_id, hmac_value, redact_tool_input

    secret = get_audit_hmac_secret()
    key_id = get_audit_hmac_key_id()
    redacted = redact_tool_input(
      tool_name,
      tool_input,
      deployment_secret=secret,
      key_id=key_id,
      redaction_scope="fresh_raw",
    )
    redacted = enrich_trade_approval_args_fn(tool_name, redacted, event_log=event_log)
    args_hash = hmac_value(tool_input, deployment_secret=secret, key_id=key_id)
    return redacted, args_hash
  except Exception:
    return {}, sha256_args_fn(tool_input)


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
  approval_policy: Any | None,
) -> RunContext:
  if run_context is not None:
    return run_context
  resolved_user_id = str(user_id or getattr(session, "user_id", "") or "unknown")
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
  cls = resolve_server_policy_tool_class_fn(
    tool_name,
    policy_tool_name=policy_tool_name,
    runtime_server=server,
    default="",
  )
  if cls:
    return cls
  try:
    from agent.shared.tool_catalog import GATED_ADDIN_TOOLS

    if tool_name in GATED_ADDIN_TOOLS:
      return "artifact_write"
  except Exception:
    pass
  return "state_write"


def effective_trade_approval_decision(
  tool_name: str,
  tool_args_redacted: dict[str, Any],
  decision: PolicyApprovalDecision,
  *,
  effective_trade_approval_expiry_seconds_fn: Any = effective_trade_approval_expiry_seconds,
  approval_wait_seconds_fn: Any = approval_settings.approval_wait_seconds,
  utc_now_fn: Any = utc_now,
) -> PolicyApprovalDecision:
  expiry_seconds = effective_trade_approval_expiry_seconds_fn(
    tool_name,
    tool_args_redacted,
    requested_expiry_seconds=decision.expiry_seconds,
    max_wait_seconds=approval_wait_seconds_fn(),
    now=utc_now_fn(),
  )
  if expiry_seconds == decision.expiry_seconds:
    return decision
  return replace(decision, expiry_seconds=expiry_seconds)
