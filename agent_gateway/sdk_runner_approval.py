from __future__ import annotations

import asyncio
from datetime import timedelta
import time
from typing import Any, Callable

from . import approval_settings
from .approval_constraints import constraint_for_catalog_tool
from .approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest as PolicyApprovalRequest,
  RunContext,
  apply_decision_to_request,
  build_approval_request,
  call_policy_safely,
  sha256_args,
  utc_now,
)
from .approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args
from .batch_approval_projection import (
  abort_unpublished_batch_approval_admission,
  acquire_batch_approval_admission,
)
from .policy_imports import resolve_server_policy_tool_class


def approval_queue_timeout_seconds(
  expiry_seconds: float | int | None,
  *,
  approval_wait_seconds_fn: Callable[[], float] = approval_settings.approval_wait_seconds,
) -> float:
  return min(float(expiry_seconds or 600), approval_wait_seconds_fn())


def approval_lifecycle_configured(*, store: Any | None, policy: Any | None, session: Any | None) -> bool:
  return store is not None and policy is not None and session is not None


def resolve_run_context(
  *,
  run_context: RunContext | None,
  usage_user_id: str,
  session: Any | None,
  approval_policy: Any | None,
  request_id: str,
  session_id: str,
  channel: str | None,
  effective_model: str,
) -> RunContext:
  if run_context is not None:
    return run_context
  return RunContext(
    user_id=str(usage_user_id or getattr(session, "user_id", "") or "unknown"),
    request_id=request_id,
    session_id=session_id,
    profile="chat",
    channel=str(channel or getattr(session, "channel", None) or "web"),
    decider_role=str(getattr(session, "role", "owner") or "owner"),
    policy_bundle_hash=str(getattr(approval_policy, "policy_bundle_hash", "unknown")),
    model_id=effective_model,
  )


def resolve_tool_class(
  tool_name: str,
  *,
  policy_tool_name_fn: Callable[[str], str],
  server_for_tool_fn: Callable[[str], str | None],
  resolve_server_policy_tool_class_fn: Callable[..., str] = resolve_server_policy_tool_class,
) -> str:
  policy_tool = policy_tool_name_fn(tool_name)
  return resolve_server_policy_tool_class_fn(
    tool_name,
    policy_tool_name=policy_tool,
    runtime_server=server_for_tool_fn(tool_name),
  )


def redact_for_approval_request(
  tool_name: str,
  tool_input: dict[str, Any],
  *,
  sha256_args_fn: Callable[[Any], str] = sha256_args,
) -> tuple[dict[str, Any], str]:
  try:
    from agent.shared.tool_redaction import get_audit_hmac_secret, get_audit_hmac_key_id, hmac_value, redact_tool_input

    secret = get_audit_hmac_secret()
    key_id = get_audit_hmac_key_id()
    return (
      redact_tool_input(tool_name, tool_input, deployment_secret=secret, key_id=key_id),
      hmac_value(tool_input, deployment_secret=secret, key_id=key_id),
    )
  except Exception:
    return {}, sha256_args_fn(tool_input)


async def await_user_approval_via_pending_tools(
  *,
  session: Any | None,
  approval_store: Any | None,
  request: PolicyApprovalRequest,
  decision: PolicyApprovalDecision,
  nonce: str,
  append_event_fn: Callable[[dict[str, Any]], None],
  timeout_seconds: float,
  log: Any,
  time_fn: Callable[[], float] = time.time,
  queue_factory: Callable[..., Any] = asyncio.Queue,
  wait_for_fn: Callable[..., Any] = asyncio.wait_for,
  batch_admission: Any | None = None,
) -> dict[str, Any] | None:
  if session is None:
    return None
  approval_queue: asyncio.Queue = queue_factory(maxsize=1)
  session.pending_tools[request.tool_call_id] = {
    "approval_id": request.approval_id,
    "nonce": nonce,
    "requested_at": int(time_fn()),
    "status": "approval_pending",
    "tool_name": request.tool_name,
    "resolved_qualifier": "",
  }
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
    "resolved_qualifier": "",
    "reason": decision.reason,
    "allow_persistent_approval": decision.allow_persistent_grant,
    "ts": time_fn(),
  }
  append_event_fn(approval_event)
  session_log = getattr(session, "agent_session_log", None)
  if session_log is not None:
    try:
      await session_log.append(approval_event)
    except Exception:
      log.warning("Failed to persist approval request event for %s", request.tool_call_id, exc_info=True)
  try:
    return await wait_for_fn(
      approval_queue.get(),
      timeout=max(0.1, timeout_seconds),
    )
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


async def can_use_tool_callback(
  runner: Any,
  tool_name: str,
  input_data: dict[str, Any],
  _context: Any,
  *,
  policy_owner_mismatch_fn: Callable[[str], tuple[str, str, str] | None],
  policy_tool_name_fn: Callable[[str], str],
  current_skill_fn: Callable[[], str | None],
  replace_fn: Callable[..., Any],
  enrich_trade_approval_args_fn: Callable[..., dict[str, Any]] = enrich_trade_approval_args,
  build_approval_request_fn: Callable[..., PolicyApprovalRequest] = build_approval_request,
  call_policy_safely_fn: Callable[..., Any] = call_policy_safely,
  apply_decision_to_request_fn: Callable[..., PolicyApprovalRequest] = apply_decision_to_request,
  effective_trade_approval_expiry_seconds_fn: Callable[..., float | int | None] = effective_trade_approval_expiry_seconds,
  approval_wait_seconds_fn: Callable[[], float] = approval_settings.approval_wait_seconds,
  utc_now_fn: Callable[[], Any] = utc_now,
  uuid_hex_fn: Callable[[], str],
  os_urandom_fn: Callable[[int], bytes],
  relay_policy_denied_sub_code: str,
  relay_policy_denied_message: str,
) -> Any:
  arguments = dict(locals())
  batch_admission = acquire_batch_approval_admission(runner._session)
  arguments["batch_admission"] = batch_admission
  try:
    return await _can_use_tool_callback_impl(**arguments)
  except BaseException:
    if batch_admission is not None and not batch_admission.published:
      await abort_unpublished_batch_approval_admission(batch_admission)
    raise
  finally:
    if batch_admission is not None:
      batch_admission.release()


async def _can_use_tool_callback_impl(
  runner: Any,
  tool_name: str,
  input_data: dict[str, Any],
  _context: Any,
  *,
  policy_owner_mismatch_fn: Callable[[str], tuple[str, str, str] | None],
  policy_tool_name_fn: Callable[[str], str],
  current_skill_fn: Callable[[], str | None],
  replace_fn: Callable[..., Any],
  enrich_trade_approval_args_fn: Callable[..., dict[str, Any]] = enrich_trade_approval_args,
  build_approval_request_fn: Callable[..., PolicyApprovalRequest] = build_approval_request,
  call_policy_safely_fn: Callable[..., Any] = call_policy_safely,
  apply_decision_to_request_fn: Callable[..., PolicyApprovalRequest] = apply_decision_to_request,
  effective_trade_approval_expiry_seconds_fn: Callable[..., float | int | None] = effective_trade_approval_expiry_seconds,
  approval_wait_seconds_fn: Callable[[], float] = approval_settings.approval_wait_seconds,
  utc_now_fn: Callable[[], Any] = utc_now,
  uuid_hex_fn: Callable[[], str],
  os_urandom_fn: Callable[[int], bytes],
  relay_policy_denied_sub_code: str,
  relay_policy_denied_message: str,
  batch_admission: Any | None = None,
) -> Any:
  import claude_agent_sdk

  allow_cls = getattr(claude_agent_sdk, "PermissionResultAllow")
  deny_cls = getattr(claude_agent_sdk, "PermissionResultDeny")
  if tool_name in runner._effective_disallowed_tools():
    return deny_cls(message=f"Tool '{tool_name}' is not available in this context")
  mismatch = policy_owner_mismatch_fn(tool_name)
  if mismatch is not None:
    runtime_server, policy_tool, policy_server = mismatch
    return deny_cls(
      message=(
        f"Tool '{tool_name}' is not available from MCP server "
        f"'{runtime_server}'; policy owner for '{policy_tool}' is '{policy_server}'"
      )
    )
  policy_tool = policy_tool_name_fn(tool_name)
  try:
    approval_constraint = constraint_for_catalog_tool(policy_tool)
  except Exception:
    return deny_cls(
      message=(
        "[approval_constraint_unavailable] Trusted approval classification "
        "is unavailable"
      )
    )
  if approval_constraint == "fresh_human_owner":
    return deny_cls(
      message=(
        "[owner_control_route_required] Exact promotion requires the "
        "authenticated owner control-plane route"
      )
    )
  if not runner._approval_lifecycle_configured():
    return allow_cls()
  store = runner._approval_store
  policy = runner._approval_policy
  assert store is not None and policy is not None
  run_context = runner._resolve_run_context()
  active_skill = current_skill_fn()
  if active_skill and run_context.skill is None:
    run_context = replace_fn(run_context, skill=active_skill)
  redacted, args_hash = runner._redact_for_approval_request(tool_name, input_data)
  redacted = enrich_trade_approval_args_fn(policy_tool, redacted, event_log=runner._log)
  request = build_approval_request_fn(
    tool_call_id=f"sdk-{uuid_hex_fn()}",
    tool_name=policy_tool,
    tool_class=runner._resolve_tool_class(tool_name),
    tool_args_redacted=redacted,
    args_hash=args_hash,
    run_context=run_context,
    approval_constraint=approval_constraint,
  )
  if batch_admission is not None:
    batch_admission.bind_request(request=request, store=store)
  await store.create(request)
  raw_args = dict(input_data)
  try:
    decision = await call_policy_safely_fn(policy, request, raw_args, run_context)
  finally:
    raw_args.clear()
    del raw_args
  expiry_seconds = effective_trade_approval_expiry_seconds_fn(
    policy_tool,
    request.tool_args_redacted,
    requested_expiry_seconds=decision.expiry_seconds,
    max_wait_seconds=approval_wait_seconds_fn(),
    now=utc_now_fn(),
  )
  if expiry_seconds != decision.expiry_seconds:
    decision = replace_fn(decision, expiry_seconds=expiry_seconds)
  request = apply_decision_to_request_fn(request, decision)
  await store.update_request(request)

  if decision.outcome == "auto_approve":
    request = await store.transition_state(request.approval_id, "auto_approved", expected_state_version=request.state_version)
    await policy.on_resolve(request=request)
    if decision.modified_tool_args is not None:
      return allow_cls(updated_input=decision.modified_tool_args)
    return allow_cls()
  if decision.outcome == "auto_deny":
    request = await store.transition_state(request.approval_id, "auto_denied", expected_state_version=request.state_version)
    await policy.on_resolve(request=request)
    return deny_cls(message=decision.reason)
  if decision.outcome == "route_external":
    await store.transition_state(
      request.approval_id,
      "routed_external",
      route_target=decision.route_target,
      route_target_type=decision.route_target_type,
      expires_at=utc_now_fn() + timedelta(seconds=decision.expiry_seconds or 600),
      expected_state_version=request.state_version,
    )
    return deny_cls(message="external approval route is pending")

  request = await store.transition_state(
    request.approval_id,
    "pending_user",
    route_target_type="pending_tools",
    expires_at=utc_now_fn() + timedelta(seconds=decision.expiry_seconds or 600),
    expected_state_version=request.state_version,
  )
  approval = await runner._await_user_approval_via_pending_tools(
    request,
    decision,
    nonce=os_urandom_fn(8).hex(),
    batch_admission=batch_admission,
  )
  if approval and approval.get("approved"):
    if decision.modified_tool_args is not None:
      return allow_cls(updated_input=decision.modified_tool_args)
    return allow_cls()
  if approval and approval.get("denied_by") == "relay_policy":
    return deny_cls(message=f"[{relay_policy_denied_sub_code}] {relay_policy_denied_message}")
  return deny_cls(message="user denied")


__all__ = [
  "approval_lifecycle_configured",
  "approval_queue_timeout_seconds",
  "await_user_approval_via_pending_tools",
  "can_use_tool_callback",
  "redact_for_approval_request",
  "resolve_run_context",
  "resolve_tool_class",
]
