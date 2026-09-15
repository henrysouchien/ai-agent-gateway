from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, cast

from .approval_constraints import constraint_for_catalog_tool
from .approval_policy import RunContext
from .approval_enrichment import enrich_trade_approval_args
from .approval_route import DurableLocalApprovalRoute
from .mcp_client import registered_mcp_dispatch_scope
from .policy_imports import resolve_effective_role, resolve_server_policy_tool_class
from .sdk_runner_helpers import (
  catalogless_policy_owner_mismatch,
  catalogless_tool_name,
  server_for_tool,
)
from .skill_limits import ActiveSkillAdmission
from . import tool_dispatcher_approval_lifecycle as _approval_lifecycle_helpers
from .tool_redaction import resolve_redaction_provider

_redaction = resolve_redaction_provider()

log = logging.getLogger("agent_gateway.sdk_runner_approval")


def resolve_run_context(
  *,
  run_context: RunContext | None,
  run_id: str | None = None,
  usage_user_id: str,
  session: Any | None,
  approval_policy: Any | None,
  request_id: str,
  session_id: str,
  channel: str | None,
) -> RunContext:
  if run_context is not None:
    owner_user_id = str(getattr(session, "owner_user_id", None) or "").strip()
    if owner_user_id and run_context.user_id != owner_user_id:
      return replace(run_context, user_id=owner_user_id)
    return run_context
  return RunContext(
    user_id=str(
      getattr(session, "owner_user_id", None)
      or usage_user_id
      or getattr(session, "user_id", "")
      or "unknown"
    ),
    request_id=request_id,
    session_id=session_id,
    profile="chat",
    channel=str(channel or getattr(session, "channel", None) or "web"),
    decider_role=resolve_effective_role(getattr(session, "role", None)),
    policy_bundle_hash=str(getattr(approval_policy, "policy_bundle_hash", "unknown")),
    run_id=run_id,
  )


def resolve_catalogless_approval_identity(
  tool_name: str,
  *,
  resolve_server_policy_tool_class_fn: Callable[..., str] = resolve_server_policy_tool_class,
) -> tuple[str, str]:
  """Return the legacy lifecycle identity for an explicit catalog-free route."""

  policy_tool = catalogless_tool_name(tool_name)
  return (
    policy_tool,
    resolve_server_policy_tool_class_fn(
      tool_name,
      policy_tool_name=policy_tool,
      runtime_server=server_for_tool(tool_name),
    ),
  )


def is_catalogless_approval_route(runner: Any, tool_name: str) -> bool:
  """Select the construction-owned SDK-local and builtin approval route."""

  return (
    runner._registered_mcp_descriptor_for_sdk_tool is None
    or not tool_name.startswith("mcp__")
    or tool_name in runner._catalogless_mcp_tool_ids
  )


def registered_policy_owner_mismatch(
  runner: Any,
  tool_name: str,
) -> tuple[str, str, str] | None:
  """Retain catalog-free ownership checks only for its explicit routes."""

  if is_catalogless_approval_route(runner, tool_name):
    return catalogless_policy_owner_mismatch(tool_name)
  return None


def redact_for_approval_request(
  tool_name: str,
  tool_input: dict[str, Any],
) -> tuple[dict[str, Any], str]:
  secret = _redaction.get_audit_hmac_secret()
  key_id = _redaction.get_audit_hmac_key_id()
  return (
    _redaction.redact_tool_input(tool_name, tool_input, deployment_secret=secret, key_id=key_id),
    _redaction.hmac_value(tool_input, deployment_secret=secret, key_id=key_id),
  )


def approval_args_hash(tool_input: dict[str, Any]) -> str:
  secret = _redaction.get_audit_hmac_secret()
  key_id = _redaction.get_audit_hmac_key_id()
  return _redaction.hmac_value(
    tool_input,
    deployment_secret=secret,
    key_id=key_id,
  )


async def can_use_tool_callback(
  runner: Any,
  tool_name: str,
  input_data: dict[str, Any],
  _context: Any,
  *,
  current_skill_admission_fn: Callable[[], ActiveSkillAdmission | None],
  enrich_trade_approval_args_fn: Callable[..., dict[str, Any]] = enrich_trade_approval_args,
  uuid_hex_fn: Callable[[], str],
) -> Any:
  import claude_agent_sdk

  allow_cls = getattr(claude_agent_sdk, "PermissionResultAllow")
  deny_cls = getattr(claude_agent_sdk, "PermissionResultDeny")
  if tool_name in runner._effective_disallowed_tools():
    return deny_cls(message=f"Tool '{tool_name}' is not available in this context")
  mismatch = runner._sdk_policy_owner_mismatch(tool_name)
  if mismatch is not None:
    runtime_server, policy_tool, policy_server = mismatch
    return deny_cls(
      message=(
        f"Tool '{tool_name}' is not available from MCP server "
        f"'{runtime_server}'; policy owner for '{policy_tool}' is '{policy_server}'"
      )
    )
  registered_call = None
  if is_catalogless_approval_route(runner, tool_name):
    policy_tool, tool_class = runner._resolve_sdk_approval_identity(tool_name)
    prepared_input = input_data
  else:
    preparer = cast(
      Callable[..., Any],
      runner._prepare_registered_mcp_tool_call_for_sdk_tool,
    )
    trusted_scope = registered_mcp_dispatch_scope(
      user_id=runner._resolve_run_context().user_id,
      dispatch_scope=getattr(runner._session, "dispatch_scope", None),
    )
    registered_call = preparer(
      tool_name,
      input_data,
      trusted_scope,
      runner._registered_approval_overlay,
    )
    descriptor = registered_call.descriptor
    policy_tool = descriptor.identity.logical_name
    tool_class = descriptor.declaration.semantics.effect
    prepared_input = registered_call.prepared_call.materialize_input()
    if not registered_call.approval_required:
      return allow_cls(updated_input=prepared_input)
  prepared_authorization = (
    registered_call.prepared_authorization
    if registered_call is not None
    else None
  )
  try:
    approval_constraint = constraint_for_catalog_tool(policy_tool)
  except Exception as exc:
    log.error(
      "Approval constraint classification failed for tool %r (policy tool %r); "
      "owner: agent_gateway.approval_constraints over the trusted FMS action "
      "catalog (fms.action_catalog): %s",
      tool_name,
      policy_tool,
      exc,
      exc_info=True,
    )
    return deny_cls(
      message=(
        "[approval_constraint_unavailable] Trusted approval classification "
        f"is unavailable for tool '{policy_tool}': {exc}"
      )
    )
  if approval_constraint == "fresh_human_owner":
    return deny_cls(
      message=(
        "[owner_control_route_required] Exact promotion requires the "
        "authenticated owner control-plane route"
      )
    )
  route = runner._approval_route
  if not isinstance(route, DurableLocalApprovalRoute):
    if getattr(runner, "_approval_lifecycle", "required") == "not_required":
      return allow_cls(updated_input=prepared_input)
    return deny_cls(
      message=(
        "[approval_route_absent] This run has no admitted approval route "
        f"for '{tool_name}'"
      )
    )
  approval_input = (
    prepared_authorization.materialize_approval_arguments()
    if prepared_authorization is not None
    else prepared_input
  )
  redacted, default_args_hash = runner._redact_for_approval_request(
    tool_name,
    approval_input,
  )
  args_hash = (
    prepared_authorization.approval_arguments_hash
    if prepared_authorization is not None
    else default_args_hash
  )
  redacted = enrich_trade_approval_args_fn(policy_tool, redacted, event_log=runner._log)
  approval_reuse_key = (
    registered_call.approval_reuse_key
    if registered_call is not None
    else None
  )
  tool_call_id = f"sdk-{uuid_hex_fn()}"
  try:
    lifecycle = await _approval_lifecycle_helpers.run_approval_lifecycle(
      route=route,
      session=runner._session,
      tool_call_id=tool_call_id,
      tool_name=policy_tool,
      tool_input=prepared_input,
      qualifier="",
      reason="",
      allow_persistent=(
        True
        if registered_call is None
        else approval_reuse_key is not None
      ),
      approval_constraint=approval_constraint,
      approval_reuse_mode=(
        "legacy"
        if registered_call is None
        else "exact"
        if approval_reuse_key is not None
        else "disabled"
      ),
      approval_reuse_key=approval_reuse_key,
      approval_identity=(
        prepared_authorization.approval_identity
        if prepared_authorization is not None
        else None
      ),
      prepared_authorization_payload=(
        prepared_authorization.prepared_payload
        if prepared_authorization is not None
        else None
      ),
      approval_args_redacted=redacted,
      approval_args_hash=args_hash,
      resolve_run_context_fn=runner._resolve_run_context,
      current_skill_admission_fn=current_skill_admission_fn,
      redact_for_approval_request_fn=runner._redact_for_approval_request,
      resolve_tool_class_fn=lambda _policy_tool: tool_class,
      effective_trade_approval_decision_fn=runner._effective_trade_approval_decision,
      await_user_approval_via_pending_tools_fn=runner._await_user_approval_via_pending_tools,
      approval_queue_timeout_seconds_fn=runner._approval_queue_timeout_seconds,
      secret_boundary=getattr(runner, "_secret_boundary", None),
    )
  except _approval_lifecycle_helpers.ApprovalSkillAdmissionMismatch:
    return deny_cls(
      message="[skill_admission_mismatch] Trusted skill admission is unavailable"
    )
  if lifecycle["approved"]:
    lifecycle_input = lifecycle["tool_input"]
    if registered_call is None or prepared_authorization is None:
      return allow_cls(updated_input=lifecycle_input)
    return allow_cls(
      updated_input=registered_call.materialize_authorized_input(tool_call_id)
    )
  if lifecycle.get("timeout"):
    return deny_cls(
      message=(
        "[approval_timeout] approval expired; a fresh tool call and "
        "approval are required"
      ),
      interrupt=True,
    )
  request = lifecycle["request"]
  if request.state == "auto_denied":
    return deny_cls(message=str(request.reason or ""))
  return deny_cls(message="user denied")


__all__ = [
  "approval_args_hash",
  "can_use_tool_callback",
  "redact_for_approval_request",
  "is_catalogless_approval_route",
  "registered_policy_owner_mismatch",
  "resolve_catalogless_approval_identity",
  "resolve_run_context",
]
