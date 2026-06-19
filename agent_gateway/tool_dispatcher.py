from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from datetime import timedelta
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, TYPE_CHECKING

from . import approval_settings
from .approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalPolicy,
  ApprovalRequest as PolicyApprovalRequest,
  RunContext,
  apply_decision_to_request,
  build_approval_request,
  call_policy_safely,
  sha256_args,
  utc_now,
)
from .approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args
from .event_log import EventLog
from .skill_context import current_skill
from . import tool_dispatcher_source_pack as _source_pack_helpers
from .tool_dispatcher_helpers import (
  ApprovalCallback as ApprovalCallback,
  ApprovalDecision as ApprovalDecision,
  ApprovalKeyQualifier as ApprovalKeyQualifier,
  ApprovalRequest as ApprovalRequest,
  HeadlessAskCallback as HeadlessAskCallback,
  InterceptContext as InterceptContext,
  InterceptDecision as InterceptDecision,
  InterceptResult as InterceptResult,
  LocalToolHandler as LocalToolHandler,
  NeedsApprovalCallback as NeedsApprovalCallback,
  RELAY_POLICY_DENIED_MESSAGE as RELAY_POLICY_DENIED_MESSAGE,
  RELAY_POLICY_DENIED_SUB_CODE as RELAY_POLICY_DENIED_SUB_CODE,
  ToolExecutionContext as ToolExecutionContext,
  ToolInterceptor as ToolInterceptor,
  ToolResult as ToolResult,
  TransportApprovalRequest as TransportApprovalRequest,
  TransportApprovalResult as TransportApprovalResult,
  _approval_queue_timeout_seconds as _approval_queue_timeout_seconds,
  resolve_denied_provenance as resolve_denied_provenance,
)

if TYPE_CHECKING:
  from .mcp_client import McpClientManager


log = logging.getLogger("agent_gateway.dispatcher")


class ToolDispatcher:
  """Route tool calls to local handlers or MCP servers.

  The dispatcher is the policy boundary between model output and real tool
  execution. For each tool call it can:

  1. run interceptors
  2. request human approval
  3. execute a local Python handler
  4. fall back to an MCP server tool
  5. return structured warnings or errors
  """

  def __init__(
    self,
    mcp_client: "McpClientManager",
    local_tool_handlers: Dict[str, LocalToolHandler] | None = None,
    needs_approval: Callable[..., bool] | None = None,
    request_approval: ApprovalCallback | None = None,
    approved_tool_types: Set[str] | None = None,
    event_log: EventLog | None = None,
    approval_key_qualifier: ApprovalKeyQualifier | None = None,
    interceptors: Sequence[ToolInterceptor] | None = None,
    session_id: str = "",
    should_avoid_permission_prompts: bool = False,
    on_headless_ask: HeadlessAskCallback | None = None,
    mcp_session_inject_servers: set[str] | None = None,
    mcp_meta_inject_servers: frozenset[str] | None = None,
    user_id: str | None = None,
    risk_user_id: int | None = None,
    channel: str | None = None,
    role: str | None = None,
    credentials_resolver_active: bool = False,
    session_cache_denied_tools: frozenset[str] | None = None,
    session: Any | None = None,
    store: Any | None = None,
    policy: ApprovalPolicy | None = None,
    run_context: RunContext | None = None,
  ) -> None:
    self._mcp = mcp_client
    self._local = local_tool_handlers or {}
    self._needs_approval = self._normalize_needs_approval(needs_approval)
    self._request_approval = request_approval
    self._approved_tool_types = approved_tool_types if approved_tool_types is not None else set()
    self._event_log = event_log
    self._approval_key_qualifier = approval_key_qualifier
    self._interceptors: Sequence[ToolInterceptor] = list(interceptors or [])
    self._session_id = session_id
    self._should_avoid_permission_prompts = should_avoid_permission_prompts
    self._on_headless_ask = on_headless_ask
    self._mcp_session_inject_servers = mcp_session_inject_servers or set()
    self._mcp_meta_inject_servers = mcp_meta_inject_servers or frozenset()
    self._user_id = user_id
    self._risk_user_id = risk_user_id
    self._channel = channel
    self._role = role or "owner"
    self._credentials_resolver_active = credentials_resolver_active
    self._session_cache_denied = session_cache_denied_tools or frozenset()
    self._source_pack_session = session
    self._session = session
    self._approval_store = store or getattr(session, "approval_store", None)
    self._approval_policy = policy or getattr(session, "approval_policy", None)
    self._run_context = run_context
    self._mcp_accepts_abort_event = self._callable_accepts_kw(getattr(self._mcp, "call_tool", None), "abort_event")

  async def _run_interceptors(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
  ) -> InterceptResult:
    """Run interceptor chain and return a structured result."""
    if not self._interceptors:
      return InterceptResult(proceed=True)

    ctx = InterceptContext(
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_input=tool_input,
      session_id=self._session_id,
    )
    warnings: List[str] = []
    pending_ask: InterceptDecision | None = None

    for interceptor in self._interceptors:
      try:
        decision = await interceptor(ctx)
      except Exception as exc:
        is_critical = getattr(interceptor, "__intercept_critical__", False)
        if is_critical:
          log.error("Critical interceptor %s failed: %s — denying", interceptor, exc)
          if self._event_log is not None:
            self._event_log.append(
              {
                "type": "interceptor_decision",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "action": "deny",
                "code": "interceptor_error",
                "message": f"Critical safety interceptor failed: {exc}",
              }
            )
          return InterceptResult(
            proceed=False,
            error={
              "code": "interceptor_error",
              "message": f"Safety check failed due to an internal error. Tool '{tool_name}' was blocked.",
            },
          )
        log.warning("Interceptor %s error (non-fatal): %s", interceptor, exc)
        continue

      if decision.action == "deny":
        if self._event_log is not None:
          self._event_log.append(
            {
              "type": "interceptor_decision",
              "tool_call_id": tool_call_id,
              "tool_name": tool_name,
              "action": "deny",
              "code": decision.code,
              "message": decision.message,
            }
          )
        return InterceptResult(
          proceed=False,
          error={
            "code": decision.code,
            "message": decision.message or f"Tool '{tool_name}' was blocked by a runtime policy",
          },
        )
      if decision.action == "warn":
        warnings.append(decision.message)
        if self._event_log is not None:
          self._event_log.append(
            {
              "type": "interceptor_decision",
              "tool_call_id": tool_call_id,
              "tool_name": tool_name,
              "action": "warn",
              "code": decision.code,
              "message": decision.message,
            }
          )
        continue
      if decision.action == "ask":
        if self._event_log is not None:
          self._event_log.append(
            {
              "type": "interceptor_decision",
              "tool_call_id": tool_call_id,
              "tool_name": tool_name,
              "action": "ask",
              "code": decision.code,
              "message": decision.message,
            }
          )
        if pending_ask is None:
          pending_ask = decision

    return InterceptResult(proceed=True, warnings=warnings, pending_ask=pending_ask)

  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    call_index: int = 0,
    abort_event: asyncio.Event | None = None,
    skill_run_id: str | None = None,
    workspace_dir: str | None = None,
  ) -> ToolResult:
    """Execute one tool call and return `(result, error)`.

    Args:
      tool_call_id: Provider-emitted tool id.
      tool_name: Tool name selected by the model.
      tool_input: JSON-like tool payload.
      call_index: Zero-based tool index for the current turn.

    Returns:
      A tuple of `(result, error)` where exactly one side is usually `None`.

    Notes:
      - Local handlers receive `tool_ctx` and `call_index` keyword arguments.
      - Approved tool types are cached in-session through `allow_tool_type`.
      - Interceptor warnings are attached to successful dict results under
        `_interceptor_warnings`.
    """
    if abort_event is not None and abort_event.is_set():
      raise asyncio.CancelledError()

    ir = await self._run_interceptors(
      tool_call_id,
      tool_name,
      tool_input,
    )
    if not ir.proceed:
      return None, ir.error

    qualifier = ""
    if self._approval_key_qualifier is not None:
      try:
        qualifier = self._approval_key_qualifier(tool_name, tool_input) or ""
      except Exception:
        qualifier = ""

    static_needs_approval = self._should_request_approval(tool_name, tool_input, qualifier)
    dynamic_ask = ir.pending_ask is not None
    final_tool_input = tool_input
    approval_request_record: PolicyApprovalRequest | None = None

    if (
      not static_needs_approval
      and not dynamic_ask
      and self._tool_was_cache_hit(tool_name, qualifier)
    ):
      self._emit_approval_decided(
        tool_call_id,
        tool_name,
        outcome="approved",
        decision_source="session_cache_approved",
        allow_tool_type_applied=False,
      )

    if static_needs_approval or dynamic_ask:
      if self._should_avoid_permission_prompts:
        if static_needs_approval:
          reason_text = (
            ir.pending_ask.message
            if ir.pending_ask is not None
            else f"Tool '{tool_name}' requires static approval in headless context"
          )
          if self._event_log is not None:
            self._event_log.append(
              {
                "type": "headless_auto_deny",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "reason": reason_text,
                "source": "static",
              }
            )
          self._emit_approval_decided(
            tool_call_id,
            tool_name,
            outcome="denied",
            decision_source="headless_auto_deny",
            allow_tool_type_applied=False,
          )
          return None, {
            "code": "headless_auto_deny",
            "message": f"Tool '{tool_name}' blocked (static approval required): {reason_text}",
          }

        hook_result = "deny"
        if self._on_headless_ask is not None and ir.pending_ask is not None:
          headless_ctx = InterceptContext(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            session_id=self._session_id,
          )
          try:
            raw = self._on_headless_ask(headless_ctx, ir.pending_ask)
            if inspect.isawaitable(raw):
              raw = await raw
            hook_result = raw if raw in ("allow", "deny") else "deny"
          except Exception as exc:
            log.warning("Headless ask hook failed: %s — auto-denying", exc)
            hook_result = "deny"

        if hook_result != "allow":
          reason_text = (
            ir.pending_ask.message
            if ir.pending_ask is not None
            else "Approval required in headless context"
          )
          if self._event_log is not None:
            self._event_log.append(
              {
                "type": "headless_auto_deny",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "reason": reason_text,
                "source": "interceptor",
              }
            )
          self._emit_approval_decided(
            tool_call_id,
            tool_name,
            outcome="denied",
            decision_source="headless_auto_deny",
            allow_tool_type_applied=False,
          )
          return None, {
            "code": "headless_auto_deny",
            "message": f"Tool '{tool_name}' blocked: {reason_text}",
          }
        self._emit_approval_decided(
          tool_call_id,
          tool_name,
          outcome="approved",
          decision_source="headless_hook_approved",
          allow_tool_type_applied=False,
        )
      else:
        if self._approval_lifecycle_configured():
          allow_persistent = not dynamic_ask
          approval_reason = ir.pending_ask.message if ir.pending_ask is not None else ""
          lifecycle = await self._run_approval_lifecycle(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            qualifier=qualifier,
            reason=approval_reason,
            allow_persistent=allow_persistent,
          )
          approval_request_record = lifecycle.get("request")
          if lifecycle.get("timeout"):
            self._emit_approval_decided(
              tool_call_id,
              tool_name,
              outcome="timeout",
              decision_source="approval_timeout",
              allow_tool_type_applied=False,
            )
            return None, {"code": "approval_timeout", "message": "User did not respond within timeout"}
          if not lifecycle.get("approved"):
            decision_source, error_dict = resolve_denied_provenance(lifecycle.get("denied_by"))
            self._emit_approval_decided(
              tool_call_id,
              tool_name,
              outcome="denied",
              decision_source=decision_source,
              allow_tool_type_applied=False,
            )
            return None, error_dict
          final_tool_input = lifecycle.get("tool_input") or tool_input
          will_install = (
            bool(lifecycle.get("allow_tool_type"))
            and allow_persistent
            and tool_name not in self._session_cache_denied
          )
          self._emit_approval_decided(
            tool_call_id,
            tool_name,
            outcome="approved",
            decision_source=(
              "delegated_auto_approved"
              if getattr(approval_request_record, "state", None) == "auto_approved"
              else "user_approved"
            ),
            allow_tool_type_applied=will_install,
          )
          if will_install:
            self._approved_tool_types.add(self._qualified_key(tool_name, qualifier))
        elif self._request_approval is None:
          return None, {
            "code": "approval_required",
            "message": f"Tool '{tool_name}' requires approval but no approval handler is configured",
          }
        else:
          allow_persistent = not dynamic_ask
          approval_reason = ir.pending_ask.message if ir.pending_ask is not None else ""
          approval_tool_input = self._approval_transport_input(tool_name, tool_input)
          decision = await self._request_approval(
            ApprovalRequest(
              tool_call_id=tool_call_id,
              nonce=os.urandom(8).hex(),
              tool_name=tool_name,
              tool_input=approval_tool_input,
              resolved_qualifier=qualifier,
              reason=approval_reason,
              allow_persistent_approval=allow_persistent,
            )
          )
          if decision is None:
            self._emit_approval_decided(
              tool_call_id,
              tool_name,
              outcome="timeout",
              decision_source="approval_timeout",
              allow_tool_type_applied=False,
            )
            return None, {"code": "approval_timeout", "message": "User did not respond within timeout"}
          will_install = (
            decision.approved
            and decision.allow_tool_type
            and allow_persistent
            and tool_name not in self._session_cache_denied
          )
          self._emit_approval_decided(
            tool_call_id,
            tool_name,
            outcome="approved" if decision.approved else "denied",
            decision_source="user_approved" if decision.approved else resolve_denied_provenance(decision.denied_by)[0],
            allow_tool_type_applied=will_install,
          )
          if not decision.approved:
            return None, resolve_denied_provenance(decision.denied_by)[1]
          if will_install:
            self._approved_tool_types.add(self._qualified_key(tool_name, qualifier))

    result: Optional[Any]
    error: Optional[Dict[str, Any]]
    if tool_name in self._local:
      tool_ctx = None
      if self._event_log is not None:
        tool_ctx = ToolExecutionContext(
          tool_call_id=tool_call_id,
          tool_name=tool_name,
          event_log=self._event_log,
          resolved_qualifier=qualifier,
          abort_event=abort_event,
          skill_run_id=skill_run_id,
          workspace_dir=workspace_dir,
        )
      result, error = await self._local[tool_name](final_tool_input, call_index=call_index, tool_ctx=tool_ctx)
    elif self._mcp.is_mcp_tool(tool_name):
      server = self._mcp.get_server_for_tool(tool_name)
      if server and server in self._mcp_meta_inject_servers:
        resolved_risk_user_id = self._risk_user_id
        if resolved_risk_user_id is None and self._user_id is not None and str(self._user_id).isdigit():
          resolved_risk_user_id = int(str(self._user_id))
        if self._credentials_resolver_active and not resolved_risk_user_id:
          raise RuntimeError("MCP meta user_id is required in strict mode")
        meta = {
          "session_id": self._session_id,
          "user_id": str(resolved_risk_user_id) if resolved_risk_user_id is not None else None,
          "channel": self._channel,
          "role": self._role,
        }
        if skill_run_id is not None:
          meta["skill_run_id"] = skill_run_id
        if workspace_dir is not None:
          meta["workspace_dir"] = workspace_dir
        result, error = await self._call_mcp_tool(tool_name, final_tool_input, meta=meta, abort_event=abort_event)
      elif server and server in self._mcp_session_inject_servers:
        final_tool_input = {**final_tool_input, "_session_id": self._session_id}
        result, error = await self._call_mcp_tool(tool_name, final_tool_input, abort_event=abort_event)
      else:
        result, error = await self._call_mcp_tool(tool_name, final_tool_input, abort_event=abort_event)
    else:
      result, error = None, {"code": "unknown_tool", "message": f"Unknown tool: {tool_name}"}

    if ir.warnings and error is None and result is not None and isinstance(result, dict):
      result = dict(result)
      result["_interceptor_warnings"] = ir.warnings

    await self._emit_execution_audit(
      approval_request_record,
      final_tool_input,
      outcome="tool_error" if error is not None else "success",
      error_summary=str(error)[:500] if error is not None else None,
    )
    return result, error

  def requires_approval(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
    """Return True if dispatching this tool would block on user approval."""
    if self._request_approval is None and not self._approval_lifecycle_configured():
      return False
    qualifier = ""
    if self._approval_key_qualifier is not None:
      try:
        qualifier = self._approval_key_qualifier(tool_name, tool_input) or ""
      except Exception:
        qualifier = ""
    return self._should_request_approval(tool_name, tool_input, qualifier)

  def _approval_lifecycle_configured(self) -> bool:
    return self._approval_store is not None and self._approval_policy is not None and self._session is not None

  async def _run_approval_lifecycle(
    self,
    *,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    qualifier: str,
    reason: str,
    allow_persistent: bool,
  ) -> dict[str, Any]:
    store = self._approval_store
    policy = self._approval_policy
    if store is None or policy is None:
      raise RuntimeError("approval lifecycle is not configured")
    run_context = self._resolve_run_context()
    active_skill = current_skill()
    if active_skill and run_context.skill is None:
      run_context = replace(run_context, skill=active_skill)
    redacted, args_hash = self._redact_for_approval_request(tool_name, tool_input)
    request = build_approval_request(
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_class=self._resolve_tool_class(tool_name),
      tool_args_redacted=redacted,
      args_hash=args_hash,
      run_context=run_context,
      reason=reason or None,
    )
    await store.create(request)

    raw_args = dict(tool_input)
    try:
      decision: PolicyApprovalDecision = await call_policy_safely(policy, request, raw_args, run_context)
    finally:
      raw_args.clear()
      del raw_args

    decision = self._effective_trade_approval_decision(tool_name, request.tool_args_redacted, decision)
    request = apply_decision_to_request(request, decision)
    await store.update_request(request)
    final_tool_input = decision.modified_tool_args if decision.modified_tool_args is not None else tool_input

    if decision.outcome == "auto_approve":
      request = await store.transition_state(
        request.approval_id,
        "auto_approved",
        expected_state_version=request.state_version,
      )
      await policy.on_resolve(request=request)
      return {
        "approved": True,
        "allow_tool_type": False,
        "request": request,
        "tool_input": final_tool_input,
      }

    if decision.outcome == "auto_deny":
      request = await store.transition_state(
        request.approval_id,
        "auto_denied",
        expected_state_version=request.state_version,
      )
      await policy.on_resolve(request=request)
      return {"approved": False, "allow_tool_type": False, "request": request}

    if decision.outcome == "route_external":
      expires_at = utc_now() + timedelta(seconds=decision.expiry_seconds or 600)
      request = await store.transition_state(
        request.approval_id,
        "routed_external",
        route_target=decision.route_target,
        route_target_type=decision.route_target_type,
        expires_at=expires_at,
        expected_state_version=request.state_version,
      )
      return {"approved": False, "allow_tool_type": False, "request": request, "timeout": True}

    expires_at = utc_now() + timedelta(seconds=decision.expiry_seconds or 600)
    request = await store.transition_state(
      request.approval_id,
      "pending_user",
      route_target_type="pending_tools",
      expires_at=expires_at,
      expected_state_version=request.state_version,
    )
    nonce = os.urandom(8).hex()
    approval = await self._await_user_approval_via_pending_tools(
      request,
      decision,
      nonce=nonce,
      resolved_qualifier=qualifier,
      allow_persistent=allow_persistent,
      timeout_seconds=_approval_queue_timeout_seconds(decision.expiry_seconds),
    )
    if approval is None:
      return {"approved": False, "allow_tool_type": False, "request": request, "timeout": True}
    latest = await store.get(request.approval_id)
    if latest is not None:
      request = latest
    return {
      "approved": bool(approval.get("approved")),
      "allow_tool_type": bool(approval.get("allow_tool_type")),
      "denied_by": approval.get("denied_by"),
      "request": request,
      "tool_input": final_tool_input,
    }

  async def _await_user_approval_via_pending_tools(
    self,
    request: PolicyApprovalRequest,
    decision: PolicyApprovalDecision,
    *,
    nonce: str,
    resolved_qualifier: str,
    allow_persistent: bool,
    timeout_seconds: float,
  ) -> dict[str, Any] | None:
    session = self._session
    if session is None:
      return None
    approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    session.pending_tools[request.tool_call_id] = {
      "approval_id": request.approval_id,
      "nonce": nonce,
      "requested_at": int(time.time()),
      "status": "approval_pending",
      "tool_name": request.tool_name,
      "resolved_qualifier": resolved_qualifier,
    }
    session.approval_queues[request.tool_call_id] = approval_queue

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
    if self._event_log is not None:
      self._event_log.append(approval_event)
    session_log = getattr(session, "agent_session_log", None)
    if session_log is not None:
      try:
        await session_log.append(approval_event)
      except Exception:
        log.warning("Failed to persist approval request event for %s", request.tool_call_id, exc_info=True)

    try:
      return await asyncio.wait_for(approval_queue.get(), timeout=max(0.1, timeout_seconds))
    except asyncio.TimeoutError:
      store = self._approval_store
      if store is not None:
        try:
          latest = await store.get(request.approval_id)
          if latest is not None and latest.state == "pending_user":
            await store.transition_state(
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

  def _resolve_run_context(self) -> RunContext:
    if self._run_context is not None:
      return self._run_context
    session = self._session
    user_id = str(self._user_id or getattr(session, "user_id", "") or "unknown")
    channel = str(self._channel or getattr(session, "channel", None) or "web")
    return RunContext(
      user_id=user_id,
      request_id=str(getattr(session, "request_id", "") or self._session_id or "request-unknown"),
      session_id=self._session_id or getattr(session, "session_id", None),
      profile="chat",
      channel=channel,
      decider_role=self._role,
      policy_bundle_hash=str(getattr(self._approval_policy, "policy_bundle_hash", "unknown")),
    )

  def _resolve_tool_class(self, tool_name: str) -> str:
    try:
      from agent.shared.server_policies import get_server_for_policy_tool, get_tool_class

      server = self._mcp.get_server_for_tool(tool_name) if self._mcp.is_mcp_tool(tool_name) else None
      server = server or get_server_for_policy_tool(tool_name)
      if server:
        cls = get_tool_class(server, tool_name)
        if cls is not None:
          return cls
    except Exception:
      pass
    try:
      from agent.shared.tool_catalog import GATED_ADDIN_TOOLS

      if tool_name in GATED_ADDIN_TOOLS:
        return "artifact_write"
    except Exception:
      pass
    return "state_write"

  def _redact_for_approval_request(self, tool_name: str, tool_input: Dict[str, Any]) -> tuple[dict[str, Any], str]:
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
      redacted = enrich_trade_approval_args(tool_name, redacted, event_log=self._event_log)
      args_hash = hmac_value(tool_input, deployment_secret=secret, key_id=key_id)
      return redacted, args_hash
    except Exception:
      return {}, sha256_args(tool_input)

  def _approval_transport_input(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    return enrich_trade_approval_args(tool_name, dict(tool_input), event_log=self._event_log)

  def _effective_trade_approval_decision(
    self,
    tool_name: str,
    tool_args_redacted: Dict[str, Any],
    decision: PolicyApprovalDecision,
  ) -> PolicyApprovalDecision:
    expiry_seconds = effective_trade_approval_expiry_seconds(
      tool_name,
      tool_args_redacted,
      requested_expiry_seconds=decision.expiry_seconds,
      max_wait_seconds=approval_settings.approval_wait_seconds(),
      now=utc_now(),
    )
    if expiry_seconds == decision.expiry_seconds:
      return decision
    return replace(decision, expiry_seconds=expiry_seconds)

  async def _emit_execution_audit(
    self,
    request: PolicyApprovalRequest | None,
    raw_tool_args: Dict[str, Any],
    *,
    outcome: str,
    error_summary: str | None = None,
  ) -> None:
    if request is None or self._approval_store is None:
      return
    emitter = getattr(self._approval_store, "audit_emitter", None)
    emit = getattr(emitter, "emit_execution_outcome", None) if emitter is not None else None
    if emit is None:
      return
    raw_args = dict(raw_tool_args)
    try:
      await emit(
        request=request,
        raw_tool_args=raw_args,
        outcome=outcome,
        error_summary=error_summary,
      )
    finally:
      raw_args.clear()
      del raw_args

  async def _call_mcp_tool(
    self,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    meta: Dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
  ) -> ToolResult:
    kwargs: Dict[str, Any] = {}
    if meta is not None:
      kwargs["meta"] = meta
    if abort_event is not None and self._mcp_accepts_abort_event:
      kwargs["abort_event"] = abort_event
    result, error = await self._mcp.call_tool(tool_name, tool_input, **kwargs)
    if tool_name == "get_filing_evidence" and result is not None and error is None:
      self._capture_filing_source_pack(result, tool_input)
    return result, error

  def _capture_filing_source_pack(self, result: Any, tool_input: Dict[str, Any]) -> None:
    _source_pack_helpers.capture_filing_source_pack(
      self._source_pack_session,
      result,
      tool_input,
      log,
      planner_result_payload_fn=self._planner_result_payload,
      derive_fiscal_period_fn=self._derive_fiscal_period,
      payload_get_fn=self._payload_get,
    )

  @classmethod
  def _planner_result_payload(cls, result: Any) -> Any | None:
    for candidate in cls._planner_result_candidates(result):
      if cls._looks_like_source_pack_payload(candidate):
        return cls._coerce_planner_result_payload(candidate)
    return None

  @classmethod
  def _planner_result_candidates(cls, result: Any) -> list[Any]:
    return _source_pack_helpers.planner_result_candidates(result)

  @staticmethod
  def _coerce_planner_result_payload(payload: Any) -> Any:
    return _source_pack_helpers.coerce_planner_result_payload(payload)

  @staticmethod
  def _looks_like_source_pack_payload(payload: Any) -> bool:
    return _source_pack_helpers.looks_like_source_pack_payload(payload)

  @staticmethod
  def _payload_get(payload: Any, key: str) -> Any:
    return _source_pack_helpers.payload_get(payload, key)

  @classmethod
  def _derive_fiscal_period(cls, tool_input: Dict[str, Any], planner_result: Any) -> str | None:
    for key in ("fiscal_period", "period"):
      value = tool_input.get(key) or cls._payload_get(planner_result, key)
      if value:
        return str(value)
    year = tool_input.get("year") or tool_input.get("fiscal_year") or cls._payload_get(planner_result, "year")
    quarter = tool_input.get("quarter") or tool_input.get("fiscal_quarter") or cls._payload_get(planner_result, "quarter")
    if year and quarter:
      quarter_text = str(quarter).upper()
      if not quarter_text.startswith("Q"):
        quarter_text = f"Q{quarter_text}"
      return f"FY{year} {quarter_text}"
    if year:
      return f"FY{year}"
    return None

  @staticmethod
  def _callable_accepts_kw(callback: Any, keyword: str) -> bool:
    if callback is None:
      return False
    try:
      params = inspect.signature(callback).parameters
    except (TypeError, ValueError):
      return False
    return keyword in params or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())

  @staticmethod
  def _normalize_needs_approval(
    needs_approval: Callable[..., bool] | None,
  ) -> NeedsApprovalCallback:
    if needs_approval is None:
      return lambda _name, _tool_input, _qualifier: False

    try:
      arg_count = len(inspect.signature(needs_approval).parameters)
    except (TypeError, ValueError):
      return needs_approval  # type: ignore[return-value]

    if arg_count == 1:
      return lambda name, _tool_input, _qualifier: needs_approval(name)
    if arg_count == 2:
      return lambda name, tool_input, _qualifier: needs_approval(name, tool_input)
    return needs_approval  # type: ignore[return-value]

  @staticmethod
  def _qualified_key(tool_name: str, qualifier: str) -> str:
    return f"{tool_name}:{qualifier}" if qualifier else tool_name

  def _should_request_approval(
    self,
    tool_name: str,
    tool_input: Dict[str, Any],
    qualifier: str,
  ) -> bool:
    if tool_name not in self._session_cache_denied:
      qualified_key = self._qualified_key(tool_name, qualifier)
      if qualified_key in self._approved_tool_types:
        return False
      if not qualifier and tool_name in self._approved_tool_types:
        return False
    return self._needs_approval(tool_name, tool_input, qualifier)

  def _tool_was_cache_hit(self, tool_name: str, qualifier: str) -> bool:
    if tool_name in self._session_cache_denied:
      return False
    qualified_key = self._qualified_key(tool_name, qualifier)
    if qualified_key in self._approved_tool_types:
      return True
    if not qualifier and tool_name in self._approved_tool_types:
      return True
    return False

  def _emit_approval_decided(
    self,
    tool_call_id: str,
    tool_name: str,
    *,
    outcome: str,
    decision_source: str,
    allow_tool_type_applied: bool,
  ) -> None:
    if self._event_log is None:
      return
    self._event_log.append(
      {
        "type": "tool_approval_decided",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "outcome": outcome,
        "decision_source": decision_source,
        "allow_tool_type_applied": allow_tool_type_applied,
        "ts": time.time(),
      }
    )
