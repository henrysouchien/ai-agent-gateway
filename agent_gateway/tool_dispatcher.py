from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import time
from importlib import import_module
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set, TYPE_CHECKING

from . import approval_settings
from .approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalPolicy,
  ApprovalRequest as PolicyApprovalRequest,
  RunContext,
  sha256_args,
  utc_now,
)
from .approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args
from .event_log import EventLog
from .mcp_client_catalog import tool_argument_guidance
from .policy_imports import resolve_server_policy_tool_class
from .skill_context import current_skill
from . import tool_dispatcher_audit as _audit_helpers
from . import tool_dispatcher_approval_lifecycle as _approval_lifecycle_helpers
from . import tool_dispatcher_runtime as _runtime_helpers
from . import tool_dispatcher_skill_tools as _skill_tool_helpers
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
  PlannedWritePlanningRejected as PlannedWritePlanningRejected,
  RELAY_POLICY_DENIED_MESSAGE as RELAY_POLICY_DENIED_MESSAGE,
  RELAY_POLICY_DENIED_SUB_CODE as RELAY_POLICY_DENIED_SUB_CODE,
  ToolExecutionContext as ToolExecutionContext,
  TrustedToolPlan as TrustedToolPlan,
  TrustedToolPlanError as TrustedToolPlanError,
  ToolInterceptor as ToolInterceptor,
  ToolResult as ToolResult,
  TransportApprovalRequest as TransportApprovalRequest,
  TransportApprovalResult as TransportApprovalResult,
  _approval_queue_timeout_seconds as _approval_queue_timeout_seconds,
  active_local_tool_schema as _active_local_tool_schema_helper,
  format_expected_type as _format_expected_type_helper,
  json_type_name as _json_type_name_helper,
  matches_json_type as _matches_json_type_helper,
  resolve_denied_provenance as resolve_denied_provenance,
  run_interceptors as _run_interceptors_helper,
  tool_input_schema_error as _tool_input_schema_error_helper,
  validate_against_local_schema as _validate_against_local_schema_helper,
  validate_local_tool_input as _validate_local_tool_input_helper,
)

if TYPE_CHECKING:
  from .mcp_client import McpClientManager


log = logging.getLogger("agent_gateway.dispatcher")


_MCP_VALIDATION_ERROR_CODES = {"invalid_input", "validation_error"}
_MCP_VALIDATION_ERROR_MARKERS = (
  "validation error",
  "missing required argument",
  "unexpected keyword argument",
  "[type=missing_argument]",
  "[type=unexpected_keyword_argument]",
)
_PORTFOLIO_SCOPE_FIELDS = frozenset({"portfolio_id", "portfolio_name"})


def _is_mcp_validation_error(error: Mapping[str, Any]) -> bool:
  code = str(error.get("code") or "").strip().lower()
  sub_code = str(error.get("sub_code") or "").strip().lower()
  if code in _MCP_VALIDATION_ERROR_CODES or sub_code in _MCP_VALIDATION_ERROR_CODES:
    return True
  message = str(error.get("message") or "").lower()
  return any(marker in message for marker in _MCP_VALIDATION_ERROR_MARKERS)


def _schema_properties(schema: Any) -> dict[str, Any]:
  if not isinstance(schema, dict):
    return {}
  properties = schema.get("properties")
  if isinstance(properties, dict):
    return properties
  return {}


def _scope_text(value: Any) -> str | None:
  if not isinstance(value, str):
    return None
  return value if value.strip() else None


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
    mcp_identity_overrides: Mapping[str, int] | None = None,
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
    get_tool_definitions: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    allowed_mcp_tools_by_server: Mapping[str, Set[str]] | None = None,
    mcp_scope_context: str = "skill",
    describe_mcp_scope_block: Callable[[str | None, str], str | None] | None = None,
    commercial_work_start: Any | None = None,
    commercial_irreversible_recheck: Callable[[Any], None] | None = None,
    commercial_mcp_servers: frozenset[str] | None = None,
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
    self._mcp_identity_overrides = dict(mcp_identity_overrides or {})
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
    self._get_tool_definitions = get_tool_definitions
    self._mcp_scope_context = mcp_scope_context
    self._describe_mcp_scope_block = describe_mcp_scope_block
    self._commercial_work_start = commercial_work_start
    self._commercial_irreversible_recheck = commercial_irreversible_recheck
    self._commercial_mcp_servers = commercial_mcp_servers or frozenset()
    self._mcp_accepts_abort_event = self._callable_accepts_kw(getattr(self._mcp, "call_tool", None), "abort_event")
    if allowed_mcp_tools_by_server is None:
      self._allowed_mcp_tools_by_server = None
    elif isinstance(allowed_mcp_tools_by_server, dict):
      self._allowed_mcp_tools_by_server = allowed_mcp_tools_by_server
    else:
      self._allowed_mcp_tools_by_server = {
        str(server_name): {str(tool_name) for tool_name in tool_names}
        for server_name, tool_names in allowed_mcp_tools_by_server.items()
      }

  def ensure_gateway_local_tool_handler(self, tool_name: str) -> bool:
    return _skill_tool_helpers.ensure_gateway_local_tool_handler(
      tool_name,
      local_handlers=self._local,
      session=self._session,
      current_skill_fn=current_skill,
    )

  def _active_local_tool_schema(
    self,
    tool_name: str,
  ) -> tuple[Mapping[str, Any] | None, Dict[str, Any] | None]:
    return _active_local_tool_schema_helper(self._get_tool_definitions, tool_name)

  @staticmethod
  def _json_type_name(value: Any) -> str:
    return _json_type_name_helper(value)

  @classmethod
  def _matches_json_type(cls, value: Any, expected_type: Any) -> bool:
    return _matches_json_type_helper(value, expected_type)

  @classmethod
  def _format_expected_type(cls, expected_type: Any) -> str:
    return _format_expected_type_helper(expected_type)

  def _tool_input_schema_error(
    self,
    tool_name: str,
    *,
    message: str,
    details: Dict[str, Any],
  ) -> Dict[str, Any]:
    return _tool_input_schema_error_helper(tool_name, message=message, details=details)

  def _validate_against_local_schema(
    self,
    tool_name: str,
    tool_input: Any,
    schema: Mapping[str, Any],
  ) -> Dict[str, Any] | None:
    return _validate_against_local_schema_helper(
      tool_name,
      tool_input,
      schema,
      json_type_name_fn=self._json_type_name,
      matches_json_type_fn=self._matches_json_type,
      format_expected_type_fn=self._format_expected_type,
      tool_input_schema_error_fn=self._tool_input_schema_error,
    )

  def _validate_local_tool_input(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Any,
  ) -> Dict[str, Any] | None:
    return _validate_local_tool_input_helper(
      tool_call_id,
      tool_name,
      tool_input,
      local_tool_handlers=self._local,
      get_tool_definitions=self._get_tool_definitions,
      event_log=self._event_log,
      active_local_tool_schema_fn=self._active_local_tool_schema,
      validate_against_local_schema_fn=self._validate_against_local_schema,
    )

  async def _run_interceptors(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
  ) -> InterceptResult:
    return await _run_interceptors_helper(
      tool_call_id,
      tool_name,
      tool_input,
      interceptors=self._interceptors,
      event_log=self._event_log,
      session_id=self._session_id,
      log=log,
    )

  def _mcp_scope_error(self, tool_name: str, server_name: str | None) -> Dict[str, Any] | None:
    return _runtime_helpers.mcp_scope_error(
      tool_name,
      server_name,
      allowed_mcp_tools_by_server=self._allowed_mcp_tools_by_server,
      scope_context=self._mcp_scope_context,
      describe_scope_block=self._describe_mcp_scope_block,
    )

  @staticmethod
  def _catalog_action(tool_name: str) -> Any | None:
    """Return one row from the first supported trusted catalog layout."""

    action_catalog: Any | None = None
    for module_name in ("fms.action_catalog", "api.fms.action_catalog"):
      try:
        module = import_module(module_name)
      except ModuleNotFoundError as exc:
        parts = module_name.split(".")
        candidate_or_parent = {
          ".".join(parts[:index])
          for index in range(1, len(parts) + 1)
        }
        if exc.name not in candidate_or_parent:
          raise
        continue
      action_catalog = module.ACTION_CATALOG
      break
    if action_catalog is None:
      raise TrustedToolPlanError(
        "trusted FMS action catalog is unavailable in every supported layout"
      )
    matches = [
      action
      for action in action_catalog
      if action.local_tool_name == tool_name
    ]
    if len(matches) > 1:
      raise TrustedToolPlanError(
        f"action catalog contains duplicate local tool {tool_name!r}"
      )
    return None if not matches else matches[0]

  @staticmethod
  def _catalog_planning_identity(tool_name: str) -> str | None:
    """Read the one authoritative action catalog; never infer from a handler."""

    action = ToolDispatcher._catalog_action(tool_name)
    return None if action is None else action.planning_identity

  @staticmethod
  def _planned_handler_hooks(
    tool_name: str,
    handler: LocalToolHandler,
  ) -> tuple[str, Callable[..., Any], Callable[..., Any]] | None:
    required_identity = ToolDispatcher._catalog_planning_identity(tool_name)
    identity_marker = getattr(handler, "PLANNING_IDENTITY", None)
    planner = getattr(handler, "plan_change", None)
    executor = getattr(handler, "execute_prepared_change", None)
    declared = (
      identity_marker is not None,
      callable(planner),
      callable(executor),
    )
    if not any(declared):
      if required_identity is not None:
        raise TrustedToolPlanError(
          f"catalogued exact-write tool {tool_name!r} lost its planning hooks"
        )
      return None
    if not all(declared):
      raise TrustedToolPlanError(
        "planned local handler must declare identity, planner, and exact executor"
      )
    if identity_marker not in {"change_set", "reviewed_change_binding"}:
      raise TrustedToolPlanError(
        f"unsupported planning identity: {identity_marker!r}"
      )
    if required_identity is not None and identity_marker != required_identity:
      raise TrustedToolPlanError(
        f"handler planning identity for {tool_name!r} differs from ACTION_CATALOG"
      )
    return str(identity_marker), planner, executor

  @staticmethod
  async def _plan_local_write(
    hooks: tuple[str, Callable[..., Any], Callable[..., Any]],
    tool_input: Dict[str, Any],
    *,
    call_index: int,
    tool_ctx: ToolExecutionContext,
  ) -> tuple[TrustedToolPlan, Callable[..., Any]]:
    identity_source, planner, executor = hooks
    planned = planner(tool_input, call_index=call_index, tool_ctx=tool_ctx)
    if inspect.isawaitable(planned):
      planned = await planned
    if type(planned) is not tuple or len(planned) != 2:
      raise TrustedToolPlanError(
        "planned local handler must return exactly (identity, prepared_payload)"
      )
    identity, prepared = planned
    trusted_plan = TrustedToolPlan.create(
      identity_source=identity_source,
      identity=identity,
      prepared=prepared,
    )
    tool_ctx.trusted_plan = trusted_plan
    return trusted_plan, executor

  @staticmethod
  def _prepared_business_model_authorization(
    tool_name: str,
    trusted_plan: TrustedToolPlan,
  ) -> dict[str, Any] | None:
    if tool_name != "fms_persist_business_model":
      return None
    gateway_prepared = trusted_plan.prepared
    outer_prepared = getattr(gateway_prepared, "prepared", None)
    completion = getattr(outer_prepared, "completion", None)
    finalizer = getattr(completion, "finalizer", None)
    prepared_accept = getattr(finalizer, "prepared_accept", None)
    serializer = getattr(prepared_accept, "to_canonical_bytes", None)
    if not callable(serializer):
      raise TrustedToolPlanError(
        "business-model exact plan lost its immutable accept payload"
      )
    prepared_bytes = serializer()
    if not isinstance(prepared_bytes, bytes):
      raise TrustedToolPlanError("business-model prepared payload is not bytes")
    return {
      "caller_kind": prepared_accept.caller_kind,
      "user_scope": prepared_accept.user_scope,
      "idempotency_locator": prepared_accept.idempotency_locator,
      "intent_digest": prepared_accept.intent_digest,
      "prepared_payload": prepared_bytes,
      "prepared_payload_digest": hashlib.sha256(prepared_bytes).hexdigest(),
      "change_set_id": trusted_plan.change_set_id,
      "change_hash": trusted_plan.change_hash,
      "base_vector_hash": trusted_plan.base_vector_hash,
    }

  def _plan_raw_patch_mcp_write(
    self,
    tool_input: Dict[str, Any],
    *,
    tool_ctx: ToolExecutionContext,
  ) -> TrustedToolPlan:
    """Plan the shipped raw MCP write locally before any approval is created."""

    from research.patch_engine import prepare_raw_patch_apply
    from research.repository import get_repository_factory
    from research.route_deps import route_workspace_dir

    run_context = self._resolve_run_context()
    user_id = str(run_context.user_id or "").strip()
    if not user_id:
      raise TrustedToolPlanError("raw patch planning requires a trusted user")
    research_file_id = tool_input.get("research_file_id")
    if isinstance(research_file_id, bool) or not isinstance(research_file_id, int):
      raise TrustedToolPlanError("raw patch planning requires research_file_id")
    raw_ops = tool_input.get("ops")
    if not isinstance(raw_ops, list):
      raise TrustedToolPlanError("raw patch planning requires an ops list")
    workspace_dir = route_workspace_dir(user_id, tool_input.get("workspace_dir"))
    repo = get_repository_factory().get(user_id)
    prepared = prepare_raw_patch_apply(
      repo,
      research_file_id,
      {"ops": raw_ops},
      skill_run_id=tool_input.get("skill_run_id"),
      workspace_dir=workspace_dir,
      allow_business_overview_replacement=(
        tool_input.get("allow_business_overview_replacement") is True
      ),
    )
    trusted_plan = TrustedToolPlan.create(
      identity_source="reviewed_change_binding",
      identity=prepared.binding,
      prepared=prepared,
    )
    tool_ctx.trusted_plan = trusted_plan
    return trusted_plan

  def _new_tool_execution_context(
    self,
    *,
    tool_call_id: str,
    tool_name: str,
    qualifier: str,
    abort_event: asyncio.Event | None,
    skill_run_id: str | None,
    step_id: str | None,
    workspace_dir: str | None,
    batch_id: int | str | None,
  ) -> ToolExecutionContext:
    run_context = self._resolve_run_context()
    return ToolExecutionContext(
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      event_log=self._event_log,
      resolved_qualifier=qualifier,
      abort_event=abort_event,
      skill_run_id=skill_run_id,
      step_id=step_id,
      workspace_dir=workspace_dir,
      batch_id=batch_id,
      request_id=run_context.request_id,
      run_id=run_context.run_id,
      user_id=run_context.user_id,
    )

  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    call_index: int = 0,
    abort_event: asyncio.Event | None = None,
    skill_run_id: str | None = None,
    step_id: str | None = None,
    workspace_dir: str | None = None,
    batch_id: int | str | None = None,
    capture_readable_resource_snapshot: bool = False,
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

    self.ensure_gateway_local_tool_handler(tool_name)
    input_schema_error = self._validate_local_tool_input(tool_call_id, tool_name, tool_input)
    if input_schema_error is not None:
      return None, input_schema_error

    tool_input = self.resolve_effective_tool_input(tool_name, tool_input)

    ir = await self._run_interceptors(
      tool_call_id,
      tool_name,
      tool_input,
    )
    if not ir.proceed:
      return None, ir.error

    if tool_name not in self._local and self._mcp.is_mcp_tool(tool_name):
      scope_error = self._mcp_scope_error(tool_name, self._mcp.get_server_for_tool(tool_name))
      if scope_error is not None:
        return None, scope_error

    qualifier = ""
    if self._approval_key_qualifier is not None:
      try:
        qualifier = self._approval_key_qualifier(tool_name, tool_input) or ""
      except Exception:
        qualifier = ""

    local_handler = self._local.get(tool_name)
    resolved_step_id = str(step_id or "").strip() or (
      f"skill-step:{str(skill_run_id).strip()}"
      if str(skill_run_id or "").strip()
      else None
    )
    tool_ctx: ToolExecutionContext | None = None
    trusted_plan: TrustedToolPlan | None = None
    planned_executor: Callable[..., Any] | None = None
    prepared_business_model_record: Any | None = None
    if local_handler is not None:
      tool_ctx = self._new_tool_execution_context(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        qualifier=qualifier,
        abort_event=abort_event,
        skill_run_id=skill_run_id,
        step_id=resolved_step_id,
        workspace_dir=workspace_dir,
        batch_id=batch_id,
      )
      if (
        tool_name == "fms_persist_business_model"
        and self._approval_store is not None
        and str(skill_run_id or "").strip()
      ):
        from .prepared_business_model_store import PreparedBusinessModelLifecycle

        run_context = self._resolve_run_context()
        user_scope = str(run_context.user_id or "").strip()
        if not user_scope:
          return None, {
            "code": "planned_write_authorization_unavailable",
            "message": "BusinessModel exact planning requires a trusted user scope.",
          }
        durable = await self._approval_store.get_prepared_business_model_change(
          caller_kind="fms_persist",
          user_scope=user_scope,
          idempotency_locator=str(skill_run_id).strip(),
        )
        if durable is not None:
          if durable.lifecycle is PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT:
            return None, {
              "code": "planned_write_replan_and_reauthorize_required",
              "message": (
                "This BusinessModel skill-run already failed before commit; "
                "retry with a new skill_run_id."
              ),
            }
          if durable.lifecycle in {
            PreparedBusinessModelLifecycle.DENIED,
            PreparedBusinessModelLifecycle.EXPIRED,
          }:
            return None, {
              "code": "planned_write_authorization_state_invalid",
              "message": "The durable BusinessModel plan is terminal and cannot execute.",
            }
          tool_ctx.durable_business_model_payload = durable.prepared_payload
      try:
        planned_hooks = self._planned_handler_hooks(tool_name, local_handler)
        if planned_hooks is not None:
          trusted_plan, planned_executor = await self._plan_local_write(
            planned_hooks,
            tool_input,
            call_index=call_index,
            tool_ctx=tool_ctx,
          )
      except PlannedWritePlanningRejected as exc:
        return exc.tool_result()
      except TrustedToolPlanError as exc:
        log.error("Invalid exact-plan contract for %s: %s", tool_name, exc)
        return None, {
          "code": "planned_write_contract_invalid",
          "message": f"Tool '{tool_name}' has an invalid exact-write planning contract.",
        }
      except Exception:
        log.exception("Exact-write planning failed for %s", tool_name)
        return None, {
          "code": "planned_write_planning_failed",
          "message": f"Tool '{tool_name}' could not produce a trusted exact-write plan.",
        }
    elif (
      tool_name == "apply_patch_ops"
      and self._mcp.is_mcp_tool(tool_name)
      and "authorization_ref" not in tool_input
      and tool_input.get("dry_run") is not True
    ):
      tool_ctx = self._new_tool_execution_context(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        qualifier=qualifier,
        abort_event=abort_event,
        skill_run_id=skill_run_id,
        step_id=resolved_step_id,
        workspace_dir=workspace_dir,
        batch_id=batch_id,
      )
      try:
        trusted_plan = self._plan_raw_patch_mcp_write(
          tool_input,
          tool_ctx=tool_ctx,
        )
      except TrustedToolPlanError as exc:
        log.error("Invalid raw MCP exact-plan contract: %s", exc)
        return None, {
          "code": "planned_write_contract_invalid",
          "message": "Tool 'apply_patch_ops' has an invalid exact-write plan.",
        }
      except Exception:
        log.exception("Exact-write planning failed for apply_patch_ops")
        return None, {
          "code": "planned_write_planning_failed",
          "message": "Tool 'apply_patch_ops' could not produce a trusted exact-write plan.",
        }

    static_needs_approval = self._should_request_approval(tool_name, tool_input, qualifier)
    if tool_name == "apply_patch_ops" and (
      "authorization_ref" in tool_input or tool_input.get("dry_run") is True
    ):
      # The opaque reference is revalidated and claimed by the write service;
      # a retry must not mint a second generic approval around it.
      static_needs_approval = False
    dynamic_ask = ir.pending_ask is not None
    final_tool_input = tool_input
    approval_request_record: PolicyApprovalRequest | None = None

    if (
      trusted_plan is None
      and
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

    if trusted_plan is not None:
      if not self._approval_lifecycle_configured():
        code = (
          "planned_write_callback_transport_unsupported"
          if self._request_approval is not None
          else "planned_write_authorization_unavailable"
        )
        return None, {
          "code": code,
          "message": (
            f"Tool '{tool_name}' requires a durable approval store for its exact-write plan."
          ),
        }

      allow_persistent = not dynamic_ask
      approval_reason = ir.pending_ask.message if ir.pending_ask is not None else ""
      cache_approved = (
        not static_needs_approval
        and not dynamic_ask
        and self._tool_was_cache_hit(tool_name, qualifier)
      )
      automatic_approval_reason: str | None = None
      automatic_denial_reason: str | None = None
      deny_user_prompt = False

      if self._should_avoid_permission_prompts and not cache_approved:
        if static_needs_approval:
          automatic_denial_reason = (
            ir.pending_ask.message
            if ir.pending_ask is not None
            else f"Tool '{tool_name}' requires static approval in headless context"
          )
        elif dynamic_ask:
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
          if hook_result == "allow":
            automatic_approval_reason = (
              "Headless approval hook authorized the exact planned identity"
            )
          else:
            automatic_denial_reason = (
              ir.pending_ask.message
              if ir.pending_ask is not None
              else "Approval required in headless context"
            )
        else:
          deny_user_prompt = True

      try:
        prepared_authorization_payload: bytes | None = None
        prepared_business_model_change = self._prepared_business_model_authorization(
          tool_name,
          trusted_plan,
        )
        resume_approval_request = None
        skip_approval_lifecycle = False
        if prepared_business_model_change is not None:
          from .prepared_business_model_store import PreparedBusinessModelLifecycle

          prepared_business_model_record = await self._approval_store.get_prepared_business_model_change(
            caller_kind=str(prepared_business_model_change["caller_kind"]),
            user_scope=str(prepared_business_model_change["user_scope"]),
            idempotency_locator=str(
              prepared_business_model_change["idempotency_locator"]
            ),
          )
          if prepared_business_model_record is not None:
            supplied_intent = str(prepared_business_model_change["intent_digest"])
            immutable_matches = (
              prepared_business_model_record.intent_digest == supplied_intent
              and prepared_business_model_record.prepared_payload
              == prepared_business_model_change["prepared_payload"]
              and prepared_business_model_record.change_set_id
              == str(prepared_business_model_change["change_set_id"])
              and prepared_business_model_record.change_hash
              == str(prepared_business_model_change["change_hash"])
              and prepared_business_model_record.base_vector_hash
              == str(prepared_business_model_change["base_vector_hash"])
              and prepared_business_model_record.prepared_payload_digest
              == str(prepared_business_model_change["prepared_payload_digest"])
            )
            if (
              prepared_business_model_record.lifecycle
              is PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT
            ):
              if prepared_business_model_record.intent_digest != supplied_intent:
                raise TrustedToolPlanError(
                  "FMS skill-run locator conflicts with its terminal intent"
                )
              return None, {
                "code": "planned_write_replan_and_reauthorize_required",
                "message": (
                  "This BusinessModel skill-run already failed before commit; "
                  "retry with a new skill_run_id."
                ),
              }
            if prepared_business_model_record.lifecycle in {
              PreparedBusinessModelLifecycle.DENIED,
              PreparedBusinessModelLifecycle.EXPIRED,
            }:
              return None, {
                "code": "planned_write_authorization_state_invalid",
                "message": "The durable BusinessModel plan is terminal and cannot execute.",
              }
            if not immutable_matches:
              raise TrustedToolPlanError(
                "FMS skill-run locator conflicts with its prepared BusinessModel plan"
              )
            resume_approval_request = await self._approval_store.get(
              prepared_business_model_record.approval_id
            )
            if resume_approval_request is None:
              raise TrustedToolPlanError(
                "durable FMS BusinessModel plan lost its approval request"
              )
            if (
              prepared_business_model_record.lifecycle
              is PreparedBusinessModelLifecycle.PENDING
            ):
              approval_state = str(resume_approval_request.state)
              if approval_state in {"approved", "auto_approved"}:
                prepared_business_model_record = await self._approval_store.transition_prepared_business_model_change(
                  caller_kind=prepared_business_model_record.caller_kind,
                  user_scope=prepared_business_model_record.user_scope,
                  idempotency_locator=prepared_business_model_record.idempotency_locator,
                  expected=PreparedBusinessModelLifecycle.PENDING,
                  target=PreparedBusinessModelLifecycle.AUTHORIZED,
                  approval_id=prepared_business_model_record.approval_id,
                  approval_chain_id=prepared_business_model_record.approval_chain_id,
                )
                skip_approval_lifecycle = True
              elif approval_state in {"denied", "auto_denied"}:
                await self._approval_store.transition_prepared_business_model_change(
                  caller_kind=prepared_business_model_record.caller_kind,
                  user_scope=prepared_business_model_record.user_scope,
                  idempotency_locator=prepared_business_model_record.idempotency_locator,
                  expected=PreparedBusinessModelLifecycle.PENDING,
                  target=PreparedBusinessModelLifecycle.DENIED,
                  approval_id=prepared_business_model_record.approval_id,
                  approval_chain_id=prepared_business_model_record.approval_chain_id,
                )
                return None, {
                  "code": "planned_write_authorization_state_invalid",
                  "message": "The original BusinessModel approval was denied.",
                }
              elif approval_state == "expired":
                await self._approval_store.transition_prepared_business_model_change(
                  caller_kind=prepared_business_model_record.caller_kind,
                  user_scope=prepared_business_model_record.user_scope,
                  idempotency_locator=prepared_business_model_record.idempotency_locator,
                  expected=PreparedBusinessModelLifecycle.PENDING,
                  target=PreparedBusinessModelLifecycle.EXPIRED,
                  approval_id=prepared_business_model_record.approval_id,
                  approval_chain_id=prepared_business_model_record.approval_chain_id,
                )
                return None, {
                  "code": "planned_write_authorization_state_invalid",
                  "message": "The original BusinessModel approval expired.",
                }
              elif approval_state not in {"created", "pending_user", "routed_external"}:
                raise TrustedToolPlanError(
                  "durable FMS BusinessModel plan has an unknown approval state"
                )
            if prepared_business_model_record.lifecycle in {
              PreparedBusinessModelLifecycle.AUTHORIZED,
              PreparedBusinessModelLifecycle.CONSUMED,
            }:
              if resume_approval_request.state not in {
                "approved",
                "auto_approved",
              }:
                raise TrustedToolPlanError(
                  "durable FMS BusinessModel plan lost approved lineage"
                )
              skip_approval_lifecycle = True
        approval_args_redacted: dict[str, Any] | None = None
        approval_args_hash: str | None = None
        if (
          tool_name == "apply_patch_ops"
          and trusted_plan.identity_source == "reviewed_change_binding"
        ):
          from research.patch_engine import raw_patch_authorization_args

          serializer = getattr(trusted_plan.prepared, "to_canonical_bytes", None)
          if not callable(serializer):
            raise TrustedToolPlanError(
              "raw patch plan cannot serialize its prepared authorization bytes"
            )
          prepared_authorization_payload = serializer()
          if not isinstance(prepared_authorization_payload, bytes):
            raise TrustedToolPlanError(
              "raw patch prepared authorization payload is not immutable bytes"
            )
          approval_args_redacted = raw_patch_authorization_args(
            trusted_plan.prepared
          )
          approval_args_hash = sha256_args(approval_args_redacted)
        lifecycle = (
          {
            "approved": True,
            "allow_tool_type": False,
            "request": resume_approval_request,
            "tool_input": tool_input,
            "decision_source": "prepared_business_model_resume",
          }
          if skip_approval_lifecycle
          else await self._run_approval_lifecycle(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            qualifier=qualifier,
            reason=approval_reason,
            allow_persistent=allow_persistent,
            approval_identity=trusted_plan.approval_identity(),
            prepared_authorization_payload=prepared_authorization_payload,
            prepared_business_model_change=prepared_business_model_change,
            resume_approval_request=resume_approval_request,
            approval_args_redacted=approval_args_redacted,
            approval_args_hash=approval_args_hash,
            session_cache_approved=cache_approved,
            automatic_approval_reason=automatic_approval_reason,
            automatic_denial_reason=automatic_denial_reason,
            deny_user_prompt=deny_user_prompt,
          )
        )
      except Exception:
        log.exception("Durable exact-write authorization failed for %s", tool_name)
        return None, {
          "code": "planned_write_authorization_persistence_failed",
          "message": (
            f"Tool '{tool_name}' could not persist its exact-write authorization."
          ),
        }

      approval_request_record = lifecycle.get("request")
      if prepared_business_model_change is not None:
        from .prepared_business_model_store import PreparedBusinessModelLifecycle

        prepared_business_model_record = (
          await self._approval_store.get_prepared_business_model_change(
            caller_kind=str(prepared_business_model_change["caller_kind"]),
            user_scope=str(prepared_business_model_change["user_scope"]),
            idempotency_locator=str(
              prepared_business_model_change["idempotency_locator"]
            ),
          )
        )
        if prepared_business_model_record is None:
          raise TrustedToolPlanError(
            "durable FMS BusinessModel plan disappeared after authorization"
          )
        if lifecycle.get("approved"):
          if (
            prepared_business_model_record.lifecycle
            is PreparedBusinessModelLifecycle.PENDING
          ):
            prepared_business_model_record = await self._approval_store.transition_prepared_business_model_change(
              caller_kind=prepared_business_model_record.caller_kind,
              user_scope=prepared_business_model_record.user_scope,
              idempotency_locator=prepared_business_model_record.idempotency_locator,
              expected=PreparedBusinessModelLifecycle.PENDING,
              target=PreparedBusinessModelLifecycle.AUTHORIZED,
              approval_id=prepared_business_model_record.approval_id,
              approval_chain_id=prepared_business_model_record.approval_chain_id,
            )
          elif (
            prepared_business_model_record.lifecycle
            is PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT
          ):
            return None, {
              "code": "planned_write_replan_and_reauthorize_required",
              "message": (
                "This BusinessModel skill-run already failed before commit; "
                "retry with a new skill_run_id."
              ),
            }
          elif prepared_business_model_record.lifecycle in {
            PreparedBusinessModelLifecycle.DENIED,
            PreparedBusinessModelLifecycle.EXPIRED,
          }:
            return None, {
              "code": "planned_write_authorization_state_invalid",
              "message": "The durable BusinessModel plan is terminal and cannot execute.",
            }
          elif (
            prepared_business_model_record.lifecycle
            is PreparedBusinessModelLifecycle.AUTHORIZED
            and prepared_business_model_record.approval_id
            != getattr(approval_request_record, "approval_id", None)
          ):
            original_request = await self._approval_store.get(
              prepared_business_model_record.approval_id
            )
            if original_request is None or original_request.state not in {
              "approved",
              "auto_approved",
            }:
              raise TrustedToolPlanError(
                "authorized FMS BusinessModel plan lost its original approval"
              )
            approval_request_record = original_request
            lifecycle["request"] = original_request
        elif (
          not lifecycle.get("timeout")
          and prepared_business_model_record.lifecycle
          is PreparedBusinessModelLifecycle.PENDING
          and prepared_business_model_record.approval_id
          == getattr(approval_request_record, "approval_id", None)
        ):
          prepared_business_model_record = await self._approval_store.transition_prepared_business_model_change(
            caller_kind=prepared_business_model_record.caller_kind,
            user_scope=prepared_business_model_record.user_scope,
            idempotency_locator=prepared_business_model_record.idempotency_locator,
            expected=PreparedBusinessModelLifecycle.PENDING,
            target=PreparedBusinessModelLifecycle.DENIED,
            approval_id=prepared_business_model_record.approval_id,
            approval_chain_id=prepared_business_model_record.approval_chain_id,
          )
      try:
        if approval_request_record is None:
          raise TrustedToolPlanError("approval lifecycle returned no durable request")
        trusted_plan.verify_approval_request(approval_request_record)
      except TrustedToolPlanError as exc:
        log.error("Unbound exact-write approval for %s: %s", tool_name, exc)
        return None, {
          "code": "planned_write_authorization_identity_invalid",
          "message": f"Tool '{tool_name}' approval did not bind the exact planned identity.",
        }

      tool_ctx.approval_id = approval_request_record.approval_id
      tool_ctx.approval_chain_id = approval_request_record.approval_chain_id
      if lifecycle.get("policy_modified_tool_args"):
        self._emit_approval_decided(
          tool_call_id,
          tool_name,
          outcome="approved",
          decision_source="planned_write_reinvocation_required",
          allow_tool_type_applied=False,
        )
        await self._emit_execution_audit(
          approval_request_record,
          tool_input,
          outcome="reinvocation_required",
          error_summary="Policy returned modified arguments for an exact planned write",
        )
        return None, {
          "code": "planned_write_reinvocation_required",
          "message": (
            f"Tool '{tool_name}' arguments changed during authorization; "
            "submit a new invocation so the exact change can be replanned."
          ),
        }
      if lifecycle.get("timeout"):
        self._emit_approval_decided(
          tool_call_id,
          tool_name,
          outcome="timeout",
          decision_source="approval_timeout",
          allow_tool_type_applied=False,
        )
        return None, {
          "code": "approval_timeout",
          "message": "User did not respond within timeout",
        }
      if not lifecycle.get("approved"):
        decision_source, error_dict = resolve_denied_provenance(
          lifecycle.get("denied_by")
        )
        if lifecycle.get("decision_source") == "headless_auto_deny":
          error_dict = {
            "code": "headless_auto_deny",
            "message": f"Tool '{tool_name}' blocked in headless context.",
          }
        self._emit_approval_decided(
          tool_call_id,
          tool_name,
          outcome="denied",
          decision_source=lifecycle.get("decision_source") or decision_source,
          allow_tool_type_applied=False,
        )
        return None, error_dict

      final_tool_input = lifecycle.get("tool_input", tool_input)
      if tool_name == "apply_patch_ops" and planned_executor is None:
        from .raw_patch_authorization_store import encode_reference

        final_tool_input = {
          **tool_input,
          "authorization_ref": encode_reference(tool_call_id),
        }
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
          lifecycle.get("decision_source")
          or (
            "delegated_auto_approved"
            if getattr(approval_request_record, "state", None) == "auto_approved"
            else "user_approved"
          )
        ),
        allow_tool_type_applied=will_install,
      )
      if will_install:
        self._approved_tool_types.add(self._qualified_key(tool_name, qualifier))

    elif static_needs_approval or dynamic_ask:
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
          final_tool_input = lifecycle["tool_input"] if "tool_input" in lifecycle else tool_input
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
    if self._resolve_tool_class(tool_name) == "irreversible":
      if self._commercial_work_start is not None:
        recheck = self._commercial_irreversible_recheck
        if recheck is None:
          return None, {
            "code": "commercial_irreversible_authority_unavailable",
            "message": "Fresh commercial authority is unavailable.",
          }
        try:
          recheck(self._commercial_work_start)
        except Exception:
          return None, {
            "code": "commercial_irreversible_authority_invalid",
            "message": "Fresh commercial authority is invalid or expired.",
          }
    if local_handler is not None:
      input_schema_error = self._validate_local_tool_input(tool_call_id, tool_name, final_tool_input)
      if input_schema_error is not None:
        return None, input_schema_error

      if tool_ctx is None:
        raise AssertionError("local tool execution context was not created")
      local_kwargs: dict[str, Any] = {
        "call_index": call_index,
        "tool_ctx": tool_ctx,
      }
      if trusted_plan is not None:
        if (
          planned_executor is None
          or tool_ctx.trusted_plan is not trusted_plan
          or approval_request_record is None
          or not tool_ctx.approval_id
          or not tool_ctx.approval_chain_id
        ):
          return None, {
            "code": "planned_write_trusted_plan_lost",
            "message": f"Tool '{tool_name}' lost its trusted exact-write authorization.",
          }
        try:
          trusted_plan.verify_approval_request(approval_request_record)
        except TrustedToolPlanError as exc:
          log.error("Exact-write authorization drift for %s: %s", tool_name, exc)
          return None, {
            "code": "planned_write_trusted_plan_lost",
            "message": f"Tool '{tool_name}' lost its trusted exact-write authorization.",
          }
        if getattr(approval_request_record, "state", None) not in {
          "approved",
          "auto_approved",
        }:
          return None, {
            "code": "planned_write_authorization_state_invalid",
            "message": f"Tool '{tool_name}' does not have an executable approval state.",
          }
        exact_kwargs = {
          "authorized_identity": trusted_plan.identity,
          "approval_id": tool_ctx.approval_id,
          "approval_chain_id": tool_ctx.approval_chain_id,
          **local_kwargs,
        }
        executor_parameters = inspect.signature(planned_executor).parameters
        if "approval_request" in executor_parameters or any(
          parameter.kind is inspect.Parameter.VAR_KEYWORD
          for parameter in executor_parameters.values()
        ):
          exact_kwargs["approval_request"] = approval_request_record
        exact_result = planned_executor(
          trusted_plan.prepared,
          **exact_kwargs,
        )
        if inspect.isawaitable(exact_result):
          exact_result = await exact_result
        if type(exact_result) is not tuple or len(exact_result) != 2:
          raise TrustedToolPlanError(
            "exact planned-write executor must return a ToolResult pair"
          )
        result, error = exact_result
      else:
        if capture_readable_resource_snapshot:
          local_kwargs["capture_readable_resource_snapshot"] = True
        result, error = await local_handler(final_tool_input, **local_kwargs)
    elif self._mcp.is_mcp_tool(tool_name):
      server = self._mcp.get_server_for_tool(tool_name)
      if (
        self._commercial_work_start is not None
        and server not in self._commercial_mcp_servers
      ):
        return None, {
          "code": "commercial_mcp_destination_denied",
          "message": "Commercial work cannot be dispatched to this MCP destination.",
        }
      if server and callable(getattr(self._mcp, "is_per_user_server", None)) and self._mcp.is_per_user_server(server):
        resolved_risk_user_id = self._mcp_identity_overrides.get(server, self._risk_user_id)
        result, error = await self._call_mcp_tool(
          tool_name,
          final_tool_input,
          abort_event=abort_event,
          user_id=resolved_risk_user_id,
        )
      elif server and server in self._mcp_meta_inject_servers:
        resolved_risk_user_id = self._mcp_identity_overrides.get(server, self._risk_user_id)
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
        if batch_id is not None:
          meta["batch_id"] = str(batch_id)
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

    if (
      prepared_business_model_record is not None
      and prepared_business_model_record.lifecycle.value == "AUTHORIZED"
      and isinstance(result, dict)
      and isinstance(result.get("receipt"), dict)
    ):
      from .prepared_business_model_store import PreparedBusinessModelLifecycle

      receipt_payload = dict(result["receipt"])
      receipt_bytes = json.dumps(
        receipt_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
      ).encode("utf-8")
      receipt_status = str(receipt_payload.get("status") or "")
      if receipt_status == "FAILED_PRECOMMIT":
        error_data = result.get("error")
        error_data = error_data.get("data") if isinstance(error_data, dict) else None
        restoration = (
          error_data.get("restore")
          if isinstance(error_data, dict) and isinstance(error_data.get("restore"), dict)
          else None
        )
        file_restoration = (
          restoration.get("file")
          if isinstance(restoration, dict)
          and isinstance(restoration.get("file"), dict)
          else None
        )
        exact_restoration_proved = (
          isinstance(file_restoration, dict)
          and file_restoration.get("restored") is True
          and isinstance(file_restoration.get("target"), str)
          and bool(str(file_restoration.get("target")).strip())
          and isinstance(file_restoration.get("base_digest"), str)
          and len(str(file_restoration.get("base_digest"))) == 64
          and all(
            char in "0123456789abcdef"
            for char in str(file_restoration.get("base_digest"))
          )
        )
        if not exact_restoration_proved:
          result = None
          error = {
            "code": "planned_write_recovery_evidence_missing",
            "message": (
              "BusinessModel precommit failure did not prove exact restoration; "
              "the skill-run remains non-terminal for operator recovery."
            ),
          }
        else:
          restoration_bytes = json.dumps(
            restoration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
          ).encode("utf-8")
          prepared_business_model_record = await self._approval_store.transition_prepared_business_model_change(
            caller_kind=prepared_business_model_record.caller_kind,
            user_scope=prepared_business_model_record.user_scope,
            idempotency_locator=prepared_business_model_record.idempotency_locator,
            expected=PreparedBusinessModelLifecycle.AUTHORIZED,
            target=PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT,
            approval_id=prepared_business_model_record.approval_id,
            approval_chain_id=prepared_business_model_record.approval_chain_id,
            execution_receipt=receipt_bytes,
            restoration_digest=hashlib.sha256(restoration_bytes).hexdigest(),
          )
      elif receipt_status in {
        "COMMITTED",
        "COMMITTED_OUTBOX_FAILED",
        "COMMITTED_UNVERIFIED",
        "PARTIAL",
        "REPLAYED",
      }:
        child_refs = receipt_payload.get("child_refs")
        checkpoint_id = next(
          (
            str(ref.get("value"))
            for ref in child_refs
            if isinstance(ref, dict) and ref.get("kind") == "accept_checkpoint_id"
          ),
          None,
        ) if isinstance(child_refs, list) else None
        if checkpoint_id is not None:
          prepared_business_model_record = await self._approval_store.transition_prepared_business_model_change(
            caller_kind=prepared_business_model_record.caller_kind,
            user_scope=prepared_business_model_record.user_scope,
            idempotency_locator=prepared_business_model_record.idempotency_locator,
            expected=PreparedBusinessModelLifecycle.AUTHORIZED,
            target=PreparedBusinessModelLifecycle.CONSUMED,
            approval_id=prepared_business_model_record.approval_id,
            approval_chain_id=prepared_business_model_record.approval_chain_id,
            execution_receipt=receipt_bytes,
            checkpoint_id=checkpoint_id,
            consumed_at=utc_now().isoformat(),
          )

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
  ) -> dict[str, Any]:
    return await _approval_lifecycle_helpers.run_approval_lifecycle(
      store=self._approval_store,
      policy=self._approval_policy,
      session=self._session,
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_input=tool_input,
      qualifier=qualifier,
      reason=reason,
      allow_persistent=allow_persistent,
      approval_identity=approval_identity,
      prepared_authorization_payload=prepared_authorization_payload,
      prepared_business_model_change=prepared_business_model_change,
      resume_approval_request=resume_approval_request,
      approval_args_redacted=approval_args_redacted,
      approval_args_hash=approval_args_hash,
      session_cache_approved=session_cache_approved,
      automatic_approval_reason=automatic_approval_reason,
      automatic_denial_reason=automatic_denial_reason,
      deny_user_prompt=deny_user_prompt,
      resolve_run_context_fn=self._resolve_run_context,
      current_skill_fn=current_skill,
      redact_for_approval_request_fn=self._redact_for_approval_request,
      resolve_tool_class_fn=self._resolve_tool_class,
      effective_trade_approval_decision_fn=self._effective_trade_approval_decision,
      await_user_approval_via_pending_tools_fn=self._await_user_approval_via_pending_tools,
      approval_queue_timeout_seconds_fn=_approval_queue_timeout_seconds,
    )

  async def _await_user_approval_via_pending_tools(
    self,
    request: PolicyApprovalRequest,
    decision: PolicyApprovalDecision,
    *,
    nonce: str,
    resolved_qualifier: str,
    allow_persistent: bool,
    timeout_seconds: float,
    batch_admission: Any | None = None,
  ) -> dict[str, Any] | None:
    return await _approval_lifecycle_helpers.await_user_approval_via_pending_tools(
      session=self._session,
      approval_store=self._approval_store,
      event_log=self._event_log,
      request=request,
      decision=decision,
      nonce=nonce,
      resolved_qualifier=resolved_qualifier,
      allow_persistent=allow_persistent,
      timeout_seconds=timeout_seconds,
      log=log,
      batch_admission=batch_admission,
    )

  def _resolve_run_context(self) -> RunContext:
    return _approval_lifecycle_helpers.resolve_run_context(
      run_context=self._run_context,
      session=self._session,
      user_id=self._user_id,
      channel=self._channel,
      role=self._role,
      session_id=self._session_id,
      approval_policy=self._approval_policy,
    )

  def _resolve_tool_class(self, tool_name: str) -> str:
    return _approval_lifecycle_helpers.resolve_tool_class(
      tool_name, mcp=self._mcp, resolve_server_policy_tool_class_fn=resolve_server_policy_tool_class
    )

  def _redact_for_approval_request(self, tool_name: str, tool_input: Dict[str, Any]) -> tuple[dict[str, Any], str]:
    return _approval_lifecycle_helpers.redact_for_approval_request(
      tool_name,
      tool_input,
      event_log=self._event_log,
      enrich_trade_approval_args_fn=enrich_trade_approval_args,
      sha256_args_fn=sha256_args,
    )

  def _approval_transport_input(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    return _approval_lifecycle_helpers.approval_transport_input(
      tool_name,
      tool_input,
      event_log=self._event_log,
      enrich_trade_approval_args_fn=enrich_trade_approval_args,
    )

  def _mcp_tool_argument_guidance(self, tool_name: str) -> str | None:
    guidance_name = tool_name
    get_original_tool_name = getattr(self._mcp, "get_original_tool_name", None)
    if callable(get_original_tool_name):
      try:
        guidance_name = str(get_original_tool_name(tool_name) or tool_name)
      except Exception:
        guidance_name = tool_name
    return tool_argument_guidance(guidance_name)

  def resolve_effective_tool_input(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    return self._apply_dispatch_scope_default(tool_name, tool_input)

  def _apply_dispatch_scope_default(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name in self._local:
      return tool_input
    if not self._mcp.is_mcp_tool(tool_name):
      return tool_input
    server = self._mcp.get_server_for_tool(tool_name)
    if not self._is_portfolio_mcp_server(server):
      return tool_input
    if not isinstance(tool_input, dict):
      return tool_input
    if self._has_explicit_portfolio_scope(tool_input):
      return tool_input
    scope = self._portfolio_dispatch_scope()
    if scope is None:
      return tool_input
    accepted_fields = self._portfolio_scope_fields_for_tool(tool_name, server)
    if not accepted_fields:
      return tool_input
    additions: dict[str, str] = {}
    portfolio_id = _scope_text(scope.get("portfolio_id"))
    portfolio_name = _scope_text(scope.get("portfolio_name"))
    if "portfolio_id" in accepted_fields and portfolio_id is not None:
      additions["portfolio_id"] = portfolio_id
    if "portfolio_name" in accepted_fields and portfolio_name is not None:
      additions["portfolio_name"] = portfolio_name
    if not additions:
      return tool_input
    return {**tool_input, **additions}

  @staticmethod
  def _is_portfolio_mcp_server(server: str | None) -> bool:
    return (
      isinstance(server, str)
      and server.startswith("portfolio-")
      and server.endswith("-mcp")
    )

  @staticmethod
  def _has_explicit_portfolio_scope(tool_input: dict[str, Any]) -> bool:
    for key in _PORTFOLIO_SCOPE_FIELDS:
      if key not in tool_input:
        continue
      value = tool_input.get(key)
      if value is None:
        continue
      if isinstance(value, str) and not value.strip():
        continue
      return True
    return False

  def _portfolio_dispatch_scope(self) -> dict[str, Any] | None:
    scope = getattr(self._session, "dispatch_scope", None)
    if not isinstance(scope, dict):
      return None
    if scope.get("kind") != "portfolio":
      return None
    if scope.get("source") not in {"active_default", "user_selected"}:
      return None
    if _scope_text(scope.get("portfolio_name")) is None:
      return None
    return scope

  def _portfolio_scope_fields_for_tool(self, tool_name: str, server: str) -> set[str]:
    try:
      definitions = self._get_tool_definitions() if self._get_tool_definitions is not None else ()
    except Exception:
      return set()
    original_name = self._original_tool_name(tool_name)
    candidate_names = {tool_name, original_name, f"mcp__{server}__{original_name}"}
    for definition in definitions:
      if not isinstance(definition, dict):
        continue
      if str(definition.get("name") or "") not in candidate_names:
        continue
      schema = definition.get("input_schema")
      if schema is None:
        schema = definition.get("inputSchema")
      return set(_schema_properties(schema)) & _PORTFOLIO_SCOPE_FIELDS
    return set()

  def _original_tool_name(self, tool_name: str) -> str:
    get_original_tool_name = getattr(self._mcp, "get_original_tool_name", None)
    if callable(get_original_tool_name):
      try:
        original = str(get_original_tool_name(tool_name) or tool_name)
        if original:
          return original
      except Exception:
        pass
    if tool_name.startswith("mcp__"):
      parts = tool_name.split("__", 2)
      if len(parts) == 3 and parts[2]:
        return parts[2]
    return tool_name

  def _effective_trade_approval_decision(
    self,
    tool_name: str,
    tool_args_redacted: Dict[str, Any],
    decision: PolicyApprovalDecision,
  ) -> PolicyApprovalDecision:
    return _approval_lifecycle_helpers.effective_trade_approval_decision(
      tool_name,
      tool_args_redacted,
      decision,
      effective_trade_approval_expiry_seconds_fn=effective_trade_approval_expiry_seconds,
      approval_wait_seconds_fn=approval_settings.approval_wait_seconds,
      utc_now_fn=utc_now,
    )

  async def _emit_execution_audit(
    self,
    request: PolicyApprovalRequest | None,
    raw_tool_args: Dict[str, Any],
    *,
    outcome: str,
    error_summary: str | None = None,
  ) -> None:
    await _audit_helpers.emit_execution_audit(
      request,
      raw_tool_args,
      approval_store=self._approval_store,
      outcome=outcome,
      error_summary=error_summary,
    )

  async def _call_mcp_tool(
    self,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    meta: Dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    user_id: str | int | None = None,
  ) -> ToolResult:
    kwargs: Dict[str, Any] = {}
    effective_meta = dict(meta or {})
    commercial = self._commercial_work_start
    server = self._mcp.get_server_for_tool(tool_name)
    if commercial is not None and server in self._commercial_mcp_servers:
      claim = commercial.claim
      authorization = commercial.authorization
      effective_meta["hank_commercial"] = {
        "tool_name": tool_name,
        "execution_context_id": str(claim.context_id),
        "work_authorization_id": str(authorization.authorization_id),
        "workflow_run_id": str(authorization.workflow_run_id),
        "entitlement_revision": claim.entitlement_revision,
        "request_id": authorization.request_id,
        "session_id": authorization.session_id,
        "operation": authorization.operation,
        "capability_id": authorization.capability_id,
        "provider": authorization.provider,
        "billing_mode": authorization.billing_mode,
      }
    if effective_meta:
      kwargs["meta"] = effective_meta
    if abort_event is not None and self._mcp_accepts_abort_event:
      kwargs["abort_event"] = abort_event
    if user_id is not None:
      kwargs["user_id"] = user_id
    result, error = await self._mcp.call_tool(tool_name, tool_input, **kwargs)
    if isinstance(error, dict) and "tool_usage_hint" not in error and _is_mcp_validation_error(error):
      hint = self._mcp_tool_argument_guidance(tool_name)
      if hint:
        error = dict(error)
        error["tool_usage_hint"] = hint
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
    return _source_pack_helpers.planner_result_payload_with_hooks(
      result,
      candidates_fn=cls._planner_result_candidates,
      looks_like_fn=cls._looks_like_source_pack_payload,
      coerce_fn=cls._coerce_planner_result_payload,
    )

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
    return _source_pack_helpers.derive_fiscal_period(
      tool_input,
      planner_result,
      payload_get_fn=cls._payload_get,
    )

  @staticmethod
  def _callable_accepts_kw(callback: Any, keyword: str) -> bool:
    return _runtime_helpers.callable_accepts_kw(callback, keyword)

  @staticmethod
  def _normalize_needs_approval(
    needs_approval: Callable[..., bool] | None,
  ) -> NeedsApprovalCallback:
    return _runtime_helpers.normalize_needs_approval(needs_approval)

  @staticmethod
  def _qualified_key(tool_name: str, qualifier: str) -> str:
    return _runtime_helpers.qualified_key(tool_name, qualifier)

  def _should_request_approval(
    self,
    tool_name: str,
    tool_input: Dict[str, Any],
    qualifier: str,
  ) -> bool:
    return _runtime_helpers.should_request_approval(
      tool_name,
      tool_input,
      qualifier,
      session_cache_denied=self._session_cache_denied,
      approved_tool_types=self._approved_tool_types,
      needs_approval=self._needs_approval,
      qualified_key_fn=self._qualified_key,
    )

  def _tool_was_cache_hit(self, tool_name: str, qualifier: str) -> bool:
    return _runtime_helpers.tool_was_cache_hit(
      tool_name,
      qualifier,
      session_cache_denied=self._session_cache_denied,
      approved_tool_types=self._approved_tool_types,
      qualified_key_fn=self._qualified_key,
    )

  def _emit_approval_decided(
    self,
    tool_call_id: str,
    tool_name: str,
    *,
    outcome: str,
    decision_source: str,
    allow_tool_type_applied: bool,
  ) -> None:
    _audit_helpers.emit_approval_decided(
      self._event_log,
      tool_call_id,
      tool_name,
      outcome=outcome,
      decision_source=decision_source,
      allow_tool_type_applied=allow_tool_type_applied,
      time_fn=time.time,
    )
