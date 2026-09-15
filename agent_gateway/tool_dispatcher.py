from __future__ import annotations

from datetime import datetime
import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import time
from typing import AbstractSet, Any, Callable, cast, Dict, get_args, Mapping, Optional, Protocol, Sequence, Set, TYPE_CHECKING, TypeGuard

from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  ToolRegistrationCatalog,
  ToolRegistrationDeclaration,
  validate_tool_registration_catalog,
  validate_tool_registration_declaration,
)

from . import approval_settings
from .approval_policy import (
  ApprovalConstraint,
  ApprovalConstraintError,
  ApprovalReuseMode,
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest as PolicyApprovalRequest,
  RunContext,
  ToolClass,
  approval_is_executable,
  utc_now,
)
from .approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args
from .approval_route import (
  NO_APPROVAL_ROUTE,
  ApprovalRoute,
  DurableLocalApprovalRoute,
  NoApprovalRoute,
  ParentDelegatedApprovalRoute,
  route_policy,
  route_store,
)
from .event_log import EventLog
from .execution_identity import DispatchIdentity
from .investment_capability_claim import (
  INVESTMENT_CAPABILITY_CLAIM_SERVER,
  INVESTMENT_CAPABILITY_FACADE_TOOLS,
  InvestmentCapabilityClaimError,
  investment_capability_claim_unavailable_error,
  issue_investment_capability_claim,
)
from .mcp_client_catalog import tool_argument_guidance
from .mcp_client import RegisteredMcpPlannedToolCall, RegisteredMcpToolCall, registered_mcp_dispatch_scope
from .policy_imports import (
  load_server_policy_module,
  authority_policy_denies_tool,
  resolve_effective_role,
  resolve_server_policy_tool_class,
)
from .run_identity import (
  MODEL_RUN_IDENTITY_LOCAL_TOOLS,
  MODEL_RUN_IDENTITY_MCP_TOOLS,
  RunIdentityCarrier,
  RunIdentityCarrierError,
  mcp_metadata_skill_run_id,
  validate_run_identity,
)
from .secret_boundary import (
  SANITIZATION_FAILED,
  SecretBoundary,
  sanitize_boundary_value,
)
from .tool_policy_registry import (
  ApprovalCacheKeyCall,
  ApprovalPredicateCall,
  InputPreparationCall,
  OutcomeCall,
  PreparedToolCall,
  RedactionCall,
  SourceIdentityCall,
  ToolInputPreparationError,
  ToolPolicyImplementationRegistry,
)
from .skill_context import current_skill, current_skill_admission
from .skill_limits import reconcile_skill_admission
from . import tool_dispatcher_audit as _audit_helpers
from . import tool_dispatcher_approval_lifecycle as _approval_lifecycle_helpers
from . import tool_dispatcher_runtime as _runtime_helpers
from . import tool_dispatcher_skill_tools as _skill_tool_helpers
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
  run_interceptors as _run_interceptors_helper,
  tool_input_schema_error as _tool_input_schema_error_helper,
  validate_against_local_schema as _validate_against_local_schema_helper,
  validate_local_tool_input as _validate_local_tool_input_helper,
)
from .tool_dispatch_classification import (
  DispatchEntry,
  OUTCOME_OK,
  ToolResultSettlement,
  is_mcp_validation_error as _is_mcp_validation_error,
  settle_catalogless_tool_result,
)

if TYPE_CHECKING:
  from .approval_store import TargetedPreparedReconciliationResult
  from .mcp_client import McpClientManager
  from .prepared_business_model_store import (
    PreparedBusinessModelChange,
    PreparedBusinessModelLifecycle,
  )

log = logging.getLogger("agent_gateway.dispatcher")


_PORTFOLIO_SCOPE_FIELDS = frozenset({"portfolio_id", "portfolio_name"})
_CATALOG_ACTION_UNSET = object()
_TOOL_CLASSES = frozenset(get_args(ToolClass))

class ToolDispatcherApprovalStore(Protocol):
  """Prepared-plan methods called by the dispatch gate."""

  async def get(
    self,
    approval_id: str,
  ) -> PolicyApprovalRequest | None: ...

  async def get_prepared_business_model_change(
    self,
    *,
    caller_kind: str,
    user_scope: str,
    idempotency_locator: str,
  ) -> PreparedBusinessModelChange | None: ...

  async def reconcile_prepared_business_model_change(
    self,
    *,
    caller_kind: str,
    user_scope: str,
    idempotency_locator: str,
    now: datetime | None = None,
  ) -> TargetedPreparedReconciliationResult: ...

  async def transition_prepared_business_model_change(
    self,
    *,
    caller_kind: str,
    user_scope: str,
    idempotency_locator: str,
    expected: PreparedBusinessModelLifecycle,
    target: PreparedBusinessModelLifecycle,
    approval_id: str | None = None,
    approval_chain_id: str | None = None,
    execution_receipt: bytes | None = None,
    restoration_digest: str | None = None,
    checkpoint_id: str | None = None,
    consumed_at: str | None = None,
  ) -> PreparedBusinessModelChange: ...


class _ToolDispatcherApprovalAuthority(
  ToolDispatcherApprovalStore,
  _approval_lifecycle_helpers._ApprovalLifecycleAuthority,
  Protocol,
):
  """The route aggregate needed by dispatch and its lifecycle delegate."""


class RegisteredApprovalPolicyError(RuntimeError):
  """A registered approval policy could not produce a safe decision."""


def resolve_registered_addin_declaration(
  catalog: ToolRegistrationCatalog | None,
  tool_name: str,
) -> ToolRegistrationDeclaration | None:
  """Resolve one exact add-in declaration without bare-name fallback."""

  if catalog is None:
    return None
  try:
    declaration = catalog.by_identity(RegisteredToolIdentity(
      route_kind="addin_relay",
      logical_server_id=None,
      logical_name=tool_name,
    ))
  except LookupError:
    return None
  policy = declaration.semantics.input_preparation_policy
  if (
    declaration.identity.route_kind != "addin_relay"
    or policy.policy_id != "addin-workbook-context"
    or policy.version != "v1"
  ):
    raise RuntimeError(
      "registered add-in route lacks exact workbook preparation"
    )
  return declaration


def execute_registered_addin_input_preparation(
  catalog: ToolRegistrationCatalog,
  registry: ToolPolicyImplementationRegistry,
  declaration: ToolRegistrationDeclaration,
  tool_input: Mapping[str, Any],
  trusted_context: object | None,
) -> PreparedToolCall:
  """Execute one authoritative registered add-in input policy."""

  canonical = validate_tool_registration_declaration(declaration)
  exact = resolve_registered_addin_declaration(
    catalog,
    canonical.identity.logical_name,
  )
  if exact is None or exact != canonical:
    raise RuntimeError("registered add-in declaration is not authoritative")
  return registry.execute_input_preparation(
    exact.semantics.input_preparation_policy,
    InputPreparationCall(
      exact.identity,
      tool_input,
      trusted_context,
    ),
  )


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


class _BoundaryEventLogProjection:
  """Sanitize dispatcher-owned durable events while preserving log reads."""

  def __init__(self, owner: "ToolDispatcher") -> None:
    self._owner = owner

  def append(self, event: dict[str, Any]) -> Any | None:
    return self._owner._append_event(event)

  def __getattr__(self, name: str) -> Any:
    event_log = self._owner._event_log
    if event_log is None:
      raise AttributeError(name)
    return getattr(event_log, name)


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
    identity: DispatchIdentity | None = None,
    user_id: str | None = None,
    risk_user_id: int | None = None,
    channel: str | None = None,
    role: str | None = None,
    credentials_resolver_active: bool = False,
    session_cache_denied_tools: frozenset[str] | None = None,
    session: Any | None = None,
    approval_route: ApprovalRoute = NO_APPROVAL_ROUTE,
    run_context: RunContext | None = None,
    get_tool_definitions: Callable[[], list[dict[str, Any]]] | None = None,
    allowed_mcp_tools_by_server: Mapping[str, AbstractSet[str]] | None = None,
    mcp_scope_context: str = "skill",
    describe_mcp_scope_block: Callable[[str | None, str], str | None] | None = None,
    commercial_work_start: Any | None = None,
    commercial_irreversible_recheck: Callable[[Any], None] | None = None,
    commercial_mcp_servers: frozenset[str] | None = None,
    local_tool_class_resolver: Callable[[str], ToolClass] | None = None,
    local_catalog_action_resolver: Callable[[str], Any | None] | None = None,
    plan_validator: Callable[..., Mapping[str, Any]] | None = None,
    plan_review_renderer: Callable[..., dict[str, Any]] | None = None,
    tool_registration_catalog: ToolRegistrationCatalog | None = None,
    tool_policy_implementations: ToolPolicyImplementationRegistry | None = None,
    input_preparation_context_factory: (
      Callable[[ToolRegistrationDeclaration], object | None] | None
    ) = None,
    redaction_context_factory: (
      Callable[[ToolRegistrationDeclaration], object | None] | None
    ) = None,
    approval_predicate_context_factory: (
      Callable[
        [ToolRegistrationDeclaration, PreparedToolCall],
        object | None,
      ] | None
    ) = None,
    registered_approval_overlay: (
      Callable[[ToolRegistrationDeclaration, PreparedToolCall], bool] | None
    ) = None,
  ) -> None:
    self._mcp = mcp_client
    self._local = local_tool_handlers or {}
    self._plan_validator = plan_validator
    self._plan_review_renderer = plan_review_renderer
    if (local_tool_class_resolver is None) != (
      local_catalog_action_resolver is None
    ):
      raise ValueError(
        "local tool class and catalog action resolvers must be provided together"
      )
    if local_tool_class_resolver is not None and session is not None:
      raise ValueError(
        "caller-owned local tool policy is available only without a host session"
      )
    self._local_tool_classes: dict[str, ToolClass] | None = None
    self._local_catalog_actions: dict[str, Any | None] | None = None
    if (tool_registration_catalog is None) != (
      tool_policy_implementations is None
    ):
      raise ValueError(
        "tool registration catalog and policy implementations must be provided together"
      )
    self._tool_registration_catalog = (
      validate_tool_registration_catalog(tool_registration_catalog)
      if tool_registration_catalog is not None
      else None
    )
    if (
      tool_policy_implementations is not None
      and type(tool_policy_implementations)
      is not ToolPolicyImplementationRegistry
    ):
      raise TypeError(
        "tool_policy_implementations must be an exact ToolPolicyImplementationRegistry"
      )
    if tool_policy_implementations is not None:
      assert self._tool_registration_catalog is not None
      tool_policy_implementations.validate_catalog(
        self._tool_registration_catalog
      )
    if (
      input_preparation_context_factory is not None
      and not callable(input_preparation_context_factory)
    ):
      raise TypeError("input_preparation_context_factory must be callable")
    if (
      input_preparation_context_factory is not None
      and tool_policy_implementations is None
    ):
      raise ValueError(
        "input preparation context requires registered policy implementations"
      )
    if (
      redaction_context_factory is not None
      and not callable(redaction_context_factory)
    ):
      raise TypeError("redaction_context_factory must be callable")
    if (
      redaction_context_factory is not None
      and tool_policy_implementations is None
    ):
      raise ValueError(
        "redaction context requires registered policy implementations"
      )
    if (
      self._tool_registration_catalog is not None
      and redaction_context_factory is None
    ):
      raise ValueError(
        "registered tool catalog requires a redaction context factory"
      )
    if (
      approval_predicate_context_factory is not None
      and not callable(approval_predicate_context_factory)
    ):
      raise TypeError("approval_predicate_context_factory must be callable")
    if (
      approval_predicate_context_factory is not None
      and tool_policy_implementations is None
    ):
      raise ValueError(
        "approval predicate context requires registered policy implementations"
      )
    if (
      registered_approval_overlay is not None
      and not callable(registered_approval_overlay)
    ):
      raise TypeError("registered_approval_overlay must be callable")
    if (
      registered_approval_overlay is not None
      and approval_predicate_context_factory is None
    ):
      raise ValueError(
        "registered approval overlay requires registered approval runtime"
      )
    self._tool_policy_implementations = tool_policy_implementations
    self._input_preparation_context_factory = (
      input_preparation_context_factory
    )
    self._redaction_context_factory = redaction_context_factory
    self._prepared_tool_input_redactor = (
      self._redact_catalogless_prepared_tool_input
      if self._tool_registration_catalog is None
      else self._redact_registered_prepared_tool_input
    )
    self._raw_history_input_redactor = (
      self._redact_catalogless_raw_history_input
      if self._tool_registration_catalog is None
      else self._redact_registered_raw_history_input
    )
    self._approval_predicate_context_factory = (
      approval_predicate_context_factory
    )
    self._registered_approval_overlay = registered_approval_overlay
    if (
      local_tool_class_resolver is not None
      and local_catalog_action_resolver is not None
    ):
      local_tool_classes: dict[str, ToolClass] = {}
      local_catalog_actions: dict[str, Any | None] = {}
      for local_tool_name, local_handler in self._local.items():
        tool_class = local_tool_class_resolver(local_tool_name)
        if type(tool_class) is not str or tool_class not in _TOOL_CLASSES:
          raise ValueError(
            f"invalid local tool class for {local_tool_name!r}: {tool_class!r}"
          )
        catalog_action = local_catalog_action_resolver(local_tool_name)
        self._planned_handler_hooks(
          local_tool_name,
          local_handler,
          catalog_action=catalog_action,
        )
        local_tool_classes[local_tool_name] = tool_class
        local_catalog_actions[local_tool_name] = catalog_action
      self._local_tool_classes = local_tool_classes
      self._local_catalog_actions = local_catalog_actions
    self._needs_approval = self._normalize_needs_approval(needs_approval)
    self._request_approval = request_approval
    self._approved_tool_types = approved_tool_types if approved_tool_types is not None else set()
    self._event_log = event_log
    self._boundary_event_log = _BoundaryEventLogProjection(self)
    self._approval_key_qualifier = approval_key_qualifier
    self._interceptors: Sequence[ToolInterceptor] = list(interceptors or [])
    self._session_id = session_id
    self._should_avoid_permission_prompts = should_avoid_permission_prompts
    self._on_headless_ask = on_headless_ask
    self._mcp_session_inject_servers = mcp_session_inject_servers or set()
    self._mcp_meta_inject_servers = mcp_meta_inject_servers or frozenset()
    supplied_session = session
    if identity is not None:
      # D-B6-1: one identity value, not five arguments assembled per call
      # site. Passing both would let the two disagree, so it is refused.
      if (
        session is not None
        or user_id is not None
        or risk_user_id is not None
        or channel is not None
        or credentials_resolver_active
      ):
        raise ValueError(
          "dispatch identity supersedes the separate identity arguments"
        )
      if not isinstance(identity, DispatchIdentity):
        raise TypeError("dispatcher identity must be a DispatchIdentity")
      session = identity.session
      user_id = identity.user_id
      risk_user_id = identity.risk_user_id
      channel = identity.channel
      credentials_resolver_active = identity.credentials_resolver_active
    self._identity = identity
    self._execution_identity = (
      identity.execution if identity is not None else None
    )
    self._user_id = user_id
    self._risk_user_id = risk_user_id
    self._channel = channel
    self._role = resolve_effective_role(role)
    self._credentials_resolver_active = credentials_resolver_active
    self._approval_route = approval_route
    if not isinstance(self._approval_route, NoApprovalRoute):
      # D-B6-1, applied to the other value that carries a session: a live
      # approval route already names the GatewaySession whose ledger row,
      # pending-tools entry and single-slot decision queue record the
      # decision, so that session cannot also arrive separately.
      if supplied_session is not None:
        raise ValueError(
          "approval route supersedes the separate session argument"
        )
      route_session = self._approval_route.session
      if identity is not None and identity.session is not route_session:
        raise ValueError(
          "dispatch identity and approval route name different sessions"
        )
      session = route_session
    self._session_cache_denied = session_cache_denied_tools or frozenset()
    self._source_pack_session = session
    self._session = session
    self._run_context = run_context
    self._get_tool_definitions = get_tool_definitions
    self._mcp_scope_context = mcp_scope_context
    self._describe_mcp_scope_block = describe_mcp_scope_block
    self._commercial_work_start = commercial_work_start
    self._commercial_irreversible_recheck = commercial_irreversible_recheck
    self._commercial_mcp_servers = commercial_mcp_servers or frozenset()
    self._secret_boundary = SecretBoundary()
    if allowed_mcp_tools_by_server is None:
      self._allowed_mcp_tools_by_server = None
    elif isinstance(allowed_mcp_tools_by_server, Mapping):
      # Kept by reference, never copied: the interactive scope is a live
      # derivation over the session's activation fold (T3-I12), and a snapshot
      # here would re-open the desync the fold closes.
      self._allowed_mcp_tools_by_server = allowed_mcp_tools_by_server
    else:
      self._allowed_mcp_tools_by_server = {
        str(server_name): {str(tool_name) for tool_name in tool_names}
        for server_name, tool_names in allowed_mcp_tools_by_server.items()
      }

  @property
  def _approval_store(self) -> _ToolDispatcherApprovalAuthority | None:
    """The durable ledger the admitted route owns, if it owns one."""

    return route_store(self._approval_route)

  def _durable_business_model_store(self) -> ToolDispatcherApprovalStore:
    """Return the ledger admitted for the prepared BusinessModel path."""

    route = self._approval_route
    if isinstance(route, DurableLocalApprovalRoute):
      return route.store
    raise TrustedToolPlanError(
      "prepared BusinessModel lifecycle requires durable local approval custody"
    )

  @property
  def _approval_policy(
    self,
  ) -> _approval_lifecycle_helpers.ApprovalLifecyclePolicy | None:
    """The opaque policy handle the admitted route owns, if it owns one."""

    return route_policy(self._approval_route)

  def bind_secret_boundary(self, boundary: SecretBoundary) -> None:
    """Bind lifecycle-local secret knowledge supplied by the owning runner."""

    if not isinstance(boundary, SecretBoundary):
      raise TypeError("dispatcher secret boundary must be SecretBoundary")
    self._secret_boundary = boundary

  @property
  def run_context(self) -> RunContext | None:
    """The run context policy is enforced against; ``None`` outside a run."""

    return self._run_context

  @property
  def plan_validator(self) -> Callable[..., Mapping[str, Any]] | None:
    """Identity authority inherited by delegated dispatchers."""
    return self._plan_validator

  @property
  def plan_review_renderer(self) -> Callable[..., dict[str, Any]] | None:
    """Product review projection inherited with its plan validator."""
    return self._plan_review_renderer

  def with_scoped_local_handler(
    self,
    tool_name: str,
    scope: Callable[[LocalToolHandler], LocalToolHandler],
  ) -> "ToolDispatcher":
    """Return a shallow clone whose local ``tool_name`` handler is ``scope(stock)``.

    A fork narrows what its parent already dispatches locally; it never adds a
    route the parent lacks, so a missing stock handler raises ``KeyError``.
    """

    stock = self._local.get(tool_name)
    if not callable(stock):
      raise KeyError(tool_name)
    clone = copy.copy(self)
    clone._local = {**self._local, tool_name: scope(stock)}
    return clone

  def _append_event(self, event: dict[str, Any]) -> Any | None:
    if self._event_log is None:
      return None
    projected = sanitize_boundary_value(
      event,
      sink="dispatcher_event",
      boundary=self._secret_boundary,
    )
    if not isinstance(projected, dict):
      projected = {
        "type": "dispatcher_boundary_failure",
        "message": SANITIZATION_FAILED,
      }
    return self._event_log.append(projected)

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    """Return the exact tool catalog enforced by this dispatcher."""
    if self._get_tool_definitions is not None:
      return list(self._get_tool_definitions())
    if self._mcp is not None:
      return list(self._mcp.get_tool_definitions())
    return []

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
      event_log=(
        self._boundary_event_log
        if self._event_log is not None
        else None
      ),
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
      event_log=(
        self._boundary_event_log
        if self._event_log is not None
        else None
      ),
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

  def _wire_mcp_scope_error(
    self,
    tool_name: str,
    server_name: str | None,
    advertised_tool_names: AbstractSet[str] | None,
  ) -> Dict[str, Any] | None:
    """Materialize an MCP provider-request snapshot failure."""

    if advertised_tool_names is None:
      return {
        "code": "mcp_tool_not_allowed",
        "sub_code": "advertisement_unavailable",
        "message": "The advertised MCP tool snapshot is unavailable; dispatch was denied.",
      }
    advertised_scope = (
      {server_name: set(advertised_tool_names)}
      if server_name
      else {}
    )
    return _runtime_helpers.mcp_scope_error(
      tool_name,
      server_name,
      allowed_mcp_tools_by_server=advertised_scope,
      scope_context=self._mcp_scope_context,
      describe_scope_block=self._describe_mcp_scope_block,
    )

  def _deferred_local_pack_for_unadvertised_tool(
    self,
    tool_name: str,
    advertised_tool_names: AbstractSet[str],
  ) -> str | None:
    """Return the one catalog-owned pack this session can load for a local tool."""

    if (
      self._session is None
      or self._role != "owner"
      or tool_name not in self._local
      or "load_tools" not in self._local
      or "load_tools" not in advertised_tool_names
    ):
      return None
    profile_name = (
      str(getattr(self._run_context, "profile", "") or "") or None
    )
    if (
      authority_policy_denies_tool(
        session=self._session,
        role=self._role,
        tool_name=tool_name,
        is_local=True,
        profile_name=profile_name,
      )
      or authority_policy_denies_tool(
        session=self._session,
        role=self._role,
        tool_name="load_tools",
        is_local=True,
        profile_name=profile_name,
      )
    ):
      return None
    loaded_local_tools = getattr(
      self._session,
      "loaded_local_tools",
      None,
    )
    if loaded_local_tools is None or tool_name in loaded_local_tools:
      return None

    policy = load_server_policy_module()
    resolve_pack = getattr(policy, "deferred_local_tool_pack", None)
    return resolve_pack(tool_name, self._channel) if resolve_pack is not None else None

  def _request_advertisement_error(
    self,
    tool_name: str,
    advertised_tool_names: AbstractSet[str] | None,
    *,
    is_mcp: bool,
    server_name: str | None,
  ) -> Dict[str, Any] | None:
    """Apply one provider-request tool-name snapshot to local and MCP routes."""

    if advertised_tool_names is None:
      return (
        self._wire_mcp_scope_error(
          tool_name,
          server_name,
          advertised_tool_names,
        )
        if is_mcp
        else None
      )
    if tool_name in advertised_tool_names:
      return None
    if is_mcp:
      return self._wire_mcp_scope_error(
        tool_name,
        server_name,
        advertised_tool_names,
      )
    deferred_pack = self._deferred_local_pack_for_unadvertised_tool(
      tool_name,
      advertised_tool_names,
    )
    return {
      "code": "tool_not_advertised",
      "message": (
        f"Tool '{tool_name}' was not advertised for this provider request"
      ),
      "details": {"tool_name": tool_name},
      "fix": (
        "Call only tools advertised for this provider request."
        if deferred_pack is None
        else (
          f"This tool is in deferred tool pack '{deferred_pack}'; "
          f'call load_tools(pack="{deferred_pack}") before retrying.'
        )
      ),
    }

  @staticmethod
  def _catalog_action(tool_name: str) -> Any | None:
    """Read the embedding application's explicitly bound action catalog."""
    policy = load_server_policy_module()
    resolver = getattr(policy, "catalog_action_for_tool", None)
    return resolver(tool_name) if resolver is not None else None

  def _resolved_catalog_action(self, tool_name: str) -> Any | None:
    if self._local_catalog_actions is not None and tool_name in self._local:
      return self._local_catalog_actions[tool_name]
    return self._catalog_action(tool_name)

  @staticmethod
  def _catalog_planning_identity(tool_name: str) -> str | None:
    """Read the one authoritative action catalog; never infer from a handler."""

    action = ToolDispatcher._catalog_action(tool_name)
    return None if action is None else action.planning_identity

  @staticmethod
  def _planned_handler_hooks(
    tool_name: str,
    handler: LocalToolHandler,
    *,
    catalog_action: Any = _CATALOG_ACTION_UNSET,
  ) -> tuple[str, Callable[..., Any], Callable[..., Any]] | None:
    action = (
      ToolDispatcher._catalog_action(tool_name)
      if catalog_action is _CATALOG_ACTION_UNSET
      else catalog_action
    )
    required_identity = None if action is None else action.planning_identity
    identity_marker = getattr(handler, "PLANNING_IDENTITY", None)
    planner = getattr(handler, "plan_change", None)
    executor = getattr(handler, "execute_prepared_change", None)
    if (
      identity_marker is None
      and not callable(planner)
      and not callable(executor)
    ):
      if required_identity is not None:
        raise TrustedToolPlanError(
          f"catalogued exact-write tool {tool_name!r} lost its planning hooks"
        )
      return None
    if (
      identity_marker is None
      or not callable(planner)
      or not callable(executor)
    ):
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

  async def _plan_local_write(
    self,
    hooks: tuple[str, Callable[..., Any], Callable[..., Any]],
    tool_input: Dict[str, Any],
    *,
    call_index: int,
    tool_ctx: ToolExecutionContext,
    own_prepared: Callable[[Any], None] | None = None,
  ) -> tuple[TrustedToolPlan, Callable[..., Any]]:
    if self._plan_validator is None:
      raise TrustedToolPlanError("planned local handler requires an identity validator")
    identity_source, planner, executor = hooks
    planned = planner(tool_input, call_index=call_index, tool_ctx=tool_ctx)
    if inspect.isawaitable(planned):
      planned = await planned
    if type(planned) is not tuple or len(planned) != 2:
      raise TrustedToolPlanError(
        "planned local handler must return exactly (identity, prepared_payload)"
      )
    identity, prepared = planned
    if own_prepared is not None:
      own_prepared(prepared)
    trusted_plan = TrustedToolPlan.create(
      identity_source=identity_source,
      identity=identity,
      prepared=prepared,
      validator=self._plan_validator,
      review_renderer=self._plan_review_renderer,
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
    if prepared_accept is None or not callable(serializer):
      # Blocked/non-success verdicts intentionally carry no accepted business
      # model. They still execute through the generic exact-plan lifecycle;
      # only an actual accept is eligible for the stronger durable replay row.
      if (
        finalizer is None
        and prepared_accept is None
        and hasattr(completion, "projection")
      ):
        return None
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
      event_log=(
        self._boundary_event_log
        if self._event_log is not None
        else None
      ),
      resolved_qualifier=qualifier,
      abort_event=abort_event,
      skill_run_id=skill_run_id,
      step_id=step_id,
      workspace_dir=workspace_dir,
      batch_id=batch_id,
      request_id=run_context.request_id,
      run_id=run_context.run_id,
      user_id=run_context.user_id,
      ui_blocks_run=run_context.ui_blocks_run,
    )

  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    call_index: int = 0,
    advertised_tool_names: AbstractSet[str] | None = None,
    abort_event: asyncio.Event | None = None,
    skill_run_id: str | None = None,
    step_id: str | None = None,
    workspace_dir: str | None = None,
    batch_id: int | str | None = None,
    capture_readable_resource_snapshot: bool = False,
    allow_uncertain_mcp_replay: bool = True,
    on_executed_prepared_call: (
      Callable[[PreparedToolCall], None] | None
    ) = None,
  ) -> ToolResult:
    """Execute one tool call while owning any private planning snapshot."""
    if type(allow_uncertain_mcp_replay) is not bool:
      raise TypeError("allow_uncertain_mcp_replay must be an exact bool")
    try:
      prepared_call = self.prepare_tool_call(
        tool_name,
        tool_input,
      )
    except ToolInputPreparationError as exc:
      return None, exc.materialize_error()
    except Exception as exc:
      log.error(
        "Tool input preparation failed for %s | exception_type=%s",
        tool_name,
        type(exc).__name__,
      )
      return None, {
        "code": "tool_input_preparation_failed",
        "message": f"Tool '{tool_name}' input could not be prepared for dispatch.",
      }
    return await self.dispatch_prepared(
      tool_call_id,
      tool_name,
      prepared_call,
      call_index=call_index,
      advertised_tool_names=advertised_tool_names,
      abort_event=abort_event,
      skill_run_id=skill_run_id,
      step_id=step_id,
      workspace_dir=workspace_dir,
      batch_id=batch_id,
      capture_readable_resource_snapshot=capture_readable_resource_snapshot,
      allow_uncertain_mcp_replay=allow_uncertain_mcp_replay,
      on_executed_prepared_call=on_executed_prepared_call,
    )

  def prepare_tool_call(
    self,
    tool_name: str,
    tool_input: Mapping[str, Any],
  ) -> PreparedToolCall:
    """Prepare one immutable call before redaction, approval, and retry."""

    if (
      tool_name not in self._local
      and self._mcp.is_mcp_tool(tool_name)
      and callable(getattr(self._mcp, "uses_registered_tool_catalog", None))
      and self._mcp.uses_registered_tool_catalog()
    ):
      return self._mcp.prepare_registered_tool_input(
        tool_name,
        tool_input,
        self._portfolio_dispatch_scope(),
      )
    declaration = self._registered_code_declaration(tool_name)
    if declaration is not None:
      registry = self._tool_policy_implementations
      context_factory = self._input_preparation_context_factory
      if registry is None or context_factory is None:
        raise RuntimeError(
          "registered code input-preparation runtime is not configured"
        )
      return registry.execute_input_preparation(
        declaration.semantics.input_preparation_policy,
        InputPreparationCall(
          declaration.identity,
          tool_input,
          context_factory(declaration),
        ),
      )
    return PreparedToolCall(
      self.resolve_effective_tool_input(tool_name, dict(tool_input))
    )

  def _registered_code_declaration(
    self,
    tool_name: str,
  ) -> ToolRegistrationDeclaration | None:
    if tool_name != "code_execute" or tool_name not in self._local:
      return None
    declaration = self.registered_tool_declaration(tool_name)
    if declaration is None:
      return None
    if declaration.identity.route_kind != "local_handler":
      raise RuntimeError(
        "registered code_execute route must be an exact local_handler"
      )
    if (
      declaration.semantics.input_preparation_policy.policy_id
      != "code-execution-backend"
    ):
      raise RuntimeError(
        "registered code_execute route lacks exact backend preparation"
      )
    return declaration

  def registered_tool_declaration(
    self,
    tool_name: str,
  ) -> ToolRegistrationDeclaration | None:
    """Resolve one exact installed local or live MCP declaration."""

    catalog = self._tool_registration_catalog
    if catalog is None:
      return None
    if tool_name in self._local:
      try:
        return catalog.by_identity(RegisteredToolIdentity(
          route_kind="local_handler",
          logical_server_id=None,
          logical_name=tool_name,
        ))
      except LookupError:
        return None
    if (
      not self._mcp.is_mcp_tool(tool_name)
      or not callable(getattr(self._mcp, "uses_registered_tool_catalog", None))
      or not self._mcp.uses_registered_tool_catalog()
    ):
      return None
    if TYPE_CHECKING:
      get_descriptor = self._mcp.get_registered_mcp_tool_descriptor
    else:
      get_descriptor = getattr(
        self._mcp,
        "get_registered_mcp_tool_descriptor",
        None,
      )
    if not callable(get_descriptor):
      raise RuntimeError("registered MCP descriptor lookup is not configured")
    declaration = validate_tool_registration_declaration(
      get_descriptor(tool_name).declaration
    )
    if catalog.by_identity(declaration.identity) != declaration:
      raise RuntimeError(
        "live registered MCP declaration does not match dispatcher catalog"
      )
    return declaration

  def redact_prepared_tool_input(
    self,
    tool_name: str,
    prepared_call: PreparedToolCall,
  ) -> dict[str, object]:
    """Redact through the live route, or generically when no route is active."""

    if type(prepared_call) is not PreparedToolCall:
      raise TypeError("prepared_call must be an exact PreparedToolCall")
    return self._prepared_tool_input_redactor(tool_name, prepared_call)

  def redact_raw_tool_input_for_history(
    self,
    tool_name: str,
    tool_input: Mapping[str, object],
  ) -> dict[str, object]:
    """Project raw provider input through the construction-selected owner."""

    return self._raw_history_input_redactor(tool_name, tool_input)

  @staticmethod
  def _redact_catalogless_prepared_tool_input(
    tool_name: str,
    prepared_call: PreparedToolCall,
  ) -> dict[str, object]:
    from .runner_tool_audit import redact_tool_input_for_event

    return redact_tool_input_for_event(
      tool_name,
      prepared_call.materialize_input(),
    )

  @staticmethod
  def _redact_catalogless_raw_history_input(
    tool_name: str,
    tool_input: Mapping[str, object],
  ) -> dict[str, object]:
    from .runner_tool_audit import redact_tool_input_for_event

    return redact_tool_input_for_event(tool_name, dict(tool_input))

  def _redact_registered_prepared_tool_input(
    self,
    tool_name: str,
    prepared_call: PreparedToolCall,
  ) -> dict[str, object]:
    self.ensure_gateway_local_tool_handler(tool_name)
    if tool_name in self._local:
      declaration = self._required_registered_tool_declaration(tool_name)
      return self._redact_registered_declaration_input(
        declaration,
        prepared_call.materialize_input(),
      )
    if self._mcp.is_mcp_tool(tool_name):
      return self._mcp.redact_registered_tool_input(
        tool_name,
        prepared_call,
      )
    return self._redact_catalogless_prepared_tool_input(
      tool_name,
      prepared_call,
    )

  def _redact_registered_raw_history_input(
    self,
    tool_name: str,
    tool_input: Mapping[str, object],
  ) -> dict[str, object]:
    self.ensure_gateway_local_tool_handler(tool_name)
    if tool_name not in self._local:
      return self._mcp.redact_registered_raw_tool_input(
        tool_name,
        tool_input,
      )
    declaration = self._required_registered_tool_declaration(tool_name)
    return self._redact_registered_declaration_input(
      declaration,
      tool_input,
    )

  def _redact_registered_declaration_input(
    self,
    declaration: ToolRegistrationDeclaration,
    tool_input: Mapping[str, object],
  ) -> dict[str, object]:
    registry = cast(
      ToolPolicyImplementationRegistry,
      self._tool_policy_implementations,
    )
    context_factory = cast(
      Callable[[ToolRegistrationDeclaration], object | None],
      self._redaction_context_factory,
    )
    result = registry.execute_redaction(
      declaration.semantics.redaction_policy,
      RedactionCall(
        declaration.identity,
        tool_input,
        context_factory(declaration),
      ),
    )
    return result.materialize_input()

  def _required_registered_tool_declaration(
    self,
    tool_name: str,
  ) -> ToolRegistrationDeclaration:
    catalog = cast(ToolRegistrationCatalog, self._tool_registration_catalog)
    if tool_name in self._local:
      return catalog.by_identity(RegisteredToolIdentity(
        route_kind="local_handler",
        logical_server_id=None,
        logical_name=tool_name,
      ))
    descriptor = self._mcp.get_registered_mcp_tool_descriptor(tool_name)
    declaration = validate_tool_registration_declaration(
      descriptor.declaration
    )
    if catalog.by_identity(declaration.identity) != declaration:
      raise RuntimeError(
        "live registered MCP declaration does not match dispatcher catalog"
      )
    return declaration

  def settle_tool_result(
    self,
    tool_name: str,
    dispatch_entry: DispatchEntry | None,
    result: Any,
    error: Mapping[str, Any] | None,
    semantic_error: Mapping[str, Any] | None = None,
    *,
    prepared_call: PreparedToolCall,
  ) -> ToolResultSettlement:
    """Settle outcome and sources through one route-owned policy mode."""

    declaration = self.registered_tool_declaration(tool_name)
    if declaration is None:
      return settle_catalogless_tool_result(
        entry=dispatch_entry,
        result=result,
        error=error,
        semantic_error=semantic_error,
      )
    outcome = self.settle_registered_outcome(
      declaration,
      result,
      error,
      semantic_error,
    )
    if outcome != OUTCOME_OK:
      return ToolResultSettlement(outcome=outcome)
    registry = self._tool_policy_implementations
    assert registry is not None
    sources = registry.execute_source_identity(
      declaration.semantics.source_identity_policy,
      SourceIdentityCall(
        declaration.identity,
        result,
        prepared_call.materialize_input(),
        tool_name,
      ),
    )
    return ToolResultSettlement(outcome=outcome, sources=sources.identities)

  def registered_outcomes_configured(self) -> bool:
    """Return whether this dispatcher was built with registered policy owners."""

    return self._tool_policy_implementations is not None

  def uses_registered_tool_catalog(self) -> bool:
    """Return whether this dispatcher owns registered tool policy."""

    return self._tool_registration_catalog is not None

  def settle_registered_outcome(
    self,
    declaration: ToolRegistrationDeclaration,
    result: Any,
    error: Mapping[str, Any] | None,
    semantic_error: Mapping[str, Any] | None = None,
  ) -> str:
    """Execute the outcome policy owned by one exact registered route."""

    registry = self._tool_policy_implementations
    if registry is None:
      raise RuntimeError("registered outcome implementations are not configured")
    return registry.execute_outcome(
      declaration.semantics.outcome_policy,
      OutcomeCall(result, error, semantic_error),
    )

  def settle_registered_addin_outcome(
    self,
    tool_name: str,
    result: Any,
    error: Mapping[str, Any] | None,
    semantic_error: Mapping[str, Any] | None = None,
  ) -> str:
    """Execute outcome semantics for one exact registered add-in route."""

    catalog = cast(ToolRegistrationCatalog, self._tool_registration_catalog)
    declaration = catalog.by_identity(RegisteredToolIdentity(
      route_kind="addin_relay",
      logical_server_id=None,
      logical_name=tool_name,
    ))
    return self.settle_registered_outcome(
      declaration,
      result,
      error,
      semantic_error,
    )

  def _registered_approval_declaration(
    self,
    tool_name: str,
  ) -> ToolRegistrationDeclaration | None:
    """Return the exact installed route declaration when approval is bound."""

    if self._approval_predicate_context_factory is None:
      return None
    return self.registered_tool_declaration(tool_name)

  def registered_approval_requirement(
    self,
    declaration: ToolRegistrationDeclaration,
    prepared_call: PreparedToolCall,
    trusted_plan: TrustedToolPlan | None = None,
  ) -> tuple[bool, str | None, bool]:
    """Return (approval required, safe cache key, cache hit)."""

    registry = self._tool_policy_implementations
    context_factory = self._approval_predicate_context_factory
    if registry is None:
      raise RuntimeError("registered approval runtime is not configured")
    if type(prepared_call) is not PreparedToolCall:
      raise TypeError("prepared_call must be an exact PreparedToolCall")

    policy = declaration.semantics.approval
    if policy.mode == "never":
      intrinsic_required = False
    elif policy.mode == "always":
      intrinsic_required = True
    else:
      assert policy.predicate is not None
      if context_factory is None:
        raise RuntimeError(
          "registered approval predicate context is not configured"
        )
      try:
        intrinsic_required = registry.execute_approval_predicate(
          policy.predicate,
          ApprovalPredicateCall(
            declaration.identity,
            prepared_call.prepared_input,
            context_factory(declaration, prepared_call),
          ),
        )
      except Exception as exc:
        log.error(
          "Registered approval predicate failed for %s | exception_type=%s",
          declaration.identity.registration_key,
          type(exc).__name__,
        )
        raise RegisteredApprovalPolicyError(
          "registered approval predicate failed"
        ) from exc

    overlay_required = False
    overlay = self._registered_approval_overlay
    if overlay is not None:
      try:
        overlay_result = overlay(declaration, prepared_call)
        if type(overlay_result) is not bool:
          raise TypeError("registered approval overlay must return an exact bool")
        overlay_required = overlay_result
      except Exception as exc:
        log.error(
          "Registered approval overlay failed for %s | exception_type=%s",
          declaration.identity.registration_key,
          type(exc).__name__,
        )
        raise RegisteredApprovalPolicyError(
          "registered approval overlay failed"
        ) from exc

    approval_required = intrinsic_required or overlay_required
    if not approval_required:
      return False, None, False
    if (
      policy.cache_key is None
      or overlay_required
    ):
      return True, None, False

    try:
      prepared_plan = (
        trusted_plan.approval_identity()
        if trusted_plan is not None
        else None
      )
      cache_key = registry.execute_approval_cache_key(
        policy.cache_key,
        ApprovalCacheKeyCall(
          declaration.identity,
          prepared_call.prepared_input,
          prepared_call.exact_backend,
          prepared_plan,
        ),
      )
    except Exception as exc:
      log.error(
        "Registered approval cache key failed for %s | exception_type=%s",
        declaration.identity.registration_key,
        type(exc).__name__,
      )
      raise RegisteredApprovalPolicyError(
        "registered approval cache key failed"
      ) from exc

    cache_hit = cache_key in self._approved_tool_types
    return not cache_hit, cache_key, cache_hit

  def registered_addin_declaration(
    self,
    tool_name: str,
  ) -> ToolRegistrationDeclaration | None:
    """Return the exact registered add-in declaration when configured."""

    return resolve_registered_addin_declaration(
      self._tool_registration_catalog,
      tool_name,
    )

  def required_registered_addin_declaration(
    self,
    tool_name: str,
  ) -> ToolRegistrationDeclaration:
    declaration = resolve_registered_addin_declaration(
      self._tool_registration_catalog,
      tool_name,
    )
    if declaration is None:
      raise LookupError(f"registered add-in tool not found: {tool_name}")
    return declaration

  def redact_registered_addin_tool_input(
    self,
    declaration: ToolRegistrationDeclaration,
    prepared_call: PreparedToolCall,
  ) -> dict[str, object]:
    exact = self.required_registered_addin_declaration(
      declaration.identity.logical_name
    )
    if exact != declaration:
      raise RuntimeError(
        "registered add-in declaration is not authoritative"
      )
    return self._redact_registered_declaration_input(
      exact,
      prepared_call.materialize_input(),
    )

  def redact_registered_addin_raw_tool_input(
    self,
    tool_name: str,
    tool_input: Mapping[str, object],
  ) -> dict[str, object]:
    declaration = self.required_registered_addin_declaration(tool_name)
    return self._redact_registered_declaration_input(
      declaration,
      tool_input,
    )

  def prepare_registered_addin_tool_call(
    self,
    declaration: ToolRegistrationDeclaration,
    tool_input: Mapping[str, Any],
    trusted_context: object | None,
  ) -> PreparedToolCall:
    """Execute one exact add-in input policy in the gateway registry."""

    catalog = self._tool_registration_catalog
    registry = self._tool_policy_implementations
    if catalog is None or registry is None:
      raise RuntimeError(
        "registered add-in policy implementations are not configured"
      )
    return execute_registered_addin_input_preparation(
      catalog,
      registry,
      declaration,
      tool_input,
      trusted_context,
    )

  async def dispatch_prepared(
    self,
    tool_call_id: str,
    tool_name: str,
    prepared_call: PreparedToolCall,
    *,
    call_index: int = 0,
    advertised_tool_names: AbstractSet[str] | None = None,
    abort_event: asyncio.Event | None = None,
    skill_run_id: str | None = None,
    step_id: str | None = None,
    workspace_dir: str | None = None,
    batch_id: int | str | None = None,
    capture_readable_resource_snapshot: bool = False,
    allow_uncertain_mcp_replay: bool = True,
    on_executed_prepared_call: (
      Callable[[PreparedToolCall], None] | None
    ) = None,
  ) -> ToolResult:
    """Dispatch one exact prepared call without executing preparation again."""

    if type(prepared_call) is not PreparedToolCall:
      raise TypeError("prepared_call must be an exact PreparedToolCall")
    if type(allow_uncertain_mcp_replay) is not bool:
      raise TypeError("allow_uncertain_mcp_replay must be an exact bool")
    prepared_to_close: Any | None = None

    def own_prepared(prepared: Any) -> None:
      nonlocal prepared_to_close
      prepared_to_close = prepared

    try:
      return await self._dispatch(
        tool_call_id,
        tool_name,
        prepared_call.materialize_input(),
        call_index=call_index,
        advertised_tool_names=advertised_tool_names,
        abort_event=abort_event,
        skill_run_id=skill_run_id,
        step_id=step_id,
        workspace_dir=workspace_dir,
        batch_id=batch_id,
        capture_readable_resource_snapshot=capture_readable_resource_snapshot,
        allow_uncertain_mcp_replay=allow_uncertain_mcp_replay,
        on_executed_prepared_call=on_executed_prepared_call,
        registered_prepared_call=(
          prepared_call
          if (
            self._registered_code_declaration(tool_name) is not None
            or self._registered_approval_declaration(tool_name) is not None
            or (
              tool_name not in self._local
              and self._mcp.is_mcp_tool(tool_name)
              and callable(
                getattr(self._mcp, "uses_registered_tool_catalog", None)
              )
              and self._mcp.uses_registered_tool_catalog()
            )
          )
          else None
        ),
        own_prepared=own_prepared,
      )
    finally:
      close = getattr(prepared_to_close, "close", None)
      if callable(close):
        close()

  def route_origin_for_tool(self, tool_name: str) -> str | None:
    """Return the exact live dispatcher route kind, or ``None`` if ambiguous."""

    is_local = tool_name in self._local
    try:
      is_mcp = bool(self._mcp.is_mcp_tool(tool_name))
    except Exception:
      return None
    if is_local == is_mcp:
      return None
    return "local" if is_local else "mcp"

  async def _dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    call_index: int = 0,
    advertised_tool_names: AbstractSet[str] | None = None,
    abort_event: asyncio.Event | None = None,
    skill_run_id: str | None = None,
    step_id: str | None = None,
    workspace_dir: str | None = None,
    batch_id: int | str | None = None,
    capture_readable_resource_snapshot: bool = False,
    allow_uncertain_mcp_replay: bool = True,
    on_executed_prepared_call: (
      Callable[[PreparedToolCall], None] | None
    ) = None,
    registered_prepared_call: PreparedToolCall | None = None,
    own_prepared: Callable[[Any], None],
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
    self.ensure_gateway_local_tool_handler(tool_name)
    is_local_tool = tool_name in self._local
    is_mcp_tool = (
      not is_local_tool
      and self._mcp.is_mcp_tool(tool_name)
    )
    mcp_server_name = (
      self._mcp.get_server_for_tool(tool_name)
      if is_mcp_tool
      else None
    )
    advertisement_error = self._request_advertisement_error(
      tool_name,
      advertised_tool_names,
      is_mcp=is_mcp_tool,
      server_name=mcp_server_name,
    )
    if advertisement_error is not None:
      return None, advertisement_error
    lifecycle_tool_name = tool_name
    get_original_tool_name = getattr(self._mcp, "get_original_tool_name", None)
    if callable(get_original_tool_name):
      try:
        lifecycle_tool_name = str(
          get_original_tool_name(tool_name) or tool_name
        )
      except Exception:
        lifecycle_tool_name = tool_name
    if authority_policy_denies_tool(
      session=self._session,
      role=self._role,
      tool_name=tool_name,
      is_local=is_local_tool,
      profile_name=str(getattr(self._run_context, "profile", "") or "") or None,
    ):
      return None, {
        "code": "role_policy_denied",
        "message": f"Role '{self._role}' is not authorized to execute '{tool_name}'",
      }

    if abort_event is not None and abort_event.is_set():
      raise asyncio.CancelledError()

    if lifecycle_tool_name in (
      MODEL_RUN_IDENTITY_MCP_TOOLS | MODEL_RUN_IDENTITY_LOCAL_TOOLS
    ):
      try:
        carrier = RunIdentityCarrier.from_optional(skill_run_id)
        if carrier is None:
          raise RunIdentityCarrierError(
            "run_identity_required",
            f"Tool '{lifecycle_tool_name}' requires a server-owned run identity.",
          )
        run_context_id = validate_run_identity(
          self._resolve_run_context().run_id
        )
        if run_context_id != carrier.run_id:
          raise RunIdentityCarrierError(
            "run_identity_mismatch",
            f"Tool '{lifecycle_tool_name}' received conflicting run identities.",
          )
      except RunIdentityCarrierError as exc:
        return None, {"code": exc.code, "message": str(exc)}

    input_schema_error = self._validate_local_tool_input(tool_call_id, tool_name, tool_input)
    if input_schema_error is not None:
      return None, input_schema_error

    ir = await self._run_interceptors(
      tool_call_id,
      tool_name,
      tool_input,
    )
    if not ir.proceed:
      return None, ir.error
    if (
      registered_prepared_call is not None
      and tool_input != registered_prepared_call.materialize_input()
    ):
      return None, {
        "code": "tool_input_preparation_failed",
        "message": (
          f"Tool '{tool_name}' input changed after preparation; dispatch was denied."
        ),
      }

    if is_mcp_tool:
      scope_error = self._mcp_scope_error(tool_name, mcp_server_name)
      if scope_error is not None:
        return None, scope_error

    registered_code_declaration = self._registered_code_declaration(tool_name)
    registered_approval_declaration = (
      self._registered_approval_declaration(tool_name)
    )
    if (
      registered_approval_declaration is not None
      and registered_prepared_call is None
    ):
      return None, {
        "code": "tool_input_preparation_failed",
        "message": (
          f"Tool '{tool_name}' lacks its exact prepared call; dispatch was denied."
        ),
      }
    qualifier = (
      registered_prepared_call.exact_backend
      if registered_code_declaration is not None
      and registered_prepared_call is not None
      else ""
    )
    if qualifier is None:
      return None, {
        "code": "tool_input_preparation_failed",
        "message": "Registered code execution lost its exact backend.",
      }
    if (
      registered_code_declaration is None
      and self._approval_key_qualifier is not None
    ):
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
    registered_mcp_call: RegisteredMcpToolCall | None = None
    planned_executor: Callable[..., Any] | None = None
    prepared_business_model_record: PreparedBusinessModelChange | None = None
    prepared_business_model_context: (
      tuple[ToolDispatcherApprovalStore, dict[str, Any]] | None
    ) = None
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
        and isinstance(self._approval_route, DurableLocalApprovalRoute)
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
        durable = await self._approval_route.store.get_prepared_business_model_change(
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
        catalog_action = self._resolved_catalog_action(tool_name)
        planned_hooks = self._planned_handler_hooks(
          tool_name,
          local_handler,
          catalog_action=catalog_action,
        )
        if planned_hooks is not None:
          trusted_plan, planned_executor = await self._plan_local_write(
            planned_hooks,
            tool_input,
            call_index=call_index,
            tool_ctx=tool_ctx,
            own_prepared=own_prepared,
          )
      except PlannedWritePlanningRejected as exc:
        return exc.tool_result()
      except (ApprovalConstraintError, TrustedToolPlanError) as exc:
        log.error(
          "Invalid exact-plan contract for %s | exception_type=%s",
          tool_name,
          type(exc).__name__,
        )
        return None, {
          "code": "planned_write_contract_invalid",
          "message": f"Tool '{tool_name}' has an invalid exact-write planning contract.",
        }
      except Exception as exc:
        log.error(
          "Exact-write planning failed for %s | exception_type=%s",
          tool_name,
          type(exc).__name__,
        )
        return None, {
          "code": "planned_write_planning_failed",
          "message": f"Tool '{tool_name}' could not produce a trusted exact-write plan.",
        }
    elif (
      registered_approval_declaration is not None
      and registered_approval_declaration.identity.route_kind == "mcp"
      and registered_approval_declaration.semantics.planning_policy.policy_id
      != "none"
    ):
      try:
        assert registered_prepared_call is not None
        trusted_scope = registered_mcp_dispatch_scope(
          user_id=self._resolve_run_context().user_id,
          dispatch_scope=self._portfolio_dispatch_scope(),
        )
        registered_mcp_call = (
          self._mcp.classify_registered_mcp_prepared_tool_call(
            tool_name,
            registered_prepared_call,
            trusted_scope,
            self._registered_approval_overlay,
          )
        )
      except Exception as exc:
        log.error(
          "Registered MCP planning failed for %s | exception_type=%s",
          tool_name,
          type(exc).__name__,
        )
        return None, {
          "code": "planned_write_planning_failed",
          "message": (
            f"Tool '{tool_name}' could not produce its registered exact plan."
          ),
        }

    registered_approval_cache_key: str | None = None
    registered_approval_cache_hit = False
    if registered_approval_declaration is not None:
      assert registered_prepared_call is not None
      try:
        if registered_mcp_call is not None:
          static_needs_approval = registered_mcp_call.approval_required
          registered_approval_cache_key = (
            registered_mcp_call.approval_reuse_key
          )
          registered_approval_cache_hit = (
            registered_approval_cache_key is not None
            and registered_approval_cache_key in self._approved_tool_types
          )
          if registered_approval_cache_hit:
            static_needs_approval = False
        else:
          (
            static_needs_approval,
            registered_approval_cache_key,
            registered_approval_cache_hit,
          ) = self.registered_approval_requirement(
            registered_approval_declaration,
            registered_prepared_call,
            trusted_plan,
          )
      except RegisteredApprovalPolicyError:
        return None, {
          "code": "registered_approval_policy_failed",
          "message": (
            f"Tool '{tool_name}' approval policy could not be evaluated."
          ),
        }
    else:
      static_needs_approval = self._should_request_approval(
        tool_name,
        tool_input,
        qualifier,
      )
    dynamic_ask = ir.pending_ask is not None
    approval_reuse_mode: ApprovalReuseMode
    approval_reuse_key: str | None
    if registered_approval_declaration is None:
      approval_reuse_mode = "legacy"
      approval_reuse_key = None
    elif dynamic_ask or registered_approval_cache_key is None:
      approval_reuse_mode = "disabled"
      approval_reuse_key = None
    else:
      approval_reuse_mode = "exact"
      approval_reuse_key = registered_approval_cache_key
    session_approval_cache_key = (
      approval_reuse_key
      if approval_reuse_mode == "exact"
      else (
        (
          None
          if tool_name in self._session_cache_denied
          else self._qualified_key(tool_name, qualifier)
        )
        if approval_reuse_mode == "legacy"
        else None
      )
    )
    final_tool_input = tool_input
    approval_modified_prepared_input = False
    approval_request_record: PolicyApprovalRequest | None = None
    registered_prepared_authorization = (
      registered_mcp_call.prepared_authorization
      if registered_mcp_call is not None
      else None
    )
    exact_plan_active = (
      trusted_plan is not None
      or registered_prepared_authorization is not None
    )

    if (
      not exact_plan_active
      and not static_needs_approval
      and not dynamic_ask
      and (
        registered_approval_cache_hit
        if registered_approval_declaration is not None
        else self._tool_was_cache_hit(tool_name, qualifier)
      )
    ):
      self._emit_approval_decided(
        tool_call_id,
        tool_name,
        outcome="approved",
        decision_source="session_cache_approved",
        allow_tool_type_applied=False,
      )

    if exact_plan_active:
      if isinstance(self._approval_route, NoApprovalRoute):
        return None, {
          "code": "approval_route_absent",
          "message": (
            f"Tool '{tool_name}' has no admitted approval route for its "
            "exact-write plan."
          ),
        }
      if isinstance(
        self._approval_route,
        ParentDelegatedApprovalRoute,
      ) and self._plan_requires_prepared_custody(
        tool_name,
        trusted_plan,
        registered_prepared_authorization,
      ):
        return None, {
          "code": "planned_write_prepared_custody_unsupported",
          "message": (
            f"Tool '{tool_name}' needs durable prepared-payload custody, "
            "which the parent-delegated approval route does not carry."
          ),
        }

      allow_persistent = not dynamic_ask
      approval_reason = ir.pending_ask.message if ir.pending_ask is not None else ""
      cache_approved = (
        not static_needs_approval
        and not dynamic_ask
        and (
          registered_approval_cache_hit
          if registered_approval_declaration is not None
          else self._tool_was_cache_hit(tool_name, qualifier)
        )
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
              log.warning(
                "Headless ask hook failed; auto-denying | exception_type=%s",
                type(exc).__name__,
              )
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
          automatic_approval_reason = (
            "Autonomous tool policy authorized the exact planned identity"
          )

      try:
        prepared_authorization_payload: bytes | None = None
        prepared_business_model_change = (
          self._prepared_business_model_authorization(
            tool_name,
            trusted_plan,
          )
          if trusted_plan is not None
          else None
        )
        resume_approval_request = None
        skip_approval_lifecycle = False
        if prepared_business_model_change is not None:
          prepared_business_model_store = self._durable_business_model_store()
          prepared_business_model_context = (
            prepared_business_model_store,
            prepared_business_model_change,
          )
          from .approval_store import PreparedReconciliationConflict
          from .prepared_business_model_store import PreparedBusinessModelLifecycle

          async def reconcile_pending_business_model_record(
            record: PreparedBusinessModelChange,
          ) -> PreparedBusinessModelChange:
            reconciliation = await prepared_business_model_store.reconcile_prepared_business_model_change(
              caller_kind=record.caller_kind,
              user_scope=record.user_scope,
              idempotency_locator=record.idempotency_locator,
            )
            if reconciliation.conflict is PreparedReconciliationConflict.MISSING_APPROVAL:
              raise TrustedToolPlanError(
                "durable FMS BusinessModel plan lost its approval request"
              )
            if reconciliation.conflict is PreparedReconciliationConflict.LINEAGE_CONFLICT:
              raise TrustedToolPlanError(
                "durable FMS BusinessModel plan conflicts with its approval lineage"
              )
            if reconciliation.conflict is PreparedReconciliationConflict.UNKNOWN_APPROVAL_STATE:
              raise TrustedToolPlanError(
                "durable FMS BusinessModel plan has an unknown approval state"
              )
            if reconciliation.conflict is PreparedReconciliationConflict.CAS_CONFLICT:
              raise TrustedToolPlanError(
                "durable FMS BusinessModel plan reconciliation compare-and-swap lost"
              )
            if reconciliation.record is None:
              raise TrustedToolPlanError(
                "durable FMS BusinessModel plan disappeared during reconciliation"
              )
            return reconciliation.record

          prepared_business_model_record = await prepared_business_model_store.get_prepared_business_model_change(
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
            if (
              prepared_business_model_record.lifecycle
              is PreparedBusinessModelLifecycle.PENDING
            ):
              prepared_business_model_record = (
                await reconcile_pending_business_model_record(
                  prepared_business_model_record
                )
              )
              if prepared_business_model_record.lifecycle in {
                PreparedBusinessModelLifecycle.DENIED,
                PreparedBusinessModelLifecycle.EXPIRED,
              }:
                return None, {
                  "code": "planned_write_authorization_state_invalid",
                  "message": (
                    "The original BusinessModel approval was denied."
                    if prepared_business_model_record.lifecycle
                    is PreparedBusinessModelLifecycle.DENIED
                    else "The durable BusinessModel plan expired."
                  ),
                }
            resume_approval_request = await prepared_business_model_store.get(
              prepared_business_model_record.approval_id
            )
            if resume_approval_request is None:
              raise TrustedToolPlanError(
                "durable FMS BusinessModel plan lost its approval request"
              )
            if (
              prepared_business_model_record.lifecycle
              is PreparedBusinessModelLifecycle.PENDING
              and str(resume_approval_request.state)
              in {
                "approved",
                "auto_approved",
                "denied",
                "auto_denied",
                "cancelled",
                "expired",
              }
            ):
              prepared_business_model_record = (
                await reconcile_pending_business_model_record(
                  prepared_business_model_record
                )
              )
              if prepared_business_model_record.lifecycle in {
                PreparedBusinessModelLifecycle.DENIED,
                PreparedBusinessModelLifecycle.EXPIRED,
              }:
                return None, {
                  "code": "planned_write_authorization_state_invalid",
                  "message": (
                    "The original BusinessModel approval was denied."
                    if prepared_business_model_record.lifecycle
                    is PreparedBusinessModelLifecycle.DENIED
                    else "The durable BusinessModel plan expired."
                  ),
                }
              resume_approval_request = await prepared_business_model_store.get(
                prepared_business_model_record.approval_id
              )
              if resume_approval_request is None:
                raise TrustedToolPlanError(
                  "durable FMS BusinessModel plan lost its approval request"
                )
            if prepared_business_model_record.lifecycle in {
              PreparedBusinessModelLifecycle.AUTHORIZED,
              PreparedBusinessModelLifecycle.CONSUMED,
            }:
              if not approval_is_executable(resume_approval_request):
                raise TrustedToolPlanError(
                  "durable FMS BusinessModel plan lost approved lineage"
                )
              skip_approval_lifecycle = True
        approval_args_redacted: dict[str, Any] | None = None
        approval_args_hash: str | None = None
        if registered_prepared_authorization is not None:
          prepared_authorization_payload = (
            registered_prepared_authorization.prepared_payload
          )
          approval_args_redacted = (
            registered_prepared_authorization.materialize_approval_arguments()
          )
          approval_args_hash = (
            registered_prepared_authorization.approval_arguments_hash
          )
        elif (
          trusted_plan is not None
          and trusted_plan.identity_source == "change_set"
        ):
          if registered_approval_declaration is not None:
            assert registered_prepared_call is not None
            approval_args_redacted, approval_args_hash = (
              self._redact_registered_prepared_for_approval(
                tool_name,
                registered_prepared_call,
              )
            )
          else:
            approval_args_redacted, approval_args_hash = (
              self._redact_for_approval_request(tool_name, tool_input)
            )
          approval_args_redacted = {
            **approval_args_redacted,
            "planned_change": trusted_plan.approval_review(),
          }
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
            approval_reuse_mode=approval_reuse_mode,
            approval_reuse_key=approval_reuse_key,
            approval_identity=(
              registered_prepared_authorization.approval_identity
              if registered_prepared_authorization is not None
              else trusted_plan.approval_identity()
              if trusted_plan is not None
              else None
            ),
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
      except Exception as exc:
        log.error(
          "Durable exact-write authorization failed for %s | exception_type=%s",
          tool_name,
          type(exc).__name__,
        )
        return None, {
          "code": "planned_write_authorization_persistence_failed",
          "message": (
            f"Tool '{tool_name}' could not persist its exact-write authorization."
          ),
        }

      approval_request_record = lifecycle.get("request")
      if prepared_business_model_context is not None:
        prepared_business_model_store, prepared_business_model_change = (
          prepared_business_model_context
        )
        from .prepared_business_model_store import PreparedBusinessModelLifecycle

        prepared_business_model_record = (
          await prepared_business_model_store.get_prepared_business_model_change(
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
            prepared_business_model_record = await prepared_business_model_store.transition_prepared_business_model_change(
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
            original_request = await prepared_business_model_store.get(
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
          prepared_business_model_record = await prepared_business_model_store.transition_prepared_business_model_change(
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
        if trusted_plan is not None:
          trusted_plan.verify_approval_request(approval_request_record)
        else:
          assert registered_prepared_authorization is not None
          expected_identity = dict(
            registered_prepared_authorization.approval_identity
          )
          actual_identity = {
            field_name: getattr(
              approval_request_record,
              field_name,
              None,
            )
            for field_name in expected_identity
          }
          if actual_identity != expected_identity:
            raise TrustedToolPlanError(
              "approval row is not bound to the registered MCP plan"
            )
      except TrustedToolPlanError as exc:
        log.error(
          "Unbound exact-write approval for %s | exception_type=%s",
          tool_name,
          type(exc).__name__,
        )
        return None, {
          "code": "planned_write_authorization_identity_invalid",
          "message": f"Tool '{tool_name}' approval did not bind the exact planned identity.",
        }

      if tool_ctx is not None:
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
        decision_source = "user_denied"
        error_dict = {"code": "user_denied", "message": "User denied execution"}
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
      approval_modified_prepared_input = final_tool_input != tool_input
      if (
        registered_prepared_call is not None
        and approval_modified_prepared_input
      ):
        return None, {
          "code": "tool_input_preparation_failed",
          "message": (
            f"Tool '{tool_name}' input changed after preparation; dispatch was denied."
          ),
        }
      if isinstance(registered_mcp_call, RegisteredMcpPlannedToolCall):
        final_tool_input = registered_mcp_call.materialize_authorized_input(
          tool_call_id
        )
      will_install = (
        bool(lifecycle.get("allow_tool_type"))
        and allow_persistent
        and session_approval_cache_key is not None
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
        assert session_approval_cache_key is not None
        self._approved_tool_types.add(session_approval_cache_key)

    elif static_needs_approval or dynamic_ask:
      if self._should_avoid_permission_prompts:
        if static_needs_approval:
          reason_text = (
            ir.pending_ask.message
            if ir.pending_ask is not None
            else f"Tool '{tool_name}' requires static approval in headless context"
          )
          if self._event_log is not None:
            self._append_event(
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
            log.warning(
              "Headless ask hook failed; auto-denying | exception_type=%s",
              type(exc).__name__,
            )
            hook_result = "deny"

        if hook_result != "allow":
          reason_text = (
            ir.pending_ask.message
            if ir.pending_ask is not None
            else "Approval required in headless context"
          )
          if self._event_log is not None:
            self._append_event(
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
        if not isinstance(self._approval_route, NoApprovalRoute):
          allow_persistent = not dynamic_ask
          approval_reason = ir.pending_ask.message if ir.pending_ask is not None else ""
          approval_args_redacted: dict[str, Any] | None = None
          approval_args_hash: str | None = None
          if registered_approval_declaration is not None:
            assert registered_prepared_call is not None
            approval_args_redacted, approval_args_hash = (
              self._redact_registered_prepared_for_approval(
                tool_name,
                registered_prepared_call,
              )
            )
          lifecycle = await self._run_approval_lifecycle(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            qualifier=qualifier,
            reason=approval_reason,
            allow_persistent=allow_persistent,
            approval_reuse_mode=approval_reuse_mode,
            approval_reuse_key=approval_reuse_key,
            approval_args_redacted=approval_args_redacted,
            approval_args_hash=approval_args_hash,
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
            decision_source = "user_denied"
            error_dict = {"code": "user_denied", "message": "User denied execution"}
            self._emit_approval_decided(
              tool_call_id,
              tool_name,
              outcome="denied",
              decision_source=decision_source,
              allow_tool_type_applied=False,
            )
            return None, error_dict
          final_tool_input = lifecycle["tool_input"] if "tool_input" in lifecycle else tool_input
          approval_modified_prepared_input = final_tool_input != tool_input
          if (
            registered_prepared_call is not None
            and approval_modified_prepared_input
          ):
            return None, {
              "code": "tool_input_preparation_failed",
              "message": (
                f"Tool '{tool_name}' input changed after preparation; dispatch was denied."
              ),
            }
          will_install = (
            bool(lifecycle.get("allow_tool_type"))
            and allow_persistent
            and session_approval_cache_key is not None
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
            assert session_approval_cache_key is not None
            self._approved_tool_types.add(session_approval_cache_key)
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
            and session_approval_cache_key is not None
          )
          self._emit_approval_decided(
            tool_call_id,
            tool_name,
            outcome="approved" if decision.approved else "denied",
            decision_source="user_approved" if decision.approved else "user_denied",
            allow_tool_type_applied=will_install,
          )
          if not decision.approved:
            return None, {"code": "user_denied", "message": "User denied execution"}
          if will_install:
            assert session_approval_cache_key is not None
            self._approved_tool_types.add(session_approval_cache_key)

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
    if (
      registered_prepared_call is not None
      and approval_modified_prepared_input
    ):
      return None, {
        "code": "tool_input_preparation_failed",
        "message": (
          f"Tool '{tool_name}' input changed after preparation; dispatch was denied."
        ),
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
        executor_parameters = inspect.signature(
          planned_executor
        ).parameters
        assert approval_request_record is not None
        assert tool_ctx.approval_id
        assert tool_ctx.approval_chain_id
        try:
          trusted_plan.verify_approval_request(
            approval_request_record
          )
        except TrustedToolPlanError as exc:
          log.error(
            "Exact-write authorization drift for %s | exception_type=%s",
            tool_name,
            type(exc).__name__,
          )
          return None, {
            "code": "planned_write_trusted_plan_lost",
            "message": f"Tool '{tool_name}' lost its trusted exact-write authorization.",
          }
        if not approval_is_executable(approval_request_record):
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
      if approval_modified_prepared_input:
        return None, {
          "code": "tool_input_preparation_failed",
          "message": (
            f"Tool '{tool_name}' input changed after preparation; dispatch was denied."
          ),
        }
      server = self._mcp.get_server_for_tool(tool_name)
      if (
        self._commercial_work_start is not None
        and server not in self._commercial_mcp_servers
      ):
        return None, {
          "code": "commercial_mcp_destination_denied",
          "message": "Commercial work cannot be dispatched to this MCP destination.",
        }
      per_user_server = bool(
        server
        and callable(getattr(self._mcp, "is_per_user_server", None))
        and self._mcp.is_per_user_server(server)
      )
      if server and server in self._mcp_meta_inject_servers:
        resolved_risk_user_id = (
          self._risk_user_id
          if (
            isinstance(self._risk_user_id, int)
            and not isinstance(self._risk_user_id, bool)
            and self._risk_user_id > 0
          )
          else None
        )
        if (
          resolved_risk_user_id is None
          and self._user_id is not None
          and str(self._user_id).isdigit()
        ):
          numeric_user_id = int(str(self._user_id))
          resolved_risk_user_id = numeric_user_id if numeric_user_id > 0 else None
        if self._credentials_resolver_active and resolved_risk_user_id is None:
          raise RuntimeError("MCP meta user_id is required in strict mode")
        meta = {
          "session_id": self._session_id,
          "user_id": str(resolved_risk_user_id) if resolved_risk_user_id is not None else None,
          "channel": self._channel,
          "role": self._role,
        }
        caller_session_token = getattr(self._session, "session_token", None)
        if caller_session_token is not None:
          meta["session_token"] = caller_session_token
        routed_skill_run_id = mcp_metadata_skill_run_id(
          lifecycle_tool_name,
          skill_run_id,
        )
        if routed_skill_run_id is not None:
          meta["skill_run_id"] = routed_skill_run_id
        if workspace_dir is not None:
          meta["workspace_dir"] = workspace_dir
        if batch_id is not None:
          meta["batch_id"] = str(batch_id)
        if (
          server == INVESTMENT_CAPABILITY_CLAIM_SERVER
          and lifecycle_tool_name in INVESTMENT_CAPABILITY_FACADE_TOOLS
        ):
          run_context = self._resolve_run_context()
          if (
            lifecycle_tool_name == "start_investment_run"
            and "external_refs" in final_tool_input
          ):
            return None, {
              "code": "investment_external_refs_not_allowed",
              "message": (
                "Investment run provenance references cannot be supplied by "
                "the model."
              ),
            }
          try:
            reconciled_admission = reconcile_skill_admission(
              skill_name=run_context.skill,
              execution_limits=(
                run_context.admitted_skill_execution_limits
              ),
              active_admission=current_skill_admission(),
            )
          except (TypeError, ValueError):
            return None, investment_capability_claim_unavailable_error(
              subject=f"tool '{lifecycle_tool_name}'",
            )
          if reconciled_admission is None:
            return None, investment_capability_claim_unavailable_error(
              subject=f"tool '{lifecycle_tool_name}'",
            )
          trusted_skill = reconciled_admission.skill_name
          trusted_research_file_id: int | None = None
          if lifecycle_tool_name == "start_quant_research":
            trusted_research_file_id = run_context.research_file_id
            request_payload = final_tool_input.get("request")
            request_research_file_id = (
              request_payload.get("research_file_id")
              if isinstance(request_payload, dict)
              else None
            )
            if (
              isinstance(trusted_research_file_id, bool)
              or not isinstance(trusted_research_file_id, int)
              or not 1 <= trusted_research_file_id < (1 << 63)
              or isinstance(request_research_file_id, bool)
              or not isinstance(request_research_file_id, int)
              or request_research_file_id != trusted_research_file_id
            ):
              return None, investment_capability_claim_unavailable_error(
                subject=f"tool '{lifecycle_tool_name}'",
              )
          try:
            meta["investment_capability_claim"] = (
              issue_investment_capability_claim(
                user_id=str(resolved_risk_user_id or ""),
                session_id=str(self._session_id or run_context.session_id or ""),
                skill_run_id=str(skill_run_id or ""),
                channel=str(self._channel or run_context.channel or ""),
                tool_name=lifecycle_tool_name,
                request_id=str(run_context.request_id or ""),
                jti=str(tool_call_id or ""),
                skill=trusted_skill,
                admitted_skill_execution_limits=(
                  reconciled_admission.execution_limits
                ),
                policy_bundle_hash=str(run_context.policy_bundle_hash or ""),
                research_file_id=trusted_research_file_id,
              )
            )
          except InvestmentCapabilityClaimError:
            return None, investment_capability_claim_unavailable_error(
              subject=f"tool '{lifecycle_tool_name}'",
            )
        result, error = await ToolDispatcher._call_mcp_tool(
          self,
          tool_name,
          final_tool_input,
          prepared_call=registered_prepared_call,
          meta=meta,
          abort_event=abort_event,
          gateway_session=self._session if per_user_server else None,
          allow_uncertain_replay=allow_uncertain_mcp_replay,
          on_executed_prepared_call=on_executed_prepared_call,
        )
      elif server and per_user_server:
        result, error = await ToolDispatcher._call_mcp_tool(
          self,
          tool_name,
          final_tool_input,
          prepared_call=registered_prepared_call,
          abort_event=abort_event,
          gateway_session=self._session,
          allow_uncertain_replay=allow_uncertain_mcp_replay,
          on_executed_prepared_call=on_executed_prepared_call,
        )
      elif server and server in self._mcp_session_inject_servers:
        final_tool_input = {**final_tool_input, "_session_id": self._session_id}
        result, error = await ToolDispatcher._call_mcp_tool(
          self,
          tool_name,
          final_tool_input,
          prepared_call=registered_prepared_call,
          abort_event=abort_event,
          allow_uncertain_replay=allow_uncertain_mcp_replay,
          on_executed_prepared_call=on_executed_prepared_call,
        )
      else:
        result, error = await ToolDispatcher._call_mcp_tool(
          self,
          tool_name,
          final_tool_input,
          prepared_call=registered_prepared_call,
          abort_event=abort_event,
          allow_uncertain_replay=allow_uncertain_mcp_replay,
          on_executed_prepared_call=on_executed_prepared_call,
        )
    else:
      result, error = None, {"code": "unknown_tool", "message": f"Unknown tool: {tool_name}"}

    if ir.warnings and error is None and result is not None and isinstance(result, dict):
      result = dict(result)
      result["_interceptor_warnings"] = ir.warnings

    if (
      prepared_business_model_context is not None
      and prepared_business_model_record is not None
      and prepared_business_model_record.lifecycle.value == "AUTHORIZED"
      and isinstance(result, dict)
      and isinstance(result.get("receipt"), dict)
    ):
      prepared_business_model_store, _ = prepared_business_model_context
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
          prepared_business_model_record = await prepared_business_model_store.transition_prepared_business_model_change(
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
          prepared_business_model_record = await prepared_business_model_store.transition_prepared_business_model_change(
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

    await _audit_helpers.emit_execution_audit(
      approval_request_record,
      final_tool_input,
      approval_store=self._approval_store,
      outcome="tool_error" if error is not None else "success",
      error_summary=str(error)[:500] if error is not None else None,
      boundary_sanitizer=lambda value, sink: sanitize_boundary_value(
        value,
        sink=sink,
        boundary=self._secret_boundary,
      ),
    )
    return result, error

  def requires_approval(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
    """Return True when dispatch may enter an approval-owned wait.

    Exact-write handlers enter the durable approval lifecycle after planning
    even when their static tool policy is already allowed. The runner must not
    wrap that lifecycle in its generic tool timeout; the approval lifecycle has
    its own expiry and returns an approval-specific result.
    """
    route_absent = isinstance(self._approval_route, NoApprovalRoute)
    if self._request_approval is None and route_absent:
      return False
    try:
      registered_declaration = self._registered_approval_declaration(tool_name)
    except Exception:
      return True
    if registered_declaration is not None:
      registered_static_needs_approval = (
        registered_declaration.semantics.approval.mode != "never"
        or self._registered_approval_overlay is not None
      )
      return self._requires_approval_with_qualifier(
        tool_name,
        tool_input,
        "",
        registered_static_needs_approval=registered_static_needs_approval,
      )
    qualifier = ""
    if (
      self._registered_code_declaration(tool_name) is None
      and self._approval_key_qualifier is not None
    ):
      try:
        qualifier = self._approval_key_qualifier(tool_name, tool_input) or ""
      except Exception:
        qualifier = ""
    return self._requires_approval_with_qualifier(
      tool_name,
      tool_input,
      qualifier,
    )

  def requires_approval_prepared(
    self,
    tool_name: str,
    prepared_call: PreparedToolCall,
  ) -> bool:
    """Classify one already-prepared call without selecting its route again."""

    if type(prepared_call) is not PreparedToolCall:
      raise TypeError("prepared_call must be an exact PreparedToolCall")
    tool_input = prepared_call.materialize_input()
    try:
      registered_declaration = self._registered_approval_declaration(tool_name)
      if registered_declaration is not None:
        if (
          registered_declaration.semantics.planning_policy.policy_id != "none"
        ):
          # Exact-write planning happens inside dispatch.  Timeout
          # classification runs before that boundary and therefore cannot
          # derive a plan-bound approval reuse key yet.  Such calls may enter
          # the approval lifecycle, so classify them conservatively without
          # executing the cache-key policy against an absent plan.
          return True
        registered_static_needs_approval = (
          self.registered_approval_requirement(
            registered_declaration,
            prepared_call,
          )[0]
        )
        return self._requires_approval_with_qualifier(
          tool_name,
          tool_input,
          prepared_call.exact_backend or "",
          registered_static_needs_approval=registered_static_needs_approval,
        )
    except Exception as exc:
      log.error(
        "Registered approval classification failed for %s | exception_type=%s",
        tool_name,
        type(exc).__name__,
      )
      return True
    if self._registered_code_declaration(tool_name) is not None:
      qualifier = prepared_call.exact_backend
      if qualifier is None:
        raise RuntimeError("registered code call lacks its exact backend")
    else:
      qualifier = ""
      if self._approval_key_qualifier is not None:
        try:
          qualifier = self._approval_key_qualifier(tool_name, tool_input) or ""
        except Exception:
          qualifier = ""
    return self._requires_approval_with_qualifier(
      tool_name,
      tool_input,
      qualifier,
    )

  def _requires_approval_with_qualifier(
    self,
    tool_name: str,
    tool_input: Dict[str, Any],
    qualifier: str,
    *,
    registered_static_needs_approval: bool | None = None,
  ) -> bool:
    if registered_static_needs_approval is True:
      return True
    if (
      registered_static_needs_approval is None
      and self._should_request_approval(tool_name, tool_input, qualifier)
    ):
      return True
    if isinstance(self._approval_route, NoApprovalRoute):
      return False

    try:
      action = self._resolved_catalog_action(tool_name)
    except TrustedToolPlanError:
      # Dispatch will return the precise invalid-contract error. Keep the
      # generic timeout from masking that approval-bound planning path.
      return True
    if action is not None and action.planning_identity is not None:
      return True

    local_handler = self._local.get(tool_name)
    if local_handler is None:
      return False
    try:
      return self._planned_handler_hooks(
        tool_name,
        local_handler,
        catalog_action=action,
      ) is not None
    except TrustedToolPlanError:
      return True

  def _plan_requires_prepared_custody(
    self,
    tool_name: str,
    trusted_plan: TrustedToolPlan | None,
    registered_prepared_authorization: Any | None,
  ) -> bool:
    """Report whether this door needs a prepared payload to survive attempts.

    These plans resume across attempts from a durable prepared row only the
    process owning the ledger can write, so a route without that custody must
    refuse rather than mint an approval with nothing behind it.
    """

    if tool_name == "fms_persist_business_model":
      return True
    if registered_prepared_authorization is not None:
      return True
    return (
      trusted_plan is not None
      and trusted_plan.identity_source == "reviewed_change_binding"
    )

  async def _run_approval_lifecycle(
    self,
    *,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
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
    session_cache_approved: bool = False,
    automatic_approval_reason: str | None = None,
    automatic_denial_reason: str | None = None,
    deny_user_prompt: bool = False,
  ) -> dict[str, Any]:
    return await _approval_lifecycle_helpers.run_approval_lifecycle(
      route=self._approval_route,
      session=self._session,
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_input=tool_input,
      qualifier=qualifier,
      reason=reason,
      allow_persistent=allow_persistent,
      approval_constraint=approval_constraint,
      required_owner_user_id=required_owner_user_id,
      approval_reuse_mode=approval_reuse_mode,
      approval_reuse_key=approval_reuse_key,
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
      current_skill_admission_fn=current_skill_admission,
      redact_for_approval_request_fn=self._redact_for_approval_request,
      resolve_tool_class_fn=self._resolve_tool_class,
      effective_trade_approval_decision_fn=self._effective_trade_approval_decision,
      await_user_approval_via_pending_tools_fn=self._await_user_approval_via_pending_tools,
      approval_queue_timeout_seconds_fn=_approval_queue_timeout_seconds,
      secret_boundary=getattr(self, "_secret_boundary", None),
    )

  async def run_registered_addin_approval_lifecycle(
    self,
    *,
    declaration: ToolRegistrationDeclaration,
    tool_call_id: str,
    prepared_call: PreparedToolCall,
    reason: str,
    allow_persistent: bool,
    approval_reuse_mode: ApprovalReuseMode,
    approval_reuse_key: str | None,
    session_cache_approved: bool = False,
  ) -> dict[str, Any]:
    """Run approval with the exact add-in declaration's registered effect."""

    tool_input = prepared_call.materialize_input()
    approval_args_redacted, approval_args_hash = (
      self._redact_registered_declaration_for_approval(
        declaration,
        prepared_call,
      )
    )

    return await _approval_lifecycle_helpers.run_approval_lifecycle(
      route=self._approval_route,
      session=self._session,
      tool_call_id=tool_call_id,
      tool_name=declaration.identity.logical_name,
      tool_input=tool_input,
      qualifier="",
      reason=reason,
      allow_persistent=allow_persistent,
      approval_reuse_mode=approval_reuse_mode,
      approval_reuse_key=approval_reuse_key,
      approval_args_redacted=approval_args_redacted,
      approval_args_hash=approval_args_hash,
      session_cache_approved=session_cache_approved,
      resolve_run_context_fn=self._resolve_run_context,
      current_skill_admission_fn=current_skill_admission,
      redact_for_approval_request_fn=self._redact_for_approval_request,
      resolve_tool_class_fn=(
        lambda _tool_name: cast(ToolClass, declaration.semantics.effect)
      ),
      effective_trade_approval_decision_fn=self._effective_trade_approval_decision,
      await_user_approval_via_pending_tools_fn=self._await_user_approval_via_pending_tools,
      approval_queue_timeout_seconds_fn=_approval_queue_timeout_seconds,
      secret_boundary=getattr(self, "_secret_boundary", None),
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
    durable_request: dict[str, Any] | None = None,
  ) -> dict[str, Any] | None:
    return await _approval_lifecycle_helpers.await_user_approval_via_pending_tools(
      session=self._session,
      approval_store=self._approval_store,
      append_event_fn=self._boundary_event_log.append,
      request=request,
      decision=decision,
      nonce=nonce,
      resolved_qualifier=resolved_qualifier,
      allow_persistent=allow_persistent,
      timeout_seconds=timeout_seconds,
      log=log,
      batch_admission=batch_admission,
      durable_request=durable_request,
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
    if self._local_tool_classes is not None and tool_name in self._local:
      return self._local_tool_classes[tool_name]
    return _approval_lifecycle_helpers.resolve_tool_class(
      tool_name, mcp=self._mcp, resolve_server_policy_tool_class_fn=resolve_server_policy_tool_class
    )

  def _redact_for_approval_request(self, tool_name: str, tool_input: Dict[str, Any]) -> tuple[dict[str, Any], str]:
    return _approval_lifecycle_helpers.redact_for_approval_request(
      tool_name,
      tool_input,
      event_log=self._event_log,
      enrich_trade_approval_args_fn=enrich_trade_approval_args,
    )

  def _redact_registered_prepared_for_approval(
    self,
    tool_name: str,
    prepared_call: PreparedToolCall,
  ) -> tuple[dict[str, Any], str]:
    return self._registered_approval_projection(
      tool_name,
      prepared_call,
      self.redact_prepared_tool_input(tool_name, prepared_call),
    )

  def _redact_registered_declaration_for_approval(
    self,
    declaration: ToolRegistrationDeclaration,
    prepared_call: PreparedToolCall,
  ) -> tuple[dict[str, Any], str]:
    return self._registered_approval_projection(
      declaration.identity.logical_name,
      prepared_call,
      self._redact_registered_declaration_input(
        declaration,
        prepared_call.materialize_input(),
      ),
    )

  def _registered_approval_projection(
    self,
    tool_name: str,
    prepared_call: PreparedToolCall,
    redacted: Mapping[str, object],
  ) -> tuple[dict[str, Any], str]:
    enriched = enrich_trade_approval_args(
      tool_name,
      dict(redacted),
      event_log=self._event_log,
    )
    return (
      enriched,
      _approval_lifecycle_helpers.hash_approval_arguments(
        prepared_call.materialize_input()
      ),
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
  def _is_portfolio_mcp_server(server: str | None) -> TypeGuard[str]:
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

  async def _call_mcp_tool(
    self,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    prepared_call: PreparedToolCall | None = None,
    meta: Dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: Any | None = None,
    allow_uncertain_replay: bool = True,
    on_executed_prepared_call: (
      Callable[[PreparedToolCall], None] | None
    ) = None,
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
    if abort_event is not None:
      kwargs["abort_event"] = abort_event
    if gateway_session is not None:
      kwargs["gateway_session"] = gateway_session
    kwargs["allow_uncertain_replay"] = allow_uncertain_replay
    transport_input: Dict[str, Any] | PreparedToolCall = tool_input
    if prepared_call is not None:
      if type(prepared_call) is not PreparedToolCall:
        raise TypeError("prepared_call must be an exact PreparedToolCall")
      transport_input = PreparedToolCall(
        tool_input,
        prepared_call.exact_backend,
      )
    if on_executed_prepared_call is not None:
      executed_prepared_call = (
        transport_input
        if isinstance(transport_input, PreparedToolCall)
        else PreparedToolCall(transport_input)
      )
      on_executed_prepared_call(executed_prepared_call)
    result, error = await self._mcp.call_tool(
      tool_name,
      transport_input,
      **kwargs,
    )
    if isinstance(error, dict) and "tool_usage_hint" not in error and _is_mcp_validation_error(error):
      hint = self._mcp_tool_argument_guidance(tool_name)
      if hint:
        error = dict(error)
        error["tool_usage_hint"] = hint
    if result is not None and error is None:
      policy = load_server_policy_module()
      capture = getattr(policy, "capture_tool_source_pack", None)
      if capture is not None:
        capture(tool_name, self._source_pack_session, result, tool_input, log)
    return result, error


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
      self._boundary_event_log,
      tool_call_id,
      tool_name,
      outcome=outcome,
      decision_source=decision_source,
      allow_tool_type_applied=allow_tool_type_applied,
      time_fn=time.time,
    )
