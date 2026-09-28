"""Gateway-side MCP client for stdio and streamable HTTP.

Owns per-user sessions and catalog publication; mcp_client_config owns reconnect
classification. Reconnecting must never authorize unsafe tool-call replay.
Every connection's transport contexts are entered and exited by one host task
that mcp_client_connections owns, never by a caller's: exiting them from
anywhere else cancels the task that entered them.
See packages/agent-gateway/docs/architecture.md.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import hmac
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import AbstractSet, Any, Callable, cast, Coroutine, Dict, List, Mapping, Sequence, Set, Tuple, TypedDict

import httpx
import httpx2
from fastmcp.client.auth.oauth import OAuth as FastMCPOAuth
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from typing_extensions import TypeIs

from agent_workflow_contracts.tool_registration import (
  McpInputPreparationRoute,
  ToolRegistrationCatalog,
  ToolRegistrationDeclaration,
  index_mcp_input_preparation_routes,
  validate_tool_registration_catalog,
)

from . import mcp_client_catalog as _catalog_helpers
from . import mcp_client_connections as _connection_helpers
from . import mcp_client_config as _config_helpers
from . import mcp_client_errors as _error_helpers
from . import mcp_client_oauth_storage as _oauth_storage
from . import mcp_client_policy_owner as _policy_owner_helpers
from . import mcp_client_runtime as _runtime_helpers
from . import mcp_client_startup as _startup_helpers
from .approval_policy import sha256_args
from .policy_imports import load_server_policy_helpers, update_mcp_tool_metadata
from .tool_definition import LiveToolRouteBinding, OriginatedToolDefinition
from .tool_dispatch_classification import (
  OUTCOME_OK,
  ToolResultSettlement,
)
from .tool_registration import (
  RegisteredMcpToolDescriptor,
  UnknownRegisteredMcpToolDescriptorError,
  compile_registered_mcp_tool_descriptors,
)
from .tool_policy_registry import (
  ApprovalCacheKeyCall,
  ApprovalPredicateCall,
  InputPreparationCall,
  OutcomeCall,
  PlanDecision,
  PlanningCall,
  PreparedToolCall,
  RedactionCall,
  SourceIdentityCall,
  ToolPolicyImplementationRegistry,
  ToolPolicyResultError,
)

log = logging.getLogger("agent_gateway.mcp_client")
_UNSET: _config_helpers.McpConfigPathUnset = _config_helpers.UNSET
_STREAMABLE_HTTP_TYPES = _config_helpers.STREAMABLE_HTTP_TYPES
_SUPPORTED_SERVER_TYPES = _config_helpers.SUPPORTED_SERVER_TYPES
_DEFAULT_ENV_ALLOWLIST = _config_helpers.DEFAULT_ENV_ALLOWLIST
_MCP_CLOSE_TIMEOUT_SECONDS = 5.0
_MCP_TOOL_CANCEL_GRACE_SECONDS = 1.0
GSHEETS_BROKER_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
_GSHEETS_SERVER_NAME = "gsheets-mcp"
_GSHEETS_BROKER_READ_TOOLS = frozenset({
  "gsheets_list_tabs",
  "gsheets_read_range",
})
_GSHEETS_BROKER_WRITE_TOOLS = frozenset({
  "gsheets_append_rows",
  "gsheets_clear_range",
  "gsheets_copy_spreadsheet",
  "gsheets_create_spreadsheet",
  "gsheets_recalculate_range",
  "gsheets_write_range",
})
_GSHEETS_BROKER_TOOLS = _GSHEETS_BROKER_READ_TOOLS | _GSHEETS_BROKER_WRITE_TOOLS
_READ_ONLY_POLICY_CLASSES = frozenset({"read", "pure_transform"})
_AUTOMATIC_REPLAY_EFFECTS = frozenset({"read", "pure_transform", "support"})
PER_USER_SESSION_TTL_SECONDS = 60 * 60
PER_USER_EXPIRY_MARGIN_SECONDS = 5 * 60
PER_USER_IDLE_REAP_SECONDS = 30 * 60
PER_USER_REAPER_INTERVAL_SECONDS = 60.0
PER_USER_INSTANCE_CAP = 32
PER_USER_DRAIN_POLL_SECONDS = 0.05


class _McpCallKwargs(TypedDict, total=False):
  read_timeout_seconds: float
  meta: dict[str, object]


def _is_exact_prepared_tool_call(value: object) -> TypeIs[PreparedToolCall]:
  return type(value) is PreparedToolCall
def _resolve_mcp_config_path(
  config_path: _config_helpers.McpConfigPathInput = _UNSET,
) -> Path | None:
  return _config_helpers.resolve_mcp_config_path(
    config_path,
    unset=_UNSET,
    environ=os.environ,
  )


_McpStdioTerminationFallbackFilter = _runtime_helpers.McpStdioTerminationFallbackFilter


def _suppress_mcp_stdio_termination_fallback_warnings():
  return _runtime_helpers.suppress_mcp_stdio_termination_fallback_warnings(logging.getLogger, _McpStdioTerminationFallbackFilter)


def _build_mcp_env(server_env: Dict[str, Any] | None) -> Dict[str, str]:
  return _config_helpers.build_mcp_env(
    server_env if isinstance(server_env, dict) else None,
    environ=os.environ,
  )


def _build_http_headers(headers: Dict[str, Any] | None) -> Dict[str, str]:
  return _config_helpers.build_http_headers(
    headers if isinstance(headers, dict) else None,
    environ=os.environ,
  )


def _stdio_connect_retries() -> int:
  return _config_helpers.stdio_connect_retries(environ=os.environ, logger=log)


def _stdio_connect_retry_delay(attempt: int) -> float:
  return _config_helpers.stdio_connect_retry_delay(
    attempt,
    environ=os.environ,
    logger=log,
    jitter_fn=random.uniform,
  )


def _stdio_connect_stabilize_delay() -> float:
  return _config_helpers.stdio_connect_stabilize_delay(environ=os.environ, logger=log)


def _startup_concurrency_limit() -> int:
  return _config_helpers.startup_concurrency_limit(environ=os.environ, logger=log)


def _is_retryable_stdio_connect_error(exc: BaseException) -> bool:
  return _config_helpers.is_retryable_stdio_connect_error(exc)


def _is_retryable_stdio_startup_error(exc: BaseException) -> bool:
  return _config_helpers.is_retryable_stdio_startup_error(exc)


def _consume_mcp_tool_call_result(task: asyncio.Task[Any]) -> None:
  _runtime_helpers.consume_mcp_tool_call_result(task, logger=log)


def _safe_cache_name(name: str) -> str:
  return _config_helpers.safe_cache_name(name)


class _JsonFileKeyValue(_oauth_storage.JsonFileKeyValue):
  def _replace_file(self, tmp_path: Path, path: Path) -> None:
    os.replace(tmp_path, path)

  def _time(self) -> float:
    return time.time()



def _classify_exception(exc: Exception, msg: str) -> str:
  return _error_helpers.classify_exception(exc, msg)


def _classify_mcp_error(message: str) -> str:
  return _error_helpers.classify_mcp_error(message)


def _tool_error_from_exception(exc: Exception) -> dict[str, str]:
  """Project a failed MCP call's exception; the message always names its class.

  Transport exceptions such as anyio's ClosedResourceError stringify empty, so
  ``str(exc)`` alone leaves the caller no why-not.
  """
  detail = str(exc)
  return {
    "code": "tool_error",
    "sub_code": _classify_exception(exc, detail),
    "message": ": ".join(filter(None, (type(exc).__name__, detail))),
  }



def _is_sheets_transport_failure(exc: BaseException) -> bool:
  return _is_retryable_stdio_startup_error(exc)


def _sheets_structured_error(
  result: _connection_helpers.McpToolCallResult,
  *,
  expected_operation: str,
) -> dict[str, Any] | None:
  payload = result.structured_content
  if not isinstance(payload, dict) or payload.get("status") != "error":
    return None
  operation = payload.get("operation")
  error = payload.get("error")
  if operation != expected_operation or not isinstance(error, dict):
    return None
  code = error.get("code")
  message = error.get("message")
  outcome = error.get("outcome")
  retry = error.get("retry")
  if not isinstance(code, str) or not code or not isinstance(message, str) or not message:
    return None
  if not isinstance(outcome, dict) or not isinstance(retry, dict):
    return None
  if outcome.get("state") not in {"not_started", "unchanged", "uncertain", "partial", "restored"}:
    return None
  if not isinstance(outcome.get("phase"), str) or not isinstance(outcome.get("mutation_may_have_occurred"), bool):
    return None
  if not isinstance(retry.get("safe"), bool) or not isinstance(retry.get("automatic"), bool):
    return None
  if not isinstance(retry.get("action"), str) or "retry_after_seconds" not in retry:
    return None
  if "validation" not in error or "recovery" not in error:
    return None
  return copy.deepcopy(payload)


def _sheets_gateway_error(payload: dict[str, Any]) -> dict[str, Any]:
  error = payload["error"]
  return {
    "code": "mcp_tool_error",
    "sub_code": error["code"],
    "message": error["message"],
    "data": copy.deepcopy(payload),
  }


def _gateway_sheets_error_payload(
  operation: str,
  *,
  code: str,
  message: str,
  outcome_state: str,
  phase: str,
  mutation_may_have_occurred: bool,
  retry_safe: bool,
  retry_automatic: bool,
  retry_action: str,
) -> dict[str, Any]:
  return {
    "status": "error",
    "operation": operation,
    "error": {
      "code": code,
      "message": message,
      "outcome": {
        "state": outcome_state,
        "phase": phase,
        "mutation_may_have_occurred": mutation_may_have_occurred,
      },
      "retry": {
        "safe": retry_safe,
        "automatic": retry_automatic,
        "action": retry_action,
        "retry_after_seconds": None,
      },
      "validation": None,
      "recovery": None,
    },
  }


def _sheets_error_allows_automatic_read_retry(
  payload: dict[str, Any],
  policy_class: str | None,
) -> bool:
  error = payload.get("error")
  if not isinstance(error, dict) or policy_class not in _READ_ONLY_POLICY_CLASSES:
    return False
  outcome = error.get("outcome")
  retry = error.get("retry")
  return bool(
    isinstance(outcome, dict)
    and outcome.get("state") == "not_started"
    and isinstance(retry, dict)
    and retry.get("safe") is True
    and retry.get("automatic") is True
  )


def _startup_failure_from_exception(exc: BaseException) -> Dict[str, Any]:
  return _error_helpers.startup_failure_from_exception(
    exc,
    is_retryable_stdio_connect_error=_is_retryable_stdio_connect_error,
  )


def _preflight_stdio_executable(
  command: str,
  args: Sequence[str],
  env: Mapping[str, str],
) -> None:
  missing = _config_helpers.resolve_missing_stdio_executable(command, args, env)
  if missing is not None:
    raise _error_helpers.McpExecutableMissingError(missing)


@dataclass
class _ServerState:
  name: str
  session: _connection_helpers.McpClientSession
  exit_contexts: List[Any]
  tool_definitions: List[Dict[str, Any]]
  tool_names: Set[str]
  tool_prefix: str = ""
  config: Dict[str, Any] | None = None
  exported_tool_names: frozenset[str] | None = None
  tool_metadata: Mapping[str, Mapping[str, Any] | None] = field(default_factory=dict)
  stdio_eof: asyncio.Event | None = None
  stdio_receive_done: asyncio.Event | None = None
  stdio_watch_task: asyncio.Task[Any] | None = None
  reconnect_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  published_tool_definitions: List[Dict[str, Any]] = field(init=False)

  def __post_init__(self) -> None:
    names = (
      self.tool_names
      if self.exported_tool_names is None
      else self.exported_tool_names
    )
    self.exported_tool_names = frozenset(names)
    self.published_tool_definitions = self.tool_definitions


@dataclass
class _ConnectedServerState(_ServerState):
  session: _connection_helpers.McpClientSession


def _consume_future_exception(future: "asyncio.Future[Any]") -> None:
  """Mark a spawn result retrieved so an abandoned spawn logs no stray warning."""
  if not future.cancelled():
    future.exception()


class _PerUserChildHost:
  """One task owns a per-user child's lifecycle, from its startup to its close.

  The transport contexts themselves are entered and exited by the connection
  layer's own host task (`mcp_client_connections._TransportHost`). This host
  carries what is per-user: the caller's hand-off (`ready`), the retirement
  request, and a task that outlives an abandoned startup, which holds the
  child's instance-cap slot until the child is let go.
  """

  def __init__(self, server_name: str, user_id: str) -> None:
    self.server_name = server_name
    self.user_id = user_id
    self.task: asyncio.Task[None] | None = None
    self.ready: asyncio.Future[_ConnectedServerState] = (
      asyncio.get_running_loop().create_future()
    )
    self.ready.add_done_callback(_consume_future_exception)
    self.close_requested = asyncio.Event()
    self.closed = asyncio.Event()

  def start(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    task = asyncio.ensure_future(coro)
    task.add_done_callback(self._settle)
    self.task = task
    return task

  def _settle(self, _task: asyncio.Task[None]) -> None:
    """However the host task ends, everyone waiting on this child ends with it."""
    if not self.ready.done():
      self.ready.cancel()
    self.closed.set()

  def request_close(self) -> None:
    """Ask for this child to be retired.

    Never a cancellation of the host task, and never a deadline over it:
    cancelling a host mid-teardown interrupts the very cleanup that reaps the
    child, and one cancelled before its coroutine runs leaves its waiters
    unsettled. A startup still in flight runs to its own end and the close is
    waiting for it there. The connection layer's startup timeouts are per
    request, so they do not bound a server that pages its tool catalog without
    end; that loop carries its own page ceiling in
    `mcp_client_connections.initialize_session_state`, which is where the bound
    belongs — not here, where a deadline would surround the startup's own
    teardown.
    """
    self.close_requested.set()

  async def close(self) -> None:
    self.request_close()
    await self.closed.wait()


@dataclass
class _PerUserServerState:
  server: _ServerState
  expires_at: float
  last_used_at: float
  active_calls: int = 0
  draining: bool = False
  binding_fingerprint: bytes | None = None
  host: _PerUserChildHost | None = None


@dataclass(frozen=True, slots=True)
class _PerUserGatewaySubject:
  """Identity derived from an authenticated gateway session, never a caller id."""

  user_id: str
  user_email: str | None = None

  @classmethod
  def from_gateway_session(cls, session: Any) -> "_PerUserGatewaySubject":
    from .session import GatewaySession

    if not isinstance(session, GatewaySession):
      raise ValueError("per-user MCP requires a gateway session")
    risk_user_id = session.risk_user_id
    owner_user_id = str(session.owner_user_id or "").strip()
    if (
      isinstance(risk_user_id, bool)
      or not isinstance(risk_user_id, int)
      or risk_user_id <= 0
      or owner_user_id != str(risk_user_id)
    ):
      raise ValueError("gateway session has no canonical per-user subject")
    user_email = str(session.user_email or "").strip().lower() or None
    return cls(user_id=owner_user_id, user_email=user_email)


class _PerUserMcpError(RuntimeError):
  def __init__(self, code: str, message: str | None = None) -> None:
    super().__init__(message or code)
    self.code = code


def registered_mcp_dispatch_scope(
  *,
  user_id: str,
  dispatch_scope: Mapping[str, object] | None,
) -> Mapping[str, object]:
  """Bind trusted runtime identity to the existing session dispatch scope."""

  normalized_user_id = str(user_id or "").strip()
  if not normalized_user_id:
    raise ValueError("registered MCP dispatch scope requires a trusted user")
  if dispatch_scope is not None and not isinstance(dispatch_scope, Mapping):
    raise TypeError("registered MCP session dispatch scope must be a mapping")
  return MappingProxyType({
    **dict(dispatch_scope or {}),
    "user_id": normalized_user_id,
  })


@dataclass(frozen=True, slots=True)
class RegisteredMcpRawPatchAuthorization:
  """Exact durable authorization material emitted by raw-patch planning."""

  approval_identity: Mapping[str, object]
  approval_arguments: Mapping[str, object]
  approval_arguments_hash: str
  prepared_payload: bytes

  def materialize_approval_arguments(self) -> dict[str, object]:
    return PreparedToolCall(self.approval_arguments).materialize_input()

  @classmethod
  def from_plan_decision(
    cls,
    decision: PlanDecision,
  ) -> "RegisteredMcpRawPatchAuthorization":
    if decision.kind != "prepared_plan" or decision.prepared_plan is None:
      raise ValueError(
        "prepared MCP authorization requires an exact prepared-plan decision"
      )
    plan = decision.prepared_plan
    if (
      plan.get("schema_version")
      != "registered-mcp-raw-patch-authorization.v1"
    ):
      raise ValueError("prepared raw-patch authorization schema is unsupported")
    approval_identity = plan.get("approval_identity")
    approval_arguments = plan.get("approval_arguments")
    encoded_payload = plan.get("prepared_payload_base64")
    if not isinstance(approval_identity, Mapping):
      raise TypeError("prepared MCP approval identity must be a mapping")
    if not isinstance(approval_arguments, Mapping):
      raise TypeError("prepared MCP approval arguments must be a mapping")
    if type(encoded_payload) is not str or not encoded_payload:
      raise TypeError("prepared MCP authorization payload must be base64 text")
    try:
      prepared_payload = base64.b64decode(encoded_payload, validate=True)
    except (ValueError, TypeError) as exc:
      raise ValueError(
        "prepared MCP authorization payload is not canonical base64"
      ) from exc
    if not prepared_payload:
      raise ValueError("prepared MCP authorization payload must not be empty")
    frozen_approval_arguments = PreparedToolCall(approval_arguments)
    return cls(
      approval_identity=approval_identity,
      approval_arguments=frozen_approval_arguments.prepared_input,
      approval_arguments_hash=sha256_args(
        frozen_approval_arguments.materialize_input()
      ),
      prepared_payload=prepared_payload,
    )


@dataclass(frozen=True, slots=True)
class RegisteredMcpDirectToolCall:
  """One immutable registered MCP call without an executable plan."""

  descriptor: RegisteredMcpToolDescriptor
  prepared_call: PreparedToolCall
  planning: PlanDecision
  approval_required: bool
  approval_reuse_key: str | None

  @property
  def prepared_authorization(self) -> None:
    return None


@dataclass(frozen=True, slots=True)
class RegisteredMcpPlannedToolCall:
  """One immutable registered MCP plan that always uses durable approval."""

  descriptor: RegisteredMcpToolDescriptor
  prepared_call: PreparedToolCall
  planning: PlanDecision
  prepared_authorization: RegisteredMcpRawPatchAuthorization
  approval_reuse_key: str | None

  @property
  def approval_required(self) -> bool:
    return True

  def materialize_authorized_input(
    self,
    tool_call_id: str,
  ) -> dict[str, object]:
    """Mint the registered post-approval credential into provider input."""

    prepared = self.prepared_call.materialize_input()
    from .raw_patch_authorization_store import encode_reference

    return {
      **prepared,
      "authorization_ref": encode_reference(tool_call_id),
    }


RegisteredMcpToolCall = (
  RegisteredMcpDirectToolCall | RegisteredMcpPlannedToolCall
)
RegisteredSdkMcpToolCall = RegisteredMcpToolCall


class McpClientManager:
  """Manage MCP server lifecycles and tool routing.

  The manager can load servers from inline config, from `MCP_CONFIG_PATH`, or
  from an explicit alternate config path. With none of those inputs, it loads
  zero file-backed servers. On startup it connects to each allowed server,
  lists its tools, filters name collisions, and exposes a merged tool catalog
  to the runner.

  `inline_servers` is the easiest way to ship self-contained examples because it
  avoids any dependency on a separate config file.

  `provider_symbol_resolver` is an optional application-owned port. Without it,
  input-preparation routes leave symbols unchanged; the package never imports
  a product resolver or reads a product symbol cache.
  """

  def __init__(
    self,
    allowed_servers: AbstractSet[str] | None = None,
    builtin_tool_names: Set[str] | None = None,
    config_path: _config_helpers.McpConfigPathInput = _UNSET,
    inline_servers: Dict[str, Dict[str, Any]] | None = None,
    timeout_overrides: Mapping[str, float] | None = None,
    tool_timeout_overrides: Mapping[str, float] | None = None,
    server_aliases: Mapping[str, str] | None = None,
    logical_server_routes: Mapping[str, str] | None = None,
    logical_tool_aliases: Mapping[str, Mapping[str, str]] | None = None,
    input_preparation_routes: Sequence[McpInputPreparationRoute] = (),
    provider_ids_by_server: Dict[str, str] | None = None,
    server_env_passthrough: Mapping[str, AbstractSet[str]] | None = None,
    per_user_env_resolver: Callable[
      [str, str, str | None], Mapping[str, str]
    ] | None = None,
    startup_timeout: int = 15,
    default_tool_timeout: int = 30,
    strip_input_fields: set[str] | None = None,
    tool_registration_catalog: ToolRegistrationCatalog | None = None,
    tool_policy_implementations: ToolPolicyImplementationRegistry | None = None,
    input_preparation_context_factory: Callable[
      [RegisteredMcpToolDescriptor, Mapping[str, object] | None], object
    ] | None = None,
    planning_context_factory: Callable[
      [RegisteredMcpToolDescriptor, PreparedToolCall, object | None], object
    ] | None = None,
    redaction_context_factory: Callable[
      [RegisteredMcpToolDescriptor], object
    ] | None = None,
    provider_symbol_resolver: Callable[[Any], Any] | None = None,
  ) -> None:
    self._lock = asyncio.Lock()
    self._started = False
    self._configured_transport_server_names: Set[str] | None = None
    self._servers: Dict[str, _ServerState] = {}
    self._per_user_servers: Dict[tuple[str, str], _PerUserServerState] = {}
    self._per_user_spawn_locks: Dict[tuple[str, str], asyncio.Lock] = {}
    self._per_user_spawn_reservations: Dict[str, int] = {}
    self._per_user_reaper_task: asyncio.Task[Any] | None = None
    self._per_user_hosts: Set[_PerUserChildHost] = set()
    self._drain_tasks: Set[asyncio.Task[Any]] = set()
    self._tool_definitions: List[Dict[str, Any]] = []
    self._tool_to_server: Dict[str, str] = {}
    self._prefixed_to_original: Dict[str, str] = {}
    self._mcp_tool_names: Set[str] = set()
    self._startup_diagnostics: Dict[str, Dict[str, Any]] = {}
    self._server_aliases = dict(server_aliases or {})
    self._logical_server_routes = dict(logical_server_routes or {})
    self._logical_tool_aliases = {
      server_name: dict(aliases)
      for server_name, aliases in dict(logical_tool_aliases or {}).items()
    }
    self._input_preparation_routes = index_mcp_input_preparation_routes(
      input_preparation_routes
    )
    self._provider_ids_by_server = {
      self._canonical_server_name(server_name): str(provider_id).strip()
      for server_name, provider_id in dict(provider_ids_by_server or {}).items()
      if str(provider_id).strip()
    }
    self._server_env_passthrough = {
      self._canonical_server_name(server_name): {str(env_name) for env_name in env_names}
      for server_name, env_names in dict(server_env_passthrough or {}).items()
    }
    self._per_user_env_resolver = per_user_env_resolver
    self._per_user_binding_hmac_key = os.urandom(32)
    self._logical_tool_definitions: Dict[str, List[Dict[str, Any]]] = {}
    self._logical_alias_generation: tuple[
      _catalog_helpers.LogicalToolAliasProvenance,
      ...,
    ] = ()
    self._tool_registration_catalog = (
      validate_tool_registration_catalog(tool_registration_catalog)
      if tool_registration_catalog is not None
      else None
    )
    if (tool_policy_implementations is None) != (
      input_preparation_context_factory is None
    ):
      raise ValueError(
        "tool policy implementations and input-preparation context factory "
        "must be provided together"
      )
    if (
      tool_policy_implementations is not None
      and type(tool_policy_implementations)
      is not ToolPolicyImplementationRegistry
    ):
      raise TypeError(
        "tool_policy_implementations must be an exact "
        "ToolPolicyImplementationRegistry"
      )
    if (
      input_preparation_context_factory is not None
      and not callable(input_preparation_context_factory)
    ):
      raise TypeError("input_preparation_context_factory must be callable")
    if planning_context_factory is not None and not callable(
      planning_context_factory
    ):
      raise TypeError("planning_context_factory must be callable")
    if (
      planning_context_factory is not None
      and tool_policy_implementations is None
    ):
      raise ValueError(
        "registered planning context requires tool policy implementations"
      )
    if redaction_context_factory is not None and not callable(
      redaction_context_factory
    ):
      raise TypeError("redaction_context_factory must be callable")
    if (
      redaction_context_factory is not None
      and tool_policy_implementations is None
    ):
      raise ValueError(
        "registered redaction context requires tool policy implementations"
      )
    if (
      tool_policy_implementations is not None
      and redaction_context_factory is None
    ):
      raise ValueError(
        "registered MCP policy runtime requires a redaction context factory"
      )
    if (
      tool_policy_implementations is not None
      and self._tool_registration_catalog is None
    ):
      raise ValueError(
        "registered input preparation requires a tool registration catalog"
      )
    if self._tool_registration_catalog is not None and self._input_preparation_routes:
      raise ValueError(
        "registered MCP input preparation cannot use legacy preparation routes"
      )
    if tool_policy_implementations is not None:
      assert self._tool_registration_catalog is not None
      tool_policy_implementations.validate_catalog(
        self._tool_registration_catalog
      )
    self._tool_policy_implementations = tool_policy_implementations
    self._input_preparation_context_factory = (
      input_preparation_context_factory
    )
    self._planning_context_factory = planning_context_factory
    self._redaction_context_factory = redaction_context_factory
    self._provider_symbol_resolver = provider_symbol_resolver
    if self._tool_registration_catalog is not None and (
      timeout_overrides or tool_timeout_overrides
    ):
      raise ValueError(
        "registered MCP timeouts must come only from the tool registration catalog"
      )
    self._registered_mcp_descriptors_by_exposed_name: Mapping[
      str,
      RegisteredMcpToolDescriptor,
    ] = MappingProxyType({})
    self._dispatch_to_original: Dict[str, str] = {}
    self._allowed_servers = self._canonical_server_names(allowed_servers) if allowed_servers is not None else None
    self._builtin_tool_names = set(builtin_tool_names or set())
    self._inline_servers = dict(inline_servers or {})
    self._config_path = _resolve_mcp_config_path(config_path)
    self._timeout_overrides = {
      self._canonical_server_name(server_name): timeout
      for server_name, timeout in dict(timeout_overrides or {}).items()
    }
    self._tool_timeout_overrides = {
      self._canonical_tool_timeout_key(tool_name): timeout
      for tool_name, timeout in dict(tool_timeout_overrides or {}).items()
    }
    self._startup_timeout = startup_timeout
    self._default_tool_timeout = default_tool_timeout
    self._strip_input_fields = strip_input_fields or set()

  def _canonical_server_name(self, server_name: str) -> str:
    return self._server_aliases.get(server_name, server_name)

  def _canonical_server_names(self, server_names: AbstractSet[str]) -> Set[str]:
    return {self._canonical_server_name(server_name) for server_name in server_names}

  def _transport_server_name(self, server_name: str) -> str:
    canonical_name = self._canonical_server_name(server_name)
    return self._logical_server_routes.get(canonical_name, canonical_name)

  def _transport_server_names(self, server_names: Set[str]) -> Set[str]:
    return {self._transport_server_name(server_name) for server_name in server_names}

  def _transport_only_server_names(self) -> Set[str]:
    if self._allowed_servers is None:
      return set()
    return {
      self._transport_server_name(logical_server)
      for logical_server in self._logical_server_routes
      if logical_server in self._allowed_servers
      and self._transport_server_name(logical_server) not in self._allowed_servers
    }

  def _is_transport_only_server(self, server_name: str) -> bool:
    return self._canonical_server_name(server_name) in self._transport_only_server_names()

  def _canonical_tool_timeout_key(self, tool_name: str) -> str:
    if "." not in tool_name:
      return tool_name
    server_name, original_name = tool_name.split(".", 1)
    return f"{self._canonical_server_name(server_name)}.{original_name}"

  def _set_startup_diagnostic(
    self,
    server_name: str,
    *,
    category: str,
    message: str,
    retryable: bool,
    error_type: str | None = None,
  ) -> None:
    canonical_name = self._canonical_server_name(server_name)
    payload: Dict[str, Any] = {
      "server": canonical_name,
      "category": category,
      "retryable": bool(retryable),
      "message": message,
    }
    if error_type:
      payload["error_type"] = error_type
    self._startup_diagnostics[canonical_name] = payload

  def _timeout_for_tool(self, server_name: str, exposed_name: str, original_name: str) -> float:
    if self._tool_registration_catalog is not None:
      descriptor = self.get_registered_mcp_tool_descriptor(exposed_name)
      if descriptor.server.transport_server_id != server_name:
        raise ValueError(
          "registered MCP timeout server does not match the live transport"
        )
      return descriptor.server.per_tool_timeout_seconds.get(
        descriptor.identity.logical_name,
        descriptor.server.default_timeout_seconds,
      )
    for key in (
      f"{server_name}.{original_name}",
      f"{server_name}.{exposed_name}",
      original_name,
      exposed_name,
    ):
      timeout = self._tool_timeout_overrides.get(key)
      if timeout is not None:
        return timeout
    return self._timeout_overrides.get(server_name, self._default_tool_timeout)

  def _canonicalize_server_configs(
    self,
    mcp_servers: Dict[str, Dict[str, Any]],
  ) -> Dict[str, Dict[str, Any]]:
    return _startup_helpers.canonicalize_server_configs(
      mcp_servers,
      canonical_server_name=self._canonical_server_name,
      logger=log,
    )

  def _configured_server_configs(self) -> Dict[str, Dict[str, Any]]:
    config = self._read_claude_config()
    mcp_servers = config.get("mcpServers", {})
    if not isinstance(mcp_servers, dict):
      mcp_servers = {}
    configured = dict(mcp_servers)
    configured.update(self._inline_servers)
    return self._canonicalize_server_configs(configured)

  def get_configured_transport_server_names(self) -> Set[str]:
    """Return the transports captured from this manager's startup config."""

    configured = self._configured_transport_server_names
    if configured is None:
      raise RuntimeError(
        "configured MCP transports are unavailable before manager startup"
      )
    return set(configured)

  async def startup(self, allowed_servers: Set[str] | None = None) -> None:
    async with self._lock:
      if self._started:
        return
      await _startup_helpers.startup_manager(
        self,
        allowed_servers,
        supported_server_types=set(_SUPPORTED_SERVER_TYPES),
        logger=log,
      )

  def _connection_runtime(self) -> _connection_helpers.McpConnectionRuntime:
    return _connection_helpers.McpConnectionRuntime(
      startup_concurrency_limit=_startup_concurrency_limit,
      startup_failure_from_exception=_startup_failure_from_exception,
      streamable_http_types=set(_STREAMABLE_HTTP_TYPES),
      stdio_connect_retries=_stdio_connect_retries,
      stdio_connect_retry_delay=_stdio_connect_retry_delay,
      stdio_connect_stabilize_delay=_stdio_connect_stabilize_delay,
      is_retryable_stdio_startup_error=_is_retryable_stdio_startup_error,
      build_mcp_env=_build_mcp_env,
      preflight_stdio_executable=_preflight_stdio_executable,
      build_http_headers=_build_http_headers,
      parse_allowed_tools=_config_helpers.parse_allowed_tools,
      safe_cache_name=_safe_cache_name,
      close_contexts=self._close_contexts,
      server_state_factory=_ConnectedServerState,
      stdio_server_parameters_factory=StdioServerParameters,
      stdio_client_factory=stdio_client,
      client_session_factory=ClientSession,
      httpx_module=httpx2,
      streamable_http_client_factory=streamable_http_client,
      json_file_key_value_factory=_JsonFileKeyValue,
      fastmcp_oauth_factory=FastMCPOAuth,
      path_factory=Path,
      environ=os.environ,
      logger=log,
    )

  async def _connect_startup_servers(
    self,
    connect_jobs: Sequence[tuple[str, Dict[str, Any]]],
  ) -> list[_ConnectedServerState | None]:
    return await _connection_helpers.connect_startup_servers(
      self,
      connect_jobs,
      self._connection_runtime(),
    )

  async def _publish_server_states(
    self,
    states: Sequence[_ServerState],
    *,
    replacing: _ServerState | None = None,
  ) -> bool:
    """Compile and publish under the caller-held lifecycle lock."""
    if replacing is not None and self._servers.get(replacing.name) is not replacing:
      for state in states:
        await self._close_contexts(state.exit_contexts)
      return False

    # Compile only projections; live connection identities and their advertised
    # catalogs survive unrelated publications and rejected replacements.
    servers = dict(self._servers)
    servers.update((state.name, state) for state in states)
    candidate = copy.copy(self)
    candidate._servers = {
      name: replace(state, tool_names=set(state.tool_names))
      for name, state in servers.items()
    }
    candidate._startup_diagnostics = dict(self._startup_diagnostics)
    candidate._apply_collision_filtering()

    for state in states:
      displaced = self._servers.get(state.name)
      if displaced is not None and displaced is not state:
        await self._close_server(displaced)
    for name, state in servers.items():
      state.published_tool_definitions = candidate._servers[name].published_tool_definitions
      state.tool_names = candidate._servers[name].tool_names
    for state in states:
      update_mcp_tool_metadata(state.name, state.tool_metadata)
    self._servers = servers
    self._startup_diagnostics = candidate._startup_diagnostics
    self._tool_definitions = candidate._tool_definitions
    self._tool_to_server = candidate._tool_to_server
    self._prefixed_to_original = candidate._prefixed_to_original
    self._dispatch_to_original = candidate._dispatch_to_original
    self._mcp_tool_names = candidate._mcp_tool_names
    self._logical_tool_definitions = candidate._logical_tool_definitions
    self._logical_alias_generation = candidate._logical_alias_generation
    self._registered_mcp_descriptors_by_exposed_name = (
      candidate._registered_mcp_descriptors_by_exposed_name
    )
    for state in states:
      if state.stdio_eof is not None and state.stdio_watch_task is None:
        state.stdio_watch_task = asyncio.create_task(self._reconnect_stdio_on_eof(state))
    return True

  async def _connect_or_warn(self, name: str, config: Dict[str, Any]) -> _ConnectedServerState | None:
    return await _connection_helpers.connect_or_warn(
      self,
      name,
      config,
      self._connection_runtime(),
    )

  async def _connect(self, name: str, config: Dict[str, Any]) -> _ConnectedServerState:
    return await _connection_helpers.connect(
      self,
      name,
      config,
      self._connection_runtime(),
    )

  async def _connect_stdio_with_retries(self, name: str, config: Dict[str, Any]) -> _ConnectedServerState:
    return await _connection_helpers.connect_stdio_with_retries(
      self,
      name,
      config,
      self._connection_runtime(),
    )

  async def _connect_stdio(self, name: str, config: Dict[str, Any]) -> _ConnectedServerState:
    return await _connection_helpers.connect_stdio(
      self,
      name,
      config,
      self._connection_runtime(),
    )

  async def _connect_streamable_http(self, name: str, config: Dict[str, Any]) -> _ConnectedServerState:
    return await _connection_helpers.connect_streamable_http(
      self,
      name,
      config,
      self._connection_runtime(),
    )

  def _build_http_auth(self, name: str, url: str, config: Dict[str, Any]) -> Any | None:
    return _connection_helpers.build_http_auth(
      name,
      url,
      config,
      self._connection_runtime(),
    )

  async def _initialize_session_state(
    self,
    *,
    name: str,
    session: _connection_helpers.McpConnectionSession,
    exit_contexts: List[Any],
    tool_prefix: str,
    allowed_tools: tuple[str, ...] | None = None,
  ) -> _ConnectedServerState:
    return await _connection_helpers.initialize_session_state(
      self,
      name=name,
      session=session,
      exit_contexts=exit_contexts,
      tool_prefix=tool_prefix,
      allowed_tools=allowed_tools,
      runtime=self._connection_runtime(),
    )

  async def _verify_stdio_session_stable(
    self,
    session: _connection_helpers.McpConnectionSession,
  ) -> None:
    await _connection_helpers.verify_stdio_session_stable(
      self,
      session,
      self._connection_runtime(),
    )

  def get_tool_definitions(self) -> List[Dict[str, Any]]:
    if self._tool_registration_catalog is not None:
      return [
        self.get_registered_mcp_tool_descriptor(
          str(definition.get("name"))
        ).materialize_provider_definition()
        for definition in self._tool_definitions
      ]
    return copy.deepcopy(self._tool_definitions)

  def get_registered_mcp_tool_descriptor(
    self,
    exposed_name: str,
  ) -> RegisteredMcpToolDescriptor:
    """Return the exact live descriptor or fail closed for unknown routes."""

    if type(exposed_name) is not str or not exposed_name:
      raise UnknownRegisteredMcpToolDescriptorError(
        "registered MCP exposed name must be a non-empty exact str"
      )
    if self._tool_registration_catalog is None:
      raise UnknownRegisteredMcpToolDescriptorError(
        "MCP registration catalog is not configured"
      )
    descriptor = self._registered_mcp_descriptors_by_exposed_name.get(
      exposed_name
    )
    if descriptor is None:
      raise UnknownRegisteredMcpToolDescriptorError(
        f"unknown registered live MCP tool: {exposed_name}"
      )
    return descriptor

  def get_registered_mcp_tool_descriptor_for_sdk_tool(
    self,
    sdk_tool_name: str,
  ) -> RegisteredMcpToolDescriptor:
    """Resolve one exact SDK MCP identity through the live manager topology."""

    _mcp_marker, configured_server, provider_name = sdk_tool_name.split("__", 2)
    exposed_name = cast(
      str,
      self.resolve_tool_name(configured_server, provider_name),
    )
    return self.get_registered_mcp_tool_descriptor(exposed_name)

  def prepare_registered_mcp_tool_call_for_sdk_tool(
    self,
    sdk_tool_name: str,
    raw_input: Mapping[str, object],
    trusted_dispatch_scope: Mapping[str, object] | None,
    registered_approval_overlay: (
      Callable[[ToolRegistrationDeclaration, PreparedToolCall], bool] | None
    ) = None,
  ) -> RegisteredMcpToolCall:
    """Prepare and classify one SDK call through its live registration."""

    descriptor = self.get_registered_mcp_tool_descriptor_for_sdk_tool(
      sdk_tool_name
    )
    prepared_call = self.prepare_registered_tool_input(
      descriptor.exposed_name,
      raw_input,
      trusted_dispatch_scope,
    )
    return self.classify_registered_mcp_prepared_tool_call(
      descriptor.exposed_name,
      prepared_call,
      trusted_dispatch_scope,
      registered_approval_overlay,
    )

  def classify_registered_mcp_prepared_tool_call(
    self,
    exposed_name: str,
    prepared_call: PreparedToolCall,
    trusted_dispatch_scope: object | None,
    registered_approval_overlay: (
      Callable[[ToolRegistrationDeclaration, PreparedToolCall], bool] | None
    ) = None,
  ) -> RegisteredMcpToolCall:
    """Plan and classify one already-prepared exact registered MCP call."""

    if type(prepared_call) is not PreparedToolCall:
      raise TypeError("prepared_call must be an exact PreparedToolCall")
    descriptor = self.get_registered_mcp_tool_descriptor(exposed_name)
    registry = cast(
      ToolPolicyImplementationRegistry,
      self._tool_policy_implementations,
    )
    planning_policy = descriptor.declaration.semantics.planning_policy
    planning_context = None
    if planning_policy.policy_id != "none":
      context_factory = self._planning_context_factory
      if context_factory is None:
        raise RuntimeError(
          "registered MCP planning context is not configured"
        )
      planning_context = context_factory(
        descriptor,
        prepared_call,
        trusted_dispatch_scope,
      )
    planning = registry.execute_planning(
      planning_policy,
      PlanningCall(
        descriptor.identity,
        prepared_call.prepared_input,
        planning_context,
      ),
    )
    if planning.kind == "authorized_intent":
      raise ToolPolicyResultError(
        "registered MCP planning must return none or an exact prepared plan"
      )
    if planning.kind == "prepared_plan":
      if planning_policy.policy_id != "raw-patch-ops":
        raise ToolPolicyResultError(
          "registered MCP prepared planning is not executable"
        )
      prepared_authorization = (
        RegisteredMcpRawPatchAuthorization.from_plan_decision(planning)
      )
    else:
      prepared_authorization = None
    policy = descriptor.declaration.semantics.approval
    if policy.mode == "never":
      intrinsic_approval_required = False
    elif policy.mode == "always":
      intrinsic_approval_required = True
    else:
      assert policy.predicate is not None
      intrinsic_approval_required = registry.execute_approval_predicate(
        policy.predicate,
        ApprovalPredicateCall(
          descriptor.identity,
          prepared_call.prepared_input,
        ),
      )
    overlay_approval_required = False
    if registered_approval_overlay is not None:
      overlay_result = registered_approval_overlay(
        descriptor.declaration,
        prepared_call,
      )
      if type(overlay_result) is not bool:
        raise TypeError(
          "registered approval overlay must return an exact bool"
        )
      overlay_approval_required = overlay_result
    approval_required = (
      intrinsic_approval_required or overlay_approval_required
    )
    approval_reuse_key = None
    if (
      approval_required
      and not overlay_approval_required
      and policy.cache_key is not None
    ):
      approval_reuse_key = registry.execute_approval_cache_key(
        policy.cache_key,
        ApprovalCacheKeyCall(
          descriptor.identity,
          prepared_call.prepared_input,
          prepared_call.exact_backend,
          (
            planning.prepared_plan
            if planning.kind == "prepared_plan"
            else planning.authorized_intent
          ),
        ),
      )
    if prepared_authorization is not None:
      return RegisteredMcpPlannedToolCall(
        descriptor,
        prepared_call,
        planning,
        prepared_authorization,
        approval_reuse_key,
      )
    return RegisteredMcpDirectToolCall(
      descriptor,
      prepared_call,
      planning,
      approval_required,
      approval_reuse_key,
    )

  def settle_registered_mcp_tool_result_for_sdk_tool(
    self,
    sdk_tool_name: str,
    tool_input: Mapping[str, object],
    result: object,
    error: Mapping[str, object] | None,
    semantic_error: Mapping[str, object] | None = None,
  ) -> ToolResultSettlement:
    """Settle one SDK MCP result through its exact live registration."""

    descriptor = self.get_registered_mcp_tool_descriptor_for_sdk_tool(
      sdk_tool_name
    )
    registry = self._tool_policy_implementations
    if registry is None:
      raise RuntimeError(
        "registered MCP outcome implementations are not configured"
      )
    outcome = registry.execute_outcome(
      descriptor.declaration.semantics.outcome_policy,
      OutcomeCall(result, error, semantic_error),
    )
    if outcome != OUTCOME_OK:
      return ToolResultSettlement(outcome=outcome)
    sources = registry.execute_source_identity(
      descriptor.declaration.semantics.source_identity_policy,
      SourceIdentityCall(
        descriptor.identity,
        result,
        tool_input,
        sdk_tool_name,
      ),
    )
    return ToolResultSettlement(
      outcome=outcome,
      sources=sources.identities,
    )

  def redact_registered_mcp_tool_input_for_sdk_tool(
    self,
    sdk_tool_name: str,
    tool_input: Mapping[str, object],
  ) -> dict[str, object]:
    """Redact one SDK MCP input through its exact live registration."""

    return self._redact_registered_mcp_descriptor_input(
      self.get_registered_mcp_tool_descriptor_for_sdk_tool(sdk_tool_name),
      tool_input,
    )

  def redact_registered_tool_input(
    self,
    exposed_name: str,
    prepared_call: PreparedToolCall,
  ) -> dict[str, object]:
    """Redact one native MCP call through its exact live registration."""

    if type(prepared_call) is not PreparedToolCall:
      raise TypeError("prepared_call must be an exact PreparedToolCall")
    return self._redact_registered_mcp_descriptor_input(
      self.get_registered_mcp_tool_descriptor(exposed_name),
      prepared_call.materialize_input(),
    )

  def redact_registered_raw_tool_input(
    self,
    exposed_name: str,
    tool_input: Mapping[str, object],
  ) -> dict[str, object]:
    """Redact raw native input through its exact live registration."""

    return self._redact_registered_mcp_descriptor_input(
      self.get_registered_mcp_tool_descriptor(exposed_name),
      tool_input,
    )

  def _redact_registered_mcp_descriptor_input(
    self,
    descriptor: RegisteredMcpToolDescriptor,
    tool_input: Mapping[str, object],
  ) -> dict[str, object]:
    registry = cast(
      ToolPolicyImplementationRegistry,
      self._tool_policy_implementations,
    )
    context_factory = cast(
      Callable[[RegisteredMcpToolDescriptor], object],
      self._redaction_context_factory,
    )
    result = registry.execute_redaction(
      descriptor.declaration.semantics.redaction_policy,
      RedactionCall(
        descriptor.identity,
        tool_input,
        context_factory(descriptor),
      ),
    )
    return result.materialize_input()

  def prepare_registered_tool_input(
    self,
    exposed_name: str,
    raw_input: Mapping[str, object],
    trusted_dispatch_scope: Mapping[str, object] | None,
  ) -> PreparedToolCall:
    """Prepare one exact registered MCP call before policy and retry handling."""

    descriptor = self.get_registered_mcp_tool_descriptor(exposed_name)
    registry = self._tool_policy_implementations
    context_factory = self._input_preparation_context_factory
    if registry is None or context_factory is None:
      raise RuntimeError(
        "registered MCP input-preparation runtime is not configured"
      )
    trusted_context = context_factory(
      descriptor,
      trusted_dispatch_scope,
    )
    return registry.execute_input_preparation(
      descriptor.declaration.semantics.input_preparation_policy,
      InputPreparationCall(
        descriptor.identity,
        raw_input,
        trusted_context,
      ),
    )

  def _refresh_registered_mcp_tool_descriptors(self) -> None:
    """Rebuild the exact live join after the exposed topology is finalized."""

    self._registered_mcp_descriptors_by_exposed_name = MappingProxyType({})
    catalog = self._tool_registration_catalog
    if catalog is None:
      return
    descriptors = compile_registered_mcp_tool_descriptors(
      catalog,
      self.get_server_tool_route_bindings(self.get_server_names()),
    )
    by_exposed_name = {
      descriptor.exposed_name: descriptor
      for descriptor in descriptors
    }
    if set(by_exposed_name) != self._mcp_tool_names:
      raise ValueError(
        "registered MCP descriptors do not match the live tool surface"
      )
    self._registered_mcp_descriptors_by_exposed_name = MappingProxyType(
      by_exposed_name
    )

  def get_server_tool_definitions(self, server_names: Set[str]) -> List[Dict[str, Any]]:
    return [
      record.materialize()
      for record in self.get_server_tool_definition_records(server_names)
    ]

  def get_server_tool_definition_records(
    self,
    server_names: Set[str],
  ) -> tuple[OriginatedToolDefinition, ...]:
    canonical_server_names = self._canonical_server_names(set(server_names))
    records: list[OriginatedToolDefinition] = []
    for server_name, state in self._servers.items():
      if server_name in canonical_server_names and not self._is_transport_only_server(server_name):
        for definition in state.published_tool_definitions:
          record = OriginatedToolDefinition(
            definition=definition,
            origin="mcp",
            server_id=server_name,
          )
          if self._tool_to_server.get(record.name) != server_name:
            raise ValueError("MCP tool definition owner mapping is incoherent")
          records.append(record)
    for server_name in canonical_server_names:
      for definition in self._logical_tool_definitions.get(server_name, []):
        record = OriginatedToolDefinition(
          definition=definition,
          origin="mcp",
          server_id=server_name,
        )
        if self._tool_to_server.get(record.name) != server_name:
          raise ValueError("MCP tool definition owner mapping is incoherent")
        records.append(record)
    if self._tool_registration_catalog is not None:
      registered_records: list[OriginatedToolDefinition] = []
      for record in records:
        descriptor = self.get_registered_mcp_tool_descriptor(record.name)
        registered_record = descriptor.live_binding.originated_definition
        if registered_record != record:
          raise ValueError(
            "registered MCP definition diverges from the live surface"
          )
        registered_records.append(registered_record)
      return tuple(registered_records)
    return tuple(records)

  def get_server_tool_route_bindings(
    self,
    server_names: Set[str],
  ) -> tuple[LiveToolRouteBinding, ...]:
    """Return exact live MCP route identities with their provider definitions."""

    canonical_server_names = self._canonical_server_names(set(server_names))
    selected_logical_definitions = tuple(
      (server_name, definitions)
      for server_name, definitions in self._logical_tool_definitions.items()
      if server_name in canonical_server_names
    )
    generation_by_alias: dict[
      tuple[str, str],
      list[_catalog_helpers.LogicalToolAliasProvenance],
    ] = {}
    for raw_provenance in self._logical_alias_generation:
      if type(raw_provenance) is not _catalog_helpers.LogicalToolAliasProvenance:
        raise ValueError("MCP logical alias generation is incoherent")
      provenance = _catalog_helpers.LogicalToolAliasProvenance(
        logical_server_id=raw_provenance.logical_server_id,
        exposed_name=raw_provenance.exposed_name,
        transport_server_id=raw_provenance.transport_server_id,
        provider_original_name=raw_provenance.provider_original_name,
        physical_definition=raw_provenance.physical_definition,
        logical_definition=raw_provenance.logical_definition,
      )
      generation_by_alias.setdefault((
        provenance.logical_server_id,
        provenance.exposed_name,
      ), []).append(provenance)
    required_physical_servers = {
      server_name
      for server_name in canonical_server_names
      if server_name in self._servers
    }
    for server_name, _definitions in selected_logical_definitions:
      if server_name not in self._logical_server_routes:
        raise ValueError("MCP logical transport mapping is missing")
      transport_server_id = self._logical_server_routes[server_name]
      if (
        type(transport_server_id) is not str
        or transport_server_id not in self._servers
      ):
        raise ValueError("MCP logical transport mapping is incoherent")
      required_physical_servers.add(transport_server_id)

    physical_records: list[
      tuple[str, OriginatedToolDefinition, str]
    ] = []
    physical_by_route: dict[
      tuple[str, str],
      list[OriginatedToolDefinition],
    ] = {}
    for server_name, state in self._servers.items():
      if server_name not in required_physical_servers:
        continue
      tool_prefix = state.tool_prefix
      if type(tool_prefix) is not str:
        raise ValueError("MCP physical tool prefix is incoherent")
      for definition in state.published_tool_definitions:
        originated_definition = OriginatedToolDefinition(
          definition=definition,
          origin="mcp",
          server_id=server_name,
        )
        exposed_name = originated_definition.name
        if tool_prefix:
          if exposed_name not in self._prefixed_to_original:
            raise ValueError("MCP prefixed tool original mapping is missing")
          provider_original_name = self._prefixed_to_original[exposed_name]
          if exposed_name != f"{tool_prefix}{provider_original_name}":
            raise ValueError("MCP prefixed tool original mapping is incoherent")
        else:
          if exposed_name in self._prefixed_to_original:
            raise ValueError("MCP unprefixed tool original mapping is incoherent")
          provider_original_name = exposed_name
        route = (server_name, provider_original_name)
        physical_by_route.setdefault(route, []).append(originated_definition)
        physical_records.append((
          server_name,
          originated_definition,
          provider_original_name,
        ))

    bindings: list[LiveToolRouteBinding] = []
    seen_exposed_names: set[str] = set()
    seen_binding_ids: set[tuple[str, str, str]] = set()

    def append_binding(binding: LiveToolRouteBinding) -> None:
      binding_id = (
        binding.route_kind,
        binding.logical_server_id,
        binding.logical_name,
      )
      if binding.exposed_name in seen_exposed_names:
        raise ValueError("duplicate MCP live exposed route")
      if binding_id in seen_binding_ids:
        raise ValueError("duplicate MCP live route binding")
      seen_exposed_names.add(binding.exposed_name)
      seen_binding_ids.add(binding_id)
      bindings.append(binding)

    for (
      server_name,
      originated_definition,
      provider_original_name,
    ) in physical_records:
      if (
        server_name not in canonical_server_names
        or self._is_transport_only_server(server_name)
      ):
        continue
      exposed_name = originated_definition.name
      if self._tool_to_server.get(exposed_name) != server_name:
        raise ValueError("MCP tool route owner mapping is incoherent")
      dispatch_original = self._dispatch_to_original.get(exposed_name)
      if (
        dispatch_original is not None
        and dispatch_original != provider_original_name
      ):
        raise ValueError("MCP physical tool original mapping is incoherent")
      surface_matches = [
        surface_definition
        for surface_definition in self._tool_definitions
        if isinstance(surface_definition, Mapping)
        and surface_definition.get("name") == exposed_name
      ]
      if len(surface_matches) != 1:
        raise ValueError(
          "MCP physical tool requires unique surface provenance"
        )
      surface_definition = OriginatedToolDefinition(
        definition=surface_matches[0],
        origin="mcp",
        server_id=server_name,
      )
      if surface_definition != originated_definition:
        raise ValueError("MCP physical tool definition diverges from surface")
      append_binding(LiveToolRouteBinding(
        originated_definition=originated_definition,
        route_kind="physical",
        logical_name=provider_original_name,
        transport_server_id=server_name,
        provider_original_name=provider_original_name,
        provider_id=self._provider_ids_by_server.get(server_name),
      ))

    for server_name, definitions in selected_logical_definitions:
      for definition in definitions:
        originated_definition = OriginatedToolDefinition(
          definition=definition,
          origin="mcp",
          server_id=server_name,
        )
        exposed_name = originated_definition.name
        if self._tool_to_server.get(exposed_name) != server_name:
          raise ValueError("MCP tool route owner mapping is incoherent")
        transport_server_id = self._logical_server_routes[server_name]
        if exposed_name not in self._dispatch_to_original:
          raise ValueError("MCP logical tool original mapping is missing")
        provider_original_name = self._dispatch_to_original[exposed_name]
        declared_aliases = self._logical_tool_aliases.get(server_name, {})
        declared_original = declared_aliases.get(exposed_name, exposed_name)
        if provider_original_name != declared_original:
          raise ValueError("MCP logical tool original mapping is incoherent")

        generation_matches = generation_by_alias.get(
          (server_name, exposed_name),
          [],
        )
        if len(generation_matches) != 1:
          raise ValueError(
            "MCP logical tool requires unique alias generation"
          )
        generation = generation_matches[0]
        if (
          generation.transport_server_id != transport_server_id
          or generation.provider_original_name != provider_original_name
        ):
          raise ValueError("MCP logical route diverges from alias generation")
        if generation.logical_definition != originated_definition:
          raise ValueError(
            "MCP logical definition diverges from alias generation"
          )

        physical_provenance = physical_by_route.get(
          (transport_server_id, provider_original_name),
          [],
        )
        transport_state = self._servers[transport_server_id]
        advertised_definitions = [
          advertised
          for advertised in transport_state.tool_definitions
          if advertised.get("name") == provider_original_name
        ]
        if len(physical_provenance) != 1 or len(advertised_definitions) != 1:
          raise ValueError(
            "MCP logical tool requires unique physical provenance"
          )
        advertised_projection = OriginatedToolDefinition(
          definition=_catalog_helpers.materialize_published_tool_definition(
            advertised_definitions[0],
            prefix=transport_state.tool_prefix,
            strip_input_fields=self._strip_input_fields,
          ),
          origin="mcp",
          server_id=transport_server_id,
        )
        if (
          physical_provenance[0] != generation.physical_definition
          or advertised_projection != generation.physical_definition
        ):
          raise ValueError(
            "MCP physical definition diverges from alias generation"
          )
        physical_definition = physical_provenance[0].materialize()
        logical_definition = originated_definition.materialize()
        for route_field in ("name", "description"):
          physical_definition.pop(route_field, None)
          logical_definition.pop(route_field, None)
        if logical_definition != physical_definition:
          raise ValueError("MCP logical tool schema diverges from its transport")

        surface_matches = [
          surface_definition
          for surface_definition in self._tool_definitions
          if isinstance(surface_definition, Mapping)
          and surface_definition.get("name") == exposed_name
        ]
        if len(surface_matches) != 1:
          raise ValueError(
            "MCP logical tool requires unique surface provenance"
          )
        surface_definition = OriginatedToolDefinition(
          definition=surface_matches[0],
          origin="mcp",
          server_id=server_name,
        )
        if surface_definition != generation.logical_definition:
          raise ValueError("MCP logical tool definition diverges from surface")

        append_binding(LiveToolRouteBinding(
          originated_definition=originated_definition,
          route_kind="logical",
          logical_name=exposed_name,
          transport_server_id=transport_server_id,
          provider_original_name=provider_original_name,
          provider_id=self._provider_ids_by_server.get(transport_server_id),
        ))
    return tuple(bindings)

  def get_server_names(self) -> Set[str]:
    logical_servers = {
      logical_server
      for logical_server, physical_server in self._logical_server_routes.items()
      if physical_server in self._servers and self._logical_tool_definitions.get(logical_server)
    }
    physical_servers = {
      server_name
      for server_name in self._servers
      if not self._is_transport_only_server(server_name)
    }
    return physical_servers | logical_servers

  def get_exported_server_tool_names(self) -> Dict[str, Set[str]]:
    """Return per-server ListTools exports before registration or catalog merge."""
    exported: Dict[str, Set[str]] = {}
    transport_only_servers = self._transport_only_server_names()
    for server_name, state in self._servers.items():
      if server_name in transport_only_servers:
        continue
      exported[server_name] = {
        f"{state.tool_prefix}{tool_name}" if state.tool_prefix else tool_name
        for tool_name in state.exported_tool_names or ()
      }

    for logical_server, physical_server in self._logical_server_routes.items():
      state = self._servers.get(physical_server)
      if state is None:
        continue
      original_names = set(state.exported_tool_names or ())
      aliases = self._logical_tool_aliases.get(logical_server, {})
      if physical_server in transport_only_servers:
        aliases_by_original = {
          original_name: alias_name
          for alias_name, original_name in aliases.items()
        }
        logical_names = {
          aliases_by_original.get(original_name, original_name)
          for original_name in original_names
        }
      elif aliases:
        logical_names = {
          alias_name
          for alias_name, original_name in aliases.items()
          if original_name in original_names
        }
      else:
        continue
      exported[logical_server] = logical_names
    return exported

  def get_server_catalog(self) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for server_name, state in self._servers.items():
      if self._is_transport_only_server(server_name):
        continue
      tool_names = sorted(tool["name"] for tool in state.published_tool_definitions if isinstance(tool.get("name"), str))
      catalog[server_name] = {
        "tool_count": len(tool_names),
        "tools": tool_names,
      }
    for server_name, definitions in self._logical_tool_definitions.items():
      if self._logical_server_routes.get(server_name) not in self._servers:
        continue
      tool_names = sorted(
        tool["name"]
        for tool in definitions
        if isinstance(tool.get("name"), str)
      )
      catalog[server_name] = {
        "tool_count": len(tool_names),
        "tools": tool_names,
      }
    return catalog

  def get_startup_diagnostics(self) -> Dict[str, Dict[str, Any]]:
    return copy.deepcopy(self._startup_diagnostics)

  def is_mcp_tool(self, name: str) -> bool:
    if self._tool_registration_catalog is not None:
      return name in self._registered_mcp_descriptors_by_exposed_name
    return name in self._mcp_tool_names

  def uses_registered_tool_catalog(self) -> bool:
    """Return whether live MCP routes require exact registered semantics."""

    return self._tool_registration_catalog is not None

  def get_server_for_tool(self, name: str) -> str | None:
    if self._tool_registration_catalog is not None:
      descriptor = self._registered_mcp_descriptors_by_exposed_name.get(name)
      return (
        descriptor.live_binding.logical_server_id
        if descriptor is not None
        else None
      )
    return self._tool_to_server.get(name)

  def get_policy_tool_name(self, name: str) -> str | None:
    """Return the exact logical policy name attested by the live route binding."""

    server_name = self._tool_to_server.get(name)
    if server_name is None:
      return None
    try:
      matches = tuple(
        binding.logical_name
        for binding in self.get_server_tool_route_bindings({server_name})
        if binding.exposed_name == name
      )
    except (TypeError, ValueError):
      return None
    return matches[0] if len(matches) == 1 else None

  def get_provider_id_for_tool(self, name: str) -> str | None:
    """Return the trusted provider selected by this tool's transport route."""
    if name.startswith("mcp__"):
      parts = name.split("__", 2)
      if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
      logical_server = self._canonical_server_name(parts[1])
      if self._is_transport_only_server(logical_server):
        return None
      exposed_name = parts[2]
      catalog_owner = self._tool_to_server.get(exposed_name)
      if catalog_owner is not None and catalog_owner != logical_server:
        return None
      logical_aliases = self._logical_tool_aliases.get(logical_server)
      if (
        catalog_owner is None
        and logical_aliases is not None
        and exposed_name not in logical_aliases
      ):
        return None
      transport_server = self._transport_server_name(logical_server)
      return self._provider_ids_by_server.get(transport_server)

    logical_server = self._tool_to_server.get(name)
    if logical_server is None:
      return None
    transport_server = self._transport_server_name(logical_server)
    return self._provider_ids_by_server.get(transport_server)

  def is_per_user_server(self, server_name: str) -> bool:
    state = self._servers.get(self._transport_server_name(server_name))
    config = getattr(state, "config", None)
    return bool(config and config.get("per_user") is True)

  @staticmethod
  def _declared_per_user_env(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("per_user_env")
    if raw is None:
      return ()
    if (
      not isinstance(raw, list)
      or not raw
      or any(not isinstance(name, str) or not name.strip() for name in raw)
    ):
      raise ValueError("per_user_env must be a non-empty list of environment names")
    names = tuple(name.strip() for name in raw)
    if len(set(names)) != len(names):
      raise ValueError("per_user_env must not contain duplicate environment names")
    if config.get("per_user") is not True:
      raise ValueError("per_user_env requires per_user=true")
    return names

  def _prepare_server_config_for_startup(
    self,
    config: Mapping[str, Any],
  ) -> Dict[str, Any]:
    prepared = copy.deepcopy(dict(config))
    allowed_tools = _config_helpers.parse_allowed_tools(
      prepared.get("allowed_tools")
    )
    if allowed_tools is not None:
      prepared["allowed_tools"] = list(allowed_tools)
    dynamic_env_names = self._declared_per_user_env(prepared)
    if not dynamic_env_names:
      return prepared
    env = dict(prepared.get("env") or {})
    for name in dynamic_env_names:
      env.pop(name, None)
    prepared["env"] = env
    return prepared

  def _resolve_per_user_env(
    self,
    server_name: str,
    subject: _PerUserGatewaySubject,
    config: Mapping[str, Any],
  ) -> tuple[Dict[str, str], bytes] | None:
    env_names = self._declared_per_user_env(config)
    if not env_names:
      return None
    if self._per_user_env_resolver is None:
      raise _PerUserMcpError(
        "mcp_user_authority_unavailable",
        "User-scoped MCP authority is not configured.",
      )
    try:
      resolved = self._per_user_env_resolver(
        server_name,
        subject.user_id,
        subject.user_email,
      )
    except Exception as exc:
      raise _PerUserMcpError(
        "mcp_user_authority_unavailable",
        "User-scoped MCP authority could not be resolved.",
      ) from exc
    if not isinstance(resolved, Mapping):
      raise _PerUserMcpError(
        "mcp_user_authority_unavailable",
        "User-scoped MCP authority is invalid.",
      )
    projection = {
      str(name): str(value).strip()
      for name, value in resolved.items()
      if isinstance(name, str) and isinstance(value, str)
    }
    if set(projection) != set(env_names) or any(not projection[name] for name in env_names):
      raise _PerUserMcpError(
        "mcp_user_authority_unavailable",
        "User-scoped MCP authority is incomplete.",
      )
    canonical = json.dumps(
      {name: projection[name] for name in sorted(env_names)},
      sort_keys=True,
      separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hmac.new(
      self._per_user_binding_hmac_key,
      canonical,
      hashlib.sha256,
    ).digest()
    return projection, fingerprint

  @staticmethod
  def _canonical_broker_body(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

  async def _mint_gsheets_broker_session(
    self,
    subject: _PerUserGatewaySubject,
  ) -> tuple[str, float, str]:
    hmac_key = os.environ.get("GATEWAY_GOOGLE_SHEETS_BROKER_HMAC_KEY", "").strip()
    base_url = os.environ.get("GOOGLE_SHEETS_BROKER_URL", "").strip().rstrip("/")
    if not hmac_key or not base_url:
      raise _PerUserMcpError("sheets_unavailable", "Google Sheets broker is not configured")
    timestamp = int(time.time())
    payload = {
      "user_id": subject.user_id,
      "scopes": [GSHEETS_BROKER_SCOPE],
      "request_id": uuid.uuid4().hex,
      "ttl_s": PER_USER_SESSION_TTL_SECONDS,
    }
    message = str(timestamp).encode("ascii") + b"\n" + self._canonical_broker_body(payload)
    signature = hmac.new(hmac_key.encode("utf-8"), message, hashlib.sha256).hexdigest()
    try:
      async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
          f"{base_url}/api/internal/google/sheets-broker-session",
          json=payload,
          headers={
            "X-Resolver-Timestamp": str(timestamp),
            "X-Resolver-Signature": signature,
          },
        )
    except Exception as exc:
      raise _PerUserMcpError("sheets_unavailable", "Google Sheets broker is unavailable") from exc
    try:
      body = response.json()
    except Exception:
      body = {}
    error_code = str(body.get("error") or "") if isinstance(body, dict) else ""
    if response.status_code == 404 and error_code == "sheets_not_connected":
      raise _PerUserMcpError("sheets_not_connected", "Connect Google Sheets before using this tool")
    if response.status_code != 200:
      unavailable_code = error_code if error_code in {"broker_rate_limited", "replay_rejected"} else "sheets_unavailable"
      raise _PerUserMcpError(unavailable_code, "Google Sheets is temporarily unavailable")
    token = body.get("session_token") if isinstance(body, dict) else None
    expires_at = body.get("expires_at") if isinstance(body, dict) else None
    if not isinstance(token, str) or not token or not isinstance(expires_at, (int, float)):
      raise _PerUserMcpError("sheets_unavailable", "Google Sheets broker returned an invalid response")
    return token, float(expires_at), base_url

  async def _spawn_per_user_server(
    self,
    server_name: str,
    subject: _PerUserGatewaySubject,
    broker_session: tuple[str, float, str] | None = None,
    projected_env: tuple[Dict[str, str], bytes] | None = None,
  ) -> _PerUserServerState:
    definition = self._servers.get(server_name)
    if definition is None or not definition.config:
      raise _PerUserMcpError("sheets_unavailable", f"MCP server unavailable: {server_name}")
    config = copy.deepcopy(definition.config)
    env = dict(config.get("env") or {})
    binding_fingerprint: bytes | None = None
    if projected_env is not None:
      projection, binding_fingerprint = projected_env
      env.update(projection)
      expires_at = float("inf")
    else:
      if broker_session is None:
        broker_session = await self._mint_gsheets_broker_session(subject)
      token, expires_at, broker_url = broker_session
      env.update({
        "GSHEETS_TOKEN_MODE": "broker",
        "GSHEETS_HEADLESS": "1",
        "GSHEETS_BROKER_URL": broker_url,
        "GSHEETS_BROKER_SESSION_TOKEN": token,
      })
    config["env"] = env
    # The tier-1 credential is read only by this gateway process and is never
    # copied into the child config/environment.
    host = _PerUserChildHost(server_name, subject.user_id)
    task = host.start(self._host_per_user_child(host, config))
    self._per_user_hosts.add(host)
    task.add_done_callback(lambda _task: self._per_user_hosts.discard(host))
    try:
      state = await asyncio.shield(host.ready)
    except BaseException:
      # The child belongs to its host from the moment the host task starts,
      # so an abandoned spawn is closed by the host and not by this caller.
      host.request_close()
      self._hold_spawn_slot_until_closed(host)
      raise
    return _PerUserServerState(
      state,
      expires_at,
      time.time(),
      binding_fingerprint=binding_fingerprint,
      host=host,
    )

  def _hold_spawn_slot_until_closed(self, host: _PerUserChildHost) -> None:
    """Keep an abandoned child's instance-cap slot until its host lets it go.

    The caller that asked for the child releases its own reservation as it
    unwinds, but the child outlives it by however long its startup takes to
    notice. A slot that is free while the process is still running is a slot
    the cap can hand out twice.
    """
    task = host.task
    assert task is not None
    if task.done():
      # The startup failed rather than being abandoned; there is no child.
      return
    server_name = host.server_name
    self._per_user_spawn_reservations[server_name] = (
      self._per_user_spawn_reservations.get(server_name, 0) + 1
    )

    def _release(_task: "asyncio.Task[None]") -> None:
      remaining = self._per_user_spawn_reservations.get(server_name, 0) - 1
      if remaining > 0:
        self._per_user_spawn_reservations[server_name] = remaining
      else:
        self._per_user_spawn_reservations.pop(server_name, None)

    task.add_done_callback(_release)

  async def _host_per_user_child(
    self,
    host: _PerUserChildHost,
    config: Dict[str, Any],
  ) -> None:
    try:
      state = await self._connect_stdio_with_retries(f"{host.server_name}[user]", config)
    except BaseException as exc:
      if not host.ready.done():
        host.ready.set_exception(exc)
      return
    if host.close_requested.is_set() or host.ready.done():
      # Retired, or abandoned, before it was ever published. A child whose
      # close has been asked for is never handed to a caller.
      host.close_requested.set()
      if not host.ready.done():
        host.ready.cancel()
    else:
      host.ready.set_result(state)
    try:
      await host.close_requested.wait()
    finally:
      await self._close_contexts(state.exit_contexts)

  async def _close_per_user_when_drained(
    self,
    state: _PerUserServerState,
    reason: str,
  ) -> None:
    # The state is already out of `_per_user_servers`, so no further caller can
    # obtain it and `active_calls` falls to zero on its own. There is no deadline:
    # a tool call carries its own timeout, and closing under one strands it.
    started = time.monotonic()
    while state.active_calls:
      await asyncio.sleep(PER_USER_DRAIN_POLL_SECONDS)
    host = state.host
    assert host is not None
    log.info(
      "per-user MCP child %s closed | user=%s site=%s drained_in=%.2fs",
      host.server_name,
      host.user_id,
      reason,
      time.monotonic() - started,
    )
    await host.close()

  def _schedule_drain(self, state: _PerUserServerState, reason: str) -> None:
    if state.draining:
      return
    state.draining = True
    task = asyncio.create_task(self._close_per_user_when_drained(state, reason))
    self._drain_tasks.add(task)
    task.add_done_callback(self._drain_tasks.discard)

  def _retire_per_user_spawn_lock(self, key: tuple[str, str]) -> None:
    lock = self._per_user_spawn_locks.get(key)
    waiters = getattr(lock, "_waiters", None) if lock is not None else None
    if (
      key not in self._per_user_servers
      and lock is not None
      and not lock.locked()
      and not waiters
    ):
      self._per_user_spawn_locks.pop(key, None)

  def _reap_idle_per_user_servers(self, now: float) -> None:
    for key, state in list(self._per_user_servers.items()):
      if state.active_calls == 0 and now - state.last_used_at > PER_USER_IDLE_REAP_SECONDS:
        if self._per_user_servers.pop(key, None) is state:
          self._schedule_drain(state, "idle_reap")
          self._retire_per_user_spawn_lock(key)

  async def _run_per_user_reaper(self) -> None:
    while True:
      await asyncio.sleep(PER_USER_REAPER_INTERVAL_SECONDS)
      self._reap_idle_per_user_servers(time.time())

  def _ensure_per_user_reaper(self) -> None:
    if self._per_user_reaper_task is None or self._per_user_reaper_task.done():
      self._per_user_reaper_task = asyncio.create_task(self._run_per_user_reaper())

  async def _get_per_user_server(
    self,
    server_name: str,
    subject: _PerUserGatewaySubject,
    *,
    force: bool = False,
    discard_current_on_failure: bool = False,
  ) -> _PerUserServerState:
    key = (server_name, subject.user_id)
    lock = self._per_user_spawn_locks.setdefault(key, asyncio.Lock())
    try:
      async with lock:
        now = time.time()
        definition = self._servers.get(server_name)
        if definition is None or not definition.config:
          raise _PerUserMcpError(
            "mcp_user_authority_unavailable",
            f"MCP server unavailable: {server_name}",
          )
        projected_env = self._resolve_per_user_env(
          server_name,
          subject,
          definition.config,
        )
        binding_fingerprint = projected_env[1] if projected_env is not None else None
        current = self._per_user_servers.get(key)
        replacement_reason = "first_spawn"
        if current is not None:
          if current.draining:
            replacement_reason = "already_draining"
          elif not current.server.exit_contexts:
            replacement_reason = "dead_transport"
          elif force:
            replacement_reason = "forced_refresh"
          elif current.expires_at - now <= PER_USER_EXPIRY_MARGIN_SECONDS:
            replacement_reason = "near_expiry"
          elif current.binding_fingerprint != binding_fingerprint:
            replacement_reason = "binding_changed"
          else:
            current.last_used_at = now
            return current

        # Mint before changing capacity accounting or evicting a healthy child.
        # A forced broker-expiry replacement is the exception: the current
        # child is known invalid and must not remain cached when minting fails.
        broker_session = None
        if projected_env is None:
          try:
            broker_session = await self._mint_gsheets_broker_session(subject)
          except BaseException:
            if force and discard_current_on_failure:
              expired = self._per_user_servers.pop(key, None)
              if expired is not None:
                self._schedule_drain(expired, "broker_mint_failed")
            raise
        current = self._per_user_servers.get(key)
        old_state = None
        if current is not None:
          old_state = self._per_user_servers.pop(key, None)
        else:
          server_count = sum(
            candidate_server == server_name
            for candidate_server, _candidate_user in self._per_user_servers
          )
          server_count += self._per_user_spawn_reservations.get(server_name, 0)
          if server_count >= PER_USER_INSTANCE_CAP:
            idle = [
              (candidate.last_used_at, candidate_key, candidate)
              for candidate_key, candidate in self._per_user_servers.items()
              if candidate_key[0] == server_name and candidate.active_calls == 0
            ]
            if not idle:
              projected_authority = projected_env is not None
              raise _PerUserMcpError(
                (
                  "mcp_user_authority_unavailable"
                  if projected_authority
                  else "sheets_unavailable"
                ),
                (
                  "Per-user MCP instance capacity reached"
                  if projected_authority
                  else "Google Sheets per-user instance capacity reached"
                ),
              )
            _, evict_key, evicted = min(idle)
            self._per_user_servers.pop(evict_key, None)
            self._schedule_drain(evicted, "instance_cap_eviction")
            self._retire_per_user_spawn_lock(evict_key)
        self._per_user_spawn_reservations[server_name] = (
          self._per_user_spawn_reservations.get(server_name, 0) + 1
        )
        try:
          spawn_kwargs: Dict[str, Any] = {"broker_session": broker_session}
          if projected_env is not None:
            spawn_kwargs["projected_env"] = projected_env
          replacement = await self._spawn_per_user_server(
            server_name,
            subject,
            **spawn_kwargs,
          )
        except BaseException:
          if (
            not discard_current_on_failure
            and old_state is not None
            and not old_state.draining
            and bool(old_state.server.exit_contexts)
          ):
            self._per_user_servers[key] = old_state
          elif old_state is not None:
            self._schedule_drain(old_state, "replacement_spawn_failed")
          raise
        else:
          self._per_user_servers[key] = replacement
        finally:
          remaining = self._per_user_spawn_reservations.get(server_name, 0) - 1
          if remaining > 0:
            self._per_user_spawn_reservations[server_name] = remaining
          else:
            self._per_user_spawn_reservations.pop(server_name, None)
        self._ensure_per_user_reaper()
        if old_state is not None and old_state is not replacement:
          self._schedule_drain(old_state, replacement_reason)
        return replacement
    finally:
      self._retire_per_user_spawn_lock(key)

  def get_original_tool_name(self, name: str) -> str:
    return self._dispatch_to_original.get(
      name,
      self._prefixed_to_original.get(name, name),
    )

  def resolve_tool_name(self, server_name: str, original_name: str) -> str | None:
    """Return the exposed tool name for a server-owned tool."""
    server_name = self._canonical_server_name(server_name)
    if server_name in self._logical_server_routes:
      for exposed_name, owner in self._tool_to_server.items():
        if owner == server_name and self.get_original_tool_name(exposed_name) == original_name:
          return exposed_name
      return None
    state = self._servers.get(server_name)
    if state is None:
      return None
    exposed_name = f"{state.tool_prefix}{original_name}" if state.tool_prefix else original_name
    return exposed_name if self._tool_to_server.get(exposed_name) == server_name else None

  def _translate_provider_symbol(
    self,
    logical_server_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
  ) -> Dict[str, Any]:
    try:
      route = self._input_preparation_routes.get(
        (logical_server_id, tool_name)
      )
      if route is None:
        return tool_input

      resolver = self._provider_symbol_resolver
      if resolver is None:
        return tool_input

      def translate(value: Any) -> Any:
        return resolver(value) or value

      if route.mode == "consistent-present-keys":
        translated = dict(tool_input)
        present_keys = [key for key in route.keys if key in translated]
        translated_values = {key: translate(translated[key]) for key in present_keys}
        if (
          len(present_keys) == len(route.keys)
          and translated_values[route.keys[0]] != translated_values[route.keys[1]]
        ):
          return translated
        for key, value in translated_values.items():
          translated[key] = value
        return translated

      if route.mode == "scalar":
        translated = dict(tool_input)
        for key in route.keys:
          if key in translated:
            translated[key] = translate(translated[key])
        return translated

      if route.mode == "comma-separated" and len(route.keys) == 1:
        translated = dict(tool_input)
        comma_key = route.keys[0]
        value = translated.get(comma_key)
        if isinstance(value, str):
          translated[comma_key] = ",".join(str(translate(token)) for token in value.split(","))
        return translated

      return tool_input
    except Exception:
      return tool_input

  @staticmethod
  def _policy_tool_class(server_name: str, tool_name: str) -> str | None:
    fallback_class = None
    if server_name == _GSHEETS_SERVER_NAME:
      if tool_name in _GSHEETS_BROKER_READ_TOOLS:
        fallback_class = "read"
      elif tool_name in _GSHEETS_BROKER_WRITE_TOOLS:
        fallback_class = "external_write"
    try:
      _forbidden, _owner, get_tool_class = load_server_policy_helpers()
      if get_tool_class is None:
        return fallback_class
      value = get_tool_class(server_name, tool_name)
      return str(value) if value is not None else fallback_class
    except Exception as exc:
      log.error(
        "Unable to resolve MCP policy class for %s.%s: %s",
        server_name,
        tool_name,
        type(exc).__name__,
      )
      return fallback_class

  async def call_tool(
    self,
    name: str,
    tool_input: Dict[str, Any] | PreparedToolCall,
    meta: Dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: Any | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: Mapping[str, object] | None = None,
  ) -> Tuple[Any | None, Dict[str, Any] | None]:
    if type(allow_uncertain_replay) is not bool:
      raise TypeError("allow_uncertain_replay must be an exact bool")
    registered_descriptor = (
      self._registered_mcp_descriptors_by_exposed_name.get(name)
      if self._tool_registration_catalog is not None
      else None
    )
    effective_allow_uncertain_replay = (
      allow_uncertain_replay
      and (
        registered_descriptor is None
        or (
          registered_descriptor.declaration.semantics.idempotent
          and registered_descriptor.declaration.semantics.effect
          in _AUTOMATIC_REPLAY_EFFECTS
        )
      )
    )
    server_name = self._tool_to_server.get(name)
    if not server_name:
      return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

    if _is_exact_prepared_tool_call(tool_input):
      if registered_descriptor is None:
        return None, {
          "code": "tool_input_preparation_failed",
          "message": "Prepared MCP input requires an exact registered route.",
        }
      effective_input = tool_input.materialize_input()
    elif registered_descriptor is not None:
      try:
        effective_input = self.prepare_registered_tool_input(
          name,
          tool_input,
          trusted_dispatch_scope,
        ).materialize_input()
      except Exception as exc:
        log.error(
          "Registered MCP input preparation failed for %s | exception_type=%s",
          name,
          type(exc).__name__,
        )
        return None, {
          "code": "tool_input_preparation_failed",
          "message": f"Tool '{name}' input could not be prepared for dispatch.",
        }
    else:
      try:
        policy_name = self.get_policy_tool_name(name)
      except Exception:
        policy_name = None
      effective_input = self._translate_provider_symbol(
        server_name,
        policy_name or "",
        tool_input,
      )

    original_name = self.get_original_tool_name(name)
    is_sheets = server_name == _GSHEETS_SERVER_NAME
    policy_class = (
      registered_descriptor.declaration.semantics.effect
      if is_sheets and registered_descriptor is not None
      else self._policy_tool_class(server_name, original_name)
      if is_sheets
      else None
    )
    sheets_is_read_only = (
      registered_descriptor.declaration.semantics.effect
      in _AUTOMATIC_REPLAY_EFFECTS
      if registered_descriptor is not None
      else policy_class in _READ_ONLY_POLICY_CLASSES
    )
    sheets_is_mutation = is_sheets and not sheets_is_read_only

    transport_server_name = self._transport_server_name(server_name)
    server = self._servers.get(transport_server_name)
    if not server:
      if is_sheets:
        payload = _gateway_sheets_error_payload(
          original_name,
          code="sheets_unavailable",
          message="Google Sheets is unavailable before the request was dispatched.",
          outcome_state="not_started",
          phase="gateway_startup",
          mutation_may_have_occurred=False,
          retry_safe=True,
          retry_automatic=False,
          retry_action="retry",
        )
        return None, _sheets_gateway_error(payload)
      return None, {
        "code": "mcp_tool_error",
        "message": f"MCP server unavailable: {server_name}",
      }
    per_user_state: _PerUserServerState | None = None
    per_user_subject: _PerUserGatewaySubject | None = None
    if self.is_per_user_server(transport_server_name):
      try:
        per_user_subject = _PerUserGatewaySubject.from_gateway_session(
          gateway_session
        )
      except ValueError:
        if not is_sheets:
          return None, {
            "code": "mcp_tool_error",
            "sub_code": "missing_user_identity",
            "message": "User-scoped MCP requires an authenticated user identity.",
          }
        payload = _gateway_sheets_error_payload(
          original_name,
          code="missing_user_identity",
          message="Google Sheets requires an authenticated user identity.",
          outcome_state="not_started",
          phase="gateway_identity",
          mutation_may_have_occurred=False,
          retry_safe=True,
          retry_automatic=False,
          retry_action="authenticate_user",
        )
        return None, _sheets_gateway_error(payload)
      try:
        per_user_state = await self._get_per_user_server(
          transport_server_name,
          per_user_subject,
        )
      except _PerUserMcpError as exc:
        if not is_sheets:
          return None, {
            "code": "mcp_tool_error",
            "sub_code": exc.code,
            "message": str(exc),
          }
        action = "connect_sheets" if exc.code == "sheets_not_connected" else "retry"
        payload = _gateway_sheets_error_payload(
          original_name,
          code=exc.code,
          message=str(exc),
          outcome_state="not_started",
          phase="gateway_startup",
          mutation_may_have_occurred=False,
          retry_safe=True,
          retry_automatic=False,
          retry_action=action,
        )
        return None, _sheets_gateway_error(payload)
      except Exception:
        if not is_sheets:
          return None, {
            "code": "mcp_tool_error",
            "sub_code": "mcp_user_authority_unavailable",
            "message": "User-scoped MCP could not be started before dispatch.",
          }
        payload = _gateway_sheets_error_payload(
          original_name,
          code="sheets_unavailable",
          message="Google Sheets could not be started before the request was dispatched.",
          outcome_state="not_started",
          phase="gateway_startup",
          mutation_may_have_occurred=False,
          retry_safe=True,
          retry_automatic=False,
          retry_action="retry",
        )
        return None, _sheets_gateway_error(payload)
      server = per_user_state.server
    timeout_seconds = self._timeout_for_tool(transport_server_name, name, original_name)
    try:
      if per_user_state is not None:
        per_user_state.active_calls += 1
        per_user_state.last_used_at = time.time()
      result = await self._call_tool_once(
        server=server,
        original_name=original_name,
        tool_input=effective_input,
        meta=meta,
        abort_event=abort_event,
        timeout_seconds=timeout_seconds,
      )
    except Exception as exc:
      if per_user_state is not None:
        assert per_user_subject is not None
        key = (transport_server_name, per_user_subject.user_id)
        if self._per_user_servers.get(key) is per_user_state:
          self._per_user_servers.pop(key, None)
          self._retire_per_user_spawn_lock(key)
        self._schedule_drain(per_user_state, "dispatch_transport_failure")
        if is_sheets:
          transport_failure = _is_sheets_transport_failure(exc)
          payload = _gateway_sheets_error_payload(
            original_name,
            code=(
              "mutation_outcome_uncertain"
              if sheets_is_mutation
              else ("sheets_transport_error" if transport_failure else "sheets_internal_error")
            ),
            message=(
              "Google Sheets did not confirm the dispatched mutation; its outcome is uncertain."
              if sheets_is_mutation
              else (
                "The Google Sheets connection was lost before a read result was received."
                if transport_failure
                else "Google Sheets could not complete the read."
              )
            ),
            outcome_state=("uncertain" if sheets_is_mutation else "unchanged"),
            phase="dispatch",
            mutation_may_have_occurred=sheets_is_mutation,
            retry_safe=bool(not sheets_is_mutation and transport_failure),
            retry_automatic=False,
            retry_action=(
              "inspect_spreadsheet"
              if sheets_is_mutation
              else ("retry" if transport_failure else "report_incident")
            ),
          )
          return None, _sheets_gateway_error(payload)
        return None, _tool_error_from_exception(exc)
      if is_sheets:
        transport_failure = _is_sheets_transport_failure(exc)
        if transport_failure:
          await self._reconnect_stdio_server_for_future(
            server_name=server_name,
            server=server,
            original_name=original_name,
            cause=exc,
          )
        payload = _gateway_sheets_error_payload(
          original_name,
          code=(
            "mutation_outcome_uncertain"
            if sheets_is_mutation
            else ("sheets_transport_error" if transport_failure else "sheets_internal_error")
          ),
          message=(
            "Google Sheets did not confirm the dispatched mutation; its outcome is uncertain."
            if sheets_is_mutation
            else (
              "The Google Sheets connection was lost before a read result was received."
              if transport_failure
              else "Google Sheets could not complete the read."
            )
          ),
          outcome_state=("uncertain" if sheets_is_mutation else "unchanged"),
          phase="dispatch",
          mutation_may_have_occurred=sheets_is_mutation,
          retry_safe=bool(not sheets_is_mutation and transport_failure),
          retry_automatic=False,
          retry_action=(
            "inspect_spreadsheet"
            if sheets_is_mutation
            else ("retry" if transport_failure else "report_incident")
          ),
        )
        return None, _sheets_gateway_error(payload)
      if not effective_allow_uncertain_replay:
        await self._reconnect_stdio_server_for_future(
          server_name=transport_server_name,
          server=server,
          original_name=original_name,
          cause=exc,
        )
        return None, _tool_error_from_exception(exc)
      try:
        retry_result = await self._retry_stdio_tool_call_after_reconnect(
          server_name=transport_server_name,
          server=server,
          original_name=original_name,
          tool_input=effective_input,
          meta=meta,
          abort_event=abort_event,
          timeout_seconds=timeout_seconds,
          cause=exc,
        )
      except Exception as retry_exc:
        return None, _tool_error_from_exception(retry_exc)
      if retry_result is not None:
        result = retry_result
      else:
        return None, _tool_error_from_exception(exc)
    except asyncio.CancelledError:
      raise
    finally:
      if per_user_state is not None:
        per_user_state.active_calls = max(0, per_user_state.active_calls - 1)
        per_user_state.last_used_at = time.time()

    sheets_error = (
      _sheets_structured_error(result, expected_operation=original_name)
      if is_sheets
      else None
    )
    if (
      per_user_state is not None
      and sheets_error is not None
      and sheets_error["error"]["code"] == "broker_session_expired"
    ):
      replay_safe = _sheets_error_allows_automatic_read_retry(sheets_error, policy_class)
      assert per_user_subject is not None
      try:
        replacement = await self._get_per_user_server(
          transport_server_name,
          per_user_subject,
          force=True,
          discard_current_on_failure=True,
        )
      except _PerUserMcpError:
        return None, _sheets_gateway_error(sheets_error)
      except Exception:
        return None, _sheets_gateway_error(sheets_error)

      if replay_safe and effective_allow_uncertain_replay:
        replacement.active_calls += 1
        try:
          result = await self._call_tool_once(
            server=replacement.server,
            original_name=original_name,
            tool_input=effective_input,
            meta=meta,
            abort_event=abort_event,
            timeout_seconds=timeout_seconds,
          )
        except Exception as exc:
          key = (server_name, per_user_subject.user_id)
          if self._per_user_servers.get(key) is replacement:
            self._per_user_servers.pop(key, None)
            self._retire_per_user_spawn_lock(key)
          self._schedule_drain(replacement, "sheets_replay_transport_failure")
          payload = _gateway_sheets_error_payload(
            original_name,
            code="sheets_transport_error",
            message="The Google Sheets connection was lost before a read result was received.",
            outcome_state="unchanged",
            phase="dispatch",
            mutation_may_have_occurred=False,
            retry_safe=True,
            retry_automatic=False,
            retry_action="retry",
          )
          if not _is_sheets_transport_failure(exc):
            payload["error"]["code"] = "sheets_unavailable"
            payload["error"]["message"] = "Google Sheets could not complete the retried read."
          return None, _sheets_gateway_error(payload)
        finally:
          replacement.active_calls = max(0, replacement.active_calls - 1)
          replacement.last_used_at = time.time()
        sheets_error = _sheets_structured_error(
          result,
          expected_operation=original_name,
        )
        if (
          sheets_error is not None
          and sheets_error["error"]["code"] == "broker_session_expired"
        ):
          key = (server_name, per_user_subject.user_id)
          if self._per_user_servers.get(key) is replacement:
            self._per_user_servers.pop(key, None)
            self._retire_per_user_spawn_lock(key)
          self._schedule_drain(replacement, "broker_session_expired")

    if is_sheets and sheets_error is not None:
      return None, _sheets_gateway_error(sheets_error)

    if result.is_error:
      if is_sheets:
        payload = _gateway_sheets_error_payload(
          original_name,
          code="invalid_sheets_error_contract",
          message="Google Sheets returned an invalid structured error.",
          outcome_state=("uncertain" if sheets_is_mutation else "unchanged"),
          phase="dispatch",
          mutation_may_have_occurred=sheets_is_mutation,
          retry_safe=False,
          retry_automatic=False,
          retry_action=("inspect_spreadsheet" if sheets_is_mutation else "retry"),
        )
        return None, _sheets_gateway_error(payload)
      message = self._result_message(result)
      return None, {
        "code": "mcp_tool_error",
        "sub_code": _classify_mcp_error(message or ""),
        "message": message or f"MCP tool failed: {name}",
      }

    if is_sheets:
      if (
        isinstance(result.structured_content, dict)
        and result.structured_content.get("status") == "ok"
        and result.structured_content.get("operation") == original_name
      ):
        return result.structured_content, None
      error_contract = bool(
        isinstance(result.structured_content, dict)
        and result.structured_content.get("status") == "error"
      )
      payload = _gateway_sheets_error_payload(
        original_name,
        code=("invalid_sheets_error_contract" if error_contract else "invalid_sheets_result_contract"),
        message=(
          "Google Sheets returned an invalid structured error."
          if error_contract
          else "Google Sheets returned no valid structured result."
        ),
        outcome_state=("uncertain" if sheets_is_mutation else "unchanged"),
        phase="dispatch",
        mutation_may_have_occurred=sheets_is_mutation,
        retry_safe=False,
        retry_automatic=False,
        retry_action=("inspect_spreadsheet" if sheets_is_mutation else "retry"),
      )
      return None, _sheets_gateway_error(payload)

    if result.structured_content is not None:
      return result.structured_content, None

    text_payload = self._extract_text(result.content)
    if text_payload:
      try:
        return json.loads(text_payload), None
      except json.JSONDecodeError:
        return {"text": text_payload}, None

    return {}, None

  async def _call_tool_once(
    self,
    *,
    server: _connection_helpers._McpCallableServerState,
    original_name: str,
    tool_input: Dict[str, Any],
    meta: Dict[str, Any] | None,
    abort_event: asyncio.Event | None,
    timeout_seconds: float,
  ) -> _connection_helpers.McpToolCallResult:
    if abort_event is not None and abort_event.is_set():
      raise asyncio.CancelledError()
    session = server.session
    call_kwargs: _McpCallKwargs = {
      "read_timeout_seconds": timeout_seconds,
    }
    if meta is not None:
      call_kwargs["meta"] = meta
    call_task = asyncio.create_task(session.call_tool(
      original_name,
      tool_input,
      **call_kwargs,
    ))
    abort_task: asyncio.Task[Any] | None = None
    try:
      wait_tasks: set[asyncio.Task[Any]] = {call_task}
      if abort_event is not None:
        abort_task = asyncio.create_task(abort_event.wait())
        wait_tasks.add(abort_task)
      timeout = max(0.0, float(timeout_seconds))
      done, _pending = await asyncio.wait(
        wait_tasks,
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
      )
      if abort_task in done and abort_event is not None and abort_event.is_set():
        await self._cancel_mcp_tool_call(
          call_task,
          tool_name=original_name,
          reason="abort",
        )
        raise asyncio.CancelledError()
      if call_task in done:
        return await call_task
      await self._cancel_mcp_tool_call(
        call_task,
        tool_name=original_name,
        reason="timeout",
      )
      raise asyncio.TimeoutError(
        f"MCP tool {original_name} timed out after {timeout:g}s"
      )
    except asyncio.CancelledError:
      await self._cancel_mcp_tool_call(
        call_task,
        tool_name=original_name,
        reason="caller_cancelled",
      )
      raise
    finally:
      if abort_task is not None:
        abort_task.cancel()

  @staticmethod
  async def _cancel_mcp_tool_call(
    task: asyncio.Task[Any],
    *,
    tool_name: str,
    reason: str,
  ) -> None:
    await _runtime_helpers.cancel_mcp_tool_call(
      task,
      tool_name=tool_name,
      reason=reason,
      grace_seconds=_MCP_TOOL_CANCEL_GRACE_SECONDS,
      consume_result=_consume_mcp_tool_call_result,
      current_task=asyncio.current_task,
      logger=log,
      shield=asyncio.shield,
      wait_for=asyncio.wait_for,
    )

  async def _retry_stdio_tool_call_after_reconnect(
    self,
    *,
    server_name: str,
    server: _ServerState,
    original_name: str,
    tool_input: Dict[str, Any],
    meta: Dict[str, Any] | None,
    abort_event: asyncio.Event | None,
    timeout_seconds: float,
    cause: Exception,
  ) -> _connection_helpers.McpToolCallResult | None:
    config = server.config
    if not config:
      return None
    server_type = str(config.get("type", "stdio")).strip().lower()
    if server_type != "stdio" or not _is_retryable_stdio_connect_error(cause):
      return None
    message = str(cause).strip() or type(cause).__name__
    log.warning(
      "MCP stdio server %s tool call %s failed with transient transport error; "
      "reconnecting once: %s",
      server_name,
      original_name,
      message,
    )
    try:
      retry_server = await self._replace_stdio_server(server_name, server)
    except Exception as reconnect_exc:
      reconnect_message = str(reconnect_exc).strip() or type(reconnect_exc).__name__
      log.warning(
        "MCP stdio server %s failed to reconnect after tool transport error: %s",
        server_name,
        reconnect_message,
      )
      async with self._lock:
        current = self._servers.get(server_name)
        retry_server = current if current is not server else None

    if retry_server is None:
      return None
    return await self._call_tool_once(
      server=retry_server,
      original_name=original_name,
      tool_input=tool_input,
      meta=meta,
      abort_event=abort_event,
      timeout_seconds=timeout_seconds,
    )

  async def _reconnect_stdio_server_for_future(
    self,
    *,
    server_name: str,
    server: _ServerState,
    original_name: str,
    cause: BaseException,
  ) -> bool:
    config = server.config
    if not config:
      return False
    server_type = str(config.get("type", "stdio")).strip().lower()
    if server_type != "stdio" or not _is_sheets_transport_failure(cause):
      return False
    log.warning(
      "MCP stdio server %s tool call %s lost transport; reconnecting for future calls only (%s)",
      server_name,
      original_name,
      type(cause).__name__,
    )
    try:
      return await self._replace_stdio_server(server_name, server) is not None
    except Exception as reconnect_exc:
      log.warning(
        "MCP stdio server %s could not reconnect for future calls (%s)",
        server_name,
        type(reconnect_exc).__name__,
      )
      return False

  async def _replace_stdio_server(
    self, server_name: str, server: _ServerState,
  ) -> _ServerState | None:
    # EOF and an in-flight call can observe the same disconnect. One owner
    # closes/spawns/publishes; the other uses the already published generation.
    async with server.reconnect_lock:
      async with self._lock:
        current = self._servers.get(server_name)
        if current is not server:
          return current
        config = server.config
        assert config is not None
        if (
          server.stdio_eof is not None
          and server.stdio_eof.is_set()
          and server.stdio_receive_done is not None
        ):
          # EOF wakes this owner before ClientSession has delivered connection
          # errors. Closing its task group now would strand the pending calls.
          await server.stdio_receive_done.wait()
        await self._close_server(server)
      replacement = await self._connect_stdio_with_retries(server_name, config)
      try:
        async with self._lock:
          await self._publish_server_states([replacement], replacing=server)
          return self._servers.get(server_name)
      except BaseException:
        await self._close_contexts(replacement.exit_contexts)
        raise

  async def _reconnect_stdio_on_eof(self, server: _ServerState) -> None:
    assert server.stdio_eof is not None
    await server.stdio_eof.wait()
    log.warning("MCP stdio server %s closed stdout; reconnecting", server.name)
    try:
      await self._replace_stdio_server(server.name, server)
    except Exception as exc:
      log.warning(
        "MCP stdio server %s could not reconnect after EOF (%s)",
        server.name, type(exc).__name__,
      )

  async def _close_server(self, server: _ServerState) -> None:
    watch = server.stdio_watch_task
    if watch is not None and watch is not asyncio.current_task():
      watch.cancel()
      await asyncio.gather(watch, return_exceptions=True)
    await self._close_contexts(server.exit_contexts)

  async def shutdown(self) -> None:
    async with self._lock:
      self._registered_mcp_descriptors_by_exposed_name = MappingProxyType({})
      if not self._started and not self._servers:
        return

      reaper_task = self._per_user_reaper_task
      self._per_user_reaper_task = None
      if reaper_task is not None:
        reaper_task.cancel()
        await asyncio.gather(reaper_task, return_exceptions=True)

      for server in reversed(list(self._servers.values())):
        await self._close_server(server)

      for task in list(self._drain_tasks):
        task.cancel()
      if self._drain_tasks:
        await asyncio.gather(*list(self._drain_tasks), return_exceptions=True)
      for host in list(self._per_user_hosts):
        log.info(
          "per-user MCP child %s closed | user=%s site=manager_shutdown",
          host.server_name,
          host.user_id,
        )
        await host.close()

      self._servers.clear()
      self._per_user_servers.clear()
      self._per_user_hosts.clear()
      self._per_user_spawn_locks.clear()
      self._per_user_spawn_reservations.clear()
      self._tool_definitions = []
      self._tool_to_server = {}
      self._prefixed_to_original = {}
      self._dispatch_to_original = {}
      self._mcp_tool_names = set()
      self._logical_tool_definitions = {}
      self._logical_alias_generation = ()
      self._startup_diagnostics = {}
      self._configured_transport_server_names = None
      self._started = False

  def _apply_collision_filtering(
    self,
    *,
    policy_server_for_tool: Callable[[str], str | None] | None = None,
  ) -> None:
    for state in self._servers.values():
      state.published_tool_definitions = state.tool_definitions
    self._registered_mcp_descriptors_by_exposed_name = MappingProxyType({})
    self._logical_alias_generation = ()
    if policy_server_for_tool is None:
      _get_forbidden_tools_for_session, get_server_for_policy_tool, _get_tool_class = load_server_policy_helpers()
      if get_server_for_policy_tool is None:
        log.warning(
          "Shared MCP policy module unavailable; enforcing the built-in Google Sheets broker surface and owner mapping"
        )
        self._prefilter_gsheets_without_shared_policy()
      else:
        policy_server_for_tool = get_server_for_policy_tool

    # Preserve the Google Sheets broker's built-in owner mapping when the
    # optional shared policy module cannot be imported during gateway startup.
    if policy_server_for_tool is None:
      def sheets_fallback_policy_server(tool_name: str) -> str | None:
        if tool_name in _GSHEETS_BROKER_TOOLS:
          return _GSHEETS_SERVER_NAME
        for logical_server, aliases in self._logical_tool_aliases.items():
          if tool_name in aliases:
            return logical_server
        return None

      policy_server_for_tool = sheets_fallback_policy_server

    if policy_server_for_tool is not None:
      self._prefilter_policy_owner_mismatches(
        policy_server_for_tool=policy_server_for_tool,
      )

    result = _catalog_helpers.apply_collision_filtering(
      servers=self._servers,
      builtin_tool_names=self._builtin_tool_names,
      strip_input_fields=self._strip_input_fields,
      logger=log,
    )
    self._tool_definitions = result.tool_definitions
    self._tool_to_server = result.tool_to_server
    self._prefixed_to_original = result.prefixed_to_original
    self._mcp_tool_names = result.mcp_tool_names
    if policy_server_for_tool is not None:
      self._apply_policy_owner_invariant(
        policy_server_for_tool=policy_server_for_tool,
      )
    alias_result = _catalog_helpers.add_logical_tool_aliases(
      tool_definitions=self._tool_definitions,
      tool_to_server=self._tool_to_server,
      prefixed_to_original=self._prefixed_to_original,
      mcp_tool_names=self._mcp_tool_names,
      logical_server_routes=self._logical_server_routes,
      logical_tool_aliases=self._logical_tool_aliases,
      policy_server_for_tool=policy_server_for_tool,
      transport_only_servers=self._transport_only_server_names(),
    )
    self._tool_definitions = alias_result.tool_definitions
    self._tool_to_server = alias_result.tool_to_server
    self._dispatch_to_original = alias_result.dispatch_to_original
    self._mcp_tool_names = alias_result.mcp_tool_names
    self._logical_tool_definitions = alias_result.logical_tool_definitions
    self._logical_alias_generation = alias_result.alias_generation
    self._refresh_registered_mcp_tool_descriptors()

  def _prefilter_gsheets_without_shared_policy(self) -> None:
    state = self._servers.get(_GSHEETS_SERVER_NAME)
    if state is None:
      return

    kept_tool_definitions = [
      tool_def
      for tool_def in state.published_tool_definitions
      if not str(tool_def.get("name") or "").strip()
      or str(tool_def.get("name") or "").strip() in _GSHEETS_BROKER_TOOLS
    ]
    if len(kept_tool_definitions) == len(state.published_tool_definitions):
      return

    state.published_tool_definitions = kept_tool_definitions
    state.tool_names = {
      f"{state.tool_prefix}{tool_def['name']}" if state.tool_prefix else tool_def["name"]
      for tool_def in kept_tool_definitions
      if isinstance(tool_def.get("name"), str)
    }


  def _prefilter_policy_owner_mismatches(
    self,
    *,
    policy_server_for_tool: Callable[[str], str | None],
  ) -> None:
    for server_name, state in self._servers.items():
      kept_tool_definitions: list[dict[str, Any]] = []
      mismatches: list[tuple[str, str]] = []
      for tool_def in state.published_tool_definitions:
        original_name = str(tool_def.get("name") or "").strip()
        if not original_name:
          kept_tool_definitions.append(tool_def)
          continue
        policy_server = policy_server_for_tool(original_name)
        policy_runtime_server = self._transport_server_name(policy_server) if policy_server else None
        if policy_server and policy_runtime_server != server_name:
          mismatches.append((original_name, policy_server))
          continue
        kept_tool_definitions.append(tool_def)

      if not mismatches:
        continue

      state.published_tool_definitions = kept_tool_definitions
      state.tool_names = {
        f"{state.tool_prefix}{tool_def['name']}" if state.tool_prefix else tool_def["name"]
        for tool_def in kept_tool_definitions
        if isinstance(tool_def.get("name"), str)
      }
      mismatch_summary = ", ".join(
        f"{original_name}->{policy_server}"
        for original_name, policy_server in mismatches
      )
      message = (
        "MCP runtime owner does not match gateway policy owner; "
        f"pre-filtering tools before catalog merge: {mismatch_summary}"
      )
      category = "policy_owner_mismatch"
      error_type = "PolicyOwnerMismatch"
      self._set_startup_diagnostic(
        server_name,
        category=category,
        message=message,
        retryable=False,
        error_type=error_type,
      )
      log.error("%s on runtime server %s", message, server_name)

  def _apply_policy_owner_invariant(
    self,
    *,
    policy_server_for_tool: Callable[[str], str | None] | None = None,
  ) -> None:
    resolved_policy_server_for_tool = policy_server_for_tool
    if resolved_policy_server_for_tool is None:
      _get_forbidden_tools_for_session, get_server_for_policy_tool, _get_tool_class = load_server_policy_helpers()
      if get_server_for_policy_tool is None:
        log.warning("Skipping MCP policy-owner invariant: server policy module unavailable")
        return
      resolved_policy_server_for_tool = get_server_for_policy_tool

    result = _policy_owner_helpers.apply_policy_owner_invariant(
      servers=self._servers,
      tool_definitions=self._tool_definitions,
      tool_to_server=self._tool_to_server,
      prefixed_to_original=self._prefixed_to_original,
      mcp_tool_names=self._mcp_tool_names,
      policy_server_for_tool=resolved_policy_server_for_tool,
      transport_server_for_policy_server=self._transport_server_name,
      set_startup_diagnostic=self._set_startup_diagnostic,
      logger=log,
    )
    self._tool_definitions = result.tool_definitions
    self._tool_to_server = result.tool_to_server
    self._prefixed_to_original = result.prefixed_to_original
    self._mcp_tool_names = result.mcp_tool_names
    self._dispatch_to_original = {
      exposed_name: original_name
      for exposed_name, original_name in self._dispatch_to_original.items()
      if exposed_name in self._tool_to_server
    }

  @staticmethod
  def _extract_text(content: Any) -> str:
    return _runtime_helpers.extract_text(content)

  def _result_message(self, result: _connection_helpers.McpToolCallResult) -> str:
    message = self._extract_text(result.content)
    structured_content = result.structured_content
    if not message and structured_content is not None:
      message = json.dumps(structured_content, default=str)
    return message

  @staticmethod
  async def _close_contexts(
    contexts: List[Any],
    *,
    close_timeout_seconds: float = _MCP_CLOSE_TIMEOUT_SECONDS,
  ) -> None:
    await _runtime_helpers.close_contexts(
      contexts,
      close_timeout_seconds=close_timeout_seconds,
      logger=log,
      suppress_warnings=_suppress_mcp_stdio_termination_fallback_warnings,
      wait_for=asyncio.wait_for,
    )

  def _read_claude_config(self) -> Dict[str, Any]:
    return _runtime_helpers.read_claude_config(self._config_path, json_load=json.load, logger=log)
