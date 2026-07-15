from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

from . import mcp_client_catalog as _catalog_helpers
from . import mcp_client_connections as _connection_helpers
from . import mcp_client_config as _config_helpers
from . import mcp_client_errors as _error_helpers
from . import mcp_client_oauth_storage as _oauth_storage
from . import mcp_client_policy_owner as _policy_owner_helpers
from . import mcp_client_runtime as _runtime_helpers
from . import mcp_client_startup as _startup_helpers
from .policy_imports import load_server_policy_helpers, load_server_policy_module

try:
  from mcp.client.session import ClientSession
  from mcp.client.stdio import StdioServerParameters, stdio_client
  MCP_IMPORT_ERROR: Exception | None = None
except Exception as exc:
  ClientSession = Any  # type: ignore[assignment]
  StdioServerParameters = Any  # type: ignore[assignment]
  MCP_IMPORT_ERROR = exc

  def stdio_client(*args: Any, **kwargs: Any) -> Any:
    _ = args, kwargs
    raise RuntimeError(f"MCP client runtime unavailable: {MCP_IMPORT_ERROR}")

try:
  from mcp.client.streamable_http import streamable_http_client
  STREAMABLE_HTTP_IMPORT_ERROR: Exception | None = None
except Exception as exc:
  streamable_http_client = None  # type: ignore[assignment]
  STREAMABLE_HTTP_IMPORT_ERROR = exc

try:
  import httpx
  HTTPX_IMPORT_ERROR: Exception | None = None
except Exception as exc:
  httpx = Any  # type: ignore[assignment]
  HTTPX_IMPORT_ERROR = exc

try:
  from fastmcp.client.auth.oauth import OAuth as FastMCPOAuth
  FASTMCP_OAUTH_IMPORT_ERROR: Exception | None = None
except Exception as exc:
  FastMCPOAuth = None  # type: ignore[assignment]
  FASTMCP_OAUTH_IMPORT_ERROR = exc


log = logging.getLogger("agent_gateway.mcp_client")
_UNSET = _config_helpers.UNSET
_ENV_REF_RE = _config_helpers.ENV_REF_RE
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
PER_USER_SESSION_TTL_SECONDS = 60 * 60
PER_USER_EXPIRY_MARGIN_SECONDS = 5 * 60
PER_USER_IDLE_REAP_SECONDS = 30 * 60
PER_USER_REAPER_INTERVAL_SECONDS = 60.0
PER_USER_INSTANCE_CAP = 32
PER_USER_DRAIN_TIMEOUT_SECONDS = 60.0
_MCP_STDIO_CONNECT_RETRIES_ENV = _config_helpers.MCP_STDIO_CONNECT_RETRIES_ENV
_MCP_STDIO_CONNECT_BACKOFF_ENV = _config_helpers.MCP_STDIO_CONNECT_BACKOFF_ENV
_MCP_STDIO_CONNECT_STABILIZE_ENV = _config_helpers.MCP_STDIO_CONNECT_STABILIZE_ENV
_MCP_STARTUP_CONCURRENCY_ENV = _config_helpers.MCP_STARTUP_CONCURRENCY_ENV
_MCP_STDIO_CONNECT_RETRIES_DEFAULT = _config_helpers.MCP_STDIO_CONNECT_RETRIES_DEFAULT
_MCP_STDIO_CONNECT_BACKOFF_DEFAULT = _config_helpers.MCP_STDIO_CONNECT_BACKOFF_DEFAULT
_MCP_STDIO_CONNECT_STABILIZE_DEFAULT = _config_helpers.MCP_STDIO_CONNECT_STABILIZE_DEFAULT
_MCP_STDIO_RETRYABLE_EXCEPTION_NAMES = _config_helpers.MCP_STDIO_RETRYABLE_EXCEPTION_NAMES
_MCP_STDIO_RETRYABLE_MESSAGE_MARKERS = _config_helpers.MCP_STDIO_RETRYABLE_MESSAGE_MARKERS
_PROVIDER_SYMBOL_SYMBOL_TOOLS = frozenset({
  "check_market_cap",
  "compare_peers",
  "fetch_financials",
  "get_earnings_transcript",
  "get_etf_holdings",
  "get_insider_trades",
  "get_institutional_ownership",
  "get_price_performance_windows",
  "get_technical_analysis",
  "industry_peer_comparison",
})
_PROVIDER_SYMBOL_TICKER_TOOLS = frozenset({
  "cite_concept",
  "concept_trend",
  "describe_filing",
  "get_concept",
  "get_estimate_revisions",
  "get_event_filings",
  "get_extraction_series",
  "get_filing_cover_facts",
  "get_filing_document",
  "get_filing_evidence",
  "get_filing_extractions",
  "get_filing_sections",
  "get_filing_tables",
  "get_filings",
  "get_financials",
  "get_metric",
  "get_metric_series",
  "get_operational_kpi_drivers",
  "get_statement",
  "list_metrics",
  "search_extractions",
  "search_filing_tables",
  "search_filing_text",
  "search_metrics",
})
_PROVIDER_SYMBOL_SCALAR_KEYS: dict[str, tuple[str, ...]] = {
  **{tool: ("symbol",) for tool in _PROVIDER_SYMBOL_SYMBOL_TOOLS},
  **{tool: ("ticker",) for tool in _PROVIDER_SYMBOL_TICKER_TOOLS},
}
_PROVIDER_SYMBOL_COMMA_KEYS: dict[str, str] = {
  "get_events_calendar": "symbols",
  "get_news": "symbols",
  "get_sector_overview": "symbols",
  "screen_estimate_revisions": "tickers",
}
_PROVIDER_SYMBOL_TOOL_NAMES = (
  frozenset(_PROVIDER_SYMBOL_SCALAR_KEYS)
  | frozenset(_PROVIDER_SYMBOL_COMMA_KEYS)
  | frozenset({"fetch_company_profile"})
)


def _resolve_mcp_config_path(config_path: Path | str | None | object = _UNSET) -> Path | None:
  return _config_helpers.resolve_mcp_config_path(
    config_path,
    unset=_UNSET,
    environ=os.environ,
    home_factory=Path.home,
  )


_McpStdioTerminationFallbackFilter = _runtime_helpers.McpStdioTerminationFallbackFilter


def _suppress_mcp_stdio_termination_fallback_warnings():
  return _runtime_helpers.suppress_mcp_stdio_termination_fallback_warnings(logging.getLogger, _McpStdioTerminationFallbackFilter)


def _build_mcp_env(server_env: Dict[str, Any] | None) -> Dict[str, str]:
  return _config_helpers.build_mcp_env(
    server_env if isinstance(server_env, dict) else None,
    environ=os.environ,
  )


def _expand_env_refs(value: Any) -> str:
  return _config_helpers.expand_env_refs(value, environ=os.environ)


def _build_http_headers(headers: Dict[str, Any] | None) -> Dict[str, str]:
  return _config_helpers.build_http_headers(
    headers if isinstance(headers, dict) else None,
    environ=os.environ,
  )


def _env_nonnegative_int(name: str, default: int) -> int:
  return _config_helpers.env_nonnegative_int(name, default, environ=os.environ, logger=log)


def _env_nonnegative_float(name: str, default: float) -> float:
  return _config_helpers.env_nonnegative_float(name, default, environ=os.environ, logger=log)


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


def _iter_exception_tree(exc: BaseException):
  yield from _config_helpers.iter_exception_tree(exc)


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

  @staticmethod
  def _active_value(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
      return None
    expires_at = entry.get("expires_at")
    if expires_at is not None:
      try:
        if float(expires_at) <= time.time():
          return None
      except (TypeError, ValueError):
        return None
    value = entry.get("value")
    return dict(value) if isinstance(value, dict) else None


def _classify_exception(exc: Exception, msg: str) -> str:
  return _error_helpers.classify_exception(exc, msg)


def _classify_mcp_error(message: str) -> str:
  return _error_helpers.classify_mcp_error(message)


def _is_sheets_transport_failure(exc: BaseException) -> bool:
  return any(
    isinstance(candidate, (asyncio.TimeoutError, TimeoutError))
    for candidate in _iter_exception_tree(exc)
  ) or _is_retryable_stdio_connect_error(exc)


def _sheets_structured_error(
  result: Any,
  *,
  expected_operation: str,
) -> dict[str, Any] | None:
  payload = getattr(result, "structuredContent", None)
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


@dataclass
class _ServerState:
  name: str
  session: ClientSession
  exit_contexts: List[Any]
  tool_definitions: List[Dict[str, Any]]
  tool_names: Set[str]
  tool_prefix: str = ""
  config: Dict[str, Any] | None = None


@dataclass
class _PerUserServerState:
  server: _ServerState
  expires_at: float
  last_used_at: float
  active_calls: int = 0
  draining: bool = False


class _PerUserMcpError(RuntimeError):
  def __init__(self, code: str, message: str | None = None) -> None:
    super().__init__(message or code)
    self.code = code


class McpClientManager:
  """Manage MCP server lifecycles and tool routing.

  The manager can load servers from inline config, from `MCP_CONFIG_PATH`, from
  `~/.claude.json`, or from an alternate config path. On startup it connects to
  each allowed server, lists its tools, filters name collisions, and exposes a
  merged tool catalog to the runner.

  `inline_servers` is the easiest way to ship self-contained examples because it
  avoids any dependency on the user's Claude desktop config.
  """

  def __init__(
    self,
    allowed_servers: Set[str] | None = None,
    builtin_tool_names: Set[str] | None = None,
    config_path: Path | str | None | object = _UNSET,
    inline_servers: Dict[str, Dict[str, Any]] | None = None,
    timeout_overrides: Dict[str, int] | None = None,
    tool_timeout_overrides: Dict[str, int] | None = None,
    server_aliases: Dict[str, str] | None = None,
    logical_server_routes: Dict[str, str] | None = None,
    logical_tool_aliases: Dict[str, Dict[str, str]] | None = None,
    provider_ids_by_server: Dict[str, str] | None = None,
    startup_timeout: int = 15,
    default_tool_timeout: int = 30,
    strip_input_fields: set[str] | None = None,
  ) -> None:
    self._lock = asyncio.Lock()
    self._started = False
    self._servers: Dict[str, _ServerState] = {}
    self._per_user_servers: Dict[tuple[str, str], _PerUserServerState] = {}
    self._per_user_spawn_locks: Dict[tuple[str, str], asyncio.Lock] = {}
    self._per_user_spawn_reservations: Dict[str, int] = {}
    self._per_user_reaper_task: asyncio.Task[Any] | None = None
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
    self._provider_ids_by_server = {
      self._canonical_server_name(server_name): str(provider_id).strip()
      for server_name, provider_id in dict(provider_ids_by_server or {}).items()
      if str(provider_id).strip()
    }
    self._logical_tool_definitions: Dict[str, List[Dict[str, Any]]] = {}
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

  def _canonical_server_names(self, server_names: Set[str]) -> Set[str]:
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

  def _timeout_for_tool(self, server_name: str, exposed_name: str, original_name: str) -> int:
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

  async def startup(self, allowed_servers: Set[str] | None = None) -> None:
    async with self._lock:
      if self._started:
        return
      await _startup_helpers.startup_manager(
        self,
        allowed_servers,
        mcp_import_error=MCP_IMPORT_ERROR,
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
      build_http_headers=_build_http_headers,
      safe_cache_name=_safe_cache_name,
      close_contexts=self._close_contexts,
      server_state_factory=_ServerState,
      stdio_server_parameters_factory=StdioServerParameters,
      stdio_client_factory=stdio_client,
      client_session_factory=ClientSession,
      httpx_module=httpx,
      streamable_http_client_factory=streamable_http_client,
      json_file_key_value_factory=_JsonFileKeyValue,
      fastmcp_oauth_factory=FastMCPOAuth,
      path_factory=Path,
      httpx_import_error=HTTPX_IMPORT_ERROR,
      streamable_http_import_error=STREAMABLE_HTTP_IMPORT_ERROR,
      fastmcp_oauth_import_error=FASTMCP_OAUTH_IMPORT_ERROR,
      environ=os.environ,
      logger=log,
    )

  async def _connect_startup_servers(
    self,
    connect_jobs: Sequence[tuple[str, Dict[str, Any]]],
  ) -> list[_ServerState | None]:
    return await _connection_helpers.connect_startup_servers(
      self,
      connect_jobs,
      self._connection_runtime(),
    )

  async def _connect_or_warn(self, name: str, config: Dict[str, Any]) -> _ServerState | None:
    return await _connection_helpers.connect_or_warn(
      self,
      name,
      config,
      self._connection_runtime(),
    )

  async def _connect(self, name: str, config: Dict[str, Any]) -> _ServerState:
    return await _connection_helpers.connect(
      self,
      name,
      config,
      self._connection_runtime(),
    )

  async def _connect_stdio_with_retries(self, name: str, config: Dict[str, Any]) -> _ServerState:
    return await _connection_helpers.connect_stdio_with_retries(
      self,
      name,
      config,
      self._connection_runtime(),
    )

  async def _connect_stdio(self, name: str, config: Dict[str, Any]) -> _ServerState:
    return await _connection_helpers.connect_stdio(
      self,
      name,
      config,
      self._connection_runtime(),
    )

  async def _connect_streamable_http(self, name: str, config: Dict[str, Any]) -> _ServerState:
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
    session: ClientSession,
    exit_contexts: List[Any],
    tool_prefix: str,
  ) -> _ServerState:
    return await _connection_helpers.initialize_session_state(
      self,
      name=name,
      session=session,
      exit_contexts=exit_contexts,
      tool_prefix=tool_prefix,
      runtime=self._connection_runtime(),
    )

  async def _verify_stdio_session_stable(self, session: ClientSession) -> None:
    await _connection_helpers.verify_stdio_session_stable(
      self,
      session,
      self._connection_runtime(),
    )

  def get_tool_definitions(self) -> List[Dict[str, Any]]:
    return copy.deepcopy(self._tool_definitions)

  def get_server_tool_definitions(self, server_names: Set[str]) -> List[Dict[str, Any]]:
    canonical_server_names = self._canonical_server_names(set(server_names))
    tool_definitions: List[Dict[str, Any]] = []
    for server_name, state in self._servers.items():
      if server_name in canonical_server_names and not self._is_transport_only_server(server_name):
        tool_definitions.extend(copy.deepcopy(state.tool_definitions))
    for server_name in canonical_server_names:
      tool_definitions.extend(copy.deepcopy(self._logical_tool_definitions.get(server_name, [])))
    return tool_definitions

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

  def get_server_catalog(self) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for server_name, state in self._servers.items():
      if self._is_transport_only_server(server_name):
        continue
      tool_names = sorted(tool["name"] for tool in state.tool_definitions if isinstance(tool.get("name"), str))
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
    return name in self._mcp_tool_names

  def get_server_for_tool(self, name: str) -> str | None:
    return self._tool_to_server.get(name)

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
  def _canonical_broker_body(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

  async def _mint_gsheets_broker_session(self, user_id: str) -> tuple[str, float, str]:
    hmac_key = os.environ.get("GATEWAY_GOOGLE_SHEETS_BROKER_HMAC_KEY", "").strip()
    base_url = os.environ.get("GOOGLE_SHEETS_BROKER_URL", "").strip().rstrip("/")
    if not hmac_key or not base_url:
      raise _PerUserMcpError("sheets_unavailable", "Google Sheets broker is not configured")
    timestamp = int(time.time())
    payload = {
      "user_id": user_id,
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
    user_id: str,
    broker_session: tuple[str, float, str] | None = None,
  ) -> _PerUserServerState:
    definition = self._servers.get(server_name)
    if definition is None or not definition.config:
      raise _PerUserMcpError("sheets_unavailable", f"MCP server unavailable: {server_name}")
    if broker_session is None:
      broker_session = await self._mint_gsheets_broker_session(user_id)
    token, expires_at, broker_url = broker_session
    config = copy.deepcopy(definition.config)
    env = dict(config.get("env") or {})
    env.update({
      "GSHEETS_TOKEN_MODE": "broker",
      "GSHEETS_HEADLESS": "1",
      "GSHEETS_BROKER_URL": broker_url,
      "GSHEETS_BROKER_SESSION_TOKEN": token,
    })
    config["env"] = env
    # The tier-1 credential is read only by this gateway process and is never
    # copied into the child config/environment.
    state = await self._connect_stdio_with_retries(f"{server_name}[user]", config)
    return _PerUserServerState(state, expires_at, time.time())

  async def _close_per_user_when_drained(self, state: _PerUserServerState) -> None:
    deadline = time.monotonic() + PER_USER_DRAIN_TIMEOUT_SECONDS
    while state.active_calls and time.monotonic() < deadline:
      await asyncio.sleep(0.05)
    await self._close_contexts(state.server.exit_contexts)

  def _schedule_drain(self, state: _PerUserServerState) -> None:
    if state.draining:
      return
    state.draining = True
    task = asyncio.create_task(self._close_per_user_when_drained(state))
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
          self._schedule_drain(state)
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
    user_id: str,
    *,
    force: bool = False,
    discard_current_on_failure: bool = False,
  ) -> _PerUserServerState:
    key = (server_name, user_id)
    lock = self._per_user_spawn_locks.setdefault(key, asyncio.Lock())
    try:
      async with lock:
        now = time.time()
        self._reap_idle_per_user_servers(now)
        current = self._per_user_servers.get(key)
        alive = current is not None and not current.draining and bool(current.server.exit_contexts)
        if alive and not force and current.expires_at - now > PER_USER_EXPIRY_MARGIN_SECONDS:
          current.last_used_at = now
          return current

        # Mint before changing capacity accounting or evicting a healthy child.
        # A forced broker-expiry replacement is the exception: the current
        # child is known invalid and must not remain cached when minting fails.
        try:
          broker_session = await self._mint_gsheets_broker_session(user_id)
        except BaseException:
          if force and discard_current_on_failure:
            expired = self._per_user_servers.pop(key, None)
            if expired is not None:
              self._schedule_drain(expired)
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
              raise _PerUserMcpError(
                "sheets_unavailable",
                "Google Sheets per-user instance capacity reached",
              )
            _, evict_key, evicted = min(idle)
            self._per_user_servers.pop(evict_key, None)
            self._schedule_drain(evicted)
            self._retire_per_user_spawn_lock(evict_key)
        self._per_user_spawn_reservations[server_name] = (
          self._per_user_spawn_reservations.get(server_name, 0) + 1
        )
        replacement = None
        spawned = False
        try:
          replacement = await self._spawn_per_user_server(
            server_name,
            user_id,
            broker_session=broker_session,
          )
          spawned = True
        finally:
          remaining = self._per_user_spawn_reservations.get(server_name, 0) - 1
          if remaining > 0:
            self._per_user_spawn_reservations[server_name] = remaining
          else:
            self._per_user_spawn_reservations.pop(server_name, None)
          if spawned:
            self._per_user_servers[key] = replacement
          elif (
            not discard_current_on_failure
            and old_state is not None
            and not old_state.draining
            and bool(old_state.server.exit_contexts)
          ):
            self._per_user_servers[key] = old_state
          elif old_state is not None:
            self._schedule_drain(old_state)
        self._ensure_per_user_reaper()
        if old_state is not None and old_state is not replacement:
          self._schedule_drain(old_state)
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
    tool_name: str,
    tool_input: Dict[str, Any],
  ) -> Dict[str, Any]:
    try:
      if tool_name not in _PROVIDER_SYMBOL_TOOL_NAMES:
        return tool_input

      from research.source_html import sec_native_symbol_cached_only

      def translate(value: Any) -> Any:
        return sec_native_symbol_cached_only(value) or value

      if tool_name == "fetch_company_profile":
        translated = dict(tool_input)
        present_keys = [key for key in ("symbol", "ticker") if key in translated]
        translated_values = {key: translate(translated[key]) for key in present_keys}
        if len(present_keys) == 2 and translated_values["symbol"] != translated_values["ticker"]:
          return translated
        for key, value in translated_values.items():
          translated[key] = value
        return translated

      scalar_keys = _PROVIDER_SYMBOL_SCALAR_KEYS.get(tool_name)
      if scalar_keys is not None:
        translated = dict(tool_input)
        for key in scalar_keys:
          if key in translated:
            translated[key] = translate(translated[key])
        return translated

      comma_key = _PROVIDER_SYMBOL_COMMA_KEYS.get(tool_name)
      if comma_key is not None:
        translated = dict(tool_input)
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
    tool_input: Dict[str, Any],
    meta: Dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    user_id: str | int | None = None,
  ) -> Tuple[Any | None, Dict[str, Any] | None]:
    server_name = self._tool_to_server.get(name)
    if not server_name:
      return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

    original_name = self.get_original_tool_name(name)
    is_sheets = server_name == _GSHEETS_SERVER_NAME
    policy_class = self._policy_tool_class(server_name, original_name) if is_sheets else None
    sheets_is_read_only = policy_class in _READ_ONLY_POLICY_CLASSES
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
      return None, {"code": "mcp_tool_error", "message": f"MCP server unavailable: {server_name}"}
    per_user_state: _PerUserServerState | None = None
    normalized_user_id: str | None = None
    if self.is_per_user_server(transport_server_name):
      normalized_user_id = str(user_id or "").strip()
      if not normalized_user_id or not normalized_user_id.isdigit() or int(normalized_user_id) <= 0:
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
        per_user_state = await self._get_per_user_server(transport_server_name, normalized_user_id)
      except _PerUserMcpError as exc:
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
    translation_name = name if name in _PROVIDER_SYMBOL_TOOL_NAMES else original_name
    effective_input = self._translate_provider_symbol(translation_name, tool_input)

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
        key = (transport_server_name, normalized_user_id or "")
        if self._per_user_servers.get(key) is per_user_state:
          self._per_user_servers.pop(key, None)
          self._retire_per_user_spawn_lock(key)
        self._schedule_drain(per_user_state)
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
        msg = str(exc)
        return None, {
          "code": "tool_error",
          "sub_code": _classify_exception(exc, msg),
          "message": msg,
        }
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
        msg = str(retry_exc)
        return None, {
          "code": "tool_error",
          "sub_code": _classify_exception(retry_exc, msg),
          "message": msg,
        }
      if retry_result is not None:
        result = retry_result
      else:
        msg = str(exc)
        return None, {
          "code": "tool_error",
          "sub_code": _classify_exception(exc, msg),
          "message": msg,
        }
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
      try:
        replacement = await self._get_per_user_server(
          transport_server_name,
          normalized_user_id or "",
          force=True,
          discard_current_on_failure=True,
        )
      except _PerUserMcpError:
        return None, _sheets_gateway_error(sheets_error)
      except Exception:
        return None, _sheets_gateway_error(sheets_error)

      if replay_safe:
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
          key = (server_name, normalized_user_id or "")
          if self._per_user_servers.get(key) is replacement:
            self._per_user_servers.pop(key, None)
            self._retire_per_user_spawn_lock(key)
          self._schedule_drain(replacement)
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
          key = (server_name, normalized_user_id or "")
          if self._per_user_servers.get(key) is replacement:
            self._per_user_servers.pop(key, None)
            self._retire_per_user_spawn_lock(key)
          self._schedule_drain(replacement)

    if is_sheets and sheets_error is not None:
      return None, _sheets_gateway_error(sheets_error)

    if result.isError:
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
        isinstance(result.structuredContent, dict)
        and result.structuredContent.get("status") == "ok"
        and result.structuredContent.get("operation") == original_name
      ):
        return result.structuredContent, None
      error_contract = bool(
        isinstance(result.structuredContent, dict)
        and result.structuredContent.get("status") == "error"
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

    if result.structuredContent is not None:
      return result.structuredContent, None

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
    server: _ServerState,
    original_name: str,
    tool_input: Dict[str, Any],
    meta: Dict[str, Any] | None,
    abort_event: asyncio.Event | None,
    timeout_seconds: int,
  ) -> Any:
    if abort_event is not None and abort_event.is_set():
      raise asyncio.CancelledError()
    call_kwargs = {
      "read_timeout_seconds": timedelta(seconds=timeout_seconds),
    }
    if meta is not None:
      call_kwargs["meta"] = meta
    call_task = asyncio.create_task(server.session.call_tool(
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
    timeout_seconds: int,
    cause: Exception,
  ) -> Any | None:
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
      await self._close_contexts(server.exit_contexts)
      replacement = await self._connect_stdio_with_retries(server_name, config)
    except Exception as reconnect_exc:
      reconnect_message = str(reconnect_exc).strip() or type(reconnect_exc).__name__
      log.warning(
        "MCP stdio server %s failed to reconnect after tool transport error: %s",
        server_name,
        reconnect_message,
      )
      return None

    server.session = replacement.session
    server.exit_contexts = replacement.exit_contexts
    server.config = replacement.config
    return await self._call_tool_once(
      server=server,
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
      await self._close_contexts(server.exit_contexts)
      replacement = await self._connect_stdio_with_retries(server_name, config)
    except Exception as reconnect_exc:
      log.warning(
        "MCP stdio server %s could not reconnect for future calls (%s)",
        server_name,
        type(reconnect_exc).__name__,
      )
      return False

    server.session = replacement.session
    server.exit_contexts = replacement.exit_contexts
    server.config = replacement.config
    return True

  async def shutdown(self) -> None:
    async with self._lock:
      if not self._started and not self._servers:
        return

      reaper_task = self._per_user_reaper_task
      self._per_user_reaper_task = None
      if reaper_task is not None:
        reaper_task.cancel()
        await asyncio.gather(reaper_task, return_exceptions=True)

      for server in reversed(list(self._servers.values())):
        await self._close_contexts(server.exit_contexts)

      for state in list(self._per_user_servers.values()):
        await self._close_contexts(state.server.exit_contexts)
      if self._drain_tasks:
        await asyncio.gather(*list(self._drain_tasks), return_exceptions=True)

      self._servers.clear()
      self._per_user_servers.clear()
      self._per_user_spawn_locks.clear()
      self._per_user_spawn_reservations.clear()
      self._tool_definitions = []
      self._tool_to_server = {}
      self._prefixed_to_original = {}
      self._dispatch_to_original = {}
      self._mcp_tool_names = set()
      self._logical_tool_definitions = {}
      self._startup_diagnostics = {}
      self._started = False

  def _apply_collision_filtering(
    self,
    *,
    policy_server_for_tool: Callable[[str], str | None] | None = None,
    policy_tool_class: Callable[[str, str], str | None] | None = None,
    strict_runtime_tool_set_for_server: Callable[[str], bool] | None = None,
  ) -> None:
    if (
      policy_server_for_tool is None
      or policy_tool_class is None
      or strict_runtime_tool_set_for_server is None
    ):
      _get_forbidden_tools_for_session, get_server_for_policy_tool, get_tool_class = load_server_policy_helpers()
      if get_server_for_policy_tool is None:
        log.warning(
          "Shared MCP policy module unavailable; enforcing the built-in Google Sheets cutover policy only"
        )
      elif policy_server_for_tool is None:
        policy_server_for_tool = get_server_for_policy_tool
      if policy_tool_class is None:
        policy_tool_class = get_tool_class
      policy_module = load_server_policy_module()
      policies = getattr(policy_module, "MCP_SERVER_POLICIES", {}) if policy_module is not None else {}
      if (
        strict_runtime_tool_set_for_server is None
        and policy_module is not None
        and isinstance(policies, dict)
      ):
        def policy_uses_strict_runtime_tool_set(server_name: str) -> bool:
          return (
            server_name == _GSHEETS_SERVER_NAME
            or bool(getattr(policies.get(server_name), "strict_runtime_tool_set", False))
          )

        strict_runtime_tool_set_for_server = policy_uses_strict_runtime_tool_set

    # Google Sheets is a security-sensitive full cutover. Keep its broker
    # surface closed-world even if the optional shared policy module cannot be
    # imported during gateway startup. Other MCP servers retain the existing
    # best-effort policy-import behavior.
    if policy_server_for_tool is None:
      def sheets_fallback_policy_server(tool_name: str) -> str | None:
        if tool_name in _GSHEETS_BROKER_TOOLS:
          return _GSHEETS_SERVER_NAME
        for logical_server, aliases in self._logical_tool_aliases.items():
          if tool_name in aliases:
            return logical_server
        return None

      policy_server_for_tool = sheets_fallback_policy_server
    if policy_tool_class is None:
      def sheets_fallback_tool_class(server_name: str, tool_name: str) -> str | None:
        if server_name != _GSHEETS_SERVER_NAME:
          return None
        if tool_name in _GSHEETS_BROKER_READ_TOOLS:
          return "read"
        if tool_name in _GSHEETS_BROKER_WRITE_TOOLS:
          return "external_write"
        return None

      policy_tool_class = sheets_fallback_tool_class
    if strict_runtime_tool_set_for_server is None:
      def sheets_fallback_strict_runtime_tool_set(server_name: str) -> bool:
        return server_name == _GSHEETS_SERVER_NAME

      strict_runtime_tool_set_for_server = sheets_fallback_strict_runtime_tool_set

    if policy_server_for_tool is not None:
      self._prefilter_policy_owner_mismatches(
        policy_server_for_tool=policy_server_for_tool,
        policy_tool_class=policy_tool_class,
        strict_runtime_tool_set_for_server=strict_runtime_tool_set_for_server,
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
        policy_tool_class=policy_tool_class,
        strict_runtime_tool_set_for_server=strict_runtime_tool_set_for_server,
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

  def _prefilter_policy_owner_mismatches(
    self,
    *,
    policy_server_for_tool: Callable[[str], str | None],
    policy_tool_class: Callable[[str, str], str | None] | None = None,
    strict_runtime_tool_set_for_server: Callable[[str], bool] | None = None,
  ) -> None:
    for server_name, state in self._servers.items():
      kept_tool_definitions: list[dict[str, Any]] = []
      mismatches: list[tuple[str, str, str]] = []
      for tool_def in state.tool_definitions:
        original_name = str(tool_def.get("name") or "").strip()
        if not original_name:
          kept_tool_definitions.append(tool_def)
          continue
        policy_server = policy_server_for_tool(original_name)
        policy_runtime_server = self._transport_server_name(policy_server) if policy_server else None
        if policy_server and policy_runtime_server != server_name:
          mismatches.append((original_name, policy_server, "owner_mismatch"))
          continue
        policy_context_server = policy_server or server_name
        strict_runtime = bool(
          strict_runtime_tool_set_for_server is not None
          and strict_runtime_tool_set_for_server(policy_context_server)
        )
        if strict_runtime and (
          policy_tool_class is None
          or policy_tool_class(policy_context_server, original_name) is None
        ):
          mismatches.append((original_name, "unclassified", "unclassified"))
          continue
        kept_tool_definitions.append(tool_def)

      if not mismatches:
        continue

      state.tool_definitions = kept_tool_definitions
      state.tool_names = {
        f"{state.tool_prefix}{tool_def['name']}" if state.tool_prefix else tool_def["name"]
        for tool_def in kept_tool_definitions
        if isinstance(tool_def.get("name"), str)
      }
      mismatch_summary = ", ".join(
        f"{original_name}->{policy_server}"
        for original_name, policy_server, _reason in mismatches
      )
      has_unclassified = any(reason == "unclassified" for _name, _server, reason in mismatches)
      if has_unclassified:
        message = (
          "MCP runtime exposed tools outside its strict gateway policy set; "
          f"pre-filtering tools before catalog merge: {mismatch_summary}"
        )
        category = "strict_runtime_tool_set_mismatch"
        error_type = "StrictRuntimeToolSetMismatch"
      else:
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
    policy_tool_class: Callable[[str, str], str | None] | None = None,
    strict_runtime_tool_set_for_server: Callable[[str], bool] | None = None,
  ) -> None:
    if policy_server_for_tool is None:
      _get_forbidden_tools_for_session, get_server_for_policy_tool, get_tool_class = load_server_policy_helpers()
      if get_server_for_policy_tool is None:
        log.warning("Skipping MCP policy-owner invariant: server policy module unavailable")
        return
      policy_server_for_tool = get_server_for_policy_tool
      if policy_tool_class is None:
        policy_tool_class = get_tool_class

    result = _policy_owner_helpers.apply_policy_owner_invariant(
      servers=self._servers,
      tool_definitions=self._tool_definitions,
      tool_to_server=self._tool_to_server,
      prefixed_to_original=self._prefixed_to_original,
      mcp_tool_names=self._mcp_tool_names,
      policy_server_for_tool=policy_server_for_tool,
      policy_tool_class=policy_tool_class,
      strict_runtime_tool_set_for_server=strict_runtime_tool_set_for_server,
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

  def _result_message(self, result: Any) -> str:
    message = self._extract_text(getattr(result, "content", None))
    structured_content = getattr(result, "structuredContent", None)
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
