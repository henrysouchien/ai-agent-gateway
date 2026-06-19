from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from . import mcp_client_catalog as _catalog_helpers
from . import mcp_client_config as _config_helpers
from . import mcp_client_errors as _error_helpers
from . import mcp_client_oauth_storage as _oauth_storage
from . import mcp_client_runtime as _runtime_helpers

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
_MCP_STDIO_CONNECT_RETRIES_ENV = _config_helpers.MCP_STDIO_CONNECT_RETRIES_ENV
_MCP_STDIO_CONNECT_BACKOFF_ENV = _config_helpers.MCP_STDIO_CONNECT_BACKOFF_ENV
_MCP_STDIO_CONNECT_STABILIZE_ENV = _config_helpers.MCP_STDIO_CONNECT_STABILIZE_ENV
_MCP_STARTUP_CONCURRENCY_ENV = _config_helpers.MCP_STARTUP_CONCURRENCY_ENV
_MCP_STDIO_CONNECT_RETRIES_DEFAULT = _config_helpers.MCP_STDIO_CONNECT_RETRIES_DEFAULT
_MCP_STDIO_CONNECT_BACKOFF_DEFAULT = _config_helpers.MCP_STDIO_CONNECT_BACKOFF_DEFAULT
_MCP_STDIO_CONNECT_STABILIZE_DEFAULT = _config_helpers.MCP_STDIO_CONNECT_STABILIZE_DEFAULT
_MCP_STDIO_RETRYABLE_EXCEPTION_NAMES = _config_helpers.MCP_STDIO_RETRYABLE_EXCEPTION_NAMES
_MCP_STDIO_RETRYABLE_MESSAGE_MARKERS = _config_helpers.MCP_STDIO_RETRYABLE_MESSAGE_MARKERS


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
    startup_timeout: int = 15,
    default_tool_timeout: int = 30,
    strip_input_fields: set[str] | None = None,
  ) -> None:
    self._lock = asyncio.Lock()
    self._started = False
    self._servers: Dict[str, _ServerState] = {}
    self._tool_definitions: List[Dict[str, Any]] = []
    self._tool_to_server: Dict[str, str] = {}
    self._prefixed_to_original: Dict[str, str] = {}
    self._mcp_tool_names: Set[str] = set()
    self._startup_diagnostics: Dict[str, Dict[str, Any]] = {}
    self._server_aliases = dict(server_aliases or {})
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
    canonical_configs: Dict[str, Dict[str, Any]] = {}
    source_names: Dict[str, str] = {}
    for server_name, server_config in mcp_servers.items():
      canonical_name = self._canonical_server_name(server_name)
      existing_source = source_names.get(canonical_name)
      if existing_source is not None:
        if server_name == canonical_name and existing_source != canonical_name:
          log.info(
            "MCP server %s overrides alias %s for canonical server %s",
            server_name,
            existing_source,
            canonical_name,
          )
          canonical_configs[canonical_name] = server_config
          source_names[canonical_name] = server_name
        else:
          log.warning(
            "Skipping MCP server %s: resolves to already configured canonical server %s",
            server_name,
            canonical_name,
          )
        continue

      if server_name != canonical_name:
        log.info("Using MCP server alias %s as %s", server_name, canonical_name)
      canonical_configs[canonical_name] = server_config
      source_names[canonical_name] = server_name
    return canonical_configs

  async def startup(self, allowed_servers: Set[str] | None = None) -> None:
    async with self._lock:
      if self._started:
        return

      self._startup_diagnostics = {}
      requested_servers = (
        self._canonical_server_names(set(allowed_servers))
        if allowed_servers is not None
        else None
      )

      if MCP_IMPORT_ERROR is not None:
        log.warning("MCP runtime unavailable; skipping startup: %s", MCP_IMPORT_ERROR)
        for server_name in sorted(requested_servers or set()):
          self._set_startup_diagnostic(
            server_name,
            category="runtime_unavailable",
            message=f"MCP runtime unavailable: {MCP_IMPORT_ERROR}",
            retryable=False,
            error_type=type(MCP_IMPORT_ERROR).__name__,
          )
        self._started = True
        return

      config = self._read_claude_config()
      mcp_servers = config.get("mcpServers", {})
      if not isinstance(mcp_servers, dict):
        mcp_servers = {}
      mcp_servers = dict(mcp_servers)
      mcp_servers.update(self._inline_servers)

      effective_allowed_servers = self._allowed_servers
      if requested_servers is not None:
        if effective_allowed_servers is None:
          effective_allowed_servers = requested_servers
        else:
          not_allowed = requested_servers - effective_allowed_servers
          for server_name in sorted(not_allowed):
            self._set_startup_diagnostic(
              server_name,
              category="not_allowed",
              message="server requested for startup but absent from the MCP allowlist",
              retryable=False,
            )
          effective_allowed_servers = set(effective_allowed_servers) & requested_servers

      if not mcp_servers:
        missing_config_targets = effective_allowed_servers if requested_servers is not None else set()
        for server_name in sorted(missing_config_targets or set()):
          self._set_startup_diagnostic(
            server_name,
            category="config_missing",
            message="server requested for startup but absent from the MCP config",
            retryable=False,
          )
        self._started = True
        return
      mcp_servers = self._canonicalize_server_configs(mcp_servers)

      if requested_servers is not None and effective_allowed_servers is not None:
        for server_name in sorted(effective_allowed_servers - set(mcp_servers)):
          self._set_startup_diagnostic(
            server_name,
            category="config_missing",
            message="server requested for startup but absent from the MCP config",
            retryable=False,
          )

      connect_jobs: list[tuple[str, Dict[str, Any]]] = []
      for server_name, server_config in mcp_servers.items():
        if effective_allowed_servers is not None and server_name not in effective_allowed_servers:
          continue
        if not isinstance(server_config, dict):
          log.warning("Skipping MCP server %s: invalid config", server_name)
          self._set_startup_diagnostic(
            server_name,
            category="invalid_config",
            message="invalid MCP server config",
            retryable=False,
          )
          continue

        server_type = str(server_config.get("type", "stdio")).strip().lower()
        if server_type not in _SUPPORTED_SERVER_TYPES:
          log.info("Skipping MCP server %s: unsupported type %s", server_name, server_type)
          self._set_startup_diagnostic(
            server_name,
            category="unsupported_type",
            message=f"unsupported MCP server type: {server_type}",
            retryable=False,
          )
          continue

        connect_jobs.append((server_name, server_config))

      if connect_jobs:
        for state in await self._connect_startup_servers(connect_jobs):
          if state is not None:
            self._servers[state.name] = state

      self._apply_collision_filtering()
      self._started = True

  async def _connect_startup_servers(
    self,
    connect_jobs: Sequence[tuple[str, Dict[str, Any]]],
  ) -> list[_ServerState | None]:
    concurrency = _startup_concurrency_limit()
    if concurrency <= 0 or concurrency >= len(connect_jobs):
      return await asyncio.gather(
        *(
          self._connect_or_warn(server_name, server_config)
          for server_name, server_config in connect_jobs
        )
      )

    log.info(
      "MCP startup concurrency limited to %d for %d server(s)",
      concurrency,
      len(connect_jobs),
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def _connect_limited(
      server_name: str,
      server_config: Dict[str, Any],
    ) -> _ServerState | None:
      async with semaphore:
        return await self._connect_or_warn(server_name, server_config)

    return await asyncio.gather(
      *(_connect_limited(server_name, server_config) for server_name, server_config in connect_jobs)
    )

  async def _connect_or_warn(self, name: str, config: Dict[str, Any]) -> _ServerState | None:
    try:
      state = await self._connect(name, config)
      self._startup_diagnostics.pop(self._canonical_server_name(name), None)
      return state
    except Exception as exc:
      message = str(exc).strip() or type(exc).__name__
      diagnostic = _startup_failure_from_exception(exc)
      self._set_startup_diagnostic(
        name,
        category=str(diagnostic["category"]),
        message=str(diagnostic["message"]),
        retryable=bool(diagnostic["retryable"]),
        error_type=str(diagnostic["error_type"]) if diagnostic.get("error_type") else None,
      )
      log.warning("MCP server %s failed to connect: %s", name, message)
      return None

  async def _connect(self, name: str, config: Dict[str, Any]) -> _ServerState:
    server_type = str(config.get("type", "stdio")).strip().lower()
    if server_type in _STREAMABLE_HTTP_TYPES:
      return await self._connect_streamable_http(name, config)
    if server_type == "stdio":
      return await self._connect_stdio_with_retries(name, config)
    raise ValueError(f"unsupported type {server_type}")

  async def _connect_stdio_with_retries(self, name: str, config: Dict[str, Any]) -> _ServerState:
    total_attempts = 1 + _stdio_connect_retries()
    for attempt in range(1, total_attempts + 1):
      try:
        return await self._connect_stdio(name, config)
      except Exception as exc:
        if attempt >= total_attempts or not _is_retryable_stdio_connect_error(exc):
          raise
        delay = _stdio_connect_retry_delay(attempt)
        message = str(exc).strip() or type(exc).__name__
        log.warning(
          "MCP stdio server %s connect attempt %d/%d failed with transient transport error; "
          "retrying in %.2fs: %s",
          name,
          attempt,
          total_attempts,
          delay,
          message,
        )
        if delay > 0:
          await asyncio.sleep(delay)
    raise RuntimeError("unreachable stdio connect retry state")

  async def _connect_stdio(self, name: str, config: Dict[str, Any]) -> _ServerState:
    exit_contexts: List[Any] = []
    success = False
    try:
      command = str(config.get("command", "")).strip()
      if not command:
        raise ValueError("missing command")

      args_raw = config.get("args", [])
      args = [str(arg) for arg in args_raw] if isinstance(args_raw, list) else []

      env_raw = config.get("env")
      env = _build_mcp_env(env_raw if isinstance(env_raw, dict) else None)

      cwd = config.get("cwd")
      tool_prefix = str(config.get("tool_prefix", "") or "").strip()
      server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
        cwd=cwd,
      )

      devnull = open(os.devnull, "w")
      stdio_cm = stdio_client(server_params, errlog=devnull)
      read_stream, write_stream = await stdio_cm.__aenter__()
      exit_contexts.append(stdio_cm)

      session = ClientSession(read_stream, write_stream)
      await session.__aenter__()
      exit_contexts.append(session)

      state = await self._initialize_session_state(
        name=name,
        session=session,
        exit_contexts=exit_contexts,
        tool_prefix=tool_prefix,
      )
      await self._verify_stdio_session_stable(session)
      state.config = dict(config)
      success = True
      return state
    finally:
      if not success:
        await self._close_contexts(exit_contexts)

  async def _connect_streamable_http(self, name: str, config: Dict[str, Any]) -> _ServerState:
    if HTTPX_IMPORT_ERROR is not None:
      raise RuntimeError(f"HTTP MCP transport unavailable: {HTTPX_IMPORT_ERROR}")
    if STREAMABLE_HTTP_IMPORT_ERROR is not None or streamable_http_client is None:
      raise RuntimeError(f"HTTP MCP transport unavailable: {STREAMABLE_HTTP_IMPORT_ERROR}")

    exit_contexts: List[Any] = []
    success = False
    try:
      url = str(config.get("url") or "").strip()
      if not url:
        raise ValueError("missing url")

      headers_raw = config.get("headers")
      headers = _build_http_headers(headers_raw if isinstance(headers_raw, dict) else None)
      timeout_seconds = float(config.get("timeout", self._startup_timeout))
      sse_read_timeout_seconds = float(config.get("sse_read_timeout", 300))
      terminate_on_close = bool(config.get("terminate_on_close", True))
      tool_prefix = str(config.get("tool_prefix", "") or "").strip()
      auth = self._build_http_auth(name, url, config)

      http_client = httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(timeout_seconds, read=sse_read_timeout_seconds),
        auth=auth,
      )
      await http_client.__aenter__()
      exit_contexts.append(http_client)

      stream_cm = streamable_http_client(
        url,
        http_client=http_client,
        terminate_on_close=terminate_on_close,
      )
      read_stream, write_stream, _get_session_id = await stream_cm.__aenter__()
      exit_contexts.append(stream_cm)

      session = ClientSession(read_stream, write_stream)
      await session.__aenter__()
      exit_contexts.append(session)

      state = await self._initialize_session_state(
        name=name,
        session=session,
        exit_contexts=exit_contexts,
        tool_prefix=tool_prefix,
      )
      success = True
      return state
    finally:
      if not success:
        await self._close_contexts(exit_contexts)

  def _build_http_auth(self, name: str, url: str, config: Dict[str, Any]) -> Any | None:
    oauth_raw = config.get("oauth")
    if not oauth_raw:
      return None
    if FASTMCP_OAUTH_IMPORT_ERROR is not None or FastMCPOAuth is None:
      raise RuntimeError(f"OAuth MCP transport unavailable: {FASTMCP_OAUTH_IMPORT_ERROR}")
    if oauth_raw is True:
      oauth_config: dict[str, Any] = {}
    elif isinstance(oauth_raw, dict):
      oauth_config = dict(oauth_raw)
    else:
      raise ValueError("oauth must be true or an object")

    cache_path = oauth_config.get("cache_path")
    if cache_path is None:
      cache_dir = Path(
        os.environ.get(
          "AGENT_GATEWAY_MCP_OAUTH_CACHE_DIR",
          str(Path.home() / ".cache" / "agent-gateway" / "mcp-oauth"),
        )
      )
      cache_path = cache_dir / f"{_safe_cache_name(name)}.json"
    storage = _JsonFileKeyValue(Path(str(cache_path)).expanduser())
    scopes = oauth_config.get("scopes")
    callback_port = oauth_config.get("callback_port")
    return FastMCPOAuth(
      mcp_url=url,
      scopes=scopes,
      client_name=str(oauth_config.get("client_name") or f"agent-gateway:{name}"),
      token_storage=storage,
      callback_port=int(callback_port) if callback_port is not None else None,
      client_metadata_url=oauth_config.get("client_metadata_url"),
      client_id=oauth_config.get("client_id"),
      client_secret=oauth_config.get("client_secret"),
    )

  async def _initialize_session_state(
    self,
    *,
    name: str,
    session: ClientSession,
    exit_contexts: List[Any],
    tool_prefix: str,
  ) -> _ServerState:
    await asyncio.wait_for(session.initialize(), timeout=self._startup_timeout)

    tools: List[Any] = []
    cursor: str | None = None
    while True:
      listed = await asyncio.wait_for(
        session.list_tools(cursor=cursor),
        timeout=self._startup_timeout,
      )
      tools.extend(listed.tools or [])
      if listed.nextCursor is None:
        break
      cursor = listed.nextCursor

    tool_definitions: List[Dict[str, Any]] = []
    for tool in tools:
      input_schema = tool.inputSchema or {"type": "object", "properties": {}}
      tool_definitions.append(
        {
          "name": tool.name,
          "description": tool.description or "",
          "input_schema": copy.deepcopy(input_schema),
        }
      )

    return _ServerState(
      name=name,
      session=session,
      exit_contexts=exit_contexts,
      tool_definitions=tool_definitions,
      tool_names={tool["name"] for tool in tool_definitions},
      tool_prefix=tool_prefix,
    )

  async def _verify_stdio_session_stable(self, session: ClientSession) -> None:
    delay = _stdio_connect_stabilize_delay()
    if delay > 0:
      await asyncio.sleep(delay)
    await asyncio.wait_for(
      session.list_tools(cursor=None),
      timeout=self._startup_timeout,
    )

  def get_tool_definitions(self) -> List[Dict[str, Any]]:
    return copy.deepcopy(self._tool_definitions)

  def get_server_tool_definitions(self, server_names: Set[str]) -> List[Dict[str, Any]]:
    canonical_server_names = self._canonical_server_names(set(server_names))
    tool_definitions: List[Dict[str, Any]] = []
    for server_name, state in self._servers.items():
      if server_name in canonical_server_names:
        tool_definitions.extend(copy.deepcopy(state.tool_definitions))
    return tool_definitions

  def get_server_names(self) -> Set[str]:
    return set(self._servers.keys())

  def get_server_catalog(self) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for server_name, state in self._servers.items():
      tool_names = sorted(tool["name"] for tool in state.tool_definitions if isinstance(tool.get("name"), str))
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

  def resolve_tool_name(self, server_name: str, original_name: str) -> str | None:
    """Return the exposed tool name for a server-owned tool."""
    server_name = self._canonical_server_name(server_name)
    state = self._servers.get(server_name)
    if state is None:
      return None
    exposed_name = f"{state.tool_prefix}{original_name}" if state.tool_prefix else original_name
    return exposed_name if self._tool_to_server.get(exposed_name) == server_name else None

  async def call_tool(
    self,
    name: str,
    tool_input: Dict[str, Any],
    meta: Dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
  ) -> Tuple[Any | None, Dict[str, Any] | None]:
    server_name = self._tool_to_server.get(name)
    if not server_name:
      return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

    server = self._servers.get(server_name)
    if not server:
      return None, {"code": "mcp_tool_error", "message": f"MCP server unavailable: {server_name}"}
    original_name = self._prefixed_to_original.get(name, name)

    timeout_seconds = self._timeout_for_tool(server_name, name, original_name)

    try:
      result = await self._call_tool_once(
        server=server,
        original_name=original_name,
        tool_input=tool_input,
        meta=meta,
        abort_event=abort_event,
        timeout_seconds=timeout_seconds,
      )
    except Exception as exc:
      try:
        retry_result = await self._retry_stdio_tool_call_after_reconnect(
          server_name=server_name,
          server=server,
          original_name=original_name,
          tool_input=tool_input,
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

    if result.isError:
      message = self._extract_text(result.content)
      if not message and result.structuredContent is not None:
        message = json.dumps(result.structuredContent, default=str)
      return None, {
        "code": "mcp_tool_error",
        "sub_code": _classify_mcp_error(message or ""),
        "message": message or f"MCP tool failed: {name}",
      }

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

  async def shutdown(self) -> None:
    async with self._lock:
      if not self._started and not self._servers:
        return

      for server in reversed(list(self._servers.values())):
        await self._close_contexts(server.exit_contexts)

      self._servers.clear()
      self._tool_definitions = []
      self._tool_to_server = {}
      self._prefixed_to_original = {}
      self._mcp_tool_names = set()
      self._startup_diagnostics = {}
      self._started = False

  def _apply_collision_filtering(self) -> None:
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

  @staticmethod
  def _extract_text(content: Any) -> str:
    return _runtime_helpers.extract_text(content)

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
