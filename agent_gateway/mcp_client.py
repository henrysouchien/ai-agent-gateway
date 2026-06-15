from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, SupportsFloat, Tuple

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
_UNSET = object()
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_STREAMABLE_HTTP_TYPES = {"streamable-http", "streamable_http", "http", "streamable"}
_SUPPORTED_SERVER_TYPES = {"stdio"} | _STREAMABLE_HTTP_TYPES
# Default-deny: MCP subprocesses inherit only a small set of safe env vars.
_DEFAULT_ENV_ALLOWLIST = {
  "PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR", "USER",
  "PYTHONPATH", "NODE_PATH", "VIRTUAL_ENV",
}
_MCP_STDIO_TERMINATION_LOGGER = "mcp.os.posix.utilities"
_MCP_STDIO_PG_FALLBACK_PREFIX = "Process group termination failed for PID "
_MCP_STDIO_PG_FALLBACK_MARKER = "falling back to simple terminate"
_MCP_CLOSE_TIMEOUT_SECONDS = 5.0


def _resolve_mcp_config_path(config_path: Path | str | None | object = _UNSET) -> Path | None:
  if config_path is _UNSET:
    env_path = os.getenv("MCP_CONFIG_PATH", "").strip()
    return Path(env_path).expanduser() if env_path else Path.home() / ".claude.json"
  if config_path is None:
    return None
  return Path(config_path).expanduser()


class _McpStdioTerminationFallbackFilter(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool:
    if record.name != _MCP_STDIO_TERMINATION_LOGGER:
      return True
    message = record.getMessage()
    return not (
      message.startswith(_MCP_STDIO_PG_FALLBACK_PREFIX)
      and _MCP_STDIO_PG_FALLBACK_MARKER in message
    )


@contextmanager
def _suppress_mcp_stdio_termination_fallback_warnings():
  """Suppress the MCP SDK's expected process-group fallback warning during close."""
  upstream_logger = logging.getLogger(_MCP_STDIO_TERMINATION_LOGGER)
  log_filter = _McpStdioTerminationFallbackFilter()
  upstream_logger.addFilter(log_filter)
  try:
    yield
  finally:
    upstream_logger.removeFilter(log_filter)


def _build_mcp_env(server_env: Dict[str, Any] | None) -> Dict[str, str]:
  env = {k: v for k, v in os.environ.items() if k in _DEFAULT_ENV_ALLOWLIST}
  if not isinstance(server_env, dict):
    return env

  for key, value in server_env.items():
    if value is None:
      continue
    env[str(key)] = _expand_env_refs(value)
  return env


def _expand_env_refs(value: Any) -> str:
  raw = str(value)
  return _ENV_REF_RE.sub(lambda match: os.environ.get(match.group(1), ""), raw)


def _build_http_headers(headers: Dict[str, Any] | None) -> Dict[str, str]:
  if not isinstance(headers, dict):
    return {}
  return {
    str(key): _expand_env_refs(value)
    for key, value in headers.items()
    if value is not None
  }


def _safe_cache_name(name: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "server"


class _JsonFileKeyValue:
  """Minimal AsyncKeyValue-compatible JSON store for FastMCP OAuth tokens."""

  def __init__(self, path: Path | str, *, default_collection: str = "default") -> None:
    self._path = Path(path).expanduser()
    self._default_collection = default_collection

  def _collection(self, collection: str | None) -> str:
    return str(collection or self._default_collection)

  def _load(self) -> dict[str, dict[str, dict[str, Any]]]:
    try:
      data = json.loads(self._path.read_text(encoding="utf-8"))
    except FileNotFoundError:
      return {}
    except (OSError, json.JSONDecodeError):
      return {}
    return data if isinstance(data, dict) else {}

  def _save(self, data: dict[str, dict[str, dict[str, Any]]]) -> None:
    self._path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
    tmp_path.write_text(
      json.dumps(data, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    try:
      tmp_path.chmod(0o600)
    except OSError:
      pass
    os.replace(tmp_path, self._path)

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

  async def get(self, key: str, *, collection: str | None = None) -> dict[str, Any] | None:
    data = self._load()
    coll = self._collection(collection)
    value = self._active_value(data.get(coll, {}).get(str(key)))
    if value is None and str(key) in data.get(coll, {}):
      await self.delete(str(key), collection=coll)
    return value

  async def ttl(
    self,
    key: str,
    *,
    collection: str | None = None,
  ) -> tuple[dict[str, Any] | None, float | None]:
    data = self._load()
    coll = self._collection(collection)
    entry = data.get(coll, {}).get(str(key))
    value = self._active_value(entry)
    if value is None:
      if str(key) in data.get(coll, {}):
        await self.delete(str(key), collection=coll)
      return None, None
    expires_at = entry.get("expires_at") if isinstance(entry, dict) else None
    ttl_seconds = None if expires_at is None else max(float(expires_at) - time.time(), 0.0)
    return value, ttl_seconds

  async def put(
    self,
    key: str,
    value: Mapping[str, Any],
    *,
    collection: str | None = None,
    ttl: SupportsFloat | None = None,
  ) -> None:
    data = self._load()
    coll = self._collection(collection)
    expires_at = None if ttl is None else time.time() + float(ttl)
    data.setdefault(coll, {})[str(key)] = {
      "value": dict(value),
      "expires_at": expires_at,
    }
    self._save(data)

  async def delete(self, key: str, *, collection: str | None = None) -> bool:
    data = self._load()
    coll = self._collection(collection)
    bucket = data.get(coll, {})
    existed = str(key) in bucket
    if existed:
      bucket.pop(str(key), None)
      if not bucket:
        data.pop(coll, None)
      self._save(data)
    return existed

  async def get_many(
    self,
    keys: Sequence[str],
    *,
    collection: str | None = None,
  ) -> list[dict[str, Any] | None]:
    return [await self.get(str(key), collection=collection) for key in keys]

  async def ttl_many(
    self,
    keys: Sequence[str],
    *,
    collection: str | None = None,
  ) -> list[tuple[dict[str, Any] | None, float | None]]:
    return [await self.ttl(str(key), collection=collection) for key in keys]

  async def put_many(
    self,
    keys: Sequence[str],
    values: Sequence[Mapping[str, Any]],
    *,
    collection: str | None = None,
    ttl: SupportsFloat | None = None,
  ) -> None:
    if len(keys) != len(values):
      raise ValueError("keys and values must have the same length")
    for key, value in zip(keys, values):
      await self.put(str(key), value, collection=collection, ttl=ttl)

  async def delete_many(self, keys: Sequence[str], *, collection: str | None = None) -> int:
    deleted = 0
    for key in keys:
      if await self.delete(str(key), collection=collection):
        deleted += 1
    return deleted


def _classify_exception(exc: Exception, msg: str) -> str:
  lower = msg.lower()
  if isinstance(exc, asyncio.TimeoutError) or "timeout" in lower or "timed out" in lower:
    return "timeout"
  if "connection" in lower or "refused" in lower:
    return "connection_error"
  return "unknown"


def _classify_mcp_error(message: str) -> str:
  lower = message.lower()
  if "not found" in lower or "no filing" in lower or "no data" in lower:
    return "not_found"
  if "parse" in lower or "invalid" in lower or "malformed" in lower:
    return "parse_error"
  if "timeout" in lower or "timed out" in lower:
    return "timeout"
  return "unknown"


@dataclass
class _ServerState:
  name: str
  session: ClientSession
  exit_contexts: List[Any]
  tool_definitions: List[Dict[str, Any]]
  tool_names: Set[str]
  tool_prefix: str = ""


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

      if MCP_IMPORT_ERROR is not None:
        log.warning("MCP runtime unavailable; skipping startup: %s", MCP_IMPORT_ERROR)
        self._started = True
        return

      config = self._read_claude_config()
      mcp_servers = config.get("mcpServers", {})
      if not isinstance(mcp_servers, dict):
        mcp_servers = {}
      mcp_servers = dict(mcp_servers)
      mcp_servers.update(self._inline_servers)
      if not mcp_servers:
        self._started = True
        return
      mcp_servers = self._canonicalize_server_configs(mcp_servers)

      effective_allowed_servers = self._allowed_servers
      if allowed_servers is not None:
        requested_servers = self._canonical_server_names(set(allowed_servers))
        if effective_allowed_servers is None:
          effective_allowed_servers = requested_servers
        else:
          effective_allowed_servers = set(effective_allowed_servers) & requested_servers

      connect_tasks = []
      for server_name, server_config in mcp_servers.items():
        if effective_allowed_servers is not None and server_name not in effective_allowed_servers:
          continue
        if not isinstance(server_config, dict):
          log.warning("Skipping MCP server %s: invalid config", server_name)
          continue

        server_type = str(server_config.get("type", "stdio")).strip().lower()
        if server_type not in _SUPPORTED_SERVER_TYPES:
          log.info("Skipping MCP server %s: unsupported type %s", server_name, server_type)
          continue

        connect_tasks.append(self._connect_or_warn(server_name, server_config))

      if connect_tasks:
        for state in await asyncio.gather(*connect_tasks):
          if state is not None:
            self._servers[state.name] = state

      self._apply_collision_filtering()
      self._started = True

  async def _connect_or_warn(self, name: str, config: Dict[str, Any]) -> _ServerState | None:
    try:
      return await self._connect(name, config)
    except Exception as exc:
      message = str(exc).strip() or type(exc).__name__
      log.warning("MCP server %s failed to connect: %s", name, message)
      return None

  async def _connect(self, name: str, config: Dict[str, Any]) -> _ServerState:
    server_type = str(config.get("type", "stdio")).strip().lower()
    if server_type in _STREAMABLE_HTTP_TYPES:
      return await self._connect_streamable_http(name, config)
    if server_type == "stdio":
      return await self._connect_stdio(name, config)
    raise ValueError(f"unsupported type {server_type}")

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
        if abort_event is None:
          result = await call_task
        else:
          abort_task = asyncio.create_task(abort_event.wait())
          done, _pending = await asyncio.wait(
            {call_task, abort_task},
            return_when=asyncio.FIRST_COMPLETED,
          )
          if abort_task in done and abort_event.is_set():
            call_task.cancel()
            await asyncio.gather(call_task, return_exceptions=True)
            raise asyncio.CancelledError()
          result = await call_task
      finally:
        if abort_task is not None:
          abort_task.cancel()
    except Exception as exc:
      msg = str(exc)
      return None, {
        "code": "tool_error",
        "sub_code": _classify_exception(exc, msg),
        "message": msg,
      }

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
      self._started = False

  def _apply_collision_filtering(self) -> None:
    existing_names = set(self._builtin_tool_names)
    seen_mcp_names: Dict[str, str] = {}
    merged: List[Dict[str, Any]] = []

    self._tool_to_server = {}
    self._prefixed_to_original = {}

    for server_name, state in self._servers.items():
      prefix = state.tool_prefix
      filtered: List[Dict[str, Any]] = []
      filtered_names: Set[str] = set()

      for tool in state.tool_definitions:
        original_name = tool["name"]
        tool_name = f"{prefix}{original_name}" if prefix else original_name
        if tool_name in existing_names:
          log.warning(
            "Skipping MCP tool %s from %s: collides with built-in tool. "
            "Fix: add \"tool_prefix\" to the \"%s\" server config to namespace its tools.",
            tool_name,
            server_name,
            server_name,
          )
          continue

        first_server = seen_mcp_names.get(tool_name)
        if first_server:
          log.warning(
            "Skipping MCP tool %s from %s: collides with MCP tool from %s. "
            "Fix: add \"tool_prefix\" to the \"%s\" server config to namespace its tools.",
            tool_name,
            server_name,
            first_server,
            server_name,
          )
          continue

        seen_mcp_names[tool_name] = server_name
        self._tool_to_server[tool_name] = server_name
        if prefix:
          self._prefixed_to_original[tool_name] = original_name
          tool = {**tool, "name": tool_name}
        filtered.append(tool)
        filtered_names.add(tool_name)

      state.tool_definitions = filtered
      state.tool_names = filtered_names
      merged.extend(filtered)

      log.info(
        "MCP server %s connected | %d tools: %s",
        server_name,
        len(filtered_names),
        sorted(filtered_names),
      )

    self._tool_definitions = merged
    if self._strip_input_fields:
      for tool_def in self._tool_definitions:
        schema = tool_def.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for field in self._strip_input_fields:
          props.pop(field, None)
          if field in required:
            required.remove(field)
    self._mcp_tool_names = set(self._tool_to_server.keys())

  @staticmethod
  def _extract_text(content: Any) -> str:
    if not content:
      return ""

    chunks: List[str] = []
    for item in content:
      if isinstance(item, dict):
        text = item.get("text")
      else:
        text = getattr(item, "text", None)
      if isinstance(text, str) and text.strip():
        chunks.append(text)
    return "\n".join(chunks).strip()

  @staticmethod
  async def _close_contexts(
    contexts: List[Any],
    *,
    close_timeout_seconds: float = _MCP_CLOSE_TIMEOUT_SECONDS,
  ) -> None:
    while contexts:
      ctx = contexts.pop()
      try:
        with _suppress_mcp_stdio_termination_fallback_warnings():
          await asyncio.wait_for(
            ctx.__aexit__(None, None, None),
            timeout=close_timeout_seconds,
          )
      except asyncio.TimeoutError:
        log.warning(
          "MCP context close timed out after %.1fs; continuing shutdown",
          close_timeout_seconds,
        )
      except Exception as exc:
        log.debug("MCP context close failed: %s", exc)

  def _read_claude_config(self) -> Dict[str, Any]:
    if self._config_path is None or not self._config_path.exists():
      return {}

    try:
      with open(self._config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as exc:
      log.warning("Failed to read %s: %s", self._config_path, exc)
      return {}

    return data if isinstance(data, dict) else {}
