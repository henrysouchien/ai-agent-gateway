from __future__ import annotations

import asyncio
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping, Sequence


@dataclass(frozen=True)
class McpConnectionRuntime:
  startup_concurrency_limit: Callable[[], int]
  startup_failure_from_exception: Callable[[BaseException], dict[str, Any]]
  streamable_http_types: set[str]
  stdio_connect_retries: Callable[[], int]
  stdio_connect_retry_delay: Callable[[int], float]
  stdio_connect_stabilize_delay: Callable[[], float]
  is_retryable_stdio_connect_error: Callable[[BaseException], bool]
  build_mcp_env: Callable[[dict[str, Any] | None], dict[str, str]]
  build_http_headers: Callable[[dict[str, Any] | None], dict[str, str]]
  safe_cache_name: Callable[[str], str]
  close_contexts: Callable[[list[Any]], Any]
  server_state_factory: Callable[..., Any]
  stdio_server_parameters_factory: Callable[..., Any]
  stdio_client_factory: Callable[..., Any]
  client_session_factory: Callable[..., Any]
  httpx_module: Any
  streamable_http_client_factory: Any
  json_file_key_value_factory: Callable[[Path], Any]
  fastmcp_oauth_factory: Any
  path_factory: Any
  httpx_import_error: Exception | None
  streamable_http_import_error: Exception | None
  fastmcp_oauth_import_error: Exception | None
  environ: MutableMapping[str, str]
  logger: Any


class _SyncCloseContext:
  def __init__(self, close: Callable[[], Any]) -> None:
    self._close = close

  async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    self._close()


async def connect_startup_servers(
  manager: Any,
  connect_jobs: Sequence[tuple[str, dict[str, Any]]],
  runtime: McpConnectionRuntime,
) -> list[Any | None]:
  concurrency = runtime.startup_concurrency_limit()
  if concurrency <= 0 or concurrency >= len(connect_jobs):
    return await asyncio.gather(
      *(
        manager._connect_or_warn(server_name, server_config)
        for server_name, server_config in connect_jobs
      )
    )

  runtime.logger.info(
    "MCP startup concurrency limited to %d for %d server(s)",
    concurrency,
    len(connect_jobs),
  )
  semaphore = asyncio.Semaphore(concurrency)

  async def _connect_limited(
    server_name: str,
    server_config: dict[str, Any],
  ) -> Any | None:
    async with semaphore:
      return await manager._connect_or_warn(server_name, server_config)

  return await asyncio.gather(
    *(_connect_limited(server_name, server_config) for server_name, server_config in connect_jobs)
  )


async def connect_or_warn(
  manager: Any,
  name: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any | None:
  try:
    state = await manager._connect(name, config)
    manager._startup_diagnostics.pop(manager._canonical_server_name(name), None)
    return state
  except Exception as exc:
    message = str(exc).strip() or type(exc).__name__
    diagnostic = runtime.startup_failure_from_exception(exc)
    manager._set_startup_diagnostic(
      name,
      category=str(diagnostic["category"]),
      message=str(diagnostic["message"]),
      retryable=bool(diagnostic["retryable"]),
      error_type=str(diagnostic["error_type"]) if diagnostic.get("error_type") else None,
    )
    runtime.logger.warning("MCP server %s failed to connect: %s", name, message)
    return None


async def connect(
  manager: Any,
  name: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any:
  server_type = str(config.get("type", "stdio")).strip().lower()
  if server_type in runtime.streamable_http_types:
    return await manager._connect_streamable_http(name, config)
  if server_type == "stdio":
    return await manager._connect_stdio_with_retries(name, config)
  raise ValueError(f"unsupported type {server_type}")


async def connect_stdio_with_retries(
  manager: Any,
  name: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any:
  total_attempts = 1 + runtime.stdio_connect_retries()
  for attempt in range(1, total_attempts + 1):
    try:
      return await manager._connect_stdio(name, config)
    except Exception as exc:
      if attempt >= total_attempts or not runtime.is_retryable_stdio_connect_error(exc):
        raise
      delay = runtime.stdio_connect_retry_delay(attempt)
      message = str(exc).strip() or type(exc).__name__
      runtime.logger.warning(
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


async def connect_stdio(
  manager: Any,
  name: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any:
  exit_contexts: list[Any] = []
  success = False
  try:
    command = str(config.get("command", "")).strip()
    if not command:
      raise ValueError("missing command")

    args_raw = config.get("args", [])
    args = [str(arg) for arg in args_raw] if isinstance(args_raw, list) else []

    env_raw = config.get("env")
    env = runtime.build_mcp_env(env_raw if isinstance(env_raw, dict) else None)

    cwd = config.get("cwd")
    tool_prefix = str(config.get("tool_prefix", "") or "").strip()
    server_params = runtime.stdio_server_parameters_factory(
      command=command,
      args=args,
      env=env,
      cwd=cwd,
    )

    devnull = open(os.devnull, "w")
    exit_contexts.append(_SyncCloseContext(devnull.close))
    stdio_cm = runtime.stdio_client_factory(server_params, errlog=devnull)
    read_stream, write_stream = await stdio_cm.__aenter__()
    exit_contexts.append(stdio_cm)

    session = runtime.client_session_factory(read_stream, write_stream)
    await session.__aenter__()
    exit_contexts.append(session)

    state = await manager._initialize_session_state(
      name=name,
      session=session,
      exit_contexts=exit_contexts,
      tool_prefix=tool_prefix,
    )
    await manager._verify_stdio_session_stable(session)
    state.config = dict(config)
    success = True
    return state
  finally:
    if not success:
      await runtime.close_contexts(exit_contexts)


async def connect_streamable_http(
  manager: Any,
  name: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any:
  if runtime.httpx_import_error is not None:
    raise RuntimeError(f"HTTP MCP transport unavailable: {runtime.httpx_import_error}")
  if runtime.streamable_http_import_error is not None or runtime.streamable_http_client_factory is None:
    raise RuntimeError(f"HTTP MCP transport unavailable: {runtime.streamable_http_import_error}")

  exit_contexts: list[Any] = []
  success = False
  try:
    url = str(config.get("url") or "").strip()
    if not url:
      raise ValueError("missing url")

    headers_raw = config.get("headers")
    headers = runtime.build_http_headers(headers_raw if isinstance(headers_raw, dict) else None)
    timeout_seconds = float(config.get("timeout", manager._startup_timeout))
    sse_read_timeout_seconds = float(config.get("sse_read_timeout", 300))
    terminate_on_close = bool(config.get("terminate_on_close", True))
    tool_prefix = str(config.get("tool_prefix", "") or "").strip()
    auth = manager._build_http_auth(name, url, config)

    http_client = runtime.httpx_module.AsyncClient(
      headers=headers,
      timeout=runtime.httpx_module.Timeout(timeout_seconds, read=sse_read_timeout_seconds),
      auth=auth,
    )
    await http_client.__aenter__()
    exit_contexts.append(http_client)

    stream_cm = runtime.streamable_http_client_factory(
      url,
      http_client=http_client,
      terminate_on_close=terminate_on_close,
    )
    read_stream, write_stream, _get_session_id = await stream_cm.__aenter__()
    exit_contexts.append(stream_cm)

    session = runtime.client_session_factory(read_stream, write_stream)
    await session.__aenter__()
    exit_contexts.append(session)

    state = await manager._initialize_session_state(
      name=name,
      session=session,
      exit_contexts=exit_contexts,
      tool_prefix=tool_prefix,
    )
    success = True
    return state
  finally:
    if not success:
      await runtime.close_contexts(exit_contexts)


def build_http_auth(
  name: str,
  url: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any | None:
  oauth_raw = config.get("oauth")
  if not oauth_raw:
    return None
  if runtime.fastmcp_oauth_import_error is not None or runtime.fastmcp_oauth_factory is None:
    raise RuntimeError(f"OAuth MCP transport unavailable: {runtime.fastmcp_oauth_import_error}")
  if oauth_raw is True:
    oauth_config: dict[str, Any] = {}
  elif isinstance(oauth_raw, dict):
    oauth_config = dict(oauth_raw)
  else:
    raise ValueError("oauth must be true or an object")

  cache_path = oauth_config.get("cache_path")
  if cache_path is None:
    cache_dir = runtime.path_factory(
      runtime.environ.get(
        "AGENT_GATEWAY_MCP_OAUTH_CACHE_DIR",
        str(runtime.path_factory.home() / ".cache" / "agent-gateway" / "mcp-oauth"),
      )
    )
    cache_path = cache_dir / f"{runtime.safe_cache_name(name)}.json"
  storage = runtime.json_file_key_value_factory(runtime.path_factory(str(cache_path)).expanduser())
  scopes = oauth_config.get("scopes")
  callback_port = oauth_config.get("callback_port")
  return runtime.fastmcp_oauth_factory(
    mcp_url=url,
    scopes=scopes,
    client_name=str(oauth_config.get("client_name") or f"agent-gateway:{name}"),
    token_storage=storage,
    callback_port=int(callback_port) if callback_port is not None else None,
    client_metadata_url=oauth_config.get("client_metadata_url"),
    client_id=oauth_config.get("client_id"),
    client_secret=oauth_config.get("client_secret"),
  )


async def initialize_session_state(
  manager: Any,
  *,
  name: str,
  session: Any,
  exit_contexts: list[Any],
  tool_prefix: str,
  runtime: McpConnectionRuntime,
) -> Any:
  await asyncio.wait_for(session.initialize(), timeout=manager._startup_timeout)

  tools: list[Any] = []
  cursor: str | None = None
  while True:
    listed = await asyncio.wait_for(
      session.list_tools(cursor=cursor),
      timeout=manager._startup_timeout,
    )
    tools.extend(listed.tools or [])
    if listed.nextCursor is None:
      break
    cursor = listed.nextCursor

  tool_definitions: list[dict[str, Any]] = []
  for tool in tools:
    input_schema = tool.inputSchema or {"type": "object", "properties": {}}
    tool_definitions.append(
      {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": copy.deepcopy(input_schema),
      }
    )

  return runtime.server_state_factory(
    name=name,
    session=session,
    exit_contexts=exit_contexts,
    tool_definitions=tool_definitions,
    tool_names={tool["name"] for tool in tool_definitions},
    tool_prefix=tool_prefix,
  )


async def verify_stdio_session_stable(
  manager: Any,
  session: Any,
  runtime: McpConnectionRuntime,
) -> None:
  delay = runtime.stdio_connect_stabilize_delay()
  if delay > 0:
    await asyncio.sleep(delay)
  await asyncio.wait_for(
    session.list_tools(cursor=None),
    timeout=manager._startup_timeout,
  )


__all__ = [
  "McpConnectionRuntime",
  "build_http_auth",
  "connect",
  "connect_or_warn",
  "connect_startup_servers",
  "connect_stdio",
  "connect_stdio_with_retries",
  "connect_streamable_http",
  "initialize_session_state",
  "verify_stdio_session_stable",
]
