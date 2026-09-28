from __future__ import annotations

import asyncio
import copy
import os
import threading
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence

from anyio import EndOfStream
from mcp.types import PaginatedRequestParams


class _StdioReadStream:
  """Observe peer EOF without consuming messages ahead of ClientSession."""

  def __init__(self, stream: Any) -> None:
    self._stream = stream
    self.eof = asyncio.Event()
    self.receive_done = asyncio.Event()

  async def __aenter__(self) -> _StdioReadStream:
    await self._stream.__aenter__()
    return self

  async def __aexit__(self, *args: Any) -> None:
    try:
      await self._stream.__aexit__(*args)
    finally:
      # The SDK dispatcher synchronously wakes pending requests after this exits,
      # before the reconnect task waiting on receive_done can resume.
      self.receive_done.set()

  def __aiter__(self) -> _StdioReadStream:
    return self

  async def __anext__(self) -> Any:
    try:
      return await self.receive()
    except EndOfStream:
      raise StopAsyncIteration from None

  async def receive(self) -> Any:
    try:
      return await self._stream.receive()
    except EndOfStream:
      self.eof.set()
      raise

  async def aclose(self) -> None:
    await self._stream.aclose()



class McpToolCallResult(Protocol):
  @property
  def is_error(self) -> bool: ...

  @property
  def content(self) -> object: ...

  @property
  def structured_content(self) -> object | None: ...


class McpListedTool(Protocol):
  @property
  def name(self) -> str: ...

  @property
  def description(self) -> str | None: ...

  @property
  def input_schema(self) -> Mapping[str, object] | None: ...

  @property
  def meta(self) -> Mapping[str, object] | None: ...


class McpListToolsResult(Protocol):
  @property
  def tools(self) -> Sequence[McpListedTool] | None: ...

  @property
  def next_cursor(self) -> str | None: ...


class McpClientSession(Protocol):
  async def call_tool(
    self,
    name: str,
    arguments: dict[str, object],
    *,
    read_timeout_seconds: float,
    meta: dict[str, object] | None = None,
  ) -> McpToolCallResult: ...


class _McpCallableServerState(Protocol):
  session: McpClientSession


class McpConnectionSession(McpClientSession, Protocol):
  async def initialize(self) -> object: ...

  async def list_tools(
    self,
    *,
    params: PaginatedRequestParams | None = None,
  ) -> McpListToolsResult: ...


@dataclass(frozen=True)
class McpConnectionRuntime:
  startup_concurrency_limit: Callable[[], int]
  startup_failure_from_exception: Callable[[BaseException], dict[str, Any]]
  streamable_http_types: set[str]
  stdio_connect_retries: Callable[[], int]
  stdio_connect_retry_delay: Callable[[int], float]
  stdio_connect_stabilize_delay: Callable[[], float]
  is_retryable_stdio_startup_error: Callable[[BaseException], bool]
  build_mcp_env: Callable[[dict[str, Any] | None], dict[str, str]]
  # Raises when the configured stdio executable decisively cannot exist in the
  # spawn env; a no-op for anything it cannot decisively resolve.
  preflight_stdio_executable: Callable[[str, Sequence[str], Mapping[str, str]], None]
  build_http_headers: Callable[[dict[str, Any] | None], dict[str, str]]
  parse_allowed_tools: Callable[[Any], tuple[str, ...] | None]
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
  environ: MutableMapping[str, str]
  logger: Any


# Our write end is closed before the drain is awaited, so a drain still running
# after this grace period means a surviving child descendant holds the stderr
# pipe open and no EOF is coming. The wait has to expire well inside the
# caller's context-close timeout (_MCP_CLOSE_TIMEOUT_SECONDS) so shutdown stays
# bounded and the stderr tail still gets logged.
_STDERR_DRAIN_GRACE_SECONDS = 1.0

# How many `list_tools` pages one server's catalog may take before the walk in
# initialize_session_state gives up. A mechanical I/O ceiling on a client loop,
# never a wall-clock deadline on the work behind it. A client cannot know the
# page size a server picks, so the ceiling is set where it cannot reject a real
# catalog: all 14 servers this product configures return their whole catalog in
# one page, the largest 67 tools (portfolio-reads-mcp, measured live
# 2026-09-22), and 256 pages admits four times that even from a server perverse
# enough to page one tool at a time. It exists to end a server that pages
# without end, not to size a catalog.
_LIST_TOOLS_PAGE_CEILING = 256


class _StdioStderr:
  """Drain child stderr without blocking it or retaining an unbounded log."""

  def __init__(
    self,
    name: str,
    logger: Any,
    *,
    drain_grace_seconds: float = _STDERR_DRAIN_GRACE_SECONDS,
  ) -> None:
    read_fd, write_fd = os.pipe()
    self.errlog = os.fdopen(write_fd, "w", encoding="utf-8")
    self._reader = os.fdopen(read_fd, "rb", buffering=0)
    self._tail = bytearray()
    self._name = name
    self._logger = logger
    self.failed = True
    self._drain_grace_seconds = drain_grace_seconds
    # Signalled by the reader thread instead of joining it: a join is neither
    # cancellable nor interruptible, and an abandoned one parks a shared
    # executor worker that the event loop waits for at shutdown.
    self._drained: futures.Future[None] = futures.Future()
    self._drained.set_running_or_notify_cancel()
    self._thread = threading.Thread(target=self._drain, daemon=True)
    self._thread.start()

  def _drain(self) -> None:
    try:
      with self._reader:
        while chunk := self._reader.read(4096):
          self._tail.extend(chunk)
          del self._tail[:-8192]
    finally:
      self._drained.set_result(None)

  async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    self.errlog.close()
    try:
      await asyncio.wait_for(
        asyncio.wrap_future(self._drained),
        timeout=self._drain_grace_seconds,
      )
    except asyncio.TimeoutError:
      self._logger.debug(
        "MCP stdio server %s stderr drain still open after %.1fs; "
        "a child descendant holds the pipe, reporting the tail read so far",
        self._name,
        self._drain_grace_seconds,
      )
    if self.failed and self._tail:
      tail = "\n".join(bytes(self._tail).decode("utf-8", errors="replace").splitlines()[-20:])
      self._logger.warning(
        "MCP stdio server %s failed to connect; stderr (last 20 lines, up to 8192 bytes):\n%s",
        self._name,
        tail,
      )


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
      if attempt >= total_attempts or not runtime.is_retryable_stdio_startup_error(exc):
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


class _TransportHost:
  """One task enters a connection's transport contexts and the same task exits them.

  The SDK's `stdio_client`, `streamable_http_client` and `ClientSession` each
  enter an anyio task group. anyio delivers a group's cancellation to the task
  that entered it, and only that task can exit it. A connection opened in a
  caller's task would hand that caller, for the connection's whole life, a
  cancellation aimed at whatever it is doing when an EOF reconnect, a
  replacement or shutdown later closes the connection from another task —
  a pipeline stage or a parent turn, cancelled by a reconnect it never asked
  for. This host's task opens the contexts, waits to be closed, and exits them
  itself; a connection's `exit_contexts` holds only the host.
  """

  def __init__(self, close_contexts: Callable[[list[Any]], Any]) -> None:
    self._close_contexts = close_contexts
    self._close_requested = asyncio.Event()
    self._task: asyncio.Task[None] | None = None

  async def open(self, opener: Callable[[list[Any]], Any]) -> Any:
    """Run `opener(exit_contexts)` in the host task and return its result.

    A caller that stops waiting only asks for the close: an open still in
    flight runs to its own end in the host task, which then exits whatever it
    entered.
    """
    opened: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    opened.add_done_callback(_retrieve_exception)
    task = asyncio.create_task(self._run(opener, opened))
    # The loop holds tasks weakly; an abandoned open still has to finish and
    # exit what it entered.
    _HOST_TASKS.add(task)
    task.add_done_callback(_HOST_TASKS.discard)
    self._task = task
    try:
      return await asyncio.shield(opened)
    except BaseException:
      self._close_requested.set()
      raise

  async def _run(self, opener: Callable[[list[Any]], Any], opened: asyncio.Future[Any]) -> None:
    exit_contexts: list[Any] = []
    try:
      opened.set_result(await opener(exit_contexts))
    except BaseException as exc:
      await self._close_contexts(exit_contexts)
      if isinstance(exc, asyncio.CancelledError):
        opened.cancel()
      else:
        opened.set_exception(exc)
      return
    try:
      await self._close_requested.wait()
    finally:
      await self._close_contexts(exit_contexts)

  async def __aexit__(self, *_exc_info: Any) -> None:
    self._close_requested.set()
    assert self._task is not None
    # Waiting, never awaiting the task itself: a caller's close deadline must
    # not cancel the host out of its own teardown.
    await asyncio.wait({self._task})


def _retrieve_exception(future: asyncio.Future[Any]) -> None:
  """Mark an open's failure retrieved when its caller already stopped waiting."""
  if not future.cancelled():
    future.exception()


_HOST_TASKS: set[asyncio.Task[None]] = set()


async def connect_stdio(
  manager: Any,
  name: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any:
  host = _TransportHost(runtime.close_contexts)

  async def open_stdio(exit_contexts: list[Any]) -> Any:
    command = str(config.get("command", "")).strip()
    if not command:
      raise ValueError("missing command")

    args_raw = config.get("args", [])
    args = [str(arg) for arg in args_raw] if isinstance(args_raw, list) else []

    env_raw = config.get("env")
    env = runtime.build_mcp_env(env_raw if isinstance(env_raw, dict) else None)
    runtime.preflight_stdio_executable(command, args, env)

    cwd = config.get("cwd")
    tool_prefix = str(config.get("tool_prefix", "") or "").strip()
    server_params = runtime.stdio_server_parameters_factory(
      command=command,
      args=args,
      env=env,
      cwd=cwd,
    )

    stderr = _StdioStderr(name, runtime.logger)
    exit_contexts.append(stderr)
    stdio_cm = runtime.stdio_client_factory(server_params, errlog=stderr.errlog)
    read_stream, write_stream = await stdio_cm.__aenter__()
    exit_contexts.append(stdio_cm)
    read_stream = _StdioReadStream(read_stream)

    session = runtime.client_session_factory(read_stream, write_stream)
    await session.__aenter__()
    exit_contexts.append(session)

    state = await manager._initialize_session_state(
      name=name,
      session=session,
      exit_contexts=[host],
      tool_prefix=tool_prefix,
      allowed_tools=runtime.parse_allowed_tools(config.get("allowed_tools")),
    )
    await manager._verify_stdio_session_stable(session)
    state.config = dict(config)
    state.stdio_eof = read_stream.eof
    state.stdio_receive_done = read_stream.receive_done
    stderr.failed = False
    return state

  return await host.open(open_stdio)


async def connect_streamable_http(
  manager: Any,
  name: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any:
  host = _TransportHost(runtime.close_contexts)

  async def open_streamable_http(exit_contexts: list[Any]) -> Any:
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
    read_stream, write_stream = await stream_cm.__aenter__()
    exit_contexts.append(stream_cm)

    session = runtime.client_session_factory(read_stream, write_stream)
    await session.__aenter__()
    exit_contexts.append(session)

    return await manager._initialize_session_state(
      name=name,
      session=session,
      exit_contexts=[host],
      tool_prefix=tool_prefix,
      allowed_tools=runtime.parse_allowed_tools(config.get("allowed_tools")),
    )

  return await host.open(open_streamable_http)


def build_http_auth(
  name: str,
  url: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any | None:
  oauth_raw = config.get("oauth")
  if not oauth_raw:
    return None
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
  session: McpConnectionSession,
  exit_contexts: list[Any],
  tool_prefix: str,
  allowed_tools: tuple[str, ...] | None,
  runtime: McpConnectionRuntime,
) -> Any:
  await asyncio.wait_for(session.initialize(), timeout=manager._startup_timeout)

  tools: list[McpListedTool] = []
  cursor: str | None = None
  # Bounded by construction: the `wait_for` below bounds one request, never the
  # walk, so a server that answers every page with a fresh cursor used to be
  # enumerated forever here — withholding every other server's catalog and
  # holding the manager lock against shutdown for good.
  for _page in range(_LIST_TOOLS_PAGE_CEILING):
    listed = await asyncio.wait_for(
      session.list_tools(params=PaginatedRequestParams(cursor=cursor)),
      timeout=manager._startup_timeout,
    )
    tools.extend(listed.tools or [])
    if listed.next_cursor is None:
      break
    cursor = listed.next_cursor
  else:
    # Reached only when the ceiling ran out with a cursor still pending: a
    # protocol fault of the same class as the allowed_tools violation below,
    # and it reaches the analyst by the same carrier.
    raise ValueError(
      "MCP server did not finish paginating list_tools within "
      f"{_LIST_TOOLS_PAGE_CEILING} pages"
    )

  # Carry audience metadata with the candidate until its catalog is accepted.
  # It must never enter Anthropic/OpenAI tool definitions.
  tool_metadata = {str(tool.name): tool.meta for tool in tools}

  advertised_names = {str(tool.name) for tool in tools}
  if allowed_tools is not None:
    missing = [name for name in allowed_tools if name not in advertised_names]
    if missing:
      raise ValueError(
        "MCP server did not advertise configured allowed_tools: "
        + ", ".join(missing)
      )
    allowed_names = frozenset(allowed_tools)
    tools = [tool for tool in tools if str(tool.name) in allowed_names]

  tool_definitions: list[dict[str, Any]] = []
  for tool in tools:
    input_schema = tool.input_schema or {"type": "object", "properties": {}}
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
    exported_tool_names=frozenset(advertised_names),
    tool_metadata=tool_metadata,
    tool_prefix=tool_prefix,
  )


async def verify_stdio_session_stable(
  manager: Any,
  session: McpConnectionSession,
  runtime: McpConnectionRuntime,
) -> None:
  delay = runtime.stdio_connect_stabilize_delay()
  if delay > 0:
    await asyncio.sleep(delay)
  await asyncio.wait_for(
    session.list_tools(params=PaginatedRequestParams()),
    timeout=manager._startup_timeout,
  )


__all__ = [
  "McpClientSession",
  "McpConnectionSession",
  "McpListToolsResult",
  "McpListedTool",
  "McpToolCallResult",
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
