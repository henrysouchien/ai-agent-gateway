from __future__ import annotations

import asyncio
import copy
import errno
import logging
import os
import threading
import time
import uuid
from concurrent import futures
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, MutableMapping, Protocol, Sequence

from anyio import EndOfStream
import httpx2
import psutil
from mcp.types import PaginatedRequestParams

from .claim_mint_client import mint_gateway_user_claim
from .claim_signing_authority import UserClaimHttpAuth, agent_api_claim_http_headers
from .mcp_client_config import expand_http_template, iter_exception_tree
from .mcp_client_oauth_storage import oauth_token_cache_path
from .mcp_client_errors import McpStdioStartupBoundError, StdioStartupOutcome


class _RemoteUserClaimHttpAuth(UserClaimHttpAuth):
  """Keep remote claim I/O on the async request path only."""

  def __init__(
    self, mint_claim: Callable[[], Awaitable[Mapping[str, str]]],
  ) -> None:
    super().__init__(self._require_async)
    self._mint_remote_claim = mint_claim

  @staticmethod
  def _require_async() -> Mapping[str, str]:
    raise RuntimeError("remote claim mint requires asynchronous HTTP")

  async def async_auth_flow(
    self, request: httpx2.Request,
  ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
    request.headers.update(
      agent_api_claim_http_headers(await self._mint_remote_claim())
    )
    yield request


def _remote_user_claim_http_auth(
  scoped_env: Mapping[str, str],
) -> UserClaimHttpAuth | None:
  if "AGENT_API_CLAIM_MINT_URL" not in scoped_env:
    return None
  mint_url = scoped_env["AGENT_API_CLAIM_MINT_URL"]
  if not mint_url.strip():
    raise ValueError("AGENT_API_CLAIM_MINT_URL must not be empty")
  api_key = scoped_env.get("GATEWAY_API_KEY")
  if api_key is None or not api_key.strip():
    raise ValueError("remote claim mint requires GATEWAY_API_KEY")

  async def mint_claim() -> dict[str, str]:
    return await mint_gateway_user_claim(base_url=mint_url, api_key=api_key)

  return _RemoteUserClaimHttpAuth(mint_claim)


#: Bounded backoff between loopback reconnect attempts of one idempotent
#: call. A loopback MCP endpoint that is transiently rebinding refuses the
#: connect (ECONNREFUSED) before any request bytes are sent, so replaying an
#: idempotent call inside this window cannot duplicate an effect; the
#: window is a measurement of a local rebind, never a deadline on the work.
LOOPBACK_REFUSAL_RETRY_DELAYS_SECONDS = (0.05, 0.15, 0.45)


def is_loopback_connection_refused(exc: BaseException) -> bool:
  """Whether ``exc`` is a loopback connect refused before dispatch."""

  seen: set[int] = set()
  pending: list[BaseException] = [exc]
  while pending:
    candidate = pending.pop()
    if id(candidate) in seen:
      continue
    seen.add(id(candidate))
    if isinstance(candidate, ConnectionRefusedError):
      return True
    if isinstance(candidate, OSError) and candidate.errno == errno.ECONNREFUSED:
      return True
    for child in iter_exception_tree(candidate):
      if child is not candidate:
        pending.append(child)
    for linked in (candidate.__cause__, candidate.__context__):
      if isinstance(linked, BaseException):
        pending.append(linked)
  return False


log = logging.getLogger("agent_gateway.mcp_client")


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
    read_timeout_seconds: float | None,
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
  build_mcp_env: Callable[..., dict[str, str]]
  # Raises when the configured stdio executable decisively cannot exist in the
  # spawn env; a no-op for anything it cannot decisively resolve.
  preflight_stdio_executable: Callable[[str, Sequence[str], Mapping[str, str]], None]
  build_http_headers: Callable[..., dict[str, str]]
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

#: Set in each stdio child's environment to a value unique to that spawn. The
#: SDK's stdio transport does not expose the process it starts, so the startup
#: bound finds the child among this process's children by this value.
STDIO_SPAWN_MARKER_ENV = "AGENT_GATEWAY_MCP_SPAWN_ID"

# CPU seconds a starting stdio child and its descendants must spend inside one
# startup window to count as working. An interpreter blocked on a read spends
# none; an import under heavy machine load still spends far more than this per
# window, while a periodic timer in an otherwise stuck server stays below it.
_STDIO_STARTUP_PROGRESS_CPU_SECONDS = 0.1

# A working stdio child gets further startup windows until this many startup
# bounds have passed since its spawn (360 s at the product's 120 s bound,
# inside the 420 s readiness wait of a deploy). It ends a child that computes
# without ever answering.
_STDIO_STARTUP_CEILING_BOUNDS = 3


@dataclass(frozen=True)
class _StdioChildSample:
  state: Literal["alive", "died", "unobserved"]
  pid: int | None
  cpu_seconds: float = 0.0


class StdioChildProgress:
  """Whether one starting stdio child is working, measured as the CPU it spends.

  Created at the spawn; the startup bound samples it when a step starts and
  when the step's window ends. Wall seconds alone cannot tell a child importing
  on a saturated machine from one that will never answer.
  """

  def __init__(self, spawn_id: str, *, spawned_wall: float, ceiling_at: float, ceiling_seconds: float) -> None:
    self._spawn_id = spawn_id
    self._spawned_wall = spawned_wall
    self.ceiling_at = ceiling_at
    self.ceiling_seconds = ceiling_seconds
    self._process: psutil.Process | None = None

  def _find(self) -> psutil.Process | None:
    if self._process is not None:
      return self._process
    for child in psutil.Process().children():
      try:
        # The clock tick of a process start time is coarse; a second of slack.
        if child.create_time() < self._spawned_wall - 1.0:
          continue
        if child.environ().get(STDIO_SPAWN_MARKER_ENV) == self._spawn_id:
          self._process = child
          return child
      except psutil.Error:
        continue
    return None

  def sample(self) -> _StdioChildSample:
    """Blocking: the child's and its descendants' CPU seconds, or why there are none."""
    process = self._find()
    if process is None:
      return _StdioChildSample("unobserved", None)
    try:
      if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
        return _StdioChildSample("died", process.pid)
      times = process.cpu_times()
      total = times.user + times.system + times.children_user + times.children_system
      for descendant in process.children(recursive=True):
        try:
          descendant_times = descendant.cpu_times()
        except psutil.Error:
          continue
        total += descendant_times.user + descendant_times.system
    except psutil.NoSuchProcess:
      return _StdioChildSample("died", process.pid)
    except psutil.Error:
      return _StdioChildSample("unobserved", process.pid)
    return _StdioChildSample("alive", process.pid, total)


def _stdio_startup_failure(
  outcome: StdioStartupOutcome,
  pid: int | None,
  *,
  bound: float,
  ceiling: float,
) -> str:
  if outcome == "stalled":
    return f"stdio child pid {pid} alive but stalled (no CPU progress in its last {bound:g}s window)"
  if outcome == "died":
    return f"stdio child pid {pid} died"
  if outcome == "busy_past_ceiling":
    return f"stdio child pid {pid} still working at the {ceiling:g}s startup ceiling"
  return f"startup resource bound {bound:g}s; stdio child process not observed"



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
  started = time.monotonic()
  try:
    state = await manager._connect(name, config)
    runtime.logger.info(
      "MCP server %s connected after %.3fs", name, time.monotonic() - started,
    )
    manager._startup_diagnostics.pop(manager._canonical_server_name(name), None)
    manager._stdio_startup_failures.pop(name, None)
    return state
  except Exception as exc:
    elapsed = time.monotonic() - started
    message = ": ".join(
      str(cause).strip() or type(cause).__name__
      for cause in iter_exception_tree(exc)
    )
    diagnostic = runtime.startup_failure_from_exception(exc)
    manager._set_startup_diagnostic(
      name,
      category=str(diagnostic["category"]),
      message=(
        f"MCP server {name} startup failed after {elapsed:.3f}s: "
        f"{message}"
      ),
      retryable=bool(diagnostic["retryable"]),
      error_type=str(diagnostic["error_type"]) if diagnostic.get("error_type") else None,
    )
    if str(config.get("type", "stdio")).strip().lower() == "stdio":
      # The one owner that restores a stdio child holds its prepared config:
      # its startup-recovery task reconnects it off every call's path.
      manager._stdio_startup_failures[name] = config
    runtime.logger.warning(
      "MCP server %s failed to connect after %.3fs: %s",
      name, elapsed, message,
    )
    return None


async def connect(
  manager: Any,
  name: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any:
  server_type = str(config.get("type", "stdio")).strip().lower()
  if server_type in runtime.streamable_http_types:
    # A per-user server's startup connection only lists its catalog; the
    # per-user connections that serve a user's calls are made per user.
    projected_raw = config.get("_projected_env")
    projected = projected_raw if isinstance(projected_raw, Mapping) else None
    scoped_env, _ = manager._scoped_server_env(
      name,
      config,
      deployment_environ=runtime.environ,
      projected_env=projected,
    )
    user_claim_auth = _remote_user_claim_http_auth(scoped_env)
    if user_claim_auth is None and config.get("per_user") is True:
      user_claim_auth = manager._catalog_claim_auth()
    return await manager._connect_streamable_http(
      name,
      config,
      user_claim_auth=user_claim_auth,
    )
  if server_type == "stdio":
    return await manager._connect_stdio_with_retries(name, config)
  raise ValueError(f"unsupported type {server_type}")


async def connect_stdio_with_retries(
  manager: Any,
  name: str,
  config: dict[str, Any],
  runtime: McpConnectionRuntime,
) -> Any:
  # A stdio child that dies during startup (broken pipe, EOF) is a flapping
  # launch, not a missing server; the count bounds a launch that never settles
  # so one dead command cannot hold startup open. It is a measurement of
  # effort, not a verdict: every attempt is logged below with its number and
  # the last attempt's real error reaches `connect_or_warn` unchanged.
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
      unwound = await self._close_contexts(exit_contexts)
      if isinstance(exc, asyncio.CancelledError):
        # A transport's task group cancels the task that entered it when its
        # own task fails (a refused connection), and reports that failure only
        # when the group is exited. A cancellation whose exit surfaced a
        # failure is that failure, not a caller's cancel.
        if unwound is None:
          opened.cancel()
        else:
          opened.set_exception(unwound)
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
    projected_raw = config.get("_projected_env")
    projected = projected_raw if isinstance(projected_raw, Mapping) else None
    scoped_env, absent_env = manager._scoped_server_env(
      name,
      config,
      deployment_environ=runtime.environ,
      projected_env=projected,
    )
    env = runtime.build_mcp_env(
      env_raw if isinstance(env_raw, dict) else None,
      environ=scoped_env,
      base_environ=runtime.environ,
      absent=absent_env,
    )
    runtime.preflight_stdio_executable(command, args, env)
    spawn_id = uuid.uuid4().hex
    env[STDIO_SPAWN_MARKER_ENV] = spawn_id

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
    spawned_wall = time.time()
    ceiling_seconds = _STDIO_STARTUP_CEILING_BOUNDS * manager._startup_timeout
    read_stream, write_stream = await stdio_cm.__aenter__()
    exit_contexts.append(stdio_cm)
    child = StdioChildProgress(
      spawn_id,
      spawned_wall=spawned_wall,
      ceiling_at=time.monotonic() + ceiling_seconds,
      ceiling_seconds=ceiling_seconds,
    )
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
      stdio_child=child,
    )
    await manager._verify_stdio_session_stable(session, stdio_child=child)
    state.config = {k: v for k, v in config.items() if k != "_projected_env"}
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
  *,
  user_claim_auth: Any | None = None,
) -> Any:
  """Connect one streamable-HTTP server.

  `user_claim_auth` is the per-request user-claim `httpx.Auth` of a per-user
  connection; it is the connection's only authentication (a per-user entry
  cannot also configure OAuth).
  """
  host = _TransportHost(runtime.close_contexts)

  async def open_streamable_http(exit_contexts: list[Any]) -> Any:
    configured_url = str(config.get("url") or "").strip()
    if not configured_url:
      raise ValueError("missing url")

    headers_raw = config.get("headers")
    projected_raw = config.get("_projected_env")
    projected = projected_raw if isinstance(projected_raw, Mapping) else None
    scoped_env, _ = manager._scoped_server_env(
      name,
      config,
      deployment_environ=runtime.environ,
      projected_env=projected,
    )
    url = expand_http_template(configured_url, environ=scoped_env)
    auth = _remote_user_claim_http_auth(scoped_env)
    if auth is None:
      auth = (
        user_claim_auth
        if user_claim_auth is not None
        else manager._build_http_auth(name, url, config)
      )
    headers = runtime.build_http_headers(
      headers_raw if isinstance(headers_raw, dict) else None,
      environ=scoped_env,
    )
    timeout_seconds = float(config.get("timeout", manager._startup_timeout))
    # GP19/GA-N2: no clock may end an in-flight call, transport included; the
    # configured seconds are an expectation logged once when exceeded, and the
    # SDK read_timeout is None at dispatch. Connect/write/pool keep the clock:
    # a response that never starts is a connection, not a slow call.
    sse_read_expected_seconds = float(config.get("sse_read_timeout", 300))
    terminate_on_close = bool(config.get("terminate_on_close", True))
    tool_prefix = str(config.get("tool_prefix", "") or "").strip()

    log.info(
      "streamable-http for %s: no socket-read deadline; read expectation %gs",
      name,
      sse_read_expected_seconds,
    )
    http_client = runtime.httpx_module.AsyncClient(
      headers=headers,
      timeout=runtime.httpx_module.Timeout(timeout_seconds, read=None),
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

    state = await manager._initialize_session_state(
      name=name,
      session=session,
      exit_contexts=[host],
      tool_prefix=tool_prefix,
      allowed_tools=runtime.parse_allowed_tools(config.get("allowed_tools")),
    )
    state.config = dict(config)
    return state

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
    cache_path = oauth_token_cache_path(
      cache_dir,
      name,
      credential_principal=str(oauth_config.get("credential_principal") or ""),
      credential_owner=(
        None if oauth_config.get("credential_owner") is None
        else str(oauth_config.get("credential_owner"))
      ),
    )
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


async def _await_while_child_works(
  request: asyncio.Future[Any],
  child: StdioChildProgress,
  bound: float,
  name: str | None,
) -> tuple[StdioStartupOutcome, int | None] | None:
  """Wait for ``request`` while the child keeps spending CPU.

  Returns None once the request is done, or why the wait ended with its pid:
  a window of ``bound`` seconds in which the child spent no CPU, a child gone
  or not found, or one still working at its ceiling. The request is cancelled
  and settled before any return or exception leaves.
  """
  window_end = time.monotonic() + bound
  baseline = await asyncio.to_thread(child.sample)
  try:
    while True:
      remaining = window_end - time.monotonic()
      if remaining > 0:
        await asyncio.wait({request}, timeout=remaining)
      if request.done():
        return None
      current = await asyncio.to_thread(child.sample)
      if request.done():
        return None
      if current.state != "alive":
        return current.state, current.pid
      spent = current.cpu_seconds - (baseline.cpu_seconds if baseline.state == "alive" else 0.0)
      if spent < _STDIO_STARTUP_PROGRESS_CPU_SECONDS:
        return "stalled", current.pid
      now = time.monotonic()
      if now >= child.ceiling_at:
        return "busy_past_ceiling", current.pid
      log.info(
        "MCP stdio server %s (pid %s) is still starting and spent %.2fs CPU in its last "
        "%gs window; extending its startup window (ceiling %gs after spawn)",
        name, current.pid, spent, bound, child.ceiling_seconds,
      )
      baseline = current
      window_end = min(now + bound, child.ceiling_at)
  finally:
    if not request.done():
      request.cancel()
      await asyncio.wait({request})


async def _bounded_startup_step(
  manager: Any,
  step: str,
  awaitable: Any,
  *,
  name: str | None = None,
  stdio_child: StdioChildProgress | None = None,
) -> Any:
  """Contain an unanswered handshake that otherwise holds startup and shutdown's lock.

  The step is bounded by ``manager._startup_timeout``. For a stdio child the
  bound is a window of no progress, not wall seconds: a child that spent CPU in
  the window gets another, up to ``stdio_child.ceiling_at``; a child that is
  idle, gone or unobservable ends the step at the bound with
  ``McpStdioStartupBoundError`` naming which.
  """
  started = time.monotonic()
  request = asyncio.ensure_future(awaitable)
  server = f"MCP server {name}" if name is not None else "MCP server"
  bound = manager._startup_timeout
  try:
    if stdio_child is None:
      return await asyncio.wait_for(request, timeout=bound)
    ended = await _await_while_child_works(request, stdio_child, bound, name)
    if ended is None:
      return request.result()
  except asyncio.TimeoutError as exc:
    elapsed = time.monotonic() - started
    # A peer's own TimeoutError is its cause, not our no-response diagnosis.
    cause = (
      f"no response (startup resource bound {bound:g}s)"
      if request.cancelled() else str(exc).strip() or type(exc).__name__
    )
    raise TimeoutError(
      f"{server} startup {step} failed after {elapsed:.3f}s: {cause}"
    ) from exc
  except Exception as exc:
    log.warning(
      "%s startup %s failed after %.3fs: %s",
      server, step, time.monotonic() - started, str(exc).strip() or type(exc).__name__,
    )
    raise
  outcome, pid = ended
  detail = _stdio_startup_failure(
    outcome, pid, bound=bound, ceiling=stdio_child.ceiling_seconds,
  )
  raise McpStdioStartupBoundError(
    f"{server} startup {step} failed after {time.monotonic() - started:.3f}s: "
    f"no response; {detail}",
    server=name,
    step=step,
    outcome=outcome,
    pid=pid,
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
  stdio_child: StdioChildProgress | None = None,
) -> Any:
  await _bounded_startup_step(
    manager, "initialize", session.initialize(), name=name, stdio_child=stdio_child,
  )

  tools: list[McpListedTool] = []
  cursor: str | None = None
  # Bounded by construction: the `wait_for` below bounds one request, never the
  # walk, so a server that answers every page with a fresh cursor used to be
  # enumerated forever here — withholding every other server's catalog and
  # holding the manager lock against shutdown for good.
  for _page in range(_LIST_TOOLS_PAGE_CEILING):
    listed = await _bounded_startup_step(
      manager,
      "list_tools",
      session.list_tools(params=PaginatedRequestParams(cursor=cursor)),
      name=name,
      stdio_child=stdio_child,
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
  code_dispositions: dict[str, str | None] = {}
  for tool in tools:
    meta = tool.meta or {}
    disposition: str | None = None
    served_disposition = meta.get("hank/code-disposition")
    reason = meta.get("hank/code-reason")
    if "hank/code-disposition" in meta or "hank/code-reason" in meta:
      if (
        not isinstance(served_disposition, str)
        or served_disposition not in {
          "pure_local", "read_composable", "artifact_composable", "write_composable",
          "tool_only", "internal_only", "unclassified",
        }
        or not isinstance(reason, str)
        or not reason.strip()
      ):
        runtime.logger.warning(
          "Malformed code disposition for MCP tool %s/%s; retaining existing policy coverage",
          name, tool.name,
        )
      elif served_disposition != "unclassified":
        disposition = served_disposition
    code_dispositions[tool.name] = disposition
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
    code_dispositions=code_dispositions,
    tool_prefix=tool_prefix,
  )


async def verify_stdio_session_stable(
  manager: Any,
  session: McpConnectionSession,
  runtime: McpConnectionRuntime,
  stdio_child: StdioChildProgress | None = None,
) -> None:
  delay = runtime.stdio_connect_stabilize_delay()
  if delay > 0:
    await asyncio.sleep(delay)
  await _bounded_startup_step(
    manager,
    "stability check",
    session.list_tools(params=PaginatedRequestParams()),
    stdio_child=stdio_child,
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
  "STDIO_SPAWN_MARKER_ENV",
  "StdioChildProgress",
  "LOOPBACK_REFUSAL_RETRY_DELAYS_SECONDS",
  "is_loopback_connection_refused",
  "verify_stdio_session_stable",
]
