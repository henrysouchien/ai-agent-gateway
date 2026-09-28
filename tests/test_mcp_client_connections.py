import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from anyio import ClosedResourceError
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED, CallToolResult, ListToolsResult, PaginatedRequestParams, Tool

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module  # noqa: E402
import agent_gateway.mcp_client_connections as mcp_client_connections  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from agent_gateway.tool_registration import RegisteredMcpToolCompilationError  # noqa: E402
from agent_workflow_contracts.tool_registration import (  # noqa: E402
  ToolRegistrationCatalog,
)


class _ListedToolsSession:
  def __init__(self, names: list[str]) -> None:
    self.names = names

  async def initialize(self) -> object:
    return None

  async def list_tools(
    self,
    *,
    params: PaginatedRequestParams | None = None,
  ) -> ListToolsResult:
    assert params is None or params.cursor is None
    return ListToolsResult(
      tools=[
        Tool(
          name=name,
          description=f"Tool {name}",
          input_schema={"type": "object", "properties": {}},
        )
        for name in self.names
      ],
      next_cursor=None,
    )

  async def call_tool(
    self,
    name: str,
    arguments: dict[str, object],
    *,
    read_timeout_seconds: float,
    meta: dict[str, object] | None = None,
  ) -> CallToolResult:
    _ = name, arguments, read_timeout_seconds, meta
    raise AssertionError("tool calls are not used by list-tools tests")


@pytest.fixture
def reconnect_transport(monkeypatch):
  pending = deque()
  opened_contexts = []

  class Session(_ListedToolsSession):
    def __init__(self, names, generation):
      super().__init__(names)
      self.generation = generation
      self.closed = False
      self.before_call: Callable[[], Awaitable[None]] | None = None

    async def __aenter__(self):
      opened_contexts.append(self)
      return self

    async def __aexit__(self, *_args):
      self.closed = True

    async def call_tool(self, name, arguments, **kwargs):
      if self.before_call is not None:
        await self.before_call()
      assert not self.closed
      assert name in self.names
      return CallToolResult(
        content=[],
        structured_content={"tool": name, "generation": self.generation},
      )

  class StdioContext:
    def __init__(self, errlog):
      self.closed = False
      self.errlog = errlog

    async def __aenter__(self):
      opened_contexts.append(self)
      return object(), object()

    async def __aexit__(self, *_args):
      self.closed = True

  def queue(names, *, generation=1):
    session = Session(names, generation)
    pending.append(session)
    return session

  monkeypatch.setattr(
    mcp_client_module, "stdio_client",
    lambda _params, errlog: StdioContext(errlog),
  )
  monkeypatch.setattr(
    mcp_client_module, "ClientSession", lambda *_args: pending.popleft(),
  )
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_retries", lambda: 0)
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_stabilize_delay", lambda: 0)
  return queue, opened_contexts


async def _reconnect_for_future(manager, name, tool):
  return await manager._reconnect_stdio_server_for_future(
    server_name=name,
    server=manager._servers[name],
    original_name=tool,
    cause=EOFError("connection closed"),
  )


def test_reconnect_restores_tools_hidden_by_temporary_collision(reconnect_transport):
  async def scenario():
    queue, _opened_contexts = reconnect_transport
    queue(["alpha"])
    queue(["beta"])
    manager = McpClientManager(config_path=None, inline_servers={
      name: {"command": sys.executable} for name in ("first", "second")
    })
    await manager.startup()
    try:
      queue(["beta"], generation=2)
      assert await _reconnect_for_future(manager, "first", "alpha")
      result, error = await manager.call_tool("beta", {})
      assert error is None
      assert result == {"tool": "beta", "generation": 2}

      queue(["alpha"], generation=3)
      assert await _reconnect_for_future(manager, "first", "beta")
      result, error = await manager.call_tool("beta", {})
      assert error is None
      assert result == {"tool": "beta", "generation": 1}
      assert manager.get_server_for_tool("beta") == "second"
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


def test_rejected_startup_closes_all_connected_transports(reconnect_transport):
  async def scenario():
    queue, opened_contexts = reconnect_transport
    queue(["alpha"])
    queue(["beta"])
    manager = McpClientManager(
      config_path=None,
      inline_servers={
        name: {"command": sys.executable} for name in ("first", "second")
      },
      tool_registration_catalog=ToolRegistrationCatalog(declarations=(), servers=()),
    )
    try:
      with pytest.raises(RegisteredMcpToolCompilationError):
        await manager.startup()
      assert len(opened_contexts) == 4
      assert all(context.closed for context in opened_contexts)
      assert all(
        context.errlog.closed
        for context in opened_contexts
        if hasattr(context, "errlog")
      )
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


@pytest.mark.parametrize("allow_uncertain_replay", [False, True], ids=["future", "replay"])
def test_eof_and_delayed_tool_failures_keep_one_live_generation(
  reconnect_transport, allow_uncertain_replay,
):
  async def scenario():
    queue, opened_contexts = reconnect_transport
    first = queue(["alpha"])
    manager = McpClientManager(config_path=None, inline_servers={
      "first": {"command": sys.executable},
    })
    await manager.startup()
    failure_releases = [asyncio.Event(), asyncio.Event()]
    calls_entered = [asyncio.Event(), asyncio.Event()]
    call_count = 0

    async def fail_when_released():
      nonlocal call_count
      index = call_count
      call_count += 1
      calls_entered[index].set()
      await failure_releases[index].wait()
      raise ClosedResourceError()

    first.before_call = fail_when_released
    calls = []
    try:
      for entered in calls_entered:
        calls.append(asyncio.create_task(manager.call_tool(
          "alpha", {}, allow_uncertain_replay=allow_uncertain_replay,
        )))
        await entered.wait()
      second = queue(["alpha"], generation=2)
      queue(["alpha"], generation=3)
      first_server = manager._servers["first"]
      assert first_server.stdio_eof is not None
      assert first_server.stdio_receive_done is not None
      first_server.stdio_eof.set()
      first_server.stdio_receive_done.set()
      failure_releases[0].set()
      first_outcome = await calls[0]
      failure_releases[1].set()
      second_outcome = await calls[1]
      if allow_uncertain_replay:
        expected = ({"tool": "alpha", "generation": 2}, None)
        assert first_outcome == expected
        assert second_outcome == expected

      result, error = await manager.call_tool("alpha", {})
      assert error is None
      assert result == {"tool": "alpha", "generation": 2}
      assert manager._servers["first"].session is second
      assert first.closed and not second.closed
      assert len([context for context in opened_contexts if not context.closed]) == 2
    finally:
      for release in failure_releases:
        release.set()
      await asyncio.gather(*calls, return_exceptions=True)
      await manager.shutdown()
    assert all(context.closed for context in opened_contexts)

  asyncio.run(scenario())


def test_pending_retry_awaits_passive_eof_reconnect(reconnect_transport, monkeypatch):
  async def scenario():
    queue, opened_contexts = reconnect_transport
    queue(["alpha"])
    manager = McpClientManager(config_path=None, inline_servers={
      "first": {"command": sys.executable},
    })
    await manager.startup()
    server = manager._servers["first"]
    connect_started = asyncio.Event()
    connect_release = asyncio.Event()
    retry_started = asyncio.Event()
    connect = manager._connect_stdio_with_retries

    async def delayed_connect(name, config):
      connect_started.set()
      await connect_release.wait()
      return await connect(name, config)

    async def retry():
      retry_started.set()
      return await manager._retry_stdio_tool_call_after_reconnect(
        server_name="first", server=server, original_name="alpha",
        tool_input={}, meta=None, abort_event=None, timeout_seconds=1,
        cause=EOFError("connection closed"),
      )

    monkeypatch.setattr(manager, "_connect_stdio_with_retries", delayed_connect)
    pending = None
    try:
      queue(["alpha"], generation=2)
      assert server.stdio_eof is not None
      assert server.stdio_receive_done is not None
      server.stdio_eof.set()
      server.stdio_receive_done.set()
      await connect_started.wait()
      pending = asyncio.create_task(retry())
      await retry_started.wait()
      connect_release.set()
      result = await pending
      assert result is not None
      assert result.structured_content == {"tool": "alpha", "generation": 2}
      assert await manager.call_tool("alpha", {}) == (
        {"tool": "alpha", "generation": 2}, None,
      )
      assert len(opened_contexts) == 4
    finally:
      connect_release.set()
      if pending is not None:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
      await manager.shutdown()
    assert all(context.closed for context in opened_contexts)

  asyncio.run(scenario())


@pytest.mark.parametrize("allow_uncertain_replay", [False, True], ids=["future", "replay"])
def test_pending_stdio_calls_survive_passive_eof_reconnect(
  tmp_path, allow_uncertain_replay,
):
  server_script = tmp_path / "mcp_server.py"
  server_script.write_text("""
import asyncio
import json
import os
from pathlib import Path
from fastmcp import FastMCP

generation = Path("generation").read_text()
with Path("starts").open("a") as stream:
    stream.write(json.dumps({"pid": os.getpid(), "generation": generation}) + "\\n")
mcp = FastMCP("pending-read")

@mcp.tool()
async def read_generation() -> dict:
    with Path("calls").open("a") as stream:
        stream.write(generation + "\\n")
    if generation == "first":
        await asyncio.Event().wait()
    return {"generation": generation}

mcp.run()
""")
  generation_file = tmp_path / "generation"
  generation_file.write_text("first")
  calls_file = tmp_path / "calls"
  starts_file = tmp_path / "starts"

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      default_tool_timeout=2,
      inline_servers={
        "generation": {
          "command": sys.executable, "args": [str(server_script)], "cwd": str(tmp_path),
        },
      },
    )
    await manager.startup()
    calls = []
    try:
      for _ in range(2):
        calls.append(asyncio.create_task(manager.call_tool(
          "read_generation", {}, allow_uncertain_replay=allow_uncertain_replay,
        )))
      async with asyncio.timeout(10):
        while not calls_file.exists() or len(calls_file.read_text().splitlines()) < 2:
          await asyncio.sleep(0.01)
      first = json.loads(starts_file.read_text().splitlines()[0])
      generation_file.write_text("second")
      os.kill(first["pid"], signal.SIGTERM)
      outcomes = await asyncio.wait_for(asyncio.gather(*calls), timeout=10)
      if allow_uncertain_replay:
        assert outcomes == [({"generation": "second"}, None)] * 2
      else:
        for result, error in outcomes:
          assert result is None
          assert error["sub_code"] == "connection_error"
      assert await manager.call_tool("read_generation", {}) == (
        {"generation": "second"}, None,
      )
      assert [json.loads(line)["generation"] for line in starts_file.read_text().splitlines()] == [
        "first", "second",
      ]
      assert calls_file.read_text().splitlines() == (
        ["first", "first"] + ["second"] * (3 if allow_uncertain_replay else 1)
      )
    finally:
      for call in calls:
        call.cancel()
      await asyncio.gather(*calls, return_exceptions=True)
      await manager.shutdown()

  asyncio.run(scenario())


def test_idle_stdio_child_reconnects_after_generation_swap_and_rollback(tmp_path):
  server_script = tmp_path / "mcp_server.py"
  generation_file = tmp_path / "generation"
  starts = tmp_path / "starts"
  calls = tmp_path / "calls"
  server_script.write_text("""
import json
import os
from pathlib import Path
from fastmcp import FastMCP

generation = Path("generation").read_text()
identity = {"pid": os.getpid(), "generation": generation, "cwd": os.getcwd()}
with Path("starts").open("a") as stream:
    stream.write(json.dumps(identity) + "\\n")
mcp = FastMCP("generation")

@mcp.tool(name="read_" + generation)
def read_generation() -> dict:
    with Path("calls").open("a") as stream:
        stream.write(generation + "\\n")
    return identity

mcp.run()
""")
  generation_file.write_text("first")

  async def scenario():
    manager = McpClientManager(config_path=None, inline_servers={
      "generation": {
        "command": sys.executable, "args": [str(server_script)], "cwd": str(tmp_path),
      },
    })
    await manager.startup()

    async def wait_for_catalog(tool):
      async with asyncio.timeout(15):
        while manager.get_server_for_tool(tool) != "generation":
          await asyncio.sleep(0.01)

    try:
      await wait_for_catalog("read_first")
      first = json.loads(starts.read_text().splitlines()[0])
      generation_file.write_text("second")
      os.kill(first["pid"], signal.SIGTERM)
      await wait_for_catalog("read_second")
      assert manager.get_server_for_tool("read_first") is None
      first, second = [json.loads(line) for line in starts.read_text().splitlines()]
      assert second["pid"] != first["pid"]
      assert second["cwd"] == str(tmp_path)
      assert not calls.exists()

      generation_file.write_text("first")
      os.kill(second["pid"], signal.SIGTERM)
      await wait_for_catalog("read_first")
      assert manager.get_server_for_tool("read_second") is None
      first, second, restored = [json.loads(line) for line in starts.read_text().splitlines()]
      assert restored["pid"] not in {first["pid"], second["pid"]}
      assert not calls.exists()
      result, error = await manager.call_tool("read_first", {})
      assert error is None
      assert result == restored
      assert calls.read_text().splitlines() == ["first"]
    finally:
      await manager.shutdown()
    await asyncio.sleep(0.05)
    assert len(starts.read_text().splitlines()) == 3
    for line in starts.read_text().splitlines():
      with pytest.raises(ProcessLookupError):
        os.kill(json.loads(line)["pid"], 0)

  asyncio.run(scenario())


def test_eof_reconnect_never_cancels_the_task_that_opened_the_generation(tmp_path):
  """A generation's transport scopes belong to no caller.

  The caller below reconnects the server on the tool-call path, so before the
  connection layer hosted its own transports that caller entered the new
  generation's anyio task groups. When the child then died mid-call, the EOF
  watcher's close cancelled those groups' scopes, and anyio delivered the
  cancellation to the task that entered them: the caller, whose call should
  only have seen the server's transport error.
  """
  server_script = tmp_path / "mcp_server.py"
  server_script.write_text("""
import asyncio
import os
from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("held")

@mcp.tool()
async def hold() -> dict:
    with Path("calls").open("a") as stream:
        stream.write(str(os.getpid()) + "\\n")
    await asyncio.Event().wait()
    return {}

mcp.run(show_banner=False)
""")
  calls_file = tmp_path / "calls"

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      default_tool_timeout=30,
      inline_servers={
        "held": {"command": sys.executable, "args": [str(server_script)], "cwd": str(tmp_path)},
      },
    )
    await manager.startup()
    opened = manager._servers["held"]

    async def stage():
      assert await _reconnect_for_future(manager, "held", "hold")
      assert manager._servers["held"] is not opened
      outcome = await manager.call_tool("hold", {}, allow_uncertain_replay=False)
      # anyio keeps redelivering a scope's cancellation to its host task, so
      # the caller must also survive its next await.
      await asyncio.sleep(0.2)
      return outcome

    task = asyncio.create_task(stage())
    try:
      async with asyncio.timeout(15):
        while not calls_file.exists() or not calls_file.read_text().strip():
          await asyncio.sleep(0.01)
      os.kill(int(calls_file.read_text().split()[0]), signal.SIGTERM)
      result, error = await asyncio.wait_for(asyncio.shield(task), timeout=15)
      assert result is None
      assert error is not None
      assert error["sub_code"] == "connection_error"
    finally:
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)
      await manager.shutdown()

  asyncio.run(scenario())


def test_session_allowed_tools_filter_definitions_and_dispatch_surface() -> None:
  manager = McpClientManager(config_path=None)
  state = asyncio.run(manager._initialize_session_state(
    name="idea-workbench-mcp",
    session=_ListedToolsSession(["get_investment_artifact", "generic_write"]),
    exit_contexts=[],
    tool_prefix="",
    allowed_tools=("get_investment_artifact",),
  ))

  asyncio.run(manager._publish_server_states([state]))

  assert [tool["name"] for tool in state.tool_definitions] == [
    "get_investment_artifact"
  ]
  assert manager.get_server_for_tool("get_investment_artifact") == (
    "idea-workbench-mcp"
  )
  assert manager.get_server_for_tool("generic_write") is None


def test_session_allowed_tools_reject_missing_remote_definition() -> None:
  manager = McpClientManager(config_path=None)

  with pytest.raises(ValueError, match="start_quant_research"):
    asyncio.run(manager._initialize_session_state(
      name="idea-workbench-mcp",
      session=_ListedToolsSession(["get_investment_artifact"]),
      exit_contexts=[],
      tool_prefix="",
      allowed_tools=("get_investment_artifact", "start_quant_research"),
    ))


class _PagingSession:
  """List tools one page at a time; `pages=None` never stops offering a cursor."""

  def __init__(self, pages: int | None) -> None:
    self.pages = pages
    self.cursors: list[str | None] = []

  async def initialize(self) -> object:
    return None

  async def list_tools(
    self,
    *,
    params: PaginatedRequestParams | None = None,
  ) -> ListToolsResult:
    cursor = None if params is None else params.cursor
    self.cursors.append(cursor)
    page = len(self.cursors)
    last = self.pages is not None and page >= self.pages
    return ListToolsResult(
      tools=[
        Tool(
          name=f"tool_{page}",
          description=f"Tool from page {page}",
          input_schema={"type": "object", "properties": {}},
        )
      ],
      next_cursor=None if last else f"page-{page}",
    )

  async def call_tool(self, name, arguments, **kwargs) -> CallToolResult:
    raise AssertionError("pagination tests never dispatch a tool")


def test_list_tools_pagination_collects_every_page_under_the_ceiling() -> None:
  manager = McpClientManager(config_path=None)
  session = _PagingSession(pages=3)

  state = asyncio.run(manager._initialize_session_state(
    name="idea-workbench-mcp",
    session=session,
    exit_contexts=[],
    tool_prefix="",
  ))

  assert session.cursors == [None, "page-1", "page-2"]
  assert [tool["name"] for tool in state.tool_definitions] == [
    "tool_1", "tool_2", "tool_3",
  ]


def test_list_tools_pagination_ends_at_the_page_ceiling() -> None:
  manager = McpClientManager(config_path=None)
  session = _PagingSession(pages=None)

  with pytest.raises(ValueError, match="did not finish paginating list_tools") as raised:
    asyncio.run(manager._initialize_session_state(
      name="idea-workbench-mcp",
      session=session,
      exit_contexts=[],
      tool_prefix="",
    ))

  assert len(session.cursors) == mcp_client_connections._LIST_TOOLS_PAGE_CEILING
  # The endless pager reaches the analyst by the carrier the layer already has.
  diagnostic = mcp_client_module._startup_failure_from_exception(raised.value)
  assert diagnostic["category"] == "startup_error"
  assert diagnostic["retryable"] is False


def test_catalog_republication_preserves_other_server_prefix() -> None:
  manager = McpClientManager(config_path=None)

  async def initialize(name, tools, prefix=""):
    return await manager._initialize_session_state(
      name=name,
      session=_ListedToolsSession(tools),
      exit_contexts=[],
      tool_prefix=prefix,
    )

  async def scenario():
    first = await initialize("first", ["lookup"], "first_")
    second = await initialize("second", ["previous_read"])
    await manager._publish_server_states([first, second])
    replacement = await initialize("second", ["replacement_read"])
    await manager._publish_server_states([replacement])

    assert {tool["name"] for tool in manager.get_tool_definitions()} == {
      "first_lookup", "replacement_read",
    }
    assert manager.get_server_for_tool("first_lookup") == "first"
    assert manager.get_original_tool_name("first_lookup") == "lookup"
    assert manager.get_server_for_tool("previous_read") is None
    assert manager.get_server_for_tool("replacement_read") == "second"

  asyncio.run(scenario())


def test_invalid_allowed_tools_leaves_optional_server_unadvertised() -> None:
  manager = McpClientManager(
    config_path=None,
    inline_servers={
      "idea-workbench-mcp": {
        "command": "unused",
        "allowed_tools": [],
      }
    },
  )

  asyncio.run(manager.startup())

  assert manager.get_server_names() == set()
  assert manager.get_startup_diagnostics()["idea-workbench-mcp"]["category"] == (
    "invalid_config"
  )


def test_failed_stdio_child_reports_bounded_stderr_tail(monkeypatch, caplog) -> None:
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_retries", lambda: 0)
  script = (
    "import sys\n"
    "for index in range(30): print(f'startup-line-{index:02}', file=sys.stderr)\n"
    "raise RuntimeError('child dependency failed')\n"
  )

  async def scenario():
    manager = McpClientManager(config_path=None)
    try:
      assert await manager._connect_or_warn(
        "broken-child", {"command": sys.executable, "args": ["-c", script]},
      ) is None
    finally:
      await manager.shutdown()

  asyncio.run(scenario())
  diagnostics = [
    record.getMessage() for record in caplog.records
    if "broken-child" in record.getMessage() and "stderr" in record.getMessage()
  ]
  assert len(diagnostics) == 1
  stderr = diagnostics[0].split("\n", 1)[1]
  assert len(stderr.splitlines()) == 20
  assert "startup-line-00" not in stderr
  assert "startup-line-29" in stderr
  assert stderr.endswith("RuntimeError: child dependency failed")


def test_failed_stdio_child_with_long_stderr_line_does_not_block(monkeypatch, caplog) -> None:
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_retries", lambda: 0)
  script = "import sys; sys.stderr.write('x' * 1_000_000 + 'fatal startup error\\n')"

  async def scenario():
    manager = McpClientManager(config_path=None)
    try:
      assert await asyncio.wait_for(manager._connect_or_warn(
        "noisy-child", {"command": sys.executable, "args": ["-c", script]},
      ), timeout=15) is None
    finally:
      await manager.shutdown()

  asyncio.run(scenario())
  diagnostic = next(
    record.getMessage() for record in caplog.records
    if "noisy-child" in record.getMessage() and "stderr" in record.getMessage()
  )
  stderr = diagnostic.split("\n", 1)[1]
  assert len(stderr) <= 8192
  assert stderr.endswith("fatal startup error")


def test_failed_stdio_child_with_surviving_descendant_shuts_down_bounded(monkeypatch, caplog) -> None:
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_retries", lambda: 0)
  # The child leaves a descendant holding the inherited stderr pipe well past
  # every cleanup deadline, so the drain never sees EOF.
  script = (
    "import os, subprocess, sys\n"
    "devnull = os.open(os.devnull, os.O_RDWR)\n"
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],"
    " stdin=devnull, stdout=devnull)\n"
    "print('child fatal: dependency missing', file=sys.stderr, flush=True)\n"
    "raise SystemExit(3)\n"
  )

  async def scenario():
    manager = McpClientManager(config_path=None)
    try:
      assert await manager._connect_or_warn(
        "orphan-child", {"command": sys.executable, "args": ["-c", script]},
      ) is None
    finally:
      await manager.shutdown()

  started = time.monotonic()
  # asyncio.run also joins the default executor, so a cleanup that abandoned an
  # uncancellable thread join there shows up in this elapsed time.
  asyncio.run(scenario())
  elapsed = time.monotonic() - started

  assert elapsed < 4.0, f"connection cleanup outlived its timeout: {elapsed:.2f}s"
  diagnostic = next(
    record.getMessage() for record in caplog.records
    if "orphan-child" in record.getMessage() and "stderr" in record.getMessage()
  )
  assert diagnostic.endswith("child fatal: dependency missing")


def test_stdio_stability_probe_rejects_closed_transport(reconnect_transport):
  async def scenario():
    queue, opened_contexts = reconnect_transport
    session = queue(["alpha"])
    calls = 0

    async def list_tools(*, params: PaginatedRequestParams | None = None):
      nonlocal calls
      calls += 1
      if calls > 1:
        raise MCPError(code=CONNECTION_CLOSED, message="Connection closed")
      return ListToolsResult(tools=[Tool(name="alpha", input_schema={"type": "object"})])

    session.list_tools = list_tools
    manager = McpClientManager(config_path=None, inline_servers={
      "unstable": {"command": sys.executable},
    })
    try:
      await manager.startup()
      assert manager.get_server_for_tool("alpha") is None
      assert manager.get_startup_diagnostics()["unstable"]["category"] == "transient_transport"
      assert calls == 2
      assert all(context.closed for context in opened_contexts)
    finally:
      await manager.shutdown()

  asyncio.run(scenario())
