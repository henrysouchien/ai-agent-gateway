import asyncio
from collections import deque
import json
import os
import signal
import sys
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
      self.before_call = None

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
        await asyncio.wait_for(entered.wait(), timeout=1)
      second = queue(["alpha"], generation=2)
      queue(["alpha"], generation=3)
      manager._servers["first"].stdio_eof.set()
      manager._servers["first"].stdio_receive_done.set()
      failure_releases[0].set()
      first_outcome = await asyncio.wait_for(calls[0], timeout=1)
      failure_releases[1].set()
      second_outcome = await asyncio.wait_for(calls[1], timeout=1)
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
      server.stdio_eof.set()
      server.stdio_receive_done.set()
      await asyncio.wait_for(connect_started.wait(), timeout=1)
      pending = asyncio.create_task(retry())
      await asyncio.wait_for(retry_started.wait(), timeout=1)
      connect_release.set()
      result = await asyncio.wait_for(pending, timeout=1)
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
