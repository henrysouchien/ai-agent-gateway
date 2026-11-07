import asyncio
import contextlib
import errno
from collections import deque
from collections.abc import Awaitable, Callable
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import httpx2
from hank_contracts.agent_api import AGENT_API_CLAIM_ENV_VARS, AGENT_API_CLAIM_HEADERS
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
from agent_gateway.mcp_client_errors import McpStdioStartupBoundError  # noqa: E402
from agent_gateway.tool_policy_registry import (  # noqa: E402
  PlanDecision,
  PreparedToolCall,
  RedactionResult,
  SourceIdentityResult,
  ToolPolicyImplementation,
  ToolPolicyImplementationRegistry,
  reject_policy_parameters,
)
from agent_gateway.tool_registration import RegisteredMcpToolCompilationError  # noqa: E402
from gateway_test_support.host_policy import owner_session_host_policy  # noqa: E402,F401
from agent_workflow_contracts.tool_registration import (  # noqa: E402
  PolicyKind,
  RegisteredToolIdentity,
  RegisteredToolServerDescriptor,
  ToolApprovalPolicy,
  ToolIntrinsicSemantics,
  ToolRegistrationCatalog,
  ToolRegistrationDeclaration,
  VersionedPolicyRef,
)


@pytest.fixture
def http_claim_connection(monkeypatch):
  clients = []
  legacy_calls = []
  scoped_env = {}
  legacy_auth = object()

  class Context:
    async def __aenter__(self) -> Any:
      return self

    async def __aexit__(self, *_args):
      pass

  class Client(Context):
    def __init__(self, **kwargs):
      self.options = kwargs
      clients.append(self)

  class Stream(Context):
    async def __aenter__(self):
      return object(), object()

  async def close_contexts(contexts):
    for context in reversed(contexts):
      await context.__aexit__(None, None, None)

  async def initialize(**kwargs):
    return SimpleNamespace(exit_contexts=kwargs["exit_contexts"])

  def build_auth(*args):
    legacy_calls.append(args)
    return legacy_auth

  manager = McpClientManager(config_path=None)
  monkeypatch.setattr(
    manager, "_scoped_server_env",
    lambda *_args, **_kwargs: (scoped_env, set()),
  )
  monkeypatch.setattr(manager, "_build_http_auth", build_auth)
  monkeypatch.setattr(manager, "_initialize_session_state", initialize)
  runtime = SimpleNamespace(
    environ={},
    streamable_http_types={"http", "streamable-http"},
    close_contexts=close_contexts,
    build_http_headers=lambda *_args, **_kwargs: {},
    httpx_module=SimpleNamespace(AsyncClient=Client, Timeout=httpx2.Timeout),
    streamable_http_client_factory=lambda *_args, **_kwargs: Stream(),
    client_session_factory=lambda *_args: Context(),
    parse_allowed_tools=lambda _value: None,
  )
  monkeypatch.setattr(manager, "_connection_runtime", lambda: runtime)

  async def connect(user_claim_auth=None):
    return await mcp_client_connections.connect_streamable_http(
      manager, "remote", {"url": "https://mcp.example/mcp"},
      cast(mcp_client_connections.McpConnectionRuntime, runtime),
      user_claim_auth=user_claim_auth,
    )

  return scoped_env, clients, legacy_calls, legacy_auth, connect, manager


def test_http_remote_claim_auth_mints_fresh_claims_without_second_auth(
  http_claim_connection, monkeypatch,
):
  scoped_env, clients, legacy_calls, _legacy_auth, connect, _manager = http_claim_connection
  scoped_env.update({
    "AGENT_API_CLAIM_MINT_URL": "https://gateway.example",
    "GATEWAY_API_KEY": "roster-key",
  })
  calls = []

  async def mint(**kwargs):
    calls.append(kwargs)
    return {
      env_name: f"{field}-{len(calls)}"
      for field, env_name in AGENT_API_CLAIM_ENV_VARS.items()
    }

  monkeypatch.setattr(mcp_client_connections, "mint_gateway_user_claim", mint)

  async def scenario():
    state = await connect(user_claim_auth=object())
    try:
      assert calls == []
      auth = clients[0].options["auth"]
      assert isinstance(auth, mcp_client_connections.UserClaimHttpAuth)
      for generation in (1, 2):
        request = httpx2.Request("POST", "https://mcp.example/mcp")
        flow = auth.async_auth_flow(request)
        signed = await anext(flow)
        assert signed.headers[AGENT_API_CLAIM_HEADERS["signature"]] == (
          f"signature-{generation}"
        )
        await flow.aclose()
      assert calls == [
        {"base_url": "https://gateway.example", "api_key": "roster-key"},
        {"base_url": "https://gateway.example", "api_key": "roster-key"},
      ]
      assert legacy_calls == []
      assert set(clients[0].options) == {"headers", "timeout", "auth"}
    finally:
      await state.exit_contexts[0].__aexit__(None, None, None)

  asyncio.run(scenario())


def test_manager_connect_remote_claim_auth_never_requires_local_signer(
  http_claim_connection, monkeypatch,
):
  scoped_env, clients, legacy_calls, _legacy_auth, _connect, manager = http_claim_connection
  scoped_env.update({
    "AGENT_API_CLAIM_MINT_URL": "https://gateway.example",
    "GATEWAY_API_KEY": "roster-key",
  })
  signer_calls = []
  mint_calls = []
  claims = {
    env_name: f"remote-{field}"
    for field, env_name in AGENT_API_CLAIM_ENV_VARS.items()
  }

  def unexpected_catalog_auth():
    signer_calls.append(True)
    raise AssertionError("remote startup must not access the local signer")

  async def mint(**kwargs):
    mint_calls.append(kwargs)
    return claims

  monkeypatch.setattr(manager, "_catalog_claim_auth", unexpected_catalog_auth)
  monkeypatch.setattr(mcp_client_connections, "mint_gateway_user_claim", mint)

  async def scenario():
    state = await manager._connect(
      "remote",
      {"type": "http", "per_user": True, "url": "https://mcp.example/mcp"},
    )
    try:
      assert mint_calls == []
      auth = clients[0].options["auth"]
      assert isinstance(auth, mcp_client_connections.UserClaimHttpAuth)
      request = httpx2.Request("POST", "https://mcp.example/mcp")
      flow = auth.async_auth_flow(request)
      signed = await anext(flow)
      try:
        for field, header_name in AGENT_API_CLAIM_HEADERS.items():
          assert signed.headers[header_name] == claims[AGENT_API_CLAIM_ENV_VARS[field]]
      finally:
        await flow.aclose()
      assert mint_calls == [
        {"base_url": "https://gateway.example", "api_key": "roster-key"},
      ]
      assert signer_calls == []
      assert legacy_calls == []
    finally:
      await state.exit_contexts[0].__aexit__(None, None, None)

  asyncio.run(scenario())


@pytest.mark.parametrize("supplied_claim_auth", [False, True])
def test_http_absent_mint_url_preserves_legacy_auth(
  http_claim_connection, supplied_claim_auth,
):
  _env, clients, legacy_calls, legacy_auth, connect, _manager = http_claim_connection
  claim_auth = object() if supplied_claim_auth else None

  async def scenario():
    state = await connect(user_claim_auth=claim_auth)
    try:
      assert clients[0].options["auth"] is (
        claim_auth if supplied_claim_auth else legacy_auth
      )
      assert len(legacy_calls) == (0 if supplied_claim_auth else 1)
    finally:
      await state.exit_contexts[0].__aexit__(None, None, None)

  asyncio.run(scenario())


@pytest.mark.parametrize("per_user", [False, True])
def test_manager_connect_absent_mint_url_preserves_catalog_auth(
  http_claim_connection, monkeypatch, per_user,
):
  _env, clients, legacy_calls, legacy_auth, _connect, manager = http_claim_connection
  catalog_auth = object()
  signer_calls = []

  def catalog_claim_auth():
    signer_calls.append(True)
    return catalog_auth

  monkeypatch.setattr(manager, "_catalog_claim_auth", catalog_claim_auth)

  async def scenario():
    state = await manager._connect(
      "remote",
      {"type": "http", "per_user": per_user, "url": "https://mcp.example/mcp"},
    )
    try:
      assert clients[0].options["auth"] is (catalog_auth if per_user else legacy_auth)
      assert signer_calls == ([True] if per_user else [])
      assert len(legacy_calls) == (0 if per_user else 1)
    finally:
      await state.exit_contexts[0].__aexit__(None, None, None)

  asyncio.run(scenario())


@pytest.mark.parametrize("environment", [
  {"AGENT_API_CLAIM_MINT_URL": "", "GATEWAY_API_KEY": "roster-key"},
  {"AGENT_API_CLAIM_MINT_URL": "  ", "GATEWAY_API_KEY": "roster-key"},
  {"AGENT_API_CLAIM_MINT_URL": "https://gateway.example"},
  {"AGENT_API_CLAIM_MINT_URL": "https://gateway.example", "GATEWAY_API_KEY": ""},
  {"AGENT_API_CLAIM_MINT_URL": "https://gateway.example", "GATEWAY_API_KEY": "  "},
])
@pytest.mark.parametrize("manager_connect", [False, True])
def test_http_invalid_remote_mint_configuration_never_falls_back(
  http_claim_connection, environment, monkeypatch, manager_connect,
):
  scoped_env, clients, legacy_calls, _legacy_auth, connect, manager = http_claim_connection
  scoped_env.update(environment)

  async def unexpected_mint(**_kwargs):
    raise AssertionError("invalid configuration must fail before minting")

  def unexpected_catalog_auth():
    raise AssertionError("invalid configuration must fail before local signing")

  monkeypatch.setattr(manager, "_catalog_claim_auth", unexpected_catalog_auth)

  monkeypatch.setattr(mcp_client_connections, "mint_gateway_user_claim", unexpected_mint)
  with pytest.raises(ValueError, match="AGENT_API_CLAIM_MINT_URL|GATEWAY_API_KEY"):
    if manager_connect:
      asyncio.run(manager._connect(
        "remote",
        {"type": "http", "per_user": True, "url": "https://mcp.example/mcp"},
      ))
    else:
      asyncio.run(connect(user_claim_auth=object()))
  assert clients == []
  assert legacy_calls == []


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
    read_timeout_seconds: float | None,
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


def test_startup_contains_unregistered_routes_and_refuses_calls_to_them(reconnect_transport):
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
      await manager.startup()
      assert {
        (route.logical_server_id, route.logical_name)
        for route in manager.get_refused_mcp_routes()
      } == {("first", "alpha"), ("second", "beta")}
      assert manager.get_tool_definitions() == []
      result, error = await manager.call_tool("alpha", {})
      assert result is None
      assert error is not None
      assert error["code"] == "mcp_tool_not_registered"
      assert "alpha" in error["message"]
      assert "first" in error["message"]
      assert not any(context.closed for context in opened_contexts)
    finally:
      await manager.shutdown()
    assert manager.get_refused_mcp_routes() == ()

  asyncio.run(scenario())


def test_rejected_startup_closes_all_connected_transports(reconnect_transport, monkeypatch):
  def incoherent_compile(*_args, **_kwargs):
    raise RegisteredMcpToolCompilationError("duplicate live MCP exposed name")

  monkeypatch.setattr(
    mcp_client_module, "compile_registered_mcp_tool_descriptors", incoherent_compile,
  )

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
        tool_input={}, meta=None, abort_event=None, expected_seconds=1,
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

# The banner's PyPI version check is a network call: where the resolver does not answer
# (a no-egress gate box) it stalls every spawn of this server for about 10 s.
mcp.run(show_banner=False)
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

mcp.run(show_banner=False)
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


def test_startup_timeout_measures_waiting_for_cancellation_settlement(monkeypatch) -> None:
  manager = SimpleNamespace(_startup_timeout=0.01)
  clock = SimpleNamespace(now=100.0)
  # Replace this owner's clock, not time.monotonic used by asyncio's scheduler.
  monkeypatch.setattr(
    mcp_client_connections, "time", SimpleNamespace(monotonic=lambda: clock.now),
  )

  async def scenario():
    cancelled = asyncio.Event()
    acknowledge_cancellation = asyncio.Event()

    async def unanswered_step():
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        cancelled.set()
        await acknowledge_cancellation.wait()
        raise

    startup = asyncio.create_task(mcp_client_connections._bounded_startup_step(
      manager, "initialize", unanswered_step(), name="slow-cancellation",
    ))
    try:
      await asyncio.wait_for(cancelled.wait(), timeout=1)
      clock.now += manager._startup_timeout
      await asyncio.sleep(0)
      assert not startup.done()
      clock.now += 0.125
      acknowledge_cancellation.set()
      with pytest.raises(TimeoutError) as raised:
        await asyncio.wait_for(startup, timeout=1)
      return str(raised.value)
    finally:
      acknowledge_cancellation.set()
      startup.cancel()
      await asyncio.gather(startup, return_exceptions=True)

  message = asyncio.run(scenario())
  assert "MCP server slow-cancellation startup initialize" in message
  measured = re.search(r"after ([0-9.]+)s: no response", message)
  assert measured is not None
  assert float(measured.group(1)) == pytest.approx(0.135, abs=0.001)
  assert float(measured.group(1)) > manager._startup_timeout


@pytest.mark.parametrize("step", ["initialize", "list_tools"])
def test_startup_preserves_native_timeout_cause(monkeypatch, step) -> None:
  manager = SimpleNamespace(_startup_timeout=15)
  clock = SimpleNamespace(now=100.0)
  monkeypatch.setattr(
    mcp_client_connections, "time", SimpleNamespace(monotonic=lambda: clock.now),
  )

  async def upstream_timeout():
    clock.now += 0.375
    raise TimeoutError("upstream socket timed out")

  with pytest.raises(TimeoutError) as raised:
    asyncio.run(mcp_client_connections._bounded_startup_step(
      manager, step, upstream_timeout(), name="upstream-child",
    ))
  message = str(raised.value)
  assert f"MCP server upstream-child startup {step}" in message
  assert "after 0.375s: upstream socket timed out" in message
  assert "no response" not in message
  diagnostic = mcp_client_module._startup_failure_from_exception(raised.value)
  assert diagnostic["category"] == "transient_timeout"
  assert diagnostic["retryable"] is True


@pytest.mark.parametrize(
  ("step", "method"), [("initialize", "initialize"), ("list_tools", "tools/list")],
)
def test_hung_stdio_startup_publishes_healthy_sibling_and_reaps_children(
  tmp_path, monkeypatch, caplog, step, method,
) -> None:
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_retries", lambda: 0)
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_stabilize_delay", lambda: 0)
  caplog.set_level(logging.INFO, logger="agent_gateway.mcp_client")
  server_script = tmp_path / "startup_server.py"
  server_script.write_text("""
import json
import os
import sys
from pathlib import Path

name, unanswered_method = sys.argv[1:]
Path(f"{name}.pid").write_text(str(os.getpid()))
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    with Path(f"{name}.requests").open("a") as stream:
        stream.write(str(method) + "\\n")
    if "id" not in request or method == unanswered_method:
        continue
    if method == "initialize":
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": name, "version": "1.0"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": "healthy_echo",
            "description": "Report the responding server",
            "inputSchema": {"type": "object", "properties": {}},
        }]}
    elif method == "tools/call":
        result = {"content": [], "structuredContent": {"server": name}}
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
""")
  hung_name = f"hung-{step}"
  healthy_name = "healthy-sibling"

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      startup_timeout=1,
      inline_servers={
        name: {
          "command": sys.executable,
          "args": [str(server_script), name, unanswered_method],
          "cwd": str(tmp_path),
        }
        for name, unanswered_method in ((hung_name, method), (healthy_name, "none"))
      },
    )
    try:
      started = time.monotonic()
      await asyncio.wait_for(manager.startup(), timeout=10)
      elapsed = time.monotonic() - started
      diagnostic = manager.get_startup_diagnostics()[hung_name]
      assert diagnostic["server"] == hung_name
      assert diagnostic["category"] == "transient_timeout"
      assert diagnostic["retryable"] is True
      assert manager.get_server_names() == {healthy_name}
      assert {tool["name"] for tool in manager.get_tool_definitions()} == {"healthy_echo"}
      assert manager.get_server_for_tool("healthy_echo") == healthy_name
      assert await asyncio.wait_for(manager.call_tool("healthy_echo", {}), timeout=2) == (
        {"server": healthy_name}, None,
      )
      assert method in (tmp_path / f"{hung_name}.requests").read_text().splitlines()
      with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / f"{hung_name}.pid").read_text()), 0)
      return diagnostic, elapsed
    finally:
      await asyncio.wait_for(manager.shutdown(), timeout=5)
      assert manager.get_server_names() == set()
      for name in (hung_name, healthy_name):
        pid_file = tmp_path / f"{name}.pid"
        if pid_file.exists():
          with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)

  diagnostic, elapsed = asyncio.run(scenario())
  message = diagnostic["message"]
  assert f"MCP server {hung_name}" in message
  step_duration = re.search(
    rf"startup {step} failed after ([0-9.]+)s: no response", message,
  )
  total_duration = re.search(r"startup failed after ([0-9.]+)s:", message)
  assert step_duration is not None
  assert total_duration is not None
  assert float(step_duration.group(1)) >= 0.999
  assert float(total_duration.group(1)) >= float(step_duration.group(1))
  assert float(total_duration.group(1)) == pytest.approx(elapsed, abs=0.05)
  connected = next(
    record.getMessage() for record in caplog.records
    if record.getMessage().startswith(f"MCP server {healthy_name} connected after ")
  )
  success_duration = re.search(r"connected after ([0-9.]+)s", connected)
  assert success_duration is not None
  assert 0 < float(success_duration.group(1)) <= elapsed


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
      started = time.monotonic()
      assert await manager._connect_or_warn(
        "broken-child", {"command": sys.executable, "args": ["-c", script]},
      ) is None
      elapsed = time.monotonic() - started
      diagnostic = manager.get_startup_diagnostics()["broken-child"]
      return diagnostic, elapsed
    finally:
      await manager.shutdown()

  diagnostic, elapsed = asyncio.run(scenario())
  message = diagnostic["message"]
  assert "MCP server broken-child" in message
  assert "Connection closed" in message
  assert "exit code unavailable" not in message.lower()
  measured = re.search(r"startup failed after ([0-9.]+)s:", message)
  assert measured is not None
  assert 0 < float(measured.group(1))
  assert float(measured.group(1)) == pytest.approx(elapsed, abs=0.02)
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


def test_gateway_spawned_fastmcp_child_attempts_no_update_check(tmp_path, monkeypatch) -> None:
  # A product server's plain `mcp.run()`: FastMCP's banner makes a blocking PyPI
  # version check unless the gateway's child env turns it off.
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_retries", lambda: 0)
  egress = tmp_path / "egress.log"
  server_script = tmp_path / "plain_run_server.py"
  server_script.write_text("""
import os
import socket
import sys
from pathlib import Path

egress = Path(os.environ["EGRESS_LOG"])

def refuse_egress(event, args):
    # Record every name lookup and non-local connect, and fail it at once, as a
    # box whose resolver does not answer would, minus the wait.
    if event == "socket.getaddrinfo":
        target = args[0]
    elif event == "socket.connect" and args[0].family != socket.AF_UNIX:
        target = args[1]
    else:
        return
    with egress.open("a") as stream:
        stream.write(f"{event} {target!r}\\n")
    raise socket.gaierror(socket.EAI_NONAME, "egress refused by the test")

sys.addaudithook(refuse_egress)

from fastmcp import FastMCP

mcp = FastMCP("plain-run")

@mcp.tool()
def ping() -> dict:
    return {"pong": True}

mcp.run()
""")

  async def scenario():
    manager = McpClientManager(config_path=None, inline_servers={
      "plain-run": {
        "command": sys.executable,
        "args": [str(server_script)],
        "cwd": str(tmp_path),
        # No version cache from an earlier run stands in for the check.
        "env": {"EGRESS_LOG": str(egress), "FASTMCP_HOME": str(tmp_path / "fastmcp-home")},
      },
    })
    try:
      await manager.startup()
      assert manager.get_server_names() == {"plain-run"}
      assert await manager.call_tool("ping", {}) == ({"pong": True}, None)
    finally:
      await manager.shutdown()

  asyncio.run(scenario())
  assert not egress.exists(), egress.read_text()


_SLOW_START_SERVER = """
import json
import os
import subprocess
import sys
import time
from pathlib import Path

name, mode, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
Path(f"{name}.pid").write_text(str(os.getpid()))
if mode == "busy":
    # Spend CPU before reading stdin, as a cold import on a loaded machine does.
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass
elif mode == "died":
    # A descendant keeps stdout open, so the gateway sees no EOF from the exit.
    descendant = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL,
    )
    Path(f"{name}.descendant").write_text(str(descendant.pid))
    raise SystemExit(0)
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request or mode == "stalled":
        continue
    method = request.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": name, "version": "1.0"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": "slow_echo",
            "description": "Report the responding server",
            "inputSchema": {"type": "object", "properties": {}},
        }]}
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


def _connect_slow_start_child(tmp_path, name, mode, seconds, *, startup_timeout):
  server_script = tmp_path / "slow_start_server.py"
  server_script.write_text(_SLOW_START_SERVER)

  async def scenario():
    manager = McpClientManager(config_path=None, startup_timeout=startup_timeout)
    try:
      state = await manager._connect_or_warn(name, {
        "command": sys.executable,
        "args": [str(server_script), name, mode, str(seconds)],
        "cwd": str(tmp_path),
      })
      return state, manager.get_startup_diagnostics().get(name)
    finally:
      await manager.shutdown()
      descendant = tmp_path / f"{name}.descendant"
      if descendant.exists():
        with contextlib.suppress(ProcessLookupError):
          os.kill(int(descendant.read_text()), signal.SIGKILL)

  return asyncio.run(scenario())


def test_busy_stdio_child_keeps_its_startup_window_past_the_bound(tmp_path, monkeypatch, caplog) -> None:
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_retries", lambda: 0)
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_stabilize_delay", lambda: 0)
  caplog.set_level(logging.INFO, logger="agent_gateway.mcp_client")

  # Busy for one and a half bounds before it reads `initialize`.
  state, diagnostic = _connect_slow_start_child(tmp_path, "busy-child", "busy", 3.0, startup_timeout=2)

  assert diagnostic is None
  assert state is not None
  assert state.tool_names == {"slow_echo"}
  pid = (tmp_path / "busy-child.pid").read_text()
  assert any(
    record.getMessage().startswith(f"MCP stdio server busy-child (pid {pid}) is still starting")
    for record in caplog.records
  )


@pytest.mark.parametrize(
  ("mode", "outcome", "detail"),
  [("stalled", "stalled", "alive but stalled"), ("died", "died", "died")],
)
def test_idle_or_dead_stdio_child_fails_at_the_startup_bound(
  tmp_path, monkeypatch, mode, outcome, detail,
) -> None:
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_retries", lambda: 0)
  name = f"{mode}-child"
  observed: list[BaseException] = []
  connect_stdio = McpClientManager._connect_stdio

  async def observe(self, server_name, config):
    try:
      return await connect_stdio(self, server_name, config)
    except BaseException as exc:
      observed.append(exc)
      raise

  monkeypatch.setattr(McpClientManager, "_connect_stdio", observe)

  state, diagnostic = _connect_slow_start_child(tmp_path, name, mode, 0, startup_timeout=2)

  assert state is None
  pid = int((tmp_path / f"{name}.pid").read_text())
  [error] = observed
  assert isinstance(error, McpStdioStartupBoundError)
  assert (error.server, error.step, error.outcome, error.pid) == (name, "initialize", outcome, pid)
  assert diagnostic is not None
  assert diagnostic["category"] == "transient_timeout"
  assert diagnostic["error_type"] == "McpStdioStartupBoundError"
  message = diagnostic["message"]
  assert f"stdio child pid {pid} {detail}" in message
  measured = re.search(r"startup initialize failed after ([0-9.]+)s: no response", message)
  assert measured is not None
  # At the bound, never extended toward the 6 s ceiling.
  assert 2.0 <= float(measured.group(1)) < 4.0


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


def test_shutdown_completes_while_stdio_watch_awaits_receive_done():
  async def scenario():
    manager = McpClientManager(config_path=None)
    receive_done = asyncio.Event()
    eof = asyncio.Event()
    eof.set()
    server = mcp_client_module._ServerState(
      name="stuck",
      session=_ListedToolsSession(["alpha"]),
      exit_contexts=[],
      tool_definitions=[],
      tool_names=set(),
      config={"command": sys.executable, "type": "stdio"},
      stdio_eof=eof,
      stdio_receive_done=receive_done,
    )
    manager._servers["stuck"] = server
    manager._started = True
    parked = asyncio.Event()
    real_wait = receive_done.wait

    async def marked_wait():
      parked.set()
      return await real_wait()

    receive_done.wait = marked_wait
    watch = asyncio.create_task(manager._replace_stdio_server("stuck", server))
    server.stdio_watch_task = watch
    await asyncio.wait_for(parked.wait(), timeout=1.0)
    assert manager._lock.locked() is False
    assert watch.done() is False
    started = time.monotonic()
    await asyncio.wait_for(manager.shutdown(), timeout=1.0)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert watch.done()
    assert receive_done.is_set() is False
    assert "stuck" not in manager._servers

  asyncio.run(scenario())


class _LoopbackRefusingSession:
  """Refuse the first ``refusals`` loopback connects, then answer."""

  def __init__(self, refusals: int) -> None:
    self.refusals = refusals
    self.calls = 0

  async def call_tool(self, name, arguments, **_kwargs):
    self.calls += 1
    if self.calls <= self.refusals:
      raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")
    return SimpleNamespace(
      is_error=False,
      structured_content={"tool": name, "ok": True},
      content=[],
    )


def _loopback_manager(session) -> McpClientManager:
  manager = McpClientManager(config_path=None)
  manager._servers["loopback"] = mcp_client_module._ConnectedServerState(
    name="loopback",
    session=session,
    exit_contexts=[],
    tool_definitions=[],
    tool_names={"read_door", "write_door"},
    config={"type": "streamable_http", "url": "http://127.0.0.1:9"},
  )
  manager._tool_to_server["read_door"] = "loopback"
  manager._tool_to_server["write_door"] = "loopback"
  return manager


def test_loopback_connection_refused_retries_idempotent_call_then_succeeds():
  session = _LoopbackRefusingSession(refusals=2)
  manager = _loopback_manager(session)

  result, error = asyncio.run(manager.call_tool("read_door", {}))

  assert error is None
  assert result == {"tool": "read_door", "ok": True}
  assert session.calls == 3


def test_loopback_connection_refused_never_retries_when_replay_forbidden():
  session = _LoopbackRefusingSession(refusals=2)
  manager = _loopback_manager(session)

  result, error = asyncio.run(
    manager.call_tool("write_door", {}, allow_uncertain_replay=False)
  )

  assert result is None
  assert error is not None
  assert session.calls == 1


def test_loopback_connection_refused_exhaustion_returns_refusal():
  session = _LoopbackRefusingSession(refusals=99)
  manager = _loopback_manager(session)

  result, error = asyncio.run(manager.call_tool("read_door", {}))

  assert result is None
  assert error is not None
  assert session.calls > 1


def test_loopback_refusal_ignores_enodata_under_linux_numbering(monkeypatch):
  # Linux numbers ENODATA 61 and ECONNREFUSED 111 (Darwin is the reverse).
  # The predicate must compare against errno.ECONNREFUSED alone.
  monkeypatch.setattr(errno, "ECONNREFUSED", 111)
  monkeypatch.setattr(errno, "ENODATA", 61)
  no_data = OSError("no data")
  no_data.errno = errno.ENODATA
  refused = OSError("refused")
  refused.errno = errno.ECONNREFUSED

  assert mcp_client_connections.is_loopback_connection_refused(no_data) is False
  assert mcp_client_connections.is_loopback_connection_refused(refused) is True


_FLAKY_STDIO_SERVER = """
import os
import sys
import time
from pathlib import Path

while Path("hold").exists():
    time.sleep(0.02)
if Path("refuse").exists():
    sys.stderr.write("refusing to start\\n")
    sys.exit(3)
from fastmcp import FastMCP

with Path("starts").open("a") as stream:
    stream.write(str(os.getpid()) + "\\n")
mcp = FastMCP("flaky")

@mcp.tool()
def read_probe() -> dict:
    return {"pid": os.getpid()}

mcp.run(show_banner=False)
"""


def _flaky_stdio_manager(
  tmp_path, monkeypatch, *, server_source: str | None = None, **manager_kwargs,
) -> McpClientManager:
  monkeypatch.setenv("MCP_STDIO_CONNECT_RETRIES", "0")
  monkeypatch.setenv("MCP_STDIO_CONNECT_STABILIZE_S", "0")
  monkeypatch.setattr(
    mcp_client_module, "_stdio_recovery_delay", lambda _attempt: 0.05, raising=False,
  )
  script = tmp_path / "mcp_server.py"
  script.write_text(_FLAKY_STDIO_SERVER if server_source is None else server_source)
  return McpClientManager(
    config_path=None,
    inline_servers={
      "probe": {"command": sys.executable, "args": [str(script)], "cwd": str(tmp_path)},
    },
    **manager_kwargs,
  )


def _flaky_starts(tmp_path) -> list[int]:
  starts = tmp_path / "starts"
  return [int(line) for line in starts.read_text().split()] if starts.exists() else []


async def _until(condition, *, seconds: float = 30):
  async with asyncio.timeout(seconds):
    while not condition():
      await asyncio.sleep(0.02)


def test_stdio_server_that_failed_startup_comes_back_without_a_call(tmp_path, monkeypatch):
  (tmp_path / "refuse").write_text("x")
  manager = _flaky_stdio_manager(tmp_path, monkeypatch)

  async def scenario():
    await manager.startup()
    try:
      assert "probe" not in manager._servers
      assert manager.get_startup_diagnostics()["probe"]["category"] == "transient_transport"
      result, error = await manager.call_tool("read_probe", {})
      assert result is None and error is not None
      assert _flaky_starts(tmp_path) == []

      # The cause is gone; no call is made. The startup-recovery round publishes it.
      (tmp_path / "refuse").unlink()
      await _until(lambda: manager.get_server_for_tool("read_probe") == "probe")
      assert manager.get_startup_diagnostics() == {}
      watch = manager._servers["probe"].stdio_watch_task
      assert watch is not None and not watch.done()
      result, error = await manager.call_tool("read_probe", {})
      assert error is None
      assert result == {"pid": _flaky_starts(tmp_path)[0]}

      # The revived child is under the same EOF supervision as any other.
      os.kill(_flaky_starts(tmp_path)[0], signal.SIGTERM)
      await _until(lambda: len(_flaky_starts(tmp_path)) == 2)
      await _until(lambda: manager._servers["probe"].stdio_watch_task is not watch)
      result, error = await manager.call_tool("read_probe", {})
      assert error is None
      assert result == {"pid": _flaky_starts(tmp_path)[1]}
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


def _registered_probe_catalog() -> ToolRegistrationCatalog:
  return _registered_catalog("probe", {"read_probe": ("read", True)})


def _registered_catalog(
  server_id: str, tools: dict[str, tuple[str, bool]],
) -> ToolRegistrationCatalog:
  def policy(kind: str) -> VersionedPolicyRef:
    return VersionedPolicyRef(kind=kind, policy_id=f"{kind}.identity", version="1")  # type: ignore[arg-type]

  return ToolRegistrationCatalog(
    declarations=tuple(
      ToolRegistrationDeclaration(
        identity=RegisteredToolIdentity(
          route_kind="mcp", logical_server_id=server_id, logical_name=name,
        ),
        semantics=ToolIntrinsicSemantics(
          effect=effect,  # type: ignore[arg-type]
          idempotent=idempotent,
          semantic_capability="market-data.read/v1",
          approval=ToolApprovalPolicy(mode="never"),
          audience="ordinary",
          redaction_policy=policy("redaction"),
          planning_policy=policy("planning"),
          input_preparation_policy=policy("input_preparation"),
          outcome_policy=policy("outcome"),
          source_identity_policy=policy("source_identity"),
        ),
      )
      for name, (effect, idempotent) in tools.items()
    ),
    servers=(
      RegisteredToolServerDescriptor(
        logical_server_id=server_id,
        transport_server_id=server_id,
        default_timeout_seconds=30,
        per_tool_timeout_seconds={},
        session_injection_policy=policy("session_injection"),
      ),
    ),
  )


async def _call_at_once(manager):
  """A call to a down server answers typed while every recovery round is parked (the
  child's start waits on ``hold``): it ran no round of its own and waited on none. A call
  that did would not return."""
  async with asyncio.timeout(10):
    return await manager.call_tool("read_probe", {})


def test_call_to_a_registered_tool_of_a_down_stdio_server_fails_typed(tmp_path, monkeypatch):
  (tmp_path / "refuse").write_text("x")
  manager = _flaky_stdio_manager(
    tmp_path, monkeypatch, tool_registration_catalog=_registered_probe_catalog(),
  )

  async def scenario():
    await manager.startup()
    try:
      # Recovery rounds park in the child's start; "at once" is a typed answer while they do.
      (tmp_path / "hold").write_text("x")
      (tmp_path / "refuse").unlink()
      result, error = await _call_at_once(manager)
      assert result is None
      assert error is not None
      assert error["code"] == "mcp_tool_error"
      assert error["sub_code"] == "mcp_server_unavailable"
      assert "probe" in error["message"]
      assert "probe" in manager.get_startup_diagnostics()
      (tmp_path / "hold").unlink()
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


def test_server_catalog_lists_a_startup_failed_servers_declared_tools(tmp_path, monkeypatch):
  """b05-72 end to end: a server that failed its startup connect published no definitions,
  so nothing named its tools and the tool door answered 404 before the manager could
  answer. The catalog the surface authorizes against lists its declared tools."""
  (tmp_path / "refuse").write_text("x")
  manager = _flaky_stdio_manager(
    tmp_path, monkeypatch, tool_registration_catalog=_registered_probe_catalog(),
  )

  async def scenario():
    await manager.startup()
    try:
      assert manager.get_declared_tools_of_startup_failed_servers() == {"read_probe": "probe"}
      catalog = manager.get_server_catalog()
      assert catalog["probe"] == {"tool_count": 1, "tools": ["read_probe"], "startup_failed": True}
      assert manager.get_tool_definitions() == []
      # The dispatcher routes on these two facts: a declared tool of a down server is
      # this manager's and belongs to its declared server, so dispatch reaches
      # `call_tool` instead of answering unknown_tool (the door's 404).
      assert manager.is_mcp_tool("read_probe")
      assert manager.get_server_for_tool("read_probe") == "probe"
      assert not manager.is_mcp_tool("never_declared")
      assert manager.get_server_for_tool("never_declared") is None
      # A call to such a tool fails typed at once; the startup-recovery task, not the
      # call, reconnects the server (the tool door dispatches into this same call).
      _result, error = await manager.call_tool("read_probe", {})
      assert error is not None and error["sub_code"] == "mcp_server_unavailable"
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


def test_dispatch_of_a_declared_tool_of_a_startup_failed_server_settles_typed_in_production_shape(
  tmp_path, monkeypatch, owner_session_host_policy,
):
  """b05-72 in the production shape: the registration catalog and policy implementations
  on both the manager and the dispatcher (api/mcp_client.py, api/main.py). A declared tool
  of a startup-failed server has no live descriptor. Every route the door, a chat turn and
  dispatch take must answer the manager's typed mcp_server_unavailable, never raise on a
  missing descriptor (500) and never unknown_tool (404)."""
  from agent_gateway.approval_policy import RunContext
  from agent_gateway.event_log import EventLog
  from agent_gateway.registry_scope import RegistryScope
  from agent_gateway.session import SessionStore
  from agent_gateway.tool_dispatcher import ToolDispatcher
  from agent_gateway.tool_policy_registry import ToolInputPreparationError
  from gateway_test_support.capability_execution_test_support import stub_capability_execution_resolver

  (tmp_path / "refuse").write_text("x")
  registry = _redacting_policy_registry()
  manager = _flaky_stdio_manager(
    tmp_path, monkeypatch,
    tool_registration_catalog=_registered_probe_catalog(),
    tool_policy_implementations=registry,
    input_preparation_context_factory=lambda _descriptor, _scope: None,
    redaction_context_factory=lambda _registration: None,
  )

  async def scenario():
    await manager.startup()
    try:
      session = SessionStore(ttl=3600).create_session(
        api_key_hash="hash", user_id="alice", role="owner",
      )
      execution = stub_capability_execution_resolver().resolve("session.driver")
      dispatcher = ToolDispatcher(
        capability_bind=execution.bind,
        mcp_client=manager,
        needs_approval=lambda *_args: False,
        event_log=EventLog(),
        session=session,
        role="owner",
        run_context=RunContext(
          user_id="alice", request_id="request-1", session_id=session.session_id,
          registry_scope=RegistryScope("run", "test-run", str(tmp_path / "store")),
        ),
        tool_registration_catalog=_registered_probe_catalog(),
        tool_policy_implementations=registry,
        redaction_context_factory=lambda _declaration: None,
      )
      assert dispatcher.registered_tool_declaration("read_probe") is not None

      # dispatch(): settled typed at preparation, with the failed server's diagnostic.
      result, error = await dispatcher.dispatch("call-1", "read_probe", {"q": 1})
      assert result is None
      assert error is not None and error.get("sub_code") == "mcp_server_unavailable", error
      assert "Connection closed" in error["message"]

      # The door's sequence: prepare, then requires_approval_prepared on a prepared call.
      with pytest.raises(ToolInputPreparationError) as refused:
        dispatcher.prepare_tool_call("read_probe", {"q": 1})
      assert refused.value.materialize_error()["sub_code"] == "mcp_server_unavailable"
      # Classification reads the catalog declaration (its planning policy is not
      # "none", so the planning branch answers True) instead of raising on the
      # missing descriptor.
      assert dispatcher.requires_approval_prepared(
        "read_probe", PreparedToolCall({"q": 1}),
      ) is True

      # A chat turn's history line after preparation failed, and a prepared line:
      # both run the declaration's redaction policy (the registry's redactor marks
      # every value), never the raw input.
      assert dispatcher.redact_raw_tool_input_for_history("read_probe", {"q": 1}) == {"q": "<redacted>"}
      assert dispatcher.redact_prepared_tool_input("read_probe", PreparedToolCall({"q": 1})) == {"q": "<redacted>"}
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


def test_down_stdio_server_fails_fast_and_returns_without_a_call(tmp_path, monkeypatch):
  manager = _flaky_stdio_manager(tmp_path, monkeypatch)

  async def scenario():
    await manager.startup()
    try:
      first_pid = _flaky_starts(tmp_path)[0]
      (tmp_path / "refuse").write_text("x")
      os.kill(first_pid, signal.SIGTERM)
      await _until(
        lambda: manager.get_startup_diagnostics().get("probe", {}).get("category")
        == "reconnect_failed",
      )

      # The next recovery round parks in the child's start: a call that ran, or waited on, a
      # round would not return.
      (tmp_path / "hold").write_text("x")
      (tmp_path / "refuse").unlink()
      for _ in range(2):
        result, error = await _call_at_once(manager)
        assert result is None
        assert error is not None
        assert error["sub_code"] == "mcp_server_unavailable"
        assert "Connection closed" in error["message"]

      # The recovery rounds keep running off the call path; the cause clears.
      (tmp_path / "hold").unlink()
      await _until(lambda: manager.get_startup_diagnostics() == {})
      pids = _flaky_starts(tmp_path)
      assert len(pids) == 2 and pids[1] != first_pid
      watch = manager._servers["probe"].stdio_watch_task
      assert watch is not None and not watch.done()
      result, error = await manager.call_tool("read_probe", {})
      assert error is None
      assert result == {"pid": pids[1]}
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


def test_child_loss_is_reported_reconnecting_at_once_and_calls_fail_fast(tmp_path, monkeypatch):
  manager = _flaky_stdio_manager(tmp_path, monkeypatch)

  async def scenario():
    await manager.startup()
    try:
      first_pid = _flaky_starts(tmp_path)[0]
      # The replacement child parks before its handshake: only a record made when EOF fires
      # can show the server down before any round has ended.
      (tmp_path / "hold").write_text("x")
      os.kill(first_pid, signal.SIGTERM)
      await _until(lambda: "probe" in manager.get_startup_diagnostics())
      diagnostic = manager.get_startup_diagnostics()["probe"]
      assert diagnostic["category"] == "reconnecting"
      assert diagnostic["state"] == "reconnecting"

      result, error = await _call_at_once(manager)
      assert result is None
      assert error is not None
      assert error["sub_code"] == "mcp_server_unavailable"

      # The cause clears: the replacement publishes, readiness data and calls recover.
      (tmp_path / "hold").unlink()
      await _until(lambda: manager.get_startup_diagnostics() == {})
      pids = _flaky_starts(tmp_path)
      assert len(pids) == 2 and pids[1] != first_pid
      result, error = await manager.call_tool("read_probe", {})
      assert error is None
      assert result == {"pid": pids[1]}
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


def test_a_replacement_whose_publish_raises_is_retried_until_it_compiles(tmp_path, monkeypatch):
  """Production 2026-11-05 22:49Z (EC2 gateway, before the recovery work): EOF from
  edgar-parser-mcp, the replacement connected, its publish raised
  RegisteredMcpToolCompilationError, the failure was logged once and never retried, so
  the child count stayed 0 until a gateway.service restart. Every round must re-run the
  publish; once it compiles the server comes back with no call."""
  manager = _flaky_stdio_manager(
    tmp_path, monkeypatch, tool_registration_catalog=_registered_probe_catalog(),
  )
  compile_registered = McpClientManager._refresh_registered_mcp_tool_descriptors
  rejected: list[int] = []

  def compile_unless_skewed(self):
    if (tmp_path / "skew").exists():
      rejected.append(1)
      raise RegisteredMcpToolCompilationError("live MCP route has no exact static registration")
    compile_registered(self)

  monkeypatch.setattr(McpClientManager, "_refresh_registered_mcp_tool_descriptors", compile_unless_skewed)

  async def scenario():
    await manager.startup()
    try:
      first_pid = _flaky_starts(tmp_path)[0]
      (tmp_path / "skew").write_text("x")
      os.kill(first_pid, signal.SIGTERM)

      # The replacements start and connect, and every publish is refused: the server is
      # reported down, with the reason, and the rounds keep coming.
      await _until(lambda: len(rejected) >= 3)
      diagnostic = manager.get_startup_diagnostics()["probe"]
      assert diagnostic["state"] == "reconnecting"
      assert diagnostic["category"] in {"reconnecting", "reconnect_failed"}
      assert len(_flaky_starts(tmp_path)) >= 3
      if diagnostic["category"] == "reconnect_failed":
        assert "no exact static registration" in diagnostic["message"]
      assert manager._servers["probe"].stdio_eof is not None and manager._servers["probe"].stdio_eof.is_set()

      # The route compiles again (EDGAR rolled back): no call is made, the server returns.
      (tmp_path / "skew").unlink()
      await _until(lambda: manager.get_startup_diagnostics() == {})
      published = manager._servers["probe"]
      assert published.stdio_eof is not None and not published.stdio_eof.is_set()
      watch = published.stdio_watch_task
      assert watch is not None and not watch.done()
      pids = _flaky_starts(tmp_path)
      assert pids[-1] != first_pid
      served = await published.session.call_tool("read_probe", {}, read_timeout_seconds=None)
      assert served.structured_content == {"pid": pids[-1]}
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


_SHAPED_STDIO_SERVER = """
import os
from pathlib import Path

from fastmcp import FastMCP

shape = set(Path("shape").read_text().split(",")) if Path("shape").exists() else set()
with Path("starts").open("a") as stream:
    stream.write(str(os.getpid()) + "\\n")
mcp = FastMCP("shaped")

if "missing" not in shape and "schema" in shape:
    @mcp.tool()
    def read_probe(ticker: str) -> dict:
        return {"pid": os.getpid(), "ticker": ticker}
elif "missing" not in shape:
    @mcp.tool()
    def read_probe() -> dict:
        return {"pid": os.getpid()}

if "extra" in shape:
    @mcp.tool()
    def get_restatements() -> dict:
        return {"pid": os.getpid()}

mcp.run(show_banner=False)
"""


def test_one_undeclared_tool_is_refused_alone_at_startup_and_after_an_eof_republish(
  tmp_path, monkeypatch,
):
  """The production shape (edgar-parser-mcp gaining get_restatements): one declared tool and
  one undeclared tool on the same server. The declared tool stays published and callable;
  only the undeclared one is refused. A declared tool that goes missing, or whose schema
  changes, is not contained: the published surface follows the live server."""
  monkeypatch.setenv("MCP_STDIO_CONNECT_RETRIES", "0")
  monkeypatch.setenv("MCP_STDIO_CONNECT_STABILIZE_S", "0")
  monkeypatch.setattr(
    mcp_client_module, "_stdio_recovery_delay", lambda _attempt: 0.05, raising=False,
  )
  script = tmp_path / "mcp_server.py"
  script.write_text(_SHAPED_STDIO_SERVER)
  (tmp_path / "shape").write_text("extra")
  manager = McpClientManager(
    config_path=None,
    inline_servers={
      "probe": {"command": sys.executable, "args": [str(script)], "cwd": str(tmp_path)},
    },
    tool_registration_catalog=_registered_probe_catalog(),
    tool_policy_implementations=_identity_policy_registry(),
    input_preparation_context_factory=lambda _descriptor, _scope: None,
    redaction_context_factory=lambda _descriptor: None,
  )

  def surface() -> tuple[list[str], set[tuple[str, str]]]:
    return (
      sorted(str(definition["name"]) for definition in manager.get_tool_definitions()),
      {(route.logical_server_id, route.logical_name) for route in manager.get_refused_mcp_routes()},
    )

  async def republish(shape: str):
    previous = manager._servers["probe"]
    (tmp_path / "shape").write_text(shape)
    os.kill(_flaky_starts(tmp_path)[-1], signal.SIGTERM)
    await _until(
      lambda: manager._servers["probe"] is not previous and manager.get_startup_diagnostics() == {},
    )

  async def scenario():
    await manager.startup()
    try:
      for label in ("startup", "after-eof-republish"):
        if label != "startup":
          await republish("extra")
        assert surface() == (["read_probe"], {("probe", "get_restatements")}), label
        result, error = await manager.call_tool("read_probe", {})
        assert error is None, label
        assert result == {"pid": _flaky_starts(tmp_path)[-1]}, label
        result, error = await manager.call_tool("get_restatements", {})
        assert result is None, label
        assert error is not None and error["code"] == "mcp_tool_not_registered", label
        assert "get_restatements" in error["message"]

      # A declared tool that went missing is simply gone: no definition, no route to call.
      await republish("extra,missing")
      assert surface() == ([], {("probe", "get_restatements")})
      result, error = await manager.call_tool("read_probe", {})
      assert result is None
      assert error is not None and error["code"] == "unknown_tool"

      # A declared tool whose schema changed is published with the live schema, and a
      # call that does not satisfy it is refused by the server, never answered.
      await republish("extra,schema")
      definitions = {str(item["name"]): item for item in manager.get_tool_definitions()}
      assert definitions["read_probe"]["input_schema"]["required"] == ["ticker"]
      result, error = await manager.call_tool("read_probe", {})
      assert result is None
      assert error is not None
      result, error = await manager.call_tool("read_probe", {"ticker": "AAPL"})
      assert error is None
      assert result == {"pid": _flaky_starts(tmp_path)[-1], "ticker": "AAPL"}
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


_HOLDING_STDIO_SERVER = _FLAKY_STDIO_SERVER.replace("import os\n", "import asyncio\nimport os\n", 1).replace(
  "mcp.run(show_banner=False)",
  """@mcp.tool()
async def hold() -> dict:
    Path("holding").write_text(str(os.getpid()))
    await asyncio.Event().wait()
    return {}

mcp.run(show_banner=False)""",
)


def test_a_call_in_flight_at_the_kill_leaves_the_recovery_owner_running(tmp_path, monkeypatch):
  """A read in flight when the child dies fails with a transport error and queues behind the
  watch's first connect round. When that round fails the call must not run its own round: closing
  the generation from the call path cancelled the watch task, and with no task left every later
  call answered mcp_server_unavailable while nothing reconnected the server (audit P1 d0c3f8c7d34a)."""
  manager = _flaky_stdio_manager(tmp_path, monkeypatch, server_source=_HOLDING_STDIO_SERVER)
  monkeypatch.setenv("MCP_STDIO_CONNECT_RETRIES", "2")
  monkeypatch.setenv("MCP_STDIO_CONNECT_BACKOFF_S", "0.3")

  async def scenario():
    await manager.startup()
    try:
      first_pid = _flaky_starts(tmp_path)[0]
      call = asyncio.create_task(manager.call_tool("hold", {}))
      await _until(lambda: (tmp_path / "holding").exists())
      (tmp_path / "refuse").write_text("x")
      os.kill(first_pid, signal.SIGTERM)

      result, error = await asyncio.wait_for(call, timeout=60)
      assert result is None
      assert error is not None
      watch = manager._servers["probe"].stdio_watch_task
      assert watch is not None and not watch.done()
      assert manager.get_startup_diagnostics()["probe"]["state"] == "reconnecting"

      # The cause clears: the owner's next round publishes, no call needed.
      (tmp_path / "refuse").unlink()
      await _until(lambda: manager.get_startup_diagnostics() == {})
      pids = _flaky_starts(tmp_path)
      assert len(pids) == 2 and pids[1] != first_pid
      result, error = await manager.call_tool("read_probe", {})
      assert error is None
      assert result == {"pid": pids[1]}
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


_DROPPING_HTTP_SERVER = """
import sys
from pathlib import Path

import uvicorn
from fastmcp import FastMCP

mcp = FastMCP("dropping")


def record(name: str) -> int:
    with Path(name).open("a") as stream:
        stream.write("x\\n")
    return len(Path(name).read_text().split())


@mcp.tool()
def read_probe() -> dict:
    return {"runs": record("reads")}


@mcp.tool()
def write_probe() -> dict:
    return {"runs": record("writes")}


inner = mcp.http_app(path="/mcp")


async def app(scope, receive, send):
    if scope["type"] != "http" or scope["method"] != "POST":
        await inner(scope, receive, send)
        return
    messages = []
    more = True
    while more:
        message = await receive()
        messages.append(message)
        more = message.get("more_body", False)
    pending = iter(messages)

    async def replayed():
        return next(pending, None) or await receive()

    body = b"".join(message.get("body", b"") for message in messages)
    drops = Path("drops")
    remaining = int(drops.read_text()) if drops.exists() else 0
    if b'"tools/call"' not in body or remaining <= 0:
        await inner(scope, replayed, send)
        return
    # The server runs the tool and its response never reaches the client: the
    # response stream closes empty, with no event and no result.
    drops.write_text(str(remaining - 1))

    async def discard(_message):
        return None

    await inner(scope, replayed, discard)
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"text/event-stream")]})
    await send({"type": "http.response.body", "body": b"", "more_body": False})


uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
"""


def _redacting_policy_registry() -> ToolPolicyImplementationRegistry:
  """The identity registry with a redactor that marks every value, so a redaction
  path that skipped execute_redaction and returned raw input would fail."""
  def entry(kind: PolicyKind, implementation: Any) -> ToolPolicyImplementation:
    return ToolPolicyImplementation(
      kind, f"{kind}.identity", "1", reject_policy_parameters, implementation,
    )

  return ToolPolicyImplementationRegistry((
    entry("redaction", lambda _ref, call: RedactionResult(
      {key: "<redacted>" for key in call.prepared_input},
    )),
    entry("planning", lambda _ref, _call: PlanDecision("none")),
    entry("input_preparation", lambda _ref, call: PreparedToolCall(call.raw_input)),
    entry("outcome", lambda _ref, _call: "ok"),
    entry("source_identity", lambda _ref, _call: SourceIdentityResult(())),
    entry("session_injection", lambda _ref, _call: (_ for _ in ()).throw(AssertionError("unused"))),
  ))


def _identity_policy_registry() -> ToolPolicyImplementationRegistry:
  def entry(kind: PolicyKind, implementation: Any) -> ToolPolicyImplementation:
    return ToolPolicyImplementation(
      kind, f"{kind}.identity", "1", reject_policy_parameters, implementation,
    )

  return ToolPolicyImplementationRegistry((
    entry("redaction", lambda _ref, call: RedactionResult(call.prepared_input)),
    entry("planning", lambda _ref, _call: PlanDecision("none")),
    entry("input_preparation", lambda _ref, call: PreparedToolCall(call.raw_input)),
    entry("outcome", lambda _ref, _call: "ok"),
    entry("source_identity", lambda _ref, _call: SourceIdentityResult(())),
    entry("session_injection", lambda _ref, _call: (_ for _ in ()).throw(AssertionError("unused"))),
  ))


def _registered_http_manager(tmp_path, port: int, **kwargs) -> McpClientManager:
  registry = _identity_policy_registry()
  return McpClientManager(
    config_path=None,
    inline_servers={"dropping": {"type": "streamable-http", "url": f"http://127.0.0.1:{port}/mcp"}},
    tool_registration_catalog=_registered_catalog(
      "dropping",
      {"read_probe": ("read", True), "write_probe": ("external_write", False)},
    ),
    tool_policy_implementations=registry,
    input_preparation_context_factory=lambda _descriptor, _scope: None,
    redaction_context_factory=lambda _descriptor: None,
    **kwargs,
  )


async def _dropping_http_server(tmp_path, drops: int):
  import socket

  with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
  script = tmp_path / "http_server.py"
  script.write_text(_DROPPING_HTTP_SERVER)
  (tmp_path / "drops").write_text(str(drops))
  process = await asyncio.create_subprocess_exec(
    sys.executable, str(script), str(port), cwd=str(tmp_path),
    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
  )
  async with asyncio.timeout(30):
    while True:
      try:
        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
      except OSError:
        await asyncio.sleep(0.05)
        continue
      writer.close()
      return process, port


def _runs(tmp_path, name: str) -> int:
  path = tmp_path / name
  return len(path.read_text().split()) if path.exists() else 0


@pytest.mark.parametrize(
  ("tool", "drops", "ran", "succeeds"),
  [
    ("read_probe", 1, 2, True),
    ("read_probe", 2, 2, False),
    ("write_probe", 1, 1, False),
  ],
  ids=["read-retried-once", "read-fails-after-one-retry", "write-never-replayed"],
)
def test_dropped_response_stream_retries_only_a_declared_read_once(
  tmp_path, tool, drops, ran, succeeds,
):
  async def scenario():
    process, port = await _dropping_http_server(tmp_path, drops)
    manager = _registered_http_manager(tmp_path, port)
    try:
      await manager.startup()
      assert manager.get_server_for_tool(tool) == "dropping"
      first = manager._servers["dropping"]
      result, error = await manager.call_tool(tool, {})
      if succeeds:
        assert error is None
        assert result == {"runs": 2}
        assert manager._servers["dropping"] is not first
      else:
        assert result is None
        assert error is not None
        assert error["code"] == "tool_error"
        assert error["sub_code"] == "connection_error"
        assert "SSE stream ended without a response" in error["message"]
      assert _runs(tmp_path, "reads" if tool == "read_probe" else "writes") == ran
    finally:
      await manager.shutdown()
      process.terminate()
      await process.wait()

  asyncio.run(scenario())


def test_dropped_per_user_stream_retries_a_read_on_a_fresh_per_user_session():
  drop = MCPError(CONNECTION_CLOSED, "SSE stream ended without a response")
  calls: list[str] = []

  class Session:
    def __init__(self, label, outcome):
      self.label, self.outcome = label, outcome

    async def call_tool(self, name, arguments, **_kwargs):
      calls.append(self.label)
      if isinstance(self.outcome, BaseException):
        raise self.outcome
      return self.outcome

  def state(label, outcome):
    server = SimpleNamespace(
      session=Session(label, outcome), config={"type": "streamable-http", "per_user": True},
    )
    return SimpleNamespace(server=server, active_calls=0, last_used_at=0.0, draining=False)

  ok = CallToolResult(content=[], structured_content={"from": "fresh"})

  async def scenario(replay_allowed):
    calls.clear()
    manager = McpClientManager(config_path=None)
    subject = SimpleNamespace(user_id="7")
    old = state("old", drop)
    fresh = state("fresh", ok)
    manager._per_user_servers[("remote", "7")] = cast(Any, old)
    drained: list[tuple[Any, str]] = []
    manager._schedule_drain = lambda target, reason: drained.append((target, reason))  # type: ignore[method-assign]

    async def get_fresh(*_args, **_kwargs):
      return fresh

    manager._get_per_user_server = get_fresh  # type: ignore[method-assign]
    try:
      result = await manager._call_tool_once_on_fresh_session_after_dropped_stream(
        server=cast(Any, old.server), per_user_state=cast(Any, old),
        per_user_subject=cast(Any, subject), server_name="remote",
        replay_allowed=replay_allowed, original_name="read_probe", tool_input={},
        meta=None, abort_event=None, expected_seconds=5, invocation_report=None,
        retry_refusals=True,
      )
    except MCPError as exc:
      return exc, drained, manager
    return result, drained, manager

  result, drained, manager = asyncio.run(scenario(True))
  assert result is ok
  assert calls == ["old", "fresh"]
  assert [reason for _state, reason in drained] == ["response_stream_dropped"]
  assert ("remote", "7") not in manager._per_user_servers

  result, drained, _manager = asyncio.run(scenario(False))
  assert result is drop
  assert calls == ["old"]
  assert drained == []
