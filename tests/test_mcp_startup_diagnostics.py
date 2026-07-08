import asyncio
import json
import logging
import time

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager


def test_connect_or_warn_logs_timeout_exception_class(caplog):
  manager = McpClientManager(config_path=None)

  async def _raise_timeout(_name, _config):
    raise asyncio.TimeoutError()

  manager._connect = _raise_timeout

  async def _run():
    return await manager._connect_or_warn("portfolio-reads-mcp", {"command": "python3"})

  caplog.set_level(logging.WARNING, logger="agent_gateway.mcp_client")

  result = asyncio.run(_run())

  assert result is None
  assert "MCP server portfolio-reads-mcp failed to connect: TimeoutError" in caplog.text
  assert manager.get_startup_diagnostics()["portfolio-reads-mcp"] == {
    "server": "portfolio-reads-mcp",
    "category": "transient_timeout",
    "retryable": True,
    "message": "TimeoutError",
    "error_type": "TimeoutError",
  }


def test_startup_records_missing_requested_server_config(tmp_path):
  config_path = tmp_path / "mcp.json"
  config_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
  manager = McpClientManager(
    config_path=config_path,
    allowed_servers={"research-corpus-mcp"},
  )

  asyncio.run(manager.startup(allowed_servers={"research-corpus-mcp"}))

  assert manager.get_startup_diagnostics()["research-corpus-mcp"] == {
    "server": "research-corpus-mcp",
    "category": "config_missing",
    "retryable": False,
    "message": "server requested for startup but absent from the MCP config",
  }


def test_startup_without_requested_servers_does_not_mark_allowlist_missing(tmp_path):
  config_path = tmp_path / "mcp.json"
  config_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
  manager = McpClientManager(
    config_path=config_path,
    allowed_servers={"research-corpus-mcp"},
  )

  asyncio.run(manager.startup())

  assert manager.get_startup_diagnostics() == {}


def test_startup_records_disallowed_requested_server(tmp_path):
  config_path = tmp_path / "mcp.json"
  config_path.write_text(
    json.dumps({"mcpServers": {"research-corpus-mcp": {"command": "python3"}}}),
    encoding="utf-8",
  )
  manager = McpClientManager(
    config_path=config_path,
    allowed_servers={"portfolio-reads-mcp"},
  )

  asyncio.run(manager.startup(allowed_servers={"research-corpus-mcp"}))

  assert manager.get_startup_diagnostics()["research-corpus-mcp"] == {
    "server": "research-corpus-mcp",
    "category": "not_allowed",
    "retryable": False,
    "message": "server requested for startup but absent from the MCP allowlist",
  }


def test_startup_records_invalid_and_unsupported_configs(tmp_path):
  config_path = tmp_path / "mcp.json"
  config_path.write_text(
    json.dumps({
      "mcpServers": {
        "bad-shape": [],
        "bad-type": {"type": "websocket", "url": "ws://localhost"},
      }
    }),
    encoding="utf-8",
  )
  manager = McpClientManager(
    config_path=config_path,
    allowed_servers={"bad-shape", "bad-type"},
  )

  asyncio.run(manager.startup(allowed_servers={"bad-shape", "bad-type"}))

  diagnostics = manager.get_startup_diagnostics()
  assert diagnostics["bad-shape"] == {
    "server": "bad-shape",
    "category": "invalid_config",
    "retryable": False,
    "message": "invalid MCP server config",
  }
  assert diagnostics["bad-type"] == {
    "server": "bad-type",
    "category": "unsupported_type",
    "retryable": False,
    "message": "unsupported MCP server type: websocket",
  }


class McpError(Exception):
  pass


class _FakeExceptionGroup(Exception):
  def __init__(self, *exceptions: Exception):
    super().__init__("unhandled errors in a TaskGroup")
    self.exceptions = exceptions


def test_stdio_connect_retries_transient_connection_closed_group(monkeypatch, caplog):
  monkeypatch.setenv("MCP_STDIO_CONNECT_RETRIES", "2")
  monkeypatch.setenv("MCP_STDIO_CONNECT_BACKOFF_S", "0")
  manager = McpClientManager(config_path=None)
  connected_state = object()
  attempts: list[int] = []

  async def _flaky_connect_stdio(_name, _config):
    attempts.append(1)
    if len(attempts) < 3:
      raise _FakeExceptionGroup(McpError("Connection closed"))
    return connected_state

  manager._connect_stdio = _flaky_connect_stdio
  caplog.set_level(logging.WARNING, logger="agent_gateway.mcp_client")

  result = asyncio.run(manager._connect("fmp-mcp", {"type": "stdio", "command": "python3"}))

  assert result is connected_state
  assert len(attempts) == 3
  assert "MCP stdio server fmp-mcp connect attempt 1/3 failed" in caplog.text
  assert "MCP stdio server fmp-mcp connect attempt 2/3 failed" in caplog.text


def test_stdio_connect_exhausts_bounded_retries(monkeypatch, caplog):
  monkeypatch.setenv("MCP_STDIO_CONNECT_RETRIES", "2")
  monkeypatch.setenv("MCP_STDIO_CONNECT_BACKOFF_S", "0")
  manager = McpClientManager(config_path=None)
  attempts: list[int] = []

  async def _always_closed(_name, _config):
    attempts.append(1)
    raise _FakeExceptionGroup(McpError("Connection closed"))

  manager._connect_stdio = _always_closed
  caplog.set_level(logging.WARNING, logger="agent_gateway.mcp_client")

  async def _run():
    return await manager._connect_or_warn("fmp-mcp", {"type": "stdio", "command": "python3"})

  result = asyncio.run(_run())

  assert result is None
  assert len(attempts) == 3
  assert caplog.text.count("retrying in 0.00s") == 2
  assert "MCP server fmp-mcp failed to connect: unhandled errors in a TaskGroup" in caplog.text
  assert manager.get_startup_diagnostics()["fmp-mcp"]["category"] == "transient_transport"
  assert manager.get_startup_diagnostics()["fmp-mcp"]["retryable"] is True


def test_stdio_missing_command_is_not_retried(monkeypatch, caplog):
  monkeypatch.setenv("MCP_STDIO_CONNECT_RETRIES", "5")
  monkeypatch.setenv("MCP_STDIO_CONNECT_BACKOFF_S", "0.01")
  sleep_calls: list[float] = []

  async def _fake_sleep(delay):
    sleep_calls.append(delay)

  monkeypatch.setattr(mcp_client_module.asyncio, "sleep", _fake_sleep)
  manager = McpClientManager(config_path=None)
  caplog.set_level(logging.WARNING, logger="agent_gateway.mcp_client")

  async def _run():
    return await manager._connect_or_warn("bad-stdio", {"type": "stdio"})

  result = asyncio.run(_run())

  assert result is None
  assert sleep_calls == []
  assert "missing command" in caplog.text
  assert "retrying" not in caplog.text
  assert manager.get_startup_diagnostics()["bad-stdio"]["category"] == "config_error"
  assert manager.get_startup_diagnostics()["bad-stdio"]["retryable"] is False


class ClosedResourceError(Exception):
  pass


class _ToolResult:
  isError = False
  content = []
  structuredContent = {"ok": True}


def test_stdio_tool_call_reconnects_once_on_closed_transport(monkeypatch, caplog):
  manager = McpClientManager(config_path=None)

  class _ClosedSession:
    async def call_tool(self, _name, _tool_input, **_kwargs):
      raise ClosedResourceError()

  class _ReplacementSession:
    async def call_tool(self, _name, _tool_input, **_kwargs):
      return _ToolResult()

  class _CloseContext:
    closed = False

    async def __aexit__(self, exc_type, exc, tb):
      self.closed = True

  close_context = _CloseContext()
  config = {"type": "stdio", "command": "fake-mcp-server"}
  manager._servers = {
    "fmp-mcp": mcp_client_module._ServerState(
      name="fmp-mcp",
      session=_ClosedSession(),
      exit_contexts=[close_context],
      tool_definitions=[],
      tool_names={"fmp_profile"},
      config=config,
    )
  }
  manager._tool_to_server = {"fmp_profile": "fmp-mcp"}
  reconnects: list[tuple[str, dict]] = []

  async def _fake_reconnect(name, reconnect_config):
    reconnects.append((name, reconnect_config))
    return mcp_client_module._ServerState(
      name=name,
      session=_ReplacementSession(),
      exit_contexts=[],
      tool_definitions=[],
      tool_names={"fmp_profile"},
      config=reconnect_config,
    )

  manager._connect_stdio_with_retries = _fake_reconnect
  caplog.set_level(logging.WARNING, logger="agent_gateway.mcp_client")

  result, error = asyncio.run(manager.call_tool("fmp_profile", {"ticker": "WM"}))

  assert result == {"ok": True}
  assert error is None
  assert reconnects == [("fmp-mcp", config)]
  assert close_context.closed is True
  assert "tool call fmp_profile failed with transient transport error" in caplog.text


def test_close_contexts_suppresses_known_sdk_process_group_fallback_warning(caplog):
  manager = McpClientManager(config_path=None)

  class _NoisyCloseContext:
    async def __aexit__(self, exc_type, exc, tb):
      logging.getLogger("mcp.os.posix.utilities").warning(
        "Process group termination failed for PID 15032: [Errno 1] Operation not permitted, "
        "falling back to simple terminate"
      )
      logging.getLogger("mcp.os.posix.utilities").warning("Unexpected MCP shutdown warning")

  caplog.set_level(logging.WARNING)

  asyncio.run(manager._close_contexts([_NoisyCloseContext()]))

  assert "Process group termination failed for PID 15032" not in caplog.text
  assert "Unexpected MCP shutdown warning" in caplog.text


def test_close_contexts_suppresses_known_sdk_process_exit_race_noise(caplog):
  manager = McpClientManager(config_path=None)

  class _NoisyCloseContext:
    async def __aexit__(self, exc_type, exc, tb):
      logger = logging.getLogger("mcp.os.posix.utilities")
      logger.warning("Process termination failed for PID 92894, attempting force kill")
      try:
        raise ProcessLookupError()
      except ProcessLookupError:
        logger.exception("Failed to kill process 92894")
      try:
        raise PermissionError("permission denied")
      except PermissionError:
        logger.exception("Failed to kill process 92895")

  caplog.set_level(logging.WARNING)

  asyncio.run(manager._close_contexts([_NoisyCloseContext()]))

  assert "Process termination failed for PID 92894" not in caplog.text
  assert "Failed to kill process 92894" not in caplog.text
  assert "Failed to kill process 92895" in caplog.text


def test_close_contexts_times_out_hanging_context(caplog):
  manager = McpClientManager(config_path=None)

  class _HangingCloseContext:
    async def __aexit__(self, exc_type, exc, tb):
      await asyncio.sleep(60)

  caplog.set_level(logging.WARNING, logger="agent_gateway.mcp_client")
  started = time.monotonic()

  asyncio.run(manager._close_contexts([_HangingCloseContext()], close_timeout_seconds=0.01))

  assert time.monotonic() - started < 1
  assert "MCP context close timed out" in caplog.text


def test_close_contexts_suppresses_cancelled_cleanup(caplog):
  manager = McpClientManager(config_path=None)

  class _CancelledCloseContext:
    async def __aexit__(self, exc_type, exc, tb):
      raise asyncio.CancelledError("Cancelled via cancel scope")

  caplog.set_level(logging.DEBUG, logger="agent_gateway.mcp_client")

  asyncio.run(manager._close_contexts([_CancelledCloseContext()]))

  assert "MCP context close cancelled during cleanup" in caplog.text
