import asyncio
import logging
import time

from agent_gateway.mcp_client import McpClientManager


def test_connect_or_warn_logs_timeout_exception_class(caplog):
  manager = McpClientManager(config_path=None)

  async def _raise_timeout(_name, _config):
    raise asyncio.TimeoutError()

  manager._connect = _raise_timeout

  async def _run():
    return await manager._connect_or_warn("portfolio-mcp", {"command": "python3"})

  caplog.set_level(logging.WARNING, logger="agent_gateway.mcp_client")

  result = asyncio.run(_run())

  assert result is None
  assert "MCP server portfolio-mcp failed to connect: TimeoutError" in caplog.text


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
