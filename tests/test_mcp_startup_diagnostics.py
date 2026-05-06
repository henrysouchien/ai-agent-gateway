import asyncio
import logging

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
