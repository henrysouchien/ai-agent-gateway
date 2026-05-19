import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.mcp_client import McpClientManager
from agent_gateway.tool_dispatcher import ToolDispatcher


def _run(coro):
  return asyncio.run(coro)


class _FakeMcpClient:
  def __init__(self, server_name: str = "portfolio-mcp") -> None:
    self.server_name = server_name
    self.calls: list[dict[str, Any]] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name == "portfolio_tool"

  def get_server_for_tool(self, name: str) -> str | None:
    return self.server_name if name == "portfolio_tool" else None

  async def call_tool(self, name: str, tool_input: dict[str, Any], meta: dict[str, Any] | None = None):
    self.calls.append({"name": name, "tool_input": tool_input, "meta": meta})
    return {"ok": True}, None


class _FakeSession:
  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []

  async def call_tool(self, name: str, tool_input: dict[str, Any], *, read_timeout_seconds, meta=None):
    self.calls.append(
      {
        "name": name,
        "tool_input": tool_input,
        "read_timeout_seconds": read_timeout_seconds,
        "meta": meta,
      }
    )
    return SimpleNamespace(
      isError=False,
      structuredContent={"ok": True},
      content=None,
    )


@pytest.mark.parametrize("server_name", ["portfolio-mcp", "research-corpus-mcp"])
def test_tool_dispatcher_injects_user_id_into_mcp_meta(server_name: str) -> None:
  mcp = _FakeMcpClient(server_name=server_name)
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    user_id="alice",
    risk_user_id=42,
    channel="excel",
    role="invite",
    mcp_meta_inject_servers=frozenset({"portfolio-mcp", "research-corpus-mcp"}),
  )

  result, error = _run(dispatcher.dispatch("call-1", "portfolio_tool", {"ticker": "AAPL"}))

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls == [
    {
      "name": "portfolio_tool",
      "tool_input": {"ticker": "AAPL"},
      "meta": {"session_id": "sess-1", "user_id": "42", "channel": "excel", "role": "invite"},
    }
  ]


def test_tool_dispatcher_session_param_injection_still_works() -> None:
  mcp = _FakeMcpClient(server_name="session-param-server")
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    mcp_session_inject_servers={"session-param-server"},
  )

  result, error = _run(dispatcher.dispatch("call-1", "portfolio_tool", {"ticker": "AAPL"}))

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["tool_input"] == {"ticker": "AAPL", "_session_id": "sess-1"}
  assert mcp.calls[0]["meta"] is None


def test_tool_dispatcher_fails_closed_without_user_id_in_strict_mode() -> None:
  mcp = _FakeMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    user_id=None,
    risk_user_id=None,
    mcp_meta_inject_servers=frozenset({"portfolio-mcp", "research-corpus-mcp"}),
    credentials_resolver_active=True,
  )

  with pytest.raises(RuntimeError, match="MCP meta user_id is required in strict mode"):
    _run(dispatcher.dispatch("call-1", "portfolio_tool", {"ticker": "AAPL"}))

  assert mcp.calls == []


def test_mcp_client_call_tool_forwards_meta_to_underlying_session() -> None:
  manager = McpClientManager(config_path=None)
  session = _FakeSession()
  manager._tool_to_server = {"portfolio_tool": "portfolio-mcp"}
  manager._prefixed_to_original = {"portfolio_tool": "portfolio_tool"}
  manager._servers = {
    "portfolio-mcp": SimpleNamespace(session=session),
  }

  result, error = _run(
    manager.call_tool(
      "portfolio_tool",
      {"ticker": "AAPL"},
      meta={"session_id": "sess-1", "user_id": "42", "channel": "excel", "role": "invite"},
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert session.calls[0]["meta"] == {"session_id": "sess-1", "user_id": "42", "channel": "excel", "role": "invite"}
