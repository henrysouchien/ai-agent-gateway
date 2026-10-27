from __future__ import annotations
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND

import asyncio
import sys
from pathlib import Path
from typing import Mapping


PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.mcp_client import McpClientManager, _ServerState  # noqa: E402
from agent_gateway.mcp_client_connections import McpToolCallResult  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from agent_gateway.tool_dispatcher import ToolDispatcher  # noqa: E402
from gateway_test_support.host_policy import owner_session_host_policy


SERVER = "research-corpus-mcp"


class _UnusedClientSession:
  async def call_tool(
    self,
    name: str,
    arguments: Mapping[str, object],
    *,
    read_timeout_seconds: float | None,
    meta: Mapping[str, object] | None = None,
  ) -> McpToolCallResult:
    raise AssertionError(f"unexpected physical MCP call: {name}")

def _definition() -> _ServerState:
  return _ServerState(
    SERVER,
    _UnusedClientSession(),
    [],
    [],
    {"thesis_list"},
    config={
      "command": "synthetic-research-server",
      "per_user": True,
      "env": {"MCP_SUBPROCESS": "true"},
    },
  )


def _session(user_id: int, email: str) -> GatewaySession:
  return GatewaySession(
    session_id=f"session-{user_id}",
    api_key_hash="synthetic-hash",
    created_at=1,
    expires_at=2,
    user_id=str(user_id),
    user_email=email,
    risk_user_id=user_id,
    owner_user_id=str(user_id),
    role="owner",
  )


def _manager() -> McpClientManager:
  manager = McpClientManager(config_path=None)
  manager._servers = {SERVER: _definition()}
  manager._tool_to_server = {"thesis_list": SERVER}
  manager._mcp_tool_names = {"thesis_list"}
  return manager


def test_dispatcher_preserves_meta_and_authenticated_session_for_user_server(monkeypatch, owner_session_host_policy) -> None:
  async def scenario() -> None:
    manager = _manager()
    gateway_session = _session(7, "user@example.com")
    captured = {}

    async def call_tool(name, tool_input, **kwargs):
      captured.update(name=name, tool_input=tool_input, kwargs=kwargs)
      return {"ok": True}, None

    monkeypatch.setattr(manager, "call_tool", call_tool)
    dispatcher = ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=manager,
      session=gateway_session,
      session_id="session-7",
      user_id="7",
      risk_user_id=7,
      channel="cli",
      role="owner",
      mcp_meta_inject_servers=frozenset({SERVER}),
    )
    result, error = await dispatcher.dispatch(
      "call-1",
      "thesis_list",
      {},
      advertised_tool_names=frozenset({"thesis_list"}),
    )

    assert error is None
    assert result == {"ok": True}
    assert captured["kwargs"]["gateway_session"] is gateway_session
    assert captured["kwargs"]["meta"] == {
      "session_id": "session-7",
      "user_id": "7",
      "channel": "cli",
      "role": "owner",
    }
    assert "session_token" not in captured["kwargs"]["meta"]

  asyncio.run(scenario())
