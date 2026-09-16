from __future__ import annotations

import asyncio
from typing import Any, Mapping

from agent_gateway.mcp_client import McpClientManager
from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway.tool_policy_registry import PreparedToolCall

def _run(coro):
  return asyncio.run(coro)


def _dispatch(
  dispatcher: ToolDispatcher,
  tool_call_id: str,
  tool_name: str,
  tool_input: dict[str, Any],
  **kwargs: Any,
):
  return dispatcher.dispatch(
    tool_call_id,
    tool_name,
    tool_input,
    advertised_tool_names=frozenset({tool_name}),
    **kwargs,
  )
class _FakeMcpClient(McpClientManager):
  def __init__(
    self,
    server_name: str = "portfolio-reads-mcp",
    *,
    tool_name: str = "portfolio_tool",
    original_names: dict[str, str] | None = None,
  ) -> None:
    super().__init__(config_path=None)
    self.server_name = server_name
    self.tool_name = tool_name
    self.original_names = original_names or {}
    self.calls: list[dict[str, Any]] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name == self.tool_name

  def get_server_for_tool(self, name: str) -> str | None:
    return self.server_name if name == self.tool_name else None

  def get_original_tool_name(self, name: str) -> str:
    return self.original_names.get(name, name)

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any] | PreparedToolCall,
    meta: dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: Mapping[str, object] | None = None,
  ):
    self.calls.append({"name": name, "tool_input": tool_input, "meta": meta})
    return {"ok": True}, None

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return [_portfolio_tool_def(name=self.tool_name)]
def _portfolio_tool_def(
  name: str = "portfolio_tool",
  *,
  properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
  return {
    "name": name,
    "description": "Portfolio tool",
    "input_schema": {
      "type": "object",
      "properties": properties
      if properties is not None
      else {
        "format": {"type": "string"},
        "portfolio_id": {"type": "string"},
        "portfolio_name": {"type": "string"},
      },
    },
  }
