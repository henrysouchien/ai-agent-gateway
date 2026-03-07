from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
  from .mcp_client import McpClientManager


ToolResult = Tuple[Optional[Any], Optional[Dict[str, Any]]]


@dataclass
class ApprovalRequest:
  tool_call_id: str
  nonce: str
  tool_name: str
  tool_input: Dict[str, Any]
  timeout: float = 120.0


@dataclass
class ApprovalDecision:
  approved: bool
  allow_tool_type: bool = False


ApprovalCallback = Callable[[ApprovalRequest], Awaitable[Optional[ApprovalDecision]]]
LocalToolHandler = Callable[..., Awaitable[ToolResult]]


class ToolDispatcher:
  def __init__(
    self,
    mcp_client: "McpClientManager",
    local_tool_handlers: Dict[str, LocalToolHandler] | None = None,
    needs_approval: Callable[[str], bool] | None = None,
    request_approval: ApprovalCallback | None = None,
    approved_tool_types: Set[str] | None = None,
  ) -> None:
    self._mcp = mcp_client
    self._local = local_tool_handlers or {}
    self._needs_approval = needs_approval or (lambda _: False)
    self._request_approval = request_approval
    self._approved_tool_types = approved_tool_types if approved_tool_types is not None else set()

  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    call_index: int = 0,
  ) -> ToolResult:
    if self._should_request_approval(tool_name):
      if self._request_approval is None:
        return None, {
          "code": "approval_required",
          "message": f"Tool '{tool_name}' requires approval but no approval handler is configured",
        }
      decision = await self._request_approval(
        ApprovalRequest(
          tool_call_id=tool_call_id,
          nonce=os.urandom(8).hex(),
          tool_name=tool_name,
          tool_input=tool_input,
        )
      )
      if decision is None:
        return None, {"code": "approval_timeout", "message": "User did not respond within timeout"}
      if not decision.approved:
        return None, {"code": "user_denied", "message": "User denied execution"}
      if decision.allow_tool_type:
        self._approved_tool_types.add(tool_name)

    if tool_name in self._local:
      return await self._local[tool_name](tool_input, call_index=call_index)

    if self._mcp.is_mcp_tool(tool_name):
      return await self._mcp.call_tool(tool_name, tool_input)

    return None, {"code": "unknown_tool", "message": f"Unknown tool: {tool_name}"}

  def _should_request_approval(self, tool_name: str) -> bool:
    if tool_name in self._approved_tool_types:
      return False
    return self._needs_approval(tool_name)
