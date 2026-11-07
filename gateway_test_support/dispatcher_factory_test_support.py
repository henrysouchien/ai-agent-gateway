from __future__ import annotations

from typing import Any

from agent_gateway.dispatcher_factory import GatewayDispatcherDeps
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.session import GatewaySession
from agent_gateway.tool_dispatcher_helpers import (
  ApprovalDecision,
  ApprovalRequest,
  InterceptContext,
  InterceptDecision,
  ToolResult,
)


def _never_needs_approval(
  _name: str,
  _tool_input: object,
  _qualifier: str,
) -> bool:
  return False


async def _local_handler(_tool_input: object) -> ToolResult:
  return {"ok": True}, None


async def _request_approval(
  _request: ApprovalRequest,
) -> ApprovalDecision:
  return ApprovalDecision(approved=False)


async def _allow_interceptor(
  _context: InterceptContext,
) -> InterceptDecision:
  return InterceptDecision(action="allow")


def _session(
  *,
  kind: str = "chat",
  channel: str = "web",
) -> GatewaySession:
  return GatewaySession(
    session_id=f"session-{kind}-{channel}",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="42",
    risk_user_id=42,
    role="owner",
    kind=kind,  # type: ignore[arg-type]
    channel=channel,
    auth_config={"api_key": "secret"},
  )


def _deps(
  *,
  mcp_client: McpClientManager | None = None,
  approval_store: Any = None,
  approval_policy: Any = None,
  mcp_meta_inject_servers: frozenset[str] = frozenset({
    "portfolio-reads-mcp"
  }),
) -> GatewayDispatcherDeps:
  return GatewayDispatcherDeps(
    mcp_client=mcp_client or McpClientManager(config_path=None),
    approval_store=approval_store,
    approval_policy=approval_policy,
    mcp_meta_inject_servers=mcp_meta_inject_servers,
  )
