# Build A Domain Agent

Use `create_agent()` for simple agents. Use `create_gateway_app()` when the
agent is part of a larger product with its own auth, domain tools, storage,
approval rules, and background workflows.

This guide describes the consumer pattern used by domain systems such as a
portfolio research assistant.

## Runtime Shape

The product owns domain behavior. `ai-agent-gateway` owns the runtime shell.

```text
domain app
  auth, user profile, memory, domain tools
        |
        v
GatewayServerConfig
        |
        v
build_chat_runtime(session, request, channel, auth_manager)
        |
        v
ChatRuntime
        |
        v
AgentRunner + ToolDispatcher
```

Keep these boundaries explicit:

- `GatewayServerConfig` describes HTTP/session/auth hooks.
- `build_chat_runtime` chooses the prompt, model, tools, limits, and hooks for one request.
- `ChatRuntime` is the per-request wiring bundle returned to the gateway.
- `AgentRunner` runs the model/tool loop.
- `ToolDispatcher` is the policy boundary before any local or MCP tool executes.

## Minimal App

```python
from __future__ import annotations

import os
from typing import Any

from agent_gateway import (
  AgentRunner,
  AnthropicProvider,
  ChatRequest,
  ChatRuntime,
  GatewayServerConfig,
  ToolDispatcher,
  create_gateway_app,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.session import AuthManager, GatewaySession


provider = AnthropicProvider()
auth_config = {
  "auth_mode": "api",
  "api_key": os.environ["ANTHROPIC_API_KEY"],
  "model": "claude-sonnet-4-6",
}
mcp_client = McpClientManager(
  config_path=None,
  inline_servers={
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
    }
  },
)


async def build_chat_runtime(
  session: GatewaySession,
  request: ChatRequest,
  channel: str | None,
  auth_manager: AuthManager,
) -> ChatRuntime:
  user_id = request.user_id or session.user_id
  prompt = (
    "You are a domain assistant. Use tools for facts, cite tool outputs, "
    "and ask for approval before mutating external state."
  )

  def build_runner(event_log, session_id: str, started_at: float) -> AgentRunner:
    dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      needs_approval=lambda tool_name, tool_input=None, **_: tool_name.startswith("write_"),
      event_log=event_log,
      session_id=session_id,
      user_id=user_id,
      channel=channel,
    )
    return AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id=session_id,
      started_at=started_at,
      provider=provider,
      auth_config=session.auth_config or auth_config,
      get_tool_definitions=mcp_client.get_tool_definitions,
      mcp_client=mcp_client,
      user_id=user_id,
      channel=channel,
      rate_table_version="default",
      billing_mode="byok",
    )

  return ChatRuntime(
    system_prompt=prompt,
    build_runner=build_runner,
    get_tool_definitions=mcp_client.get_tool_definitions,
    provider=provider,
    max_turns=8,
  )


app = create_gateway_app(
  GatewayServerConfig(
    valid_api_keys={os.environ["DOMAIN_GATEWAY_API_KEY"]},
    build_chat_runtime=build_chat_runtime,
    mcp_client=mcp_client,
    prefix="/api",
  )
)
```

The gateway calls `build_chat_runtime` for every chat turn. Use request context
there to select the domain profile, channel, model, tool set, and memory scope.

## Add Domain Tools

Use MCP servers when tools are already separate processes or need their own
release cadence. Use local handlers for small in-process product operations.

```python
async def read_position(input: dict[str, Any], **_: Any) -> tuple[dict[str, Any], None]:
  ticker = str(input["ticker"]).upper()
  return {"ticker": ticker, "weight": 0.04}, None


tool_handlers = {"read_position": read_position}
tool_definitions = [
  {
    "name": "read_position",
    "description": "Read the current portfolio weight for one ticker.",
    "input_schema": {
      "type": "object",
      "properties": {"ticker": {"type": "string"}},
      "required": ["ticker"],
    },
  }
]
```

Pass those into `ToolDispatcher(local_tool_handlers=tool_handlers)` and return
`tool_definitions + mcp_client.get_tool_definitions()` from `ChatRuntime`.

## Validate Tool Use

Put product policy in dispatch, not in prompt text alone.

Use `ToolDispatcher` controls for:

- `needs_approval`: request human approval before sensitive tools.
- `interceptors`: warn, ask, allow, or deny before execution.
- `allowed_mcp_tools_by_server`: narrow a server surface for one runtime.
- `session_cache_denied_tools`: avoid repeated asks for denied tool classes.
- `on_tool_result`: sanitize, summarize, or record structured tool metadata.

For example, a portfolio app can allow read-only market data tools while asking
for approval before order, notification, or artifact-write tools.

## Memory

The gateway includes generic memory primitives such as `MemoryStore` and
`MarkdownSyncManager`. A domain app should own the schema of its domain memory.

Recommended pattern:

1. Store durable product facts in product tables or files.
2. Mirror agent-readable summaries into `MemoryStore` or markdown sync.
3. Scope reads by `user_id`, tenant, portfolio, channel, or profile.
4. Treat memory writes as tool calls so they pass through policy and audit.

This keeps the gateway generic while giving the domain app control over
retention, privacy, and meaning.

## Autonomous Workflows

For cron or batch work, reuse the same runtime pieces:

- build the same provider and MCP clients
- use the same domain prompts and tool policy
- call `run_autonomous()` or `run_autonomous_sync()`
- deliver results through the product's notification layer

The key design rule is that background runs and interactive chats should share
domain validation, memory scoping, and tool approval semantics. Only delivery
and scheduling should differ.

## Production Checklist

- Define the product-owned `build_chat_runtime` callback.
- Keep tool schemas close to their handlers or MCP servers.
- Put mutation policy in dispatcher/interceptor code.
- Set explicit `user_id`, `channel`, `rate_table_version`, and `billing_mode`
  when using `AgentRunner` or `AgentSDKRunner`.
- Wire `on_usage`, `on_session_summary`, and `on_late_usage_event` before launch.
- Store transcripts and tool audit records in product-owned locations.
- Keep domain memory schema outside the gateway package.
- Use `create_agent()` only for simple deployments; use `create_gateway_app()`
  for product integrations.
