# claude-gateway

A Python package for building streaming Claude agent applications with MCP (Model Context Protocol) tool support. Handles the agent loop, tool dispatch, MCP server management, and SSE event streaming — so you can focus on your domain logic.

## Install

```bash
pip install claude-gateway
```

## Quick Start

```python
import asyncio
from claude_gateway import AgentRunner, EventLog, McpClientManager, ToolDispatcher

async def main():
    # 1. Set up MCP client (reads ~/.claude.json for server configs)
    mcp = McpClientManager(
        allowed_servers={"my-mcp-server"},
        builtin_tool_names=set(),
    )
    await mcp.startup()

    # 2. Create a tool dispatcher with local handlers + MCP tools
    async def my_tool(tool_input, *, call_index=0):
        return {"result": f"Hello from {tool_input.get('name', 'world')}"}, None

    dispatcher = ToolDispatcher(
        mcp_client=mcp,
        local_tool_handlers={"greet": my_tool},
    )

    # 3. Create event log for SSE streaming
    event_log = EventLog(session_id="session-001")

    # 4. Run the agent
    runner = AgentRunner(
        event_log=event_log,
        dispatcher=dispatcher,
        session_id="session-001",
        auth_config={"auth_mode": "api", "api_key": "sk-ant-..."},
        mcp_client=mcp,
        get_tool_definitions=lambda: [
            {"name": "greet", "description": "Greet someone", "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }},
            *mcp.get_tool_definitions(),
        ],
    )

    # Start streaming (non-blocking, populates event_log)
    asyncio.create_task(runner.run(
        messages=[{"role": "user", "content": "Say hello"}],
        system_prompt="You are a helpful assistant.",
    ))

    # Consume SSE events
    async for entry in event_log.iter_from():
        print(entry.event)
        if entry.event.get("type") == "stream_complete":
            break

    await mcp.shutdown()

asyncio.run(main())
```

## API Reference

### `AgentRunner`

Core streaming agent loop. Sends messages to Claude, processes tool calls, and emits SSE events.

- **Hooks**: `on_tool_result`, `on_usage`, `on_tool_timing` — async callbacks for observability
- **Sub-agents**: `sub_agent_config` + `spawn_sub_agent()` for parallel task delegation
- **Streaming**: Emits `thinking_delta`, `text_delta`, `tool_use`, `tool_result`, `stream_complete` events

### `ToolDispatcher`

Routes tool calls to local handlers or MCP servers. Supports approval gates for sensitive operations.

- `local_tool_handlers`: `Dict[str, handler]` — local async functions
- `needs_approval` / `request_approval`: Optional approval flow for gated tools
- `approved_tool_types`: Session-persistent set of pre-approved tools

### `McpClientManager`

Manages MCP server connections via stdio. Reads server configs from `~/.claude.json`.

- `allowed_servers`: Whitelist of server names to connect
- `builtin_tool_names`: Tool names to skip (avoids collisions with local tools)
- `timeout_overrides`: Per-server call timeouts

### `EventLog`

Async event stream for SSE. Append events, iterate from any sequence number.

- `append(event)` — add an event
- `iter_from(after_seq)` — async iterator, blocks until new events arrive
- `close()` — emit terminal event and seal the log

### Supporting Types

- `ToolResult = Tuple[Optional[Any], Optional[Dict[str, Any]]]` — `(result, error)` pair
- `ApprovalRequest` / `ApprovalDecision` — approval flow dataclasses
- `SubAgentConfig` — excluded tools, system prompt, max turns for sub-agents
- `ToolResultContext` — full context passed to `on_tool_result` hook
- `LogEntry` — sequence number, timestamp, event dict

## License

MIT
