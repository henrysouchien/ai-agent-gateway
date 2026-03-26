# ai-agent-gateway

Deploy AI agents as production services.

Other frameworks help you define what your agent does.
This handles everything around it — the HTTP server, session management, SSE streaming, tool dispatch, human-in-the-loop approval, and code execution sandboxing.

Start with a system prompt. Add MCP tools, local Python tools, skills, and code execution as you need them. Requires Python >= 3.10.

## Install + Quick Start

`create_agent()` is the fastest path to a working agent server. It uses Anthropic models by default. For OpenAI or custom providers, see [`create_gateway_app()`](./docs/api-reference.md#server).

Install the package and `uvicorn`:

```bash
pip install "ai-agent-gateway[anthropic]" uvicorn
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Create `agent.py`:

```python
from claude_gateway import create_agent

app = create_agent("You are a concise research assistant.")
```

Run the server:

```bash
uvicorn agent:app --reload --port 8000
```

Create a session token:

```bash
SESSION_TOKEN=$(curl -s http://127.0.0.1:8000/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"local-demo-key"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_token"])')
```

Chat with the agent:

```bash
curl -N http://127.0.0.1:8000/api/chat \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Give me three bullet points on why SSE is useful for chat UIs."}]}'
```

You will get an SSE stream like:

```text
data: {"type":"text_delta","text":"- SSE lets the server push tokens as they are generated.\n"}

data: {"type":"text_delta","text":"- The browser can render partial output without polling.\n"}

data: {"type":"stream_complete","usage":{"input_tokens":...,"output_tokens":...}}
```

Full 5-minute walkthrough: [Quickstart](./docs/quickstart.md)

## Features

- FastAPI server factory with `/api/chat/init`, `/api/chat`, `/api/chat/tool-result`, `/api/chat/tool-approval`, and `/api/health`
- SSE event stream for text deltas, thinking deltas, tool calls, approval requests, tool output chunks, retries, and completion
- JWT sessions with scoped approvals and isolated code execution directories
- MCP tool discovery from inline config or `~/.claude.json`
- Local Python tool handlers with the same dispatch loop as MCP tools
- Code execution with Docker preferred and subprocess fallback
- Markdown skill files (prompt + config per task) and sub-agents via the built-in `run_agent` tool
- Anthropic and OpenAI provider abstractions when you use `create_gateway_app()`

You bring your system prompt, your tools (MCP servers, local Python handlers, or both), and your runtime policy. The gateway handles everything else.

## Progressive Examples

### Tier 1: System Prompt Only

```python
from claude_gateway import create_agent

app = create_agent("You are a helpful assistant for spreadsheet users.")
```

### Tier 2: Add MCP Tools

This uses an inline MCP server config. The example below assumes Node.js is installed because it runs an `npx`-based MCP server.

```python
from claude_gateway import create_agent

app = create_agent(
  "You can inspect and edit files when needed.",
  mcp_servers={
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
    }
  },
)
```

### Tier 3: Add Local Tools

```python
from claude_gateway import create_agent


async def summarize_csv(tool_input, **_kwargs):
  path = tool_input["path"]
  return {"summary": f"Would summarize {path}"}, None


app = create_agent(
  "Use the summarize_csv tool when the user asks for a file summary.",
  tool_handlers={"summarize_csv": summarize_csv},
  tool_definitions=[
    {
      "name": "summarize_csv",
      "description": "Summarize a CSV file on disk.",
      "input_schema": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "Path to the CSV file."}
        },
        "required": ["path"],
      },
    }
  ],
)
```

### Tier 4: Add Code Execution and Skills

`code_execution=True` prefers Docker when available and falls back to local subprocess execution otherwise.

```python
from claude_gateway import create_agent

app = create_agent(
  "Use code execution for calculations and run_agent for focused subtasks.",
  code_execution=True,       # Adds code_execute tool (Docker preferred, subprocess fallback)
  skills_dir="skills",       # Each .md file becomes a named skill for run_agent
)
```

### Graduate: Switch to `create_gateway_app()`

Use `create_gateway_app()` when you need OpenAI, custom approval logic, channel-aware runtimes, interceptors, or production hooks.

```python
from claude_gateway import (
  AnthropicProvider, ChatRuntime, GatewayServerConfig, create_gateway_app,
)

# Full control: custom providers, approval logic, channel routing, interceptors.
# See examples/07-full-production/ for the complete version.
app = create_gateway_app(
  GatewayServerConfig(
    build_chat_runtime=my_runtime_factory,
    default_provider=AnthropicProvider(),
  )
)
```

Runnable versions of these examples live in [`examples/`](./examples/).

## Architecture

```text
Client
  |
  |-- POST /api/chat/init --> JWT session token
  |
  |-- POST /api/chat (Bearer token, SSE stream)
  |       |
  |       v
  |   ChatRuntime (built per-request)
  |       |
  |       v
  |   AgentRunner (model loop: stream -> tool calls -> dispatch -> resume)
  |       |
  |       v
  |   ToolDispatcher
  |       |-- interceptors (rate limits, custom policies)
  |       |-- approval check (session-scoped)
  |       |-- local Python handler
  |       |-- MCP server (stdio)
  |       |-- code_execute (Docker / subprocess)
  |       |-- run_agent (sub-agent with own runner)
  |       |
  |       v
  |   EventLog --> SSE events to client
  |
  |-- POST /api/chat/tool-approval (human-in-the-loop)
```

The same backend can serve multiple frontends. Pass `context.channel` to shape runtime behavior per client without rewriting the agent loop.

## Comparison

| Category | ai-agent-gateway | LangGraph | LangChain | CrewAI | mcp-agent |
| --- | --- | --- | --- | --- | --- |
| Primary purpose | Deploying agents as services | Stateful workflow graphs | LLM app building blocks | Multi-agent role/task orchestration | MCP-centric workflow orchestration |
| Agent logic | Model-driven prompts with tools | Code-defined graph nodes and edges | Code-defined chains and agents | Code-defined crews and tasks | Code-defined workflows |
| Tool system | MCP-native plus local handlers | Bring your own adapters | Bring your own adapters | Custom tool abstractions | MCP-native |
| Server/runtime | FastAPI + SSE in core | Bring your own or LangGraph Platform | LangServe is separate | Bring your own | Bring your own |
| Sessions/auth | JWT sessions in core | Bring your own | Bring your own | Bring your own | Bring your own |
| Human approval | Built into tool dispatch | Available through interrupt/checkpoint patterns | Not a core runtime feature | Human-input patterns available | Not a core runtime feature |
| Best for | Shipping a chat-facing agent backend quickly | Explicit workflow control flow | Reusable LLM components | Team-style agent simulations | MCP-heavy automation flows |

When to use this package: you want users or clients talking to an agent over HTTP, and you do not want to build the session, SSE, approval, and tool-serving infrastructure yourself.

When not to use this package: you want explicit graph orchestration, or you are building a one-off notebook or script that does not need a reusable server runtime.

You can also combine them. For example, a LangGraph workflow can sit behind an `ai-agent-gateway` HTTP surface.

## Documentation

- [Quickstart](./docs/quickstart.md)
- [HTTP API](./docs/http-api.md)
- [Architecture](./docs/architecture.md)
- [Comparison](./docs/comparison.md)
- [API Reference](./docs/api-reference.md)
- [Contributing](./CONTRIBUTING.md)

## License

MIT
