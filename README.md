# ai-agent-gateway

Source directory in this monorepo: `packages/claude-gateway/`
Import package: `claude_gateway`
Published package name: `ai-agent-gateway`

Python package for building streaming agent backends with MCP tools, approval gates, memory, skills, and multi-model provider support.

## Install

```bash
pip install ai-agent-gateway[anthropic]
```

Or:

```bash
pip install ai-agent-gateway[openai]
```

For local development from this repo:

```bash
pip install -e .
```

## Quick Prompt Example

```python
from claude_gateway import send_prompt_sync

text = send_prompt_sync(
    "Summarize the key risk in one sentence.",
    model="claude-sonnet-4-6",
    system_prompt="You are a concise analyst.",
)

print(text)
```

`send_prompt_sync()` is the lightest-weight entry point when you only need a single model call.

## Main Building Blocks

- `AgentRunner`: streaming multi-turn tool-calling loop
- `ToolDispatcher`: local + MCP tool routing with approval hooks
- `McpClientManager`: stdio MCP client lifecycle and tool discovery
- `create_gateway_app()`: FastAPI server factory for SSE chat backends
- `AnthropicProvider` / `OpenAIProvider`: model-provider implementations
- `MemoryStore` / `MarkdownSyncManager`: persistent memory primitives
- `SkillLoader` / `SkillStateStore`: markdown skill loading and state tracking
- `send_prompt()` / `send_prompt_sync()`: convenience helpers for single prompts

## Typical Usage

Use this package when you want one or more of these:

- a reusable SSE chat backend instead of writing the HTTP/session layer yourself
- MCP tool discovery and dispatch from `~/.claude.json`
- approval-gated tool execution
- multi-channel agents that share the same backend runtime
- persistent memory and markdown sync
- a provider abstraction over Anthropic and OpenAI

## Server Factory

For a full backend, compose `GatewayServerConfig`, `ChatRuntime`, and `create_gateway_app()`:

- `GatewayServerConfig` owns auth, sessions, CORS, model allowlists, and hook wiring
- `ChatRuntime` supplies the system prompt, tool list, dispatcher builder, and optional runtime hooks
- `create_gateway_app()` returns a FastAPI app with `/api/chat/init`, `/api/chat`, `/api/chat/tool-result`, and tool-approval endpoints

The AI-excel-addin app in this repo is the main production example of that pattern.

## License

MIT
