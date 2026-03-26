# API Reference

This is a hand-written guide to the main public APIs in `claude_gateway`.

For endpoint payloads and SSE event schemas, see [HTTP API](./http-api.md). For architecture and control flow, see [Architecture](./architecture.md).

## Getting Started

### `create_agent()`

Fastest way to create a gateway server from a system prompt.

Use it when you want:

- Anthropic-backed chat
- automatic session and SSE endpoints
- optional MCP tools
- optional local tools
- optional code execution
- optional skills

Important limitation:

- `create_agent()` currently uses `AnthropicProvider` only

```python
from claude_gateway import create_agent

app = create_agent("You are a concise assistant.")
```

### `send_prompt()` and `send_prompt_sync()`

Single-call helpers for Anthropic text generation without the HTTP server.

Use them when:

- you want one prompt/response
- you do not need sessions or SSE
- you still want the same provider normalization as the gateway

## Server

### `create_gateway_app(config)`

Low-level FastAPI server factory.

Use it when you need:

- OpenAI
- custom approval logic
- interceptors
- multiple runtime profiles
- advanced auth, CORS, or transcript behavior

### `GatewayServerConfig`

Top-level app configuration.

Key fields:

- `build_chat_runtime`: required async factory that returns `ChatRuntime`
- `valid_api_keys`, `jwt_secret`, `session_ttl`: auth and session envelope
- `allowed_models`: request-time allowlist
- `cors_origins`, `prefix`: HTTP surface
- `on_event`, `on_startup`, `on_shutdown`: app lifecycle hooks
- `transcript_dir`: JSONL transcript output

### `ChatRuntime`

Per-request runtime description.

Key fields:

- `system_prompt`
- `build_runner`
- `get_tool_definitions`
- `provider`
- `model_override`
- `max_turns`
- `execution_location`

### `RequestContext`

Container for mutable request-scoped objects such as the active `Session`, `EventLog`, and approval callback.

## Agent Loop

### `AgentRunner`

The main tool-calling loop used by the gateway.

Responsibilities:

- stream model output
- execute tools
- retry transient stream failures
- emit client-visible events
- track usage and estimated cost
- stop on budget or max-turn limits

### `SubAgentConfig`

Defaults applied when the runner spawns sub-agents through `run_agent`.

### `ToolResultContext`

Payload passed to `on_tool_result` hooks after a tool completes.

Useful when you want to:

- sanitize large tool payloads
- inject follow-up content blocks
- log structured tool metadata

### `AgentSDKRunner` and `AgentSDKConfig`

Alternative execution path that keeps the same gateway HTTP surface but delegates tool-loop behavior to the pinned Anthropic agent SDK.

Use this when SDK parity matters more than native-runner control.

## Tools And Approval

### `ToolDispatcher`

Routes tool calls to local handlers or MCP servers and applies:

- interceptors
- approval rules
- in-session approval caching

### `ToolExecutionContext`

Optional context object passed to local tool handlers. Use `emit()` to stream extra structured events such as `tool_output_chunk`.

### `ToolResult`

Standard tool return shape: `(result, error)`.

### `ApprovalRequest` and `ApprovalDecision`

Types used by the approval loop.

`ApprovalDecision.allow_tool_type` persists approval for that tool type within the session.

### `InterceptContext`, `InterceptDecision`, and `ToolInterceptor`

Interceptor contract for runtime tool policy.

Use interceptors when you want to warn or deny before any tool executes.

## Code Execution

### `CodeExecutionConfig`

Configures built-in code execution:

- backend registration
- Docker image selection
- environment preparation
- work directory root
- timeouts and output limits
- tool description customization

### `CodeExecutionBundle`

Return type from `build_code_execution()`.

Contains:

- local handlers
- tool definitions
- approval helpers
- result sanitization hook

### `build_code_execution(session, config=None)`

Inject built-in code execution tools into a runtime.

Behavior:

- prefers Docker when available
- falls back to subprocess when enabled
- stores background tasks and work directories on the session

### `cleanup_code_execution(session)`

Cancel background tasks and remove the session work directory. Call this on session expiry or app shutdown if you manage runtimes yourself.

### `DockerBackend` and `SubprocessBackend`

Concrete execution backends.

- `DockerBackend` is treated as sandboxed
- `SubprocessBackend` is treated as unsandboxed

### `OutputRingBuffer` and `BackgroundTask`

Helpers used by background code execution.

### `make_code_execute_tool_def()` and `make_code_execute_status_tool_def()`

Factories for the public tool schemas exposed to the model.

### `strip_code_execute_base64_hook()`

Sanitization hook that replaces inline image base64 blobs with filename markers.

## Skills And Sub-Agents

### `SkillLoader`

Loads named markdown skill files from a directory.

### `SkillProfile`

Parsed skill metadata plus body prompt.

Common fields:

- `name`
- `system_prompt`
- `model`
- `max_turns`
- `timeout`
- `metadata`

### `SkillStateStore`

Simple JSON persistence for per-skill state.

### `parse_skill_file(path)`

Parse a standalone skill markdown file into a `SkillProfile`.

### `make_run_agent_handler(...)`

Factory for the local `run_agent` handler.

### `make_run_agent_tool_def(...)`

Factory for the tool schema exposed to the model.

## Providers

### `ModelProvider`

Provider interface used by runners.

It standardizes:

- client creation
- request parameter building
- message normalization
- stream event normalization
- cost estimation

### `AnthropicProvider`

Anthropic adapter used by `create_agent()`.

### `OpenAIProvider`

OpenAI-compatible adapter used when you build the app yourself.

### `ModelInfo`

Per-model metadata such as context window, thinking support, and token pricing.

### `CostEstimate`

Estimated request cost broken down by input, output, and cache token categories.

### `StreamEvent`

Normalized stream event produced by providers and consumed by runners.

### `ThinkingLevel`

Provider-agnostic reasoning intensity hint.

## Sessions And Events

### `Session`

Per-user mutable runtime state.

### `SessionStore`

TTL-based in-memory session registry.

### `AuthManager`

Issues and verifies JWT session tokens.

### `EventLog`

Append-only event buffer used by the runner and the SSE layer.

### `LogEntry`

Single event record with sequence number and timestamp.

## Memory

### `MemoryStore`

SQLite-backed persistent memory with optional embeddings and tags.

### `MarkdownSyncManager`

Synchronize memory entities to and from markdown files.

### `EmbeddingProvider`

Protocol for pluggable embedding backends.

## MCP

### `McpClientManager`

Starts stdio MCP servers, lists tool definitions, filters collisions, and routes tool calls to the right server.

Use `inline_servers` when you want self-contained examples or application-local MCP setup. Use `config_path` when you want to reuse a Claude desktop config.
