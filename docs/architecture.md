# Architecture

`ai-agent-gateway` is a server runtime for tool-using agents. It combines session management, SSE streaming, tool dispatch, approval handling, and provider abstraction in one package so you can ship an agent backend without rebuilding that infrastructure yourself.

## Request Lifecycle

```text
Client
  -> POST /api/chat/init
  -> session token
  -> POST /api/chat
      -> build_chat_runtime(...)
      -> ChatRuntime
      -> AgentRunner or AgentSDKRunner
      -> ToolDispatcher
          -> local handler
          -> MCP server
          -> code execution
          -> run_agent
      -> EventLog
  <- SSE events
```

A single request usually moves through these stages:

1. The client exchanges an API key for a JWT session token.
2. The client sends a chat request plus optional `context`, including `context.channel`.
3. The server calls `build_chat_runtime(session, request, channel, auth_manager)`.
4. `ChatRuntime.build_runner()` constructs the runner for this request.
5. The runner streams model output, dispatches tools, and appends events to `EventLog`.
6. The server forwards those events to the client as SSE until `stream_complete` or `error`.

## Core Concepts

### `create_agent()`

`create_agent()` is the convenience entry point.

It wires together:

- a resolved `ModelProvider` (`AnthropicProvider`, `OpenAIProvider`, or your own instance)
- `GatewayServerConfig`
- `ChatRuntime`
- `AgentRunner`
- optional MCP client startup/shutdown
- optional code execution
- optional skills and `run_agent`

Use it when you want the shortest path to a working Anthropic- or OpenAI-backed agent server.

### `run_autonomous()`

`run_autonomous()` is the headless entry point for one-shot agent runs.

It wires together the same components as `create_agent()` — provider, MCP, tools, skills — but without the HTTP server. Returns `RunOutput` directly. Handles its own MCP startup/shutdown lifecycle.

Use it for cron jobs, batch tasks, or as the building block for `HeartbeatLoop`.

### `HeartbeatLoop`

`HeartbeatLoop` wraps a `run_fn` (typically a `functools.partial` of `run_autonomous()`) and calls it at regular intervals.

Key behaviors:

- **Quiet suppression**: if the agent replies with `HEARTBEAT_OK`, the response is suppressed (no delivery)
- **Active hours**: skip ticks outside a configured time window
- **Backoff**: on errors (`RunOutput.error`, `timed_out`, or exceptions), retry with exponential backoff that replaces the normal interval; reset on success
- **Checklist skip**: if `HEARTBEAT.md` exists but is empty (only headers/blank lines), skip the tick entirely to save API cost
- **Callback safety**: all callbacks wrapped in try/except — failures are logged but never kill the loop

```text
HeartbeatLoop
    |
    |-- check active hours → skip if outside window
    |-- check checklist → skip if empty
    |-- call run_fn() → RunOutput
    |-- strip HEARTBEAT_OK → classify as alert or quiet
    |-- fire on_alert / on_quiet / on_error callback
    |-- persist heartbeat_state.json
    |-- sleep (interval or backoff)
    |-- repeat
```

### `create_gateway_app()`

`create_gateway_app()` is the low-level server factory.

Use it when you need:

- custom `needs_approval` rules
- interceptors
- channel-aware runtime selection
- multiple runtime profiles
- advanced auth, CORS, hook, or budget behavior
- direct control over `AgentRunner` or `AgentSDKRunner`

### `ChatRuntime`

`ChatRuntime` is the per-request wiring bundle. It carries:

- the system prompt
- a `build_runner` callback
- a tool-definition callback
- provider metadata
- model override
- request-scoped limits such as `max_turns`

### `AgentRunner`

`AgentRunner` is the main model loop. It:

- streams model events
- collects tool calls
- dispatches tools
- appends SSE-facing events
- tracks usage and estimated cost
- retries transient stream failures
- stops on budgets or max-turn limits

## Tool Dispatch

`ToolDispatcher` is the execution boundary between model output and real tool code.

Dispatch order:

1. Run interceptors
2. Decide whether approval is required
3. Execute a local Python handler if present
4. Otherwise execute an MCP tool if it exists
5. Otherwise return `unknown_tool`

The dispatcher does not care whether the tool came from:

- `tool_handlers`
- `build_code_execution()`
- `make_run_agent_handler()`
- `McpClientManager`

They all converge at the same dispatch layer.

## MCP Tools Vs Local Tools

### MCP Tools

`McpClientManager` starts stdio MCP servers, lists their tool definitions, and routes tool calls back to the correct server.

Good fit when:

- you already have an MCP ecosystem
- the tool runtime is external
- you want server discovery from inline config or `~/.claude.json`

### Local Tools

Local tools are plain async Python callables registered in `tool_handlers`.

Good fit when:

- the tool logic lives in your app already
- you want low-overhead custom behavior
- you do not need a separate MCP server process

## Approval Flow

Approval is built into `ToolDispatcher`.

Flow:

1. `needs_approval(tool_name, tool_input, qualifier)` returns `True`
2. The dispatcher calls `request_approval(...)`
3. The server appends a `tool_approval_request` event
4. The client calls `/api/chat/tool-approval`
5. The dispatcher resumes or returns a `user_denied` / `approval_timeout` error

`allow_tool_type=true` stores a session-scoped approval key so later calls of the same tool type can skip the prompt.

Important nuance:

- `create_agent()` only installs built-in approval behavior for `code_execute`, and only when the resolved backend is unsandboxed subprocess execution.
- If you want approval for arbitrary tools, use `create_gateway_app()` and construct `ToolDispatcher` with your own `needs_approval` callback. Example: [`../examples/06-tool-approval/`](../examples/06-tool-approval/)

## Sessions And Auth

Sessions are first-class runtime state.

`Session` currently tracks:

- `pending_tools`
- `approved_tool_types`
- `loaded_mcp_servers`
- `approval_queues`
- code execution work directory
- background code execution tasks
- whether a stream is already active

`AuthManager` issues and verifies JWT session tokens. `SessionStore` owns TTL, cleanup, and expiry hooks.

This matters because approvals, code execution state, and loaded MCP servers are all session-scoped, not global.

## Channels

The server itself is channel-agnostic. Channel information comes from `ChatRequest.context["channel"]` and is passed into `build_chat_runtime(...)`.

That lets you use one backend for multiple clients:

- web
- CLI
- Telegram
- internal operators

Typical channel-specific decisions:

- prompt shaping
- tool filtering
- approval policy
- deferred MCP server loading
- code execution enablement

## Interceptors

Interceptors are runtime tool policies that run before dispatch.

A `ToolInterceptor` returns an `InterceptDecision` with:

- `allow`
- `warn`
- `deny`

Warnings are attached to successful tool results. Denials stop the tool before it runs and emit `interceptor_decision` events.

If an interceptor has `__intercept_critical__ = True` and throws, the dispatcher fails closed and blocks the tool.

## Code Execution

`build_code_execution(session, config)` injects two tools:

- `code_execute`
- `code_execute_status`

It also returns:

- local handlers
- tool definitions
- approval qualifier logic
- approval predicate
- a result-sanitization hook

Backend selection:

1. Prefer Docker if it is registered and available
2. Fall back to subprocess if registered and available
3. Error if no backend is available

Operational details:

- the working directory persists within a session
- stdout and stderr can stream back as `tool_output_chunk`
- background tasks are stored on the session and polled with `code_execute_status`
- `cleanup_code_execution(session)` cancels tasks and removes the work directory

## Skills And Sub-Agents

Skills are markdown files loaded by `SkillLoader`.

Each skill can specify:

- a system prompt
- a model override
- max turns
- timeout
- metadata

When `skills_dir` is configured:

- `create_agent()` installs the `run_agent` tool automatically
- `make_run_agent_handler()` spawns sub-agents through the parent runner
- named skills are resolved by file name

Sub-agents are intentionally constrained:

- they get their own runner and event log
- they inherit provider and budget context
- they cannot recursively spawn more sub-agents because `run_agent` is excluded by default

### Background Sub-Agents

The agent can run sub-agents in the background for parallel research by passing `background=true` to `run_agent`. This returns immediately with a `task_id` so the parent agent can continue working, then collect results later with `get_background_result`.

Key details:

- background tasks are stored on the `AgentRunner`, not the session
- a concurrency semaphore limits parallel sub-agents (default 3)
- the runner injects a system prompt reminder listing active background tasks after compaction pauses, so the model stays aware of pending work
- on runner shutdown, pending background tasks are awaited (up to 30 seconds) or cancelled
- `on_before_background` and `on_background_complete` callbacks let consumers hook into the lifecycle

## Providers

The provider abstraction lives under `ModelProvider`.

Built-in providers:

- `AnthropicProvider`
- `OpenAIProvider`

Both normalize messages, build request params, stream events, and estimate costs through the same contract.

`create_agent()` can resolve the built-in provider strings or accept a `ModelProvider` instance directly. `create_gateway_app()` remains the escape hatch when you need custom runtime assembly around that provider.

There is also an `AgentSDKRunner` path for SDK-based execution when you want the Anthropic agent SDK instead of the native runner.

## Event Streaming

`EventLog` is the handoff between runtime execution and HTTP streaming.

Why it exists:

- runners append structured events without knowing about HTTP
- the server can add heartbeat events
- the server can enrich tool events with `execution_location`
- the same event stream can be consumed by SSE clients or internal observers

## Upgrade Path

Start with `create_agent()` when you want:

- one prompt
- Anthropic or OpenAI
- optional MCP tools
- optional local tools
- optional code execution
- optional skills

Move to `create_gateway_app()` when you need:

- custom approval for non-code tools
- channel-aware runtimes
- interceptors
- custom budgets and hooks
- multiple runtime profiles in one backend
- full control over provider lifecycle and runtime assembly

Practical sequence:

1. Start with [`../examples/01-minimal/`](../examples/01-minimal/)
2. Add tools with [`../examples/02-mcp-tools/`](../examples/02-mcp-tools/) or [`../examples/03-local-tools/`](../examples/03-local-tools/)
3. Add execution and skills with [`../examples/04-code-execution/`](../examples/04-code-execution/) and [`../examples/05-skills/`](../examples/05-skills/)
4. Graduate to full runtime control with [`../examples/06-tool-approval/`](../examples/06-tool-approval/), [`../examples/07-full-production/`](../examples/07-full-production/), and [`../examples/08-multi-provider/`](../examples/08-multi-provider/)
