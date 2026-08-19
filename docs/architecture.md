# Architecture

**Last verified:** 2026-08-05 (docs-program unit pass against package source)

`ai-agent-gateway` is a server runtime for tool-using agents. It combines session management, SSE streaming, tool dispatch, approval handling, and provider abstraction in one package so you can ship an agent backend without rebuilding that infrastructure yourself.

## Request Lifecycle

```text
Client
  -> POST /api/chat/init
  -> session token
  -> POST /api/chat
      -> build server-owned AuthContext
      -> resolve + materialize session.driver once
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
3. The server constructs `AuthContext` from the authenticated session and
   server-owned credential handles, then resolves and materializes one
   immutable `session.driver` execution.
4. The server calls `build_chat_runtime(session, request, channel, auth_manager)`.
5. `ChatRuntime.build_runner()` constructs the runner with that exact bind,
   admitting registry, provider adapter, and credential snapshot.
6. The runner streams model output, dispatches tools, and appends events to `EventLog`.
7. The server forwards those events to the client as SSE until `stream_complete`
   or `error`. `stream_complete.terminal_disposition` distinguishes successful
   completion from an intentional interruption while preserving one canonical
   transport-closing event.

## Core Concepts

### `create_agent()`

`create_agent()` is the convenience entry point.

It wires together:

- a resolved `ModelProvider` (`AnthropicProvider`, `OpenAIProvider`, or your own instance)
- a server-owned model registry, selection policy, and credential provenance
- `GatewayServerConfig`
- `ChatRuntime`
- `AgentRunner`
- optional MCP client startup/shutdown
- optional code execution
- optional skills and `run_agent`

Use it when you want the shortest path to a working Anthropic- or OpenAI-backed agent server.

### `run_autonomous()`

`run_autonomous()` is the execution-only headless entry point for one-shot
autonomous/cron agent runs.

The application supplies an immutable `session.driver` capability bind, its
admitting registry, the exact provider adapter, and its bound credential
snapshot. The function verifies those inputs before session/MCP/client setup,
then wires together MCP, tools, skills, and `AgentRunner` without the HTTP
server. It returns `RunOutput` directly and handles its own MCP
startup/shutdown lifecycle. Provider/model/credential selection is deliberately
outside this boundary.

Use it for cron jobs, autonomous tasks, or as the building block for
`HeartbeatLoop`.

### Capability binding

`CapabilityExecutionResolver` is the single selection and credential
materialization boundary. It combines a versioned
`ProductModelSelectionPolicy`, server-owned `AuthContext`, versioned
`ProductModelRegistry`, credential materializer, and adapter resolver. A
successful resolution returns `BoundCapabilityExecution`: the immutable bind,
exact admitting registry, provider adapter, and credential snapshot consumed by
the runner.

Bindings are capability-specific. `session.driver` uses a trusted session
selection followed by its policy default; `plan.author` is policy-owned; and
`node.*` capabilities use their request/skill/role/parent-worker/policy
precedence. Credential principal and run mode are recorded in every
`CapabilityBind`. Retries reuse and reauthorize the same execution.

The production autonomous launcher uses the same boundary before subprocess
creation. It persists the complete secret-free bind in the task manifest,
then signs a short-lived launch envelope bound to task, control-run, and owner
identity. The child verifies and removes the envelope, materializes the exact
bind, and never re-selects a model or credential. Resume requires the persisted
bind; a scheduled run receives a fresh `cron` bind.

### Workflow output delivery

Workflow presentation is a projection over an exact published output, not a
second authored source of truth. The shared `agent_workflow_contracts` package
owns the version-discriminated delivery reader used by journal recovery,
gateway attachment materialization, the CLI, and generated TypeScript clients.

Historical V1 starts retain their absent-version `WorkflowDeliverySpecV1`
bytes and pair only with a `DeliveryEnvelopeV1` authored-summary envelope. V2
starts require an explicit `WorkflowDeliverySpecV2(schema_version="2.0")` and
pair only with `DeliveryEnvelopeV2`: one exact `PublishedOutputRef` plus a
bounded deterministic UTF-8 preview. The preview is non-authoritative; clients
keep the exact-output read recipe and verify the returned identity, byte count,
and digest.

Every reader in the assembled artifact accepts V1 and V2 before the single
writer is activated. The service admits only explicit V2 starts, derives the
preview from owner-authorized exact bytes, and appends one V2 envelope;
already-recorded V1 or V2 envelopes replay through the shared reader without
regeneration. An unsettled historical V1 specification resolves to a typed
failed delivery under the current protocol; it never revives the V1 writer.
Historical compatibility is product behavior. Writer exclusivity
is external release evidence: stop the prior gateway, prove the managed process
is absent, inspect durable journals for unsettled runs, and only then start the
coherent reader-and-writer artifact. Product health and workflow APIs do not
carry deployment switches, drain counters, or rollout state.

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
- the exact `session.driver` `BoundCapabilityExecution`
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
- you want server discovery from inline config, an explicit config path, or `MCP_CONFIG_PATH` (with none configured, no file-backed MCP servers are loaded)

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

### Persistent grants and per-call authorization

A persistent grant is eligibility to mint authorization for a new invocation; it is not itself executable authority. The default policy looks up an active grant by exact `user_id`, tool name, and derived scope hint. A finite `expires_at` and revocation close that lookup. When a grant matches, the gateway still creates a new durable approval request for the invocation and records `authorization_mode=PERSISTENT_GRANT` plus the source grant ID in `grant_reference`.

The per-call row binds the current actor, tool, redacted argument projection and argument hash, policy identity, and request context. For exact-planned tools, the row also binds the current plan/change identity; the trusted carrier separately proves that identity is linked to the prepared payload. The dispatcher revalidates the row against the carrier immediately before execution. For FMS ChangeSet writes, the effect interpreter rechecks the live base before the first effect. A grant therefore never authorizes reuse of a prior plan or bypasses the current validation owned by an exact executor.

Current grant matching has two deliberate facts and one known limitation:

- The default scope hint includes the tool class, tool name, and the first available `ticker`, `symbol`, `portfolio_id`, or `account_id`. The reminted approval and execution authorization carry the current tenant context, but the persistent-grant lookup key itself is not tenant-keyed.
- `args_predicate` is persisted on a grant, but `SingleUserApprovalPolicy` does not currently evaluate it during persistent-grant lookup. It is stored metadata, not an active authorization boundary in the default policy. Tightening this would change which existing grants are accepted and requires a separately reviewed compatibility and policy decision.
- Revocation is linearized by the active-grant lookup. A revocation committed before that query prevents the match. Once lookup has returned a grant, a later revocation prevents future matches but does not retract the per-call decision already in progress.

Grant lifetime and invocation lifetime are separate:

| Record | Current time semantics |
| --- | --- |
| Persistent grant | `expires_at=NULL` means indefinite until revoked. Ordinary grant creation currently writes `NULL`; a finite timestamp, when supplied, is enforced at lookup. |
| Default human prompt | The policy requests 600 seconds. The lifecycle may reduce the effective window, notably to stay within a trade preview's lifetime, and stores the resulting expiry on the pending row. |
| Prepared BusinessModel approval | A missing per-call expiry is replaced with a 15-minute expiry before the prepared row is stored. |
| Generic grant-reminted approval | May retain `expires_at=NULL`. That means no approval-row time deadline; argument/identity, exact-plan, proposal TTL, and live-base checks still apply where those contracts exist. |

Adding a default TTL to grants or all reminted approvals would be a product-policy, migration, and client-UI change. It is not implied by the current schema and must not be introduced as a documentation-only hardening change. `grant_reference` provides durable approval/execution lineage to the source grant; not every projected audit-event JSON currently includes that field.

## Sessions And Auth

Sessions are first-class runtime state.

`GatewaySession` (module `agent_gateway.session`) currently tracks:

- `pending_tools`
- `approved_tool_types`
- `loaded_mcp_servers`
- `approval_queues`
- code execution work directory
- background code execution tasks
- whether a stream is already active (`stream_active` / `active_turn`)
- session kind (`chat` vs `control`)

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
- they use one immutable capability bind; for a named operation, the registered
  skill profile's finite budget is frozen into the immutable execution snapshot
  and enforced again on durable resume
- unnamed delegation keeps the legacy observation-only cost accumulator, and
  `run_agent` has no model-authored hard-budget input
- they cannot recursively spawn more sub-agents because `run_agent` is excluded by default

### Background Sub-Agents

The agent runs sub-agents in the background by default. `run_agent` returns immediately with a `task_id`; pass `background=false` only when the current turn must wait for the validated `ChildReturn`. With automatic notifications enabled, current-run results arrive through typed notifications and must not be polled. `get_background_result` is reserved for historical tasks and current-run payloads explicitly omitted from a notification; notification-disabled channels may make one exact bounded wait.

Key details:

- background tasks are stored on the `AgentRunner`, not the session
- a universal concurrency semaphore limits parallel sub-agents (default 4)
- the runner injects a system prompt reminder listing active background tasks after compaction pauses, so the model stays aware of pending work
- on runner shutdown, pending background tasks are awaited (up to 30 seconds) or cancelled
- `on_before_background` and `on_background_complete` callbacks let consumers hook into the lifecycle

### Typed Event Contract (0.15.0+)

Skill-framework sub-agent runs emit typed lifecycle and result events onto the parent `EventLog`, where they flow through the standard SSE channel alongside `text_delta` / `tool_call_complete` / etc. The contract is opt-in — events only fire when both `skill_run_id` and `SkillProfile` are wired into the sub-agent dispatch.

The six events split into two scopes:

- **Run-scoped** (carry `skill_run_id` for correlation): `SkillRunStartedEvent`, `skill_result_captured`, `ArtifactReadyEvent`, `AggregateReadyEvent`, `ArtifactFailedEvent`.
- **Renderer-only** (no skill run): `ArtifactUnavailableEvent`, surfaced when a UI lookup for `(ticker, skill)` finds nothing.

Why typed: renderers (Excel taskpane, web demo surface, etc.) need a stable shape to react to skill state transitions without parsing free-form `tool_call_complete` blocks. The dataclasses are frozen where event classes exist, the `type` discriminator is fixed, and `event_to_dict` / `event_from_dict` round-trip the dataclass wire format. Renderers can drive UI state machines off `skill_run_started` → `skill_result_captured` → `artifact_ready` without doing tool-call introspection.

Verdict display data comes from `skill_result_captured.verdict_echo` or structured FMS artifact events. Runtime code does not parse final markdown, fenced YAML, or `memory_write` payloads to infer verdict state.

See `agent_gateway.events` for the dataclasses, `docs/api-reference.md` for the public surface, and `docs/http-api.md` → "Skill Framework Events" for wire format.

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
