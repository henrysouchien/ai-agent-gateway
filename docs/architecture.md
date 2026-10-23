# Architecture

**Last verified:** 2026-08-21 (docs-program unit pass against package source)

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
4. The server calls `build_chat_runtime(session, request, channel, auth_manager, storage_root=...)` — four positional arguments and a keyword-only `storage_root` (`None` unless the app has a storage root).
5. `ChatRuntime.build_runner()` constructs the runner with that exact bind,
   admitting registry, provider adapter, and credential snapshot.
6. The runner streams model output, dispatches tools, and appends events to `EventLog`.
7. The server forwards those events to the client as SSE until `stream_complete`
   or `error`. `stream_complete.terminal_disposition` distinguishes successful
   completion from an intentional interruption while preserving one canonical
   transport-closing event.
   Terminal success is published only after the lifecycle owner's off-loop
   durable closure (including named-skill result, interruption, and detach).
   Repeated cancellation drains that closure before propagating; learning
   receipts are acknowledged only for a successfully persisted success.
   Persistence failures append error history without resetting staged evidence.

For native sessions using `AgentSessionLog`, the session lifecycle writer
settles unanswered assistant tool calls as durable interrupted results before
interruption/detach; attach recovery also includes calls not yet dispatched.
An interrupted result reports missing completion, not that side effects were
rolled back. `SessionContextBuilder` replays each batch as one user message
with all tool results before supplemental text, matching the live tool loop.
Replay is parent-only: entries whose envelope `role` is `sub_agent` are
excluded. There is no tail budget or time window. After the tail is flattened,
`truncate_to_last_compaction` starts at the last assistant message holding a
`{type: "compaction"}` block.

A session's durable log outlives its TTL and the process. Every attach records
the log's owner (`user_id`, `channel`, `api_key_hash` of the writing session).
`POST /api/chat/init` with `resume_session_id` mints a new session (new id,
token and TTL, fresh credential bind) and asks the application's
`session_resume_resolver` for the caller's own log for that id: it checks only
the caller's filename and requires the log's first attach to name the resolved
caller. A caller's idle session still holding that log is expired as the new
one binds. A refusal is HTTP 409 `{"error": "session_resume_refused",
"reason": ...}` — `not_found` (including a log owned by someone else),
`ambiguous`, `no_ownership_record`, or `in_use` (the holding session is mid-turn)
— and registers no session and touches no file. On success the response echoes
`resumed_from_session_id`, the session carries `resume_log_path`, and the
next turn appends to and replays that log. A log keeps its first session's id,
so every later resume presents that id.

## Core Concepts

### `create_agent()`

`create_agent()` is the convenience entry point.

It wires together:

- a resolved `ModelProvider` (`AnthropicProvider`, `CodexProvider`, `OpenAIProvider`, `XAIProvider`, or your own instance)
- a server-owned model registry, selection policy, and credential provenance
- `GatewayServerConfig`
- `ChatRuntime`
- `AgentRunner`
- optional MCP client startup/shutdown
- optional code execution
- optional skills and `run_agent`

Use it when you want the shortest path to a working Anthropic-, Codex-,
OpenAI-, or XAI-backed agent server.

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
- request-scoped hooks and settings such as `on_usage` and `compaction_trigger`

### `AgentRunner`

`AgentRunner` is the main model loop. It:

- streams model events
- collects tool calls
- dispatches tools
- appends SSE-facing events
- tracks usage and estimated cost
- retries transient stream failures
- runs the one soft-budget ladder (`runner_soft_budget`) on a delegated run:
  steer at the `soft_turn_threshold(kind)` threshold (a key of the operator settings file, read through
  `ContainmentSettings`), forced final at `soft_budget_final_factor` (1.5×), and an abort only when the child
  still has not answered after `soft_budget_grace_turns`; a primary run has no ladder unless
  `writer_soft_turn_budget` is set, and cost is a measurement, never a stop
- takes every containment bound from its `containment` argument
  (`ContainmentSettings`, `agent_gateway/containment_settings.py`, a live view over `agent_gateway/run_settings.py`); child
  runners inherit the parent's object

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

`McpClientManager` connects stdio and streamable-HTTP MCP servers, lists their
tool definitions, and routes tool calls back to the correct server.
`mcp_client_connections` owns transport setup, OAuth configuration, and catalog
enumeration; `mcp_client_config` owns reconnect eligibility.

MCP 2, FastMCP 4, and `httpx2` are required dependencies; OAuth is an optional
per-server authentication choice, not an optional SDK installation. The client
uses `ClientSession.initialize()` for its handshake, snake_case SDK attributes
(without changing wire JSON names), float-second timeouts, and
`PaginatedRequestParams` for catalog pagination and the stdio stability probe.
Streamable HTTP yields read and write streams; its transport and OAuth clients
use `httpx2`, while ordinary tool HTTP remains on `httpx`.

`mcp_client_connections` opens every connection in a transport host task of its
own. This covers shared and per-user connections, over stdio or streamable HTTP.
The host task enters the SDK's transport and `ClientSession` contexts, hands the
connected state to the caller, and later exits those contexts itself. The
connection's `exit_contexts` holds only the host, so closing it is a request the
host carries out. Entering the contexts in a caller's task would arm the
connection with a cancellation aimed at that caller. Each of those contexts
cancels its anyio scope on exit, and anyio delivers that cancellation to the
task that entered the scope. An EOF reconnect, a replacement or a shutdown closes
the connection from some other task, so the close would cancel whatever
pipeline stage or turn opened that generation. Exiting a scope outside the task
that entered it also leaves the scope entered for good.

A server marked `per_user` in its config entry serves definitions only; calls
route to a child keyed by `(server, user)` that `McpClientManager` spawns on
first dispatch. Each such child also gets a per-user host task that owns its
lifecycle: it hands the connected state to the caller, keeps the child until it
is retired, and then closes it.

A child is retired by the periodic idle reaper, by a replacement (forced
refresh, near broker expiry, changed credential binding, dead transport), by
instance-cap eviction, or by a transport failure on a dispatch. Retirement
removes the child from the per-user map first, so no further caller can obtain
it, and then waits for its in-flight calls to finish before the host closes it —
with no deadline: a tool call ends at its result or its caller's abort. Manager
shutdown does not drain: it drops the pending drains and closes every live host
at once. Each close logs the server, the user and the site that closed it. A
close is only ever a request, never a cancellation of the host and never a
deadline over it: cancelling a host mid-teardown interrupts the cleanup that
reaps the child, and one cancelled before its coroutine runs leaves its waiters
unsettled. A startup still in flight runs to its own end and the close is
waiting for it there. The connection layer owns every startup timeout and
applies them per request, so the catalog walk carries its own bound instead: a
server answering every `list_tools` page with another cursor is enumerated for
at most `_LIST_TOOLS_PAGE_CEILING` pages (256) in
`mcp_client_connections.initialize_session_state`, and the walk then raises. On
manager startup that raise becomes a non-retryable `startup_error` diagnostic
against that one server, and the other servers' catalogs publish without it; for
a per-user child it settles the waiting caller's `ready` with the failure rather
than leaving a host that never arrives. That loop
is where the bound belongs — it hangs a shared server's startup the same way as
a per-user child's — and a deadline over the per-user host is not, because such
a deadline surrounds the startup's own teardown and cancels the SDK
mid-termination. The ceiling is a mechanical I/O bound on a client loop, never a
wall-clock deadline on the work: the per-request `wait_for` is unchanged. An
abandoned child holds its instance-cap slot until its host has let it go, and a
child whose close has been asked for is never handed to a caller.

Reconnect eligibility uses typed `MCPError` codes. Request timeout `-32001`
permits startup retry or reconnection for future calls, never in-flight replay.
Connection-closed errors can replay only when the call's existing replay policy
allows it; registered mutating calls and uncertain Sheets mutations are not
replayed.

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
5. The dispatcher resumes on approval or returns a `user_denied` error on denial; the wait has no timer

On the durable approval route, a call whose job context lets the lifecycle record its private executable binding does not wait in the turn: after the ask is published it raises `ApprovalPark`, the turn closes `awaiting_approval` with the request still `pending_user`, and the vote on `/api/chat/tool-approval` reaches the original call through the root or child Continue hook, which executes it once and continues the run. A call with no job or tenant context has no binding to continue from and waits on the live queue as above.

There is no session approval cache: each approval decides one call. `allow_tool_type=true` requests a persistent grant, which only an approval policy that permits grants mints.

Important nuance:

- `create_agent()` only installs built-in approval behavior for `code_execute`, and only when the resolved backend is unsandboxed subprocess execution.
- If you want approval for arbitrary tools, use `create_gateway_app()` and construct `ToolDispatcher` with your own `needs_approval` callback. Example: [`../examples/06-tool-approval/`](../examples/06-tool-approval/)

### Persistent grants and per-call authorization

A persistent grant is eligibility to mint authorization for a new invocation; it is not itself executable authority. The default `SingleUserApprovalPolicy` neither looks up nor mints grants: it asks a human only for the `irreversible` class (a live order), decides every such call fresh with `allow_persistent_grant=False`, and auto-approves every other class. The store keeps grant rows and their revocation path (`revoke_persistent_grant`) for existing rows and for a custom policy (`GATEWAY_APPROVAL_POLICY_CLASS`). A custom policy that matches a grant returns its ID as `grant_reference`; the gateway still creates a new durable approval request for the invocation and records `authorization_mode=PERSISTENT_GRANT` plus that ID in `grant_reference`.

The per-call row binds the current actor, tool, redacted argument projection and argument hash, policy identity, and request context. For exact-planned tools, the row also binds the current plan/change identity; the trusted carrier separately proves that identity is linked to the prepared payload. The dispatcher revalidates the row against the carrier immediately before execution. For FMS ChangeSet writes, the effect interpreter rechecks the live base before the first effect. A grant therefore never authorizes reuse of a prior plan or bypasses the current validation owned by an exact executor.

A custom policy that reuses grants owns its matching; the store gives it these facts:

- The scope hint is whatever the policy returns as `persistent_grant_scope_hint`; the reminted approval and execution authorization carry the current tenant context, but the grant lookup key (`user_id`, tool name, scope hint) is not tenant-keyed.
- `args_predicate` is persisted on a grant but evaluated only by a policy that chooses to; it is stored metadata, not an authorization boundary the store enforces.
- Revocation is linearized by the active-grant lookup (`find_persistent_grant`). A revocation committed before that query prevents the match. Once lookup has returned a grant, a later revocation prevents future matches but does not retract the per-call decision already in progress.

Grant lifetime and invocation lifetime are separate:

| Record | Current time semantics |
| --- | --- |
| Persistent grant | `expires_at=NULL` means indefinite until revoked. Grant creation on approval writes `NULL`; a finite timestamp, when supplied, is enforced at lookup. |
| Default human prompt | The policy requests no expiry for a live order. A trade's request records its preview's remaining window, less dispatch margin, as `expires_at`; the timestamp is informational, and no clock expires the request or refuses its vote. |
| Prepared BusinessModel approval | The prepared row carries the approval's expiry, `NULL` when the policy set none; elapsed time does not lapse it. |
| Generic grant-reminted approval | May retain `expires_at=NULL`. That means no approval-row time deadline; argument/identity, exact-plan, proposal TTL, and live-base checks still apply where those contracts exist. |

Adding a default TTL to grants or all reminted approvals would be a product-policy, migration, and client-UI change. It is not implied by the current schema and must not be introduced as a documentation-only hardening change. `grant_reference` provides durable approval/execution lineage to the source grant; not every projected audit-event JSON currently includes that field.

## Sessions And Auth

Sessions are first-class runtime state.

`GatewaySession` (module `agent_gateway.session`) currently tracks:

- `pending_tools`
- `loaded_mcp_servers`
- `approval_queues`
- code execution work directory
- background code execution tasks
- whether a stream is already active (`stream_active` / `active_turn`)
- session kind (`chat` vs `control`)

`AuthManager` issues and verifies JWT session tokens. `SessionStore` owns the session records and explicit expiry hooks. A session token authenticates until the session is explicitly expired (`expire_session*`, session delete, or run teardown); there is no wall-clock TTL — a run's session never expires mid-workflow, and a session whose turn has ended can start another turn. `visible_sessions_snapshot` filters only explicitly expired sessions.

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

An interceptor that throws is logged and the tool proceeds; an interceptor that holds an invariant returns its own `deny`.

## Code Execution

`build_code_execution(session, config, *, job_runner=None)` injects `code_execute`.

It also returns:

- local handlers
- tool definitions
- approval qualifier logic
- the prepared-call hook and work-directory helper
- a result-sanitization hook

Backend selection:

1. Prefer Docker if it is registered and available
2. Fall back to subprocess if registered and available
3. Error if no backend is available

Operational details:

- the working directory persists within a session
- stdout and stderr can stream back as `tool_output_chunk`
- a `background=true` cell is a `code` job on the task registry of the runner `job_runner()` answers; it is inspected, waited on and killed through that registry, and refused (`code_execution_unavailable`) when there is no runner
- `cleanup_code_execution(session)` kills the live turn's running `code` jobs, then releases interpreters and removes the work directory

## Skills And Sub-Agents

Skills are markdown files loaded by `SkillLoader`.

Each skill can specify:

- a system prompt
- max turns
- timeout
- metadata

The capability resolver, not skill frontmatter, owns provider and model
selection.

When `skills_dir` is configured:

- `create_agent()` installs the `run_agent` tool automatically
- `make_run_agent_handler()` spawns sub-agents through the parent runner
- named skills are resolved by file name

Sub-agents are intentionally constrained:

- they get their own runner and event log
- they use one immutable capability bind; a named operation declares no turn or
  cost limit, and every child meets the one soft turn ladder
- cost is recorded on the child's `CostAccumulator` as a measurement, and
  `run_agent` has no model-authored budget input
- they cannot recursively spawn more sub-agents because `run_agent` is excluded by default

### Background Sub-Agents

The agent runs sub-agents in the background by default. `run_agent` returns immediately with a `task_id`; pass `background=false` only when the current turn must wait for the validated `ChildReturn`. With automatic notifications enabled, current-run results arrive through typed notifications and must not be polled. `get_background_result` is reserved for historical tasks and current-run payloads explicitly omitted from a notification; notification-disabled channels may make one exact bounded wait.

Key details:

- background tasks are stored on the `AgentRunner`, not the session
- a universal concurrency semaphore limits parallel sub-agents (default 4)
- the runner injects a system prompt reminder listing active background tasks after compaction pauses, so the model stays aware of pending work
- on runner shutdown, pending background tasks are awaited (up to 30 seconds) or cancelled
- `on_before_background` and `on_background_complete` callbacks let consumers hook into the lifecycle

### Typed Event Contract

Typed lifecycle and result events flow through the parent `EventLog` alongside
ordinary runner events. The 14-event union covers skill runs, artifacts, UI
blocks, direct-parent agent completion, workflow output, recommendations,
approvals, and session recaps. Seven event types are run-scoped; the remainder
have their own correlation keys. The canonical union and scope sets live in
`agent_gateway.events`.

Why typed: renderers (Excel taskpane, web demo surface, etc.) need a stable shape to react to skill state transitions without parsing free-form `tool_call_complete` blocks. The dataclasses are frozen where event classes exist, the `type` discriminator is fixed, and `event_to_dict` / `event_from_dict` round-trip the dataclass wire format. Renderers can drive UI state machines off `skill_run_started` → `skill_result_captured` → `artifact_ready` without doing tool-call introspection.

Verdict display data comes from `skill_result_captured.verdict_echo` or structured FMS artifact events. Runtime code does not parse final markdown, fenced YAML, or `memory_write` payloads to infer verdict state.

See `agent_gateway.events` for the dataclasses, [API reference](./api-reference.md)
for the public surface, and [HTTP API](./http-api.md) for the SSE envelope and
wire format.

## Providers

The provider abstraction lives under `ModelProvider`.

Built-in providers:

- `AnthropicProvider`
- `CodexProvider`
- `OpenAIProvider`
- `XAIProvider`

All four normalize messages, build request params, stream events, and estimate costs through the same contract.

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
- Anthropic, Codex, OpenAI, or XAI
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
