# API Reference

This is a hand-written guide to the main public APIs in `agent_gateway`.

For endpoint payloads and SSE event schemas, see [HTTP API](./http-api.md). For architecture and control flow, see [Architecture](./architecture.md).

## Getting Started

### `create_agent()`

Fastest way to create a gateway server from a system prompt.

Use it when you want:

- Anthropic- or OpenAI-backed chat
- automatic session and SSE endpoints
- optional MCP tools
- optional local tools
- optional code execution
- optional skills

Key parameters:

- `model_key`: stable product-registry key for the `session.driver` default
- `effort`: optional effort supported by that exact registry entry
- `provider`: optional credential-family/adapter consistency assertion; it
  cannot override the stable key's execution identity
- `provider_config`: credential/endpoint configuration only; model-selection
  fields are rejected
- `skills_dir`: directory of markdown skill files used by the built-in `run_agent` tool
- `skill_state_file`: optional JSON file for callable skills with `persist_state: true`; prior state is injected into the skill prompt, and final `## STATE_UPDATE_JSON` updates are merged back into the file
- `outputs_dir`: directory for named-skill output files; stale same-day outputs are cleaned before background `run_agent` launch

```python
from agent_gateway import create_agent

app = create_agent("You are a concise assistant.")
```

### `run_autonomous()`

Execute one already-resolved headless `session.driver` capability without the
HTTP server and return a `RunOutput`. Capability/provider/model/effort,
credential, and session identity selection happen before this execution
boundary.

Use it when you want:

- one autonomous task or cron job
- optional MCP tools and local tools
- optional skills, state persistence, and completion delivery

Key parameters:

- `capability_execution`: immutable `BoundCapabilityExecution` for
  `session.driver`, with an `autonomous` or `cron` run mode; it carries the
  exact admitting registry, adapter, credential snapshot, and complete bind
- `session`: exact `GatewaySession` that owns the authenticated runtime
  identity and mutable lifecycle state
- `user_id`: required stable user identity for usage accounting
- `billing_mode`: required usage billing mode, either `byok` or `metered`
- `rate_table_version`: required usage rate-table/version label
- `skills_dir`: directory of markdown skill files used by `run_agent`
- `outputs_dir`: directory for named-skill output files; stale same-day outputs are cleaned before background `run_agent` launch
- `state_dir`: optional directory for persisted run state (JSON load/save between runs)
- `delivery`: optional `DeliveryConfig` for webhook, Telegram, briefing file, or callback output
- `max_concurrent_sub_agents`: optional concurrency limit for sub-agents
- `compaction_instructions`: optional instructions for context compaction

Raw `model`, `api_key`, `auth_token`, `auth_config`, `provider_config`, and
`max_tokens` selectors are not accepted. Split execution inputs are also
rejected. Bind/config/adapter/registry disagreements and unavailable
credentials fail before session scoping, MCP construction/startup, or
provider-client creation. The executor snapshots the bound mapping before its
first await and passes `allow_stub_response=False`.

Returns `RunOutput` with `response`, `tools_used`, `usage`, `error`, `timed_out`, `budget_exceeded`, `max_turns_reached`.

### `run_autonomous_sync()`

Synchronous wrapper around `run_autonomous()` for cron scripts and autonomous
jobs. It requires the same prebound execution inputs and returns the same type.

### `RunOutput`

Result of an autonomous run. Fields:

- `response: str` — final model response text
- `tools_used: list[str]` — tool names called during the run
- `usage: dict` — token usage from the provider
- `error: str | None` — error message if the run failed
- `timed_out: bool` — True if the run hit the wall-clock timeout
- `budget_exceeded: bool` — True if estimated cost exceeded `max_budget_usd`
- `max_turns_reached: bool` — True if the model loop hit `max_turns`

### `DeliveryConfig`

Where results go after a run completes.

- `on_complete`: async/sync callback receiving `(RunOutput, state_dict)`
- `telegram_bot_token`, `telegram_chat_id`, `telegram_label`: built-in Telegram delivery
- `briefing_file`: path to an artifact file to send as follow-up content
- `webhook_url`: HTTP POST endpoint for the run summary
- `format_message`: optional custom formatter `(RunOutput, state) -> str`

### `run_output_exit_code()` and `run_output_outcome()`

Map a `RunOutput` to process exit codes (0/1/2/3/124) or semantic outcome strings (`"success"`, `"timeout"`, `"error"`, `"budget_exceeded"`, `"max_turns"`). Exit codes are for process semantics; outcomes are for persisted state tracking.

### `HeartbeatLoop`

Persistent agent loop that calls a run function at regular intervals with quiet suppression.

Use it when you want:

- periodic "anything need attention?" check-ins
- automatic suppression of "nothing to report" responses via `HEARTBEAT_OK` token
- active hours window (e.g., only run during market hours)
- exponential backoff on errors with automatic recovery

```python
from functools import partial
from agent_gateway import (
    BoundCapabilityExecution,
    GatewaySession,
    HeartbeatConfig,
    HeartbeatLoop,
    run_autonomous,
)

capability_execution: BoundCapabilityExecution
session: GatewaySession
capability_execution, session = prepare_autonomous_execution()

loop = HeartbeatLoop(
    run_fn=partial(
        run_autonomous,
        system_prompt="...",
        initial_message="...",
        capability_execution=capability_execution,
        session=session,
        user_id="heartbeat-agent",
        billing_mode="byok",
        rate_table_version="current",
    ),
    config=HeartbeatConfig(interval_seconds=1800, active_hours=(6, 22)),
    on_alert=my_delivery_fn,
)
await loop.start()  # blocks until loop.stop()
```

Callbacks:

- `on_alert(output, state)`: called when the agent has something actionable to report
- `on_quiet(output, state)`: called when `HEARTBEAT_OK` is detected and suppressed
- `on_error(error_or_output, state)`: called on `RunOutput.error`, `timed_out`, or exceptions
- `on_tick(tick_result, state)`: called after every tick (including skipped/error)

All callbacks are exception-safe — failures are logged but never kill the loop.

### `HeartbeatConfig`

- `interval_seconds`: time between ticks (default 1800 = 30 minutes)
- `active_hours`: `(start, end)` half-open `[start, end)`, supports overnight `(22, 6)`, `None` = always active
- `timezone`: IANA timezone for active hours (default `"UTC"`)
- `quiet_threshold`: max chars after stripping `HEARTBEAT_OK` to classify as quiet (default 20)
- `backoff_steps`: delay sequence on errors, replaces interval (default `[30, 60, 300, 900, 3600]`)
- `checklist_path`: path to HEARTBEAT.md; skip run if file is empty (saves API cost)
- `state_dir`: persist `heartbeat_state.json` (tick counts, errors)
- `max_ticks`: optional cap for testing or bounded runs

### `TickResult`

Result of a single heartbeat tick. Fields: `output`, `skipped`, `skip_reason`, `alert`, `error`, `stripped_response`, `tick_number`, `started_at`, `duration_seconds`.

### State and Delivery Helpers

- `load_state(state_dir, state_file)` / `save_state(state_dir, state, state_file)`: atomic JSON persistence
- `extract_state_update(text)`: parse `## STATE_UPDATE_JSON` fenced blocks from model response
- `build_state_payload(prev, model_state, output, ...)`: merge state with run metadata
- `deliver(config, output, state)`: dispatch to Telegram + webhook + callback
- `format_run_summary(output, ...)`: generic run summary formatter
- `send_telegram(message, bot_token, chat_id)`: async Telegram message via httpx
- `send_telegram_file(path, bot_token, chat_id)`: send file content as chunked messages
- `strip_heartbeat_ok(text)`: strip leading/trailing HEARTBEAT_OK token, return `(stripped, had_token)`

### `send_prompt()` and `send_prompt_sync()`

Execution-only, single-call text helpers for callers that already resolved a
capability. Both require:

- an immutable `BoundCapabilityExecution` containing the exact capability bind,
  admitting registry, provider adapter, and bound auth-config snapshot

They do not select a provider/model, resolve credentials, or read credential
environment variables. Bind, registry, adapter, credential, model, and effort
disagreements are rejected before provider-client creation. Usage callbacks and
commercial usage production use the bound provider/model identity.

## Capability Binding

### `CapabilityExecutionResolver`

The canonical server-owned resolver for `session.driver`, `plan.author`, and
the `node.*` capability classes. Construct it with:

- a versioned `ProductModelSelectionPolicy`
- an `AuthContext` derived from authenticated identity and secret-free
  user/service credential handles
- a versioned `ProductModelRegistry`
- trusted credential-materializer and adapter-resolver callbacks

`resolve(capability_id, ...)` applies that capability's precedence and returns
one immutable `BoundCapabilityExecution`. `materialize_bind(bind)` restores an
exact durable bind without choosing a replacement. Both paths validate policy,
registry, principal eligibility, provider adapter, credential material, and
effort before returning.

### `BoundCapabilityExecution`

Carries the exact `CapabilityBind`, admitting `ProductModelRegistry`,
`ModelProvider` adapter, and immutable credential auth-config snapshot consumed
by a runner. Call `validate()` at an execution boundary to recheck agreement.

## Workflow Delivery Contracts

The public `agent_workflow_contracts` package exposes one version-tolerant
reader surface and explicit construction types:

- `WorkflowDeliverySpec` is the read union of
  `WorkflowDeliverySpecV1 | WorkflowDeliverySpecV2`.
- `DeliveryEnvelope` is the discriminated read union of
  `DeliveryEnvelopeV1 | DeliveryEnvelopeV2`.
- `parse_workflow_delivery_spec(value)` admits only the deployed absent-version
  v1 spec or an explicit `schema_version="2.0"` spec. An explicit version on a
  v1 spec is rejected so historical bytes cannot silently change meaning.
- `parse_delivery_envelope(value)` admits only explicit envelope versions
  `"1.0"` and `"2.0"`; missing and unknown versions fail closed.

Use the explicit V1 classes only to inspect or test historical records; live
starts and settlement write V2 exclusively. V1 preserves the authored-summary
delivery form. V2 removes summary coupling:
its primary carries an exact `PublishedOutputRef` plus a bounded
`DeliveryPreview` whose UTF-8 byte interval, total bytes, completeness, and
omitted bytes are validated. `DeliverySettlement` accepts only matching
V1/V1 or V2/V2 spec/envelope pairs. Exact output retrieval remains authoritative
for both versions.

`DELIVERY_PREVIEW_POLICY_VERSION` and `DELIVERY_PREVIEW_MAX_BYTES` identify the
currently supported deterministic prefix policy. The wheel also includes the
generated JSON Schemas, TypeScript declarations, frozen historical-v1 golden,
and complete/truncated v2 goldens under
`agent_workflow_contracts/generated/`.

### Autonomous capability handoff

`AutonomousCapabilityBindingRequest` is passed only to the trusted
`GatewayServerConfig.autonomous_capability_binding_resolver` callback.
Fresh starts and schedules select once; resumes include the required persisted
bind. The callback returns `AutonomousCapabilityBinding`, a secret-free exact
bind and, only for a run-scoped user
principal, an opaque credential-handle identity with repr-hidden,
memory-only material for the launch pipe.

Before spawning a child, the gateway persists that pair and uses
`sign_autonomous_launch_envelope()` to bind it to the task id, control-run id,
owner identity, expiry, and nonce. The child calls
`verify_autonomous_launch_envelope()`, removes the envelope, handoff marker,
and signing secret from its environment, validates any user credential read
from the anonymous stdin pipe against the signed handle and launch identity,
and materializes the verified bind without re-selection. Invalid, expired,
replayed, or identity-mismatched envelopes and credential handoffs fail before
model or MCP setup.

## Server

### `create_gateway_app(config)`

Low-level FastAPI server factory.

Use it when you need:

- custom approval logic
- interceptors
- multiple runtime profiles
- provider lifecycle or runtime wiring beyond what `create_agent()` exposes
- advanced auth, CORS, or transcript behavior

### `GatewayServerConfig`

Top-level app configuration.

Key fields:

- `build_chat_runtime`: required async factory that returns `ChatRuntime`
- `auth_manager` or `valid_api_keys`, `jwt_secret`, and `session_ttl`: auth and
  session envelope
- `tenant_id`: server-owned tenant bound into credential provenance
- `model_registry`, `model_selection_policy`: canonical stable execution
  identity registry and product selection policy
- `model_preference_store`: durable saved preferences keyed by authenticated
  tenant, actor, and capability
- `service_provider_handles`, `service_auth_config_resolver`,
  `capability_adapter_resolver`: server-owned credential and adapter
  materialization
- `autonomous_capability_binding_resolver`: trusted pre-spawn resolver for
  autonomous start, schedule, and exact resume
- `dispatch_scope_validator`: optional dispatch-time validator/canonicalizer for redacted structured portfolio scope
- `cors_origins`, `prefix`: HTTP surface
- `on_event`, `on_startup`, `on_shutdown`: app lifecycle hooks
- `transcript_dir`: JSONL transcript output

### `ChatRuntime`

Per-request runtime description.

Key fields:

- `system_prompt`
- `build_runner`
- `capability_execution`
- `get_tool_definitions`
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
For routed MCP tools, `provider_id` carries the gateway router's trusted
adapter selection; it is never sourced from provider-returned payload content.
Legacy or unrouted results leave the field unset.

Useful when you want to:

- sanitize large tool payloads
- inject follow-up content blocks
- log structured tool metadata

### `AgentSDKRunner` and `AgentSDKConfig`

Alternative execution path that keeps the same gateway HTTP surface but delegates tool-loop behavior to the pinned Anthropic agent SDK.

Use this when SDK parity matters more than native-runner control.

`AgentSDKConfig` accepts the same identity fields as `AgentRunner` (`user_id`, `channel`, `rate_table_version`, `billing_mode`); both runners require explicit usage identity before emitting billing records.

## Usage Callbacks and Billing

### `UsageEvent`

Per-turn billing event emitted by both `AgentRunner` and `AgentSDKRunner`. Carries token counts, cost, identity (`user_id`, `channel`, `request_id`), and a unique `event_id` (UUID, default-generated). Fed to `UsageLedger.record()` and to user-supplied `on_usage` callbacks.

### `SessionUsageSummary`

Aggregated billing record emitted **once per chat** after background-task drain. Fields include summed tokens/cost, `turns`, `request_id`, plus `drain_complete: bool` and `in_flight_task_count: int` for detecting incomplete drains. Sub-agent `UsageEvent`s roll into the parent runner's summary via the internal aggregator.

### `normalize_identity()`

Helper that validates and normalizes identity across `AgentRunner` and `AgentSDKRunner`. `user_id`, `rate_table_version`, and `billing_mode` are required; `_default` is reserved-invalid; `channel` is stripped or returned as `None`.

### Runner callbacks

Both runners accept three usage-related callbacks at construction:

- `on_usage: Callable[[UsageEvent], Awaitable[None] | None]` — fires per-turn for live cost streaming. Use for SSE telemetry, real-time UI updates.
- `on_session_summary: Callable[[SessionUsageSummary], Awaitable[None] | None]` — fires once per chat after drain. Use for billing writes (`record_cost`).
- `on_late_usage_event: Callable[[UsageEvent], Awaitable[None] | None]` — fires when a `UsageEvent` arrives **after** `on_session_summary` already emitted (e.g., a background sub-agent finished post-stream). Wire this to a spool table for reconciliation.

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

### `InterceptContext`, `InterceptDecision`, `InterceptResult`, and `ToolInterceptor`

Interceptor contract for runtime tool policy.

`InterceptDecision.action` supports `allow`, `warn`, `ask`, and `deny`.

`InterceptResult` is the structured return from the dispatcher's interceptor
pipeline and exposes `proceed`, `warnings`, `error`, and `pending_ask`.

Use interceptors when you want to warn, ask for approval, or deny before any
tool executes.

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

For a named operation, the handler freezes the registered skill profile's
finite `max_budget_usd` into `AgentExecutionSnapshot`. Initial execution and
durable resume enforce that frozen value while continuing to forward cost to
the telemetry-only observation accumulator. The public tool input cannot set
or increase this hard budget; unnamed delegation retains observation-only cost
tracking.

Key parameters:

- `runner_ref`: single-element list holding the active `AgentRunner`
- `skill_loader`: optional `SkillLoader` for named agent profiles
- `mcp_client`: MCP client manager shared with the parent
- `local_tool_handlers`: local handlers forwarded to the sub-agent dispatcher
- `excluded_tools`: additional tool names to block in sub-agents
- `on_before_background`: sync callback invoked just before a background task starts
- `on_background_complete`: async callback invoked when a background task finishes

The handler supports `background=true` in the tool input. When set, the sub-agent runs asynchronously and the tool returns immediately with a `task_id`. With automatic notifications enabled, the terminal result is delivered through a typed task notification; current-run tasks must not be polled.

### `make_run_agent_tool_def(...)`

Factory for the tool schema exposed to the model. Includes the `background`
boolean property and deliberately excludes `max_budget_usd`; hard budget
authority comes from the registered named profile and execution snapshot.

### `make_get_background_result_handler(runner_ref)`

Factory for the local `get_background_result` handler. Delegates to `AgentRunner.get_background_result()`. The handler retrieves historical results and payloads explicitly omitted from automatic notifications. When automatic notifications are disabled, an exact task request may perform one explicit bounded wait.

### `make_get_background_result_tool_def()`

Factory for the `get_background_result` tool schema. The tool accepts an exact `task_id`, or `"*"` for a bounded task-ID/status directory that never returns aggregate payloads. Optional `wait` and `timeout` fields support the notification-disabled path, with timeout clamped to 120 seconds. Oversized exact results are delivered as bounded canonical-JSON text pages; callers pass each opaque `cursor` back unchanged. An omitted payload remains retained after its final exact page is returned. It is released only after a later successful provider request contains that page's tool result and the resulting assistant response is durably recorded.

## Typed Events

Added in 0.15.0 and extended by later result-capture work. Frozen dataclasses cover the run/artifact lifecycle events emitted on the `/api/chat` SSE stream when the host wires up a `SkillProfile` and a `skill_run_id`; structured skill results are emitted as `skill_result_captured` wire events. The dataclass event classes plus `event_to_dict` are re-exported from `agent_gateway` top-level; the rest live in `agent_gateway.events`. Wire format (JSON shape on the SSE stream) is documented in `http-api.md` → "Skill Framework Events".

### `SkillRunStartedEvent`

Emitted once at the start of a skill-framework sub-agent run.

Fields: `skill_run_id`, `skill`, `ticker`, `ts`. `type` is fixed to `"skill_run_started"`.

### `SkillResultCapturedEvent`

Emitted when a skill run completes with a structured runtime result. This is the display/control-plane source for status, gate code, artifact refs, proposal ids, FMS result envelopes, and verdict echo data.

Fields: `skill_run_id`, `skill`, `ticker`, `exit_code`, `outcome`, `status`, `gate_code`, `artifact_refs`, `proposal_ids`, `verdict_echo`, `fms_results`, `artifact_events`, `output_memory_file`, `cost_usd`, `duration_s`, `compaction_count` (a non-negative per-run count; legacy captures default to `0`), `error`, `warnings`.

### `ArtifactReadyEvent`

Emitted when a structured report door or artifact-producing tool writes a JSON sidecar (and optionally a binary artifact such as a `.docx` letter) to per-user workspace storage. Pairs with the artifact read endpoints documented in `http-api.md`.

Fields: `skill_run_id`, `ticker`, `skill`, `artifact_id`, `artifact_path`, `binary_artifact_path` (`str | None`), `contract_name`, `data_source` (`"live" | "fixture"`), `ts`.

### `AggregateReadyEvent`

Emitted when an aggregate view-model has all its source artifacts.

Fields: `skill_run_id`, `ticker`, `view_model_id`, `trigger` (`AggregateReadyTrigger` with `kind: "artifact_ready" | "tool_response"` + `source: str`), `sources_complete`, `ts`.

### `ArtifactFailedEvent`

Emitted when a structured report door or artifact-producing tool fails to produce an artifact.

Fields: `skill_run_id`, `ticker`, `skill`, `error_code` (`"validation" | "missing_contract" | "schema_drift" | "tool_write_failed" | "other"`), `error_detail`, `source_path`, `ts`.

### `ArtifactUnavailableEvent`

Renderer-side event. No `skill_run_id` — surfaced when the UI/aggregator looks up an artifact for `(ticker, skill)` and finds it absent.

Fields: `ticker`, `skill`, `reason` (`"no_runs_yet" | "stale" | "fixture_only" | "auth_blocked"`), `affordance` (short user-facing hint), `ts`.

### `event_to_dict(event)` and `event_from_dict(payload)`

Serialize a typed event to a JSON-ready dict (with `type` set from the class's frozen `type` field) and parse one back. Round-trips `data_source` literals, `AggregateReadyTrigger` nesting, and optional `binary_artifact_path` / `materiality_cushion` / `confidence`. `event_from_dict` raises `ValueError` on unknown `type`.

### Helper constants

- `TYPED_EVENT_TYPES` — frozenset of all six event-type strings.
- `RUN_SCOPED_EVENT_TYPES` — frozenset of the five that carry a `skill_run_id` (excludes `artifact_unavailable`).
- `TypedEvent` — union over the six event classes.

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

Anthropic adapter used by `create_agent(provider="anthropic")`.

### `OpenAIProvider`

OpenAI-compatible adapter used by `create_agent(provider="openai")` or when you build the app yourself.

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
