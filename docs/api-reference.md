# API Reference

This is a hand-written guide to the main public APIs in `agent_gateway`.

For endpoint payloads and SSE event schemas, see [HTTP API](./http-api.md). For architecture and control flow, see [Architecture](./architecture.md).

## Getting Started

### `create_agent()`

Fastest way to create a gateway server from a system prompt.

Use it when you want:

- Anthropic-, Codex-, OpenAI-, or XAI-backed chat
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
- `admitted_skill_execution_limits`: required keyword-only argument with no
  default, typed `agent_gateway.skill_limits.SkillExecutionLimits |
  None`; callers with no admitted skill limits must still pass `None`
- `session`: exact `GatewaySession` that owns the authenticated runtime
  identity and mutable lifecycle state
- `user_id`: required stable user identity for usage accounting
- `billing_mode`: required usage billing mode, either `byok` or `metered`
- `rate_table_version`: required usage rate-table/version label
- `skills_dir`: directory of markdown skill files used by `run_agent`
- `outputs_dir`: directory for named-skill output files; stale same-day outputs are cleaned before background `run_agent` launch
- `state_dir`: optional directory for persisted run state (JSON load/save between runs)
- `delivery`: optional `DeliveryConfig` for webhook, Telegram, briefing file, or callback output
- `containment`: optional `ContainmentSettings` (spawn depth, concurrent children,
  soft-ladder knobs, hung-child silence, context thresholds); `None` resolves
  `resolve_containment_settings(os.environ)` when the call starts; an unreadable
  `HANK_MAX_SPAWN_DEPTH` raises `ContainmentSettingError` naming the setting, and
  any other invalid value logs a warning and keeps its default
- `compaction_instructions`: optional instructions for context compaction
- `interceptors`: optional trusted interceptor sequence; automatic `run_agent`
  children retain the same stateful instances in the same order as the parent
  dispatcher

Raw selectors are not accepted: `api_key`, `auth_config`, `auth_token`,
`bound_auth_config`, `capability_bind`, `execution_transport`, `max_tokens`,
`model`, `provider`, and `provider_config`. `run_autonomous()` takes no
`**kwargs`, so they raise `TypeError`; `run_autonomous_sync()` rejects the same
set explicitly. Provider, model,
effort, transport, credentials, and runtime identity are therefore all decided
upstream of this boundary. Split execution inputs are also rejected.
Bind/config/adapter/registry disagreements and unavailable credentials fail
before session scoping, MCP construction/startup, or provider-client creation.
The executor snapshots the bound mapping before its first await and passes
`allow_stub_response=False`.

No wall clock bounds the run or its `run_agent` children. Liveness is the
event-gap stall watchdog plus the parent-side child activity guard. The same
no-progress bound (`runner_streaming.effective_stream_stall_timeout`; heartbeats
are not progress) covers the direct provider streams of portable compaction
(`provider_summarize`) and `send_prompt`, which raise `StreamStallError`; time a
request spends queued for local Anthropic rate-limit capacity
(`rate_limit.locally_queued`) is not a stall. The deadline is checked before
consuming ready events, so continuously ready heartbeats cannot bypass it. A
delegated run (a `run_agent` child or a learning fork) meets the soft-budget ladder
(`agent_gateway/runner_soft_budget.py`): a wrap-up steer at the threshold, a
forced final report at 1.5x, and an `interrupted` run with reason
`soft_budget_abort` only if the child still has not finished two provider
requests later. The ladder counts provider requests (turns plus `max_tokens`
continuation segments). Its threshold has one source, the
`ContainmentSettings.soft_turn_budget` table keyed by run kind (`default` 100,
`autonomous` 100, `fork` 10); nothing a skill, profile or caller declares feeds
it. `HANK_SOFT_TURN_BUDGET` (`soft_turn_budget_cap`) can only
lower it, and `0` disables the ladder. A primary run has no ladder, and no run
carries a cost ceiling: spend is measured (`stream_complete.usage`, usage
events), never enforced.

Returns a `RunOutput`; see the complete field list below.

### `run_autonomous_sync()`

Synchronous wrapper around `run_autonomous()` for cron scripts and autonomous
jobs. It requires the same prebound execution inputs and returns the same type.

### `RunOutput`

Result of an autonomous run. Fields:

- `response: str` — final model response text
- `tools_used: list[str]` — tool names called during the run
- `usage: dict` — token usage from the provider
- `error: str | None` — error message if the run failed
- `timed_out: bool` — timeout marker retained in the result contract; the
  autonomous entry point applies no wall clock, so it stays False
- `operator_paused: bool` — True when an operator intentionally paused the run
- `max_tokens_reached: bool` — REMOVED. A provider length stop is not a result field; it is the `max_tokens_stops` measurement on `session_recap.tool_calls_summary.measurements`.
- `exit_reason: str | None` — more specific terminal reason, when one exists
- `post_run_guard: dict | None` — structured post-run guard failure details

### `DeliveryConfig`

Where results go after a run completes.

- `on_complete`: async/sync callback receiving `(RunOutput, state_dict)`
- `telegram_bot_token`, `telegram_chat_id`, `telegram_label`: built-in Telegram delivery
- `briefing_file`: path to an artifact file to send as follow-up content
- `webhook_url`: HTTP POST endpoint for the run summary
- `format_message`: optional custom formatter `(RunOutput, state) -> str`

### `run_output_exit_code()` and `run_output_outcome()`

Map a `RunOutput` to process exit codes: `0` for success or operator pause, `1`
for errors and post-run guard failures, and `124` for timeout. Semantic outcomes
include `success`, `timeout`, `error`, `operator_pause`, or a more specific
`exit_reason`. Exit codes are for process semantics; outcomes are for persisted
state tracking. A provider length stop changes neither: it is counted in
`session_recap.tool_calls_summary.measurements["max_tokens_stops"]`.

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

# Application/server-owned preparation step; `prepare_autonomous_execution` is
# the caller's own resolver call, not an `agent_gateway` export.
capability_execution: BoundCapabilityExecution
session: GatewaySession
capability_execution, session = prepare_autonomous_execution()

loop = HeartbeatLoop(
    run_fn=partial(
        run_autonomous,
        system_prompt="...",
        initial_message="...",
        capability_execution=capability_execution,
        admitted_skill_execution_limits=None,
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
- `agent_gateway.autonomous.send_telegram_file(path, bot_token, chat_id)`: send
  file content as chunked messages; this helper is module-level rather than a
  top-level `agent_gateway` export
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

`select_bind()` selects a secret-free `CapabilityBind` without materializing a
credential or adapter. Both selection methods accept optional `role`, exact
`effort`, and `selection_snapshot: SelectionSnapshot | None`. Admission supplies
the frozen selection snapshot; this adapter does not derive it from a runtime
session. `authorize_bind()`
and `materialize_bind()` retain exact frozen-bind reauthorization without
reselection.

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
`packages/agent-workflow-contracts/agent_workflow_contracts/generated/`.

### Ordinary agent completion

Foreground results and background notifications use `AgentCompletionEnvelope`.
`settlement_projection` carries the canonical execution status and, when the
child did not succeed, its `terminal_reason`. A child the soft-budget ladder
drove to its forced final report and that answered settles as an ordinary
completed result carrying its usage. A child aborted by the ladder settles
`failed` with `terminal_reason` `soft_budget_abort`; its envelope still
publishes with `parent_materialization: null` and any child evidence.
Intermediate tool-use prose is not promoted to a completed report. Successful
results continue to carry exact inline content or a readable content handle.

`get_agent_result_content` reads the canonical `source.content_id` of inline
narrative and projection materializations as well as overflow handles. Authority
comes from the executing job's admitted outputs or explicitly passed handles,
independent of the parent's presentation choice. Optional `read_grant_id` selects
an existing grant-bearing materialization; it does not grant access or select a
grantless inline materialization. Canonical task-result and byte integrity checks
apply to every readable source.

When the child settled on a terminal door's answer, `stage_refs` carries the
refs that door minted (`read_back_refs` of its result: a persisted set's ref, a
skill artifact's `{skill, artifact_id, content_sha256}`) in every arm, so a
result that overflowed to a handle still names what its door persisted; the
field is omitted when the door minted none.

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
- `transcript_dir`, `transcript_retention_days`: JSONL transcript output and
  the age horizon that sweeps it. The horizon defaults to `0`, which deletes
  nothing; a positive value unlinks transcripts and their `.meta.json`
  sidecars older than that many days
- `server_policy`: explicitly bound product policy callbacks; one policy per
  gateway process, installed before runtime construction (no checkout discovery)
- `identity_resolver`, `mcp_user_key_lookup`: application identity and per-user
  credential projection for control sessions and autonomous children
- `batch_backend`: application-owned batch registry/controller and captured-run
  operations; omit when the embedding application does not offer batches
- `operator_schedule_backend`: application-owned operator scheduler/jobs API;
  independent of the generic per-user agent-run schedule store

Schema-backed artifact stores, emitters and HTTP routes are application code,
not exports of this package. Applications mount those routes after creating
the gateway app and supply an `artifact_tools_installer` for product sub-agents.

### `ChatRuntime`

Per-request runtime description.

Key fields:

- `system_prompt`
- `build_runner`
- `capability_execution`
- `get_tool_definitions`
- `execution_location`

### `RequestContext`

Container for mutable request-scoped objects such as the active
`GatewaySession`, `EventLog`, and approval callback.

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

Per-provider-request billing event (a portable compaction request emits its own event before the following request) emitted by both `AgentRunner` and `AgentSDKRunner`. Carries token counts, cost, identity (`user_id`, `channel`, `request_id`), and a unique `event_id` (UUID, default-generated). Fed to `UsageLedger.record()` and to user-supplied `on_usage` callbacks.

### `SessionUsageSummary`

Billing record emitted **once per session run** after background-task drain: one for the root run and one for each delegated sub-agent run (`session_id` `sub<N>:<root session id>`). Fields include summed tokens/cost, `turns`, `request_id`, `unit_kind` / `unit_id` (the root's declared unit, shared by every run in the dispatch tree), plus `drain_complete: bool` and `in_flight_task_count: int` for detecting incomplete drains. A summary counts only its own run's `UsageEvent`s and its own model; the root's summary does not include its sub-agents' usage. `session_recap.usage` and the commercial usage reconciliation read the whole dispatch tree's total.

### `normalize_identity()`

Helper that validates and normalizes identity across `AgentRunner` and `AgentSDKRunner`. `user_id`, `rate_table_version`, and `billing_mode` are required; `_default` is reserved-invalid; `channel` is stripped or returned as `None`.

### Runner callbacks

Both runners accept three usage-related callbacks at construction:

- `on_usage: Callable[[UsageEvent], Awaitable[None] | None]` — fires per provider request (including compaction requests) for live cost streaming; sum events for a turn or session total. Use for SSE telemetry, real-time UI updates.
- `on_session_summary: Callable[[SessionUsageSummary], Awaitable[None] | None]` — fires once per chat after drain. Use for billing writes (`record_cost`).
- `on_late_usage_event: Callable[[UsageEvent], Awaitable[None] | None]` — fires when a `UsageEvent` arrives **after** `on_session_summary` already emitted (e.g., a background sub-agent finished post-stream). Wire this to a spool table for reconciliation.

## Tools And Approval

### `ToolDispatcher`

Routes tool calls to local handlers or MCP servers and applies:

- interceptors
- approval rules
- in-session approval caching

Pass the actual executing session through `session`, or bundle it with its
authenticated user and credential identity in `DispatchIdentity`. Local-handler
context, role policy, per-user MCP authentication, and source-pack capture use
that execution authority. A `DispatchIdentity` cannot also be supplied alongside
separate execution-identity arguments. Build it with `dispatch_identity(session=...,
execution=..., context=..., attempt_lineage=None)`: `execution` is the identity the
caller resolved, and `None` means a resolved absence, never derived from the
session. `context` is the executing admitted context. `execution`, `context`, and
`attempt_lineage` are required keyword arguments. Pass `attempt_lineage=None` for
a non-resume attempt; a successor supplies its trusted async resume-ancestry lookup.

`approval_route` is independent: its ledger and policy authorize the money
boundary, and its session owns pending approval cards and decision queues. An
in-process child can therefore execute under its own session while
`InProcessDelegatedApprovalRoute` records and waits for decisions on its parent's
ledger and queue. The route never supplies or replaces execution identity.

### `ToolExecutionContext`

Optional context object passed to local tool handlers. Use `emit()` to stream extra structured events such as `tool_output_chunk`.

### `ToolResult`

Standard tool return shape: `(result, error)`.

### `ApprovalRequest` and `ApprovalDecision`

Types used by the approval loop.

`ApprovalDecision.allow_tool_type` requests a persistent grant; only an approval policy that permits grants mints one. Each approval otherwise decides one call.

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
- session-owned host frame reader shared by request bundles

`bundle.bind_dispatcher(...)` binds the turn's dispatcher, a `code_callable(operation)`
predicate the embedding application derives from the exact installed route's producer
policy and served disposition retained on its registered live binding (a same-named
operation on another server is decided by that server and route). Callable unless
the served review is `tool_only` or `internal_only`, the operation is in that
server's internal-only audience, or its effect class is `irreversible`.
Absent, malformed or `unclassified` producer metadata does not exclude the route;
malformed metadata logs once at registration without rejecting the tool. Optional
`catalog_metadata(operation)` supplies route identities, alongside the application's source extraction
and identity callbacks. Cells compose those granted operations through the injected
`hank.search(query)`, `hank.describe(operation)` and `hank.call(operation, **arguments)`;
catalog rows carry the route metadata unchanged, and grant is rechecked per call.
The host reader dispatches through the canonical dispatcher and settles producer
source identities. Complete deduplicated `sources` stay inline; model-output
spilling handles large envelopes. Nested payloads go to the cell, not directly
to the model. A cell registration permits repeated frames until that cell ends.

### `build_code_execution(session, config=None, *, job_runner=None)`

Inject built-in code execution tools into a runtime.

Behavior:

- `host=auto` prefers Docker when available
- falls back to a registered host kernel, then subprocess; skipped-backend reasons are logged at `WARNING`
- every `code_execute` result names where it ran: `backend`, `sandboxed`, and `skipped_backends` (`[{backend, sandboxed, reason}]`, empty unless `host=auto` skipped one)
- stores reader, interpreter states and work directory on the session
- `job_runner()` answers the `AgentRunner` that owns the request's task registry, or `None`, at call time; a `background=true` cell registers there as a `code` job and is inspected, waited on and killed through the registry, and with no runner it returns `code_execution_unavailable`

### `cleanup_code_execution(session)`

Kill and await the running `code` jobs of the session's live turn runner, await
interpreter shutdown and nested dispatch closure,
then remove the session work directory. Call this on final session expiry,
task teardown or app shutdown if you manage runtimes yourself. Ordinary cell
completion or interruption does not dispose a persistent interpreter.

### `docker_backend_unavailable_reason(config=None, *, ttl_s=15.0)`

Read-only availability probe for the Docker backend `config` names: returns that `DockerBackend`'s `unavailable_reason()` (`None` when it can run). The answer is memoised per image name for `ttl_s` seconds; `clear_docker_availability_cache()` forgets it. A runtime that must offer `code_execute` only under the sandboxed backend calls this before `build_code_execution()`.

### `DockerBackend`, `KernelBackend` and `SubprocessBackend`

Concrete execution backends of `ExecutionBackend`.

- `DockerBackend` is treated as sandboxed
- `KernelBackend` is an unsandboxed persistent host interpreter, registered by default (`CodeExecutionConfig.register_kernel=True`); sandboxed-only deployments must set `register_kernel=False`
- `SubprocessBackend` is treated as unsandboxed
- `unavailable_reason()` returns why a backend cannot run now, or `None`; `available()` derives from it, so a custom backend overrides `unavailable_reason()` only

Subprocess carries the bridge over inherited pipes. Docker and the host kernel
run the same dependency-free `_kernel_runner.py` with the shared client and
codec shipped as source. The container receives cell tokens over stdin, not a
new environment allowlist or socket mount; its direct networking remains
disabled and domain reads execute through the gateway dispatcher.

Every placement uses the same 2 MiB wire-frame and 64 MiB executing-cell
aggregate bounds. Oversize requests, replies and aggregate overflow report
`protocol_error`, not truncated success or silent source loss. Exhausting one
cell does not grant further dispatch or destroy the session interpreter.

Docker/kernel cells retain Python locals and imports. A non-`None` trailing
expression returns an implicit MIME summary: `text/plain` is capped at 16 KiB
including its truncation marker; serializable containers also receive an
`application/json` copy only when its encoding fits within 16 KiB and the
representation was not truncated. The value is bound to `_` (kernel: available
in later cells; the truncation marker names it). Implicit output does not invoke rich
representations or alter the original value. Explicit
`display()` retains its existing rich MIME behavior and wire-frame bounds.
Results include full stdout/stderr text up to `max_output_bytes`, a separate
`stdout_tail` containing the last 20 lines, and workdir-relative spill paths
holding the complete output. Those logs remain until session cleanup.
Cell interruption cancels nested work and preserves previously assigned locals;
session teardown awaits interpreter/container disposal. Request bundles bind
each new cell to the current dispatcher and granted surface, so persistent
locals or cached discovery cannot retain a wider old grant.
Subprocess timeout measures process exit independently of stream drainage:
readers stay active through termination grace, then have a two-second drain
bound so a descendant holding a writer cannot stall teardown.

### `make_code_execute_tool_def()`

Factory for the public `code_execute` tool schema exposed to the model. A
`background=true` cell runs as a `code` job on the task registry of the runner
`build_code_execution(..., job_runner=...)` answers; its result is delivered as
any background task result.

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
- `metadata`

### `SkillStateStore`

Simple JSON persistence for per-skill state.

### `parse_skill_file(path)`

Parse a standalone skill markdown file into a `SkillProfile`.

### `make_run_agent_handler(...)`

Factory for the local `run_agent` handler.

The child carries no turn or cost number of its own: its soft-budget ladder
threshold is the bundled table's, and its spend is recorded against the
telemetry-only `cost_observation_threshold_usd`, which never admits,
interrupts or gates anything.

Key parameters:

- `runner_ref`: single-element list holding the active `AgentRunner`
- `skill_loader`: optional `SkillLoader` for named agent profiles
- `mcp_client`: MCP client manager shared with the parent
- `local_tool_handlers`: local handlers forwarded to the sub-agent dispatcher
- `excluded_tools`: additional tool names to block in sub-agents
- `interceptors`: optional trusted sequence forwarded to the child dispatcher
  without cloning or reordering its stateful elements
- `on_before_background`: sync callback invoked just before a background task starts
- `on_background_complete`: async callback invoked when a background task finishes

The handler supports `background=true` in the tool input. When set, the sub-agent runs asynchronously and the tool returns immediately with a `task_id`. With automatic notifications enabled, the terminal result is delivered through a typed task notification; current-run tasks must not be polled.

Optional `handles` is a list of canonical content-handle strings granting the
child read access. Only handles already granted to the executing caller or
published by that caller's job or its descendants in the durable session log
may be passed. Spill membership comes from durable successful-publication
observations binding the exact content reference to its producer job and tenant.
The reader resolves the producer's spill root internally; model-facing handles
do not expose private filesystem paths. Historical own/descendant rights and
the canonical admitted-resumer custody event determine accessible producers.
Duplicates are
collapsed; unowned handles are refused by shared admission. A nested spawn
retains its executing child's scope, not the root session's scope. Message
citations do not grant reads.

### `make_resume_handler(...)`

Factory for the local `resume_background_agent` handler. Its optional trusted
`interceptors` sequence is forwarded to every resumed child dispatcher without
cloning or reordering its stateful elements. Production initial and resume
wiring supplies the same finalized instances. A caller-supplied custom
`run_agent` handler remains caller-owned and is not wrapped or replaced.

The required keyword `operation_resume_bindings` is either `None` or a callback
from the parked `AdmittedTask` to `OperationResumeBinding | None`. The frozen
binding is exported from `agent_gateway.sub_agent` and carries
`local_handler_scopes`, `usage_aggregator`, and `on_settled`. Scopes wrap existing
stock local handlers, raising `KeyError` for a missing handler. A non-`None`
aggregator receives successor usage instead of the parent's; `None` inside a
binding gives the child independent accounting. Returning no binding preserves
ordinary resume behavior. The optional async settlement callback receives the
canonical entry with cumulative `TaskResult` usage after activity release, before
the workflow host completion callback. These bindings are runtime-only; identity
is selected from the parked admission, never an added persisted field.

Successful continuation records one additive custody event on the canonical
successor `AttemptRef.resumer_job_id`: the actual admitted resumer job receives
read custody over the resumed job and its descendants. This is not a union of
the resumer's ancestors, a session/runner alias grant, or a rewrite of historical
rights. Original execution lineage, executable grants and message-roster rights
remain unchanged. Ordinary-result and spill output owners project that event;
spill reads retain exact-reference, tenant and content-integrity checks.
Workflow continuation's restore API remains incomplete; this contract does not
claim that workflow continuation or all handle-family journeys are restored.

### `make_run_agent_tool_def(...)`

Factory for the tool schema exposed to the model. Includes optional `handles`
and the `background` boolean property; it carries no turn or cost ceiling.

### `make_get_background_result_handler(runner_ref)`

Factory for the local `get_background_result` handler. Delegates to `AgentRunner.get_background_result()`. The handler retrieves historical results and payloads explicitly omitted from automatic notifications. When automatic notifications are disabled, an exact task request may perform one explicit bounded wait.

### `make_get_background_result_tool_def()`

Factory for the `get_background_result` tool schema. The tool accepts an exact `task_id`, or `"*"` for a bounded task-ID/status directory that never returns aggregate payloads. Optional `wait` and `timeout` fields support the notification-disabled path, with timeout clamped to 120 seconds. Oversized exact results are delivered as bounded canonical-JSON text pages; callers pass each opaque `cursor` back unchanged. An omitted payload remains retained after its final exact page is returned. It is released only after a later successful provider request contains that page's tool result and the resulting assistant response is durably recorded.

## Typed Events

Frozen dataclasses cover skill runs, artifacts, approvals, workflow output,
agent completion, UI blocks, and session recaps. Import the complete contract
from `agent_gateway.events`; selected classes and `event_to_dict` /
`event_from_dict` are also available at package top level. On SSE, the typed
event dictionary is the `event` member of the transport envelope documented in
[HTTP API](./http-api.md).

| Wire type | Purpose |
| --- | --- |
| `skill_run_started` | Start of a named skill run |
| `skill_result_captured` | Structured skill result and execution summary |
| `artifact_ready` / `artifact_updated` / `artifact_failed` / `artifact_unavailable` | Artifact lifecycle |
| `ui_blocks_ready` | Renderable UI-block payload |
| `agent_completion` | Durable direct-parent child result publication |
| `workflow_output_attached` | Workflow output attached to a chat turn |
| `typed_recommendations_extracted` | Validated recommendation extraction result |
| `aggregate_ready` | Aggregate view-model readiness |
| `tool_approval_request` / `tool_approval_decided` | Approval lifecycle |
| `session_recap` | Typed recap of a session or turn range |

### `SkillRunStartedEvent`

Emitted once at the start of a skill-framework sub-agent run.

Fields: `skill_run_id`, `skill`, optional `ticker`, `scope`, optional
`portfolio_id`, and `ts`. `type` is fixed to `"skill_run_started"`.

### `SkillResultCapturedEvent`

Emitted when a skill run completes with a structured runtime result. This is the display/control-plane source for status, gate code, artifact refs, proposal ids, FMS result envelopes, and verdict echo data.

Fields include `skill_run_id`, `skill`, optional `ticker`, `scope`, optional
`portfolio_id`, execution status/result fields, artifact and proposal refs,
cost/duration, warnings, `compaction_count`, and optional approval outcome,
approval id, and approval tool name.

### `ArtifactReadyEvent`

Emitted when a structured report door or artifact-producing tool writes a JSON sidecar (and optionally a binary artifact such as a `.docx` letter) to per-user workspace storage. Pairs with the artifact read endpoints documented in `http-api.md`.

Fields: `skill_run_id`, optional `ticker`, `scope`, optional `portfolio_id`,
`skill`, `artifact_id`, `artifact_path`, optional `binary_artifact_path`,
`contract_name`, `data_source` (`"live" | "fixture"`), and `ts`.

### `AggregateReadyEvent`

Emitted when an aggregate view-model has all its source artifacts.

Fields: `skill_run_id`, `ticker`, `view_model_id`, `trigger` (`AggregateReadyTrigger` with `kind: "artifact_ready" | "tool_response"` + `source: str`), `sources_complete`, `ts`.

### `ArtifactFailedEvent`

Emitted when a structured report door or artifact-producing tool fails to produce an artifact.

Fields: `skill_run_id`, optional `ticker`, `skill`, `error_code`
(`"validation" | "missing_contract" | "schema_drift" | "tool_write_failed" |
"other"`), `error_detail`, optional `source_path`, optional `tool_call_id`, and
`ts`.

### `ArtifactUnavailableEvent`

Renderer-side event. No `skill_run_id` — surfaced when the UI/aggregator looks up an artifact for `(ticker, skill)` and finds it absent.

Fields: optional `ticker`, `skill`, `reason` (`"no_runs_yet" | "stale" |
"fixture_only" | "auth_blocked"`), `affordance` (short user-facing hint), and
`ts`.

### `event_to_dict(event)` and `event_from_dict(payload)`

Serialize a typed event to a JSON-ready dict and parse a known type back into
its frozen dataclass. `event_from_dict` raises `ValueError` on an unknown
`type`.

### Helper constants

- `TYPED_EVENT_TYPES` — frozenset of all 14 event-type strings.
- `RUN_SCOPED_EVENT_TYPES` — the seven events associated with a skill run or
  run-local UI/artifact update.
- `TypedEvent` — union over all typed event classes.

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

### `CodexProvider`

Codex adapter used by `create_agent(provider="codex")`; authentication is
delegated to the Codex CLI login flow.

### `XAIProvider`

XAI adapter used by `create_agent(provider="xai")`.

### `ModelInfo`

Per-model metadata such as context window, thinking support, and token pricing.

### `CostEstimate`

Estimated request cost broken down by input, output, and cache token categories.

### `StreamEvent`

Normalized stream event produced by providers and consumed by runners.

### `ThinkingLevel`

Provider-agnostic reasoning intensity hint.

## Sessions And Events

### `GatewaySession`

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
