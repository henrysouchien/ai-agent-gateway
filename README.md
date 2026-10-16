# ai-agent-gateway

Deploy an AI agent as an HTTP/SSE service without rebuilding session, tool,
approval, and execution infrastructure around the model loop. Requires Python
3.10 or newer.

## Ownership

This package owns the generic FastAPI chat surface, sessions and JWTs, SSE event
stream, provider/run loop, tool dispatch, approvals, code execution, skills,
sub-agents, autonomous execution, and heartbeat support.

It does not own an embedding product's profiles, prompts, channel policy,
domain tools, research data, or business schemas. Applications supply those
through `create_gateway_app()` or the higher-level `create_agent()` inputs.

The wheel imports without an application checkout. Product integrations bind
policy and identity callbacks plus optional batch/operator-schedule backends in
`GatewayServerConfig`; schema-backed artifact tools and routes belong to the
embedding application. `server_policy` is process-scoped: configure it before
building runtimes and use one product policy per gateway process.

`sub_agent.continue_approved_child()` is a host-only continuation entry through
the configured resume handler. It reserves the parked attempt's ordinary resume
successor, retains its exact tool grant and tenant, and executes the original
approval binding after the child's runner and job registry are bound, before
requesting the model. The runner publishes the approved call's result durably
through the same tool-execution path used for model-issued calls.
The required `approved_call=sub_agent.ApprovedCallRef(job_id, tool_call_id, nonce)`
identifies that exact durable original call.
The stored call's real result (including typed errors) replaces only its orphan
tool result; other orphan calls remain synthetic. Its tool-result hook extras are
projected as for model-issued calls: event-only blocks never reach the model and
visible extras follow every tool result in that message. Registration
reservations that have not started a worker remain retryable, and the retry that
starts the worker installs its own lifecycle callbacks. Existing workers are
replayed before credential materialization. Completion releases activity,
including an untransferred child lease, before notifying the host.
Model-facing resume still reissues the grant and synthesizes interrupted calls.
Callers supply `on_complete` and the required `on_park` keyword (`None` when no
park callback is needed). Configured operation-resume bindings apply to this
successor just as they do to the other resume paths.

## Main entrypoints

| Source | Primary symbol | Use it for |
| --- | --- | --- |
| [`agent_gateway/easy.py`](agent_gateway/easy.py) | `create_agent()` | A small server from a prompt, tools, skills, and provider configuration |
| [`agent_gateway/server.py`](agent_gateway/server.py) | `create_gateway_app()` | Product-owned runtime factories, auth, policy, and lifecycle integration |
| [`agent_gateway/gateway_server.py`](agent_gateway/gateway_server.py) | `run_gateway_server()` / `main()` | Privileged single-worker TCP launcher, claim-signing fd adoption, and startup-failure logging |
| [`agent_gateway/autonomous.py`](agent_gateway/autonomous.py) | `run_autonomous()` / `run_autonomous_sync()` | A prebound headless one-shot with no HTTP server |
| [`agent_gateway/admission.py`](agent_gateway/admission.py) | `admit()` | Shared spawn, continuation, message and handle authority |
| [`agent_gateway/containment_settings.py`](agent_gateway/containment_settings.py) | `ContainmentSettings` / `resolve_containment_settings()` | Every containment bound (spawn depth, concurrent children, soft ladder, hung-child silence, context thresholds) as one object, resolved once at process start from its `HANK_*` env table |
| [`agent_gateway/heartbeat.py`](agent_gateway/heartbeat.py) | `HeartbeatLoop` | Repeated prebound autonomous work with quiet windows and backoff |
| [`agent_gateway/cli.py`](agent_gateway/cli.py) | `main()` (`agent init` / `agent run`) | Scaffold and run a package project |
| [`agent_gateway/send_prompt.py`](agent_gateway/send_prompt.py) | `send_prompt()` / `send_prompt_sync()` | One already-bound prompt→completion call returning text. `completion_record_root` is required keyword-only: pass a directory to append one `completion_record` v1 row per provider attempt, or `None` to state the call is not captured |
| [`agent_gateway/completion_record.py`](agent_gateway/completion_record.py) | `build_completion_record()` / `append_completion_record()` | The `completion_record` v1 row for one-shot companion-repo model calls (contract: `contracts/completion-record-v1/completion_record.schema.json`); day-sharded `<root>/<YYYY-MM-DD>.jsonl`, append-only, never fails the call it records |
| [`agent_gateway/gateway_address.py`](agent_gateway/gateway_address.py) | `gateway_base_url()` | The origin of the gateway a process talks to: the one reader of `GATEWAY_BASE_URL` and the one place its default is spelled |

`run_gateway_server()` is the serving entry on both local supported launches, not
only the privileged host one: the immutable runtime's serving child calls it
directly, so claim-signing adoption, `uvicorn.Config`/`Server` construction and
the launch-failure journal have one owner.

The package does not infer a new model or credential inside autonomous
execution. The application resolves and passes the exact bound capability,
session, billing mode, and skill limits before calling `run_autonomous()`.

Model selection and model request facts come from `model_authority`
(`hank-model-authority`; in the monorepo `packages/model-authority/`), which this
package depends on and carries no copy of: the registry, the selection policy,
each entry's per-family `compat` block and the rate tables.
`CapabilityExecutionResolver.select_bind()` in `capability_execution.py` selects
and authorizes a secret-free bind through that owner, without materializing
credentials or constructing an adapter. `resolve()` selects and materializes
once; `materialize_bind()` executes an already selected bind without reselection.
Both selection methods accept already-lowered `role`/exact `effort` and the
admission owner's frozen `selection_snapshot`. They use the shared
model-authority resolver; this adapter does not capture principal preferences
or own a map registry.
Bound executions retain their resolver for mid-run credential rotation.
`rebind_rotated_credential()` reauthorizes a new bind naming the selected
Anthropic source generation. Fresh spawns select and materialize through the
current scoped runner's execution resolver, so later descendants inherit the
rotated generation. Generation-bound executions do not rotate to material
without source-generation metadata.
When a rate-limited pool has no unblocked account, or an outage outlasts the
transient retry budget, `rebind_model_fallback()` binds the next eligible
entry of the frozen selection snapshot's role chain (through the same
resolver, with that entry's own credential) for the turn's remaining
attempts; the runner emits `model_fallback` and the next turn binds the
primary again.
[`agent_gateway/providers/`](agent_gateway/providers/README.md) builds each request
from the bound entry. Mechanism: `docs/reference/model-authority.md`.

`admit(SpawnRequest(...), parent, owners=AdmissionOwners(...))` returns the
selected context and exact `ResolvedAuthority`, without materializing credentials.
A `SpawnRequest.model_preset` names a published preset whose frozen selection
snapshot binds the child and rides its wire, so the child's descendants inherit it
while siblings keep the parent's (subtree scope); a frozen bind ignores it.
The admitted wire's `selection_provenance` records the role, its source, the
effort source, and the preset name, scope and policy revision beside the exact
bind, and is omitted when a frozen bind records none.
`admit(ContinueRequest(...), parent, owners=ContinuationOwners(...))` instead
retains the original `AdmittedTask`: no model reselection, semantic recompile, or
replacement tool grant. Its context preserves lineage and recorded spend;
`placement_lost` refreshes only backend placement/environment facts. Both results
carry the child's approval route from `child_approval_route()`: a child never
holds `DurableLocalApprovalRoute`; an in-process child records into its parent's
ledger through `InProcessDelegatedApprovalRoute`, a detached child adopts its
inherited channel as `ParentDelegatedApprovalRoute`, and a sessionless root's
children carry its `ControlPlaneApprovalRoute`. The existing task registry, transcript
reconstruction, attempt creation and credential materialization remain outside
this pure step. A valid older task without the context projection keeps its
durable authority; missing durable routes/definitions retain the resume owner's
existing handling.
Detached approvals may carry a private capture writer on their admitted route.
For a parked call, the lifecycle sends original prepared input, exact backend,
dispatch context and admitted job/tenant/credential identity through that writer
before publishing the public approval request. A route without a writer publishes
no executable capture; public events and pending-tool maps never carry it.
`AdmissionOwners.frozen_bind` retains a plan-frozen model bind without
reselection; `materialize_bind()` is the single reauthorization owner. Fresh spawns select effort within the registered role; profile
overlays remain deferred. Declared operation children determine nested spawning.
Role narrowing reads the authenticated admitted tenant, not the presence or role
of an optional runtime session. The product policy still reads real session/profile
capability facts for its scoped exceptions; sessionless launch does not mint a
synthetic session or gain those exceptions.
Operation-private routes may extend the parent's visible surface, but cannot
bypass its withheld routes. Admission matches withheld canonical route IDs and
their registered policy names before freezing the executable grant.
The admitted wire retains `withheld_tools` across descendants and durable
continuation. Resume keeps its original executable grant while descendants
retain recorded exclusions plus the current parent's known exclusions.
`MessageRequest` and `HandleRequest` retain the caller's job context, using
spawn-passed handles plus current own/descendant outputs supplied by the durable
output owner. Message references are citations, never grants to the recipient.
The authenticated turn edge calls `root_context()` after approval, verified
research binding and withheld tools are ready; `current_parent_context()` reads
that captured scope without minting authority. A child session installs
`context_for_child(..., job_id=<physical task id>)` with `capture_job_context()`
and cannot regain root scope.
Sessionless control/schedule edges use `root_context_from_identity()` with their
authenticated identity, live roster and approval owner.
An autonomous skill launch passes its launch edge's user facts as
`AdmissionOwners.identity` (`AdmittedChildIdentity`). A spawn admits that
identity only as the parent tenant's owner (`user_id` is the tenant
`storage_id`, exact `role` equal to the tenant role), else refuses
`tenant_mismatch`, and returns it on `AdmittedContext.identity`; the wire
carries no user identity. `AutonomousRegistry.start()` fills the run record and
the signed session authority from that admitted identity; a launch admission
does not run for (once/task/pack, a parentless skill, resume) keeps the launch
edge's identity facts.
`TaskRegistry` recovery validates the canonical task and keeps its model bind as
execution authority. Duplicate registration/message bind receipts are optional
observations, not prerequisites or competing authority; canonical corruption and
task/result/message identity checks still apply.
Completion replay is dormant: even a non-agent ordinary result cannot reconstruct
a pending/running worker from `task_completed`. Such a corrupted final state
projects as failed with `task_result_settlement_mismatch`; the result and journal
remain unchanged. `adopt_reconstructed()` retains one registry entry and lock,
returning an existing entry rather than replacing it with a fresh replay copy.

Workflow-node continuation belongs to the workflow owner, not model-facing
`revive`. A trusted host first calls
`sub_agent.reserve_admitted_successor(runner, parked_task_id=...)` to reserve the
kernel's deterministic physical successor, then supplies its workflow admission
to `sub_agent.resume_admitted_successor(runner, parked_task_id=..., successor=...,
approved_call=..., on_complete=..., on_park=...)`. Pass an `ApprovedCallRef` to
execute the approved original call with this supplied successor's dispatcher
before the provider runs, replacing only that call's synthetic result. Pass
`approved_call=None` to retain synthetic results for every interrupted call.
The supplied successor and its grant are unchanged. The async park callback is installed before the
successor worker starts and observes that same entry once its parked result is
durably committed; pass `None` when no park handling is needed.
`attach_background_task(task_id, on_complete=..., on_park=...)` binds missing
lifecycle callbacks on live or adopted entries without replacing existing ones.
The runner's dispatcher must contain the handler built by
`make_resume_handler`; this reuses its configured child-runtime wiring without
exposing supplied admissions through tool inputs. The admission retains the
parked logical task, authenticated tenant and frozen capability bind, names the
parked physical attempt in `resume_of_task_id`, uses the reserved successor ID as
both physical and admitted task ID, and carries the original grant unchanged
(including its grant ID and digest). Workflow attempt IDs and resume lifecycle events remain
workflow-owned. Completion releases the gateway's activity lease before invoking
the host callback, including an awaitable callback. Supplied successor installation
serializes with parked STOP under the source entry's finalization lock until
registration settles, including initialization left running by caller cancellation.
If STOP wins before installation, resume returns `source_terminated` and releases
the unstarted successor reservation; a registered successor is the source's resumed
marker. Ordinary `revive` still accepts only ordinary delegation identity and mints
its existing ordinary successor admission. When the retained transcript holds no
model turn (the attempt was interrupted before its first assistant message), the
successor starts as a first attempt does: its admission keeps the original
`objective`, and its model input is the `admitted_task_message_from_handles` a first attempt
receives, followed only by unconsumed parent messages and any caller-supplied
`additional_context`, never the default resume instruction.

`make_resume_handler` requires the `operation_resume_bindings` keyword; pass
`None` for ordinary resume wiring. An application may supply a callback from the
parked `AdmittedTask` to the frozen, runtime-only `OperationResumeBinding` in
`agent_gateway.sub_agent`. The binding scopes existing local handlers (a missing
stock handler raises `KeyError`), selects a usage aggregator, and optionally
awaits `on_settled(entry)` after successor completion and activity release, before
the workflow host's completion callback. A binding's `usage_aggregator=None`
keeps the child's own accounting and charges nothing to its parent; a callback
returning no binding preserves ordinary parent accounting: the resumed child
delivers its own session summary under the parent's unit. The lower-level
`resume_sub_agent` also requires an explicit `usage_aggregator`: pass the parent's
own aggregator for that ordinary delegation, any other aggregator to share its
accounting, or `None` to keep it independent. The entry's
canonical `TaskResult` contains cumulative usage across attempts. Operation identity comes
only from the parked admission; no binding is persisted.
`await sub_agent.resume_binding_for(runner, task_id)` attaches the existing
admitted task without installing lifecycle callbacks and returns its entry,
evidence recovered from that runner's own durable child lineage, and the
configured operation binding. Unknown or unadmitted tasks return `None`; no
configured operation-binding callback yields a `None` binding.
Continued `node.fork` operations retain fork mode and soft-budget measurements,
with their resumed suffix marked after the reconstructed child history. Resume
uses the child's journal and frozen execution snapshot, never a new parent
conversation or system prompt.
Final narratives use the exact durable logical-response lineage for the run's
role: child callers supply `role="sub_agent"` and their session ID as
`sub_agent_id`; independent writer runs supply `role="writer"` and
`sub_agent_id=None`, with their exact `runner_id`. Both retain the session ID
as the narrative artifact identity.

## Quick start

Install a provider extra and set its credential:

```bash
pip install "ai-agent-gateway[anthropic]"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export USER_DATA_DIR="$PWD/.agent-data"
mkdir -p "$USER_DATA_DIR/gateway"
chmod 700 "$USER_DATA_DIR" "$USER_DATA_DIR/gateway"
```

Create and run a project:

```bash
agent init my-agent
cd my-agent
agent run
```

Or create a FastAPI app directly:

```python
from agent_gateway import create_agent

app = create_agent(
    "You are a concise research assistant.",
    skills_dir="skills",
)
```

Serve the module with `uvicorn agent:app`. The default is the stable
`session.driver` entry in the configured model registry; applications can pass
another eligible stable `model_key`.

For OpenAI, install `ai-agent-gateway[openai]` and set `OPENAI_API_KEY`.
Managed provider login flows are available through `agent auth login
<provider> [--api-key]`. Supported doors:
- `gemini` (default: Antigravity Google OAuth loopback; `--api-key`: Google AI Studio key prompt)
- `google` (Google AI Studio OAuth PKCE loopback when an OAuth client is configured, or API key prompt)
- `kimi-code` (default: Kimi Code OAuth device-code flow; `--api-key`: Kimi Code API key prompt); seats land in `$USER_DATA_DIR/users/<risk_user_id>/kimi_code/oauth.json` (per user) or `$USER_DATA_DIR/kimi_code/oauth.json` (deployment default), API keys in `credentials.json` beside them
- `muse` (default: Meta Muse Code device-code flow minting Model API key; `--api-key`: Meta Model API key prompt)
- `ollama-cloud` (Ollama Cloud API key prompt)
- `ollama` (delegated `ollama signin` on local daemon)
- `copilot` (GitHub device-code flow)
- `anthropic` (Claude Code OAuth enrollment)
- `codex` (CAAM-managed ChatGPT subscription enrollment)
- `xai` (xAI device-code flow)
- `openai` (informational guidance to OPENAI_API_KEY / codex)
Use `agent auth status <provider>` to inspect stored credentials without secret leakage.

Repeated `agent auth login <provider>` enrollments append accounts to the
provider's multi-account credential pool, or update an existing account by
identity without overwriting other seats. `providers/credential_pool.py` owns
the shared `"accounts"` store envelope, legacy single-record migration, and
cooldowns for every pool provider except Anthropic. A scoped user's accounts
lead the deployment pool; usage limits and refresh failures park a seat so the
running request can rotate to an unblocked sibling, keeping that seat's token
and provider-specific fields together.
Public `google` OAuth uses PKCE S256; `gemini` Antigravity uses confidential-client
loopback OAuth with `ANTIGRAVITY_CLIENT_SECRET`, not PKCE.

Anthropic selects by quota instead. Each store keeps a `<store>.selection.json`
sidecar (mode `0600`) holding per-account policy, quota windows, usage-probe
backoff, scoped cooldowns and session affinity, keyed by an opaque `acct_` locator
id. `ModelProvider.prepare_credential` runs only at episode-starting client builds:
the first request of a turn, the rebuild after a model-fallback hop, the
reactive-compaction rebuild and the compaction summarizer. A same-model transient
retry keeps its binding, and rotated material is not re-prepared. Affinity is keyed
by the durable conversation identity (`undo_token_lifetime.conversation_id`), so
resume keeps warm affinity; sub-agents keep their own id. Selection ranks private
accounts before org accounts, keeps the conversation's warm account (one hour),
then orders by reserve, the 85% short-window guard, measured quota before unknown,
priority, and weekly then short-window drain. Quota comes from
`anthropic-ratelimit-unified-*` response headers and a bounded
`api.anthropic.com` usage probe (direct route only; a 401/403 makes telemetry
unavailable for an hour and never parks inference). A quota 429 parks its scope
(shared or model family) until its reset; a concurrency 429 or 529 backs off
without parking. Set an account's policy with
`python3 -m agent_gateway.cli auth policy anthropic --identity ACCOUNT
[--user N|--store PATH] (--priority INT --reserve-pct FLOAT | --reset)`; inspect
it with `auth list anthropic --usage` (cached view only). Mechanism:
`docs/reference/provider-authentication-and-token-methods.md`.

The complete session-token, chat request, and SSE walkthrough is in the
[quickstart](docs/quickstart.md).

## Request shape

```text
HTTP client
  -> create_agent() or create_gateway_app()
  -> ChatRuntime
  -> AgentRunner.run() (provider stream and model/tool loop)
  -> ToolDispatcher.dispatch()
       |-- local Python handler
       |-- MCP stdio server
       |-- approval policy
       |-- code execution
       `-- run_agent sub-agent
  -> EventLog -> SSE response
```

Headless calls use the same runner, tools, skills, and provider abstractions,
but return `RunOutput` instead of exposing an HTTP/SSE session.

Code execution settles native and bridged nested calls through the cell ledger.
Their `dispatch.sources` remain in the outer result's `sources`; native computation
provenance is projected into `computations` with its `nested_call_id`, without
entry or byte caps. Computation JSON sidecars are no longer read or consumed.
Each bridged nested dispatch inherits the enclosing `code_execute` call's skill
run, step, workspace and batch context, including for background cells; the
dispatcher supplies the request's control-run identity as for direct calls.

A docker or kernel cell's `timeout_ms` bounds its compute only. The interpreter
runner brackets each `hank` request made on the cell's own (main interpreter)
thread with `host_wait_begin`/`host_wait_end` frames; `_run_cell` pauses the
compute deadline while one is open, so a blocked host call never times the cell
out or interrupts it. A `hank` call from a cell-spawned thread still works but
sends no frames and does not pause the clock, since the cell thread may still be
computing. Each result carries `compute_ms` (unpaused time the deadline counted)
and `host_wait_ms` (the cell thread's summed blocked time; spawned threads' waits
are not counted). The optional `host_wait_ms` operand
(`CodeExecutionConfig.default_host_wait_ms`) is recorded and echoed as
`requested_host_wait_ms`, never armed as a wait. The subprocess backend still
bounds the whole process wall time.

Native calls parked for local approval capture their original `PreparedToolCall`
and dispatch operands privately on the route's ledger store, before publication.
Custody uses the executing job's admitted context (including in-process children),
and an Anthropic generation reference comes only from the dispatcher's frozen
capability bind. A captured root turn links that same park to its root admission.
Automatic decisions and detached delegated approval do not record a native
binding; a missing admitted job or tenant is observed with a warning and approval continues.
Public approval projections remain redacted and carry none of these private operands.

## Develop and verify

From the repository root, update a gateway dependency with the constrained
target (the dev lock is compiled `--constraint api/requirements.txt` so the two
locks cannot disagree; `docs/reference/python-environments-and-packaging.md`):

```bash
make gateway-deps-update PKG=<name>   # then: make gateway-deps-install
```

For coupled targets, use multiple space-separated `--upgrade-package <name>==<ver>`
flags. In the monorepo checkout, the shortcut is
`make gateway-deps-update PKG='<name>==<ver>'` (`PKG` accepts a space-separated list).

From the package root, run:

```bash
pytest tests
```

The suite and its `gateway_test_support/` fixtures are package-local; no embedding
application checkout or product configuration is required. Product integration
scenarios belong in the embedding application's tests, not this package suite.

Use the focused test for the boundary you change. Runnable examples live under
[`examples/`](examples/); the production-style assembly example is
[`examples/07-full-production/`](examples/07-full-production/), and the
headless fixture is [`examples/09-autonomous/`](examples/09-autonomous/).

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [HTTP and SSE API](docs/http-api.md)
- [Python API reference](docs/api-reference.md)
- [MCP server configuration](docs/mcp-server-catalog.md)
- [Framework comparison](docs/comparison.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT
