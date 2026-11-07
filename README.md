# ai-agent-gateway

Deploy an AI agent as an HTTP/SSE service without rebuilding session, tool,
approval, and execution infrastructure around the model loop. Requires Python
3.10 or newer.

## Ownership

Owner doc: [`docs/architecture.md`](docs/architecture.md) — the package's architecture page; product wiring and approval policy are the embedding app's (`docs/reference/AGENT_INFRASTRUCTURE.md`, `docs/reference/agent-tool-policy-architecture.md`); run-loop and approval mechanism: `docs/architecture.md` and `agent_gateway/README.md`.

This package owns the generic FastAPI chat surface, sessions and JWTs, SSE event
stream, model/tool run loop, tool dispatch, approvals, code execution, skills,
sub-agents, autonomous execution, and heartbeat support.

It does not own an embedding product's profiles, prompts, channel policy,
domain tools, research data, or business schemas. Applications supply those
through `create_gateway_app()` or the higher-level `create_agent()` inputs.

Provider adapters, credential stores, transport types, and one-shot completion
calls belong to [`hank_model_transport`](../model-transport/hank_model_transport/),
not this package. The gateway retains `AuthConfig`, capability resolvers, and
the host adapter `transport_runtime.GATEWAY_SEND_PROMPT_RUNTIME`. Host package
metadata lives in `agent_gateway.gateway_info`; transport `package_info` describes
only the transport package identity and version.

The wheel imports without an application checkout. Product integrations bind
policy and identity callbacks plus optional batch/operator-schedule backends in
`GatewayServerConfig`; schema-backed artifact tools and routes belong to the
embedding application. `server_policy` is process-scoped: configure it before
building runtimes and use one product policy per gateway process.

Storage-owner syntax belongs to `user_identity_mapping.parse_storage_identity`.
Raw signed claims, persisted schedule owners, scanned desk-directory names, and
runtime bootstrap inputs enter through that parser. Session-backed artifact and
schedule paths reuse `session_owner_user_id()` without parsing again; desk and
workspace path helpers only locate an already admitted owner. Runtime placement
additionally requires the parser's numeric form before binding a Risk identity.


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
Foreground children register in the shared journal before the child attaches for
its first model turn. The one `task_registered` names the child's gateway session
as `sub_agent_id`, so lineage reconstruction retains assistant tool calls made
before an approval park. Foreground settlement appends `task_completed` and
`agent_completion`, not another registration; approved continuation executes the
original durable binding once and carries its real result into the successor.
The warm foreground registry claim retains the same admitted task and bind as
the journal, so delivery through a warm host and a restored host uses the same
continuation authority.
Registered foreground attempts settle their exact admitted identity before
readable-envelope materialization. Cancellation retains the child's observed
tools and usage; dispatch/lifecycle failures get canonical failed settlements,
while genuine approval parks retain `awaiting_approval`. Ambiguous-write retries
reuse the original completion time. A settlement-write failure is logged and
recorded as uncertain without replacing the original response or cancellation.


Root approval continuations retain the parked turn's sequence space, even after
its active-turn reattach window expires. The prior turn is settled and cleared
before `EventLog(first_seq=session.last_completed_turn.event_log.next_seq)`
starts the successor; a cold restore without a predecessor starts at sequence `1`.
`iter_from(after_seq)` always returns entries strictly after that absolute
sequence, not after a local list offset. Each continuation closes its own turn
and flushes its entries through the same wire projection and per-turn transcript
writer as `/chat`, including when no subscriber is connected. Hosts explicitly
supply the restorer's `transcript_dir`. Already-delivered checks run after the
previous turn settles, under the continuation lock with the session slot held.
A no-op releases that empty slot without a terminal or transcript entry.
A second approval park is a turn terminal, not the end of the continued run.
Authenticated `GET /api/capability-choices/session.driver` uses the same live
selection projection as init without creating or replacing a chat session.
Preference controls can refresh it while an approval is parked. Account
preferences apply to new turns; they do not replace a parked turn's binding.

Session replay pairs every assistant tool call with a result in the immediately
following user message. A parked call reads as awaiting approval until its real
result arrives; unanswered siblings, including calls dispatch never started
before the park, read as explicit `tool_interrupted` errors. These are replay
projections only: they never settle durable calls, and later recorded results
replace them.

The native `AgentRunner` accepts an optional async `final_answer_check(text)`
hook and `final_answer_check_mode` (`hold` by default, `append`, or `off`),
directly or from the dispatcher's `RunContext`. Every terminal no-tool answer
emits `citation_check_started` and awaits the hook with the whole answer the
client holds (a `max_tokens`-continued answer's segments together). A positive
`runner_state.FinalAnswerVerdict(mode, violation_count, revise_message,
text_transform)` retains the assistant draft plus the supplied user revision
instruction in model context and emits `answer_revision_started` before
continuing the same run; both events name the draft's turn. A run revises at
most once, so the revised answer is checked again but never revised; `off`
checks every answer and revises none. The verdict's optional `text_transform`
is applied once to the reply that answers the revision — through any tool turns
or retries before it; a reply cut at `max_tokens` is held and spliced whole with
its continuation; a reply that fails mid-stream settles on `text_transform("")`.
`None`, a zero-violation verdict, or a logged check exception leaves normal
completion unchanged. Domain validation and revision wording remain host-owned.

## Main entrypoints

| Source | Primary symbol | Use it for |
| --- | --- | --- |
| [`agent_gateway/easy.py`](agent_gateway/easy.py) | `create_agent()` | A small server from a prompt, tools, skills, and provider configuration |
| [`agent_gateway/server.py`](agent_gateway/server.py) | `create_gateway_app()` | Product-owned runtime factories, auth, policy, and lifecycle integration |
| [`agent_gateway/gateway_server.py`](agent_gateway/gateway_server.py) | `run_gateway_server()` / `main()` | Privileged single-worker TCP launcher, claim-signing fd adoption, and startup-failure logging |
| [`agent_gateway/autonomous.py`](agent_gateway/autonomous.py) | `run_autonomous()` / `run_autonomous_sync()` | A prebound headless one-shot with no HTTP server |
| [`agent_gateway/admission.py`](agent_gateway/admission.py) | `admit()` | Shared spawn, continuation, message and handle authority |
| [`agent_gateway/containment_settings.py`](agent_gateway/containment_settings.py) | `ContainmentSettings` | Every containment bound (spawn depth, concurrent children, soft ladder per session kind, hung-child silence, context thresholds) as a live view over the operator run-settings file (`agent_gateway/run_settings.py`, `HANK_RUN_SETTINGS_PATH`); `ContainmentSettings(**overrides)` is the pinned form |
| [`agent_gateway/heartbeat.py`](agent_gateway/heartbeat.py) | `HeartbeatLoop` | Repeated prebound autonomous work with quiet windows and backoff |
| [`agent_gateway/cli.py`](agent_gateway/cli.py) | `main()` (`agent init` / `agent run`) | Scaffold and run a package project |
| [`packages/model-transport/hank_model_transport/send_prompt.py`](../model-transport/hank_model_transport/send_prompt.py) | `send_prompt()` / `send_prompt_sync()` | One already-bound prompt→completion call returning text. `completion_record_root` is required keyword-only: pass a directory to append one `completion_record` v1 row per provider attempt, or `None` to state the call is not captured |
| [`packages/model-transport/hank_model_transport/completion_record.py`](../model-transport/hank_model_transport/completion_record.py) | `build_completion_record()` / `append_completion_record()` | The `completion_record` v1 row for one-shot companion-repo model calls (contract: `packages/model-transport/hank_model_transport/contracts/completion-record-v1/completion_record.schema.json`); day-sharded `<root>/<YYYY-MM-DD>.jsonl`, append-only, never fails the call it records |
| [`agent_gateway/runner_usage.py`](agent_gateway/runner_usage.py) | `apply_provider_diagnostics()` / `apply_thinking_event()` | Fold each provider call's `provider_response_id`, `provider_request_id`, `service_tier`, `rate_limit` (headers read once by `packages/model-transport/hank_model_transport/providers/response_diagnostics.py`) and the thinking tallies `thinking_chars` / `redacted_thinking_blocks` into a turn's usage, so `assistant_message.usage`, `turn_complete.usage`, the spend-row attrs and the completion record carry them; thinking blocks, `redacted_thinking` included, stay uncapped in `content_blocks` as the provider replay record |
| [`agent_gateway/gateway_address.py`](agent_gateway/gateway_address.py) | `gateway_base_url()` | The origin of the gateway a process talks to: the one reader of `GATEWAY_BASE_URL` and the one place its default is spelled |

`run_gateway_server()` is the serving entry on both local supported launches, not
only the privileged host one: the immutable runtime's serving child calls it
directly, so claim-signing adoption, `uvicorn.Config`/`Server` construction and
the launch-failure journal have one owner.

The package does not infer a new model or credential inside autonomous
execution. The application resolves and passes the exact bound capability,
session, billing mode, and skill limits before calling `run_autonomous()`.

Autonomous model processes inherit `OLLAMA_CLOUD_BASE_URL` and `OLLAMA_BASE_URL`
through the child-environment owner, including detached profile roots, scheduled
runs, admitted autonomous/cron children and their continuations. Admission retains
the endpoint names in the frozen environment projection; launch re-hands endpoints
even for older persisted projections. `HANK_RUN_ROUTE_FILE` is not allowlisted:
launch explicitly hands every child the route path or `""` when the parent has no
route, so the child never reads the operator's current selection. These routing
names are not granted to `run_bash` or
`code_execute`. The generic projection never forwards `OLLAMA_CLOUD_API_KEY`:
the bound model credential still travels through the signed launch's private
credential handoff.

The owned-process sentinel reports its JSON status on a dedicated pipe the
parent creates and passes by descriptor number (a non-inheritable read end in
the parent, the write end inherited only by the sentinel, never by its child).
The sentinel's stderr is the run log from process start, so interpreter-startup
and import-time output (`sitecustomize`, `.pth` processing, warnings) lands in
the log; the parent still parses every status line strictly.
Reaper-induced abort of a pending event reader, when the reaper is cleaning up
an already-failed or cancelled run, is cleanup collateral, not a channel
failure that can replace the process failure.

Model selection and model request facts come from `model_authority`
(`hank-model-authority`; in the monorepo `packages/model-authority/`), which this
package depends on and carries no copy of: the registry, the selection policy,
each entry's per-family `compat` block and the rate tables.
`CapabilityExecutionResolver.select_bind()` in `packages/model-transport/hank_model_transport/capability_execution.py` selects
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
attempts; the runner emits `model_fallback`. The authenticated resolver
remembers failed chain positions for fresh child runners, which resume at the
first remaining entry without re-probing them. Declared resets and credential
changes revive their entries; a newly prepared root turn starts at its primary.
[`hank_model_transport.providers`](../model-transport/hank_model_transport/providers/)
from the bound entry. Mechanism: [model authority](../../docs/reference/model-authority.md).

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
A durable-park pending entry is marked before the approval request is published.
An original-token vote therefore always uses durable continuation, even while the
ask is being persisted and the live pending entry still exists; it never enters
the live queue that the parked call will not consume.
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
Operation snapshots retain canonical essential-route identities: local tool IDs
stay unchanged and MCP IDs use `mcp__<server_id>__<tool_id>`. Admission maps these
identities to the current exposed route names before resolving authority.
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
Both root helpers require the keyword `selection_provenance`; fresh roots capture
it from the same selection-policy object as their snapshot. The existing
`default_root_provenance(policy)` projection records `policy_default` even when
there is no named preset. Explicit `None` means the original selection source is
unknown (historical/frozen jobs or a tool-only bridge with no model selection),
not permission to reconstruct a layer from the current process. Durable root
turns retain the original provenance alongside their snapshot for Continue.
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

`POST /chat/background/{task_id}/interrupt` uses the authenticated session's
current live runner and the same `TaskRegistry.interrupt()` owner as
`run_agent(interrupt=task_id)`. It requests `user_interrupted`, never parent-turn
cancellation. Unknown, foreign, settled or already-interrupting targets return
`interrupt_requested: false`; a session with no live runner returns HTTP 409.
An acknowledgement is not durable settlement: wait for the child's interrupted
notification, then ask the parent to `run_agent(resume=task_id)`. Resume retains
the original transcript, capability bind and grant and returns a successor ID.
The endpoint accepts no session override, model selection or replacement grant.

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

## Completion reads (schema version 1)

`agent_gateway.completion_reads` composes raw completion facts from the existing
runner, task registry, EventLog and durable session-log owners. Control-plane
chat and autonomous run responses expose this as `completion`; in-process hosts
can call `read_runner_completion(runner, pending_approval=...)` directly.
There is deliberately no `settled`, `quiescent`, live-child classification or
new completion event. A terminal, closed stream or zero observed tasks is not a
success/quiescence assertion.

The version-1 envelope has `schema_version: 1`, `source`
(`chat`, `workflow`, or `runner`), `consistency`, `terminal`, `receipt`,
`journal`, `durable`, `children`, `obligations`, `pending_approval` and `unknown`.
Facts are copied, not references to mutable registry entries. `consistency` is
`changed` when the two owner observations differ, `unknown` when an owner or
epoch is unavailable, otherwise `current`. `current` means only that the
observations matched, not that execution completed. There is no retry or wait
for a stable/successful answer. The native background-success fingerprint is
compared as well, so work changing during composition cannot masquerade as an
unchanged observation.

| Field | Native meaning |
| --- | --- |
| `journal` | EventLog's actual `closed`, absolute current `cursor` (`next_seq - 1`), `next_seq`, revision and last terminal `{seq,event}`. Closing without a terminal keeps terminal null. |
| `terminal` | The observed journal terminal, independent of durable receipt and detach. Approval-park and interruption dispositions remain unchanged. |
| `durable` | AgentSessionLog's strict current cursor/revision, runner-scoped root receipt rows, attach, terminal and detach rows, pending appends and read status. No second tailer or cached success store. |
| `receipt` | Chat/in-process: `{source: "agent_session_log", status, refs}` naming the existing durable receipt-equivalent rows, never a newly minted proof. |
| `children` | Registry epoch, coverage, `observed_count`, authoritative `count`, raw `state_counts` for every TaskState and copied child refs. The runner's admitted self entry is excluded. |
| `obligations.background` | The existing success fence's blockers and raw initialization/reconciliation, completion-append, notification queue, acknowledgement and delivery-credit `{count,refs}` observations. |
| `obligations.session` | Native attach/closure, staged receipts, required top-level skill result/terminal/detach settlement and persistence facts. |
| `pending_approval` | Existing pending approval count and tool-call/approval refs, not a new approval decision. |

Child rows retain `pending`, `running`, `completed`, `failed`, `killed`,
`interrupted`, `awaiting_approval`, `idle` and `parked` without mapping them
to product liveness. Rows also retain registration/completion persistence
state/error, detached finalizers and pending final state, required skill-result
settled/projected facts, initialization refs, notification generation/delivery
state and callback presence/invocation. Queue omission and payload omission
remain unresolved delivery facts; an unread result handle is not reclassified
as running work.

### Receipt equivalents and unavailable facts

Chat has no separate `AutonomousTerminalReceipt`. Its receipt equivalents are
the existing durable `task_completed`, `agent_completion`,
`skill_result_captured`, `workflow_output_attached`, `stream_complete`, `error`
and `session_recap` rows. Durable `detach` is separate. The read scopes to the
exact runner and its attached sub-agent identity (null for root chat), excluding
other runners/turns and descendant event envelopes. A workflow attached to an
in-process runner uses those same native workflow-output and skill-result
rows; their publication is not a child-liveness judgment.

On the control plane, autonomous/subprocess runs use `source: "workflow"`.
Their existing `AutonomousTerminalReceipt` is projected alongside
`terminal_manifest_committed`, event-evidence status and the native
channel acknowledgement's channel/count/digest. A terminal receipt may exist
before its manifest is committed: `receipt.committed` exposes that distinction.
The parent does not own the subprocess EventLog, task registry, durable detach
or settlement queues, so these remain null/unknown rather than inferred from
exit code, a channel ACK, terminal manifest or event count.

`null` means not observed/available, never zero or successful settlement.
A durable read has explicit `observed`, `missing`, `no_target` or `unknown`
status; `missing` means no matching row at the read cursor, not success.
Unknown readers are listed in `unknown` with their owner/error type.
Restoring a registry preserves its observed raw task rows but makes coverage
`unknown` and authoritative `count: null`, even for an empty replay. Its new
process-local epoch is not continuity proof. A lost chat runner likewise leaves
children/obligations unknown while retaining an available journal and pending
approval map. Unavailable task/notification inspections use null counts/refs,
not an empty collection. No absent receipt, detached writer, restart or empty
registry is promoted to a completed obligation.
The session's `detach_status` distinguishes acknowledged (`observed`),
unacknowledged-write outcome (`unknown`), no acknowledgement (`missing`) and
no durable target (`no_target`). Detach and receipt persistence errors remain
visible independently, including after a later detach acknowledgement.
`durable.pending_appends` is the log owner's current log-wide count, not
attribution of those appends to this runner.

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

For a one-shot call, pass an already-bound execution from the gateway's
capability resolver and explicitly supply the gateway host runtime:

```python
from hank_model_transport.send_prompt import send_prompt
from agent_gateway.transport_runtime import GATEWAY_SEND_PROMPT_RUNTIME

async def complete(prompt, capability_execution, user_id):
    return await send_prompt(
        prompt,
        capability_execution=capability_execution,
        runtime=GATEWAY_SEND_PROMPT_RUNTIME,
        user_id=user_id,
        completion_record_root=None,
    )
```

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
identity without overwriting other seats. `hank_model_transport.providers.credential_pool` owns
the shared `"accounts"` store envelope, legacy single-record migration, and
cooldowns for every pool provider except Anthropic. A scoped user's accounts
lead the deployment pool; a usage limit that declares its reset parks that seat
until then so the running request can rotate to an unblocked sibling, keeping
that seat's token and provider-specific fields together. A limit with no declared
reset, or a failed refresh, skips the seat for that selection or rotation only.
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
[provider authentication and token methods](../../docs/reference/provider-authentication-and-token-methods.md).

The complete session-token, chat request, and SSE walkthrough is in the
[quickstart](docs/quickstart.md).

### Durable base-chat history

`GET /api/chat/history` uses the existing `/api/chat/init` bearer. It accepts
only `limit` (default `50`, range `1`–`200`) and optional `before_seq` (a positive,
exclusive sequence cursor). There is no caller-selected user, session, log id or
filesystem path. The authenticated live session supplies its durable
`agent_session_log`, or its init-bound `resume_log_path` / `fork_source_log_path`
before the first turn. A fresh session has an empty history. Resume ownership
continues to belong to the configured init-time resolver, not to this read.

```json
{
  "schema_version": 1,
  "entries": [
    {"seq": 2, "role": "user", "content": "Hello", "timestamp": 1793318400.0},
    {"seq": 3, "role": "assistant", "content": "Hello.", "timestamp": 1793318401.0}
  ],
  "next_before_seq": null
}
```

Pages start at the newest durable writer `user_message` / `assistant_message`
records and return display entries in ascending sequence order. `seq` is the
original durable sequence; `timestamp` is numeric Unix epoch seconds. User
display text comes only from the durable `user_message.analyst_content` string
or content-block list, captured before per-turn context injection by both native
and SDK runners. Non-injected writers default it to their supplied content.
`content` remains the unchanged model-facing message, including appended context,
and native/SDK replay continues to use it. No provenance keys enter model messages.
Text blocks join in order with no separator. Operator messages, sub-agent records
and textless messages are invisible; thinking, tool results/calls, attachments
and actionable payloads are never returned or replayed. Consecutive assistant
entries remain separate for clients to group into a response bubble.

If a visible legacy user record lacks `analyst_content`, the read returns HTTP
`409` with `detail: {"code": "history_analyst_content_unavailable", "seq": ...}`.
It never displays the unknown model-facing content or guesses which headings to
strip. Present malformed provenance is an error, not a fallback to `content`.
This is additive within the existing durable event schema version.

`next_before_seq` is the oldest raw sequence in the page when an older raw page
exists, otherwise `null`. Visibility filtering can yield an empty `entries`
array with a non-null cursor. Follow every cursor until `null`, prepending older
pages and deduplicating by durable sequence so an in-flight newer turn survives
rehydration. The read uses the existing log and its rotated segments, without
opening a writer, creating active log files or repairing rotation metadata for
resume/fork reads. Responses carry `Cache-Control: private, no-store`; there is
no separate transcript store.

A fork reads its authorized source before its first turn, then reads its own
live log (also after resuming that fork). The inherited `fork_seed.messages`
are model context with no analyst/operator display provenance, so they are
not projected and `source_log_path` is never followed. Inherited display history
after a fork's first turn requires provenance at the fork-seed owner; it cannot
be reconstructed safely by this endpoint.

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

A docker or kernel cell's `timeout_ms` bounds its compute only: the clock is armed
by the runner's `started` frame, so a session's first cell never spends its budget
on interpreter boot. Boot (the cell frame's write to `started`) has its own bound,
`CodeExecutionConfig.kernel_boot_timeout_ms` (default 60 s); a boot that outlasts
it discards the kernel and returns `error.code` `kernel_boot_failed` with
`timed_out` false, never `return_code` 124. The interpreter
runner brackets each `hank` request made on the cell's own (main interpreter)
thread, and each `await hank.acall(...)` worker's, with
`host_wait_begin`/`host_wait_end` frames; `_run_cell` pauses the compute
deadline while one is open, so a blocked host call never times the cell out or
interrupts it. CPU the main thread spends on other coroutines while an `acall`
wait is open also escapes the deadline. A `hank` call from any other
cell-spawned thread (`threading.Thread`) still works but sends no frames and
does not pause the clock, since the cell thread may still be computing. Each
result carries `compute_ms` (unpaused time the deadline counted) and
`host_wait_ms` (the union of the intervals during which at least one host wait
was open, so overlapping concurrent waits are not summed). The optional
`host_wait_ms` operand
(`CodeExecutionConfig.default_host_wait_ms`) is recorded and echoed as
`requested_host_wait_ms`, never armed as a wait.

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
pytest --import-mode=importlib tests
```

In this monorepo checkout, the repository `pytest.ini` owns import roots and default filters for targeted and whole-suite runs, including the add-in `api/` root; the standalone command above keeps `--import-mode=importlib` explicit.

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
