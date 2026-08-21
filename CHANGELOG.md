# Changelog

## 0.17.4 (2026-08-21)

### Fixed

- Descriptor-confined directory walks on Linux now use lookup-only `O_PATH`
  descriptors for intermediate ancestors and retain a readable descriptor for
  the exact target. Signed v2 session-log storage therefore remains usable
  after child Landlock enforcement without granting directory-enumeration
  access to the storage root's ancestors or sibling streams.

## 0.17.3 (2026-08-21)

### Fixed

- Descriptor-bound v1 session-log inventory now preserves coherent historical
  telemetry lineage after an authorized physical-root relocation. Historical
  paths remain metadata only: the current file is still admitted and read by
  its bound descriptor, family and filename identity must match, active and
  rotated source IDs must be self-consistent, and conflicting lineage fails
  closed. V2 canonical paths remain physically exact.

## 0.17.2 (2026-08-20)

### Fixed

- Private-document erasure now invalidates the affected current session-log
  projections without deleting durable history or crossing owner boundaries.
- Autonomous canonical session logs can use a signed, descriptor-bound v2
  stream directory whose exact root is admitted to the child Landlock policy;
  the temporary `AGENT_SESSION_LOG_LAYOUT` choice remains `v1` in production
  until the v2 cutover is separately validated.
- Error-signal, telemetry, retention, workflow-erasure, and current-projection
  consumers now share descriptor-bound session-log inventory and fail closed on
  ambiguous metadata instead of following untrusted paths.

### Changed

- Added the dependency-neutral immutable operation-catalog projection used by
  new Gateway admissions; persisted execution snapshots remain authoritative
  for retry and resume.

## 0.17.1 (2026-08-20)

### Fixed

- Added an explicit application API-directory configuration for autonomous
  dispatch. Wheel-installed Gateway applications no longer derive the
  authoritative `user_identity.py` location or autonomous child working
  directory from the package's `site-packages` path.

## 0.17.0 (2026-08-20)

This is a pre-1.0 breaking-minor release. It describes the complete public
package delta from 0.16.2, not only the final release-preparation changes.

### Added

- Added explicit `WorkflowDeliverySpecV1` / `WorkflowDeliverySpecV2` and
  `DeliveryEnvelopeV1` / `DeliveryEnvelopeV2` construction models, shared
  version-tolerant readers, deterministic V2 preview models, and generated JSON
  Schema, TypeScript, and golden resources. Exact published output remains the
  authority behind every preview.
- Added typed selected-content and workflow-content paging contracts, including
  owner-bound read authorizations, integrity-checked cursors, Investment
  artifact views, and exact read recipes for parent and child runs.
- Added durable workflow continuation brackets, recorded `max_phases`, blocked
  boundary recovery, a single `WorkflowView`, typed recovery hints, and
  multi-valued evidence ports for tool-less operations that need upstream
  evidence.
- Added a platform tool catalog, capability resolver, execution identity, and
  declared capability requirements. Operation admission, dispatcher scope, and
  tool activation now project from those shared authorities.
- Added an append-only MCP activation fold, producer-owned `control-run-v1`
  lifecycle contract, and versioned `commercial-authority-v1` invalidation-feed
  contract as installed package resources.
- Added durable tool-dispatch source and outcome facts, mechanical settlement
  qualifiers, and child-evidence projection into the parent's citation
  registry. Child retrieval remains usable even when the child later fails;
  child computation is not promoted to parent retrieval evidence.

### Changed

- Workflow starts and settlements write only explicit V2 delivery contracts.
  Historical V1 records remain byte-exact and readable, but V1 is reader-only;
  missing, unknown, or cross-version envelopes fail closed. Unsettled V1 starts
  finish as typed delivery failures instead of reviving a V1 writer. Lifecycle
  readers accept outer session-log versions 1 and 2, while durable task rebuild
  remains version-2-only.
- `WorkflowResult` is now schema version 2.0 and composes the canonical
  `WorkflowView`. Failed or interrupted phases park at actionable boundaries;
  continuation, finish, authoring recovery, evidence, cost, and legal actions
  are projected consistently across API, CLI, attachment, and UI consumers.
- Named `run_agent` operations freeze the registered skill's finite budget in
  the execution snapshot and enforce it on initial execution and resume. Model
  input can no longer set or increase the hard budget. Trusted dispatcher
  interceptors propagate, in order and by identity, into delegated and resumed
  agents.
- Operation authority comes from catalog routes and declared capability
  requirements rather than inferred read effects, copied allowlists, or
  `data_requirements`. Unavailable operations are reported explicitly, and the
  advertised MCP surface and dispatcher allowlist derive from the same durable
  activation fold.
- Background-task registration and delivery acknowledgement are transactional.
  Active runs continue admitting tasks while results arrive; unread
  handle-shaped results retain their identity and receive one stop-boundary
  reminder. Notification renders and delivery nudges are durably observable.
  Expired sessions reject new use while already-owned workflow resources remain
  available until that work settles.
- Control-run active and terminal states now come from the producer-owned
  lifecycle table. Foreground interrupt issues an authoritative cancel,
  approval expiry is distinct from user denial, budget stops project as
  `budget_limited`, and autonomous terminal state and retention ownership no
  longer maintain competing state sets.
- Oversized tool results now preserve their typed structure while eliding bulk
  arrays, mappings, strings, or deep nesting in place. The former serialized
  byte-prefix and shallow scalar preview are removed.
- Final answers are no longer intercepted and rewritten by a runtime
  `FinalAnswerGuard`. Methodology, evidence requirements, and completion
  decisions are owned by admitted operations and their explicit contracts.
- Hosted model operations resolve the authenticated current-model identity and
  verify descriptor-relative artifact bytes. Hosted parent, child, resume, and
  autonomous schemas reject raw user/path selectors and expose categorical
  identities instead of server file locations. Producer-owned Edgar and Risk
  files remain behind their source services.

### Fixed

- Preserved session-owner identity across hosted and multi-user runner
  construction, dispatch, spend attribution, workbook tenancy, children, and
  resumes; ambiguous or mismatched identities fail closed.
- Routed SEC access through the canonical machine egress client and kept its
  machine configuration out of child environments.
- Made commercial authority invalidation consume an explicitly versioned feed
  with strict format parsing instead of relying on sibling-specific payload
  assumptions.
- Kept deterministic workflow delivery replay, content paging, notification
  settlement, task reconstruction, nested approvals, and cancellation coherent
  across restart and recovery seams.

### Removed

- Removed public exports `FixtureProvider`, `DataRequirement`,
  `RELAY_POLICY_DENIED_MESSAGE`, `RELAY_POLICY_DENIED_SUB_CODE`, and
  `resolve_denied_provenance`.
- Removed the caller-supplied `AgentRunner(final_answer_guard=...)` surface.
- Removed the development-only commercial contract verifier exports
  (`CONTRACT_FILES`, `MANIFEST_FILE`, `USAGE_V3_CONTRACT_FILES`,
  `USAGE_V3_MANIFEST_FILE`, `packaged_contract_directory`,
  `verify_contract_directory`, and `verify_usage_v3_contract_directory`) from
  `agent_gateway.commercial_contract`. The verifier now lives in the source
  checkout at `scripts/verify_commercial_contracts.py`; installed consumers
  should validate producer-owned manifests and use the packaged contract
  resources directly.
- Removed product-embedded fixture, development-mode, QA, market-scan,
  orchestration-dispatch, and Excel-relay runtime facades. Tests and product
  integrations must use their real provider, control-plane, or application
  boundaries.
- Removed legacy MCP configuration facades, copied live-tool views, inferred
  research-evidence capability, and the `data_requirements` skill vocabulary.

### Security and behavioral changes

- Non-owner MCP execution is deny-by-default for unknown tools, and canonical
  owner identity now scopes relay workbooks, model spend, dispatch authority,
  and persisted execution state.
- Hosted reads use bounded, no-follow, stable file access with digest checks;
  raw paths, stale producer modes, private extraction tools, and foreign source
  coordinates are refused before execution.
- Source, credential, approval, lifecycle, notification, and child-evidence
  facts are sanitized and durably bound at their producing boundaries rather
  than reconstructed from model-visible output.

### Breaking changes and migration

- Regenerate workflow clients for schema version 2.0 and read status, actions,
  recovery, evidence, and phase state from `WorkflowResult.view`. Continue to
  parse historical delivery records through the public V1/V2 reader unions;
  construct all new starts and envelopes with the explicit V2 classes.
- Replace `SkillProfile.data_requirements` with declared capability
  requirements. Tool-less fan-in operations must declare `evidence_ports`
  with an explicit cardinality floor and ceiling.
- Pass one resolved `DispatchIdentity`/`ExecutionIdentity` to new
  `ToolDispatcher` integrations, and derive operation tool scope through the
  platform capability resolver. Do not combine `identity=` with the retained
  separate compatibility inputs.
- Remove `max_budget_usd` from model-authored `run_agent` input. Set a finite
  budget on the registered named skill when a hard child budget is required.
- Replace removed fixture/dev/relay imports with test-owned fixtures or the real
  provider and application boundary. There is no runtime replacement for the
  retired final-answer guard.
- Required runtime dependencies are unchanged from 0.16.2. The `dev` extra and
  hashed development lock now include Matplotlib so code-execution plot capture
  is available to the package test suite.

## 0.16.2 (2026-08-14)

Completes the model-selection authority conformance work
(implementation-review items C1-C9 and the §3 sweep).

### Added

- The model registry and selection policy are now packaged
  `product-model-registry/v1` / `product-model-selection/v1` YAML artifacts
  parsed by strict typed loaders (unknown fields, duplicate mapping keys,
  missing required fields, and incoherent lifecycle combinations fail
  construction loudly) and admitted once at import. Deployment may select an
  alternative admitted artifact file via
  `AGENT_GATEWAY_MODEL_REGISTRY_FILE` / `AGENT_GATEWAY_MODEL_SELECTION_FILE`;
  full admission runs on whatever loads. Lifecycle states
  (`active`/`hidden`/`deprecated`/`disabled`/`revoked`) are authorable.
- Installed adapters declare their real protocol support through
  `ModelProvider.adapter_route_support()`; admission validates registry
  entries against installed declarations. Capabilities executed outside this
  package carry an explicit process designation; gateway-process resolvers
  refuse them with the typed `capability_externally_executed` code.
- Typed `capability_catalog_stale` refusal when a request's observed catalog
  revision is stale and its key is no longer eligible.

### Changed

- `inherit_parent` copies the whole parent binding (credential principal and
  reference, run mode, revisions) rather than reselecting credentials.
- Effort for an explicit selection without an effort falls to the capability
  policy effort, not the registry entry default.
- Selection refusals carry the current eligible model keys; the resolver
  enforces capability exposure (hidden models are refused as user choices).
- Server construction closes over the configured registry: gateway-executed
  entries must resolve installed adapters declaring their profile/route, or
  startup fails.

### Removed

- The hand-written `INITIAL_ADAPTER_ROUTE_SUPPORT` table (replaced by
  installed-adapter declarations; breaking for any consumer importing it).
- The xAI silent effort clamp: unsupported efforts now refuse instead of
  degrading.

### Fixed

- Usage outbox ships only current-schema payloads; pre-v3 rows dead-letter
  loudly. The schema gate covers the agent session log and batch registry.
- Fork admitted-task digest unified on the canonical definition; autonomous
  manifest bind writes are unconditional; the model preference store closes
  its sqlite handles; `easy.py` no longer encodes caller selection into
  policy revisions.

## 0.16.1 (2026-08-14)

### Fixed

- Preserved the tracked Canvas Kit authoring type bundle through the standalone
  dist and public-repository sync paths, so installed wheels include the
  component and formatter declarations required by `authoring_manifest()` and
  Canvas typechecking. Other runtime `node_modules` trees remain excluded.
- Saved model preferences on a model that is no longer eligible (deprecated,
  revoked, hidden, disallowed, or unsupported effort) now resolve the eligible
  capability default with a typed not-applied notice carrying the reason,
  instead of hard-refusing every turn in the session. The stored preference is
  never mutated or deleted.
- Session-log task events are now versioned (`EVENT_SCHEMA_VERSION` 2) and the
  registry rebuild loudly warn-skips pre-cutover or bind-less records instead
  of rebuilding them into resumable entries that fail late; durable task
  registration requires a capability bind receipt end to end.
- Excel dispatch (`mint_and_submit`, the orchestration route, and the
  autonomous `message_excel_agent` handler) now forwards stable
  `model_key`/`effort`/`catalog_revision` selection intent instead of silently
  discarding it, refuses raw `model` with the typed `chat_model_not_accepted`
  code before any grant is minted, and advertises the selection triple in the
  shared tool definition.

## 0.16.0 (2026-08-13)

This is a pre-1.0 breaking-minor release. It describes the complete package
delta from the immutable public source baseline
`aa74b5030384cd130118f6fd9293f2b13fd64bae`, not only the final release-prep
commits.

### Added

- Added canonical capability policy, model catalog, credential-handle, and
  `BoundCapabilityExecution` contracts, plus `GatewaySession`-owned identity
  and credential provenance for interactive, autonomous, cron, and node runs.
- Added portable compaction with provider-aware history normalization and
  durable, provenance-preserving summaries across live and resumed sessions.
- Added the Canvas artifact contract, validation/store/event pipeline, and the
  pinned TypeScript/esbuild build resources shipped in the wheel.
- Added durable child-agent and workflow delivery contracts, generated JSON
  Schema/TypeScript artifacts in `agent_workflow_contracts`, typed result and
  settlement materialization, idempotent journal replay, and parent-visible
  workflow evidence.
- Added owner-scoped control-plane admission, approval, readable-resource,
  cancellation, resume, and autonomous launch-envelope boundaries.

### Changed

- `run_autonomous()`, `run_autonomous_sync()`, `send_prompt()`, and
  `send_prompt_sync()` now consume one pre-resolved `BoundCapabilityExecution`;
  autonomous entry points additionally require an exact `GatewaySession`.
- `AgentRunner` now consumes a pre-resolved `BoundCapabilityExecution` instead
  of selecting provider, model, effort, transport, or credential material.
- OpenAI execution is Responses-API-only, targets the supported GPT-5.x model
  family, and requires `openai>=2.31.0`; durable history is fenced to the
  Responses epoch.
- MCP startup is explicit: inline or file-backed configuration requires a
  launcher-owned allowlist, and absent configuration no longer discovers an
  ambient file.
- SSE, terminal-state, and child-result handling now use typed final-state and
  `ChildReturn` contracts rather than interpreting transport completion as the
  workflow outcome.
- Role and owner authorization is fail closed across approval, dispatch,
  cancellation, persisted grants, and rehydration paths.

### Fixed

- Made parent-to-child message acknowledgement return the exact message
  identity and distinguish durable `accepted` delivery from in-memory
  `queued` delivery.
- Persisted parent-to-child message acceptance before inbox delivery, made
  exact message-ID replay idempotent without duplicate delivery, rejected
  reuse of an accepted message ID with different content, and restored
  accepted identities from the durable journal.
- Kept the generic gateway install and import independent of the optional
  Excel relay package while preserving exact typed restart admission handling
  in deployments that assemble the Excel capability.
- Preserved authority, run identity, usage, approval, compaction, and workflow
  evidence across retry, resume, reconstruction, cancellation, and terminal
  delivery seams.

### Removed

- Removed the public `ProviderResolver`, `ResolvedProvider`, and
  `sub_agent_default_model` selection surfaces.
- Removed the retired `runtime_auth_context`, `sub_agent_model_resolution`,
  `providers.openai_helpers`, and control-plane diligence-PR modules.
- Removed `emit_html_artifact`; product-facing rich output now uses Canvas.
- Removed ambient `~/.claude.json` MCP discovery and the deleted diligence-PR
  surface has no replacement.

### Security

- Capability, credential principal, owner, run, and tool authority are bound at
  trusted admission boundaries and revalidated fail closed at execution and
  recovery seams; secret-bearing values are kept out of child environments,
  logs, events, and durable public contracts.
- Security Wave 0 is `VERIFIED_NOT_VALIDATED`: automated and closest-local
  user-MCP verification passed, but no deployed live-provider journey was run.
  OpenAI's atomic Responses-epoch deployment remains an operational
  requirement, Canvas deployed build/toolchain validation is pending, and the
  orchestration product end-to-end journey remains open.

### Breaking changes and migration

- Replace `ProviderResolver` with `CapabilityProviderResolver` or the complete
  `CapabilityExecutionResolver`; replace `ResolvedProvider` with
  `BoundCapabilityExecution`.
- Replace split execution arguments with one `capability_execution`. Pass an
  exact `GatewaySession` to autonomous entry points.
- Replace raw model/provider defaults and copied allowlists with a
  `ProductModelRegistry`, `ProductModelSelectionPolicy`, and stable
  `model_key` intent. Call `CapabilityExecutionResolver.resolve()` once for a
  new execution or `materialize_bind()` for the exact durable receipt.
- Replace `runtime_auth_context` with `GatewaySession` credential provenance
  plus the bound execution snapshot.
- Replace `providers.openai_helpers` imports with
  `providers.openai_responses_helpers`, migrate supported OpenAI workloads to
  the Responses API and GPT-5.x, and upgrade the OpenAI extra to
  `openai>=2.31.0`.
- Replace HTML artifact emission with Canvas artifacts. There is no migration
  target for the removed diligence-PR API.
- Configure MCP servers explicitly and update SSE/child consumers to honor the
  typed `ChildReturn` and final-state contracts before upgrading.

## 0.15.13 (2026-07-15)

### Added

- Added finite positive, skill-only autonomous budget overrides across the
  control API and agents MCP, including spawned-command receipts, durable
  manifest persistence, status projection, rehydration, and resume.

### Fixed

- Armed direct autonomous CLI runs with the canonical durable approval store
  and allowed unattended dev runs to persist and execute policy-authorized
  exact plans while retaining fail-closed behavior for approval-gated tools.

## 0.15.12 (2026-07-15)

### Added

- Added autonomous, run-scoped recovery for oversized tool results. Capability-
  aware spill sets provide bounded `file_read`/`file_grep` projections for
  multiline and single-line payloads, while per-run budgets, no-clobber atomic
  publication, registry manifests, cross-process leases, and retention give
  artifact-backed and direct launches explicit lifecycle owners.
- Added approval presentation metadata that exposes exact planned changes and
  available Undo actions without changing the authoritative approval payload.

### Fixed

- Bound batch approvals to validated positive stage-run identity across SDK
  pending entries, session logs, projections, and control-plane resolution.
- Honored each approval's advertised expiry throughout the batch lifecycle and
  preserved terminal expiry and cancellation outcomes in typed events.
- Anchored the default approval database below the configured gateway user-data
  root so durable write state participates in the authoritative snapshot and
  migration boundary.

## 0.15.11 (2026-07-15)

### Fixed

- Kept durable planned-write approval waits outside the runner's generic tool
  timeout so the approval lifecycle owns expiry and returns an
  approval-specific outcome instead of cancelling exact-write execution as a
  generic `tool_timeout`.

## 0.15.10 (2026-07-15)

### Added

- Added a fail-closed, pre-spend corpus readiness gate for valuation-ready
  batch admission, retry, and force-rerun flows. Callers declare exact filing
  and transcript periods for every ticker; successful dispatches include the
  readiness proof, while invalid, incomplete, unavailable, and not-ready
  checks return typed errors before a batch run is acquired.
- Added optional sub-agent event observation and caller-supplied anonymous
  system-prompt templates without replacing the gateway's durable event log.

### Fixed

- Bound each authenticated batch dispatch's provider credentials to its own
  async task tree, rejected cross-provider fallback, and failed closed when a
  configured credential resolver cannot supply the dispatching user's
  credential.
- Surfaced missing provider credentials as explicit runner errors instead of
  silent stub turns, resolved control-plane routes across lazy nested routers,
  and preserved application state during batch-registry shutdown.
- Reconciled pending prepared BusinessModel changes with their exact approval
  lifecycle before retry and maintenance, including TTL precedence, lineage
  checks, and compare-and-swap conflict handling.

## 0.15.9 (2026-07-15)

### Fixed

- Required a fresh approval from the exact human owner for promotion-saga
  actions, preserving the approval constraint and reviewed-change identity
  through persistence, reopen, replacement, and lifecycle transitions.
- Prevented delegated or persistent grants, auto-approval, mismatched owners,
  and non-owner roles from satisfying owner-only promotion approval.
- Derived approval constraints from the trusted FMS action catalog and failed
  closed for unavailable, duplicate, or unsupported catalog definitions.

## 0.15.8 (2026-07-15)

### Fixed

- Preserved semantic workflow outcomes independently from execution termination
  and exposed both fields consistently through batch control responses.
- Closed approval-cancellation admission races by fencing new producers,
  aborting unpublished durable approvals, and cleaning notification,
  projection, and persistent-grant residue before cancellation completes.
- Kept exact-write BusinessModel authorization and recovery state intact through
  the approval-admission wrapper, including resume and failed-precommit paths.
- Made workflow budget reservations independent of profile import order by using
  explicit skill caps and a fixed per-stage fallback.
- Restored ambient import-time profile configuration after deterministic context
  rendering so introspection cannot leak its pinned environment into the process.

### Changed

- Enforced one canonical `agent` package identity across source checkouts and
  installed distributions, with lazy `agent.batch` exports and stricter guards
  against executable `api.agent` compatibility aliases.

## 0.15.4 (2026-07-08)

### Added

- Forwarded a stable code-execution work-dir environment variable into subprocess
  and Docker sandboxes, and collected bounded helper computation sidecars into
  terminal `code_execute` / `code_execute_status` results for citation minting.

## 0.15.3 (2026-06-19)

### Fixed

- Republished the dashboard artifact runtime registry in the PyPI wheel. The
  source package metadata already includes `agent_gateway/dashboard_artifact/*.json`;
  this release makes `dashboard_artifact/registry_description.json` available to
  installed `ai-agent-gateway` consumers and keeps the publish smoke reading it
  through `importlib.resources`.
- Deferred schema-backed dashboard QA imports so standalone wheel imports do not
  require the monorepo `schema` package unless the QA helpers are explicitly used.
- Honored skill-specific `max_tokens` budgets and typed excluded FMS writer
  blockers in the gateway runner path.
- Preserved rotated agent-session log ranges and hardened autonomous run
  lifecycle cleanup, rehydrated control-event replay, model-writer autonomous
  resume blocking, and Excel dispatch handling for non-live workbook sessions.
- Reported MCP startup diagnostics more clearly and kept `SkillProfile.provider`
  last to preserve positional-construction compatibility.

### Changed

- Extracted gateway streaming, MCP client config / OAuth storage / error /
  catalog / runtime, and tool-dispatcher source-pack helpers into smaller
  modules while preserving the public package surface.

## 0.15.2 (2026-06-15)

### Added — Tool-result spill to code-execution work dir (2026-06-03)

When a model-bound tool result exceeds the truncation cap
(`AGENT_GATEWAY_MAX_MODEL_TOOL_RESULT_CHARS`, default 60K), the full payload is now
written verbatim to the session's code-execution work dir (bind-mounted into the
sandbox) and the model receives a bare filename to read inside `code_execute` — so
large MCP/data-tool pulls skip the context round-trip. Automatic and universal
(every tool), complementing the per-tool opt-in `output="file"` mode.

- `AgentRunner` — new `code_execution_spill_dir_provider` constructor param.
  `_compact_model_tool_result_entry` now returns `(live_entry, durable_entry)`: the
  spill pointer lives only in the live in-memory entry, so the durable event-log
  copy stays pointer-free (no dangling reads on resume). Spill filename is a
  full-sanitized `{tool_name}_{tool_use_id}.{json|txt}`, written exclusive-create
  with a uuid retry; spill is best-effort (never breaks a turn) and skipped for
  error results.
- `code_execution.CodeExecutionBundle.ensure_work_dir` — exposed and lock-guarded
  so spill and `code_execute` share one work dir; sub-agents inherit the parent
  provider (with `approval_key_qualifier` propagated so their `code_execute`
  resolves a backend).
- `easy.py` runner construction passes the provider; app-layer consumers (e.g. the
  interactive runtime) do the same.
- Env kill-switch `AGENT_GATEWAY_SPILL_TRUNCATED_TOOL_RESULTS` (default on).

Spec: `docs/design/completed/tool-result-spill-to-code-exec-task.md` (Codex review PASS).
Live-verified: real model + gateway + FMP — a 232 KB `fmp_fetch` result spilled to
`fmp_fetch_<id>.json` and was read back in `code_execute` over the full 1,254 rows.

### Added — Agent Control Plane v1 (10 PRs + 1 fix, 2026-05-20 → 2026-05-28)

Channel-agnostic HTTP control surface under `/control/...` for skill discovery, run dispatch (chat + autonomous), monitoring, schedule management, approval handling, and artifact browsing. The TUI (and future Telegram + Excel add-in tab) consume this surface; agents-mcp's autonomous tools (`agent_run_start/status/wait/logs/cancel`) become HTTP relays to the same endpoints (Claude's behavior unchanged).

**New endpoint families:**
- `POST /control/session` — mints a lightweight `kind="control"` session JWT from a channel-bound API key (15-min default TTL).
- `GET /control/health` — returns `{status, version, endpoints}`; emits `X-Control-Plane-Version: 1` response header on all `/control/*` responses.
- `GET /control/skills`, `GET /control/skills/:name` — read-only catalog of frontmatter + resolved body (excludes `catalog: false`).
- `POST /control/runs` — discriminated dispatch for chat or autonomous runs (returns `ChatDispatchResponse` with new chat-session token, or `AutonomousDispatchResponse` with `task_id`).
- `POST /control/runs/:id/messages` — chat continuation.
- `GET /control/runs`, `GET /control/runs/:id`, `GET /control/runs/:id/logs`, `DELETE /control/runs/:id` — unified read + cancel across chat and autonomous.
- `GET /control/events` — unified SSE stream subscribing to the new `UserEventBus`; replays per-run buffered events on attach.
- `GET /control/schedules`, `:name`, `:name/logs`, `POST`, `PUT :name/enabled`, `DELETE :name?confirm=true` — full schedule CRUD across launchd + jobs-mcp backends (operator-global scope).
- `GET /control/approvals`, `POST /control/runs/:run_id/approvals/:approval_id` — list + path-scoped resolve over the shipped F131 `ApprovalRequestStore`.
- `GET /control/artifacts` — cross-cutting recent artifact list across all skills (capped at 50, filterable by ticker + skill).

**Internal additions:**
- `agent_gateway.session.GatewaySession.kind` — `"chat" | "control"` discriminator. `SessionStore.create_session(..., ttl_seconds=N)` per-session TTL override.
- `agent_gateway.event_log.UserEventBus` — in-process per-user pub/sub with per-session ordering, bounded 1000-event subscriber queues (oldest-drop with `events_dropped` sentinel), per-run replay buffer (5000 events keyed by `(user_id, control_run_id)`, 60s grace after run termination), shielded `BackgroundTask` cleanup.
- `agent_gateway.session_event_history.SessionEventHistory` — append-only bounded 5000-event retention attached to `GatewaySession`; survives across chat turns until session expiry. Distinct from per-stream `EventLog`.
- `agent_gateway.autonomous_runner` (relocated from `mcp_servers/agents_mcp/subprocess_runner.py`) — gateway-owned `AutonomousRegistry` with per-user identity (`user_id`, `user_email`, `events_path`, `control_run_id` on `AutonomousTask`). Eager-tail tail task on spawn; events written to `{task_id}.events.jsonl` by the child runner and published into `UserEventBus`.
- `agent_gateway.approvals._record_vote_and_unblock` — shared helper extracted from `/chat/tool-approval`; called by both the legacy chat path and the new control-plane endpoint. Role-class precheck via `ApprovalPolicy.role_authorized_for_class` gated to approvals only (denials/cancellations skip the check so low-role users can cancel `irreversible`-class held approvals).
- `_dispatch_chat_turn()` non-ASGI helper extracted from the `/chat` handler; both `/chat` and `POST /control/runs {kind:"chat"}` call it. Stream-lifecycle (`stream_active` flag) managed inside the helper with `try/finally`.

### Changed

- Existing `/artifacts/{ticker}/{skill}/...` and `/letters/...` endpoints now accept **bearer JWT OR signed-claim** auth via a shared `_artifact_auth_dependency` (bearer-first; signed-claim only when no `Authorization` header is present). Existing signed-claim callers continue to work unchanged.
- `agents-mcp` autonomous tools (`agent_run_start/status/wait/logs/cancel`) — internal cutover to HTTP relays against the control plane. Claude-facing behavior unchanged (`agent_run_start` still returns `task_id` etc.).
- `mcp_servers/agents_mcp/subprocess_runner.py` — DELETED (logic relocated to `packages/agent-gateway/agent_gateway/autonomous_runner.py`).
- `/chat/tool-approval` handler refactored to call the shared `_record_vote_and_unblock` helper; role-class precheck added (strictly more restrictive on approvals; unchanged on denials).
- `tui` channel added to `GATEWAY_USER_KEYS` v2 channel taxonomy. Wired through channel registry, credential resolver, `WEB_TOOL_CHANNELS`, agents-mcp validator, and chat_costs attribution. The follow-up `sk_henry_tui_<token>` SSM entries were provisioned for dev and prod on 2026-05-30.

### Fixed

- **Autonomous subprocess boot regression.** PR6 declared `from mcp_servers.scheduler_mcp import server` and `from investment_tools.jobs import api` at module top in `control_plane/schedules.py`. The autonomous child (`python -m agent.autonomous`) transitively imports the gateway code chain and hit `ModuleNotFoundError` because `mcp_servers` isn't on the child's restricted `PYTHONPATH`. Fix: convert both to `@functools.cache` lazy module-level getters (`_scheduler_mcp()` / `_jobs_api()`); schedule endpoints lazy-load on first call (gateway has the modules on path); other consumers can import `control_plane/schedules.py` without triggering the chain.

### Spec / docs

- `docs/design/completed/agent-control-plane-spec.md` — round 8 CODEX PASS.
- `docs/design/completed/agent-control-plane-impl-plan.md` — round 8 CODEX PASS.
- `docs/design/completed/skill-run-id-rename-spec.md` — round 2 CODEX PASS. Renamed demo-surface's per-skill-invocation `run_id` field to `skill_run_id` (frees the unqualified `run_id` for the control plane outer Run ID).
- `docs/design/completed/schedules-lazy-imports-task.md` — round 2 CODEX PASS. Lazy-imports fix for PR6 autonomous subprocess boot regression.

## 0.15.0

### Added

- **Typed event contract for skill-framework runs.** New module
  `agent_gateway.events` exports `SkillRunStartedEvent`, `ArtifactReadyEvent`,
  `AggregateReadyEvent`, `ArtifactFailedEvent`, and `ArtifactUnavailableEvent`;
  structured skill results flow as `skill_result_captured` wire events. All carry
  `run_id` correlation except `ArtifactUnavailableEvent` (renderer-side only, no
  skill run). Re-exported from `agent_gateway` top-level. Spec:
  `docs/design/demo-surface-spec.md` §2.3.
- **Artifact read API.** Four new GET endpoints serving artifact JSON sidecars
  and `.docx` binaries from per-user workspace storage:
  - `/api/artifacts/{ticker}/{skill}/latest`
  - `/api/artifacts/{ticker}/{skill}/{artifact_id}`
  - `/api/artifacts/{ticker}`
  - `/api/letters/{ticker}/{artifact_id}`

  Signed-claim auth (reuses the credentials-resolver verifier). Path safety:
  rejects `..`-traversal (raw and URL-encoded), symlink escape, and
  cross-user access (returns 404 not 403 to avoid info-leak). Read-only —
  server-side materialization writes the files. Spec:
  `docs/design/demo-surface-spec.md` §2.5.
- **`/chat/init` now returns `user_id: str`** on `ChatInitResponse`, populated
  from the credentials-resolver-resolved user_id. Lets consumers thread the
  resolved identity onto subsequent `/chat` calls (fixes `cross_user_reuse`
  for consumers that proxy chat). Spec:
  `docs/design/theme-a-user-claim-spec.md` §4.

### Changed

- **BREAKING (CORS):** Default CORS `allow_headers` no longer includes
  `X-MCP-Secret`. Consumers that relied on browser clients sending this
  header must either re-add it in their own CORS config or migrate to JWT
  auth via `/api/chat/init`.
- Internal `code_execution` env-var denylist swapped from `EXCEL_MCP_SECRET`
  to `EXCEL_MCP_API_KEY` + `ADDIN_DISPATCH_API_KEY` to match the new auth
  model.

### Docs

- Polished gateway built-in tool docstrings.

## 0.14.1

### Changed

- Propagated parent user identity through framework sub-agent handler dispatchers and sub-sessions.

## 0.14.0

### Added

- Added `Session.channel` / `GatewaySession.channel` and `Session.is_public` / `GatewaySession.is_public` for server-derived channel binding.
- Added `GatewayServerConfig.on_session_created(session, api_key, request)` so deployments can validate or enrich newly-created sessions before a token is issued.
- Added `ChatInitRequest` BYOK fields: `anthropic_auth_mode`, `anthropic_api_key`, and `anthropic_auth_token`.
- Added `channel` and `is_public` JWT claims. Tokens issued before 0.14.0 remain valid and decode with `channel=None` and `is_public=False`.

### Changed

- BREAKING: `CredentialsResolver` now returns `Awaitable[ResolverResult(user_id, channel, auth_config)]` instead of `Awaitable[AuthConfig]`. Resolver implementations now own session identity and channel selection. `finance_cli` and `risk_module` consumers must migrate before adopting 0.14.0.
