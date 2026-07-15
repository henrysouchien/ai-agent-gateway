# Changelog

## Unreleased (post-0.15.9)

_No unreleased changes recorded yet._

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
