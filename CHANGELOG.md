# Changelog

## 0.27.0 (2026-10-16)

### Breaking

- `AnthropicCredentialPool` and `ANTHROPIC_CREDENTIAL_POOL` are removed from `providers.anthropic_oauth`, with the process-local cooldowns and first-usable selection they carried. Account selection is quota-aware: `select_anthropic_credential_source()` and `rotate_anthropic_credential_source()` lose their `pool=` argument and gain `model` / `session_id` (select) and `model` / `scope` / `reason` (rotate); `anthropic_credential_blocked_until(credential, *, model)` replaces `pool.blocked_until(key)`. Cooldowns, quota windows, probe backoff, policy and warm affinity live in a per-store `<store>.selection.json` sidecar (`providers/anthropic_quota.py`), keyed by locator (`AccountLocator`, opaque `acct_` id), so a private and an org record with the same identity no longer collapse; cooldowns are visible to every process sharing the store. A caller that imported the pool or passed `pool=` must drop it: `evals`, `risk_module` and `investment_tools` consumers migrate before adopting this release. `ProviderCredentialFailure` gains `limit_scope`, and an Anthropic concurrency 429 or 529 no longer parks an account.

- `AdmissionOwners` and `ContinuationOwners` no longer accept `placement` or `env_contract`. Admission derives placement and environment projection from the bind's `run_mode`; callers must stop supplying those fields.

- Custom dispatchers must implement `prepare_tool_call_async()` and `dispatch_prepared()`. The runner no longer falls back to direct dispatch when a dispatcher lacks the preparation contract.

### Added

- Quota-aware Anthropic account selection. `ModelProvider.prepare_credential(config, *, model, session_id)` (no-op by default) runs only at episode-starting client builds: the first request of a turn, the rebuild after a model-fallback hop, the reactive-compaction rebuild and the compaction summarizer. A same-model transient retry keeps its binding, and rotated material is not re-prepared. Affinity uses the durable conversation identity (`undo_token_lifetime.conversation_id`), so resume keeps warm affinity; sub-agents keep their own id. `AnthropicProvider` ranks the pool private-first, keeps one hour of warm affinity, then orders by reserve, an 85% short-window guard, measured quota before unknown, priority, and weekly then short-window drain. It probes stale accounts at `api.anthropic.com/api/oauth/usage` (direct route only, 2 s / 4 concurrent; a 401/403 marks telemetry unavailable for an hour and never parks inference) and ingests `anthropic-ratelimit-unified-*` headers on success and failure. A quota 429 parks its scope (shared or model family) until its reset; a concurrency 429 or 529 backs off without parking; any other 403 keeps `billing`. An automatic resolver-issued `credential_source` is an affinity hint, an exact one is honoured. Refresh keeps the store-wide `_store_lock` flock protocol used by published gateway versions, now with a bounded wait, held across reread → exchange → guarded write. New CLI: `auth policy anthropic --identity ACCOUNT [--user N|--store PATH] (--priority INT --reserve-pct FLOAT | --reset)` and `auth list anthropic --usage` (cached view only).

- Gateway-owned workbook undo lifetime (`UndoTokenLifetimeRegistry`, `<log_dir>/workbook-undo.sqlite3`): restore tokens stay active while an owning conversation holds them, without TTL or count eviction. `POST /workbook-undo/status` returns `unknown` / `consumed` / `active` / `terminal`; `POST /chat/close` and `DELETE /chat/conversations/{conversation}` explicitly retire the conversation. Forks share token holders; auth expiry alone does not end them.

- Child-scoped background controls: `run_agent(interrupt=<task_id>)` pauses one owned child without cancelling siblings; `run_agent(resume=<task_id>, additional_context=...)` continues its durable transcript. `TaskRegistry.interrupt()` and `interrupt_background_task()` distinguish `parent_interrupted` from `user_interrupted`; the interrupt acknowledgement is not settlement.

- Detached autonomous children park as `awaiting_approval` on a human ask. `AutonomousRegistry.continue_approval(approval_id=..., tenant_storage_id=...)` relaunches from the captured, signed call binding; the parent executes the decided call without granting the child access to its approval store. Continued results retain the transcript and deduplicate delivery by approval id.

- Approval decisions accept an optional `reason` through HTTP, transport and in-memory approval paths. `ToolApprovalDecidedEvent` carries `approval_id` / `decision_reason`, and a denial's tool result includes the user's reason.

- Offline `session_archive.archive_session()` / `ArchiveResult` and the CLI's `hank archive-session` export a self-contained session directory: transcripts, lineage-linked autonomous children, content-addressed prompts, dereferenced spills and a SHA-256 manifest. Missing or malformed inputs are warnings; the output directory must be new.

- `SessionUsageSummary.provider_model_usage` exposes per-provider/model/bind `ProviderModelUsage` buckets, including the `rate_table_version` that priced each hop. `UsageEvent` and `SessionUsageSummary` also carry an optional child-attempt `stage_id`.

- Code-exec provenance: `hank.input()`, `hank.derive(name, value, expression=..., inputs=...)` and `hank.carry()` retain selected inputs' issued `source_refs`; bridge replies and ledger entries carry them, and `summary()` reports derived values separately from sources. Returned dict/list subclasses carry lineage; unissued inputs produce `input_unsourced` findings.

- Read-only parent-session capability claims for `get_investment_run`, `list_investment_artifacts` and `get_investment_artifact`, without an active skill admission. `issue_investment_capability_claim()` accepts `skill=None` for these reads; mutations still require their admitted skill.

- Admitted children with a non-empty tool surface gain the `tool_result_read` companion for their own spilled results. A child advertises spill reads only when its dispatcher can execute them.

- `decision_record.build_decision_record()` and `run_pointer_from_context()` provide a store-independent decision envelope with an open actor/verdict vocabulary and available run join keys.

### Changed

- `agent_workflow_contracts` moves out of the gateway wheel into the separate `agent-workflow-contracts>=1,<2` dependency; Python import names and generated contract resources stay the same. Source-checkout consumers must include the new package root. The model-authority floor is `hank-model-authority>=1.6.1,<2`.

- `AuthConfig` separates `auth_mechanism` (`api_key` / `oauth_bearer` / `none`) from `entitlement_kind`; `from_dict()` preserves the raw round trip instead of injecting derived keys. Public placement helpers (`placement_for_channel()`, `evaluate_placement_eligibility()`, `filter_fallback_entries_for_placement()`) distinguish local use from `hosted_external` (currently web): hosted execution excludes local daemons and personal subscription seats, while commercial BYOK and org/platform seats retain their own billing identities.

- Anthropic resolution includes user-store API keys and materializes an API-key pool member as an API key, not a bearer token. Muse subscriptions in the org store are operator-personal service/metered seats, not the requesting user's consumer subscription.

- Admission records the role selected by the bind in `selection_provenance`. `autonomous` / `cron` use `local_stub` placement and the autonomous environment projection; `interactive` / `batch` / `fleet` remain in-process.

- `run_agent` reuses matching successful children only within the current spawning runner, after admission, keyed by objective, handles, role, effort and resolved research inputs. A foreground concurrent duplicate waits for the same completion; a background duplicate receives its running acknowledgement. Failed, killed or interrupted children spawn fresh on retry.

- Changed-argument prepared-BusinessModel retries replace only an unclaimed PENDING plan. The predecessor approval is expired through the approval transition owner with `decision_reason="superseded by <approval_id>"`; authorized, claimed or completed plans cannot be overwritten, and denied/expired originals retain their authorization error.

- `ContainmentSettings.compaction_trigger_pct` defaults to 25% instead of 80% (about 250k tokens for a 1M window), retaining the configured token floor and `HANK_COMPACTION_TRIGGER_PCT` override. `run_autonomous()` / `run_autonomous_sync()` accept an optional `compaction_trigger`.

- Autonomous manifests fold declared tool surfaces by union: visible names determine the union count when complete, otherwise the maximum declared count is retained; a missing `allowed_tools` ceiling stays absent, and a later legacy declaration cannot leave an incomplete name set marked complete.

- The seven research context/compare doors can bind authenticated local handlers, including when loaded by autonomous agents, without changing their callable names or arguments. Forecast context returns a compact navigation index and invalid-key recovery with `valid_keys` / `next_call`. Source-identity extraction uses the producer's declared descriptor kind; `get_concept`, `get_statement` and `get_financials` gain citation readers.

- Tool cards omit the internal `_workbook` session UUID, render concept/skill identifiers as readable titles, and supply labels for additional research, memory and FMS tools.

- `CommercialWorkStartError` is owned by `hank_contracts.commercial_work`; the top-level gateway export resolves to that class. It is no longer listed in `commercial_work_start.__all__`.

### Fixed

- Fallback hops select a placement-eligible account from the pool rather than binding an ineligible first member and skipping the route. The bound credential's actual entitlement governs selection, including direct and generation-bound one-candidate pools; a fully parked eligible pool remains distinguishable from `placement_ineligible`.

- Per-hop cost rows retain the pricing table version used for that hop. Sequential and resumed children get separate attempt-scoped cost rows instead of overwriting `sub0` siblings; `session.cached_usage` keeps the dispatch-tree total while summary callbacks remain per-run. `send_prompt()` completion records name the credential bind that actually served after rotation.

- Anthropic refresh fallback credentials are marked prepared only after refresh preflight, so an all-blocked fallback still refreshes before use. Current-generation reads do not wait behind a refresh exchange, and preparation preserves the requested placement.

- Anthropic quota updates never overwrite an unreadable selection sidecar; policy writes report the failure. Per-account `claim_usage_probe()` / `UsageProbeState.in_progress_until` prevent duplicate parent/child probes, freshness follows the jittered probe deadline, and exported `window_exhausted()` / `claim_scope()` keep exhaustion and model-family scope consistent. Overflowing telemetry numbers are ignored.

- Ollama Cloud monthly-quota bodies classify as billing again; a monthly-limit / add-credits 429 without `Retry-After` parks the credential for 24 hours instead of the short default.

- Kimi Code materialization preserves both OAuth and API-key store paths for later rotation. Chat-completions adapters admit `effort="none"` for off-only reasoning models, including adapters without `get_model_info()`.

- Explicit `/chat/init` `model_preset` outranks a saved driver preference. Assistant-tail continuations append a durable user nudge before an Anthropic request; excluded tools do not receive request-1 deferral framing.

- `run_agent` registration failures release unstarted spawn claims, including shielded setup failures; background completion honors the child's `parent_result_policy`, and terminal interruption events use the closure decision's recorded reason. A late handler cannot overwrite a task that has already settled COMPLETED, FAILED or KILLED.

- Detached approval continuations claim the executable attempt after spawn and before credential handoff. A failed launch leaves the call retryable; unknown execution outcomes remain `execution_outcome_unknown`, never a fabricated completion or same-call retry.

- `run_agent`'s schema no longer uses a top-level `oneOf`; an unconfigured resume handler returns `invalid_input` instead of raising `StopIteration`. Approval payload objects without `reason` remain supported.

- Undo result observation unwraps `_runner` / `_inner` wrappers and chains the existing result hook. Explicit close cancels active turns, denies parked approvals with `conversation_closed`, revokes their grants, and prevents cold resume or fork from a closed conversation; token-recording failures warn without breaking the tool result.

- Interactive `invoke_skill` rotates the runner's skill-run identity, and `run_agent` children inherit the parent's source-registry scope so parent-issued source handles remain resolvable.

- Compaction retains noop `verdict_echo` findings, warnings, explicit nulls, empty containers, `argument_skeleton` and `next_call`. Navigation indexes preserve keys/items through projection; an oversized list degrades to a marked prefix with an elision count instead of an omission stub.

- Uncoded owner failures (`planned_write_planning_failed` / `local_tool_failed`) report `retryable=false` and `tool_call_id` without manufacturing a same-call `next_call`; named-field and owner-supplied repair calls are retained.

- Autonomous children receive `CANVAS_BUILD_DIR` / `CANVAS_NODE_BINARY`. `GATEWAY_SSL_VERIFY` is classified as tool environment so declared sibling MCP servers can receive it from their own dotenv.

- MCP stdio shutdown retires watch tasks before taking the manager lock, avoiding the shutdown/replacement deadlock. On Darwin, `killpg` EPERM for an all-zombie group is treated as no live member rather than a cleanup failure.

## 0.26.0 (2026-10-15)

### Breaking

- `resolve_auth_config()` no longer reads Anthropic material from the process environment: `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_MODE` are ignored, the `read_env_auth_mode` parameter is removed, and an Anthropic record missing from args and `auth_config` resolves from the gateway credential pool only (its error messages now say to enroll an account or pass `api_key=`/`auth_token=`).

- Anthropic `agent auth login` returns to the console-callback paste flow. `anthropic_authorize_url()` is removed from `providers.anthropic_oauth`; `start_anthropic_login()` (new, with `generate_pkce()` and `ANTHROPIC_OAUTH_REDIRECT_URI`) builds the request; `AnthropicLoginRequest` carries `authorize_url` instead of `redirect_uri`; `complete_anthropic_login(request, pasted, ...)` takes the pasted `code#state`/callback URL, and the CLI prompts for it (`--no-browser` skips opening the browser). The loopback listener flow is gone because claude.ai refused its redirect.

- `AnthropicCredentialPool` is now a `ProviderCredentialPool` subclass and the Anthropic pool admits API keys as members: `ANTHROPIC_API_KEY` is registered only by the gateway materializer (`get_anthropic_config`), carried as the secret-free `deployment_api_key_source` generation ref (`ANTHROPIC_DEPLOYMENT_API_KEY_FIELD`), and ranked after the user's own enrolled accounts and before env setup tokens and org accounts. The "OAuth pool never admits API billing credentials" (`incompatible_source`) refusals are removed; rotation returns the replacement's `auth_mode` and material, and an API key's billing denial rotates to the seats; `merge_refreshed_auth_config` treats `auth_mode` and the provider pool fields (`project_id`, `account_identity`, `auth_store_path`, `auth_user_scope`, `credential_source`, `refresh_token`, `token_expires_at`) as credential material, not bound values. `ANTHROPIC_API_KEY` leaves `ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES`, so child processes no longer receive it.

- `SessionUsageSummary` (and billing rows) gain `unit_kind` (default `"none"`), `unit_id` and `cost_known`; unpriced routes (`google`, `google_antigravity`, `muse`, `ollama`, `ollama_cloud`; routes `google.antigravity`, `google.public`, …) now report `cost_known=False` instead of a priced zero.

- `claim-contract-v1/commercial-work-authorization.schema.json` is re-vendored from `CommercialWorkAuthorizationV3` (title `CommercialWorkAuthorizationV3`; integer bound written `exclusiveMinimum: 0` for `minimum: 1`; required list reordered) and `commercial-contract-digests-v1.json` carries the new schema digest `27437a3c…`; a consumer pinning the old digest must update.

### Added

- Multi-account credential pool for every file-backed provider: `providers/credential_pool.py` (`ProviderCredentialPool`, `load_pool_accounts`, `upsert_pool_account`, `remove_pool_account`, `rotate_provider_credential`, `register_provider_pool_hooks`, store locks and path resolvers) owns the shared `{"version": 1, "accounts": [...]}` store envelope, transparent migration of legacy single-record stores on next write, `user_scope` → org ordering, cooldown parking and mid-run rotation. `google_antigravity`, `google`, `copilot`, `muse`, `xai`, `ollama_cloud` and the API-key store delegate store I/O to it; `agent auth login` appends seats (same identity updates in place) and OAuth callbacks with a missing or mismatched `state` are rejected.

- `agent auth list` / `remove` accept every pool provider (previously `anthropic` only); `agent auth status` / `logout` / `list` / `remove` take `--api-key` to target the API-key store and `logout` takes `--account` to remove one seat; secrets are never printed.

- `kimi_code` provider (`KimiCodeProvider`, route `kimi.code` at `https://api.kimi.com/coding/v1`, `providers/kimi_code_oauth.py` device-code OAuth, `KimiCodeLoginFlow`, `agent auth login kimi-code [--api-key]`) with its seats in `$USER_DATA_DIR/kimi_code/oauth.json` (per-user under `users/<risk_user_id>/`); the rotating OAuth grant refreshes under the store lockfile and skips the POST when another process already refreshed.

- Hosted user runtimes: `user_runtime_credentials` (edge-minted bundle; the runtime admits one user and signs with a derived claim key), `runtime_placement` (`runtime_host.sqlite3` placements, local-process backend, per-placement state roots and log), `runtime_placement_agentcore` (AgentCore microVM backend with STS credentials narrowed to `users/<id>/*`), and `runtime_tunnel` (session WebSocket; frames over 60 KB fragmented, paced to 200 frames/s, a started message is sent whole, one cancelled before its first frame is dropped). Running runtimes re-mint when the org claim key rotates or the authenticated identity changes; desk sync now mirrors the user's non-database files with the databases. The AgentCore SDK (`boto3`, `websockets>=15`) sits behind the new `agentcore` extra.

- Each delegated sub-agent writes its own cost row (see the existing per-run `on_session_summary` bullet); the agent declares the unit through a session-state `declare_unit(outcome_id)` tool, recorded on the root's aggregator and written to `chat_costs` as `unit_kind`/`unit_id` (any string is recorded, never refused).

- `skills`: `result_requires_persist` (authored persist expectation) is projected onto `AgentOperationSnapshot` and the workflow-contract schema; omitted from dumps at its default so recorded digests replay.

- The autonomous run manifest (`bg_<id>.task.json`) records the child's launch tool surface (`active_servers`, `allowed_mcp_tools_by_server`, `allowed_tools`, `visible_tool_count`) from a new non-wire `tool_surface_declared` event; old manifests rehydrate with the fields null.

- `HANK_MODEL_PRESET` (default model-selection preset) and `HANK_LOCAL_FALLBACK_ENABLED` join the autonomous-child projected environment, as do `INVESTMENT_TOOLS_PYTHON`, `EDGAR_UPDATER_PYTHON`, `INVESTMENT_TOOLS_STATE_DIR`, `GSHEETS_MCP_EXECUTABLE` and `INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY` (paths and a verify-only key; no credential).

- `/chat/subscribe` after the turn's grace clear replays the finished log (`GatewaySession.last_completed_turn`) instead of 404 "No active turn".

- Explicit session state for prompt caching: `GatewaySession.request_tool_prefix` (`RequestToolPrefix`) and `last_turn_context_message`; `ChatRequest.turn_context_message` carries Excel client context.

### Changed

- Excel client context moves from the static system prompt to the user message, attached only when it differs from the previous turn, so the cached system prefix is no longer rewritten.

- Anthropic requests keep a session-scoped prefix-stable tool array: later tools are sent `defer_loading` and surfaced by `tool_reference` blocks on the exposing `tool_result` (which carries only those blocks; its own result follows in the same user message); a tool is deferred only when visible history references it, request 1 carries real deferral framing, and a request with no prior assistant message re-anchors. System-prompt tool sections render from the session's starting surface; `session_recap` replay is a compact deterministic projection.

- Credential rotations onto a new generation and `model_fallback` hops wait 0 s (only a retry of the same credential backs off), a watchdog stall that exhausts its retry budget takes the `provider_outage` chain hop, `rebind_model_fallback` fills `skipped_out` and the evaluated candidates and refusal codes are logged when no hop exists, and fallback hops/rotation cover every pooled provider. Anthropic 400 "usage limits" is classified `billing` with the parsed "regain access on …" UTC reset parking the key.

- Both Gemini adapters (public API key and Code Assist) share one tool-schema reducer, `providers/google_tool_schema.py`: it drops `patternProperties`, `additionalProperties`, `propertyNames`, `if/then/else`, `not`, `contains`, `multipleOf` and the other draft-7 keywords the Schema proto lacks, and defaults a missing array `items` to `string`.

- `run_agent`'s eager tool description lists operations as `namespace/name@version` plus the one-line description (no sha256 digests; the schema still requires the digest) and points to `list_skills`; the `<untrusted-child-results>` reminder attaches only when `run_agent` is advertised or a child notification is queued.

- `skill_result_captured` settles on context-door (`get_*_context`) refusals as well as FMS doors, and a writer child with declared `result_requires_persist` that persisted nothing settles `blocked`/`no_gate_produced` instead of success; classification on the durable-settlement path reads the doors that answered.

- A resumable pipeline stage that exhausts its turns parks (`interrupted`/`turns_exhausted`) and is continued once through `TaskRegistry.continue_task`; `execution_settlement` is now public in `sub_agent_narrative_result`.

- Leaf model calls from `code_execute` bill through the runner's commercial usage producer with the real billing identity; nested bridge dispatch inherits the enclosing call's skill run, step and workspace and forwards only the `dispatch_prepared` contract fields.

- FMS local handlers declare `__gateway_authoritative_input_validation__`, so the schema gate defers to their own validation and errors carry the filled example and `next_call`; model-visible tool examples and hints use `<TICKER>`/`<ACCESSION>` or the call's own ticker rather than a real issuer.

- Anthropic replay drops a foreign model's `thoughtSignature` from `tool_use`.

- `on_session_summary` fires once per session run, not once per request: each delegated sub-agent (spawned or resumed under the parent's own aggregator) delivers its own `SessionUsageSummary` — `session_id` `sub<N>:<root session id>`, its own model, provider, tokens, cost and turns, the root's `unit_kind` / `unit_id` — through the hook the root uses. The root's summary counts the root run alone; `session_recap.usage` and the commercial usage reconciliation still read the whole dispatch tree's total (`_UsageAggregator.fleet_snapshot()`). A child's `declare_unit` records on the root's aggregator.

- The tool loop appends one `context_composition` event per provider request;
  `session_recap` folds its `context_chars.*`, `context_tools.*` and
  `context_skills.invoked` fields into `tool_calls_summary.measurements`
  (peak per name; the `invoked` counts are cumulative). Measurement only.

### Fixed

- Anthropic 400 "Extra inputs are not permitted" on cross-provider replay; 400 on mixed tool_result content.
- Mid-stream in-band `INVALID_ARGUMENT` abort from Cloud Code Assist after streamed progress is a dropped attempt and retries within budget; a first-event error stays non-retryable.
- An attached child's `interrupted` is mirrored onto the live event log, so an exhausted child settles `turns_exhausted` → PARKED, not `runtime_error`; committed interruptions project on the owning event loop.
- `TaskRegistry`: discarding an uncommitted successor restores the source's routing.
- `send_telegram` splits at 4096 characters and autonomous run summaries are delivered whole (no 1200-char truncation).
- Autonomous runs: events file is created only after the durable starting manifest, orphans cleaned up; a successful stream terminal is rewritten to error when the skill result failed; `clean_detach_reason="error"` on stream failure.
- `prepare_model_build`/`build` failures record `prepare_failed`/`build_failed` instead of leaving the attempt live; `check_model` next_call names revisable driver keys; nested `prepare_model_build` carries `skill_run_id`.
- Pool/credential hardening: non-blocking pool read lock (no event-loop stall behind an async xAI refresh), xAI lockfile unified, quarantined seat no longer contaminates siblings, request-scoped credentials never parked, custom-store rotation stays within its accounts, whole-store removal takes the write lock, Muse `--api-key` status/logout target the API-key store, kimi-code CLI list/remove target `oauth.json`/`credentials.json`, `kimi_code` joins the dual-store family.
- Cost records: child stage rows isolated, `cache_hit_ratio` fixed, `cost_known` tracked, operator-byok rate-version overwrite removed.
- `GOOGLE_AUTH_STORE_PATH` API-key collision in login flows.

## 0.25.0 (2026-10-11)

### Breaking

- `send_prompt()` / `send_prompt_sync()` take a required keyword-only
  `completion_record_root: Path | None`. A path appends one `completion_record`
  v1 row per provider attempt (`capture_level="stream"`: the request params, the
  provider's raw content blocks, model identity, `CapabilityBind`, usage and
  outcome) under `<root>/<YYYY-MM-DD>.jsonl`; `None` states the call is not
  captured. The returned text is unchanged and a write failure is logged
  (`completion_record_write_failed`), never raised. New
  `agent_gateway.completion_record` and the packaged
  `contracts/completion-record-v1` schema own the row format.

- Pipeline stage refs and a named child's declared refs move as granted
  content handles. `declared_ref_lines` is deleted.
  `agent_gateway.stage_ref_handles` admits each handle and renders the same
  `key=<json>` lines from the producer's durable bag. `read_admitted_result_content`
  is the durable read that admission already authorized.


- `RunOutput.max_tokens_reached` is removed, and `run_output_exit_code` /
  `run_output_outcome` no longer return `4` / `"max_tokens"`: a provider
  length stop is not a run outcome. `build_state_payload` no longer writes
  `max_tokens_reached`, and `format_run_summary` has no "max tokens reached"
  status. Each `assistant_message` with `stop_reason: "max_tokens"` counts
  toward `max_tokens_stops` (`MAX_TOKENS_STOPS_MEASUREMENT`) in
  `session_recap.tool_calls_summary.measurements` and `fold_run_measurements`.
  The never-produced `max_tokens_reached` event is no longer read.

- Every containment bound is one setting of `ContainmentSettings`
  (`agent_gateway.containment_settings`), resolved once at process start by
  `resolve_containment_settings(environ)` from `HANK_MAX_SPAWN_DEPTH`,
  `HANK_SOFT_TURN_BUDGET`,
  `HANK_SOFT_BUDGET_FINAL_FACTOR`, `HANK_SOFT_BUDGET_GRACE_TURNS`,
  `HANK_CHILD_SILENCE_TIMEOUT_SECONDS`, `HANK_MODEL_CONTEXT_LIMIT`,
  `HANK_CONTEXT_WARNING_PCT`, `HANK_CONTEXT_PRESSURE_REMINDER_PCT`,
  `HANK_CONTEXT_PRESSURE_REMINDER_STEP_PCT` and `HANK_COMPACTION_TRIGGER_PCT`
  (`max_concurrent_children` has no env name: no concurrency override returns);
  defaults are the ruled values (depth 2, six children, the soft ladder's
  100/100/10 table, 1.5x and two grace requests, 600 s child silence, 200000
  tokens, 80/60/10/80 percent). Any unreadable or out-of-range value, `HANK_MAX_SPAWN_DEPTH`
  included, logs one warning naming the setting, env name and value and keeps
  its default (admission's `int()` of the depth used to raise at spawn time;
  it no longer refuses), as an
  unreadable `HANK_SOFT_TURN_BUDGET` always did.
  `AdmissionOwners` requires `containment` (admission no longer reads
  `HANK_MAX_SPAWN_DEPTH` itself). `AgentRunner(containment=...)` replaces
  `max_concurrent_sub_agents`, and child runners and learning forks inherit the
  parent's object; `run_autonomous(containment=None)` replaces
  `max_concurrent_sub_agents` and resolves from `os.environ` when `None`.
  `HANK_CHILD_SILENCE_TIMEOUT_SECONDS=0` disables the hung-child guard. The
  autonomous child's launch projection carries every containment env name, so
  a child process resolves its parent's values (it previously did not receive
  `HANK_MAX_SPAWN_DEPTH`). Removed: `runner.DEFAULT_MAX_BACKGROUND_TASKS`,
  `runner_sub_agents.SUB_AGENT_ACTIVITY_GAP`, `runner_soft_budget`'s
  `SOFT_TURN_BUDGET`, `SOFT_BUDGET_FINAL_FACTOR`, `SOFT_BUDGET_GRACE_TURNS`,
  `SOFT_TURN_BUDGET_ENV`, `configured_soft_turn_budget()` and
  `resolve_soft_turn_budget()` (use `ContainmentSettings.soft_turn_threshold(kind)`;
  `SoftBudgetLadder` takes `final_factor` and `grace_turns`), and
  `runner_limits`' `MODEL_CONTEXT_LIMIT`, `CONTEXT_WARNING_PCT`,
  `CONTEXT_PRESSURE_REMINDER_PCT`, `CONTEXT_PRESSURE_REMINDER_STEP_PCT` and
  `COMPACTION_TRIGGER_PCT`; `model_context_window()` and
  `effective_compaction_trigger()` take the settings, and
  `context_pressure_reminder_decision()` requires `initial_threshold_pct` and
  `step_pct`.

- A background `code_execute` cell's result is the bounded model projection a
  foreground cell gets (base64 stripped, spilled through the runner's sink,
  compacted under the registry's inline limit) and settles through
  `on_tool_result` under the launching call when its job completes.
  `FrameReader.aclose()` kills the open cells' own background jobs before
  closing them; `cleanup_code_execution` no longer looks up a runner.
  `compact_model_tool_result_entry` and `AgentRunner._compact_model_tool_result_entry`
  take an optional `max_chars`. `code_execution_factory` callables are called
  as `factory(session, job_runner=...)`.

- `ToolDispatcher` no longer accepts `should_avoid_permission_prompts` or `on_headless_ask`, and `HeadlessAskCallback` is removed (GP47, 2026-10-09: background, delegated and resumed runs park on a human ask). `run_approval_lifecycle` drops `deny_user_prompt`, `automatic_approval_reason` and `automatic_denial_reason`. A human ask on a ledger route parks (`ApprovalPark`). With no route and no `request_approval` callback it returns `approval_required`. Nothing emits `headless_auto_deny` or `headless_hook_approved` any more; both stay in `ApprovalDecisionSource` and the event adapter so recorded logs parse.
- `ForkTaskRegistry.submit()` requires the keyword-only `on_settled` (a `Callable[[str], None]` or `None`), called with the fork id after a succeeded admission settles; the registry no longer reads `on_admission_settled` off the handoff. `submit()` now calls the spawn callable synchronously, after the admission is marked started and before the task is created, and a spawn exception abandons the admission and propagates. `ForkSpawn` returns `Awaitable[TaskEntry]` instead of an awaitable cost.
- `spawn_learning_fork(fork_id, work_item)` is a synchronous factory returning `Awaitable[TaskEntry]`. It admits a new `learning-fork` job through `admit()` with its own grant `tool-grant:<fork_id>`, whose routes are the parent's (a root parent's turn wire tools minus the turn's withheld tools, or a child parent's admitted grant) except that `run_agent` is omitted at the `HANK_MAX_SPAWN_DEPTH` tier; an admission refusal raises `RuntimeError`. The job is registered in the parent's task registry under `fork_id` and runs as its own top-level `GatewaySession` (session id `fork_id`) on its own journal, with that entry as the runner's admitted self task. Its bind is the `node.fork` bind the capability resolver selects from the parent's bind.
- `learning_fork_result_provenance()` is removed; a learning fork's provenance is its admitted task's. The private `_bind_learning_fork_execution` relabel and the fixed `_LEARNING_FORK_OPERATION` ref are removed with it.
- `LearningForkConfig` requires `operation_snapshot_builder` (directive, required capabilities, workspace scope → `AgentOperationSnapshot`) and `child_session_log_builder` (child `GatewaySession` → `AgentSessionLog`).
- `LearningForkConfig` requires `fork_journal_reader` (fork id → that fork's own `AgentSessionLog` opened read-only, or `None` when it has none). The fork id alone keys the lookup: a mapped user's journal is named for `user_id` while the ledger owner is `owner_user_id`, and reconciliation reads every dead owner's rows whichever session creates the runtime, so neither the reconciling session's owner, channel nor API key decides it.
- `ForkLedger.reconcile_startup()` requires the keyword-only `fork_journal` (fork id → `AgentSessionLog | None`). A dead instance's `started` admission whose fork's latest attempt in its own journal is a committed `awaiting_approval` park moves to the reconciling instance, so the fork's Continue settles it through the ordinary owner-checked `settle_admission`; every other dead `reserved`/`started` admission is abandoned at its reservation, as before. The park is read from the journal's committed prefix: a torn trailing line a dead writer left is skipped, as that writer's recovery would truncate it. The return value still counts released receipt claims and abandoned admissions.
- `ToolDispatcher.fork_child()` and the `AgentRunnerDispatcher.fork_child` Protocol member require the keyword-only `identity` (`DispatchIdentity | None`), `session` (`GatewaySession | None`) and `approval_route` (`ApprovalRoute | None`); a non-`None` identity makes the clone dispatch under that identity, session and approval route instead of the parent's. The clone keeps the parent dispatcher's credential mode (`credentials_resolver_active`) whatever the identity carries, so a fork of a strict parent refuses an MCP meta-inject call with no resolvable user id as its parent does.
- `spawn_fork_agent()` is a plain function returning the run coroutine. It requires `task_entry` (no longer optional) and the keyword-only `sub_session`, `identity`, `approval_route` and `bind_job_runner`; `event_log_observer` and `independent_accounting` are removed.
- `ApprovalLifecycleStore` implementations must provide `park_executable_binding()` (same keyword arguments as `record_executable_binding()`, returning the nonce the park publishes) in place of `record_executable_binding()`; the native approval lifecycle writes its binding through it. `SQLiteApprovalStore.park_executable_binding()` records a first park's nonce and lets a re-park reuse the durable identity: a call already bound keeps its nonce, an undecided approval whose bound call never executed moves (with its request's `tool_call_id`) to the call now pending, and a call bound to a denied or expired, never-executed approval is bound to the new approval.
- `run_autonomous` no longer accepts `skill_local_tool_ceiling` or `skill_mcp_tool_ceiling` (DO54, 2026-10-09: a skill's declared needs union the granted tools; they are never an exclusive ceiling). The dispatcher's MCP allowlist is `None` and every supplied local handler and connected MCP tool is offered; tenant and role policy still decide each call. A named top-level skill no longer needs an approval policy carrying a 64-hex `policy_bundle_hash` (DO55): the run context carries the session policy's `policy_bundle_hash`, or `"unknown"` when the session has no policy.
- `tool_dispatch_declarations` no longer carries any per-tool dispatch semantics (GP54, 2026-10-09): `tool_dispatch_semantics_overrides()`, `tool_dispatch_semantics_for()`, `build_tool_dispatch_declarations()` and `tool_dispatch_declarations()` read the bound server policy module's `get_dispatch_semantics_overrides()` (exact `RegisteredToolIdentity` → `ToolDispatchSemantics`), and with no policy bound they declare nothing, so every route reads `DEFAULT_TOOL_DISPATCH_SEMANTICS`. Each registration owner declares the idempotence, success signal and source-identity descriptors of its own tools; the hand-written exact-identity table is removed.
- A `code_execute(background=true)` cell is a `code` job on the owning runner's `TaskRegistry`, inspected, waited on and killed through the registry (task notifications, `get_background_result`, `AgentRunner.cancel_background_task`). `build_code_execution(session, config, *, job_runner=None)` takes `job_runner`, a callable answering the owning `AgentRunner` (or `None`) at call time; with no runner a background cell returns `code_execution_unavailable`. Removed: the `code_execute_status` tool and `make_code_execute_status_tool_def()`, `GatewaySession.background_tasks`, and `BackgroundTask`/`OutputRingBuffer` (`agent_gateway.code_execution._background`). `cleanup_code_execution(session)` kills and awaits the running `code` jobs of the session's live turn runner (`session.active_turn.runtime_owner`) before it releases interpreters and removes the work directory.
- A registry job outlives the turn that started it, and an exhausted resumable child parks instead of failing. `TaskState` gains `IDLE` and `PARKED` (neither terminal); a resumable child whose soft ladder aborts settles `interrupted` with terminal reason `turns_exhausted` (`PARK_TERMINAL_REASONS` also admits `share_spent`), which `task_state_for_result` maps to `PARKED`. `TaskRegistry` adds `park()`, `revive_to_running()`, `continue_task()` and `restore_committed_park()` (journal replay rebuilds a committed park as `PARKED` only when its `task_result` is present). A park notifies the parent through the runner's notification listener. The turn-end shutdown of running children and the background-success fence over running children are removed: a parent turn ends while its children run. `resume_background_agent` answers a source that is neither parked nor interrupted with `not_resumable_state` (was `not_interrupted`; the runner lifecycle's own `not_interrupted` code is unchanged).

### Added

- `LogicalTaskRef` gains `PipelineStageTaskRef` (`kind="pipeline_stage"`:
  `pipeline_id`, `stage_id`, `operation`). A pipeline stage attempt is
  published by `agent_gateway.child_completion.publish_child_completion`, the
  same owner a foreground delegation uses, against the stage's existing
  registry entry.

- `hank.completion` and `hank.judge` are registration-generated bridge operations
  on the cell's `HankBridge` client, lowered through `FrameReader` to leaf model-call
  operations (`ModelCallOperation`) in `capability_execution.py` that own execution
  class and rubric schema. A cell passes a registered operation name, never a raw slot
  id or `select_bind`. Under proposal D13 each leaf call registers as a P4 job on the
  runner's `TaskRegistry` and returns a task id readable by `hank.wait` (or cancellable
  via `registry.kill`). Pre-registered operations: `general_completion`, `citation_review`,
  `verification`.

- `ChatInitResponse` and `ControlSessionResponse` carry `unknown_model_entitled_capabilities` (sorted list, empty by default): the model-entitled capabilities the resolver supplied that name no registered capability. They still grant nothing.
- `POST /api/control/runs/{id}/messages` on an autonomous run that is not active but is resumable (failed, interrupted or cancelled skill run) starts the linked resume with the message as the operator instruction and returns the resumed run (`delivery_status: "delivered"`); a settled, non-resumable run still answers 409.
- `merge_refreshed_auth_config` no longer raises on model-selection fields or on a refreshed `provider`, `auth_mode` or `max_tokens`: selection copies are dropped and the bound values overwrite.
- `AgentRunner` accepts optional `admitted_self_task` (`TaskEntry`) and `admitted_self_prior_evidence` (`SubAgentResultEvidence`). The runner durably registers its own admitted job (or adopts the journal's reconstruction of it) before tools run, excludes it from the background-success fence over pending child tasks, publishes no agent completion for it, and settles it from its own events and journal before releasing the writer lease, merging the prior evidence so usage stays cumulative; a park settles it as a nonfinal `awaiting_approval` result.
- `ForkTaskRegistry.settle_task(*, entry, on_settled)` settles a final canonical learning-fork result: `succeeded` settles the admission at `observation.usage.cost_usd` and calls `on_settled`, any other final status abandons it, and `awaiting_approval` leaves the admission and its handoff open for the continuation's settlement.
- `GatewaySession.learning_fork_operation_catalog` holds the session's root operation catalog, whose callable operations a root learning fork is offered as children.
- `ordinary_admitted_task_factory` is public (previously `_ordinary_admitted_task_factory`).
- `learning_fork_host_runner(parent, journal, *, fork_id)` rebuilds a learning fork's runner on its own journal for a Continue of a call the fork parked: from the fork's `AdmittedTask` as `journal` registered it (frozen bind, admitted context, grant, provider tool definitions) and the owning session's restored runner's process wiring, as `spawn_learning_fork` built it, with no root turn. The fork's bind is reauthorized on its exact frozen credential generation through `materialize_frozen_bind`, as an ordinary child's resume is, so a fork parked under one Anthropic generation continues on it after the owning session moved to another; a generation that can no longer be read, or that the owning session's ownership facts no longer admit, raises `CapabilityResolutionError` (`credential_unavailable`). The runner's session resumes onto the journal's file, and its dispatcher keeps the owning dispatcher's credential mode. Returns `None` when `journal` holds no registration of `fork_id`.
- `materialize_frozen_bind(resolver, bind, *, user_scope, base_config)` reauthorizes a durable bind on its exact frozen credential: an Anthropic generation-ref bind is offered its own generation's material under the resolver's current ownership facts (`with_frozen_credential`), and an unreadable generation is `CapabilityResolutionError` `credential_unavailable`. The resume of an admitted child and the learning-fork host both materialize through it.
- `session_workspace_id(session_id)` builds the `session:<id>` workspace grant id an ordinary admitted task carries when admitted beneath a chat session, and `session_of_workspace(workspace_id)` reads the session back, raising `ValueError` for any other workspace.

### Changed

- `ForkRequestHandoff` accepts any admitted parent `CapabilityBind`; it no longer requires the `session.driver` bind.
- A learning fork's owner receipt (the next-turn "Self-learning fork completed" note) names the turns the fork ran: `N memory note(s) written in T turn(s)`.
- A run whose tool call raises `ApprovalPark` closes as `awaiting_approval`: `stream_complete` carries `terminal_disposition="awaiting_approval"` with `job_id`, `approval_id`, `original_tool_call_id` (and `root_turn_id` when present), the detach reason is `awaiting_approval`, the parked call gets no `tool_call_complete`, and a cleanly parked run does not shut down its background tasks. A top-level named skill stays open with no result receipt or completion effect, and journal recovery retains an attempt parked by its committed recap/detach pair instead of reconciling it as an orphan.
- An ordinary, resumed or fork child whose run parks returns a nonfinal `awaiting_approval` `TaskResult` (reason `approval:<approval_id> call:<tool_call_id>`, usage cumulative over prior evidence on resume) and keeps its task-registry owner open.
- Chat turn state reports `approval_pending` for a turn whose terminal disposition is `awaiting_approval`, instead of `failed`.
- `ToolDispatcher` durable exact-write authorization propagates `ApprovalPark` rather than recording it as an authorization failure.
- `reconstruct_child_run_lineage()` and `reconstruct_messages_for_task()` reconstruct a task registered on its own journal by the writer runner whose session is the task (a learning fork's self task): the attempt is that runner's writer events between the registration and its `task_completed`, and the `user_message`/`assistant_message` events that runner journaled before registering form the attempt's initial context, carried on the new `ChildRunSegment.initial_context` (empty for a child registered in its parent's journal, whose reconstruction is unchanged).
- A runner with an `admitted_self_task` that starts from `resume_initial_messages` (a learning fork) journals that conversation on its own journal when it first registers the task: every message but the last before its `task_registered` (a user message as `user_message`, an assistant message as `assistant_message` with its content blocks, `model`, `provider` and `stop_reason`, and empty usage; an inherited `tool_result` stays in its user message, with no `tool_call_complete`), then the last as the `user_message` the run starts with. A run that adopts the journal's registration journals nothing again. `build_assistant_message_event()` accepts `model=None`.
- A durably routed native approval whose call has a recorded executable binding parks instead of waiting in the turn: `await_user_approval_via_pending_tools()` accepts a keyword-only `park` (`ApprovalPark | None`) that it raises after publishing the ask, and the lifecycle passes one for every bound call (`root_turn_id` from the executing root turn, `None` for a child), so the run closes `awaiting_approval` and the vote is delivered by the Continue hook. `run_approval_lifecycle()` results no longer carry `executable_nonce`; a call with no job or tenant context still waits on the live queue.

### Fixed

- Run closure publishes one live `interrupted` event on the owning asyncio loop
  only after the durable interruption and detach settle, before the terminal
  closes `EventLog`. Attached children retain `soft_budget_abort` on their live
  log so parent result classification parks a resumable child; closure failures
  publish the canonical error reason rather than a staged interruption reason.
- `TaskRegistry.find_on_roster(<agent name>)` no longer resolves a parked job that a resume successor has taken over, so a message by name reaches the live attempt and a fresh worker with the same name is not ambiguous. The superseded source keeps its own `parked` settlement and stays addressable by job id.
- A background task attached by several owners runs every owner's completion or park callback once, in attachment order, even when an earlier owner's callback raises.
- An approved or resumed successor's session id derives from the hosting runner's session, the id the runner registered the successor under, instead of the resume handler's construction-time parent session. They differ only when a learning fork's host runs the successor, whose own run is now attributed to its lineage segment.
- A native approval that parks again — a resumed BusinessModel request whose producer died while it was pending, or the same call id parking twice in one job — parks on its durable binding and re-presents the ask instead of failing with `planned_write_authorization_persistence_failed` or raising `ApprovalExecutableBindingError` (`nonce_mismatch` / `call_identity_mismatch`).
- `GET /control/approvals` lists a parked approval — a `pending_user` request whose executable binding is unclaimed, with no live `pending_tools` entry — to its tenant on the request's channel, or through a delegation grant, exactly as `POST /control/runs/:run_id/approvals/:approval_id` decides it. It is read through the new `SQLiteApprovalStore.list_parked_requests(tenant_storage_id)` and projected like a live row: no nonce, prepared input or binding operand.
- `ToolDispatcher.execute_approved_binding()` records the decision it executes or denies (`tool_approval_decided`, `decision_source` `durable_approved_binding` / `durable_approval_denied`) on the session's durable log as well as the live log, as the ask is recorded at the park; before, the decision reached only the Continue's live log, so no durable fold (session recap) held it. `tool_dispatcher_audit.emit_approval_decided()` returns the event it appended.
- A durable decision on a parked root call whose turn was admitted under an approval policy generation that is no longer current records no vote, runs no `on_resolve`, creates no grant and executes nothing; before, it recorded the vote and the continuation answered `policy_generation_changed` with no way to re-approve. The call is re-issued under the current generation through the new `SQLiteApprovalStore.reissue_parked_approval()`: a new approval and nonce for the same executable binding, linked from a new `approval_root_turns` row (the original turn row is unchanged), with the original request `expired` (its `decision_reason` names the successor). One transaction creates the successor and its turn row, moves the binding, expires the original and makes the successor pending, so a failure anywhere leaves the original pending and no successor. It returns `(successor, nonce, successor_turn_id, created)`; a concurrent decision that finds the original already re-issued gets that successor back with `created` false. The root hook still refuses to execute under another generation; when it answers `policy_generation_changed` for an approved, never-executed call (a vote recorded under the earlier generation), the decision route re-issues it the same way. `park_executable_binding()` accepts a keyword-only `policy_generation_ref` and, given one, supersedes an undecided or approved, unclaimed binding whose linked root turn was admitted under another generation in the same transaction. The root hook re-publishes the ask once, for the decision that created the successor (`RootContinueRequest.reissued_from`, built by the new `build_tool_approval_request_event()`), with one append to the job's durable log and no session restore, and the response's `continuation` carries `status: "policy_generation_changed"` and an `error` (`approval_policy_generation_mismatch`) whose `data.next_call` names `POST /control/runs/:run_id/approvals/:new_approval_id`. A repeat decision on the original approval answers 410 expired. A request whose binding moved to another approval no longer locates that approval's root-turn ref (decision lookup and root-turn lookup); it keeps the ref its own turn issued.
- A key/token a request carries (a bound Anthropic `auth_token` with no `credential_source`, or BYOK session material through `enroll_request_anthropic_credential`) registers under its own source, `request:anthropic:<kind>:<sha256 of the material>`, instead of the one shared `request:anthropic:<kind>` slot. Before, a second user's (or second session's) token retired the first's generation, so the first token's next resolve raised `AnthropicSourceError` `superseded`, permanently; now each token resolves its own generation. Existing records under the shared slot still resolve by their material.
- An Anthropic `resolve_anthropic_credentials()` with a bound `credential_source` is now the documented pure read: it reads its exact generation under a shared, read-only store lock, with no environment sync and no registration/retirement, and reports `superseded`/`removed`/`unobservable` for a changed/empty/absent env source without rewriting the store. Before, it always ran the environment sync first (`_sync_environment_tokens`) and rewrote the store, so the "exact lookup only reads" claim in `providers/README.md` and the authentication reference was false through the only public entry. Environment reconciliation now happens only on the pool path (the writers).
- Every environment-pool token's source is keyed by its own material's digest, `env:<name>:<sha256 of the token>`, for all three pool forms, not one shared `env:<name>` slot. Two processes that share the org store but carry different values for `ANTHROPIC_AUTH_TOKEN` (or `CLAUDE_CODE_OAUTH_TOKEN`) no longer retire each other's generation; a single-valued name retires its slot only when observed empty (an emptied env name still retires its slot), while `ANTHROPIC_AUTH_TOKENS` keeps array-membership reconciliation. Before, a second process's token superseded the first's and its bound run raised `AnthropicSourceError`.
- A read path that opens the Anthropic credential store read-only never creates the store, its directory or its lock file, and an unwritable or unreadable store surfaces as the typed `AnthropicStoreError` (`code` `store_unreadable`, its own `next_call`), never a raw `PermissionError`/`OSError`. Before, `_store_lock` took the write lock with `O_RDWR|O_CREAT` on every resolve, so a read-only store directory crashed `load`, `resolve`, `ensure_fresh` and `cli auth status` with a raw `PermissionError`, and every resolve created the lock file.
- A retired/removed/superseded bound Anthropic generation that reaches the runner is a typed terminal: `runner_run_loop` maps the typed `AnthropicSourceError` to `credential_unavailable: … | next: python3 -m agent_gateway.cli auth login anthropic` instead of "Provider startup failed: could not create client", and `runner_stream_turn`'s retry loop recognizes the typed error as a credential failure and rotates through the resolver to the next eligible generation exactly like a 401/429 (a frozen generation bind still fails typed at materialization, with no sibling fallback).
- An autonomous child's containment policy grants read-only access to the org Anthropic credential store directory its exact-generation resolve opens (`resolve_anthropic_auth_store_path` with the child's env), derived from the store-path owner, and creates/grant nothing when that directory does not exist. Before, every resolve wrote to the store with `O_RDWR|O_CREAT`, so a Landlock-contained child (granted only `USER_DATA_DIR/users/<id>`) hit `EACCES` on the org store lock even with a user-scoped store.
- A job spill sink's reads (`resolve_spill_producer`, `read_spill_result`, `resolve_job_spill_producer`) validate the parent root, `jobs` and the job directory as existing non-symlink directories, as publication does, without creating them; before, a symlinked `jobs` directory was followed to another tree's spills.

## 0.24.1 (2026-10-08)

### Breaking

- Investment facade claims use version 4 and `kind=facade`, signing the executing
  admitted tenant alongside the Risk subject. Consumers migrate their facade
  verifier with the package upgrade. Selected-content version 2 and
  job-principal version 3 retain their distinct forms.
- `ApprovalRequestStore` adds the required Protocol members `locate_request` and `_locate_request_for_decision`; custom store implementations must provide both.
- `TaskRegistry.adopt_interrupted()` is removed in favor of `adopt_reconstructed()`, which keeps an existing canonical live entry or adopts a durable reconstruction regardless of task state.
- `resume_admitted_successor()` requires the keyword-only `on_park` (an async callback or `None`) and installs it before the successor worker starts.
- `dispatch_identity()` requires the keyword-only `attempt_lineage` (`None` for an initial dispatch, a trusted async resume-ancestry lookup for a successor).
- `make_resume_handler()` requires the keyword-only `operation_resume_bindings` (`None` when no operation-specific resume wiring is installed), and `AgentRunner.resume_sub_agent()` requires `usage_aggregator`.
- `ToolDispatcherApprovalStore` adds the required Protocol members `_read_executable_binding_for_executor`, `_read_completed_executable_result_for_executor`, `_claim_executable_binding_for_executor` and `_complete_executable_binding_for_executor`.
- The autonomous event channel adds the `PRIVATE_EXECUTABLE_CAPTURE` frame: `AutonomousEventChannelParent.receive_next()` can return `AutonomousPrivateExecutableCapture`, and bulk `receive()` rejects a stream containing one.
- `create_delegated_autonomous_approval()` requires a verified `AutonomousClaimBroker` even when no capture is supplied.
- Approval clocks are removed: no elapsed time denies, expires or times out an approval. `agent_gateway.approval_settings` and `GATEWAY_APPROVAL_WAIT_SECONDS` are deleted; the `approval_timeout` dispatcher error and SDK `[approval_timeout]` interrupt are no longer produced.
- `await_user_approval_via_pending_tools()` drops `approval_store` and `timeout_seconds`, requires a session, and returns the decision dict (never `None`); its wait ends on the decision or cancellation. `run_approval_lifecycle()` drops `approval_queue_timeout_seconds_fn`, and its result no longer carries `timeout`. `ApprovalWaitStore` is folded into `ApprovalLifecycleStore`.
- `effective_trade_approval_expiry_seconds()` drops `max_wait_seconds` and returns `float | None` (`None` when neither the policy nor a trade preview sets a window); `effective_trade_approval_decision()` drops `approval_wait_seconds_fn`.
- `ApprovalCallback` and `RequestApproval` callbacks return a decision, never `None`.
- `SQLiteApprovalStore.expire_pending()` and `expire_prepared_business_model_changes()`, `prepared_business_model_store.expire_pending()` and `ApprovalMaintenanceResult.approvals_expired` are removed; `maintain_pending()` drops `now`, and `expire_pending_loop` is renamed `maintain_pending_loop` (prepared reconciliation only). A vote recorded after a request's `expires_at` is accepted.
- `DELEGATED_APPROVAL_EXPIRY_SECONDS` is removed: delegated requests are recorded with `expires_at=None`. `SingleUserApprovalPolicy` asks with no expiry, prepared BusinessModel rows accept `expires_at=None`, and the lifecycle no longer stamps a fifteen-minute expiry on them.
- `SQLiteApprovalStore.claim_raw_patch_authorization()` no longer refuses an approval whose `expires_at` has passed; identity, prepared-digest and single-claim checks are unchanged. `GatewaySession.approval_expire_pending_task` is removed.
- `AUTONOMOUS_APPROVAL_CHANNEL_DEFAULT_RECEIVE_TIMEOUT_SECONDS` is removed: `create_autonomous_approval_channel()` and `adopt_inherited_autonomous_approval_channel()` default `receive_timeout_seconds` to `None`, so a delegated child waits for its next decision until one arrives or the parent closes the channel. Send deadlines are unchanged.

### Added

- `ToolDispatcher.execute_approved_binding(*, job_id, tool_call_id, nonce, abort_event, on_executed_prepared_call)` executes the exact durably bound original call after approval: same job or a trusted resume successor, authenticated tenant, original grant membership and frozen credential generation; one claimed attempt, completed results replay, unknown MCP outcomes stay unknown.
- `continue_approved_child(runner, *, parked_task_id, job_id, tool_call_id, nonce, on_complete, on_park)` (host-side) mints the ordinary resume successor with the original `ToolGrant` object and, before the provider runs, executes the approved original call once through the successor's `execute_approved_binding`, replacing only that tool use's synthetic result.
- Autonomous children send the original approved call's executable operands on a private event-channel frame (`AutonomousEventChannelChild.send_private_executable_capture`); the parent verifies identity against the signed launch envelope and records the private binding before publishing the request.
- `AgentRunner.attach_background_task(task_id, *, on_complete, on_park)` attaches lifecycle callbacks to a live or adopted durable task without starting it or resetting its state; `OperationResumeBinding` supplies scoped local handlers, an operation-selected usage aggregator and an optional settlement callback that `make_resume_handler()` applies after releasing activity ownership.
- `resume_binding_for(runner, task_id)` returns the adopted entry plus prior durable journal evidence and the operation's resume bindings when a restored learning-fork child asks.
- `TaskRegistry.commit_parked_termination()` and `restore_committed_awaiting()` publish a durable parked cancellation and restore an interrupted projection to its committed approval park.
- `inherited_operation_declaration()` derives a child declaration from the parent's tool ceiling using catalog-owned capabilities and effects, picks the narrowest admitting workspace scope, and excludes effect-less routes with a warning.
- `learning_child_session_id(job_id)` derives a learning fork's session id from its logical job id.

### Changed

- `resume_admitted_successor` documents its cancellation contract: caller cancellation leaves initialization running with an unknown outcome, retrying the same successor returns its live entry, and an error result is returned only before a worker exists.
- `TaskRegistry.load_from_events()` replays ordered `task_terminated` records as cancelled parked attempts without replacing the original park receipt (a durable event-log record; no SQLite change).
- `ToolDispatcher` execution audits use canonical result settlement and semantic tool-error classification, and approval-binding and planned-write errors carry recovery guidance in `data.next_call` plus `approval_id`, `tool_call_id` and `state` when known.

### Fixed

- Parked skill settlement is nonfinal: `classify_child_outcome()` classifies
  `awaiting_approval` without error and with `is_final=False`, emitting no
  terminal `skill_result_captured` event and leaving durable run and outcome
  counters unincremented until the final outcome is published.

- A parked background task (`awaiting_approval`) stays non-terminal; its park callback fires once after the durable park commits, including when the append acknowledgement is lost after fsync; `terminate_parked_background_task` stops a parked task by journaling one `task_terminated` with the same admission and full cumulative usage under the original task finalization lock.
- Approval decisions resolve the durable request under the authenticated tenant and original call/nonce when the live queue is gone; the vote commits before the installed root or child continuation hook runs, and an absent hook reports the uninstalled owner.
- `collect_run_output`, `build_state_payload` and `format_run_summary` keep `awaiting_approval` a non-error autonomous outcome, retain approval references only while parked, and display the pending approval instead of reporting completion.
- `RunnerSessionLifecycleMixin` accepts child-authored messages addressed to the root mailbox and reconciles their consumption through the owning task and the consumer's attach provenance.
- `make_resume_handler` adopts reconstructed parked tasks into the canonical registry, serializes admitted-successor installation with a parked source's STOP, discards an unstarted successor whose source has terminated, and holds the source lock until shielded initialization settles after caller cancellation.
- A credential rotation that lands on an untried pool member no longer spends a transient stream retry, so a run can cycle through all parked members before the turn fails.

## 0.24.0 (2026-10-08)

### Breaking

- `ToolDispatcher` requires the keyword-only `capability_bind`, the frozen bind of the execution it dispatches for; `fork_child()` requires the fork execution's `capability_bind` as well. Investment callers migrate together with the `>=0.24,<0.25` dependency range: production `research/task_runner.py` (two sites) and `research/web.py`, and five sites in `tests/unit/test_capability_task_runner_policy.py`. `build_tool_dispatcher()` requires the same argument.
- `CapabilityExecutionResolver.select_bind()` and `.resolve()` remove `role_map`, `role_map_source` and `requested_role_source`; pass the frozen `selection_snapshot` instead.
- `RootTurnRecord` removes `routes_json` and replaces `grant_routes_digest` with `grant_digest`; `SQLiteApprovalStore.record_root_turn()` removes `routes`. On open, `SQLiteApprovalStore` rebuilds the 0.23.1 `approval_root_turns` table, preserving rows, executable references and indexes while recomputing grant digests. Snapshot digests are recomputed only after validating the original snapshot; altered rows retain their original digest and remain unreadable as `snapshot_mismatch`.
- `AgentRunnerDispatcher` implementations must provide `replace_capability_bind()` and `local_tool_handlers()`, and accept the required keyword-only `capability_bind` in `fork_child()`.
- `issue_investment_capability_claim()` requires the keyword-only `tenant` argument and an authenticated `AdmittedTenant`; the emitted facade claim remains version 1.
- `run_approval_lifecycle()` requires `executable_capture` (pass `None` without native capture); `ApprovalLifecycleStore` implementations must provide `record_executable_binding()` and `link_root_turn_executable_binding()`.

- `ToolDispatcher.fork_child` requires `spawning_turn`, captured synchronously
  with `capture_spawning_turn()` before scheduling the fork. Fork request
  handoffs carry that scope so learning-fork approvals retain the submitting
  job and root turn even if a later turn starts before the worker runs.
  Independently admitted child identities retain their own job context and
  do not attach approval bindings to the parent's root turn.
- Code execution no longer reads or consumes legacy computation JSON sidecars.
  Native nested-call provenance and settled `dispatch.sources` remain available
  in terminal results; the sidecar-only size/count limits and deduplication are removed.

### Added

- `RootTurnAdmission` and `admit_root_turn()`, `capture_root_turn()`, `current_root_turn()`, `record_root_turn_admission()` and `link_root_park()` capture each root turn’s exact authority and link parked calls to its durable snapshot.
- `reserve_admitted_successor()` and `resume_admitted_successor()` provide a trusted host entry for workflow-owner-admitted successors, separate from model-driven resume.
- `child_gateway_session()` constructs child sessions from admitted tenancy and materialized child credentials.
- `BoundCapabilityExecution.resolver` retains its materializing resolver; `CapabilityExecutionResolver.rebind_rotated_credential()` authorizes a rotated credential generation under a new exact bind.
- `ToolDispatcher.replace_capability_bind()` updates the current credential bind without changing execution authority; `local_tool_handlers()` exposes configured job-rebound local handlers.

### Changed

- `ToolDispatcher.dispatch_prepared()` captures the original native prepared call and dispatch context in private approval custody before publishing a park, including the executing job, tenant and credential generation.
- `run_autonomous()` records root-turn admission before root tool dispatch; SDK approval calls can link durable requests to root turns without native executable capture.

### Fixed

- An uncaptured `GatewaySession` declares absent admitted job context as `None`. Investment claim issuance returns its typed authority gap rather than failing on an undeclared attribute.
- `ToolDispatcher.fork_child()` retains the spawning turn’s admitted job context and root-turn admission even after the shared session advances, and carries the fork execution's capability bind so learning forks rotate runner and dispatcher together.
- `AgentRunner` and `send_prompt()` rebind rotated credentials through the execution’s resolver; `make_run_agent_handler()` resolves descendants through the live execution’s resolver rather than stale construction-time credentials. `run_autonomous()` pairs a prebound execution that lacks a resolver with an explicitly supplied resolver before constructing its runner.
- `resume_admitted_successor()` tells the model the workflow owner’s frozen `resume_instruction`.
- `SQLiteApprovalStore` serializes secure connection open and close across stores and processes to prevent concurrent WAL/SHM unlink races; its executable-result lookup reports claimed execution as `execution_outcome_unknown` rather than an absent result.
- `code_execute` prepends `extra_env` values only for `PATH` and `PYTHONPATH`; scalar environment variables replace inherited values.
- `SessionContextBuilder.build()` drains the initial fork-seed existence lookup as well as the append before cancellation completes.
- `AgentRunner` preserves exact root-writer mailbox consumption lineage without requiring or fabricating a child id.
- `AutonomousRegistryStartMixin` registers prepared session-log launch inputs through the admitted child environment projection instead of injecting them afterward.

## 0.23.1 (2026-10-08)

This heading covers every change since 0.21.1, because 0.22.0, 0.22.2 and 0.23.0 were
published without their own. 0.23.1 was cut as a patch, but it carries the breaking
changes listed next. No in-forest consumer (Risk, Investment) calls the changed APIs.

### Breaking

- `dispatch_identity()` requires the keyword-only arguments `execution` and `context`.
  `execution` is taken as given (`None` is a resolved absence and is no longer derived
  from the session), and `DispatchIdentity` gains a required `context` field.
- `agent_gateway.sub_agent_capability` is removed; operations author their execution
  class.
- (0.23.0) `select_anthropic_credential()` and `rotate_anthropic_credential()` are
  removed; use `select_anthropic_credential_source()` and
  `rotate_anthropic_credential_source()`.

### Changed

- Investment job controls can carry an exact-operation signed job-principal
  claim without a business skill grant. The four controls are submit, steer,
  pipeline start and benchmark start. Facade v1 and selected-content v2 remain
  unchanged in this additive release; tenant-bearing facade v4 ships separately.
- `DispatchIdentity` requires the executing admitted context independently of
  its credential session. Spawn, resume and native child dispatch retain their
  own job identity, including sessionless admitted execution.
- Root mailbox consumption accepts root-writer- and child-authored sent facts
  without rewriting sender provenance or fabricating a child namespace.
  Recipient authority comes from the bound registry owner; the consumer writer
  comes from an earlier matching attach, including historical replay.
  Child-recipient lineage and exact binding/conflict checks remain enforced,
  and consumption audits remain replay-idempotent.

- Approval executable bindings are private, separate from redacted approval
  projections. The existing SQLite owner preserves original prepared operands,
  backend, job/call/nonce/tenant identity and credential generation reference.
  Original-call reservation uses a compare-and-swap; completed calls recover
  their exact private result, while a lost or unfinished outcome has no timed
  takeover. Canonical task/completion model results retain their types and child
  evidence. This storage change does not remove approval clocks or publish a
  parked continuation.
  Secure connection open/close now holds a verified parent-directory lock across
  WAL/SHM preflight and cleanup, preventing concurrent sidecar unlink races.
  Read-only result lookup reports an existing claim as outcome unknown; valid
  completed results still recover exactly without a second invocation.

- Mailbox sends and ordinary-result, workflow-output and spill reads admit the
  executing job's scope. Child handlers are rebound at spawn and resume; linked
  per-runner registries address parents and siblings without a second registry.
  `run_agent.handles` explicitly delegates readable handles, while
  `send_message.handle_refs` remains citation-only. Child spills live under their
  own job roots and retain pointer/manifest byte-integrity verification.
  Supported package entries capture authenticated root authority before the
  first handle read and expose their root through the existing drained mailbox.
  Autonomous admitted jobs retain their physical job identity. Completed
  foreground registries release their tree links after live descendants and
  completion reconciliation settle. Terminal workflow author failures remain
  readable without an adopted phase or fabricated output publications.
  Catalog-backed `run_agent` admission forwards the operation owner's declared
  effort to the existing provider-bind selection.
  Canonical ordinary content reads include inline narrative and projection
  sources, not only overflow handles; presentation does not create a second
  access rule. Explicit read-grant selectors and exact-byte verification remain.
  Closed background-owner registries release obsolete roster links after their
  parent task is evicted, without rewriting omitted notifications or losing
  durable results; live descendants and detached finalizers retain their links
  until actual settlement.

- BREAKING: run hard caps are gone (`61babe3020a6`..`c308d64d5cdd`). `AgentRunner`
  and `AgentRunner.run()` lose `max_budget_usd`, `max_turns`, `on_max_turns`;
  `AgentSDKConfig`, `ChatRuntime` and `SubAgentConfig` lose the same fields.
  Primary runs no longer carry any ceiling; a delegated run meets one soft
  ladder (`runner_soft_budget.SoftBudgetLadder`) that steers at its threshold,
  forces a final report at 1.5×, and aborts only after grace. The events are
  `soft_budget_steer` / `soft_budget_final`; the terminal is
  `interrupted` with `reason: "soft_budget_abort"`. `max_turns_reached` and
  `budget_exceeded` events and the `turns_exhausted` / `budget_exhausted`
  settlement classes are deleted. Cost stays measured on
  `stream_complete.usage.estimated_cost`. Model: `docs/engineering/soft-budget-containment-reference.md`.
- BREAKING: a background child is never refused for the in-flight count.
  `TaskRegistry.admit` loses `reject_over_capacity`, and
  `runner_background_tasks.background_task_limit_error` (code
  `max_background_tasks`) is deleted. Every admitted child registers, and
  execution waits on the sub-agent semaphore (`max_concurrent_sub_agents`).
  `TaskRegistry(max_inflight=)` now only sizes the notification-retrieval
  retention limit.
- BREAKING: `AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_BYTES` is deleted, and the signed
  launch envelope no longer refuses sign or decode by size. The
  `AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_TTL_SECONDS` constant is deleted (it had no
  reader). The 256-character dispatch-scope caps (`AutonomousDispatchScope`, the
  manifest scope normalizer, the lifecycle `portfolio_id`) are gone.
- Autonomous capability binding no longer checks `run_mode` equality between the
  persisted bind and the resume/launch request. The exact persisted-bind equality
  and the credential actor match are unchanged.
- `spawn_fork_agent` no longer checks that the bind is `node.fork` or that the result
  mode is narrative. The canvas build no longer refuses a finished compile whose
  output exceeds 128 KiB; the timeout and the Linux rlimits are the bounds.
  `BROKERAGE_CONNECT_REQUIRED` is gone from the child environment passthrough.

- BREAKING: an MCP tool call no longer has a deadline (GP19/GA-N2). The
  per-tool and per-server seconds (`default_tool_timeout`, `timeout_overrides`,
  `tool_timeout_overrides`, a registration's `*_timeout_seconds`) are expected
  durations: past one, `agent_gateway.mcp_client` logs one warning with the tool,
  expected and elapsed seconds, and the call keeps waiting for its result. The
  SDK gets `read_timeout_seconds=None` (`McpClientSession.call_tool` now takes
  `float | None`); only the caller's `abort_event` cancels a call. The same
  holds for local tools: `AgentRunner(tool_call_timeout=)` is the expected
  duration of a local tool call (logged on `agent_gateway.runner` when exceeded),
  never a `tool_timeout` cancellation; transient-failure retries are bounded by
  the attempt count alone (`retry_decision` loses `wall_clock_exhausted`).
- The per-profile provider client timeout and stream-stall override are
  deleted (`gateway-policy-denials-derive-from-registration`, TC9/TC37): no
  profile sets `client_timeout` or `stream_stall_timeout`, xAI and Codex no
  longer default a 120s read deadline, and the stream watchdog
  (`runner_streaming.py`) is the one hung-stream bound, applied to every run
  alike. Its terminal error names the attempts.
- `investment_capability_server_load_error` is deleted (GP22 sibling): loading
  `idea-workbench-mcp` is not grant-gated; the claim-gated facade tools are
  authorized per call by claim issuance. The issuer no longer keeps a second
  hand list of facade tools beside the skill note's declared grant.

- BREAKING: investment-capability grants come from each skill's own note
  (`gateway-policy-denials-derive-from-registration`, GP21–GP23).
  `INVESTMENT_CAPABILITY_SKILL_GRANTS` and `InvestmentCapabilitySkillGrant`
  are removed; the application binds the skill-name → `InvestmentCapabilityGrant`
  projection of its notes' `semantic_metadata.investment_grant` with
  `configure_investment_capability_grants` (unbound, no skill holds a grant).
  The claim issuer and inline `invoke_skill` facade grant read that binding.
  Quant claims carry `max_wall_clock_seconds` and `max_tokens` only;
  `QUANT_RESEARCH_APPROVED_BUDGET_CEILINGS` is removed and no turn or cost
  number is minted. Package `0.22.1` requires `hank-model-authority>=1.3,<2`
  so a published-dependency install has `require_registered_capability_bind`.
- The Anthropic provider rate bucket's wait is measured, not silent:
  `AnthropicTokenBucket.acquire` returns the seconds waited and the call's
  turn usage carries `provider_wait_ms` when a wait happened (GP14).
- BREAKING: control-plane refusals stand only with a written reason
  (`control-plane-refusals-justified-or-deleted`). Schedules: a stored record
  with no `dispatch_role` fires as `invite` (never `owner`) and records
  `last_fire_warning`; `DELETE /control/schedules/:name` takes no `confirm`;
  web sessions may create task-mode schedules and read operator schedule logs
  (`web_control_raw_schedule_forbidden`, `web_control_schedule_logs_forbidden`
  removed); the launchd create shape `CreateLaunchdScheduleRequest` is removed;
  dispatch text length refusals are gone and `dispatch_text_lengths` is on the
  record. The dispatch-scope validator owns its outcome: it returns a redacted
  scope or raises its own `HTTPException`, and any other failure surfaces as a
  crash (`dispatch_scope_validation_failed` removed). Valuation-ready batches warn
  `corpus_not_ready` instead of refusing, and `missing_corpus_requirements`,
  `CORPUS_GATED_PIPELINE_TEMPLATES` and the valuation-ready skill guard are
  removed. Readable resources carry any `content_type` and any size; an
  unknown event `schema_version` projects as v1. Commercial claim and
  work-authorization lifetimes are not capped at 300 s; commercial headers on
  a gateway with no or a disabled commercial gate are ignored, not 403, with no
  log line (`commercial_work_start_disabled` removed); unknown model-entitled
  capabilities grant nothing instead of raising; `AuthConfig` no longer
  refuses selection fields; usage/work-start drift no longer fails the turn.
  `/chat` admits the normalizer purpose; saved model preferences are not
  refused by capability class; `_default` is an ordinary user id.
- BREAKING: child, fork and autonomous runs carry no name-based exclusions or
  scope receipts (`child-run-receipts-and-exclusions-deleted`). A `run_agent`
  child's surface is its grant with nothing subtracted by name: the default
  `run_agent` / `get_agent_result_content` / `get_background_result` /
  `send_message` exclusion, the skill `extra_excluded_tools` subtraction on the
  profile path and the "cannot spawn further sub-agents" prompt sentence are
  gone. A fork runs on its parent's wire surface: `FORK_ORCHESTRATION_TOOLS`,
  `LEARNING_FORK_ALLOWED_TOOLS`, `ForkPolicyDispatcher` (`fork_policy_denied`),
  `build_side_quest_tool_decisions`, `build_learning_fork_tool_decisions` and
  the `fork_scope_receipt` module and event are removed.
- BREAKING: fork caps are measurements. `learn_fork_enabled` and
  `HANK_LEARN_FORK_ENABLED`, `HANK_LEARN_FORK_GLOBAL_CONCURRENCY`,
  `HANK_LEARN_FORK_DAILY_INVOCATION_QUOTA`, `HANK_LEARN_FORK_DAILY_BUDGET_USD`
  and `HANK_FORK_HANDOFF_MAX_BYTES` (with their `DEFAULT_*` constants and
  readers) are removed; `ForkAdmissionQuotaExceeded` and
  `ForkAdmissionBudgetExceeded` are removed and `ForkLedger.reserve_admission`
  returns an `AdmissionReservation` carrying the day's invocation count and
  charged microusd. `learning_fork_launched` telemetry carries `in_flight`,
  `day_invocations`, `day_charged_microusd` and the handoff's retained bytes.
  The fork suffix ceiling is removed: `FORK_SUFFIX_WRAP_UP_REMINDER`, the
  forced-final clamp, `HANK_FORK_SUFFIX_MAX_TOKENS`,
  `DEFAULT_FORK_SUFFIX_MAX_TOKENS`, `fork_suffix_max_tokens()` and the
  `spawn_fork_agent(suffix_ceiling=)` parameter are gone, and the
  `fork_measurements` record carries `suffix_tokens`.
- BREAKING: autonomous launch refuses less. The `autonomous_concurrency_limit`
  429 is gone (the in-flight count is recorded as
  `autonomous_in_flight_at_start` on the task and its manifest);
  `AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES` and its refusals are removed
  (`credential_handoff_bytes` is recorded on the manifest); a resumed run no
  longer has to repeat its origin's mode and skill, and a capability binding no
  longer couples its source to the run mode. The child environment projection
  is one list in `child_environment.py` (`AUTONOMOUS_CHILD_ENVIRONMENT`),
  applied to every autonomous child alike instead of varying by profile.
  `deliver=False` (`--no-deliver`) is accepted for every mode and suppresses
  once/pack Telegram delivery.
- `spawn_sub_agent` / `resume_sub_agent` no longer require a `node.*`
  capability bind; an unmapped delegation role resolves from registration
  instead of raising (`DelegationRoleResolutionError` is removed); a persisted
  `ToolGrant` is no longer re-verified against its digest or route order.
  The coordinator's `worker_excluded_tools` no longer subtracts from a child.
  `resume` no longer abandons a non-resumable or model-writer operation
  (`not_resumable`, `model_writer_resume_unsupported`) or a chain past its
  depth (`max_resume_chain_depth`); the successor's `tool_input` records
  `resume_chain_depth`, and `resume_task_id_override` returns the successor id
  alone. Child evidence past its canonical bound no longer fails the child
  (`admission_rejected` is removed): the result settles on its own terms and
  its mechanical outcome is `partial` with `child_evidence` unmet.
- `code_execute` accepts any `timeout_ms` at or above its floor and reports
  the requested `timeout_ms` on its result; the container kill is the bound.
  The canvas import allowlist is read from the canvas-kit manifest's
  `externals` instead of three hand-written copies.
- Agent results carry the reason and the count in turn
  (`agent-results-carry-the-reason-and-the-count-in-turn`).
  `make_error_result` always writes `error.data.next_call`: the producer's
  value, or an explicit `null` when it names no next action. A
  `tool_excluded` result carries `data.excluding_owner` (`profile`, `skill_or_operation_grant`
  or `role`) and `next_call: null` on every hit; a repeated excluded call no
  longer forces a final turn. The FMS-commit and output-file run exclusions
  are removed (money stays at the approval policy, hosted file output at its
  interceptor). The soft-budget steer notice names the run's `observed`
  provider requests and its `threshold` (`soft_budget_steer_notice`).
  `session_recap.tool_calls_summary.measurements` folds the approval
  decisions sent (`decisions_sent`, `frame_bytes`), the learning fork's
  `in_flight`, `day_invocations`, `day_charged_microusd` and
  `retained_bytes`, the skill run's `lease_fence_bytes`, and a `model_calls`
  total of `model_*` tool calls (`RUN_MEASUREMENT_FIELDS`,
  `fold_run_measurements`); an autonomous child's recap also carries
  `autonomous_in_flight_at_start` and `credential_handoff_bytes`.
  `ResolverResult.unknown_model_entitled_capabilities` names the
  model-entitled capabilities the resolver supplied that match no registered
  capability. A commercial work-start 403 (`commercial_work_authority_invalid`)
  names the verifier's refusal code in its message. A sub-agent
  `invalid_context` result names the failing input in `details.field` and
  `details.reason`, with `next_call: null`.

## 0.21.1 (2026-09-29)

### Changed

- Requires `hank-model-authority>=1.2,<2` (schema `model-authority/2.1`).
- The Anthropic adapter turns thinking off with the entry's
  `compat.thinking.off_value`: at effort `none`, a non-null off value yields
  `{"thinking": {"type": <off_value>}}` alone (no `display`, no
  `output_config`) — `disabled` for Sonnet 5, `between_tools` for an entry
  that names it (Claude Sonnet 5.5 refuses `disabled`) — and no thinking field
  when the off value is null.
- `compat.tool_choice_forcing` controls lone-tool forcing: `any` always forces,
  `thinking_off_only` forces only when thinking is off, and `never` sends auto.
  Multiple offered tools use auto.

## 0.21.0 (2026-09-29)

### Changed

- BREAKING: the model authority — registry, selection policy, capability
  resolver, rate tables, effort vocabulary and `CapabilityBind` — moved to the
  new distribution `hank-model-authority` (import `model_authority`), a new
  dependency (`hank-model-authority>=1.0,<2`). `agent_gateway.model_registry`,
  `agent_gateway.capability_binding`, `agent_gateway.rates`,
  `agent_gateway.thinking` and the package-root re-exports of their names are
  removed with no aliases; import from `model_authority.current`,
  `.registry`, `.selection`, `.binding`, `.bind`, `.capabilities`, `.rates` and
  `.thinking`. `agent_workflow_contracts.CapabilityBind` is removed: use
  `model_authority.bind.CapabilityBind`; the wire schema
  (`capability-bind.schema.json`) is unchanged. The package no longer ships
  `model_authority/*.yaml` or `rates/*.json`.
- BREAKING: adapters derive `ModelInfo` from the model's registry entry, its
  `compat:` block and the rate tables. The model-ID tag tables in the
  Anthropic, OpenAI Responses, Codex and xAI providers are gone, including
  `providers/codex_model_info.py`; a model ID with no registry entry is not
  callable.
- BREAKING: `HANK_MODEL_AUTHORITY_DIR` — a directory holding
  `model-authority.yaml` and `rates/*.json` — replaces
  `AGENT_GATEWAY_MODEL_REGISTRY_FILE`, `AGENT_GATEWAY_MODEL_SELECTION_FILE` and
  `AGENT_GATEWAY_RATES_FILE`, which are no longer read. Unset, the copy
  packaged in `hank-model-authority` loads. The autonomous runner projects
  `HANK_MODEL_AUTHORITY_DIR` into its child environment where it projected
  the rates-file variable. A registry or selection revision now ships as a
  deploy of that directory, not as a gateway release.
- A model turn the provider ends with `stop_reason=refusal` completes as a
  refusal instead of a failure. The runner keeps the delivered text as the
  turn's output, persists the `assistant_message` with `stop_reason: refusal`,
  appends a client-visible `refusal` event carrying the provider's
  `stop_details.category` and `explanation` verbatim plus guidance to start a
  fresh conversation, and closes with `stream_complete` `completed`. A refused
  `run_agent` child settles `interrupted` with `terminal_reason`
  `refusal: <category>` and keeps its narrative. Previously a refusal fell
  through to `terminal_outcome_unproven`: an `error` event, a `run_error`
  settlement, and a child `runtime_error` with its text discarded.
- BREAKING: `agent.yaml` loading refuses unknown top-level keys again:
  `load_agent_project_config` raises `AgentProjectError` naming each one, so a
  typo such as `max_buget_usd` fails at load instead of running with no budget
  cap. Keys an earlier `agent init` scaffold wrote and the schema later retired
  (`per_turn_timeout`, `provider`, `model`) still load, with a warning naming
  each retired key; their values are ignored. Reverses the 0.20.0 log-and-ignore
  rule for keys the product never wrote.
- Fork children (`spawn_fork_agent`) run under the same parent-side activity
  guard as spawned and resumed sub-agents: a fork silent past
  `SUB_AGENT_ACTIVITY_GAP` with no tool in flight is cancelled and settles
  `interrupted` with `terminal_reason` `stalled: …`, instead of hanging and
  holding its fork-registry session and concurrency slots.
- A child streaming a portable-compaction summary emits `heartbeat` events
  through its event path while the summary streams, so the parent activity
  guard no longer settles a healthy child `stalled` mid-compaction; a summary
  that sends nothing is still settled `stalled`.

## 0.20.1 (2026-09-28)

### Fixed

- `send_prompt` / `send_prompt_sync` rotate credentials the way a runner turn
  does. A failure the provider classifies as a credential failure that a
  sibling can answer (`classify_credential_failure` → `next_credential`: 429
  and auth-class rejections) parks the bound credential and retries the same
  bind on the next pool member, once per member; with no unblocked sibling the
  last provider rejection is raised unchanged. Previously one refused or
  rate-limited pool member failed every out-of-gateway prompt.
- A 403 carrying `oauth_not_allowed_for_organization` classifies as `auth`
  (rotatable) from its error code. Previously it rotated only because the
  provider's prose message contained the word "authentication"; a body that
  carried the code alone fell through to the 403 `billing` branch and was
  never rotated.

## 0.20.0 (2026-09-28)

### Fixed

- A chat turn that runs past its session's TTL keeps its tools. Token
  verification, session cleanup and turn admission read the TTL through one
  predicate, `session_ttl_elapsed`, which keeps a session live while it holds
  a turn that has not published its terminal event. Previously every tool call
  that authenticated with the session's token was refused `Session expired`
  once the turn crossed `expires_at`, and the cleanup loop cancelled the turn.
  The TTL still applies after the turn ends: a turn attaches only while the
  TTL has not elapsed, so an expired session cannot start a new turn, even
  with a token accepted while the previous turn was still answering.
- A fork child records its dispatcher boundary events (interceptor,
  input-validation and approval audit events) in its own event log under its
  own session id. Previously the child's dispatcher was a shallow copy whose
  boundary-event projection still wrote to the parent's log, and on a
  wrapping dispatcher the copy never rebound the log or session at all.
- A `run_agent` child's first message carries its admitted binding: the
  parent's objective under `Ticker: <T>` and `RESEARCH_FILE_ID=<n>`, the values
  its run context holds. Previously the child got the objective alone, so a
  child whose context tool takes `research_file_id` as an argument could admit
  the file and still report it missing.
- `anthropic.claude-opus-5-5` no longer advertises effort `none`: Opus 5.5
  rejects `thinking: {"type": "disabled"}`. `claude-opus-5-5` has its own
  model-info row (thinking cannot be disabled, omitted-effort default
  `medium`) and rate row ($4 / $20 per MTok, cache read $0.20, 5-minute cache
  write $5); it previously resolved by prefix to the Opus 5 rows. The
  `claude-sonnet-5` model-info row carries the published $2 / $10 standard
  price (cache read $0.20, write $2.50) instead of the cancelled $3 / $15
  increase, matching its rate row. Anthropic rate table version `2026-09-28`,
  source now names the pricing page.

### Added

- Model registry lifecycle `retired`: the provider no longer serves the
  entry's `upstream_model`. A selection policy that names a retired key as a
  default or allowed model fails admission. Resolution refuses a retired key
  with the new `CAPABILITY_RESOLUTION_CODES` member `capability_model_retired`
  (checked before policy allowance, for user-driven and internal selection
  alike), and reauthorizing a durable binding on a retired key refuses with
  the same code; the entry stays in the registry so the binding names why it
  cannot run. A saved preference on a retired key is reported `model_hidden`
  and the capability default applies.
- `scripts/model_registry_liveness.py` (repository development check, not
  shipped in the wheel and not a runtime gate) lists registry entries whose
  Anthropic `upstream_model` the Models API no longer serves; exit 1 when any
  are found. Providers without a liveness adapter are reported skipped.

### Changed

- Adaptive thinking is requested with `display: "summarized"` on every
  adaptive Anthropic model. Thinking text streams as `thinking_delta`, is
  stored in the assistant message's thinking block beside its signature, and
  keeps the stream stall guard fed through long thinking phases (the silent
  phase reached 337 s on opus-5 at effort high). Budget-mode thinking is
  unchanged. Resumed sessions carry the logged `provider` onto replayed
  assistant messages, so same-provider same-model thinking blocks replay with
  their signatures; messages from logs that never recorded `provider` still
  convert thinking to text.
- `agent_workflow_contracts.research_file_contract.context_text_with_run_binding(
  context_text, *, ticker, research_file_id)` is the one composer of the binding
  header a model reads; embedding applications render run and skill contexts
  through it.
- `GatewayServerConfig.transcript_retention_days` now defaults to `0`, which
  deletes nothing. An application that configures `transcript_dir` and says
  nothing about retention previously swept transcripts and their `.meta.json`
  sidecars at seven days. The sweeper is unchanged and a positive value still
  enforces that horizon.
- BREAKING: `ToolDispatcher.with_scoped_local_handler(tool_name, scope)` is
  replaced by `ToolDispatcher.fork_child(*, event_log, session_id,
  local_handler_scopes=None)`, which binds the child's event log, session id
  and scoped local handlers in one clone; `AgentRunnerDispatcher` declares it,
  and `spawn_fork_agent` takes the parent dispatcher plus
  `local_handler_scopes` and calls it.
- BREAKING: `per_turn_timeout` is removed from `AgentRunner`, `create_agent`,
  `run_autonomous`, `GatewayServerConfig`, `AgentProjectConfig`, the sub-agent
  and fork spawners and the `PARENT_PER_TURN_TIMEOUT` child-env allow-list. No
  model turn is cancelled for how long it has run; a turn ends on its own stop
  reason, the stream-gap stall watchdog, `max_turns` / `max_budget_usd`, a
  tool's own I/O timeout or the operator's cancel. While a turn streams the
  runner logs `Turn N streaming: elapsed=…s last_progress=…s ago events=…`
  every 60 s. `load_agent_project_config` now logs and ignores keys it does
  not read instead of raising `Unsupported agent config keys`, so an
  `agent.yaml` the scaffold wrote with `per_turn_timeout: 300` still loads.
- BREAKING: no wall clock bounds a `run_agent` child. Removed:
  `DEFAULT_SUB_AGENT_TIMEOUT_SECONDS`, the 2100 s `run_agent` dispatch cap
  (`run_agent` is now exempt from `tool_call_timeout`), the `timeout` argument
  of `spawn_sub_agent` / `resume_sub_agent` / `spawn_fork_agent`,
  `default_timeout` on `make_run_agent_handler` / `make_resume_handler`,
  `SkillProfile.timeout` (skill frontmatter `timeout:` is no longer read),
  `OperationRuntimePolicy.timeout_seconds`, `timeout_seconds` on `run_session` /
  `run_autonomous`, and the `SUB_AGENT_TIMEOUT`, `SUB_AGENT_SKILL_TIMEOUT` and
  `*_AGENT_TIMEOUT_SECONDS` child-env allow-list names. A wedged child ends by
  a parent-side activity guard instead: with no child event for
  `SUB_AGENT_ACTIVITY_GAP` (2 × `STREAM_THINKING_STALL_TIMEOUT`, 600 s) and no
  child tool call in flight, the child is cancelled and settles `interrupted`
  with `terminal_reason` `stalled: …` (`task_result_from_execution` takes
  `stalled` in place of `timed_out` / `timeout`). The stream guard's 60 s
  progress line also appends a `heartbeat` event (`elapsed_s`,
  `last_progress_s`, `events`) so a child composing a long tool input stays
  live to its parent.
- The chat SSE projection keeps the `heartbeat` liveness fields (`elapsed_s`,
  `last_progress_s`, `events`), so a client can show `working · 3m12s · last
  progress 4s ago` during a turn that streams no text; the TUI working row
  renders it. Provider pings still never reach the wire.
- Model registry and selection policy revision `2026-09-28.1` move every
  workload lane off superseded Anthropic ids:
  - `investment.biotech_review`: `anthropic.claude-sonnet-4-20250514-sdk` →
    `anthropic.claude-sonnet-5` at effort `none` (thinking disabled).
  - `investment.research_agent`: `anthropic.claude-sonnet-4-6-sdk` →
    `anthropic.claude-sonnet-5` at effort `none`.
  - `investment.newsletter`, `investment.earnings_transcript`:
    `anthropic.claude-haiku-4-5-20251001-gateway` →
    `anthropic.claude-haiku-4-5` (undated `claude-haiku-4-5`).
  - `risk.completion`, `risk.interpretation`: `anthropic.claude-sonnet-4-6-sdk`
    → new `anthropic.claude-sonnet-5-sdk` (`anthropic.sdk.messages`,
    `messages.adaptive`, `anthropic.byok`) at effort `none`.
  - `risk.asset_classification`, `risk.overview_editorial`:
    `anthropic.claude-haiku-4-5-20251001-sdk` → new
    `anthropic.claude-haiku-4-5-sdk` (`anthropic.sdk.messages`,
    `messages.adaptive`, `anthropic.byok`) at effort `none`.
  - `risk.document_ingest`: `anthropic.claude-opus-4-8-oauth` → new
    `anthropic.claude-opus-5-5-oauth` (`anthropic.sdk.messages`,
    `messages.oauth`, `anthropic.oauth`) at effort `none`: no thinking field
    is sent, so Opus 5.5 runs its default adaptive thinking.
  - Lifecycle: `anthropic.claude-sonnet-4-20250514-sdk` is `retired`;
    `anthropic.claude-sonnet-4-6-sdk`,
    `anthropic.claude-haiku-4-5-20251001-sdk`,
    `anthropic.claude-haiku-4-5-20251001-gateway` and
    `anthropic.claude-opus-4-8-oauth` are `deprecated`. No policy allows any
    of them.

## 0.19.0 (2026-09-16)

### Changed

- Require FastMCP 4 and MCP 2. Catalog pagination, request metadata,
  structured results, timeout/cancellation, and streamable-HTTP OAuth now use
  the MCP 2 SDK and httpx2 transport directly.
- Classify MCP transport failures by typed SDK error codes without replaying
  uncertain tool mutations.
- Provider extras support Anthropic 1.6, OpenAI 3.14, and Claude Agent SDK
  0.2.153; unrelated HTTP integrations continue to use httpx.

## 0.18.4 (2026-09-15)

### Fixed

- Bound MCP dependencies to supported SDK majors (`mcp<2`, `fastmcp<4`);
  newer majors remove the session APIs used by the gateway transport.
- Made package tests independent of the embedding product checkout: product
  integration scenarios live with the product, and generic fixtures and pytest
  configuration are package-owned.

## 0.18.2 (2026-09-15)

### Fixed

- Removed checkout-tree imports from the wheel, including lazy policy, artifact,
  batch, schedule, learning-fork, and reviewed-plan paths. Applications now supply
  their product implementations through explicit runtime configuration.
- Moved Hank's schema-backed artifact persistence, rendering and routes into the
  product API. Generic gateway installations no longer require Hank's schema,
  research, memory, scheduler, or Investment checkout.
- Added whole-package isolated import coverage against declared dependencies.

## 0.18.1 (2026-09-15)

### Fixed

- Autonomous children spawned from a pinned local runtime generation
  (`LOCAL_GATEWAY_RUNTIME_VERSION_ROOT`) keep the generation's import roots ahead of
  any inherited `PYTHONPATH`, and those roots now name the generation's installed
  Risk distributions directory (`ai-excel-addin/risk-dists`) instead of the Risk
  source tree. Production deployments without a pinned generation are unchanged.

## 0.18.0 (2026-09-15)

This is a pre-1.0 breaking-minor release. It describes the complete public
package delta from 0.17.4, not only the final release-preparation changes.
This section covers the packaged source through commit `35b5ce9d6`.

### Added

- Added immutable, dependency-neutral originated tool-definition records and
  an MCP catalog API that preserves each definition's exact local or MCP
  origin and MCP server owner. The existing dictionary-returning API remains a
  compatibility wrapper and returns fresh mutable copies in the same order.
- Added a lazy directory-backed control-skill catalog that identity-checks
  direct skill files, validates frontmatter and semantic metadata, and returns
  typed non-enumerating refusals for invalid or unavailable selectors.
- Added typed admitted-skill context and policy projections for autonomous,
  batch, control, delegated, and retry/resume execution. New admissions carry
  the compiler-owned skill identity, limits, capability requirements, prompt
  projection, and exact tool/server ceiling instead of reparsing product skill
  files inside the package.
- Added the closed `pack` control-run mode so a named autonomous pack can cross
  the HTTP/control carrier into the existing runner without an out-of-band
  process wrapper. Pack, task, skill, and once fields remain mutually
  exclusive.
- Added immutable live MCP route bindings, registered MCP tool descriptors,
  and policy-implementation registries. `McpClientManager` can expose the
  exact physical or logical route, provider definition, transport, alias
  provenance, and registration joined for each advertised tool.
- Added MCP input-preparation routes to `create_agent()` and
  `run_autonomous()`. Autonomous and server configuration can also inject the
  registered tool catalog, policy implementations, and preparation,
  redaction, planning, and approval context factories; server configuration
  can inject canonical control-profile list and load providers.
- Added explicit `DurableLocalApprovalRoute`, `ParentDelegatedApprovalRoute`,
  and `NoApprovalRoute` contracts plus the exported `ApprovalReuseMode`
  vocabulary for legacy, disabled, and exact reuse.
- Added owner-minted `NamedRefusal` transport errors and an
  `AutonomousFileControlAuthority` projection with required path, device, and
  inode identities for file-mode launch consumers.
- Added content-bound `spill:v1:` references and the bounded
  `tool_result_read` page/search contract. Autonomous spill roots can be
  selected with `AGENT_AUTONOMOUS_TOOL_RESULT_SPILL_DIR` and expire according
  to `AGENT_AUTONOMOUS_SPILL_TTL_HOURS`.
- Added `record_paths` to autonomous run and status responses, covering the
  task manifest, log, event stream, and tool-result spill directory.
- Added public `ToolDispatcher.run_context` and
  `with_scoped_local_handler()`. Scoped clones replace one existing local
  handler without mutating the parent and raise `KeyError` for an absent
  route; Excel dispatcher clones retain their Excel transport.
- Added root exports for `CAPABILITY_RESOLUTION_CODES`,
  `NON_INTERACTIVE_RUN_MODES`, and `SESSION_DRIVER_CAPABILITY`; capability
  resolution now includes `capability_selection_invalid`.
- Added durable learning-fork decision diagnostics, provider rate-limit
  scalars on `stream_retry` and `error` events, and the chars/4
  `est_system_tokens` and `est_tools_tokens` projections on
  `stream_complete.usage`.

### Changed

- **Breaking:** Replaced the exported `ControlSkillDefinition` contract with
  separate `ControlSkillSummary` and `ControlSkillDetail` contracts. Catalog
  listing now returns body-free summaries, while exact resolution returns a
  selected detail with its body; selector validation and the list/resolve
  protocol return types changed accordingly.
- Canvas artifact sources now use the source ledger's generated `SourceRecord`
  schema as their sole tool-definition and runtime authority. Caller-authored
  source mappings are normalized and validated before bundle construction, and
  malformed ledgers return typed source-stage diagnostics instead of surfacing
  as internal errors.
- Gateway tool assembly now consumes exact originated definitions, rejects
  incoherent MCP ownership and local/MCP name collisions, and retains provider
  payloads, catalog order, and fresh-copy behavior at compatibility boundaries.
- Product composition can now inject one compiled summary/detail catalog and
  one captured skill-context projection across prompt, control, admission, and
  lifecycle consumers. The generic package directory catalog remains available
  only for explicitly configured external clients; it is not unioned with or
  used as a fallback for the injected product authority.
- Autonomous launch envelopes, persisted run state, capability claims, and SDK
  runners now preserve the admitted skill facts needed by retry/resume rather
  than consulting mutable live definitions. Missing or incoherent new-admission
  facts fail closed while the explicit legacy resolver remains responsible for
  already persisted historical state.
- **Breaking:** `issue_investment_capability_claim()` requires the
  keyword-only `admitted_skill_execution_limits` argument. Quant-research
  claims require an exact `SkillExecutionLimits` value; other capability
  claims may pass `None`.
- **Breaking:** Moved the admitted-skill contracts from
  `agent_gateway.skill_admission` to `agent_gateway.skill_limits`, renamed
  `AdmittedSkillExecutionLimits` to `SkillExecutionLimits`, and renamed its
  mapping conversion functions accordingly.
- **Breaking:** Renamed `CapabilityResolutionError.receipt()` to
  `to_error()`; the returned error payload fields are unchanged.
- **Breaking:** `ToolDispatcher` now receives one `approval_route` instead of
  independent `store` and `policy` arguments, no longer accepts
  `mcp_identity_overrides`, and accepts registered catalog, policy,
  preparation, redaction, and approval callbacks as explicit construction
  inputs.
- **Breaking:** `BuildChatRuntime` is now one protocol: four positional-only
  arguments `(session, request, channel, auth_manager)` followed by the
  keyword-only `storage_root`. Signature sniffing and the legacy
  keyword-call variants were removed.
- The product MCP environment-passthrough table is now an immutable mapping
  of `frozenset` values. `McpClientManager` and startup accept
  `Mapping[str, AbstractSet[str]]`; other manager collection inputs accept
  abstract sets and mappings, including `Mapping[str, float]` timeout
  overrides.
- Public annotations now accept abstract excluded-tool sets, iterable state
  server collections, cached prompt blocks in `AgentSDKRunner.run()`, and
  arbitrary input to `WorkflowTaskMetadata.from_payload()`.
  `VerifiedWorkAuthorization.billing_mode` and ordinary autonomous authority
  kind expose their enforced literal vocabularies, while `parse_effort()`
  publishes return-type overloads.
- `product_config.gateway_product_id()` now reads `PRODUCT_ID` on every call.
  It is no longer LRU-cached, so consumers must not call
  `gateway_product_id.cache_clear()`.
- Dispatch applies the provider-request advertised-name snapshot before
  policy, schema validation, and interceptors for both local and MCP tools.
  A name absent from the snapshot returns `tool_not_advertised`; an advertised
  name with no live route remains `unknown_tool`, and a missing live local
  definition or input schema returns `tool_schema_unavailable`.
- Tool preparation, approval, redaction, retry eligibility, dispatch,
  settlement, and source extraction now derive from the exact registered
  local or MCP route. Child and resumed runs preserve the admitted route and
  provider-definition snapshots rather than rebuilding them from a live
  catalog.
- Autonomous execution is progress-bound: `timeout_seconds` remains accepted
  but no longer cancels LLM work, and the default concurrent background-task
  ceiling is six. Turn, token, budget, and event-gap limits remain active.
- Durable approval requests now carry `approval_reuse_mode` and an optional
  exact reuse key. Existing rows use the legacy path; registered requests
  disable reuse without a safe key and cannot turn one-time or mismatched
  grants into reusable authority.
- `run_agent` operation descriptions no longer include non-authoritative
  grant previews. When a host defers a local tool, `tool_not_advertised`
  identifies its loadable pack when exactly one pack owns it.
- The autonomous admission ledger is now the launch-nonce store with schema
  version 2. Its durable schema no longer embeds the transient launch-envelope
  audience, and the public store types and functions use `LaunchNonce*`
  names.
- Static type checkers now resolve the lazy root exports `create_agent`,
  `create_gateway_app`, `ChatRuntime`, `ChatTurnInputs`, `ChatTurnResult`,
  `GatewayServerConfig`, `RequestContext`, and `SessionExecutionPolicy`
  instead of treating them as `Any`; runtime lazy loading is unchanged.
- `PreparedBusinessModelChange.approval_id` is required, and dispatcher
  definition callbacks now return concrete lists of definition dictionaries.

### Fixed

- Terminal settlement now drains receipt, named-skill, and detach writes together
  under repeated cancellation without blocking other sessions. Learning receipts
  are acknowledged only after persisted success; error fallback appends failure
  history without discarding earlier durable records or staged evidence.
- Publish ordinary child settlements even when budget exhaustion or another
  terminal failure leaves no canonical content. Foreground and background
  completions preserve the execution reason and available evidence instead of
  losing the result to parent materialization.
- Durable multi-turn replay now keeps a tool batch's results together before
  source annotations. Interrupted native runs persist terminal error results
  for unanswered tool calls, including calls not yet dispatched, before
  releasing the session writer; attach recovery applies the same settlement.
- Source checkouts now report the adjacent `pyproject.toml` package version in
  health payloads; installed wheels continue to use distribution metadata.
- Autonomous event-channel JSON accepts shared, non-cyclic container objects
  while continuing to reject genuine circular references.
- Control-run message delivery resolves exactly one chat or autonomous owner,
  and another user's autonomous run remains indistinguishable from a missing
  run with a 404 response before payload, token-kind, or state checks.
- Autonomous `research_producer` children now receive their
  `RESEARCH_PRODUCER_*` settings, `LOCAL_GATEWAY_CONTROL_STATE_ROOT`, and the
  production claim-launcher `GATEWAY_URL`.
- Per-user Risk MCP calls receive only a positive Risk user id in `meta.user_id`;
  non-positive identities are omitted or refused in strict mode.
- Workspace-less parents can spawn and resume siblings without constructing
  `Path(None)`, and inline completion envelopes are published even when no
  durable workspace exists.
- Recoverable FMS judgment errors are no longer treated as terminal solely
  because their envelope also carries `gate_code=STOP`; bounded
  `error.data.got` details are preserved for the repair turn.
- MCP readiness now derives required tools only from configured transports
  and their actual `ListTools` exports. Optional or mode-specific classified
  tools no longer make a healthy runtime report an incomplete catalog.
- Claim-signing secret transfer now uses the portable non-inheritable
  `os.pipe()` contract, so autonomous claim launch works on macOS as well as
  Linux while exposing only the intended read descriptor to the child.
- Named child runs settle immediately from an accepted terminal tool result,
  preserve the deterministic projection across resume, and do not repeat the
  provider call after a later budget stop.
- Session-log inventory ignores foreign `.jsonl` files in a segment directory
  instead of rejecting every valid segment in that stream.

### Removed

- Removed the root and `agent_gateway.retry` exports `RetryConfig`,
  `classify_outcome`, and `run_autonomous_with_retry`, along with
  `SkillProfile.max_retries`. Autonomous and standalone skill runs are
  single-attempt; bounded retry remains at registered idempotent/read tool
  dispatch.
- Removed the replay-oriented MCP activation exports
  `MCP_SERVER_ACTIVATED_EVENT`, `ACTIVATION_SOURCE_STAGE_SCOPE`,
  `mcp_server_activated_event()`, and `fold_mcp_activations()`. Named-operation
  activation now returns the typed `RunAgentMcpActivationError` contract.
- Removed the retired `HANK_FORK_BUDGET_USD` and `HANK_FORK_MAX_TURNS`
  controls and their `DEFAULT_FORK_*`, `fork_budget_default()`, and
  `fork_max_turns()` exports.

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
