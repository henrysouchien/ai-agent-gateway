# Plan: Canonical agent-gateway dev clients (CLI + TUI)

**Status:** v4-final, **Codex PASS (round 10)**
**Owner:** hc@henrychien.com
**Date:** 2026-05-01
**Supersedes:** `finance_cli/docs/planning/PLAN_CASHNERD_DEV_TUI.md`
**Canonical home:** `AI-excel-addin/packages/agent-gateway/docs/planning/PLAN_AGENT_GATEWAY_DEV_CLIENTS.md`
**Recon worksheet:** [recon-agent-gateway-dev-clients.md](./recon-agent-gateway-dev-clients.md)

**Review trail:** R1 FAIL(9) → v2. R2 FAIL(10) → v3. R3 FAIL(5) → R4 FAIL(3) → R5 FAIL(1+2 stale) → **R6 PASS** (v3-final). Phase 0 recon executed → 6 plan-revision items → v4. R7 FAIL(6 mech) → R8 FAIL(3 stale) → R9 FAIL(3 stale) → **R10 PASS** (v4-final). Diff vs. each version at end of doc.

## Problem

Three near-identical reference clients of the agent-gateway protocol exist in product repos today:

| Surface | Location | LOC | Owner repo |
|---|---|---|---|
| Hank CLI | `AI-excel-addin/api/dev/chat_cli.py` | 1118 | AI-excel-addin |
| CashNerd CLI | `finance_cli/finance_cli/dev/chat_cli.py` | 1082 | finance_cli |
| Hank TUI | `AI-excel-addin/tui/src/` (TS, pi-tui) | ~2800 | AI-excel-addin |

All three are clients of the same `/api/chat/init` + `/api/chat` + `/api/chat/tool-approval` protocol exposed by `ai-agent-gateway`. The two CLIs are ~80% identical scaffolding with already-visible micro-divergence. The TUI exists only in Hank but its protocol layer is product-agnostic.

Symptom on entry: CashNerd has no rich dev TUI, blocking terminal-based dev/test. Wrong fix would be to build a CashNerd-specific TUI; correct fix is to extract canonical clients into the protocol owner's monorepo.

## Decision

Extract canonical dev clients into the `ai-agent-gateway` monorepo (`AI-excel-addin/packages/`) as siblings of the existing gateway package:

```
AI-excel-addin/packages/
├── agent-gateway/            (existing, Python, the protocol owner)
├── ...other MCP/utility packages...
├── agent-gateway-cli/        (NEW — Python CLI client)
└── agent-gateway-tui/        (NEW — Node/TS TUI client)
```

Both shipped product-agnostic. Hank and CashNerd consume them. Product-specific behavior (slash commands like Hank's modes, default skill names) extends the TUI through a **typed extension-hook interface**, not a JSON config schema.

**Honest framing:** drift doesn't become impossible. It moves. New drift surfaces are: package version skew, session-schema skew, product shim drift, release coordination across PyPI/npm. The plan addresses each with explicit policy below. Net win is replacing 3 forks with 2 versioned canonical packages and a single conformance test suite.

## Goals

- One Python CLI codebase consumed by Hank + CashNerd + future agent-gateway products.
- One Node TUI codebase consumed by the same.
- Net code reduction (~3900 LOC duplicated → ~3000 LOC canonical).
- Existing `agent-gateway/docs/http-api.md` becomes the wire-contract source of truth, **extended** in this plan to cover currently-undocumented fields used in production (Phase 0.5).
- Product-specific TUI commands plug in through a typed interface, defined now, not deferred.

## Non-goals

- Not changing the agent-gateway runtime/protocol semantics (only documenting what already exists).
- Not building a Python TUI (decision: pi-tui Node port).
- Not extending the agent-gateway-dist mirror to the new packages (Python core only). New packages publish to PyPI/npm directly. NOTE: v4 DOES add a new CI workflow (`agent-gateway-tests.yml`) per Recon 0.8 — required because no test CI exists in either AI-excel-addin or agent-gateway-dist today; this is a Phase 4 prerequisite for the conformance gate to function, not a "non-goal" exception.
- Not adding gateway-side compaction/reflection endpoints.
- Not Windows packaging (Mac + Linux only).
- Not back-compat shims for old import paths in product repos. Clean cut, deprecation in CHANGELOG.

---

## Phase 0 — Reconnaissance (executable, must complete before Phase 0.5)

Each item is a hard yes/no the plan depends on. Recon work outputs go into a worksheet committed alongside the plan.

| # | Question | How to answer | Drives |
|---|---|---|---|
| 0.1 | Is `AI-excel-addin/packages/agent-gateway/` the canonical source for `ai-agent-gateway` v0.8.1? | Already confirmed in session: yes, MIT, Henry Chien LLC. | Repo placement. |
| 0.2 | Does `/api/chat/init` handler READ `user_id` (route, auth, log on it) or ACCEPT-AND-IGNORE / reject? | Read `agent_gateway/server.py` init handler; check pydantic model on init request AND any downstream code that references `request.user_id`. | Whether `user_id` is added to `/init` spec + sent by clients (READS), or omitted (IGNORES/rejects). Per Phase 0.5a single rule. |
| 0.3 | Does `/api/chat/init` response emit `model_catalog`? | Same handler audit + spot-check existing tests in `agent-gateway/tests/`. | Whether `/model` command works against allowlist; whether spec needs the field. |
| 0.4 | Does `tool_approval_request` event emit an `expires_at` field? CashNerd CLI reads it; spec doesn't list it. | grep `expires_at` in `agent_gateway/tool_dispatcher.py`; check what gets serialized into the SSE payload. | Whether approval countdown UX stays or gets removed. |
| 0.5 | Does the gateway/server route or analyze on `context.channel` value? Specifically, is `"telegram"` (current CashNerd value from a CLI source) load-bearing anywhere? | grep `context.channel\|context\["channel"\]\|channel ==` across `agent-gateway/`, `finance-web/`, finance_cli, AI-excel-addin server-side. | Whether `channel="cli"` swap is a clean change or breaks routing. |
| 0.6 | pi-tui v0.62.0 — license + last-publish-date check. | npmjs page for `@mariozechner/pi-tui`. | Vendor or peer-dep. |
| 0.7 | What dev-loop pattern do existing `packages/` siblings (browser-mcp, excel-mcp, etc.) use for local linkage? | Read each package's `pyproject.toml` / `package.json` and product consumer's pin. | Local install pattern for new packages. |
| 0.8 | What CI runs against `agent-gateway/tests/` today? | Read `.github/workflows/` (or equivalent) in AI-excel-addin. | Where conformance tests run. |
| 0.9 | Hank TUI — any non-protocol Excel deps in protocol layer? Spot-check `tui/src/backend-client.ts` and `event-adapter.ts`. | Already partly verified in session; finish read. | Confirm clean port. |
| 0.10 | Does `/api/chat` handler READ `user_id` (route, auth, log on it) or ACCEPT-AND-IGNORE / reject? | Read `agent_gateway/server.py` chat handler; check pydantic model on chat request AND any downstream code that references `request.user_id`. Today's CashNerd CLI sends it on `/chat`; need to confirm what gateway does. | Whether `user_id` is added to `/chat` spec + sent by clients (READS), or omitted (IGNORES/rejects). Per Phase 0.5a single rule. |
| 0.11 | Exact on-disk session/config shape in CashNerd + Hank today (so v0 migration loader handles both correctly). | Read `~/.cache/cashnerd/sessions/*.json` shape (already known via `chat_cli.py:_load_session`); read Hank's equivalent. | Confirms v0→v1 loader handles both. |
| 0.12 | Hank reflection feature deps: what does `buildReflectMessages` reference, what does the `memory_write` tool path do, what's in `reflections/`? | Read `tui/src/history.ts` (`buildReflectMessages` def), read Hank's memory_write tool implementation, list `reflections/` contents. | Validates Phase 2 extension hook covers reflection completely. |
| 0.13 | PyPI `ai-agent-gateway-cli` and npm `@ai-agent-gateway/tui` name availability. | `pip index versions ai-agent-gateway-cli` (or PyPI search); `npm view @ai-agent-gateway/tui`. | Pick alternates if taken. |
| 0.14 | Existing imports/docs/scripts referencing soon-deleted entrypoints (`finance_cli.dev.chat_cli`, `api.dev.chat_cli`, `tui/`). | `grep -rn 'finance_cli.dev.chat_cli\|api.dev.chat_cli\|tui/' --include='*.py' --include='*.ts' --include='*.md' --include='*.sh'` in both products. | Phase 3 task list — every reference needs updating. |

**Recon worksheet output:** `docs/planning/recon-agent-gateway-dev-clients.md`. Recon must commit before Phase 0.5 starts. If any answer surprises, plan-revision PR.

---

## Phase 0.5 — Protocol spec extension + schema freeze (BLOCKING for all later phases)

Codex round 1 correctly flagged that the canonical clients can't assert behavior `http-api.md` doesn't document. This phase reconciles before any extraction.

### 0.5a — Extend `agent-gateway/docs/http-api.md`

Edit the spec to document fields that exist in production but were previously implicit. **Each edit gated on Phase 0 recon answer.** All field additions require gateway-side confirmation; the spec records what the server actually accepts/emits, not aspirational behavior.

- **`user_id` on `/api/chat/init` and `/api/chat`** — single rule applied everywhere:
  - Phase 0.2 audits `/init` handler: does it READ `user_id` (route/auth/log on it) or merely ACCEPT-AND-IGNORE?
  - Phase 0.10 audits `/chat` handler: same question.
  - **If handler READS the field:** spec extension documents it on that endpoint with precise semantics (what gateway does with it); canonical clients send when configured.
  - **If handler ACCEPTS-AND-IGNORES (or rejects):** spec does NOT document the field on that endpoint; canonical clients DO NOT send. (Sending a server-ignored field encodes no behavior into the canonical contract — Codex round-2 line.)
  - The two endpoints are decided independently. It's possible `/init` reads `user_id` while `/chat` ignores it, or vice versa. Each gets its own send/skip decision based on its handler audit.
  - This single rule is referenced from Phase 1 (Python CLI behavior) and Phase 2 (TUI behavior); no client sends or documents `user_id` outside this rule.
- **`/api/chat/init` response:** add `model_catalog` (optional object) IF Phase 0.3 confirms emission. **Single behavior across plan:** clients tolerate absence and blind-forward `model` name to gateway; `/model` slash command falls back to displaying current model + accepting any string when catalog absent. No "remove handling" branch — handling is conditional on field presence.
- **`tool_approval_request` event:** Recon 0.4 confirmed `expires_at` is NOT emitted. Existing CashNerd CLI countdown UX is dead code. Action: REMOVE approval-countdown UX from merged CLI (no fallback timer, no synthetic deadline). Recon 0.4 ALSO surfaced two emitted-but-undocumented fields: `reason` (string, optional context for why approval is needed) and `allow_persistent_approval` (bool, whether the gateway will accept `allow_tool_type=true` for this approval). Phase 0.5a adds BOTH to the documented event schema in `http-api.md`. Clients render `reason` in the approval prompt; `allow_persistent_approval` gates whether the prompt offers the "approve all of this tool type" option.
- **`context.channel`:** add a "recommended values" subsection: `web`, `cli`, `telegram`, `bot`, plus convention note: "free-form, surfaces source for analytics or routing." Mark `cli` as the canonical value for both dev surfaces (the agent-gateway-cli + agent-gateway-tui packages). Phase 0.5 separately audits whether existing `"telegram"` value from CashNerd CLI is load-bearing in any gateway/server route (recon item 0.5).
- **`stream_complete` `usage` schema:** confirm fields the merged CLI's `_usage_int` / `_usage_float` helpers consume (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `estimated_cost`); spec already has these but verify completeness.

### 0.5b — Freeze SSE event-handling policy

Every documented event gets explicit per-client policy. `http-api.md` documents 22 event types after the Phase 0.5 follow-up; the table covers all 22 plus an unknown-event fallback. **Render policy specifies the exact adapter output** so conformance tests can assert it; vague descriptions like "render warning" are replaced with structured-event names.

Adapter output is a normalized event the test harness asserts on. Each client adapter (`event-adapter.ts` for TUI; `sse.py` dispatch for CLI) emits these:

- `assistant_text_append({text})` — append to current assistant message
- `thinking_text_append({text})` — append to thinking buffer (TUI renders; CLI debug-logs)
- `tool_start({tool_call_id, tool_name, tool_input})` — open tool-execution UI
- `tool_chunk({tool_call_id, stream, text, seq})` — append to tool stream pane (TUI); debug-log (CLI)
- `tool_complete({tool_call_id, tool_name, result, error, duration_ms})` — close tool UI, record for history summary
- `tool_interrupted({tool_call_id, tool_name, tool_input, message})` — close open tool UI with error state (TUI); debug-log only (CLI)
- `tool_approval_needed({tool_call_id, nonce, tool_name, tool_input, resolved_qualifier, reason, allow_persistent_approval})` — render approval prompt; block until user responds. `reason` rendered as context line; `allow_persistent_approval` gates the "approve all of this type" option in the prompt UI
- `interceptor_warning({tool_call_id, action, code, message})` — render diagnostic warning line
- `interruption_warning({reason, runner_id, role, last_completed_seq})` — render runner interruption/recovery diagnostic line (TUI); debug-log only (CLI)
- `headless_auto_deny_warning({tool_call_id, tool_name, reason, source})` — render headless auto-deny diagnostic line (TUI); debug-log only (CLI)
- `stream_retry_notice({attempt, error})` — render diagnostic retry line
- `compaction_notice({chars})` — render compaction line
- `terminal_max_turns({turn_count, max_turns})` — terminal: stop loop, render message
- `terminal_budget_exceeded({total_cost, budget})` — terminal: stop loop, render message
- `terminal_complete({usage})` — terminal: success path
- `terminal_error({error, source})` — terminal: error path; `source` distinguishes `error` vs `stream_error`
- (no normalized output for `heartbeat` — silent keep-alive)
- (no normalized output for `turn_complete`, `task_registered`, `task_completed`, or `parent_message_sent` by default — debug-log lifecycle details; `task_registered` and `task_completed` may be surfaced only by an explicit verbose lifecycle mode)
- (no normalized output for unknown events — debug-log raw payload, never crash)

Mapping table (22 documented events + unknown), including the Phase 0.5 follow-up events surfaced by gateway 0.13.1:

| Wire event (`type`) | Normalized output | Terminal? | Notes |
|---|---|---|---|
| `text_delta` | `assistant_text_append` | No | Hot path |
| `thinking_delta` | `thinking_text_append` | No | TUI renders pane; CLI debug-logs |
| `tool_call_start` | `tool_start` | No | Both clients |
| `tool_call_complete` | `tool_complete` | No | Both clients |
| `tool_call_interrupted` | `tool_interrupted` (TUI); debug-log only (CLI) | No | Phase 0.5 follow-up; synthesized after recovery |
| `tool_output_chunk` | `tool_chunk` | No | TUI renders stream; CLI debug-logs |
| `tool_approval_request` | `tool_approval_needed` | No | Blocks until POST /tool-approval. Per Recon 0.4: NO `expires_at` field — no countdown UX, no synthetic deadline. Includes `reason` + `allow_persistent_approval` (per Phase 0.5a additions) |
| `headless_auto_deny` | `headless_auto_deny_warning` (TUI); debug-log only (CLI) | No | Phase 0.5 follow-up; headless approval auto-deny |
| `interceptor_decision` | `interceptor_warning` | No | Both clients |
| `turn_complete` | (no output) | No | Phase 0.5 follow-up; per-turn lifecycle usage, debug-log only |
| `interrupted` | `interruption_warning` (TUI); debug-log only (CLI) | No | Phase 0.5 follow-up; runner recovery/interruption signal |
| `stream_retry` | `stream_retry_notice` | No | Both clients |
| `compaction` | `compaction_notice` | No | Both clients |
| `max_turns_reached` | `terminal_max_turns` | **Yes** | Both clients |
| `budget_exceeded` | `terminal_budget_exceeded` | **Yes** | Both clients |
| `task_registered` | (no output) | No | Phase 0.5 follow-up; background-task lifecycle, silent unless verbose lifecycle mode |
| `task_completed` | (no output) | No | Phase 0.5 follow-up; background-task lifecycle, silent unless verbose lifecycle mode |
| `parent_message_sent` | (no output) | No | Phase 0.5 follow-up; sub-agent internal coordination |
| `heartbeat` | (no output) | No | Keep-alive only |
| `stream_complete` | `terminal_complete` | **Yes** | Both clients |
| `error` | `terminal_error` (source=error) | **Yes** | Both clients |
| `stream_error` | `terminal_error` (source=stream_error) | **Yes** | Both clients |
| _unknown event_ | (debug-log raw payload) | No | Forward-compat; never crash |

Conformance tests assert: emit each wire event → adapter emits the corresponding normalized output (or no output for silent lifecycle/heartbeat/unknown), terminal events stop the loop, known follow-up events do not hit the unknown-event path, and unknown events don't crash.

### 0.5c — Freeze shared config + session JSON schemas

These freeze before either client package extracts code. Both clients write/read this exact schema; conformance tests assert it.

**`cli_config.json`** (mode 0600):
```jsonc
{
  "schema_version": 1,
  "gateway_api_key": "string",
  "user_id": "string",          // optional in config; clients send per the Phase 0.5a `user_id` rule (only on endpoints whose handler READS it)
  "base_url": "string",         // validated per Phase 1 policy
  "config_namespace": "string"  // e.g., "cashnerd", "excel-addin" — drives default file paths
}
```

**`sessions/<name>.json`** (mode 0600):
```jsonc
{
  "schema_version": 1,
  "name": "string",
  "created_at": <unix int>,
  "updated_at": <unix int>,
  "messages": [
    {"role": "user|assistant|system", "content": "string"}  // system role added per protocol
  ]
}
```

**v0 → v1 migration story (specified now, not deferred):**

Recon 0.11 surfaced TWO distinct v0 shapes — Hank's session file is a flat JSON array, not a dict. Loader handles both:

- **Hank v0 shape** (`tui/data/chat_history.json`): a JSON array `[{role, content}, ...]`. No envelope, no `schema_version`, no per-session naming. Single shared file at `tui/data/`.
- **CashNerd v0 shape** (`~/.cache/cashnerd/sessions/<name>.json`): a JSON object `{name, created_at, updated_at, messages: [{role, content}]}`. No `schema_version`. Per-session named files. `role ∈ {user, assistant}`.

**Loader rules** (parse-error handling first, then top-level type discrimination):
1. **File doesn't exist or is empty (zero bytes)** → return new empty session. No quarantine, no error.
2. **JSON parse fails (malformed file)** → quarantine (rename to `<file>.corrupt.<timestamp>.json`) and return new empty session. Matches Hank's existing quarantine pattern.
3. **Parsed value is a JSON Array** → Hank v0. Default `name` to the file basename (or `"default"` if loaded directly without a name), use file mtime for both `created_at` and `updated_at`, take messages as-is. Wrap in v1 envelope on save.
4. **Parsed value is a JSON Object missing `schema_version`** → CashNerd v0. Promote in place, accept existing `role ∈ {user, assistant}`, set `schema_version: 1` on next save.
5. **Parsed value is a JSON Object with `schema_version: 1`** → v1. Accept `role ∈ {user, assistant, system}`.
6. **Parsed value is `null` / a JSON primitive / an Object with unknown `schema_version`** → quarantine and return new empty session.

The rules are checked in order; the first match wins. Rules are non-overlapping by construction (rule 3 = Array; rules 4–5 = Object; rule 6 = catchall for primitives and future-version files).

v1 reader accepts both v0 shapes transparently. v0-only readers (older client versions) cannot read v1 files (`role: "system"` will error their validator). Acceptable: clients pin via peer-dep range; mismatched-version reads are out-of-scope for dev tooling.

Schema-version bumps beyond v1 require explicit migration code, documented in CHANGELOG.

**Concurrent writes:** last-writer-wins, documented; no locking attempted. CLI and TUI sharing same `--session` name in parallel is the only realistic conflict path; an advisory `.lock` sentinel file is parked for v1.1 if it bites in practice (per round-2 question 8).

### 0.5d — Codex review of Phase 0.5 outputs before any extraction

Phase 0.5 deliverables (extended http-api.md + SSE policy table + frozen schemas) go to Codex for a focused review before Phase 1 starts. This catches contract gaps Codex round 1 flagged.

---

## Phase 1 — Extract Python CLI to `packages/agent-gateway-cli/`

Best-of-both merge of Hank's + CashNerd's `chat_cli.py`. Concrete structural decisions resolving Codex round-1 traps:

### Behavior-preserving merges (no widening)

- **`_validate_base_url`:** keep CashNerd's strict default (`http://127.0.0.1` only, no path/query). Hank's HTTPS-on-localhost is the unusual case — Hank's consuming product overrides via `BaseUrlPolicy(allowed_schemes={"http","https"}, allowed_hostnames={"127.0.0.1","localhost"})` constructor injection. **No widening of defaults.** Conformance test asserts default policy.
- **Session message roles:** extend to `{"user", "assistant", "system"}` per protocol. Migration: existing CashNerd session files (only user/assistant) remain valid.
- **Approval countdown UX:** REMOVED. Recon 0.4 confirmed `expires_at` is NOT emitted on `tool_approval_request`. CashNerd's existing countdown reads a missing field and renders "unknown" — dead code. No fallback timer, no synthetic deadline. Replaced with `reason` rendering and `allow_persistent_approval` gating per Phase 0.5a.
- **`context.channel`:** clients send `"cli"` (per Phase 0.5a canonical mark). CashNerd's existing `"telegram"` value gets audited in Phase 0.5 (`grep` recon item 0.5); if load-bearing, add a back-compat `--legacy-channel telegram` flag for one release. If not, clean swap.
- **`user_id` field:** sent per the single Phase 0.5a rule — only on endpoints where the handler READS the field (per Phase 0.2 / Phase 0.10 audit). No other behavior. Clients do NOT send `user_id` on endpoints whose handler ignores it.

### Adopt from CashNerd
- Named-session subsystem: `--session`, `--new`, atomic tempfile writes, corrupted-file detection.
- `_cache_root` honoring `XDG_CACHE_HOME`.
- `_validate_session_name` regex.

### Adopt from Hank
- `_truncate` helper.
- `_usage_int` / `_usage_float` helpers for usage display from `stream_complete.usage`.
- `logging.getLogger("dev.chat_cli")` integration.

### Productize: config-namespace flag

`python -m agent_gateway_cli --config-namespace cashnerd login` writes to `~/.cache/cashnerd/cli_config.json`. Default namespace is `agent-gateway`. Each consuming product picks its own.

### SSE handlers added per Phase 0.5b table

CLI's existing event handler covers ~6 events. Phase 1 adds: `thinking_delta` (log-only debug), `tool_output_chunk` (log-only debug), `interceptor_decision` (render warning), `stream_retry` (render diagnostic), `compaction` (render notice), `max_turns_reached` + `budget_exceeded` (render terminator + treat as terminal).

### Package layout
```
packages/agent-gateway-cli/
├── agent_gateway_cli/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py        (CLIConfig, BaseUrlPolicy, namespace resolution)
│   ├── session.py       (SessionState + load/save, schema_version=1)
│   ├── transport.py     (httpx wrapper, init/chat/approval)
│   └── sse.py           (parse_sse_events + per-event handler dispatch)
├── tests/
│   ├── test_contract.py (asserts against Phase 0.5 frozen contract; mock gateway)
│   ├── test_session.py
│   └── test_config.py
├── pyproject.toml       (name=ai-agent-gateway-cli, dep on httpx; peer-version range vs ai-agent-gateway)
└── README.md
```

**Versioning:** independent semver. Pyproject declares `Requires-Dist: ai-agent-gateway>=0.8,<0.9` (peer-dep range). Bumping gateway minor = bumping client peer range. Lockstep is rejected; client bumps independently when client logic changes.

### Test coverage transfer

`finance_cli/tests/test_dev_chat_cli.py` covers ~330 lines of CashNerd CLI behavior; AI-excel-addin has equivalents. Both port into `tests/test_contract.py` and `tests/test_session.py` in the new package, deduplicated. Net coverage measured before deleting product copies (must not regress).

---

## Phase 2 — Extract Node TUI to `packages/agent-gateway-tui/`

Lift from `AI-excel-addin/tui/`. ~2.8K LOC, well-structured.

### Direct ports (config-only changes)

`backend-client.ts`, `stream-assembler.ts`, `event-adapter.ts`, `formatters.ts`, `tool-summary.ts`, `tool-display.ts`, `components/*`, `theme/*`, `utils/*`. Verify each event in the adapter against Phase 0.5b SSE policy; add missing handlers (parallel with CLI Phase 1 SSE additions).

### Rework: typed command extension hook with explicit sandbox contract

Codex round 1 correctly flagged JSON config as premature/underpowered. Round 2 correctly flagged the v2 hook as missing capabilities (reflection needs `getHistory`, `streamChat`, `setStatus`) and as having vague sandbox semantics (`writeFile` "sandboxed" was just a comment; `callBackend` had no constraints). v3 expands the interface to cover reflection (validated against Hank's actual `runReflectCommand`) and replaces freeform `callBackend` with the specific primitive reflection needs (`streamChat`):

```typescript
// agent-gateway-tui/src/extensions.ts

export type ChatMessage = { role: "user" | "assistant" | "system"; content: string };
export type Status = "idle" | "sending" | "streaming" | "compacting" | "error";

// Grouped sub-namespaces so the public surface is reviewable and the
// extension API stays factored as it grows.
export type CommandHandlerContext = {
  ui: {
    getStatus: () => Status;
    setStatus: (s: Status) => void;
    prompt: (label: string) => Promise<string>;
    promptChoice: <T extends string>(label: string, choices: ReadonlyArray<T>) => Promise<T>;
    showFeedback: (text: string, opts?: { isError?: boolean }) => void;
  };

  model: {
    get: () => string;
    set: (m: string) => void;
  };

  // Request-time `context.*` payload included on the next gateway call.
  requestContext: {
    get: () => Readonly<Record<string, unknown>>;
    set: (patch: Record<string, unknown>) => void;
    clear: (keys?: string[]) => void;
  };

  history: {
    get: () => ReadonlyArray<ChatMessage>;
    append: (msg: ChatMessage) => void;
    truncate: () => void;
  };

  session: {
    getId: () => string | null;
    switch: (name: string) => Promise<void>;
    invalidate: () => void;
  };

  // Out-of-band gateway calls. Used by reflection/compaction/extension turns
  // that must not mutate main history. Extension owns event consumption AND
  // owns whether to mutate history via ctx.history.append.
  gateway: {
    streamChat: (
      messages: ReadonlyArray<ChatMessage>,
      opts?: { context?: Record<string, unknown>; signal?: AbortSignal; timeoutMs?: number },
    ) => AsyncIterable<BackendEvent>;
  };

  // ctx.fs.* (sandboxed filesystem) is DEFERRED to v0.2.
  // Recon 0.12 confirmed Hank's reflection feature does NOT require fs primitives —
  // reflection is implemented entirely via ctx.history + ctx.gateway.streamChat plus
  // server-side memory_write tool. No current first-party use case needs fs in v0.1.
  // Sandbox contract is preserved below as v0.2 reference; if a future extension
  // needs it, the v0.1 → v0.2 minor bump adds the ctx.fs.* sub-namespace per that spec.
};

export type CommandHandler = (
  args: string,
  ctx: CommandHandlerContext,
) => Promise<{ feedback?: string; isError?: boolean; shouldExit?: boolean }>;

export type CommandRegistration = {
  name: string;
  description: string;
  handler: CommandHandler;
};

export interface TuiExtension {
  registerCommands(): CommandRegistration[];
}
```

**Sandbox contract (DEFERRED to v0.2 — preserved below as reference):**

This spec was developed across Codex rounds 3–6 and is solid. It does not ship in v0.1 because Recon 0.12 confirmed no current first-party use case needs `ctx.fs.*`. If/when a future extension needs filesystem access, the v0.1 → v0.2 minor bump adds the `ctx.fs.*` sub-namespace per the spec below. Conformance tests for `ctx.fs.*` are deferred to v0.2.

The TUI captures the consuming product's working directory at startup (`workdirRoot = realpathSync(process.cwd())`) and treats it as the immutable sandbox root. The contract is **footgun prevention, not adversarial isolation** — extensions are first-party product code. The rules below catch accidental escapes (typos, bad relative paths, symlinks) but are not designed to resist a malicious extension.

For each fs primitive (`writeWorkdirFile`, `readWorkdirFile`, `listWorkdir`):

1. **Reject absolute paths AND any `..` segment in the raw input.** Reject if `path.isAbsolute(relPath)` or if `relPath.split(/[\\/]/).includes("..")` (the character class splits on both `/` and `\` BEFORE any normalization, since `path.normalize` collapses `a/../b` to `b` and would defeat the check). Throw `ExtensionSandboxError`. This is a lexical check on the raw input before any filesystem touch.
2. **Resolve target path:** `target = path.join(workdirRoot, relPath)`.
3. **Canonicalize the target's PARENT directory:** `parentReal = realpathSync(path.dirname(target))`. This step resolves any symlinks in ancestors. If `dirname` doesn't exist yet (writes only), walk upward to the nearest existing ancestor and canonicalize that.
4. **Reject if canonical parent escapes workdir:** if `!parentReal.startsWith(workdirRoot + path.sep)` and `parentReal !== workdirRoot` → throw `ExtensionSandboxError`. This catches symlinked parent directories pointing outside.
5. **For `readWorkdirFile`:** also canonicalize the target file itself if it exists; if `realpath(target)` escapes workdir, reject. (Symlinks pointing outside are rejected even if the parent is inside.)
6. **For `listWorkdir`:** also canonicalize the target directory itself; if `realpath(target)` escapes workdir, reject. (Listing through a symlink that points outside workdir is rejected even if the parent is inside.)
7. **For `writeWorkdirFile`:** parent directories created as needed (mode 0755). File written atomically via tempfile + rename (mode 0644). If target is an existing symlink, the implementation must EXPLICITLY decide rename-vs-follow semantics: POSIX `rename()` over a symlink REPLACES the symlink rather than following it, so the implementation calls `realpath(target)` first and writes to the canonical path (after the parent-escape check from step 4 + the symlink-target-escape check below). If the symlink's canonical target is outside workdir, reject. This avoids accidental symlink-replacement footguns.
8. **Errors caught and surfaced via `showFeedback({isError: true})`** in the registry-level handler wrapping the extension call. Extensions do NOT need to catch sandbox errors themselves.

**Out of scope (footgun-prevention scope):** TOCTOU races between the canonicalization check and the write are not defended against; first-party extensions have no incentive to race themselves. Hardlink-based escapes are not blocked. If adversarial isolation becomes a real requirement, that's a separate plan.

- `streamChat(messages, opts)`: uses the same auth session as the main TUI loop. Issues `POST /api/chat` with provided messages and merged `context` (request-time context from `ctx.requestContext.get()` overlaid with `opts.context`). Returns the SSE event AsyncIterable; extension owns event consumption. Default timeout 180s, override via `opts.timeoutMs`. **Does not mutate main history** — extension calls `ctx.history.append` if it wants the turn recorded.

**Extension scope for v0.1:** the extension model is limited to **chat-turn extensions and slash commands that mutate UI/session/context state**. Extensions that need non-`/api/chat` backend calls (custom analytics endpoints, direct tool-result submission, etc.) are explicitly out of scope for v0.1. If a future extension surfaces this need, the plan adds a constrained primitive (e.g., `ctx.gateway.fetch(path, opts)` with an allowlist of permitted paths) rather than restoring a freeform `callBackend`.

**Validation against Hank's reflection** (per Phase 0.12 recon):
- `runReflectCommand` checks `isStreaming` → uses `ctx.ui.getStatus()`.
- Refuses if `historyLength < 5` → uses `ctx.history.get().length`.
- Sets status to "compacting" → uses `ctx.ui.setStatus("compacting")`.
- Builds reflect messages (`buildReflectMessages(history)`) → extension imports its own helper; uses `ctx.history.get()`.
- Streams reflection prompt against gateway (`streamCompactionText`) → uses `ctx.gateway.streamChat(reflectMessages, {timeoutMs: 120_000})`.
- Reflection writes to `reflections/{date}.md` via the `memory_write` tool inside the reflection turn — that's a server-side tool dispatch during the streamed turn, NOT an extension capability. Extension just streams the turn; tool execution happens server-side.
- Restores status to "idle" → `ctx.ui.setStatus("idle")`.

This covers the reflection use case without inventing speculative capabilities. Hank ships reflection as ~80 lines of TS in its repo using this interface.

**Trust model:** extensions are first-party product code (Hank's own TS, CashNerd's own TS), not third-party plugins. Sandbox is for accidental footgun avoidance (no `..` traversal), not for adversarial isolation. `streamChat` uses the same auth as the main loop because there's no privilege boundary to enforce — extensions could fetch the bearer token from `BackendClient` directly if they wanted, the typed surface just makes the common path safe.

**API stability promise (hard decision):** `TuiExtension` and `CommandHandlerContext` are part of `@ai-agent-gateway/tui`'s public API.

- **v0.x band (initial release through v0.x):** breaking changes ALLOWED in minor bumps. Following standard semver pre-1.0 convention. Extension authors pin to specific minor versions.
- **v1.0 onward:** breaking changes require major bump. New fields can be added in minor versions (TS catches missing-field errors at consumer compile time).
- Either way, every breaking change is documented in `CHANGELOG.md` with a migration note.
- Conformance tests pin the exact public surface and fail on accidental signature drift, regardless of band.

Hank ships its modes (`/research`, `/training`, `/tutor`, `/dev`) and reflection (`/reflect`) as TS modules implementing `TuiExtension` — registered at TUI startup. The TUI core knows nothing about modes or reflection.

CashNerd's initial extension is empty (no product-specific commands). `/skill <name>` is a built-in command (sets `context.skill`) so CashNerd doesn't need to ship an extension for that.

### Reflection — hard call

Hank ships reflection as a `TuiExtension` plugin in its own repo, using the v0.1 extension hook (`ctx.history.get()` + `ctx.gateway.streamChat()`). Recon 0.12 confirmed reflection does NOT need `ctx.fs.*` — `reflections/{date}.md` is written by the server-side `memory_write` MCP tool during the streamed reflection turn, not by TUI-extension filesystem code. Hank's plugin is ~80 lines of TS using only `ctx.history` + `ctx.gateway.streamChat`. Decision lands before TUI extraction PR merges.

### Built-in commands in core

`/quit`, `/clear` (truncate session), `/compact` (client-side history compaction; kept), `/model` (works against allowlist if Phase 0.3 confirms `model_catalog`; otherwise blind-forwards model name), `/status`, `/tools` (verbose toggle), `/skill <name>`, `/session <name>` (switch session file), `/new` (truncate current).

### Compaction stays client-side

Decision (no longer "or"): client-side compaction matching Hank's `history.ts` logic. Reasoning: dev-tool sessions are local files; server-side compaction is a different mechanism (used by Telegram/web bot-store) with different invocation. Plan does not introduce gateway-side compaction endpoints.

### Package layout
```
packages/agent-gateway-tui/
├── src/
│   ├── index.ts
│   ├── tui.ts                  (runTui, accepts TuiExtension[])
│   ├── extensions.ts           (interface + helpers)
│   ├── backend-client.ts
│   ├── event-adapter.ts        (per Phase 0.5b SSE policy)
│   ├── stream-assembler.ts
│   ├── commands.ts             (built-in commands)
│   ├── command-registry.ts     (built-ins + extension merge)
│   ├── history.ts              (Phase 0.5c session schema)
│   ├── config.ts               (Phase 0.5c config schema)
│   ├── formatters.ts
│   ├── tool-summary.ts
│   ├── tool-display.ts
│   ├── components/...
│   ├── theme/...
│   └── utils/...
├── tests/
│   ├── contract.test.ts        (asserts Phase 0.5 frozen contract; mock gateway)
│   └── extensions.test.ts      (asserts extension API surface)
├── package.json                (name=@ai-agent-gateway/tui, dep pinned to @mariozechner/pi-tui@0.62.0 per Recon 0.6 — matches Hank exactly to minimize port risk; bump to v0.71.1 is a follow-up PR after extraction stabilizes)
├── tsconfig.json
└── README.md
```

**Versioning:** same approach as CLI — independent semver, peer-dep range against gateway.

---

## Phase 3 — Product migrations

**Honest framing:** independent product PRs landing at different times always create a brief window where one product is canonical and the other isn't. Mitigations: (a) Phase 0.5 schema freeze before extraction, (b) maintainer runs tests before merging each PR, (c) products pin to specific extracted package versions so a later upstream change doesn't break them.

Migration PRs land independently. Order between Hank and CashNerd doesn't matter; whichever PR lands first lands first.

### 3a — CashNerd migrates to upstream CLI

**Sequencing: TWO commits, channel-set update lands FIRST as an independently-revertible PR.**

**Step 3a.1 — Channel-set widening (lands first, before package migration):**
- Update `finance_cli/gateway/server.py:526`: `user_scoped_channel = channel in {"web", "telegram"}` → `user_scoped_channel = channel in {"web", "telegram", "cli"}`. Recon 0.5 confirmed CashNerd's gateway routes user-scoping based on this set. Adding `"cli"` is backward-compatible for existing `"telegram"` traffic (no behavior change for that path) and prepares the gateway to handle the migrated CLI's `channel="cli"` traffic correctly.
- Why separate PR: the gateway behavior fix is independently revertible from the package extraction. If the channel-set update breaks something downstream, revert it without touching the chat_cli deletion. Codex round-7 explicit recommendation.
- This PR can land any time after Phase 0.5 schema freeze; no dependency on agent-gateway-cli being released.

**Step 3a.2 — Package migration (lands after step 3a.1 is in production):**
- Add `ai-agent-gateway-cli` to `finance_cli/pyproject.toml` (pinned to specific extracted version).
- Delete `finance_cli/finance_cli/dev/chat_cli.py` and `finance_cli/finance_cli/tests/test_dev_chat_cli.py` (coverage moved to upstream package).
- Update `finance_cli/CLAUDE.md` (per Recon 0.14 line 91 reference) and `docs/AGENT_WORKFLOWS.md`: `python3 -m finance_cli.dev.chat_cli {login,chat}` → `python3 -m agent_gateway_cli --config-namespace cashnerd {login,chat}`.
- Update `finance_cli/docs/developer/SKILL_CREATION_PLAYBOOK.md` lines 352, 355 (per Recon 0.14) to use new entrypoint.
- Audit other tests for `from finance_cli.dev.chat_cli import` (none expected per Recon 0.14, but verify).
- Smoke test: existing `~/.cache/cashnerd/sessions/*.json` files still load post-migration (Phase 0.5c schema-loader rule 4 handles CashNerd v0 shape — Object missing `schema_version`).

### 3b — Hank migrates to upstream CLI + TUI
- Add `ai-agent-gateway-cli` and `@ai-agent-gateway/tui` to AI-excel-addin's deps.
- Delete `AI-excel-addin/api/dev/chat_cli.py` and `AI-excel-addin/tui/`.
- Implement `hank-tui-extension` (TS module in AI-excel-addin) registering modes + reflection (or drop reflection if Hank chooses).
- Update Hank's CLAUDE.md and docs.

### 3c — Rollback policy

If either migration regresses production dev workflow, revert the consumer-product migration commit (the upstream packages stay extracted; consumer just unpins for one cycle). No back-compat shim in upstream — clean cut, documented in CHANGELOG.

---

## Phase 4 — Conformance tests (paired with Phase 1 + Phase 2)

Conformance tests live in the same PRs as their packages — Phase 1 PR includes the Python tests, Phase 2 PR includes the Node tests. Maintainer runs tests locally before merging. Tests are the documentation of the wire contract; their primary value is catching drift, not gating merges. The CI workflow (already added) runs them on every PR as a backstop, but the gate is the maintainer.

**CI gap surfaced by Recon 0.8:** AI-excel-addin currently has only `grep-guards.yml` in `.github/workflows/`; agent-gateway tests run only locally. `agent-gateway-dist` has no CI either. Phase 4 adds a NEW workflow as an explicit deliverable:

```yaml
# AI-excel-addin/.github/workflows/agent-gateway-tests.yml
name: agent-gateway-tests
on: [pull_request, push]
jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e 'packages/agent-gateway[dev]' -e 'packages/agent-gateway-cli[dev]'
      - run: pytest packages/agent-gateway/ packages/agent-gateway-cli/
  node-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "22" }
      - run: npm install --workspaces=false --prefix packages/agent-gateway-tui
      - run: npm test --prefix packages/agent-gateway-tui
```

The workflow runs on every PR. Failing test = client drifted from `http-api.md` or schema freeze. **Without this workflow, the conformance gate that prevents Phase 3 product deletions cannot function.** This is a hard prerequisite, not a nice-to-have.

**Pre-flight before merging the workflow:** confirm `[dev]` optional-deps group exists in both `packages/agent-gateway/pyproject.toml` AND `packages/agent-gateway-cli/pyproject.toml` (Phase 1 deliverable). If the gateway's existing `pyproject.toml` doesn't define `[dev]`, add it as part of the Phase 0.5 / pre-Phase-1 prep. The workflow's editable-extras syntax `'packages/...[dev]'` requires the extras to be defined.

### Python: `packages/agent-gateway-cli/tests/test_contract.py`
- Mock gateway responding to all three endpoints + emitting all 15 documented SSE event types plus an unknown-type event.
- Assert request shape per Phase 0.5: `context.channel="cli"`, top-level `user_id` on `/init` AND `/chat` (Recon 0.2 + 0.10 confirmed both handlers READ the field), `model` when set, approval payload with `tool_call_id`/`nonce`/`approved`/`allow_tool_type`.
- Assert SSE handling per Phase 0.5b: every wire event maps to its declared normalized output; terminal events stop the loop; unknown events debug-log without crashing.
- Assert `tool_approval_request` rendering surfaces `reason` (when present) and conditionally exposes the "approve all of this type" option only when `allow_persistent_approval=true`.
- Assert config + session JSON schemas per Phase 0.5c: schema_version=1, mode 0600 on write, BOTH v0 shapes (CashNerd dict + Hank flat array) load and forward-promote on save (loader rules 3 + 4: Array → Hank v0; Object missing `schema_version` → CashNerd v0). Also assert rules 1, 2, 6 (empty file → empty session; parse failure → quarantine; primitive/unknown-version → quarantine).

### Node: `packages/agent-gateway-tui/tests/contract.test.ts`
- Same wire-shape and SSE-handling assertions as the Python suite.
- Approval rendering: assert `reason` is surfaced and `allow_persistent_approval` gates the "approve type" option.
- Session loader: assert BOTH v0 shapes (CashNerd dict + Hank flat array) load and forward-promote on save (loader rules 3 + 4). Also assert rules 1, 2, 6 (empty file → empty session; parse failure → quarantine; primitive/unknown-version → quarantine).

### Node: `packages/agent-gateway-tui/tests/extensions.test.ts`
- Assert the v0.1 `TuiExtension` + `CommandHandlerContext` public surface (sub-namespaces: `ui`, `model`, `requestContext`, `history`, `session`, `gateway`). NO `ctx.fs.*` in v0.1 — assertion that the type does not include the `fs` key.
- Assert `ctx.gateway.streamChat` semantics: uses main session auth, returns SSE iterable, does not mutate main history unless extension calls `ctx.history.append`.
- Assert command-registry behavior: built-in commands + extension commands merge; extension command names cannot collide with built-ins.

### CI gate
Both test suites run on every agent-gateway monorepo PR via the new `agent-gateway-tests.yml` workflow (Phase 4 deliverable above). Failing test = client drifted from `http-api.md` or schema freeze. Drift detection lives at the source.

---

## Phase 5 — Docs + release

- `agent-gateway/README.md` — add "Reference clients" section linking the two new packages.
- `agent-gateway/docs/http-api.md` — already extended in Phase 0.5a; minor edits if Phase 4 surfaces ambiguity.
- New: `packages/agent-gateway-cli/README.md`, `packages/agent-gateway-tui/README.md`, `packages/agent-gateway-cli/CHANGELOG.md`, `packages/agent-gateway-tui/CHANGELOG.md`.
- PyPI release for `ai-agent-gateway-cli` v0.1.0; npm release for `@ai-agent-gateway/tui` v0.1.0.
- `agent-gateway-dist/` — does NOT extend to new packages. Existing dist mirror remains for the Python core only. New packages ship via PyPI/npm directly.
- Deprecation notes in finance_cli/CLAUDE.md and AI-excel-addin/CLAUDE.md.

---

## Order of work

1. **Phase 0** — Recon worksheet (commit before 0.5). All 14 recon items answered.
2. **Phase 0.5** — Spec extension + SSE policy + schema freeze. **Codex review of Phase 0.5 outputs before Phase 1+2 start.**
3. **Phase 1 + Phase 4-Python (paired)** — Python CLI extraction + conformance test suite. PR exits this phase only when conformance is green in CI.
4. **Phase 2 + Phase 4-Node (paired, parallel to Phase 1)** — Node TUI extraction + conformance test suite. PR exits this phase only when conformance is green in CI.
5. **Phase 3** — Product migrations (Hank + CashNerd, independent PRs). **Gated on Phase 1+2 conformance PASS.** Migration PR is allowed to delete the product's local CLI/TUI copy only after the upstream package's CI conformance is green. **Exception: Phase 3a.1 (CashNerd channel-set widening)** is a server-side gateway fix that lands independently and is gated only on Phase 0.5 schema freeze — not on Phase 1+2 conformance. It can land in parallel with Phase 1+2, since it doesn't touch any client package.
6. **Phase 5** — Docs, release notes, deprecation announcements.

---

## Open questions remaining

Recon resolved most prior open questions. Remaining:

1. **Spec extension ownership:** Phase 0.5a now edits `http-api.md` to document `user_id` (init + chat), `model_catalog` (init response), `reason` and `allow_persistent_approval` (tool_approval_request). Vote: separate PR landing first as a "spec catch-up" change, marking the contract change explicitly. Confirm OK before Phase 1+2 PRs reference the extended spec.
2. **`ctx.fs.*` deferral to v0.2:** Recon 0.12 confirmed reflection doesn't need fs primitives. Plan defers `ctx.fs.*` (`writeWorkdirFile`, `readWorkdirFile`, `listWorkdir`) to v0.2; sandbox contract is preserved as v0.2 reference. Reasonable simplification, or is there a known v0.1 use case we'd be removing?
3. ~~**CashNerd channel-set update timing**~~ — **CLOSED in round 7.** Codex hard-recommended separate-PR-first approach. Plan v4 splits Phase 3a into 3a.1 (channel-set widening, lands first as backward-compatible PR) + 3a.2 (package migration, lands after 3a.1 is in production). Independently revertible.
4. **Concurrent session writes** between CLI and TUI on same `--session` name: last-writer-wins documented. Advisory `.lock` sentinel parked for v1.1. Acceptable to ship v1 without it, or does the plan need to ship locking now?
5. **CI workflow location:** Phase 4 adds `AI-excel-addin/.github/workflows/agent-gateway-tests.yml`. AI-excel-addin is the right home (where `packages/` live), but should this be coordinated with adding CI to `agent-gateway-dist/` too, or is that a separate concern?

---

## Risks (updated)

- **Cross-repo coordination:** mitigated by Phase 3 concurrent migrations (not sequential), no shared schema drift between products.
- **Spec extension landing first:** Phase 0.5a edits the protocol doc. If those edits are wrong, every downstream piece is wrong. Codex review of Phase 0.5 outputs before Phase 1 (per order of work) is the gate.
- **Extension hook expressiveness:** v0.1 `TuiExtension` interface covers reflection (validated against Hank's `runReflectCommand` source) and slash-command-style mode toggles via `ctx.ui` / `ctx.requestContext` / `ctx.history` / `ctx.session` / `ctx.gateway`. Does NOT cover filesystem access — `ctx.fs.*` is deferred to v0.2 per Recon 0.12 (no current first-party use case needs it). If a v0.1 extension surfaces an actual fs need before extraction merges, plan adds the v0.2 sandbox spec to v0.1 rather than shipping an underpowered v0.1.
- **pi-tui upstream:** Hank already depends; we share blast radius. Phase 0.6 verifies maintenance signal.
- **Test coverage during migration:** Phase 1/2 ports product tests to upstream package. Phase 3 deletes product copies. Net coverage measured both directions; must not regress.
- **Drift remains, just relocated:** version skew between gateway core and client packages, schema-version skew, extension API breaking changes. Each addressed via peer-dep ranges + schema_version field + semver discipline. Honest framing replaces round-1's "structurally impossible" claim.

---

## Out of scope

- Gateway-side compaction/reflection endpoints (separate plan if Phase 0 surfaces need).
- Windows packaging.
- Brew formulas, standalone binaries.
- TUI auth beyond `chat_cli login` flow.
- Telegram/web chat UI changes.
- Onboarding-shell or higher-level dev workflows.

---

## Diff vs. v1 (Codex round-1 fixes — addressed in v2)

| # | Round-1 finding | v2 fix |
|---|---|---|
| 1 | Protocol contract violations (`user_id`, `model_catalog`, `channel="cli"` not in spec) | Phase 0.5a extends `http-api.md` to document each, gated on Phase 0 recon. |
| 2 | SSE coverage too narrow | Phase 0.5b table introduced. |
| 3 | "Drift structurally impossible" overstated | Reframed: drift moves to version skew, schema skew, shim drift. |
| 4 | Phase ordering wrong (sequential migration creates half-canonical state) | Phase 0.5 schema freeze introduced. |
| 5 | TUI command schema premature/underpowered | Replaced JSON schema with typed TS extension hook (`TuiExtension`). |
| 6 | Python CLI merge regression traps | Each call made explicit: base URL stays strict by default, session roles extended to system, approval `expires_at` gated on recon, `channel` swap gated on recon. |
| 7 | Reflection: hard decision needed | Hank ships reflection as a `TuiExtension` plugin in its own repo. |
| 8 | Deferral smell | Compaction = client-side; command schema = extension hook now; reflection = Hank plugin; versioning = independent semver + peer-dep; dist mirror = doesn't extend. |
| 9 | Missing questions | Schema freeze content, local dev linkage, CI ownership, gateway version support, `user_id` extension status, base URL policy, release coordination, rollback, compat shims (none). |

## Diff vs. v2 (Codex round-2 fixes — addressed in v3)

| # | Round-2 finding | v3 fix |
|---|---|---|
| 1 | `user_id` only spec'd on `/init`, but Phase 1 sends on `/chat` too | v3 extends BOTH `/init` and `/chat`, each gated on a recon item. v3 round 3 hardened to a single rule: send only on endpoints whose handler READS the field; do not send (and do not document) on endpoints that ignore it. |
| 2 | SSE event count off-by-one (said 16, actually 15 documented + unknown) | Language corrected to "15 documented + unknown handler" everywhere. |
| 3 | Render policy too vague for testing ("render warning line") | Phase 0.5b now specifies exact normalized adapter outputs (`assistant_text_append`, `tool_approval_needed`, etc.) with structured payloads. Conformance tests assert these. |
| 4 | Phase ordering bug: order-of-work put Phase 3 before Phase 4 conformance | Phase 4 split into Phase 4-Python (paired with Phase 1) and Phase 4-Node (paired with Phase 2). Phase 3 explicitly gated on Phase 4 PASS. Order-of-work updated. |
| 5 | "Concurrent migrations" claim misleading | Phase 3 reworded with honest framing: concurrency doesn't eliminate half-canonical state; the real mitigations are schema freeze + conformance gate + pinned versions. |
| 6 | Extension hook missing capabilities (history, session mutation, structured prompts, status) | `CommandHandlerContext` expanded and (in v3 round 3) factored into sub-objects: `ctx.ui`, `ctx.model`, `ctx.requestContext`, `ctx.history`, `ctx.session`, `ctx.gateway`, `ctx.fs`. Validated against `runReflectCommand` source. |
| 7 | Sandbox contract too vague (writeFile sandbox was a comment; callBackend unconstrained) | Sandbox contract specified with explicit path-resolution + `realpath` canonicalization rules (v3 round 3+5): rejects absolute paths, rejects any raw `..` segment lexically BEFORE normalization (split on `/` and `\`, never call `path.normalize` first since it collapses `a/../b`), rejects symlinked-parent escapes via realpath check, atomic writes via tempfile+rename. Trust model: footgun prevention (not adversarial isolation). `streamChat` replaces freeform `callBackend`; v0.1 extension scope explicitly limited to chat-turn extensions. |
| 8 | Schema migration deferred (v0 → v1) | v0 → v1 loader rule specified: missing `schema_version` → treat as v0 → forward-promote on save. v1 reader accepts v0 transparently. Phase 0.11 recon confirms exact v0 shapes in both products. |
| 9 | `model_catalog` two contradictory positions (line 89 vs. line 263) | Single behavior: clients always tolerate absence and blind-forward `model` name; `/model` displays current + accepts any string when catalog absent. No "remove handling" branch. Specified in Phase 0.5a and Phase 1 consistently. |
| 10 | Recon missing items (chat handler, on-disk shapes, reflection deps, package names, deleted-entrypoint refs) | Recon table extended to 14 items (added 0.10 chat handler, 0.11 on-disk shapes, 0.12 reflection deps, 0.13 package name availability, 0.14 deleted-entrypoint refs). |

## Diff vs. v4-initial (Codex round-7 fixes — applied within v4)

| # | R7 finding | v4 fix |
|---|---|---|
| 1 | Phase 0.5b table still referenced `expires_at per Phase 0.5a` (stale) | Replaced: explicit "NO `expires_at` field — no countdown UX, no synthetic deadline. Includes `reason` + `allow_persistent_approval`". |
| 2 | Non-goals contradicted v4 by saying "no CI changes" while v4 adds a workflow | Non-goal rewritten: only the dist-mirror extension is excluded; the new CI workflow is explicitly called out as a Phase 4 prerequisite, not a non-goal. |
| 3 | Channel-set update bundled with chat_cli deletion (coupled revert paths) | Phase 3a split into 3a.1 (channel-set widening — separate PR, lands first as backward-compatible change) and 3a.2 (package migration — lands after 3a.1 is in production). Open question 3 closed. |
| 4 | CI YAML `pip install -e packages/...[dev]` unquoted (shell-glob bait) | Quoted: `pip install -e 'packages/agent-gateway[dev]' -e 'packages/agent-gateway-cli[dev]'`. Plus pre-flight note to confirm `[dev]` extras exist in both packages' pyproject.toml. |
| 5 | Loader rules didn't handle parse errors (empty file, malformed JSON) | Loader rules expanded to 6 explicit cases: empty file → empty session; parse failure → quarantine; Array → Hank v0; Object-no-version → CashNerd v0; Object-v1 → v1; primitive/unknown-version → quarantine. Order documented; first match wins; non-overlapping by construction. |
| 6 | Risks section still claimed hook covers "fs writes" (stale; v0.1 defers fs) | Risk rewritten: v0.1 covers reflection + slash-commands; `ctx.fs.*` explicitly out of scope per Recon 0.12; if a v0.1 fs use case surfaces before extraction merges, the v0.2 sandbox spec gets pulled into v0.1. |

## Diff vs. v3-final (recon-driven changes — applied in v4)

| # | Recon item | v4 change |
|---|---|---|
| 1 | 0.4 — `expires_at` not emitted; `reason` + `allow_persistent_approval` ARE emitted but undocumented | Phase 0.5a documents both new fields. Phase 0.5b normalized `tool_approval_needed` payload includes them. Conformance tests assert rendering of `reason` and conditional UI for `allow_persistent_approval`. CashNerd's dead countdown UX stays removed. |
| 2 | 0.5 — `channel="telegram"` load-bearing in `finance_cli/gateway/server.py:526` (`user_scoped_channel` set) | Phase 3a split into 3a.1 (channel-set widening: add `"cli"` to the set, lands FIRST as a separate backward-compatible PR, gated only on Phase 0.5) and 3a.2 (package migration: lands AFTER 3a.1 is in production, gated on Phase 1+2 conformance). Independently revertible. (Originally bundled with chat_cli deletion in v4-initial; split in v4-round-7 fixes per Codex recommendation.) |
| 3 | 0.6 — pi-tui v0.71.1 latest, Hank pins 0.62.0 | Phase 2 pins to 0.62.0 to match Hank exactly; bump to 0.71.1 is a separate follow-up PR after extraction stabilizes. Avoids compounding port risk with version bump. |
| 4 | 0.8 — no CI for agent-gateway today | Phase 4 adds `AI-excel-addin/.github/workflows/agent-gateway-tests.yml` as an explicit deliverable. Without this, the conformance gate can't function. Workflow runs both Python (pytest) and Node (npm test) suites on every PR. |
| 5 | 0.11 — Hank v0 is a flat JSON array, NOT a dict | Phase 0.5c loader expanded with type-discrimination rules: Array → Hank-shape; Object missing `schema_version` → CashNerd-shape; Object with `schema_version=1` → v1; anything else → quarantine. Both v0 shapes load transparently and forward-promote on save. |
| 6 | 0.12 — reflection doesn't need `ctx.fs.*` | **v0.1 scope reduction:** `ctx.fs.*` (`writeWorkdirFile`, `readWorkdirFile`, `listWorkdir`) deferred to v0.2. Sandbox contract preserved in plan as v0.2 reference. Hank reflection uses only `ctx.history.get()` + `ctx.gateway.streamChat()`. Saves significant test surface (sandbox path-resolution, symlink edge cases, atomic-write semantics) for v0.1. |

## Diff vs. v3-initial (Codex round-3 fixes — applied within v3)

| # | Round-3 finding | v3 fix (final) |
|---|---|---|
| 1 | `user_id` had three contradictory positions in body | Collapsed to a single rule in Phase 0.5a (handler READS the field → send + document on that endpoint; handler IGNORES → do not send, do not document). All other plan references (Phase 1, config schema comment at line 159, conformance test assertion at line 457) reference the rule rather than restating it, so the rule is in one place and applied consistently. |
| 2 | Sandbox traversal had symlink hole | Sandbox contract now specifies: (a) lexical pre-normalization rejection of absolute paths and any raw `..` segment (split raw input on `/` and `\`, NEVER `path.normalize` first since it would collapse `a/../b` to `b` and defeat the check), (b) `realpath` canonicalization on parent directories rejecting symlinked-ancestor escapes, (c) target-itself canonicalization on `readWorkdirFile` AND `listWorkdir` (closing the round-4 list-symlink hole), (d) `writeWorkdirFile` rejects symlinks pointing outside workdir. Documents what's out of scope (TOCTOU, hardlinks). |
| 3 | Extension API too flat (17 methods on one object) | Factored into seven sub-namespaces: `ctx.ui`, `ctx.model`, `ctx.requestContext`, `ctx.history`, `ctx.session`, `ctx.gateway`, `ctx.fs`. API stability: v0.x band allows breaks in minor bumps (semver pre-1.0); v1.0+ requires major bump. Hard call replaces open question. |
| 4 | `streamChat` covers chat extensions only — was an open question | Now an explicit body constraint: v0.1 extension scope is "chat-turn extensions and slash commands that mutate UI/session/context state." Non-`/api/chat` use cases are out of scope and require a future constrained primitive (not freeform `callBackend`). |
| 5 | Diff table overclaimed issues 1 and 7 | Both rows updated above to reflect the v3-final positions including the round-3 hardening. |
