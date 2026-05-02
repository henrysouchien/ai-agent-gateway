# Recon Worksheet: Agent-Gateway Dev-Client Extraction

**Plan:** [PLAN_AGENT_GATEWAY_DEV_CLIENTS.md](./PLAN_AGENT_GATEWAY_DEV_CLIENTS.md) (Codex PASS R6)
**Date:** 2026-05-01
**Owner:** hc@henrychien.com
**Status:** All 14 recon items answered. Plan-revision items flagged at end.

---

## Findings

### 0.1 — Canonical agent-gateway source ✓

`AI-excel-addin/packages/agent-gateway/` is canonical. PyPI `ai-agent-gateway` v0.8.1, MIT, Henry Chien LLC. Dist mirror at `henrysouchien/ai-agent-gateway` (synced).

### 0.2 — Does `/api/chat/init` handler READ `user_id`? **YES, heavily.**

`agent_gateway/server.py:51` — `ChatInitRequest.user_id: str | None = None`. Handler at line 495+:
- Strips whitespace (line 499–501)
- Falls back to `payload.context.user_id` with deprecation warning (line 507–511)
- Falls back to `_default` (line 512–513)
- In strict multi-user mode: rejects `_default` with HTTP error (line 517–522)
- Calls credential resolver with `resolved_user_id` (line 528)
- Stores in session at line 548

**Implication:** `user_id` is THE primary identity field on `/init`. Plan position holds: clients send `user_id` on `/init` when configured. Phase 0.5a documents this as the canonical contract.

### 0.3 — Does `/api/chat/init` response emit `model_catalog`? **YES, optional.**

`agent_gateway/server.py:241` — `ChatInitResponse.model_catalog: Optional[ModelCatalog] = None`. Set from `config.model_catalog` at line 558.

**Implication:** Plan position holds: clients tolerate absence, blind-forward `model` name when catalog absent. `/model` slash command displays current model + accepts any string when catalog absent.

### 0.4 — Does `tool_approval_request` event emit `expires_at`? **NO.**

`agent_gateway/server.py:402–410` payload contains: `type`, `tool_call_id`, `nonce`, `tool_name`, `tool_input`, `resolved_qualifier`, `reason`, `allow_persistent_approval`. **NO `expires_at` field.**

CashNerd CLI's `_derive_approval_countdown_seconds` reads `event.get("expires_at")` which always returns None → countdown displays "unknown". **Existing UX is dead code.**

**Two new fields surface beyond `http-api.md` documentation:** `reason` and `allow_persistent_approval`. These are emitted but not in the published spec.

**Implication:** Plan revisions:
- REMOVE approval-countdown UX from merged CLI (no synthetic timer fallback).
- Phase 0.5a additionally documents `reason` and `allow_persistent_approval` on the `tool_approval_request` event.

### 0.5 — Is `context.channel="telegram"` load-bearing? **YES.**

`finance_cli/gateway/server.py:526`:
```python
user_scoped_channel = channel in {"web", "telegram"}
if user_scoped_channel:
    raw_user_id = ...
    if not user_id:
        raise HTTPException(status_code=400, detail=f"user_id is required for {channel or 'chat'} chat")
    provision_user(...)
```

CashNerd CLI today sends `channel="telegram"` → triggers user-scoped path: `user_id` required, `provision_user` called. Switching to `channel="cli"` without other changes would BYPASS user-scoping (different behavior).

**Implication:** Plan revision — Phase 0.5 / Phase 3 CashNerd migration commit must include:
- Update `finance_cli/gateway/server.py:526` to `user_scoped_channel = channel in {"web", "telegram", "cli"}` (preferred — semantic fix, "cli" IS a user-scoped surface like "web" and "telegram").
- OR keep CashNerd CLI on `"telegram"` via `--legacy-channel telegram` flag for one release.
- Recommend the gateway-side update; the channel-set is a CashNerd-specific concept (not in agent-gateway core), so no upstream coordination needed.

### 0.6 — pi-tui v0.62.0 license + maintenance ✓

`@mariozechner/pi-tui` v0.71.1 latest (Hank pins **0.62.0** — 9 minor versions behind).
- License: MIT ✓
- Active: published 5 hours ago, 309 versions
- Maintainers: Mario Zechner (badlogic) + Armin Ronacher (mitsuhiko, Flask creator)

**Implication:** Peer-dep is fine, no need to vendor.

**New question for Phase 2:** upgrade to v0.71.1 during port, or pin to Hank's 0.62.0 to minimize port risk? Recommend: port to 0.62.0 first to match Hank exactly, then bump in a separate PR after extraction stabilizes.

### 0.7 — Dev-loop pattern in existing `packages/` siblings ✓

`AI-excel-addin/api/requirements-local.txt`:
```
-e ../packages/agent-gateway
```

Standard `pip install -e` editable install pointing at the sibling package directory. Same pattern works for `agent-gateway-cli`.

For Node `agent-gateway-tui`: AI-excel-addin doesn't have an existing npm-link sibling pattern. Two options:
- (a) Set up npm workspaces at AI-excel-addin root (cleaner long-term).
- (b) `"@ai-agent-gateway/tui": "file:../packages/agent-gateway-tui"` in consumer's package.json (simpler, matches the pip pattern).

**Recommend:** option (b) for v0.1 (matches pip pattern, avoids workspace migration scope).

### 0.8 — CI infrastructure for agent-gateway tests **GAP.**

`AI-excel-addin/.github/workflows/` only has `grep-guards.yml` (runs `make grep-guards`). `agent-gateway-dist/.github/workflows/` does not exist.

**agent-gateway tests apparently run only locally during dev — no automated CI.**

**Implication:** Plan revision — Phase 4 conformance tests need a NEW workflow:
- Add `AI-excel-addin/.github/workflows/agent-gateway-tests.yml` running `pytest` in `packages/agent-gateway/`, `packages/agent-gateway-cli/`, plus Node test in `packages/agent-gateway-tui/`.
- This must land before product migrations gate on conformance PASS.
- New scope item for the plan.

### 0.9 — Hank TUI clean-port verification ✓

Zero Excel-related strings in `tui/src/backend-client.ts` and `tui/src/event-adapter.ts` (0 matches each for `excel|spreadsheet|workbook|cell`, case-insensitive). Protocol layer is genuinely product-agnostic. Direct port is feasible.

### 0.10 — Does `/api/chat` handler READ `user_id`? **YES, validates session identity.**

`agent_gateway/server.py:87` — `ChatRequest.user_id: str | None = None`. Handler at line 561+:
- Strips (line 564–566)
- Validates `body.user_id == jwt_user_id` (from session JWT at line 569) — **mismatch returns HTTP error at line 584**
- Falls through to `body.user_id = body.user_id or jwt_user_id` at line 596

**Implication:** Clients MUST send `user_id` on `/chat` matching the session-issued JWT identity. Plan position holds: clients send `user_id` on `/chat` when configured. If not sent, the JWT identity is used (forward-compat). If sent and mismatched, gateway rejects.

### 0.11 — On-disk session shapes in CashNerd + Hank — **two distinct v0 shapes.**

**CashNerd v0** (`~/.cache/cashnerd/sessions/<name>.json`):
```jsonc
{
  "name": "string",
  "created_at": <unix int>,
  "updated_at": <unix int>,
  "messages": [{"role": "user|assistant", "content": "string"}]
}
```
No `schema_version`. Roles limited to user/assistant. Per-session named files.

**Hank v0** (`tui/data/chat_history.json`):
```jsonc
[
  {"role": "user", "content": "string"},
  {"role": "assistant", "content": "string"}
]
```
Just a flat array. Single shared file (no per-session naming). No envelope, no metadata, no schema_version.

**Implication:** Plan revision — v0 → v1 loader must accept BOTH:
- File contains a JSON Array → Hank-shape. Default `name="default"`, use file mtime for `created_at`/`updated_at`, wrap in v1 envelope on save.
- File contains a JSON Object missing `schema_version` → CashNerd-shape. Promote to `schema_version: 1` on save.
- File contains JSON Object with `schema_version: 1` → v1.

Update Phase 0.5c schema-migration spec accordingly.

### 0.12 — Hank reflection feature deps ✓

`tui/src/history.ts:151` `buildReflectMessages` takes last 10 history messages, formats with date-stamped `REFLECT_PROMPT`. The prompt instructs the agent to:
1. `memory_write reflections/{date}.md` (server-side MCP tool)
2. `run_agent` with task to read findings + analyze code

**Implication:** Reflection is an AGENT-PROMPT feature, not a TUI capability. Extension API needs ONLY `ctx.history.get()` + `ctx.gateway.streamChat()` to implement reflection. `ctx.fs.*` is NOT required.

**This shrinks v0.1 extension surface area:** if reflection is the only motivating use case, `ctx.fs.*` could be deferred to v0.2. Open question for Phase 2 design.

### 0.13 — Package name availability ✓

- **PyPI `ai-agent-gateway-cli`:** AVAILABLE (no matching distribution).
- **npm `@ai-agent-gateway/tui`:** scope/package not registered (404).

**Implication:** Both names usable. ACTION: register `@ai-agent-gateway` npm scope (via `npm org create` or web) before publishing TUI.

### 0.14 — Refs to soon-deleted entrypoints ✓

**Active refs requiring update during Phase 3 migration:**

In `finance_cli`:
- `CLAUDE.md:91` — references `python3 -m finance_cli.dev.chat_cli`
- `docs/developer/SKILL_CREATION_PLAYBOOK.md:352, 355` — uses chat_cli for skill testing
- `docs/planning/PLAN_ONBOARDING_COPY_FIX.md` (multiple lines) — references chat_cli paths
- `finance_cli/tests/test_dev_chat_cli.py` — DELETE entirely (coverage moves to upstream package)
- `finance_cli/dev/chat_cli.py` — DELETE entirely

In `AI-excel-addin`:
- `tui/package.json` — DELETE the entire `tui/` directory
- `tui/src/*` — DELETE all
- `api/dev/chat_cli.py` — DELETE entirely
- `api/dev/__init__.py` — likely DELETE

**Inactive (historical) refs in `docs/completed/`** — no update needed.

---

## Plan-revision items

Recon surfaced six items requiring updates to v3-final plan before extraction starts:

| # | Recon item | Plan revision needed |
|---|---|---|
| 1 | 0.4 — `expires_at` not emitted | Plan removes approval-countdown UX (already specified). Phase 0.5a additionally documents `reason` and `allow_persistent_approval` on `tool_approval_request` event (NEW — not in v3-final). |
| 2 | 0.5 — `channel="telegram"` load-bearing | Phase 3 CashNerd migration adds line `user_scoped_channel = channel in {"web", "telegram", "cli"}` to finance_cli/gateway/server.py:526 (NEW migration step — not in v3-final task list). |
| 3 | 0.6 — pi-tui v0.71.1 vs Hank's 0.62.0 | Phase 2 pins to 0.62.0 for parity, defers upgrade to follow-up (NEW Phase 2 decision — not in v3-final). |
| 4 | 0.8 — no CI for agent-gateway today | Phase 4 adds `AI-excel-addin/.github/workflows/agent-gateway-tests.yml` (NEW Phase 4 deliverable — not in v3-final). Without this, the conformance gate can't function. |
| 5 | 0.11 — Hank v0 is a flat array, not a dict | Phase 0.5c schema-migration loader must accept BOTH JSON Array (Hank) AND Object-missing-schema_version (CashNerd) (REFINEMENT — v3-final spec only mentioned CashNerd shape). |
| 6 | 0.12 — reflection doesn't need `ctx.fs.*` | Phase 2 v0.1 design: defer `ctx.fs.*` (`writeWorkdirFile`, `readWorkdirFile`, `listWorkdir`) to v0.2 unless another use case surfaces. Sandbox contract spec stays in plan but marked v0.2 not v0.1. (POSSIBLE SIMPLIFICATION — needs Codex review.) |

## Recon worksheet — done

All 14 items answered. Plan revisions enumerated above. Recommended next step: revise plan v3-final → v4 with the six items, send to Codex for round-7 review (which should be quick since these are concrete recon-driven additions, not design changes).

Decision points before plan-revision:
- Plan-revision item 6 (defer `ctx.fs.*` to v0.2) is a SCOPE REDUCTION — needs explicit user decision before locking in. The motivation (reflection doesn't need it) is weaker than "no use case currently needs it" but matches the plan's "v0.1 minimal" philosophy.
- All other items are mechanical additions; no scope-changing decisions needed.
