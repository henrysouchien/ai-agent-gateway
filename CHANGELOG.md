# Changelog

## 0.15.0

### Added

- **Typed event contract for skill-framework runs.** New module
  `agent_gateway.events` exports `SkillRunStartedEvent`, `VerdictEmittedEvent`,
  `ArtifactReadyEvent`, `AggregateReadyEvent`, `ArtifactFailedEvent`, and
  `ArtifactUnavailableEvent`. All carry `run_id` correlation except
  `ArtifactUnavailableEvent` (renderer-side only, no skill run). Re-exported
  from `agent_gateway` top-level. Spec:
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
