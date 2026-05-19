# Changelog

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
