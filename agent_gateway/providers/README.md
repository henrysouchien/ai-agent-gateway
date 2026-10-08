# Provider adapters

One `ModelProvider` subclass per wire protocol the gateway executes: it builds the
request, streams the response into `StreamEvent`s, classifies failures, rotates
credentials and estimates cost. It does not choose a model.

## Model selection and request facts come from `model_authority`

Which model a capability calls is resolved before a provider is reached:
`CapabilityExecutionResolver` (`../capability_execution.py`) binds the capability
through `hank-model-authority` and hands the adapter a `CapabilityBind`. The adapter
reads everything model-specific from the bound registry entry: limits and the
`compat` block (thinking form, tool-choice forcing, reasoning control, output-token
parameter) through `model_info_from_entry()` in `base.py`, and prices through
`authority_rate_table()`. No module here keeps a table keyed by model name; a new
model or a changed request fact is an artifact change, not an adapter change.
Mechanism: `docs/reference/model-authority.md` (monorepo root).

Each adapter declares what it can execute in `adapter_route_support()` (adapter id,
protocol profiles, routes). `installed_adapter_route_support()` in `__init__.py`
derives the installed set from those declarations; there is no hand-written adapter
table, and importing this package refuses a registry entry for a gateway-executed
capability that no installed adapter supports.

## Modules

| Path | Primary symbol | Role |
| --- | --- | --- |
| `base.py` | `ModelProvider`, `ModelInfo`, `model_info_from_entry()` | Adapter contract; `ModelInfo` built from the registry entry, its `compat` and rate table |
| `anthropic.py`, `anthropic_helpers.py` | `AnthropicProvider` | Messages API (`anthropic.messages` compat) |
| `anthropic_oauth.py` | `resolve_anthropic_credentials()`, `select_anthropic_credential_source()`, `rotate_anthropic_credential_source()` | The Anthropic OAuth credential pool Risk and Investment also draw from |
| `openai.py`, `openai_responses_helpers.py` | `OpenAIProvider`, `responses_compat()` | Responses API (`openai.responses` compat) |
| `codex.py`, `codex_helpers.py` | `CodexProvider` | Codex route over the Responses family |
| `xai.py`, `xai_helpers.py`, `xai_oauth.py` | `XAIProvider` | xAI Responses route and its OAuth |
| `agent_sdk.py` | `AgentSDKConfig` | Claude Agent SDK runner configuration |
| `__init__.py` | `installed_adapter_route_support()` | Installed adapters and their declared support |

Credentials are materialized by the embedding application (`api/credentials.py` in
Hank) from the bind's `route`; an adapter only reports `has_active_credential()`.

## Anthropic credential sources

`anthropic_oauth.py` owns the version-3 source store and generation helpers.
`select_anthropic_credential_source()` and
`rotate_anthropic_credential_source()` return an `AnthropicCredential`, not a
bare token: its token and original source record describe the same generation,
with `account_identity`, `grant_hash`, `generation_id` and opaque
`generation_ref` metadata. Consumers carry selected token and source metadata
together. OAuth refresh updates material within that generation; replacement
of a static key/token or reconsent enrolls a new generation.

The store keeps active credential records and generation retirement reasons.
Retirement discards that generation's secrets; a narrow
`retired_auth_token_grants` map retains only retired static auth-token grant hashes
and generation ids so token re-entry cannot reenroll retired material as a
request source. Legacy stores acquire generation metadata under the store lock. Old
writers cannot safely share an upgraded store: isolate their store paths
operationally rather than relying on mixed-version writes.

`resolve_anthropic_credentials()` with `credential_source` metadata reads the
exact original bearer record for the bound pool member, without replacement
enrollment; selection can refresh that original grant. The caller supplies the
authenticated user scope, authorizing lookup in that user's private store and
the org store (only the org store without a user scope). A reference encodes no
store path. Exact lookup only reads: an absent environment name reports
`unobservable`, an empty source reports `removed`, and a changed source reports
`superseded`. Environment-pool reconciliation alone retires observed empty or
changed pool sources. Environment credentials remain org sources, not copies
enrolled into each user's store.

These source helpers do not by themselves demonstrate gateway binding, durable
session reopen, or a completed consumer cutover. Mutable-pool rotation remains
a separate consumer behavior described in the authentication reference: it is
not an exact-generation lookup fallback and does not authorize replacing a
frozen generation binding.
