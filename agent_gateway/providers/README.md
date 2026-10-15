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
table. Importing this package observes the registry against it and never refuses: a
gateway-executed entry no installed adapter supports is logged (the artifact deploys
ahead of code), and selection skips it with `provider_unavailable`
(`model_authority.binding.require_capability_execution_bind`) or `default_not_eligible`.

## Modules

| Path | Primary symbol | Role |
| --- | --- | --- |
| `base.py` | `ModelProvider`, `ModelInfo`, `model_info_from_entry()` | Adapter contract; `ModelInfo` built from the registry entry, its `compat` and rate table |
| `anthropic.py`, `anthropic_helpers.py` | `AnthropicProvider` | Messages API (`anthropic.messages` compat) |
| `anthropic_oauth.py` | `resolve_anthropic_credentials()`, `select_anthropic_credential_source()`, `rotate_anthropic_credential_source()` | The Anthropic OAuth credential pool Risk and Investment also draw from |
| `openai.py`, `openai_responses_helpers.py` | `OpenAIProvider`, `responses_compat()` | Responses API (`openai.responses` compat) |
| `codex.py`, `codex_helpers.py` | `CodexProvider` | Codex route over the Responses family |
| `xai.py`, `xai_helpers.py`, `xai_oauth.py` | `XAIProvider` | xAI Responses route and its OAuth |
| `google.py` | `GoogleGenerateContentProvider` | Gemini API `streamGenerateContent` (`google.generate_content` compat), route `google.public`; Bearer OAuth token or API key |
| `google_antigravity.py` | `GoogleAntigravityProvider` | Gemini over Cloud Code Assist (`daily-cloudcode-pa`), route `google.antigravity`; Bearer OAuth token plus `project_id` from the Antigravity login (Henry 2026-10-10) |
| `copilot.py` | `CopilotResponsesProvider` | GitHub Copilot proxy over the Responses family, route `copilot.api` |
| `muse.py` | `MuseResponsesProvider` | Meta Model API over the Responses family, route `muse.api` |
| `ollama.py` | `OllamaLocalProvider` | Chat Completions against the local Ollama daemon, route `ollama.local`; `-cloud` tags reach Ollama's cloud through a signed-in daemon |
| `ollama_cloud.py` | `OllamaCloudProvider` | Ollama Cloud API-key entry, Chat Completions on `https://ollama.com/v1`, route `ollama.cloud` |
| `agent_sdk.py` | `AgentSDKConfig` | Claude Agent SDK runner configuration |
| `__init__.py` | `installed_adapter_route_support()` | Installed adapters and their declared support |

Credentials are materialized by the embedding application (`api/credentials.py` in
Hank) per provider family, not per route: a handle names the family, and
`create_client` receives no route. So each provider family declares one route; a
second route that needs other material is a second family with its own class. An
adapter only reports `has_active_credential()`. Each adapter classifies its own
failures (`classify_credential_failure`): a usage limit or HTTP 429 is `rate_limit`,
a hard 401/403 denial is `auth` or `billing`, and a 400, content refusal or context
overflow is neither, so the runner's fallback hop names the right reason.
`agent_gateway.auth.AuthConfig` admits exactly the provider families these
declarations name.

## Anthropic credential sources

`anthropic_oauth.py` owns the version-3 source store and generation helpers.
`select_anthropic_credential_source()` and
`rotate_anthropic_credential_source()` return an `AnthropicCredential`, not a
bare token: its token and original source record describe the same generation,
with `account_identity`, `grant_hash`, `generation_id` and opaque
`generation_ref` metadata. Consumers carry selected token and source metadata
together. OAuth refresh updates material within that generation; replacement
of a static key/token or reconsent enrolls a new generation. A key/token a
request carries (bound `auth_token` without `credential_source`, BYOK session
material) is registered under its own source, named by its material's digest,
in the store of the scope that carried it: two requests' material never shares
a slot, so one never supersedes another, and re-presented material resolves its
own generation.

The store keeps active credential records and generation retirement reasons.
Retirement discards that generation's secrets; a narrow
`retired_auth_token_grants` map retains only retired static auth-token grant hashes
and generation ids so token re-entry cannot reenroll material an environment
change or `auth remove` retired as a request source. Legacy stores acquire
generation metadata under the store lock. Old
writers cannot safely share an upgraded store: isolate their store paths
operationally rather than relying on mixed-version writes.

`resolve_anthropic_credentials()` with `credential_source` metadata reads the
exact original bearer record for the bound pool member, without replacement
enrollment; selection can refresh that original grant. The lookup is a pure
read: it opens the store read-only under a shared lock (never creating the
store, its directory or its lock file) and runs no environment sync, so it
registers and retires nothing. The caller supplies the authenticated user
scope, authorizing lookup in that user's private store and the org store (only
the org store without a user scope). A reference encodes no store path. Exact
lookup only reads: an absent environment name reports `unobservable`, an empty
source reports `removed`, and a changed source reports `superseded`. An
unwritable or unreadable store is the typed `AnthropicStoreError` (`code`
`store_unreadable` with its own `next_call`), never a raw `OSError`.
Environment-pool reconciliation alone (the pool resolve, not the exact lookup)
retires observed empty or changed pool sources: each environment-pool token's
source is keyed by its own material's digest (`env:<name>:<sha256>`), so two
processes with different values for one name never retire each other's
generation, while a name observed empty still retires that name's slot.
Environment credentials remain org sources, not copies enrolled into each
user's store.

These source helpers do not by themselves demonstrate gateway binding, durable
session reopen, or a completed consumer cutover. Mutable-pool rotation remains
a separate consumer behavior described in the authentication reference: it is
not an exact-generation lookup fallback and does not authorize replacing a
frozen generation binding.
