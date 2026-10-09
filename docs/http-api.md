# HTTP API

This page documents the wire protocol exposed by `create_gateway_app()` and `create_agent()`.

By default the API is mounted under `/api`. If you set `GatewayServerConfig.prefix`, prepend that value to every route in this document.

## Transport

- `POST /chat/init` returns JSON
- `POST /chat/tool-result` returns JSON
- `POST /chat/tool-approval` returns JSON
- `POST /chat/cancel` returns JSON
- `POST /chat/recap` returns JSON
- `GET /chat/subscribe` returns an SSE stream for an active turn
- `GET /workflow-outputs/{workflow_run_id}/{output_id}` returns canonical output bytes
- artifact, UI-block, letter, and control-plane reads return JSON or file content
- `GET /health` returns JSON
- `POST /chat` returns an SSE stream

The chat stream uses standard server-sent events with `Content-Type: text/event-stream`. The server writes JSON payloads as `data:` lines and does not currently set named SSE event types.
Stream schema version is selected once at `POST /chat/init` and echoed on every stream envelope. `GET /chat/subscribe` reuses the session's stored schema version; clients must create a new session to change versions.

## Auth Flow

1. Your client sends an API key to `POST /api/chat/init`.
2. The server validates that key against `GatewayServerConfig.valid_api_keys`.
3. On success, the server issues a JWT `session_token`.
4. Your client sends that token as `Authorization: Bearer <session_token>` on chat and tool loop requests.

If `valid_api_keys` is empty, any non-empty API key is accepted.
Without a credentials resolver, session init must also supply a stable
top-level `user_id`; a configured resolver owns that identity instead.

## Endpoints

### POST /api/chat/init

Start a session and receive a JWT session token.

Request body:

```json
{
  "api_key": "demo-key",
  "schema_version": 1,
  "user_id": "alice"
}
```

Schema:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `api_key` | string | yes | Must be non-empty. |
| `schema_version` | integer | no | Wire schema version requested for this session. Defaults to `1`. Unsupported versions return HTTP `400` with `{"error":"unsupported_schema_version", "message":"Unsupported schema_version <N>; supported: [1]"}`. |
| `user_id` | string | yes without resolver | Stable user identity. The gateway uses this for credential resolution and as the session identity. When a credentials resolver is configured, the resolver may derive the user from the API key; otherwise clients must send a top-level user_id. `_default` is reserved-invalid. |

Response body:

```json
{
  "user_id": "alice",
  "session_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "session_id": "sess_1234abcd5678",
  "expires_at": 1770000000,
  "schema_version": 1,
  "capability_choices": {
    "session.driver": {
      "capability": "session.driver",
      "catalog_revision": "registry-r7",
      "policy_revision": "policy-r4",
      "selected": {
        "model_key": "anthropic.claude-opus-5-5",
        "label": "Opus 5.5",
        "effort": "high",
        "reason": "platform_default"
      },
      "notices": [],
      "choices": [
        {
          "model_key": "anthropic.claude-opus-5-5",
          "label": "Opus 5.5",
          "supported_efforts": ["none", "low", "medium", "high", "xhigh", "max"],
          "default_effort": "high",
          "lifecycle": "active"
        }
      ]
    }
  }
}
```

Schema:

| Field | Type | Notes |
| --- | --- | --- |
| `user_id` | string | Resolved end-user identity. When the request body `user_id` is set, this echoes it. When a credentials resolver derives identity from the API key, this is the resolver's resolved value. Clients should thread this value onto subsequent `POST /api/chat` calls so the gateway can enforce strict-mode identity checks. Added in 0.15.0. |
| `session_token` | string | JWT bearer token for later requests |
| `session_id` | string | Server-generated session id |
| `expires_at` | integer | Unix timestamp. The session's TTL; work still in progress (a turn, or a batch admitted on this token) keeps the token valid past it, but a new turn starts only before it (see Session Lifecycle) |
| `schema_version` | integer | Negotiated wire schema version for this session. Every `/chat` and `/chat/subscribe` SSE envelope echoes this value. |
| `capability_choices` | object | Required authenticated, session-executable choices keyed by capability id. Each entry includes exact registry/policy revisions, the selected stable key or `null`, notices, and eligible stable-key choices. Clients must not synthesize catalogs, aliases, or defaults. |

Example:

```bash
curl -s http://127.0.0.1:8000/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"demo-key","schema_version":1,"user_id":"alice"}'
```

### POST /api/chat

Open an SSE stream for one chat turn.

Headers:

- `Authorization: Bearer <session_token>`
- `Content-Type: application/json`

Request body:

```json
{
  "messages": [
    {"role": "user", "content": "Summarize the launch plan."}
  ],
  "context": {
    "channel": "web"
  },
  "user_id": "alice",
  "model_key": "anthropic.claude-sonnet-5-5",
  "effort": "high",
  "catalog_revision": "registry-r7"
}
```

Schema:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `messages` | array of `ChatMessage` | yes | Full message list for the current turn |
| `context` | object | no | Free-form request context; `channel` is commonly used |
| `user_id` | string | no | If sent, the value is used as the request identity. In strict multi-user mode (resolver configured), it must match the session's JWT identity; mismatch returns an HTTP error. In non-strict mode, the supplied value is accepted as-is. If absent: strict mode rejects the request; non-strict mode falls back to the JWT identity. Reference clients should always send `user_id` to be forward-compatible with strict mode. |
| `model_key` | string or null | no | Authenticated stable-key override. Omit to use the saved preference or product default. |
| `effort` | string or null | no | Explicit effort. Requires `model_key`. |
| `catalog_revision` | string or null | no | Concurrency context presented with the authenticated choice. Requires `model_key`. It is not selection authority: the server rechecks current eligibility, so harmless registry refreshes do not invalidate a still-eligible stable key. |

`ChatMessage` schema:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `role` | string | yes | Typically `user`, `assistant`, or `system` |
| `content` | string | yes | Plain-text message content |

Notes:

- One active stream is allowed per session. A second concurrent `POST /api/chat` returns HTTP `409`.
- A turn starts only while the session's TTL has not elapsed. A `POST /api/chat` on a session past `expires_at` returns HTTP `401` `Session expired`, including one whose token was accepted while the previous turn, or a batch it dispatched, was still running and that finds no turn in progress by the time it would start (see Session Lifecycle).
- Stream envelopes have the shape `{seq, session_id, schema_version, event}`. For schema v1, the server projects each event through the v1 adapter, strips fields added after the v1 freeze, and skips event types not in the v1 wire contract while preserving cursor sequence gaps.
- The server resolves only eligible stable keys from its exact model registry and
  selection policy. Raw upstream model names are rejected.
- Recommended `context.channel` values: `web`, `cli`, `telegram`, `bot`. The
  field is free-form; gateways may route or scope behavior on it. Reference
  TUI and CLI clients use `"cli"`.
- `create_agent()` resolves the runtime for you. `create_gateway_app()` calls your `build_chat_runtime(session, request, channel, auth_manager, storage_root=...)`; a builder must accept the keyword-only `storage_root` (default `None`).

Example:

```bash
curl -N http://127.0.0.1:8000/api/chat \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain tool approval in one paragraph."}
    ],
    "context": {"channel": "web"},
    "user_id": "alice"
  }'
```

### GET /api/chat/subscribe

Resume or add a subscriber to the bearer-token session's active SSE turn.
Optional query parameters are `session_id` (must match the token), `after_seq`
(non-negative cursor, default `0`), and `client_label`. The negotiated schema
version comes from session init and cannot be changed here. A session with no
active turn returns `404`.

```bash
curl -N "http://127.0.0.1:8000/api/chat/subscribe?after_seq=17" \
  -H "Authorization: Bearer $SESSION_TOKEN"
```

### POST /api/chat/recap

Return a typed `session_recap` for the bearer-token session. `scope` is
`active_turn` (default) or `session_cumulative`.

```json
{"session_id":"sess_1234abcd5678","scope":"active_turn"}
```

`tool_calls_summary.measurements` maps run-count names to integers. Each name
holds the largest value seen, except `model_calls`, which is a total; none of
them is a bound. A name appears only when its source was seen in the run.

| Name | Source |
| --- | --- |
| interceptor counts (`calls_in_window`, `structured_reads`, `concurrent_sub_agents`, …) | `interceptor_measurement` events |
| `decisions_sent`, `frame_bytes` | `approval_decision_sent` events: approval decisions sent to an autonomous child and their frame bytes |
| `in_flight`, `day_invocations`, `day_charged_microusd`, `retained_bytes` | `learning_fork_launched` events |
| `lease_fence_bytes` | `skill_run_started` events |
| `model_calls` | count of `model_*` tool calls (MCP prefix ignored) |
| `autonomous_in_flight_at_start`, `credential_handoff_bytes` | added by the autonomous parent to its child's recap from the run record |

At terminal closure, the `session_recap` event's `usage` is the existing
authoritative `SessionUsageSummary`, captured after child-drain handling and before
the terminal event is emitted. It includes billed attempts and retries; recap
construction does not recount provider events. Canonical settlement persists
the first terminal recap and `detach` together under the existing session-log
append mutex, rolling back both on append failure before deciding the error
fallback. The recap precedes `detach`, which remains the final durable lifecycle
event. Live terminal projection reuses the committed JSON payload, including
list-valued usage event IDs.
On the next turn, the context builder projects the record as a previous-turn
recap, making its usage and tool measurements available to the agent.
An ordinary exception escaping the runner, including a failed compaction summary,
is converted to its canonical error terminal before that recap is persisted.
The original exception still propagates; a later server error emission cannot
replace the first terminal or its accounting.

### GET /api/workflow-outputs/{workflow_run_id}/{output_id}

Return the exact canonical bytes for a durable workflow-output attachment in
the bearer-token session. Use `?download=true` for attachment disposition.
The response includes `ETag`, `X-Content-SHA256`, and `X-Workflow-Output-Id`;
missing, unavailable, and integrity-failed output paths return structured
`404`, `503`, and `409` responses respectively.

### PUT /api/model-preferences/{capability_id}

Save an authenticated stable-key preference for a capability whose product
policy permits saved preferences. The bearer token must name a chat session.

```json
{
  "model_key": "anthropic.claude-opus-5-5",
  "effort": "high",
  "catalog_revision": "registry-r7"
}
```

The response is the stored receipt:

```json
{
  "capability": "session.driver",
  "model_key": "anthropic.claude-opus-5-5",
  "effort": "high"
}
```

The server validates the key and effort against the authenticated session's
current eligible choices. Clients should submit the `catalog_revision` supplied
in `capability_choices` as concurrency context and must not substitute aliases
or raw upstream names. Revision inequality alone does not reject a still-valid
key; a key or effort that is no longer eligible does fail closed.

### DELETE /api/model-preferences/{capability_id}

Clear the authenticated saved preference. The response contains the capability
with `model_key` and `effort` set to `null`. A subsequent turn that omits
selection uses the product policy default.

### POST /api/chat/cancel

Explicitly cancel the active turn for the bearer-token session. Passive SSE disconnects remain resumable through `/chat/subscribe`; use this endpoint for user-initiated Stop/cancel actions.

Headers:

- `Authorization: Bearer <session_token>`
- `Content-Type: application/json`

Request body:

```json
{
  "session_id": "sess_1234abcd5678"
}
```

### POST /api/chat/tool-result

Submit the result of a client-executed tool call.

This endpoint is part of the server contract, but the built-in `create_agent()` flow usually does not need it because MCP tools, local tools, code execution, and `run_agent` all execute on the backend. Use this endpoint only if your custom runtime creates pending client-side tool calls.

Headers:

- `Authorization: Bearer <session_token>`
- `Content-Type: application/json`

Request body:

```json
{
  "tool_call_id": "call_123",
  "nonce": "a1b2c3d4e5f6",
  "result": {"ok": true},
  "error": null
}
```

Schema:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `tool_call_id` | string | yes | Matches the pending tool call |
| `nonce` | string | yes | Anti-replay nonce issued by the server |
| `result` | object or null | no | Tool result payload |
| `error` | object or null | no | Tool error payload |

Success response:

```json
{"status": "ok"}
```

Current server behavior:

- The request succeeds only if the session contains a pending tool with status `pending`.
- If the tool is unknown, expired, or already submitted, the endpoint returns `404`, `410`, or `409`.

Example:

```bash
curl -s http://127.0.0.1:8000/api/chat/tool-result \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_call_id":"call_123",
    "nonce":"a1b2c3d4e5f6",
    "result":{"ok":true}
  }'
```

### POST /api/chat/tool-approval

Approve or deny a tool call after receiving a `tool_approval_request` SSE event.

Headers:

- `Authorization: Bearer <session_token>`
- `Content-Type: application/json`

Request body:

```json
{
  "tool_call_id": "call_123",
  "nonce": "a1b2c3d4e5f6",
  "approved": true,
  "allow_tool_type": false
}
```

Schema:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `tool_call_id` | string | yes | Matches the approval request event |
| `nonce` | string | yes | Anti-replay nonce from the approval request |
| `approved` | boolean | yes | `true` to execute, `false` to deny |
| `allow_tool_type` | boolean | no | Request a persistent grant for this approval; honored only when the request's `allow_persistent_approval` is true and the session's approval policy mints grants |

When the call has a durable approval record, the response contains its canonical
`approval` projection. A live SDK waiter receives one queue decision and the
response has no `continuation` field. Calls without a durable ledger retain the
`{"status": "ok"}` response.

After a native call has parked and its live queue has been removed, the gateway
resolves the original call and nonce under the current bearer's tenant. It
records the vote before invoking the installed host continuation adapter. The
receiving session address is not execution authority, and a missing cached
queue does not make the original approval disappear.

The response's `continuation` reports the installed owning adapter's delivery
status. Without an installed adapter it instead reports `available: false`,
`unavailable_owner`, and the safe original job and capture state; it does not
invent a delivery outcome. An accepted durable vote is not a claim that execution
resumed. An ended SDK call can accept its durable vote but cannot be restored as
a native prepared call. `outcome_unknown` means the effect outcome is unresolved,
not an invitation to automatically retry it.

The route calls the configured root adapter first. Only its `not_root` outcome
dispatches the host-private child adapter, using the approval ID and verified
tenant rather than a model-supplied task or receiving address. That host owner
restores the actual job and selects workflow versus ordinary child continuation
from its original admitted logical task. The HTTP route does not select a runner
or send arbitrary child approvals directly to a workflow service.

The control decision route keeps its control-session, tenant, channel/delegation,
and decision-role policy. Its run address likewise does not replace the
canonical tenant-bound approval and original parked job.

Example:

```bash
curl -s http://127.0.0.1:8000/api/chat/tool-approval \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_call_id":"call_123",
    "nonce":"a1b2c3d4e5f6",
    "approved":true,
    "allow_tool_type":false
  }'
```

### GET /api/health

Simple health check.

Response:

```json
{"status": "ok"}
```

Example:

```bash
curl -s http://127.0.0.1:8000/api/health
```

### Artifact endpoints

Read-only GET endpoints serve artifact JSON sidecars, UI-block payloads, and
letter binaries from per-user workspace storage. Artifact files are written
server-side by structured report doors or artifact-producing tools.

**Auth.** Requests may use the gateway session bearer token. Clients without a
session may instead send all seven signed `X-Agent-Claim-*` headers
(`Audience`, `Issued-At`, `Expiry`, `User-Id`, `User-Email`, `Nonce`,
`Signature`); the configured claim-signing authority verifies their HMAC and
expiry.

**Path safety.** Artifact and letter endpoints reject:

- `..`-traversal (raw and URL-encoded)
- Symlink escape outside the user's workspace
- Cross-user access (404, not 403, to avoid info-leak)

Two additional path-traversal-guard routes (`GET /api/artifacts/{path:path}` and `GET /api/letters/{path:path}`) catch all unsafe paths and return 400 or 404.

#### GET /api/artifacts/{ticker}/{skill}/latest

Return the JSON sidecar of the most recent artifact for `(ticker, skill)`.

Response: the raw JSON sidecar payload (shape depends on the skill's contract).

Status codes:
- `200` — artifact found, body is the JSON content
- `400` — unsafe path
- `404` — no artifact for this `(ticker, skill)`

Response headers:
- `Cache-Control: private, max-age=0`
- `ETag: W/"<mtime>-<size>"` (weak ETag for cheap re-fetch checks)

#### GET /api/artifacts/{ticker}/{skill}/{artifact_id}

Return the JSON sidecar of a specific artifact.

Response: the raw JSON sidecar payload.

Status codes: as above. `artifact_id` matches the JSON filename stem.

#### GET /api/artifacts/{ticker}

Return an index of available skills, latest artifact ids, and bounded recent
artifact ids for a ticker. Use `recent_artifact_ids` with
`GET /api/artifacts/{ticker}/{skill}/{artifact_id}` when a downstream reviewer
needs a non-latest producer sidecar without filesystem globbing.

Response body:

```json
[
  {
    "skill": "fundamental-research",
    "latest_artifact_id": "20260520T173401",
    "artifact_count": 2,
    "recent_artifact_ids": ["20260520T173401", "20260519T104455"]
  },
  {
    "skill": "earnings-scenarios",
    "latest_artifact_id": "20260518T091200",
    "artifact_count": 1,
    "recent_artifact_ids": ["20260518T091200"]
  }
]
```

Returns an empty array (`200`) when the ticker directory does not exist.

#### GET /api/letters/{ticker}/{artifact_id}

Return the `.docx` letter binary.

Response: `FileResponse` with media type `application/vnd.openxmlformats-officedocument.wordprocessingml.document`.

Status codes:
- `200` — letter found
- `400` — unsafe path
- `404` — letter not found

Response headers: `Cache-Control: private, max-age=0` + weak `ETag` (same scheme as JSON endpoints).

#### GET /api/ui-blocks/{ui_blocks_id}

Return the stored UI-block JSON payload for an authenticated user. It uses the
same bearer-or-signed-claim authentication and path-safety boundary as the
artifact reads.

### Control plane

`create_gateway_app()` also mounts the authenticated `/api/control` surface.
It owns control sessions, profiles, skills, schedules, runs and continuations,
batches, approvals, event streams, readable resources, artifacts, and health.
Dashboard, canvas, and HTML artifact routers are mounted under `/api` as well.
Use the generated OpenAPI document for exact control-plane request and response
models; those routes are intended for operator clients rather than the basic
chat flow.

## SSE Event Types

Each SSE message is a JSON object under a `data:` line with the envelope
`{"seq": ..., "session_id": ..., "schema_version": 1, "event": {...}}`.
The schemas below describe the inner `event` object.

### Core Events

#### `text_delta`

Emitted whenever text tokens arrive from the model.

Schema:

```json
{
  "type": "text_delta",
  "text": "partial text"
}
```

#### `thinking_delta`

Emitted when the provider exposes a streamed reasoning/thinking channel.

Schema:

```json
{
  "type": "thinking_delta",
  "text": "partial reasoning text"
}
```

#### `tool_calls_planned`

Emitted once per model turn that ends with tool use, immediately after that
turn's `turn_complete` and before its first `tool_call_start`. It names every
tool call the model planned, in model order, before any of them executes.

Schema:

```json
{
  "type": "tool_calls_planned",
  "turn": 1,
  "calls": [
    {
      "tool_call_id": "toolu_123",
      "tool_name": "get_quote",
      "tool_input": {"ticker": "NVDA"},
      "display": {"label": "Get quote", "detail": "NVDA"}
    }
  ]
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `turn` | integer | The `turn_complete` turn these calls belong to |
| `calls` | array | One entry per planned call: `tool_call_id`, `tool_name`, `tool_input` (the redacted provider input, as persisted on the assistant message), and `display` (the label object `tool_call_start` carries, built the same way, or `null`) |

#### `tool_call_start`

Emitted before a tool runs.

Schema:

```json
{
  "type": "tool_call_start",
  "tool_call_id": "toolu_123",
  "tool_name": "code_execute",
  "tool_input": {"code": "print(1 + 1)"},
  "execution_location": "backend",
  "call_index": 0
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `tool_call_id` | string | Provider tool call id |
| `tool_name` | string | Tool name |
| `tool_input` | object | Tool payload |
| `execution_location` | string, optional | Added when the runtime resolves location metadata |
| `call_index` | integer | Zero-based call index for the turn |

#### `tool_call_complete`

Emitted after a tool finishes.

Schema:

```json
{
  "type": "tool_call_complete",
  "tool_call_id": "toolu_123",
  "tool_name": "code_execute",
  "result": {"stdout": "2\n", "stderr": "", "return_code": 0},
  "error": null,
  "duration_ms": 143,
  "server": null,
  "execution_location": "backend"
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `tool_call_id` | string | Provider tool call id |
| `tool_name` | string | Tool name |
| `result` | object or null | Tool result payload |
| `error` | object or null | Tool error payload |
| `duration_ms` | integer | End-to-end tool duration |
| `server` | string or null | MCP server name when relevant |
| `execution_location` | string, optional | Added when the runtime resolves location metadata |

A model-visible tool error built by the gateway is the tool result content
`{"error": {"code", "message", "sub_code"?, "data"}}`. `data` always carries
`next_call`: the producer's suggested next action, or `null` when there is no
next action. A `tool_excluded` error also names the excluding set in
`data.excluding_owner` (`profile`, `skill_or_operation_grant` or `role`) and returns the same
result, with `next_call: null`, on every repeat.

#### `tool_call_interrupted`

Emitted when a runtime knows that a started tool call will not produce a
completion, or synthesized when recovery finds such an unresolved call.

Schema:

```json
{
  "type": "tool_call_interrupted",
  "tool_call_id": "toolu_123",
  "tool_name": "code_execute",
  "tool_input": {"code": "print(1 + 1)"},
  "original_started_at": 1770000000.1,
  "discovered_at": 1770000060.2,
  "reason": "cancelled",
  "tool_risk": "dangerous",
  "runner_id": "runner_abc",
  "role": "writer",
  "sub_agent_id": "sub0:sess_1234"
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `tool_call_id` | string | Provider tool call id from the original `tool_call_start` |
| `tool_name` | string or null | Original tool name |
| `tool_input` | object or null | Redacted original tool payload |
| `original_started_at` | number or null | Original `tool_call_start.started_at` timestamp |
| `discovered_at` | number | Unix timestamp when the runtime or recovery recorded this event |
| `reason` | string, optional | Immediate runtime reason such as `cancelled` or `tool_error`; absent when recovery cannot know the cause |
| `tool_risk` | string | Risk classification for the tool name |
| `runner_id` | string or null | Runner that originally started the tool |
| `role` | string | Runner role from the original tool start event. Defaults to `writer` when absent. |
| `sub_agent_id` | string, optional | Present when the original tool start event included a sub-agent id. |

#### `stream_complete`

Terminal run-settlement event. `terminal_disposition` distinguishes a successfully
completed run from an intentional interruption. This is not necessarily transport
EOF: citation validation can follow the terminal event. Continue reading through
EOF to receive those settlements; do not persist or present an interrupted run as
successful.

Schema:

```json
{
  "type": "stream_complete",
  "terminal_disposition": "completed",
  "usage": {
    "input_tokens": 123,
    "output_tokens": 45,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "estimated_cost": 0.0012,
    "est_system_tokens": 5344,
    "est_tools_tokens": 12980
  }
}
```

`est_system_tokens` and `est_tools_tokens` come from the server's
`TokenEstimateSnapshot`. They are character-derived `chars/4` proxies for the
turn's system-prompt and tool-definition sections, respectively, not provider
token counts.

`terminal_disposition` is `"completed"` or `"interrupted"`. Interrupted
closures also carry a machine-readable `reason`, such as `"operator_pause"`
or `"soft_budget_abort"`.

The first terminal event fixes the run outcome; trailing events cannot replace it. An unknown non-null
`terminal_disposition` is a protocol error and must fail closed; it must never
be projected as successful completion. A missing disposition is interpreted
as completed only when reading pre-cutover event history.

#### `turn_complete`

Emitted at the end of each agent turn within a stream. This is a per-turn lifecycle event and is distinct from `stream_complete`, which closes the whole stream.

Schema:

```json
{
  "type": "turn_complete",
  "turn": 1,
  "usage": {
    "input_tokens": 123,
    "output_tokens": 45,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "estimated_cost": 0.0012
  },
  "est_input_tokens": 14210,
  "request_input_tokens": 15000,
  "context_window": 200000
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `turn` | integer | One-based turn counter within the stream |
| `usage` | object | Per-turn billing usage, additive across provider calls and server-side iterations, including compaction summary requests. Current runner emission includes `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens`; this aggregate is not a context measurement. |
| `est_input_tokens` | integer | Character-derived estimate of the last submitted request's prompt |
| `request_input_tokens` | integer or null | Provider-reported absolute prompt tokens of the final invocation, counting cached tokens once. For Anthropic, this is the last server-side iteration's input plus cache-read and cache-creation tokens, not the sum of iterations. Zero is a valid measurement; `null` means unavailable. |
| `context_window` | integer | The model's context window |

The TUI uses `request_input_tokens` as its sole actual context measurement.
An unavailable measurement (including an absent field in older events) clears
the displayed context rather than substituting aggregate `usage` or the
character-derived estimate. A reported zero remains known zero.

#### `error`

Terminal error event for the SSE stream.

Schema:

```json
{
  "type": "error",
  "error": "RuntimeError: something failed"
}
```

### Tool Flow Events

#### `tool_approval_request`

Emitted when `ToolDispatcher` requires human approval.

Schema:

```json
{
  "type": "tool_approval_request",
  "tool_call_id": "toolu_123",
  "nonce": "a1b2c3d4e5f6",
  "tool_name": "code_execute",
  "tool_input": {"code": "print(1 + 1)"},
  "resolved_qualifier": "subprocess",
  "reason": "Code execution requires explicit approval.",
  "allow_persistent_approval": true
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `tool_call_id` | string | Tool call id you must echo back |
| `nonce` | string | Anti-replay nonce |
| `tool_name` | string | Tool name |
| `tool_input` | object | Proposed tool input |
| `resolved_qualifier` | string | Tool-type qualifier (for code execution, the backend) |
| `reason` | string | Free-form context for why approval is needed. May be empty. Clients render this in the approval prompt UI when non-empty. |
| `allow_persistent_approval` | boolean | Whether the gateway will accept `allow_tool_type=true` for this approval. When false, the client should not offer the "approve all of this tool type" option. |

#### `headless_auto_deny`

Emitted when a tool approval request is automatically denied because the runtime is configured to avoid permission prompts in a headless context.

Schema:

```json
{
  "type": "headless_auto_deny",
  "tool_call_id": "toolu_123",
  "tool_name": "code_execute",
  "reason": "Tool 'code_execute' requires static approval in headless context",
  "source": "static"
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `tool_call_id` | string | Provider tool call id |
| `tool_name` | string | Tool name |
| `reason` | string | Human-readable denial reason |
| `source` | string | `static` for static approval rules or `interceptor` for interceptor approval requests |

#### `tool_output_chunk`

Emitted by tools that stream their own output, currently used by built-in code execution.

Schema:

```json
{
  "type": "tool_output_chunk",
  "tool_call_id": "toolu_123",
  "tool_name": "code_execute",
  "stream": "stdout",
  "text": "line 1\n",
  "seq": 1
}
```

#### `interceptor_decision`

Emitted when a runtime interceptor warns or denies a tool.

Schema:

```json
{
  "type": "interceptor_decision",
  "tool_call_id": "toolu_123",
  "tool_name": "write_file",
  "action": "warn",
  "code": "policy_warning",
  "message": "Writing outside the workspace is discouraged."
}
```

### Lifecycle Events

#### `heartbeat`

Liveness event appended every 60 seconds while a model turn streams. It reports
how long the turn has run and how recently it made progress, so a client can show
that a long turn is still working. Clients may render it (for example
`working · 3m12s · last progress 4s ago`) and must not treat it as content or as
a terminal event.

Fields:

- `elapsed_s`: integer seconds since the turn started
- `last_progress_s`: integer seconds since the last progress event
- `events`: integer count of progress events so far

Schema:

```json
{
  "type": "heartbeat",
  "elapsed_s": 192,
  "last_progress_s": 4,
  "events": 37,
  "timestamp": 1770000000
}
```

Transport keepalive is separate: after 15 seconds without a wire emission, the
stream yields the SSE comment `:keepalive`. Filtered internal log entries do not
reset that idle interval. The comment is not a data event and carries no payload.

#### `refusal`

Emitted when the provider ends a model turn with `stop_reason: "refusal"`. It
arrives after the turn's last `text_delta` and before `stream_complete`. The text
the model delivered before the stop stays the turn's output; the gateway does not
retry, trim, or reset the conversation. The run then closes with
`stream_complete.terminal_disposition = "completed"`; this event, not the
disposition, is what marks the turn as refused. No `error` event is emitted for a
refusal. The persisted `assistant_message` keeps its text blocks and
`stop_reason: "refusal"`.

Fields:

- `category`: the provider's `stop_details.category` verbatim (for Anthropic:
  `cyber`, `bio`, `frontier_llm`, `reasoning_extraction`, `general_harms`), or
  `null` when the provider sent none
- `explanation`: the provider's `stop_details.explanation` verbatim, or `null`
- `guidance`: fixed text telling the user that Anthropic recommends starting a
  fresh conversation before continuing

Schema:

```json
{
  "type": "refusal",
  "category": "reasoning_extraction",
  "explanation": "The request asks the model to reproduce its internal reasoning.",
  "guidance": "Anthropic recommends starting a fresh conversation before continuing; further turns in this context may be refused again."
}
```

A `run_agent` child whose turn is refused settles `interrupted` with
`terminal_reason` `refusal: <category>` (`refusal` when no category was sent)
and keeps the text it delivered as its terminal narrative.

#### `interrupted`

Lifecycle audit event emitted when a runner is interrupted or recovered. This
event may occur before later stream activity and is therefore not a transport
terminal. A run that ultimately closes as interrupted does so with
`stream_complete.terminal_disposition = "interrupted"`.

Schema:

```json
{
  "type": "interrupted",
  "reason": "recovered_on_attach",
  "runner_id": "runner_previous",
  "role": "writer",
  "last_completed_seq": 42,
  "recovered_by_runner_id": "runner_current",
  "recovered_at": 1770000060.2
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `reason` | string | Interruption reason such as `soft_budget_abort` (with `kind`, `observed`, `threshold`, `grace_turns`), `graceful_shutdown`, `sub_agent_cancelled`, or `recovered_on_attach` |
| `runner_id` | string or null | Interrupted runner id |
| `role` | string or null | Runner role, typically `writer` or `sub_agent` |
| `last_completed_seq` | integer or null | Last durable sequence number considered safe by the runner |
| `recovered_by_runner_id` | string, optional | Present when another runner recovered the interrupted runner |
| `recovered_at` | number, optional | Unix timestamp when recovery occurred |

#### `stream_retry`

Emitted when the runner retries a transient stream failure or watchdog stall.

Schema:

```json
{
  "type": "stream_retry",
  "attempt": 1,
  "error": "Stream watchdog: no stream events for 61s"
}
```

#### `stream_error`

Emitted when SSE serialization itself fails.

Schema:

```json
{
  "type": "stream_error",
  "error": "SSE serialization failed: ..."
}
```

#### `compaction`

Emitted when the provider returns a compaction block and the runner keeps going.

Schema:

```json
{
  "type": "compaction",
  "chars": 842
}
```

#### `soft_budget_steer` and `soft_budget_final`

Informational markers from the one soft containment guard on delegated runs
(`runner_soft_budget.py`; `docs/engineering/soft-budget-containment-reference.md`).
Neither stops the run. `soft_budget_steer` says the run's provider requests
(turns plus `max_tokens` continuation segments) reached the threshold from the
bundled `SOFT_TURN_BUDGET` table, and a wrap-up notice naming the `observed`
request count and the `threshold` rides the child's next turn;
`soft_budget_final` says 1.5x the threshold was reached and the next turn
is the forced final report. `kind` is always `"turns"`; `observed` is the
request count. Only a run that still has not answered after the grace turns is
aborted: an `interrupted` event with `reason: "soft_budget_abort"` carries
`kind`, `observed`, `threshold` and `grace_turns`, and the stream then closes
with an interrupted `stream_complete` whose `reason` is `soft_budget_abort`.
A primary run emits none of these.

For interactive children, the durable steer/final record also contains
`notice`: the exact text injected into the subsequent provider request,
alongside `kind`, `observed`, and `threshold`. These measurements remain
readable after the in-memory run has ended.

Schema (both):

```json
{
  "type": "soft_budget_steer",
  "kind": "turns",
  "observed": 10,
  "threshold": 10
}
```

### Background Task Events

The background-task events share task-correlation fields.

Task-correlation fields:

| Field | Type | Notes |
| --- | --- | --- |
| `task_id` | string | Background task id, for example `bg_0` |
| `owner_runner_id` | string or null | Runner that owns the task |
| `owner_role` | string or null | Role of the owner runner |
| `sub_agent_id` | string or null | Derived sub-agent session id for the task |
| `parent_turn_id` | string or null | Parent turn id associated with the task |
| `call_index` | integer or null | Tool-call index associated with the task |
| `task_type` | string | Task type. Current background-agent tasks use `background`. |
| `provider_name` | string or null | Provider selected for the task |
| `model` | string or null | Model selected for the task |
| `original_task_id` | string, optional | Present on resumed tasks |

#### `task_registered`

Emitted when a background task is registered for a multi-agent or `run_agent` workflow.

Schema:

```json
{
  "type": "task_registered",
  "task_id": "bg_0",
  "owner_runner_id": "runner_abc",
  "owner_role": "writer",
  "sub_agent_id": "sub0:sess_1234",
  "parent_turn_id": "turn-1",
  "call_index": 0,
  "task_type": "background",
  "provider_name": "anthropic",
  "model": "claude-sonnet-4-6",
  "agent_name": "researcher",
  "parent_session_id": "sess_1234",
  "metadata": {
    "owner_runner_id": "runner_abc",
    "owner_role": "writer",
    "sub_agent_id": "sub0:sess_1234",
    "parent_turn_id": "turn-1",
    "call_index": 0,
    "task_type": "background",
    "provider_name": "anthropic",
    "model": "claude-sonnet-4-6",
    "resumable": true
  },
  "started_at": 1770000000.1
}
```

Additional fields:

| Field | Type | Notes |
| --- | --- | --- |
| `agent_name` | string or null | Background agent name supplied by the caller |
| `parent_session_id` | string | Parent gateway session id |
| `metadata` | object | Task metadata snapshot. Includes the correlation fields and may include `resumable` or `original_task_id`. |
| `started_at` | number | Task registration timestamp |
| `resumable` | boolean, optional | Included inside `metadata` when the tool input included `resumable` |

#### `task_completed`

Emitted when a background task reaches a completed or failed final state.

Schema:

```json
{
  "type": "task_completed",
  "task_id": "bg_0",
  "owner_runner_id": "runner_abc",
  "owner_role": "writer",
  "sub_agent_id": "sub0:sess_1234",
  "parent_turn_id": "turn-1",
  "call_index": 0,
  "task_type": "background",
  "provider_name": "anthropic",
  "model": "claude-sonnet-4-6",
  "final_state": "completed",
  "completed_at": 1770000030.2,
  "result": {
    "kind": "report",
    "version": "1",
    "report": {
      "summary": "done",
      "findings": [],
      "artifacts": [],
      "caveats": []
    },
    "usage": {"input_tokens": 120, "output_tokens": 24},
    "tools_used": ["memory_recall"],
    "fms_results": null,
    "artifact_events": null,
    "warning": null
  },
  "error": null
}
```

Additional fields:

| Field | Type | Notes |
| --- | --- | --- |
| `final_state` | string | Final task state. Current runner emission uses `completed`, `failed`, or `killed`. |
| `completed_at` | number | Unix timestamp when the completion event was appended |
| `result` | object or null | Terminal result retained by the task contract. Consumers should use its typed fields and referenced content rather than parse free-form relay text. |
| `error` | object or null | Background task error payload |

#### `parent_message_sent`

Emitted when an admitted message reaches a running sub-agent or the root job's
mailbox. `task_id` identifies the recipient job; `runner_id` and `role`
identify the authoring runner. On a send to a child, `sub_agent_id` identifies
the recipient child and is supplied by task correlation. On a send to the
root, the author's durable namespace supplies any `sub_agent_id`.
The example below is a parent-authored send to a child.

Schema:

```json
{
  "type": "parent_message_sent",
  "task_id": "bg_0",
  "owner_runner_id": "runner_abc",
  "owner_role": "writer",
  "sub_agent_id": "sub0:sess_1234",
  "parent_turn_id": "turn-1",
  "call_index": 0,
  "task_type": "background",
  "provider_name": "anthropic",
  "model": "claude-sonnet-4-6",
  "message_id": "msg_123",
  "sender": {
    "session_id": "sess_1234",
    "user_id": "alice"
  },
  "sent_at": 1770000010.1,
  "message": "Please narrow the search."
}
```

Additional fields:

| Field | Type | Notes |
| --- | --- | --- |
| `message_id` | string | Caller-supplied id or generated UUID |
| `sender` | object | Sender metadata with `session_id` and `user_id`, each string or null |
| `sent_at` | number | Unix timestamp when the parent message was sent |
| `message` | string | Message text delivered to the sub-agent |

Consumption is bound to the durable `assistant_message` fact through
`parent_message_consumptions`: exact `task_id`, `message_id`,
`parent_message_seq` and `consumer_turn`. Its durable
`parent_message_consumed` audit also references `assistant_message_seq`.
Root consumption binds the recipient to the bound registry owner's `job_id`.
The sent fact retains its author's runner, role and optional child namespace;
these need not equal the root consumer's writer identity. The consuming
assistant's writer must have an earlier matching `attach`, including during
historical replay. Root consumers do not acquire a child namespace.
Child consumption retains its existing task-correlated lineage. Exact
message/body/sequence and duplicate/conflict checks remain enforced; replay
does not duplicate the consumption audit.

### Typed Domain Events

The canonical frozen dataclasses and complete 14-type union live in
`agent_gateway.events` (see [API reference](./api-reference.md)). Seven types
are run-scoped: `skill_run_started`, `skill_result_captured`, `artifact_ready`,
`ui_blocks_ready`, `artifact_updated`, `aggregate_ready`, and
`artifact_failed`. The remaining types are `agent_completion`,
`workflow_output_attached`, `typed_recommendations_extracted`,
`artifact_unavailable`, `tool_approval_request`, `tool_approval_decided`, and
`session_recap`. The representative payloads below cover the core skill and
artifact lifecycle; use the dataclasses for exact fields of every type.

#### `skill_run_started`

Emitted once at the start of a skill-framework sub-agent run.

```json
{
  "type": "skill_run_started",
  "skill_run_id": "run_8a3f",
  "skill": "fundamental-research",
  "ticker": "AAPL",
  "scope": "ticker",
  "portfolio_id": null,
  "ts": 1770000000.0
}
```

#### `skill_result_captured`

Emitted when the runtime captures a structured skill result envelope.

```json
{
  "type": "skill_result_captured",
  "skill_run_id": "run_8a3f",
  "skill": "fundamental-research",
  "ticker": "AAPL",
  "scope": "ticker",
  "portfolio_id": null,
  "exit_code": 0,
  "outcome": "success",
  "status": "noop",
  "gate_code": "PROCEED",
  "artifact_refs": ["artifacts/AAPL/fundamental-research/result.json"],
  "proposal_ids": [],
  "verdict_echo": {
    "verdict": "MATERIAL_POSITIVE",
    "confidence": "HIGH",
    "one_line_summary": "Capital allocation continues to compound at about 22% ROIC."
  },
  "fms_results": [],
  "artifact_events": [],
  "output_memory_file": null,
  "cost_usd": 0.12,
  "duration_s": 18.4,
  "compaction_count": 1,
  "error": null,
  "warnings": []
}
```

`compaction_count` is the non-negative number of context-compaction events observed during the skill run; legacy captures that omit it decode as zero. `verdict_echo` is optional; when present it is the UI/control-plane verdict summary source. It is produced by structured tools and FMS envelopes, not by parsing final markdown.
The event may also carry `approval_outcome`, `approval_id`, and
`approval_tool_name`.

#### `artifact_ready`

Emitted when a structured report door or artifact-producing tool writes a JSON sidecar artifact (and optionally a binary artifact such as a `.docx`) to per-user workspace storage. Pairs with the artifact-read endpoints documented above.

```json
{
  "type": "artifact_ready",
  "skill_run_id": "run_8a3f",
  "ticker": "AAPL",
  "skill": "fundamental-research",
  "artifact_id": "20260520T173401",
  "artifact_path": "data/users/alice/workspace/artifacts/AAPL/fundamental-research/20260520T173401.json",
  "binary_artifact_path": "data/users/alice/workspace/letters/AAPL/20260520T173401.docx",
  "contract_name": "fundamental_research_v1",
  "data_source": "live",
  "scope": "ticker",
  "portfolio_id": null,
  "ts": 1770000010.0
}
```

Field types: `binary_artifact_path` is a string or null. `data_source` is `"live" | "fixture"`.

#### `aggregate_ready`

Emitted when an aggregate view-model reaches the sources-complete state.

```json
{
  "type": "aggregate_ready",
  "skill_run_id": "run_8a3f",
  "ticker": "AAPL",
  "view_model_id": "ticker_summary_v1",
  "trigger": {"kind": "artifact_ready", "source": "fundamental-research"},
  "sources_complete": true,
  "ts": 1770000011.0
}
```

`trigger.kind` is `"artifact_ready" | "tool_response"`.

#### `artifact_failed`

Emitted when a structured report door or artifact-producing tool fails to produce an artifact.

```json
{
  "type": "artifact_failed",
  "skill_run_id": "run_8a3f",
  "ticker": "AAPL",
  "skill": "fundamental-research",
  "error_code": "schema_drift",
  "error_detail": "expected field 'confidence' missing from structured artifact payload",
  "source_path": "artifacts/AAPL/fundamental-research/20260520T173401.json",
  "ts": 1770000010.0
}
```

`error_code` is one of `"validation" | "missing_contract" | "schema_drift" | "tool_write_failed" | "other"`; `tool_call_id` may also be present.

#### `artifact_unavailable`

Renderer-side event with no associated skill run. Surfaced when the UI/aggregator looks up an artifact for `(ticker, skill)` and finds it absent. The `affordance` field is a short user-facing hint.

```json
{
  "type": "artifact_unavailable",
  "ticker": "AAPL",
  "skill": "fundamental-research",
  "reason": "no_runs_yet",
  "affordance": "Run /research to populate this view.",
  "ts": 1770000020.0
}
```

`reason` is one of `"no_runs_yet" | "stale" | "fixture_only" | "auth_blocked"`.

## Session Lifecycle

Typical flow:

1. `POST /api/chat/init`
2. `POST /api/chat`
3. Receive SSE events
4. If needed, handle `tool_approval_request`
5. Call `POST /api/chat/tool-approval`
6. Observe the run disposition at `stream_complete` or `error`; consume through EOF for trailing settlements.

The same session can be reused for multiple turns until `expires_at`. The TTL bounds
the session between units of work, not one in progress: a turn accepted before
`expires_at` keeps the token its tool calls authenticate with valid until the turn
publishes its terminal event (`stream_complete` or `error`), and a batch admitted on
the session's token keeps it valid until the batch finishes or is cancelled, even
after the turn that dispatched it has ended. Admission of a new turn is the clock
alone: a running turn is refused with 409 before admission is read, and once a turn
has ended the TTL applies, so an expired session cannot start another turn even
while a batch it dispatched still runs.

## Tool Approval Flow

When a tool needs approval:

1. The server emits `tool_approval_request`.
2. Your client decides whether to approve or deny.
3. Your client calls `POST /api/chat/tool-approval`.
4. A live SDK waiter resumes through its existing queue. A parked native call is
   delivered by its installed host owner, if available; otherwise the response
   reports the missing continuation owner without claiming execution.

There is no session approval cache: each approval decides one call. For code execution, the qualifier usually resolves to `docker` or `subprocess`, naming the backend the approval covers.

The default `SingleUserApprovalPolicy` never mints a persistent grant: it asks only for a live order, and decides each one per call. When a custom policy permits it, the same approval can also mint a durable persistent grant. A later grant match still creates a new per-call approval/authorization record; clients must not treat the grant ID as executable authority or assume it reuses an earlier plan. Grant and per-call expiry, revocation-race, scope, and predicate facts are documented in [Persistent grants and per-call authorization](architecture.md#persistent-grants-and-per-call-authorization). The API does not currently let a client select a default grant TTL.

## Notes For Frontend Clients

- Treat `stream_complete`, `error`, and `stream_error` as terminal events.
- On `stream_complete`, use `terminal_disposition` to distinguish success from interruption; both close the transport.
- Do not treat lifecycle and background-task events such as `turn_complete`, `interrupted`, `task_registered`, or `task_completed` as terminal.
- `heartbeat` is liveness, not content: render it as a progress indicator or ignore it, but never append it to the assistant message.
- `refusal` is a notice, not a failure and not a terminal: keep the text already delivered, show the category, explanation and guidance under it, and wait for `stream_complete`.
- Do not assume every stream has `thinking_delta`, `tool_output_chunk`, or `tool_approval_request`.
- Preserve `tool_call_id` and `nonce` exactly when you answer approval requests.
