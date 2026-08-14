# HTTP API

This page documents the wire protocol exposed by `create_gateway_app()` and `create_agent()`.

By default the API is mounted under `/api`. If you set `GatewayServerConfig.prefix`, prepend that value to every route in this document.

## Transport

- `POST /chat/init` returns JSON
- `POST /chat/tool-result` returns JSON
- `POST /chat/tool-approval` returns JSON
- `POST /chat/cancel` returns JSON
- `GET /chat/subscribe` returns an SSE stream for an active turn
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
        "model_key": "anthropic.claude-opus-5",
        "label": "Opus 5",
        "effort": "high",
        "reason": "platform_default"
      },
      "notices": [],
      "choices": [
        {
          "model_key": "anthropic.claude-opus-5",
          "label": "Opus 5",
          "supported_efforts": ["none", "low", "medium", "high"],
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
| `expires_at` | integer | Unix timestamp |
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
  "model_key": "anthropic.claude-sonnet-5",
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
- Stream envelopes have the shape `{seq, session_id, schema_version, event}`. For schema v1, the server projects each event through the v1 adapter, strips fields added after the v1 freeze, and skips event types not in the v1 wire contract while preserving cursor sequence gaps.
- The server resolves only eligible stable keys from its exact model registry and
  selection policy. Raw upstream model names are rejected.
- Recommended `context.channel` values: `web`, `cli`, `telegram`, `bot`. The field is free-form; gateways may route or scope behavior on the channel value, and analytics commonly use it. Planned reference dev clients (the in-flight `@ai-agent-gateway/tui` and `ai-agent-gateway-cli` packages) will send `"cli"` as the canonical dev-surface value.
- `create_agent()` resolves the runtime for you. `create_gateway_app()` calls your `build_chat_runtime(session, request, channel, auth_manager)`.

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

### PUT /api/model-preferences/{capability_id}

Save an authenticated stable-key preference for a capability whose product
policy permits saved preferences. The bearer token must name a chat session.

```json
{
  "model_key": "anthropic.claude-opus-5",
  "effort": "high",
  "catalog_revision": "registry-r7"
}
```

The response is the stored receipt:

```json
{
  "capability": "session.driver",
  "model_key": "anthropic.claude-opus-5",
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
| `allow_tool_type` | boolean | no | Persist approval for this tool type in the current session |

Success response:

```json
{"status": "ok"}
```

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

Added in 0.15.0. Four read-only GET endpoints serve artifact JSON sidecars and `.docx` letter binaries from per-user workspace storage (`data/users/<user>/workspace/artifacts/` and `.../letters/`). The artifact files are written server-side by structured report doors or artifact-producing tools; these endpoints are read-only.

**Auth: signed end-user claim, not session JWT.** Each request must carry seven `X-Agent-Claim-*` headers (`Audience`, `Issued-At`, `Expiry`, `User-Id`, `User-Email`, `Nonce`, `Signature`). The signature is HMAC-SHA256 over `audience\nissued_at\nexpiry\nuser_id\nuser_email\nnonce` using a key the gateway operator pre-shares with the artifact client. This is the same signed-claim scheme Theme A introduced for `POST /api/chat/init`; the verifier is shared.

**Path safety.** All four endpoints reject:
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

## SSE Event Types

Each SSE message is a JSON object under a `data:` line.

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

Terminal transport-closure event for the SSE stream. `terminal_disposition`
distinguishes a successfully completed run from an intentional interruption.
Clients must close the stream for either disposition, but must not persist or
present an interrupted run as successful.

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
    "estimated_cost": 0.0012
  }
}
```

`terminal_disposition` is `"completed"` or `"interrupted"`. Interrupted
closures also carry a machine-readable `reason`, such as `"operator_pause"`,
`"budget_exceeded"`, or `"max_turns_reached"`.

Consumers must stop at the first terminal event. An unknown non-null
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
  }
}
```

Fields:

| Field | Type | Notes |
| --- | --- | --- |
| `turn` | integer | One-based turn counter within the stream |
| `usage` | object | Per-turn token delta emitted by the runner. Current runner emission includes `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens`. |

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
| `resolved_qualifier` | string | Tool-type qualifier used for session approval caching |
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

Keep-alive event emitted every 15 seconds while the stream is open.

Schema:

```json
{
  "type": "heartbeat",
  "timestamp": 1770000000
}
```

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
| `reason` | string | Interruption reason such as `max_turns_reached`, `budget_exceeded`, `graceful_shutdown`, `sub_agent_cancelled`, or `recovered_on_attach` |
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

#### `max_turns_reached`

Nonterminal reason marker emitted when the runner stops because it exceeded
`max_turns`. The stream then closes with an interrupted `stream_complete`.

Schema:

```json
{
  "type": "max_turns_reached",
  "turn_count": 6,
  "max_turns": 5
}
```

#### `budget_exceeded`

Nonterminal reason marker emitted when the accumulated estimated cost crosses
`max_budget_usd`. The stream then closes with an interrupted
`stream_complete`.

Schema:

```json
{
  "type": "budget_exceeded",
  "total_cost": 0.42,
  "budget": 0.4
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
| `result` | `ChildReturn v1` object or null | Canonical background child result. `kind: report` is authoritative; `kind: unstructured` carries a typed terminal failure reason. |
| `error` | object or null | Background task error payload |

#### `parent_message_sent`

Emitted when the parent sends a message to a running background sub-agent.

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

### Skill Framework Events

Added in 0.15.0. Six typed events emitted when the host wires up the skill-framework profile contract — running an embedder that does not set `skill_run_id` + `profile` on sub-agent calls will never see these. Five carry a `skill_run_id` for run correlation; `artifact_unavailable` is renderer-side only and has no run. The Python dataclasses live in `agent_gateway.events` (see api-reference.md).

#### `skill_run_started`

Emitted once at the start of a skill-framework sub-agent run.

```json
{
  "type": "skill_run_started",
  "skill_run_id": "run_8a3f",
  "skill": "fundamental-research",
  "ticker": "AAPL",
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

`error_code` is one of `"validation" | "missing_contract" | "schema_drift" | "tool_write_failed" | "other"`.

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
6. Continue consuming SSE events until `stream_complete` or `error`

The same session can be reused for multiple turns until `expires_at`.

## Tool Approval Flow

When a tool needs approval:

1. The server emits `tool_approval_request`.
2. Your client decides whether to approve or deny.
3. Your client calls `POST /api/chat/tool-approval`.
4. The runner resumes and emits the rest of the tool lifecycle events.

If `allow_tool_type=true`, the server stores that approval in `session.approved_tool_types` for the rest of the session. For code execution, the qualifier usually resolves to `docker` or `subprocess`, so approval can be scoped to a specific backend.

When the configured policy permits it, the same approval can also mint a durable persistent grant. A later grant match still creates a new per-call approval/authorization record; clients must not treat the grant ID as executable authority or assume it reuses an earlier plan. Grant and per-call expiry, revocation-race, scope, and current predicate limitations are documented in [Persistent grants and per-call authorization](architecture.md#persistent-grants-and-per-call-authorization). The API does not currently let a client select a default grant TTL.

## Notes For Frontend Clients

- Treat `stream_complete`, `error`, and `stream_error` as terminal events.
- On `stream_complete`, use `terminal_disposition` to distinguish success from interruption; both close the transport.
- Do not treat lifecycle and background-task events such as `turn_complete`, `interrupted`, `task_registered`, or `task_completed` as terminal.
- Ignore `heartbeat` for rendering; it exists to keep the connection warm.
- Do not assume every stream has `thinking_delta`, `tool_output_chunk`, or `tool_approval_request`.
- Preserve `tool_call_id` and `nonce` exactly when you answer approval requests.
