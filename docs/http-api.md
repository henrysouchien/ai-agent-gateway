# HTTP API

This page documents the wire protocol exposed by `create_gateway_app()` and `create_agent()`.

By default the API is mounted under `/api`. If you set `GatewayServerConfig.prefix`, prepend that value to every route in this document.

## Transport

- `POST /chat/init` returns JSON
- `POST /chat/tool-result` returns JSON
- `POST /chat/tool-approval` returns JSON
- `GET /health` returns JSON
- `POST /chat` returns an SSE stream

The chat stream uses standard server-sent events with `Content-Type: text/event-stream`. The server writes JSON payloads as `data:` lines and does not currently set named SSE event types.

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
  "api_key": "demo-key"
}
```

Schema:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `api_key` | string | yes | Must be non-empty. |

Response body:

```json
{
  "session_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "session_id": "sess_1234abcd5678",
  "expires_at": 1770000000
}
```

Schema:

| Field | Type | Notes |
| --- | --- | --- |
| `session_token` | string | JWT bearer token for later requests |
| `session_id` | string | Server-generated session id |
| `expires_at` | integer | Unix timestamp |

Example:

```bash
curl -s http://127.0.0.1:8000/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"demo-key"}'
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
  "model": "claude-sonnet-4-6"
}
```

Schema:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `messages` | array of `ChatMessage` | yes | Full message list for the current turn |
| `context` | object | no | Free-form request context; `channel` is commonly used |
| `model` | string or null | no | Per-request model override |

`ChatMessage` schema:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `role` | string | yes | Typically `user`, `assistant`, or `system` |
| `content` | string | yes | Plain-text message content |

Notes:

- One active stream is allowed per session. A second concurrent `POST /api/chat` returns HTTP `409`.
- The server verifies that `model` is in `GatewayServerConfig.allowed_models` when an allowlist is configured.
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
    "context": {"channel": "web"}
  }'
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

#### `stream_complete`

Terminal success event for the SSE stream.

Schema:

```json
{
  "type": "stream_complete",
  "usage": {
    "input_tokens": 123,
    "output_tokens": 45,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "estimated_cost": 0.0012
  }
}
```

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
  "resolved_qualifier": "subprocess"
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

Emitted when the runner stops because it exceeded `max_turns`.

Schema:

```json
{
  "type": "max_turns_reached",
  "turn_count": 6,
  "max_turns": 5
}
```

#### `budget_exceeded`

Emitted when the accumulated estimated cost crosses `max_budget_usd`.

Schema:

```json
{
  "type": "budget_exceeded",
  "total_cost": 0.42,
  "budget": 0.4
}
```

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

## Notes For Frontend Clients

- Treat both `stream_complete` and `error` as terminal events.
- Ignore `heartbeat` for rendering; it exists to keep the connection warm.
- Do not assume every stream has `thinking_delta`, `tool_output_chunk`, or `tool_approval_request`.
- Preserve `tool_call_id` and `nonce` exactly when you answer approval requests.
