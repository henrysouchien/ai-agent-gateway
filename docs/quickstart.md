# Quickstart

This gets you from `pip install` to a streaming agent response in about five minutes.

`create_agent()` uses Anthropic by default. Switch to OpenAI with `provider="openai"` after installing the OpenAI extra, or move to `create_gateway_app()` when you need more runtime control.

## 1. Install

```bash
pip install "ai-agent-gateway[anthropic]" uvicorn
```

OpenAI variant:

```bash
pip install "ai-agent-gateway[openai]" uvicorn
```

## 2. Set Your Provider Credential

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

If you prefer OAuth-style auth, `create_agent()` also supports `ANTHROPIC_AUTH_TOKEN`.

For OpenAI:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 3. Create `agent.py`

Save this as `agent.py`:

```python
from agent_gateway import create_agent

app = create_agent(
  "You are a concise assistant. Answer clearly and use short paragraphs."
)
```

To use OpenAI instead:

```python
from agent_gateway import create_agent

app = create_agent(
  "You are a concise assistant. Answer clearly and use short paragraphs.",
  provider="openai",
)
```

That is enough to start a working gateway. By default:

- the HTTP API is mounted under `/api`
- session init accepts any non-empty API key unless you configure `valid_api_keys`
- the server streams responses over SSE

## 4. Run It With Uvicorn

```bash
uvicorn agent:app --reload --port 8000
```

The server is now listening on `http://127.0.0.1:8000`.

## 5. Initialize a Session

The chat API is session-based. First, exchange an API key for a JWT session token.

```bash
curl -s http://127.0.0.1:8000/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"local-demo-key"}'
```

Example response:

```json
{
  "session_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "session_id": "sess_1234abcd...",
  "expires_at": 1770000000
}
```

Capture the token for the next step:

```bash
SESSION_TOKEN=$(curl -s http://127.0.0.1:8000/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"local-demo-key"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_token"])')
```

## 6. Send a Chat Request

Use the session token as a bearer token and post to `/api/chat`. Keep `-N` so `curl` prints the SSE stream as it arrives.

```bash
curl -N http://127.0.0.1:8000/api/chat \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain in two sentences what this package does."}
    ],
    "context": {"channel": "web"}
  }'
```

## 7. Watch the Response Stream

You will see SSE events like this:

```text
data: {"type":"text_delta","text":"This package turns an agent into an HTTP service with sessions, streaming, and tool dispatch built in. "}

data: {"type":"text_delta","text":"It lets you start simple with a prompt and then add MCP tools, local tools, skills, approvals, and code execution."}

data: {"type":"stream_complete","usage":{"input_tokens":123,"output_tokens":45,"estimated_cost":0.0012}}
```

The most important client-visible event types are:

- `text_delta`
- `thinking_delta`
- `tool_call_start`
- `tool_call_complete`
- `tool_approval_request`
- `tool_output_chunk`
- `stream_complete`
- `error`

Full event and endpoint details: [HTTP API](./http-api.md)

## 8. Next Steps

- Add MCP tools: see [`../examples/02-mcp-tools/`](../examples/02-mcp-tools/)
- Add local Python tools: see [`../examples/03-local-tools/`](../examples/03-local-tools/)
- Add code execution: see [`../examples/04-code-execution/`](../examples/04-code-execution/)
- Add skills and sub-agents: see [`../examples/05-skills/`](../examples/05-skills/)
- Run headless (cron/batch): see [`../examples/09-autonomous/`](../examples/09-autonomous/)
- Move to `create_gateway_app()`: see [Architecture](./architecture.md) and [API Reference](./api-reference.md)

## Troubleshooting

### The agent returns a stub response

`create_agent()` emits a stub response when no provider credential is configured. For Anthropic, set `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`. For OpenAI, set `OPENAI_API_KEY` or pass `api_key=...`.

### My MCP server will not start

If you use an `npx`-based MCP server, install Node.js first and verify the command works outside the gateway.

### Code execution does not use Docker

That is expected when Docker is unavailable or the configured image is missing. The gateway prefers Docker and falls back to subprocess execution if subprocess support is enabled.

### Session init fails with 401

If you configured `valid_api_keys`, `/api/chat/init` will reject any API key not in that allowlist.
