# 02 MCP Tools

This example adds an inline filesystem MCP server so the agent can read and write files in the current directory.

## Prerequisites

- Anthropic API access
- Node.js, because the MCP server runs through `npx`

## Install

```bash
pip install "ai-agent-gateway[anthropic]" uvicorn
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Create A File To Read

```bash
printf 'Quarterly revenue grew 18 percent year over year.\nMargins improved by 220 basis points.\n' > notes.txt
```

## Run

```bash
uvicorn agent:app --reload --port 8000
```

## Chat

```bash
SESSION_TOKEN=$(curl -s http://127.0.0.1:8000/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"demo-key"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_token"])')
```

```bash
curl -N http://127.0.0.1:8000/api/chat \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Read notes.txt and rewrite it as two concise bullet points."
      }
    ],
    "context": {"channel": "web"}
  }'
```

## What It Shows

- `mcp_servers=` accepts inline MCP server definitions.
- MCP tools are discovered on startup and merged into the tool list.
- Filesystem access is scoped to the example directory because the server is launched with `.`.
