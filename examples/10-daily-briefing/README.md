# 10 Daily Briefing Project

This is a launch-ready `agent.yaml` project example. It reads local briefing inputs through a filesystem MCP server, then uses a callable skill to turn them into a concise daily update.

## Prerequisites

- Anthropic API access
- Node.js, because the filesystem MCP server runs through `npx`

## Install

```bash
pip install "ai-agent-gateway[anthropic]"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export USER_DATA_DIR="$PWD/.agent-data"
mkdir -p "$USER_DATA_DIR/gateway"
chmod 700 "$USER_DATA_DIR" "$USER_DATA_DIR/gateway"
```

## Run

From the package root:

```bash
cd examples/10-daily-briefing
agent run
```

## Try It

```bash
SESSION_TOKEN=$(curl -s http://127.0.0.1:8010/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"local-demo-key","user_id":"demo-user"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_token"])')
```

```bash
curl -N http://127.0.0.1:8010/api/chat \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "demo-user",
    "messages": [
      {
        "role": "user",
        "content": "Read feeds/product-updates.md and feeds/ops-notes.md, then use the daily-briefing skill to produce today'\''s briefing."
      }
    ],
    "context": {"channel": "web"}
  }'
```

## What It Shows

- A complete project config loaded by `agent run`
- MCP filesystem wiring through `agent.yaml`
- A callable markdown skill with persisted state enabled
- A 5-minute path for adapting the project to real RSS or notification MCP servers
