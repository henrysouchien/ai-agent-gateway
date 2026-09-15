# 11 Research Report Project

This `agent.yaml` project turns a small local source pack into a short research report. It demonstrates a parent agent coordinating MCP reads with two callable skills: one for source triage and one for report drafting.

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
cd examples/11-research-report
agent run
```

## Try It

```bash
SESSION_TOKEN=$(curl -s http://127.0.0.1:8011/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"local-demo-key","user_id":"demo-user"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_token"])')
```

```bash
curl -N http://127.0.0.1:8011/api/chat \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "demo-user",
    "messages": [
      {
        "role": "user",
        "content": "Use the files in sources/ to prepare a short report on whether a small team should adopt an internal docs assistant."
      }
    ],
    "context": {"channel": "web"}
  }'
```

## What It Shows

- A complete project config loaded by `agent run`
- A source pack that works without external web access
- Multiple callable skills under one project
- Persistent skill state for remembering the last report topic
