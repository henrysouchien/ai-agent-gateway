# 03 Local Tools

This example uses local Python tool handlers instead of MCP.

## Install

```bash
pip install "ai-agent-gateway[anthropic]" uvicorn
export ANTHROPIC_API_KEY="your-anthropic-api-key"
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
        "content": "Write a note named launch-plan.txt that says ship docs, examples, and docstrings. Then read it back."
      }
    ],
    "context": {"channel": "web"}
  }'
```

## What It Shows

- `tool_definitions` describe the tool schema exposed to the model.
- `tool_handlers` implement the actual Python behavior.
- Local tools can safely keep state in the example directory without running an MCP server.
