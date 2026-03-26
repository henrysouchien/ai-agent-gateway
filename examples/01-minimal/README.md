# 01 Minimal

The smallest possible `ai-agent-gateway` server: one system prompt, no tools.

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
      {"role": "user", "content": "Explain what server-sent events are in two sentences."}
    ],
    "context": {"channel": "web"}
  }'
```

## What It Shows

- `create_agent()` turns a system prompt into a FastAPI app.
- The app exposes session init plus SSE chat endpoints automatically.
- No tool wiring is required to get a streaming agent backend.
