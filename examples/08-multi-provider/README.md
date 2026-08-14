# 08 Multi Provider

This example serves the same gateway shape through `provider="openai"` on `create_agent()`.

It shows the first-party, Responses-only OpenAI setup:
`create_agent(..., provider="openai", model="gpt-5.6")`.

## Install

```bash
pip install "ai-agent-gateway[openai]" uvicorn
export OPENAI_API_KEY="your-openai-api-key"
```

`OpenAIProvider` uses `POST /v1/responses` at the official OpenAI API. Other
vendors should use a first-class provider with its own protocol and credentials.

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
      {"role": "user", "content": "Explain in two sentences why provider abstractions matter."}
    ],
    "context": {"channel": "web"},
    "model": "gpt-5.6"
  }'
```

## What It Shows

- The HTTP surface does not change when you switch providers.
- `create_agent()` can switch to OpenAI without dropping to `create_gateway_app()`.
- OpenAI requests use the Responses API without a Chat Completions fallback.
