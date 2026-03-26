# 08 Multi Provider

This example serves the same gateway shape through `OpenAIProvider` instead of `AnthropicProvider`.

It uses `create_gateway_app()` because `create_agent()` currently supports Anthropic only.

## Install

```bash
pip install "ai-agent-gateway[openai]" uvicorn
export OPENAI_API_KEY="your-openai-api-key"
```

Optional: point the OpenAI-compatible client at another base URL.

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
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
      {"role": "user", "content": "Explain in two sentences why provider abstractions matter."}
    ],
    "context": {"channel": "web"},
    "model": "gpt-4o-mini"
  }'
```

## What It Shows

- The HTTP surface does not change when you switch providers.
- `AgentRunner` works with both `AnthropicProvider` and `OpenAIProvider`.
- `GatewayServerConfig.allowed_models` can expose a provider-specific allowlist.
