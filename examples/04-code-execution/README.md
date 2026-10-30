# 04 Code Execution

This example enables the built-in `code_execute` tool.

## Prerequisites

- Anthropic API access
- `python3`
- Docker only when explicitly configuring container placement

This example uses the default local persistent Python kernel. Set
`CodeExecutionConfig(placement="docker", ...)` to place that kernel in Docker;
an unavailable configured backend returns an error rather than selecting another.

## Install

```bash
pip install "ai-agent-gateway[anthropic]" uvicorn
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export USER_DATA_DIR="$PWD/.agent-data"
mkdir -p "$USER_DATA_DIR/gateway"
chmod 700 "$USER_DATA_DIR" "$USER_DATA_DIR/gateway"
```


## Run

```bash
uvicorn agent:app --reload --port 8000
```

## Chat

```bash
SESSION_TOKEN=$(curl -s http://127.0.0.1:8000/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"demo-key","user_id":"demo-user"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_token"])')
```

```bash
curl -N http://127.0.0.1:8000/api/chat \
  -H "Authorization: Bearer $SESSION_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "demo-user",
    "messages": [
      {
        "role": "user",
        "content": "Use code execution to calculate the first 10 Fibonacci numbers and print them as a Python list."
      }
    ],
    "context": {"channel": "web"}
  }'
```

## What It Shows

- `code_execution=True` adds built-in code execution tools.
- Foreground runs stream `tool_output_chunk` events as stdout and stderr arrive.
- Background runs return a `task_id`; the cell runs as a background job on the turn's task registry and its result is delivered like any background task result.
- Local and Docker placements share the persistent kernel; placement is explicit, not an availability fallback.
