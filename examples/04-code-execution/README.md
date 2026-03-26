# 04 Code Execution

This example enables the built-in `code_execute` and `code_execute_status` tools.

## Prerequisites

- Anthropic API access
- `python3`
- Docker if you want the preferred sandboxed backend

The gateway prefers Docker when a usable Docker image is available and falls back to subprocess execution otherwise.

## Install

```bash
pip install "ai-agent-gateway[anthropic]" uvicorn
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Optional: override the Docker image the backend checks for.

```bash
export CODE_EXECUTE_DOCKER_IMAGE="ai-excel-addin-code-exec:latest"
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
        "content": "Use code execution to calculate the first 10 Fibonacci numbers and print them as a Python list."
      }
    ],
    "context": {"channel": "web"}
  }'
```

## What It Shows

- `code_execution=True` adds built-in code execution tools.
- Foreground runs stream `tool_output_chunk` events as stdout and stderr arrive.
- Background runs return a `task_id` that can be polled with `code_execute_status`.
- Docker is preferred for sandboxing; subprocess is the fallback for development environments.
