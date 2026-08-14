# 06 Tool Approval

This example uses `create_gateway_app()` with a custom `needs_approval` rule. Every `write_note` tool call requires approval.

`create_agent()` only installs built-in approval behavior for unsandboxed `code_execute` calls. Use `create_gateway_app()` when you want approval for arbitrary tools.

## Install

```bash
pip install "ai-agent-gateway[anthropic]" uvicorn
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Run

```bash
uvicorn agent:app --reload --port 8000
```

## Initialize A Session

```bash
SESSION_TOKEN=$(curl -s http://127.0.0.1:8000/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"demo-key","user_id":"demo-user"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_token"])')
export SESSION_TOKEN
```

## Stream And Approve From A Client

This client watches the SSE stream, waits for `tool_approval_request`, and then posts an approval.

```bash
python3 - <<'PY'
import json
import os
import urllib.request

session_token = os.environ["SESSION_TOKEN"]
chat_request = urllib.request.Request(
  "http://127.0.0.1:8000/api/chat",
  data=json.dumps(
    {
      "messages": [
        {
          "role": "user",
          "content": "Write a note named release-plan.txt that says ship README, examples, and API docs."
        }
      ],
      "context": {"channel": "web"},
    }
  ).encode("utf-8"),
  headers={
    "Authorization": f"Bearer {session_token}",
    "Content-Type": "application/json",
  },
)

with urllib.request.urlopen(chat_request) as response:
  for raw_line in response:
    line = raw_line.decode("utf-8").strip()
    if not line.startswith("data: "):
      continue

    event = json.loads(line[6:])
    print(event)

    if event.get("type") == "tool_approval_request":
      approval_request = urllib.request.Request(
        "http://127.0.0.1:8000/api/chat/tool-approval",
        data=json.dumps(
          {
            "tool_call_id": event["tool_call_id"],
            "nonce": event["nonce"],
            "approved": True,
            "allow_tool_type": False,
          }
        ).encode("utf-8"),
        headers={
          "Authorization": f"Bearer {session_token}",
          "Content-Type": "application/json",
        },
      )
      with urllib.request.urlopen(approval_request) as approval_response:
        print(json.loads(approval_response.read().decode("utf-8")))
PY
```

## Verify The Side Effect

```bash
cat approved_notes/release-plan.txt
```

## What It Shows

- `ToolDispatcher` can enforce approval on any tool, not just code execution.
- Clients approve tools by reacting to `tool_approval_request` SSE events.
- Approval state is session-scoped and flows through `/api/chat/tool-approval`.
- A server-owned `session.driver` policy binds the model, effort, provider,
  credential principal, and native transport before runtime construction.
- Service credential material is resolved from an opaque handle only after the
  capability bind selects Anthropic; the runtime consumes that exact prepared
  provider and auth configuration.
