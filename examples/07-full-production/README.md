# 07 Full Production

This example shows the "graduate from `create_agent()`" setup:

- `create_gateway_app()` directly
- explicit API key allowlist
- session TTL
- CORS configuration
- canonical `session.driver` capability and model policy
- opaque service credential handles with deferred materialization
- custom tool approval for a write tool
- usage and tool timing hooks
- transcript logging

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

This example enforces `valid_api_keys={"demo-key"}`.

```bash
SESSION_TOKEN=$(curl -s http://127.0.0.1:8000/api/chat/init \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"demo-key","user_id":"demo-user"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_token"])')
export SESSION_TOKEN
```

## Stream, Approve, And Persist A Report

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
          "content": "Create a markdown launch checklist named launch-checklist.md and save it as a report."
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
            "allow_tool_type": True,
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

## Inspect The Artifacts

```bash
ls -1 reports logs transcripts
```

```bash
cat reports/launch-checklist.md
```

## What It Shows

- App-level auth and session settings live in `GatewayServerConfig`.
- The gateway resolves one exact `session.driver` bind before runtime
  construction; the runtime reuses its provider, model, effort, bound auth, and
  native execution transport.
- Observability hooks can write JSONL logs without changing the runner core.
- Tool approval can be scoped to side-effecting tools such as `write_report`.
- Transcript files are written automatically when `transcript_dir` is configured.
