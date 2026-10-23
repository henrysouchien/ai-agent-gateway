# 05 Skills

This example enables markdown-defined skills. Setting `skills_dir=` automatically exposes the `run_agent` tool.

## Install

```bash
pip install "ai-agent-gateway[anthropic]" uvicorn
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export USER_DATA_DIR="$PWD/.agent-data"
mkdir -p "$USER_DATA_DIR/gateway"
chmod 700 "$USER_DATA_DIR" "$USER_DATA_DIR/gateway"
```

## Included Skills

- `researcher.md`
- `summarizer.md`

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
        "content": "Use the researcher skill to collect three ideas for launching an API docs site, then use the summarizer skill to turn them into an executive summary."
      }
    ],
    "context": {"channel": "web"}
  }'
```

## What It Shows

- Skills are markdown files whose YAML frontmatter declares `name` and the
  required `execution_class` (the work-kind capability row, e.g.
  `node.implement`).
- `run_agent` spawns focused sub-agents. A delegated child meets the one soft
  turn ladder (`agent_gateway/runner_soft_budget.py`, thresholds in the operator settings file read through `agent_gateway/run_settings.py`); a
  named skill declares no turn or cost limit.
- Provider and model
  selection remain owned by the capability resolver.
