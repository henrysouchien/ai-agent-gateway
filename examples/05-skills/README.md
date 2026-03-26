# 05 Skills

This example enables markdown-defined skills. Setting `skills_dir=` automatically exposes the `run_agent` tool.

## Install

```bash
pip install "ai-agent-gateway[anthropic]" uvicorn
export ANTHROPIC_API_KEY="your-anthropic-api-key"
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
        "content": "Use the researcher skill to collect three ideas for launching an API docs site, then use the summarizer skill to turn them into an executive summary."
      }
    ],
    "context": {"channel": "web"}
  }'
```

## What It Shows

- Skills are plain markdown files with optional YAML frontmatter.
- `run_agent` spawns focused sub-agents with their own turn budgets.
- Named skills can override model, timeout, and max-turn defaults.
