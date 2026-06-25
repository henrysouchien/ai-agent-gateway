# Build a Daily Briefing Agent

This tutorial uses the complete project in `examples/10-daily-briefing/`.

## 1. Open The Example

```bash
cd packages/agent-gateway/examples/10-daily-briefing
```

The project includes:

- `agent.yaml` for provider, port, MCP, skills, and output settings
- `feeds/` sample source files
- `skills/daily-briefing.md` as a callable skill with persistent state enabled

## 2. Run The Server

```bash
agent run
```

The example listens on `http://127.0.0.1:8010/api`.

## 3. Ask For A Briefing

Ask the agent:

```text
Read feeds/product-updates.md and feeds/ops-notes.md, then use the daily-briefing skill to produce today's briefing.
```

The parent agent reads local source files through the filesystem MCP server, then delegates the synthesis step to the callable briefing skill.

## 4. Adapt It

- Replace `feeds/` with generated RSS exports, webhook payloads, or checked-in status notes.
- Swap the filesystem MCP server for an RSS or notification MCP server once packaged.
- Keep the briefing skill focused on synthesis so source access remains owned by the parent project configuration.
