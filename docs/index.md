# ai-agent-gateway

`ai-agent-gateway` is a production runtime for tool-using agents. It gives you the HTTP server, sessions, SSE streaming, tool dispatch, human approval, MCP orchestration, skills, memory hooks, and code execution infrastructure around your model calls.

Start with the [Quickstart](quickstart.md) if you want a running agent in a few minutes.

## Core Paths

- [Architecture](architecture.md) explains the request lifecycle and runtime boundaries.
- [API Reference](api-reference.md) covers the main Python entry points.
- [HTTP API](http-api.md) documents the session, chat, approval, and health endpoints.
- [MCP Server Catalog](mcp-server-catalog.md) lists the MCP servers intended for the public mono-repo.

## Tutorials

- [Build a Daily Briefing Agent](tutorials/daily-briefing-agent.md)
- [Add a Custom MCP Server](tutorials/add-mcp-server.md)
