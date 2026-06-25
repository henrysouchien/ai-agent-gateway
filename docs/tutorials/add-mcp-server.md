# Add a Custom MCP Server

This tutorial starts from a generated project and registers a filesystem MCP server.

## 1. Create A Project

```bash
agent init mcp-demo
cd mcp-demo
```

## 2. Register The Server

```bash
agent add mcp filesystem npx -y @modelcontextprotocol/server-filesystem .
```

This updates `agent.yaml`:

```yaml
mcp_servers:
  filesystem:
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-filesystem"
      - .
```

## 3. Add A Source File

```bash
printf 'The release checklist has docs, tests, and package smoke.\n' > notes.txt
```

## 4. Run The Agent

```bash
agent run
```

Ask the agent to read `notes.txt` and summarize it. The gateway starts the MCP server, merges its tool definitions into the agent turn, and routes the tool call back to the server.

## Production Notes

- Keep MCP server commands deterministic and explicit in `agent.yaml`.
- Prefer project-relative filesystem scopes over broad home-directory scopes.
- Use separate MCP servers when a tool runtime has its own dependencies or process lifecycle.
