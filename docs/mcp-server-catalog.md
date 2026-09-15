# MCP Server Catalog

The public distribution is designed around a standalone gateway core plus
plugin-style MCP servers.

| Distribution | Plugin directory | Purpose |
|---|---|---|
| `services-mcp` | `plugins/services-mcp/` | Service integrations exposed as MCP tools |
| `ai-agent-scheduler-mcp` | `plugins/scheduler-mcp/` | Scheduling tools for agent projects |
| `financial-model-engine` | `plugins/model-engine/` | Financial-model schemas, tools, and MCP adapters |

Some plugin directory names intentionally differ from the PyPI distribution name
when the shorter name is already taken or the public path is kept stable.

Financial-domain MCP servers such as FMP, IBKR, portfolio, EDGAR, and SheetsFinance are packaged or tracked separately because they carry domain-specific dependencies and release paths.

## Adding A Server

1. Keep the server source independent of any embedding gateway application.
2. Add `pyproject.toml` with a console script.
3. Include a README with install, run, MCP config, and tool reference sections.
4. Add package smoke tests that prove the console entry point and core tool contracts.
5. Add it to the distribution's plugin manifest and release automation.
