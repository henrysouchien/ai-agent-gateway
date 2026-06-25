# MCP Server Catalog

The public mono-repo sync is designed around standalone gateway core plus plugin-style MCP servers.

| Distribution | Source | Public Target | Status |
|---|---|---|---|
| `services-mcp` | `mcp_servers/services_mcp/` | `plugins/services-mcp/` | Package scaffold exists; PyPI publish remains |
| `ai-agent-scheduler-mcp` | `mcp_servers/scheduler_mcp/` | `plugins/scheduler-mcp/` | Package scaffold exists; PyPI publish remains |
| `financial-model-engine` | `packages/model-engine-package/`, `schema/`, `packages/financial-modeling-tools-compat/`, plus `mcp_servers/model_engine*` | `plugins/model-engine/` | Package scaffold exists; PyPI publish remains |

Some plugin directory names intentionally differ from the PyPI distribution name
when the shorter name is already taken or the public path is kept stable.

Financial-domain MCP servers such as FMP, IBKR, portfolio, EDGAR, and SheetsFinance are packaged or tracked separately because they carry domain-specific dependencies and release paths.

## Adding A Server

1. Keep the server source independent of the private gateway application.
2. Add `pyproject.toml` with a console script.
3. Include a README with install, run, MCP config, and tool reference sections.
4. Add package smoke tests that prove the console entry point and core tool contracts.
5. Add the source-to-public mapping to `scripts/sync_public_repo.sh` when the package belongs in the public mono-repo.
