from __future__ import annotations

from typing import Any, Callable, Mapping


def apply_server_env_passthrough(
  mcp_servers: dict[str, Any],
  passthrough: Mapping[str, set[str]],
  *,
  environ: Mapping[str, str],
) -> dict[str, Any]:
  """Overlay explicit parent-env references for selected stdio servers."""
  resolved = {
    name: dict(config) if isinstance(config, dict) else config
    for name, config in mcp_servers.items()
  }
  for server_name, env_names in passthrough.items():
    config = resolved.get(server_name)
    if not isinstance(config, dict):
      continue
    configured_env = config.get("env")
    server_env = dict(configured_env) if isinstance(configured_env, dict) else {}
    for env_name in env_names:
      if environ.get(env_name):
        server_env[env_name] = f"${{{env_name}}}"
      elif server_env.get(env_name) == f"${{{env_name}}}":
        server_env.pop(env_name, None)
    config["env"] = server_env
  return resolved


def canonicalize_server_configs(
  mcp_servers: dict[str, dict[str, Any]],
  *,
  canonical_server_name: Callable[[str], str],
  logger: Any,
) -> dict[str, dict[str, Any]]:
  canonical_configs: dict[str, dict[str, Any]] = {}
  source_names: dict[str, str] = {}
  for server_name, server_config in mcp_servers.items():
    canonical_name = canonical_server_name(server_name)
    existing_source = source_names.get(canonical_name)
    if existing_source is not None:
      if server_name == canonical_name and existing_source != canonical_name:
        logger.info(
          "MCP server %s overrides alias %s for canonical server %s",
          server_name,
          existing_source,
          canonical_name,
        )
        canonical_configs[canonical_name] = server_config
        source_names[canonical_name] = server_name
      else:
        logger.warning(
          "Skipping MCP server %s: resolves to already configured canonical server %s",
          server_name,
          canonical_name,
        )
      continue

    if server_name != canonical_name:
      logger.info("Using MCP server alias %s as %s", server_name, canonical_name)
    canonical_configs[canonical_name] = server_config
    source_names[canonical_name] = server_name
  return canonical_configs


async def startup_manager(
  manager: Any,
  allowed_servers: set[str] | None = None,
  *,
  mcp_import_error: Exception | None,
  supported_server_types: set[str],
  logger: Any,
) -> None:
  manager._startup_diagnostics = {}
  requested_servers = (
    manager._canonical_server_names(set(allowed_servers))
    if allowed_servers is not None
    else None
  )

  if mcp_import_error is not None:
    logger.warning("MCP runtime unavailable; skipping startup: %s", mcp_import_error)
    for server_name in sorted(requested_servers or set()):
      manager._set_startup_diagnostic(
        server_name,
        category="runtime_unavailable",
        message=f"MCP runtime unavailable: {mcp_import_error}",
        retryable=False,
        error_type=type(mcp_import_error).__name__,
      )
    manager._started = True
    return

  config = manager._read_claude_config()
  mcp_servers = config.get("mcpServers", {})
  if not isinstance(mcp_servers, dict):
    mcp_servers = {}
  mcp_servers = dict(mcp_servers)
  mcp_servers.update(manager._inline_servers)

  effective_allowed_servers = manager._allowed_servers
  if requested_servers is not None:
    if effective_allowed_servers is None:
      effective_allowed_servers = requested_servers
    else:
      not_allowed = requested_servers - effective_allowed_servers
      for server_name in sorted(not_allowed):
        manager._set_startup_diagnostic(
          server_name,
          category="not_allowed",
          message="server requested for startup but absent from the MCP allowlist",
          retryable=False,
        )
      effective_allowed_servers = set(effective_allowed_servers) & requested_servers

  transport_allowed_servers = (
    manager._transport_server_names(effective_allowed_servers)
    if effective_allowed_servers is not None
    else None
  )

  if not mcp_servers:
    missing_config_targets = transport_allowed_servers if requested_servers is not None else set()
    for server_name in sorted(missing_config_targets or set()):
      manager._set_startup_diagnostic(
        server_name,
        category="config_missing",
        message="server requested for startup but absent from the MCP config",
        retryable=False,
      )
    manager._started = True
    return

  mcp_servers = manager._canonicalize_server_configs(mcp_servers)
  mcp_servers = apply_server_env_passthrough(
    mcp_servers,
    manager._server_env_passthrough,
    environ=manager._connection_runtime().environ,
  )

  if requested_servers is not None and transport_allowed_servers is not None:
    for server_name in sorted(transport_allowed_servers - set(mcp_servers)):
      manager._set_startup_diagnostic(
        server_name,
        category="config_missing",
        message="server requested for startup but absent from the MCP config",
        retryable=False,
      )

  connect_jobs: list[tuple[str, dict[str, Any]]] = []
  for server_name, server_config in mcp_servers.items():
    if transport_allowed_servers is not None and server_name not in transport_allowed_servers:
      continue
    if not isinstance(server_config, dict):
      logger.warning("Skipping MCP server %s: invalid config", server_name)
      manager._set_startup_diagnostic(
        server_name,
        category="invalid_config",
        message="invalid MCP server config",
        retryable=False,
      )
      continue

    server_type = str(server_config.get("type", "stdio")).strip().lower()
    if server_type not in supported_server_types:
      logger.info("Skipping MCP server %s: unsupported type %s", server_name, server_type)
      manager._set_startup_diagnostic(
        server_name,
        category="unsupported_type",
        message=f"unsupported MCP server type: {server_type}",
        retryable=False,
      )
      continue

    try:
      prepared_config = manager._prepare_server_config_for_startup(server_config)
    except ValueError as exc:
      logger.warning("Skipping MCP server %s: invalid config", server_name)
      manager._set_startup_diagnostic(
        server_name,
        category="invalid_config",
        message=str(exc),
        retryable=False,
      )
      continue

    connect_jobs.append((server_name, prepared_config))

  if connect_jobs:
    for state in await manager._connect_startup_servers(connect_jobs):
      if state is not None:
        manager._servers[state.name] = state

  manager._apply_collision_filtering()
  manager._started = True


__all__ = [
  "apply_server_env_passthrough",
  "canonicalize_server_configs",
  "startup_manager",
]
