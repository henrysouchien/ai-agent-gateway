from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

from .project import (
  DEFAULT_AGENT_CONFIG,
  PROJECT_CONFIG_ENV,
  AgentProjectError,
  load_agent_project_config,
  load_agent_project_payload,
  save_agent_project_payload,
  write_agent_project,
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="agent",
    description="Scaffold and run ai-agent-gateway projects.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  init_parser = subparsers.add_parser("init", help="Create a new agent project.")
  init_parser.add_argument("name", help="Project name and default target directory.")
  init_parser.add_argument(
    "--dir",
    dest="target_dir",
    default=None,
    help="Target directory. Defaults to the project name.",
  )
  init_parser.add_argument(
    "--provider",
    default="anthropic",
    choices=["anthropic", "openai", "codex"],
    help="Default model provider.",
  )
  init_parser.add_argument("--model", default=None, help="Optional default model.")
  init_parser.add_argument("--force", action="store_true", help="Overwrite generated files.")

  run_parser = subparsers.add_parser("run", help="Run the current agent project.")
  run_parser.add_argument(
    "--config",
    default=DEFAULT_AGENT_CONFIG,
    help=f"Path to agent config. Defaults to {DEFAULT_AGENT_CONFIG}.",
  )
  run_parser.add_argument("--host", default=None, help="Override host from agent.yaml.")
  run_parser.add_argument("--port", type=int, default=None, help="Override port from agent.yaml.")
  run_parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")

  add_parser = subparsers.add_parser("add", help="Update agent.yaml.")
  add_sub = add_parser.add_subparsers(dest="add_command", required=True)

  add_mcp = add_sub.add_parser("mcp", help="Register an MCP server.")
  add_mcp.add_argument("name")
  add_mcp.add_argument("server_command")
  add_mcp.add_argument("args", nargs=argparse.REMAINDER)
  add_mcp.add_argument(
    "--config",
    default=DEFAULT_AGENT_CONFIG,
    help=f"Path to agent config. Defaults to {DEFAULT_AGENT_CONFIG}.",
  )

  add_provider = add_sub.add_parser("provider", help="Set the default model provider.")
  add_provider.add_argument("provider", choices=["anthropic", "openai", "codex"])
  add_provider.add_argument("--model", default=None, help="Optional default model.")
  add_provider.add_argument(
    "--config",
    default=DEFAULT_AGENT_CONFIG,
    help=f"Path to agent config. Defaults to {DEFAULT_AGENT_CONFIG}.",
  )

  list_parser = subparsers.add_parser("list", help="Show the resolved project config.")
  list_parser.add_argument(
    "--config",
    default=DEFAULT_AGENT_CONFIG,
    help=f"Path to agent config. Defaults to {DEFAULT_AGENT_CONFIG}.",
  )
  return parser


def main(
  argv: list[str] | None = None,
  *,
  stdout: TextIO | None = None,
  stderr: TextIO | None = None,
  uvicorn_run: Callable[..., Any] | None = None,
) -> int:
  stdout = stdout or sys.stdout
  stderr = stderr or sys.stderr
  parser = build_parser()
  args = parser.parse_args(argv)

  try:
    if args.command == "init":
      target = Path(args.target_dir or args.name)
      written = write_agent_project(
        target,
        name=args.name,
        provider=args.provider,
        model=args.model,
        force=args.force,
      )
      stdout.write(f"Created agent project in {target}\n")
      for path in written:
        stdout.write(f"  {path.relative_to(target)}\n")
      return 0

    if args.command == "run":
      return _run_project(args, stdout=stdout, stderr=stderr, uvicorn_run=uvicorn_run)

    if args.command == "add":
      if args.add_command == "mcp":
        return _add_mcp(args, stdout=stdout)
      if args.add_command == "provider":
        return _add_provider(args, stdout=stdout)

    if args.command == "list":
      return _list_project(args, stdout=stdout)
  except AgentProjectError as exc:
    stderr.write(f"agent: {exc}\n")
    return 2

  parser.error(f"unknown command: {args.command}")
  return 2


def _run_project(
  args: argparse.Namespace,
  *,
  stdout: TextIO,
  stderr: TextIO,
  uvicorn_run: Callable[..., Any] | None,
) -> int:
  config = load_agent_project_config(args.config)
  host = args.host or config.host
  port = args.port if args.port is not None else config.port
  if port <= 0:
    raise AgentProjectError("port must be a positive integer")

  run = uvicorn_run or _load_uvicorn_run()
  previous_config = os.environ.get(PROJECT_CONFIG_ENV)
  os.environ[PROJECT_CONFIG_ENV] = str(config.path)
  stdout.write(f"Starting {config.name} on http://{host}:{port}{config.api_prefix}\n")
  stdout.flush()
  try:
    run(
      "agent_gateway.project:create_app_from_env",
      factory=True,
      host=host,
      port=port,
      reload=bool(args.reload),
      reload_dirs=[str(config.path.parent)] if args.reload else None,
    )
  finally:
    if previous_config is None:
      os.environ.pop(PROJECT_CONFIG_ENV, None)
    else:
      os.environ[PROJECT_CONFIG_ENV] = previous_config
    stderr.flush()
  return 0


def _load_uvicorn_run() -> Callable[..., Any]:
  try:
    from uvicorn import run
  except ImportError as exc:
    raise AgentProjectError(
      "uvicorn is required for `agent run`; install ai-agent-gateway with its runtime dependencies"
    ) from exc
  return run


def _add_mcp(args: argparse.Namespace, *, stdout: TextIO) -> int:
  server_args, config_path = _extract_remainder_config(list(args.args), args.config)
  payload = load_agent_project_payload(config_path)
  servers = payload.get("mcp_servers")
  if servers is None:
    servers = {}
  if not isinstance(servers, dict):
    raise AgentProjectError("mcp_servers must be a mapping")
  name = str(args.name).strip()
  if not name:
    raise AgentProjectError("MCP server name cannot be blank")
  servers[name] = {"command": args.server_command, "args": server_args}
  payload["mcp_servers"] = servers
  save_agent_project_payload(payload, config_path)
  stdout.write(f"Registered MCP server {name}\n")
  return 0


def _extract_remainder_config(args: list[str], default_config: str) -> tuple[list[str], str]:
  if args and args[0] == "--":
    args = args[1:]
  if "--config" not in args:
    return args, default_config

  index = args.index("--config")
  try:
    config_path = args[index + 1]
  except IndexError as exc:
    raise AgentProjectError("--config requires a path") from exc
  if not config_path.strip():
    raise AgentProjectError("--config requires a path")
  stripped_args = args[:index] + args[index + 2 :]
  return stripped_args, config_path


def _add_provider(args: argparse.Namespace, *, stdout: TextIO) -> int:
  payload = load_agent_project_payload(args.config)
  payload["provider"] = args.provider
  payload["model"] = args.model
  save_agent_project_payload(payload, args.config)
  stdout.write(f"Set provider to {args.provider}\n")
  return 0


def _list_project(args: argparse.Namespace, *, stdout: TextIO) -> int:
  config = load_agent_project_config(args.config)
  stdout.write(f"name: {config.name}\n")
  stdout.write(f"provider: {config.provider}\n")
  stdout.write(f"model: {config.model or ''}\n")
  stdout.write(f"address: http://{config.host}:{config.port}{config.api_prefix}\n")
  stdout.write(f"skills_dir: {config.skills_dir or ''}\n")
  stdout.write(f"mcp_servers: {', '.join(sorted(config.mcp_servers))}\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
