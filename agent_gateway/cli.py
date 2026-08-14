from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import subprocess
import sys
import webbrowser

import httpx
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
    "--model-key",
    default=None,
    help="Optional stable product model key.",
  )
  init_parser.add_argument(
    "--effort",
    default=None,
    choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
    help="Optional effort for --model-key.",
  )
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

  add_model = add_sub.add_parser("model", help="Set stable model-key intent.")
  add_model.add_argument("model_key", help="Stable product model key.")
  add_model.add_argument(
    "--effort",
    default=None,
    choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
  )
  add_model.add_argument(
    "--config",
    default=DEFAULT_AGENT_CONFIG,
    help=f"Path to agent config. Defaults to {DEFAULT_AGENT_CONFIG}.",
  )

  auth_parser = subparsers.add_parser("auth", help="Manage provider authentication.")
  auth_sub = auth_parser.add_subparsers(dest="auth_command", required=True)
  auth_login = auth_sub.add_parser("login", help="Sign in to a provider.")
  auth_login.add_argument("provider", choices=["anthropic", "codex", "openai", "xai"])
  auth_login.add_argument("--no-browser", action="store_true", help="Do not open the verification URL.")
  auth_login.add_argument("--store", default=None, help="Override the OAuth token-store path.")
  auth_login.add_argument("--profile", default=None, help="CAAM profile name for a new Codex enrollment.")
  auth_login.add_argument("--email", default=None, help="Expected ChatGPT email for a new Codex enrollment.")
  auth_status = auth_sub.add_parser("status", help="Show provider authentication status.")
  auth_status.add_argument("provider", choices=["anthropic", "codex", "openai", "xai"])
  auth_status.add_argument("--store", default=None, help="Override the OAuth token-store path.")
  auth_logout = auth_sub.add_parser("logout", help="Remove persisted provider authentication.")
  auth_logout.add_argument("provider", choices=["anthropic", "xai"])
  auth_logout.add_argument("--store", default=None, help="Override the OAuth token-store path.")

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
        model_key=args.model_key,
        effort=args.effort,
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
      if args.add_command == "model":
        return _add_model(args, stdout=stdout)

    if args.command == "auth":
      if args.auth_command == "login":
        return _auth_login(args, stdout=stdout)
      if args.auth_command == "status":
        return _auth_status(args, stdout=stdout)
      if args.auth_command == "logout":
        return _auth_logout(args, stdout=stdout)

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


def _add_model(args: argparse.Namespace, *, stdout: TextIO) -> int:
  payload = load_agent_project_payload(args.config)
  model_key = str(args.model_key or "").strip()
  if not model_key:
    raise AgentProjectError("model_key cannot be blank")
  payload["model_key"] = model_key
  if args.effort is None:
    payload.pop("effort", None)
  else:
    payload["effort"] = args.effort
  save_agent_project_payload(payload, args.config)
  stdout.write(f"Set model key to {model_key}\n")
  return 0


def _list_project(args: argparse.Namespace, *, stdout: TextIO) -> int:
  config = load_agent_project_config(args.config)
  stdout.write(f"name: {config.name}\n")
  stdout.write(f"model_key: {config.model_key or ''}\n")
  stdout.write(f"effort: {config.effort or ''}\n")
  stdout.write(f"address: http://{config.host}:{config.port}{config.api_prefix}\n")
  stdout.write(f"skills_dir: {config.skills_dir or ''}\n")
  stdout.write(f"mcp_servers: {', '.join(sorted(config.mcp_servers))}\n")
  return 0


def _xai_auth_config(args: argparse.Namespace) -> dict[str, str]:
  store = str(getattr(args, "store", None) or "").strip()
  return {"auth_store_path": store} if store else {}


def _auth_login(args: argparse.Namespace, *, stdout: TextIO) -> int:
  if args.provider == "anthropic":
    return _auth_login_anthropic(args, stdout=stdout)
  if args.provider == "codex":
    return _auth_login_codex(args, stdout=stdout)
  if args.provider == "openai":
    raise AgentProjectError(
      "The OpenAI API does not provide ChatGPT subscription OAuth login. "
      "Use OPENAI_API_KEY for provider=openai, or run "
      "`python3 -m agent_gateway.cli auth login codex` "
      "for ChatGPT subscription authentication. No credentials were changed."
    )

  from .providers.xai_oauth import XAIDeviceCode, login_xai_device_code

  async def on_verification(device: XAIDeviceCode) -> None:
    url = device.verification_uri_complete or device.verification_uri
    stdout.write("\nxAI OAuth device authorization\n")
    stdout.write(f"Open: {url}\n")
    stdout.write(f"Code: {device.user_code}\n")
    stdout.write(f"Expires in approximately {max(1, round(device.expires_in / 60))} minutes.\n")
    stdout.write("Waiting for authorization...\n")
    stdout.flush()
    if not args.no_browser:
      webbrowser.open(url)

  try:
    _record, path = asyncio.run(
      login_xai_device_code(config=_xai_auth_config(args), on_verification=on_verification)
    )
  except (RuntimeError, ValueError, httpx.HTTPError) as exc:
    raise AgentProjectError(str(exc)) from exc
  stdout.write(f"xAI OAuth login complete. Token store: {path}\n")
  return 0


def _auth_login_anthropic(args: argparse.Namespace, *, stdout: TextIO) -> int:
  from .providers.anthropic_oauth import (
    import_claude_setup_token,
    resolve_anthropic_auth_store_path,
  )

  token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
  if not token:
    stdout.write(
      "Launching `claude setup-token`. This creates a separate one-year inference token "
      "and does not replace Claude Code's saved login.\n"
    )
    stdout.flush()
    try:
      result = subprocess.run(["claude", "setup-token"], check=False)
    except OSError as exc:
      raise AgentProjectError("Unable to launch `claude setup-token`; install Claude Code first") from exc
    if result.returncode != 0:
      raise AgentProjectError(f"`claude setup-token` failed with exit code {result.returncode}")
    token = getpass.getpass("Paste the setup token printed by Claude Code: ").strip()
  path = resolve_anthropic_auth_store_path(_xai_auth_config(args))
  try:
    import_claude_setup_token(token, path=path)
  except (OSError, RuntimeError, ValueError) as exc:
    raise AgentProjectError(str(exc)) from exc
  stdout.write(f"Anthropic OAuth token stored for new gateway sessions: {path}\n")
  stdout.write("Existing Claude Code and gateway sessions were not modified.\n")
  return 0


def _auth_login_codex(args: argparse.Namespace, *, stdout: TextIO) -> int:
  try:
    status = subprocess.run(
      ["codex", "login", "status"],
      check=False,
      capture_output=True,
      text=True,
    )
  except OSError as exc:
    raise AgentProjectError("Unable to launch `codex`; install Codex first") from exc
  if status.returncode == 0:
    detail = str(status.stdout or "").strip() or "already logged in"
    stdout.write(f"Codex: {detail}. Existing credentials and sessions were not modified.\n")
    return 0
  profile = str(args.profile or "").strip()
  email = str(args.email or "").strip()
  if not profile or not email:
    raise AgentProjectError(
      "No active Codex login found. Safe enrollment requires both --profile and --email, "
      "and runs through `cx enroll` so CAAM can protect and restore vaulted credentials."
    )
  stdout.write(f"No active Codex login found; starting transactional CAAM enrollment for {profile}.\n")
  stdout.flush()
  result = subprocess.run(["cx", "enroll", profile, "--email", email], check=False)
  if result.returncode != 0:
    raise AgentProjectError(f"`cx enroll` failed with exit code {result.returncode}")
  stdout.write("Codex enrollment complete; CAAM restored the managed global credential.\n")
  return 0


def _auth_status(args: argparse.Namespace, *, stdout: TextIO) -> int:
  if args.provider == "anthropic":
    return _auth_status_anthropic(args, stdout=stdout)
  if args.provider == "codex":
    try:
      status = subprocess.run(
        ["codex", "login", "status"],
        check=False,
        capture_output=True,
        text=True,
      )
    except OSError as exc:
      raise AgentProjectError("Unable to launch `codex`; install Codex first") from exc
    detail = str(status.stdout or status.stderr or "").strip()
    stdout.write(f"Codex: {detail or 'not logged in'}\n")
    return 0 if status.returncode == 0 else 1
  if args.provider == "openai":
    api_key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    access_token_present = bool(os.environ.get("OPENAI_AUTH_TOKEN", "").strip())
    if api_key_present or access_token_present:
      kind = "API key" if api_key_present else "bearer access token"
      stdout.write(f"OpenAI API: {kind} configured in the environment; no gateway OAuth store.\n")
      return 0
    stdout.write(
      "OpenAI API: no credential configured. Set OPENAI_API_KEY; for ChatGPT subscription "
      "auth use provider=codex and `python3 -m agent_gateway.cli auth login codex`.\n"
    )
    return 1

  from .providers.xai_oauth import (
    load_xai_token_record,
    resolve_xai_oauth_settings,
    token_needs_refresh,
    token_store_is_private,
    xai_record_requires_reauth,
  )

  settings = resolve_xai_oauth_settings(_xai_auth_config(args))
  record = load_xai_token_record(settings.store_path)
  if not record:
    stdout.write(f"xAI OAuth: not logged in ({settings.store_path})\n")
    return 1
  if xai_record_requires_reauth(record):
    if record.get("reauth_required"):
      stdout.write("xAI OAuth: logged out — re-authentication required\n")
    else:
      stdout.write(
        "xAI OAuth: previous refresh did not complete — re-authentication required\n"
      )
    return 1
  # This check is local-only: it reads the stored record and its expiry, and never contacts xAI.
  # A refresh token the server has REVOKED still looks fine here, so do not report it as "active"
  # — that is a claim this code cannot support, and it has misled an operator into trusting a dead
  # credential. Say exactly what was checked.
  state = "refresh required" if token_needs_refresh(record) else "stored; not server-validated"
  permissions = "0600" if token_store_is_private(settings.store_path) else "insecure permissions"
  stdout.write(f"xAI OAuth: {state}; store={settings.store_path}; permissions={permissions}\n")
  return 0


def _auth_status_anthropic(args: argparse.Namespace, *, stdout: TextIO) -> int:
  from .providers.anthropic_oauth import (
    anthropic_token_is_expiring,
    anthropic_token_store_is_private,
    load_anthropic_oauth_record,
    resolve_anthropic_auth_store_path,
  )

  path = resolve_anthropic_auth_store_path(_xai_auth_config(args))
  record = load_anthropic_oauth_record(path)
  if not record:
    stdout.write(f"Anthropic OAuth: no gateway token store ({path})\n")
    return 1
  state = "renew soon" if anthropic_token_is_expiring(record) else "active"
  permissions = "0600" if anthropic_token_store_is_private(path) else "insecure permissions"
  stdout.write(f"Anthropic OAuth: {state}; store={path}; permissions={permissions}\n")
  return 0


def _auth_logout(args: argparse.Namespace, *, stdout: TextIO) -> int:
  if args.provider == "anthropic":
    from .providers.anthropic_oauth import resolve_anthropic_auth_store_path

    path = resolve_anthropic_auth_store_path(_xai_auth_config(args))
    provider_label = "Anthropic"
  else:
    from .providers.xai_oauth import (
      _store_lock,
      _write_reauth_tombstone,
      resolve_xai_oauth_settings,
    )

    path = resolve_xai_oauth_settings(_xai_auth_config(args)).store_path
    provider_label = "xAI"

    async def logout_xai() -> None:
      async with _store_lock(path):
        _write_reauth_tombstone(path)

    try:
      asyncio.run(logout_xai())
    except OSError as exc:
      raise AgentProjectError(f"Unable to update {provider_label} OAuth token store: {path}") from exc
    stdout.write(f"Removed {provider_label} OAuth token store: {path}\n")
    return 0
  try:
    path.unlink()
  except FileNotFoundError:
    pass
  except OSError as exc:
    raise AgentProjectError(f"Unable to remove {provider_label} OAuth token store: {path}") from exc
  stdout.write(f"Removed {provider_label} OAuth token store: {path}\n")
  if args.provider == "anthropic":
    stdout.write("Claude Code's saved login was not modified.\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
