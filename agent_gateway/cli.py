"""Command-line composition root for package projects and provider login.

``main`` parses the installed ``agent`` command and delegates project
scaffolding/configuration to ``project.py``, serving to Uvicorn, and provider
credentials to their token-store owners. It emits an integer process status and
does not own the gateway run loop or product policy. See
``packages/agent-gateway/README.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import subprocess
import sys
import time
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
  _USER_SCOPE_HELP = (
    "risk_user_id whose credential pool this account belongs to; omit for the "
    "deployment's org pool."
  )
  from .login_flows import registered_login_flows
  _flow_choices = sorted(registered_login_flows().keys())
  auth_login = auth_sub.add_parser("login", help="Sign in to a provider.")
  auth_login.add_argument("provider", choices=_flow_choices)
  auth_login.add_argument(
    "--api-key",
    action="store_true",
    help="Log in with a direct API key instead of subscription OAuth/device code.",
  )
  auth_login.add_argument("--no-browser", action="store_true", help="Do not open the verification URL.")
  auth_login.add_argument("--store", default=None, help="Override the OAuth token-store path.")
  auth_login.add_argument("--user", default=None, help=_USER_SCOPE_HELP)
  auth_login.add_argument("--profile", default=None, help="CAAM profile name for a new Codex enrollment.")
  auth_login.add_argument("--email", default=None, help="Expected ChatGPT email for a new Codex enrollment.")
  auth_enroll = auth_sub.add_parser(
    "enroll",
    help="Enroll an existing provider credential in a credential pool.",
  )
  auth_enroll.add_argument("provider", choices=["anthropic"])
  auth_enroll.add_argument(
    "--setup-token",
    action="store_true",
    required=True,
    help="Read a `claude setup-token` value from stdin (never echoed, never an argument).",
  )
  auth_enroll.add_argument("--store", default=None, help="Override the OAuth token-store path.")
  auth_enroll.add_argument("--user", default=None, help=_USER_SCOPE_HELP)
  auth_status = auth_sub.add_parser("status", help="Show provider authentication status.")
  auth_status.add_argument("provider", choices=_flow_choices)
  auth_status.add_argument(
    "--api-key",
    action="store_true",
    help="Inspect the API-key store instead of subscription OAuth/device code.",
  )
  auth_status.add_argument("--store", default=None, help="Override the OAuth token-store path.")
  auth_status.add_argument("--user", default=None, help=_USER_SCOPE_HELP)
  auth_logout = auth_sub.add_parser("logout", help="Remove persisted provider authentication.")
  auth_logout.add_argument("provider", choices=_flow_choices)
  auth_logout.add_argument(
    "--account",
    default=None,
    help="Account identity to remove (default: remove all accounts in target store).",
  )
  auth_logout.add_argument(
    "--api-key",
    action="store_true",
    help="Remove from the API-key store instead of subscription OAuth/device code.",
  )
  auth_logout.add_argument("--store", default=None, help="Override the OAuth token-store path.")
  auth_logout.add_argument("--user", default=None, help=_USER_SCOPE_HELP)
  auth_list = auth_sub.add_parser("list", help="List enrolled provider accounts.")
  auth_list.add_argument("provider", choices=_flow_choices)
  auth_list.add_argument(
    "--api-key",
    action="store_true",
    help="List accounts in the API-key store instead of subscription OAuth/device code.",
  )
  auth_list.add_argument("--store", default=None, help="Override the OAuth token-store path.")
  auth_list.add_argument("--user", default=None, help=_USER_SCOPE_HELP)
  auth_list.add_argument(
    "--usage",
    action="store_true",
    help=(
      "Anthropic only: also show each account's policy, cached quota windows and "
      "cooldowns. Reads the cached selection state only; never contacts Anthropic."
    ),
  )
  auth_remove = auth_sub.add_parser("remove", help="Remove one enrolled provider account.")
  auth_remove.add_argument("provider", choices=_flow_choices)
  auth_remove.add_argument("identity", help="Account identity reported by `auth list`.")
  auth_remove.add_argument(
    "--api-key",
    action="store_true",
    help="Remove from the API-key store instead of subscription OAuth/device code.",
  )
  auth_remove.add_argument("--store", default=None, help="Override the OAuth token-store path.")
  auth_remove.add_argument("--user", default=None, help=_USER_SCOPE_HELP)
  auth_policy = auth_sub.add_parser(
    "policy",
    help="Set or reset one enrolled Anthropic account's selection policy.",
  )
  auth_policy.add_argument("provider", choices=["anthropic"])
  auth_policy.add_argument(
    "--identity",
    required=True,
    help="Account identity reported by `auth list`.",
  )
  auth_policy.add_argument(
    "--priority",
    type=int,
    default=None,
    help="Integer priority; a higher value is preferred. Requires --reserve-pct.",
  )
  auth_policy.add_argument(
    "--reserve-pct",
    type=float,
    default=None,
    help=(
      "Soft reserve, 0-100: percent of quota kept back for sibling accounts that "
      "have headroom; 0 disables it. Requires --priority."
    ),
  )
  auth_policy.add_argument(
    "--reset",
    action="store_true",
    help="Remove the account's configured policy so it uses the defaults.",
  )
  auth_policy.add_argument("--store", default=None, help="Override the OAuth token-store path.")
  auth_policy.add_argument("--user", default=None, help=_USER_SCOPE_HELP)
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
  """Dispatch one ``agent`` CLI invocation and return its process status.

  Project validation failures are rendered to ``stderr`` with status 2.
  Runtime startup and provider-specific authentication effects are delegated to
  their owning helpers.
  """
  stdout_stream: TextIO = sys.stdout if stdout is None else stdout
  stderr_stream: TextIO = sys.stderr if stderr is None else stderr
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
      stdout_stream.write(f"Created agent project in {target}\n")
      for path in written:
        stdout_stream.write(f"  {path.relative_to(target)}\n")
      return 0

    if args.command == "run":
      return _run_project(
        args,
        stdout=stdout_stream,
        stderr=stderr_stream,
        uvicorn_run=uvicorn_run,
      )

    if args.command == "add":
      if args.add_command == "mcp":
        return _add_mcp(args, stdout=stdout_stream)
      if args.add_command == "model":
        return _add_model(args, stdout=stdout_stream)

    if args.command == "auth":
      if args.auth_command == "login":
        return _auth_login(args, stdout=stdout_stream)
      if args.auth_command == "status":
        return _auth_status(args, stdout=stdout_stream)
      if args.auth_command == "logout":
        return _auth_logout(args, stdout=stdout_stream)
      if args.auth_command == "list":
        return _auth_list(args, stdout=stdout_stream)
      if args.auth_command == "enroll":
        return _auth_enroll(args, stdout=stdout_stream)
      if args.auth_command == "remove":
        return _auth_remove(args, stdout=stdout_stream)
      if args.auth_command == "policy":
        return _auth_policy(args, stdout=stdout_stream)

    if args.command == "list":
      return _list_project(args, stdout=stdout_stream)
  except AgentProjectError as exc:
    stderr_stream.write(f"agent: {exc}\n")
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


def _anthropic_store_path(args: argparse.Namespace) -> Path:
  """The store these `auth` commands read and write: one user's, or the org's.

  `--user <risk_user_id>` names the pool that serves that user's turns and the
  autonomous runs launched from them; without it the command operates on the
  deployment's org pool, which every user's pool ends with.
  """

  from .providers.anthropic_oauth import resolve_anthropic_auth_store_path

  return resolve_anthropic_auth_store_path(
    _xai_auth_config(args),
    user_scope=getattr(args, "user", None),
  )


def _auth_login(args: argparse.Namespace, *, stdout: TextIO) -> int:
  from .login_flows import get_login_flow
  try:
    flow = get_login_flow(args.provider)
    return flow.login(args, stdout=stdout)
  except AgentProjectError:
    raise
  except Exception as exc:
    raise AgentProjectError(str(exc)) from exc

def _auth_login_anthropic(args: argparse.Namespace, *, stdout: TextIO) -> int:
  """Enroll one Claude account in this gateway's own credential pool.

  This is the product's own OAuth client: the grant it mints here (access plus
  a refresh token it owns) is the only Anthropic grant the gateway ever
  refreshes. Run it once per account; each run adds a sibling and leaves the
  ones already enrolled alone.
  """

  from .providers.anthropic_oauth import (
    AnthropicOAuthError,
    complete_anthropic_login,
    mask_anthropic_token,
    start_anthropic_login,
  )

  path = _anthropic_store_path(args)
  request = start_anthropic_login()
  stdout.write("\nAnthropic OAuth login — one Claude account per run\n")
  stdout.write(f"Open: {request.authorize_url}\n")
  stdout.write(
    "Sign in as the account you want to enroll, approve access, then copy the\n"
    "value the callback page shows (it looks like `<code>#<state>`).\n"
  )
  stdout.flush()
  if not args.no_browser:
    webbrowser.open(request.authorize_url)
  pasted = getpass.getpass("Paste the callback URL or code#state: ").strip()
  try:
    record = complete_anthropic_login(request, pasted, store_path=path)
  except (AnthropicOAuthError, OSError) as exc:
    raise AgentProjectError(str(exc)) from exc
  stdout.write(
    f"Enrolled {record.identity} ({mask_anthropic_token(record.access_token)}) in the "
    f"gateway credential pool: {path}\n"
  )
  stdout.write("Existing Claude Code and gateway sessions were not modified.\n")
  return 0


def _auth_enroll(args: argparse.Namespace, *, stdout: TextIO) -> int:
  """Enroll a `claude setup-token` credential in a credential pool.

  The token is read from stdin — never an argument, never echoed — because an
  account created with `claude setup-token` has no refresh grant for this
  product's OAuth client to mint and is therefore the one credential form the
  login flow cannot produce. It joins the pool as an ordinary sibling: sent
  as-is, parked by its account key when a limiter rejects it.
  """

  from .providers.anthropic_oauth import (
    AnthropicOAuthError,
    enroll_anthropic_setup_token,
    mask_anthropic_token,
  )

  path = _anthropic_store_path(args)
  if sys.stdin.isatty():
    token = getpass.getpass("Paste the Anthropic setup token: ")
  else:
    token = sys.stdin.read()
  try:
    record = enroll_anthropic_setup_token(token, store_path=path)
  except (AnthropicOAuthError, OSError) as exc:
    raise AgentProjectError(str(exc)) from exc
  stdout.write(
    f"Enrolled {record.identity} ({mask_anthropic_token(record.access_token)}) in the "
    f"credential pool: {path}\n"
  )
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
  from .login_flows import get_login_flow
  try:
    flow = get_login_flow(args.provider)
    return flow.status(args, stdout=stdout)
  except AgentProjectError:
    raise
  except Exception as exc:
    raise AgentProjectError(str(exc)) from exc
def _auth_status_anthropic(args: argparse.Namespace, *, stdout: TextIO) -> int:
  from .providers.anthropic_oauth import (
    anthropic_token_store_is_private,
    load_anthropic_oauth_store,
  )

  path = _anthropic_store_path(args)
  records = load_anthropic_oauth_store(path)
  if not records:
    stdout.write(f"Anthropic OAuth: no enrolled accounts ({path})\n")
    return 1
  permissions = "0600" if anthropic_token_store_is_private(path) else "insecure permissions"
  # Local-only: this reads the store and its expiries and never contacts
  # Anthropic, so a grant the server has revoked still looks enrolled here.
  stdout.write(
    f"Anthropic OAuth: {len(records)} enrolled account(s); store={path}; "
    f"permissions={permissions}; not server-validated\n"
  )
  return 0


def _resolve_pool_store_target(
  args: argparse.Namespace,
) -> tuple[Path, str, tuple[str, ...]]:
  from .providers.api_key_store import resolve_api_key_store_path
  from .providers.copilot_oauth import resolve_copilot_oauth_settings
  from .providers.credential_pool import USER_SCOPE_FIELD, resolve_provider_store_path
  from .providers.google_antigravity_oauth import resolve_antigravity_store_path
  from .providers.google_oauth import resolve_google_oauth_settings
  from .providers.kimi_code_oauth import resolve_kimi_code_store_path
  from .providers.muse_oauth import resolve_muse_store_path
  from .providers.xai_oauth import resolve_xai_oauth_settings

  provider = str(getattr(args, "provider", "") or "").strip().lower()
  store = str(getattr(args, "store", None) or "").strip()
  api_key_mode = bool(getattr(args, "api_key", False))
  user_scope = getattr(args, "user", None)

  if provider == "kimi-code":
    if api_key_mode:
      cfg = {"api_key_store_path": store} if store else {}
      return resolve_api_key_store_path("kimi_code", cfg, user_scope=user_scope), "kimi_code", ("api_key",)
    cfg = {"auth_store_path": store} if store else {}
    return resolve_kimi_code_store_path(cfg, user_scope=user_scope), "kimi_code", ("access_token", "refresh_token")
  if provider == "gemini":
    if api_key_mode:
      cfg = {"api_key_store_path": store} if store else {}
      return resolve_api_key_store_path("google", cfg, user_scope=user_scope), "google", ("api_key",)
    cfg = {"auth_store_path": store} if store else {}
    return resolve_antigravity_store_path(cfg, user_scope=user_scope), "antigravity", ("access_token", "refresh_token")
  if provider == "google":
    if api_key_mode:
      cfg = {"api_key_store_path": store} if store else {}
      return resolve_api_key_store_path("google", cfg, user_scope=user_scope), "google", ("api_key",)
    cfg = {"auth_store_path": store} if store else {}
    return resolve_google_oauth_settings(cfg, user_scope=user_scope).store_path, "google", ("access_token", "refresh_token")
  if provider == "copilot":
    cfg = {"auth_store_path": store} if store else {}
    return resolve_copilot_oauth_settings(cfg, user_scope=user_scope).store_path, "copilot", ("access_token", "refresh_token")
  if provider == "muse":
    if api_key_mode:
      cfg = {"api_key_store_path": store} if store else {}
      return resolve_api_key_store_path("muse", cfg, user_scope=user_scope), "muse", ("api_key",)
    cfg = {"auth_store_path": store} if store else {}
    return resolve_muse_store_path(cfg, user_scope=user_scope), "muse", ("api_key",)
  if provider == "ollama-cloud":
    cfg = {"api_key_store_path": store} if store else {}
    return resolve_api_key_store_path("ollama_cloud", cfg, user_scope=user_scope), "ollama_cloud", ("api_key",)
  if provider == "xai":
    cfg_xai: dict[str, Any] = {"auth_store_path": store} if store else {}
    if user_scope is not None:
      cfg_xai[USER_SCOPE_FIELD] = user_scope
    return resolve_xai_oauth_settings(cfg_xai).store_path, "xai", ("access_token", "refresh_token")
  cfg_generic = {"auth_store_path": store} if store else {}
  return (
    resolve_provider_store_path(provider, "oauth.json", cfg_generic, user_scope=user_scope),
    provider,
    ("access_token", "refresh_token", "api_key"),
  )


def _format_duration(seconds: float) -> str:
  total = max(0, int(round(seconds)))
  days, rest = divmod(total, 86400)
  hours, rest = divmod(rest, 3600)
  minutes, secs = divmod(rest, 60)
  if days:
    return f"{days}d {hours}h"
  if hours:
    return f"{hours}h {minutes:02d}m"
  if minutes:
    return f"{minutes}m {secs:02d}s"
  return f"{secs}s"


def _write_anthropic_usage(
  stdout: TextIO,
  usage: dict[str, Any],
) -> None:
  """Render one `describe_account_usage` view; cached data only."""

  reserve = float(usage["reserve_pct"])
  quota = str(usage["usage"])
  if quota == "unavailable" and usage.get("unavailable_reason"):
    quota += f" ({usage['unavailable_reason']})"
  stdout.write(
    f"  account={usage['account']}  priority={usage['priority']}  "
    f"reserve={reserve:g}%  policy={usage['policy_source']}  quota={quota}\n"
  )
  for window in usage["windows"]:
    used = window["used_pct"]
    used_text = "usage unknown" if used is None else f"{used:.0f}% used"
    resets = window["resets_in_s"]
    reset_text = "reset unknown" if resets is None else f"resets in {_format_duration(resets)}"
    stale = "" if window["fresh"] else " (stale)"
    stdout.write(
      f"    {window['id']} [{window['scope']}]: {used_text}, {reset_text}, "
      f"observed {_format_duration(window['age_s'])} ago via {window['source']}{stale}\n"
    )
  for cooldown in usage["cooldowns"]:
    stdout.write(
      f"    cooldown [{cooldown['scope']}]: {cooldown['reason']}, "
      f"{_format_duration(cooldown['remaining_s'])} remaining\n"
    )


def _auth_list(args: argparse.Namespace, *, stdout: TextIO) -> int:
  if args.usage and args.provider != "anthropic":
    raise AgentProjectError("--usage is only available for the anthropic provider")
  if args.provider == "anthropic":
    from .providers.anthropic_oauth import (
      load_anthropic_oauth_store,
      mask_anthropic_token,
    )

    path = _anthropic_store_path(args)
    records = load_anthropic_oauth_store(path)
    if not records:
      stdout.write(f"Anthropic OAuth: no enrolled accounts ({path})\n")
      return 1
    now = time.time()
    state = None
    from .providers.anthropic_quota import (
      describe_account_usage,
      load_selection_state,
      locator_for_record,
    )

    if args.usage:
      state = load_selection_state(path, now=now)
      if state.problems:
        stdout.write(f"Selection state problems: {', '.join(state.problems)}\n")
    for record in records:
      remaining = record.expires_at - now
      if not record.refresh_token:
        expiry = "no refresh token (enroll again to make it refreshable)"
      elif remaining <= 0:
        expiry = "access token expired; refreshes before next use"
      else:
        expiry = f"access token valid for {remaining / 60:.0f} min"
      line = f"{record.identity}  {mask_anthropic_token(record.access_token)}  {expiry}"
      if record.last_error:
        line += f"  last error: {record.last_error}"
      stdout.write(f"{line}\n")
      if state is not None:
        _write_anthropic_usage(
          stdout,
          describe_account_usage(state, locator_for_record(path, record.identity), now=now),
        )
    return 0

  from .providers.credential_pool import load_pool_accounts

  path, prefix, req_keys = _resolve_pool_store_target(args)
  accounts = load_pool_accounts(path, default_prefix=prefix, required_any_keys=req_keys)
  if not accounts:
    stdout.write(f"{args.provider}: no enrolled accounts ({path})\n")
    return 1
  for acc in accounts:
    identity = str(acc.get("identity") or "")
    details: list[str] = [identity]
    if acc.get("project_id"):
      details.append(f"project={acc['project_id']}")
    if acc.get("source"):
      details.append(f"source={acc['source']}")
    stdout.write("  ".join(details) + "\n")
  return 0


def _auth_policy(args: argparse.Namespace, *, stdout: TextIO) -> int:
  """Write or clear one enrolled account's selection policy in its store's sidecar."""

  from .providers.anthropic_oauth import load_anthropic_oauth_store
  from .providers.anthropic_quota import (
    locator_for_record,
    reset_account_policy,
    selection_sidecar_path,
    set_account_policy,
  )

  identity = str(args.identity or "").strip()
  if not identity:
    raise AgentProjectError("--identity must name an enrolled account")
  if args.reset:
    if args.priority is not None or args.reserve_pct is not None:
      raise AgentProjectError("--reset cannot be combined with --priority or --reserve-pct")
  elif args.priority is None or args.reserve_pct is None:
    raise AgentProjectError("set a policy with both --priority and --reserve-pct, or pass --reset")

  path = _anthropic_store_path(args)
  if identity not in {record.identity for record in load_anthropic_oauth_store(path)}:
    raise AgentProjectError(f"no enrolled Anthropic account {identity!r} in {path}")
  locator = locator_for_record(path, identity)
  sidecar = selection_sidecar_path(path)
  try:
    if args.reset:
      reset_account_policy(path, locator)
      stdout.write(f"Reset policy for {identity} ({locator.id}) in {sidecar}\n")
      return 0
    policy = set_account_policy(
      path, locator, priority=args.priority, reserve_pct=args.reserve_pct
    )
  except ValueError as exc:
    raise AgentProjectError(f"invalid policy: {exc}") from exc
  except OSError as exc:
    raise AgentProjectError(f"could not write policy to {sidecar}: {exc}") from exc
  stdout.write(
    f"Set policy for {identity} ({locator.id}): priority={policy.priority} "
    f"reserve_pct={policy.reserve_pct:g} in {sidecar}\n"
  )
  return 0


def _auth_remove(args: argparse.Namespace, *, stdout: TextIO) -> int:
  identity = str(args.identity or "").strip()
  if args.provider == "anthropic":
    from .providers.anthropic_oauth import remove_anthropic_oauth_record

    path = _anthropic_store_path(args)
    if not remove_anthropic_oauth_record(path, identity):
      stdout.write(f"Anthropic OAuth: no enrolled account {identity!r} in {path}\n")
      return 1
    stdout.write(f"Removed Anthropic OAuth account {identity} from {path}\n")
    stdout.write("Claude Code's saved login was not modified.\n")
    return 0

  from .providers.credential_pool import remove_pool_account

  path, prefix, req_keys = _resolve_pool_store_target(args)
  if not remove_pool_account(path, identity, default_prefix=prefix, required_any_keys=req_keys):
    stdout.write(f"{args.provider}: no enrolled account {identity!r} in {path}\n")
    return 1
  stdout.write(f"Removed {args.provider} account {identity} from {path}\n")
  return 0

def _auth_logout(args: argparse.Namespace, *, stdout: TextIO) -> int:
  from .login_flows import get_login_flow
  try:
    flow = get_login_flow(args.provider)
    return flow.logout(args, stdout=stdout)
  except AgentProjectError:
    raise
  except Exception as exc:
    raise AgentProjectError(str(exc)) from exc

if __name__ == "__main__":
  raise SystemExit(main())
