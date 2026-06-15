from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .anthropic import AnthropicProvider
from .base import CostEstimate


SDK_PINNED_VERSION = "0.1.50"

# Fail-closed inventory of reviewed SDK built-ins for the pinned version.
SDK_KNOWN_BUILTINS = {
  "Agent",
  "AskUserQuestion",
  "Bash",
  "BashOutput",
  "Edit",
  "EnterPlanMode",
  "ExitPlanMode",
  "Glob",
  "Grep",
  "KillBash",
  "ListMcpResources",
  "NotebookEdit",
  "Read",
  "ReadMcpResource",
  "Skill",
  "Task",
  "TaskCreate",
  "TaskGet",
  "TaskList",
  "TaskOutput",
  "TaskStop",
  "TaskUpdate",
  "TodoWrite",
  "ToolSearch",
  "WebFetch",
  "WebSearch",
  "Write",
}

SDK_SAFE_BUILTINS = {"Read", "Glob", "Grep"}
SDK_WEB_BUILTINS = {"WebSearch", "WebFetch"}
SDK_GATED_BUILTINS = {"Write", "Edit", "Bash", "NotebookEdit", "BashOutput", "KillBash"}
_UNSET = object()


@dataclass
class AgentSDKConfig:
  """Configuration for the optional Anthropic agent SDK runner."""

  api_key: str
  auth_mode: str | None = None
  auth_token: str | None = None
  model: str | None = None
  max_budget_usd: float | None = None
  cwd: str | Path | None = None
  disallowed_tools: list[str] = field(default_factory=list)
  user_id: str | None = None
  channel: str | None = None
  rate_table_version: str | None = None
  billing_mode: str | None = None
  request_id: str | None = None


def _sdk_web_tool_channels() -> set[Optional[str]]:
  try:
    from api.tool_catalog import WEB_TOOL_CHANNELS
  except Exception:
    return {"web", "telegram", "cli", "tui"}
  return set(WEB_TOOL_CHANNELS)


def _resolve_channel_tier(
  channel: Optional[str],
  channel_tiers: Dict[Optional[str], Dict[str, set[str]]],
) -> Dict[str, set[str]]:
  default_tier = channel_tiers.get(None, {"always": set(), "defer": set()})
  return channel_tiers.get(channel, default_tier)


def _resolve_mcp_config_path(config_path: Path | str | None | object = _UNSET) -> Path | None:
  if config_path is _UNSET:
    env_path = os.getenv("MCP_CONFIG_PATH", "").strip()
    return Path(env_path).expanduser() if env_path else Path.home() / ".claude.json"
  if config_path is None:
    return None
  return Path(config_path).expanduser()


def build_disallowed_tools(
  channel: Optional[str],
  channel_tiers: Dict[Optional[str], Dict[str, set[str]]],
  extra_blocked: set[str] | None = None,
  session: Any | None = None,
) -> List[str]:
  tier = _resolve_channel_tier(channel, channel_tiers)
  allowed = set(SDK_SAFE_BUILTINS)
  if channel in _sdk_web_tool_channels():
    allowed |= SDK_WEB_BUILTINS

  blocked = set(SDK_KNOWN_BUILTINS) - allowed
  # NOTE (deferred-tier limitation, agent-sdk only): the SDK fixes its MCP server
  # set at session start and has no mid-session `load_tools`, so deferred-tier
  # servers are hard-blocked here and never started by load_mcp_config_for_sdk
  # (always-only, below). A server placed in a channel's `defer` set (e.g.
  # fred-mcp, macro-mcp) is therefore unreachable under AGENT_PROVIDER=agent-sdk.
  # The `anthropic` provider does NOT have this limitation — it honors load_tools,
  # so `defer` works there. To make a deferred server usable under agent-sdk,
  # promote it to the channel's `always` tier in CHANNEL_TIERS, or add
  # deferred-loading support to this provider.
  for server_name in tier.get("defer", set()):
    blocked.add(f"mcp__{server_name}__*")
  if extra_blocked:
    blocked |= extra_blocked
  try:
    from agent.shared.server_policies import get_forbidden_tools_for_session, get_server_for_policy_tool
  except Exception:
    try:
      from api.agent.shared.server_policies import get_forbidden_tools_for_session, get_server_for_policy_tool
    except Exception:
      get_forbidden_tools_for_session = None  # type: ignore[assignment]
      get_server_for_policy_tool = None  # type: ignore[assignment]

  if get_forbidden_tools_for_session is not None and get_server_for_policy_tool is not None:
    for tool_name in get_forbidden_tools_for_session(session):
      server_name = get_server_for_policy_tool(tool_name) or "portfolio-mcp"
      blocked.add(f"mcp__{server_name}__{tool_name}")
  return sorted(blocked)


def load_mcp_config_for_sdk(
  channel: Optional[str],
  channel_tiers: Dict[Optional[str], Dict[str, set[str]]],
  config_path: Path | str | None | object = _UNSET,
  *,
  session: Any | None = None,
) -> Dict[str, Dict[str, Any]]:
  _ = session
  resolved_config_path = _resolve_mcp_config_path(config_path)
  if resolved_config_path is None or not resolved_config_path.exists():
    return {}

  try:
    with open(resolved_config_path, "r", encoding="utf-8") as handle:
      data = json.load(handle)
  except Exception:
    return {}

  if not isinstance(data, dict):
    return {}

  mcp_servers = data.get("mcpServers")
  if not isinstance(mcp_servers, dict):
    return {}

  tier = _resolve_channel_tier(channel, channel_tiers)
  # always-only: the SDK has no mid-session load_tools, so deferred-tier servers
  # are not started here (and are blocked in build_disallowed_tools — see note there).
  allowed_servers = tier.get("always", set())
  sdk_configs: Dict[str, Dict[str, Any]] = {}

  for server_name in sorted(allowed_servers):
    raw_config = mcp_servers.get(server_name)
    if not isinstance(raw_config, dict):
      continue

    server_type = str(raw_config.get("type", "stdio") or "stdio").strip().lower()
    if server_type == "stdio":
      command = str(raw_config.get("command", "")).strip()
      if not command:
        continue
      entry: Dict[str, Any] = {"command": command}
      args_raw = raw_config.get("args")
      if isinstance(args_raw, list):
        entry["args"] = [str(arg) for arg in args_raw]
      env_raw = raw_config.get("env")
      if isinstance(env_raw, dict):
        env = {str(key): str(value) for key, value in env_raw.items() if value is not None}
        if env:
          entry["env"] = env
      cwd = raw_config.get("cwd")
      if cwd:
        entry["cwd"] = str(cwd)
      sdk_configs[server_name] = entry
      continue

    if server_type in {"sse", "http"}:
      url = str(raw_config.get("url", "")).strip()
      if not url:
        continue
      entry = {"type": server_type, "url": url}
      headers_raw = raw_config.get("headers")
      if isinstance(headers_raw, dict):
        headers = {str(key): str(value) for key, value in headers_raw.items() if value is not None}
        if headers:
          entry["headers"] = headers
      sdk_configs[server_name] = entry

  return sdk_configs


def _validate_sdk_version() -> None:
  try:
    import claude_agent_sdk
  except ImportError as exc:
    raise RuntimeError("claude-agent-sdk dependency is required when AGENT_PROVIDER=agent-sdk") from exc

  installed = str(getattr(claude_agent_sdk, "__version__", "") or "")
  if installed != SDK_PINNED_VERSION:
    raise RuntimeError(
      f"claude-agent-sdk {installed or '<unknown>'} != pinned {SDK_PINNED_VERSION}. "
      "Review new built-in tools, update SDK_KNOWN_BUILTINS, then bump SDK_PINNED_VERSION."
    )


def has_active_credential() -> bool:
  return bool(os.getenv("AGENT_SDK_API_KEY", "").strip())


def estimate_cost(
  model: str,
  input_tokens: int,
  output_tokens: int,
  cache_read_tokens: int = 0,
  cache_creation_tokens: int = 0,
) -> CostEstimate:
  provider = AnthropicProvider()
  return provider.estimate_cost(
    model,
    input_tokens,
    output_tokens,
    cache_read_tokens=cache_read_tokens,
    cache_creation_tokens=cache_creation_tokens,
  )


__all__ = [
  "AgentSDKConfig",
  "SDK_GATED_BUILTINS",
  "SDK_KNOWN_BUILTINS",
  "SDK_PINNED_VERSION",
  "SDK_SAFE_BUILTINS",
  "SDK_WEB_BUILTINS",
  "build_disallowed_tools",
  "estimate_cost",
  "has_active_credential",
  "load_mcp_config_for_sdk",
]
