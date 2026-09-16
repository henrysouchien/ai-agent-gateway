from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .anthropic import AnthropicProvider
from .base import CostEstimate
from ..policy_imports import (
  load_server_policy_helpers,
  load_server_policy_module,
  resolve_effective_role,
)


SDK_PINNED_VERSION = "0.2.153"

# Bundled CLI registry, including conditionally enabled tools and legacy names.
# Keep unreviewed built-ins denied even when CLI capabilities change at runtime.
SDK_KNOWN_BUILTINS = {
  "Agent",
  "AppifactRepl",
  "Artifact",
  "ArtifactCheck",
  "ArtifactComments",
  "ArtifactData",
  "AskUserQuestion",
  "Bash",
  "BashOutput",
  "ClaudeDesign",
  "CronCreate",
  "CronDelete",
  "CronList",
  "DesignSync",
  "Edit",
  "EndConversation",
  "EnterPlanMode",
  "EnterWorktree",
  "ExitPlanMode",
  "ExitWorktree",
  "FetchInboxMessage",
  "Glob",
  "Grep",
  "KillBash",
  "LSP",
  "ListAgents",
  "ListConnectors",
  "ListMcpResources",
  "ListMcpResourcesTool",
  "ListPlugins",
  "ListSkills",
  "Monitor",
  "NotebookEdit",
  "ObserverReport",
  "Poll",
  "PowerShell",
  "Projects",
  "ProposeGoal",
  "PushNotification",
  "REPL",
  "Read",
  "ReadMcpResource",
  "ReadMcpResourceDirTool",
  "ReadMcpResourceTool",
  "ReadNotifications",
  "RefreshMcpTools",
  "RemoteTrigger",
  "ReportFindings",
  "ScheduleWakeup",
  "SearchMcpRegistry",
  "SearchPlugins",
  "SearchSkills",
  "SendFeedback",
  "SendFile",
  "SendMessage",
  "SendUserFile",
  "SendUserMessage",
  "ShareOnboardingGuide",
  "ShowOnboardingRolePicker",
  "Skill",
  "SubagentHandback",
  "SuggestConnectors",
  "SuggestPluginInstall",
  "SuggestSkills",
  "Task",
  "TaskCreate",
  "TaskGet",
  "TaskList",
  "TaskOutput",
  "TaskStop",
  "TaskUpdate",
  "TodoWrite",
  "ToolSearch",
  "WaitForMcpServers",
  "WebFetch",
  "WebSearch",
  "Workflow",
  "Write",
  "enable__mcp__claude-in-chrome",
  "enable__mcp__remote-devices__Claude_Browser",
  "enable__mcp__remote-devices__computer",
  "memory_list",
  "memory_read",
  "memory_write",
  "propose_skills",
  "self_hosted_runner_get_pool",
  "self_hosted_runner_list_runners",
  "self_hosted_runner_list_secrets",
  "self_hosted_runner_list_sessions",
  "self_hosted_runner_read_health",
  "self_hosted_runner_read_metrics",
  "self_hosted_runner_requeue_session",
  "self_hosted_runner_spawn_local",
  "self_hosted_runner_tail_log",
}

SDK_SAFE_BUILTINS = {"Read", "Glob", "Grep"}
SDK_WEB_BUILTINS = {"WebSearch", "WebFetch"}
SDK_GATED_BUILTINS = {"Write", "Edit", "Bash", "NotebookEdit", "BashOutput", "KillBash"}
# Intentionally differs from api/agent/shared/tool_catalog.py:WEB_TOOL_CHANNELS,
# which includes "discord". Whether Discord agent-SDK sessions receive hosted
# WebSearch/WebFetch remains an open product decision; ruled 2026-08-31 to keep
# today's behavior.
SDK_WEB_TOOL_CHANNELS: frozenset[str] = frozenset({"web", "telegram", "cli", "tui"})

class _Unset(Enum):
  TOKEN = 0


@dataclass
class AgentSDKConfig:
  """Non-routing configuration for the optional Anthropic agent SDK runner."""

  max_budget_usd: float | None = None
  cwd: str | Path | None = None
  disallowed_tools: list[str] = field(default_factory=list)
  user_id: str | None = None
  channel: str | None = None
  rate_table_version: str | None = None
  billing_mode: str | None = None
  request_id: str | None = None




def _resolve_channel_tier(
  channel: Optional[str],
  channel_tiers: Dict[Optional[str], Dict[str, set[str]]],
) -> Dict[str, set[str]]:
  default_tier = channel_tiers.get(None, {"always": set(), "defer": set()})
  return channel_tiers.get(channel, default_tier)


def _resolve_mcp_config_path(
  config_path: Path | str | None | _Unset = _Unset.TOKEN,
) -> Path | None:
  if config_path is _Unset.TOKEN:
    env_path = os.getenv("MCP_CONFIG_PATH", "").strip()
    return Path(env_path).expanduser() if env_path else None
  if config_path is None:
    return None
  return Path(config_path).expanduser()


def _authority_role(session: Any | None) -> str:
  raw_role = (
    session.get("role")
    if isinstance(session, dict)
    else getattr(session, "role", None)
  )
  return resolve_effective_role(raw_role if isinstance(raw_role, str) else None)


def _unclassified_tool_denials(available_tools: set[str]) -> set[str]:
  """Deny every available identity that carries no reviewed policy class.

  The SDK enforces availability through `disallowed_tools`, so deny-by-default
  can only be expressed against the identities the query actually exposes.  An
  identity whose bare tool name resolves to neither an MCP class nor a local
  effect is unreviewed, and unreviewed reach is not invite authority.
  """
  policy_module = load_server_policy_module()
  is_classified = (
    getattr(policy_module, "tool_is_policy_classified", None)
    if policy_module is not None
    else None
  )
  denials: set[str] = set()
  for raw_identity in available_tools:
    identity = str(raw_identity or "").strip()
    if not identity:
      continue
    # No host policy means no classification is knowable, so nothing is
    # reviewed and everything available is denied.
    if is_classified is None or not is_classified(identity):
      denials.add(identity)
  return denials


def build_disallowed_tools(
  channel: Optional[str],
  channel_tiers: Dict[Optional[str], Dict[str, set[str]]],
  extra_blocked: set[str] | None = None,
  session: Any | None = None,
  available_tools: set[str] | None = None,
) -> List[str]:
  tier = _resolve_channel_tier(channel, channel_tiers)
  allowed = set(SDK_SAFE_BUILTINS)
  if channel in SDK_WEB_TOOL_CHANNELS:
    allowed |= SDK_WEB_BUILTINS

  blocked = set(SDK_KNOWN_BUILTINS) - allowed
  # Deferred servers stay blocked in the initial SDK query. The interactive SDK
  # runtime retains their configs separately and rebuilds the query after its
  # in-process load_tools bridge admits a server or pack.
  for server_name in tier.get("defer", set()):
    blocked.add(f"mcp__{server_name}__*")
  if extra_blocked:
    blocked |= extra_blocked
  get_forbidden_tools_for_session, get_server_for_policy_tool, _get_tool_class = load_server_policy_helpers()

  if get_forbidden_tools_for_session is None or get_server_for_policy_tool is None:
    blocked.add("mcp__*")
  else:
    for tool_name in get_forbidden_tools_for_session(session):
      server_name = get_server_for_policy_tool(tool_name)
      if server_name is None:
        blocked.add(tool_name)
        continue
      blocked.add(f"mcp__{server_name}__{tool_name}")
  if available_tools and _authority_role(session) != "owner":
    blocked |= _unclassified_tool_denials(set(available_tools))
  return sorted(blocked)


def load_mcp_config_for_sdk(
  channel: Optional[str],
  channel_tiers: Dict[Optional[str], Dict[str, set[str]]],
  config_path: Path | str | None | _Unset = _Unset.TOKEN,
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
  # Initial-query configs are always-only. The interactive runtime makes a
  # separate all-eligible call to retain deferred configs for query rebuilds.
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
    raise RuntimeError("claude-agent-sdk dependency is required for the selected SDK adapter") from exc

  installed = str(getattr(claude_agent_sdk, "__version__", "") or "")
  if installed != SDK_PINNED_VERSION:
    raise RuntimeError(
      f"claude-agent-sdk {installed or '<unknown>'} != pinned {SDK_PINNED_VERSION}. "
      "Review new built-in tools, update SDK_KNOWN_BUILTINS, then bump SDK_PINNED_VERSION."
    )


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
  "load_mcp_config_for_sdk",
]
