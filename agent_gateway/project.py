from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI

from .easy import create_agent

DEFAULT_AGENT_CONFIG = "agent.yaml"
PROJECT_CONFIG_ENV = "AGENT_GATEWAY_PROJECT_CONFIG"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and use short paragraphs."
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TOP_LEVEL_KEYS = {
  "name",
  "system_prompt",
  "model_key",
  "effort",
  "host",
  "port",
  "api_prefix",
  "skills_dir",
  "skill_state_file",
  "outputs_dir",
  "mcp_servers",
  "code_execution",
  "max_tokens",
  "max_turns",
  "max_budget_usd",
  "per_turn_timeout",
}


class AgentProjectError(ValueError):
  """Raised when an agent project config or scaffold request is invalid."""


@dataclass(frozen=True, slots=True)
class AgentProjectConfig:
  path: Path
  name: str
  system_prompt: str
  model_key: str | None
  effort: str | None
  host: str
  port: int
  api_prefix: str
  skills_dir: Path | None
  skill_state_file: Path | None
  outputs_dir: Path | None
  mcp_servers: dict[str, dict[str, Any]]
  code_execution: bool
  max_tokens: int
  max_turns: int | None
  max_budget_usd: float | None
  per_turn_timeout: int


def validate_project_name(name: str) -> str:
  cleaned = name.strip()
  if not _NAME_RE.fullmatch(cleaned):
    raise AgentProjectError(
      "Project name must be 1-64 letters, numbers, underscores, or hyphens, "
      "and start with a letter or number."
    )
  return cleaned


def default_agent_config_payload(
  *,
  name: str,
  model_key: str | None = None,
  effort: str | None = None,
) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "name": validate_project_name(name),
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "host": "127.0.0.1",
    "port": 8000,
    "api_prefix": "/api",
    "skills_dir": "skills",
    "skill_state_file": "skill_state.json",
    "outputs_dir": "outputs",
    "mcp_servers": {},
    "code_execution": False,
    "max_tokens": 16000,
    "max_turns": None,
    "max_budget_usd": None,
    "per_turn_timeout": 300,
  }
  if model_key is not None:
    payload["model_key"] = model_key
  if effort is not None:
    payload["effort"] = effort
  return payload


def default_agent_py() -> str:
  return (
    "from agent_gateway.project import create_agent_from_yaml\n\n"
    "app = create_agent_from_yaml(\"agent.yaml\")\n"
  )


def default_skill_markdown() -> str:
  return (
    "---\n"
    "name: general\n"
    "agent_callable: true\n"
    "agent_description: General-purpose helper skill for small delegated tasks.\n"
    "persist_state: true\n"
    "---\n\n"
    "Work through the request directly. Keep the response concise and cite any assumptions.\n"
  )


def default_project_readme(name: str) -> str:
  return (
    f"# {name}\n\n"
    "Run this agent locally with:\n\n"
    "```bash\n"
    "agent run\n"
    "```\n\n"
    "The gateway listens on the host and port configured in `agent.yaml`.\n"
  )


def write_agent_project(
  target_dir: str | Path,
  *,
  name: str,
  model_key: str | None = None,
  effort: str | None = None,
  force: bool = False,
) -> list[Path]:
  root = Path(target_dir).expanduser()
  project_name = validate_project_name(name)
  payload = default_agent_config_payload(
    name=project_name,
    model_key=model_key,
    effort=effort,
  )
  files = {
    root / DEFAULT_AGENT_CONFIG: yaml.safe_dump(payload, sort_keys=False),
    root / "agent.py": default_agent_py(),
    root / "skills" / "general.md": default_skill_markdown(),
    root / "README.md": default_project_readme(project_name),
    root / ".gitignore": "outputs/\n__pycache__/\n*.pyc\n",
  }

  existing = [path for path in files if path.exists()]
  if existing and not force:
    rels = ", ".join(str(path.relative_to(root)) for path in existing)
    raise AgentProjectError(f"Refusing to overwrite existing files: {rels}")

  written: list[Path] = []
  for path, content in files.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    written.append(path)
  (root / "outputs").mkdir(parents=True, exist_ok=True)
  return written


def load_agent_project_payload(config_path: str | Path = DEFAULT_AGENT_CONFIG) -> dict[str, Any]:
  path = Path(config_path).expanduser()
  try:
    raw = path.read_text(encoding="utf-8")
  except FileNotFoundError as exc:
    raise AgentProjectError(f"Agent config not found: {path}") from exc
  except OSError as exc:
    raise AgentProjectError(f"Failed to read agent config: {path}") from exc

  try:
    payload = yaml.safe_load(raw) or {}
  except yaml.YAMLError as exc:
    raise AgentProjectError(f"Invalid YAML in agent config: {path}") from exc
  if not isinstance(payload, dict):
    raise AgentProjectError(f"Agent config must be a mapping: {path}")
  return dict(payload)


def save_agent_project_payload(
  payload: dict[str, Any],
  config_path: str | Path = DEFAULT_AGENT_CONFIG,
) -> None:
  path = Path(config_path).expanduser()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_agent_project_config(config_path: str | Path = DEFAULT_AGENT_CONFIG) -> AgentProjectConfig:
  path = Path(config_path).expanduser()
  payload = load_agent_project_payload(path)
  unknown = sorted(set(payload) - _TOP_LEVEL_KEYS)
  if unknown:
    raise AgentProjectError(f"Unsupported agent config keys: {', '.join(unknown)}")

  base_dir = path.parent
  name = validate_project_name(_string(payload.get("name"), "agent"))
  mcp_servers = _mcp_servers(payload.get("mcp_servers"))
  return AgentProjectConfig(
    path=path,
    name=name,
    system_prompt=_string(payload.get("system_prompt"), DEFAULT_SYSTEM_PROMPT),
    model_key=_optional_string(payload.get("model_key")),
    effort=_optional_string(payload.get("effort")),
    host=_string(payload.get("host"), "127.0.0.1"),
    port=_int(payload.get("port"), 8000, key="port"),
    api_prefix=_api_prefix(_string(payload.get("api_prefix"), "/api")),
    skills_dir=_optional_path(base_dir, payload.get("skills_dir")),
    skill_state_file=_optional_path(base_dir, payload.get("skill_state_file")),
    outputs_dir=_optional_path(base_dir, payload.get("outputs_dir")),
    mcp_servers=mcp_servers,
    code_execution=_bool(payload.get("code_execution"), False),
    max_tokens=_int(payload.get("max_tokens"), 16000, key="max_tokens"),
    max_turns=_optional_int(payload.get("max_turns"), key="max_turns"),
    max_budget_usd=_optional_float(payload.get("max_budget_usd"), key="max_budget_usd"),
    per_turn_timeout=_int(payload.get("per_turn_timeout"), 300, key="per_turn_timeout"),
  )


def create_agent_from_yaml(config_path: str | Path = DEFAULT_AGENT_CONFIG) -> FastAPI:
  config = load_agent_project_config(config_path)
  selection: dict[str, str] = {}
  if config.model_key is not None:
    selection["model_key"] = config.model_key
  if config.effort is not None:
    selection["effort"] = config.effort
  return create_agent(
    config.system_prompt,
    **selection,
    max_tokens=config.max_tokens,
    mcp_servers=config.mcp_servers or None,
    skills_dir=config.skills_dir,
    skill_state_file=config.skill_state_file,
    outputs_dir=config.outputs_dir,
    code_execution=config.code_execution,
    max_turns=config.max_turns,
    max_budget_usd=config.max_budget_usd,
    per_turn_timeout=config.per_turn_timeout,
    prefix=config.api_prefix,
  )


def create_app_from_env() -> FastAPI:
  return create_agent_from_yaml(os.getenv(PROJECT_CONFIG_ENV, DEFAULT_AGENT_CONFIG))


def _string(value: Any, default: str) -> str:
  if value is None:
    return default
  if not isinstance(value, str):
    raise AgentProjectError(f"Expected string value, got {type(value).__name__}")
  cleaned = value.strip()
  return cleaned if cleaned else default


def _optional_string(value: Any) -> str | None:
  if value is None:
    return None
  if not isinstance(value, str):
    raise AgentProjectError(f"Expected optional string value, got {type(value).__name__}")
  cleaned = value.strip()
  return cleaned or None


def _bool(value: Any, default: bool) -> bool:
  if value is None:
    return default
  if not isinstance(value, bool):
    raise AgentProjectError(f"Expected boolean value, got {type(value).__name__}")
  return value


def _int(value: Any, default: int, *, key: str) -> int:
  if value is None:
    return default
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise AgentProjectError(f"{key} must be a positive integer")
  return value


def _optional_int(value: Any, *, key: str) -> int | None:
  if value is None:
    return None
  return _int(value, 1, key=key)


def _optional_float(value: Any, *, key: str) -> float | None:
  if value is None:
    return None
  if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
    raise AgentProjectError(f"{key} must be a positive number")
  return float(value)


def _optional_path(base_dir: Path, value: Any) -> Path | None:
  text = _optional_string(value)
  if text is None:
    return None
  path = Path(text).expanduser()
  return path if path.is_absolute() else base_dir / path


def _api_prefix(value: str) -> str:
  cleaned = value.strip()
  if not cleaned:
    return "/api"
  prefixed = cleaned if cleaned.startswith("/") else f"/{cleaned}"
  return prefixed.rstrip("/") or "/api"


def _mcp_servers(value: Any) -> dict[str, dict[str, Any]]:
  if value is None:
    return {}
  if not isinstance(value, dict):
    raise AgentProjectError("mcp_servers must be a mapping")
  result: dict[str, dict[str, Any]] = {}
  for name, server in value.items():
    if not isinstance(name, str) or not name.strip():
      raise AgentProjectError("mcp_servers keys must be non-empty strings")
    if not isinstance(server, dict):
      raise AgentProjectError(f"mcp_servers.{name} must be a mapping")
    command = server.get("command")
    if not isinstance(command, str) or not command.strip():
      raise AgentProjectError(f"mcp_servers.{name}.command must be a non-empty string")
    args = server.get("args", [])
    if args is None:
      args = []
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
      raise AgentProjectError(f"mcp_servers.{name}.args must be a list of strings")
    result[name.strip()] = dict(server)
    result[name.strip()]["command"] = command.strip()
    result[name.strip()]["args"] = list(args)
  return result
