from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .fixture_gate import fixture_provider_available, is_fixture_skill_name, require_fixture_provider_available
from ._io import _atomic_write_json, _read_json_object

log = logging.getLogger("agent_gateway.skills")
_FRONTMATTER_DELIMITER = "---"
_BLOCK_REF_RE = re.compile(r"(?<!\\)\{\{([A-Z][A-Z0-9_]*)\}\}")
_ESCAPE_SENTINEL = "\x00BLOCK_ESC\x00"
AGENT_DESCRIPTION_MAX_CHARS = 240
AGENT_DESCRIPTION_PLACEHOLDER = "(no description)"
SKILL_STATE_CLASSES = frozenset({
  "producer",
  "advisor-with-decision-log",
  "advisor-no-state",
  "deprecated",
})
Mode = Literal["read_only", "preview", "apply", "model_writer"]


@dataclass
class SkillProfile:
  """Parsed markdown skill definition.

  Skill files may include YAML frontmatter for metadata and a markdown body that
  becomes the sub-agent system prompt.
  """

  name: str
  system_prompt: str
  version: str | None = None
  model: str | None = None
  max_turns: int | None = None
  timeout: float | None = None
  tool_packs: list[str] | None = None
  persist_state: bool = False
  scope: str | None = None
  interactive: bool = False
  metadata: dict[str, Any] | None = None
  mcp_servers: list[str] | None = None
  mcp_tools: dict[str, list[str]] | None = None
  session_inject_servers: list[str] | None = None
  timeout_overrides: dict[str, int] | None = None
  state_dir: str | None = None
  max_budget_usd: float | None = None
  max_tokens: int | None = None
  thinking: bool | None = None
  max_retries: int | None = None
  initial_message: str | None = None
  delivery_label: str | None = None
  agent_callable: bool = False
  agent_description: str | None = None
  resumable: bool = False
  resume_mcp_session_reset_ok: bool = False
  mode: str = "full"
  extra_excluded_tools: set[str] = field(default_factory=set)
  tool_packs_enabled: bool = True
  state_class: str | None = None
  mutation_mode: Mode | str | None = None
  # Append new fields BELOW this line. Positional construction is part of the
  # contract (test_skill_profile_positional_construction_compat pins the order
  # through mutation_mode); inserting a field mid-struct shifts later fields and
  # breaks positional callers.
  provider: str | None = None


def _clean_string(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value).strip()
  return text or None


def _coerce_optional_int(value: Any, *, field_name: str, path: Path) -> int | None:
  if value is None:
    return None
  if isinstance(value, bool):
    raise ValueError(f"{path}: '{field_name}' must be an integer")
  if isinstance(value, int):
    return value
  if isinstance(value, float) and value.is_integer():
    return int(value)
  if isinstance(value, str):
    text = value.strip()
    if text:
      try:
        return int(text)
      except ValueError as exc:
        raise ValueError(f"{path}: '{field_name}' must be an integer") from exc
  raise ValueError(f"{path}: '{field_name}' must be an integer")


def _coerce_optional_float(value: Any, *, field_name: str, path: Path) -> float | None:
  if value is None:
    return None
  if isinstance(value, bool):
    raise ValueError(f"{path}: '{field_name}' must be a number")
  if isinstance(value, (int, float)):
    return float(value)
  if isinstance(value, str):
    text = value.strip()
    if text:
      try:
        return float(text)
      except ValueError as exc:
        raise ValueError(f"{path}: '{field_name}' must be a number") from exc
  raise ValueError(f"{path}: '{field_name}' must be a number")


def _coerce_optional_bool(value: Any, *, field_name: str, path: Path) -> bool:
  if value is None:
    return False
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1", "on"}:
      return True
    if normalized in {"false", "no", "0", "off"}:
      return False
  raise ValueError(f"{path}: '{field_name}' must be a boolean")


def _coerce_optional_string_list(
  value: Any,
  *,
  field_name: str,
  path: Path,
  strict_shape: bool = False,
  strict_items: bool = False,
) -> list[str] | None:
  if value is None:
    return None
  raw_items: list[Any]
  if isinstance(value, str):
    if strict_shape:
      raise ValueError(f"{path}: '{field_name}' must be a list of strings")
    raw_items = [value]
  elif isinstance(value, (list, tuple, set)):
    raw_items = list(value)
  else:
    raise ValueError(f"{path}: '{field_name}' must be a list of strings")

  items: list[str] = []
  for item in raw_items:
    if strict_items and not isinstance(item, str):
      raise ValueError(f"{path}: '{field_name}' entries must be strings")
    text = _clean_string(item)
    if text:
      items.append(text)
  return items or None


def _coerce_optional_string_set(
  value: Any,
  *,
  field_name: str,
  path: Path,
  strict_shape: bool = False,
  strict_items: bool = False,
) -> set[str]:
  if value is None:
    return set()
  raw_items: list[Any]
  if isinstance(value, str):
    if strict_shape:
      raise ValueError(f"{path}: '{field_name}' must be a list of strings")
    raw_items = [value]
  elif isinstance(value, (list, tuple, set)):
    raw_items = list(value)
  else:
    raise ValueError(f"{path}: '{field_name}' must be a list of strings")

  items: set[str] = set()
  for item in raw_items:
    if strict_items and not isinstance(item, str):
      raise ValueError(f"{path}: '{field_name}' entries must be strings")
    text = _clean_string(item)
    if text:
      items.add(text)
  return items


def _coerce_optional_mcp_tools(
  value: Any,
  *,
  field_name: str,
  path: Path,
) -> dict[str, list[str]] | None:
  if value is None:
    return None
  if not isinstance(value, dict):
    raise ValueError(f"{path}: '{field_name}' must be a mapping of server names to tool lists")

  declared: dict[str, list[str]] = {}
  for raw_server, raw_tools in value.items():
    if raw_server is not None and not isinstance(raw_server, str):
      raise ValueError(f"{path}: '{field_name}' server names must be strings")
    server_name = _clean_string(raw_server)
    if server_name is None:
      continue
    tools = _coerce_optional_string_list(
      raw_tools,
      field_name=f"{field_name}.{server_name}",
      path=path,
      strict_shape=True,
      strict_items=True,
    )
    if not tools:
      continue
    deduped_tools: list[str] = []
    seen: set[str] = set()
    for tool_name in tools:
      if tool_name in seen:
        continue
      seen.add(tool_name)
      deduped_tools.append(tool_name)
    declared[server_name] = deduped_tools
  return declared or None


def _coerce_optional_timeout_overrides(
  value: Any,
  *,
  field_name: str,
  path: Path,
) -> dict[str, int] | None:
  if value is None:
    return None
  if not isinstance(value, dict):
    raise ValueError(f"{path}: '{field_name}' must be a mapping of server names to timeout seconds")

  overrides: dict[str, int] = {}
  for raw_name, raw_timeout in value.items():
    server_name = _clean_string(raw_name)
    if server_name is None:
      continue
    timeout_value = _coerce_optional_int(
      raw_timeout,
      field_name=f"{field_name}.{server_name}",
      path=path,
    )
    if timeout_value is not None:
      overrides[server_name] = timeout_value
  return overrides or None


def _coerce_optional_scope(value: Any, *, field_name: str, path: Path) -> str | None:
  text = _clean_string(value)
  if text is None:
    return None
  if text not in {"ticker", "portfolio", "industry"}:
    raise ValueError(
      f"{path}: Invalid scope '{text}': must be 'ticker', 'portfolio', or 'industry'"
    )
  return text


def _coerce_mode(value: Any, *, field_name: str, path: Path) -> str:
  text = _clean_string(value)
  if text is None:
    return "full"
  if text not in {"full", "recommend"}:
    raise ValueError(f"{path}: '{field_name}' must be 'full' or 'recommend'")
  return text


def _coerce_optional_state_class(value: Any, *, field_name: str, path: Path) -> str | None:
  text = _clean_string(value)
  if text is None:
    return None
  if text not in SKILL_STATE_CLASSES:
    allowed = ", ".join(sorted(SKILL_STATE_CLASSES))
    raise ValueError(f"{path}: '{field_name}' must be one of: {allowed}")
  return text


def _split_frontmatter(text: str, *, path: Path) -> tuple[dict[str, Any], str]:
  lines = text.splitlines()
  if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
    return {}, text

  for index, line in enumerate(lines[1:], start=1):
    if line.strip() != _FRONTMATTER_DELIMITER:
      continue
    frontmatter_text = "\n".join(lines[1:index])
    body = "\n".join(lines[index + 1 :])
    payload = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(payload, dict):
      raise ValueError(f"{path}: skill frontmatter must be a YAML mapping")
    return payload, body

  return {}, text


def _read_block(block_name: str, blocks_dir: Path) -> str:
  block_path = blocks_dir / f"{block_name.lower().replace('_', '-')}.md"
  if not block_path.exists():
    raise FileNotFoundError(
      f"Block '{block_name}' not found: expected {block_path}"
    )
  return block_path.read_text(encoding="utf-8")


def resolve_blocks(content: str, blocks_dir: Path) -> str:
  """Resolve {{BLOCK_NAME}} references in skill content from block files."""
  working = content.replace("\\{{", _ESCAPE_SENTINEL)
  working = _BLOCK_REF_RE.sub(lambda match: _read_block(match.group(1), blocks_dir), working)
  return working.replace(_ESCAPE_SENTINEL, "{{")


def parse_skill_file(path: Path) -> SkillProfile:
  """Parse a markdown skill file into a `SkillProfile`.

  The parser accepts optional YAML frontmatter delimited by `---` and treats the
  remaining markdown body as the skill prompt.
  """
  text = path.read_text(encoding="utf-8")
  frontmatter, body = _split_frontmatter(text, path=path)

  raw_name = frontmatter.pop("name", None)
  raw_version = frontmatter.pop("version", None)
  raw_model = frontmatter.pop("model", None)
  raw_provider = frontmatter.pop("provider", None)
  raw_max_turns = frontmatter.pop("max_turns", None)
  raw_timeout = frontmatter.pop("timeout", None)
  raw_tool_packs = frontmatter.pop("tool_packs", None)
  raw_persist_state = frontmatter.pop("persist_state", None)
  raw_scope = frontmatter.pop("scope", None)
  raw_interactive = frontmatter.pop("interactive", None)
  raw_agent_callable = frontmatter.pop("agent_callable", None)
  raw_agent_description = frontmatter.pop("agent_description", None)
  raw_resumable = frontmatter.pop("resumable", None)
  raw_resume_mcp_session_reset_ok = frontmatter.pop("resume_mcp_session_reset_ok", None)
  raw_mode = frontmatter.pop("mode", None)
  raw_state_class = frontmatter.pop("state_class", None)
  raw_mutation_mode = frontmatter.pop("mutation_mode", None)
  raw_extra_excluded_tools = frontmatter.pop("extra_excluded_tools", None)
  raw_tool_packs_enabled = frontmatter.pop("tool_packs_enabled", None)
  raw_metadata = frontmatter.pop("metadata", None)

  if raw_metadata is None:
    metadata: dict[str, Any] = {}
  elif isinstance(raw_metadata, dict):
    metadata = dict(raw_metadata)
  else:
    raise ValueError(f"{path}: 'metadata' must be a mapping when provided")

  metadata.update(frontmatter)
  original_metadata_keys = set(metadata.keys())

  raw_mcp_servers = metadata.pop("mcp_servers", None)
  raw_session_inject_servers = metadata.pop("session_inject_servers", None)
  raw_timeout_overrides = metadata.pop("timeout_overrides", None)
  raw_state_dir = metadata.pop("state_dir", None)
  raw_max_budget_usd = metadata.pop("max_budget_usd", None)
  raw_max_tokens = metadata.pop("max_tokens", None)
  raw_thinking = metadata.pop("thinking", None)
  raw_max_retries = metadata.pop("max_retries", None)
  raw_initial_message = metadata.pop("initial_message", None)
  raw_delivery_label = metadata.pop("delivery_label", None)
  raw_mcp_tools = metadata.pop("mcp_tools", None)

  coerced_mcp_servers = _coerce_optional_string_list(
    raw_mcp_servers,
    field_name="mcp_servers",
    path=path,
    strict_shape=True,
    strict_items=True,
  )
  coerced_mcp_tools = _coerce_optional_mcp_tools(
    raw_mcp_tools,
    field_name="mcp_tools",
    path=path,
  )
  coerced_session_inject_servers = _coerce_optional_string_list(
    raw_session_inject_servers,
    field_name="session_inject_servers",
    path=path,
    strict_shape=True,
    strict_items=True,
  )
  coerced_timeout_overrides = _coerce_optional_timeout_overrides(
    raw_timeout_overrides,
    field_name="timeout_overrides",
    path=path,
  )
  coerced_state_dir = _clean_string(raw_state_dir)
  coerced_max_budget_usd = _coerce_optional_float(
    raw_max_budget_usd,
    field_name="max_budget_usd",
    path=path,
  )
  coerced_max_tokens = _coerce_optional_int(
    raw_max_tokens,
    field_name="max_tokens",
    path=path,
  )
  coerced_thinking = (
    None
    if raw_thinking is None
    else _coerce_optional_bool(raw_thinking, field_name="thinking", path=path)
  )
  coerced_max_retries = _coerce_optional_int(
    raw_max_retries,
    field_name="max_retries",
    path=path,
  )
  coerced_initial_message = _clean_string(raw_initial_message)
  coerced_delivery_label = _clean_string(raw_delivery_label)
  coerced_agent_callable = _coerce_optional_bool(
    raw_agent_callable,
    field_name="agent_callable",
    path=path,
  )
  coerced_agent_description = _clean_string(raw_agent_description)
  coerced_resumable = _coerce_optional_bool(raw_resumable, field_name="resumable", path=path)
  coerced_resume_mcp_session_reset_ok = _coerce_optional_bool(
    raw_resume_mcp_session_reset_ok,
    field_name="resume_mcp_session_reset_ok",
    path=path,
  )
  coerced_mode = _coerce_mode(raw_mode, field_name="mode", path=path)
  coerced_state_class = _coerce_optional_state_class(
    raw_state_class,
    field_name="state_class",
    path=path,
  )
  coerced_extra_excluded_tools = _coerce_optional_string_set(
    raw_extra_excluded_tools,
    field_name="extra_excluded_tools",
    path=path,
    strict_shape=True,
    strict_items=True,
  )
  coerced_tool_packs_enabled = (
    True
    if raw_tool_packs_enabled is None
    else _coerce_optional_bool(
      raw_tool_packs_enabled,
      field_name="tool_packs_enabled",
      path=path,
    )
  )

  for key, coerced in [
    ("mcp_servers", coerced_mcp_servers),
    ("mcp_tools", coerced_mcp_tools),
    ("session_inject_servers", coerced_session_inject_servers),
    ("timeout_overrides", coerced_timeout_overrides),
    ("state_dir", coerced_state_dir),
    ("max_budget_usd", coerced_max_budget_usd),
    ("thinking", coerced_thinking),
    ("max_retries", coerced_max_retries),
    ("initial_message", coerced_initial_message),
    ("delivery_label", coerced_delivery_label),
  ]:
    if key in original_metadata_keys:
      metadata[key] = coerced

  if coerced_resumable and coerced_session_inject_servers and not coerced_resume_mcp_session_reset_ok:
    raise ValueError(
      f"{path}: 'resumable: true' with 'session_inject_servers' requires "
      "'resume_mcp_session_reset_ok: true'"
    )

  return SkillProfile(
    name=_clean_string(raw_name) or path.stem,
    system_prompt=body.strip(),
    version=_clean_string(raw_version),
    model=_clean_string(raw_model),
    max_turns=_coerce_optional_int(raw_max_turns, field_name="max_turns", path=path),
    timeout=_coerce_optional_float(raw_timeout, field_name="timeout", path=path),
    tool_packs=_coerce_optional_string_list(raw_tool_packs, field_name="tool_packs", path=path),
    persist_state=_coerce_optional_bool(raw_persist_state, field_name="persist_state", path=path),
    scope=_coerce_optional_scope(raw_scope, field_name="scope", path=path),
    interactive=_coerce_optional_bool(raw_interactive, field_name="interactive", path=path),
    metadata=metadata or None,
    mcp_servers=coerced_mcp_servers,
    mcp_tools=coerced_mcp_tools,
    session_inject_servers=coerced_session_inject_servers,
    timeout_overrides=coerced_timeout_overrides,
    state_dir=coerced_state_dir,
    max_budget_usd=coerced_max_budget_usd,
    max_tokens=coerced_max_tokens,
    thinking=coerced_thinking,
    max_retries=coerced_max_retries,
    initial_message=coerced_initial_message,
    delivery_label=coerced_delivery_label,
    agent_callable=coerced_agent_callable,
    agent_description=coerced_agent_description,
    resumable=coerced_resumable,
    resume_mcp_session_reset_ok=coerced_resume_mcp_session_reset_ok,
    mode=coerced_mode,
    extra_excluded_tools=coerced_extra_excluded_tools,
    tool_packs_enabled=coerced_tool_packs_enabled,
    provider=_clean_string(raw_provider),
    state_class=coerced_state_class,
    mutation_mode=_clean_string(raw_mutation_mode),
  )


def _cap_agent_description(description: str) -> str:
  if len(description) <= AGENT_DESCRIPTION_MAX_CHARS:
    return description
  return description[: AGENT_DESCRIPTION_MAX_CHARS - 1].rstrip() + "…"


def _warn_agent_description_if_needed(profile: SkillProfile) -> None:
  if not profile.agent_callable:
    return
  if not profile.agent_description:
    log.warning(
      "Callable skill '%s' is missing agent_description; rendering %s",
      profile.name,
      AGENT_DESCRIPTION_PLACEHOLDER,
    )
    return
  if len(profile.agent_description) > AGENT_DESCRIPTION_MAX_CHARS:
    log.warning(
      "Callable skill '%s' agent_description is %d chars; max is %d and rendered output will be truncated",
      profile.name,
      len(profile.agent_description),
      AGENT_DESCRIPTION_MAX_CHARS,
    )


class SkillLoader:
  """Load named skill files from a directory.

  Each skill must be a markdown file named `<skill>.md`. `load()` validates the
  file and returns a parsed `SkillProfile`.
  """

  def __init__(self, skills_dir: str | Path):
    self.skills_dir = Path(skills_dir)

  def _skill_path(self, name: str) -> Path:
    skill_name = str(name).strip()
    if not skill_name:
      raise ValueError("Skill name is required")

    resolved_dir = self.skills_dir.resolve()
    path = (resolved_dir / f"{skill_name}.md").resolve()
    if path.parent != resolved_dir:
      raise ValueError(f"Invalid skill name '{skill_name}'")
    return path

  def load(self, name: str) -> SkillProfile:
    if is_fixture_skill_name(name):
      require_fixture_provider_available("fixture skill", error_type=ValueError)
    path = self._skill_path(name)
    if not path.exists():
      available = self.list_skills()
      available_label = ", ".join(available) if available else "(none)"
      raise FileNotFoundError(f"Skill '{name}' not found. Available: {available_label}")
    profile = parse_skill_file(path)
    _warn_agent_description_if_needed(profile)
    return profile

  def list_skills(self) -> list[str]:
    if not self.skills_dir.exists():
      return []
    names = [path.stem for path in self.skills_dir.glob("*.md") if path.is_file()]
    if not fixture_provider_available():
      names = [name for name in names if not is_fixture_skill_name(name)]
    return sorted(names)

  def list_callable_skills_with_descriptions(self) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for name in self.list_skills():
      profile = self.load(name)
      if not profile.agent_callable:
        continue
      description = profile.agent_description or AGENT_DESCRIPTION_PLACEHOLDER
      entries.append((profile.name, _cap_agent_description(description)))
    return sorted(entries, key=lambda item: item[0])

  def exists(self, name: str) -> bool:
    try:
      return self._skill_path(name).is_file()
    except ValueError:
      return False


class SkillStateStore:
  """Persist per-skill JSON state in a single file."""

  def __init__(self, state_file: str | Path):
    self.state_file = Path(state_file)

  def _read_all(self) -> dict[str, Any]:
    return _read_json_object(self.state_file)

  def get(self, skill_name: str) -> dict[str, Any]:
    payload = self._read_all()
    state = payload.get(skill_name)
    if isinstance(state, dict):
      return dict(state)
    return {}

  def set(self, skill_name: str, state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
      raise TypeError("state must be a dict")
    payload = self._read_all()
    payload[str(skill_name)] = dict(state)
    _atomic_write_json(self.state_file, payload)

  def clear(self, skill_name: str) -> None:
    if not self.state_file.exists():
      return
    payload = self._read_all()
    if skill_name not in payload:
      return
    payload.pop(skill_name, None)
    _atomic_write_json(self.state_file, payload)


__all__ = [
  "AGENT_DESCRIPTION_MAX_CHARS",
  "AGENT_DESCRIPTION_PLACEHOLDER",
  "Mode",
  "SkillLoader",
  "SkillProfile",
  "SkillStateStore",
  "parse_skill_file",
  "resolve_blocks",
]
