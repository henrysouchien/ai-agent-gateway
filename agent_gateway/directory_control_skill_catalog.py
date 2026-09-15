"""Lazy external-directory adapter for the control skill catalog wire."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import stat
from typing import Any

import yaml

from .control_skill_catalog import (
  ControlSkillDetail,
  ControlSkillSummary,
  ControlSkillUnavailableError,
  is_control_skill_name,
)
from .skills import SkillLoader, SkillProfile, resolve_blocks


log = logging.getLogger("agent_gateway.directory_control_skill_catalog")

_FRONTMATTER_DELIMITER = "---"
_PORTFOLIO_CONTEXT_NAMES = frozenset({
  "portfolio",
  "portfolio_context",
  "portfolio_name",
})


def _clean_text(value: object) -> str | None:
  if value is None:
    return None
  text = str(value).strip()
  return text or None


def _coerce_bool(
  value: object,
  *,
  field_name: str,
  default: bool = False,
) -> bool:
  if value is None:
    return default
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1", "on"}:
      return True
    if normalized in {"false", "no", "0", "off"}:
      return False
  raise ValueError(f"{field_name} must be a boolean")


def _text_list(value: object, *, field_name: str) -> tuple[str, ...]:
  if value is None:
    return ()
  if isinstance(value, str):
    raw_items = (value,)
  elif isinstance(value, (list, tuple, set)):
    raw_items = tuple(value)
  else:
    raise ValueError(f"{field_name} must be a text list")
  result: list[str] = []
  seen: set[str] = set()
  for raw_item in raw_items:
    item = _clean_text(raw_item)
    if item is None or item in seen:
      continue
    seen.add(item)
    result.append(item)
  return tuple(result)


def _first_present(
  values: Mapping[str, Any],
  *field_names: str,
) -> object:
  for field_name in field_names:
    if field_name in values:
      return values[field_name]
  return None


def _is_frontmatter_delimiter(line: bytes) -> bool:
  return line.decode("utf-8").strip() == _FRONTMATTER_DELIMITER


def _decode_skill_source(source: bytes) -> str:
  return source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _read_universal_binary_line(handle: Any) -> bytes:
  line = bytearray()
  while True:
    byte = handle.read(1)
    if not byte:
      return bytes(line)
    line.extend(byte)
    if byte in {b"\r", b"\n"}:
      return bytes(line)


def _read_pinned_skill_source(path: Path) -> str | None:
  before = path.lstat()
  if not stat.S_ISREG(before.st_mode):
    raise ValueError("skill source must be a regular file")
  if path.resolve() != path:
    raise ValueError("canonical skill source path changed before read")
  with path.open("rb", buffering=0) as handle:
    after = os.fstat(handle.fileno())
    if (
      not stat.S_ISREG(after.st_mode)
      or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
    ):
      raise ValueError("skill source identity changed before read")
    first_line = _read_universal_binary_line(handle)
    captured = [first_line]
    if _is_frontmatter_delimiter(first_line):
      while True:
        line = _read_universal_binary_line(handle)
        if not line:
          raise ValueError("skill frontmatter is missing its closing delimiter")
        captured.append(line)
        if _is_frontmatter_delimiter(line):
          break
      frontmatter_source = _decode_skill_source(b"".join(captured))
      if not _is_catalog_visible(frontmatter_source):
        return None
    captured.append(handle.read())
    return _decode_skill_source(b"".join(captured))


def _raw_frontmatter(source: str) -> dict[str, Any]:
  lines = source.splitlines(keepends=True)
  if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
    return {}
  frontmatter_lines: list[str] = []
  for line in lines[1:]:
    if line.strip() != _FRONTMATTER_DELIMITER:
      frontmatter_lines.append(line)
      continue
    payload = yaml.safe_load("".join(frontmatter_lines)) or {}
    if not isinstance(payload, dict):
      raise ValueError("skill frontmatter must be a YAML mapping")
    return payload
  raise ValueError("skill frontmatter is missing its closing delimiter")


def _is_catalog_visible(source: str) -> bool:
  frontmatter = _raw_frontmatter(source)
  return _coerce_bool(
    frontmatter.get("catalog"),
    field_name="catalog",
    default=True,
  )


def _contained_skill_path(root: Path, candidate: Path) -> Path | None:
  resolved_root = root.resolve()
  resolved_candidate = candidate.resolve()
  if resolved_candidate.parent != resolved_root:
    return None
  return resolved_candidate


def _profile_metadata(profile: SkillProfile) -> Mapping[str, Any]:
  metadata = profile.metadata
  if metadata is None:
    return {}
  if not isinstance(metadata, Mapping):
    raise ValueError("profile metadata must be a mapping")
  return metadata


def _semantic_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
  semantic = metadata.get("semantic_metadata")
  if semantic is None:
    return {}
  if not isinstance(semantic, Mapping):
    raise ValueError("semantic_metadata must be a mapping")
  return semantic


def _outputs(semantic: Mapping[str, Any]) -> tuple[str, ...]:
  raw_outputs = semantic.get("output_contracts")
  if raw_outputs is None:
    return ()
  if isinstance(raw_outputs, (str, bytes)) or not isinstance(
    raw_outputs,
    (list, tuple),
  ):
    raise ValueError("semantic output_contracts must be a sequence of mappings")
  outputs: list[str] = []
  for raw_output in raw_outputs:
    if not isinstance(raw_output, Mapping):
      raise ValueError("semantic output_contracts entries must be mappings")
    owner = _clean_text(raw_output.get("owner"))
    contract_name = _clean_text(raw_output.get("contract_name"))
    schema_version = _clean_text(raw_output.get("schema_version"))
    if owner is None or contract_name is None or schema_version is None:
      raise ValueError("semantic output_contracts entries must be complete")
    outputs.append(f"{owner}:{contract_name}@{schema_version}")
  return tuple(outputs)


def _action_class(semantic: Mapping[str, Any]) -> str:
  effects = set(_text_list(
    semantic.get("allowed_effects"),
    field_name="semantic allowed_effects",
  ))
  if effects.intersection({"external_write", "irreversible"}):
    return "external"
  if effects.intersection({
    "artifact_write",
    "portfolio_config",
    "state_write",
  }):
    return "state_write"
  return "read_only"


def _approval_policy(semantic: Mapping[str, Any]) -> str:
  constraints = set(_text_list(
    semantic.get("approval_constraints"),
    field_name="semantic approval_constraints",
  ))
  if "explicit_user_approval" in constraints:
    return "explicit_user_approval"
  if "human_review_before_apply" in constraints:
    return "human_review_before_apply"
  return "runtime_policy"


def _schedule_eligible(semantic: Mapping[str, Any]) -> bool:
  scheduling = semantic.get("scheduling")
  if scheduling is None:
    return False
  if not isinstance(scheduling, Mapping):
    raise ValueError("semantic scheduling must be a mapping")
  return _clean_text(scheduling.get("eligibility")) == "eligible"


def _default_skill_label(name: str) -> str:
  return " ".join(
    part.capitalize()
    for part in name.replace("_", "-").split("-")
    if part
  )


def _project_summary(
  profile: SkillProfile,
  *,
  metadata: Mapping[str, Any],
  path: Path,
) -> ControlSkillSummary:
  name = profile.name
  if not is_control_skill_name(name):
    raise ValueError("profile name must match the control skill name grammar")
  semantic = _semantic_metadata(metadata)
  required_context = _text_list(
    metadata.get("required_context"),
    field_name="required_context",
  )
  requires_portfolio_context = _coerce_bool(
    metadata.get("requires_portfolio_context"),
    field_name="requires_portfolio_context",
  )
  if (
    requires_portfolio_context
    and not _PORTFOLIO_CONTEXT_NAMES.intersection(required_context)
  ):
    raise ValueError(
      "portfolio context requirement must be present in required_context"
    )
  blocked_reason = _clean_text(metadata.get("blocked_reason"))
  can_launch = profile.agent_callable and blocked_reason is None
  schedule_eligible = _schedule_eligible(semantic)
  return ControlSkillSummary(
    name=name,
    label=(
      _clean_text(_first_present(
        metadata,
        "label",
        "display_label",
        "title",
      ))
      or _default_skill_label(name)
    ),
    description=_clean_text(metadata.get("description")) or "",
    agent_description=profile.agent_description,
    version=profile.version or "",
    scope=profile.scope or "global",
    requires_portfolio_context=requires_portfolio_context,
    required_context=required_context,
    agent_callable=profile.agent_callable,
    resumable=profile.resumable,
    max_turns=profile.max_turns,
    max_budget_usd=profile.max_budget_usd,
    persist_state=profile.persist_state,
    typed_contract=_clean_text(metadata.get("typed_contract")),
    catalog=True,
    profiles=_text_list(
      semantic.get("allowed_profiles"),
      field_name="semantic allowed_profiles",
    ),
    modes=_text_list(
      _first_present(metadata, "modes", "compatible_modes"),
      field_name="modes",
    ),
    outputs=_outputs(semantic),
    action_class=_action_class(semantic),
    approval_policy=_approval_policy(semantic),
    tier_availability=_text_list(
      _first_present(metadata, "tier_availability", "tier"),
      field_name="tier_availability",
    ),
    credential_requirements=_text_list(
      semantic.get("credential_requirements"),
      field_name="semantic credential_requirements",
    ),
    schedule_eligible=schedule_eligible,
    can_launch=can_launch,
    can_schedule=schedule_eligible and can_launch,
    blocked_reason=blocked_reason,
    path=path.as_posix(),
  )


def _project_detail(
  summary: ControlSkillSummary,
  *,
  body: str,
) -> ControlSkillDetail:
  return ControlSkillDetail(
    name=summary.name,
    label=summary.label,
    description=summary.description,
    agent_description=summary.agent_description,
    version=summary.version,
    scope=summary.scope,
    requires_portfolio_context=summary.requires_portfolio_context,
    required_context=summary.required_context,
    agent_callable=summary.agent_callable,
    resumable=summary.resumable,
    max_turns=summary.max_turns,
    max_budget_usd=summary.max_budget_usd,
    persist_state=summary.persist_state,
    typed_contract=summary.typed_contract,
    catalog=summary.catalog,
    profiles=summary.profiles,
    modes=summary.modes,
    outputs=summary.outputs,
    action_class=summary.action_class,
    approval_policy=summary.approval_policy,
    tier_availability=summary.tier_availability,
    credential_requirements=summary.credential_requirements,
    schedule_eligible=summary.schedule_eligible,
    can_launch=summary.can_launch,
    can_schedule=summary.can_schedule,
    blocked_reason=summary.blocked_reason,
    path=summary.path,
    body=body,
  )


@dataclass(frozen=True, slots=True)
class DirectoryControlSkillCatalog:
  """Live, lazy control projection over one external skill directory."""

  _root: Path

  def __init__(self, root: str | Path) -> None:
    object.__setattr__(self, "_root", Path(root))

  def list_skills(self) -> tuple[ControlSkillSummary, ...]:
    try:
      if not self._root.exists():
        return ()
      paths = tuple(sorted(
        path
        for path in self._root.glob("*.md")
        if path.is_file() and is_control_skill_name(path.stem)
      ))
    except Exception as exc:
      raise ControlSkillUnavailableError(
        code="invalid",
        selector=None,
      ) from exc
    projected: list[ControlSkillSummary] = []
    for path in paths:
      try:
        contained_path = _contained_skill_path(self._root, path)
        if contained_path is None or not contained_path.is_file():
          continue
        source = _read_pinned_skill_source(contained_path)
        if source is None:
          continue
        profile = SkillLoader(self._root).load_source(
          source,
          path=contained_path,
        )
        metadata = _profile_metadata(profile)
        projected.append(_project_summary(
          profile,
          metadata=metadata,
          path=path,
        ))
      except Exception as exc:
        # One malformed member must not refuse the listing of every valid
        # control skill; a direct read of this selector still reports its
        # typed 'invalid' error via resolve_skill.
        log.warning(
          "control skill %r failed to project during listing; skipping"
          " it: %s",
          path.stem,
          exc,
        )
        continue
    return tuple(sorted(projected, key=lambda summary: summary.name))

  def resolve_skill(self, selector: object) -> ControlSkillDetail:
    if not is_control_skill_name(selector):
      raise ControlSkillUnavailableError(
        code="invalid_selector",
        selector=selector,
      ) from None
    path = self._root / f"{selector}.md"
    try:
      contained_path = _contained_skill_path(self._root, path)
      if contained_path is None or not contained_path.is_file():
        raise ControlSkillUnavailableError(
          code="unknown",
          selector=selector,
        ) from None
    except ControlSkillUnavailableError:
      raise
    except Exception as exc:
      raise ControlSkillUnavailableError(
        code="invalid",
        selector=selector,
      ) from exc
    try:
      source = _read_pinned_skill_source(contained_path)
      if source is None:
        raise ControlSkillUnavailableError(
          code="unknown",
          selector=selector,
        ) from None
      profile = SkillLoader(self._root).load_source(
        source,
        path=contained_path,
      )
      metadata = _profile_metadata(profile)
      summary = _project_summary(
        profile,
        metadata=metadata,
        path=path,
      )
      body = resolve_blocks(profile.system_prompt, self._root / "_blocks")
      return _project_detail(summary, body=body)
    except ControlSkillUnavailableError:
      raise
    except Exception as exc:
      raise ControlSkillUnavailableError(
        code="invalid",
        selector=selector,
      ) from exc


__all__ = ["DirectoryControlSkillCatalog"]
