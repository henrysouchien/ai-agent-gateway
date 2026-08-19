from __future__ import annotations

from copy import deepcopy
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

from agent_workflow_contracts import (
  AgentOperationRef,
  AgentOperationSnapshot,
  CatalogToolEntry,
  ContractRef,
  EvidencePort,
  OperationUnavailable,
  PlatformToolCatalog,
  SemanticCapabilityRequirement,
  canonical_json_bytes,
  sha256_digest,
)

from .capability_resolution import (
  OperationDeclaration,
  resolve_operation_authority,
)
from .canonical_json_target_lock import (
  LockedCanonicalJsonTarget,
  lock_canonical_json_path,
  read_locked_canonical_json_target,
  write_locked_canonical_json_target,
)
from ._io import (
  _atomic_write_json as _atomic_write_json,
  _read_json_object,
)
from .semantic_capability_routing import capability_for_tool
from .sub_agent_capability import (
  DelegationRoleResolutionError,
  resolve_delegation_role_capability,
  resolve_sub_agent_capability,
)
from .sub_agent_scope_receipt import _server_owned_effect
from .thinking import resolve_effort_pair

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
OperationResultMode = Literal["narrative"]
DEFAULT_OPERATION_PROJECTION_CONTRACT = "report-base-v1"
GENERIC_EXPLORE_OPERATION_NAME = "explore"
GENERIC_EXPLORE_OPERATION_VERSION = "1.0"
GENERIC_EXPLORE_TOOL_IDS = frozenset({
  "docs_fetch",
  "docs_search",
  "file_glob",
  "file_grep",
  "file_read",
  "memory_list",
  "memory_read",
  "memory_recall",
  "web_fetch",
  "web_search",
})
#: The evidence domains the generic explore ceiling can reach, declared
#: instead of inferred from the operation's name.
#:
#: Both are optional, deliberately.  The retired coarse row meant "at least one
#: read route, of any kind", which the granular vocabulary cannot state: making
#: either domain required would refuse `explore` in a parent that happens to
#: offer only the other one, and `explore` is the fallback operation of last
#: resort.  The "explore must have a live evidence route" rule keeps its own
#: enforcement in the live catalog build, which refuses an empty tool scope for
#: this operation by name.
GENERIC_EXPLORE_CAPABILITIES: tuple[tuple[str, bool], ...] = (
  ("corpus.read/v1", False),
  ("web.read/v1", False),
)
_GENERIC_EXPLORE_DESCRIPTION = (
  "Investigate a focused question with the available read-only evidence routes."
)
_GENERIC_EXPLORE_INSTRUCTIONS = (
  "You are a focused research sub-agent. Investigate the assigned objective "
  "thoroughly, use available evidence tools when the claim depends on current "
  "or external facts, identify source gaps explicitly, and return the complete "
  "substantive result in your terminal assistant message. You cannot delegate "
  "to another agent."
)
_COMMON_OPERATION_INSTRUCTIONS = (
  "You are a focused sub-agent working on behalf of another agent. Complete "
  "the admitted objective thoroughly. If evidence access fails or returns "
  "suspicious data, identify the limitation explicitly instead of silently "
  "proceeding. You cannot delegate to another agent."
)


@dataclass(frozen=True, slots=True)
class ResolvedAgentOperation:
  """One immutable executable operation plus its methodology source.

  ``SkillProfile`` remains available for non-authoritative runtime ergonomics
  such as timeouts and lifecycle labels. Execution authority is carried only
  by ``snapshot`` and the grants compiled during admission.
  """

  snapshot: AgentOperationSnapshot
  methodology_profile: "SkillProfile"


def _contract_ref(
  *,
  namespace: str,
  name: str,
  version: str,
  payload: dict[str, Any],
) -> ContractRef:
  return ContractRef(
    namespace=namespace,
    name=name,
    version=version,
    digest=sha256_digest(payload),
  )


def _operation_semantic_digest(payload: dict[str, Any]) -> str:
  canonical = deepcopy(payload)
  operation = canonical.get("operation")
  if not isinstance(operation, dict):
    raise ValueError("operation semantic payload requires an operation object")
  canonical["operation"] = {
    key: value for key, value in operation.items() if key != "digest"
  }
  return "sha256:" + hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def _workspace_scope(profile: "SkillProfile") -> Literal[
  "read_only", "workspace_write", "model_write"
]:
  mode = str(profile.mutation_mode or "read_only").strip().lower().replace("-", "_")
  if mode == "read_only":
    return "read_only"
  if mode in {"preview", "apply"}:
    return "workspace_write"
  if mode in {"model_writer", "thesis_writer"}:
    semantic = (
      getattr(profile, "metadata", {}).get("semantic_metadata")
      if isinstance(getattr(profile, "metadata", None), dict)
      else None
    )
    declared_effects = (
      frozenset(
        str(item).strip().lower()
        for item in semantic.get("allowed_effects", ())
        if str(item).strip()
      )
      if isinstance(semantic, dict)
      and isinstance(semantic.get("allowed_effects"), (list, tuple))
      else frozenset()
    )
    if (
      mode == "model_writer"
      and "artifact_write" in declared_effects
      and "state_write" not in declared_effects
      and "external_write" not in declared_effects
    ):
      # Some model workflows persist only server-owned immutable artifacts.
      # Keep their model-writer tool exclusions while compiling the narrower
      # artifact-proposal authority declared by the semantic effect contract.
      return "workspace_write"
    return "model_write"
  # Missing/unknown legacy metadata must not silently grant write authority.
  return "read_only"


def _declared_semantic_requirements(
  profile: "SkillProfile",
) -> tuple[SemanticCapabilityRequirement, ...]:
  """Read the route-independent requirements the methodology *declares*.

  Nothing is inferred any more (B-8).  There is no coarse
  ``research-evidence.read/v1`` minted from "this profile has tool refs", no
  ``data_requirements`` side channel, and no ``profile.name in {"explore",
  "verify-finding"}`` shim: a methodology that needs a capability says so.
  The server-owned capability router still decides which exact live routes
  satisfy the declaration.
  """

  semantic = (
    getattr(profile, "metadata", {}).get("semantic_metadata")
    if isinstance(getattr(profile, "metadata", None), dict)
    else None
  )
  raw_requirements = (
    semantic.get("capability_requirements")
    if isinstance(semantic, dict)
    else None
  )
  requirements: dict[str, SemanticCapabilityRequirement] = {}
  if raw_requirements is not None:
    if isinstance(raw_requirements, (str, bytes)) or not isinstance(
      raw_requirements, (list, tuple)
    ):
      raise ValueError("semantic capability_requirements must be a list")
    for raw in raw_requirements:
      if isinstance(raw, str):
        name = raw.strip()
        required = True
        binding_modes = ("live_tool",)
      elif isinstance(raw, dict):
        name = str(raw.get("name") or "").strip()
        required = raw.get("required", True)
        raw_modes = raw.get("binding_modes", ("live_tool",))
        raw_contracts = raw.get("compatible_input_contracts", ())
        if isinstance(raw_modes, str) or not isinstance(raw_modes, (list, tuple)):
          raise ValueError("semantic capability binding_modes must be a list")
        if isinstance(raw_contracts, (str, bytes)) or not isinstance(
          raw_contracts,
          (list, tuple),
        ):
          raise ValueError(
            "semantic capability compatible_input_contracts must be a list"
          )
        binding_modes = tuple(sorted(set(str(item).strip() for item in raw_modes)))
      else:
        raise ValueError("semantic capability requirements must be strings or mappings")
      if not name:
        raise ValueError("semantic capability requirement name must be non-empty")
      requirement = SemanticCapabilityRequirement(
        name=name,
        required=required,
        binding_modes=binding_modes,
        compatible_input_contracts=tuple(
          ContractRef.model_validate(item)
          for item in (
            raw_contracts if isinstance(raw, dict) else ()
          )
        ),
      )
      if name in requirements and requirements[name] != requirement:
        raise ValueError(f"conflicting semantic capability requirement: {name}")
      requirements[name] = requirement

  return tuple(requirements[name] for name in sorted(requirements))


def declared_ceiling_catalog(profile: "SkillProfile") -> PlatformToolCatalog:
  """Describe the operation's declared ceiling as a platform catalog.

  Compile time has no live routing table, so the ceiling is projected as if
  every declared tool were routable.  The columns that decide authority —
  effect and capability — come from the same server-owned derivations the live
  snapshot uses, which is what makes the compile-time check and the
  admission-time check the same computation.
  """

  metadata = getattr(profile, "metadata", None)
  semantic = (
    metadata.get("semantic_metadata") if isinstance(metadata, dict) else None
  )
  raw_refs = semantic.get("tool_refs") if isinstance(semantic, dict) else None
  entries: dict[str, CatalogToolEntry] = {}
  for raw in raw_refs if isinstance(raw_refs, (list, tuple)) else ():
    if not isinstance(raw, dict):
      continue
    tool_id = str(raw.get("tool_id") or "").strip()
    if not tool_id or tool_id in entries:
      continue
    is_local = raw.get("kind") != "mcp"
    server_id = (
      None if is_local else (str(raw.get("server_id") or "").strip() or None)
    )
    if not is_local and server_id is None:
      continue
    effect = _server_owned_effect(tool_id, server_id, is_local)
    if effect is None:
      continue
    entries[tool_id] = CatalogToolEntry(
      tool_id=tool_id,
      canonical_name=tool_id,
      effect=effect,
      server_id=server_id,
      capability=capability_for_tool(
        canonical_name=tool_id,
        server_id=server_id,
        effect=effect,
      ),
    )
  return PlatformToolCatalog(
    tools=tuple(entries[name] for name in sorted(entries))
  )


def validate_operation_tool_coherence(profile: "SkillProfile") -> None:
  """Reject a declaration its own ceiling cannot satisfy.

  This is the *resolver*, run at compile time over the declared ceiling
  instead of a live catalog (B-8).  It used to be a separate hand-written rule
  ("required live-tool capability plus empty ceiling"), which meant compile
  time and admission time could disagree about what "satisfied" means — and
  they did, because the rule never looked at effects or capabilities at all.
  One computation now answers both.

  Methodology prose stays invisible here: authority is never inferred from
  text, so a profile whose only tool-use signal is prose is indistinguishable
  from a genuinely tool-free operation and cannot be rejected deterministically.
  """

  requirements = _declared_semantic_requirements(profile)
  if not requirements:
    return
  ceiling = operation_tool_ids(profile)
  catalog = declared_ceiling_catalog(profile)
  if len(catalog.tools) != len(ceiling):
    # At least one declared tool has no resolvable effect here — either the
    # effect table is not importable in this process, or the tool is not one
    # the platform knows.  A platform that cannot fully describe the ceiling
    # cannot refute a declaration over it, so this is silence, not a verdict.
    return
  verdict = resolve_operation_authority(
    OperationDeclaration(
      operation_name=profile.name,
      grant_id=f"coherence:{profile.name}",
      workspace_scope=_workspace_scope(profile),
      required_capabilities=requirements,
      tool_ceiling=ceiling,
    ),
    catalog=catalog,
  )
  if isinstance(verdict, OperationUnavailable):
    unmet = ", ".join(
      sorted(item.capability for item in verdict.unsatisfied)
    ) or verdict.detail
    raise ValueError(
      f"Methodology '{profile.name}' declares semantic capabilities its own "
      f"declared tool ceiling cannot satisfy ({unmet}); declare the exact "
      "tools in semantic_metadata.tool_refs or remove the capability "
      "requirements"
    )


def _declared_evidence_ports(
  profile: "SkillProfile",
  required_context: tuple[str, ...],
) -> tuple[EvidencePort, ...]:
  """Read the operation's declared evidence ports from skill frontmatter.

  ``evidence_ports`` is a list of ``{name, min_selections, max_selections}``
  mappings.  It states a fact the catalog owns: this operation receives
  upstream workflow results at these named ports, at least ``min_selections``
  of them.  Tool-less operations must declare one; the live catalog refuses
  to offer a tool-less operation with no port because every admissible plan
  would starve it.
  """

  metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
  raw = metadata.get("evidence_ports")
  if raw is None:
    return ()
  if not isinstance(raw, (list, tuple)):
    raise ValueError(
      f"Methodology '{profile.name}' evidence_ports must be a list of port mappings"
    )
  ports: list[EvidencePort] = []
  for item in raw:
    if not isinstance(item, dict):
      raise ValueError(
        f"Methodology '{profile.name}' evidence_ports entries must be mappings"
      )
    ports.append(EvidencePort.model_validate(item))
  ports.sort(key=lambda port: port.name)
  names = [port.name for port in ports]
  if len(set(names)) != len(names):
    raise ValueError(
      f"Methodology '{profile.name}' evidence_ports must have unique names"
    )
  if set(names) & set(required_context):
    raise ValueError(
      f"Methodology '{profile.name}' evidence_ports cannot reuse required_context keys"
    )
  return tuple(ports)


def compile_agent_operation(
  profile: "SkillProfile",
  *,
  execution_class: str,
  resolved_instructions: str | None = None,
) -> AgentOperationSnapshot:
  """Compile one callable methodology into its canonical runnable snapshot."""

  if not profile.agent_callable:
    raise ValueError(f"Methodology '{profile.name}' is not agent-callable")
  validate_operation_tool_coherence(profile)
  version = str(profile.version or "1.0").strip()
  methodology_text = profile.system_prompt.strip()
  instructions = (resolved_instructions or methodology_text).strip()
  if not methodology_text or not instructions:
    raise ValueError(f"Methodology '{profile.name}' has empty instructions")
  prompt_text = f"{instructions}\n\n{_COMMON_OPERATION_INSTRUCTIONS}"
  methodology = _contract_ref(
    namespace="skill-methodology",
    name=profile.name,
    version=version,
    payload={
      "namespace": "skill-methodology",
      "name": profile.name,
      "version": version,
      "instructions": methodology_text,
    },
  )
  prompt = _contract_ref(
    namespace="agent-prompt",
    name=profile.name,
    version=version,
    payload={
      "namespace": "agent-prompt",
      "name": profile.name,
      "version": version,
      "instructions": prompt_text,
    },
  )
  required_capabilities = _declared_semantic_requirements(profile)
  result_modes: tuple[OperationResultMode, ...] = ("narrative",)
  projection_contracts: tuple[ContractRef, ...] = ()
  required_context: tuple[str, ...] = ()
  semantic = (
    profile.metadata.get("semantic_metadata")
    if isinstance(profile.metadata, dict)
    else None
  )
  raw_context = (
    semantic.get("required_context")
    if isinstance(semantic, dict) and "required_context" in semantic
    else (
      profile.metadata.get("required_context", ())
      if isinstance(profile.metadata, dict)
      else ()
    )
  )
  if isinstance(raw_context, (list, tuple)):
    required_context = tuple(sorted(set(
      str(item).strip() for item in raw_context if str(item).strip()
    )))
  evidence_ports = _declared_evidence_ports(profile, required_context)
  description = (
    profile.agent_description
    or (profile.metadata or {}).get("description")
    or f"Run the {profile.name} methodology."
  )
  payload = {
    "operation": {
      "namespace": "agent-operation",
      "name": profile.name,
      "version": version,
      "digest": "sha256:" + "0" * 64,
    },
    "methodology": methodology.model_dump(mode="json"),
    "prompt": prompt.model_dump(mode="json"),
    "description": str(description).strip(),
    "instructions": prompt_text,
    "execution_class": execution_class,
    "required_capabilities": [
      item.model_dump(mode="json") for item in required_capabilities
    ],
    "workspace_scope": _workspace_scope(profile),
    "required_context": list(required_context),
    **(
      {"evidence_ports": [port.model_dump(mode="json") for port in evidence_ports]}
      if evidence_ports
      else {}
    ),
    "resumable": profile.resumable,
    "result_modes": list(result_modes),
    "projection_contracts": [
      item.model_dump(mode="json") for item in projection_contracts
    ],
  }
  operation = AgentOperationRef(
    namespace="agent-operation",
    name=profile.name,
    version=version,
    digest=_operation_semantic_digest(payload),
  )
  payload["operation"] = operation.model_dump(mode="json")
  return AgentOperationSnapshot.model_validate(payload)


def generic_explore_profile() -> "SkillProfile":
  """The code-owned explore operation, with its needs *declared* (B-8).

  Its requirements used to come from the same name-based inference fallback
  the skill files relied on (`profile.name in {"explore", "verify-finding"}`).
  That fallback is gone: this profile declares the two evidence domains its
  fixed ceiling can actually reach, exactly as a skill file does.
  """

  return SkillProfile(
    name=GENERIC_EXPLORE_OPERATION_NAME,
    system_prompt=_GENERIC_EXPLORE_INSTRUCTIONS,
    version=GENERIC_EXPLORE_OPERATION_VERSION,
    agent_callable=True,
    agent_description=_GENERIC_EXPLORE_DESCRIPTION,
    resumable=False,
    mutation_mode="read_only",
    delegation_role="explore",
    metadata={
      "semantic_metadata": {
        "tool_refs": [
          {"kind": "local", "tool_id": tool_id}
          for tool_id in sorted(GENERIC_EXPLORE_TOOL_IDS)
        ],
        "capability_requirements": [
          {
            "name": name,
            "required": required,
            "binding_modes": ["live_tool"],
          }
          for name, required in GENERIC_EXPLORE_CAPABILITIES
        ],
      }
    },
  )


def operation_tool_ids(profile: "SkillProfile") -> frozenset[str]:
  """Return the operation's private declared route ceiling, never a grant."""

  metadata = getattr(profile, "metadata", None)
  semantic = (
    metadata.get("semantic_metadata")
    if isinstance(metadata, dict)
    else None
  )
  raw_refs = semantic.get("tool_refs") if isinstance(semantic, dict) else None
  if raw_refs is None:
    if getattr(profile, "name", None) == GENERIC_EXPLORE_OPERATION_NAME:
      return GENERIC_EXPLORE_TOOL_IDS
    return frozenset()
  if isinstance(raw_refs, (str, bytes)) or not isinstance(raw_refs, (list, tuple)):
    raise ValueError("semantic tool_refs must be a list")
  tool_ids: set[str] = set()
  for raw in raw_refs:
    if not isinstance(raw, dict):
      raise ValueError("semantic tool_refs entries must be mappings")
    tool_id = raw.get("tool_id")
    if type(tool_id) is not str or not tool_id or tool_id != tool_id.strip():
      raise ValueError("semantic tool_refs require canonical tool_id")
    tool_ids.add(tool_id)
  return frozenset(tool_ids)


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
  effort: str | None = None
  provider: str | None = None
  max_structured_reads: int | None = None
  delegation_role: str | None = None


def _clean_string(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value).strip()
  return text or None


def _coerce_optional_delegation_role(value: Any, *, path: Path) -> str | None:
  if value is None:
    return None
  if not isinstance(value, str):
    raise ValueError(f"{path}: 'delegation_role' must be a string")
  role = value.strip()
  if not role:
    raise ValueError(f"{path}: 'delegation_role' must be non-empty when provided")
  try:
    resolve_delegation_role_capability(role)
  except DelegationRoleResolutionError:
    log.warning(
      "%s: delegation_role %r is not mapped and will be rejected at dispatch",
      path,
      role,
    )
  return role


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
  raw_max_structured_reads = frontmatter.pop("max_structured_reads", None)
  raw_subagent_return_contract = frontmatter.pop("subagent_return_contract", None)
  raw_subagent_result_mode = frontmatter.pop("subagent_result_mode", None)
  if raw_subagent_return_contract is not None or raw_subagent_result_mode is not None:
    raise ValueError(
      f"{path}: agent completion is always a normal terminal message; "
      "subagent_return_contract/subagent_result_mode are not supported"
    )
  raw_delegation_role = frontmatter.pop("delegation_role", None)
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
  raw_effort = metadata.pop("effort", None)
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
  coerced_effort_level = resolve_effort_pair(effort=raw_effort, thinking=coerced_thinking)
  coerced_effort = coerced_effort_level.value if coerced_effort_level is not None else None
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
  coerced_max_structured_reads = _coerce_optional_int(
    raw_max_structured_reads,
    field_name="max_structured_reads",
    path=path,
  )
  if coerced_max_structured_reads is not None and coerced_max_structured_reads < 1:
    raise ValueError(f"{path}: 'max_structured_reads' must be greater than or equal to 1")

  for key, coerced in [
    ("mcp_servers", coerced_mcp_servers),
    ("mcp_tools", coerced_mcp_tools),
    ("session_inject_servers", coerced_session_inject_servers),
    ("timeout_overrides", coerced_timeout_overrides),
    ("state_dir", coerced_state_dir),
    ("max_budget_usd", coerced_max_budget_usd),
    ("thinking", coerced_thinking),
    ("effort", coerced_effort),
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
    state_class=coerced_state_class,
    mutation_mode=_clean_string(raw_mutation_mode),
    effort=coerced_effort,
    provider=_clean_string(raw_provider),
    max_structured_reads=coerced_max_structured_reads,
    delegation_role=_coerce_optional_delegation_role(
      raw_delegation_role,
      path=path,
    ),
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
    path = self._skill_path(name)
    if not path.exists():
      available = self.list_skills()
      available_label = ", ".join(available) if available else "(none)"
      raise FileNotFoundError(f"Skill '{name}' not found. Available: {available_label}")
    profile = parse_skill_file(path)
    _warn_agent_description_if_needed(profile)
    return profile

  def _load_operation_profile(self, name: str) -> SkillProfile:
    if name == GENERIC_EXPLORE_OPERATION_NAME and not self.exists(name):
      return generic_explore_profile()
    return self.load(name)

  def resolve_operation(
    self,
    operation: AgentOperationRef | dict[str, Any] | None,
  ) -> ResolvedAgentOperation:
    """Resolve a public full operation identity to immutable server bytes.

    Omitting the selector is an explicit request for the registered generic
    ``explore`` operation. A bare skill name is intentionally not accepted.
    """

    if operation is None:
      requested: AgentOperationRef | None = None
      operation_name = GENERIC_EXPLORE_OPERATION_NAME
    else:
      try:
        requested = AgentOperationRef.model_validate(operation)
      except Exception as exc:
        raise ValueError("operation must be a full AgentOperationRef") from exc
      operation_name = requested.name
    profile = self._load_operation_profile(operation_name)
    if not profile.agent_callable:
      raise ValueError(
        f"Operation methodology '{operation_name}' is not agent-callable"
      )
    execution_class = resolve_delegation_role_capability(
      profile.delegation_role
    ) if profile.delegation_role is not None else resolve_sub_agent_capability(
      validated_agent_name=profile.name,
      mutation_mode=profile.mutation_mode,
      delegation_role=None,
    )
    instructions = resolve_blocks(
      profile.system_prompt,
      self.skills_dir / "_blocks",
    )
    snapshot = compile_agent_operation(
      profile,
      execution_class=execution_class,
      resolved_instructions=instructions,
    )
    if requested is not None and requested != snapshot.operation:
      raise ValueError(
        "operation identity/version/digest does not match the registered snapshot"
      )
    return ResolvedAgentOperation(
      snapshot=snapshot,
      methodology_profile=profile,
    )

  def list_callable_operations(
    self,
  ) -> list[ResolvedAgentOperation]:
    names = set(self.list_skills())
    names.add(GENERIC_EXPLORE_OPERATION_NAME)
    operations: list[ResolvedAgentOperation] = []
    for name in sorted(names):
      try:
        profile = self._load_operation_profile(name)
        if not profile.agent_callable:
          continue
        resolved = self.resolve_operation(
          compile_agent_operation(
            profile,
            execution_class=resolve_sub_agent_capability(
              validated_agent_name=profile.name,
              mutation_mode=profile.mutation_mode,
              delegation_role=profile.delegation_role,
            ),
            resolved_instructions=resolve_blocks(
              profile.system_prompt,
              self.skills_dir / "_blocks",
            ),
          ).operation
        )
      except (FileNotFoundError, TypeError, ValueError):
        log.warning("Failed to compile callable operation %s", name, exc_info=True)
        continue
      operations.append(resolved)
    return operations

  def list_callable_operations_with_descriptions(
    self,
  ) -> list[tuple[AgentOperationRef, str]]:
    return [
      (
        item.snapshot.operation,
        _cap_agent_description(item.snapshot.description),
      )
      for item in self.list_callable_operations()
    ]

  def list_skills(self) -> list[str]:
    if not self.skills_dir.exists():
      return []
    names = [path.stem for path in self.skills_dir.glob("*.md") if path.is_file()]
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

  def _read_all_for_mutation(
    self,
    target: LockedCanonicalJsonTarget,
  ) -> tuple[dict[str, Any], bytes | None, bool]:
    snapshot = read_locked_canonical_json_target(target)
    if not snapshot.exists:
      return {}, None, True
    if snapshot.parse_error is not None:
      log.warning(
        "Failed to parse state file %s: %s",
        self.state_file,
        snapshot.parse_error,
      )
      return {}, snapshot.raw, False
    payload = snapshot.value
    if not isinstance(payload, dict):
      log.warning(
        "State file %s is not a JSON object",
        self.state_file,
      )
      return {}, snapshot.raw, False
    return payload, snapshot.raw, True

  @staticmethod
  def _write_backup_once(
    *,
    target: LockedCanonicalJsonTarget,
    backup_name: str,
    raw: bytes,
  ) -> None:
    if (
      not backup_name
      or backup_name in {".", ".."}
      or os.path.basename(backup_name) != backup_name
    ):
      raise ValueError("backup path must be adjacent to the state file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
      fd = os.open(
        backup_name,
        flags,
        0o600,
        dir_fd=target.parent_fd,
      )
    except FileExistsError:
      return
    try:
      os.set_inheritable(fd, False)
      offset = 0
      while offset < len(raw):
        written = os.write(fd, raw[offset:])
        if written <= 0:
          raise OSError("short write while backing up skill state")
        offset += written
      os.fsync(fd)
    finally:
      os.close(fd)
    os.fsync(target.parent_fd)

  def mutate(
    self,
    mutation: Callable[
      [dict[str, Any]],
      dict[str, Any],
    ],
    *,
    normalize: Callable[
      [dict[str, Any]],
      dict[str, Any],
    ] | None = None,
    backup_path: str | Path | None = None,
  ) -> dict[str, Any]:
    """Atomically read, transform, and replace the complete state payload."""

    if not callable(mutation):
      raise TypeError("mutation must be callable")
    if normalize is not None and not callable(normalize):
      raise TypeError("normalize must be callable")
    backup = Path(backup_path) if backup_path is not None else None
    if backup is not None and backup.parent != self.state_file.parent:
      raise ValueError("backup path must be adjacent to the state file")

    with lock_canonical_json_path(
      self.state_file,
      create_parents=True,
    ) as target:
      payload, raw, source_is_object = (
        self._read_all_for_mutation(target)
      )
      original = deepcopy(payload)
      if normalize is not None:
        payload = normalize(deepcopy(payload))
        if not isinstance(payload, dict):
          raise TypeError("normalize must return a dict")
      requires_backup = raw is not None and (
        not source_is_object or payload != original
      )
      if backup is not None and requires_backup:
        self._write_backup_once(
          target=target,
          backup_name=backup.name,
          raw=raw,
        )
      replacement = mutation(deepcopy(payload))
      if not isinstance(replacement, dict):
        raise TypeError("mutation must return a dict")
      write_locked_canonical_json_target(
        target,
        replacement,
      )
      return deepcopy(replacement)

  def update(
    self,
    skill_name: str,
    mutation: Callable[
      [dict[str, Any]],
      dict[str, Any],
    ],
  ) -> dict[str, Any]:
    """Atomically update one skill while preserving every peer entry."""

    normalized_name = str(skill_name)
    if not callable(mutation):
      raise TypeError("mutation must be callable")
    updated: dict[str, Any] | None = None

    def _mutate(payload: dict[str, Any]) -> dict[str, Any]:
      nonlocal updated
      previous = payload.get(normalized_name)
      previous_state = (
        dict(previous) if isinstance(previous, dict) else {}
      )
      replacement = mutation(previous_state)
      if not isinstance(replacement, dict):
        raise TypeError("skill state mutation must return a dict")
      updated = dict(replacement)
      payload[normalized_name] = dict(replacement)
      return payload

    self.mutate(_mutate)
    if updated is None:
      raise RuntimeError("skill state mutation did not produce a result")
    return dict(updated)

  def get(self, skill_name: str) -> dict[str, Any]:
    payload = self._read_all()
    state = payload.get(skill_name)
    if isinstance(state, dict):
      return dict(state)
    return {}

  def set(self, skill_name: str, state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
      raise TypeError("state must be a dict")
    replacement = dict(state)
    self.update(
      skill_name,
      lambda _previous: dict(replacement),
    )

  def clear(self, skill_name: str) -> None:
    normalized_name = str(skill_name)

    def _clear(payload: dict[str, Any]) -> dict[str, Any]:
      payload.pop(normalized_name, None)
      return payload

    self.mutate(_clear)


__all__ = [
  "AGENT_DESCRIPTION_MAX_CHARS",
  "AGENT_DESCRIPTION_PLACEHOLDER",
  "GENERIC_EXPLORE_CAPABILITIES",
  "GENERIC_EXPLORE_OPERATION_NAME",
  "GENERIC_EXPLORE_TOOL_IDS",
  "Mode",
  "SkillLoader",
  "SkillProfile",
  "SkillStateStore",
  "parse_skill_file",
  "operation_tool_ids",
  "resolve_blocks",
  "validate_operation_tool_coherence",
]
