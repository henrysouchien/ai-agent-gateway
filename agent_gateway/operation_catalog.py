"""Injected operation definitions for new Gateway admissions.

The application owns skill-source compilation.  Gateway receives only this
dependency-neutral, immutable projection and must not import or reparse the
application's source model.  Persisted execution snapshots and tool grants
remain the authority for retry and resume.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from agent_workflow_contracts import AgentOperationRef, AgentOperationSnapshot


SemanticScope = Literal["global", "ticker", "portfolio", "industry"]
StateClass = Literal[
  "producer",
  "advisor-with-decision-log",
  "advisor-no-state",
  "deprecated",
]
MutationMode = Literal[
  "read_only",
  "preview",
  "apply",
  "model_writer",
  "thesis_writer",
]
RunMode = Literal["full", "recommend"]
Effort = Literal[
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]

_SEMANTIC_SCOPES = frozenset({"global", "ticker", "portfolio", "industry"})
_STATE_CLASSES = frozenset({
  "producer",
  "advisor-with-decision-log",
  "advisor-no-state",
  "deprecated",
})
_MUTATION_MODES = frozenset({
  "read_only",
  "preview",
  "apply",
  "model_writer",
  "thesis_writer",
})
_RUN_MODES = frozenset({"full", "recommend"})
_EFFORTS = frozenset({
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
})
_WORKSPACE_SCOPES_BY_MUTATION_MODE: Mapping[str, frozenset[str]] = (
  MappingProxyType({
    "read_only": frozenset({"read_only"}),
    "preview": frozenset({"workspace_write"}),
    "apply": frozenset({"workspace_write"}),
    # Model writers that only materialize immutable artifacts legitimately use
    # the narrower workspace scope; other model/thesis writers use model scope.
    "model_writer": frozenset({"workspace_write", "model_write"}),
    "thesis_writer": frozenset({"model_write"}),
  })
)
_SERVER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_TOOL_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MCP_TOOL_ID_RE = re.compile(
  r"^mcp__(?P<server>[a-z0-9][a-z0-9.-]*)__"
  r"(?P<tool>[A-Za-z_][A-Za-z0-9_]*)$"
)
_STATE_DIR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _require_exact_bool(value: object, *, field_name: str) -> None:
  if type(value) is not bool:
    raise TypeError(f"{field_name} must be a bool")


def _require_enum(
  value: object,
  *,
  field_name: str,
  allowed: frozenset[str],
) -> None:
  if type(value) is not str or value not in allowed:
    raise ValueError(
      f"{field_name} must be one of: {', '.join(sorted(allowed))}"
    )


def _require_optional_identifier(value: object, *, field_name: str) -> None:
  if value is not None and (
    type(value) is not str or not value or value != value.strip()
  ):
    raise ValueError(f"{field_name} must be a canonical non-empty string")


def _require_optional_positive_int(value: object, *, field_name: str) -> None:
  if value is not None and (type(value) is not int or value < 1):
    raise ValueError(f"{field_name} must be a positive integer")


def _require_optional_nonnegative_int(value: object, *, field_name: str) -> None:
  if value is not None and (type(value) is not int or value < 0):
    raise ValueError(f"{field_name} must be a non-negative integer")


def _require_optional_positive_number(value: object, *, field_name: str) -> None:
  if value is None:
    return
  if (
    type(value) not in {int, float}
    or not math.isfinite(value)
    or value <= 0
  ):
    raise ValueError(f"{field_name} must be a finite positive number")


def _canonical_string_set(
  values: object,
  *,
  field_name: str,
) -> frozenset[str]:
  if isinstance(values, (str, bytes)):
    raise TypeError(f"{field_name} must be a collection of strings")
  try:
    copied = tuple(values)  # type: ignore[arg-type]
  except TypeError as exc:
    raise TypeError(f"{field_name} must be a collection of strings") from exc
  for value in copied:
    if type(value) is not str or not value or value != value.strip():
      raise ValueError(
        f"{field_name} must contain canonical non-empty strings"
      )
  if len(set(copied)) != len(copied):
    raise ValueError(f"{field_name} must contain unique strings")
  return frozenset(copied)


@dataclass(frozen=True, slots=True)
class OperationRuntimePolicy:
  """Immutable non-wire facts needed to admit one operation.

  ``exact_tool_ids`` is the operation's exact ceiling.  Server collections
  describe runtime availability and session mechanics only; they never widen
  that ceiling.
  """

  semantic_scope: SemanticScope
  state_class: StateClass
  persist_state: bool
  resumable: bool
  resume_mcp_session_reset_ok: bool
  state_dir: str | None
  mutation_mode: MutationMode
  run_mode: RunMode
  exact_tool_ids: frozenset[str]
  mcp_tools_by_server: Mapping[str, frozenset[str]]
  runtime_server_refs: frozenset[str]
  session_inject_servers: frozenset[str]
  timeout_overrides: Mapping[str, int]
  extra_excluded_tools: frozenset[str]
  tool_packs: tuple[str, ...] = ()
  tool_packs_enabled: bool = True
  model: str | None = None
  provider: str | None = None
  max_turns: int | None = None
  timeout_seconds: float | None = None
  max_tokens: int | None = None
  max_budget_usd: float | None = None
  effort: Effort | None = None
  max_retries: int | None = None
  max_structured_reads: int | None = None
  initial_message: str | None = None
  delivery_label: str | None = None

  def __post_init__(self) -> None:
    _require_enum(
      self.semantic_scope,
      field_name="semantic_scope",
      allowed=_SEMANTIC_SCOPES,
    )
    _require_enum(
      self.state_class,
      field_name="state_class",
      allowed=_STATE_CLASSES,
    )
    _require_enum(
      self.mutation_mode,
      field_name="mutation_mode",
      allowed=_MUTATION_MODES,
    )
    _require_enum(self.run_mode, field_name="run_mode", allowed=_RUN_MODES)
    if self.effort is not None:
      _require_enum(self.effort, field_name="effort", allowed=_EFFORTS)
    for field_name in (
      "persist_state",
      "resumable",
      "resume_mcp_session_reset_ok",
      "tool_packs_enabled",
    ):
      _require_exact_bool(getattr(self, field_name), field_name=field_name)
    if self.persist_state and self.state_class == "advisor-no-state":
      raise ValueError("advisor-no-state operations cannot persist state")
    if self.resumable and self.session_inject_servers:
      if not self.resume_mcp_session_reset_ok:
        raise ValueError(
          "resumable operations with session injection must allow MCP "
          "session reset"
        )

    if self.state_dir is not None:
      state_dir = self.state_dir
      if type(state_dir) is not str:
        raise TypeError("state_dir must be a string or None")
      state_path = PurePosixPath(state_dir)
      if (
        not _STATE_DIR_RE.fullmatch(state_dir)
        or state_path.is_absolute()
        or any(part in {"", ".", ".."} for part in state_path.parts)
        or state_path.as_posix() != state_dir
      ):
        raise ValueError(
          "state_dir must be a canonical relative POSIX path without traversal"
        )

    _require_optional_identifier(self.model, field_name="model")
    _require_optional_identifier(self.provider, field_name="provider")
    _require_optional_identifier(self.initial_message, field_name="initial_message")
    _require_optional_identifier(self.delivery_label, field_name="delivery_label")
    _require_optional_positive_int(self.max_turns, field_name="max_turns")
    _require_optional_positive_int(self.max_tokens, field_name="max_tokens")
    _require_optional_nonnegative_int(self.max_retries, field_name="max_retries")
    _require_optional_positive_int(
      self.max_structured_reads,
      field_name="max_structured_reads",
    )
    _require_optional_positive_number(
      self.timeout_seconds,
      field_name="timeout_seconds",
    )
    _require_optional_positive_number(
      self.max_budget_usd,
      field_name="max_budget_usd",
    )

    exact_tool_ids = _canonical_string_set(
      self.exact_tool_ids,
      field_name="exact_tool_ids",
    )
    for exact_tool_id in exact_tool_ids:
      if exact_tool_id.startswith("mcp__"):
        if _MCP_TOOL_ID_RE.fullmatch(exact_tool_id) is None:
          raise ValueError(
            f"invalid canonical MCP tool identity: {exact_tool_id!r}"
          )
      elif _TOOL_ID_RE.fullmatch(exact_tool_id) is None:
        raise ValueError(f"invalid local tool identity: {exact_tool_id!r}")

    if not isinstance(self.mcp_tools_by_server, Mapping):
      raise TypeError("mcp_tools_by_server must be a mapping")
    mcp_tools_by_server: dict[str, frozenset[str]] = {}
    for server_id, raw_tool_ids in self.mcp_tools_by_server.items():
      if type(server_id) is not str or _SERVER_ID_RE.fullmatch(server_id) is None:
        raise ValueError(f"invalid MCP server identity: {server_id!r}")
      tool_ids = _canonical_string_set(
        raw_tool_ids,
        field_name=f"mcp_tools_by_server[{server_id!r}]",
      )
      if not tool_ids:
        raise ValueError("mcp_tools_by_server entries must not be empty")
      invalid_tool_ids = sorted(
        tool_id for tool_id in tool_ids if _TOOL_ID_RE.fullmatch(tool_id) is None
      )
      if invalid_tool_ids:
        raise ValueError(
          f"invalid MCP tool identities for {server_id!r}: "
          + ", ".join(invalid_tool_ids)
        )
      mcp_tools_by_server[server_id] = tool_ids

    projected_mcp_ids = frozenset(
      f"mcp__{server_id}__{tool_id}"
      for server_id, tool_ids in mcp_tools_by_server.items()
      for tool_id in tool_ids
    )
    exact_mcp_ids = frozenset(
      tool_id for tool_id in exact_tool_ids if tool_id.startswith("mcp__")
    )
    if projected_mcp_ids != exact_mcp_ids:
      raise ValueError(
        "mcp_tools_by_server must project exactly to the MCP identities in "
        "exact_tool_ids"
      )

    runtime_server_refs = _canonical_string_set(
      self.runtime_server_refs,
      field_name="runtime_server_refs",
    )
    invalid_servers = sorted(
      server_id
      for server_id in runtime_server_refs
      if _SERVER_ID_RE.fullmatch(server_id) is None
    )
    if invalid_servers:
      raise ValueError(
        "invalid runtime server identities: " + ", ".join(invalid_servers)
      )
    exact_servers = frozenset(mcp_tools_by_server)
    if not exact_servers.issubset(runtime_server_refs):
      raise ValueError(
        "every exact MCP tool server must be present in runtime_server_refs"
      )

    session_inject_servers = _canonical_string_set(
      self.session_inject_servers,
      field_name="session_inject_servers",
    )
    if not session_inject_servers.issubset(runtime_server_refs):
      raise ValueError(
        "session_inject_servers must be a subset of runtime_server_refs"
      )

    if not isinstance(self.timeout_overrides, Mapping):
      raise TypeError("timeout_overrides must be a mapping")
    timeout_overrides: dict[str, int] = {}
    for server_id, timeout in self.timeout_overrides.items():
      if type(server_id) is not str or _SERVER_ID_RE.fullmatch(server_id) is None:
        raise ValueError(f"invalid timeout server identity: {server_id!r}")
      if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout overrides must be positive integer seconds")
      timeout_overrides[server_id] = timeout
    if not set(timeout_overrides).issubset(runtime_server_refs):
      raise ValueError("timeout override servers must be runtime server refs")

    extra_excluded_tools = _canonical_string_set(
      self.extra_excluded_tools,
      field_name="extra_excluded_tools",
    )
    for excluded_tool_id in extra_excluded_tools:
      if excluded_tool_id.startswith("mcp__"):
        if _MCP_TOOL_ID_RE.fullmatch(excluded_tool_id) is None:
          raise ValueError(
            f"invalid excluded MCP tool identity: {excluded_tool_id!r}"
          )
      elif _TOOL_ID_RE.fullmatch(excluded_tool_id) is None:
        raise ValueError(
          f"invalid excluded local tool identity: {excluded_tool_id!r}"
        )
    bare_mcp_tool_ids = frozenset(
      tool_id
      for tool_ids in mcp_tools_by_server.values()
      for tool_id in tool_ids
    )
    if (exact_tool_ids | bare_mcp_tool_ids) & extra_excluded_tools:
      raise ValueError(
        "exact_tool_ids and extra_excluded_tools must not overlap"
      )

    if isinstance(self.tool_packs, (str, bytes)):
      raise TypeError("tool_packs must be a tuple of strings")
    tool_packs = tuple(self.tool_packs)
    if tuple(sorted(set(tool_packs))) != tool_packs or any(
      type(tool_pack) is not str
      or not tool_pack
      or tool_pack != tool_pack.strip()
      for tool_pack in tool_packs
    ):
      raise ValueError("tool_packs must be sorted, unique, canonical strings")

    object.__setattr__(self, "exact_tool_ids", exact_tool_ids)
    object.__setattr__(
      self,
      "mcp_tools_by_server",
      MappingProxyType(dict(sorted(mcp_tools_by_server.items()))),
    )
    object.__setattr__(self, "runtime_server_refs", runtime_server_refs)
    object.__setattr__(self, "session_inject_servers", session_inject_servers)
    object.__setattr__(
      self,
      "timeout_overrides",
      MappingProxyType(dict(sorted(timeout_overrides.items()))),
    )
    object.__setattr__(self, "extra_excluded_tools", extra_excluded_tools)
    object.__setattr__(self, "tool_packs", tool_packs)


@dataclass(frozen=True, slots=True)
class ResolvedOperationRuntime:
  """One exact operation snapshot paired with its new-admission policy."""

  snapshot: AgentOperationSnapshot
  policy: OperationRuntimePolicy

  def __post_init__(self) -> None:
    if type(self.snapshot) is not AgentOperationSnapshot:
      raise TypeError("snapshot must be an exact AgentOperationSnapshot")
    if type(self.policy) is not OperationRuntimePolicy:
      raise TypeError("policy must be an exact OperationRuntimePolicy")
    if self.snapshot.resumable != self.policy.resumable:
      raise ValueError("snapshot and runtime policy must agree on resumability")
    allowed_workspace_scopes = _WORKSPACE_SCOPES_BY_MUTATION_MODE[
      self.policy.mutation_mode
    ]
    if self.snapshot.workspace_scope not in allowed_workspace_scopes:
      raise ValueError(
        "snapshot workspace_scope is incompatible with runtime policy "
        "mutation_mode"
      )


@runtime_checkable
class AgentOperationCatalog(Protocol):
  """Application-supplied operation catalog used only for new admissions."""

  def resolve_operation(
    self,
    selector: AgentOperationRef | Mapping[str, Any] | None,
  ) -> ResolvedOperationRuntime: ...

  def list_callable_operations_with_descriptions(
    self,
  ) -> Sequence[tuple[AgentOperationRef, str]]: ...


__all__ = [
  "AgentOperationCatalog",
  "Effort",
  "MutationMode",
  "OperationRuntimePolicy",
  "ResolvedOperationRuntime",
  "RunMode",
  "SemanticScope",
  "StateClass",
]
