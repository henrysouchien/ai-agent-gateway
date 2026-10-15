"""Dispatch-semantics value types and the bare-name dispatch projection.

``agent_gateway`` never imports product policy modules — every policy read goes
through :mod:`agent_gateway.policy_imports` explicit application binding — and
it owns no per-tool list: the dispatch semantics of each exact registered
route are declared by that route's registration owner and read from the bound
policy module (``get_dispatch_semantics_overrides``).

Each row declares, for one tool:

``success_signal``
  What a successful payload looks like, as a descriptor (never a callable).
  ``None`` means the tool declares no signal: a non-error result classifies
  ``ok`` (D-B1-5).

``source_identity``
  A declarative descriptor of the source identities a successful payload
  carries, interpreted by
  :mod:`agent_gateway.tool_dispatch_source_identity`.  ``None`` means the tool
  contributes no source identities at this boundary.

``effect``
  **Derived**, never restated: resolved from the existing per-tool effect
  table (``agent.shared.server_policies.get_local_tool_effect`` for local
  tools, the server policy tool class otherwise) and normalized by the same
  ``_normalized_effect`` the operation-admission receipt uses.  A tool whose
  effect cannot be resolved carries ``None`` and is never retried.

``idempotent``
  Whether repeating the call is known to be safe.  ``None`` means unknown;
  retry eligibility requires ``idempotent is not False`` *and* ``effect ==
  "read"``.

The exact-identity declarations the bound policy module returns are the sole
declaration source.  The existing bare-name ``ToolDispatchDecl`` API remains a
one-way compatibility projection for current runtime readers until descriptor
activation replaces it.  Bare names are verified unique before that projection
is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from agent_workflow_contracts.models import CatalogToolEffect
from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  VersionedPolicyRef,
  validate_registered_tool_identity,
)

from .policy_imports import load_server_policy_module


@dataclass(frozen=True, slots=True)
class ToolDispatchDecl:
  """One declarative dispatch row (see the module docstring)."""

  success_signal: Mapping[str, Any] | None = None
  source_identity: Mapping[str, Any] | None = None
  effect: CatalogToolEffect | None = None
  idempotent: bool | None = None


@dataclass(frozen=True, slots=True)
class ToolDispatchSemantics:
  """Final dispatch semantics for one exact registered route."""

  idempotent: bool
  outcome_policy: VersionedPolicyRef
  source_identity_policy: VersionedPolicyRef

  def __post_init__(self) -> None:
    if type(self.idempotent) is not bool:
      raise TypeError("idempotent must be an exact bool")
    if type(self.outcome_policy) is not VersionedPolicyRef:
      raise TypeError("outcome_policy must be an exact VersionedPolicyRef")
    if self.outcome_policy.kind != "outcome":
      raise ValueError("outcome_policy has the wrong policy kind")
    if type(self.source_identity_policy) is not VersionedPolicyRef:
      raise TypeError(
        "source_identity_policy must be an exact VersionedPolicyRef"
      )
    if self.source_identity_policy.kind != "source_identity":
      raise ValueError("source_identity_policy has the wrong policy kind")


DEFAULT_TOOL_DISPATCH_SEMANTICS = ToolDispatchSemantics(
  idempotent=False,
  outcome_policy=VersionedPolicyRef(
    kind="outcome",
    policy_id="current_generic_non_error",
    version="v1",
    parameters={},
  ),
  source_identity_policy=VersionedPolicyRef(
    kind="source_identity",
    policy_id="current_no_gateway_extraction",
    version="v1",
    parameters={},
  ),
)


def tool_dispatch_semantics_overrides(
) -> Mapping[RegisteredToolIdentity, ToolDispatchSemantics]:
  """Return the exact-identity semantics the bound policy module declares.

  Empty when no policy is bound: the package has no product tools of its own.
  """

  policy = load_server_policy_module()
  resolve: Callable[
    [],
    Mapping[RegisteredToolIdentity, ToolDispatchSemantics],
  ] | None = (
    getattr(policy, "get_dispatch_semantics_overrides", None)
    if policy is not None
    else None
  )
  if resolve is None:
    return MappingProxyType({})
  return resolve()


def tool_dispatch_semantics_for(
  identity: RegisteredToolIdentity,
) -> ToolDispatchSemantics:
  """Return final dispatch semantics for one exact registered identity."""

  canonical = validate_registered_tool_identity(identity)
  return tool_dispatch_semantics_overrides().get(
    canonical,
    DEFAULT_TOOL_DISPATCH_SEMANTICS,
  )


def canonical_dispatch_tool_name(tool_name: str) -> str:
  """Strip the ``mcp__<server>__`` prefix the model-facing names carry."""

  name = str(tool_name or "")
  if name.startswith("mcp__"):
    parts = name.split("__", 2)
    if len(parts) == 3:
      return parts[2]
  return name


def _derive_tool_effect(tool_name: str) -> CatalogToolEffect | None:
  """Derive the effect from the existing per-tool effect table.

  Never restates an effect: local tools resolve through
  ``get_local_tool_effect`` and server-owned tools through the server policy
  tool class, both normalized by the operation-admission receipt's own
  ``_normalized_effect``.
  """

  from .sub_agent_scope_receipt import _normalized_effect
  from .policy_imports import (
    load_server_policy_module,
    resolve_server_policy_tool_class,
  )

  policy = load_server_policy_module()
  get_local_effect = (
    getattr(policy, "get_local_tool_effect", None) if policy is not None else None
  )
  raw = get_local_effect(tool_name) if callable(get_local_effect) else None
  if raw:
    return _normalized_effect(raw)
  return _normalized_effect(
    resolve_server_policy_tool_class(tool_name, default="")
  )


def build_tool_dispatch_declarations(
  *,
  effect_resolver: Callable[[str], CatalogToolEffect | None] | None = None,
) -> Mapping[str, ToolDispatchDecl]:
  """Project exact semantics to the current bare-name declaration API."""

  resolver = effect_resolver if effect_resolver is not None else _derive_tool_effect
  rows: dict[str, ToolDispatchDecl] = {}
  for identity, semantics in tool_dispatch_semantics_overrides().items():
    tool_name = identity.logical_name
    if tool_name in rows:
      raise RuntimeError("dispatch compatibility names must be unique")
    success_signal: Mapping[str, object] | None = None
    if semantics.outcome_policy.policy_id == "current-declared-status-equals":
      success_signal = MappingProxyType({
        "kind": "status_equals",
        **semantics.outcome_policy.parameters,
      })
    source_identity: Mapping[str, object] | None = None
    if (
      semantics.source_identity_policy.policy_id
      not in {
        "current_no_gateway_extraction",
        "vendor_sources",
      }
    ):
      source_identity = MappingProxyType({
        "kind": semantics.source_identity_policy.policy_id,
        **semantics.source_identity_policy.parameters,
      })
    rows[tool_name] = ToolDispatchDecl(
      success_signal=success_signal,
      source_identity=source_identity,
      effect=resolver(tool_name),
      idempotent=semantics.idempotent,
    )
  return MappingProxyType(rows)


_CACHED_DECLARATIONS: Mapping[str, ToolDispatchDecl] | None = None


def tool_dispatch_declarations() -> Mapping[str, ToolDispatchDecl]:
  """Return the process-wide declaration table, built on first use.

  The build is deferred because the rows and the ``effect`` derivation both
  read the bound server policy module, which is not bound until the host
  application is up.
  """

  global _CACHED_DECLARATIONS
  if _CACHED_DECLARATIONS is not None:
    return _CACHED_DECLARATIONS
  table = build_tool_dispatch_declarations()
  # A table whose every effect is unresolved means the policy module was not
  # importable yet; do not freeze that answer into the process.
  if any(row.effect is not None for row in table.values()):
    _CACHED_DECLARATIONS = table
  return table


__all__ = [
  "DEFAULT_TOOL_DISPATCH_SEMANTICS",
  "ToolDispatchDecl",
  "ToolDispatchSemantics",
  "build_tool_dispatch_declarations",
  "canonical_dispatch_tool_name",
  "tool_dispatch_semantics_for",
  "tool_dispatch_semantics_overrides",
  "tool_dispatch_declarations",
]
