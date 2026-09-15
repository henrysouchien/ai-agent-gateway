"""Exact static dispatch semantics and their legacy gateway projection.

``agent_gateway`` never imports product policy modules — every policy read goes
through :mod:`agent_gateway.policy_imports` explicit application binding — so the declaration
the dispatch boundary needs lives here, beside the boundary that reads it.

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

The immutable exact-identity map is the sole declaration source.  The existing
bare-name ``ToolDispatchDecl`` API remains a one-way compatibility projection
for current runtime readers until descriptor activation replaces it.  Bare
names are verified unique before that projection is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from agent_workflow_contracts.models import CatalogToolEffect

from agent_workflow_contracts.tool_registration import (
  PolicyKind,
  RegisteredToolIdentity,
  VersionedPolicyRef,
  validate_registered_tool_identity,
)


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


_STATUS_SUCCESS: Mapping[str, Any] = MappingProxyType(
  {"kind": "status_equals", "field": "status", "values": ("success",)}
)
_STATUS_OK: Mapping[str, Any] = MappingProxyType(
  {"kind": "status_equals", "field": "status", "values": ("ok",)}
)


def _search_hits(default_source_kind: str) -> Mapping[str, Any]:
  return MappingProxyType(
    {
      "kind": "search_hits",
      "container": "hits",
      "default_source_kind": default_source_kind,
    }
  )


def _documents(source_kind: str) -> Mapping[str, Any]:
  return MappingProxyType(
    {"kind": "documents", "container": "documents", "source_kind": source_kind}
  )


def _single_document(source_kind: str) -> Mapping[str, Any]:
  return MappingProxyType({"kind": "single_document", "source_kind": source_kind})


def _vendor_sources(provider: str) -> Mapping[str, Any]:
  return MappingProxyType({"kind": "vendor_sources", "provider": provider})


def _parser_items(container: str, default_source_kind: str) -> Mapping[str, Any]:
  return MappingProxyType(
    {
      "kind": "parser_items",
      "container": container,
      "default_source_kind": default_source_kind,
    }
  )


def _policy_ref(
  kind: PolicyKind,
  policy_id: str,
  *,
  parameters: Mapping[str, object] | None = None,
) -> VersionedPolicyRef:
  return VersionedPolicyRef(
    kind=kind,
    policy_id=policy_id,
    version="v1",
    parameters=parameters or {},
  )


_CURRENT_GENERIC_NON_ERROR = _policy_ref(
  "outcome",
  "current_generic_non_error",
)
_CURRENT_NO_GATEWAY_EXTRACTION = _policy_ref(
  "source_identity",
  "current_no_gateway_extraction",
)
_SANDBOX_COMPUTATIONS = _policy_ref(
  "source_identity",
  "sandbox_computations",
)
_FMS_COMPUTATION = _policy_ref(
  "source_identity",
  "fms-computation",
)

DEFAULT_TOOL_DISPATCH_SEMANTICS = ToolDispatchSemantics(
  idempotent=False,
  outcome_policy=_CURRENT_GENERIC_NON_ERROR,
  source_identity_policy=_CURRENT_NO_GATEWAY_EXTRACTION,
)


def _idempotent_semantics(
  *,
  success_signal: Mapping[str, Any] | None = None,
  source_identity: Mapping[str, Any] | None = None,
) -> ToolDispatchSemantics:
  outcome_policy = _CURRENT_GENERIC_NON_ERROR
  if success_signal is not None:
    if success_signal.get("kind") != "status_equals":
      raise ValueError("current success signal must be status_equals")
    outcome_policy = _policy_ref(
      "outcome",
      "current-declared-status-equals",
      parameters={
        key: value
        for key, value in success_signal.items()
        if key != "kind"
      },
    )
  source_identity_policy = _CURRENT_NO_GATEWAY_EXTRACTION
  if source_identity is not None:
    policy_id = source_identity.get("kind")
    if type(policy_id) is not str:
      raise TypeError("current source identity kind must be an exact str")
    source_identity_policy = _policy_ref(
      "source_identity",
      policy_id,
      parameters={
        key: value
        for key, value in source_identity.items()
        if key != "kind"
      },
    )
  return ToolDispatchSemantics(
    idempotent=True,
    outcome_policy=outcome_policy,
    source_identity_policy=source_identity_policy,
  )


_IDEMPOTENT_GENERIC = _idempotent_semantics()
_NON_IDEMPOTENT_SANDBOX_COMPUTATIONS = ToolDispatchSemantics(
  idempotent=False,
  outcome_policy=_CURRENT_GENERIC_NON_ERROR,
  source_identity_policy=_SANDBOX_COMPUTATIONS,
)
_IDEMPOTENT_FMS_COMPUTATION = ToolDispatchSemantics(
  idempotent=True,
  outcome_policy=_policy_ref(
    "outcome",
    "current-declared-status-equals",
    parameters={"field": "status", "values": ("ok",)},
  ),
  source_identity_policy=_FMS_COMPUTATION,
)


def _local(logical_name: str) -> RegisteredToolIdentity:
  return RegisteredToolIdentity(
    route_kind="local_handler",
    logical_name=logical_name,
  )


def _mcp(logical_server_id: str, logical_name: str) -> RegisteredToolIdentity:
  return RegisteredToolIdentity(
    route_kind="mcp",
    logical_server_id=logical_server_id,
    logical_name=logical_name,
  )


def _build_exact_dispatch_semantics_overrides(
) -> Mapping[RegisteredToolIdentity, ToolDispatchSemantics]:
  source_rows = (
    (_local("web_fetch"), MappingProxyType({"kind": "web_fetch"})),
    (_mcp("research-corpus-mcp", "filings_search"), _search_hits("filing")),
    (
      _mcp("research-corpus-mcp", "transcripts_search"),
      _search_hits("transcript"),
    ),
    (_mcp("research-corpus-mcp", "filings_list"), _documents("filing")),
    (
      _mcp("research-corpus-mcp", "transcripts_list"),
      _documents("transcript"),
    ),
    (_mcp("research-corpus-mcp", "filings_read"), _single_document("filing")),
    (
      _mcp("research-corpus-mcp", "transcripts_read"),
      _single_document("transcript"),
    ),
    (
      _mcp("research-corpus-mcp", "filings_source_excerpt"),
      _single_document("filing"),
    ),
    (
      _mcp("research-corpus-mcp", "transcripts_source_excerpt"),
      _single_document("transcript"),
    ),
    (
      _mcp("edgar-parser-mcp", "get_filings"),
      MappingProxyType({"kind": "parser_filings"}),
    ),
    (
      _mcp("edgar-parser-mcp", "get_filing_sections"),
      MappingProxyType({"kind": "parser_filing_sections"}),
    ),
    (
      _mcp("edgar-parser-mcp", "search_filing_text"),
      _parser_items("hits", "filing"),
    ),
    (
      _mcp("edgar-parser-mcp", "get_filing_evidence"),
      _parser_items("evidence", "filing_evidence"),
    ),
    (
      _mcp("edgar-parser-mcp", "cite_concept"),
      _parser_items("citations", "concept_citation"),
    ),
    (
      _mcp("edgar-parser-mcp", "get_filing_document"),
      MappingProxyType({"kind": "parser_document"}),
    ),
    (
      _mcp("edgar-parser-mcp", "get_metric"),
      MappingProxyType({"kind": "metric_citations"}),
    ),
  )
  vendor_source_rows = (
    (_mcp("market-data-mcp", "compare_peers"), "fmp", None),
    (_mcp("market-data-mcp", "fetch_financials"), "fmp", None),
    (_mcp("market-data-mcp", "fetch_company_profile"), "fmp", None),
    (_mcp("market-data-mcp", "get_economic_data"), "fmp", None),
    (_mcp("market-data-mcp", "get_estimate_revisions"), "fmp", None),
    (_mcp("market-data-mcp", "screen_estimate_revisions"), "fmp", None),
    (_mcp("market-data-mcp", "get_institutional_ownership"), "fmp", None),
    (_mcp("market-data-mcp", "get_insider_trades"), "fmp", None),
    (_mcp("market-data-mcp", "get_market_context"), "fmp", None),
    (_mcp("market-data-mcp", "get_price_performance_windows"), "fmp", None),
    (_mcp("market-data-mcp", "get_sector_overview"), "fmp", None),
    (_mcp("fred-mcp", "fred_get_multiple"), "fred", None),
    (_mcp("fred-mcp", "fred_get_series"), "fred", None),
    (_mcp("portfolio-reads-mcp", "get_positions"), "portfolio", None),
    (_mcp("portfolio-reads-mcp", "get_quote"), "portfolio", None),
    (_mcp("portfolio-reads-mcp", "get_risk_analysis"), "portfolio", None),
    (
      _mcp("portfolio-reads-mcp", "industry_peer_comparison"),
      "fmp",
      None,
    ),
    (_mcp("portfolio-reads-mcp", "run_whatif"), "portfolio", None),
    (_mcp("gsheets-mcp", "gsheets_read_range"), "gsheets", _STATUS_OK),
  )
  generic_identities = (
    _mcp("fred-mcp", "fred_list_series"),
    _mcp("fred-mcp", "fred_search"),
  )

  rows: list[tuple[RegisteredToolIdentity, ToolDispatchSemantics]] = [
    (
      identity,
      _idempotent_semantics(
        success_signal=_STATUS_SUCCESS,
        source_identity=source_identity,
      ),
    )
    for identity, source_identity in source_rows
  ]
  rows.extend(
    (
      identity,
      _idempotent_semantics(
        success_signal=success_signal,
        source_identity=_vendor_sources(provider),
      ),
    )
    for identity, provider, success_signal in vendor_source_rows
  )
  rows.extend((identity, _IDEMPOTENT_GENERIC) for identity in generic_identities)
  rows.append((
    _local("fms_compute_quantifying_risk"),
    _IDEMPOTENT_FMS_COMPUTATION,
  ))
  rows.extend(
    (identity, _NON_IDEMPOTENT_SANDBOX_COMPUTATIONS)
    for identity in (
      _local("code_execute"),
      _local("code_execute_status"),
    )
  )

  by_identity: dict[RegisteredToolIdentity, ToolDispatchSemantics] = {}
  for identity, semantics in rows:
    if identity in by_identity:
      raise RuntimeError("duplicate exact dispatch-semantics identity")
    by_identity[identity] = semantics
  if len(by_identity) != 40:
    raise RuntimeError("exact dispatch-semantics override count must remain 40")
  return MappingProxyType(by_identity)


_TOOL_DISPATCH_SEMANTICS_OVERRIDES = (
  _build_exact_dispatch_semantics_overrides()
)


def tool_dispatch_semantics_overrides(
) -> Mapping[RegisteredToolIdentity, ToolDispatchSemantics]:
  """Return the immutable sparse exact-identity override map."""

  return _TOOL_DISPATCH_SEMANTICS_OVERRIDES


def tool_dispatch_semantics_for(
  identity: RegisteredToolIdentity,
) -> ToolDispatchSemantics:
  """Return final dispatch semantics for one exact registered identity."""

  canonical = validate_registered_tool_identity(identity)
  return _TOOL_DISPATCH_SEMANTICS_OVERRIDES.get(
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
  for identity, semantics in _TOOL_DISPATCH_SEMANTICS_OVERRIDES.items():
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

  The build is deferred because ``effect`` derivation soft-imports the server
  policy module, which is not importable until the host application is up.
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
