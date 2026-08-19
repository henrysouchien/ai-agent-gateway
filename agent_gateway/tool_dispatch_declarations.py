"""The gateway-local tool dispatch declaration table.

``agent_gateway`` never statically imports ``agent.*`` — every policy read goes
through :mod:`agent_gateway.policy_imports` soft imports — so the declaration
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

End state reached (D-B1-2, B-5 stage c): ``classify_tool_outcome`` consumes
the *catalog entry* now, and this table is construction-only input to
``agent_gateway.capability_resolution.snapshot_platform_catalog`` — its one
and only reader.  There is no per-tool dispatch-time lookup here any more;
``lookup_catalog_entry`` is the dispatch boundary's accessor.  A single-source
grep gate pins this at WP9 — never two live declaration sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ToolDispatchDecl:
  """One declarative dispatch row (see the module docstring)."""

  success_signal: Mapping[str, Any] | None = None
  source_identity: Mapping[str, Any] | None = None
  effect: str | None = None
  idempotent: bool | None = None


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


def _parser_items(container: str, default_source_kind: str) -> Mapping[str, Any]:
  return MappingProxyType(
    {
      "kind": "parser_items",
      "container": container,
      "default_source_kind": default_source_kind,
    }
  )


# The recognized-source population: the corpus + parser extraction chain the
# citation envelope reads today, keyed by canonical tool name.
_SOURCE_DECLARATIONS: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {
  "web_fetch": (_STATUS_SUCCESS, MappingProxyType({"kind": "web_fetch"})),
  "filings_search": (_STATUS_SUCCESS, _search_hits("filing")),
  "transcripts_search": (_STATUS_SUCCESS, _search_hits("transcript")),
  "filings_list": (_STATUS_SUCCESS, _documents("filing")),
  "transcripts_list": (_STATUS_SUCCESS, _documents("transcript")),
  "filings_read": (_STATUS_SUCCESS, _single_document("filing")),
  "transcripts_read": (_STATUS_SUCCESS, _single_document("transcript")),
  "filings_source_excerpt": (_STATUS_SUCCESS, _single_document("filing")),
  "transcripts_source_excerpt": (_STATUS_SUCCESS, _single_document("transcript")),
  "get_filings": (_STATUS_SUCCESS, MappingProxyType({"kind": "parser_filings"})),
  "get_filing_sections": (
    _STATUS_SUCCESS,
    MappingProxyType({"kind": "parser_filing_sections"}),
  ),
  "search_filing_text": (_STATUS_SUCCESS, _parser_items("hits", "filing")),
  "get_filing_evidence": (
    _STATUS_SUCCESS,
    _parser_items("evidence", "filing_evidence"),
  ),
  "cite_concept": (_STATUS_SUCCESS, _parser_items("citations", "concept_citation")),
  "get_filing_document": (
    _STATUS_SUCCESS,
    MappingProxyType({"kind": "parser_document"}),
  ),
  "get_metric": (_STATUS_SUCCESS, MappingProxyType({"kind": "metric_citations"})),
}

# The handle-eligible vendor and computation population.  These tools mint
# their citations through the api-side ledger path (provider provenance,
# stable values, run-scoped handles) which cannot be reproduced at this
# boundary, so they declare no ``source_identity`` — but they DO belong in the
# table, because their outcome classification is what closes the 429-minting
# hole: a rate-limited payload settles ``error_rate_limited`` and never
# reaches the ``ok`` arm.
_VENDOR_TOOLS: tuple[str, ...] = (
  "compare_peers",
  "fetch_financials",
  "fetch_company_profile",
  "fred_get_multiple",
  "fred_get_series",
  "get_economic_data",
  "get_estimate_revisions",
  "screen_estimate_revisions",
  "get_institutional_ownership",
  "get_insider_trades",
  "get_market_context",
  "get_price_performance_windows",
  "get_positions",
  "get_quote",
  "get_risk_analysis",
  "get_sector_overview",
  "industry_peer_comparison",
  "run_whatif",
)

# The two tools whose success token is verified in the extraction chain today
# (``status == "ok"``); the remaining vendor tools declare no success signal
# rather than guess one.
_STATUS_OK_TOOLS: tuple[str, ...] = (
  "gsheets_read_range",
  "fms_compute_quantifying_risk",
)


def canonical_dispatch_tool_name(tool_name: str) -> str:
  """Strip the ``mcp__<server>__`` prefix the model-facing names carry."""

  name = str(tool_name or "")
  if name.startswith("mcp__"):
    parts = name.split("__", 2)
    if len(parts) == 3:
      return parts[2]
  return name


def _derive_tool_effect(tool_name: str) -> str | None:
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
  effect_resolver: Callable[[str], str | None] | None = None,
) -> Mapping[str, ToolDispatchDecl]:
  """Build the declaration table, deriving every ``effect`` column."""

  resolver = effect_resolver if effect_resolver is not None else _derive_tool_effect
  rows: dict[str, ToolDispatchDecl] = {}
  for tool_name, (success_signal, source_identity) in _SOURCE_DECLARATIONS.items():
    rows[tool_name] = ToolDispatchDecl(
      success_signal=success_signal,
      source_identity=source_identity,
      effect=resolver(tool_name),
      idempotent=True,
    )
  for tool_name in _VENDOR_TOOLS:
    rows.setdefault(
      tool_name,
      ToolDispatchDecl(
        success_signal=None,
        source_identity=None,
        effect=resolver(tool_name),
        idempotent=True,
      ),
    )
  for tool_name in _STATUS_OK_TOOLS:
    rows[tool_name] = ToolDispatchDecl(
      success_signal=_STATUS_OK,
      source_identity=None,
      effect=resolver(tool_name),
      idempotent=True,
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


def reset_tool_dispatch_declarations_cache() -> None:
  """Drop the cached table (tests and policy-module reloads)."""

  global _CACHED_DECLARATIONS
  _CACHED_DECLARATIONS = None


__all__ = [
  "ToolDispatchDecl",
  "build_tool_dispatch_declarations",
  "canonical_dispatch_tool_name",
  "reset_tool_dispatch_declarations_cache",
  "tool_dispatch_declarations",
]
