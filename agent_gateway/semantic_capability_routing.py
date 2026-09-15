"""Which semantic capability one platform tool serves (T3-I13).

Before B-8 a tool route carried no capability at all: ``SemanticToolRoute``
and every ``CatalogToolEntry`` were built with ``capability=None``, so
satisfaction was decided purely by *effect* — any ``read`` tool satisfied the
single coarse ``research-evidence.read/v1`` requirement, ``file_read``
included.  That is why a methodology could declare "I need evidence" and be
admitted by a tool that cannot possibly supply it.

This module is the one place that answers "which capability does this tool
serve".  Static registration derives from an exact registered identity first,
then its owning MCP server and intrinsic effect.  The existing bare-name
helper remains a compatibility projection for current runtime readers only.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  ToolEffect,
  ToolRegistrationContractError,
  validate_registered_tool_identity,
)


# ---------------------------------------------------------------------------
# The capability vocabulary.  ``research-evidence.read/v1`` is deliberately
# absent: it was retired at the flip, and nothing may reintroduce a capability
# that every read tool satisfies.
# ---------------------------------------------------------------------------

MARKET_DATA_READ = "market-data.read/v1"
FILINGS_READ = "filings.read/v1"
TRANSCRIPTS_READ = "transcripts.read/v1"
WEB_READ = "web.read/v1"
CORPUS_READ = "corpus.read/v1"
COMPUTATION_EXECUTE = "computation.execute/v1"
WORKSPACE_WRITE = "workspace.write/v1"
ARTIFACT_PROPOSE = "artifact.propose/v1"
STATE_MUTATE = "state.mutate/v1"

#: The dataset-shaped requirement the autonomous prefetch hook reads.  It
#: binds no live tool (its registry spec declares no compatible effect); it
#: exists so a methodology can declare *which datasets it needs warmed*
#: without the deleted ``data_requirements`` block (D-B8-2).
MARKET_DATA_HISTORY = "market-data.history/v1"


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


#: Exact registered identity -> read capability.  Identity overrides precede
#: the owning-server default and cannot leak to a same-bare-name route.
_READ_CAPABILITY_BY_IDENTITY: Mapping[RegisteredToolIdentity, str] = (
  MappingProxyType({
    _local("docs_fetch"): WEB_READ,
    _local("docs_search"): WEB_READ,
    _local("web_fetch"): WEB_READ,
    _local("web_search"): WEB_READ,
    _mcp("research-corpus-mcp", "filings_list"): FILINGS_READ,
    _mcp("research-corpus-mcp", "filings_read"): FILINGS_READ,
    _mcp("research-corpus-mcp", "filings_search"): FILINGS_READ,
    _mcp("research-corpus-mcp", "filings_source_excerpt"): FILINGS_READ,
    _mcp("market-data-mcp", "get_earnings_transcript"): TRANSCRIPTS_READ,
    _mcp("research-corpus-mcp", "transcripts_list"): TRANSCRIPTS_READ,
    _mcp("research-corpus-mcp", "transcripts_read"): TRANSCRIPTS_READ,
    _mcp("research-corpus-mcp", "transcripts_search"): TRANSCRIPTS_READ,
    _mcp("research-corpus-mcp", "transcripts_source_excerpt"): TRANSCRIPTS_READ,
    _local("file_glob"): CORPUS_READ,
    _local("file_grep"): CORPUS_READ,
    _local("file_read"): CORPUS_READ,
    _local("memory_list"): CORPUS_READ,
    _local("memory_read"): CORPUS_READ,
    _local("memory_recall"): CORPUS_READ,
    _local("fms_compute_quantifying_risk"): COMPUTATION_EXECUTE,
    _local("invoke_skill"): COMPUTATION_EXECUTE,
    _local("load_tools"): COMPUTATION_EXECUTE,
    _local("valuation_ready_batch_read"): COMPUTATION_EXECUTE,
  })
)

# Current runtime readers still pass a bare name.  This compatibility view is
# projected one way from exact overrides and is not consulted by registration.
_LEGACY_READ_CAPABILITY_BY_TOOL: Mapping[str, str] = MappingProxyType({
  identity.logical_name: capability
  for identity, capability in _READ_CAPABILITY_BY_IDENTITY.items()
})

#: MCP server -> read capability for every tool it owns that the tool map
#: above does not claim.
_READ_CAPABILITY_BY_SERVER: Mapping[str, str] = MappingProxyType({
  "edgar-parser-mcp": FILINGS_READ,
  "research-corpus-mcp": CORPUS_READ,
  "market-data-mcp": MARKET_DATA_READ,
  "fred-mcp": MARKET_DATA_READ,
  "macro-mcp": MARKET_DATA_READ,
  "positioning-mcp": MARKET_DATA_READ,
  "sheetsfinance": MARKET_DATA_READ,
  "idea-workbench-mcp": MARKET_DATA_READ,
  "model-engine": COMPUTATION_EXECUTE,
  "portfolio-reads-mcp": COMPUTATION_EXECUTE,
  "gsheets-mcp": COMPUTATION_EXECUTE,
})

#: A read route the platform cannot place in a named domain still gets a
#: capability: an unplaceable route must never fall back to "satisfies
#: anything", which is exactly the vacuous admission B-8 removes.
_DEFAULT_READ_CAPABILITY = CORPUS_READ


# Compatibility corrections that the old four-effect normalization could not
# express.  These are immutable derivation inputs, not another tool catalog.
SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT: Mapping[str, str] = MappingProxyType({
  "irreversible": STATE_MUTATE,
  "portfolio_config": STATE_MUTATE,
})
SEMANTIC_CAPABILITY_CORRECTIONS_BY_IDENTITY: Mapping[
  RegisteredToolIdentity, str
] = (
  MappingProxyType({
    _mcp("timesfm", "timesfm_forecast"): COMPUTATION_EXECUTE,
  })
)

_READ_INTRINSIC_EFFECTS = frozenset({"read", "pure_transform", "support"})
_PROPOSE_INTRINSIC_EFFECTS = frozenset({"preview", "artifact_write"})


def capability_for_tool(
  *,
  canonical_name: str,
  server_id: str | None,
  effect: str | None,
) -> str | None:
  """The exact semantic capability one route serves, or ``None``.

  ``None`` is returned only when the platform could not resolve an effect for
  the tool — an undescribable tool is never authority, so it needs no
  capability.
  """

  name = str(canonical_name or "").strip()
  if effect == "read":
    by_tool = _LEGACY_READ_CAPABILITY_BY_TOOL.get(name)
    if by_tool is not None:
      return by_tool
    if server_id is not None:
      return _READ_CAPABILITY_BY_SERVER.get(server_id, _DEFAULT_READ_CAPABILITY)
    return _DEFAULT_READ_CAPABILITY
  if effect == "propose":
    return ARTIFACT_PROPOSE
  if effect == "write":
    # A local handler writes the analyst's own workspace; a server-owned write
    # leaves it.  The two are separable authority and are kept separable.
    return WORKSPACE_WRITE if server_id is None else STATE_MUTATE
  if effect == "external_effect":
    return STATE_MUTATE
  return None


def semantic_capability_for_registration(
  identity: RegisteredToolIdentity,
  effect: ToolEffect,
) -> str:
  """Derive the existing versioned capability for static intrinsic semantics.

  The exact identity is authoritative.  Identity overrides cannot leak across
  routes that share a bare name, and this function never calls the legacy
  bare-name compatibility resolver.
  """

  canonical = validate_registered_tool_identity(identity)
  if type(effect) is not str:
    raise TypeError("effect must be an exact str")
  by_effect = SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT.get(effect)
  if by_effect is not None:
    return by_effect
  if effect in _READ_INTRINSIC_EFFECTS:
    correction = SEMANTIC_CAPABILITY_CORRECTIONS_BY_IDENTITY.get(canonical)
    if correction is not None:
      return correction
    by_identity = _READ_CAPABILITY_BY_IDENTITY.get(canonical)
    if by_identity is not None:
      return by_identity
    if canonical.route_kind == "mcp":
      assert canonical.logical_server_id is not None
      return _READ_CAPABILITY_BY_SERVER.get(
        canonical.logical_server_id,
        _DEFAULT_READ_CAPABILITY,
      )
    return _DEFAULT_READ_CAPABILITY
  if effect in _PROPOSE_INTRINSIC_EFFECTS:
    return ARTIFACT_PROPOSE
  if effect == "state_write":
    return WORKSPACE_WRITE if canonical.route_kind != "mcp" else STATE_MUTATE
  if effect == "external_write":
    return STATE_MUTATE
  raise ToolRegistrationContractError(
    f"unsupported intrinsic tool effect: {effect}"
    )


__all__ = [
  "ARTIFACT_PROPOSE",
  "COMPUTATION_EXECUTE",
  "CORPUS_READ",
  "FILINGS_READ",
  "MARKET_DATA_HISTORY",
  "MARKET_DATA_READ",
  "STATE_MUTATE",
  "SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT",
  "SEMANTIC_CAPABILITY_CORRECTIONS_BY_IDENTITY",
  "TRANSCRIPTS_READ",
  "WEB_READ",
  "WORKSPACE_WRITE",
  "capability_for_tool",
  "semantic_capability_for_registration",
]
