"""Which semantic capability one platform tool serves (T3-I13).

Before B-8 a tool route carried no capability at all: ``SemanticToolRoute``
and every ``CatalogToolEntry`` were built with ``capability=None``, so
satisfaction was decided purely by *effect* — any ``read`` tool satisfied the
single coarse ``research-evidence.read/v1`` requirement, ``file_read``
included.  That is why a methodology could declare "I need evidence" and be
admitted by a tool that cannot possibly supply it.

This module is the one place that answers "which capability does this tool
serve".  It is a **derivation**, not a second declaration table: the effect
comes from the server-owned effect table (via the caller), and the domain
comes from the route the platform already knows the tool by — its canonical
name first, then its owning MCP server.  Both the live catalog snapshot and
the skill-metadata codegen read this one function, so a skill's declared
requirement and the route that satisfies it can never be derived from two
different opinions.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


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


#: Canonical tool name -> read capability.  Checked before the server map so a
#: filings/transcripts route keeps its domain no matter which server hosts it.
_READ_CAPABILITY_BY_TOOL: Mapping[str, str] = MappingProxyType({
  "docs_fetch": WEB_READ,
  "docs_search": WEB_READ,
  "web_fetch": WEB_READ,
  "web_search": WEB_READ,
  "filings_list": FILINGS_READ,
  "filings_read": FILINGS_READ,
  "filings_search": FILINGS_READ,
  "filings_source_excerpt": FILINGS_READ,
  "get_earnings_transcript": TRANSCRIPTS_READ,
  "transcripts_list": TRANSCRIPTS_READ,
  "transcripts_read": TRANSCRIPTS_READ,
  "transcripts_search": TRANSCRIPTS_READ,
  "transcripts_source_excerpt": TRANSCRIPTS_READ,
  "file_glob": CORPUS_READ,
  "file_grep": CORPUS_READ,
  "file_read": CORPUS_READ,
  "memory_list": CORPUS_READ,
  "memory_read": CORPUS_READ,
  "memory_recall": CORPUS_READ,
  "fms_compute_quantifying_risk": COMPUTATION_EXECUTE,
  "invoke_skill": COMPUTATION_EXECUTE,
  "load_tools": COMPUTATION_EXECUTE,
  "valuation_ready_batch_read": COMPUTATION_EXECUTE,
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
    by_tool = _READ_CAPABILITY_BY_TOOL.get(name)
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


__all__ = [
  "ARTIFACT_PROPOSE",
  "COMPUTATION_EXECUTE",
  "CORPUS_READ",
  "FILINGS_READ",
  "MARKET_DATA_HISTORY",
  "MARKET_DATA_READ",
  "STATE_MUTATE",
  "TRANSCRIPTS_READ",
  "WEB_READ",
  "WORKSPACE_WRITE",
  "capability_for_tool",
]
