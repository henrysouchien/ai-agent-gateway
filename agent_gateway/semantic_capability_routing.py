"""Which semantic capability one platform tool serves (T3-I13).

Before B-8 a tool route carried no capability at all: ``SemanticToolRoute``
and every ``CatalogToolEntry`` were built with ``capability=None``, so
satisfaction was decided purely by *effect* — any ``read`` tool satisfied the
single coarse ``research-evidence.read/v1`` requirement, ``file_read``
included.  That is why a methodology could declare "I need evidence" and be
admitted by a tool that cannot possibly supply it.

A read route's capability is the one its registration owner declares (the
server policy for an MCP route, the local-effect policy for a local handler);
a write route's capability follows from its intrinsic effect.  This module
holds the vocabulary and that derivation, and no per-tool or per-server list.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  ToolEffect,
  ToolRegistrationContractError,
  validate_registered_tool_identity,
)

from .policy_imports import load_server_policy_module


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


#: A read route whose owner declares no capability still gets one: an
#: unplaceable route must never fall back to "satisfies anything", which is
#: exactly the vacuous admission B-8 removes.
_DEFAULT_READ_CAPABILITY = CORPUS_READ


# Compatibility corrections that the old four-effect normalization could not
# express.  These are immutable derivation inputs, not another tool catalog.
SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT: Mapping[str, str] = MappingProxyType({
  "irreversible": STATE_MUTATE,
  "portfolio_config": STATE_MUTATE,
})

_READ_INTRINSIC_EFFECTS = frozenset({"read", "pure_transform", "support"})
_PROPOSE_INTRINSIC_EFFECTS = frozenset({"preview", "artifact_write"})


def _declared_read_capability(name: str, server_id: str | None) -> str | None:
  """The read capability the bound registration owner declares, if any."""

  policy = load_server_policy_module()
  resolve: Callable[[str, str | None], str | None] | None = (
    getattr(policy, "get_read_capability", None)
    if policy is not None
    else None
  )
  return resolve(name, server_id) if resolve is not None else None


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
    return _declared_read_capability(name, server_id) or _DEFAULT_READ_CAPABILITY
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
  *,
  read_capability: str | None,
) -> str:
  """Derive the versioned capability for static intrinsic semantics.

  ``read_capability`` is the registration owner's declaration for this exact
  identity; it decides every read-like effect.  Write-like effects derive
  from the effect itself.
  """

  canonical = validate_registered_tool_identity(identity)
  if type(effect) is not str:
    raise TypeError("effect must be an exact str")
  by_effect = SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT.get(effect)
  if by_effect is not None:
    return by_effect
  if effect in _READ_INTRINSIC_EFFECTS:
    return read_capability or _DEFAULT_READ_CAPABILITY
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
  "TRANSCRIPTS_READ",
  "WEB_READ",
  "WORKSPACE_WRITE",
  "capability_for_tool",
  "semantic_capability_for_registration",
]
