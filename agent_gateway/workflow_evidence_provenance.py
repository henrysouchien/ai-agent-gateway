"""Compact verified workflow-evidence projection for parent citation provenance.

Workflow children retain typed source observations and tool names in their
durable ``TaskResult.evidence``. Parent citation validation needs the observed
sources even though the workflow result keeps this runtime provenance private.

This module owns the runtime-only bridge:

- the workflow result/delivery boundary builds one compact projection from the
  durable, provenance-checked node ``TaskResult`` values (never from child
  prose or transcripts);
- the projection rides the ``workflow_run`` tool result under a private key
  that the runner strips before the model or the durable log can see it;
- the runner passes the projection to the tool-result hook so the parent source
  registry can be seeded from what the children actually read.

Agents never author any part of this projection. It is a bounded projection
over already-durable run state, not a new receipt or ledger artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY = "_workflow_evidence_projection"

_MAX_OBSERVED_SOURCES = 256
_MAX_EVIDENCE_TOOLS = 128

def collect_child_evidence(
  results: Sequence[Any],
) -> tuple[list[str], list[dict[str, Any]]]:
  """Fold durable child evidence into bounded tool names and source records.

  Every *settled* result contributes, whatever its execution status. Retrieval
  a child actually performed is a fact about the run, not a reward for
  succeeding: a node that read three filings and then failed did read them,
  and its typed source identities are exactly as durable as a succeeded
  node's. Filtering them out is what forced the parent to re-retrieve
  documents the child already had (D-B4-1).

  Ordering is stable (first observation wins) and both lists are capped.
  """

  evidence_tools: list[str] = []
  seen_tools: set[str] = set()
  observed_sources: list[dict[str, Any]] = []
  seen_sources: set[tuple[tuple[str, Any], ...]] = set()
  for result in results:
    evidence = getattr(result, "evidence", None)
    if evidence is None:
      continue
    for tool_name in getattr(evidence, "tools_used", ()) or ():
      name = str(tool_name or "").strip()
      if not name or name in seen_tools:
        continue
      if len(evidence_tools) >= _MAX_EVIDENCE_TOOLS:
        break
      seen_tools.add(name)
      evidence_tools.append(name)
    for ref in getattr(evidence, "observed_sources", ()) or ():
      if getattr(ref, "kind", None) != "observed_source":
        continue
      if len(observed_sources) >= _MAX_OBSERVED_SOURCES:
        break
      record = {
        key: value
        for key, value in (
          ("source_kind", getattr(ref, "source_kind", None)),
          ("document_id", getattr(ref, "document_id", None)),
          ("produced_by_tool", getattr(ref, "produced_by_tool", None)),
          ("source_url", getattr(ref, "source_url", None)),
          ("excerpt_handle_id", getattr(ref, "excerpt_handle_id", None)),
        )
        if value is not None
      }
      if "source_kind" not in record or "document_id" not in record:
        continue
      key = tuple(sorted(record.items()))
      if key in seen_sources:
        continue
      seen_sources.add(key)
      observed_sources.append(record)
  return evidence_tools, observed_sources


def build_workflow_evidence_projection(
  workflow_run_id: str,
  node_results: Sequence[Any],
) -> dict[str, Any] | None:
  """Project verified child evidence from settled workflow task results.

  Every settled node contributes its durable, provenance-verified evidence —
  a failed or interrupted node's retrieval happened just as much as a
  succeeded node's (D-B4-1). Returns ``None`` when no evidence exists so
  callers attach nothing.
  """

  run_id = str(workflow_run_id or "").strip()
  if not run_id:
    return None
  evidence_tools, observed_sources = collect_child_evidence(node_results)
  if not evidence_tools and not observed_sources:
    return None
  return {
    "workflow_run_id": run_id,
    "evidence_tools": evidence_tools,
    "observed_sources": observed_sources,
  }


def build_child_evidence_projection(result: Any) -> dict[str, Any] | None:
  """Project one settled child result's evidence for direct-parent delivery.

  The direct-dispatch (``run_agent``) lane has no workflow run identity, so
  the projection is the same bounded tool/source fold without the run key.
  Returns ``None`` when the child observed nothing.
  """

  evidence_tools, observed_sources = collect_child_evidence((result,))
  if not evidence_tools and not observed_sources:
    return None
  return {
    "evidence_tools": evidence_tools,
    "observed_sources": observed_sources,
  }


__all__ = [
  "WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY",
  "build_child_evidence_projection",
  "build_workflow_evidence_projection",
  "collect_child_evidence",
]
