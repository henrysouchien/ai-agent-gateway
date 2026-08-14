"""Compact verified workflow-evidence projection for parent-runner provenance.

Workflow children retain typed source observations and tool names in their
durable ``TaskResult.evidence``. The parent runner's final-answer guard sees
only the parent's flat ``tools_used`` list, so child evidence was invisible and
the parent re-did retrieval that the workflow had already performed.

This module owns the runtime-only bridge:

- the workflow result/delivery boundary builds one compact projection from the
  durable, provenance-checked node ``TaskResult`` values (never from child
  prose or transcripts);
- the projection rides the ``workflow_run`` tool result under a private key
  that the runner strips before the model or the durable log can see it;
- the runner registers the projection as provenance and merges its evidence
  tool names into the guard's view of ``tools_used``.

Agents never author any part of this projection. It is a bounded projection
over already-durable run state, not a new receipt or ledger artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY = "_workflow_evidence_projection"

_MAX_OBSERVED_SOURCES = 256
_MAX_EVIDENCE_TOOLS = 128

# Child arithmetic verification proves the child's own computation, not the
# parent's final derived arithmetic, so it never carries over as provenance.
_NON_TRANSFERABLE_EVIDENCE_TOOLS = frozenset({"code_execute"})


def build_workflow_evidence_projection(
  workflow_run_id: str,
  node_results: Sequence[Any],
) -> dict[str, Any] | None:
  """Project verified child evidence from settled workflow task results.

  Only successfully settled results contribute: those are the durable,
  provenance-verified settlements the workflow accepted. Returns ``None``
  when no evidence exists so callers attach nothing.
  """

  run_id = str(workflow_run_id or "").strip()
  if not run_id:
    return None
  evidence_tools: list[str] = []
  seen_tools: set[str] = set()
  observed_sources: list[dict[str, Any]] = []
  seen_sources: set[tuple[tuple[str, Any], ...]] = set()
  for result in node_results:
    execution = getattr(result, "execution", None)
    if getattr(execution, "status", None) != "succeeded":
      continue
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
  if not evidence_tools and not observed_sources:
    return None
  return {
    "workflow_run_id": run_id,
    "evidence_tools": evidence_tools,
    "observed_sources": observed_sources,
  }


def register_workflow_evidence_projection(
  store: dict[str, dict[str, Any]],
  payload: Any,
) -> None:
  """Record one runtime-built projection, keyed by workflow run identity.

  Invalid payloads are dropped whole: registration fails closed to "no
  provenance" rather than admitting a partially trusted record.
  """

  if not isinstance(payload, Mapping):
    return
  run_id = str(payload.get("workflow_run_id") or "").strip()
  if not run_id:
    return
  raw_tools = payload.get("evidence_tools")
  raw_sources = payload.get("observed_sources")
  if not isinstance(raw_tools, list) or not isinstance(raw_sources, list):
    return
  evidence_tools: list[str] = []
  seen_tools: set[str] = set()
  for tool_name in raw_tools[:_MAX_EVIDENCE_TOOLS]:
    name = str(tool_name or "").strip()
    if name and name not in seen_tools:
      seen_tools.add(name)
      evidence_tools.append(name)
  observed_sources = [
    dict(record)
    for record in raw_sources[:_MAX_OBSERVED_SOURCES]
    if isinstance(record, Mapping)
    and str(record.get("source_kind") or "").strip()
    and str(record.get("document_id") or "").strip()
  ]
  if not evidence_tools and not observed_sources:
    return
  store[run_id] = {
    "workflow_run_id": run_id,
    "evidence_tools": evidence_tools,
    "observed_sources": observed_sources,
  }


def guard_visible_tools_used(
  tools_used: Iterable[str],
  projections: Iterable[Mapping[str, Any]],
) -> list[str]:
  """Merge registered workflow evidence into the guard's tools view.

  The final-answer guard reasons over tool-name sets, so verified child
  retrieval provenance is exposed as additional tool names. The merge is
  order-preserving and deduplicated; non-transferable verification tools are
  excluded so the guard still demands parent-side arithmetic verification.
  """

  merged = [str(name) for name in tools_used]
  seen = set(merged)
  for projection in projections:
    raw_tools = projection.get("evidence_tools")
    for tool_name in raw_tools if isinstance(raw_tools, list) else []:
      name = str(tool_name or "").strip()
      if (
        not name
        or name in seen
        or name in _NON_TRANSFERABLE_EVIDENCE_TOOLS
      ):
        continue
      seen.add(name)
      merged.append(name)
  return merged


__all__ = [
  "WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY",
  "build_workflow_evidence_projection",
  "guard_visible_tools_used",
  "register_workflow_evidence_projection",
]
