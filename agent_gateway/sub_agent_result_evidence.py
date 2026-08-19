from __future__ import annotations

from collections import ChainMap
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .sub_agent_result_contract import (
  child_evidence_fits_externalization_bound,
)
from .tool_dispatch_classification import OUTCOME_OK


_EVIDENCE_ADMISSION_WARNING = (
  "child evidence exceeded the canonical admission contract"
)


class UsageEvidenceMergeError(ValueError):
  """Raised when numeric usage evidence cannot be merged deterministically."""


@dataclass(frozen=True)
class SubAgentResultEvidence:
  """Side data retained across one or more child-run incarnations."""

  usage: dict[str, Any]
  tools_used: tuple[str, ...]
  fms_results: tuple[Mapping[str, Any], ...]
  artifact_events: tuple[Mapping[str, Any], ...]
  warning_parts: tuple[str, ...]
  admission_rejected: bool = False
  observed_sources: tuple[Mapping[str, Any], ...] = ()

  @classmethod
  def empty(cls) -> SubAgentResultEvidence:
    return cls(
      usage={},
      tools_used=(),
      fms_results=(),
      artifact_events=(),
      warning_parts=(),
      admission_rejected=False,
      observed_sources=(),
    )


def _rejected_evidence(
  warning_parts: Iterable[str] = (),
) -> SubAgentResultEvidence:
  warnings = tuple(warning_parts)
  if _EVIDENCE_ADMISSION_WARNING not in warnings:
    warnings = (*warnings, _EVIDENCE_ADMISSION_WARNING)
  return SubAgentResultEvidence(
    usage={},
    tools_used=(),
    fms_results=(),
    artifact_events=(),
    warning_parts=warnings,
    admission_rejected=True,
    observed_sources=(),
  )


def _observed_source_record(
  *,
  source_kind: Any,
  document_id: Any,
  produced_by_tool: Any,
  source_url: Any,
  excerpt_handle_id: Any,
) -> dict[str, Any] | None:
  """Build one typed observed-source record; reject malformed identities."""

  kind_text = str(source_kind or "").strip()
  document_text = str(document_id or "").strip()
  if not kind_text or not document_text:
    return None
  record: dict[str, Any] = {
    "kind": "observed_source",
    "source_kind": kind_text,
    "document_id": document_text,
  }
  tool_text = str(produced_by_tool or "").strip()
  if tool_text:
    record["produced_by_tool"] = tool_text
  url_text = str(source_url or "").strip()
  if url_text:
    record["source_url"] = url_text
  handle_text = str(excerpt_handle_id or "").strip()
  if handle_text:
    record["excerpt_handle_id"] = handle_text
  return record


def _records_from_source_envelope(
  block: Mapping[str, Any],
) -> list[dict[str, Any]]:
  envelope_tool = block.get("tool_name")
  raw_handles = block.get("excerpt_handles_for_call")
  handles = [
    handle
    for handle in (raw_handles if isinstance(raw_handles, list) else [])
    if isinstance(handle, Mapping)
  ]
  handle_by_document = {
    str(handle.get("document_id") or "").strip(): handle
    for handle in handles
    if str(handle.get("document_id") or "").strip()
  }
  records: list[dict[str, Any]] = []
  matched_handle_documents: set[str] = set()
  raw_sources = block.get("sources_for_call")
  for source in raw_sources if isinstance(raw_sources, list) else []:
    if not isinstance(source, Mapping):
      continue
    document_text = str(source.get("document_id") or "").strip()
    handle = handle_by_document.get(document_text)
    if handle is not None:
      matched_handle_documents.add(document_text)
    record = _observed_source_record(
      source_kind=source.get("source_kind"),
      document_id=source.get("document_id"),
      produced_by_tool=source.get("produced_by_tool") or envelope_tool,
      source_url=source.get("source_url"),
      excerpt_handle_id=handle.get("handle_id") if handle is not None else None,
    )
    if record is not None:
      records.append(record)
  for handle in handles:
    document_text = str(handle.get("document_id") or "").strip()
    if document_text in matched_handle_documents:
      continue
    record = _observed_source_record(
      source_kind=handle.get("source_kind") or handle.get("handle_class"),
      document_id=handle.get("document_id"),
      produced_by_tool=envelope_tool,
      source_url=None,
      excerpt_handle_id=handle.get("handle_id"),
    )
    if record is not None:
      records.append(record)
  return records


def _records_from_source_observation(
  block: Mapping[str, Any],
) -> list[dict[str, Any]]:
  observation_tool = block.get("tool_name")
  raw_observed = block.get("observed_sources")
  records: list[dict[str, Any]] = []
  for observed in raw_observed if isinstance(raw_observed, list) else []:
    if not isinstance(observed, Mapping):
      continue
    record = _observed_source_record(
      source_kind=observed.get("source_kind"),
      document_id=observed.get("document_id"),
      produced_by_tool=observed.get("produced_by_tool") or observation_tool,
      source_url=observed.get("source_url"),
      excerpt_handle_id=None,
    )
    if record is not None:
      records.append(record)
  return records


def _dedup_observed_source_records(
  records: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
  deduped: list[Mapping[str, Any]] = []
  seen: set[tuple[tuple[str, Any], ...]] = set()
  for record in records:
    key = tuple(sorted((str(name), value) for name, value in record.items()))
    if key in seen:
      continue
    seen.add(key)
    deduped.append(record)
  return tuple(deduped)


def _records_from_dispatch(event: Mapping[str, Any]) -> list[dict[str, Any]]:
  """Read the settled dispatch record's source identities off one event."""

  dispatch = event.get("dispatch")
  if not isinstance(dispatch, Mapping):
    return []
  raw_sources = dispatch.get("sources")
  event_tool = event.get("tool_name")
  records: list[dict[str, Any]] = []
  for source in raw_sources if isinstance(raw_sources, list) else []:
    if not isinstance(source, Mapping):
      continue
    record = _observed_source_record(
      source_kind=source.get("source_kind"),
      document_id=source.get("document_id"),
      produced_by_tool=source.get("produced_by_tool") or event_tool,
      source_url=source.get("source_url"),
      excerpt_handle_id=source.get("excerpt_handle_id"),
    )
    if record is not None:
      records.append(record)
  return records


def _records_from_result_blocks(event: Mapping[str, Any]) -> list[dict[str, Any]]:
  blocks = event.get("final_tool_result_blocks")
  if not isinstance(blocks, list):
    return []
  records: list[dict[str, Any]] = []
  for block in blocks:
    if not isinstance(block, Mapping):
      continue
    block_type = block.get("type")
    if block_type == "source_envelope":
      records.extend(_records_from_source_envelope(block))
    elif block_type == "source_observation":
      records.extend(_records_from_source_observation(block))
  return records


def fold_observed_sources(
  events: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
  """Fold the child's observed sources out of its durable tool events.

  Evidence is a fold, not a side effect (T3-I06). The settled ``dispatch``
  record is the authority for any event that carries one:

  * outcome other than ``ok`` — the call observed **nothing**, and the legacy
    event-only blocks are not consulted. This is the 429 shape: a rate-limited
    vendor payload can still carry a ``source_envelope`` naming the source it
    never delivered, and reading it here would re-open the minting hole the
    classifier closes at the boundary.
  * outcome ``ok`` with source identities — those identities win over any
    blocks on the same event.
  * outcome ``ok`` with none of its own — the block readers supply that
    event's records, because vendor and computation citations still mint on
    the api-side ledger path and legitimately settle no dispatch source.

  Pre-train events carry no dispatch record at all and keep the historical
  ``source_envelope`` / ``source_observation`` block reading.
  """

  records: list[dict[str, Any]] = []
  for event in events:
    if event.get("type") != "tool_call_complete":
      continue
    dispatch = event.get("dispatch")
    if isinstance(dispatch, Mapping) and dispatch.get("outcome") is not None:
      if str(dispatch.get("outcome")) != OUTCOME_OK:
        continue
      dispatch_records = _records_from_dispatch(event)
      if dispatch_records:
        records.extend(dispatch_records)
        continue
    records.extend(_records_from_result_blocks(event))
  return _dedup_observed_source_records(records)


def fold_dispatch_failures(
  events: Iterable[Mapping[str, Any]],
) -> Mapping[str, tuple[int, int]]:
  """Fold per-tool ``(ok, failed)`` counts out of the durable tool events.

  Reads the normalized ``dispatch.outcome`` where it exists and falls back to
  the coarse ``is_error`` bit for pre-train logs (D-B3-5).
  """

  counts: dict[str, list[int]] = {}
  for event in events:
    if event.get("type") != "tool_call_complete":
      continue
    tool_id = str(event.get("tool_name") or "").strip()
    if not tool_id:
      continue
    dispatch = event.get("dispatch")
    if isinstance(dispatch, Mapping) and dispatch.get("outcome") is not None:
      succeeded = str(dispatch.get("outcome")) == "ok"
    else:
      succeeded = not bool(event.get("is_error"))
    bucket = counts.setdefault(tool_id, [0, 0])
    bucket[0 if succeeded else 1] += 1
  return {tool_id: (bucket[0], bucket[1]) for tool_id, bucket in counts.items()}


def _collect_observed_source_records(
  event_list: list[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
  """Collect typed source observations from durable child tool events.

  Retained as the fold's call name at this seam; see
  :func:`fold_observed_sources` for the dispatch-first rule.
  """

  return fold_observed_sources(event_list)


def collect_sub_agent_result_evidence(
  entries: Iterable[Any],
  *,
  durable: bool,
) -> SubAgentResultEvidence:
  """Collect one child-run segment without double-counting usage.

  Live runner logs expose per-turn ``turn_complete`` deltas and normally finish
  with an authoritative ``stream_complete`` total. Durable logs instead retain
  assistant-message usage; historical final-answer drafts carry the same delta
  on ``runtime_guard`` because no assistant message was committed for them.
  """

  entry_list = list(entries)
  event_list = [
    event
    for event in (_entry_event(entry) for entry in entry_list)
    if event is not None
  ]
  raw_fms_results = [
    result
    for event in event_list
    if (
      event.get("type") == "tool_call_complete"
      and isinstance(event.get("tool_name"), str)
      and str(event.get("tool_name")).startswith("fms_")
      and isinstance((result := event.get("result")), Mapping)
    )
  ]
  raw_artifact_events = [
    event
    for event in event_list
    if event.get("type") == "artifact_ready"
  ]
  raw_usage_payloads = [
    usage
    for event in event_list
    if isinstance(
      (
        usage := (
          event.get("usage")
          if event.get("type") in {
            "stream_complete",
            "turn_complete",
            "assistant_message",
          }
          else event.get("draft_usage")
          if event.get("type") == "runtime_guard"
          else None
        )
      ),
      Mapping,
    )
  ]
  raw_observed_sources = _collect_observed_source_records(event_list)
  if not child_evidence_fits_externalization_bound(
    usage={},
    tools_used=[],
    fms_results=[
      *raw_fms_results,
      *raw_artifact_events,
      *raw_usage_payloads,
      *raw_observed_sources,
    ],
    artifact_events=[],
  ):
    return _rejected_evidence()
  tools_used = tuple(
    str(event.get("tool_name") or "")
    for event in event_list
    if event.get("type") == "tool_call_start"
    and str(event.get("tool_name") or "")
  )

  stream_usage: dict[str, Any] | None = None
  turn_usages: list[Mapping[str, Any]] = []
  durable_usages: list[Mapping[str, Any]] = []
  warning_parts: list[str] = []
  for event in event_list:
    event_type = str(event.get("type") or "")
    if event_type == "stream_complete":
      usage = event.get("usage")
      if isinstance(usage, Mapping):
        stream_usage = dict(usage)
    elif event_type == "turn_complete":
      usage = event.get("usage")
      if isinstance(usage, Mapping):
        turn_usages.append(usage)
    elif event_type == "assistant_message":
      usage = event.get("usage")
      if isinstance(usage, Mapping):
        durable_usages.append(usage)
    elif event_type == "runtime_guard":
      usage = event.get("draft_usage")
      if isinstance(usage, Mapping):
        durable_usages.append(usage)
    elif event_type in {"run_error", "interrupted"}:
      warning = _warning_from_event(event)
      if warning:
        warning_parts.append(warning)

  try:
    if stream_usage is not None:
      usage_payload = stream_usage
    elif durable and durable_usages:
      usage_payload = merge_usage_payloads(*durable_usages)
    else:
      usage_payload = merge_usage_payloads(*turn_usages)
  except UsageEvidenceMergeError:
    return _rejected_evidence(warning_parts)

  fms_results = tuple(
    ChainMap(
      {"tool_name": event.get("tool_name")},
      result,
    )
    for event in event_list
    if (
      event.get("type") == "tool_call_complete"
      and isinstance(event.get("tool_name"), str)
      and str(event.get("tool_name")).startswith("fms_")
      and isinstance((result := event.get("result")), Mapping)
    )
  )
  artifact_events = tuple(raw_artifact_events)
  evidence = SubAgentResultEvidence(
    usage=usage_payload,
    tools_used=tools_used,
    fms_results=fms_results,
    artifact_events=artifact_events,
    warning_parts=tuple(warning_parts),
    admission_rejected=False,
    observed_sources=raw_observed_sources,
  )
  if not child_evidence_fits_externalization_bound(
    usage=evidence.usage,
    tools_used=evidence.tools_used,
    fms_results=[*evidence.fms_results, *evidence.observed_sources],
    artifact_events=evidence.artifact_events,
  ):
    return _rejected_evidence(evidence.warning_parts)
  return evidence


def merge_sub_agent_result_evidence(
  *items: SubAgentResultEvidence | None,
) -> SubAgentResultEvidence:
  """Merge child-run evidence in lineage order."""

  evidence_items = [item for item in items if item is not None]
  if not evidence_items:
    return SubAgentResultEvidence.empty()
  warning_parts = tuple(
    warning
    for item in evidence_items
    for warning in item.warning_parts
  )
  if any(item.admission_rejected for item in evidence_items):
    return _rejected_evidence(warning_parts)
  try:
    usage = merge_usage_payloads(
      *(item.usage for item in evidence_items)
    )
  except UsageEvidenceMergeError:
    return _rejected_evidence(warning_parts)
  evidence = SubAgentResultEvidence(
    usage=usage,
    tools_used=tuple(
      tool_name
      for item in evidence_items
      for tool_name in item.tools_used
    ),
    fms_results=tuple(
      result
      for item in evidence_items
      for result in item.fms_results
    ),
    artifact_events=tuple(
      event
      for item in evidence_items
      for event in item.artifact_events
    ),
    warning_parts=warning_parts,
    admission_rejected=False,
    observed_sources=_dedup_observed_source_records(
      record
      for item in evidence_items
      for record in item.observed_sources
    ),
  )
  if not child_evidence_fits_externalization_bound(
    usage=evidence.usage,
    tools_used=evidence.tools_used,
    fms_results=[*evidence.fms_results, *evidence.observed_sources],
    artifact_events=evidence.artifact_events,
  ):
    return _rejected_evidence(evidence.warning_parts)
  return evidence


def merge_usage_payloads(
  *payloads: Mapping[str, Any],
) -> dict[str, Any]:
  """Add usage counters while retaining the newest non-numeric metadata."""

  merged: dict[str, Any] = {}
  for payload in payloads:
    for key, value in payload.items():
      if key == "provider_unit_deltas" and isinstance(value, Mapping):
        existing = merged.get(key)
        unit_totals = (
          dict(existing)
          if isinstance(existing, Mapping)
          else {}
        )
        for unit_name, unit_value in value.items():
          if _is_number(unit_value):
            unit_totals[str(unit_name)] = (
              _merge_numeric_values(
                unit_totals.get(str(unit_name), 0),
                unit_value,
              )
            )
        merged[key] = unit_totals
      elif _is_number(value):
        existing = merged.get(key)
        merged[key] = (
          _merge_numeric_values(existing, value)
          if _is_number(existing)
          else value
        )
      else:
        merged[key] = value
  return merged


def _merge_numeric_values(left: Any, right: Any) -> int | float:
  try:
    merged = left + right
  except (ArithmeticError, TypeError, ValueError) as exc:
    raise UsageEvidenceMergeError(
      "numeric usage evidence could not be merged safely"
    ) from exc
  if not _is_number(merged):
    raise UsageEvidenceMergeError(
      "numeric usage evidence produced a non-numeric result"
    )
  return merged


def _entry_event(entry: Any) -> Mapping[str, Any] | None:
  raw_event = getattr(entry, "event", entry)
  return raw_event if isinstance(raw_event, Mapping) else None


def _warning_from_event(event: Mapping[str, Any]) -> str | None:
  event_type = str(event.get("type") or "")
  raw_detail = (
    event.get("message")
    or event.get("error")
    or event.get("reason")
  )
  detail = str(raw_detail).strip() if raw_detail is not None else ""
  if event_type == "interrupted":
    return (
      f"Prior child run was interrupted: {detail}"
      if detail
      else "Prior child run was interrupted"
    )
  if event_type == "run_error":
    return (
      f"Prior child run error: {detail}"
      if detail
      else "Prior child run recorded an error"
    )
  return None


def _is_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
  "SubAgentResultEvidence",
  "UsageEvidenceMergeError",
  "collect_sub_agent_result_evidence",
  "fold_dispatch_failures",
  "fold_observed_sources",
  "merge_sub_agent_result_evidence",
  "merge_usage_payloads",
]
