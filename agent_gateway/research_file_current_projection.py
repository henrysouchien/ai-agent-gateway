"""Current session-log projection cutoff for erased research evidence.

The durable JSONL remains the archive.  This module owns the much narrower
current-product view used by replay, task recovery, and current session query.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
from uuid import UUID

from agent_workflow_contracts import AdmittedTask

from .agent_session_log import (
  AgentSessionLog,
  AgentSessionLogEnumerationError,
  AgentSessionLogLocation,
  LogEntry,
  try_acquire_agent_session_log_write_leases,
)
from .agent_session_log_inventory import (
  SessionLogInventoryError,
  enumerate_selected_agent_session_logs,
)
from .agent_session_log_layout import (
  SESSION_LOG_LAYOUT_V1,
  resolve_agent_session_log_archive_product_ids,
  resolve_agent_session_log_layout,
)
from .product_config import gateway_product_id
from .agent_session_log_records import EVENT_SCHEMA_VERSION, slugify


RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE = (
  "research_file_current_projection_unavailable"
)
_CURRENT_PROJECTION_EVENT_VERSION = 1
_DISCOVERY_LIMIT = 10_000
_DOCUMENT_ID_RE = re.compile(r"^doc:[0-9a-f]{32}$")
_MARKER_KEYS = frozenset({
  "type",
  "event_schema_version",
  "current_projection_event_version",
  "invalidated_through_seq",
  "research_file_ids",
  "document_id",
  "document_generation",
  "reason",
})


class ResearchFileCurrentProjectionUnavailable(RuntimeError):
  """The source-owned current session projection cannot be determined."""


@dataclass(frozen=True, slots=True)
class ResearchFileCurrentProjection:
  """Pure current-view boundary for one integrity-checked log snapshot."""

  cutoff_seq: int
  marker_seqs: frozenset[int]

  def excludes(self, entry: LogEntry) -> bool:
    return entry.seq <= self.cutoff_seq or entry.seq in self.marker_seqs


def build_current_projection_unavailable_event(
  *,
  invalidated_through_seq: int,
  research_file_ids: Iterable[int],
  document_id: str,
  document_generation: str,
) -> dict[str, object]:
  if type(invalidated_through_seq) is not int or invalidated_through_seq < 0:
    raise ValueError("invalidated_through_seq must be a non-negative integer")
  canonical_ids = tuple(sorted(set(research_file_ids)))
  if any(type(value) is not int or value <= 0 for value in canonical_ids):
    raise ValueError("research_file_ids must contain only positive integers")
  canonical_document_id = _canonical_document_id(document_id)
  canonical_generation = _canonical_document_generation(document_generation)
  return {
    "type": RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE,
    "current_projection_event_version": _CURRENT_PROJECTION_EVENT_VERSION,
    "invalidated_through_seq": invalidated_through_seq,
    "research_file_ids": list(canonical_ids),
    "document_id": canonical_document_id,
    "document_generation": canonical_generation,
    "reason": "document_evidence_erased",
  }


def project_research_file_current_projection(
  marker_entries: Iterable[LogEntry],
) -> ResearchFileCurrentProjection:
  marker_seqs: set[int] = set()
  cutoff_seq = 0
  for entry in marker_entries:
    event = entry.event
    if frozenset(event) != _MARKER_KEYS:
      raise ResearchFileCurrentProjectionUnavailable(
        "current projection cutoff has an invalid schema"
      )
    if event.get("type") != RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE:
      raise ResearchFileCurrentProjectionUnavailable(
        "current projection discovery returned an unexpected event"
      )
    if event.get("current_projection_event_version") != (
      _CURRENT_PROJECTION_EVENT_VERSION
    ):
      raise ResearchFileCurrentProjectionUnavailable(
        "current projection cutoff has an unsupported version"
      )
    if event.get("event_schema_version") != EVENT_SCHEMA_VERSION:
      raise ResearchFileCurrentProjectionUnavailable(
        "current projection cutoff has an unsupported event schema"
      )
    if event.get("reason") != "document_evidence_erased":
      raise ResearchFileCurrentProjectionUnavailable(
        "current projection cutoff has an invalid reason"
      )
    invalidated_through_seq = event.get("invalidated_through_seq")
    if (
      type(invalidated_through_seq) is not int
      or invalidated_through_seq < 0
      or invalidated_through_seq >= entry.seq
    ):
      raise ResearchFileCurrentProjectionUnavailable(
        "current projection cutoff has an invalid sequence boundary"
      )
    raw_ids = event.get("research_file_ids")
    if (
      not isinstance(raw_ids, list)
      or any(type(value) is not int or value <= 0 for value in raw_ids)
      or raw_ids != sorted(set(raw_ids))
    ):
      raise ResearchFileCurrentProjectionUnavailable(
        "current projection cutoff has invalid research file identities"
      )
    try:
      _canonical_document_id(event.get("document_id"))
    except ValueError as exc:
      raise ResearchFileCurrentProjectionUnavailable(
        "current projection cutoff has no exact document identity"
      ) from exc
    try:
      _canonical_document_generation(event.get("document_generation"))
    except ValueError as exc:
      raise ResearchFileCurrentProjectionUnavailable(
        "current projection cutoff has an invalid document generation"
      ) from exc
    marker_seqs.add(entry.seq)
    cutoff_seq = max(cutoff_seq, invalidated_through_seq)
  return ResearchFileCurrentProjection(
    cutoff_seq=cutoff_seq,
    marker_seqs=frozenset(marker_seqs),
  )


async def load_research_file_current_projection(
  session_log: AgentSessionLog,
  *,
  snapshot_max_seq: int | None = None,
) -> ResearchFileCurrentProjection:
  if snapshot_max_seq is None:
    snapshot_max_seq = await session_log.latest_seq_current_strict()
  entries, cursor = await session_log.query_current_strict(
    event_types={RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE},
    before_seq=snapshot_max_seq,
    order="asc",
    limit=_DISCOVERY_LIMIT,
  )
  if cursor is not None:
    raise ResearchFileCurrentProjectionUnavailable(
      "current projection cutoffs exceed their discovery bound"
    )
  return project_research_file_current_projection(entries)


def load_research_file_current_projection_sync(
  session_log: AgentSessionLog,
  *,
  snapshot_max_seq: int | None = None,
) -> ResearchFileCurrentProjection:
  """Synchronous strict projection read for an owned terminal barrier."""

  if snapshot_max_seq is None:
    snapshot_max_seq = session_log.latest_seq_current_strict_sync()
  entries, cursor = session_log.query_current_strict_sync(
    event_types={RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE},
    before_seq=snapshot_max_seq,
    order="asc",
    limit=_DISCOVERY_LIMIT,
  )
  if cursor is not None:
    raise ResearchFileCurrentProjectionUnavailable(
      "current projection cutoffs exceed their discovery bound"
    )
  return project_research_file_current_projection(entries)


async def task_registration_is_current(
  session_log: AgentSessionLog,
  task_id: str,
) -> bool:
  """Return whether an exact task registration exists after the last cutoff."""

  snapshot_max_seq = await session_log.latest_seq_current_strict()
  projection = await load_research_file_current_projection(
    session_log,
    snapshot_max_seq=snapshot_max_seq,
  )
  if projection.cutoff_seq == 0 and not projection.marker_seqs:
    return True
  entries, cursor = await session_log.query_current_strict(
    event_types={"task_registered"},
    after_seq=projection.cutoff_seq + 1,
    before_seq=snapshot_max_seq,
    contains_text=task_id,
    order="asc",
    limit=_DISCOVERY_LIMIT,
    exclude_entry=projection.excludes,
  )
  if cursor is not None:
    raise ResearchFileCurrentProjectionUnavailable(
      "current task registrations exceed their discovery bound"
    )
  return any(entry.event.get("task_id") == task_id for entry in entries)


async def invalidate_owner_current_session_projections(
  base_dir: Path,
  *,
  owner_user_id: str,
  owner_user_id_aliases: Iterable[str] = (),
  research_file_ids: Iterable[int],
  document_id: str,
  document_generation: str,
) -> None:
  """Append one conservative current-view cutoff to every exact owner log."""

  if type(owner_user_id) is not str or not owner_user_id.strip():
    raise ValueError("owner_user_id is required")
  owner_user_ids = tuple(sorted({owner_user_id, *owner_user_id_aliases}))
  if any(type(value) is not str or not value.strip() for value in owner_user_ids):
    raise ValueError("owner_user_id aliases must be non-empty strings")
  canonical_ids = tuple(sorted(set(research_file_ids)))
  # Validate before touching storage, including the deliberately allowed empty
  # set used when an erased document had not yet reached an RF repository row.
  build_current_projection_unavailable_event(
    invalidated_through_seq=0,
    research_file_ids=canonical_ids,
    document_id=document_id,
    document_generation=document_generation,
  )
  selected = _select_owner_locations(
    base_dir,
    owner_user_ids=owner_user_ids,
  )
  leases = try_acquire_agent_session_log_write_leases(selected)
  if leases is None:
    raise ResearchFileCurrentProjectionUnavailable(
      "owner session-log writers are active"
    )
  try:
    confirmed = _select_owner_locations(
      base_dir,
      owner_user_ids=owner_user_ids,
    )
    if confirmed != selected:
      raise ResearchFileCurrentProjectionUnavailable(
        "owner session-log selection changed during invalidation"
      )
    for location in confirmed:
      session_log = AgentSessionLog(location)
      latest_seq = await session_log.latest_seq_current_strict()
      current_projection = await load_research_file_current_projection(
        session_log,
        snapshot_max_seq=latest_seq,
      )
      marker_entries, cursor = await session_log.query_current_strict(
        event_types={RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE},
        before_seq=latest_seq,
        order="desc",
        limit=_DISCOVERY_LIMIT,
      )
      if cursor is not None:
        raise ResearchFileCurrentProjectionUnavailable(
          "current projection cutoffs exceed their discovery bound"
        )
      prior_scope_markers = [
        entry
        for entry in marker_entries
        if (
          entry.event.get("document_id") == document_id
          and entry.event.get("document_generation")
          == document_generation
        )
      ]
      if prior_scope_markers:
        latest_scope_marker = prior_scope_markers[0]
        latest_current_boundary = max(
          current_projection.cutoff_seq,
          max(current_projection.marker_seqs, default=0),
        )
        post_cutoff_entries, post_cutoff_cursor = (
          await session_log.query_current_strict(
            after_seq=max(
              latest_scope_marker.seq,
              latest_current_boundary,
            ) + 1,
            before_seq=latest_seq,
            order="asc",
            limit=_DISCOVERY_LIMIT,
          )
        )
        if post_cutoff_cursor is not None:
          raise ResearchFileCurrentProjectionUnavailable(
            "post-cutoff current events exceed their discovery bound"
          )
        if not any(
          _event_matches_research_file_scope(entry.event, canonical_ids)
          for entry in post_cutoff_entries
        ):
          # Same-incarnation retries leave unrelated later work current.
          continue
      await session_log.append(build_current_projection_unavailable_event(
        invalidated_through_seq=latest_seq,
        research_file_ids=canonical_ids,
        document_id=document_id,
        document_generation=document_generation,
      ))
  except ResearchFileCurrentProjectionUnavailable:
    raise
  except Exception as exc:
    raise ResearchFileCurrentProjectionUnavailable(
      "owner current session projections could not be invalidated"
    ) from exc
  finally:
    leases.release()


def _select_owner_locations(
  base_dir: Path,
  *,
  owner_user_ids: tuple[str, ...],
) -> tuple[AgentSessionLogLocation, ...]:
  try:
    selected_layout = resolve_agent_session_log_layout()
    trusted_product_id = gateway_product_id()
    allowed_product_ids = frozenset({trusted_product_id})
    if selected_layout == SESSION_LOG_LAYOUT_V1:
      allowed_product_ids = frozenset({
        trusted_product_id,
        *resolve_agent_session_log_archive_product_ids(),
      })
    inventory = enumerate_selected_agent_session_logs(
      base_dir,
      layout=selected_layout,
      trusted_product_id=trusted_product_id,
      allowed_product_ids=allowed_product_ids,
      allowed_stream_kinds=frozenset({
        "canonical", "batch", "pipeline", "ephemeral",
      }),
    )
  except (AgentSessionLogEnumerationError, SessionLogInventoryError) as exc:
    raise ResearchFileCurrentProjectionUnavailable(
      "owner session logs cannot be enumerated"
    ) from exc
  selected: list[AgentSessionLogLocation] = []
  for item in inventory:
    location = item.location
    if item.storage_layout == "v2":
      metadata_owner = (
        item.sidecar_payload.get("user_id")
        if item.sidecar_payload is not None
        else None
      )
      if item.stream_kind == "canonical" and metadata_owner in owner_user_ids:
        selected.append(location)
      continue
    # Autonomous/captured logs are canonical and their owner namespace is
    # encoded in the v1 source-owned path. A v1 sidecar remains telemetry and
    # never authorizes a reset. Interactive ephemeral logs deliberately do not
    # match.
    agent_slug = location.path.parent.name
    expected_stems = {
      f"agentsess_{agent_slug}_{slugify(owner_user_id)}"
      for owner_user_id in owner_user_ids
    }
    if location.path.stem in expected_stems:
      selected.append(location)
  return tuple(selected)


def _event_matches_research_file_scope(
  event: dict[str, object],
  research_file_ids: tuple[int, ...],
) -> bool:
  if not research_file_ids:
    return False
  target_ids = set(research_file_ids)
  direct_id = event.get("context_research_file_id")
  if type(direct_id) is int and direct_id in target_ids:
    return True
  if event.get("type") != "task_registered":
    return False
  metadata = event.get("metadata")
  if not isinstance(metadata, dict):
    return False
  admitted_task = metadata.get("admitted_task")
  if admitted_task is None:
    return False
  try:
    typed_task = AdmittedTask.model_validate(admitted_task)
  except Exception as exc:
    raise ResearchFileCurrentProjectionUnavailable(
      "current task registration has malformed admission metadata"
    ) from exc
  matches = tuple(
    binding
    for binding in typed_task.inputs
    if binding.name == "research_file_id"
  )
  if len(matches) > 1:
    raise ResearchFileCurrentProjectionUnavailable(
      "current task registration has ambiguous research file identity"
    )
  if not matches:
    return False
  admitted_id = matches[0].context.content
  return type(admitted_id) is int and admitted_id in target_ids


def _canonical_document_generation(value: object) -> str:
  if type(value) is not str:
    raise ValueError("document_generation must be a canonical UUID")
  try:
    parsed = UUID(value)
  except (AttributeError, TypeError, ValueError) as exc:
    raise ValueError("document_generation must be a canonical UUID") from exc
  canonical = str(parsed)
  if value != canonical:
    raise ValueError("document_generation must be a canonical UUID")
  return canonical


def _canonical_document_id(value: object) -> str:
  if type(value) is not str or _DOCUMENT_ID_RE.fullmatch(value) is None:
    raise ValueError("document_id must be canonical doc identity")
  return value


__all__ = [
  "RESEARCH_FILE_CURRENT_PROJECTION_UNAVAILABLE_EVENT_TYPE",
  "ResearchFileCurrentProjection",
  "ResearchFileCurrentProjectionUnavailable",
  "build_current_projection_unavailable_event",
  "invalidate_owner_current_session_projections",
  "load_research_file_current_projection",
  "load_research_file_current_projection_sync",
  "project_research_file_current_projection",
  "task_registration_is_current",
]
