from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_workflow_contracts import (
  AdmittedTask,
  AnalyticalOutcome,
  AttemptRef,
  ExecutionSettlement,
  LogicalTaskRef,
  OrdinaryDelegationTaskRef,
  ResultRequirement,
  TaskResult,
  TaskResultProvenance,
  WorkflowNodeTaskRef,
)

from .final_narrative_artifact import (
  publish_final_narrative,
  read_final_narrative_by_content_handle,
)
from .mechanical_outcome import derive_mechanical_outcome
from .sub_agent_result_contract import (
  FinalNarrativeArtifactReference,
  build_task_result,
)
from .sub_agent_result_evidence import (
  SubAgentResultEvidence,
  collect_sub_agent_result_evidence,
  fold_dispatch_failures,
  merge_sub_agent_result_evidence,
)


_NARRATIVE_QUERY_PAGE_SIZE = 64
_TERMINAL_REASON_PRECEDENCE = (
  "cancelled",
  "killed",
  "timeout",
  "budget_exhausted",
  "turns_exhausted",
  "retries_exhausted",
  "runtime_error",
)


@dataclass(frozen=True, slots=True)
class FinalChildVisibleText:
  text: str
  final_narrative: FinalNarrativeArtifactReference | None


def _visible_text_from_content_blocks(value: Any) -> str:
  if not isinstance(value, list):
    return ""
  return "".join(
    str(block.get("text") or "")
    for block in value
    if (
      isinstance(block, dict)
      and block.get("type") == "text"
      and isinstance(block.get("text"), str)
    )
  )


def _is_visible_max_tokens_continuation(event: Mapping[str, Any]) -> bool:
  """Return whether an assistant segment is a visible-text continuation."""

  if event.get("stop_reason") != "max_tokens":
    return False
  content_blocks = event.get("content_blocks")
  if not isinstance(content_blocks, list):
    return False
  return not any(
    isinstance(block, dict) and block.get("type") == "tool_use"
    for block in content_blocks
  )


def _entry_seq(entry: Any) -> int | None:
  value = getattr(entry, "seq", None)
  return value if type(value) is int and value > 0 else None


def _logical_response_identity(
  event: Mapping[str, Any],
) -> tuple[str, int] | None:
  response_id = event.get("logical_response_id")
  ordinal = event.get("logical_response_segment_ordinal")
  if response_id is None and ordinal is None:
    return None
  if not isinstance(response_id, str) or not response_id.strip():
    raise RuntimeError(
      "assistant logical response has an invalid logical_response_id"
    )
  if type(ordinal) is not int or ordinal < 0:
    raise RuntimeError(
      "assistant logical response has an invalid segment ordinal"
    )
  return response_id.strip(), ordinal


def _narrative_query_filters(
  *,
  sub_session_id: str,
  runner_id: str | None,
) -> dict[str, Any]:
  filters: dict[str, Any] = {
    "event_types": {"assistant_message"},
    "sub_agent_id": sub_session_id,
  }
  if runner_id is not None:
    normalized_runner_id = str(runner_id).strip()
    if not normalized_runner_id:
      raise ValueError("runner_id must be non-empty when provided")
    filters["runner_id"] = normalized_runner_id
  return filters


async def _materialize_lineaged_response(
  query: Any,
  *,
  terminal_entry: Any,
  terminal_event: Mapping[str, Any],
  query_filters: Mapping[str, Any],
) -> str:
  identity = _logical_response_identity(terminal_event)
  if identity is None:
    raise RuntimeError("logical response metadata unexpectedly absent")
  response_id, terminal_ordinal = identity
  terminal_seq = _entry_seq(terminal_entry)
  if terminal_seq is None:
    raise RuntimeError(
      "assistant logical response terminal event has no durable sequence"
    )
  segments: dict[int, tuple[int, Mapping[str, Any]]] = {
    terminal_ordinal: (terminal_seq, terminal_event)
  }
  cursor = None
  prior_seq = terminal_seq
  while len(segments) < terminal_ordinal + 1:
    query_kwargs = {
      **query_filters,
      "before_seq": terminal_seq - 1,
      "order": "desc",
      "limit": _NARRATIVE_QUERY_PAGE_SIZE,
    }
    if cursor is not None:
      query_kwargs["cursor"] = cursor
    entries, next_cursor = await query(**query_kwargs)
    ordered_entries = sorted(
      entries,
      key=lambda item: _entry_seq(item) or 0,
      reverse=True,
    )
    page_progress = False
    for entry in ordered_entries:
      seq = _entry_seq(entry)
      event = getattr(entry, "event", None)
      if seq is None or seq >= prior_seq:
        continue
      page_progress = True
      prior_seq = seq
      if (
        not isinstance(event, dict)
        or event.get("logical_response_id") != response_id
      ):
        continue
      segment_identity = _logical_response_identity(event)
      if segment_identity is None:
        continue
      _, ordinal = segment_identity
      if ordinal > terminal_ordinal:
        raise RuntimeError(
          "assistant logical response contains an out-of-range segment"
        )
      existing = segments.get(ordinal)
      if existing is not None and existing[0] != seq:
        raise RuntimeError(
          "assistant logical response contains duplicate segment ordinals"
        )
      segments[ordinal] = (seq, event)
    if next_cursor is None:
      break
    if not page_progress:
      raise RuntimeError(
        "assistant logical response pagination made no backward progress"
      )
    cursor = next_cursor

  expected_ordinals = set(range(terminal_ordinal + 1))
  if set(segments) != expected_ordinals:
    missing = sorted(expected_ordinals.difference(segments))
    raise RuntimeError(
      "assistant logical response is missing durable segments: "
      + ", ".join(str(value) for value in missing)
    )

  ordered = [segments[ordinal] for ordinal in range(terminal_ordinal + 1)]
  for ordinal, (seq, event) in enumerate(ordered):
    continued_from = event.get("continued_from_assistant_message_seq")
    if ordinal == 0:
      if continued_from is not None:
        raise RuntimeError(
          "assistant logical response origin has continuation metadata"
        )
    elif continued_from != ordered[ordinal - 1][0]:
      raise RuntimeError(
        "assistant logical response continuation does not identify its "
        "durable predecessor"
      )
    if ordinal < terminal_ordinal:
      if not _is_visible_max_tokens_continuation(event):
        raise RuntimeError(
          "assistant logical response crosses a tool or terminal boundary"
        )
    elif event.get("stop_reason") != "end_turn":
      raise RuntimeError(
        "assistant logical response does not end at an end_turn segment"
      )
  return "".join(
    _visible_text_from_content_blocks(event.get("content_blocks"))
    for _, event in ordered
  )


async def final_child_visible_text(
  session_log: Any,
  *,
  sub_session_id: str,
  workspace_dir: str,
  runner_id: str | None = None,
) -> FinalChildVisibleText:
  """Materialize the child's final durable logical assistant response."""

  query = getattr(session_log, "query", None)
  if not callable(query):
    raise TypeError(
      "narrative child execution requires a queryable durable session log"
    )
  query_filters = _narrative_query_filters(
    sub_session_id=sub_session_id,
    runner_id=runner_id,
  )
  entries, _ = await query(
    **query_filters,
    order="desc",
    limit=_NARRATIVE_QUERY_PAGE_SIZE,
  )
  ordered_entries = sorted(
    entries,
    key=lambda item: _entry_seq(item) or 0,
    reverse=True,
  )
  if not ordered_entries:
    return FinalChildVisibleText(
      text="",
      final_narrative=None,
    )
  entry = ordered_entries[0]
  event = getattr(entry, "event", None)
  # The newest assistant event for this exact attempt is authoritative. If it
  # is not terminal, partial-output handling must run; searching backward for
  # an older end_turn would publish a stale answer as this attempt's result.
  if not isinstance(event, dict) or event.get("stop_reason") != "end_turn":
    return FinalChildVisibleText(
      text="",
      final_narrative=None,
    )
  if _logical_response_identity(event) is None:
    raise RuntimeError(
      "terminal assistant message has no durable logical-response lineage"
    )
  text = await _materialize_lineaged_response(
    query,
    terminal_entry=entry,
    terminal_event=event,
    query_filters=query_filters,
  )
  if text.strip():
    event_seq = _entry_seq(entry)
    final_narrative = publish_final_narrative(
      workspace_dir=workspace_dir,
      sub_agent_id=sub_session_id,
      terminal_event_seq=event_seq,
      text=text,
    )
    return FinalChildVisibleText(
      text=text,
      final_narrative=final_narrative,
    )
  return FinalChildVisibleText(
    text="",
    final_narrative=None,
  )


def _execution_settlement(
  signals: Sequence[str],
  *,
  error_detail: str | None,
) -> ExecutionSettlement:
  if not signals:
    return ExecutionSettlement(status="succeeded")
  observed = set(signals)
  if "resume_abandoned" in observed:
    if len(observed) != 1:
      raise ValueError(
        "resume_abandoned is mutually exclusive with in-run terminal signals"
      )
    reason = "resume_abandoned"
  else:
    reason = next(
      (candidate for candidate in _TERMINAL_REASON_PRECEDENCE if candidate in observed),
      None,
    )
    if reason is None:
      raise ValueError("at least one terminal signal is required")
  detail = str(error_detail or "").strip()
  terminal_reason = f"{reason}: {detail}" if detail else reason
  if reason == "cancelled":
    status = "cancelled"
  elif reason in {"killed", "timeout", "resume_abandoned"}:
    status = "interrupted"
  else:
    status = "failed"
  return ExecutionSettlement(
    status=status,
    terminal_reason=terminal_reason,
  )


def task_result_from_execution(
  entries: Iterable[Any],
  *,
  logical_task: LogicalTaskRef,
  attempt: AttemptRef,
  requirement: ResultRequirement,
  provenance: TaskResultProvenance,
  final_narrative: FinalNarrativeArtifactReference | None,
  timed_out: bool,
  timeout: float | None,
  outcome: AnalyticalOutcome | None = None,
  budget_exceeded_reason: str | None = None,
  runtime_error_detail: str | None = None,
  external_terminal_signals: Sequence[str] = (),
  prior_evidence: SubAgentResultEvidence | None = None,
  admitted_task: AdmittedTask | None = None,
) -> TaskResult:
  """Materialize a typed runtime result from the exact terminal message.

  The child never authors this envelope.  Logical/physical identity,
  provenance, observations, and content handles are all deterministic runtime
  values; terminal prose is treated only as opaque content.

  This is the **sole** constructor of a mechanically derived outcome
  (T3-I08).  ``admitted_task`` carries the authority frozen at admission —
  ``tool_grant`` and ``capability_bindings``, never the ambient catalog.  When
  it is absent no admission is in scope, so no assessment occurred and the
  outcome stays ``None``.
  """

  if not isinstance(
    logical_task,
    (OrdinaryDelegationTaskRef, WorkflowNodeTaskRef),
  ):
    raise TypeError("task result requires an exact LogicalTaskRef")
  if not isinstance(attempt, AttemptRef):
    raise TypeError("task result requires an exact AttemptRef")
  if not isinstance(requirement, ResultRequirement):
    raise TypeError("task result requires a ResultRequirement")
  if not isinstance(provenance, TaskResultProvenance):
    raise TypeError("task result requires exact admitted provenance")

  entry_list = list(entries)
  allowed_signals = {
    "turns_exhausted",
    "retries_exhausted",
    "timeout",
    "cancelled",
    "killed",
    "resume_abandoned",
    "budget_exhausted",
    "runtime_error",
  }
  signals = [str(signal) for signal in external_terminal_signals]
  if any(signal not in allowed_signals for signal in signals):
    raise ValueError("task result received an unknown terminal signal")
  error_detail = (
    str(runtime_error_detail).strip()
    if runtime_error_detail is not None
    else None
  )
  event_list: list[Mapping[str, Any]] = []
  for entry in entry_list:
    event = getattr(entry, "event", entry)
    if not isinstance(event, Mapping):
      continue
    event_list.append(event)
    event_type = event.get("type")
    if event_type == "max_turns_reached":
      signals.append("turns_exhausted")
    elif event_type == "budget_exceeded":
      signals.append("budget_exhausted")
    elif event_type in {"run_error", "error"}:
      signals.append("runtime_error")
      if error_detail is None:
        raw = str(event.get("error") or event.get("message") or "").strip()
        error_detail = raw or None
  if timed_out:
    signals.append("timeout")
    if error_detail is None:
      error_detail = (
        f"sub-agent timed out after {timeout}s"
        if timeout is not None
        else "sub-agent timed out"
      )
  if runtime_error_detail is not None:
    signals.append("runtime_error")
  if budget_exceeded_reason is not None:
    signals.append("budget_exhausted")

  evidence = merge_sub_agent_result_evidence(
    prior_evidence,
    collect_sub_agent_result_evidence(entry_list, durable=False),
  )
  if evidence.admission_rejected:
    signals.append("runtime_error")
    error_detail = "child evidence failed canonical admission"

  projection = None
  acquired_outcome = outcome
  if not signals and requirement.mode != "narrative":
    signals.append("runtime_error")
    error_detail = (
      "agent execution accepts terminal-message results only; structured "
      "records require a deterministic runtime materializer"
    )

  if (
    not signals
    and requirement.terminal_narrative == "required"
    and final_narrative is None
  ):
    signals.append("runtime_error")
    error_detail = "durable terminal assistant narrative was not available"
  if (
    not signals
    and requirement.terminal_narrative == "forbidden"
  ):
    signals.append("runtime_error")
    error_detail = "agent execution cannot forbid its terminal assistant message"

  settled_signals = tuple(dict.fromkeys(signals))
  # The honest partial (design §4.5): a child that hit its turn ceiling and
  # still published a durable terminal narrative did real work. Exhaustion is
  # the SOLE signal here — cancelled/killed/timeout/budget/runtime failures
  # co-occurring keep the old failed settlement, and ``ExecutionSettlement``
  # forbids a ``terminal_reason`` on ``succeeded``, so the exhaustion fact
  # rides the outcome's ``unmet_requirements`` instead of the settlement.
  turns_exhausted_only = set(settled_signals) == {"turns_exhausted"}
  honest_partial = (
    turns_exhausted_only
    and final_narrative is not None
    and requirement.mode == "narrative"
  )
  if honest_partial:
    execution = ExecutionSettlement(status="succeeded")
  else:
    execution = _execution_settlement(
      settled_signals,
      error_detail=error_detail,
    )
  if execution.status != "succeeded":
    final_narrative = None
    projection = None
    acquired_outcome = None
  runtime_outcome = None
  if execution.status == "succeeded" and acquired_outcome is None:
    runtime_outcome = derive_mechanical_outcome(
      grant=(
        admitted_task.tool_grant if admitted_task is not None else None
      ),
      bindings=(
        admitted_task.capability_bindings
        if admitted_task is not None
        else ()
      ),
      failures=fold_dispatch_failures(event_list),
      sources=evidence.observed_sources,
      narrative_present=final_narrative is not None,
      turns_exhausted=honest_partial,
      # D-B3-1: the only "unavailable input" fact at this HEAD settles through
      # ``terminal_task_result`` (status ``skipped``), which forbids an
      # outcome; the derivation's missing-inputs arm stays live but unreached.
      missing_inputs=(),
    )
  return build_task_result(
    logical_task=logical_task,
    attempt=attempt,
    requirement=requirement,
    provenance=provenance,
    execution=execution,
    outcome=acquired_outcome,
    terminal_narrative=final_narrative,
    projection=projection,
    observed_sources=evidence.observed_sources,
    tools_used=evidence.tools_used,
    usage=evidence.usage,
    runtime_outcome=runtime_outcome,
  )


def task_result_payload_from_execution(*args: Any, **kwargs: Any) -> dict[str, Any]:
  """Serialize :func:`task_result_from_execution` at dictionary seams."""

  return task_result_from_execution(*args, **kwargs).model_dump(mode="json")


def read_task_result_terminal_narrative(
  task_result: TaskResult,
  *,
  workspace_dir: str,
) -> str:
  """Read exact terminal prose through the canonical content handle."""

  if not isinstance(task_result, TaskResult):
    raise TypeError("terminal narrative read requires a TaskResult")
  content = task_result.values.terminal_narrative
  if content is None:
    raise ValueError("task result has no terminal narrative")
  return read_final_narrative_by_content_handle(
    workspace_dir=workspace_dir,
    content=content,
  )


__all__ = [
  "FinalChildVisibleText",
  "final_child_visible_text",
  "read_task_result_terminal_narrative",
  "task_result_from_execution",
  "task_result_payload_from_execution",
]
