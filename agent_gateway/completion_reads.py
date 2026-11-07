"""Versioned raw completion reads composed from the execution owners.

These reads deliberately have no settled/success/quiescent predicate. Missing
process-local owners remain unknown, including after a gateway restart.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict


class CompletionRead(BaseModel):
  model_config = ConfigDict(extra="forbid", strict=True)

  schema_version: Literal[1] = 1
  source: Literal["chat", "workflow", "runner"]
  consistency: Literal["current", "changed", "unknown"]
  terminal: dict[str, Any] | None
  receipt: dict[str, Any] | None
  journal: dict[str, Any] | None
  durable: dict[str, Any] | None
  children: dict[str, Any] | None
  obligations: dict[str, Any] | None
  pending_approval: dict[str, Any] | None
  unknown: list[str]


def _observe(
  name: str,
  read: Callable[[], Any],
  unknown: list[str],
) -> Any:
  try:
    return read()
  except Exception as exc:
    # A degraded read is data, not evidence that the obligation disappeared.
    unknown.append(f"{name}:{type(exc).__name__}")
    return None


def _runner_sources(runner: Any, journal: Any) -> tuple[dict[str, Any], Any]:
  unknown: list[str] = []
  background = _observe("background", lambda: runner.background_completion_read(), unknown)
  children = _observe(
    "children",
    lambda: runner._task_registry.completion_read(
      exclude_task_id=(
        runner._admitted_self_task.task_id
        if runner._admitted_self_task is not None else None
      ),
    ),
    unknown,
  )
  session = _observe("session", lambda: runner.session_completion_read(), unknown)
  journal_read = _observe("journal", lambda: journal.completion_read(), unknown)
  durable_log = _observe("durable_owner", lambda: runner._agent_session_log, unknown)
  if durable_log is None:
    durable = None
    unknown.append(
      "durable:unavailable" if "durable_owner:AttributeError" in unknown
      else "durable:not_configured"
    )
  else:
    durable = _observe(
      "durable",
      lambda: durable_log.completion_read(runner_id=runner._runner_id),
      unknown,
    )
    if durable is not None and durable["status"] in {"unknown", "no_target"}:
      unknown.append(f"durable:{durable['status']}")
  if background is not None:
    for name, value in background[0].items():
      if isinstance(value, dict) and value.get("count") is None:
        unknown.append(f"background:{name}:unavailable")
  if children is not None and children["coverage"] == "unknown":
    unknown.append("children:unknown_epoch")
  facts = {
    "journal": journal_read,
    "durable": durable,
    "children": children,
    "obligations": {
      "background": background[0] if background is not None else None,
      "session": session,
    },
    "unknown": unknown,
  }
  return facts, background[1] if background is not None else None


def read_runner_completion(
  runner: Any,
  *,
  journal: Any | None = None,
  pending_approval: dict[str, Any] | None = None,
  source: Literal["chat", "workflow", "runner"] = "runner",
) -> CompletionRead:
  """Observe once, then detect a changing source without waiting or retrying.

  The native background fence supplies its own fingerprint. Comparing the
  remaining owner reads also covers journal writes, detach and raw task states.
  No snapshot is promoted to an execution/liveness verdict.
  """
  selected_journal = getattr(runner, "_log", None) if journal is None else journal
  facts, fingerprint = _runner_sources(runner, selected_journal)
  after, after_fingerprint = _runner_sources(runner, selected_journal)
  changed = facts != after or fingerprint != after_fingerprint
  unknown = list(dict.fromkeys([*facts["unknown"], *after["unknown"]]))
  if pending_approval is None:
    unknown.append("pending_approval:unavailable")
  journal_read = facts["journal"]
  durable = facts["durable"]
  return CompletionRead(
    source=source,
    consistency="changed" if changed else ("unknown" if unknown else "current"),
    terminal=journal_read["terminal"] if journal_read is not None else None,
    receipt=(
      {
        "source": "agent_session_log", "status": durable["status"],
        "refs": durable["receipts"],
      }
      if durable is not None else None
    ),
    journal=journal_read,
    durable=durable,
    children=facts["children"],
    obligations=facts["obligations"],
    pending_approval=deepcopy(pending_approval),
    unknown=unknown,
  )


def _chat_sources(session: Any) -> tuple[Any, dict[str, Any]]:
  refs = [
    {"tool_call_id": tool_call_id, "approval_id": entry.get("approval_id")}
    for tool_call_id, entry in session.pending_tools.items()
    if entry.get("status") == "approval_pending"
  ]
  turn = session.active_turn
  if turn is None:
    turn = session.last_completed_turn
  return turn, {"count": len(refs), "refs": refs}


def read_chat_completion(session: Any) -> CompletionRead:
  turn, pending = _chat_sources(session)
  if turn is not None and turn.runtime_owner is not None:
    runner = turn.runtime_owner
    result = read_runner_completion(
      runner, journal=turn.event_log, pending_approval=pending, source="chat",
    )
    after_turn, after_pending = _chat_sources(session)
    if after_turn is not turn or turn.runtime_owner is not runner or after_pending != pending:
      result = result.model_copy(update={"consistency": "changed"})
    return result
  journal = turn.event_log.completion_read() if turn is not None else None
  return CompletionRead(
    source="chat", consistency="unknown",
    terminal=journal["terminal"] if journal is not None else None,
    receipt=None, journal=journal, durable=None, children=None, obligations=None,
    pending_approval=pending,
    unknown=["runner:unavailable", "children:unknown_epoch", "durable:unavailable"],
  )


def read_workflow_completion(
  record: Any,
  *,
  terminal_receipt: dict[str, Any] | None,
  pending_approval_refs: list[dict[str, Any]],
) -> CompletionRead:
  """Read the existing subprocess record, not an invented in-process runner.

  The terminal manifest and channel acknowledgement are distinct native facts.
  The subprocess's task registry, EventLog and initialization/settlement work
  are not owned by this process and cannot be inferred from its exit status.
  """
  acknowledgement = record.event_channel_acknowledgement
  acknowledgement_read = (
    {
      "channel_id": acknowledgement.channel_id,
      "event_count": acknowledgement.event_count,
      "event_digest": acknowledgement.event_digest,
    }
    if acknowledgement is not None else None
  )
  return CompletionRead(
    source="workflow", consistency="unknown",
    terminal=deepcopy(terminal_receipt),
    receipt=(
      {"source": "autonomous_task_manifest", "value": deepcopy(terminal_receipt),
       "committed": record.terminal_manifest_committed}
      if terminal_receipt is not None else None
    ),
    journal=None,
    durable={
      "terminal_manifest_committed": record.terminal_manifest_committed,
      "event_evidence_status": record.events_evidence_status,
      "event_channel_acknowledgement": acknowledgement_read,
      "detach": None,
    },
    children=None, obligations=None,
    pending_approval={
      "count": len(pending_approval_refs), "refs": deepcopy(pending_approval_refs),
    },
    unknown=[
      "runner:remote", "journal:unavailable", "children:unknown_epoch",
      "obligations:unavailable", "durable_detach:unavailable",
    ],
  )
