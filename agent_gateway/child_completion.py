"""Publish one child completion into the parent's durable journal.

Foreground delegations and pipeline stages share this owner. The caller
supplies the registry entry that already is the job; this function does not
register a second one and does not enqueue a parent notification.
"""

from __future__ import annotations

import datetime
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from agent_workflow_contracts import (
  AdmittedTask,
  AgentCompletionEnvelope,
  ContentHandle,
  ContentReadGrant,
  TaskResult,
  sha256_digest,
)

from .runner_background_tasks import (
  agent_completion_message_id,
  build_agent_completion_envelope,
  ordinary_parent_result_policy,
  task_completed_event_payload,
  task_correlation_payload,
  task_registered_event_payload,
)
from .runner_session_events import build_agent_completion_event
from .sub_agent_narrative_result import read_task_result_terminal_narrative
from .task_registry import TaskEntry, task_state_for_result


@dataclass(frozen=True, slots=True)
class CompletionTransport:
  """The parent journal a child completion is written to.

  ``runner_id`` is the foreground runner when one exists. A batch controller
  run has none; ``principal_id`` is the parent execution's own id (the batch
  run id, or the in-turn runner id) and is what the read grant names.
  ``parent_session_id`` is required by ``task_registered``.
  """

  append: Callable[[dict[str, Any]], Awaitable[Any]]
  workspace_dir: str
  parent_session_id: str
  principal_id: str
  role: str = "writer"
  runner_id: str | None = None


def materialize_child_completion_envelope(
  result: TaskResult,
  *,
  workspace_dir: str,
  principal_id: str,
) -> AgentCompletionEnvelope:
  """Compile one child result into the same parent view a foreground child uses."""

  if not principal_id or not workspace_dir:
    raise RuntimeError(
      "child completion requires a parent principal and workspace"
    )

  policy = ordinary_parent_result_policy(result)

  def _read_terminal(canonical: TaskResult) -> str:
    return read_task_result_terminal_narrative(
      canonical,
      workspace_dir=workspace_dir,
    )

  def _read_grant(source: ContentHandle) -> ContentReadGrant:
    identity = sha256_digest({
      "task_id": result.attempt.physical_task_id,
      "content_id": source.content_id,
      "scope": "direct_parent",
      "principal_id": principal_id,
    }).removeprefix("sha256:")
    return ContentReadGrant(
      grant_id=f"content-read:{identity}",
      content_id=source.content_id,
      scope="direct_parent",
      principal_id=principal_id,
    )

  return build_agent_completion_envelope(
    result,
    policy=policy,
    terminal_narrative_reader=_read_terminal,
    read_grant_factory=_read_grant,
    message_id=agent_completion_message_id(result),
  )


def build_own_journal_registration(
  entry: TaskEntry,
  *,
  admitted_task: AdmittedTask,
  capability_bind_receipt: Mapping[str, str],
  runner_id: str,
  gateway_session_id: str,
) -> dict[str, Any]:
  """The ``task_registered`` of a job whose own writer journals it.

  The writer is the registered task's session: ``sub_agent_id`` is its gateway
  session id, which ``transcript.child_run_segment_for_task`` matches to the
  writer's attach to read the job's events after this registration as its
  attempt. The registry entry is not stamped: its terminal stamping stays
  ``publish_child_completion``'s, at the job's one settlement.
  """

  view = SimpleNamespace(
    task_id=entry.task_id,
    task_type=entry.task_type,
    started_at=entry.started_at,
    original_task_id=entry.original_task_id,
    capability_bind_receipt=dict(capability_bind_receipt),
    metadata={
      **entry.metadata,
      "sub_agent_id": gateway_session_id,
      "admitted_task": admitted_task.model_dump(mode="json"),
    },
  )
  return task_registered_event_payload(
    view,
    correlation_payload=task_correlation_payload(
      view,
      runner_id=runner_id,
      role="writer",
    ),
    agent_name=entry.agent_name,
    parent_session_id=gateway_session_id,
  )


async def publish_child_completion(
  *,
  transport: CompletionTransport,
  entry: TaskEntry,
  admitted_task: AdmittedTask,
  result: TaskResult,
  capability_bind_receipt: Mapping[str, str],
  agent_name: str | None = None,
  completed_at: float | None = None,
  settle_entry_state: bool = True,
) -> AgentCompletionEnvelope:
  """Journal ``task_registered``, ``task_completed`` and ``agent_completion``.

  ``settle_entry_state`` is false for a pipeline stage whose registry entry is
  still settled by the pipeline's own transition. Stamping a terminal state
  here would make that transition a no-op and drop the stage listener.
  """

  if not transport.parent_session_id or not transport.principal_id:
    raise RuntimeError(
      "child completion requires durable parent session identity"
    )
  if not transport.workspace_dir:
    raise RuntimeError("child completion requires a parent workspace")
  if result.attempt.physical_task_id != admitted_task.attempt.physical_task_id:
    raise RuntimeError(
      "child completion result does not settle the admitted task"
    )
  if entry.task_id != result.attempt.physical_task_id:
    raise RuntimeError(
      "child completion entry does not name the admitted task"
    )

  now = (
    completed_at
    if completed_at is not None
    else datetime.datetime.now(datetime.UTC).timestamp()
  )
  envelope = entry.completion_envelope
  if envelope is None:
    envelope = materialize_child_completion_envelope(
      result,
      workspace_dir=transport.workspace_dir,
      principal_id=transport.principal_id,
    )
  policy = entry.parent_result_policy or ordinary_parent_result_policy(result)
  result_payload = result.model_dump(mode="json")
  receipt = dict(capability_bind_receipt)
  entry.result = result_payload
  entry.capability_bind_receipt = receipt
  entry.admitted_task = admitted_task
  entry.task_result = result
  entry.parent_result_policy = policy
  entry.completion_envelope = envelope
  if entry.completed_at is None:
    entry.completed_at = now
  if settle_entry_state:
    entry.state = task_state_for_result(result)
  metadata = dict(entry.metadata)
  metadata.setdefault("admitted_task", admitted_task.model_dump(mode="json"))
  metadata.setdefault("parent_result_policy", policy.model_dump(mode="json"))
  entry.metadata = metadata

  correlation = task_correlation_payload(
    entry,
    runner_id=transport.runner_id,
    role=transport.role,
  )
  append = transport.append
  await append(task_registered_event_payload(
    entry,
    correlation_payload=correlation,
    agent_name=agent_name if agent_name is not None else entry.agent_name,
    parent_session_id=transport.parent_session_id,
  ))
  await append(task_completed_event_payload(
    entry,
    task_state_for_result(result),
    correlation_payload=correlation,
    completed_at=entry.completed_at or now,
  ))
  await append(build_agent_completion_event(
    task_id=entry.task_id,
    envelope=envelope,
    ts=entry.completed_at or now,
  ))
  return envelope


__all__ = [
  "CompletionTransport",
  "build_own_journal_registration",
  "materialize_child_completion_envelope",
  "publish_child_completion",
]
