"""Publish one child completion into the parent's durable journal.

Foreground delegations and pipeline stages share this owner. The caller
supplies the registry entry that already is the job; this function does not
register a second one and does not enqueue a parent notification.
"""

from __future__ import annotations

import asyncio
import datetime
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

from agent_workflow_contracts import (
  AdmittedTask,
  AgentCompletionEnvelope,
  ContentHandle,
  ContentReadGrant,
  ParentResultPolicy,
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
from .final_narrative_artifact import FinalNarrativeArtifactError
from .runner_session_events import build_agent_completion_event
from .sub_agent_narrative_result import read_task_result_terminal_narrative
from .skill_lifecycle import drain_owned_lifecycle_task
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
  policy: ParentResultPolicy | None = None,
) -> AgentCompletionEnvelope:
  """Compile one child result into the same parent view a foreground child uses."""

  if not principal_id:
    raise RuntimeError("child completion requires a parent principal")

  if policy is None:
    policy = ordinary_parent_result_policy(result)

  def _read_terminal(canonical: TaskResult) -> str:
    if not workspace_dir:
      raise FinalNarrativeArtifactError(
        "durable workspace root is unavailable"
      )
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
  """Build the registration naming the session that journals this attempt.

  ``sub_agent_id`` is that session's gateway id, which
  ``transcript.child_run_segment_for_task`` matches to its writer's attach.
  An own-writer job registers after attaching; a foreground delegation registers
  before the child attaches and records its distinct parent session id.
  The registry entry is not stamped until completion publication.
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


def _stamp_child_result(
  entry: TaskEntry,
  *,
  admitted_task: AdmittedTask,
  result: TaskResult,
  capability_bind_receipt: Mapping[str, str],
  completed_at: float,
  settle_entry_state: bool,
) -> None:
  policy = entry.parent_result_policy or ordinary_parent_result_policy(result)
  entry.result = result.model_dump(mode="json")
  entry.capability_bind_receipt = dict(capability_bind_receipt)
  entry.admitted_task = admitted_task
  entry.task_result = result
  entry.parent_result_policy = policy
  if entry.completed_at is None:
    entry.completed_at = completed_at
  if settle_entry_state:
    entry.state = task_state_for_result(result)
  metadata = dict(entry.metadata)
  metadata.setdefault("admitted_task", admitted_task.model_dump(mode="json"))
  metadata.setdefault("parent_result_policy", policy.model_dump(mode="json"))
  entry.metadata = metadata


async def publish_registered_child_settlement(
  *,
  transport: CompletionTransport,
  entry: TaskEntry,
  admitted_task: AdmittedTask,
  result: TaskResult,
  capability_bind_receipt: Mapping[str, str],
  completed_at: float | None = None,
) -> float:
  """Commit execution independently of readable-result materialization."""

  async def persist() -> float:
    async with entry.finalization_lock:
      if entry.completion_persistence_state == "committed":
        return cast(float, entry.completed_at)
      now = completed_at if completed_at is not None else datetime.datetime.now(datetime.UTC).timestamp()
      _stamp_child_result(
        entry, admitted_task=admitted_task, result=result,
        capability_bind_receipt=capability_bind_receipt,
        completed_at=now, settle_entry_state=True,
      )
      # Execution details live in the canonical TaskResult; a transport or
      # cancellation exception is not a second, conflicting execution error.
      entry.error = None
      await transport.append(task_completed_event_payload(
        entry, task_state_for_result(result),
        correlation_payload=task_correlation_payload(
          entry, runner_id=transport.runner_id, role=transport.role,
        ),
        completed_at=now,
      ))
      entry.completion_persistence_state = "committed"
      return now

  settlement = asyncio.create_task(persist())
  try:
    return await asyncio.shield(settlement)
  except asyncio.CancelledError:
    await drain_owned_lifecycle_task(settlement)
    raise


def _build_child_completion_publication(
  *,
  transport: CompletionTransport,
  entry: TaskEntry,
  admitted_task: AdmittedTask,
  result: TaskResult,
  capability_bind_receipt: Mapping[str, str],
  agent_name: str | None = None,
  completed_at: float | None = None,
  settle_entry_state: bool = True,
) -> tuple[AgentCompletionEnvelope, dict[str, Any], tuple[dict[str, Any], ...]]:
  """Materialize and stamp one settlement before compiling its journal events."""

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
  _stamp_child_result(
    entry, admitted_task=admitted_task, result=result,
    capability_bind_receipt=capability_bind_receipt,
    completed_at=now, settle_entry_state=settle_entry_state,
  )
  entry.completion_envelope = envelope

  correlation = task_correlation_payload(
    entry,
    runner_id=transport.runner_id,
    role=transport.role,
  )
  registration = task_registered_event_payload(
    entry,
    correlation_payload=correlation,
    agent_name=agent_name if agent_name is not None else entry.agent_name,
    parent_session_id=transport.parent_session_id,
  )
  settlement = (
    task_completed_event_payload(
      entry,
      task_state_for_result(result),
      correlation_payload=correlation,
      completed_at=entry.completed_at or now,
    ),
    build_agent_completion_event(
      task_id=entry.task_id,
      envelope=envelope,
      ts=entry.completed_at or now,
    ),
  )
  return envelope, registration, settlement


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
  """Register and settle a child at publication (pipeline-stage contract).

  ``settle_entry_state`` is false when the pipeline's own transition still
  settles the registry entry; stamping it here would drop the stage listener.
  """

  envelope, registration, settlement = _build_child_completion_publication(
    transport=transport,
    entry=entry,
    admitted_task=admitted_task,
    result=result,
    capability_bind_receipt=capability_bind_receipt,
    agent_name=agent_name,
    completed_at=completed_at,
    settle_entry_state=settle_entry_state,
  )
  for event in (registration, *settlement):
    await transport.append(event)
  return envelope


async def publish_registered_child_completion(
  *,
  transport: CompletionTransport,
  entry: TaskEntry,
  admitted_task: AdmittedTask,
  result: TaskResult,
  capability_bind_receipt: Mapping[str, str],
  agent_name: str | None = None,
  completed_at: float | None = None,
) -> AgentCompletionEnvelope:
  """Settle an attempt registered in its transcript before its first model turn."""

  settled_at = await publish_registered_child_settlement(
    transport=transport, entry=entry, admitted_task=admitted_task,
    result=result, capability_bind_receipt=capability_bind_receipt,
    completed_at=completed_at,
  )
  envelope = materialize_child_completion_envelope(
    result, workspace_dir=transport.workspace_dir,
    principal_id=transport.principal_id,
  )
  entry.completion_envelope = envelope
  await transport.append(build_agent_completion_event(
    task_id=entry.task_id, envelope=envelope, ts=settled_at,
  ))
  return envelope


__all__ = [
  "CompletionTransport",
  "build_own_journal_registration",
  "materialize_child_completion_envelope",
  "publish_child_completion",
  "publish_registered_child_completion",
  "publish_registered_child_settlement",
]
