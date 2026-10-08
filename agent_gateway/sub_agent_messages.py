"""The job mailbox: ``send_message`` admitted as a ``MessageRequest``.

Every send — parent -> child, child -> parent, sibling -> sibling — resolves
its recipient on the sender's tree roster (the linked per-runner
``TaskRegistry`` instances, companion P6), then calls the host ``admit`` owner
with the sender's own admitted context: the tenant half of the addressing
width and the ``handle_refs`` subset are admission's, the tree half is the
roster's (a miss is ``not_on_roster``). A message never authorizes: the
handles it cites are recorded with it and grant the recipient nothing.

Delivery is the recipient's existing inbox, drained at its next turn
boundary; a child's acceptance is journaled under its task identity and the
completion lock, which keeps completion ordering and accepted-message
identity intact.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from .admission import MessageOwners, MessageRequest, ParentContext, Refusal, admit
from .job_caller import CallerContext, JobScope, JobScopedHandler
from .runner_background_tasks import task_correlation_payload
from .task_registry import (
  ParentMessage,
  RosterAmbiguous,
  RosterJob,
  TaskRegistry,
  is_canonical_handle_ref,
  job_output_handles,
)


PARENT_MESSAGE_MAX_CHARS = 12_000
PARENT_MESSAGE_MAX_BYTES = 32 * 1024
PARENT_MESSAGE_ID_MAX_CHARS = 512
PARENT_MESSAGE_ID_MAX_BYTES = 512


def _message_result(
  message: ParentMessage,
  *,
  task_id: str,
) -> dict[str, Any]:
  return {
    "status": "accepted" if message.sent_seq is not None else "queued",
    "task_id": task_id,
    "message_id": message.message_id,
  }


def _invalid_message(message: Any) -> dict[str, Any] | None:
  if type(message) is not str or not message or message != message.strip():
    return {
      "code": "invalid_input",
      "message": "message must be canonical non-empty text",
    }
  if (
    len(message) > PARENT_MESSAGE_MAX_CHARS
    or len(message.encode("utf-8")) > PARENT_MESSAGE_MAX_BYTES
  ):
    return {"code": "invalid_input", "message": "message is too long"}
  return None


def _canonical_message_id(message_id: Any) -> bool:
  return (
    type(message_id) is str
    and bool(message_id)
    and message_id == message_id.strip()
    and len(message_id) <= PARENT_MESSAGE_ID_MAX_CHARS
    and len(message_id.encode("utf-8")) <= PARENT_MESSAGE_ID_MAX_BYTES
  )


def _same_acceptance(existing: ParentMessage, candidate: ParentMessage) -> bool:
  return (
    existing.text == candidate.text
    and existing.handle_refs == candidate.handle_refs
  )


async def _durable_parent_message(
  runner: Any,
  *,
  entry: Any,
  message_id: str,
) -> tuple[ParentMessage | None, dict[str, Any] | None]:
  """Resolve one exact acceptance from the existing session journal."""

  durable_log = getattr(runner, "_agent_session_log", None)
  if durable_log is None:
    return None, None
  entries, _ = await durable_log.query(
    event_types={"parent_message_sent"},
    contains_text=message_id,
    order="asc",
  )
  matches = [
    durable_entry
    for durable_entry in entries
    if (
      durable_entry.event.get("task_id") == entry.task_id
      and durable_entry.event.get("message_id") == message_id
    )
  ]
  if len(matches) > 1:
    return None, {
      "code": "message_integrity_error",
      "message": "message_id has duplicate durable acceptance facts",
    }
  if not matches:
    return None, None
  match = matches[0]
  event = match.event
  text = event.get("message")
  sent_at = event.get("sent_at")
  target_sub_agent_id = (
    entry.metadata.get("sub_agent_id")
    if isinstance(entry.metadata, dict)
    else None
  )
  if (
    _invalid_message(text) is not None
    or isinstance(sent_at, bool)
    or not isinstance(sent_at, (int, float))
    or (
      target_sub_agent_id is not None
      and event.get("sub_agent_id") != target_sub_agent_id
    )
  ):
    return None, {
      "code": "message_integrity_error",
      "message": "message_id has an invalid durable acceptance fact",
    }
  try:
    accepted = ParentMessage.from_durable_event(
      event,
      task_id=entry.task_id,
      sent_seq=match.seq,
      text=text,
      sent_at=float(sent_at),
    )
  except ValueError:
    return None, {
      "code": "message_integrity_error",
      "message": "message_id has an invalid durable acceptance fact",
    }
  return accepted, None


def _sender_payload(runner: Any, sender_job_id: str | None) -> dict[str, Any]:
  sender: dict[str, Any] = {
    "session_id": getattr(runner, "_full_session_id", None),
    "user_id": getattr(runner, "_usage_user_id", None),
  }
  if sender_job_id is not None:
    sender["job_id"] = sender_job_id
  return sender


async def _accept_on_entry(
  runner: Any,
  *,
  entry: Any,
  candidate: ParentMessage,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """Accept one message for a registered job under its completion lock.

  ``TaskEntry.finalization_lock`` is shared with completion persistence. The
  lock makes the durable ordering authoritative: a message that owns the lock
  first is accepted before completion; a completion that owns it first makes
  the message a terminal conflict.
  """

  message_id = candidate.message_id
  async with entry.finalization_lock:
    existing = entry.accepted_parent_messages.get(message_id)
    if existing is None:
      existing, durable_error = await _durable_parent_message(
        runner,
        entry=entry,
        message_id=message_id,
      )
      if durable_error is not None:
        return None, durable_error
      if existing is not None:
        entry.accepted_parent_messages[message_id] = existing
    if existing is not None:
      if not _same_acceptance(existing, candidate):
        return None, {
          "code": "message_id_conflict",
          "message": "message_id was already accepted with different content",
        }
      if not entry.completed and message_id not in entry.delivered_messages:
        await entry.message_inbox.put(existing)
        entry.delivered_messages.add(message_id)
      return _message_result(existing, task_id=entry.task_id), None
    if message_id in entry.delivered_messages:
      return None, {
        "code": "message_integrity_error",
        "message": "queued message identity lacks its accepted content",
      }
    if entry.completed:
      return None, {
        "code": "already_completed",
        "message": f"Agent {entry.task_id} already finished",
      }

    sent_seq: int | None = None
    append_durable_event = getattr(runner, "_append_durable_event", None)
    if append_durable_event is not None:
      correlation = task_correlation_payload(
        entry,
        runner_id=getattr(runner, "_runner_id", None),
        role=getattr(runner, "_role", "writer"),
      )
      if correlation.get("capability_bind") is None:
        correlation.pop("capability_bind", None)
      durable_entry = await append_durable_event({
        **correlation,
        **_durable_message_fields(runner, candidate),
      })
      durable_seq = getattr(durable_entry, "seq", None)
      if type(durable_seq) is int and durable_seq > 0:
        sent_seq = durable_seq

    accepted = _with_seq(candidate, task_id=entry.task_id, sent_seq=sent_seq)
    entry.accepted_parent_messages[message_id] = accepted
    try:
      await entry.message_inbox.put(accepted)
    except BaseException:
      if sent_seq is None:
        entry.accepted_parent_messages.pop(message_id, None)
      raise
    entry.delivered_messages.add(message_id)
    return _message_result(accepted, task_id=entry.task_id), None


def _durable_message_fields(runner: Any, candidate: ParentMessage) -> dict[str, Any]:
  fields: dict[str, Any] = {
    "type": "parent_message_sent",
    "message_id": candidate.message_id,
    "sender": _sender_payload(runner, candidate.sender_job_id),
    "sent_at": candidate.sent_at,
    "message": candidate.text,
  }
  if candidate.handle_refs:
    fields["handle_refs"] = list(candidate.handle_refs)
  if candidate.evidence:
    fields["evidence"] = True
  return fields


def _with_seq(
  candidate: ParentMessage,
  *,
  task_id: str,
  sent_seq: int | None,
) -> ParentMessage:
  return ParentMessage(
    message_id=candidate.message_id,
    text=candidate.text,
    sent_at=candidate.sent_at,
    task_id=task_id,
    sent_seq=sent_seq,
    sender_job_id=candidate.sender_job_id,
    handle_refs=candidate.handle_refs,
    evidence=candidate.evidence,
  )


async def _accept_on_root(
  runner: Any,
  *,
  registry: TaskRegistry,
  root_job_id: str,
  candidate: ParentMessage,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """Accept one message for the tree root, which has no task entry."""

  inbox = registry.owner_job_inbox()
  if inbox is None:
    return None, {
      "code": "not_available",
      "message": f"Job {root_job_id} has no mailbox in this runtime",
    }
  message_id = candidate.message_id
  async with registry.owner_message_lock:
    existing = registry.accepted_owner_messages.get(message_id)
    if existing is not None:
      if not _same_acceptance(existing, candidate):
        return None, {
          "code": "message_id_conflict",
          "message": "message_id was already accepted with different content",
        }
      return _message_result(existing, task_id=root_job_id), None
    if inbox.full():
      return None, {
        "code": "mailbox_full",
        "message": f"Job {root_job_id}'s mailbox is full; send again later",
      }
    sent_seq: int | None = None
    append_durable_event = getattr(runner, "_append_durable_event", None)
    if append_durable_event is not None:
      durable_entry = await append_durable_event({
        "task_id": root_job_id,
        **_durable_message_fields(runner, candidate),
      })
      durable_seq = getattr(durable_entry, "seq", None)
      if type(durable_seq) is int and durable_seq > 0:
        sent_seq = durable_seq
    accepted = _with_seq(candidate, task_id=root_job_id, sent_seq=sent_seq)
    registry.accepted_owner_messages[message_id] = accepted
    inbox.put_nowait(accepted)
    return _message_result(accepted, task_id=root_job_id), None


async def enqueue_parent_message(
  runner: Any,
  *,
  task_id: str,
  message: str,
  message_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """Accept one parent steer for a task in ``runner``'s own registry.

  Used by the workflow controller to steer its own node; job-to-job sends go
  through :func:`send_job_message`.
  """

  invalid = _invalid_message(message)
  if invalid is not None:
    return None, invalid
  registry = getattr(runner, "_task_registry", None)
  if registry is None:
    return None, {"code": "not_available", "message": "Task registry not configured"}
  entry = registry.get(task_id)
  if entry is None:
    return None, {"code": "not_found", "message": f"No running agent: {task_id}"}
  if message_id is not None and not _canonical_message_id(message_id):
    return None, {
      "code": "invalid_input",
      "message": "message_id must be bounded canonical non-empty text",
    }
  return await _accept_on_entry(
    runner,
    entry=entry,
    candidate=ParentMessage(
      message_id=message_id or str(uuid.uuid4()),
      text=message,
      sent_at=datetime.datetime.now(datetime.UTC).timestamp(),
    ),
  )


async def send_job_message(
  runner: Any,
  *,
  sender: ParentContext,
  target: str,
  message: str,
  handle_refs: tuple[str, ...] = (),
  message_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """Admit and deliver one job-to-job message (design §3.1 MessageRequest)."""

  invalid = _invalid_message(message)
  if invalid is not None:
    return None, invalid
  if message_id is not None and not _canonical_message_id(message_id):
    return None, {
      "code": "invalid_input",
      "message": "message_id must be bounded canonical non-empty text",
    }
  registry = getattr(runner, "_task_registry", None)
  if not isinstance(registry, TaskRegistry):
    return None, {"code": "not_available", "message": "Task registry not configured"}
  if sender.job_id is None:
    return None, {
      "code": "not_available",
      "message": "The sending job has no job identity",
    }
  try:
    recipient: RosterJob | None = registry.find_on_roster(target)
  except RosterAmbiguous as ambiguous:
    return None, {
      "code": "ambiguous_target",
      "message": (
        f"Multiple running agents named '{target}': {ambiguous}. "
        "Use task_id instead."
      ),
    }
  if recipient is None:
    return None, {
      "code": "not_on_roster",
      "message": f"No job {target!r} in this task tree",
    }

  durable_log = getattr(runner, "_agent_session_log", None)
  outputs = (
    await job_output_handles(durable_log, job_id=sender.job_id)
    if durable_log is not None
    else frozenset()
  )
  admitted = admit(
    MessageRequest(
      from_job_id=sender.job_id,
      to_job_id=recipient.job_id,
      body=message,
      handle_refs=handle_refs,
    ),
    sender,
    owners=MessageOwners(recipient_tenant=recipient.tenant, outputs=outputs),
  )
  if isinstance(admitted, Refusal):
    return None, {
      "code": admitted.reason,
      "message": admitted.detail,
      "owner": admitted.owner,
    }

  candidate = ParentMessage(
    message_id=message_id or str(uuid.uuid4()),
    text=message,
    sent_at=datetime.datetime.now(datetime.UTC).timestamp(),
    sender_job_id=sender.job_id,
    handle_refs=handle_refs,
    # The parent's own send to its child is its instruction; anything else
    # (child -> parent, sibling -> sibling) is another agent's evidence.
    evidence=recipient.parent_job_id != sender.job_id,
  )
  if recipient.entry is None:
    return await _accept_on_root(
      runner,
      registry=recipient.registry,
      root_job_id=recipient.job_id,
      candidate=candidate,
    )
  return await _accept_on_entry(runner, entry=recipient.entry, candidate=candidate)


def _parse_handle_refs(raw: Any) -> tuple[str, ...] | None:
  if raw is None:
    return ()
  if not isinstance(raw, list) or any(not is_canonical_handle_ref(ref) for ref in raw):
    return None
  return tuple(dict.fromkeys(raw))


def make_send_message_handler(
  runner_ref: list[Any],
  *,
  caller: CallerContext,
) -> JobScopedHandler:
  """Build the ``send_message`` handler for one executing job.

  ``caller`` returns the executing job's admitted context — a child's
  ``GatewaySession.admitted_job_context`` or the turn's root — never a
  closure over its parent's authority.
  """

  return JobScopedHandler(
    lambda scope: _send_message_handler(runner_ref, scope.caller),
    JobScope(caller=caller, spill_sink=lambda: None),
  )


def _send_message_handler(runner_ref: list[Any], caller: CallerContext):
  async def _handle_send_message(tool_input: dict[str, Any], **kwargs: Any):
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Runner not initialized"}
    sender = caller()
    if sender is None:
      return None, {
        "code": "not_available",
        "message": "send_message has no admitted job context in this runtime",
      }

    target = tool_input.get("to", "")
    if isinstance(target, str):
      target = target.strip()
    if not target or not isinstance(target, str):
      return None, {"code": "invalid_input", "message": "'to' is required"}

    message = tool_input.get("message", "")
    if isinstance(message, str):
      message = message.strip()
    if not message or not isinstance(message, str):
      return None, {"code": "invalid_input", "message": "'message' is required"}

    handle_refs = _parse_handle_refs(tool_input.get("handle_refs"))
    if handle_refs is None:
      return None, {
        "code": "invalid_input",
        "message": "'handle_refs' must be a list of handle strings",
      }

    tool_ctx = kwargs.get("tool_ctx")
    trusted_message_id = getattr(tool_ctx, "tool_call_id", None)
    if tool_ctx is not None and not _canonical_message_id(trusted_message_id):
      return None, {
        "code": "message_control_identity_invalid",
        "message": "send_message requires bounded trusted tool-call identity",
      }
    return await send_job_message(
      runner,
      sender=sender,
      target=target,
      message=message,
      handle_refs=handle_refs,
      message_id=(
        trusted_message_id
        if trusted_message_id is not None
        else tool_input.get("message_id")
      ),
    )

  return _handle_send_message


__all__ = [
  "PARENT_MESSAGE_MAX_BYTES",
  "PARENT_MESSAGE_MAX_CHARS",
  "PARENT_MESSAGE_ID_MAX_BYTES",
  "PARENT_MESSAGE_ID_MAX_CHARS",
  "enqueue_parent_message",
  "make_send_message_handler",
  "send_job_message",
]
