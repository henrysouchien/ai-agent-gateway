"""Authenticated, lossless delivery of canonical workflow attachments."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from .agent_session_log import AgentSessionLog
from .session import AuthManager
from .workflow_output_attachment import (
  WorkflowOutputAttachment,
  WorkflowOutputAttachmentError,
)

log = logging.getLogger("agent_gateway.server_workflow_output_routes")


class _OutputIntegrityFailure(WorkflowOutputAttachmentError):
  """Internal integrity failure naming exactly which comparison failed.

  The external response stays the generic non-sensitive 409; the reason code
  and compared logical identities exist only for retained internal logs so an
  exact-delivery failure is diagnosable without reproducing a live session.
  """

  def __init__(self, reason_code: str, message: str) -> None:
    super().__init__(message)
    self.reason_code = reason_code


async def workflow_output_response(
  request: Request,
  workflow_run_id: str,
  output_id: str,
  *,
  auth: AuthManager,
  download: bool = False,
) -> Response:
  """Serve exact canonical bytes only for a durable attachment in this session."""

  token = AuthManager.get_bearer_token(
    request.headers.get("Authorization")
  )
  session = auth.verify_token(token)
  session_log = getattr(session, "agent_session_log", None)
  if (
    type(session_log) is not AgentSessionLog
    or session_log.gateway_session is not session
  ):
    return JSONResponse(
      {
        "error": "workflow_output_unavailable",
        "message": "Workflow output delivery is unavailable for this session.",
      },
      status_code=404,
    )

  try:
    attachment = await _find_attachment(
      session_log,
      workflow_run_id=workflow_run_id,
      output_id=output_id,
    )
  except WorkflowOutputAttachmentError as exc:
    _log_integrity_failure(
      exc,
      stage="durable_attachment_lookup",
      workflow_run_id=workflow_run_id,
      output_id=output_id,
      attachment=None,
    )
    return _integrity_error()
  if attachment is None:
    return JSONResponse(
      {
        "error": "workflow_output_not_found",
        "message": "Workflow output attachment was not found in this session.",
      },
      status_code=404,
    )

  reader = session.workflow_output_reader
  if not callable(reader):
    return JSONResponse(
      {
        "error": "workflow_output_unavailable",
        "message": "Canonical workflow output is not available for reading.",
      },
      status_code=503,
    )
  try:
    content = await _materialize_verified_content(
      reader,
      attachment=attachment,
    )
  except WorkflowOutputAttachmentError as exc:
    _log_integrity_failure(
      exc,
      stage="materialize_verified_content",
      workflow_run_id=workflow_run_id,
      output_id=output_id,
      attachment=attachment,
    )
    return _integrity_error()
  except Exception:
    return JSONResponse(
      {
        "error": "workflow_output_unavailable",
        "message": "Canonical workflow output could not be materialized.",
      },
      status_code=503,
    )

  disposition = "attachment" if download else "inline"
  filename = f'{attachment.output_name}{_filename_suffix(attachment.media_type)}'
  return Response(
    content=content.encode("utf-8"),
    media_type=attachment.media_type.split(";", 1)[0],
    headers={
      "Cache-Control": "private, no-store",
      "Content-Disposition": f'{disposition}; filename="{filename}"',
      "ETag": f'"sha256:{attachment.content_sha256}"',
      "X-Content-SHA256": attachment.content_sha256,
      "X-Workflow-Output-Id": attachment.output_id,
      "Access-Control-Expose-Headers": (
        "X-Content-SHA256, X-Workflow-Output-Id, ETag, Content-Disposition"
      ),
    },
  )


async def _find_attachment(
  session_log: AgentSessionLog,
  *,
  workflow_run_id: str,
  output_id: str,
) -> WorkflowOutputAttachment | None:
  entries, _ = await session_log.query(
    event_types={"assistant_message"},
    order="desc",
  )
  found: WorkflowOutputAttachment | None = None
  for entry in entries:
    raw_attachments = entry.event.get("workflow_output_attachments")
    if not isinstance(raw_attachments, list):
      continue
    for raw in raw_attachments:
      if not isinstance(raw, Mapping):
        continue
      try:
        attachment = _attachment_from_mapping(raw)
      except WorkflowOutputAttachmentError:
        # A malformed object claiming the requested identity is an integrity
        # failure. Unrelated objects are ignored only when their normalized
        # envelope can be validated and shown to name another output.
        envelope = raw.get("delivery_envelope")
        if not isinstance(envelope, Mapping):
          continue
        if envelope.get("workflow_run_id") == workflow_run_id:
          raise
        continue
      if (
        attachment.workflow_run_id != workflow_run_id
        or attachment.output_id != output_id
      ):
        continue
      if found is not None and found != attachment:
        raise _OutputIntegrityFailure(
          "durable_attachment_identity_conflict",
          "durable workflow output attachment identity conflicts",
        )
      found = attachment
  return found


def _attachment_from_mapping(
  raw: Mapping[str, Any],
) -> WorkflowOutputAttachment:
  return WorkflowOutputAttachment.from_mapping(raw)


async def _materialize_verified_content(
  reader: Any,
  *,
  attachment: WorkflowOutputAttachment,
) -> str:
  pieces: list[str] = []
  after_char = 0
  while True:
    raw_page = await reader(
      attachment.workflow_run_id,
      attachment.output_id,
      after_char,
    )
    page = _page_mapping(raw_page)
    source = page.get("source")
    authorization = page.get("authorization")
    if (
      page.get("ok") is not True
      or page.get("action") != "output"
      or page.get("workflow_run_id") != attachment.workflow_run_id
      or page.get("output_id") != attachment.output_id
      or page.get("view") != "paged_exact_content"
    ):
      raise _OutputIntegrityFailure(
        "page_envelope_mismatch",
        "workflow output page conflicts with its durable attachment",
      )
    if (
      not isinstance(source, Mapping)
      or source != attachment.published_output_ref.content.model_dump(mode="json")
    ):
      raise _OutputIntegrityFailure(
        "source_handle_mismatch",
        "workflow output page conflicts with its durable attachment",
      )
    if (
      not isinstance(authorization, Mapping)
      or authorization.get("kind") != "published_output"
      or authorization.get("workflow_run_id") != attachment.workflow_run_id
      or authorization.get("phase_number") != attachment.delivery_phase_number
      or authorization.get("revision") != attachment.delivery_revision
      or authorization.get("output_id") != attachment.output_id
    ):
      raise _OutputIntegrityFailure(
        "authorization_mismatch",
        "workflow output page conflicts with its durable attachment",
      )
    if (
      page.get("encoding") != attachment.encoding
      or page.get("content_sha256") != attachment.content_sha256
      or page.get("content_chars") != attachment.content_chars
      or page.get("content_bytes") != attachment.content_bytes
    ):
      raise _OutputIntegrityFailure(
        "content_identity_mismatch",
        "workflow output page conflicts with its durable attachment",
      )
    if page.get("after_char") != after_char:
      raise _OutputIntegrityFailure(
        "paging_cursor_mismatch",
        "workflow output page conflicts with its durable attachment",
      )
    chunk = page.get("content")
    if not isinstance(chunk, str) or not chunk:
      raise _OutputIntegrityFailure(
        "empty_page",
        "workflow output reader returned an empty page",
      )
    pieces.append(chunk)
    next_after = after_char + len(chunk)
    end = page.get("end")
    raw_cursor = page.get("next_cursor")
    if end is True:
      if raw_cursor is not None or next_after != attachment.content_chars:
        raise _OutputIntegrityFailure(
          "terminal_cursor_invalid",
          "workflow output terminal page has an invalid cursor",
        )
      break
    if end is not False or not isinstance(raw_cursor, Mapping):
      raise _OutputIntegrityFailure(
        "continuation_cursor_invalid",
        "workflow output page has an invalid continuation cursor",
      )
    cursor_after = raw_cursor.get("after_char")
    if cursor_after != next_after or next_after >= attachment.content_chars:
      raise _OutputIntegrityFailure(
        "paging_advance_mismatch",
        "workflow output page did not advance exactly",
      )
    after_char = next_after

  content = "".join(pieces)
  payload = content.encode("utf-8")
  if (
    len(content) != attachment.content_chars
    or len(payload) != attachment.content_bytes
    or hashlib.sha256(payload).hexdigest() != attachment.content_sha256
  ):
    raise _OutputIntegrityFailure(
      "materialized_content_mismatch",
      "materialized workflow output failed identity verification",
    )
  return content


def _page_mapping(page: object) -> Mapping[str, Any]:
  if isinstance(page, Mapping):
    return page
  model_dump = getattr(page, "model_dump", None)
  if callable(model_dump):
    dumped = model_dump(mode="json")
    if isinstance(dumped, Mapping):
      return dumped
  raise _OutputIntegrityFailure(
    "invalid_page_shape",
    "workflow output reader returned an invalid page",
  )


def _log_integrity_failure(
  exc: WorkflowOutputAttachmentError,
  *,
  stage: str,
  workflow_run_id: str,
  output_id: str,
  attachment: WorkflowOutputAttachment | None,
) -> None:
  """Record which internal comparison failed, without content or client data.

  Only logical identities are logged (run, phase, revision, output name); the
  external 409 body stays generic and byte-identical.
  """

  reason_code = getattr(exc, "reason_code", "unclassified")
  log.warning(
    "workflow output integrity failed | stage=%s reason=%s "
    "workflow_run_id=%s output_id=%s phase=%s revision=%s output_name=%s",
    stage,
    reason_code,
    workflow_run_id,
    output_id,
    attachment.delivery_phase_number if attachment is not None else None,
    attachment.delivery_revision if attachment is not None else None,
    attachment.output_name if attachment is not None else None,
    extra={
      "data": {
        "event": "workflow_output_integrity_failed",
        "stage": stage,
        "reason_code": reason_code,
        "workflow_run_id": workflow_run_id,
        "output_id": output_id,
        "phase_number": (
          attachment.delivery_phase_number if attachment is not None else None
        ),
        "revision": (
          attachment.delivery_revision if attachment is not None else None
        ),
        "output_name": (
          attachment.output_name if attachment is not None else None
        ),
      }
    },
  )


def _filename_suffix(media_type: str) -> str:
  base = media_type.split(";", 1)[0].strip().lower()
  return {
    "application/json": ".json",
    "text/markdown": ".md",
    "text/plain": ".txt",
  }.get(base, ".txt")


def _integrity_error() -> JSONResponse:
  return JSONResponse(
    {
      "error": "workflow_output_integrity_failed",
      "message": "Canonical workflow output failed identity verification.",
    },
    status_code=409,
  )


__all__ = ["workflow_output_response"]
