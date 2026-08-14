"""Atomic assistant-message delivery of normalized workflow outputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Literal, Mapping

from agent_workflow_contracts import (
  DeliveryEnvelope,
  PublishedOutput,
  PublishedOutputRef,
  WorkflowResult,
)


class WorkflowOutputAttachmentError(ValueError):
  """A completed workflow result cannot produce its required attachment."""


@dataclass(frozen=True, slots=True)
class WorkflowOutputAttachment:
  """One atomic authored summary plus its lossless primary output reference.

  The envelope is retained as a unit so phase, revision, summary provenance,
  primary identity, and any additional output identities cannot drift while an
  assistant message is waiting to be persisted or rendered.
  """

  envelope: DeliveryEnvelope
  kind: Literal["workflow_primary_output"] = "workflow_primary_output"

  def __post_init__(self) -> None:
    if self.kind != "workflow_primary_output":
      raise WorkflowOutputAttachmentError(
        "workflow output attachment has the wrong kind"
      )
    if not isinstance(self.envelope, DeliveryEnvelope):
      raise WorkflowOutputAttachmentError(
        "workflow output attachment requires a DeliveryEnvelope"
      )
    if self.envelope.summary is None:
      raise WorkflowOutputAttachmentError(
        "workflow output attachment requires an authored summary"
      )

  @property
  def workflow_run_id(self) -> str:
    return self.envelope.workflow_run_id

  @property
  def output_name(self) -> str:
    return self.envelope.primary.name

  @property
  def published_output_ref(self) -> PublishedOutputRef:
    return self.envelope.primary.published_output_ref

  @property
  def output_id(self) -> str:
    return self.published_output_ref.output_id

  @property
  def content_sha256(self) -> str:
    return self.published_output_ref.content.content_sha256

  @property
  def content_chars(self) -> int:
    value = self.published_output_ref.content.content_chars
    if value is None:  # pragma: no cover - workflow output transport is textual
      raise WorkflowOutputAttachmentError(
        "workflow primary output requires textual content metadata"
      )
    return value

  @property
  def content_bytes(self) -> int:
    return self.published_output_ref.content.content_bytes

  @property
  def encoding(self) -> str:
    value = self.published_output_ref.content.encoding
    if value is None:  # pragma: no cover - workflow output transport is textual
      raise WorkflowOutputAttachmentError(
        "workflow primary output requires textual encoding metadata"
      )
    return value

  @property
  def media_type(self) -> str:
    return self.published_output_ref.content.media_type

  @property
  def delivery_phase_number(self) -> int:
    return self.envelope.phase_number

  @property
  def delivery_revision(self) -> int:
    return self.envelope.revision

  @property
  def read(self) -> dict[str, str]:
    return {
      "action": "output",
      "workflow_run_id": self.workflow_run_id,
      "output_id": self.output_id,
    }

  def to_dict(self) -> dict[str, Any]:
    return {
      "kind": self.kind,
      "delivery_envelope": self.envelope.model_dump(mode="json"),
      "read": self.read,
    }

  @classmethod
  def from_mapping(cls, value: Mapping[str, Any]) -> "WorkflowOutputAttachment":
    if not isinstance(value, Mapping):
      raise WorkflowOutputAttachmentError(
        "workflow output attachment must be a mapping"
      )
    if value.get("kind") != "workflow_primary_output":
      raise WorkflowOutputAttachmentError(
        "workflow output attachment has the wrong kind"
      )
    if set(value) != {"kind", "delivery_envelope", "read"}:
      raise WorkflowOutputAttachmentError(
        "workflow output attachment has unknown fields"
      )
    raw_envelope = value.get("delivery_envelope")
    try:
      envelope = DeliveryEnvelope.model_validate(raw_envelope)
    except Exception as exc:
      raise WorkflowOutputAttachmentError(
        "workflow output attachment has an invalid delivery envelope"
      ) from exc
    attachment = cls(envelope=envelope)
    if value.get("read") != attachment.read:
      raise WorkflowOutputAttachmentError(
        "workflow output attachment has an invalid read recipe"
      )
    return attachment


def completed_workflow_output_attachment(
  tool_name: str,
  result: object,
) -> WorkflowOutputAttachment | None:
  """Extract an attachment only from a validated aggregate workflow result.

  Inline presentation intentionally yields no attachment. Attachment
  presentation fails closed unless its authored summary and primary reference
  both match publications from the same terminal phase and revision.
  """

  if tool_name != "workflow_run" or not isinstance(result, Mapping):
    return None
  if result.get("ok") is not True or result.get("action") != "result":
    return None
  try:
    workflow_result = WorkflowResult.model_validate({
      key: value
      for key, value in result.items()
      if key not in {"ok", "action"}
    })
  except Exception as exc:
    raise WorkflowOutputAttachmentError(
      "completed workflow result violates WorkflowResult"
    ) from exc
  settlement = workflow_result.delivery
  if settlement.status != "complete":
    return None
  if settlement.spec is None:
    raise WorkflowOutputAttachmentError(
      "completed workflow delivery requires its admitted spec"
    )
  if settlement.spec.presentation == "inline":
    return None
  envelope = settlement.envelope
  if envelope is None:
    raise WorkflowOutputAttachmentError(
      "attachment delivery requires an authored summary and primary output"
    )
  if envelope.summary is None:
    # A validated complete delivery without its authored summary carries an
    # explicit warning (oversized summary fallback); the exact primary output
    # remains readable through the workflow output route, so no assistant
    # attachment is staged.
    if settlement.warning is None:
      raise WorkflowOutputAttachmentError(
        "attachment delivery requires an authored summary and primary output"
      )
    return None
  attachment = WorkflowOutputAttachment(envelope=envelope)
  publications = {
    publication.output_id: publication
    for publication in workflow_result.published_outputs
  }
  _require_publication(
    publications,
    envelope.primary.published_output_ref,
    field_name="primary",
  )
  summary_publication = _require_publication(
    publications,
    envelope.summary.source,
    field_name="summary",
  )
  _require_exact_summary(envelope.summary.text, summary_publication)
  return attachment


def accepted_workflow_continuation_run_id(
  tool_name: str,
  result: object,
) -> str | None:
  """Identify the run whose pending attachment an accepted continuation stales.

  A durably accepted ``workflow_run(action="continue")`` opens a newer phase
  revision, so any staged prior-revision attachment no longer describes the
  run's terminal delivery. The pending value must be invalidated here rather
  than waiting for a newer complete delivery to replace it: a failed
  continuation delivery produces no replacement attachment, and the stale
  revision would otherwise be emitted on the next final assistant turn.
  """

  if tool_name != "workflow_run" or not isinstance(result, Mapping):
    return None
  if result.get("ok") is not True or result.get("action") != "continue":
    return None
  workflow_run_id = result.get("workflow_run_id")
  if not isinstance(workflow_run_id, str) or not workflow_run_id:
    return None
  return workflow_run_id


def record_workflow_output_attachment(
  pending: dict[str, WorkflowOutputAttachment],
  attachment: WorkflowOutputAttachment,
) -> None:
  """Keep the latest atomic phase revision, rejecting identity conflicts."""

  existing = pending.get(attachment.workflow_run_id)
  if existing is None:
    pending[attachment.workflow_run_id] = attachment
    return
  new_revision = (
    attachment.delivery_phase_number,
    attachment.delivery_revision,
  )
  existing_revision = (
    existing.delivery_phase_number,
    existing.delivery_revision,
  )
  if new_revision < existing_revision:
    return
  if new_revision == existing_revision and attachment != existing:
    raise WorkflowOutputAttachmentError(
      "workflow output attachment changed identity within one delivery revision"
    )
  pending[attachment.workflow_run_id] = attachment


def _require_publication(
  publications: Mapping[str, PublishedOutput],
  reference: PublishedOutputRef,
  *,
  field_name: str,
) -> PublishedOutput:
  publication = publications.get(reference.output_id)
  if publication is None:
    raise WorkflowOutputAttachmentError(
      f"completed workflow {field_name} is not a published output"
    )
  expected = PublishedOutputRef(
    output_id=publication.output_id,
    contract=publication.contract,
    content=publication.content,
  )
  if expected != reference:
    raise WorkflowOutputAttachmentError(
      f"completed workflow {field_name} conflicts with published_outputs"
    )
  return publication


def _require_exact_summary(text: str, publication: PublishedOutput) -> None:
  payload = text.encode("utf-8")
  content = publication.content
  if (
    publication.inline_view is None
    or publication.inline_view.value != text
    or content.content_id != f"sha256:{hashlib.sha256(payload).hexdigest()}"
    or content.content_sha256 != hashlib.sha256(payload).hexdigest()
    or content.content_chars != len(text)
    or content.content_bytes != len(payload)
  ):
    raise WorkflowOutputAttachmentError(
      "completed workflow summary text conflicts with its published content"
    )


__all__ = [
  "WorkflowOutputAttachment",
  "WorkflowOutputAttachmentError",
  "accepted_workflow_continuation_run_id",
  "completed_workflow_output_attachment",
  "record_workflow_output_attachment",
]
