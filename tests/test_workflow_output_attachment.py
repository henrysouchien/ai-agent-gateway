from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from agent_gateway.workflow_output_attachment import (
  WorkflowOutputAttachment,
  WorkflowOutputAttachmentError,
  accepted_workflow_continuation_run_id,
  completed_workflow_output_attachment,
  record_workflow_output_attachment,
)
from agent_workflow_contracts import (
  ActivityHandle,
  AdmittedPlanRef,
  AuthoredDeliverySummary,
  ContentHandle,
  ContractRef,
  ContinuationState,
  DeliveryEnvelope,
  DeliveryPrimary,
  DeliverySettlement,
  DeliveryWarning,
  PublishedInlineView,
  PublishedOutput,
  PublishedOutputRef,
  TerminalPhaseRevision,
  TranscriptHandle,
  WorkflowDeliverySpec,
  WorkflowResult,
)


def _contract(name: str) -> ContractRef:
  digest = hashlib.sha256(f"workflow-output/{name}/1.0".encode()).hexdigest()
  return ContractRef(
    namespace="workflow-output",
    name=name,
    version="1.0",
    digest=f"sha256:{digest}",
  )


def _publication(
  name: str,
  text: str,
  *,
  phase: int = 2,
  revision: int = 2,
  inline: bool,
) -> PublishedOutput:
  payload = text.encode("utf-8")
  digest = hashlib.sha256(payload).hexdigest()
  contract = _contract(name)
  return PublishedOutput(
    name=name,
    output_id=(
      f"wout:workflow-1:phase:{phase}:revision:{revision}:{name}"
    ),
    contract=contract,
    content=ContentHandle(
      content_id=f"sha256:{digest}",
      content_sha256=digest,
      content_chars=len(text),
      content_bytes=len(payload),
      contract=contract,
      media_type="text/markdown; charset=utf-8",
      encoding="utf-8",
      retention="durable",
    ),
    inline_view=PublishedInlineView(value=text) if inline else None,
  )


def _workflow_result(*, presentation: str = "attachment") -> WorkflowResult:
  primary = _publication("synthesis", "Full report.", inline=False)
  summary_output = _publication(
    "delivery_summary",
    "Grounded authored summary.",
    inline=True,
  )
  spec = WorkflowDeliverySpec(
    presentation=presentation,
    primary_selector="synthesis",
    summary_selector=(
      "delivery_summary" if presentation == "attachment" else None
    ),
  )
  envelope = DeliveryEnvelope(
    workflow_run_id="workflow-1",
    phase_number=2,
    revision=2,
    summary=(
      AuthoredDeliverySummary(
        text="Grounded authored summary.",
        source=PublishedOutputRef.from_output(summary_output),
      )
      if presentation == "attachment"
      else None
    ),
    primary=DeliveryPrimary(
      name="synthesis",
      published_output_ref=PublishedOutputRef.from_output(primary),
    ),
  )
  return WorkflowResult(
    workflow_run_id="workflow-1",
    admitted_plan_ref=AdmittedPlanRef(
      workflow_run_id="workflow-1",
      plan_id="plan-2",
      phase_number=2,
      revision=2,
      digest=f"sha256:{'a' * 64}",
    ),
    terminal_phase_revision=TerminalPhaseRevision(
      phase_number=2,
      revision=2,
    ),
    execution_status="succeeded",
    published_outputs=(
      (primary, summary_output) if presentation == "attachment" else (primary,)
    ),
    delivery=DeliverySettlement(
      status="complete",
      phase_number=2,
      revision=2,
      spec=spec,
      envelope=envelope,
    ),
    transcript=TranscriptHandle(
      kind="workflow_transcript",
      owner_id="workflow-1",
    ),
    activity=ActivityHandle(
      kind="workflow_activity",
      owner_id="workflow-1",
    ),
    continuation_state=ContinuationState(status="exhausted"),
  )


def _completed_result(*, presentation: str = "attachment") -> dict[str, object]:
  return {
    "ok": True,
    "action": "result",
    **_workflow_result(presentation=presentation).model_dump(mode="json"),
  }


def test_completed_workflow_result_yields_atomic_authored_delivery() -> None:
  workflow_result = _workflow_result()
  attachment = completed_workflow_output_attachment(
    "workflow_run",
    {
      "ok": True,
      "action": "result",
      **workflow_result.model_dump(mode="json"),
    },
  )

  assert attachment == WorkflowOutputAttachment(
    envelope=workflow_result.delivery.envelope,
  )
  assert attachment is not None
  assert attachment.envelope.summary is not None
  assert attachment.envelope.summary.text == "Grounded authored summary."
  assert attachment.delivery_phase_number == 2
  assert attachment.delivery_revision == 2
  assert attachment.to_dict()["read"] == {
    "action": "output",
    "workflow_run_id": "workflow-1",
    "output_id": (
      "wout:workflow-1:phase:2:revision:2:synthesis"
    ),
  }


@pytest.mark.parametrize(
  ("field_name", "value"),
  [("action", "observe"), ("ok", False)],
)
def test_non_result_response_does_not_attach(
  field_name: str,
  value: object,
) -> None:
  result = _completed_result()
  result[field_name] = value

  assert completed_workflow_output_attachment("workflow_run", result) is None


def test_failed_delivery_does_not_attach() -> None:
  result = _completed_result()
  result["delivery"] = {
    "status": "failed",
    "phase_number": 2,
    "revision": 2,
    "spec": {
      "presentation": "attachment",
      "primary_selector": "synthesis",
      "summary_selector": "delivery_summary",
      "additional_selectors": [],
    },
    "envelope": None,
    "failure": {
      "code": "delivery_outputs_missing",
      "message": "Missing output.",
      "missing_outputs": ["delivery_summary"],
    },
  }

  assert completed_workflow_output_attachment("workflow_run", result) is None


def test_malformed_completed_result_fails_closed() -> None:
  result = deepcopy(_completed_result())
  result["delivery"]["envelope"]["summary"]["text"] = "Invented summary."  # type: ignore[index]

  with pytest.raises(
    WorkflowOutputAttachmentError,
    match="violates WorkflowResult",
  ):
    completed_workflow_output_attachment("workflow_run", result)


def test_inline_delivery_preserves_inline_mode_without_attachment() -> None:
  assert completed_workflow_output_attachment(
    "workflow_run",
    _completed_result(presentation="inline"),
  ) is None


def test_warned_primary_only_delivery_yields_no_attachment() -> None:
  base = _workflow_result()
  spec = base.delivery.spec
  assert spec is not None
  envelope = DeliveryEnvelope(
    workflow_run_id=base.workflow_run_id,
    phase_number=2,
    revision=2,
    summary=None,
    primary=base.delivery.envelope.primary,
  )
  degraded = base.model_copy(update={
    "delivery": DeliverySettlement(
      status="complete",
      phase_number=2,
      revision=2,
      spec=spec,
      envelope=envelope,
      warning=DeliveryWarning(
        code="delivery_summary_oversized",
        message=(
          "The authored delivery summary exceeds the inline presentation "
          "bound; only the exact primary output is delivered."
        ),
        omitted_outputs=("delivery_summary",),
      ),
    ),
  })

  assert completed_workflow_output_attachment(
    "workflow_run",
    {
      "ok": True,
      "action": "result",
      **degraded.model_dump(mode="json"),
    },
  ) is None


def test_attachment_mapping_round_trips_exactly_and_rejects_unknown_fields() -> None:
  attachment = WorkflowOutputAttachment(
    envelope=_workflow_result().delivery.envelope,
  )
  serialized = attachment.to_dict()

  assert WorkflowOutputAttachment.from_mapping(serialized) == attachment

  with pytest.raises(
    WorkflowOutputAttachmentError,
    match="unknown fields",
  ):
    WorkflowOutputAttachment.from_mapping({**serialized, "summary": "shadow"})


def test_record_orders_by_phase_and_revision_and_rejects_conflicts() -> None:
  first_result = _workflow_result()
  first = WorkflowOutputAttachment(envelope=first_result.delivery.envelope)
  pending: dict[str, WorkflowOutputAttachment] = {}
  record_workflow_output_attachment(pending, first)

  old_envelope = first.envelope.model_copy(
    update={
      "phase_number": 1,
      "revision": 1,
    }
  )
  # Reconstructing validates the phase-bound logical IDs, so use the existing
  # exact envelope only to prove same-revision conflicts and an object with a
  # lower sortable revision through model_construct for the stale fast path.
  older = WorkflowOutputAttachment(
    envelope=DeliveryEnvelope.model_construct(**old_envelope.__dict__),
  )
  record_workflow_output_attachment(pending, older)
  assert pending == {"workflow-1": first}

  conflict_summary = first.envelope.summary
  assert conflict_summary is not None
  conflict = WorkflowOutputAttachment(
    envelope=first.envelope.model_copy(
      update={
        "summary": conflict_summary.model_copy(
          update={"text": "Conflicting summary."}
        )
      }
    )
  )
  with pytest.raises(
    WorkflowOutputAttachmentError,
    match="changed identity within one delivery revision",
  ):
    record_workflow_output_attachment(pending, conflict)


def test_accepted_continuation_names_the_superseded_run() -> None:
  assert accepted_workflow_continuation_run_id(
    "workflow_run",
    {
      "ok": True,
      "action": "continue",
      "workflow_run_id": "workflow-1",
      "state": "running",
    },
  ) == "workflow-1"


@pytest.mark.parametrize(
  "result",
  [
    None,
    {"ok": True, "action": "result", "workflow_run_id": "workflow-1"},
    {"ok": False, "action": "continue", "workflow_run_id": "workflow-1"},
    {"ok": True, "action": "continue"},
    {"ok": True, "action": "continue", "workflow_run_id": ""},
    {"ok": True, "action": "continue", "workflow_run_id": 7},
  ],
)
def test_unaccepted_continuation_supersedes_nothing(result: object) -> None:
  assert accepted_workflow_continuation_run_id("workflow_run", result) is None


def test_non_workflow_tool_never_supersedes_pending_attachments() -> None:
  assert accepted_workflow_continuation_run_id(
    "run_agent",
    {"ok": True, "action": "continue", "workflow_run_id": "workflow-1"},
  ) is None
