from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import JsonValue, ValidationError

from agent_workflow_contracts import (
  AdmittedDataRef,
  AdmittedPlanRef,
  AgentCompletionEnvelope,
  AgentOperationRef,
  AnalyticalOutcome,
  AttemptRef,
  AuthoredDeliverySummary,
  CapabilityBind,
  CanonicalProjection,
  ContentEvidenceRef,
  ContentHandle,
  ContentReadGrant,
  ContinuationState,
  ContractRef,
  DELIVERY_PREVIEW_MAX_BYTES,
  DELIVERY_PREVIEW_POLICY_VERSION,
  DeliveryEnvelopeV1,
  DeliveryEnvelopeV2,
  DeliveryFailure,
  DeliveryPreview,
  DeliveryPrimary,
  DeliveryPrimaryV2,
  DeliverySettlement,
  DeliveryWarning,
  EvidenceObservation,
  ExecutionSettlement,
  LiteralSelector,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  OwnerBinding,
  ParentResultPolicy,
  PhaseOutputSelector,
  ProjectionRequirement,
  PUBLISHED_OUTPUT_INLINE_MAX_BYTES,
  PublishedInlineView,
  PublishedOutput,
  PublishedOutputRef,
  RequestedDataRef,
  ResultRequirement,
  SettlementProjection,
  SettleWithoutExecutionDisposition,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultRef,
  TaskResultValues,
  TerminalPhaseRevision,
  TerminalNarrativeInlineExact,
  TranscriptHandle,
  ActivityHandle,
  UsageObservation,
  WorkflowDeliverySpecV1,
  WorkflowDeliverySpecV2,
  WorkflowResult,
  WorkflowView,
  canonical_json_bytes,
  parse_delivery_envelope,
  parse_workflow_delivery_spec,
)


HEX = "a" * 64
DIGEST = f"sha256:{HEX}"


def contract(name: str = "report", namespace: str = "test") -> ContractRef:
  return ContractRef(namespace=namespace, name=name, version="1.0", digest=DIGEST)


def operation() -> AgentOperationRef:
  return AgentOperationRef(
    namespace="operation",
    name="research",
    version="1.0",
    digest=DIGEST,
  )


def text_handle(text: str, *, name: str = "report") -> ContentHandle:
  raw = text.encode("utf-8")
  sha = hashlib.sha256(raw).hexdigest()
  return ContentHandle(
    content_id=f"sha256:{sha}",
    content_sha256=sha,
    content_bytes=len(raw),
    content_chars=len(text),
    contract=contract(name),
    media_type="text/plain; charset=utf-8",
    encoding="utf-8",
    retention="durable",
  )


def json_handle(value: object, *, name: str = "projection") -> ContentHandle:
  raw = canonical_json_bytes(value)
  sha = hashlib.sha256(raw).hexdigest()
  return ContentHandle(
    content_id=f"sha256:{sha}",
    content_sha256=sha,
    content_bytes=len(raw),
    content_chars=len(raw.decode("utf-8")),
    contract=contract(name),
    media_type="application/json",
    encoding="utf-8",
    retention="durable",
  )


def attempt() -> AttemptRef:
  return AttemptRef(
    attempt_number=1,
    attempt_id="attempt-1",
    physical_task_id="child-1",
  )


def logical_task() -> OrdinaryDelegationTaskRef:
  return OrdinaryDelegationTaskRef(
    delegation_id="delegation-1",
    operation=operation(),
  )


def provenance() -> TaskResultProvenance:
  return TaskResultProvenance(
    admitted_task_digest=DIGEST,
    model_bind_digest=DIGEST,
    capability_binding_digest=DIGEST,
    tool_grant_digest=DIGEST,
  )


def observation() -> TaskObservation:
  return TaskObservation(
    transcript=TranscriptHandle(kind="child_transcript", owner_id="child-1"),
    activity=ActivityHandle(kind="child_activity", owner_id="child-1"),
    usage=UsageObservation(input_tokens=10, output_tokens=20),
  )


def result(text: str = "Exact final report") -> TaskResult:
  return TaskResult(
    task_result_id="task-result-1",
    logical_task=logical_task(),
    attempt=attempt(),
    execution=ExecutionSettlement(status="succeeded"),
    outcome=AnalyticalOutcome(
      disposition="complete",
      assessment_source="domain_tool",
      assessment_rationale="All declared requirements were addressed.",
    ),
    evidence=EvidenceObservation(),
    values=TaskResultValues(terminal_narrative=text_handle(text)),
    observation=observation(),
    provenance=provenance(),
  )


def test_models_are_frozen_extra_forbid_and_round_trip() -> None:
  original = result()
  encoded = original.model_dump_json()
  assert TaskResult.model_validate_json(encoded) == original

  with pytest.raises(ValidationError):
    TaskResult.model_validate({**original.model_dump(), "unknown": True})
  with pytest.raises(ValidationError):
    original.task_result_id = "changed"  # type: ignore[misc]


def test_contract_and_capability_bind_are_full_secret_free_identities() -> None:
  bind = CapabilityBind(
    schema_version="1.0",
    capability_id="node.explore",
    model_key="openai.gpt-5-6",
    provider="openai",
    upstream_model="gpt-5.6",
    adapter="openai.responses",
    protocol_profile="responses.stream",
    route="direct",
    effort="high",
    credential_principal="user",
    credential_ref="user:test-tenant:alice:openai",
    run_mode="interactive",
    registry_revision="2026-08-13.1",
    policy_revision="2026-08-13.1",
    selection_source="explicit_user",
  )
  assert CapabilityBind.from_receipt(bind.receipt()) == bind
  serialized = bind.model_dump_json().lower()
  assert "api_key" not in serialized
  assert "auth_token" not in serialized
  assert bind.credential_ref in serialized
  with pytest.raises(ValidationError):
    ContractRef(namespace="test", name="report", version="1", digest="abc")


def test_content_handle_requires_exact_identity_and_text_metadata() -> None:
  handle = text_handle("hello")
  with pytest.raises(ValidationError, match="content_id"):
    ContentHandle.model_validate({**handle.model_dump(), "content_id": DIGEST})
  with pytest.raises(ValidationError, match="content_chars"):
    ContentHandle.model_validate({**handle.model_dump(), "content_chars": None})


def test_requested_selector_is_discriminated_and_not_authority() -> None:
  requested = RequestedDataRef(
    name="prior_report",
    selector=PhaseOutputSelector(
      phase_number=1,
      revision=1,
      output_name="primary_report",
    ),
    expected_contract=contract(),
  )
  payload = requested.model_dump(mode="json")
  assert payload["selector"]["kind"] == "phase_output_selector"
  assert RequestedDataRef.model_validate(payload) == requested
  assert "workflow_run_id" not in payload["selector"]

  with pytest.raises(ValidationError, match="secret-bearing"):
    LiteralSelector(value={"api_key": "do-not-transport"})
  assert (
    LiteralSelector(value="/tmp/internal-report.json").value
    == "/tmp/internal-report.json"
  )


def test_admitted_ref_binds_owner_contract_content_and_read_grant() -> None:
  content = text_handle("report")
  request = RequestedDataRef(
    name="prior_report",
    selector=PhaseOutputSelector(
      phase_number=1,
      revision=1,
      output_name="primary_report",
    ),
    expected_contract=content.contract,
  )
  grant = ContentReadGrant(
    grant_id="grant-1",
    content_id=content.content_id,
    scope="this_task",
    principal_id="attempt-1",
  )
  admitted = AdmittedDataRef(
    request=request,
    source_kind="phase_output",
    logical_source_id="workflow-1:phase:1:revision:1:primary_report",
    owner=OwnerBinding(tenant_id="tenant-1", workflow_run_id="workflow-1"),
    actual_contract=content.contract,
    content=content,
    read_grant=grant,
  )
  assert AdmittedDataRef.model_validate_json(admitted.model_dump_json()) == admitted
  with pytest.raises(ValidationError, match="must address admitted content"):
    AdmittedDataRef.model_validate(
      {
        **admitted.model_dump(),
        "read_grant": {**grant.model_dump(), "content_id": DIGEST},
      }
    )


def test_agent_result_requirement_is_terminal_message_only() -> None:
  projection = ProjectionRequirement(contract=contract("projection"))
  requirement = ResultRequirement(
    mode="narrative",
    projection=None,
    terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )
  assert ResultRequirement.model_validate(requirement.model_dump()) == requirement
  with pytest.raises(ValidationError):
    ResultRequirement(
      mode="hybrid",
      projection=projection,
      terminal_narrative="required",
      outcome=OutcomeRequirement(required=True, source="domain_tool"),
    )
  with pytest.raises(ValidationError):
    ResultRequirement(
      mode="narrative",
      projection=projection,
      terminal_narrative="required",
      outcome=OutcomeRequirement(required=False, source="none"),
    )
  with pytest.raises(ValidationError):
    ResultRequirement(
      mode="strict_projection",
      projection=ProjectionRequirement(
        contract=contract("projection"),
        required=False,
      ),
      terminal_narrative="forbidden",
      outcome=OutcomeRequirement(required=False, source="none"),
    )


def test_non_execution_disposition_requires_canonical_unavailable_inputs() -> None:
  unavailable = SettleWithoutExecutionDisposition(
    reason="required_input_unavailable",
    unavailable_input_names=("first", "second"),
  )
  assert unavailable.kind == "settle_without_execution"

  with pytest.raises(ValidationError, match="requires unavailable input names"):
    SettleWithoutExecutionDisposition(
      reason="required_input_unavailable",
      unavailable_input_names=(),
    )
  with pytest.raises(ValidationError, match="require required_input_unavailable"):
    SettleWithoutExecutionDisposition(
      reason="workflow_cancelled",
      unavailable_input_names=("first",),
    )
  with pytest.raises(ValidationError, match="canonical sorted order"):
    SettleWithoutExecutionDisposition(
      reason="required_input_unavailable",
      unavailable_input_names=("second", "first"),
    )


def test_task_result_keeps_execution_outcome_and_evidence_separate() -> None:
  evidence_content = text_handle("filing", name="evidence")
  completed = result()
  observed = completed.model_copy(
    update={
      "evidence": EvidenceObservation(
        observed_sources=(ContentEvidenceRef(content=evidence_content),),
        tools_used=("filings.read",),
      )
    }
  )
  assert observed.execution.status == "succeeded"
  assert observed.outcome is not None and observed.outcome.disposition == "complete"
  assert observed.evidence.tools_used == ("filings.read",)

  with pytest.raises(ValidationError, match="skipped"):
    TaskResult.model_validate(
      {
        **completed.model_dump(),
        "execution": {"status": "skipped", "terminal_reason": "policy"},
      }
    )


def test_projection_inline_bytes_and_digest_are_canonical() -> None:
  value: JsonValue = {"verdict": "accepted", "score": 1}
  projection = CanonicalProjection(
    contract=contract("projection"),
    content=json_handle(value),
    inline_view=value,
  )
  assert CanonicalProjection.model_validate_json(projection.model_dump_json()) == projection
  with pytest.raises(ValidationError, match="byte count|digest"):
    CanonicalProjection(
      contract=contract("projection"),
      content=json_handle({"different": True}),
      inline_view=value,
    )


def test_exact_parent_materialization_has_no_clipping_semantics() -> None:
  terminal = "Exact terminal narrative"
  handle = text_handle(terminal)
  materialization = TerminalNarrativeInlineExact(source=handle, content=terminal)
  envelope = AgentCompletionEnvelope(
    message_id="message-1",
    task_result_ref=TaskResultRef.from_result(result(terminal)),
    settlement_projection=SettlementProjection(
      execution_status="succeeded",
      outcome_disposition="complete",
    ),
    parent_materialization=materialization,
  )
  assert AgentCompletionEnvelope.model_validate_json(envelope.model_dump_json()) == envelope
  policy_schema = json.dumps(ParentResultPolicy.model_json_schema())
  assert "truncated" not in policy_schema
  assert "clipped" not in policy_schema
  with pytest.raises(ValidationError, match="byte count"):
    TerminalNarrativeInlineExact(source=handle, content="clipped")


def test_attachment_spec_requires_summary_but_inline_does_not() -> None:
  with pytest.raises(ValidationError, match="summary"):
    WorkflowDeliverySpecV1(presentation="attachment", primary_selector="primary")
  inline = WorkflowDeliverySpecV1(presentation="inline", primary_selector="primary")
  assert inline.summary_selector is None


def test_delivery_spec_serializes_the_exact_code_owned_presentation_bound() -> None:
  spec = WorkflowDeliverySpecV1(presentation="inline", primary_selector="primary")
  assert spec.summary_inline_max_bytes == PUBLISHED_OUTPUT_INLINE_MAX_BYTES
  assert spec.model_dump(mode="json")["summary_inline_max_bytes"] == (
    PUBLISHED_OUTPUT_INLINE_MAX_BYTES
  )
  with pytest.raises(ValidationError, match="code-owned presentation bound"):
    WorkflowDeliverySpecV1(
      presentation="inline",
      primary_selector="primary",
      summary_inline_max_bytes=PUBLISHED_OUTPUT_INLINE_MAX_BYTES + 1,
    )


def test_delivery_without_authored_summary_requires_explicit_warning() -> None:
  primary_text = "Lossless report"
  primary_output = PublishedOutput(
    name="primary_report",
    output_id="wout:workflow-1:phase:2:revision:1:primary_report",
    contract=contract("report"),
    content=text_handle(primary_text),
    inline_view=None,
  )
  spec = WorkflowDeliverySpecV1(
    presentation="attachment",
    primary_selector="primary_report",
    summary_selector="delivery_summary",
  )
  envelope = DeliveryEnvelopeV1(
    schema_version="1.0",
    workflow_run_id="workflow-1",
    phase_number=2,
    revision=1,
    summary=None,
    primary=DeliveryPrimary(
      name="report",
      published_output_ref=PublishedOutputRef.from_output(primary_output),
    ),
  )
  warning = DeliveryWarning(
    code="delivery_summary_oversized",
    message="The authored delivery summary exceeds the presentation bound.",
    omitted_outputs=("delivery_summary",),
  )
  with pytest.raises(ValidationError, match="explicit delivery warning"):
    DeliverySettlement(
      status="complete",
      phase_number=2,
      revision=1,
      spec=spec,
      envelope=envelope,
    )
  degraded = DeliverySettlement(
    status="complete",
    phase_number=2,
    revision=1,
    spec=spec,
    envelope=envelope,
    warning=warning,
  )
  assert DeliverySettlement.model_validate_json(degraded.model_dump_json()) == degraded
  with pytest.raises(ValidationError, match="omitted summary selector"):
    DeliverySettlement(
      status="complete",
      phase_number=2,
      revision=1,
      spec=spec,
      envelope=envelope,
      warning=warning.model_copy(update={"omitted_outputs": ("other_output",)}),
    )
  with pytest.raises(ValidationError, match="only legal on a complete delivery"):
    DeliverySettlement(
      status="failed",
      phase_number=2,
      revision=1,
      spec=spec,
      failure=DeliveryFailure(
        code="delivery_outputs_missing",
        message="The terminal phase omitted an admitted delivery output.",
        missing_outputs=("primary_report",),
      ),
      warning=warning,
    )


def test_delivery_envelope_is_atomic_and_summary_is_exact() -> None:
  workflow_id = "workflow-1"
  summary_text = "Grounded summary"
  summary_output = PublishedOutput(
    name="delivery_summary",
    output_id=f"wout:{workflow_id}:phase:2:revision:1:delivery_summary",
    contract=contract("summary"),
    content=text_handle(summary_text, name="summary"),
    inline_view=PublishedInlineView(value=summary_text),
  )
  report_text = "Lossless report"
  primary_output = PublishedOutput(
    name="primary_report",
    output_id=f"wout:{workflow_id}:phase:2:revision:1:primary_report",
    contract=contract("report"),
    content=text_handle(report_text),
    inline_view=None,
  )
  envelope = DeliveryEnvelopeV1(
    schema_version="1.0",
    workflow_run_id=workflow_id,
    phase_number=2,
    revision=1,
    summary=AuthoredDeliverySummary(
      text=summary_text,
      source=PublishedOutputRef.from_output(summary_output),
    ),
    primary=DeliveryPrimary(
      name="report",
      published_output_ref=PublishedOutputRef.from_output(primary_output),
    ),
  )
  spec = WorkflowDeliverySpecV1(
    presentation="attachment",
    primary_selector="primary_report",
    summary_selector="delivery_summary",
  )
  settled = DeliverySettlement(
    status="complete",
    phase_number=2,
    revision=1,
    spec=spec,
    envelope=envelope,
  )
  assert DeliverySettlement.model_validate_json(settled.model_dump_json()) == settled
  with pytest.raises(ValidationError, match="one workflow phase"):
    DeliveryEnvelopeV1(
      schema_version="1.0",
      workflow_run_id=workflow_id,
      phase_number=3,
      revision=1,
      summary=envelope.summary,
      primary=envelope.primary,
    )


def test_delivery_spec_reader_keeps_absent_version_v1_exact_and_requires_v2_version() -> None:
  raw_v1 = {
    "additional_selectors": [],
    "presentation": "attachment",
    "primary_selector": "report",
    "summary_inline_max_bytes": 8000,
    "summary_selector": "summary",
  }
  v1 = parse_workflow_delivery_spec(raw_v1)
  assert isinstance(v1, WorkflowDeliverySpecV1)
  assert v1.model_dump(mode="json") == raw_v1
  assert v1.model_dump_json() == (
    '{"presentation":"attachment","primary_selector":"report",'
    '"summary_selector":"summary","additional_selectors":[],'
    '"summary_inline_max_bytes":8000}'
  )
  with pytest.raises(ValueError, match="unsupported schema_version"):
    parse_workflow_delivery_spec({**raw_v1, "schema_version": "1.0"})
  with pytest.raises(ValidationError, match="schema_version"):
    WorkflowDeliverySpecV2(
      presentation="attachment",
      primary_selector="report",
      preview_policy_version=DELIVERY_PREVIEW_POLICY_VERSION,
      preview_max_bytes=DELIVERY_PREVIEW_MAX_BYTES,
    )

  v2 = parse_workflow_delivery_spec({
    "schema_version": "2.0",
    "presentation": "attachment",
    "primary_selector": "report",
    "additional_selectors": [],
    "preview_policy_version": DELIVERY_PREVIEW_POLICY_VERSION,
    "preview_max_bytes": DELIVERY_PREVIEW_MAX_BYTES,
  })
  assert isinstance(v2, WorkflowDeliverySpecV2)


def test_v2_delivery_preview_is_a_bounded_exact_utf8_interval() -> None:
  with pytest.raises(ValidationError, match="kind|source_start_byte"):
    DeliveryPreview(
      text="brief",
      source_end_byte=5,
      source_total_bytes=5,
      complete=True,
      omitted_bytes=0,
    )
  with pytest.raises(ValidationError, match="byte range"):
    DeliveryPreview(
      kind="deterministic_text_preview",
      text="β",
      source_start_byte=0,
      source_end_byte=1,
      source_total_bytes=2,
      complete=False,
      omitted_bytes=1,
    )
  with pytest.raises(ValidationError, match="version-2 byte bound"):
    DeliveryPreview(
      kind="deterministic_text_preview",
      text="x" * (DELIVERY_PREVIEW_MAX_BYTES + 1),
      source_start_byte=0,
      source_end_byte=DELIVERY_PREVIEW_MAX_BYTES + 1,
      source_total_bytes=DELIVERY_PREVIEW_MAX_BYTES + 1,
      complete=True,
      omitted_bytes=0,
    )
  with pytest.raises(ValidationError, match="omitted bytes"):
    DeliveryPreview(
      kind="deterministic_text_preview",
      text="brief",
      source_start_byte=0,
      source_end_byte=5,
      source_total_bytes=7,
      complete=False,
      omitted_bytes=1,
    )
  with pytest.raises(ValidationError, match="completeness"):
    DeliveryPreview(
      kind="deterministic_text_preview",
      text="brief",
      source_start_byte=0,
      source_end_byte=5,
      source_total_bytes=5,
      complete=False,
      omitted_bytes=0,
    )


def test_v2_delivery_pairing_uses_the_recorded_preview_policy() -> None:
  workflow_id = "workflow-1"
  primary_text = "Lossless report"
  primary_output = PublishedOutput(
    name="primary_report",
    output_id=f"wout:{workflow_id}:phase:2:revision:1:primary_report",
    contract=contract("report"),
    content=text_handle(primary_text),
    inline_view=PublishedInlineView(value=primary_text),
  )
  envelope = DeliveryEnvelopeV2(
    schema_version="2.0",
    workflow_run_id=workflow_id,
    phase_number=2,
    revision=1,
    primary=DeliveryPrimaryV2(
      name="report",
      published_output_ref=PublishedOutputRef.from_output(primary_output),
      preview=DeliveryPreview(
        kind="deterministic_text_preview",
        text=primary_text,
        source_start_byte=0,
        source_end_byte=len(primary_text.encode("utf-8")),
        source_total_bytes=len(primary_text.encode("utf-8")),
        complete=True,
        omitted_bytes=0,
      ),
    ),
  )
  spec = WorkflowDeliverySpecV2(
    schema_version="2.0",
    presentation="attachment",
    primary_selector="primary_report",
    preview_policy_version=DELIVERY_PREVIEW_POLICY_VERSION,
    preview_max_bytes=DELIVERY_PREVIEW_MAX_BYTES,
  )
  settlement = DeliverySettlement(
    status="complete",
    phase_number=2,
    revision=1,
    spec=spec,
    envelope=envelope,
  )
  assert isinstance(parse_delivery_envelope(envelope.model_dump(mode="json")), DeliveryEnvelopeV2)
  assert DeliverySettlement.model_validate_json(settlement.model_dump_json()) == settlement
  with pytest.raises(ValidationError, match="versions must match"):
    DeliverySettlement(
      status="complete",
      phase_number=2,
      revision=1,
      spec=WorkflowDeliverySpecV1(
        presentation="inline",
        primary_selector="primary_report",
      ),
      envelope=envelope,
    )
  with pytest.raises(ValidationError, match="version-2 delivery forbids"):
    DeliverySettlement(
      status="complete",
      phase_number=2,
      revision=1,
      spec=spec,
      envelope=envelope,
      warning=DeliveryWarning(
        code="delivery_summary_oversized",
        message="not part of the version-2 contract",
        omitted_outputs=("summary",),
      ),
    )


def test_complete_v2_preview_rejects_mismatched_non_string_exact_inline_value() -> None:
  workflow_id = "workflow-1"
  inline_value: JsonValue = {"a": 1}
  raw_inline = canonical_json_bytes(inline_value)
  inline_sha = hashlib.sha256(raw_inline).hexdigest()
  primary_output = PublishedOutput(
    name="primary_report",
    output_id=f"wout:{workflow_id}:phase:2:revision:1:primary_report",
    contract=contract("report"),
    content=ContentHandle(
      content_id=f"sha256:{inline_sha}",
      content_sha256=inline_sha,
      content_bytes=len(raw_inline),
      content_chars=len(raw_inline.decode("utf-8")),
      contract=contract("report"),
      media_type="text/plain",
      encoding="utf-8",
      retention="durable",
    ),
    inline_view=PublishedInlineView(value=inline_value),
  )
  envelope = DeliveryEnvelopeV2(
    schema_version="2.0",
    workflow_run_id=workflow_id,
    phase_number=2,
    revision=1,
    primary=DeliveryPrimaryV2(
      name="report",
      published_output_ref=PublishedOutputRef.from_output(primary_output),
      preview=DeliveryPreview(
        kind="deterministic_text_preview",
        text="x" * len(raw_inline),
        source_start_byte=0,
        source_end_byte=len(raw_inline),
        source_total_bytes=len(raw_inline),
        complete=True,
        omitted_bytes=0,
      ),
    ),
  )
  spec = WorkflowDeliverySpecV2(
    schema_version="2.0",
    presentation="attachment",
    primary_selector="primary_report",
    preview_policy_version=DELIVERY_PREVIEW_POLICY_VERSION,
    preview_max_bytes=DELIVERY_PREVIEW_MAX_BYTES,
  )
  with pytest.raises(ValidationError, match="conflicts with exact inline primary"):
    WorkflowResult(
      workflow_run_id=workflow_id,
      view=WorkflowView(
        workflow_run_id=workflow_id,
        workflow_name="dynamic-workflow",
        state="terminal",
        execution_status="succeeded",
        delivery_status="complete",
        terminal_status="succeeded",
        legal_actions=(),
        observation_seq=1,
        max_phases=2,
        admitted_plan_ref=AdmittedPlanRef(
          workflow_run_id=workflow_id,
          plan_id="plan-1",
          phase_number=2,
          revision=1,
          digest=DIGEST,
        ),
        terminal_phase_revision=TerminalPhaseRevision(
          phase_number=2,
          revision=1,
        ),
        estimated_cost_usd=0.0,
        admitted_cost_estimate_usd=0.0,
        author_cost_usd=0.0,
      ),
      published_outputs=(primary_output,),
      delivery=DeliverySettlement(
        status="complete",
        phase_number=2,
        revision=1,
        spec=spec,
        envelope=envelope,
      ),
      transcript=TranscriptHandle(
        kind="workflow_transcript",
        owner_id=workflow_id,
      ),
      activity=ActivityHandle(
        kind="workflow_activity",
        owner_id=workflow_id,
      ),
      continuation_state=ContinuationState(status="not_available"),
    )
