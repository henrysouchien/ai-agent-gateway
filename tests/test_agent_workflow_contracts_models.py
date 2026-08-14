from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import JsonValue, ValidationError

from agent_workflow_contracts import (
  AdmittedDataRef,
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
  ContractRef,
  DeliveryEnvelope,
  DeliveryFailure,
  DeliveryPrimary,
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
  TerminalNarrativeInlineExact,
  TranscriptHandle,
  ActivityHandle,
  UsageObservation,
  WorkflowDeliverySpec,
  canonical_json_bytes,
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
  with pytest.raises(ValidationError, match="raw filesystem"):
    LiteralSelector(value="/tmp/internal-report.json")


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
    WorkflowDeliverySpec(presentation="attachment", primary_selector="primary")
  inline = WorkflowDeliverySpec(presentation="inline", primary_selector="primary")
  assert inline.summary_selector is None


def test_delivery_spec_serializes_the_exact_code_owned_presentation_bound() -> None:
  spec = WorkflowDeliverySpec(presentation="inline", primary_selector="primary")
  assert spec.summary_inline_max_bytes == PUBLISHED_OUTPUT_INLINE_MAX_BYTES
  assert spec.model_dump(mode="json")["summary_inline_max_bytes"] == (
    PUBLISHED_OUTPUT_INLINE_MAX_BYTES
  )
  with pytest.raises(ValidationError, match="code-owned presentation bound"):
    WorkflowDeliverySpec(
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
  spec = WorkflowDeliverySpec(
    presentation="attachment",
    primary_selector="primary_report",
    summary_selector="delivery_summary",
  )
  envelope = DeliveryEnvelope(
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
  envelope = DeliveryEnvelope(
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
  spec = WorkflowDeliverySpec(
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
    DeliveryEnvelope(
      workflow_run_id=workflow_id,
      phase_number=3,
      revision=1,
      summary=envelope.summary,
      primary=envelope.primary,
    )
