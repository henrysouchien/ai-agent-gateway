from __future__ import annotations

import hashlib

import pytest
from agent_workflow_contracts import (
  AgentOperationRef,
  AnalyticalOutcome,
  AttemptRef,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  TaskResultProvenance,
  canonical_json_bytes,
  sha256_digest,
)
from pydantic import ValidationError

from agent_gateway.sub_agent_result_contract import (
  CompactReport,
  ContractValidationError,
  ContractValidationResult,
  FinalNarrativeArtifactReference,
  FmsArtifactReference,
  REPORT_FINDINGS_MAX_ITEMS,
  VerifyFindingReport,
  build_task_result,
  canonical_projection,
  canonical_report_size_bytes,
  child_evidence_fits_externalization_bound,
  report_contract_ref,
)


def _identity():
  operation = AgentOperationRef(
    namespace="research",
    name="explore",
    version="1.0",
    digest=sha256_digest({"operation": "research.explore/1.0"}),
  )
  logical_task = OrdinaryDelegationTaskRef(
    delegation_id="delegation-1",
    operation=operation,
  )
  attempt = AttemptRef(
    attempt_number=2,
    attempt_id="attempt-2",
    physical_task_id="child-physical-2",
    restart_of_attempt_id="attempt-1",
  )
  digest = sha256_digest({"fixture": "provenance"})
  provenance = TaskResultProvenance(
    admitted_task_digest=digest,
    model_bind_digest=digest,
    capability_binding_digest=digest,
    tool_grant_digest=digest,
  )
  return logical_task, attempt, provenance


def _narrative_reference(text: str) -> FinalNarrativeArtifactReference:
  raw = text.encode("utf-8")
  digest = hashlib.sha256(raw).hexdigest()
  return FinalNarrativeArtifactReference(
    artifact_id=f"sha256:{digest}",
    artifact_ref=f"final-narrative://{digest}",
    content_sha256=digest,
    content_chars=len(text),
    content_bytes=len(raw),
    terminal_event_seq=17,
  )


def _compact_payload() -> dict[str, object]:
  return {
    "summary": "The filing supports the claim.",
    "findings": [{"claim": "Revenue accelerated.", "confidence": "high"}],
    "artifacts": [{"kind": "fms_artifact", "artifact_ref": "fms://artifact/123"}],
    "caveats": ["Management guidance remains unaudited."],
  }


def test_compact_report_preserves_typed_durable_references() -> None:
  report = CompactReport.model_validate(_compact_payload())

  assert isinstance(report.artifacts[0], FmsArtifactReference)
  assert report.artifacts[0].retention == "durable"
  assert canonical_report_size_bytes(report) > 0


def test_report_contracts_enforce_semantic_shape_and_caps() -> None:
  with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
    CompactReport.model_validate({**_compact_payload(), "verdict": "supported"})

  with pytest.raises(ValidationError):
    CompactReport(
      summary="bounded",
      findings=[{"claim": "finding"}] * (REPORT_FINDINGS_MAX_ITEMS + 1),
    )

  with pytest.raises(ValidationError):
    VerifyFindingReport(
      **_compact_payload(),
      target_claim="Revenue accelerated.",
      verdict="mostly_supported",
      recommended_action="keep",
    )


def test_canonical_projection_binds_exact_json_bytes() -> None:
  contract = report_contract_ref("report-base-v1")
  value = {"summary": "界", "findings": [], "artifacts": [], "caveats": []}

  projection = canonical_projection(contract=contract, value=value)
  encoded = canonical_json_bytes(value)

  assert projection.inline_view == value
  assert projection.content.content_bytes == len(encoded)
  assert projection.content.content_sha256 == hashlib.sha256(encoded).hexdigest()
  assert projection.content.contract == contract


def test_narrative_task_result_keeps_exact_unbounded_terminal_handle() -> None:
  logical_task, attempt, provenance = _identity()
  text = "full narrative " * 2_000
  reference = _narrative_reference(text)
  requirement = ResultRequirement(
    mode="narrative",
    terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )

  result = build_task_result(
    logical_task=logical_task,
    attempt=attempt,
    requirement=requirement,
    provenance=provenance,
    execution=ExecutionSettlement(status="succeeded"),
    outcome=None,
    terminal_narrative=reference,
    projection=None,
    tools_used=("web_search", "web_search", "read_file"),
    usage={"input_tokens": 12, "output_tokens": 8, "cost_usd": 0.125},
  )

  assert result.logical_task == logical_task
  assert result.attempt == attempt
  assert result.values.terminal_narrative is not None
  assert result.values.terminal_narrative.content_chars == len(text)
  assert result.values.terminal_narrative.content_bytes == len(text.encode("utf-8"))
  assert result.evidence.tools_used == ("web_search", "read_file")
  assert result.observation.usage.cost_usd == 0.125


def test_evidence_admission_is_bounded_and_never_materializes_cycles() -> None:
  cyclic: list[object] = []
  cyclic.append(cyclic)

  assert child_evidence_fits_externalization_bound(
    usage=None,
    tools_used=("search",),
    fms_results=(),
    artifact_events=(),
  )
  assert not child_evidence_fits_externalization_bound(
    usage={"cycle": cyclic},
    tools_used=(),
    fms_results=(),
    artifact_events=(),
  )


def test_contract_validation_result_requires_exactly_one_arm() -> None:
  report = CompactReport(summary="ok")
  error = ContractValidationError(code="invalid_report", message="invalid")

  with pytest.raises(ValueError, match="exactly one"):
    ContractValidationResult()
  with pytest.raises(ValueError, match="exactly one"):
    ContractValidationResult(report=report, error=error)


def _narrative_requirement() -> ResultRequirement:
  return ResultRequirement(
    mode="narrative",
    terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )


def _build(**overrides):
  logical_task, attempt, provenance = _identity()
  kwargs = {
    "logical_task": logical_task,
    "attempt": attempt,
    "requirement": _narrative_requirement(),
    "provenance": provenance,
    "execution": ExecutionSettlement(status="succeeded"),
    "outcome": None,
    "terminal_narrative": _narrative_reference("exact terminal prose"),
    "projection": None,
  }
  kwargs.update(overrides)
  return build_task_result(**kwargs)


def test_runtime_outcome_carries_a_mechanically_derived_qualifier() -> None:
  # D-B3-4: the mechanical door is a separate keyword, so the authored door's
  # strict rule can stay welded shut without a caller-identity check.
  derived = AnalyticalOutcome(
    disposition="partial",
    assessment_source="mechanically_derived",
    assessment_rationale="the child reached its turn ceiling",
    unmet_requirements=("turns_exhausted",),
  )

  result = _build(runtime_outcome=derived)

  assert result.outcome == derived


def test_runtime_outcome_and_authored_outcome_are_mutually_exclusive() -> None:
  derived = AnalyticalOutcome(
    disposition="complete",
    assessment_source="mechanically_derived",
  )
  authored = AnalyticalOutcome(
    disposition="not_assessed",
    assessment_source="none",
  )

  with pytest.raises(ValueError, match="both an authored and a runtime"):
    _build(outcome=authored, runtime_outcome=derived)


def test_runtime_outcome_rejects_any_non_mechanical_assessment_source() -> None:
  for source in ("domain_tool", "none"):
    disposition = "not_assessed" if source == "none" else "complete"
    with pytest.raises(ValueError, match="must be mechanically derived"):
      _build(
        runtime_outcome=AnalyticalOutcome(
          disposition=disposition,  # type: ignore[arg-type]
          assessment_source=source,  # type: ignore[arg-type]
        )
      )


def test_non_succeeded_settlement_cannot_carry_a_runtime_outcome() -> None:
  # T3-I08: non-succeeded results carry no outcome, structurally.
  with pytest.raises(ValueError, match="non-successful execution cannot carry"):
    _build(
      execution=ExecutionSettlement(
        status="failed",
        terminal_reason="run_error: boom",
      ),
      terminal_narrative=None,
      runtime_outcome=AnalyticalOutcome(
        disposition="complete",
        assessment_source="mechanically_derived",
      ),
    )


def test_authored_outcome_door_stays_welded_shut() -> None:
  # The pre-B-3 rule, verbatim: with a non-assessing requirement the public
  # ``outcome=`` parameter may only ever carry ``not_assessed``.
  with pytest.raises(ValueError, match="non-assessing result may only carry"):
    _build(
      outcome=AnalyticalOutcome(
        disposition="complete",
        assessment_source="mechanically_derived",
      )
    )
