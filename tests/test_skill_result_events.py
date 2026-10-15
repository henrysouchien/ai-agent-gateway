# ruff: noqa: E402

from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_workflow_contracts import (
  AgentOperationRef,
  AttemptRef,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  TaskResultProvenance,
  sha256_digest,
)
from agent_gateway.skill_result_events import build_skill_result_captured_event
from agent_gateway.sub_agent_result_contract import (
  FinalNarrativeArtifactReference,
  build_task_result,
)


def _report_result() -> dict[str, Any]:
  return _task_result(
    execution=ExecutionSettlement(status="succeeded"),
    include_narrative=True,
  )


def _task_result(
  *,
  execution: ExecutionSettlement,
  include_narrative: bool,
) -> dict[str, Any]:
  requirement = ResultRequirement(
    mode="narrative",
    terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )
  operation = AgentOperationRef(
    namespace="research",
    name="test-skill",
    version="1.0",
    digest=sha256_digest({"operation": "research.test-skill/1.0"}),
  )
  digest = sha256_digest({"fixture": "skill-result"})
  result = build_task_result(
    logical_task=OrdinaryDelegationTaskRef(
      delegation_id="skill-run-1",
      operation=operation,
    ),
    attempt=AttemptRef(
      attempt_number=1,
      attempt_id="skill-attempt-1",
      physical_task_id="skill-child-1",
    ),
    requirement=requirement,
    provenance=TaskResultProvenance(
      admitted_task_digest=digest,
      model_bind_digest=digest,
      capability_binding_digest=digest,
      tool_grant_digest=digest,
    ),
    execution=execution,
    outcome=None,
    terminal_narrative=(
      FinalNarrativeArtifactReference(
        artifact_id=f"sha256:{hashlib.sha256(b'done').hexdigest()}",
        artifact_ref="final-narrative://skill-child-1",
        content_sha256=hashlib.sha256(b"done").hexdigest(),
        content_chars=4,
        content_bytes=4,
        terminal_event_seq=1,
      )
      if include_narrative
      else None
    ),
    projection=None,
  )
  return result.model_dump(mode="json")


def _captured_event(
  result: dict[str, Any],
  error: dict[str, Any] | None = None,
) -> dict[str, Any]:
  event = build_skill_result_captured_event(
    skill_run_id="skill-run-1",
    skill="test-skill",
    ticker=None,
    scope="portfolio",
    portfolio_id=None,
    entries=[],
    result=result,
    error=error,
  )
  assert event is not None
  return event


def test_skill_result_event_accepts_only_validated_report_as_success() -> None:
  event = _captured_event(_report_result())

  assert event["exit_code"] == 0
  assert event["outcome"] == "success"
  assert event["status"] == "not_assessed"
  assert event["error"] is None
  assert event["approval_outcome"] is None
  assert event["approval_id"] is None
  assert event["approval_tool_name"] is None


def test_skill_result_event_uses_canonical_execution_settlement_for_failure() -> None:
  event = _captured_event(_task_result(
    execution=ExecutionSettlement(
      status="interrupted",
      terminal_reason="timeout: provider timed out",
    ),
    include_narrative=False,
  ))

  assert event["exit_code"] == 1
  assert event["outcome"] == "error"
  assert event["status"] == "interrupted"
  assert event["error"] == "Sub-agent ended with timeout: provider timed out"


def test_skill_result_event_rejects_success_claim_from_error() -> None:
  event = _captured_event(
    _report_result(),
    {
      "code": "provider_failed",
      "message": "provider failed",
      "child_outcome": "success",
    },
  )

  assert event["exit_code"] == 1
  assert event["outcome"] == "error"
  assert event["status"] == "error"
  assert event["error"] == "provider failed"


def test_skill_result_event_treats_recoverable_fms_error_as_failure() -> None:
  event = build_skill_result_captured_event(
    skill_run_id="skill-run-fms-error",
    skill="test-skill",
    ticker=None,
    scope="portfolio",
    portfolio_id=None,
    entries=[{
      "type": "tool_call_complete",
      "tool_name": "fms_persist_test_artifact",
      "result": {
        "status": "error",
        "subcommand": "persist_test_artifact",
        "mutation_mode": "model_writer",
        "error": {
          "type": "INVALID_JUDGMENT",
          "message": "active research file is required",
          "recoverable": True,
        },
      },
    }],
    result=_report_result(),
    error=None,
  )
  assert event is not None

  assert event["exit_code"] == 1
  assert event["outcome"] == "error"
  assert event["status"] == "blocked"
  assert event["fms_results"][0]["status"] == "error"
  assert event["error"] == "active research file is required"


def _gate_door(**fields: Any) -> dict[str, Any]:
  return {
    "type": "tool_call_complete",
    "tool_name": "fms_persist_scenario_set",
    "result": {
      "status": "staged",
      "subcommand": "persist_scenario_set",
      "mutation_mode": "model_writer",
      **fields,
    },
  }


def test_skill_result_event_reports_the_door_gate_code() -> None:
  event = _event_for_entries([_gate_door(gate_code="INSUFFICIENT_DATA")])

  assert event["gate_code"] == "INSUFFICIENT_DATA"


def test_skill_result_event_does_not_read_a_gate_from_the_verdict_echo() -> None:
  event = _event_for_entries([
    _gate_door(verdict_echo={"verdict": "SCENARIOS_BUILT"}),
  ])

  assert event["gate_code"] is None
  assert event["verdict_echo"] == {"verdict": "SCENARIOS_BUILT"}


def test_skill_result_event_fails_at_the_owner_on_an_unknown_gate_code() -> None:
  with pytest.raises(RuntimeError, match="gate_code must be one of"):
    _event_for_entries([_gate_door(gate_code="SCENARIOS_BUILT")])


_REFUSED_CONTEXT_DOOR = {
  "type": "tool_call_complete",
  "tool_name": "get_scenario_set_context",
  "result": {
    "status": "error",
    "error": "research_file_id must be a positive integer",
    "validation_error": True,
  },
}
_ACCEPTED_TERMINAL_DOOR = {
  "type": "tool_call_complete",
  "tool_name": "fms_persist_scenario_set",
  "result": {
    "status": "staged",
    "subcommand": "persist_scenario_set",
    "mutation_mode": "model_writer",
  },
}


def _event_for_entries(
  entries: list[dict[str, Any]],
  *,
  declared_terminal_doors: tuple[str, ...] | None = None,
  result: dict[str, Any] | None = None,
  canonical_result_evidence_authoritative: bool = False,
) -> dict[str, Any]:
  event = build_skill_result_captured_event(
    skill_run_id="skill-run-doors",
    skill="test-skill",
    ticker=None,
    scope="portfolio",
    portfolio_id=None,
    entries=entries,
    result=result if result is not None else _report_result(),
    error=None,
    declared_terminal_doors=declared_terminal_doors,
    canonical_result_evidence_authoritative=canonical_result_evidence_authoritative,
  )
  assert event is not None
  return event


def test_skill_result_event_refused_context_door_is_not_success() -> None:
  event = _event_for_entries([_REFUSED_CONTEXT_DOOR])

  assert event["exit_code"] == 1
  assert event["outcome"] == "error"
  assert event["status"] == "blocked"
  assert event["error"] == "research_file_id must be a positive integer"
  assert event["last_failed_door_answer"]["tool_name"] == "get_scenario_set_context"


def test_skill_result_event_accepted_terminal_door_supersedes_refused_context_door() -> None:
  event = _event_for_entries([_REFUSED_CONTEXT_DOOR, _ACCEPTED_TERMINAL_DOOR])

  assert event["exit_code"] == 0
  assert event["outcome"] == "success"
  assert event["status"] == "not_assessed"
  assert event["error"] is None


def test_skill_result_event_refused_context_door_after_accepted_terminal_door_is_not_success() -> None:
  event = _event_for_entries([_ACCEPTED_TERMINAL_DOOR, _REFUSED_CONTEXT_DOOR])

  assert event["exit_code"] == 1
  assert event["status"] == "blocked"


def test_skill_result_event_without_door_events_keeps_process_outcome() -> None:
  event = _event_for_entries([
    {"type": "tool_call_complete", "tool_name": "read_file", "result": {"status": "ok"}},
  ])

  assert event["exit_code"] == 0
  assert event["outcome"] == "success"
  assert event["error"] is None


_DOCS_FETCH_ONLY = [
  {"type": "tool_call_complete", "tool_name": "docs_fetch", "result": {"status": "ok"}},
]
_DECLARED = ("fms_persist_scenario_set",)


def test_skill_result_event_declared_terminal_door_never_called_is_not_success() -> None:
  event = _event_for_entries(_DOCS_FETCH_ONLY, declared_terminal_doors=_DECLARED)

  assert event["exit_code"] == 1
  assert event["outcome"] == "error"
  assert event["status"] == "blocked"
  assert event["artifact_refs"] == []
  assert event["fms_results"] == []
  assert event["error"] == (
    "skill ended without an accepted terminal door result or artifact"
  )


def test_skill_result_event_declared_terminal_door_with_artifact_is_success() -> None:
  event = _event_for_entries(
    [
      *_DOCS_FETCH_ONLY,
      {"type": "artifact_ready", "artifact_ref": "artifacts/report.json"},
    ],
    declared_terminal_doors=_DECLARED,
  )

  assert event["exit_code"] == 0
  assert event["outcome"] == "success"
  assert event["artifact_refs"] == ["artifacts/report.json"]


def test_skill_result_event_declared_terminal_door_accepted_is_success() -> None:
  event = _event_for_entries(
    [*_DOCS_FETCH_ONLY, _ACCEPTED_TERMINAL_DOOR],
    declared_terminal_doors=_DECLARED,
  )

  assert event["exit_code"] == 0
  assert event["outcome"] == "success"


def test_skill_result_event_canonical_evidence_keeps_accepted_terminal_door() -> None:
  # Canonical evidence replaces the terminal door list with its artifacts, so
  # an accepted door answer in the entries must still count as persisted.
  event = _event_for_entries(
    [*_DOCS_FETCH_ONLY, _ACCEPTED_TERMINAL_DOOR],
    declared_terminal_doors=_DECLARED,
    canonical_result_evidence_authoritative=True,
  )

  assert event["exit_code"] == 0
  assert event["outcome"] == "success"


def test_skill_result_event_authoritative_path_accepted_door_supersedes_refusal() -> None:
  # Durable settlement (canonical_result_evidence_authoritative=True) used to
  # drop the entry-derived terminal doors, so a refused context door that a
  # later accepted persist superseded still read blocked (audit P1 on
  # ace98b0426). The doors that answered decide, on every path.
  event = _event_for_entries(
    [_REFUSED_CONTEXT_DOOR, _ACCEPTED_TERMINAL_DOOR],
    canonical_result_evidence_authoritative=True,
  )

  assert event["exit_code"] == 0
  assert event["outcome"] == "success"
  assert event["error"] is None

def test_skill_result_event_canonical_without_door_or_artifact_is_not_success() -> None:
  event = _event_for_entries(
    _DOCS_FETCH_ONLY,
    declared_terminal_doors=_DECLARED,
    canonical_result_evidence_authoritative=True,
  )

  assert event["exit_code"] == 1
  assert event["status"] == "blocked"


def test_skill_result_event_declared_door_other_than_called_one_is_not_persisted() -> None:
  event = _event_for_entries(
    [*_DOCS_FETCH_ONLY, _ACCEPTED_TERMINAL_DOOR],
    declared_terminal_doors=("fms_persist_other_set",),
  )

  assert event["exit_code"] == 1
  assert event["status"] == "blocked"


def test_skill_result_event_without_declared_doors_keeps_process_outcome() -> None:
  event = _event_for_entries(_DOCS_FETCH_ONLY)

  assert event["exit_code"] == 0
  assert event["outcome"] == "success"

def test_skill_result_event_coordinator_calls_no_terminal_result_door_is_success() -> None:
  # A coordinator (diligence-fanout shape) carries writer doors on its grant
  # for its children, but its operation declares no terminal-result door of
  # its own (terminal_result_doors == ()), so delegating and reading back a
  # child's artifact is a success with nothing persisted by the coordinator.
  entries = [
    {
      "type": "tool_call_complete",
      "tool_name": "run_agent",
      "result": {"status": "success"},
    },
    {
      "type": "tool_call_complete",
      "tool_name": "get_skill_artifact",
      "result": {"status": "ok"},
    },
  ]
  event = _event_for_entries(entries, declared_terminal_doors=())

  assert event["exit_code"] == 0
  assert event["outcome"] == "success"


def test_skill_result_event_counts_compactions_from_single_pass_entries() -> None:
  event = build_skill_result_captured_event(
    skill_run_id="skill-run-1",
    skill="test-skill",
    ticker=None,
    scope="portfolio",
    portfolio_id=None,
    entries=iter([
      {"type": "compaction"},
      SimpleNamespace(event={"type": "compaction"}),
      {"type": "tool_call_complete"},
    ]),
    result=_report_result(),
    error=None,
  )
  assert event is not None

  assert event["compaction_count"] == 2


def test_skill_result_event_preserves_exact_lifecycle_identity() -> None:
  event = build_skill_result_captured_event(
    skill_run_id="skill-run-identity",
    skill="test-skill",
    ticker=None,
    scope="portfolio",
    portfolio_id="portfolio-1",
    entries=[{
      "type": "artifact_ready",
      "ticker": "PCTY",
      "artifact_path": "artifacts/PCTY/report.json",
    }],
    result=_report_result(),
    error=None,
  )
  assert event is not None

  assert {
    field_name: event[field_name]
    for field_name in (
      "skill_run_id",
      "skill",
      "scope",
      "ticker",
      "portfolio_id",
    )
  } == {
    "skill_run_id": "skill-run-identity",
    "skill": "test-skill",
    "scope": "portfolio",
    "ticker": None,
    "portfolio_id": "portfolio-1",
  }


def test_skill_result_event_deep_owns_nested_receipt_values() -> None:
  artifact_event = {
    "type": "artifact_ready",
    "artifact_path": "artifacts/report.json",
    "metadata": {"tags": ["original"]},
  }

  event = build_skill_result_captured_event(
    skill_run_id="skill-run-owned-receipt",
    skill="test-skill",
    ticker=None,
    scope="portfolio",
    portfolio_id="portfolio-1",
    entries=[artifact_event],
    result=_report_result(),
    error=None,
  )
  assert event is not None
  artifact_event["metadata"]["tags"].append("mutated")

  assert event["artifact_events"][0]["metadata"]["tags"] == ["original"]


def test_skill_result_event_rejects_non_json_nested_receipt_values() -> None:
  with pytest.raises(RuntimeError, match="contains non-JSON value object"):
    build_skill_result_captured_event(
      skill_run_id="skill-run-invalid-receipt",
      skill="test-skill",
      ticker=None,
      scope="portfolio",
      portfolio_id="portfolio-1",
      entries=[{
        "type": "artifact_ready",
        "artifact_path": "artifacts/report.json",
        "metadata": {"invalid": object()},
      }],
      result=_report_result(),
      error=None,
    )


@pytest.mark.parametrize(
  ("scope", "ticker", "portfolio_id"),
  [
    ("ticker", "PCTY", "portfolio-1"),
    ("portfolio", "PCTY", None),
    ("ticker", None, "portfolio-1"),
    ("ticker", None, None),
  ],
)
def test_skill_result_event_rejects_invalid_lifecycle_identity(
  scope: str,
  ticker: str | None,
  portfolio_id: str | None,
) -> None:
  with pytest.raises(ValueError):
    build_skill_result_captured_event(
      skill_run_id="skill-run-invalid",
      skill="test-skill",
      ticker=ticker,
      scope=scope,  # type: ignore[arg-type]
      portfolio_id=portfolio_id,
      entries=[],
      result=_report_result(),
      error=None,
    )


def test_awaiting_approval_has_no_terminal_skill_result_event() -> None:
  result = _task_result(
    execution=ExecutionSettlement(
      status="awaiting_approval",
      terminal_reason="approval_required",
    ),
    include_narrative=False,
  )

  event = build_skill_result_captured_event(
    skill_run_id="skill-run-parked",
    skill="test-skill",
    ticker=None,
    scope="portfolio",
    portfolio_id=None,
    entries=[],
    result=result,
    error=None,
  )

  assert event is None
