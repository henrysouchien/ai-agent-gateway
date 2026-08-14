# ruff: noqa: E402

from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
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
  return build_skill_result_captured_event(
    skill_run_id="skill-run-1",
    skill="test-skill",
    ticker=None,
    scope="portfolio",
    portfolio_id=None,
    entries=[],
    result=result,
    error=error,
  )


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

  assert event["exit_code"] == 1
  assert event["outcome"] == "error"
  assert event["status"] == "error"
  assert event["error"] == "active research file is required"


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
