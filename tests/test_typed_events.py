import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.events import (
  AggregateReadyEvent,
  AggregateReadyTrigger,
  ArtifactFailedEvent,
  ArtifactReadyEvent,
  ArtifactUnavailableEvent,
  SkillRunStartedEvent,
  ToolApprovalDecidedEvent,
  ToolApprovalRequestEvent,
  TypedRecommendationsExtractedEvent,
  VerdictEmittedEvent,
  event_from_dict,
  event_to_dict,
)
from agent_gateway.verdict_extractor import extract_verdict_payload, extract_verdict_payload_from_text


def test_typed_events_round_trip_with_type_discriminator() -> None:
  events = [
    SkillRunStartedEvent(skill_run_id="run-1", skill="earnings-scenarios", ticker="PCTY", ts=1.0),
    VerdictEmittedEvent(
      skill_run_id="run-1",
      skill="earnings-scenarios",
      ticker="PCTY",
      verdict_token="SCENARIOS_BUILT",
      confidence="HIGH",
      materiality_cushion=3.15,
      one_line_summary="FY28 base case clears materiality",
      ts=2.0,
    ),
    ArtifactReadyEvent(
      skill_run_id="run-1",
      ticker="PCTY",
      skill="earnings-scenarios",
      artifact_id="2026-05-20T120000.000-run-1",
      artifact_path="artifacts/research/PCTY/earnings-scenarios.json",
      binary_artifact_path=None,
      contract_name="EarningsScenarios",
      data_source="live",
      ts=3.0,
    ),
    AggregateReadyEvent(
      skill_run_id="run-1",
      ticker="PCTY",
      view_model_id="position-card",
      trigger=AggregateReadyTrigger(kind="artifact_ready", source="earnings-scenarios"),
      sources_complete=False,
      ts=4.0,
    ),
    ArtifactFailedEvent(
      skill_run_id="run-1",
      ticker="PCTY",
      skill="earnings-scenarios",
      error_code="yaml_parse",
      error_detail="Could not parse verdict YAML",
      source_path="notes/skills/earnings-scenarios/2026-05-20-PCTY.md",
      ts=5.0,
    ),
    ArtifactUnavailableEvent(
      ticker="PCTY",
      skill="earnings-scenarios",
      reason="no_runs_yet",
      affordance="Run earnings-scenarios on PCTY",
      ts=6.0,
    ),
    ToolApprovalRequestEvent(
      tool_call_id="toolu_1",
      nonce="nonce-1",
      tool_name="bash",
      tool_input={"cmd": "pytest"},
      resolved_qualifier="pytest",
      reason="Run the focused test suite",
      allow_persistent_approval=True,
      ts=7.0,
    ),
    ToolApprovalDecidedEvent(
      tool_call_id="toolu_1",
      tool_name="bash",
      outcome="approved",
      decision_source="user_approved",
      allow_tool_type_applied=True,
      ts=8.0,
    ),
  ]

  for event in events:
    payload = event_to_dict(event)
    assert payload["type"]
    assert event_to_dict(event_from_dict(payload)) == payload

  assert "skill_run_id" not in event_to_dict(events[-1])


def test_tool_approval_request_event_accepts_legacy_missing_ts() -> None:
  payload = {
    "type": "tool_approval_request",
    "tool_call_id": "toolu_1",
    "nonce": "nonce-1",
    "tool_name": "bash",
    "tool_input": {"cmd": "pytest"},
    "resolved_qualifier": "pytest",
    "reason": "Run the focused test suite",
    "allow_persistent_approval": True,
  }

  event = event_from_dict(payload)

  assert isinstance(event, ToolApprovalRequestEvent)
  assert event.ts == 0.0
  assert event_to_dict(event) == {**payload, "ts": 0.0}


def test_typed_recommendations_extracted_event_round_trip() -> None:
  event = TypedRecommendationsExtractedEvent(
    skill="risk-review",
    workflow_name="risk-review",
    scope="portfolio",
    ticker=None,
    portfolio_id=None,
    recommendations_count=2,
    verdict_code="RISK_REVIEW_MITIGATIONS_RECOMMENDED",
    validation_errors=[],
    warnings=["unknown extra field(s) preserved: foo"],
    source_artifact_path="notes/skills/risk-review/2026-05-29.md",
    ts=9.0,
  )

  payload = event_to_dict(event)

  assert payload == {
    "type": "typed_recommendations_extracted",
    "skill": "risk-review",
    "workflow_name": "risk-review",
    "scope": "portfolio",
    "ticker": None,
    "portfolio_id": None,
    "recommendations_count": 2,
    "verdict_code": "RISK_REVIEW_MITIGATIONS_RECOMMENDED",
    "validation_errors": [],
    "warnings": ["unknown extra field(s) preserved: foo"],
    "source_artifact_path": "notes/skills/risk-review/2026-05-29.md",
    "ts": 9.0,
  }
  assert event_from_dict(payload) == event


def test_extended_events_accept_portfolio_scope_without_ticker() -> None:
  events = [
    SkillRunStartedEvent(
      skill_run_id="run-portfolio",
      skill="risk-review",
      ticker=None,
      ts=1.0,
      scope="portfolio",
      portfolio_id="portfolio-1",
    ),
    VerdictEmittedEvent(
      skill_run_id="run-portfolio",
      skill="risk-review",
      ticker=None,
      verdict_token="RISK_REVIEW_MITIGATIONS_RECOMMENDED",
      confidence="HIGH",
      materiality_cushion=None,
      one_line_summary="Mitigations recommended.",
      ts=2.0,
      scope="portfolio",
      portfolio_id="portfolio-1",
    ),
    ArtifactReadyEvent(
      skill_run_id="run-portfolio",
      ticker=None,
      skill="risk-review",
      artifact_id="artifact-1",
      artifact_path="artifacts/research/portfolio/risk-review.json",
      binary_artifact_path=None,
      contract_name="RiskReviewContext",
      data_source="live",
      ts=3.0,
      scope="portfolio",
      portfolio_id="portfolio-1",
    ),
  ]

  for event in events:
    payload = event_to_dict(event)
    hydrated = event_from_dict(payload)
    assert hydrated == event
    assert event_to_dict(hydrated)["ticker"] is None
    assert event_to_dict(hydrated)["scope"] == "portfolio"


def test_extended_events_default_legacy_payloads_to_ticker_scope() -> None:
  payloads = [
    {
      "type": "skill_run_started",
      "skill_run_id": "run-1",
      "skill": "earnings-scenarios",
      "ticker": "PCTY",
      "ts": 1.0,
    },
    {
      "type": "verdict_emitted",
      "skill_run_id": "run-1",
      "skill": "earnings-scenarios",
      "ticker": "PCTY",
      "verdict_token": "SCENARIOS_BUILT",
      "confidence": "HIGH",
      "materiality_cushion": None,
      "one_line_summary": "Scenarios built.",
      "ts": 2.0,
    },
    {
      "type": "artifact_ready",
      "skill_run_id": "run-1",
      "ticker": "PCTY",
      "skill": "earnings-scenarios",
      "artifact_id": "artifact-1",
      "artifact_path": "artifacts/research/PCTY/earnings-scenarios.json",
      "binary_artifact_path": None,
      "contract_name": "EarningsScenarios",
      "data_source": "live",
      "ts": 3.0,
    },
  ]

  for payload in payloads:
    hydrated = event_from_dict(payload)
    serialized = event_to_dict(hydrated)
    assert serialized["scope"] == "ticker"
    assert serialized["portfolio_id"] is None


def test_tool_approval_decided_event_supports_all_decision_sources_and_outcomes() -> None:
  decision_sources = [
    "user_approved",
    "user_denied",
    "headless_auto_deny",
    "headless_hook_approved",
    "session_cache_approved",
    "approval_timeout",
  ]
  outcomes = ["approved", "denied", "timeout"]

  for decision_source in decision_sources:
    payload = {
      "type": "tool_approval_decided",
      "tool_call_id": f"toolu_{decision_source}",
      "tool_name": "bash",
      "outcome": "timeout" if decision_source == "approval_timeout" else "approved",
      "decision_source": decision_source,
      "allow_tool_type_applied": decision_source == "user_approved",
      "ts": 9.0,
    }
    event = event_from_dict(payload)
    assert isinstance(event, ToolApprovalDecidedEvent)
    assert event_to_dict(event) == payload

  for outcome in outcomes:
    decision_source = "approval_timeout" if outcome == "timeout" else "user_denied" if outcome == "denied" else "user_approved"
    payload = {
      "type": "tool_approval_decided",
      "tool_call_id": f"toolu_{outcome}",
      "tool_name": "bash",
      "outcome": outcome,
      "decision_source": decision_source,
      "allow_tool_type_applied": False,
      "ts": 10.0,
    }
    event = event_from_dict(payload)
    assert isinstance(event, ToolApprovalDecidedEvent)
    assert event.outcome == outcome


def test_tool_approval_decided_event_rejects_unknown_enums() -> None:
  valid_payload = {
    "type": "tool_approval_decided",
    "tool_call_id": "toolu_1",
    "tool_name": "bash",
    "outcome": "approved",
    "decision_source": "user_approved",
    "allow_tool_type_applied": False,
    "ts": 11.0,
  }

  try:
    event_from_dict({**valid_payload, "outcome": "maybe"})
  except ValueError as exc:
    assert "Unknown approval outcome" in str(exc)
  else:
    raise AssertionError("unknown approval outcome should raise")

  try:
    event_from_dict({**valid_payload, "decision_source": "background_policy"})
  except ValueError as exc:
    assert "Unknown approval decision source" in str(exc)
  else:
    raise AssertionError("unknown approval decision source should raise")


def test_verdict_extractor_uses_completed_memory_write_payloads() -> None:
  verdict_content = """## Verdict YAML

```yaml
verdict: SCENARIOS_BUILT
confidence: HIGH
spread_check:
  bull_minus_bear_pct_of_base: 53
  materiality_threshold_pct: 10
```
"""
  decision_content = """## Decision Log Entry

```yaml
decision: "SCENARIOS_BUILT: bull/base/bear adj_eps 11.50/8.50/7.00"
rationale: "Confidence: HIGH."
```
"""
  entries = [
    _memory_write_start("tool_1", verdict_content),
    _memory_write_complete("tool_1"),
    _memory_write_start("tool_2", decision_content),
    _memory_write_complete("tool_2"),
  ]

  verdict = extract_verdict_payload(entries)

  assert verdict is not None
  assert verdict.verdict_token == "SCENARIOS_BUILT"
  assert verdict.confidence == "HIGH"
  assert verdict.materiality_cushion == 5.3
  assert verdict.one_line_summary == "SCENARIOS_BUILT: bull/base/bear adj_eps 11.50/8.50/7.00"


def test_verdict_extractor_returns_none_for_malformed_verdict_yaml() -> None:
  entries = [
    _memory_write_start("tool_1", "## Verdict YAML\n\n```yaml\nverdict: [bad\n```\n"),
    _memory_write_complete("tool_1"),
  ]

  assert extract_verdict_payload(entries) is None


def test_verdict_extractor_reads_direct_markdown_text() -> None:
  verdict = extract_verdict_payload_from_text(
    """# Run output

```yaml
verdict: SCENARIOS_BUILT
confidence: HIGH
materiality_cushion: 3.15
one_line_summary: "FY28 base case clears materiality."
```
"""
  )

  assert verdict is not None
  assert verdict.verdict_token == "SCENARIOS_BUILT"
  assert verdict.confidence == "HIGH"
  assert verdict.materiality_cushion == 3.15
  assert verdict.one_line_summary == "FY28 base case clears materiality."


def _memory_write_start(tool_call_id: str, content: str) -> SimpleNamespace:
  return SimpleNamespace(
    event={
      "type": "tool_call_start",
      "tool_call_id": tool_call_id,
      "tool_name": "memory_write",
      "tool_input": {"file": "skills/earnings-scenarios/2026-05-20-PCTY.md", "content": content},
    }
  )


def _memory_write_complete(tool_call_id: str) -> SimpleNamespace:
  return SimpleNamespace(
    event={
      "type": "tool_call_complete",
      "tool_call_id": tool_call_id,
      "tool_name": "memory_write",
      "error": None,
      "is_error": False,
      "final_tool_result_blocks": [
        {
          "type": "tool_result",
          "tool_use_id": tool_call_id,
          "content": json.dumps({"written": True}),
        }
      ],
    }
  )
