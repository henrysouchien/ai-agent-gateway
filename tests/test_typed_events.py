import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.events import (
  AggregateReadyEvent,
  AggregateReadyTrigger,
  ArtifactFailedEvent,
  ArtifactReadyEvent,
  ArtifactUpdatedEvent,
  ArtifactUnavailableEvent,
  RUN_SCOPED_EVENT_TYPES,
  RecapApproval,
  RecapArtifact,
  RecapFailure,
  RecapVerdict,
  SessionRecapEvent,
  SkillResultCapturedEvent,
  SkillRunStartedEvent,
  TYPED_EVENT_TYPES,
  ToolCallsSummary,
  ToolApprovalDecidedEvent,
  ToolApprovalRequestEvent,
  TypedRecommendationsExtractedEvent,
  event_from_dict,
  event_to_dict,
)
from agent_gateway.multi_user.billing import SessionUsageSummary


def test_typed_events_round_trip_with_type_discriminator() -> None:
  events = [
    SkillRunStartedEvent(skill_run_id="run-1", skill="earnings-scenarios", ticker="PCTY", ts=1.0),
    SkillResultCapturedEvent(
      skill_run_id="run-1",
      skill="earnings-scenarios",
      ticker="PCTY",
      exit_code=0,
      outcome="success",
      status="noop",
      gate_code="PROCEED",
      artifact_refs=["artifacts/PCTY/earnings-scenarios/result.json"],
      proposal_ids=[],
      verdict_echo={"verdict": "SCENARIOS_BUILT", "confidence": "HIGH"},
      fms_results=[],
      artifact_events=[],
      output_memory_file=None,
      cost_usd=0.12,
      duration_s=4.5,
      error=None,
      warnings=[],
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
    ArtifactUpdatedEvent(
      skill_run_id="run-1",
      ticker="PCTY",
      skill="earnings-scenarios",
      artifact_id="2026-05-20T120000.000-run-1",
      contract_name="EarningsScenarios",
      partial_view_model={"scenarios": [{"name": "Base"}]},
      ts=3.5,
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
      error_code="validation",
      error_detail="Structured artifact validation failed",
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


def test_skill_result_captured_event_round_trip() -> None:
  event = SkillResultCapturedEvent(
    skill_run_id="run-risk",
    skill="risk-review",
    ticker=None,
    exit_code=0,
    outcome="success",
    status="noop",
    gate_code="PROCEED",
    artifact_refs=["notes/skills/risk-review/result.typed.json"],
    proposal_ids=[],
    verdict_echo={
      "verdict": "RISK_REVIEW_MITIGATIONS_RECOMMENDED",
      "confidence": "MEDIUM",
      "one_line_summary": "Mitigations recommended.",
    },
    fms_results=[
      {
        "status": "noop",
        "gate_code": "PROCEED",
        "artifact_ref": "notes/skills/risk-review/result.typed.json",
      }
    ],
    artifact_events=[],
    output_memory_file=None,
    cost_usd=None,
    duration_s=7.25,
    error=None,
    warnings=["source coverage limited"],
  )

  payload = event_to_dict(event)

  assert payload["type"] == "skill_result_captured"
  assert "skill_result_captured" in TYPED_EVENT_TYPES
  assert "skill_result_captured" in RUN_SCOPED_EVENT_TYPES
  assert event_from_dict(payload) == event


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


def test_artifact_failure_events_preserve_null_html_fields_and_tool_call_id() -> None:
  payload = {
    "type": "artifact_failed",
    "skill_run_id": "run-html",
    "ticker": None,
    "skill": "_html",
    "error_code": "tool_write_failed",
    "error_detail": "write failed",
    "source_path": None,
    "tool_call_id": "toolu_html",
    "ts": 7.0,
  }

  event = event_from_dict(payload)

  assert isinstance(event, ArtifactFailedEvent)
  assert event.ticker is None
  assert event.source_path is None
  assert event.tool_call_id == "toolu_html"
  assert event_to_dict(event) == payload


def test_artifact_failure_events_accept_legacy_payload_without_tool_call_id() -> None:
  payload = {
    "type": "artifact_failed",
    "skill_run_id": "run-1",
    "ticker": "PCTY",
    "skill": "earnings-scenarios",
    "error_code": "validation",
    "error_detail": "bad artifact",
    "source_path": "notes/skills/earnings-scenarios/2026-05-20-PCTY.md",
    "ts": 8.0,
  }

  event = event_from_dict(payload)

  assert isinstance(event, ArtifactFailedEvent)
  assert event.tool_call_id is None
  assert event_to_dict(event) == {**payload, "tool_call_id": None}


def test_artifact_unavailable_events_accept_null_ticker() -> None:
  payload = {
    "type": "artifact_unavailable",
    "ticker": None,
    "skill": "_html",
    "reason": "no_runs_yet",
    "affordance": "Generate an HTML artifact",
    "ts": 9.0,
  }

  event = event_from_dict(payload)

  assert isinstance(event, ArtifactUnavailableEvent)
  assert event.ticker is None
  assert event_to_dict(event) == payload


def test_artifact_updated_event_contract_round_trip() -> None:
  event = ArtifactUpdatedEvent(
    skill_run_id="run-stream",
    ticker="MSFT",
    skill="earnings-scenarios",
    artifact_id="artifact-stream-1",
    contract_name="EarningsScenarios",
    partial_view_model={
      "base": {"label": "Base", "eps": 22.26},
      "bull": {"label": "Bull", "eps": 24.47},
    },
    ts=12.5,
  )

  payload = event_to_dict(event)
  hydrated = event_from_dict(payload)

  assert payload == {
    "type": "artifact_updated",
    "skill_run_id": "run-stream",
    "ticker": "MSFT",
    "skill": "earnings-scenarios",
    "artifact_id": "artifact-stream-1",
    "contract_name": "EarningsScenarios",
    "partial_view_model": {
      "base": {"label": "Base", "eps": 22.26},
      "bull": {"label": "Bull", "eps": 24.47},
    },
    "ts": 12.5,
  }
  assert "artifact_updated" in TYPED_EVENT_TYPES
  assert "artifact_updated" in RUN_SCOPED_EVENT_TYPES
  assert hydrated == event


def test_artifact_updated_event_rejects_malformed_partial_view_model() -> None:
  payload = {
    "type": "artifact_updated",
    "skill_run_id": "run-stream",
    "ticker": "MSFT",
    "skill": "earnings-scenarios",
    "artifact_id": "artifact-stream-1",
    "contract_name": "EarningsScenarios",
    "partial_view_model": ["not", "a", "mapping"],
    "ts": 12.5,
  }

  try:
    event_from_dict(payload)
  except ValueError as exc:
    assert "artifact_updated.partial_view_model must be a mapping" in str(exc)
  else:
    raise AssertionError("malformed partial_view_model should raise")


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
    "relay_policy_denied",
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
      "outcome": (
        "timeout"
        if decision_source == "approval_timeout"
        else "denied"
        if decision_source in {"user_denied", "relay_policy_denied", "headless_auto_deny"}
        else "approved"
      ),
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


def test_session_recap_event_round_trip_with_nested_types() -> None:
  event = _session_recap_event()

  payload = event_to_dict(event)
  hydrated = event_from_dict(payload)
  json_hydrated = event_from_dict(json.loads(json.dumps(payload)))

  assert payload["type"] == "session_recap"
  assert "session_recap" in TYPED_EVENT_TYPES
  assert "session_recap" not in RUN_SCOPED_EVENT_TYPES
  assert hydrated == event
  assert json_hydrated == event


def test_session_recap_event_supports_all_failure_types() -> None:
  failure_types = [
    "terminal_error",
    "artifact_failed",
    "artifact_unavailable",
    "budget_exceeded",
    "max_turns_reached",
  ]

  for failure_type in failure_types:
    event = _session_recap_event(
      failures=[
        RecapFailure(
          failure_type=failure_type,
          detail=f"{failure_type} detail",
          emitted_at_seq=9,
          ts=10.0,
        )
      ]
    )

    hydrated = event_from_dict(event_to_dict(event))

    assert isinstance(hydrated, SessionRecapEvent)
    assert hydrated.failures[0].failure_type == failure_type


def test_session_recap_event_supports_all_triggers() -> None:
  for trigger in ["turn_end", "explicit", "session_gc"]:
    event = _session_recap_event(trigger=trigger)

    hydrated = event_from_dict(event_to_dict(event))

    assert isinstance(hydrated, SessionRecapEvent)
    assert hydrated.trigger == trigger


def test_session_recap_event_rejects_unknown_enums() -> None:
  payload = event_to_dict(_session_recap_event())

  try:
    event_from_dict({**payload, "trigger": "background"})
  except ValueError as exc:
    assert "Unknown session recap trigger" in str(exc)
  else:
    raise AssertionError("unknown session recap trigger should raise")

  bad_failure_payload = {
    **payload,
    "failures": [{**payload["failures"][0], "failure_type": "max_turn_exceeded"}],
  }
  try:
    event_from_dict(bad_failure_payload)
  except ValueError as exc:
    assert "Unknown session recap failure type" in str(exc)
  else:
    raise AssertionError("unknown session recap failure type should raise")


def _session_recap_event(
  *,
  trigger: str = "turn_end",
  failures: list[RecapFailure] | None = None,
) -> SessionRecapEvent:
  return SessionRecapEvent(
    session_id="session-1",
    seq_range=(1, 12),
    started_at=1000.0,
    ended_at=1010.0,
    trigger=trigger,  # type: ignore[arg-type]
    artifacts=[
      RecapArtifact(
        artifact_id="artifact-1",
        skill="earnings-scenarios",
        contract_name="EarningsScenarios",
        ticker=None,
        artifact_path="artifacts/research/PCTY/earnings-scenarios.json",
        emitted_at_seq=3,
        ts=1003.0,
      )
    ],
    verdicts=[
      RecapVerdict(
        skill_run_id="run-1",
        skill="earnings-scenarios",
        ticker="PCTY",
        verdict_token="SCENARIOS_BUILT",
        confidence="HIGH",
        materiality_cushion=3.15,
        one_line_summary="FY28 base case clears materiality",
        emitted_at_seq=4,
        ts=1004.0,
      )
    ],
    approvals=[
      RecapApproval(
        tool_call_id="toolu_1",
        tool_name="bash",
        outcome="approved",
        decision_source="user_approved",
        allow_tool_type_applied=True,
        emitted_at_seq=5,
        ts=1005.0,
      )
    ],
    tool_calls_summary=ToolCallsSummary(
      total_calls=2,
      successes=1,
      errors=1,
      by_tool_name={"bash": 2},
      by_server={"local": 2},
    ),
    failures=failures
    if failures is not None
    else [
      RecapFailure(
        failure_type="terminal_error",
        detail="stream failed",
        emitted_at_seq=11,
        ts=1009.0,
      )
    ],
    usage=SessionUsageSummary(
      user_id="user-1",
      session_id="session-1",
      request_id="request-1",
      input_tokens=100,
      output_tokens=50,
      cache_read_tokens=10,
      cache_creation_tokens=5,
      cost=0.01,
      turns=1,
      channel="excel",
      started_at=1000.0,
      ended_at=1010.0,
      drain_complete=True,
      in_flight_task_count=0,
      product_id="addin",
    ),
    ts=1010.0,
  )
