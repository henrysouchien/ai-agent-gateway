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
  VerdictEmittedEvent,
  event_from_dict,
  event_to_dict,
)
from agent_gateway.verdict_extractor import extract_verdict_payload


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
  ]

  for event in events:
    payload = event_to_dict(event)
    assert payload["type"]
    assert event_to_dict(event_from_dict(payload)) == payload

  assert "skill_run_id" not in event_to_dict(events[-1])


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
