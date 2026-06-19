from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

from agent_gateway.dashboard_artifact.generate_description import render_description
from agent_gateway.dashboard_artifact.qa import build_dashboard_artifact, validate_dashboard_payload
from agent_gateway.dashboard_artifact.registry import MODULE_REGISTRY
from schema.dashboard_artifact import DashboardArtifact


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "dashboard"
REGISTRY_DESCRIPTION = (
  ROOT
  / "packages"
  / "agent-gateway"
  / "agent_gateway"
  / "dashboard_artifact"
  / "registry_description.json"
)


def test_dashboard_artifact_package_import_defers_qa_module() -> None:
  script = """
import sys
from agent_gateway import dashboard_artifact

assert "agent_gateway.dashboard_artifact.qa" not in sys.modules
assert "validate_dashboard_payload" in dashboard_artifact.__all__
assert "decision_box" in dashboard_artifact.MODULE_REGISTRY
_ = dashboard_artifact.validate_dashboard_payload
assert "agent_gateway.dashboard_artifact.qa" in sys.modules
"""
  result = subprocess.run(
    [sys.executable, "-c", script],
    cwd=ROOT,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr


def test_registry_is_pinned_to_literal_expectations() -> None:
  observed = {
    name: {
      "required_keys": spec.required_keys,
      "material_keys": spec.material_keys,
      "chart_minimums": spec.chart_minimums,
    }
    for name, spec in MODULE_REGISTRY.items()
  }

  assert observed == {
    "decision_box": {
      "required_keys": ("stance", "summary"),
      "material_keys": ("summary",),
      "chart_minimums": {},
    },
    "metric_tiles": {
      "required_keys": ("items",),
      "material_keys": ("items[].value", "items[].detail"),
      "chart_minimums": {},
    },
    "table": {
      "required_keys": ("columns", "rows"),
      "material_keys": ("rows[]",),
      "chart_minimums": {},
    },
    "scenario_map": {
      "required_keys": ("cases",),
      "material_keys": ("cases[].body", "cases[].bullets[]"),
      "chart_minimums": {},
    },
    "bar_chart": {
      "required_keys": ("items",),
      "material_keys": ("items[].value",),
      "chart_minimums": {"min_items": 2},
    },
    "trend_chart": {
      "required_keys": ("periods",),
      "material_keys": ("periods[].series_values",),
      "chart_minimums": {"min_complete_periods": 2},
    },
    "timeline": {
      "required_keys": ("events",),
      "material_keys": ("events[].detail",),
      "chart_minimums": {},
    },
    "text_block": {
      "required_keys": ("body",),
      "material_keys": ("body", "bullets[]"),
      "chart_minimums": {},
    },
    "missing_evidence": {
      "required_keys": ("items",),
      "material_keys": (),
      "chart_minimums": {},
    },
  }


def test_registry_description_is_fresh() -> None:
  assert REGISTRY_DESCRIPTION.read_text(encoding="utf-8") == render_description()


def test_fixture_validation_matrix() -> None:
  full = validate_dashboard_payload(_fixture("full"), "production")
  assert full["status"] == "pass"
  assert full["warnings"] == []
  assert full["hard_failures"] == []

  tickerless = validate_dashboard_payload(_fixture("tickerless"), "production")
  assert tickerless["status"] == "pass"
  assert tickerless["warnings"] == []
  assert tickerless["hard_failures"] == []

  failing = validate_dashboard_payload(_fixture("failing"), "draft")
  assert failing["status"] == "fail"
  assert failing["production_required"] is True
  assert failing["citation_strict"] is True
  joined = "\n".join(failing["hard_failures"])
  assert "src_99" in joined
  assert "citation gap" in joined
  assert "citation_policy='strict'" in joined


def test_numeric_scalar_gap_and_inline_string_marker() -> None:
  payload = _minimal_payload(
    {
      "type": "metric_tiles",
      "title": "Metrics",
      "data": {"items": [{"label": "Upside", "value": 42}]},
    }
  )

  report = validate_dashboard_payload(payload, "production")
  assert report["status"] == "fail"
  assert any("citation gap" in failure for failure in report["hard_failures"])

  payload["sections"][0]["modules"][0]["data"]["items"][0]["value"] = "42 [src_1]"
  report = validate_dashboard_payload(payload, "production")
  assert report["status"] == "pass"
  assert report["hard_failures"] == []


def test_freeze_time_parse_and_source_freshness_are_strict_failures() -> None:
  payload = _minimal_payload(_decision_module())
  payload["metadata"]["freeze_time"] = "not-a-date"
  report = validate_dashboard_payload(payload, "production")
  assert report["status"] == "fail"
  assert any("freeze_time" in failure for failure in report["hard_failures"])

  payload = _minimal_payload(_decision_module())
  payload["metadata"]["freeze_time"] = "2026-06-06T12:00:00Z"
  payload["sources"][0]["retrieved_at"] = "2026-06-06T12:00:01Z"
  report = validate_dashboard_payload(payload, "production")
  assert report["status"] == "fail"
  assert any("postdates" in failure for failure in report["hard_failures"])


def test_placeholder_unknown_module_and_kind_mismatch_fail_closed() -> None:
  placeholder = _minimal_payload(_decision_module(summary="TODO validate 10% [src_1]"))
  report = validate_dashboard_payload(placeholder, "production")
  assert report["status"] == "fail"
  assert any("placeholder" in failure for failure in report["hard_failures"])

  unknown = _minimal_payload({"type": "sparkline", "title": "Unknown", "data": {}})
  report = validate_dashboard_payload(unknown, "production")
  assert report["status"] == "fail"
  assert any("not registered" in failure for failure in report["hard_failures"])

  wrong_kind = _minimal_payload(_decision_module())
  wrong_kind["kind"] = "other"
  report = validate_dashboard_payload(wrong_kind, "production")
  assert report["status"] == "fail"
  assert any("kind" in failure for failure in report["hard_failures"])


def test_chart_minimums_are_hard_in_strict_and_warning_in_draft() -> None:
  module = {
    "type": "bar_chart",
    "title": "One bar",
    "data": {"items": [{"label": "Only", "value": 10, "citations": ["src_1"]}]},
  }
  strict_payload = _minimal_payload(module)
  strict_report = validate_dashboard_payload(strict_payload, "production")
  assert strict_report["status"] == "fail"
  assert any("bar_chart requires at least 2 items" in failure for failure in strict_report["hard_failures"])

  draft_payload = _minimal_payload(module, readiness_posture="draft", citation_policy="warn")
  draft_payload["metadata"]["decision_context"] = None
  draft_payload["hero"] = None
  draft_payload["snapshot"] = []
  draft_report = validate_dashboard_payload(draft_payload, "draft")
  assert draft_report["status"] == "pass"
  assert any("bar_chart requires at least 2 items" in warning for warning in draft_report["warnings"])


def test_build_dashboard_artifact_returns_payload_and_partial_sidecar_fields() -> None:
  result = build_dashboard_artifact(_fixture("full"), "production", "Ready for PM review")

  assert result["payload_json"]["kind"] == "hank_dashboard.v1"
  assert result["sidecar_fields"] == {
    "title": "PCTY Decision Dashboard",
    "summary": "Ready for PM review",
    "ticker": "PCTY",
    "scope_label": None,
    "readiness_posture": "decision_ready",
    "profile": "production",
  }
  assert result["warnings"] == []

  failure = build_dashboard_artifact(_fixture("failing"), "draft", "Bad payload")
  assert failure["error"] == "dashboard_validation_failed"
  assert failure["hard_failures"]


def test_dashboard_artifact_canonical_sidecar_fixture_round_trips() -> None:
  raw = json.loads((FIXTURES / "dashboard_artifact_canonical.json").read_text(encoding="utf-8"))

  artifact = DashboardArtifact.model_validate(raw)

  assert artifact.model_dump(mode="json") == raw


def _fixture(name: str) -> dict[str, object]:
  return json.loads((FIXTURES / f"{name}.payload.json").read_text(encoding="utf-8"))


def _decision_module(summary: str = "The cited view has 10% upside [src_1].") -> dict[str, object]:
  return {
    "type": "decision_box",
    "title": "Decision",
    "data": {"stance": "constructive", "summary": summary},
  }


def _minimal_payload(
  module: dict[str, object],
  *,
  readiness_posture: str = "decision_ready",
  citation_policy: str = "strict",
) -> dict[str, object]:
  return {
    "kind": "hank_dashboard.v1",
    "title": "Minimal Dashboard",
    "subtitle": None,
    "ticker": "PCTY",
    "scope_label": None,
    "metadata": {
      "readiness_posture": readiness_posture,
      "freeze_time": "2026-06-06T18:00:00Z",
      "decision_context": "Production validation test.",
      "citation_policy": citation_policy,
    },
    "hero": {
      "title": "Hero",
      "body": "Hero includes 10% cited support [src_1].",
      "citations": ["src_1"],
    },
    "snapshot": [
      {
        "label": "Snapshot",
        "value": "10% [src_1]",
        "citations": ["src_1"],
      }
    ],
    "sections": [
      {
        "id": "main",
        "label": "Main",
        "modules": [copy.deepcopy(module)],
      }
    ],
    "sources": [
      {
        "id": "src_1",
        "type": "filing",
        "source_id": "source-1",
        "text": "Source supports the cited value.",
        "retrieved_at": "2026-06-06T12:00:00Z",
      }
    ],
  }
