from __future__ import annotations

from copy import deepcopy

import pytest

from agent_gateway import ui_blocks_validation
from agent_gateway.ui_blocks_validation import (
  FailureCode,
  build_payload_submitted,
  validate_payload,
)
from gateway_test_support.ui_blocks_fixtures import load_ui_blocks_fixtures


def _payload(*blocks: dict, **text: object) -> dict:
  return {"kind": "hank_ui_blocks.v1", "contract_version": 1, "blocks": list(blocks), **text}


def test_payload_builder_stamps_constants_omits_absent_optionals_and_is_deterministic() -> None:
  tool_input = {"blocks": [{"block": "metric-card", "props": {"label": "A", "value": "1"}}], "tail_text": None}
  first = build_payload_submitted(tool_input)
  assert first == {
    "kind": "hank_ui_blocks.v1",
    "contract_version": 1,
    "tail_text": None,
    "blocks": tool_input["blocks"],
  }
  assert "lead_text" not in first
  assert build_payload_submitted(tool_input) == first


def test_all_bundle_fixtures_match_acceptance_and_expected_code() -> None:
  for fixture in load_ui_blocks_fixtures():
    payloads = fixture.get("payloads") or [fixture.get("payload")]
    for payload in payloads:
      failures = validate_payload(payload)
      if fixture["expectation"] == "accept":
        assert failures == [], fixture["name"]
      else:
        assert failures, fixture["name"]
        assert fixture["expected_code"] in {failure["code"] for failure in failures}, fixture["name"]


@pytest.mark.parametrize("allow_layout_children", [False, True])
def test_nested_layout_children_follow_bundle_grammar(
  monkeypatch: pytest.MonkeyPatch, allow_layout_children: bool,
) -> None:
  bundle = deepcopy(ui_blocks_validation.manifest())
  if allow_layout_children:
    bundle["layout_rules"]["nested"]["allowed_children"].append("layout")
  monkeypatch.setattr(ui_blocks_validation, "manifest", lambda: bundle)
  block = {"block": "metric-card", "props": {"label": "Risk", "value": "High"}}
  payload = _payload({
    "layout": "stack",
    "children": [{"layout": "row", "children": [{"layout": "grid", "children": [block]}]}],
  })

  failures = validate_payload(payload)

  if allow_layout_children:
    assert failures == []
  else:
    assert [failure["code"] for failure in failures] == ["layout_depth_exceeded"]


@pytest.mark.parametrize(
  ("position", "kind", "child", "expected_code"),
  [
    ("top_level", "primitive", {"block": "metric-card", "props": {"label": "A", "value": "1"}}, "envelope_invalid"),
    ("nested", "primitive", {"block": "metric-card", "props": {"label": "A", "value": "1"}}, "envelope_invalid"),
    (
      "top_level", "sdk",
      {"block": "sdk:metric-grid", "props": {"source": "positions", "fields": ["totalPortfolioValue"]}},
      "envelope_invalid",
    ),
    (
      "nested", "sdk",
      {"block": "sdk:metric-grid", "props": {"source": "positions", "fields": ["totalPortfolioValue"]}},
      "envelope_invalid",
    ),
    (
      "top_level", "layout",
      {"layout": "row", "children": [{"block": "metric-card", "props": {"label": "A", "value": "1"}}]},
      "layout_depth_exceeded",
    ),
  ],
)
def test_layout_child_admission_follows_bundle_allowed_children(
  monkeypatch: pytest.MonkeyPatch, position: str, kind: str, child: dict, expected_code: str,
) -> None:
  parent = {"layout": "stack", "children": [child]}
  payload = _payload(parent if position == "top_level" else {"layout": "grid", "children": [parent]})
  assert validate_payload(payload) == []
  bundle = deepcopy(ui_blocks_validation.manifest())
  bundle["layout_rules"][position]["allowed_children"].remove(kind)
  monkeypatch.setattr(ui_blocks_validation, "manifest", lambda: bundle)

  assert [failure["code"] for failure in validate_payload(payload)] == [expected_code]


@pytest.mark.parametrize(
  ("payload", "code"),
  [
    ({"kind": "wrong", "contract_version": 1, "blocks": []}, "envelope_invalid"),
    (_payload({"block": "nope", "props": {}}), "unknown_block"),
    (_payload({"layout": "stack", "children": [{"view": "overview.concentration", "scope": {}}]}), "view_in_layout"),
    (_payload({"view": "not.admitted", "scope": {}}), "unknown_view"),
    (_payload({"block": "sdk:metric-grid", "props": {"source": "nope", "fields": ["x"]}}), "unknown_source"),
    (_payload({"block": "metric-card", "props": {"label": "missing value"}}), "props_invalid"),
    (_payload({"view": "overview.concentration", "scope": {"portfolio_id": "p"}}), "scope_forbidden_field"),
    (_payload({"view": "thesis.quantifying_risk_card", "scope": {"ticker": "AAPL"}}), "scope_missing_required_field"),
    (_payload({"view": "thesis.quantifying_risk_card", "scope": {"ticker": "AAPL", "research_file_id": "1"}}), "scope_invalid_type"),
  ],
)
def test_failure_classes(payload: dict, code: str) -> None:
  assert code in {failure["code"] for failure in validate_payload(payload)}


@pytest.mark.parametrize(
  "field",
  [
    "summary",
    {"key": "summary", "label": "Summary"},
  ],
)
def test_metric_grid_rejects_object_valued_fields(field: object) -> None:
  payload = _payload({
    "block": "sdk:metric-grid",
    "props": {"source": "portfolio-summary", "fields": [field]},
  })

  assert "field_mapping_invalid" in {
    failure["code"] for failure in validate_payload(payload)
  }


def test_metric_grid_accepts_scalar_portfolio_value_field() -> None:
  payload = _payload({
    "block": "sdk:metric-grid",
    "props": {
      "source": "positions",
      "fields": [{"key": "totalPortfolioValue", "label": "Portfolio Value", "format": "currency"}],
    },
  })

  assert validate_payload(payload) == []


def test_handler_stage_code_vocabulary_has_one_home() -> None:
  assert {code.value for code in FailureCode} >= {
    "view_artifact_missing",
    "view_artifact_stale",
    "view_preflight_error",
    "manifest_mismatch",
    "payload_too_large",
  }


def test_failure_order_is_deterministic_and_indices_follow_shuffled_blocks() -> None:
  blocks = [
    {"view": "overview.unknown", "scope": {}},
    {"block": "sdk:metric-grid", "props": {"source": "unknown", "fields": ["x"]}},
    {"view": "thesis.business_quality_card", "scope": {}},
  ]
  payload = _payload(*blocks)
  first = validate_payload(payload)
  assert validate_payload(deepcopy(payload)) == first
  assert [(failure["block_index"], failure["code"]) for failure in first] == [
    (0, "unknown_view"),
    (1, "unknown_source"),
    (2, "scope_missing_required_field"),
  ]
  shuffled = validate_payload(_payload(blocks[2], blocks[0], blocks[1]))
  assert [(failure["block_index"], failure["code"]) for failure in shuffled] == [
    (1, "unknown_view"),
    (2, "unknown_source"),
    (0, "scope_missing_required_field"),
  ]


def test_schema_path_is_preserved_for_unmapped_envelope_failure() -> None:
  failure = validate_payload(_payload({"block": "metric-card", "props": {}}, extra=True))[0]
  assert failure["code"] == "envelope_invalid"
  assert "$.extra" in failure["detail"]
