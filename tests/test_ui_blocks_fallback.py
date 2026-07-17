from __future__ import annotations

import pytest

from agent_gateway.ui_blocks_contract import fallback_projection_table
from agent_gateway.ui_blocks_fallback import text_fallback


TABLE = fallback_projection_table()


def _fallback(*blocks: dict, **extra: object) -> str:
  return text_fallback(
    {"kind": "hank_ui_blocks.v1", "contract_version": 1, "blocks": list(blocks), **extra},
    TABLE,
  )


@pytest.mark.parametrize(
  ("block", "expected"),
  [
    ({"block": "metric-card", "props": {"label": "Risk", "value": "82", "change": "+4"}}, "**Risk:** 82 (+4)"),
    ({"block": "stat-pair", "props": {"label": "Vol", "value": 12.4}}, "**Vol:** 12.4"),
    ({"block": "status-cell", "props": {"label": "Test", "value": "Pass"}}, "Test: Pass"),
    ({"block": "insight-banner", "props": {"title": "Watch", "subtitle": "Rates"}}, "> **Watch** — Rates"),
    ({"block": "section-header", "props": {"title": "Risk"}}, "### Risk"),
    ({"block": "gradient-progress", "props": {"value": 64}}, "Progress: 64%"),
    ({"block": "sparkline-chart", "props": {"label": "Trend", "data": [10, 12, 11]}}, "Trend: 3 points, 10–12"),
    ({"block": "sdk:chart-panel", "props": {"source": "performance"}}, "_[live chart-panel: source performance]_"),
    ({"view": "overview.concentration", "scope": {}}, "_[view: Concentration (current portfolio)]_"),
    ({"view": "thesis.business_quality_card", "scope": {"ticker": "AAPL", "research_file_id": 7}}, "_[view: Business Quality (research\\_file\\_id=7, ticker=AAPL)]_"),
  ],
)
def test_per_kind_projection(block: dict, expected: str) -> None:
  assert _fallback(block) == expected


def test_lead_tail_are_verbatim_once_and_interpolations_are_markdown_escaped() -> None:
  output = _fallback(
    {"block": "metric-card", "props": {"label": "A*[x]", "value": "<1|2>"}},
    lead_text="*lead*",
    tail_text="#tail",
  )
  assert output == "*lead*\n\n**A\\*\\[x\\]:** \\<1\\|2\\>\n\n#tail"


def test_layouts_flatten_depth_first_children_in_order() -> None:
  output = _fallback(
    {
      "layout": "stack",
      "children": [
        {"block": "status-cell", "props": {"label": "A", "value": "1"}},
        {"layout": "row", "children": [
          {"block": "status-cell", "props": {"label": "B", "value": "2"}},
          {"block": "status-cell", "props": {"label": "C", "value": "3"}},
        ]},
      ],
    }
  )
  assert output == "A: 1\n\nB: 2\n\nC: 3"


def test_data_table_is_gfm_escaped_and_capped_at_twenty_rows() -> None:
  rows = [{"name": f"row|{index}"} for index in range(23)]
  output = _fallback({"block": "data-table", "props": {
    "columns": [{"key": "name", "label": "Name|Label"}], "data": rows, "rowKey": "name"
  }})
  assert output.splitlines()[:3] == ["| Name\\|Label |", "| --- |", "| row\\|0 |"]
  assert "row\\|19" in output
  assert "row\\|20" not in output
  assert output.endswith("_( +3 more rows )_")


def test_character_cap_includes_bundle_suffix() -> None:
  output = _fallback({"block": "status-cell", "props": {"label": "A", "value": "x" * 9000}})
  assert len(output) == 8000
  assert output.endswith("_( …truncated )_")


def test_empty_and_partial_props_are_total_and_deterministic() -> None:
  assert _fallback({"block": "metric-card", "props": {}}) == "**:** "
  assert _fallback({"block": "sparkline-chart", "props": {}}) == "Series: 0 points, n/a–n/a"
  assert _fallback({"block": "data-table", "props": {}}) == ""
